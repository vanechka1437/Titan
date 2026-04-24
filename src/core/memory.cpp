#include "titan/core/memory.hpp"

#include <cstring>
#include <new>
#include <stdexcept>

#if defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace titan::core {

// ============================================================================
// OS-Level Pinned Memory Helpers
// ============================================================================

namespace {

[[nodiscard]] inline void* allocate_pinned_memory(std::size_t size) {
#if defined(_WIN32) || defined(_WIN64)
    void* ptr = VirtualAlloc(NULL, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!ptr) {
        throw std::bad_alloc();
    }
    // Best-effort page locking. May require increasing process working set size.
    VirtualLock(ptr, size);
    return ptr;
#else
    // MAP_POPULATE pre-faults the memory immediately to prevent page faults later
    void* ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE, 
                     MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    if (ptr == MAP_FAILED) {
        throw std::bad_alloc();
    }
    // Lock pages in physical RAM (prevent swap out)
    if (mlock(ptr, size) != 0) {
        // We log or ignore depending on strictness. For HFT, failure to lock is 
        // critical, but we'll proceed for OS environments with strict limits.
    }
    return ptr;
#endif
}

inline void free_pinned_memory(void* ptr, std::size_t size) noexcept {
    if (!ptr) return;
#if defined(_WIN32) || defined(_WIN64)
    VirtualUnlock(ptr, size);
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munlock(ptr, size);
    munmap(ptr, size);
#endif
}

[[nodiscard]] constexpr std::size_t align_up(std::size_t size, std::size_t alignment = 64) noexcept {
    return (size + alignment - 1) & ~(alignment - 1);
}

} // namespace

// ============================================================================
// LinearAllocator
// ============================================================================

LinearAllocator::LinearAllocator(std::size_t size_bytes)
    : buffer_(std::make_unique<std::byte[]>(size_bytes)), capacity_(size_bytes), offset_(0) {}

void LinearAllocator::reset() noexcept {
    offset_ = 0;
}

// ============================================================================
// OrderPoolAllocator
// ============================================================================

void OrderPoolAllocator::init(OrderNode* nodes, Handle* free_list, uint32_t capacity) noexcept {
    nodes_ = nodes;
    free_list_ = free_list;
    capacity_ = capacity;
    head_ = static_cast<std::size_t>(capacity);

    for (uint32_t i = 0; i < capacity; ++i) {
        free_list_[i] = static_cast<Handle>(capacity - 1 - i);

        nodes_[i].next = NULL_HANDLE;
        nodes_[i].prev = NULL_HANDLE;

        // Zero out payload fields for strict deterministic init
        nodes_[i].owner_id = 0;
        nodes_[i].price = 0;
        nodes_[i].quantity = 0;
        nodes_[i].side = 0;
    }
}

void OrderPoolAllocator::reset() noexcept {
    head_ = static_cast<std::size_t>(capacity_);

    for (uint32_t i = 0; i < capacity_; ++i) {
        free_list_[i] = static_cast<Handle>(capacity_ - 1 - i);
        nodes_[i].next = NULL_HANDLE;
        nodes_[i].prev = NULL_HANDLE;
    }
}

// ============================================================================
// UnifiedMemoryArena
// ============================================================================

UnifiedMemoryArena::UnifiedMemoryArena(uint32_t num_envs, uint32_t num_agents,
                                       uint32_t max_orders_per_env, uint32_t max_actions_per_step,
                                       uint32_t max_events_per_step, uint32_t max_orders_per_agent,
                                       uint32_t obs_depth, std::size_t linear_bytes)
    : num_envs_(num_envs),
      num_agents_(num_agents),
      max_orders_per_env_(max_orders_per_env),
      max_actions_per_step_(max_actions_per_step),
      max_events_per_step_(max_events_per_step),
      max_orders_per_agent_(max_orders_per_agent),
      obs_depth_(obs_depth),
      pools_(num_envs),
      linear_allocator_(linear_bytes) {

    // 1. Calculate byte sizes for each sector
    const std::size_t nodes_bytes   = static_cast<std::size_t>(num_envs) * max_orders_per_env * sizeof(OrderNode);
    const std::size_t handles_bytes = static_cast<std::size_t>(num_envs) * max_orders_per_env * sizeof(Handle);
    
    const std::size_t ready_mask_bytes = static_cast<std::size_t>(num_envs) * num_agents * sizeof(uint8_t);
    const std::size_t actions_bytes = static_cast<std::size_t>(num_envs) * max_actions_per_step * sizeof(ActionPayload);
    const std::size_t events_bytes  = static_cast<std::size_t>(num_envs) * max_events_per_step * sizeof(MarketDataEvent);
    const std::size_t active_orders_bytes = static_cast<std::size_t>(num_envs) * num_agents * max_orders_per_agent * sizeof(ActiveOrderRecord);
    
    // LOB: num_agents * depth * 2 sides (bid/ask) * 2 values (price/qty)
    const std::size_t lob_bytes = static_cast<std::size_t>(num_envs) * num_agents * obs_depth * 4 * sizeof(float);
    const std::size_t cash_bytes = static_cast<std::size_t>(num_envs) * num_agents * sizeof(float);
    const std::size_t inventory_bytes = static_cast<std::size_t>(num_envs) * num_agents * sizeof(float);
    
    // Event Cursors
    const std::size_t event_cursors_bytes = static_cast<std::size_t>(num_envs) * sizeof(uint64_t);

    // 2. Align offsets to 64 bytes (cache line) to prevent false sharing
    std::size_t offset = 0;
    
    const std::size_t nodes_offset = offset;
    offset = align_up(offset + nodes_bytes);

    const std::size_t handles_offset = offset;
    offset = align_up(offset + handles_bytes);
    
    bridge_tensors_offset_ = offset;
    
    const std::size_t ready_mask_offset = offset;
    offset = align_up(offset + ready_mask_bytes);
    
    const std::size_t actions_offset = offset;
    offset = align_up(offset + actions_bytes);

    const std::size_t events_offset = offset;
    offset = align_up(offset + events_bytes);

    const std::size_t active_orders_offset = offset;
    offset = align_up(offset + active_orders_bytes);

    const std::size_t lob_offset = offset;
    offset = align_up(offset + lob_bytes);

    const std::size_t cash_offset = offset;
    offset = align_up(offset + cash_bytes);

    const std::size_t inventory_offset = offset;
    offset = align_up(offset + inventory_bytes);

    const std::size_t event_cursors_offset = offset;
    offset = align_up(offset + event_cursors_bytes);

    total_pinned_bytes_ = offset;

    // 3. Allocate Pinned Memory Block
    pinned_block_ = allocate_pinned_memory(total_pinned_bytes_);
    
    // Zero out the entire pinned block deterministically
    std::memset(pinned_block_, 0, total_pinned_bytes_);

    // 4. Map Pointers to Offsets
    auto* base_ptr = static_cast<std::byte*>(pinned_block_);
    
    raw_nodes_         = reinterpret_cast<OrderNode*>(base_ptr + nodes_offset);
    raw_free_lists_    = reinterpret_cast<Handle*>(base_ptr + handles_offset);
    
    ready_mask_ptr_    = reinterpret_cast<uint8_t*>(base_ptr + ready_mask_offset);
    actions_ptr_       = reinterpret_cast<ActionPayload*>(base_ptr + actions_offset);
    events_ptr_        = reinterpret_cast<MarketDataEvent*>(base_ptr + events_offset);
    active_orders_ptr_ = reinterpret_cast<ActiveOrderRecord*>(base_ptr + active_orders_offset);
    lob_ptr_           = reinterpret_cast<float*>(base_ptr + lob_offset);
    cash_ptr_          = reinterpret_cast<float*>(base_ptr + cash_offset);
    inventory_ptr_     = reinterpret_cast<float*>(base_ptr + inventory_offset);
    event_cursors_ptr_ = reinterpret_cast<uint64_t*>(base_ptr + event_cursors_offset);

    // 5. Initialize Pools
    for (uint32_t env_idx = 0; env_idx < num_envs_; ++env_idx) {
        const std::size_t mem_offset = static_cast<std::size_t>(env_idx) * max_orders_per_env_;
        pools_[env_idx].init(raw_nodes_ + mem_offset, raw_free_lists_ + mem_offset, max_orders_per_env_);
    }
}

UnifiedMemoryArena::~UnifiedMemoryArena() {
    if (pinned_block_) {
        free_pinned_memory(pinned_block_, total_pinned_bytes_);
        pinned_block_ = nullptr;
    }
}

void UnifiedMemoryArena::reset(const std::vector<uint32_t>& env_indices) noexcept {
    // Partial reset for specific environments
    for (const uint32_t env_idx : env_indices) {
        pools_[env_idx].reset();

        // Size calculation for single environment slices
        const std::size_t ready_mask_slice = num_agents_ * sizeof(uint8_t);
        const std::size_t actions_slice = max_actions_per_step_ * sizeof(ActionPayload);
        const std::size_t events_slice  = max_events_per_step_ * sizeof(MarketDataEvent);
        const std::size_t active_orders_slice = num_agents_ * max_orders_per_agent_ * sizeof(ActiveOrderRecord);
        const std::size_t lob_slice = num_agents_ * obs_depth_ * 4 * sizeof(float);
        const std::size_t cash_slice = num_agents_ * sizeof(float);
        const std::size_t inventory_slice = num_agents_ * sizeof(float);

        // Zero out specific environment data
        std::memset(reinterpret_cast<std::byte*>(ready_mask_ptr_) + (env_idx * ready_mask_slice), 0, ready_mask_slice);
        std::memset(reinterpret_cast<std::byte*>(actions_ptr_) + (env_idx * actions_slice), 0, actions_slice);
        std::memset(reinterpret_cast<std::byte*>(events_ptr_) + (env_idx * events_slice), 0, events_slice);
        std::memset(reinterpret_cast<std::byte*>(active_orders_ptr_) + (env_idx * active_orders_slice), 0, active_orders_slice);
        std::memset(reinterpret_cast<std::byte*>(lob_ptr_) + (env_idx * lob_slice), 0, lob_slice);
        std::memset(reinterpret_cast<std::byte*>(cash_ptr_) + (env_idx * cash_slice), 0, cash_slice);
        std::memset(reinterpret_cast<std::byte*>(inventory_ptr_) + (env_idx * inventory_slice), 0, inventory_slice);
        
        // Reset event cursor
        event_cursors_ptr_[env_idx] = 0;
    }
}

void UnifiedMemoryArena::reset_all() noexcept {
    linear_allocator_.reset();

    for (auto& pool : pools_) {
        pool.reset();
    }

    const std::size_t zero_size = total_pinned_bytes_ - bridge_tensors_offset_;
    std::memset(static_cast<std::byte*>(pinned_block_) + bridge_tensors_offset_, 0, zero_size);
}

}  // namespace titan::core