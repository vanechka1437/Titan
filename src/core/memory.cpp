#include "titan/core/memory.hpp"

namespace titan::core {

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

        nodes_[i].generation = 0;
        nodes_[i].next = NULL_HANDLE;
        nodes_[i].prev = NULL_HANDLE;

        // Zero out payload fields for strictly deterministic initialization
        nodes_[i].owner_id = 0;
        nodes_[i].price = 0;
        nodes_[i].quantity = 0;
        nodes_[i].side = 0;
    }
}

// ============================================================================
// UnifiedMemoryArena
// ============================================================================

// Standard memory page size for x86_64 architecture (Windows/Linux)
constexpr std::size_t PAGE_SIZE = 4096;

UnifiedMemoryArena::UnifiedMemoryArena(uint32_t num_envs, uint32_t max_orders_per_env, std::size_t linear_bytes)
    : num_envs_(num_envs),
      max_orders_per_env_(max_orders_per_env),
      raw_nodes_(static_cast<std::size_t>(num_envs) * max_orders_per_env),
      raw_free_lists_(static_cast<std::size_t>(num_envs) * max_orders_per_env),
      pools_(num_envs),
      linear_allocator_(linear_bytes) {
    // ========================================================================
    // MEMORY WARM-UP (PREFAULTING)
    // ========================================================================
    // In user-space HFT applications, the OS allocates memory lazily (Virtual Memory).
    // To prevent catastrophic microsecond-level latency spikes (Page Faults) during
    // live simulation or trading, we must force the OS to map physical RAM pages
    // to our virtual addresses immediately.
    //
    // We achieve this by forcing a write operation on every 4KB page boundary.
    // The 'volatile' keyword is critical here: it strictly forbids the compiler's
    // optimizer (-O3) from optimizing away this seemingly "useless" loop.
    // ========================================================================

    // 1. Prefaulting the OrderNode array
    auto* volatile_nodes = reinterpret_cast<volatile uint8_t*>(raw_nodes_.data());
    const std::size_t total_nodes_bytes = raw_nodes_.size() * sizeof(OrderNode);

    for (std::size_t offset = 0; offset < total_nodes_bytes; offset += PAGE_SIZE) {
        volatile_nodes[offset] = 0;  // Trigger hardware page fault
    }

    // 2. Prefaulting the Free List (Handles) array
    auto* volatile_handles = reinterpret_cast<volatile uint8_t*>(raw_free_lists_.data());
    const std::size_t total_handles_bytes = raw_free_lists_.size() * sizeof(Handle);

    for (std::size_t offset = 0; offset < total_handles_bytes; offset += PAGE_SIZE) {
        volatile_handles[offset] = 0;  // Trigger hardware page fault
    }

    // ========================================================================
    // POOL INITIALIZATION
    // ========================================================================
    // Partition the unified raw memory into isolated logic pools for each RL environment.

    for (uint32_t env_idx = 0; env_idx < num_envs_; ++env_idx) {
        const std::size_t mem_offset = static_cast<std::size_t>(env_idx) * max_orders_per_env_;

        pools_[env_idx].init(raw_nodes_.data() + mem_offset, raw_free_lists_.data() + mem_offset, max_orders_per_env_);
    }
}

}  // namespace titan::core