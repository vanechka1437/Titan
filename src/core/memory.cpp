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
    head_ = capacity;

    // Initialize free list as a LIFO stack.
    // Sequential initial handles maximize spatial locality for early allocations.
    for (uint32_t i = 0; i < capacity; ++i) {
        free_list_[i] = static_cast<Handle>(capacity - 1 - i);
        nodes_[i].generation = 0;
        nodes_[i].next = NULL_HANDLE;
        nodes_[i].prev = NULL_HANDLE;
    }
}

Handle OrderPoolAllocator::allocate() noexcept {
    // [[unlikely]] minimizes instruction pipeline flushes on the hot path.
    if (head_ == 0) [[unlikely]] {
        return NULL_HANDLE;
    }
    return free_list_[--head_];
}

void OrderPoolAllocator::free(Handle handle) noexcept {
    // Zero runtime bounds checking. Caller is strictly responsible for valid handles.
    nodes_[handle].generation++;
    nodes_[handle].next = NULL_HANDLE;
    nodes_[handle].prev = NULL_HANDLE;
    free_list_[head_++] = handle;
}

// ============================================================================
// UnifiedMemoryArena
// ============================================================================

UnifiedMemoryArena::UnifiedMemoryArena(uint32_t num_envs, uint32_t max_orders_per_env, std::size_t linear_bytes)
    : num_envs_(num_envs),
      max_orders_per_env_(max_orders_per_env),
      raw_nodes_(static_cast<std::size_t>(num_envs) * max_orders_per_env),
      raw_free_lists_(static_cast<std::size_t>(num_envs) * max_orders_per_env),
      pools_(num_envs),
      linear_allocator_(linear_bytes) {
    // Partition the contiguous physical memory block among logical pool allocators.
    for (uint32_t i = 0; i < num_envs_; ++i) {
        const std::size_t offset = static_cast<std::size_t>(i) * max_orders_per_env_;

        pools_[i].init(raw_nodes_.data() + offset, raw_free_lists_.data() + offset, max_orders_per_env_);
    }
}

}  // namespace titan::core