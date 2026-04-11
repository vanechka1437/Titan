#include "titan/core/memory.hpp"

#include <cstring>

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

constexpr std::size_t PAGE_SIZE = 4096;

UnifiedMemoryArena::UnifiedMemoryArena(uint32_t num_envs, uint32_t max_orders_per_env, std::size_t linear_bytes,
                                       uint32_t num_agents, uint32_t obs_dim, uint32_t action_dim)
    : num_envs_(num_envs),
      max_orders_per_env_(max_orders_per_env),
      num_agents_(num_agents),
      obs_dim_(obs_dim),
      action_dim_(action_dim),
      raw_nodes_(static_cast<std::size_t>(num_envs) * max_orders_per_env),
      raw_free_lists_(static_cast<std::size_t>(num_envs) * max_orders_per_env),
      pools_(num_envs),
      linear_allocator_(linear_bytes),
      observation_tensor_(static_cast<std::size_t>(num_envs) * num_agents * obs_dim, 0.0f),
      action_tensor_(static_cast<std::size_t>(num_envs) * num_agents * action_dim, 0.0f),
      ready_mask_(static_cast<std::size_t>(num_envs) * num_agents, 0) {
    // ========================================================================
    // MEMORY WARM-UP (PREFAULTING)
    // ========================================================================

    // 1. Prefaulting the OrderNode array
    auto* volatile_nodes = reinterpret_cast<volatile uint8_t*>(raw_nodes_.data());
    const std::size_t total_nodes_bytes = raw_nodes_.size() * sizeof(OrderNode);
    for (std::size_t offset = 0; offset < total_nodes_bytes; offset += PAGE_SIZE) {
        volatile_nodes[offset] = 0;
    }

    // 2. Prefaulting the Free List (Handles) array
    auto* volatile_handles = reinterpret_cast<volatile uint8_t*>(raw_free_lists_.data());
    const std::size_t total_handles_bytes = raw_free_lists_.size() * sizeof(Handle);
    for (std::size_t offset = 0; offset < total_handles_bytes; offset += PAGE_SIZE) {
        volatile_handles[offset] = 0;
    }

    // 3. Prefaulting Bridge Tensors (Zero-Copy Memory)
    auto* volatile_obs = reinterpret_cast<volatile uint8_t*>(observation_tensor_.data());
    const std::size_t total_obs_bytes = observation_tensor_.size() * sizeof(float);
    for (std::size_t offset = 0; offset < total_obs_bytes; offset += PAGE_SIZE) {
        volatile_obs[offset] = 0;
    }

    auto* volatile_act = reinterpret_cast<volatile uint8_t*>(action_tensor_.data());
    const std::size_t total_act_bytes = action_tensor_.size() * sizeof(float);
    for (std::size_t offset = 0; offset < total_act_bytes; offset += PAGE_SIZE) {
        volatile_act[offset] = 0;
    }

    auto* volatile_mask = reinterpret_cast<volatile uint8_t*>(ready_mask_.data());
    const std::size_t total_mask_bytes = ready_mask_.size() * sizeof(int8_t);
    for (std::size_t offset = 0; offset < total_mask_bytes; offset += PAGE_SIZE) {
        volatile_mask[offset] = 0;
    }

    // ========================================================================
    // POOL INITIALIZATION
    // ========================================================================

    for (uint32_t env_idx = 0; env_idx < num_envs_; ++env_idx) {
        const std::size_t mem_offset = static_cast<std::size_t>(env_idx) * max_orders_per_env_;
        pools_[env_idx].init(raw_nodes_.data() + mem_offset, raw_free_lists_.data() + mem_offset, max_orders_per_env_);
    }
}

void UnifiedMemoryArena::reset() noexcept {
    linear_allocator_.reset();

    for (auto& pool : pools_) {
        pool.reset();
    }

    std::memset(observation_tensor_.data(), 0, observation_tensor_.size() * sizeof(float));
    std::memset(action_tensor_.data(), 0, action_tensor_.size() * sizeof(float));
    std::memset(ready_mask_.data(), 0, ready_mask_.size() * sizeof(int8_t));

    state_.store(0, std::memory_order_release);
}

}  // namespace titan::core