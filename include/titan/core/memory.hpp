#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#else
#include <xmmintrin.h>
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "titan/core/types.hpp"

namespace titan::core {

// ========================================================================
// LinearAllocator (Bump Allocator)
// O(1) allocations for static lifetime objects.
// ========================================================================
class LinearAllocator {
private:
    std::unique_ptr<std::byte[]> buffer_;
    std::size_t capacity_;
    std::size_t offset_{0};

public:
    explicit LinearAllocator(std::size_t size_bytes);

    LinearAllocator(const LinearAllocator&) = delete;
    LinearAllocator& operator=(const LinearAllocator&) = delete;

    template <typename T>
    [[nodiscard]] T* allocate(std::size_t count) noexcept {
        std::size_t space = capacity_ - offset_;
        void* ptr = buffer_.get() + offset_;

        if (std::align(alignof(T), count * sizeof(T), ptr, space)) {
            offset_ = capacity_ - space + (count * sizeof(T));
            return static_cast<T*>(ptr);
        }
        return nullptr;
    }

    void reset() noexcept;
};

// ========================================================================
// OrderPoolAllocator
// O(1) handle-based pool for dynamic OrderNode allocations.
// ========================================================================
class OrderPoolAllocator {
private:
    OrderNode* nodes_ = nullptr;
    Handle* free_list_ = nullptr;
    std::size_t head_ = 0;
    uint32_t capacity_ = 0;

public:
    OrderPoolAllocator() = default;

    void init(OrderNode* nodes, Handle* free_list, uint32_t capacity) noexcept;

    void reset() noexcept;

    [[nodiscard]] inline Handle allocate() noexcept {
        if (head_ == 0) [[unlikely]] {
            return NULL_HANDLE;
        }

        const Handle h = free_list_[--head_];

        nodes_[h].next = NULL_HANDLE;
        nodes_[h].prev = NULL_HANDLE;

        return h;
    }

    inline void free(Handle handle) noexcept {
        // Prefetch the node data to L1 cache.
        _mm_prefetch(reinterpret_cast<const char*>(&nodes_[handle]), _MM_HINT_T0);

        free_list_[head_++] = handle;
    }

    [[nodiscard]] inline OrderNode& get_node(Handle handle) noexcept { return nodes_[handle]; }
    [[nodiscard]] inline const OrderNode& get_node(Handle handle) const noexcept { return nodes_[handle]; }
    [[nodiscard]] inline std::size_t size() const noexcept { return head_; }
    [[nodiscard]] inline uint32_t capacity() const noexcept { return capacity_; }
};

// ========================================================================
// UnifiedMemoryArena
// Master owner of all simulation memory allocations.
// Allocates OS-level Pinned Memory (DMA-ready) for Zero-Copy Tensors.
// ========================================================================
class UnifiedMemoryArena {
private:
    // --- Limits & Config ---
    uint32_t num_envs_;
    uint32_t num_agents_;
    uint32_t max_orders_per_env_;
    uint32_t max_actions_per_step_;
    uint32_t max_events_per_step_;
    uint32_t max_orders_per_agent_;
    uint32_t obs_depth_;

    // --- Master Pinned Memory Block ---
    void* pinned_block_{nullptr};
    std::size_t total_pinned_bytes_{0};
    std::size_t bridge_tensors_offset_{0};

    // --- C++ Internal LOB Memory (Mapped into pinned_block_) ---
    OrderNode* raw_nodes_{nullptr};
    Handle* raw_free_lists_{nullptr};
    std::vector<OrderPoolAllocator> pools_;
    
    // --- Zero-Copy Tensors (Mapped into pinned_block_) ---
    // The exact dimensions PyTorch will use when wrapping these pointers
    
    // 0. SMDP Ready Mask (Triggers RL Inference)
    // Shape: [num_envs, num_agents]
    uint8_t* ready_mask_ptr_{nullptr};

    // 1. Input Command Stream
    // Shape: [num_envs, max_actions_per_step]
    ActionPayload* actions_ptr_{nullptr};
    
    // 2. Output Event Stream (Global Historical Ring Buffer per Env)
    // Shape: [num_envs, max_events_per_step]
    MarketDataEvent* events_ptr_{nullptr};
    
    // 3. Agent State Tracking
    // Shape: [num_envs, max_active_orders]
    ActiveOrderRecord* active_orders_ptr_{nullptr};
    
    // 4. Observations (Personalized realities due to latency)
    // Shape: [num_envs, num_agents, obs_depth * 4]
    float* lob_ptr_{nullptr};
    
    // Shape: [num_envs, num_agents]
    float* cash_ptr_{nullptr};
    float* inventory_ptr_{nullptr};

    // Shape: [num_envs]
    uint64_t* event_cursors_ptr_{nullptr};

    LinearAllocator linear_allocator_;

public:
    UnifiedMemoryArena(uint32_t num_envs, uint32_t num_agents, 
                       uint32_t max_orders_per_env, uint32_t max_actions_per_step, 
                       uint32_t max_events_per_step, uint32_t max_orders_per_agent, 
                       uint32_t obs_depth, std::size_t linear_bytes);
                       
    ~UnifiedMemoryArena();

    UnifiedMemoryArena(const UnifiedMemoryArena&) = delete;
    UnifiedMemoryArena& operator=(const UnifiedMemoryArena&) = delete;

    // --- C++ Internal Interfaces ---
    [[nodiscard]] inline OrderPoolAllocator& order_pool(uint32_t env_id) noexcept { return pools_[env_id]; }
    [[nodiscard]] inline LinearAllocator& get_linear_allocator() noexcept { return linear_allocator_; }

    // --- Config Accessors ---
    [[nodiscard]] inline uint32_t num_envs() const noexcept { return num_envs_; }
    [[nodiscard]] inline uint32_t num_agents() const noexcept { return num_agents_; }
    [[nodiscard]] inline uint32_t max_actions_per_step() const noexcept { return max_actions_per_step_; }
    [[nodiscard]] inline uint32_t max_events_per_step() const noexcept { return max_events_per_step_; }
    [[nodiscard]] inline uint32_t max_orders_per_agent() const noexcept { return max_orders_per_agent_; }
    [[nodiscard]] inline uint32_t max_active_orders() const noexcept { return num_agents_ * max_orders_per_agent_; }
    [[nodiscard]] inline uint32_t obs_depth() const noexcept { return obs_depth_; }

    // --- Zero-Copy Tensor Pointers (For DLPack / Nanobind) ---
    [[nodiscard]] inline uint8_t* ready_mask_ptr() noexcept { return ready_mask_ptr_; }
    [[nodiscard]] inline ActionPayload* actions_ptr() noexcept { return actions_ptr_; }
    [[nodiscard]] inline MarketDataEvent* events_ptr() noexcept { return events_ptr_; }
    [[nodiscard]] inline ActiveOrderRecord* active_orders_ptr() noexcept { return active_orders_ptr_; }
    
    [[nodiscard]] inline float* lob_ptr() noexcept { return lob_ptr_; }
    [[nodiscard]] inline float* cash_ptr() noexcept { return cash_ptr_; }
    [[nodiscard]] inline float* inventory_ptr() noexcept { return inventory_ptr_; }
    [[nodiscard]] inline uint64_t* event_cursors_ptr() noexcept { return event_cursors_ptr_; }

    // --- Resets ---
    void reset(const std::vector<uint32_t>& env_indices) noexcept;
    void reset_all() noexcept;
};

}  // namespace titan::core