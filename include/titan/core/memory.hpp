#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include <atomic>
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
        // Helps if we need to access this node's generation immediately after.
        _mm_prefetch(reinterpret_cast<const char*>(&nodes_[handle]), _MM_HINT_T0);

        nodes_[handle].generation++;
        free_list_[head_++] = handle;
    }

    [[nodiscard]] inline OrderNode& get_node(Handle handle) noexcept { return nodes_[handle]; }

    [[nodiscard]] inline const OrderNode& get_node(Handle handle) const noexcept { return nodes_[handle]; }

    [[nodiscard]] inline std::size_t size() const noexcept { return head_; }
};

// ========================================================================
// UnifiedMemoryArena
// Master owner of all simulation memory allocations.
// Combines LOB Memory Pools and Zero-Copy Python/C++ Shared Tensors.
// ========================================================================
class UnifiedMemoryArena {
private:
    // Config
    uint32_t num_envs_;
    uint32_t max_orders_per_env_;
    uint32_t num_agents_;
    uint32_t obs_dim_;
    uint32_t action_dim_;

    // LOB Memory
    std::vector<OrderNode> raw_nodes_;
    std::vector<Handle> raw_free_lists_;
    std::vector<OrderPoolAllocator> pools_;
    LinearAllocator linear_allocator_;

    // Python/C++ Bridge Memory (Zero-Copy Tensors)
    std::vector<float> observation_tensor_;
    std::vector<float> action_tensor_;
    std::vector<int8_t> ready_mask_;

    // Spinlock for execution control
    // 0 = C++ holds execution, 1 = Python holds execution
    alignas(64) std::atomic<int> state_{0};

public:
    UnifiedMemoryArena(uint32_t num_envs, uint32_t max_orders_per_env, std::size_t linear_bytes, uint32_t num_agents,
                       uint32_t obs_dim, uint32_t action_dim);

    UnifiedMemoryArena(const UnifiedMemoryArena&) = delete;
    UnifiedMemoryArena& operator=(const UnifiedMemoryArena&) = delete;

    // --- LOB Interfaces ---
    [[nodiscard]] inline OrderPoolAllocator& get_pool(uint32_t env_id) noexcept { return pools_[env_id]; }
    [[nodiscard]] inline LinearAllocator& get_linear_allocator() noexcept { return linear_allocator_; }

    [[nodiscard]] inline uint32_t num_envs() const noexcept { return num_envs_; }
    [[nodiscard]] inline uint32_t max_orders() const noexcept { return max_orders_per_env_; }

    // --- Bridge Interfaces (Python <-> C++) ---
    [[nodiscard]] inline uint32_t num_agents() const noexcept { return num_agents_; }
    [[nodiscard]] inline uint32_t obs_dim() const noexcept { return obs_dim_; }
    [[nodiscard]] inline uint32_t action_dim() const noexcept { return action_dim_; }

    [[nodiscard]] inline float* obs_ptr() noexcept { return observation_tensor_.data(); }
    [[nodiscard]] inline float* action_ptr() noexcept { return action_tensor_.data(); }
    [[nodiscard]] inline int8_t* mask_ptr() noexcept { return ready_mask_.data(); }

    // --- Synchronization ---
    inline void release_to_python() noexcept { state_.store(1, std::memory_order_release); }

    inline void wait_for_python() noexcept {
        while (state_.load(std::memory_order_acquire) != 0) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();  // Prevent pipeline flushes and save L1 cache
#endif
        }
    }

    inline void release_to_cpp() noexcept { state_.store(0, std::memory_order_release); }

    void reset() noexcept;
};

}  // namespace titan::core