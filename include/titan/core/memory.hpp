#pragma once

#include <immintrin.h>

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

    [[nodiscard]] inline std::size_t size() const noexcept { return head_; }
};

// ========================================================================
// UnifiedMemoryArena
// Master owner of all simulation memory allocations.
// ========================================================================
class UnifiedMemoryArena {
private:
    uint32_t num_envs_;
    uint32_t max_orders_per_env_;

    std::vector<OrderNode> raw_nodes_;
    std::vector<Handle> raw_free_lists_;
    std::vector<OrderPoolAllocator> pools_;

    LinearAllocator linear_allocator_;

public:
    UnifiedMemoryArena(uint32_t num_envs, uint32_t max_orders_per_env, std::size_t linear_bytes);

    UnifiedMemoryArena(const UnifiedMemoryArena&) = delete;
    UnifiedMemoryArena& operator=(const UnifiedMemoryArena&) = delete;

    [[nodiscard]] inline OrderPoolAllocator& get_pool(uint32_t env_id) noexcept { return pools_[env_id]; }
    [[nodiscard]] inline LinearAllocator& get_linear_allocator() noexcept { return linear_allocator_; }

    [[nodiscard]] inline uint32_t num_envs() const noexcept { return num_envs_; }
    [[nodiscard]] inline uint32_t max_orders() const noexcept { return max_orders_per_env_; }
};

}  // namespace titan::core