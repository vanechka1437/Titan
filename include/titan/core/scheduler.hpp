#pragma once

#include <immintrin.h>  // Required for _mm_malloc

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// 1. THE PAYLOAD (Data)
// Size: Exactly 32 bytes (Perfectly occupies half of a 64-byte Cache Line).
// This struct is stored in a stationary Pool Allocator to prevent cache churn.
// ============================================================================
struct alignas(32) ActionPayload {
    uint32_t agent_id;
    uint8_t action_type;  // 0: Limit Order, 1: Cancel, 2: Market Order
    uint8_t side;         // 0: Bid, 1: Ask
    uint8_t _padding[2];  // Explicit padding for 4-byte alignment
    Price price;
    OrderQty qty;
    OrderId target_id;  // Used for Cancel operations
};

// ============================================================================
// 2. THE ROUTING KEY (Metadata)
// Size: Exactly 16 bytes.
// Four nodes fit into a single 64-byte cache line for massive SIMD throughput.
// ============================================================================
struct HeapNode {
    uint64_t arrival_time;  // Primary Min-Heap sorting key
    uint32_t payload_idx;   // Index into the ActionPayload Arena
    uint32_t _padding;      // Padding to reach 16-byte power-of-two size
};

// ============================================================================
// 3. FAST D-ARY HEAP (Templated)
// O(logD N) Priority Queue.
// Recommended Arity: 8 for modern CPUs (exploits 128-bit spatial prefetchers).
// ============================================================================
template <uint32_t Arity>
class FastDAryHeap {
    static_assert(Arity >= 2, "Arity must be at least 2");
    static_assert((Arity & (Arity - 1)) == 0, "Arity must be a power of 2 for optimal indexing");

private:
    // Aligned to 64 bytes to prevent "Cache Line Splits" during child scans
    alignas(64) HeapNode* data_;
    std::size_t size_{0};
    std::size_t max_capacity_;

    /**
     * @brief Tournament-style branchless min-search.
     * Compiler unrolls this entirely based on the Arity template parameter.
     */
    inline std::size_t find_best_child(std::size_t first_child) const noexcept {
        if constexpr (Arity == 8) {
            // Level 1: 4 pairs
            std::size_t a = (data_[first_child + 0].arrival_time <= data_[first_child + 1].arrival_time)
                                ? first_child + 0
                                : first_child + 1;
            std::size_t b = (data_[first_child + 2].arrival_time <= data_[first_child + 3].arrival_time)
                                ? first_child + 2
                                : first_child + 3;
            std::size_t c = (data_[first_child + 4].arrival_time <= data_[first_child + 5].arrival_time)
                                ? first_child + 4
                                : first_child + 5;
            std::size_t d = (data_[first_child + 6].arrival_time <= data_[first_child + 7].arrival_time)
                                ? first_child + 6
                                : first_child + 7;
            // Level 2: 2 pairs
            std::size_t e = (data_[a].arrival_time <= data_[b].arrival_time) ? a : b;
            std::size_t f = (data_[c].arrival_time <= data_[d].arrival_time) ? c : d;
            // Level 3: Final winner
            return (data_[e].arrival_time <= data_[f].arrival_time) ? e : f;
        } else if constexpr (Arity == 4) {
            std::size_t a = (data_[first_child + 0].arrival_time <= data_[first_child + 1].arrival_time)
                                ? first_child + 0
                                : first_child + 1;
            std::size_t b = (data_[first_child + 2].arrival_time <= data_[first_child + 3].arrival_time)
                                ? first_child + 2
                                : first_child + 3;
            return (data_[a].arrival_time <= data_[b].arrival_time) ? a : b;
        } else {
            // Generic fallback for other arities
            std::size_t best = first_child;
            for (std::size_t i = 1; i < Arity; ++i) {
                if (data_[first_child + i].arrival_time < data_[best].arrival_time) {
                    best = first_child + i;
                }
            }
            return best;
        }
    }

    inline void sift_up(std::size_t idx) noexcept {
        HeapNode node = data_[idx];
        while (idx > 0) {
            std::size_t parent = (idx - 1) / Arity;
            if (data_[parent].arrival_time <= node.arrival_time)
                break;
            data_[idx] = data_[parent];
            idx = parent;
        }
        data_[idx] = node;
    }

    inline void sift_down(std::size_t idx) noexcept {
        HeapNode node = data_[idx];
        const uint64_t target_time = node.arrival_time;

        while (true) {
            const std::size_t first_child = Arity * idx + 1;
            if (first_child >= size_)
                break;

            // Tournament-style branchless selection
            std::size_t best = find_best_child(first_child);

            if (target_time <= data_[best].arrival_time)
                break;

            data_[idx] = data_[best];
            idx = best;
        }
        data_[idx] = node;
    }

public:
    explicit FastDAryHeap(std::size_t max_capacity) : max_capacity_(max_capacity) {
        // Allocate space + Arity-sized buffer for Sentinels (prevents bounds checks)
        std::size_t alloc_size = max_capacity_ + Arity;
        data_ = static_cast<HeapNode*>(_mm_malloc(alloc_size * sizeof(HeapNode), 64));

        if (!data_)
            throw std::bad_alloc();
        clear();
    }

    ~FastDAryHeap() {
        if (data_)
            _mm_free(data_);
    }

    // Standard HFT practice: Non-copyable to ensure strict ownership of the arena
    FastDAryHeap(const FastDAryHeap&) = delete;
    FastDAryHeap& operator=(const FastDAryHeap&) = delete;

    [[nodiscard]] inline bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] inline std::size_t size() const noexcept { return size_; }
    [[nodiscard]] inline const HeapNode& top() const noexcept { return data_[0]; }

    inline void push(uint64_t time, uint32_t payload_idx) {
        if (size_ >= max_capacity_) [[unlikely]] {
            throw std::runtime_error("FastDAryHeap Overflow");
        }

        const std::size_t idx = size_++;
        data_[idx] = {time, payload_idx, 0};

        // Maintain Sentinel Zone: ensures sift_down never compares against garbage memory.
        // The next Arity elements are always set to infinity.
        for (uint32_t i = 0; i < Arity; ++i) {
            data_[size_ + i].arrival_time = 0xFFFFFFFFFFFFFFFFULL;
        }

        sift_up(idx);
    }

    inline void pop() noexcept {
        if (size_ == 0) [[unlikely]]
            return;

        data_[0] = data_[--size_];
        data_[size_].arrival_time = 0xFFFFFFFFFFFFFFFFULL;

        if (size_ > 0) {
            sift_down(0);
        }
    }

    inline void clear() noexcept {
        size_ = 0;
        // Initialize the first two cache lines of sentinels
        for (uint32_t i = 0; i < Arity * 2; ++i) {
            data_[i].arrival_time = 0xFFFFFFFFFFFFFFFFULL;
        }
    }
};

// Default Projekt-wide Scheduler tuned for 64-byte cache prefetchers
using FastScheduler = FastDAryHeap<4>;

}  // namespace titan::core