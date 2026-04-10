#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// 1. THE PAYLOAD (Data)
// Size: Exactly 32 bytes (Half of a 64-byte x86_64 Cache Line).
// This struct stays stationary in memory (Pool Allocator) and is never
// copied during event sorting.
// ============================================================================
struct alignas(32) ActionPayload {
    uint32_t agent_id;
    uint8_t action_type;  // 0: Limit Order, 1: Cancel, 2: Market Order
    uint8_t side;         // 0: Bid, 1: Ask
    uint8_t _padding[2];  // Explicit padding to enforce 4-byte boundary for the next fields
    Price price;
    OrderQty qty;
    OrderId target_id;  // Required for Cancel operations
};

// ============================================================================
// 2. THE ROUTING KEY (Metadata)
// Size: Exactly 16 bytes.
// Four of these fit perfectly into one 64-byte Cache Line for SIMD scanning.
// ============================================================================
struct HeapNode {
    uint64_t arrival_time;  // Primary sorting key
    uint32_t payload_idx;   // Pointer to the ActionPayload in the Memory Arena
    uint32_t _padding;      // Explicit padding to reach 16 bytes
};

// ============================================================================
// 3. FAST 4-ARY HEAP
// O(log4 N) Priority Queue optimized for L1/L2 Cache Locality.
// Retrieves the EARLIEST arrival_time (Min-Heap).
// ============================================================================
class Fast4AryHeap {
private:
    std::vector<HeapNode> data_;
    std::size_t max_capacity_;

    inline void sift_up(std::size_t idx) noexcept {
        HeapNode node = data_[idx];
        while (idx > 0) {
            std::size_t parent = (idx - 1) / 4;

            if (data_[parent].arrival_time <= node.arrival_time) {
                break;
            }

            data_[idx] = data_[parent];
            idx = parent;
        }
        data_[idx] = node;
    }

    inline void sift_down(std::size_t idx) noexcept {
        HeapNode node = data_[idx];
        const std::size_t size = data_.size();

        while (true) {
            std::size_t first_child = 4 * idx + 1;
            if (first_child >= size) {
                break;
            }

            std::size_t min_child = first_child;
            uint64_t min_time = data_[first_child].arrival_time;

            // Find the minimum among up to 4 children
            // Modern compilers will unroll this loop and use SIMD / conditional moves
            for (std::size_t i = 1; i < 4; ++i) {
                std::size_t child = first_child + i;
                if (child < size && data_[child].arrival_time < min_time) {
                    min_time = data_[child].arrival_time;
                    min_child = child;
                }
            }

            if (node.arrival_time <= min_time) {
                break;
            }

            data_[idx] = data_[min_child];
            idx = min_child;
        }
        data_[idx] = node;
    }

public:
    explicit Fast4AryHeap(std::size_t max_capacity) : max_capacity_(max_capacity) { data_.reserve(max_capacity_); }

    [[nodiscard]] inline bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] inline std::size_t size() const noexcept { return data_.size(); }

    [[nodiscard]] inline std::size_t capacity() const noexcept { return max_capacity_; }

    [[nodiscard]] inline const HeapNode& top() const noexcept { return data_.front(); }

    inline void push(uint64_t time, uint32_t payload_idx) {
        // Strict protection against reallocation during active simulation
        if (data_.size() >= max_capacity_) [[unlikely]] {
            throw std::runtime_error(
                "Fast4AryHeap Overflow: Max capacity reached. Reallocation is strictly forbidden.");
        }

        data_.push_back({time, payload_idx, 0});
        sift_up(data_.size() - 1);
    }

    inline void pop() noexcept {
        if (data_.empty()) {
            return;
        }

        data_.front() = data_.back();
        data_.pop_back();

        if (!data_.empty()) {
            sift_down(0);
        }
    }

    // Fast reset between training epochs (keeps reserved capacity)
    inline void clear() noexcept { data_.clear(); }
};

}  // namespace titan::core