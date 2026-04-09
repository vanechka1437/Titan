#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>

#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

// Fallback configuration if CMake fails to provide the hardware profile
#ifndef TITAN_SYSTEM_L3_BYTES
#define TITAN_SYSTEM_L3_BYTES (16ull * 1024 * 1024)  // 16 MB
#endif

namespace titan::core {

using Price = uint32_t;
using Quantity = uint32_t;
using OrderId = uint64_t;

// ============================================================================
// Price Level: Cache-aligned (16 bytes) node for the Limit Order Book
// ============================================================================
struct alignas(16) PriceLevel {
    Handle head{NULL_HANDLE};
    Handle tail{NULL_HANDLE};
    Quantity total_qty{0};
    Price actual_price{0};  // Tag for O(1) lazy clearing during ring collisions
};

// ============================================================================
// Compile-Time Hardware Optimization (Zero Runtime Overhead)
// ============================================================================
namespace detail {
// 1. Allocate 1/8th of the L3 Cache to the Hot Zone
constexpr size_t TARGET_MEMORY_BYTES = TITAN_SYSTEM_L3_BYTES / 8;

// 2. Calculate capacity based on struct size
constexpr size_t TARGET_LEVELS = TARGET_MEMORY_BYTES / sizeof(PriceLevel);

// 3. Floor to nearest power of 2 for fast bitwise arithmetic
constexpr size_t OPTIMAL_RING = std::bit_floor(TARGET_LEVELS);

// 4. Safety bounds: Min 256 KB (L2 Cache), Max 8 MB
constexpr size_t FINAL_RING_SIZE = std::clamp<size_t>(OPTIMAL_RING, 16384, 524288);
}  // namespace detail

// ============================================================================
// Limit Order Book: Hybrid Architecture (O(1) Ring Buffer + O(log N) Overflow)
// ============================================================================
template <uint32_t RingSize = detail::FINAL_RING_SIZE>
class LOBState final {
private:
    // Guarantees fast bitwise modulo operations (price & RING_MASK)
    static_assert(std::has_single_bit(RingSize), "RingSize MUST be a power of 2!");

    static constexpr uint32_t RING_MASK = RingSize - 1;
    static constexpr uint32_t L1_SIZE = RingSize / 64;
    static constexpr uint32_t L2_SIZE = L1_SIZE / 64;

    // --- Hot Zone: L2/L3 Cache Resident ---
    PriceLevel hot_levels_[RingSize];

    // Separated bid/ask masks prevent pipeline stalls during parallel hardware scans
    uint64_t l1_mask_bids_[L1_SIZE]{0};
    uint64_t l2_mask_bids_[L2_SIZE]{0};

    uint64_t l1_mask_asks_[L1_SIZE]{0};
    uint64_t l2_mask_asks_[L2_SIZE]{0};

    Price anchor_price_{0};  // Base price for sliding window offset

    // --- Cold Zone: Overflow Trees ---
    std::map<Price, PriceLevel> cold_bids_;
    std::map<Price, PriceLevel> cold_asks_;

    // --- Internal Helpers ---

    [[nodiscard]] inline uint32_t get_index(Price price) const noexcept { return price & RING_MASK; }

    [[nodiscard]] inline bool is_hot(Price price) const noexcept {
        // Underflow arithmetic: validates if price is within [anchor, anchor + RingSize)
        return (price - anchor_price_) < RingSize;
    }

    [[nodiscard]] inline PriceLevel& get_level_for_write(Price target_price) noexcept {
        PriceLevel& level = hot_levels_[get_index(target_price)];

        // O(1) Lazy clear: resolves phantom data from previous ring cycles
        if (level.actual_price != target_price) {
            level.head = NULL_HANDLE;
            level.tail = NULL_HANDLE;
            level.total_qty = 0;
            level.actual_price = target_price;
        }
        return level;
    }

    // --- Bitmask Operations ---

    inline void set_active_bid(Price price) noexcept {
        const uint32_t idx = get_index(price);
        const uint32_t l1_idx = idx >> 6;
        l1_mask_bids_[l1_idx] |= (1ULL << (idx & 63));
        l2_mask_bids_[l1_idx >> 6] |= (1ULL << (l1_idx & 63));
    }

    inline void set_active_ask(Price price) noexcept {
        const uint32_t idx = get_index(price);
        const uint32_t l1_idx = idx >> 6;
        l1_mask_asks_[l1_idx] |= (1ULL << (idx & 63));
        l2_mask_asks_[l1_idx >> 6] |= (1ULL << (l1_idx & 63));
    }

    inline void set_empty_bid(Price price) noexcept {
        const uint32_t idx = get_index(price);
        const uint32_t l1_idx = idx >> 6;

        l1_mask_bids_[l1_idx] &= ~(1ULL << (idx & 63));
        if (l1_mask_bids_[l1_idx] == 0) {
            l2_mask_bids_[l1_idx >> 6] &= ~(1ULL << (l1_idx & 63));
        }
    }

    inline void set_empty_ask(Price price) noexcept {
        const uint32_t idx = get_index(price);
        const uint32_t l1_idx = idx >> 6;

        l1_mask_asks_[l1_idx] &= ~(1ULL << (idx & 63));
        if (l1_mask_asks_[l1_idx] == 0) {
            l2_mask_asks_[l1_idx >> 6] &= ~(1ULL << (l1_idx & 63));
        }
    }

    void shift_window(Price new_anchor) noexcept;

public:
    LOBState() = default;

    // Disabled copying to prevent accidental massive memory allocations
    LOBState(const LOBState&) = delete;
    LOBState& operator=(const LOBState&) = delete;

    // --- Public API ---
    void add_order(OrderId id, Price price, Quantity qty, uint8_t side, OrderPoolAllocator& pool);
    void cancel_order(OrderId id, OrderPoolAllocator& pool);

    [[nodiscard]] Price get_best_bid() const noexcept;
    [[nodiscard]] Price get_best_ask() const noexcept;
};

// ============================================================================
// Type Alias for the automatically hardware-tuned LOB
// ============================================================================
using OptimalLOBState = LOBState<detail::FINAL_RING_SIZE>;

}  // namespace titan::core