#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <map>

#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

// Fallback configuration if CMake fails to provide the hardware profile
#ifndef TITAN_SYSTEM_L2_BYTES
#define TITAN_SYSTEM_L2_BYTES (1024ull * 1024)  // 1 MB (Typical L2 cache per core)
#endif

#include <absl/container/btree_map.h>
template <typename K, typename V, typename Compare = std::less<K>>
using ColdZoneMap = absl::btree_map<K, V, Compare>;

namespace titan::core {

// ============================================================================
// Price Level: Cache-aligned (32 bytes) node for the Limit Order Book
// Exactly 2 PriceLevels fit into a standard 64-byte CPU cache line.
// ============================================================================
struct alignas(32) PriceLevel {
    // --- 8-byte fields (8 bytes) ---
    OrderQty total_qty{0};

    // --- 4-byte fields (12 bytes) ---
    Price actual_price{0};  // Tag for O(1) lazy clearing during ring collisions
    Handle head{NULL_HANDLE};
    Handle tail{NULL_HANDLE};

    // --- Padding ---
    // Total used: 20 bytes. Padding required for 32-byte alignment: 12 bytes.
    uint8_t _padding[12]{0};
};
static_assert(sizeof(PriceLevel) == 32, "PriceLevel must be exactly 32 bytes for cache alignment");

// ============================================================================
// Compile-Time Hardware Optimization (Zero Runtime Overhead)
// ============================================================================

namespace detail {
// 1. Allocate 1/4th of the L2 Cache to the Hot Zone
// (This ensures the ring buffer stays ultra-fast and doesn't pollute the entire cache)
constexpr size_t TARGET_MEMORY_BYTES = TITAN_SYSTEM_L2_BYTES / 4;

// 2. Calculate capacity based on struct size (32 bytes)
// Example: 256 KB / 32 bytes = 8192 levels
constexpr size_t TARGET_LEVELS = TARGET_MEMORY_BYTES / sizeof(PriceLevel);

// 3. Floor to nearest power of 2 for fast bitwise arithmetic (price & RING_MASK)
constexpr size_t OPTIMAL_RING = std::bit_floor(TARGET_LEVELS);

// 4. Safety bounds: Min 1024 (L1 Cache limit), Max 65536
constexpr size_t FINAL_RING_SIZE = std::clamp<size_t>(OPTIMAL_RING, 1024, 65536);
}  // namespace detail

// ============================================================================
// Limit Order Book: Hybrid Architecture (O(1) Ring Buffer + O(log N) Overflow)
// ============================================================================
template <uint32_t RingSize = detail::FINAL_RING_SIZE>
class LOBState final {
public:
    static constexpr uint32_t RING_SIZE = RingSize;

private:
    // Guarantees fast bitwise modulo operations (price & RING_MASK)
    static_assert(std::has_single_bit(RingSize), "RingSize MUST be a power of 2!");

    static constexpr uint32_t RING_MASK = RingSize - 1;
    static constexpr uint32_t L1_SIZE = RingSize / 64;
    static constexpr uint32_t L2_SIZE = (L1_SIZE + 63) / 64; // Safety padding for small rings

    // --- Hot Zone: L2/L3 Cache Resident ---
    PriceLevel hot_levels_[RingSize];

    // Separated bid/ask masks prevent pipeline stalls during parallel hardware scans
    uint64_t l1_mask_bids_[L1_SIZE]{0};
    uint64_t l2_mask_bids_[L2_SIZE]{0};

    uint64_t l1_mask_asks_[L1_SIZE]{0};
    uint64_t l2_mask_asks_[L2_SIZE]{0};

    Price anchor_price_{0};  // Base price for sliding window offset

    // --- Cold Zone: Overflow Trees ---
    ColdZoneMap<Price, PriceLevel> cold_bids_;
    ColdZoneMap<Price, PriceLevel> cold_asks_;

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

    LOBState(LOBState&&) noexcept = default;
    LOBState& operator=(LOBState&&) noexcept = default;

    // --- Public API ---

    // Core insertion. Returns the Smart OrderId (Generation + Handle).
    OrderId add_order(OwnerId owner_id, Price price, OrderQty qty, uint8_t side, OrderPoolAllocator& pool);

    // O(1) extraction of an order using its memory handle.
    void remove_order(Handle h, OrderPoolAllocator& pool) noexcept;

    // Fast retrieval of the first order at a specific price level (for spread crossing).
    [[nodiscard]] Handle get_first_order(uint8_t side, Price price) const noexcept;

    [[nodiscard]] Price get_best_bid() const noexcept;
    [[nodiscard]] Price get_best_ask() const noexcept;

    void shift_window_to_center(Price target_price) noexcept;

    [[nodiscard]] Price get_anchor_price() const noexcept { return anchor_price_; }

    void reduce_level_qty(uint8_t side, Price price, OrderQty trade_qty) noexcept;

    // Instant reset for RL episodes
    void reset() noexcept;
};

// ============================================================================
// Type Alias for the automatically hardware-tuned LOB
// ============================================================================
using OptimalLOBState = LOBState<detail::FINAL_RING_SIZE>;

}  // namespace titan::core