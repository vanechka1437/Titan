#pragma once

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

// ============================================================================
// SYSTEM HARDWARE FALLBACK CONFIGURATIONS
// These macros are overridden by CMake during compilation based on the target
// CPU architecture. If undefined, we fallback to a safe default (32KB L1).
// ============================================================================
#ifndef TITAN_SYSTEM_L1_BYTES
#define TITAN_SYSTEM_L1_BYTES (32ull * 1024)  // 32 KB L1 Data Cache
#endif

// ============================================================================
// CROSS-PLATFORM HARDWARE BIT MANIPULATION HELPERS
// Ensures MSVC doesn't hoist loops causing infinite stalls during mask scanning.
// Utilizes hardware instructions (BSR/BSF or lzcnt/tzcnt) for O(1) execution.
// ============================================================================
#ifdef _MSC_VER
#include <intrin.h>
inline uint32_t pop_msb(uint64_t& mask) noexcept {
    unsigned long index;
    _BitScanReverse64(&index, mask);
    mask &= ~(1ULL << index);  // Forcefully clear the bit to break MSVC cache
    return static_cast<uint32_t>(index);
}
inline uint32_t pop_lsb(uint64_t& mask) noexcept {
    unsigned long index;
    _BitScanForward64(&index, mask);
    mask &= (mask - 1);  // Kernighan's trick to clear the least significant bit
    return static_cast<uint32_t>(index);
}
#else
inline uint32_t pop_msb(uint64_t& mask) noexcept {
    uint32_t index = 63 - std::countl_zero(mask);
    mask &= ~(1ULL << index);
    return index;
}
inline uint32_t pop_lsb(uint64_t& mask) noexcept {
    uint32_t index = std::countr_zero(mask);
    mask &= (mask - 1);
    return index;
}
#endif

// Core engine definitions (Provides Price, OrderId, Handle, DefaultEventBuffer)
#include "titan/core/matching_engine.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// Compile-Time Hardware Optimization for ShadowLOB
// ============================================================================
namespace detail {
// 8 bytes per price level (bid_qty + ask_qty), ignoring bitmask overhead
constexpr size_t SHADOW_TARGET_BYTES = TITAN_SYSTEM_L1_BYTES;
constexpr size_t SHADOW_TARGET_LEVELS = SHADOW_TARGET_BYTES / 8;
constexpr size_t OPTIMAL_SHADOW_WINDOW = std::bit_floor(SHADOW_TARGET_LEVELS);
constexpr size_t FINAL_SHADOW_WINDOW = std::clamp<size_t>(OPTIMAL_SHADOW_WINDOW, 1024, 16384);
}  // namespace detail

// ============================================================================
// SHADOW LOB: Flat Array Bitmask Hierarchy for RL Agents
// Uses a Sliding Window (Flat Array) to prevent wrap-around parsing logic
// during tensor materialization, guaranteeing 100% correct sorting for RL.
// Fits perfectly into the L1d cache.
// ============================================================================
template <uint32_t Depth = 20, uint32_t WindowSize = detail::FINAL_SHADOW_WINDOW>
class ShadowLOB {
    static_assert((WindowSize & (WindowSize - 1)) == 0, "WindowSize must be a power of 2");

private:
    static constexpr uint32_t L1_SIZE = WindowSize / 64;
    static constexpr uint32_t L2_SIZE = (L1_SIZE + 63) / 64;
    static constexpr Price UNINITIALIZED_ANCHOR = 0xFFFFFFFF;

    // Bitmask hierarchy for Bids
    uint64_t bid_l1_[L1_SIZE]{0};
    uint64_t bid_l2_[L2_SIZE]{0};
    int32_t bid_qty_[WindowSize]{0};

    // Bitmask hierarchy for Asks
    uint64_t ask_l1_[L1_SIZE]{0};
    uint64_t ask_l2_[L2_SIZE]{0};
    int32_t ask_qty_[WindowSize]{0};

    Price anchor_price_{UNINITIALIZED_ANCHOR};

    // ========================================================================
    // Re-centering Operation (Rare event, O(N) execution time)
    // Shifts the memory window to center the new price, preserving structure.
    // ========================================================================
    inline void recenter(Price target_price) noexcept {
        Price new_anchor = (target_price > WindowSize / 2) ? (target_price - WindowSize / 2) : 0;

        if (anchor_price_ == UNINITIALIZED_ANCHOR) {
            anchor_price_ = new_anchor;
            return;
        }

        int64_t offset = static_cast<int64_t>(new_anchor) - static_cast<int64_t>(anchor_price_);

        if (std::abs(offset) >= static_cast<int64_t>(WindowSize)) {
            clear_arrays();
        } else if (offset > 0) {
            // Shift left (market going up)
            std::memmove(bid_qty_, bid_qty_ + offset, (WindowSize - offset) * sizeof(int32_t));
            std::memmove(ask_qty_, ask_qty_ + offset, (WindowSize - offset) * sizeof(int32_t));
            std::memset(bid_qty_ + (WindowSize - offset), 0, offset * sizeof(int32_t));
            std::memset(ask_qty_ + (WindowSize - offset), 0, offset * sizeof(int32_t));
        } else {
            // Shift right (market going down)
            int64_t shift = -offset;
            std::memmove(bid_qty_ + shift, bid_qty_, (WindowSize - shift) * sizeof(int32_t));
            std::memmove(ask_qty_ + shift, ask_qty_, (WindowSize - shift) * sizeof(int32_t));
            std::memset(bid_qty_, 0, shift * sizeof(int32_t));
            std::memset(ask_qty_, 0, shift * sizeof(int32_t));
        }

        anchor_price_ = new_anchor;
        rebuild_bitmasks();
    }

    // O(N) reconstruction, highly optimized for L1 cache bounds
    inline void rebuild_bitmasks() noexcept {
        std::memset(bid_l1_, 0, sizeof(bid_l1_));
        std::memset(bid_l2_, 0, sizeof(bid_l2_));
        std::memset(ask_l1_, 0, sizeof(ask_l1_));
        std::memset(ask_l2_, 0, sizeof(ask_l2_));

        for (uint32_t idx = 0; idx < WindowSize; ++idx) {
            if (bid_qty_[idx] > 0) {
                uint32_t l1_idx = idx >> 6;
                bid_l1_[l1_idx] |= (1ULL << (idx & 63));
                bid_l2_[l1_idx >> 6] |= (1ULL << (l1_idx & 63));
            }
            if (ask_qty_[idx] > 0) {
                uint32_t l1_idx = idx >> 6;
                ask_l1_[l1_idx] |= (1ULL << (idx & 63));
                ask_l2_[l1_idx >> 6] |= (1ULL << (l1_idx & 63));
            }
        }
    }

    inline void clear_arrays() noexcept {
        std::memset(bid_l1_, 0, sizeof(bid_l1_));
        std::memset(bid_l2_, 0, sizeof(bid_l2_));
        std::memset(bid_qty_, 0, sizeof(bid_qty_));
        std::memset(ask_l1_, 0, sizeof(ask_l1_));
        std::memset(ask_l2_, 0, sizeof(ask_l2_));
        std::memset(ask_qty_, 0, sizeof(ask_qty_));
    }

public:
    ShadowLOB() = default;

    // ========================================================================
    // WRITE PATH (Hot Path driven by the Engine Event Dispatcher)
    // Executes in O(1) time (~1 CPU cycle) unless a re-centering is triggered.
    // ========================================================================
    inline void apply_delta(uint8_t side, Price price, int32_t qty_delta) noexcept {
        if (anchor_price_ == UNINITIALIZED_ANCHOR || price < anchor_price_ || price >= anchor_price_ + WindowSize) {
            recenter(price);
        }

        uint32_t idx = price - anchor_price_;
        uint32_t l1_idx = idx >> 6;
        uint32_t l2_idx = l1_idx >> 6;

        if (side == 0) {  // BID
            bid_qty_[idx] += qty_delta;
            if (bid_qty_[idx] > 0) {
                bid_l1_[l1_idx] |= (1ULL << (idx & 63));
                bid_l2_[l2_idx] |= (1ULL << (l1_idx & 63));
            } else {
                bid_l1_[l1_idx] &= ~(1ULL << (idx & 63));
                if (bid_l1_[l1_idx] == 0)
                    bid_l2_[l2_idx] &= ~(1ULL << (l1_idx & 63));
            }
        } else {  // ASK
            ask_qty_[idx] += qty_delta;
            if (ask_qty_[idx] > 0) {
                ask_l1_[l1_idx] |= (1ULL << (idx & 63));
                ask_l2_[l2_idx] |= (1ULL << (l1_idx & 63));
            } else {
                ask_l1_[l1_idx] &= ~(1ULL << (idx & 63));
                if (ask_l1_[l1_idx] == 0)
                    ask_l2_[l2_idx] &= ~(1ULL << (l1_idx & 63));
            }
        }
    }

    // ========================================================================
    // READ PATH (Synchronous Barrier Export to PyTorch)
    // Employs hardware instruction sets to locate active prices from registers.
    // Guarantees mathematically strictly ordered tensors for RL models.
    // ========================================================================
    inline void export_to_tensor(float* obs_ptr) const noexcept {
        uint32_t offset = 0;
        uint32_t count = 0;

        // Bids: Scan MSB to LSB (Highest price downwards)
        for (int32_t l2 = L2_SIZE - 1; l2 >= 0 && count < Depth; --l2) {
            uint64_t mask2 = bid_l2_[l2];
            while (mask2 && count < Depth) {
                uint32_t bit2 = pop_msb(mask2);
                uint32_t l1_idx = (l2 << 6) + bit2;
                uint64_t mask1 = bid_l1_[l1_idx];

                while (mask1 && count < Depth) {
                    uint32_t bit1 = pop_msb(mask1);
                    uint32_t idx = (l1_idx << 6) + bit1;

                    obs_ptr[offset++] = static_cast<float>(anchor_price_ + idx);
                    obs_ptr[offset++] = static_cast<float>(bid_qty_[idx]);
                    count++;
                }
            }
        }
        // Zero-padding for missing bid levels
        while (count++ < Depth) {
            obs_ptr[offset++] = 0.0f;
            obs_ptr[offset++] = 0.0f;
        }

        // Asks: Scan LSB to MSB (Lowest price upwards)
        count = 0;
        for (uint32_t l2 = 0; l2 < L2_SIZE && count < Depth; ++l2) {
            uint64_t mask2 = ask_l2_[l2];
            while (mask2 && count < Depth) {
                uint32_t bit2 = pop_lsb(mask2);
                uint32_t l1_idx = (l2 << 6) + bit2;
                uint64_t mask1 = ask_l1_[l1_idx];

                while (mask1 && count < Depth) {
                    uint32_t bit1 = pop_lsb(mask1);
                    uint32_t idx = (l1_idx << 6) + bit1;

                    obs_ptr[offset++] = static_cast<float>(anchor_price_ + idx);
                    obs_ptr[offset++] = static_cast<float>(ask_qty_[idx]);
                    count++;
                }
            }
        }
        // Zero-padding for missing ask levels
        while (count++ < Depth) {
            obs_ptr[offset++] = 0.0f;
            obs_ptr[offset++] = 0.0f;
        }
    }

    inline void clear() noexcept {
        clear_arrays();
        anchor_price_ = UNINITIALIZED_ANCHOR;
    }
};

// ============================================================================
// 2. AGENT STATE
// Represents an isolated actor within the market environment.
// ============================================================================
template <uint32_t ObsDepth = 20>
class AgentState {
public:
    uint32_t id{0};

    // --- Network Physics Parameters ---
    uint64_t ingress_delay{0};     // Latency: Agent -> Exchange
    uint64_t egress_delay{0};      // Latency: Exchange -> Agent
    uint64_t compute_delay{0};     // Agent's inference/decision latency
    uint64_t next_wakeup_time{0};  // Absolute engine epoch for next allowed action

    // --- Local World View ---
    ShadowLOB<ObsDepth> shadow_lob;

    // --- Zero-Copy Arena Pointers ---
    // Direct memory mapping to PyTorch tensors for instantaneous observation reads
    float* obs_cash_ptr{nullptr};
    float* obs_inventory_ptr{nullptr};
    float* obs_pnl_ptr{nullptr};
    float* obs_event_stream_ptr{nullptr};

    // --- Precise Accounting ---
    int64_t real_cash{0};
    int32_t real_inventory{0};

    explicit AgentState() = default;

    inline void update_balance(int64_t cash_delta, int32_t inventory_delta) noexcept {
        real_cash += cash_delta;
        real_inventory += inventory_delta;

        if (obs_cash_ptr)
            *obs_cash_ptr = static_cast<float>(real_cash);
        if (obs_inventory_ptr)
            *obs_inventory_ptr = static_cast<float>(real_inventory);
    }

    // ========================================================================
    // Event Stream Zero-Copy Materialization
    // Copies centralized environment events into the agent's tensor slice.
    // Fills remaining buffer space with zero-padding to maintain matrix shape.
    // ========================================================================
    template <typename EventBuffer>
    inline void export_events_to_tensor(const EventBuffer& event_buffer, uint32_t max_events) const noexcept {
        if (!obs_event_stream_ptr)
            return;

        uint32_t buffer_size = static_cast<uint32_t>(event_buffer.size());
        uint32_t count = std::min(buffer_size, max_events);
        uint32_t offset = 0;

        uint32_t start_idx = (buffer_size > max_events) ? (buffer_size - max_events) : 0;

        for (uint32_t i = 0; i < count; ++i) {
            const auto& ev = event_buffer[start_idx + i];
            obs_event_stream_ptr[offset++] = static_cast<float>(ev.timestamp);
            obs_event_stream_ptr[offset++] = static_cast<float>(ev.action_type);
            obs_event_stream_ptr[offset++] = static_cast<float>(ev.price);
            obs_event_stream_ptr[offset++] = static_cast<float>(ev.qty);
        }

        uint32_t remaining_floats = (max_events - count) * 4;
        if (remaining_floats > 0) {
            std::memset(obs_event_stream_ptr + offset, 0, remaining_floats * sizeof(float));
        }
    }

    inline void reset() noexcept {
        next_wakeup_time = 0;
        real_cash = 0;
        real_inventory = 0;
        shadow_lob.clear();
    }
};

// ============================================================================
// 3. ENVIRONMENT STATE
// The global simulation sandbox tracking physics and causality.
// ============================================================================
template <uint32_t ObsDepth = 20>
class EnvironmentState {
public:
    uint32_t env_id;

    uint64_t current_time{0};  // Shared environment nanosecond clock

    std::vector<AgentState<ObsDepth>> agents;

    // Centralized event buffer allocated safely in heap to prevent stack overflow
    DefaultEventBuffer event_buffer;

    MatchingEngine* engine = nullptr;

    inline void reset() noexcept {
        current_time = 0;
        event_buffer.clear();
        for (auto& agent : agents) {
            agent.reset();
        }
    }
};

}  // namespace titan::core