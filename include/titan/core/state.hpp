#pragma once

#include <algorithm>
#include <boost/container/flat_map.hpp>
#include <cstdint>
#include <functional>
#include <vector>

#include "titan/core/matching_engine.hpp"
#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// Marker for empty slots in the open-addressing hash table.
// Assumes that a valid price cannot be negative.
// ============================================================================
static constexpr int32_t EMPTY_SLOT = -1;

// ============================================================================
// Shadow Price Level: Cache-aligned (8 bytes) macroscopic view of a level.
// Decoupled from the internal Matching Engine's PriceLevel (16 bytes) to
// strictly exclude intrusive linked-list handles (head/tail). This doubles
// the L1 cache capacity for the RL agent's observation state.
// ============================================================================
struct alignas(8) ShadowPriceLevel {
    int32_t price;
    int32_t qty;
};

/**
 * @brief Lazy Shadow LOB (Log-structured Dirty Buffers Paradigm)
 * * A zero-allocation, cache-friendly aggregated order book designed for Batch RL.
 * It defers the O(N) sorting overhead until the exact moment PyTorch requests
 * the observation tensor, enabling O(1) micro-tick updates during simulation.
 * * @tparam Depth Number of top levels exported to the PyTorch tensor (prevents dimensionality curse).
 * @tparam Capacity Hash table size. Default 4096 yields a 32KB footprint, fitting perfectly in L1d cache.
 */
template <uint32_t Depth = 20, uint32_t Capacity = 4096>
class LazyShadowLOB {
    // Capacity MUST be a power of 2 for the bitwise modulo operator to work
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static constexpr uint32_t MASK = Capacity - 1;

private:
    // Open-addressing hash tables with Linear Probing.
    // Memory footprint: 4096 * 8 bytes = 32 KB per side (Fits in L1/L2 Cache)
    alignas(64) ShadowPriceLevel bids_[Capacity];
    alignas(64) ShadowPriceLevel asks_[Capacity];

    // Lazy materialization flag (avoids computing identical states)
    bool is_dirty_{false};

    // Track unique price levels to monitor the load factor
    uint32_t bid_load_{0};
    uint32_t ask_load_{0};

    /**
     * @brief O(1) Cache-friendly insert or update using Linear Probing.
     * Memory shifts (memmove) are strictly prohibited here to guarantee
     * ultra-low latency on the Matching Engine's hot path.
     */
    inline void insert_or_update(ShadowPriceLevel* table, uint32_t& load, int32_t price, int32_t qty_delta) noexcept {
        // Fast bitwise modulo hashing
        uint32_t idx = static_cast<uint32_t>(price) & MASK;

        while (true) {
            if (table[idx].price == EMPTY_SLOT) {
                // Empty slot found -> Register new price level
                table[idx].price = price;
                table[idx].qty = qty_delta;
                ++load;
                return;
            } else if (table[idx].price == price) {
                // Price level exists -> O(1) in-place update
                table[idx].qty += qty_delta;

                // Note: We deliberately do NOT remove tombstones (qty <= 0) here.
                // Erasing would require shifting memory or complex tombstone handling.
                // We filter out zero-volume levels lazily during the Compaction phase.
                return;
            }
            // Hash collision -> Linear Probing
            idx = (idx + 1) & MASK;
        }
    }

public:
    LazyShadowLOB() { clear(); }

    // ========================================================================
    // WRITE PATH (Hot Path driven by the Matching Engine)
    // Executes millions of times per second (takes ~3-5 CPU cycles).
    // ========================================================================
    inline void apply_delta(uint8_t side, int32_t price, int32_t qty_delta) noexcept {
        if (side == 0) {  // BID
            insert_or_update(bids_, bid_load_, price, qty_delta);
        } else {  // ASK
            insert_or_update(asks_, ask_load_, price, qty_delta);
        }
        is_dirty_ = true;
    }

    // ========================================================================
    // READ PATH (Synchronous Barrier driven by the RL Environment)
    // Executes only when the agent's step() requires a new observation.
    // ========================================================================
    inline void export_to_tensor(float* obs_ptr) noexcept {
        // Asymmetry Magic: Skip sorting entirely if no market events occurred
        if (!is_dirty_)
            return;

        // 1. COMPACTION PHASE (Stack-allocated garbage collection)
        ShadowPriceLevel valid_bids[Capacity];
        ShadowPriceLevel valid_asks[Capacity];
        uint32_t valid_bid_count = 0;
        uint32_t valid_ask_count = 0;

        // Filter out tombstones (qty <= 0) and empty slots
        for (uint32_t i = 0; i < Capacity; ++i) {
            if (bids_[i].price != EMPTY_SLOT && bids_[i].qty > 0) {
                valid_bids[valid_bid_count++] = bids_[i];
            }
            if (asks_[i].price != EMPTY_SLOT && asks_[i].qty > 0) {
                valid_asks[valid_ask_count++] = asks_[i];
            }
        }

        // 2. QUICK-SELECT & SORT PHASE
        uint32_t bid_depth = std::min(Depth, valid_bid_count);
        if (bid_depth > 0) {
            // O(M) Ordinal statistics: Partition the array to push Top-N highest bids to the front
            std::nth_element(valid_bids, valid_bids + bid_depth - 1, valid_bids + valid_bid_count,
                             [](const ShadowPriceLevel& a, const ShadowPriceLevel& b) { return a.price > b.price; });

            // O(N log N) Sort only the isolated Top-N slice
            std::sort(valid_bids, valid_bids + bid_depth,
                      [](const ShadowPriceLevel& a, const ShadowPriceLevel& b) { return a.price > b.price; });
        }

        uint32_t ask_depth = std::min(Depth, valid_ask_count);
        if (ask_depth > 0) {
            // O(M) Partition to push Top-N lowest asks to the front
            std::nth_element(valid_asks, valid_asks + ask_depth - 1, valid_asks + valid_ask_count,
                             [](const ShadowPriceLevel& a, const ShadowPriceLevel& b) { return a.price < b.price; });

            std::sort(valid_asks, valid_asks + ask_depth,
                      [](const ShadowPriceLevel& a, const ShadowPriceLevel& b) { return a.price < b.price; });
        }

        // 3. ZERO-COPY EXPORT PHASE (SIMD / Loop Unrolling enabled by template bounds)
        uint32_t offset = 0;

#pragma GCC unroll 10
        for (uint32_t i = 0; i < Depth; ++i) {
            if (i < bid_depth) {
                obs_ptr[offset++] = static_cast<float>(valid_bids[i].price);
                obs_ptr[offset++] = static_cast<float>(valid_bids[i].qty);
            } else {
                // Padding for shallow order books
                obs_ptr[offset++] = 0.0f;
                obs_ptr[offset++] = 0.0f;
            }
        }

#pragma GCC unroll 10
        for (uint32_t i = 0; i < Depth; ++i) {
            if (i < ask_depth) {
                obs_ptr[offset++] = static_cast<float>(valid_asks[i].price);
                obs_ptr[offset++] = static_cast<float>(valid_asks[i].qty);
            } else {
                obs_ptr[offset++] = 0.0f;
                obs_ptr[offset++] = 0.0f;
            }
        }

        is_dirty_ = false;  // Reset the state flag
    }

    // ========================================================================
    // Fast reset between RL episodes without reallocating memory
    // ========================================================================
    inline void clear() noexcept {
        for (uint32_t i = 0; i < Capacity; ++i) {
            bids_[i].price = EMPTY_SLOT;
            asks_[i].price = EMPTY_SLOT;
        }
        bid_load_ = 0;
        ask_load_ = 0;
        is_dirty_ = false;
    }
};

// ============================================================================
// 2. AGENT STATE
// ============================================================================
template <uint32_t ObsDepth = 20>
class AgentState {
public:
    uint32_t id{0};

    // --- Network Physics (Private Parameters) ---
    uint64_t ingress_delay{0};     // Latency from the agent to the exchange
    uint64_t egress_delay{0};      // Latency from the exchange to the agent
    uint64_t compute_delay{0};     // Computation time (inference delay)
    uint64_t next_wakeup_time{0};  // Absolute time when the agent is allowed to act next

    // --- Local World View ---
    // The observation depth is forwarded at compile time for SIMD/Loop Unrolling.
    // Capacity is hardcoded to 4096 to ensure the hash table fits exactly into L1 Cache (32KB).
    LazyShadowLOB<ObsDepth, 4096> shadow_lob;

    // --- Pointers to Zero-Copy Arena Memory (Public Parameters) ---
    // The C++ engine will update balances by writing values directly to these
    // pointers, allowing PyTorch to instantly read the updated state.
    float* obs_cash_ptr{nullptr};
    float* obs_inventory_ptr{nullptr};
    float* obs_pnl_ptr{nullptr};

    // Hidden balances for precise math (to avoid float precision loss)
    int64_t real_cash{0};
    int32_t real_inventory{0};

    // Dynamic allocation parameter removed; memory layout is resolved at compile time
    explicit AgentState() = default;

    inline void update_balance(int64_t cash_delta, int32_t inventory_delta) noexcept {
        real_cash += cash_delta;
        real_inventory += inventory_delta;

        // Synchronize with the Python tensor
        *obs_cash_ptr = static_cast<float>(real_cash);
        *obs_inventory_ptr = static_cast<float>(real_inventory);
    }

    inline void reset() noexcept {
        next_wakeup_time = 0;
        real_cash = 0;
        real_inventory = 0;
        shadow_lob.clear();
        // Arena pointers are not changed, as the underlying memory lives forever
    }
};

// ============================================================================
// 3. ENVIRONMENT STATE
// ============================================================================
template <uint32_t ObsDepth = 20>
class EnvironmentState {
public:
    uint32_t env_id;

    // Each environment possesses its own independent nanosecond clock
    // critical for precise causality tracking across RL episodes.
    uint64_t current_time{0};

    // Array of agents residing within this specific environment instance.
    // Templated to forward the observation depth to all agents.
    std::vector<AgentState<ObsDepth>> agents;

    // The centralized event buffer for this environment.
    // Allocated safely on the Heap (as part of the parent container),
    // guaranteeing zero dynamic allocations during the hot loop while
    // completely preventing Stack Overflow exceptions.
    DefaultEventBuffer event_buffer;

    // Pointer to the localized Matching Engine governing this environment.
    MatchingEngine* engine = nullptr;

    // Instantly resets the environment physics and agents to epoch 0
    inline void reset() noexcept {
        current_time = 0;
        event_buffer.clear();
        for (auto& agent : agents) {
            agent.reset();
        }
    }
};

}  // namespace titan::core