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
// 1. SHADOW LOB
// Local, delayed copy of the reality for each individual agent.
// Uses Boost Flat Map for optimal L1/L2 cache locality and O(log N) lookups,
// gracefully scaling to 10,000+ levels without pointer-chasing overhead.
// ============================================================================
class ShadowLOB {
private:
    // Bids are automatically sorted in DESCENDING order (highest price first)
    boost::container::flat_map<Price, OrderQty, std::greater<Price>> bids_;

    // Asks are automatically sorted in ASCENDING order (lowest price first)
    boost::container::flat_map<Price, OrderQty, std::less<Price>> asks_;

public:
    explicit ShadowLOB(std::size_t reserve_capacity = 1024) {
        // Pre-allocating contiguous memory blocks to prevent reallocations
        bids_.reserve(reserve_capacity);
        asks_.reserve(reserve_capacity);
    }

    // Apply a delta (Market Data Event) to the shadow LOB
    inline void apply_delta(uint8_t side, Price price, int32_t qty_delta) noexcept {
        if (side == 0) {  // --- BID ---
            auto it = bids_.find(price);
            if (it != bids_.end()) {
                // Level exists, update the quantity
                int32_t new_qty = static_cast<int32_t>(it->second) + qty_delta;
                if (new_qty <= 0) {
                    bids_.erase(it);  // Erase the level if liquidity is depleted
                } else {
                    it->second = static_cast<OrderQty>(new_qty);
                }
            } else if (qty_delta > 0) {
                // New price level: emplace handles the sorted insertion automatically
                bids_.emplace(price, static_cast<OrderQty>(qty_delta));
            }
        } else {  // --- ASK ---
            auto it = asks_.find(price);
            if (it != asks_.end()) {
                int32_t new_qty = static_cast<int32_t>(it->second) + qty_delta;
                if (new_qty <= 0) {
                    asks_.erase(it);
                } else {
                    it->second = static_cast<OrderQty>(new_qty);
                }
            } else if (qty_delta > 0) {
                asks_.emplace(price, static_cast<OrderQty>(qty_delta));
            }
        }
    }

    // Instant Zero-Copy export of top-N levels directly to the PyTorch tensor.
    // Safe 'depth' parameter protects RL networks from the curse of dimensionality,
    // while the flat_map safely holds the full macroscopic state of the market.
    inline void export_to_tensor(float* obs_ptr, uint32_t depth) const noexcept {
        uint32_t offset = 0;

        // Write Bids (Price, Quantity)
        auto bid_it = bids_.begin();
        for (uint32_t i = 0; i < depth; ++i) {
            if (bid_it != bids_.end()) {
                obs_ptr[offset++] = static_cast<float>(bid_it->first);   // Price
                obs_ptr[offset++] = static_cast<float>(bid_it->second);  // Quantity
                ++bid_it;
            } else {
                // Pad with zeros if the book depth is shallower than requested
                obs_ptr[offset++] = 0.0f;
                obs_ptr[offset++] = 0.0f;
            }
        }

        // Write Asks (Price, Quantity)
        auto ask_it = asks_.begin();
        for (uint32_t i = 0; i < depth; ++i) {
            if (ask_it != asks_.end()) {
                obs_ptr[offset++] = static_cast<float>(ask_it->first);   // Price
                obs_ptr[offset++] = static_cast<float>(ask_it->second);  // Quantity
                ++ask_it;
            } else {
                obs_ptr[offset++] = 0.0f;
                obs_ptr[offset++] = 0.0f;
            }
        }
    }

    // Fast reset between simulation episodes
    inline void clear() noexcept {
        bids_.clear();
        asks_.clear();
    }
};

// ============================================================================
// 2. AGENT STATE
// ============================================================================
class AgentState {
public:
    uint32_t id{0};

    // --- Network Physics (Private Parameters) ---
    uint64_t ingress_delay{0};     // Latency from the agent to the exchange
    uint64_t egress_delay{0};      // Latency from the exchange to the agent
    uint64_t compute_delay{0};     // Computation time (inference delay)
    uint64_t next_wakeup_time{0};  // Absolute time when the agent is allowed to act next

    // --- Local World View ---
    ShadowLOB shadow_lob;

    // --- Pointers to Zero-Copy Arena Memory (Public Parameters) ---
    // The C++ engine will update balances by writing values directly to these
    // pointers, allowing PyTorch to instantly read the updated state.
    float* obs_cash_ptr{nullptr};
    float* obs_inventory_ptr{nullptr};
    float* obs_pnl_ptr{nullptr};

    // Hidden balances for precise math (to avoid float precision loss)
    int64_t real_cash{0};
    int32_t real_inventory{0};

    explicit AgentState(std::size_t lob_reserve_capacity) : shadow_lob(lob_reserve_capacity) {}

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
class EnvironmentState {
public:
    uint32_t env_id;

    // Each environment possesses its own independent nanosecond clock
    // critical for precise causality tracking across RL episodes.
    uint64_t current_time{0};

    // Array of agents residing within this specific environment instance.
    std::vector<AgentState> agents;

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