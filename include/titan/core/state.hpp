#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// 1. SHADOW LOB
// Local, delayed copy of the reality for each individual agent.
// ============================================================================
class ShadowLOB {
private:
    // Flat arrays for L1 cache locality. In HFT, memmove (std::vector::insert)
    // on small arrays (up to 100-200 elements) is orders of magnitude faster
    // than red-black trees (std::map) due to the absence of cache misses.
    struct Level {
        Price price;
        OrderQty qty;
    };

    std::vector<Level> bids_;
    std::vector<Level> asks_;

public:
    explicit ShadowLOB(std::size_t reserve_capacity = 1024) {
        bids_.reserve(reserve_capacity);
        asks_.reserve(reserve_capacity);
    }

    // Apply a delta (Market Data Event) to the shadow LOB
    inline void apply_delta(uint8_t side, Price price, int32_t qty_delta) noexcept {
        auto& book = (side == 0) ? bids_ : asks_;

        // Bids are sorted in descending order, Asks in ascending order
        auto cmp = [side](const Level& a, const Level& b) {
            return (side == 0) ? a.price > b.price : a.price < b.price;
        };

        auto it = std::lower_bound(book.begin(), book.end(), Level{price, 0}, cmp);

        if (it != book.end() && it->price == price) {
            // Level exists, update the quantity
            int32_t new_qty = static_cast<int32_t>(it->qty) + qty_delta;
            if (new_qty <= 0) {
                book.erase(it);  // Erase the level if liquidity is depleted
            } else {
                it->qty = static_cast<OrderQty>(new_qty);
            }
        } else if (qty_delta > 0) {
            // New price level
            book.insert(it, Level{price, static_cast<OrderQty>(qty_delta)});
        }
    }

    // Instant Zero-Copy export of top-N levels directly to the PyTorch tensor
    inline void export_to_tensor(float* obs_ptr, uint32_t depth) const noexcept {
        uint32_t offset = 0;

        // Write Bids (Price, Quantity)
        for (uint32_t i = 0; i < depth; ++i) {
            if (i < bids_.size()) {
                obs_ptr[offset++] = static_cast<float>(bids_[i].price);
                obs_ptr[offset++] = static_cast<float>(bids_[i].qty);
            } else {
                obs_ptr[offset++] = 0.0f;
                obs_ptr[offset++] = 0.0f;
            }
        }

        // Write Asks (Price, Quantity)
        for (uint32_t i = 0; i < depth; ++i) {
            if (i < asks_.size()) {
                obs_ptr[offset++] = static_cast<float>(asks_[i].price);
                obs_ptr[offset++] = static_cast<float>(asks_[i].qty);
            } else {
                obs_ptr[offset++] = 0.0f;
                obs_ptr[offset++] = 0.0f;
            }
        }
    }

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
    uint32_t id;

    // --- Network Physics (Private Parameters) ---
    uint64_t ingress_delay;     // Latency from the agent to the exchange
    uint64_t egress_delay;      // Latency from the exchange to the agent
    uint64_t compute_delay;     // Computation time (inference delay)
    uint64_t next_wakeup_time;  // Absolute time when the agent is allowed to act next

    // --- Local World View ---
    ShadowLOB shadow_lob;

    // --- Pointers to Zero-Copy Arena Memory (Public Parameters) ---
    // The C++ engine will update balances by writing values directly to these
    // pointers, allowing PyTorch to instantly read the updated state.
    float* obs_cash_ptr;
    float* obs_inventory_ptr;
    float* obs_pnl_ptr;

    // Hidden balances for precise math (to avoid float precision loss)
    int64_t real_cash;
    int32_t real_inventory;

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

    // Each environment has its own independent nanosecond clock
    uint64_t current_time{0};

    // Array of agents living within this specific environment instance
    std::vector<AgentState> agents;

    // A pointer to the MatchingEngine for this environment will be added here
    // MatchingEngine* engine = nullptr;

    inline void reset() noexcept {
        current_time = 0;
        for (auto& agent : agents) {
            agent.reset();
        }
    }
};

}  // namespace titan::core