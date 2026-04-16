#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "titan/core/types.hpp"

// Requires Boost for flat_map
#include <boost/container/flat_map.hpp>

// Requires Google Abseil for btree_map
#include <absl/container/btree_map.h>

// Native Titan includes
#include "titan/core/state.hpp"

using namespace titan::core;

// ============================================================================
// BENCHMARK EVENT STRUCTURE
// ============================================================================
struct MarketEvent {
    uint8_t side;
    Price price;
    OrderQty qty;
};

// ============================================================================
// COMPETITOR 1: Google Abseil B-Tree (absl::btree_map)
// Represents the industry standard cache-friendly tree structure.
// ============================================================================
class BTreeLOB {
private:
    absl::btree_map<Price, OrderQty, std::greater<Price>> bids_;
    absl::btree_map<Price, OrderQty, std::less<Price>> asks_;

public:
    inline void apply_delta(uint8_t side, Price price, OrderQty qty) noexcept {
        if (side == 0) {
            bids_[price] += qty;
            if (bids_[price] <= 0)
                bids_.erase(price);
        } else {
            asks_[price] += qty;
            if (asks_[price] <= 0)
                asks_.erase(price);
        }
    }

    inline void export_to_tensor(float* obs_ptr, uint32_t depth = 20) noexcept {
        uint32_t offset = 0;
        uint32_t count = 0;

        for (auto it = bids_.begin(); it != bids_.end() && count < depth; ++it, ++count) {
            obs_ptr[offset++] = static_cast<float>(it->first);
            obs_ptr[offset++] = static_cast<float>(it->second);
        }
        while (count++ < depth) {
            obs_ptr[offset++] = 0.0f;
            obs_ptr[offset++] = 0.0f;
        }

        count = 0;
        for (auto it = asks_.begin(); it != asks_.end() && count < depth; ++it, ++count) {
            obs_ptr[offset++] = static_cast<float>(it->first);
            obs_ptr[offset++] = static_cast<float>(it->second);
        }
        while (count++ < depth) {
            obs_ptr[offset++] = 0.0f;
            obs_ptr[offset++] = 0.0f;
        }
    }

    inline void clear() noexcept {
        bids_.clear();
        asks_.clear();
    }
};

// ============================================================================
// COMPETITOR 2: Flat Map (boost::container::flat_map)
// Represents a contiguous memory approach, heavily penalized by O(N) shifts.
// ============================================================================
class FlatMapLOB {
private:
    boost::container::flat_map<Price, OrderQty, std::greater<Price>> bids_;
    boost::container::flat_map<Price, OrderQty, std::less<Price>> asks_;

public:
    inline void apply_delta(uint8_t side, Price price, OrderQty qty) noexcept {
        if (side == 0) {
            bids_[price] += qty;
            if (bids_[price] <= 0)
                bids_.erase(price);
        } else {
            asks_[price] += qty;
            if (asks_[price] <= 0)
                asks_.erase(price);
        }
    }

    inline void export_to_tensor(float* obs_ptr, uint32_t depth = 20) noexcept {
        uint32_t offset = 0;
        uint32_t count = 0;

        for (auto it = bids_.begin(); it != bids_.end() && count < depth; ++it, ++count) {
            obs_ptr[offset++] = static_cast<float>(it->first);
            obs_ptr[offset++] = static_cast<float>(it->second);
        }
        while (count++ < depth) {
            obs_ptr[offset++] = 0.0f;
            obs_ptr[offset++] = 0.0f;
        }

        count = 0;
        for (auto it = asks_.begin(); it != asks_.end() && count < depth; ++it, ++count) {
            obs_ptr[offset++] = static_cast<float>(it->first);
            obs_ptr[offset++] = static_cast<float>(it->second);
        }
        while (count++ < depth) {
            obs_ptr[offset++] = 0.0f;
            obs_ptr[offset++] = 0.0f;
        }
    }

    inline void clear() noexcept {
        bids_.clear();
        asks_.clear();
    }
};

// ============================================================================
// COMPETITOR 3: Legacy Static Bitmask Hierarchy (No Sliding Window)
// Extremely fast but mathematically flawed for RL (vulnerable to collisions).
// Kept for baseline latency comparison against the robust version.
// ============================================================================
template <uint32_t RingSize = 4096>
class LegacyBitmaskLOB {
private:
    static constexpr uint32_t L1_SIZE = RingSize / 64;
    static constexpr uint32_t L2_SIZE = (L1_SIZE + 63) / 64;

    uint64_t bid_l1_[L1_SIZE]{0};
    uint64_t bid_l2_[L2_SIZE]{0};
    OrderQty bid_qty_[RingSize]{0};

    uint64_t ask_l1_[L1_SIZE]{0};
    uint64_t ask_l2_[L2_SIZE]{0};
    OrderQty ask_qty_[RingSize]{0};

    inline uint32_t get_idx(Price price) const noexcept { return static_cast<uint32_t>(price) & (RingSize - 1); }

public:
    inline void apply_delta(uint8_t side, Price price, OrderQty qty) noexcept {
        uint32_t idx = get_idx(price);
        uint32_t l1_idx = idx >> 6;
        uint32_t l2_idx = l1_idx >> 6;

        if (side == 0) {
            bid_qty_[idx] += qty;
            if (bid_qty_[idx] > 0) {
                bid_l1_[l1_idx] |= (1ULL << (idx & 63));
                bid_l2_[l2_idx] |= (1ULL << (l1_idx & 63));
            } else {
                bid_l1_[l1_idx] &= ~(1ULL << (idx & 63));
                if (bid_l1_[l1_idx] == 0)
                    bid_l2_[l2_idx] &= ~(1ULL << (l1_idx & 63));
            }
        } else {
            ask_qty_[idx] += qty;
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

    inline void export_to_tensor(float* obs_ptr, uint32_t depth = 20) noexcept {
        uint32_t offset = 0;
        uint32_t count = 0;

        for (int32_t l2 = L2_SIZE - 1; l2 >= 0 && count < depth; --l2) {
            uint64_t mask2 = bid_l2_[l2];
            while (mask2 && count < depth) {
                uint32_t bit2 = pop_msb(mask2);
                uint32_t l1_idx = (l2 << 6) + bit2;
                uint64_t mask1 = bid_l1_[l1_idx];
                while (mask1 && count < depth) {
                    uint32_t bit1 = pop_msb(mask1);
                    uint32_t idx = (l1_idx << 6) + bit1;

                    obs_ptr[offset++] = static_cast<float>(idx);
                    obs_ptr[offset++] = static_cast<float>(bid_qty_[idx]);
                    count++;
                }
            }
        }
        while (count++ < depth) {
            obs_ptr[offset++] = 0.0f;
            obs_ptr[offset++] = 0.0f;
        }

        count = 0;
        for (uint32_t l2 = 0; l2 < L2_SIZE && count < depth; ++l2) {
            uint64_t mask2 = ask_l2_[l2];
            while (mask2 && count < depth) {
                uint32_t bit2 = pop_lsb(mask2);
                uint32_t l1_idx = (l2 << 6) + bit2;
                uint64_t mask1 = ask_l1_[l1_idx];
                while (mask1 && count < depth) {
                    uint32_t bit1 = pop_lsb(mask1);
                    uint32_t idx = (l1_idx << 6) + bit1;

                    obs_ptr[offset++] = static_cast<float>(idx);
                    obs_ptr[offset++] = static_cast<float>(ask_qty_[idx]);
                    count++;
                }
            }
        }
        while (count++ < depth) {
            obs_ptr[offset++] = 0.0f;
            obs_ptr[offset++] = 0.0f;
        }
    }

    inline void clear() noexcept {
        std::memset(bid_l1_, 0, sizeof(bid_l1_));
        std::memset(bid_l2_, 0, sizeof(bid_l2_));
        std::memset(bid_qty_, 0, sizeof(bid_qty_));
        std::memset(ask_l1_, 0, sizeof(ask_l1_));
        std::memset(ask_l2_, 0, sizeof(ask_l2_));
        std::memset(ask_qty_, 0, sizeof(ask_qty_));
    }
};

// ============================================================================
// BENCHMARK HARNESS
// ============================================================================
template <typename LOBType>
void run_benchmark(const std::string& name, const std::vector<uint32_t>& lambdas,
                   const std::vector<MarketEvent>& event_stream, uint32_t rl_steps) {
    std::cout << "--- " << name << " ---\n" << std::flush;
    alignas(64) float obs_tensor[80];
    LOBType lob;

    volatile float dummy_sink = 0.0f;

    for (uint32_t lambda : lambdas) {
        lob.clear();
        uint64_t total_events = rl_steps * lambda;
        auto start_time = std::chrono::high_resolution_clock::now();
        uint64_t event_idx = 0;

        for (uint32_t step = 0; step < rl_steps; ++step) {
            // High-Frequency Market Micro-Updates
            for (uint32_t l = 0; l < lambda; ++l) {
                const auto& ev = event_stream[event_idx++];
                lob.apply_delta(ev.side, ev.price, ev.qty);
            }

            // RL Barrier: Synchronous tensor materialization
            lob.export_to_tensor(obs_tensor);

            // Prevent dead code elimination
            dummy_sink = dummy_sink + obs_tensor[0];
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        uint64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
        double ns_per_event = static_cast<double>(total_ns) / total_events;
        double ns_per_step = static_cast<double>(total_ns) / rl_steps;

        std::cout << "Lambda: " << std::setw(4) << lambda << " | Latency per Event: " << std::fixed
                  << std::setprecision(1) << std::setw(6) << ns_per_event << " ns"
                  << " | Latency per RL Step: " << std::fixed << std::setprecision(1) << std::setw(8) << ns_per_step
                  << " ns\n"
                  << std::flush;
    }
    std::cout << "\n";
}

// ============================================================================
// REALISTIC MARKET DATA GENERATOR
// Simulates non-stationary price dynamics including drift, micro-volatility,
// and aggressive flash crashes/spikes to stress-test the Sliding Window logic.
// ============================================================================
std::vector<MarketEvent> generate_realistic_market(uint64_t max_events) {
    std::vector<MarketEvent> stream(max_events);
    std::mt19937 gen(42);

    // Micro-structure dynamics
    std::normal_distribution<> tick_volatility(0.0, 3.0);
    std::uniform_int_distribution<> side_dist(0, 1);
    std::uniform_int_distribution<> qty_dist(-20, 50);

    // Macro-structure dynamics (Fat Tails / Shocks)
    std::uniform_real_distribution<> shock_prob(0.0, 1.0);
    std::normal_distribution<> shock_magnitude(0.0, 1500.0);  // Forces re-centering

    double current_price = 100000.0;  // Starting at a high price to allow drops

    for (uint64_t i = 0; i < max_events; ++i) {
        // 1. Simulate Price Action
        double shock_chance = shock_prob(gen);
        if (shock_chance < 0.001) {
            // 0.1% chance of a massive market shock (Flash Crash / Spike)
            // This guarantees the ShadowLOB must trigger heavy memmove or clear routines.
            current_price += shock_magnitude(gen);
        } else if (shock_chance < 0.05) {
            // 5% chance of a directional trend shift
            current_price += tick_volatility(gen) * 5.0;
        } else {
            // 94.9% Normal micro-volatility around the spread
            current_price += tick_volatility(gen);
        }

        // Bound validation
        if (current_price < 1.0)
            current_price = 1.0;

        // 2. Simulate Order Quantity (incorporating partial fills and cancellations)
        OrderQty qty = qty_dist(gen);
        if (qty == 0)
            qty = 1;

        stream[i] = {static_cast<uint8_t>(side_dist(gen)), static_cast<Price>(current_price), qty};
    }
    return stream;
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " Titan LOB Architecture Benchmark (Combat Simulation)\n";
    std::cout << " Testing: Micro-Volatility, Trends, and Flash Crashes\n";
    std::cout << "========================================================\n\n";

    constexpr uint32_t RL_STEPS = 10000;
    std::vector<uint32_t> lambdas = {1, 3, 5, 10, 15, 20, 35, 50, 100, 500};
    uint64_t max_total_events = RL_STEPS * lambdas.back();

    std::cout << "[*] Generating " << max_total_events << " realistic market events...\n\n" << std::flush;
    std::vector<MarketEvent> event_stream = generate_realistic_market(max_total_events);

    // Baseline 1: Standard Tree (Worst Cache Locality)
    run_benchmark<BTreeLOB>("Competitor 1: Google Abseil B-Tree", lambdas, event_stream, RL_STEPS);

    // Baseline 2: Flat Array (Heavy Shift Penalty)
    run_benchmark<FlatMapLOB>("Competitor 2: Flat Map (boost)", lambdas, event_stream, RL_STEPS);

    // Baseline 3: Legacy Static Bitmask (Ignores collisions, fastest theoretical limit)
    run_benchmark<LegacyBitmaskLOB<4096>>("Competitor 3: Legacy Bitmask (No Window Shifts)", lambdas, event_stream,
                                          RL_STEPS);

    // Native Architecture: Robust Sliding Window Bitmask from state.hpp
    run_benchmark<ShadowLOB<20, 4096>>("Native: Titan ShadowLOB (Robust Sliding Window)", lambdas, event_stream,
                                       RL_STEPS);

    return 0;
}