#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Native Titan includes
#include "titan/core/state.hpp"
#include "titan/core/types.hpp"

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
    std::normal_distribution<> shock_magnitude(0.0, 1500.0);  // Magnitude enough to force shifts on small windows

    double current_price = 100000.0;  // Starting at a high price to allow drops

    for (uint64_t i = 0; i < max_events; ++i) {
        // 1. Simulate Price Action
        double shock_chance = shock_prob(gen);
        if (shock_chance < 0.001) {
            // 0.1% chance of a massive market shock (Flash Crash / Spike)
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
    std::cout << " Titan Hardware Optimization: L1 vs L2 Cache Battle\n";
    std::cout << " Testing Window Sizes to find the Amortization Sweet Spot\n";
    std::cout << "========================================================\n\n";

    constexpr uint32_t RL_STEPS = 10000;
    std::vector<uint32_t> lambdas = {1, 5, 15, 35, 100, 500};
    uint64_t max_total_events = RL_STEPS * lambdas.back();

    std::cout << "[*] Generating " << max_total_events << " realistic market events...\n\n" << std::flush;
    std::vector<MarketEvent> event_stream = generate_realistic_market(max_total_events);

    // ------------------------------------------------------------------------
    // HYPOTHESIS 1: PURE L1 CACHE (~16 KB memory footprint)
    // Arrays fit perfectly in the fastest CPU cache. Base access is ~1ns.
    // DANGER: Window is so small that volatility will constantly trigger O(N) shifts.
    // ------------------------------------------------------------------------
    run_benchmark<ShadowLOB<20, 2048>>("Size 2048 (Pure L1 Cache - 16KB)", lambdas, event_stream, RL_STEPS);

    // ------------------------------------------------------------------------
    // HYPOTHESIS 2: MAXIMAL L1 CACHE (~32 KB memory footprint)
    // Fills the L1d cache entirely. Excellent baseline.
    // DANGER: Still vulnerable to frequent memmove penalties during trends.
    // ------------------------------------------------------------------------
    run_benchmark<ShadowLOB<20, 4096>>("Size 4096 (Max L1 Cache - 32KB)", lambdas, event_stream, RL_STEPS);

    // ------------------------------------------------------------------------
    // HYPOTHESIS 3: L1/L2 CACHE BOUNDARY (~65 KB memory footprint)
    // Spills slightly into L2. Shifts are less frequent, but base access is slightly slower.
    // ------------------------------------------------------------------------
    run_benchmark<ShadowLOB<20, 8192>>("Size 8192 (L1/L2 Boundary - 65KB)", lambdas, event_stream, RL_STEPS);

    // ------------------------------------------------------------------------
    // HYPOTHESIS 4: PURE L2 CACHE (~130 KB memory footprint)
    // Memory access takes ~4-5ns.
    // ADVANTAGE: The window is so massive that O(N) shifts and Cold Zone lookups are mathematically eliminated
    // for 99.9% of ticks.
    // ------------------------------------------------------------------------
    run_benchmark<ShadowLOB<20, 16384>>("Size 16384 (Pure L2 Cache - 130KB)", lambdas, event_stream, RL_STEPS);

    return 0;
}