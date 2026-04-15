#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

// Native Titan includes
#include "titan/core/matching_engine.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

using namespace titan::core;

// ============================================================================
// MICRO-BENCHMARK SUITE FOR MATCHING ENGINE LOGIC
// Isolates the algorithmic overhead of routing, matching, allocation,
// event generation, and Self-Trade Prevention (STP).
// ============================================================================

struct BenchResult {
    std::string name;
    double ns_per_order;
    double ns_per_matched_level;  // Relevant for sweeps
};

// ----------------------------------------------------------------------------
// TEST 1: PURE PASSIVE ROUTING (Allocation & Event Generation Overhead)
// Measures the absolute minimum time to process an order that doesn't cross
// the spread. Tests pool_.allocate(), lob_.add_order(), and EventBuffer::push.
// ----------------------------------------------------------------------------
BenchResult bench_passive_routing(MatchingEngine& engine, uint32_t count, uint64_t start_id) {
    DefaultEventBuffer events;

    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; ++i) {
        events.clear();
        // Inserting deep bids that will never match
        engine.process_order(start_id + i, 1, 0, 1000 - (i % 100), 10, events);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    return {"Passive Order Insertion (Allocation + Routing)", total_ns / count, 0};
}

// ----------------------------------------------------------------------------
// TEST 2: THE PERFECT CROSS (1-to-1 Match)
// Measures the overhead of spread validation, node retrieval, execution math,
// and generating 3 events (Trade Maker, Trade Taker, LOB Update).
// ----------------------------------------------------------------------------
BenchResult bench_perfect_cross(MatchingEngine& engine, uint32_t count, uint64_t start_id) {
    DefaultEventBuffer events;

    // Pre-fill the book with 'count' passive asks at price 2000
    for (uint32_t i = 0; i < count; ++i) {
        events.clear();
        engine.process_order(start_id + i, 2, 1, 2000, 10, events);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; ++i) {
        events.clear();
        // Send aggressive bids exactly matching the resting volume (10 lots)
        engine.process_order(start_id + count + i, 1, 0, 2000, 10, events);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    return {"Perfect Cross (1 Aggressive matches 1 Resting)", total_ns / count, total_ns / count};
}

// ----------------------------------------------------------------------------
// TEST 3: THE DEEP SWEEP (Algorithmic Loop Overhead)
// 1 massive market order consumes 100 resting levels.
// Tests the latency of the while(remaining_qty > 0) loop and pointer chasing.
// ----------------------------------------------------------------------------
BenchResult bench_deep_sweep(MatchingEngine& engine, uint32_t sweeps, uint32_t depth, uint64_t start_id) {
    DefaultEventBuffer events;
    uint64_t current_id = start_id;

    double total_sweep_ns = 0;

    for (uint32_t s = 0; s < sweeps; ++s) {
        // Setup: Build 'depth' levels of resting liquidity
        for (uint32_t i = 0; i < depth; ++i) {
            events.clear();
            engine.process_order(current_id++, 2, 1, 3000 + i, 10, events);
        }

        // Action: 1 massive aggressive order sweeps all 'depth' levels.
        // UINT32_MAX signifies an aggressive market buy.
        events.clear();
        auto start_time = std::chrono::high_resolution_clock::now();

        engine.process_order(current_id++, 1, 0, UINT32_MAX, depth * 10, events);

        auto end_time = std::chrono::high_resolution_clock::now();
        total_sweep_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    }

    return {"Deep Sweep (1 Order consumes 100 Levels)", total_sweep_ns / sweeps, total_sweep_ns / (sweeps * depth)};
}

// ----------------------------------------------------------------------------
// TEST 4: SELF-TRADE PREVENTION (STP) PENALTY
// Aggressive order hits its own passive order. Triggers specific STP logic
// (Cancel Resting). Tests the branch prediction penalty of [[unlikely]].
// ----------------------------------------------------------------------------
BenchResult bench_stp_penalty(MatchingEngine& engine, uint32_t count, uint64_t start_id) {
    DefaultEventBuffer events;

    // Pre-fill the book with 'count' passive asks from AGENT 5
    for (uint32_t i = 0; i < count; ++i) {
        events.clear();
        engine.process_order(start_id + i, 5, 1, 4000, 10, events);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    for (uint32_t i = 0; i < count; ++i) {
        events.clear();
        // Send aggressive bids ALSO from AGENT 5 to trigger STP Cancel-Resting logic
        engine.process_order(start_id + count + i, 5, 0, 4000, 10, events);
    }
    auto end_time = std::chrono::high_resolution_clock::now();

    double total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    return {"Self-Trade Prevention (STP Cancel Resting)", total_ns / count, total_ns / count};
}

int main() {
    std::cout << "========================================================\n";
    std::cout << " Titan Matching Engine: Algorithmic Overhead Profiler\n";
    std::cout << " Isolating Core Business Logic Latency\n";
    std::cout << "========================================================\n\n";

    // Memory configuration matching the UnifiedMemoryArena constructor signature
    constexpr uint32_t NUM_ENVS = 1;
    constexpr uint32_t MAX_ORDERS = 5000000;
    constexpr size_t LINEAR_BYTES = 1024 * 1024 * 16;  // 16 MB
    constexpr uint32_t NUM_AGENTS = 10;
    constexpr uint32_t OBS_DIM = 20;
    constexpr uint32_t ACT_DIM = 4;

    // Initialize the massive pre-faulted unified arena
    UnifiedMemoryArena arena(NUM_ENVS, MAX_ORDERS, LINEAR_BYTES, NUM_AGENTS, OBS_DIM, ACT_DIM);

    // Extract the pool for environment 0
    // Assuming get_pool(env_id) is the accessor defined in memory.hpp
    OrderPoolAllocator& pool = arena.get_pool(0);

    MatchingEngine engine(pool, MAX_ORDERS);

    constexpr uint32_t ITERATIONS = 100000;
    uint64_t global_id = 1;

    // Run tests
    auto res1 = bench_passive_routing(engine, ITERATIONS, global_id);
    global_id += ITERATIONS;

    auto res2 = bench_perfect_cross(engine, ITERATIONS, global_id);
    global_id += ITERATIONS * 2;

    auto res3 = bench_deep_sweep(engine, ITERATIONS / 10, 100, global_id);
    global_id += (ITERATIONS / 10) * 101;

    auto res4 = bench_stp_penalty(engine, ITERATIONS, global_id);

    // Print Results
    auto print_res = [](const BenchResult& r) {
        std::cout << std::left << std::setw(50) << r.name << " | " << std::fixed << std::setprecision(1) << std::setw(6)
                  << r.ns_per_order << " ns/order";
        if (r.ns_per_matched_level > 0) {
            std::cout << "  (" << r.ns_per_matched_level << " ns/level)";
        }
        std::cout << "\n";
    };

    print_res(res1);
    print_res(res2);
    print_res(res3);
    print_res(res4);

    return 0;
}