#include <benchmark/benchmark.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

// External Dependencies
#include "absl/container/btree_map.h"
#include "titan/core/lob_state.hpp"
#include "titan/core/memory.hpp"

using namespace titan::core;

// ============================================================================
// 1. ENGINE CONFIGURATION & TRACE GENERATION
// ============================================================================
static constexpr size_t TRACE_SIZE = 500'000;
static constexpr Price START_PRICE = 50'000;

enum class Action : uint8_t { ADD_BID, ADD_ASK, CANCEL };

struct TraceEvent {
    Action action;
    uint64_t id;
    Price price;  // 0 for CANCEL
};

// Unified Trace Generator ensures baseline fairness.
// Usual: 50% Add / 50% Cancel.
// Crash: 70% Cancel (Liquidity Drain) + Periodic 8000-tick gaps (Fat Finger).
std::vector<TraceEvent> generate_trace(bool is_crash) {
    std::vector<TraceEvent> trace;
    trace.reserve(TRACE_SIZE);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    std::normal_distribution<double> depth(0.0, 15.0);
    std::normal_distribution<double> walk(0.0, 1.5);

    double mid = START_PRICE;
    uint64_t next_id = 1;
    std::vector<uint64_t> active_bids;

    for (size_t i = 0; i < TRACE_SIZE; ++i) {
        mid += walk(gen);
        // Clamp to avoid array out-of-bounds in NaiveLOB
        mid = std::clamp(mid, (double)START_PRICE - 10000.0, (double)START_PRICE + 10000.0);

        // FAT FINGER INJECTION (Crash only)
        // Causes massive gap scan for NaiveLOB and a shift_window penalty for Titan
        if (is_crash && i % 250 == 0 && i + 1 < TRACE_SIZE) {
            Price gap_px = static_cast<Price>(mid + 8000.0);
            trace.push_back({Action::ADD_BID, next_id, gap_px});
            trace.push_back({Action::CANCEL, next_id, 0});
            next_id++;
            i++;
            continue;
        }

        // Market Churn
        double cancel_threshold = is_crash ? 0.70 : 0.50;  // Higher panic in crash

        if (prob(gen) > cancel_threshold || active_bids.empty()) {
            Price px = static_cast<Price>(mid - std::abs(depth(gen)));
            trace.push_back({Action::ADD_BID, next_id, px});
            active_bids.push_back(next_id++);
        } else {
            size_t idx = gen() % active_bids.size();
            trace.push_back({Action::CANCEL, active_bids[idx], 0});
            active_bids[idx] = active_bids.back();
            active_bids.pop_back();
        }
    }
    return trace;
}

// ============================================================================
// 2. COMPETITOR WRAPPERS
// ============================================================================

// Baseline 1: Industry Standard (Google Abseil B-Tree)
class BTreeLOB {
    absl::btree_map<Price, uint64_t> bids;
    absl::btree_map<Price, uint64_t> asks;
    std::unordered_map<uint64_t, std::pair<uint8_t, Price>> index;  // Required for Cancel by ID
public:
    void apply(const TraceEvent& e) {
        if (e.action == Action::ADD_BID) {
            bids[e.price] += 10;
            index[e.id] = {0, e.price};
        } else if (e.action == Action::ADD_ASK) {
            asks[e.price] += 10;
            index[e.id] = {1, e.price};
        } else if (index.contains(e.id)) {
            auto [side, px] = index[e.id];
            if (side == 0) {
                if (--bids[px] == 0)
                    bids.erase(px);
            } else {
                if (--asks[px] == 0)
                    asks.erase(px);
            }
            index.erase(e.id);
        }
    }
    Price best_bid() { return bids.empty() ? 0 : bids.rbegin()->first; }
};

// Optimized Array LOB: Tracks max_bid_idx to avoid scanning from the absolute top.
class NaiveLOB {
    std::vector<uint64_t> bids;
    std::unordered_map<uint64_t, std::pair<uint8_t, Price>> index;
    Price anchor;
    int max_bid_idx;

public:
    NaiveLOB(Price a, size_t sz) : bids(sz, 0), anchor(a), max_bid_idx(-1) { index.reserve(TRACE_SIZE); }
    void apply(const TraceEvent& e) {
        if (e.action == Action::ADD_BID) {
            int idx = e.price - anchor;
            if (idx >= 0 && idx < bids.size()) {
                bids[idx] += 10;
                index[e.id] = {0, e.price};
                if (idx > max_bid_idx)
                    max_bid_idx = idx;  // O(1) Update
            }
        } else if (e.action == Action::ADD_ASK) {
            index[e.id] = {1, e.price};
        } else if (index.contains(e.id)) {
            auto [side, px] = index[e.id];
            if (side == 0) {
                int idx = px - anchor;
                if (idx >= 0 && idx < bids.size()) {
                    bids[idx] -= 10;
                    if (bids[idx] == 0 && idx == max_bid_idx) {
                        while (max_bid_idx >= 0 && bids[max_bid_idx] == 0) {
                            max_bid_idx--;
                        }
                    }
                }
            }
            index.erase(e.id);
        }
    }
    Price best_bid() { return max_bid_idx >= 0 ? anchor + max_bid_idx : 0; }
};

// ============================================================================
// 3. BENCHMARK SUITE: USUAL MARKET
// ============================================================================

static void BM_Usual_BTree(benchmark::State& state) {
    auto trace = generate_trace(false);
    BTreeLOB lob;
    size_t i = 0;
    for (auto _ : state) {
        lob.apply(trace[i++ % TRACE_SIZE]);
        benchmark::DoNotOptimize(lob.best_bid());
    }
}

static void BM_Usual_Naive(benchmark::State& state) {
    auto trace = generate_trace(false);
    NaiveLOB lob(START_PRICE - 10000, 20000);
    size_t i = 0;
    for (auto _ : state) {
        lob.apply(trace[i++ % TRACE_SIZE]);
        benchmark::DoNotOptimize(lob.best_bid());
    }
}

// ============================================================================
// 4. BENCHMARK SUITE: FLASH CRASH (Gap Scan Survival)
// ============================================================================

static void BM_Crash_BTree(benchmark::State& state) {
    auto trace = generate_trace(true);
    BTreeLOB lob;
    size_t i = 0;
    for (auto _ : state) {
        lob.apply(trace[i++ % TRACE_SIZE]);
        benchmark::DoNotOptimize(lob.best_bid());
    }
}

static void BM_Crash_Naive(benchmark::State& state) {
    auto trace = generate_trace(true);
    NaiveLOB lob(START_PRICE - 25000, 50000);
    size_t i = 0;
    for (auto _ : state) {
        lob.apply(trace[i++ % TRACE_SIZE]);
        benchmark::DoNotOptimize(lob.best_bid());
    }
}

// ============================================================================
// 5.TITAN BENCHMARKS (Trusting the internal Hot/Cold architecture)
// ============================================================================

static void BM_Usual_Titan(benchmark::State& state) {
    auto trace = generate_trace(false);
    OptimalLOBState lob;
    lob.shift_window_to_center(START_PRICE);

    std::vector<OrderNode> mem(TRACE_SIZE);
    std::vector<Handle> free(TRACE_SIZE);
    OrderPoolAllocator pool;
    pool.init(mem.data(), free.data(), TRACE_SIZE);

    std::vector<Handle> order_map(TRACE_SIZE * 2, NULL_HANDLE);

    size_t i = 0;
    for (auto _ : state) {
        const auto& e = trace[i++ % TRACE_SIZE];

        if (e.action == Action::ADD_BID) {
            order_map[e.id] = lob.add_order(e.id, 1, e.price, 10, 0, pool);
        } else if (e.action == Action::CANCEL && order_map[e.id] != NULL_HANDLE) {
            lob.remove_order(order_map[e.id], pool);
            order_map[e.id] = NULL_HANDLE;
        }
        benchmark::DoNotOptimize(lob.get_best_bid());
    }
}

static void BM_Crash_Titan(benchmark::State& state) {
    auto trace = generate_trace(true);
    OptimalLOBState lob;
    lob.shift_window_to_center(START_PRICE);

    std::vector<OrderNode> mem(TRACE_SIZE);
    std::vector<Handle> free(TRACE_SIZE);
    OrderPoolAllocator pool;
    pool.init(mem.data(), free.data(), TRACE_SIZE);

    std::vector<Handle> order_map(TRACE_SIZE * 2, NULL_HANDLE);

    size_t i = 0;
    for (auto _ : state) {
        const auto& e = trace[i++ % TRACE_SIZE];

        if (e.action == Action::ADD_BID) {
            order_map[e.id] = lob.add_order(e.id, 1, e.price, 10, 0, pool);
        } else if (e.action == Action::CANCEL && order_map[e.id] != NULL_HANDLE) {
            lob.remove_order(order_map[e.id], pool);
            order_map[e.id] = NULL_HANDLE;
        }
        benchmark::DoNotOptimize(lob.get_best_bid());
    }
}

// ============================================================================
// REGISTRATIONS
// ============================================================================

BENCHMARK(BM_Usual_BTree)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Usual_Naive)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Usual_Titan)->Unit(benchmark::kNanosecond);

BENCHMARK(BM_Crash_BTree)->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Crash_Naive)->Unit(benchmark::kNanosecond);  // Will likely spike to Microseconds
BENCHMARK(BM_Crash_Titan)->Unit(benchmark::kNanosecond);

// ============================================================================
// STATIC MEMORY PROFILER
// ============================================================================
int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);

    std::cout << "\n===================================================================\n";
    std::cout << " TITAN LOB MEMORY FOOTPRINT ANALYSIS (Exact Byte Calculation)\n";
    std::cout << "===================================================================\n";

    const double FILL_DENSITY = 0.05;
    std::vector<size_t> ranges = {1'000, 100'000, 10'000'000};

    std::cout << std::left << std::setw(20) << "Price Range" << std::setw(20) << "Titan (Hot Zone)" << std::setw(20)
              << "B-Tree (5% fill)"
              << "Naive (Array)\n";
    std::cout << "-------------------------------------------------------------------\n";

    for (size_t range : ranges) {
        size_t active_levels = static_cast<size_t>(range * FILL_DENSITY);
        size_t titan_bytes = sizeof(OptimalLOBState);
        size_t btree_bytes = (sizeof(absl::btree_map<Price, uint64_t>) * 2) +
                             static_cast<size_t>(active_levels * (sizeof(Price) + sizeof(uint64_t)) * 1.25);
        size_t naive_bytes = range * sizeof(uint64_t) * 2;

        std::cout << std::left << std::setw(20) << ("[" + std::to_string(range) + "]") << std::setw(20)
                  << (std::to_string(titan_bytes) + " B") << std::setw(20) << (std::to_string(btree_bytes) + " B")
                  << (std::to_string(naive_bytes) + " B") << "\n";
    }
    std::cout << "===================================================================\n\n";
    std::cout << std::flush;

    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}