#include <benchmark/benchmark.h>

#include <algorithm>
#include <boost/pool/object_pool.hpp>
#include <numeric>
#include <random>
#include <vector>

#include "titan/core/memory.hpp"

namespace titan::core::bench {

// ============================================================================
// Helper: Generate deterministic random shuffle for scattered free benchmarks
// ============================================================================
std::vector<uint32_t> generate_random_indices(uint32_t size) {
    std::vector<uint32_t> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    // Use fixed seed for deterministic and reproducible benchmarks
    std::mt19937 gen(42);
    std::shuffle(indices.begin(), indices.end(), gen);
    return indices;
}

// ============================================================================
// SCENARIO 1: BURST (Mass Allocation -> Mass Deallocation)
// Simulates a market spike (e.g., macro news event).
// ============================================================================

static void BM_Burst_System(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    std::vector<OrderNode*> nodes(batch_size);

    for (auto _ : state) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            nodes[i] = new OrderNode();
            benchmark::DoNotOptimize(nodes[i]);
        }
        for (uint32_t i = 0; i < batch_size; ++i) {
            delete nodes[i];
        }
        benchmark::ClobberMemory();
    }
}

static void BM_Burst_Boost(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    std::vector<OrderNode*> nodes(batch_size);
    boost::object_pool<OrderNode> pool;

    for (auto _ : state) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            nodes[i] = pool.malloc();  // malloc skips constructor overhead
            benchmark::DoNotOptimize(nodes[i]);
        }
        for (uint32_t i = 0; i < batch_size; ++i) {
            pool.free(nodes[i]);
        }
        benchmark::ClobberMemory();
    }
}

static void BM_Burst_Titan(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    UnifiedMemoryArena arena(1, batch_size, 1024, 1, 10, 4);
    auto& pool = arena.get_pool(0);
    std::vector<Handle> handles(batch_size);

    for (auto _ : state) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            handles[i] = pool.allocate();
            benchmark::DoNotOptimize(handles[i]);
        }
        for (uint32_t i = 0; i < batch_size; ++i) {
            pool.free(handles[i]);
        }
        benchmark::ClobberMemory();
    }
}

// ============================================================================
// SCENARIO 2: CHURN (Interleaved Allocate & Free)
// Simulates steady-state market making (cancel & replace).
// Includes MEMORY MUTATION to prove L1 Cache Warmth superiority.
// ============================================================================

static void BM_Churn_System(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    for (auto _ : state) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            OrderNode* node = new OrderNode();
            node->price = i;
            node->quantity = i;
            benchmark::DoNotOptimize(node->price);
            delete node;
        }
        benchmark::ClobberMemory();
    }
}

static void BM_Churn_Boost(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    boost::object_pool<OrderNode> pool;
    for (auto _ : state) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            OrderNode* node = pool.malloc();
            node->price = i;
            node->quantity = i;
            benchmark::DoNotOptimize(node->price);
            pool.free(node);
        }
        benchmark::ClobberMemory();
    }
}

static void BM_Churn_Titan(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    UnifiedMemoryArena arena(1, 2, 1024, 1, 10, 4);  // Only need space for 1 concurrent order
    auto& pool = arena.get_pool(0);

    for (auto _ : state) {
        for (uint32_t i = 0; i < batch_size; ++i) {
            Handle h = pool.allocate();
            OrderNode& node = pool.get_node(h);
            node.price = i;
            node.quantity = i;
            benchmark::DoNotOptimize(node.price);
            pool.free(h);
        }
        benchmark::ClobberMemory();
    }
}

// ============================================================================
// SCENARIO 3: SCATTERED (Random Deallocation)
// Tests cache miss penalties and memory fragmentation resilience.
// ============================================================================

static void BM_Scattered_System(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    std::vector<OrderNode*> nodes(batch_size);
    std::vector<uint32_t> rand_idx = generate_random_indices(batch_size);

    for (auto _ : state) {
        state.PauseTiming();  // Pause timer while allocating sequentially
        for (uint32_t i = 0; i < batch_size; ++i) {
            nodes[i] = new OrderNode();
        }
        state.ResumeTiming();  // Measure only the random scattered deletion

        for (uint32_t i = 0; i < batch_size; ++i) {
            delete nodes[rand_idx[i]];
        }
        benchmark::ClobberMemory();
    }
}

static void BM_Scattered_Boost(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    std::vector<OrderNode*> nodes(batch_size);
    std::vector<uint32_t> rand_idx = generate_random_indices(batch_size);
    boost::object_pool<OrderNode> pool;

    for (auto _ : state) {
        state.PauseTiming();
        for (uint32_t i = 0; i < batch_size; ++i) {
            nodes[i] = pool.malloc();
        }
        state.ResumeTiming();

        for (uint32_t i = 0; i < batch_size; ++i) {
            pool.free(nodes[rand_idx[i]]);
        }
        benchmark::ClobberMemory();
    }
}

static void BM_Scattered_Titan(benchmark::State& state) {
    const uint32_t batch_size = state.range(0);
    UnifiedMemoryArena arena(1, batch_size, 1024, 1, 10, 4);
    auto& pool = arena.get_pool(0);
    std::vector<Handle> handles(batch_size);
    std::vector<uint32_t> rand_idx = generate_random_indices(batch_size);

    for (auto _ : state) {
        state.PauseTiming();
        for (uint32_t i = 0; i < batch_size; ++i) {
            handles[i] = pool.allocate();
        }
        state.ResumeTiming();

        // O(1) deallocation regardless of cache locality
        for (uint32_t i = 0; i < batch_size; ++i) {
            pool.free(handles[rand_idx[i]]);
        }
        benchmark::ClobberMemory();
    }
}

// ============================================================================
// Registration Setup
// Run each benchmark with: 100 (L1 cache), 10,000 (L3 cache), 1,000,000 (RAM)
// ============================================================================

#define REGISTER_BENCHMARKS(Category)                                             \
    BENCHMARK(BM_##Category##_System)->RangeMultiplier(100)->Range(100, 1000000); \
    BENCHMARK(BM_##Category##_Boost)->RangeMultiplier(100)->Range(100, 10000);    \
    BENCHMARK(BM_##Category##_Titan)->RangeMultiplier(100)->Range(100, 1000000);

REGISTER_BENCHMARKS(Burst)
REGISTER_BENCHMARKS(Churn)
REGISTER_BENCHMARKS(Scattered)

}  // namespace titan::core::bench

BENCHMARK_MAIN();