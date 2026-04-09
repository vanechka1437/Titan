#include <benchmark/benchmark.h>

#include <random>
#include <vector>

#include "titan/core/lob_state.hpp"

using namespace titan::core;

// ============================================================================
// BENCHMARK FIXTURE
// ============================================================================
class LOBFixture : public benchmark::Fixture {
public:
    OptimalLOBState lob;
    std::vector<OrderNode> raw_memory;
    std::vector<Handle> raw_handles;
    OrderPoolAllocator pool;

    void SetUp(const ::benchmark::State& state) override {
        const uint32_t capacity = 1'000'000;
        raw_memory.resize(capacity);
        raw_handles.resize(capacity);
        pool.init(raw_memory.data(), raw_handles.data(), capacity);

        lob.shift_window_to_center(1'000'000);
    }

    void TearDown(const ::benchmark::State& state) override {
        raw_memory.clear();
        raw_handles.clear();
    }

    inline OrderId create_dummy_id(Handle h) { return (1ULL << 32) | static_cast<uint32_t>(h); }
};

// ============================================================================
// 1. HOT ZONE INSERTION O(1)
// ============================================================================
BENCHMARK_F(LOBFixture, BM_AddOrder_HotZone)(benchmark::State& state) {
    Price base_price = lob.get_anchor_price() + 5000;

    for (auto _ : state) {
        state.PauseTiming();
        Handle h = pool.allocate();
        OrderId id = create_dummy_id(h);
        Price p = base_price + (h & 511);
        state.ResumeTiming();

        lob.add_order(id, p, 10, 0, pool);

        benchmark::ClobberMemory();
    }
}

// ============================================================================
// 2. COLD ZONE INSERTION O(log N)
// ============================================================================
BENCHMARK_F(LOBFixture, BM_AddOrder_ColdZone)(benchmark::State& state) {
    Price base_price = lob.get_anchor_price() + OptimalLOBState::RING_SIZE + 5000;

    for (auto _ : state) {
        state.PauseTiming();
        Handle h = pool.allocate();
        OrderId id = create_dummy_id(h);
        Price p = base_price + (h * 17 % 10000);
        state.ResumeTiming();

        lob.add_order(id, p, 10, 0, pool);

        benchmark::ClobberMemory();
    }
}

// ============================================================================
// 3. HARDWARE SCAN (get_best_bid)
// ============================================================================
BENCHMARK_F(LOBFixture, BM_GetBestBid_Scattered)(benchmark::State& state) {
    Price base = lob.get_anchor_price();
    for (int i = 0; i < 1000; ++i) {
        Handle h = pool.allocate();
        lob.add_order(create_dummy_id(h), base + (i * 13 % 10000), 10, 0, pool);
    }

    for (auto _ : state) {
        Price best = lob.get_best_bid();
        benchmark::DoNotOptimize(best);
    }
}

// ============================================================================
// 4. AMORTIZED PAGING (shift_window)
// ============================================================================
BENCHMARK_F(LOBFixture, BM_ShiftWindow_JumpOut)(benchmark::State& state) {
    Price base = lob.get_anchor_price();
    for (int i = 0; i < 5000; ++i) {
        Handle h = pool.allocate();
        lob.add_order(create_dummy_id(h), base + i, 10, 0, pool);
    }

    for (auto _ : state) {
        state.PauseTiming();
        Price target = (state.iterations() % 2 == 0) ? base + 50'000 : base;
        state.ResumeTiming();

        lob.shift_window_to_center(target);

        benchmark::ClobberMemory();
    }
}

BENCHMARK_MAIN();