#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "titan/core/matching_engine.hpp"
#include "titan/core/memory.hpp"

using namespace titan::core;

// ============================================================================
// TEST FIXTURE & HELPERS
// ============================================================================
class MatchingEngineTest : public ::testing::Test {
protected:
    static constexpr uint32_t MAX_ORDERS = 10000;

    std::vector<OrderNode> raw_memory;
    std::vector<Handle> free_list;
    OrderPoolAllocator pool;

    // We allocate MatchingEngine on the heap dynamically in Setup to avoid
    // object slicing or initialization order fiascos.
    std::unique_ptr<MatchingEngine> engine;

    DefaultEventBuffer events;

    void SetUp() override {
        raw_memory.resize(MAX_ORDERS);
        free_list.resize(MAX_ORDERS);
        pool.init(raw_memory.data(), free_list.data(), MAX_ORDERS);

        engine = std::make_unique<MatchingEngine>(pool, MAX_ORDERS);
    }

    void PrintScenario(const std::string& description) {
        std::cout << "\n[--------------------------------------------------]\n"
                  << "[ SCENARIO ]: " << description << "\n"
                  << "[--------------------------------------------------]\n";
    }
};

// ============================================================================
// TEST CASES (The Engine Destroyer Suite)
// ============================================================================

TEST_F(MatchingEngineTest, PassiveOrderPlacement) {
    PrintScenario(
        "Testing basic Limit Order placement. A passive order must be added to the LOB, "
        "and a LOB_UPDATE event must be generated for the agents.");

    // process_order(order_id, owner_id, side, price, qty, out_events)
    engine->process_order(1, 100, 0 /* Bid */, 1000, 50, events);

    EXPECT_EQ(engine->get_lob().get_best_bid(), 1000);
    EXPECT_EQ(events.count, 1);

    const auto& ev = events.events[0];
    EXPECT_EQ(ev.type, MarketDataEvent::Type::LOB_UPDATE);
    EXPECT_EQ(ev.price, 1000);
    EXPECT_EQ(ev.qty_delta, 50);  // Positive delta for added liquidity
}

TEST_F(MatchingEngineTest, FullExecutionGeneratesCorrectEvents) {
    PrintScenario(
        "Testing a perfect cross (Full Execution). A Maker places an order, and a Taker "
        "fully consumes it. The engine MUST generate 2 TRADE events and 1 LOB_UPDATE (liquidity removal).");

    // 1. Maker places a Bid
    engine->process_order(1, 100, 0 /* Bid */, 1000, 50, events);
    events.clear();  // Clear the placement event

    // 2. Taker crosses the spread with an Ask
    engine->process_order(2, 200, 1 /* Ask */, 1000, 50, events);

    // Order book should be empty now
    EXPECT_EQ(engine->get_lob().get_best_bid(), 0);

    // We expect 3 events: Maker Trade, Taker Trade, LOB Update (removal)
    EXPECT_EQ(events.count, 3);

    bool found_maker_trade = false;
    bool found_taker_trade = false;
    bool found_lob_update = false;

    for (uint32_t i = 0; i < events.count; ++i) {
        const auto& ev = events.events[i];
        if (ev.type == MarketDataEvent::Type::TRADE) {
            if (ev.owner_id == 100)
                found_maker_trade = true;
            if (ev.owner_id == 200)
                found_taker_trade = true;

            // Taker sells (side 1) -> Cash Delta should be positive for seller
            if (ev.owner_id == 200) {
                EXPECT_EQ(ev.cash_delta, 1000 * 50);
                EXPECT_EQ(ev.qty_delta, -50);  // Inventory decreases
            }
        } else if (ev.type == MarketDataEvent::Type::LOB_UPDATE) {
            found_lob_update = true;
            EXPECT_EQ(ev.qty_delta, -50);  // Liquidity removed
        }
    }

    EXPECT_TRUE(found_maker_trade);
    EXPECT_TRUE(found_taker_trade);
    EXPECT_TRUE(found_lob_update);
}

TEST_F(MatchingEngineTest, PythonSegfaultProtection) {
    PrintScenario(
        "Attacking the O(1) Vector Mapping. Python sends an order_id that massively exceeds "
        "the allocated capacity of the order_map_. The engine MUST silently reject it, "
        "protecting the C++ runtime from a Segmentation Fault.");

    uint64_t malicious_id = 999999999ULL;  // Way beyond MAX_ORDERS (10000)

    // Should NOT crash
    engine->process_order(malicious_id, 100, 0, 1000, 50, events);
    engine->process_cancel(malicious_id, events);

    // Book must remain untouched
    EXPECT_EQ(engine->get_lob().get_best_bid(), 0);
    EXPECT_EQ(events.count, 0);
}

TEST_F(MatchingEngineTest, SentinelMarketOrdersDoNotRest) {
    PrintScenario(
        "Testing Market Orders via Sentinel Prices (0 for Sell, UINT32_MAX for Buy). "
        "A Market Order should sweep available liquidity, but its UNFILLED remainder "
        "must EVAPORATE, never entering the LOB as a passive order.");

    // Add some liquidity: 10 lots at 1000
    engine->process_order(1, 100, 0 /* Bid */, 1000, 10, events);
    events.clear();

    // Market Sell for 100 lots (price = 0)
    engine->process_order(2, 200, 1 /* Ask */, 0 /* Market Sentinel */, 100, events);

    // 10 lots matched, 90 lots remaining.
    // The book MUST be empty, those 90 lots should NOT be resting at price = 0.
    EXPECT_EQ(engine->get_lob().get_best_ask(), UINT32_MAX);  // Empty
    EXPECT_EQ(engine->get_lob().get_best_bid(), 0);           // Empty
}

TEST_F(MatchingEngineTest, WashTradingPrevention) {
    PrintScenario(
        "Testing Self-Trade Prevention (Wash Trading). An agent attempts to match against "
        "their own resting order to artificially inflate volume. The Engine MUST cancel "
        "the resting order, generate NO trades, and drop the incoming aggressive quantity.");

    // Agent 100 places a Bid
    engine->process_order(1, 100, 0 /* Bid */, 1000, 50, events);
    events.clear();

    // Agent 100 attacks their own Bid with a Market Sell
    engine->process_order(2, 100, 1 /* Ask */, 0, 50, events);

    // The order should be gone
    EXPECT_EQ(engine->get_lob().get_best_bid(), 0);

    // We expect exactly ONE event: LOB_UPDATE indicating the resting order was cancelled.
    // There MUST NOT be any TRADE events (no fake volume generated).
    EXPECT_EQ(events.count, 1);
    EXPECT_EQ(events.events[0].type, MarketDataEvent::Type::LOB_UPDATE);
    EXPECT_EQ(events.events[0].qty_delta, -50);
}

TEST_F(MatchingEngineTest, SafeDoubleCancellation) {
    PrintScenario(
        "Testing the O(1) Cancellation map against duplicate requests. "
        "If a Python agent spams 'cancel order 5' multiple times, the first one succeeds "
        "and subsequent requests must be safely ignored without double-freeing memory.");

    engine->process_order(5, 100, 0 /* Bid */, 1000, 50, events);
    events.clear();

    // Valid Cancel
    engine->process_cancel(5, events);
    EXPECT_EQ(events.count, 1);  // LOB_UPDATE (removal)
    events.clear();

    // Invalid / Duplicate Cancel
    engine->process_cancel(5, events);

    // Should do absolutely nothing
    EXPECT_EQ(events.count, 0);
}

TEST_F(MatchingEngineTest, ResetClearsO1Map) {
    PrintScenario(
        "Testing RL Episode Reset. Calling reset() must completely sever the ties in the "
        "order_map_, preventing memory leaks or phantom cancellations across episodes.");

    engine->process_order(10, 100, 0, 1000, 50, events);
    EXPECT_EQ(engine->get_lob().get_best_bid(), 1000);

    // End of RL Episode
    engine->reset();

    // Next Episode: Try to cancel the order from Episode 1
    events.clear();
    engine->process_cancel(10, events);

    // It should be ignored, because reset() wiped the mapping
    EXPECT_EQ(events.count, 0);
}