#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "titan/core/lob_state.hpp"
#include "titan/core/memory.hpp"

using namespace titan::core;

// ============================================================================
// 1. MOCK ALLOCATOR FOR ISOLATED TESTING
// ============================================================================
class MockOrderPoolAllocator : public OrderPoolAllocator {
public:
    std::vector<OrderNode> raw_memory;
    std::vector<Handle> free_list;

    MockOrderPoolAllocator(uint32_t size = 100000) {
        raw_memory.resize(size);
        free_list.resize(size);

        // Honestly initialize the underlying HFT allocator
        this->init(raw_memory.data(), free_list.data(), size);
    }
};

// ============================================================================
// 2. TEST FIXTURE
// ============================================================================
class LOBStateTest : public ::testing::Test {
protected:
    OptimalLOBState lob;
    MockOrderPoolAllocator pool;

    // Extracted from the static constexpr inside the LOBState
    static constexpr uint32_t RING_SIZE = OptimalLOBState::RING_SIZE;

    // Helper method to print test descriptions nicely to the console
    void PrintScenario(const std::string& description) {
        std::cout << "\n[--------------------------------------------------]\n"
                  << "[ SCENARIO ]: " << description << "\n"
                  << "[--------------------------------------------------]\n";
    }
};

// ============================================================================
// 3. TEST CASES (The Destroyer Suite)
// ============================================================================

TEST_F(LOBStateTest, EmptyBookReturnsLimits) {
    PrintScenario(
        "Testing initialization. An empty order book must return 0 for Best Bid and UINT32_MAX for Best Ask to prevent "
        "matching engine from crossing non-existent liquidity.");

    EXPECT_EQ(lob.get_best_bid(), 0);
    EXPECT_EQ(lob.get_best_ask(), UINT32_MAX);
}

TEST_F(LOBStateTest, BasicO1Insertion) {
    PrintScenario(
        "Testing basic O(1) insertion into the Hot Zone. Ensuring that the bitmasks are correctly ignited and "
        "get_best_bid/ask return the exact prices using full 64-bit OrderIDs.");

    uint64_t bid_id = 5000000000ULL;  // Exceeds 32-bit limit to test uint64_t safety
    uint64_t ask_id = 5000000001ULL;

    lob.add_order(bid_id, 1, 1000, 10, 0, pool);
    lob.add_order(ask_id, 2, 1005, 10, 1, pool);

    EXPECT_EQ(lob.get_best_bid(), 1000);
    EXPECT_EQ(lob.get_best_ask(), 1005);
}

TEST_F(LOBStateTest, IntrusiveListAssassination) {
    PrintScenario(
        "Attacking the intrusive doubly-linked list. We insert 3 orders at the same price (creating a FIFO queue) and "
        "delete the middle one. The pointers MUST stitch together correctly without breaking the queue or "
        "extinguishing the bitmask prematurely.");

    Handle h1 = lob.add_order(1, 1, 1000, 10, 0, pool);
    Handle h2 = lob.add_order(2, 1, 1000, 20, 0, pool);  // Target for assassination
    Handle h3 = lob.add_order(3, 1, 1000, 30, 0, pool);

    // Remove middle node
    lob.remove_order(h2, pool);
    EXPECT_EQ(lob.get_best_bid(), 1000);

    // Remove head node
    lob.remove_order(h1, pool);
    EXPECT_EQ(lob.get_best_bid(), 1000);

    // Remove tail node. Now the mask must be extinguished.
    lob.remove_order(h3, pool);
    EXPECT_EQ(lob.get_best_bid(), 0);
}

TEST_F(LOBStateTest, ThePhantomLOBProtection) {
    PrintScenario(
        "Testing the Phantom LOB synchronization fix (reduce_level_qty). "
        "Simulating a full execution where the engine reduces the level qty FIRST, "
        "then removes the physical order when its volume reaches 0. The bitmask must "
        "safely extinguish without underflowing or leaving ghost liquidity.");

    Handle h1 = lob.add_order(1, 1, 1000, 10, 0, pool);

    // 1. Engine matches 10 lots. It calls reduce_level_qty FIRST.
    lob.reduce_level_qty(0, 1000, 10);

    // The mask should immediately extinguish
    EXPECT_EQ(lob.get_best_bid(), 0);

    // 2. Engine realizes order is empty, sets its internal qty to 0, and removes it
    pool.get_node(h1).quantity = 0;
    lob.remove_order(h1, pool);

    // Best bid must STILL be 0, no math anomalies allowed
    EXPECT_EQ(lob.get_best_bid(), 0);
}

TEST_F(LOBStateTest, NegativeUnderflowClamping) {
    PrintScenario(
        "Testing the int64_t total_qty fix. Deliberately attacking the engine by requesting "
        "a reduction larger than the available liquidity. The system must clamp to 0 and "
        "extinguish the mask, rather than underflowing into 18 quintillion.");

    Handle h1 = lob.add_order(1, 1, 1000, 50, 0, pool);

    // Attack: Attempt to reduce by 100, while only 50 exists
    lob.reduce_level_qty(0, 1000, 100);

    // The sign integer logic should clamp total_qty to 0, killing the level
    EXPECT_EQ(lob.get_best_bid(), 0);
}

TEST_F(LOBStateTest, PhysicalBufferWrapAround) {
    PrintScenario(
        "Testing the physical limits of the Ring Buffer. Forcing the logical price window to wrap around the physical "
        "boundaries of the array. The backwards scan must mathematically map higher wrapped prices before lower "
        "non-wrapped prices.");

    Price anchor = RING_SIZE - 100;
    lob.shift_window_to_center(anchor + (RING_SIZE / 2));

    // End of physical array
    Price high_physical_end = RING_SIZE - 1;
    lob.add_order(1, 1, high_physical_end, 10, 0, pool);

    // Wrapped around to the start of the physical array
    Price wrapped_higher_price = RING_SIZE + 5;
    lob.add_order(2, 1, wrapped_higher_price, 10, 0, pool);

    // wrapped_higher_price (index 5) is logically > high_physical_end (index 16383)
    EXPECT_EQ(lob.get_best_bid(), wrapped_higher_price);
}

TEST_F(LOBStateTest, FlatMapAmortizedEviction) {
    PrintScenario(
        "Testing Amortized Paging Eviction. Simulating a massive market movement upwards. "
        "An existing order in the Hot Zone must be completely purged from the hardware bitmasks "
        "and safely transferred to the Cold Zone (Boost Flat Map).");

    lob.add_order(1, 1, 1000, 10, 0, pool);

    // Jump away (Eviction triggered)
    lob.shift_window_to_center(20000);

    // Fallback search should still find it in the flat_map
    EXPECT_EQ(lob.get_best_bid(), 1000);
}

TEST_F(LOBStateTest, FlatMapBulkAbsorption) {
    PrintScenario(
        "Testing O(N) Flat Map Absorption. Simulating a market crash. The window jumps back down towards old "
        "prices. Multiple orders resting in the Cold Zone must be instantly absorbed back into the Hot Zone, "
        "igniting the O(1) bitmasks again using the optimized range-erase.");

    lob.shift_window_to_center(20000);  // Move away first

    // Populate Cold Zone directly (Market is at 20000, these go to flat_map)
    lob.add_order(1, 1, 1000, 10, 0, pool);
    lob.add_order(2, 1, 1001, 10, 0, pool);
    lob.add_order(3, 1, 1002, 10, 0, pool);

    EXPECT_EQ(lob.get_best_bid(), 1002);

    // Crash back down: Bulk absorb [1000, 1001, 1002] in one O(N) swoop
    lob.shift_window_to_center(1005);

    EXPECT_EQ(lob.get_best_bid(), 1002);
}

TEST_F(LOBStateTest, IntegerOverflowDoomsday) {
    PrintScenario(
        "CRITICAL: Testing the Doomsday Integer Overflow scenario. Emulating crypto assets quoted in nano-fractions. "
        "If a price approaches UINT32_MAX, shifting the window must NOT overflow the uint32_t arithmetic and cause an "
        "infinite scan loop or memory corruption.");

    Price high_price = UINT32_MAX - (RING_SIZE / 2);

    lob.shift_window_to_center(high_price);
    lob.add_order(1, 1, high_price, 10, 0, pool);

    // Scan should safely resolve without wrap-around bugs
    EXPECT_EQ(lob.get_best_bid(), high_price);
}