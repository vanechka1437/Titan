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
    std::vector<Handle> raw_handles;

    MockOrderPoolAllocator(uint32_t size = 100000) {
        raw_memory.resize(size);
        raw_handles.resize(size);

        // Честно инициализируем базовый HFT-аллокатор нашими векторами,
        // теперь get_node() и allocate() работают по-настоящему.
        this->init(raw_memory.data(), raw_handles.data(), size);
    }

    // Helper to generate a 64-bit OrderId with a valid memory handle and generation tag
    OrderId create_order(Price p, OrderQty q, uint8_t s, uint32_t gen = 1) {
        Handle h = this->allocate();
        OrderNode& node = this->get_node(h);
        node.generation = gen;

        // Shift generation to upper 32 bits, keep handle in lower 32 bits
        OrderId id = (static_cast<uint64_t>(gen) << 32) | static_cast<uint32_t>(h);
        return id;
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
// 3. TEST CASES
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
        "get_best_bid/ask return the exact prices.");

    OrderId bid_id = pool.create_order(1000, 10, 0 /* BID */);
    OrderId ask_id = pool.create_order(1005, 10, 1 /* ASK */);

    lob.add_order(bid_id, 1000, 10, 0, pool);
    lob.add_order(ask_id, 1005, 10, 1, pool);

    EXPECT_EQ(lob.get_best_bid(), 1000);
    EXPECT_EQ(lob.get_best_ask(), 1005);
}

TEST_F(LOBStateTest, CancelMiddleNodeIntrusiveList) {
    PrintScenario(
        "Attacking the intrusive doubly-linked list. We insert 3 orders at the same price (creating a FIFO queue) and "
        "delete the middle one. The pointers MUST stitch together correctly without breaking the queue or "
        "extinguishing the bitmask prematurely.");

    OrderId id1 = pool.create_order(1000, 10, 0);
    OrderId id2 = pool.create_order(1000, 20, 0);  // Target for assassination
    OrderId id3 = pool.create_order(1000, 30, 0);

    lob.add_order(id1, 1000, 10, 0, pool);
    lob.add_order(id2, 1000, 20, 0, pool);
    lob.add_order(id3, 1000, 30, 0, pool);

    // Remove middle node
    lob.cancel_order(id2, pool);
    EXPECT_EQ(lob.get_best_bid(), 1000);

    // Remove head node
    lob.cancel_order(id1, pool);
    EXPECT_EQ(lob.get_best_bid(), 1000);

    // Remove tail node. Now the mask must be extinguished.
    lob.cancel_order(id3, pool);
    EXPECT_EQ(lob.get_best_bid(), 0);
}

TEST_F(LOBStateTest, ABAMemoryReuseProtection) {
    PrintScenario(
        "Testing ABA problem protection. Simulating a scenario where an order is freed by the Matching Engine, the "
        "memory pool reuses the handle for a NEW order with a different generation tag, and a delayed cancel request "
        "arrives for the OLD order. The LOB MUST ignore the stale cancel request.");

    OrderId original_id = pool.create_order(1000, 10, 0, 1 /* Generation 1 */);
    lob.add_order(original_id, 1000, 10, 0, pool);

    // Simulate memory reuse by overriding the generation tag directly in the pool
    Handle h = original_id & 0xFFFFFFFF;
    pool.get_node(h).generation = 2;

    // Attempt to cancel using the old OrderId
    lob.cancel_order(original_id, pool);

    // The new order must survive
    EXPECT_EQ(lob.get_best_bid(), 1000);
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
    lob.add_order(pool.create_order(high_physical_end, 10, 0), high_physical_end, 10, 0, pool);

    // Wrapped around to the start of the physical array
    Price wrapped_higher_price = RING_SIZE + 5;
    lob.add_order(pool.create_order(wrapped_higher_price, 10, 0), wrapped_higher_price, 10, 0, pool);

    // wrapped_higher_price (index 5) is logically > high_physical_end (index 16383)
    EXPECT_EQ(lob.get_best_bid(), wrapped_higher_price);
}

TEST_F(LOBStateTest, AmortizedPagingEviction) {
    PrintScenario(
        "Testing Amortized Paging (Option 2). Simulating a massive market movement upwards. An existing order in the "
        "Hot Zone must be completely purged from the hardware bitmasks and safely transferred to the Cold Zone "
        "(Red-Black Tree).");

    lob.add_order(pool.create_order(1000, 10, 0), 1000, 10, 0, pool);

    // Jump away
    lob.shift_window_to_center(20000);

    // Fallback search should still find it
    EXPECT_EQ(lob.get_best_bid(), 1000);
}

TEST_F(LOBStateTest, AmortizedPagingAbsorption) {
    PrintScenario(
        "Testing Amortized Paging (Option 2). Simulating a market crash. The window jumps back down towards old "
        "prices. Orders resting in the Cold Zone must be instantly absorbed back into the Hot Zone, igniting the O(1) "
        "bitmasks again.");

    lob.add_order(pool.create_order(1000, 10, 0), 1000, 10, 0, pool);
    lob.shift_window_to_center(20000);  // Evict it

    // Crash back down
    lob.shift_window_to_center(1005);

    EXPECT_EQ(lob.get_best_bid(), 1000);
}

TEST_F(LOBStateTest, IntegerOverflowDoomsday) {
    PrintScenario(
        "CRITICAL: Testing the Doomsday Integer Overflow scenario. Emulating crypto assets quoted in nano-fractions. "
        "If a price approaches UINT32_MAX, shifting the window must NOT overflow the uint32_t arithmetic and cause an "
        "infinite scan loop or memory corruption.");

    Price high_price = UINT32_MAX - (RING_SIZE / 2);

    lob.shift_window_to_center(high_price);
    lob.add_order(pool.create_order(high_price, 10, 0), high_price, 10, 0, pool);

    // Scan should safely resolve without wrap-around bugs
    EXPECT_EQ(lob.get_best_bid(), high_price);
}