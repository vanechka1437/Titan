#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "titan/core/memory.hpp"

using namespace titan::core;

// ============================================================================
// TEST FIXTURE & HELPERS
// ============================================================================
class MemoryCoreTest : public ::testing::Test {
protected:
    void PrintScenario(const std::string& description) {
        std::cout << "\n[--------------------------------------------------]\n"
                  << "[ SCENARIO ]: " << description << "\n"
                  << "[--------------------------------------------------]\n";
    }
};

// ============================================================================
// 1. LINEAR ALLOCATOR TESTS (Bump Allocator)
// ============================================================================

TEST_F(MemoryCoreTest, LinearAllocator_OOM_GracefulRejection) {
    PrintScenario(
        "Attacking the LinearAllocator by requesting more memory than it owns. It must return a null pointer, not "
        "trigger a buffer overflow or segmentation fault.");

    // Create a tiny allocator (64 bytes)
    LinearAllocator allocator(64);

    // Ask for 10 uint64_t (80 bytes)
    uint64_t* ptr = allocator.allocate<uint64_t>(10);

    // MUST gracefully fail
    EXPECT_EQ(ptr, nullptr);
}

TEST_F(MemoryCoreTest, LinearAllocator_AlignmentPaddingConsumption) {
    PrintScenario(
        "Testing std::align logic. Forcing the allocator to consume capacity not just for the objects, but for the "
        "alignment padding. A tightly packed buffer should OOM if extreme alignment is required.");

    // 64 bytes total.
    LinearAllocator allocator(64);

    // Allocate 1 byte. Offset becomes 1. Remaining: 63.
    auto* byte_ptr = allocator.allocate<uint8_t>(1);
    EXPECT_NE(byte_ptr, nullptr);

    // Now ask for an object that strictly requires 64-byte alignment (e.g., an AVX-512 register block).
    // The allocator will need 63 bytes of padding just to reach the next 64-byte boundary,
    // which exceeds the remaining capacity.
    struct alignas(64) HugeAlignedStruct {
        uint8_t data[1];
    };

    auto* aligned_ptr = allocator.allocate<HugeAlignedStruct>(1);

    // Must fail because std::align cannot satisfy the 64-byte boundary requirement within 64 bytes if offset > 0.
    EXPECT_EQ(aligned_ptr, nullptr);
}

// ============================================================================
// 2. ORDER POOL ALLOCATOR TESTS
// ============================================================================

TEST_F(MemoryCoreTest, OrderPool_CapacityExhaustion) {
    PrintScenario(
        "Draining the OrderPool completely. Requesting one more handle than the capacity MUST return NULL_HANDLE, "
        "protecting the engine from allocating out of bounds.");

    std::vector<OrderNode> nodes(5);
    std::vector<Handle> free_list(5);
    OrderPoolAllocator pool;
    pool.init(nodes.data(), free_list.data(), 5);

    // Drain the pool
    for (int i = 0; i < 5; ++i) {
        Handle h = pool.allocate();
        EXPECT_NE(h, NULL_HANDLE);
    }

    // Attack: Request one more
    Handle illegal_handle = pool.allocate();
    EXPECT_EQ(illegal_handle, NULL_HANDLE);
}

TEST_F(MemoryCoreTest, OrderPool_ABAGenerationIncrement) {
    PrintScenario(
        "Verifying the ABA Protection mechanism. When a node is freed, its generation tag MUST increment. When "
        "reallocated, it retains the new generation.");

    std::vector<OrderNode> nodes(10);
    std::vector<Handle> free_list(10);
    OrderPoolAllocator pool;
    pool.init(nodes.data(), free_list.data(), 10);

    // 1. Allocate and check initial generation
    Handle h1 = pool.allocate();
    EXPECT_EQ(pool.get_node(h1).generation, 0);

    // 2. Free it (simulating order cancellation)
    pool.free(h1);

    // 3. Re-allocate. Because free_list is LIFO (stack), we should get the exact same handle back.
    Handle h2 = pool.allocate();
    EXPECT_EQ(h1, h2);

    // 4. The generation MUST have incremented to protect delayed cancel requests.
    EXPECT_EQ(pool.get_node(h2).generation, 1);
}

TEST_F(MemoryCoreTest, OrderPool_TheDoubleFreeNuke) {
    PrintScenario(
        "CRITICAL VULNERABILITY DEMONSTRATION: The Double Free. If the Matching Engine accidentally frees the same "
        "handle twice, the LIFO free list is corrupted. Subsequent allocations will hand out the exact same memory to "
        "TWO DIFFERENT orders, causing circular linked-list corruption in the LOB.");

    std::vector<OrderNode> nodes(10);
    // Deliberately over-allocate the free list buffer to avoid an immediate segfault during the attack,
    // allowing us to observe the logical corruption. In the real UnifiedMemoryArena, this would
    // overwrite the free list of the *next* RL environment!
    std::vector<Handle> free_list(20);

    OrderPoolAllocator pool;
    pool.init(nodes.data(), free_list.data(), 10);

    Handle h1 = pool.allocate();

    // The Nuke: Double Free
    pool.free(h1);
    pool.free(h1);  // Logical error from the Engine

    // Now allocate two distinct orders.
    Handle victim1 = pool.allocate();
    Handle victim2 = pool.allocate();

    // Because 'h1' was pushed to the stack twice, popping it twice yields the same physical memory block!
    EXPECT_EQ(victim1, victim2);

    // The Matching Engine now thinks it has two different orders, but they write to the same `OrderNode`.
    // The Limit Order Book bitmasks and intrusive pointers will be permanently destroyed.
}

// ============================================================================
// 3. UNIFIED MEMORY ARENA TESTS
// ============================================================================

TEST_F(MemoryCoreTest, UnifiedArena_StrictEnvironmentIsolation) {
    PrintScenario(
        "Testing memory partitioning for Reinforcement Learning. Verifying that Pool 0 and Pool 1 are physically "
        "separated by exactly the requested memory boundaries, ensuring zero crosstalk between independent RL "
        "environments.");

    const uint32_t num_envs = 4;
    const uint32_t orders_per_env = 1000;
    const std::size_t linear_size = 1024;

    UnifiedMemoryArena arena(num_envs, orders_per_env, linear_size);

    OrderPoolAllocator& pool0 = arena.get_pool(0);
    OrderPoolAllocator& pool1 = arena.get_pool(1);

    // Allocate the very first handle from both pools
    // Note: Due to LIFO stack initialization (capacity - 1 - i), the first allocated handle
    // is actually handle index 0.
    Handle p0_h = pool0.allocate();
    Handle p1_h = pool1.allocate();

    OrderNode& node0 = pool0.get_node(p0_h);
    OrderNode& node1 = pool1.get_node(p1_h);

    // Calculate physical memory distance in bytes
    std::ptrdiff_t byte_distance = reinterpret_cast<const char*>(&node1) - reinterpret_cast<const char*>(&node0);

    // The physical distance MUST be exactly the size of one environment's block
    std::ptrdiff_t expected_distance = orders_per_env * sizeof(OrderNode);

    EXPECT_EQ(byte_distance, expected_distance);
}