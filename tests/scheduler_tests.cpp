#include <gtest/gtest.h>

#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include "titan/core/scheduler.hpp"

using namespace titan::core;

// ============================================================================
// TEST FIXTURE
// ============================================================================
class SchedulerTest : public ::testing::Test {
protected:
    void PrintScenario(const std::string& description) {
        std::cout << "\n[--------------------------------------------------]\n"
                  << "[ SCENARIO ]: " << description << "\n"
                  << "[--------------------------------------------------]\n";
    }
};

// ============================================================================
// DESTRUCTIVE TEST CASES
// ============================================================================

TEST_F(SchedulerTest, ChronologicalOrdering) {
    PrintScenario("Basic Priority: Ensuring Fat Events are popped in strict chronological (nanosecond) order.");

    FastScheduler sched(1024);

    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(500, 1)));
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(100, 2)));
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(300, 3)));
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(50, 4)));

    EXPECT_EQ(sched.size(), 4u);

    EXPECT_EQ(sched.top().time, 50);
    EXPECT_EQ(sched.top().target_id, 4);
    sched.pop();

    EXPECT_EQ(sched.top().time, 100);
    EXPECT_EQ(sched.top().target_id, 2);
    sched.pop();

    EXPECT_EQ(sched.top().time, 300);
    EXPECT_EQ(sched.top().target_id, 3);
    sched.pop();

    EXPECT_EQ(sched.top().time, 500);
    EXPECT_EQ(sched.top().target_id, 1);
    sched.pop();

    EXPECT_TRUE(sched.empty());
}

TEST_F(SchedulerTest, StrictFIFOTieBreaking) {
    PrintScenario(
        "CRITICAL HFT FEATURE: Deterministic FIFO sorting for simultaneous events. "
        "If 5 events arrive at the exact same nanosecond, they MUST be processed "
        "in the exact order they were pushed to prevent non-deterministic RL environments.");

    FastScheduler sched(1024);

    sched.push(ScheduledEvent::make_agent_wakeup(1000, 10));
    sched.push(ScheduledEvent::make_agent_wakeup(1000, 11));
    sched.push(ScheduledEvent::make_agent_wakeup(1000, 12));
    sched.push(ScheduledEvent::make_agent_wakeup(1000, 13));
    sched.push(ScheduledEvent::make_agent_wakeup(1000, 14));

    for (uint32_t expected_id = 10; expected_id <= 14; ++expected_id) {
        auto ev = sched.top();
        EXPECT_EQ(ev.time, 1000);
        EXPECT_EQ(ev.target_id, expected_id);
        sched.pop();
    }
    
    EXPECT_TRUE(sched.empty());
}

TEST_F(SchedulerTest, SandboxOverflowProtection) {
    PrintScenario(
        "Sandbox Safety: Emulating an RL agent generating infinite events. "
        "Pushing beyond max_capacity must safely return false and PREVENT "
        "std::vector reallocation latency spikes.");

    const size_t CAPACITY = 5;
    FastScheduler sched(CAPACITY);

    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(10, 1)));
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(20, 2)));
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(30, 3)));
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(40, 4)));
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(50, 5)));

    EXPECT_EQ(sched.size(), CAPACITY);

    // 6th push MUST fail safely without reallocating internal vectors
    EXPECT_FALSE(sched.push(ScheduledEvent::make_agent_wakeup(60, 6)));
    EXPECT_EQ(sched.size(), CAPACITY);

    EXPECT_EQ(sched.top().time, 10);
}

TEST_F(SchedulerTest, SafeEmptyPopMemoryProtection) {
    PrintScenario(
        "Robustness: Popping an empty scheduler must not underflow the counter "
        "or corrupt the free_list_ by pushing sentinel payload indices.");

    FastScheduler sched(10);
    EXPECT_TRUE(sched.empty());

    sched.pop();  // Used to corrupt memory here
    sched.pop();

    EXPECT_TRUE(sched.empty());
    EXPECT_EQ(sched.size(), 0u);

    // Should still safely allocate and reuse pool slot 0
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(100, 99)));
    EXPECT_EQ(sched.size(), 1u);
    EXPECT_EQ(sched.top().time, 100);
    EXPECT_EQ(sched.top().target_id, 99);
}

TEST_F(SchedulerTest, SmallCapacityBufferOverflow) {
    PrintScenario(
        "Memory Integrity: Creating a scheduler with capacity < Arity (e.g. 2). "
        "The clear() loop must not overflow the allocated sentinel buffer.");

    // Arity is 4. Capacity is 2. Alloc size is 6.
    FastScheduler sched(2);
    
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(10, 1)));
    EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(20, 2)));
    
    // 3rd push should fail cleanly
    EXPECT_FALSE(sched.push(ScheduledEvent::make_agent_wakeup(30, 3)));
    
    // Clear triggers the reset loop
    sched.clear();
    
    EXPECT_TRUE(sched.empty());
    // If we reach here without a segfault or ASAN error, the fix worked.
}

TEST_F(SchedulerTest, StressTestReverseOrder) {
    PrintScenario(
        "Algorithm Stress Test: Pushing 10,000 events in completely reverse order "
        "(worst case scenario for sift_up). Ensuring the D-Ary algorithm correctly "
        "sorts them without cache-line fractures.");

    const size_t CAPACITY = 10000;
    FastScheduler sched(CAPACITY);

    // Push from 10000 down to 1
    for (uint32_t i = CAPACITY; i > 0; --i) {
        EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(i * 10, i)));
    }

    EXPECT_EQ(sched.size(), CAPACITY);

    // Verify perfect chronological extraction
    for (uint32_t expected = 1; expected <= CAPACITY; ++expected) {
        auto ev = sched.top();
        EXPECT_EQ(ev.time, expected * 10);
        EXPECT_EQ(ev.target_id, expected);
        sched.pop();
    }

    EXPECT_TRUE(sched.empty());
}

TEST_F(SchedulerTest, HeapResetAndReuseMonotonicity) {
    PrintScenario(
        "Reset Mechanics: clear() must instantly empty the heap AND reset the "
        "internal sequence_counter_ so that consecutive RL episodes behave identically.");

    FastScheduler sched(128);
    
    // Episode 1
    sched.push(ScheduledEvent::make_agent_wakeup(100, 1));
    sched.push(ScheduledEvent::make_agent_wakeup(100, 2));
    
    // Extracting sequence_id isn't directly exposed in Fat API, 
    // but FIFO ordering relies on it.
    EXPECT_EQ(sched.top().target_id, 1);
    sched.clear();
    EXPECT_TRUE(sched.empty());

    // Episode 2
    sched.push(ScheduledEvent::make_agent_wakeup(50, 99));
    sched.push(ScheduledEvent::make_agent_wakeup(50, 100));
    
    EXPECT_EQ(sched.top().target_id, 99);
    sched.pop();
    EXPECT_EQ(sched.top().target_id, 100);
}

TEST_F(SchedulerTest, FreeListReuseExhaustion) {
    PrintScenario(
        "Memory Pool Integrity: Pushing to max capacity, popping all, and pushing to max again. "
        "Ensures free_list_ doesn't lose indices and payloads_ doesn't infinitely expand.");

    const size_t CAPACITY = 1024;
    FastScheduler sched(CAPACITY);

    // Cycle 1: Fill up
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(i, i)));
    }
    EXPECT_EQ(sched.size(), CAPACITY);
    EXPECT_FALSE(sched.push(ScheduledEvent::make_agent_wakeup(9999, 0))); // Should fail

    // Cycle 1: Drain all
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        sched.pop();
    }
    EXPECT_TRUE(sched.empty());

    // Cycle 2: Fill up again
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        EXPECT_TRUE(sched.push(ScheduledEvent::make_agent_wakeup(i * 2, i)));
    }
    EXPECT_EQ(sched.size(), CAPACITY);
    EXPECT_FALSE(sched.push(ScheduledEvent::make_agent_wakeup(9999, 0))); // Should fail

    // Cycle 2: Drain all
    for (uint32_t i = 0; i < CAPACITY; ++i) {
        sched.pop();
    }
    EXPECT_TRUE(sched.empty());
}

TEST_F(SchedulerTest, ChaoticPushPopIntegrity) {
    PrintScenario(
        "Destructive Fuzzing: Chaotic mix of pushes and pops at the capacity limit. "
        "Designed to violently fragment the free_list_ and break the internal heap bounds.");

    const uint32_t CAPACITY = 500;
    FastScheduler sched(CAPACITY);
    std::mt19937_64 rng(42); // Fixed seed for reproducibility

    uint32_t active_events = 0;

    for (int i = 0; i < 100000; ++i) {
        bool should_push = (rng() % 100) < 60; // 60% chance to push

        if (should_push && active_events < CAPACITY) {
            uint64_t time = rng() % 10000;
            bool success = sched.push(ScheduledEvent::make_agent_wakeup(time, i));
            EXPECT_TRUE(success);
            active_events++;
        } else if (!sched.empty()) {
            sched.pop();
            active_events--;
        }

        EXPECT_EQ(sched.size(), active_events);
    }

    // Drain and verify no segfaults or data corruption
    uint64_t last_time = 0;
    while (!sched.empty()) {
        auto ev = sched.top();
        EXPECT_GE(ev.time, last_time); // Ensure monotonic time extraction
        last_time = ev.time;
        sched.pop();
    }
    EXPECT_EQ(sched.size(), 0u);
}

TEST_F(SchedulerTest, ZeroCapacityEdgeCase) {
    PrintScenario(
        "Extreme Edge Case: Initializing scheduler with 0 capacity. "
        "Must gracefully handle zero memory allocation without segfaults on push, pop, or clear.");

    FastScheduler sched(0);
    EXPECT_TRUE(sched.empty());
    EXPECT_EQ(sched.size(), 0u);

    // Push should fail safely
    EXPECT_FALSE(sched.push(ScheduledEvent::make_agent_wakeup(10, 1)));
    EXPECT_EQ(sched.size(), 0u);

    // Pop should do nothing safely
    sched.pop();
    EXPECT_EQ(sched.size(), 0u);

    // Clear should do nothing safely
    sched.clear();
    EXPECT_TRUE(sched.empty());
}