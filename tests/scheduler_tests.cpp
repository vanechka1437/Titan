#include <gtest/gtest.h>

#include "titan/core/scheduler.hpp"

using namespace titan::core;

class SchedulerTest : public ::testing::Test {
protected:
    void PrintScenario(const std::string& description) { std::cout << "\n[ SCENARIO ]: " << description << "\n"; }
};

TEST_F(SchedulerTest, ChronologicalOrdering) {
    PrintScenario("Ensuring events are popped in strict nanosecond order.");

    // Fix: Pass max_capacity to constructor
    Fast4AryHeap heap(1024);

    // Fix: Use push(time, payload_idx) signature
    heap.push(500, 1);  // Time 500, Payload index 1
    heap.push(100, 2);  // Time 100, Payload index 2
    heap.push(300, 3);  // Time 300, Payload index 3

    auto e1 = heap.top();
    EXPECT_EQ(e1.arrival_time, 100);
    EXPECT_EQ(e1.payload_idx, 2);
    heap.pop();

    auto e2 = heap.top();
    EXPECT_EQ(e2.arrival_time, 300);
    EXPECT_EQ(e2.payload_idx, 3);
    heap.pop();

    auto e3 = heap.top();
    EXPECT_EQ(e3.arrival_time, 500);
    EXPECT_EQ(e3.payload_idx, 1);
    heap.pop();
}

TEST_F(SchedulerTest, HeapResetAndReuse) {
    PrintScenario("Verifying that clear() allows immediate deterministic reuse.");

    Fast4AryHeap heap(128);
    heap.push(100, 1);
    heap.clear();
    EXPECT_TRUE(heap.empty());

    heap.push(50, 2);
    auto e = heap.top();
    EXPECT_EQ(e.arrival_time, 50);
}