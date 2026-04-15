#include <gtest/gtest.h>

#include <iostream>
#include <vector>

#include "titan/core/state.hpp"

using namespace titan::core;

// ============================================================================
// DUMMY EVENT BUFFER FOR AGENT STATE TESTING
// ============================================================================
struct MockEvent {
    uint64_t timestamp;
    uint8_t action_type;
    uint32_t price;
    int32_t qty;
};

// ============================================================================
// TEST FIXTURE
// ============================================================================
class StateTest : public ::testing::Test {
protected:
    void PrintScenario(const std::string& description) { std::cout << "\n[ SCENARIO ]: " << description << "\n"; }
};

// ============================================================================
// 1. BASIC LOB AND BITMASK EXPORT
// ============================================================================
TEST_F(StateTest, ShadowLOB_ExportAndPadding) {
    PrintScenario("Verifying hardware MSB/LSB scans sort prices correctly and pad empty slots.");

    // Depth = 4, Window = 1024
    ShadowLOB<4, 1024> lob;

    // Anchor will be initialized at max(0, 1000 - 512) = 488
    lob.apply_delta(0, 1000, 50);  // Bid
    lob.apply_delta(0, 1010, 30);  // Bid (Higher price, should be first)

    lob.apply_delta(1, 1020, 10);  // Ask (Lowest price, should be first)
    lob.apply_delta(1, 1030, 20);  // Ask

    std::vector<float> obs(16, -1.0f);  // 4 bids * 2 + 4 asks * 2 = 16 floats
    lob.export_to_tensor(obs.data());

    // Bids: Sorted Descending (Highest to Lowest)
    EXPECT_EQ(obs[0], 1010.0f);
    EXPECT_EQ(obs[1], 30.0f);
    EXPECT_EQ(obs[2], 1000.0f);
    EXPECT_EQ(obs[3], 50.0f);
    // Padding
    EXPECT_EQ(obs[4], 0.0f);
    EXPECT_EQ(obs[5], 0.0f);
    EXPECT_EQ(obs[6], 0.0f);
    EXPECT_EQ(obs[7], 0.0f);

    // Asks: Sorted Ascending (Lowest to Highest)
    EXPECT_EQ(obs[8], 1020.0f);
    EXPECT_EQ(obs[9], 10.0f);
    EXPECT_EQ(obs[10], 1030.0f);
    EXPECT_EQ(obs[11], 20.0f);
    // Padding
    EXPECT_EQ(obs[12], 0.0f);
    EXPECT_EQ(obs[13], 0.0f);
    EXPECT_EQ(obs[14], 0.0f);
    EXPECT_EQ(obs[15], 0.0f);
}

// ============================================================================
// 2. ZERO-QUANTITY BITMASK CLEARANCE
// ============================================================================
TEST_F(StateTest, ShadowLOB_ZeroQtyBitClearance) {
    PrintScenario("Ensuring bits are correctly cleared when level volume drops to zero.");

    ShadowLOB<2, 1024> lob;

    lob.apply_delta(0, 1000, 50);
    lob.apply_delta(0, 1010, 30);

    // Fully consume the best bid
    lob.apply_delta(0, 1010, -30);

    std::vector<float> obs(8, -1.0f);
    lob.export_to_tensor(obs.data());

    // The 1010 level should be completely gone, 1000 becomes best bid
    EXPECT_EQ(obs[0], 1000.0f);
    EXPECT_EQ(obs[1], 50.0f);
    EXPECT_EQ(obs[2], 0.0f);
    EXPECT_EQ(obs[3], 0.0f);
}

// ============================================================================
// 3. SLIDING WINDOW: RIGHT SHIFT (BULL MARKET)
// ============================================================================
TEST_F(StateTest, ShadowLOB_RecenterRight) {
    PrintScenario("Testing window shift during an upward price trend (memmove right).");

    ShadowLOB<4, 1024> lob;

    // Anchor = max(0, 1000 - 512) = 488. Window = [488, 1512)
    lob.apply_delta(0, 1000, 10);
    lob.apply_delta(0, 1400, 20);

    // Trigger right shift. Target 1800.
    // New Anchor = 1800 - 512 = 1288. Window = [1288, 2312)
    // 1000 falls out. 1400 survives. 1800 added.
    lob.apply_delta(0, 1800, 30);

    std::vector<float> obs(16, -1.0f);
    lob.export_to_tensor(obs.data());

    EXPECT_EQ(obs[0], 1800.0f);
    EXPECT_EQ(obs[1], 30.0f);
    EXPECT_EQ(obs[2], 1400.0f);
    EXPECT_EQ(obs[3], 20.0f);
    EXPECT_EQ(obs[4], 0.0f);  // 1000 was safely dropped
}

// ============================================================================
// 4. SLIDING WINDOW: LEFT SHIFT (BEAR MARKET)
// ============================================================================
TEST_F(StateTest, ShadowLOB_RecenterLeft) {
    PrintScenario("Testing window shift during a downward price crash (memmove left).");

    ShadowLOB<4, 1024> lob;

    // Anchor = 2000 - 512 = 1488. Window = [1488, 2512)
    lob.apply_delta(0, 2000, 10);
    lob.apply_delta(0, 1600, 20);

    // Trigger left shift. Target 1000.
    // New Anchor = 1000 - 512 = 488. Window = [488, 1512)
    // 2000 falls out. 1600 falls out (1600 is > 1512? No, wait. 488 + 1024 = 1512. 1600 falls out!).
    lob.apply_delta(0, 1000, 30);

    std::vector<float> obs(16, -1.0f);
    lob.export_to_tensor(obs.data());

    EXPECT_EQ(obs[0], 1000.0f);
    EXPECT_EQ(obs[1], 30.0f);
    EXPECT_EQ(obs[2], 0.0f);  // 1600 and 2000 are out of bounds and wiped
}

// ============================================================================
// 5. SLIDING WINDOW: EXTREME GAP (MARKET HALT)
// ============================================================================
TEST_F(StateTest, ShadowLOB_ExtremeRecenterWipe) {
    PrintScenario("Testing massive price jump > WindowSize. Must trigger full clear.");

    ShadowLOB<4, 1024> lob;

    // Anchor = 1000 - 512 = 488
    lob.apply_delta(0, 1000, 10);

    // Jump by 100,000. Offset is way larger than WindowSize.
    // Must trigger memset clear instead of memmove.
    lob.apply_delta(0, 100000, 50);

    std::vector<float> obs(16, -1.0f);
    lob.export_to_tensor(obs.data());

    EXPECT_EQ(obs[0], 100000.0f);
    EXPECT_EQ(obs[1], 50.0f);
    EXPECT_EQ(obs[2], 0.0f);  // Old data wiped
}

// ============================================================================
// 6. BITMASK BOUNDARY DESTRUCTION TEST (L1/L2 INTEGRITY)
// ============================================================================
TEST_F(StateTest, ShadowLOB_CrossBoundaryBitmasks) {
    PrintScenario("Testing L1/L2 mask behavior right on the 63/64 bit boundaries.");

    ShadowLOB<4, 1024> lob;

    // Target = 1000. Anchor = 488.
    // Index = Price - Anchor.
    // Price 488 + 63 = 551 (Index 63 -> L1 Block 0, highest bit)
    // Price 488 + 64 = 552 (Index 64 -> L1 Block 1, lowest bit)
    lob.apply_delta(0, 551, 10);
    lob.apply_delta(0, 552, 20);

    // Delete index 63. The L2 bit for Block 0 must go to 0.
    // L2 bit for Block 1 must remain 1.
    lob.apply_delta(0, 551, -10);

    std::vector<float> obs(16, -1.0f);
    lob.export_to_tensor(obs.data());

    // Only 552 should remain. If L2 mask bleeds, 551 would ghost, or 552 would be missed.
    EXPECT_EQ(obs[0], 552.0f);
    EXPECT_EQ(obs[1], 20.0f);
    EXPECT_EQ(obs[2], 0.0f);
}

// ============================================================================
// 7. AGENT STATE EVENT STREAM PADDING & TRUNCATION
// ============================================================================
// ============================================================================
// 7. AGENT STATE EVENT STREAM PADDING & TRUNCATION
// ============================================================================
TEST_F(StateTest, AgentState_EventStreamExport) {
    PrintScenario("Validating AgentState zero-copy event stream extraction and newest-event truncation.");

    AgentState<2> agent;

    std::vector<float> stream_tensor(16, -1.0f);  // [MAX_EVENTS=4, FEATURES=4]
    agent.obs_event_stream_ptr = stream_tensor.data();

    std::vector<MockEvent> buffer = {
        {100, 1, 5000, 10},  // Event 1 (Oldest)
        {101, 2, 5001, 5}    // Event 2 (Newest)
    };

    // 1. Padding Test: Export max 4 events. Buffer only has 2.
    agent.export_events_to_tensor(buffer, 4);

    // Event 1 (Oldest) is first
    EXPECT_EQ(stream_tensor[0], 100.0f);
    EXPECT_EQ(stream_tensor[2], 5000.0f);

    // Event 2 (Newest) is second
    EXPECT_EQ(stream_tensor[4], 101.0f);
    EXPECT_EQ(stream_tensor[6], 5001.0f);

    // Padding (Event 3 and 4)
    EXPECT_EQ(stream_tensor[8], 0.0f);
    EXPECT_EQ(stream_tensor[15], 0.0f);

    // 2. Truncation Test: Request only 1 event
    // We tell the engine the tensor is exactly 1 event (4 floats) large.
    // It must extract the NEWEST event (Event 2) and ignore the rest.
    agent.export_events_to_tensor(buffer, 1);

    // The first 4 floats must now be overwritten by Event 2!
    EXPECT_EQ(stream_tensor[0], 101.0f);
    EXPECT_EQ(stream_tensor[1], 2.0f);
    EXPECT_EQ(stream_tensor[2], 5001.0f);
    EXPECT_EQ(stream_tensor[3], 5.0f);

    // We do NOT check stream_tensor[4] because max_events=1 forbids writing there.
}

// ============================================================================
// 8. ENVIRONMENT RESET
// ============================================================================
TEST_F(StateTest, EnvironmentReset_FullCleanup) {
    PrintScenario("Ensuring EnvironmentState::reset() clears the LOBs and Time.");

    EnvironmentState<4> env;
    env.env_id = 42;
    env.current_time = 123456789;

    AgentState<4> a1;
    a1.real_cash = 1000;
    a1.shadow_lob.apply_delta(0, 1000, 10);
    env.agents.push_back(std::move(a1));

    env.reset();

    EXPECT_EQ(env.current_time, 0);
    EXPECT_EQ(env.agents[0].real_cash, 0);

    // Verify Agent's LOB is wiped
    std::vector<float> obs(16, -1.0f);
    env.agents[0].shadow_lob.export_to_tensor(obs.data());
    EXPECT_EQ(obs[0], 0.0f);  // Padded zero, indicating empty LOB
}