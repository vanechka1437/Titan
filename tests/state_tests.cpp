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
// 3. SLIDING WINDOW: RIGHT SHIFT & EVICTION TO COLD ZONE
// ============================================================================
TEST_F(StateTest, ShadowLOB_RecenterRight_Eviction) {
    PrintScenario("Testing window shift during an upward trend. Old prices must safely move to Cold Zone.");

    ShadowLOB<4, 1024> lob;

    // Anchor = max(0, 1000 - 512) = 488. Window = [488, 1511]
    lob.apply_delta(0, 1000, 10);
    lob.apply_delta(0, 1400, 20);

    // Trigger right shift. Target 1800.
    // New Anchor = 1800 - 512 = 1288. Window = [1288, 2311]
    // 1000 falls out of the new Hot Zone and is safely evicted to the Cold Zone.
    lob.apply_delta(0, 1800, 30);

    std::vector<float> obs(16, -1.0f);
    lob.export_to_tensor(obs.data());

    // Export reads Hot Zone first, then supplements with Cold Zone
    EXPECT_EQ(obs[0], 1800.0f);
    EXPECT_EQ(obs[1], 30.0f);
    EXPECT_EQ(obs[2], 1400.0f);
    EXPECT_EQ(obs[3], 20.0f);
    EXPECT_EQ(obs[4], 1000.0f);
    EXPECT_EQ(obs[5], 10.0f);  // Recovered from Cold Zone!
    EXPECT_EQ(obs[6], 0.0f);
    EXPECT_EQ(obs[7], 0.0f);
}

// ============================================================================
// 4. DEEP PASSIVE LIQUIDITY (NO-RECENTER OPTIMIZATION)
// ============================================================================
TEST_F(StateTest, ShadowLOB_DeepPassive_DirectColdZone) {
    PrintScenario("Testing that deep passive orders go directly to Cold Zone without expensive memmoves.");

    ShadowLOB<4, 1024> lob;

    // Anchor = 2000 - 512 = 1488. Window = [1488, 2511]
    lob.apply_delta(0, 2000, 10);
    lob.apply_delta(0, 1600, 20);

    // This is a deep bid (below the Hot Zone).
    // It should NOT trigger a recenter, but be placed in the Cold Zone directly.
    lob.apply_delta(0, 1000, 30);

    std::vector<float> obs(16, -1.0f);
    lob.export_to_tensor(obs.data());

    // 2000 and 1600 from Hot Zone, 1000 from Cold Zone fallback.
    EXPECT_EQ(obs[0], 2000.0f);
    EXPECT_EQ(obs[1], 10.0f);
    EXPECT_EQ(obs[2], 1600.0f);
    EXPECT_EQ(obs[3], 20.0f);
    EXPECT_EQ(obs[4], 1000.0f);
    EXPECT_EQ(obs[5], 30.0f);  // Safely tracked
    EXPECT_EQ(obs[6], 0.0f);
    EXPECT_EQ(obs[7], 0.0f);
}

// ============================================================================
// 5. SLIDING WINDOW: EXTREME GAP (FLASH CRASH)
// ============================================================================
TEST_F(StateTest, ShadowLOB_ExtremeGap_ColdZoneSurvival) {
    PrintScenario("Testing massive price jump > WindowSize. Must clear arrays but preserve liquidity in Cold Zone.");

    ShadowLOB<4, 1024> lob;

    // Anchor = 1000 - 512 = 488. Window = [488, 1511]
    lob.apply_delta(0, 1000, 10);

    // Jump by ~100,000. Offset is larger than WindowSize.
    // Triggers memset clear, but 1000 must survive by being moved to Cold Zone.
    lob.apply_delta(0, 100000, 50);

    std::vector<float> obs(16, -1.0f);
    lob.export_to_tensor(obs.data());

    EXPECT_EQ(obs[0], 100000.0f);
    EXPECT_EQ(obs[1], 50.0f);  // Hot Zone
    EXPECT_EQ(obs[2], 1000.0f);
    EXPECT_EQ(obs[3], 10.0f);  // Saved by Cold Zone
    EXPECT_EQ(obs[4], 0.0f);
    EXPECT_EQ(obs[5], 0.0f);
}

// ============================================================================
// 6. COLD ZONE ABSORPTION (MEAN REVERSION)
// ============================================================================
TEST_F(StateTest, ShadowLOB_ColdZone_Absorption) {
    PrintScenario("Testing liquidity returning from Cold Zone back to Hot Zone.");

    ShadowLOB<4, 1024> lob;

    lob.apply_delta(0, 1000, 10);  // Anchor=488. Hot=[488, 1511]

    // FIX: Using side=1 (ASK) for the 5000 price level as intended by the scenario.
    lob.apply_delta(1, 5000, 20);  // Recenter. Anchor=4488. Hot=[4488, 5511]. 1000 -> Cold Zone

    // Bring market back down
    // Ask at 1200 forces recenter down. Anchor = 1200 - 512 = 688. Hot=[688, 1711]
    // Bid 1000 (currently in Cold Zone) should be naturally absorbed into the Hot Zone arrays!
    // Ask 5000 (currently in Hot Zone) should be evicted to Cold Zone!
    lob.apply_delta(1, 1200, 30);

    std::vector<float> obs(16, -1.0f);
    lob.export_to_tensor(obs.data());

    // Bids: 1000 from Hot Zone (successfully absorbed!)
    EXPECT_EQ(obs[0], 1000.0f);
    EXPECT_EQ(obs[1], 10.0f);
    EXPECT_EQ(obs[2], 0.0f);
    EXPECT_EQ(obs[3], 0.0f);

    // Asks: 1200 from Hot Zone, 5000 from Cold Zone
    EXPECT_EQ(obs[8], 1200.0f);
    EXPECT_EQ(obs[9], 30.0f);
    EXPECT_EQ(obs[10], 5000.0f);
    EXPECT_EQ(obs[11], 20.0f);
}

// ============================================================================
// 7. BITMASK BOUNDARY DESTRUCTION TEST (L1/L2 INTEGRITY)
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
// 8. AGENT STATE EVENT STREAM PADDING & TRUNCATION
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
// 9. ENVIRONMENT RESET
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