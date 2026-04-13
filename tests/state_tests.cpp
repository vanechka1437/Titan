#include <gtest/gtest.h>

#include "titan/core/state.hpp"

using namespace titan::core;

class StateTest : public ::testing::Test {
protected:
    void PrintScenario(const std::string& description) { std::cout << "\n[ SCENARIO ]: " << description << "\n"; }
};

TEST_F(StateTest, ShadowLOB_TensorExport_Padding) {
    PrintScenario("Verifying that ShadowLOB pads missing levels with zeros in the tensor.");

    ShadowLOB lob(10);
    lob.apply_delta(0, 1000, 50);  // Bid
    lob.apply_delta(1, 1010, 30);  // Ask

    std::vector<float> obs(8, -1.0f);
    lob.export_to_tensor(obs.data(), 2);

    EXPECT_EQ(obs[0], 1000.0f);
    EXPECT_EQ(obs[1], 50.0f);
    EXPECT_EQ(obs[2], 0.0f);
    EXPECT_EQ(obs[3], 0.0f);
}

TEST_F(StateTest, AgentState_BalanceUpdate_Precision) {
    PrintScenario("Testing cash and inventory updates via sync with Python tensors.");

    // Fix: AgentState requires lob_reserve_capacity in constructor
    AgentState agent(1024);
    float obs_cash = 0.0f;
    float obs_inv = 0.0f;
    agent.obs_cash_ptr = &obs_cash;
    agent.obs_inventory_ptr = &obs_inv;

    // Update balances: update_balance(cash_delta, inventory_delta)
    agent.update_balance(-1000, 10);

    EXPECT_EQ(agent.real_inventory, 10);
    EXPECT_EQ(agent.real_cash, -1000);
    EXPECT_EQ(obs_inv, 10.0f);
    EXPECT_EQ(obs_cash, -1000.0f);
}

TEST_F(StateTest, EnvironmentReset_FullCleanup) {
    PrintScenario("Ensuring EnvironmentState::reset() clears the EventBuffer and Agents.");

    EnvironmentState env;
    env.env_id = 42;
    env.current_time = 123456789;

    // Correctly initialize agent and add to environment
    AgentState a1(1024);
    a1.real_cash = 1000;
    env.agents.push_back(std::move(a1));

    env.event_buffer.push_update(0, 1000, 1);
    EXPECT_EQ(env.event_buffer.count, 1);

    env.reset();

    EXPECT_EQ(env.current_time, 0);
    EXPECT_EQ(env.event_buffer.count, 0);
    EXPECT_EQ(env.agents[0].real_cash, 0);
}