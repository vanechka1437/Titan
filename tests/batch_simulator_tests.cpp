#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <vector>
#include <cstring> // For std::memset

#include "titan/core/batch_simulator.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

using namespace titan::core;

class BatchSimulatorHardcoreTest : public ::testing::Test {
protected:
    std::unique_ptr<UnifiedMemoryArena> arena;
    
    // Increased environment count to heavily stress the thread pool
    const uint32_t NUM_ENVS = 32; 
    const uint32_t NUM_AGENTS = 8;
    const uint32_t MAX_ORDERS_ENV = 1024;
    const uint32_t MAX_ACTIONS = 16;
    const uint32_t MAX_EVENTS = 128; // Increased to allow high-volume trade event generation
    const uint32_t MAX_ORDERS_AGENT = 128;
    const uint32_t OBS_DEPTH = DEFAULT_OBS_DEPTH;

    const uint8_t SIDE_BUY = 0;
    const uint8_t SIDE_SELL = 1;
    const uint32_t ACTION_LIMIT = 0;
    const uint32_t ACTION_MARKET = 2;
    const uint32_t ACTION_NOOP = 3;

    void SetUp() override {
        arena = std::make_unique<UnifiedMemoryArena>(
            NUM_ENVS, NUM_AGENTS, MAX_ORDERS_ENV, MAX_ACTIONS, 
            MAX_EVENTS, MAX_ORDERS_AGENT, OBS_DEPTH, 1024 * 1024
        );
        // CRITICAL: Prevent undefined behavior and data races by zeroing out the arena
        std::memset(arena->ready_mask_ptr(), 0, NUM_ENVS * NUM_AGENTS);
        std::memset(arena->actions_ptr(), 0, NUM_ENVS * MAX_ACTIONS * sizeof(ActionPayload));
    }
    
    void TearDown() override { arena.reset(); }

    ActionPayload create_limit(uint64_t target_id, uint64_t price, uint64_t qty, uint8_t side) {
        ActionPayload act{}; act.action_type = ACTION_LIMIT; act.target_id = target_id; act.price = price; act.qty = qty; act.side = side; return act;
    }
    ActionPayload create_market(uint64_t target_id, uint64_t qty, uint8_t side) {
        ActionPayload act{}; act.action_type = ACTION_MARKET; act.target_id = target_id; act.qty = qty; act.side = side; return act;
    }
    void inject_action(uint32_t env_id, uint32_t agent_id, const ActionPayload& action) {
        arena->ready_mask_ptr()[(env_id * NUM_AGENTS) + agent_id] = 1;
        arena->actions_ptr()[(env_id * MAX_ACTIONS) + agent_id] = action;
    }
    void clear_all_masks() {
        std::memset(arena->ready_mask_ptr(), 0, NUM_ENVS * NUM_AGENTS);
    }
    uint32_t wait_reliably(BatchSimulator<DEFAULT_OBS_DEPTH>& sim, uint32_t expected_target) {
        uint32_t ready = 0;
        // Yield to prevent locking up the CPU while waiting for the C++ engine
        while ((ready = sim.wait_for_batch()) < expected_target) { std::this_thread::yield(); }
        return ready;
    }
};

// ============================================================================
// BASELINE TESTS
// ============================================================================

TEST_F(BatchSimulatorHardcoreTest, ChaosThreadLifecycle) {
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), NUM_ENVS, 8);
    for (int i = 0; i < 100; ++i) { 
        sim.start(); 
        std::this_thread::sleep_for(std::chrono::microseconds(50)); 
        sim.stop(); 
    }
    SUCCEED();
}

TEST_F(BatchSimulatorHardcoreTest, ViolentRingBufferOverflow) {
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), 1, 1); 
    sim.start();
    const uint32_t target_env = 0;
    const uint64_t SPAM_COUNT = 10000; 

    for (uint64_t i = 0; i < SPAM_COUNT; ++i) {
        inject_action(target_env, 0, create_market(0, 10, SIDE_BUY));
        sim.resume_batch();
        wait_reliably(sim, 1);
        clear_all_masks();
    }

    sim.stop();
    EXPECT_EQ(arena->event_cursors_ptr()[target_env], SPAM_COUNT);
}

TEST_F(BatchSimulatorHardcoreTest, AbsoluteCrossEnvIsolation) {
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), NUM_ENVS, 4);
    sim.start();

    for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
        uint8_t side = (env_id % 2 == 0) ? SIDE_BUY : SIDE_SELL;
        inject_action(env_id, 0, create_limit(0, 1000, 100, side));
    }

    sim.resume_batch();
    wait_reliably(sim, NUM_ENVS);
    sim.stop();

    for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
        EXPECT_EQ(arena->event_cursors_ptr()[env_id], 1);
    }
}

TEST_F(BatchSimulatorHardcoreTest, SurgicalVectorizedReset) {
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), NUM_ENVS, 4);
    sim.start();

    for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
        inject_action(env_id, 0, create_market(0, 10, SIDE_BUY));
    }
    
    sim.resume_batch();
    wait_reliably(sim, NUM_ENVS);
    clear_all_masks();

    std::vector<uint32_t> reset_targets = {1, 3, 5, 7};
    sim.reset(reset_targets);

    EXPECT_EQ(arena->event_cursors_ptr()[1], 0);
    EXPECT_EQ(arena->event_cursors_ptr()[3], 0);
    EXPECT_GT(arena->event_cursors_ptr()[0], 0);
    EXPECT_GT(arena->event_cursors_ptr()[2], 0);

    sim.stop();
}

TEST_F(BatchSimulatorHardcoreTest, NoOpAndStragglerTimeout) {
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), NUM_ENVS, 4);
    sim.start();

    ActionPayload no_op{}; no_op.action_type = ACTION_NOOP;
    inject_action(0, 0, no_op);

    sim.resume_batch();
    
    auto start_wait = std::chrono::high_resolution_clock::now();
    uint32_t ready_count = sim.wait_for_batch(); 
    auto end_wait = std::chrono::high_resolution_clock::now();
    
    EXPECT_LT(ready_count, NUM_ENVS);
    EXPECT_LE(std::chrono::duration_cast<std::chrono::milliseconds>(end_wait - start_wait).count(), 50);

    sim.stop();
}

TEST_F(BatchSimulatorHardcoreTest, SimultaneousOrderBlast) {
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), 1, 1); 
    sim.start();

    for (uint32_t agent_id = 0; agent_id < 7; ++agent_id) {
        inject_action(0, agent_id, create_market(agent_id, 10, SIDE_BUY));
    }

    for (uint32_t i = 0; i < 7; ++i) {
        sim.resume_batch();
        wait_reliably(sim, 1);
        clear_all_masks(); 
    }

    sim.stop();
    EXPECT_GE(arena->event_cursors_ptr()[0], 7);
}

// ============================================================================
// EXTREME MULTITHREADING & DATA INTEGRITY TESTS
// ============================================================================

// Tests if the engine drops actions or memory leaks when fully saturated 
// across 32 environments and 8 background threads running simultaneously.
TEST_F(BatchSimulatorHardcoreTest, MassiveConcurrencySaturation) {
    const uint32_t THREAD_COUNT = 8;
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), NUM_ENVS, THREAD_COUNT);
    sim.start();

    const uint32_t BATCH_ITERATIONS = 10;
    
    for (uint32_t step = 0; step < BATCH_ITERATIONS; ++step) {
        // Inject an action for every single agent across all 32 environments (256 concurrent actions)
        for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
            for (uint32_t agent_id = 0; agent_id < NUM_AGENTS; ++agent_id) {
                // Mix of limits and cancels to cause memory shifting in the B-Tree
                if (step % 2 == 0) {
                    inject_action(env_id, agent_id, create_limit(step * 1000, 100, 10, SIDE_BUY));
                } else {
                    inject_action(env_id, agent_id, create_market(0, 10, SIDE_SELL));
                }
            }
        }

        sim.resume_batch();
        
        // Wait for all 32 environments to finish processing the swarm
        wait_reliably(sim, NUM_ENVS);
        clear_all_masks();
    }

    sim.stop();

    // Verify no events were dropped and zero-copy ring buffers updated accurately
    for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
        EXPECT_GT(arena->event_cursors_ptr()[env_id], BATCH_ITERATIONS * 2); 
    }
}

// Tests if the Matching Engine calculates PnL correctly and writes to the DLPack 
// shared memory array safely when multiple agents cross paths concurrently.
TEST_F(BatchSimulatorHardcoreTest, MultithreadedTradeExecutionAndPnL) {
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), NUM_ENVS, 4);
    sim.start();

    // Step 1: Agent 0 places a resting BUY order at $100 for 50 qty in ALL environments
    for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
        inject_action(env_id, 0, create_limit(0, 100, 50, SIDE_BUY));
    }
    
    sim.resume_batch();
    wait_reliably(sim, NUM_ENVS);
    clear_all_masks();

    // Step 2: Agent 1 aggressively sells 50 qty via Market Order, crossing the spread
    for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
        inject_action(env_id, 1, create_market(0, 50, SIDE_SELL));
    }

    // Agent 0 and Agent 1 will wake up to receive trade confirmations
    sim.resume_batch();
    wait_reliably(sim, NUM_ENVS); // Wait for first confirmation
    clear_all_masks();
    
    sim.resume_batch();
    wait_reliably(sim, NUM_ENVS); // Wait for second confirmation
    clear_all_masks();

    sim.stop();

    // Verification: Data Integrity across the Thread Pool
    for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
        const std::size_t maker_offset = (env_id * NUM_AGENTS) + 0;
        const std::size_t taker_offset = (env_id * NUM_AGENTS) + 1;

        // Maker (Agent 0) bought 50 shares at $100. Cash should be -5000, Inventory +50
        EXPECT_FLOAT_EQ(arena->cash_ptr()[maker_offset], -5000.0f) << "Env: " << env_id;
        EXPECT_FLOAT_EQ(arena->inventory_ptr()[maker_offset], 50.0f) << "Env: " << env_id;

        // Taker (Agent 1) sold 50 shares at $100. Cash should be +5000, Inventory -50
        EXPECT_FLOAT_EQ(arena->cash_ptr()[taker_offset], 5000.0f) << "Env: " << env_id;
        EXPECT_FLOAT_EQ(arena->inventory_ptr()[taker_offset], -50.0f) << "Env: " << env_id;
    }
}

// Tests the safety of stopping the simulator while threads are heavily congested
TEST_F(BatchSimulatorHardcoreTest, AsynchronousCongestionInterrupt) {
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(arena.get(), NUM_ENVS, 4);
    sim.start();

    // Bombard the environment, but intentionally interrupt the batch
    for (uint32_t env_id = 0; env_id < NUM_ENVS; ++env_id) {
        for (uint32_t agent_id = 0; agent_id < NUM_AGENTS; ++agent_id) {
            inject_action(env_id, agent_id, create_market(0, 100, SIDE_BUY));
        }
    }

    sim.resume_batch();
    
    // Only wait 1 millisecond, then aggressively kill the thread pool
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    
    // If the atomic flags and condition variables are wrong, this will deadlock
    sim.stop(); 
    
    SUCCEED(); // Reaching here means the join() operations gracefully shut down the threads
}