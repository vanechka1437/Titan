#include <benchmark/benchmark.h>
#include <vector>
#include <cstring>
#include <thread>

#include "titan/core/batch_simulator.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/types.hpp"

using namespace titan::core;

// ============================================================================
// HELPER: Pre-fill Arena with actions to avoid measuring setup time
// ============================================================================
void populate_actions(UnifiedMemoryArena& arena, uint32_t num_envs, uint32_t num_agents, uint32_t step) {
    uint8_t* ready_mask = arena.ready_mask_ptr();
    ActionPayload* actions = arena.actions_ptr();
    const uint32_t max_actions = arena.max_actions_per_step();

    std::memset(ready_mask, 0, num_envs * num_agents);

    for (uint32_t env_id = 0; env_id < num_envs; ++env_id) {
        for (uint32_t agent_id = 0; agent_id < num_agents; ++agent_id) {
            ready_mask[(env_id * num_agents) + agent_id] = 1;
            
            ActionPayload act{};
            act.target_id = (env_id * 1000) + agent_id; 
            act.qty = 10;
            
            if (step % 3 == 0) {
                act.action_type = 2; // MARKET
                act.side = (agent_id % 2 == 0) ? 0 : 1; 
            } else {
                act.action_type = 0; // LIMIT
                act.side = (agent_id % 2 == 0) ? 0 : 1;
                act.price = 10000 + (agent_id * 10) * ((act.side == 0) ? -1 : 1);
            }

            actions[(env_id * max_actions) + agent_id] = act;
        }
    }
}

// ============================================================================
// BENCHMARK 1: Core Engine Throughput & Thread Scaling (Target: 1024 Envs)
// ============================================================================
static void BM_BatchSimulator_ThroughputScaling(benchmark::State& state) {
    const uint32_t num_threads = state.range(0);
    const uint32_t num_envs = 1024; // THE REAL RL SCALE
    const uint32_t num_agents = 8;
    const uint32_t max_orders_env = 4096;
    const uint32_t max_events = 256;
    
    // 256 MB buffer to comfortably hold 1024 environments
    UnifiedMemoryArena arena(num_envs, num_agents, max_orders_env, 16, max_events, 256, DEFAULT_OBS_DEPTH, 256 * 1024 * 1024);
    
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(&arena, num_envs, num_threads);
    sim.start();

    uint32_t step = 0;

    for (auto _ : state) {
        state.PauseTiming();
        populate_actions(arena, num_envs, num_agents, step++);
        state.ResumeTiming();

        sim.resume_batch();
        
        uint32_t ready = 0;
        while ((ready = sim.wait_for_batch()) < num_envs) {
            std::this_thread::yield();
        }
    }

    sim.stop();
    state.SetItemsProcessed(state.iterations() * num_envs * num_agents);
}
// Baseline (1 thread) vs Multithreading at 1024 scale
BENCHMARK(BM_BatchSimulator_ThroughputScaling)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(10)->Arg(16)->UseRealTime();

// ============================================================================
// BENCHMARK 2: RL Batch Scalability (256 -> 2048 Environments)
// ============================================================================
static void BM_BatchSimulator_MassiveScale(benchmark::State& state) {
    const uint32_t num_threads = 10; // Optimal for 20-thread CPU (leaves room for OS/Python)
    const uint32_t num_envs = state.range(0); 
    const uint32_t num_agents = 8;
    
    // Scale memory dynamically up to 512MB for extreme 2048 scale
    UnifiedMemoryArena arena(num_envs, num_agents, 4096, 16, 256, 256, DEFAULT_OBS_DEPTH, 512 * 1024 * 1024);
    BatchSimulator<DEFAULT_OBS_DEPTH> sim(&arena, num_envs, num_threads);
    sim.start();

    uint32_t step = 0;

    for (auto _ : state) {
        state.PauseTiming();
        populate_actions(arena, num_envs, num_agents, step++);
        state.ResumeTiming();

        sim.resume_batch();
        uint32_t ready = 0;
        while ((ready = sim.wait_for_batch()) < num_envs) {
            std::this_thread::yield();
        }
    }

    sim.stop();
    state.SetItemsProcessed(state.iterations() * num_envs * num_agents);
}
// Testing 256, 512, 1024, 2048 parallel environments
BENCHMARK(BM_BatchSimulator_MassiveScale)->RangeMultiplier(2)->Range(256, 2048)->UseRealTime();

BENCHMARK_MAIN();