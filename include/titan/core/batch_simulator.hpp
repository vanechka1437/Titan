#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "titan/core/matching_engine.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/scheduler.hpp"
#include "titan/core/state.hpp"

namespace titan::core {

// ============================================================================
// BATCH SIMULATOR (Zero-Copy SMDP Event-Driven Engine)
// 
// This is the core orchestrator. It runs 1024+ independent Limit Order Book 
// environments using a highly optimized C++ Thread Pool. 
// 
// Architecture: "Asynchronous DES with Synchronous Yielding"
// 1. Each environment steps through time strictly from event to event (DES).
// 2. An environment pauses ONLY when an agent receives new data and its 
//    compute_delay has elapsed.
// 3. Paused environments flag themselves in the Zero-Copy `ready_mask`.
// 4. When `ready_count` >= BATCH_SIZE, the C++ threads notify Python.
// 5. PyTorch processes the batch, and calls `resume_batch()` to unpause.
// ============================================================================
template <uint32_t ObsDepth = 20>
class BatchSimulator {
private:
    UnifiedMemoryArena* arena_;
    uint32_t num_envs_;
    uint32_t num_agents_per_env_;
    uint32_t target_batch_size_;

    // --- Physics & State ---
    std::vector<EnvironmentState<ObsDepth>> envs_;
    std::vector<MatchingEngine> engines_;
    std::vector<FastScheduler> schedulers_;

    // --- Concurrency & Thread Pool ---
    std::vector<std::thread> workers_;
    alignas(64) std::atomic<bool> running_{false};

    // --- SMDP Batching Synchronization ---
    // Tracks how many environments are currently paused and waiting for Python.
    alignas(64) std::atomic<uint32_t> ready_count_{0};
    
    std::mutex batch_mutex_;
    std::condition_variable batch_cv_;

    // --- Environment Control Flags ---
    // True = Paused (Waiting for RL inference). False = Running DES physics.
    // Kept separate from arena's ready_mask to avoid atomic overhead on Python memory.
    std::vector<std::atomic<bool>> env_paused_;

    // --- Thread Partitioning ---
    struct alignas(64) WorkerBounds {
        uint32_t start_env;
        uint32_t end_env;
    };
    std::vector<WorkerBounds> worker_bounds_;

    // Internal loop executed by each background thread
    void worker_loop(uint32_t thread_id);

public:
    BatchSimulator(UnifiedMemoryArena* arena, uint32_t target_batch_size, uint32_t num_threads);
    ~BatchSimulator();

    // Prevent copying
    BatchSimulator(const BatchSimulator&) = delete;
    BatchSimulator& operator=(const BatchSimulator&) = delete;

    // ========================================================================
    // PYTHON RL INTERFACE
    // ========================================================================
    
    // Starts the background thread pool
    void start();

    // Stops the thread pool cleanly
    void stop();

    // Blocks the Python thread until exactly `target_batch_size_` environments 
    // have paused and are ready for inference. 
    // Returns the total number of ready agents (could be >= target_batch_size).
    uint32_t wait_for_batch() noexcept;

    // Called by Python after writing actions to the Zero-Copy tensor.
    // Reads actions where ready_mask == 1, pushes them to the scheduler, 
    // clears the ready_mask, and unpauses those environments.
    void resume_batch() noexcept;

    // Resets specific environments (e.g., when an episode ends)
    void reset(const std::vector<uint32_t>& env_indices) noexcept;
    
    // Global hard reset
    void reset_all() noexcept;
    
    // Accessors for testing/debugging
    void set_agent_latencies(uint32_t agent_id, uint64_t ingress_ns, uint64_t egress_ns, uint64_t compute_ns) noexcept;
    [[nodiscard]] inline EnvironmentState<ObsDepth>& get_env(uint32_t env_idx) { return envs_[env_idx]; }
    [[nodiscard]] inline FastScheduler& get_scheduler(uint32_t env_idx) { return schedulers_[env_idx]; }
};

}  // namespace titan::core