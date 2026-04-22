#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "titan/core/matching_engine.hpp"
#include "titan/core/memory.hpp"
#include "titan/core/scheduler.hpp"
#include "titan/core/state.hpp"
#include "titan/core/types.hpp"

namespace titan::core {

// ============================================================================
// MULTI-AGENT BATCH SIMULATOR
// The Orchestrator. Owns the Arena, Thread Pool, and N isolated sandboxes.
// Pure physical execution engine processing Command Streams.
// ============================================================================
template <uint32_t ObsDepth = 20>
class BatchSimulator {
private:
    // --- Limits & Config ---
    uint32_t num_envs_;
    uint32_t num_agents_;
    uint32_t max_orders_per_env_;
    uint32_t max_actions_per_step_;
    uint32_t max_events_per_step_;
    uint32_t max_orders_per_agent_;

    // --- Core Memory (The Monolith) ---
    UnifiedMemoryArena arena_;

    // --- Isolated Sandboxes ---
    std::vector<EnvironmentState<ObsDepth>> envs_;
    std::vector<MatchingEngine> engines_;

    // --- C++ Thread Pool Infrastructure ---
    uint32_t num_threads_;
    std::vector<std::thread> workers_;
    std::atomic<bool> terminate_pool_{false};

    // Barrier Synchronization primitives for lock-free stepping
    std::mutex sync_mutex_;
    std::condition_variable cv_start_work_;
    std::condition_variable cv_work_done_;
    
    uint32_t pending_tasks_{0};
    uint32_t completed_tasks_{0};

    // --- Execution State ---
    uint32_t current_num_commands_{0};

    // --- Internal Worker Function ---
    void worker_loop(uint32_t thread_id);

public:
    // Initializes the cluster, allocates Pinned Arena once, spins up Threads.
    BatchSimulator(uint32_t num_envs, uint32_t num_agents, 
                   uint32_t max_orders_per_env, uint32_t max_actions_per_step,
                   uint32_t max_events_per_step, uint32_t max_orders_per_agent,
                   uint32_t num_threads, std::size_t linear_bytes = 1024 * 1024);
                   
    ~BatchSimulator();

    // Non-copyable (strict ownership of hardware threads and pinned memory)
    BatchSimulator(const BatchSimulator&) = delete;
    BatchSimulator& operator=(const BatchSimulator&) = delete;

    // ========================================================================
    // THE HOT PATH (Called from Python)
    // 1. Python writes directly to arena_.actions_ptr() (Zero-Copy)
    // 2. Python calls step(num_commands)
    // 3. Wakes up the C++ Thread Pool
    // 4. Blocks Python thread natively until all C++ threads finish
    // ========================================================================
    void step(uint32_t num_commands);

    // Fast reset for specific environments
    void reset(const std::vector<uint32_t>& env_indices);

    // Global reset
    void reset_all();

    // ========================================================================
    // ZERO-COPY ACCESSORS FOR NANOBIND
    // Returns raw pointers to our pre-allocated Pinned Memory Arena.
    // ========================================================================
    [[nodiscard]] inline ActionPayload* get_actions_tensor_ptr() noexcept { return arena_.actions_ptr(); }
    [[nodiscard]] inline MarketDataEvent* get_events_tensor_ptr() noexcept { return arena_.events_ptr(); }
    [[nodiscard]] inline ActiveOrderRecord* get_active_orders_tensor_ptr() noexcept { return arena_.active_orders_ptr(); }
    
    [[nodiscard]] inline float* get_lob_tensor_ptr() noexcept { return arena_.lob_ptr(); }
    [[nodiscard]] inline float* get_cash_tensor_ptr() noexcept { return arena_.cash_ptr(); }
    [[nodiscard]] inline float* get_inventory_tensor_ptr() noexcept { return arena_.inventory_ptr(); }

    // --- Config Accessors ---
    [[nodiscard]] inline uint32_t num_envs() const noexcept { return num_envs_; }
    [[nodiscard]] inline uint32_t num_agents() const noexcept { return num_agents_; }
};

}  // namespace titan::core