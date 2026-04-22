#include "titan/core/batch_simulator.hpp"

#include <algorithm>

namespace titan::core {

// ============================================================================
// CONSTRUCTOR
// ============================================================================
template <uint32_t ObsDepth>
BatchSimulator<ObsDepth>::BatchSimulator(uint32_t num_envs, uint32_t num_agents,
                                         uint32_t max_orders_per_env, uint32_t max_actions_per_step,
                                         uint32_t max_events_per_step, uint32_t max_orders_per_agent,
                                         uint32_t num_threads, std::size_t linear_bytes)
    : num_envs_(num_envs),
      num_agents_(num_agents),
      max_orders_per_env_(max_orders_per_env),
      max_actions_per_step_(max_actions_per_step),
      max_events_per_step_(max_events_per_step),
      max_orders_per_agent_(max_orders_per_agent),
      arena_(num_envs, num_agents, max_orders_per_env, max_actions_per_step,
             max_events_per_step, max_orders_per_agent, ObsDepth, linear_bytes),
      num_threads_(num_threads) {

    // 1. Pre-allocate Sandbox Vectors
    envs_.reserve(num_envs_);
    engines_.reserve(num_envs_);

    for (uint32_t i = 0; i < num_envs_; ++i) {
        // EnvironmentState acts as the Zero-Copy writer into the Arena
        envs_.emplace_back(i, num_agents_, max_events_per_step_, max_orders_per_agent_, &arena_);
        
        // MatchingEngine binds to its isolated Pool in the Arena
        engines_.emplace_back(arena_.get_pool(i), max_orders_per_env_);
    }

    // 2. Spin up the Worker Thread Pool
    workers_.reserve(num_threads_);
    for (uint32_t i = 0; i < num_threads_; ++i) {
        workers_.emplace_back(&BatchSimulator::worker_loop, this, i);
    }
}

// ============================================================================
// DESTRUCTOR
// ============================================================================
template <uint32_t ObsDepth>
BatchSimulator<ObsDepth>::~BatchSimulator() {
    {
        std::lock_guard<std::mutex> lock(sync_mutex_);
        terminate_pool_ = true;
    }
    cv_start_work_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

// ============================================================================
// THE HOT PATH (BARRIER SYNCHRONIZATION)
// ============================================================================
template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::step(uint32_t num_commands) {
    {
        std::lock_guard<std::mutex> lock(sync_mutex_);
        current_num_commands_ = num_commands;
        completed_tasks_++; // Acts as the Generation/Epoch counter
        pending_tasks_ = num_threads_;
    }
    
    // Ignite the thread pool
    cv_start_work_.notify_all();

    {
        // Block Python until C++ workers exhaust the Command Buffer
        std::unique_lock<std::mutex> lock(sync_mutex_);
        cv_work_done_.wait(lock, [this] { return pending_tasks_ == 0; });
    }
}

// ============================================================================
// WORKER THREAD LOGIC
// ============================================================================
template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::worker_loop(uint32_t thread_id) {
    uint32_t local_epoch = 0;
    
    // Calculate static workload bounds for this specific thread
    const uint32_t envs_per_thread = num_envs_ / num_threads_;
    const uint32_t start_env = thread_id * envs_per_thread;
    const uint32_t end_env = (thread_id == num_threads_ - 1) ? num_envs_ : start_env + envs_per_thread;

    while (true) {
        // 1. Wait for the starting gun
        std::unique_lock<std::mutex> lock(sync_mutex_);
        cv_start_work_.wait(lock, [this, local_epoch] {
            return terminate_pool_ || completed_tasks_ != local_epoch;
        });

        if (terminate_pool_) {
            return;
        }

        local_epoch = completed_tasks_;
        const uint32_t cmds_per_env = current_num_commands_;
        lock.unlock();

        // 2. Consume the Command Stream
        ActionPayload* base_actions = arena_.actions_ptr();

        for (uint32_t env_id = start_env; env_id < end_env; ++env_id) {
            MatchingEngine& engine = engines_[env_id];
            EnvironmentState<ObsDepth>& state = envs_[env_id];

            // Reset step-local event counters in the EnvironmentState
            state.prepare_for_step(); 

            const std::size_t offset = static_cast<std::size_t>(env_id) * max_actions_per_step_;

            for (uint32_t i = 0; i < cmds_per_env; ++i) {
                const ActionPayload& action = base_actions[offset + i];

                // Execute based on Action Type (0: Limit, 1: Cancel, 2: Market)
                if (action.action_type == 1) {
                    engine.process_cancel(action.target_id, action.agent_id, state);
                } else {
                    engine.process_order(action.agent_id, action.side, action.price, action.qty, state);
                }
            }
            
            // Synchronize the ShadowLOB after all actions for this environment are processed
            state.update_observations(engine.get_lob());
        }

        // 3. Report completion and trigger main thread if last
        lock.lock();
        if (--pending_tasks_ == 0) {
            cv_work_done_.notify_one();
        }
    }
}

// ============================================================================
// RESETS
// ============================================================================
template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::reset(const std::vector<uint32_t>& env_indices) {
    // Ensure no step is running while resetting
    std::lock_guard<std::mutex> lock(sync_mutex_);

    arena_.reset(env_indices);
    for (const uint32_t idx : env_indices) {
        engines_[idx].reset();
        envs_[idx].reset();
    }
}

template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::reset_all() {
    std::lock_guard<std::mutex> lock(sync_mutex_);

    arena_.reset_all();
    for (uint32_t i = 0; i < num_envs_; ++i) {
        engines_[i].reset();
        envs_[i].reset();
    }
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATION
// ============================================================================
template class BatchSimulator<DEFAULT_OBS_DEPTH>;

}  // namespace titan::core