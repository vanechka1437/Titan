#include "titan/core/batch_simulator.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <mutex>

namespace titan::core {

// ============================================================================
// Initialization & Lifecycle
// ============================================================================
template <uint32_t ObsDepth>
BatchSimulator<ObsDepth>::BatchSimulator(UnifiedMemoryArena* arena, uint32_t target_batch_size, uint32_t num_threads)
    : arena_(arena),
      num_envs_(arena->num_envs()),
      num_agents_per_env_(arena->num_agents()),
      target_batch_size_(target_batch_size),
      envs_(),
      engines_(),
      schedulers_(num_envs_),
      env_paused_(num_envs_) {
      
    // 1. Initialize Environments and Engines
    envs_.reserve(num_envs_);
    engines_.reserve(num_envs_);
    for (uint32_t i = 0; i < num_envs_; ++i) {
        envs_.emplace_back(i, num_agents_per_env_, arena->max_events_per_step(), arena->max_orders_per_agent(), arena_);
        engines_.emplace_back(arena_->order_pool(i));
        
        // Environments are initialized in a paused state. 
        // This prevents worker threads from spinning on empty queues 
        // while the initial data is being loaded by the host thread.
        env_paused_[i].store(true, std::memory_order_relaxed);
    }

    // 2. Partition environments among threads evenly
    uint32_t envs_per_thread = num_envs_ / num_threads;
    uint32_t remainder = num_envs_ % num_threads;
    
    uint32_t current_start = 0;
    for (uint32_t i = 0; i < num_threads; ++i) {
        uint32_t count = envs_per_thread + (i < remainder ? 1 : 0);
        worker_bounds_.push_back({current_start, current_start + count});
        current_start += count;
    }
}

template <uint32_t ObsDepth>
BatchSimulator<ObsDepth>::~BatchSimulator() {
    stop();
}

template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::start() {
    bool expected = false;
    if (running_.compare_exchange_strong(expected, true, std::memory_order_release, std::memory_order_relaxed)) {
        for (uint32_t i = 0; i < worker_bounds_.size(); ++i) {
            workers_.emplace_back(&BatchSimulator::worker_loop, this, i);
        }
    }
}

template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::stop() {
    if (running_.exchange(false, std::memory_order_release)) {
        batch_cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }
}

// ============================================================================
// Core C++ Physics Engine (Asynchronous DES Worker)
// ============================================================================
template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::worker_loop(uint32_t thread_id) {
    const WorkerBounds& bounds = worker_bounds_[thread_id];

    while (running_.load(std::memory_order_acquire)) {
        bool idle = true;

        for (uint32_t env_id = bounds.start_env; env_id < bounds.end_env; ++env_id) {
            
            // Skip if environment is paused (waiting for external synchronization)
            if (env_paused_[env_id].load(std::memory_order_acquire)) {
                continue;
            }

            idle = false;
            EnvironmentState<ObsDepth>& env = envs_[env_id];
            MatchingEngine& engine = engines_[env_id];
            FastScheduler& scheduler = schedulers_[env_id];

            // Process events up to the next required agent decision
            while (!scheduler.empty()) {
                
                // Extract event by value and pop immediately.
                // This prevents dangling references if a newly pushed event 
                // triggers a reallocation within the underlying priority queue.
                const ScheduledEvent event = scheduler.top();
                scheduler.pop(); 
                
                // Advance the simulated clock
                env.current_time = event.time;

                // --- A. Agent is Ready to Act (SMDP Wakeup) ---
                if (event.type == ScheduledEvent::Type::AGENT_WAKEUP) {
                    const uint32_t agent_id = event.target_id;
                    const std::size_t mask_idx = (env_id * num_agents_per_env_) + agent_id;
                    
                    // Mark agent as ready in the Zero-Copy DLPack mask
                    arena_->ready_mask_ptr()[mask_idx] = 1;
                    
                    // Export localized observations
                    env.agents[agent_id].export_observations();

                    // Suspend physics for this environment
                    env_paused_[env_id].store(true, std::memory_order_release);
                    
                    // Atomically increment the global readiness counter
                    ready_count_.fetch_add(1, std::memory_order_release);
                    
                    // Notify the host thread
                    batch_cv_.notify_one(); 
                    break;
                }
                
                // --- B. Order Arrival at Exchange ---
                else if (event.type == ScheduledEvent::Type::ORDER_ARRIVAL) {
                    EventList engine_events;
                    if (event.action.is_cancel()) {
                        engine.process_cancel(event.action.target_id, event.target_id, engine_events);
                    } else {
                        engine.process_order(event.target_id, event.action.side, event.action.price, event.action.qty, engine_events);
                    }
                    
                    for (const auto& ev : engine_events) {
                        uint64_t cursor = arena_->event_cursors_ptr()[env_id];
                        uint64_t ring_idx = cursor % arena_->max_events_per_step();
                        uint64_t env_offset = env_id * arena_->max_events_per_step();
                        
                        arena_->events_ptr()[env_offset + ring_idx] = ev;
                        arena_->event_cursors_ptr()[env_id] = cursor + 1;

                        for (uint32_t a_id = 0; a_id < num_agents_per_env_; ++a_id) {
                            uint64_t arrival_time = env.current_time + env.agents[a_id].egress_delay;
                            scheduler.push(ScheduledEvent::make_market_data(arrival_time, a_id, ev));
                        }
                    }
                }

                // --- C. Market Data Delivery to Agent ---
                else if (event.type == ScheduledEvent::Type::MARKET_DATA) {
                    AgentState<ObsDepth>& agent = env.agents[event.target_id];
                    agent.apply_event(event.market_data);
                    
                    // Schedule subsequent decision-making wakeup
                    uint64_t wakeup_time = env.current_time + agent.compute_delay;
                    agent.next_wakeup_time = wakeup_time;
                    scheduler.push(ScheduledEvent::make_agent_wakeup(wakeup_time, agent.id));
                }
            }

            // Lock-Free Safety: If the queue emptied out naturally without an AGENT_WAKEUP,
            // we must pause the environment to await further Python injection.
            if (scheduler.empty()) {
                env_paused_[env_id].store(true, std::memory_order_release);
            }
        }

        if (idle) {
            std::this_thread::yield();
        }
    }
}

// ============================================================================
// Python API: Synchronous Yielding & Resuming
// ============================================================================
template <uint32_t ObsDepth>
uint32_t BatchSimulator<ObsDepth>::wait_for_batch() noexcept {
    std::unique_lock<std::mutex> lock(batch_mutex_);
    
    batch_cv_.wait_for(lock, std::chrono::milliseconds(1), [this]() {
        return ready_count_.load(std::memory_order_acquire) >= target_batch_size_;
    });

    return ready_count_.load(std::memory_order_acquire);
}

template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::resume_batch() noexcept {
    uint8_t* ready_mask = arena_->ready_mask_ptr();
    const ActionPayload* actions = arena_->actions_ptr();

    ready_count_.store(0, std::memory_order_release);

    for (uint32_t env_id = 0; env_id < num_envs_; ++env_id) {
        bool env_was_active_in_batch = false;

        for (uint32_t agent_id = 0; agent_id < num_agents_per_env_; ++agent_id) {
            const std::size_t mask_idx = (env_id * num_agents_per_env_) + agent_id;
            
            if (ready_mask[mask_idx] == 1) {
                env_was_active_in_batch = true;
                ready_mask[mask_idx] = 0; 

                const std::size_t agent_base_idx = 
                    (env_id * num_agents_per_env_ * arena_->max_actions_per_agent()) + 
                    (agent_id * arena_->max_actions_per_agent());
                
                bool sent_any_action = false;

                for (uint32_t a_slot = 0; a_slot < arena_->max_actions_per_agent(); ++a_slot) {
                    const ActionPayload& action = actions[agent_base_idx + a_slot];
                    
                    if (action.action_type != 3) { 
                        if (env_id == 0) { 
                            std::cout << "[C++ IN] Agent: " << agent_id 
                                      << " | Type: " << (int)action.action_type 
                                      << " | Side: " << (int)action.side 
                                      << " | Price: " << action.price 
                                      << " | Qty: " << action.qty << std::endl;
                        }
                        uint64_t arrival_time = envs_[env_id].current_time + envs_[env_id].agents[agent_id].ingress_delay;
                        schedulers_[env_id].push(ScheduledEvent::make_order_arrival(arrival_time, agent_id, action));
                        sent_any_action = true;
                    }
                }

                if (!sent_any_action) {
                    uint64_t next_wakeup = envs_[env_id].current_time + envs_[env_id].agents[agent_id].compute_delay;
                    schedulers_[env_id].push(ScheduledEvent::make_agent_wakeup(next_wakeup, agent_id));
                }
            }
        }

        if (env_was_active_in_batch) {
            env_paused_[env_id].store(false, std::memory_order_release);
        } else if (env_paused_[env_id].load(std::memory_order_acquire)) {
            if (!schedulers_[env_id].empty()) {
                env_paused_[env_id].store(false, std::memory_order_release);
            }
        }
    }
}

// ============================================================================
// Resets
// ============================================================================
template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::reset(const std::vector<uint32_t>& env_indices) noexcept {
    arena_->reset(env_indices);

    for (uint32_t env_id : env_indices) {
        
        // --- SECURITY PATCH: OOB Memory Protection ---
        // Protects the std::vector elements and atomic flags from invalid indices
        if (env_id >= num_envs_) [[unlikely]] {
            continue;
        }

        env_paused_[env_id].store(true, std::memory_order_relaxed); 
        
        envs_[env_id].reset();
        engines_[env_id].reset();
        schedulers_[env_id].clear();
    }
}

template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::reset_all() noexcept {
    arena_->reset_all();

    for (uint32_t i = 0; i < num_envs_; ++i) {
        env_paused_[i].store(true, std::memory_order_relaxed);
        
        envs_[i].reset();
        engines_[i].reset();
        schedulers_[i].clear();
    }
    
    ready_count_.store(0, std::memory_order_relaxed);
}

// ============================================================================
// Network & Hardware Latency Configuration
// ============================================================================
template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::set_agent_latencies(uint32_t agent_id, uint64_t ingress_ns, uint64_t egress_ns, uint64_t compute_ns) noexcept {
    for (uint32_t i = 0; i < num_envs_; ++i) {
        if (agent_id < num_agents_per_env_) [[likely]] {
            envs_[i].agents[agent_id].ingress_delay = ingress_ns;
            envs_[i].agents[agent_id].egress_delay = egress_ns;
            envs_[i].agents[agent_id].compute_delay = compute_ns;
        }
    }
}

// ============================================================================
// Explicit Instantiations
// ============================================================================
template class BatchSimulator<DEFAULT_OBS_DEPTH>;

}  // namespace titan::core