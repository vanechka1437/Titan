#include "titan/core/batch_simulator.hpp"

#include <algorithm>
#include <chrono>

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
        env_paused_[i].store(false, std::memory_order_relaxed);
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
            // Skip if environment is paused (waiting for Python)
            if (env_paused_[env_id].load(std::memory_order_acquire)) {
                continue;
            }

            idle = false;
            EnvironmentState<ObsDepth>& env = envs_[env_id];
            MatchingEngine& engine = engines_[env_id];
            FastScheduler& scheduler = schedulers_[env_id];

            // 1. Safety Valve: Buffer Overflow Protection (Data Flush)
            if (env.needs_flush()) [[unlikely]] {
                env_paused_[env_id].store(true, std::memory_order_release);
                batch_cv_.notify_one(); 
                continue;
            }

            // 2. Process events exactly up to the next required agent decision
            while (!scheduler.empty()) {
                const auto& event = scheduler.top();
                
                // Advance the simulated clock
                env.current_time = event.time;

                // --- A. Agent is Ready to Act (SMDP Wakeup) ---
                if (event.type == ScheduledEvent::Type::AGENT_WAKEUP) {
                    const uint32_t agent_id = event.target_id;
                    
                    // Mark agent as ready in Zero-Copy Mask
                    const std::size_t mask_idx = (env_id * num_agents_per_env_) + agent_id;
                    arena_->ready_mask_ptr()[mask_idx] = 1;
                    
                    // Export this agent's localized observation
                    env.agents[agent_id].export_observations();
                    
                    // Atomically increment global batch counter
                    ready_count_.fetch_add(1, std::memory_order_release);
                    
                    // Pause the environment
                    env_paused_[env_id].store(true, std::memory_order_release);
                    
                    scheduler.pop();
                    batch_cv_.notify_one(); 
                    break; // Break DES loop, move to next env
                }
                
                // --- B. Order Arrival at Exchange ---
                else if (event.type == ScheduledEvent::Type::ORDER_ARRIVAL) {
                    EventList engine_events;
                    if (event.action.is_cancel()) {
                        engine.process_cancel(event.action.target_id, event.target_id, engine_events);
                    } else {
                        engine.process_order(event.target_id, event.action.side, event.action.price, event.action.qty, engine_events);
                    }
                    
                    // Route the resulting events back to agents via Egress Delay
                    for (const auto& ev : engine_events) {
                        // Record global trajectory event for PyTorch
                        env.record_public_event(ev);

                        // Route to Maker
                        uint64_t maker_arrival_time = env.current_time + env.agents[ev.owner_id].egress_delay;
                        scheduler.push(ScheduledEvent::market_data(maker_arrival_time, ev.owner_id, ev));
                        
                        // Route to Taker (if TRADE)
                        if (ev.type == MarketDataEvent::Type::TRADE && ev.taker_id != ev.owner_id) {
                            uint64_t taker_arrival_time = env.current_time + env.agents[ev.taker_id].egress_delay;
                            scheduler.push(ScheduledEvent::market_data(taker_arrival_time, ev.taker_id, ev));
                        }
                    }
                }

                // --- C. Market Data Delivery to Agent ---
                else if (event.type == ScheduledEvent::Type::MARKET_DATA) {
                    AgentState<ObsDepth>& agent = env.agents[event.target_id];
                    agent.apply_event(event.market_data);
                    
                    // If agent receives data, schedule its compute_delay wakeup
                    uint64_t wakeup_time = env.current_time + agent.compute_delay;
                    agent.next_wakeup_time = wakeup_time;
                    scheduler.push(ScheduledEvent::agent_wakeup(wakeup_time, agent.id));
                }

                scheduler.pop();
            }
        }

        // If all assigned environments are paused, yield CPU to prevent spinning
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
    
    // Wait until BATCH_SIZE is reached, OR timeout occurs (Straggler Protection)
    batch_cv_.wait_for(lock, std::chrono::milliseconds(1), [this]() {
        return ready_count_.load(std::memory_order_acquire) >= target_batch_size_;
    });

    return ready_count_.load(std::memory_order_acquire);
}

template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::resume_batch() noexcept {
    uint8_t* ready_mask = arena_->ready_mask_ptr();
    const ActionPayload* actions = arena_->actions_ptr();

    for (uint32_t env_id = 0; env_id < num_envs_; ++env_id) {
        bool env_was_active_in_batch = false;

        // 1. Process Actions & Clear Masks
        for (uint32_t agent_id = 0; agent_id < num_agents_per_env_; ++agent_id) {
            const std::size_t idx = (env_id * num_agents_per_env_) + agent_id;
            
            if (ready_mask[idx] == 1) {
                env_was_active_in_batch = true;
                ready_mask[idx] = 0; // Clear mask

                const ActionPayload& action = actions[idx];
                
                // NO_OP handling
                if (action.action_type == 3) continue; 

                // Inject action into scheduler with Network Ingress Delay
                uint64_t arrival_time = envs_[env_id].current_time + envs_[env_id].agents[agent_id].ingress_delay;
                schedulers_[env_id].push(ScheduledEvent::order_arrival(arrival_time, agent_id, action));
            }
        }

        // 2. Unpause environment
        // If it was active in this batch, or if it was paused solely due to buffer flush
        if (env_was_active_in_batch || env_paused_[env_id].load(std::memory_order_acquire)) {
            envs_[env_id].prepare_for_step(); // Clears historical event buffer
            env_paused_[env_id].store(false, std::memory_order_release);
        }
    }

    // Reset global counter
    ready_count_.store(0, std::memory_order_release);
}

// ============================================================================
// Resets
// ============================================================================
template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::reset(const std::vector<uint32_t>& env_indices) noexcept {
    for (uint32_t env_id : env_indices) {
        env_paused_[env_id].store(true, std::memory_order_relaxed); // Temporarily halt
        
        envs_[env_id].reset();
        engines_[env_id].reset();
        schedulers_[env_id].clear();
        
        env_paused_[env_id].store(false, std::memory_order_release);
    }
}

template <uint32_t ObsDepth>
void BatchSimulator<ObsDepth>::reset_all() noexcept {
    for (uint32_t i = 0; i < num_envs_; ++i) {
        env_paused_[i].store(true, std::memory_order_relaxed);
        envs_[i].reset();
        engines_[i].reset();
        schedulers_[i].clear();
        env_paused_[i].store(false, std::memory_order_release);
    }
    ready_count_.store(0, std::memory_order_relaxed);
}

// ============================================================================
// Explicit Instantiations
// ============================================================================
template class BatchSimulator<DEFAULT_OBS_DEPTH>;

}  // namespace titan::core