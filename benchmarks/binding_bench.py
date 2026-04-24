import os
import sys
import time
import torch

# =============================================================================
# LIBRARY PATH INJECTION
# Automatically locates the compiled C++ extension in the build/Release folder.
# This handles both Windows and Linux build directory structures.
# =============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir_win = os.path.abspath(os.path.join(current_dir, '..', '..', 'build', 'Release'))
build_dir_lin = os.path.abspath(os.path.join(current_dir, '..', '..', 'build'))

if os.path.exists(build_dir_win):
    sys.path.insert(0, build_dir_win)
elif os.path.exists(build_dir_lin):
    sys.path.insert(0, build_dir_lin)
else:
    raise RuntimeError(f"Compiled extension not found. Please build the project first.")

import titan_core # type: ignore

# =============================================================================
# BENCHMARK HYPERPARAMETERS
# Massively parallel configuration designed to saturate modern multicore CPUs.
# =============================================================================
NUM_ENVS = 4096                # Number of isolated parallel LOB environments
NUM_AGENTS = 8                 # Number of HFT agents per environment
MAX_ORDERS_PER_ENV = 4096      # Max LOB depth / active orders per environment
MAX_ACTIONS = 16               # Max actions an agent can submit per step
MAX_EVENTS = 256               # Max market data events processed per step
MAX_ACTIVE_ORDERS = 128        # Max tracked orders per individual agent
OBS_DEPTH = 20                 # L2 Orderbook observation depth (Bids/Asks)
LINEAR_BYTES = 128 * 1024 * 1024  # 128 MB pinned OS memory pool

NUM_STEPS = 1000               # Total simulation steps for the benchmark loop
NUM_WARMUP_STEPS = 50          # Steps to warm up L1/L2 caches and thread pools
NUM_THREADS = os.cpu_count() or 4 # Dynamically utilize all available CPU cores

def run_zero_copy_benchmark():
    # Prevent PyTorch from spawning background threads that compete with our C++ Engine
    torch.set_num_threads(1)
    
    print(f"\n{'='*50}")
    print(f" TITAN CORE - ZERO-COPY DLPACK PROFILING ")
    print(f"{'='*50}")
    print(f"Hardware Threads : {NUM_THREADS}")
    print(f"Environments     : {NUM_ENVS:,}")
    print(f"Agents per Env   : {NUM_AGENTS}")
    print(f"Total Agents     : {NUM_ENVS * NUM_AGENTS:,}")
    print(f"Steps            : {NUM_STEPS:,}")
    print(f"{'-'*50}")

    print("[1/4] Allocating Pinned Memory Arena...")
    arena = titan_core.MemoryArena(
        num_envs=NUM_ENVS, 
        num_agents=NUM_AGENTS, 
        max_orders_per_env=MAX_ORDERS_PER_ENV,
        max_actions_per_step=MAX_ACTIONS, 
        max_events_per_step=MAX_EVENTS,
        max_orders_per_agent=MAX_ACTIVE_ORDERS, 
        obs_depth=OBS_DEPTH, 
        linear_bytes=LINEAR_BYTES
    )

    print("[2/4] Initializing Asynchronous C++ Batch Simulator...")
    sim = titan_core.Simulator(
        arena=arena, 
        target_batch_size=NUM_ENVS, 
        num_threads=NUM_THREADS
    )

    # -------------------------------------------------------------------------
    # ZERO-COPY TENSOR MAPPING
    # These tensors point directly to C++ RAM. No serialization, no deep copies.
    # -------------------------------------------------------------------------
    actions_tensor = arena.actions
    ready_mask = arena.ready_mask
    
    # Pre-allocate a dummy action tensor in Python to simulate Neural Network output.
    dummy_actions = torch.ones_like(actions_tensor)

    print("[3/4] Warming up L1/L2 caches and worker threads...")
    sim.start()
    for _ in range(NUM_WARMUP_STEPS):
        ready_mask.fill_(1)
        sim.resume_batch()
        sim.wait_for_batch()

    print("[4/4] Executing Main Profiling Loop...")
    
    # -------------------------------------------------------------------------
    # PROFILING ACCUMULATORS
    # -------------------------------------------------------------------------
    python_overhead_time = 0.0
    cpp_compute_time = 0.0

    start_time = time.perf_counter()

    for _ in range(NUM_STEPS):
        # --- PHASE 1: PYTHON & PYTORCH OVERHEAD ---
        # GIL is LOCKED. We measure the cost of memory writes and looping.
        t0 = time.perf_counter()
        
        # 1. AI Inference Simulation (O(1) contiguous memory block copy)
        actions_tensor.copy_(dummy_actions)
        
        # 2. Notify C++ that all agents in all environments are ready
        ready_mask.fill_(1)
        
        t1 = time.perf_counter()
        
        # --- PHASE 2: C++ CORE COMPUTE ---
        # GIL is RELEASED. Python thread sleeps while C++ multi-threading engine runs.
        
        # 3. Release the Python GIL and wake up C++ workers
        sim.resume_batch()
        
        # 4. Wait for C++ Engine to resolve physics (Order Matching, PnL, LOB updates)
        sim.wait_for_batch()
        
        t2 = time.perf_counter()

        # Accumulate timings
        python_overhead_time += (t1 - t0)
        cpp_compute_time += (t2 - t1)

    end_time = time.perf_counter()
    sim.stop()

    # =========================================================================
    # CALCULATE FINAL METRICS
    # =========================================================================
    total_time_sec = end_time - start_time
    
    # Ratios
    python_ratio = (python_overhead_time / total_time_sec) * 100.0
    cpp_ratio = (cpp_compute_time / total_time_sec) * 100.0
    
    # Throughput
    latency_ms = (total_time_sec / NUM_STEPS) * 1000.0
    batch_sps = NUM_STEPS / total_time_sec
    agent_sps = (NUM_STEPS * NUM_ENVS * NUM_AGENTS) / total_time_sec

    print(f"\n{'='*50}")
    print(f" PROFILING RESULTS: PYTHON vs C++ ")
    print(f"{'='*50}")
    print(f"Total Wall Time  : {total_time_sec:.4f} sec")
    print(f"Batch Latency    : {latency_ms:.4f} ms per step")
    print(f"Agent SPS        : {agent_sps:,.0f} actions/sec")
    print(f"{'-'*50}")
    print(f"Python/GIL Time  : {python_overhead_time:.4f} sec ({python_ratio:.1f}%)")
    print(f"C++ Engine Time  : {cpp_compute_time:.4f} sec ({cpp_ratio:.1f}%)")
    print(f"{'-'*50}")
    
    if cpp_ratio > 95.0:
        print("DIAGNOSIS: PERFECT. The bridge is invisible. You are completely CPU bound by C++ physics.")
    elif cpp_ratio > 80.0:
        print("DIAGNOSIS: EXCELLENT. Minor PyTorch overhead, but highly efficient.")
    else:
        print("DIAGNOSIS: BOTTLENECK. Python is taking too much time copying data. Check tensor contiguity.")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    run_zero_copy_benchmark()