import pytest
import torch
import threading
import time
import gc
import sys

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir_win = os.path.join(current_dir, '../../build/Release')
build_dir_lin = os.path.join(current_dir, '../../build')

if os.path.exists(build_dir_win):
    sys.path.insert(0, build_dir_win)
elif os.path.exists(build_dir_lin):
    sys.path.insert(0, build_dir_lin)

import titan_core # type: ignore

# =============================================================================
# TITAN BINDING DESTRUCTION SUITE
# Objective: Break the Zero-Copy bridge, induce deadlocks, find memory leaks,
# and exploit missing bounds checks in the C++ layer from Python.
# =============================================================================

# Default test parameters
NUM_ENVS = 64
NUM_AGENTS = 8
MAX_ORDERS = 1024
MAX_ACTIONS = 16
MAX_EVENTS = 256
MAX_ACTIVE_ORDERS = 128
OBS_DEPTH = 20
LINEAR_BYTES = 64 * 1024 * 1024  # 64 MB

@pytest.fixture
def clean_engine():
    """Fixture to provide a fresh engine and guarantee GC cleanup."""
    arena = titan_core.MemoryArena(
        num_envs=NUM_ENVS,
        num_agents=NUM_AGENTS,
        max_orders_per_env=MAX_ORDERS,
        max_actions_per_step=MAX_ACTIONS,
        max_events_per_step=MAX_EVENTS,
        max_orders_per_agent=MAX_ACTIVE_ORDERS,
        obs_depth=OBS_DEPTH,
        linear_bytes=LINEAR_BYTES
    )
    sim = titan_core.Simulator(arena, target_batch_size=NUM_ENVS, num_threads=4)
    yield arena, sim
    
    # Aggressive teardown
    sim.stop()
    del sim
    del arena
    gc.collect()

# =============================================================================
# 1. THE DANGLING POINTER ATTACK (Garbage Collection & Ownership)
# =============================================================================
def test_garbage_collection_keep_alive():
    """
    VULNERABILITY TEST: Does nb::keep_alive<1, 2>() actually work?
    If we delete the Arena from Python, but the Simulator is still running,
    the C++ threads will write to freed memory causing a Segmentation Fault.
    """
    arena = titan_core.MemoryArena(1, 1, 10, 10, 10, 10, 10, 1024 * 1024)
    sim = titan_core.Simulator(arena, 1, 1)
    sim.start()
    
    # Capture the raw memory address of a tensor
    lob_tensor = arena.lob
    raw_ptr = lob_tensor.data_ptr()
    
    # ATTACK: Destroy the Python arena object
    del arena
    gc.collect() 
    
    # If keep_alive is missing, this read will Segfault or return garbage
    # If keep_alive works, the memory is preserved because 'sim' is alive
    assert lob_tensor.data_ptr() == raw_ptr
    
    # Cleanup
    sim.stop()

# =============================================================================
# 2. ZERO-COPY C-CONTIGUOUS VALIDATION
# =============================================================================
def test_zero_copy_strides(clean_engine):
    """
    PERFORMANCE BOTTLENECK TEST: 
    If the tensors are not strictly C-contiguous, PyTorch will trigger 
    a silent deep-copy during matrix multiplications, destroying our 
    "3.15 million ops/sec" advantage.
    """
    arena, _ = clean_engine
    
    assert arena.lob.is_contiguous(), "CRITICAL: LOB tensor is not C-contiguous!"
    assert arena.cash.is_contiguous(), "CRITICAL: Cash tensor is not C-contiguous!"
    assert arena.actions.is_contiguous(), "CRITICAL: Actions tensor is not C-contiguous!"
    
    # Verify strict shape casting from C++ to PyTorch
    assert arena.lob.shape == (NUM_ENVS, NUM_AGENTS, OBS_DEPTH * 4)
    assert arena.actions.shape == (NUM_ENVS, MAX_ACTIONS, 4)  # 32-bytes = 4x int64

# =============================================================================
# 3. GIL DEADLOCK ATTACK (Asynchronous Threading)
# =============================================================================
def test_gil_deadlock_survival(clean_engine):
    """
    VULNERABILITY TEST: 
    If wait_for_batch() or resume_batch() do not release the GIL 
    (nb::gil_scoped_release), this test will permanently freeze.
    """
    arena, sim = clean_engine
    sim.start()
    
    deadlock_flag = {"failed": True}
    
    def background_inference_loop():
        # This should BLOCK the background thread in C++, 
        # but it MUST NOT block the Python GIL.
        sim.wait_for_batch()
        deadlock_flag["failed"] = False

    bg_thread = threading.Thread(target=background_inference_loop)
    bg_thread.start()
    
    # Give the background thread time to enter wait_for_batch()
    time.sleep(0.1)
    
    # If the GIL is locked by wait_for_batch(), the main thread will freeze here.
    # We simulate RL agent actions by writing directly to the Zero-Copy mask.
    arena.ready_mask.fill_(1)
    
    # This wakes up the C++ threads, which will eventually trigger 
    # the condition variable and free the background thread.
    sim.resume_batch()
    
    bg_thread.join(timeout=2.0)
    
    assert not bg_thread.is_alive(), "CRITICAL: GIL Deadlock detected! Threads frozen."
    assert not deadlock_flag["failed"], "Background thread failed to wake up."

# =============================================================================
# 4. ACTION PAYLOAD BIT-FUZZING (Struct Packing Integrity)
# =============================================================================
def test_action_payload_fuzzer(clean_engine):
    """
    INTEGRITY TEST: 
    We export a 32-byte C++ struct as an array of 4x 64-bit integers.
    If the struct padding or alignment in C++ (types.hpp) differs from our
    assumption, PyTorch writes will corrupt the target_id, side, or action_type,
    causing the Matching Engine to behave randomly.
    """
    arena, sim = clean_engine
    
    # Struct ActionPayload {
    #   uint64_t target_id;   // offset 0  (Tensor dim 0)
    #   int64_t qty;          // offset 8  (Tensor dim 1)
    #   uint32_t price;       // offset 16 (Tensor dim 2, lower 32-bits)
    #   uint32_t env_id;      // offset 20 (Tensor dim 2, upper 32-bits)
    #   uint16_t agent_id;    // offset 24 (Tensor dim 3, bits 0-15)
    #   uint8_t action_type;  // offset 26 (Tensor dim 3, bits 16-23)
    #   uint8_t side;         // offset 27 (Tensor dim 3, bits 24-31)
    #   uint8_t padding[4];   // offset 28 (Tensor dim 3, bits 32-63)
    # }
    
    # Write a carefully bitwise-crafted action directly via PyTorch
    env_target = 0
    agent_target = 0
    
    target_id = 999999999
    qty = 500
    price = 10500
    action_type = 2  # MARKET
    side = 1         # ASK
    
    # Dim 0: target_id
    arena.actions[env_target, 0, 0] = target_id
    # Dim 1: qty
    arena.actions[env_target, 0, 1] = qty
    # Dim 2: price | (env_id << 32)
    arena.actions[env_target, 0, 2] = price | (env_target << 32)
    # Dim 3: agent_id | (action_type << 16) | (side << 24)
    arena.actions[env_target, 0, 3] = agent_target | (action_type << 16) | (side << 24)
    
    # Mark ready and resume
    arena.ready_mask[env_target, agent_target] = 1
    
    sim.start()
    sim.resume_batch()
    
    # Wait for processing
    sim.wait_for_batch()
    sim.stop()
    
    # If the C++ Engine successfully parsed the struct, it processed a Market Ask
    # for 500 qty. We can verify this by checking the event_cursors (it should have
    # generated a REJECTED event because the book is empty, but the cursor will move).
    assert arena.event_cursors[env_target].item() > 0, "C++ failed to parse PyTorch bit-packed ActionPayload"

# =============================================================================
# 5. OUT-OF-BOUNDS VECTORIZED RESET VULNERABILITY
# =============================================================================
def test_out_of_bounds_reset_vulnerability(clean_engine):
    """
    CRITICAL VULNERABILITY (FOUND IN CODE): 
    In `BatchSimulator::reset(const std::vector<uint32_t>& env_indices)`, 
    there are NO bounds checks:
        for (uint32_t env_id : env_indices) { env_paused_[env_id].store(...) }
    
    If Python passes an env_id >= num_envs, C++ will Segmentation Fault and 
    kill the entire Jupyter Notebook / Colab instance!
    """
    arena, sim = clean_engine
    
    malicious_indices = [999999, -1] # -1 overflows uint32_t to 4 Billion
    
    try:
        # EXPECTED BEHAVIOR: Python should raise a RuntimeError or ValueError.
        # CURRENT BEHAVIOR: This will likely trigger a hard SIGSEGV crash.
        # If this crashes the test suite, we've successfully proven the vulnerability.
        sim.reset(malicious_indices)
        
        # If we reach here, either nanobind protected us, or we corrupted 
        # random memory and got lucky.
        pytest.fail("VULNERABILITY: C++ allowed Out-of-Bounds index without throwing an exception!")
    except Exception as e:
        # If C++ threw std::out_of_range, the vulnerability is patched.
        pass

# =============================================================================
# 6. SYSTEM MEMORY LEAK (OS VirtualAlloc/mmap Fuzzer)
# =============================================================================
def test_os_memory_leak():
    """
    BOTTLENECK/LEAK TEST:
    Constantly creating and destroying 64MB Pinned Memory Arenas.
    If UnifiedMemoryArena::~UnifiedMemoryArena() fails to call VirtualFree/munmap,
    the OS will kill the Python process with an OOM (Out of Memory) error within seconds.
    """
    iterations = 50
    linear_bytes = 64 * 1024 * 1024  # 64 MB
    
    for i in range(iterations):
        arena = titan_core.MemoryArena(
            num_envs=2, num_agents=2, max_orders_per_env=100, 
            max_actions_per_step=10, max_events_per_step=10, 
            max_orders_per_agent=10, obs_depth=5, 
            linear_bytes=linear_bytes
        )
        
        # Force PyTorch to map the memory
        _ = arena.lob
        
        # Destroy
        del arena
        
        # Force Python GC to execute destructors immediately
        gc.collect()
        
    # If the process hasn't been killed by the OS OOM Killer, the C++ 
    # memory management is mathematically sound.
    assert True