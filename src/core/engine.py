import os
import sys
import torch
from dataclasses import dataclass
from typing import List, Optional

# =============================================================================
# DYNAMIC C++ CORE IMPORT
# Automatically locates and loads the compiled nanobind extension.
# =============================================================================
try:
    import titan_core # type: ignore
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir_win = os.path.abspath(os.path.join(current_dir, '../../build/Release'))
    build_dir_lin = os.path.abspath(os.path.join(current_dir, '../../build'))
    
    if os.path.exists(build_dir_win):
        sys.path.insert(0, build_dir_win)
    elif os.path.exists(build_dir_lin):
        sys.path.insert(0, build_dir_lin)
    else:
        raise RuntimeError("titan_core extension not found. Please build the C++ project.")
    
    import titan_core # type: ignore


@dataclass
class EngineConfig:
    """
    Low-level hardware and memory limits configuration.
    These parameters directly dictate the size of the pre-allocated OS Pinned Memory.
    Must perfectly align with the C++ UnifiedMemoryArena constructor.
    """
    num_envs: int = 4096
    num_agents: int = 8
    num_threads: int = os.cpu_count() or 4
    target_batch_size: int = 4096
    max_orders_per_env: int = 4096
    max_actions_per_step: int = 16
    max_events_per_step: int = 256
    max_active_orders_per_agent: int = 128
    obs_depth: int = 20
    linear_bytes: int = 256 * 1024 * 1024  # 256 MB


class TitanEngine:
    """
    Low-level C++ Bridge Manager.
    Strictly encapsulates memory allocation (MemoryArena), thread lifecycle (Simulator),
    and Zero-Copy DLPack tensor references. 
    
    This class does NOT handle business logic (like OFI calculation or PnL with fees).
    It only routes raw bytes between PyTorch and the C++ Discrete Event Simulator.
    """
    
    def __init__(self, config: EngineConfig):
        self.config = config
        
        # CRITICAL: Prevent PyTorch from spawning background CPU threads for intra-op 
        # parallelism. This ensures the C++ thread pool gets 100% of the CPU cores 
        # without context-switching contention.
        torch.set_num_threads(1)

        # 1. OS-Pinned Memory Allocation
        self._arena = titan_core.MemoryArena(
            num_envs=config.num_envs,
            num_agents=config.num_agents,
            max_orders_per_env=config.max_orders_per_env,
            max_actions_per_step=config.max_actions_per_step,
            max_events_per_step=config.max_events_per_step,
            max_orders_per_agent=config.max_active_orders_per_agent,
            obs_depth=config.obs_depth,
            linear_bytes=config.linear_bytes
        )

        # 2. Asynchronous Thread Pool Initialization
        self._sim = titan_core.Simulator(
            arena=self._arena,
            target_batch_size=config.target_batch_size,
            num_threads=config.num_threads
        )

        # =====================================================================
        # 3. ZERO-COPY DLPACK TENSOR CACHING
        # These properties invoke the nanobind `.def_prop_ro` bindings.
        # They return torch.Tensor objects that point directly to the C++ RAM.
        # No serialization, no deep copies. O(1) instantaneous access.
        # =====================================================================
        
        # [num_envs, num_agents] - Write 1 to wake up C++ for a specific agent
        self.ready_mask: torch.Tensor = self._arena.ready_mask
        
        # [num_envs, max_actions_per_step, 4] - 32-byte ActionPayloads
        self.actions: torch.Tensor = self._arena.actions
        
        # [num_envs, max_events_per_step, 4] - 32-byte MarketDataEvents (Historical ring buffer)
        self.events: torch.Tensor = self._arena.events
        
        # [num_envs, max_active_orders, 2] - 16-byte ActiveOrderRecords
        self.active_orders: torch.Tensor = self._arena.active_orders
        
        # [num_envs] - Tracks the current write index in the global events ring buffer
        self.event_cursors: torch.Tensor = self._arena.event_cursors
        
        # [num_envs, num_agents, obs_depth * 4] - Flattened ShadowLOB state
        self.lob: torch.Tensor = self._arena.lob
        
        # [num_envs, num_agents] - Raw agent cash (Mark-to-Market base)
        self.cash: torch.Tensor = self._arena.cash
        
        # [num_envs, num_agents] - Raw agent inventory (Position)
        self.inventory: torch.Tensor = self._arena.inventory


    def get_current_times(self) -> torch.Tensor:
        """
        Dynamically extracts the nanosecond epoch clocks for all environments.
        Since current_time is stored inside interleaved C++ objects (EnvironmentState),
        this function dynamically allocates a 1D tensor and copies the data.
        
        Returns:
            torch.Tensor: Shape [num_envs], dtype int64.
        """
        return self._sim.get_current_times(self.config.num_envs)


    def start(self) -> None:
        """
        Ignites the background C++ thread pool. 
        Must be called before the first `step()`.
        """
        self._sim.start()


    def stop(self) -> None:
        """
        Safely joins all C++ threads and halts the physics engine.
        Prevents deadlocks during script termination.
        """
        self._sim.stop()


    def step(self) -> int:
        """
        Executes a single SMDP forward pass.
        
        Workflow:
        1. Notifies C++ to read `self.actions` where `self.ready_mask == 1`.
        2. Drops the Python GIL (gil_scoped_release).
        3. C++ processes event queues asynchronously.
        4. Blocks the main thread until `target_batch_size` agents hit an AGENT_WAKEUP event.
        5. Reacquires the GIL and returns execution to Python.
        
        Returns:
            int: The exact number of agents currently paused and awaiting Python inference.
                 (Can be >= target_batch_size).
        """
        self._sim.resume_batch()
        ready_count = self._sim.wait_for_batch()
        return ready_count


    def reset(self, env_indices: Optional[List[int]] = None) -> None:
        """
        Hardware-level memory reset.
        
        Args:
            env_indices: List of specific environment IDs to wipe (used at the end 
                         of individual RL episodes). If None, performs a global 
                         O(1) memset wipe across all environments.
        """
        if env_indices is None:
            self._sim.reset_all()
        else:
            self._sim.reset(env_indices)


    def __del__(self):
        """
        Aggressive cleanup hook. Ensures that if the Python Garbage Collector 
        destroys this object, the C++ threads are safely terminated first to 
        prevent dangling pointer segfaults.
        """
        if hasattr(self, '_sim'):
            self.stop()