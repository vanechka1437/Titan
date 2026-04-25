import torch
from typing import Tuple

from titan.core.engine import TitanEngine, EngineConfig #type: ignore
from titan.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView #type: ignore
from titan.core.actions import ActionBuilder #type: ignore
from titan.agents.population import Population #type: ignore


class Simulation:
    """
    High-level Simulation Manager for the Titan Digital Twin.
    
    This class orchestrates the interaction between the C++ physics engine
    and the Python vectorized agents (ABM). It initializes all Zero-Copy 
    tensor views and manages the SMDP execution loop.
    """
    def __init__(self, 
                 config: EngineConfig, 
                 population: Population, 
                 tick_size: float = 0.01,
                 device: torch.device = torch.device('cpu')):
        """
        Args:
            config: Hardware and memory limits for the C++ Arena.
            population: The registered pool of trading agents.
            tick_size: Minimum price increment for the simulated asset.
            device: Target execution device.
        """
        self.config = config
        self.population = population
        self.tick_size = tick_size
        self.device = device

        # 1. Initialize the C++ backend and allocate Pinned Memory
        self.engine = TitanEngine(config)
        
        # 2. Initialize Zero-Copy Data Parsers (Views)
        # These objects do NOT copy memory; they simply provide a Pythonic 
        # interface to the raw byte arrays residing in C++ RAM.
        self.lob_view = ShadowLOBView(
            lob_tensor=self.engine.lob, 
            tick_size=self.tick_size, 
            obs_depth=self.config.obs_depth
        )
        
        self.event_view = EventStreamView(
            events_tensor=self.engine.events, 
            cursors_tensor=self.engine.event_cursors, 
            tick_size=self.tick_size
        )
        
        self.orders_view = ActiveOrdersView(
            active_orders_tensor=self.engine.active_orders
        )
        
        # 3. Initialize the Action Injector
        self.action_builder = ActionBuilder(
            actions_tensor=self.engine.actions
        )

        # 4. State tracking for SMDP time jumps
        self.last_times = torch.zeros((config.num_envs,), dtype=torch.int64, device=device)

    def start(self) -> None:
        """Ignites the C++ thread pool. Must be called before stepping."""
        self.engine.start()

    def stop(self) -> None:
        """Safely halts the C++ thread pool to prevent deadlocks."""
        self.engine.stop()

    def reset(self) -> Tuple[ShadowLOBView, EventStreamView]:
        """
        Performs a hardware-level wipe of the simulation state.
        
        Returns:
            Tuple containing the LOB and Event viewers for initial observation.
        """
        self.engine.reset()
        self.last_times = self.engine.get_current_times()
        
        # In a real setup, we might want to run a "warmup" phase here 
        # to populate the LOB with initial liquidity before returning.
        return self.lob_view, self.event_view

    def step(self) -> Tuple[ShadowLOBView, EventStreamView, torch.Tensor]:
        """
        Executes a single Semi-Markov Decision Process (SMDP) cycle.
        
        Workflow:
        1. Resumes C++ physics execution.
        2. Blocks Python until `target_batch_size` agents require inference.
        3. Calculates the stochastic time delta (dt) for RL discounting.
        4. Invokes the Population orchestrator to generate new agent actions.
        
        Returns:
            lob: Updated Shadow LOB Viewer.
            events: Updated Event Stream Viewer.
            dt: Tensor of shape [num_envs] containing elapsed time in nanoseconds 
                since the last step for each parallel environment.
        """
        # 1. C++ Engine Run: Drop GIL and simulate until the next batch is ready
        self.engine.step()

        # 2. Calculate SMDP Time Jumps
        current_times = self.engine.get_current_times()
        dt = current_times - self.last_times
        self.last_times = current_times.clone()

        # 3. Agent Inference: Poll the population and write actions to C++ memory
        self.population.poll_and_act(
            ready_mask=self.engine.ready_mask,
            lob=self.lob_view,
            events=self.event_view,
            active_orders=self.orders_view,
            action_builder=self.action_builder
        )

        # Actions are now written to memory. 
        # The next call to self.engine.step() will ingest them.
        
        return self.lob_view, self.event_view, dt

    def __del__(self):
        """Ensures safe cleanup if garbage collected."""
        if hasattr(self, 'engine'):
            self.stop()