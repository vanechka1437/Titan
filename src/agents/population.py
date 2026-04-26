import torch
from typing import List, Optional

from Titan.src.agents.base_agent import BaseAgent
from Titan.src.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView
from Titan.src.core.actions import ActionBuilder 


class Population:
    """
    Simulation Manager for Agent-Based Modeling (ABM).
    
    Acts as the orchestrator connecting the C++ Discrete Event Simulator 
    with the PyTorch vectorized agent implementations.
    
    Responsibilities:
    1. Holds initialized agent instances and maps them to their assigned `agent_id`.
    2. Allows assigning agents to specific geographic/logical subsets of environments 
       via `env_mask` (e.g., simulating a fragmented market).
    3. Performs O(1) vectorized polling of the `ready_mask` to trigger 
       decision-making exclusively for agents whose `next_wakeup_time` has arrived.
    """
    def __init__(self):
        # Ordered list of active agent instances
        self.agents: List[BaseAgent] = []
        
        # Mapping: env_masks[agent_id] stores a boolean tensor [num_envs] 
        # defining which environments this specific agent is allowed to trade in.
        # If None, the agent is active across all environments globally.
        self.env_masks: List[Optional[torch.Tensor]] = []

    def add_agent(self, agent: BaseAgent, env_mask: Optional[torch.Tensor] = None) -> int:
        """
        Registers an initialized agent into the simulation.
        Automatically injects the assigned `agent_id` into the agent's state.
        
        Args:
            agent: An instance of a class inheriting from BaseAgent (e.g., HawkesZITrader).
            env_mask: A boolean 1D tensor of shape [num_envs]. True means the agent 
                      is active in that environment. Defaults to None (active everywhere).
                      
        Returns:
            int: The assigned agent_id (matches the C++ Arena index).
        """
        agent_id = len(self.agents)
        agent.agent_id = agent_id
        
        self.agents.append(agent)
        self.env_masks.append(env_mask)
        
        return agent_id

    def poll_and_act(self, 
                     ready_mask: torch.Tensor, 
                     lob: ShadowLOBView, 
                     events: EventStreamView, 
                     active_orders: ActiveOrdersView, 
                     action_builder: ActionBuilder) -> None:
        """
        The core ABM router. 
        Invoked every time the C++ engine yields execution back to Python.
        
        Args:
            ready_mask: C++ Zero-Copy Tensor of shape [num_envs, num_agents]. 
                        Value == 1 indicates the agent must take an action.
            lob: Viewer for the L2 Limit Order Book.
            events: Viewer for the historical tape (trades/cancellations).
            active_orders: Viewer for agents' currently resting limit orders.
            action_builder: Interface for writing new actions back to C++.
        """
        # CRITICAL: Wipe the action memory buffer from the previous SMDP step 
        # to prevent ghost/stale orders from being resubmitted to the matching engine.
        action_builder.clear()

        # The loop runs strictly over the number of AGENT ROLES (e.g., 8), 
        # NOT over the environments (4096+). This ensures Python overhead is O(1).
        for i, agent in enumerate(self.agents):
            
            # 1. Extract the wakeup signals specifically for this agent 
            # across all parallel universes. Shape: [num_envs]
            is_ready = (ready_mask[:, agent.agent_id] == 1)
            
            # 2. Intersect with the logical subset mask (if applied)
            env_mask = self.env_masks[i]
            if env_mask is not None:
                is_ready = is_ready & env_mask

            # 3. Convert the boolean mask to a dense tensor of environment IDs
            # e.g., [0, 15, 1024, 4005]
            active_env_indices = is_ready.nonzero(as_tuple=False).squeeze(-1)
            
            # 4. Route execution to the agent's vectorized PyTorch logic
            if active_env_indices.shape[0] > 0:
                agent.act(
                    active_env_indices=active_env_indices,
                    lob=lob,
                    events=events,
                    active_orders=active_orders,
                    action_builder=action_builder
                )