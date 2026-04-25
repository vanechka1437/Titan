import abc
import torch
from typing import Optional

from titan.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView # type: ignore
from titan.core.actions import ActionBuilder, Side, TimeInForce # type: ignore

class BaseAgent(abc.ABC):
    """
    Abstract base class for all background agents (ZI, Market Makers, RL) in the digital twin.
    
    In a vectorized architecture, a single BaseAgent instance physically controls
    the behavior of its role across ALL parallel environments simultaneously.
    """
    def __init__(self) -> None:
        # Vectorized agent ID (from 0 to num_agents - 1).
        # Automatically injected by the Simulation Manager (Population) during initialization.
        self.agent_id: int = -1

    @abc.abstractmethod
    def act(self, 
            active_env_indices: torch.Tensor, 
            lob: ShadowLOBView, 
            events: EventStreamView, 
            active_orders: ActiveOrdersView, 
            action_builder: ActionBuilder) -> None:
        """
        Main decision-making routine. Invoked by the Simulation Manager during an SMDP step.
        
        Args:
            active_env_indices: 1D Tensor of environment IDs where the agent has "woken up" 
                                and must perform an action.
            lob: Zero-Copy viewer for the Shadow Limit Order Book.
            events: Zero-Copy viewer for the historical MarketDataEvent ring buffer.
            active_orders: Zero-Copy viewer for the agent's currently active orders.
            action_builder: Interface for direct (Zero-Copy) action injection into C++ memory.
        """
        pass

    # =========================================================================
    # VECTORIZED HELPERS (Syntactic Sugar)
    # =========================================================================

    def submit_limit_orders(self,
                            action_builder: ActionBuilder,
                            env_indices: torch.Tensor,
                            action_slot: int,
                            sides: torch.Tensor,
                            prices: torch.Tensor,
                            qtys: torch.Tensor,
                            tifs: Optional[torch.Tensor] = None) -> None:
        """
        Submits a batched array of Limit Orders directly into C++ memory.
        
        Args:
            action_builder: The injected action builder instance.
            env_indices: The target environments for these orders.
            action_slot: The index in the action array (0 to max_actions_per_step - 1).
                         Allows an agent to submit multiple orders in a single tick.
            sides: Tensor of Side (BID/ASK).
            prices: Tensor of prices in integer ticks.
            qtys: Tensor of order quantities.
            tifs: Optional Tensor of TimeInForce policies (defaults to GTC).
        """
        action_indices = torch.full_like(env_indices, action_slot)
        action_builder.make_limit_order(
            env_indices=env_indices,
            action_indices=action_indices,
            sides=sides,
            prices=prices,
            qtys=qtys,
            tifs=tifs
        )

    def submit_market_orders(self,
                             action_builder: ActionBuilder,
                             env_indices: torch.Tensor,
                             action_slot: int,
                             sides: torch.Tensor,
                             qtys: torch.Tensor) -> None:
        """
        Submits a batched array of Market Orders (Aggressive Liquidity Taking).
        """
        action_indices = torch.full_like(env_indices, action_slot)
        action_builder.make_market_order(
            env_indices=env_indices,
            action_indices=action_indices,
            sides=sides,
            qtys=qtys
        )

    def submit_cancellations(self,
                             action_builder: ActionBuilder,
                             env_indices: torch.Tensor,
                             action_slot: int,
                             order_ids: torch.Tensor) -> None:
        """
        Submits a batched array of Cancellation commands for previously placed orders.
        """
        action_indices = torch.full_like(env_indices, action_slot)
        action_builder.make_cancel_order(
            env_indices=env_indices,
            action_indices=action_indices,
            order_ids=order_ids
        )