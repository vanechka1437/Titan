import math
import torch
from dataclasses import dataclass
from typing import Optional

from titan.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView # type: ignore
from titan.core.actions import ActionBuilder, Side, TimeInForce # type: ignore
from titan.agents.base_agent import BaseAgent, NetworkConfig # type: ignore

@dataclass
class AvellanedaStoikovConfig:
    gamma: float           
    sigma: float           
    kappa: float           
    order_qty: int         
    min_price_ticks: int = 1


class AvellanedaStoikovMM(BaseAgent):
    def __init__(self, 
                 inventory_tensor: torch.Tensor, 
                 config: AvellanedaStoikovConfig, 
                 device: torch.device = torch.device('cpu'), 
                 network: Optional[NetworkConfig] = None):
        
        super().__init__(network)
        self.inventory_tensor = inventory_tensor
        self.config = config
        self.device = device
        
        # Precompute the constant spread term
        self.spread_term = (config.gamma * (config.sigma ** 2)) + \
                           ((2.0 / config.gamma) * math.log(1.0 + (config.gamma / config.kappa)))

    def act(self, 
            active_env_indices: torch.Tensor, 
            lob: ShadowLOBView, 
            events: EventStreamView, 
            active_orders: ActiveOrdersView, 
            action_builder: ActionBuilder) -> None:
        
        num_active = active_env_indices.shape[0]
        if num_active == 0:
            return

        # =====================================================================
        # 1. CANCEL OLD ORDERS (100% synced with C++ ActiveOrderRecord)
        # =====================================================================
        self._cancel_all_active_orders(active_env_indices, active_orders, action_builder)

        # =====================================================================
        # 2. CALCULATE NEW QUOTES
        # =====================================================================
        q = self.inventory_tensor[active_env_indices, self.agent_id]
        mid_float = lob.get_midprice(active_env_indices)[:, self.agent_id]
        
        valid_mid_mask = mid_float > 0
        if not valid_mid_mask.any():
            return
            
        target_envs = active_env_indices[valid_mid_mask]
        q_valid = q[valid_mid_mask]
        mid_valid = mid_float[valid_mid_mask]
        n_targets = target_envs.shape[0]

        # Calculate reservation price shifted by inventory risk
        reservation_price = mid_valid - (q_valid * self.config.gamma * (self.config.sigma ** 2))
        
        half_spread = self.spread_term / 2.0
        bid_float = reservation_price - half_spread
        ask_float = reservation_price + half_spread
        
        bid_ticks = (bid_float / lob.tick_size).to(torch.int64)
        ask_ticks = (ask_float / lob.tick_size).to(torch.int64)
        mid_ticks = (mid_valid / lob.tick_size).to(torch.int64)
        
        # Ensure we don't cross the spread (passive maker logic)
        bid_ticks = torch.minimum(bid_ticks, mid_ticks - 1)
        bid_ticks = torch.clamp(bid_ticks, min=self.config.min_price_ticks)
        ask_ticks = torch.maximum(ask_ticks, mid_ticks + 1)
        
        qtys = torch.full((n_targets,), self.config.order_qty, dtype=torch.int64, device=self.device)

        # =====================================================================
        # 3. SUBMIT NEW ORDERS (Slots 2 and 3, as 0 and 1 are used for cancels)
        # =====================================================================
        bids_side = torch.full((n_targets,), Side.BID, dtype=torch.int64, device=self.device)
        self.submit_limit_orders(
            action_builder=action_builder,
            env_indices=target_envs,
            action_slot=2, 
            sides=bids_side,
            prices=bid_ticks,
            qtys=qtys,
            tifs=torch.full_like(bids_side, TimeInForce.GTC)
        )
        
        asks_side = torch.full((n_targets,), Side.ASK, dtype=torch.int64, device=self.device)
        self.submit_limit_orders(
            action_builder=action_builder,
            env_indices=target_envs,
            action_slot=3, 
            sides=asks_side,
            prices=ask_ticks,
            qtys=qtys,
            tifs=torch.full_like(asks_side, TimeInForce.GTC)
        )

    def _cancel_all_active_orders(self, 
                                  active_env_indices: torch.Tensor, 
                                  active_orders: ActiveOrdersView, 
                                  action_builder: ActionBuilder) -> None:
        """
        Guaranteed O(1) cancellation of all active orders avoiding hash maps.
        """
        raw_orders = active_orders._orders[active_env_indices]
        
        # STRICT C++ ALIGNMENT (ActiveOrderRecord): Slot 0 is OrderId!
        order_ids = raw_orders[:, :, 0] 
        
        # Sort IDs descending. 
        # Real orders (ID > 0) will bubble up to columns 0 and 1, empty slots (0) move to the right.
        sorted_ids, _ = torch.sort(order_ids, descending=True, dim=1)
        
        # Market maker holds max 2 orders (Bid and Ask), so we only process the first 2 columns
        for slot in range(2):
            if slot < sorted_ids.shape[1]:
                ids_to_cancel = sorted_ids[:, slot]
                
                # Submit cancellation commands to Slot 0 and Slot 1 of the action builder.
                # If ID == 0, C++ MatchingEngine safely ignores the request.
                if (ids_to_cancel > 0).any():
                    self.submit_cancellations(
                        action_builder=action_builder,
                        env_indices=active_env_indices,
                        action_slot=slot, 
                        order_ids=ids_to_cancel
                    )