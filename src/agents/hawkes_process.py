import torch
from dataclasses import dataclass
from typing import Optional

from Titan.src.agents.base_agent import BaseAgent, NetworkConfig 
from Titan.src.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView, EventType 
from Titan.src.core.actions import ActionBuilder, Side, TimeInForce 
from Titan.src.core.distributions import Distribution 

@dataclass
class HawkesNoiseConfig:
    mu: float           
    alpha: float        
    beta: float         
    order_qty: Distribution
    limit_prob: float = 0.5
    price_offset_ticks: int = 2

class HawkesNoiseTrader(BaseAgent):
    def __init__(self, num_envs: int, config: HawkesNoiseConfig, device: torch.device = torch.device('cpu'), network: Optional[NetworkConfig] = None):
        super().__init__(network)
        self.config = config
        self.device = device
        self.lambdas = torch.full((num_envs,), config.mu, dtype=torch.float32, device=device)

    def act(self, 
            active_env_indices: torch.Tensor, 
            lob: ShadowLOBView, 
            events: EventStreamView, 
            active_orders: ActiveOrdersView, 
            action_builder: ActionBuilder) -> None:
        
        num_active = active_env_indices.shape[0]
        if num_active == 0:
            return

        raw_events = events._events[active_env_indices]
        cursors = events._cursors[active_env_indices].unsqueeze(1)
        max_events = raw_events.shape[1]
        
        arange = torch.arange(max_events, device=self.device).unsqueeze(0).expand(num_active, max_events)
        valid_mask = arange < cursors
        
        event_types = raw_events[:, :, 3] & 0xFF
        is_trade = (event_types == EventType.TRADE) & valid_mask
        
        trade_counts = is_trade.sum(dim=1).float()

        old_lambdas = self.lambdas[active_env_indices]
        new_lambdas = (old_lambdas * self.config.beta) + (trade_counts * self.config.alpha) + self.config.mu
        self.lambdas[active_env_indices] = new_lambdas

        num_orders = torch.poisson(new_lambdas).to(torch.int64)
        num_orders = torch.clamp(num_orders, max=action_builder.max_actions_per_agent)
        
        max_orders_in_batch = int(num_orders.max().item())
        if max_orders_in_batch == 0:
            return

        mid_prices = lob.get_midprice(active_env_indices)[:, self.agent_id]
        mid_ticks = (mid_prices / lob.tick_size).to(torch.int64)

        for slot in range(max_orders_in_batch):
            slot_mask = num_orders > slot
            target_envs = active_env_indices[slot_mask]
            n_targets = target_envs.shape[0]
            
            if n_targets == 0:
                continue
                
            sides = torch.randint(0, 2, (n_targets,), device=self.device, dtype=torch.int64)
            qtys = self.config.order_qty.sample((n_targets,), self.device).to(torch.int64)
            qtys = torch.clamp(qtys, min=1)
            
            # Split into Market and Limit orders
            rand_val = torch.rand((n_targets,), device=self.device)
            is_limit = rand_val < self.config.limit_prob
            is_market = ~is_limit

            # Process Market Orders
            if is_market.any():
                self.submit_market_orders(
                    action_builder=action_builder,
                    env_indices=target_envs[is_market],
                    action_slot=slot,
                    sides=sides[is_market],
                    qtys=qtys[is_market]
                )

            # Process Limit Orders
            if is_limit.any():
                # Side 0 (BUY) -> price = mid - offset
                # Side 1 (SELL) -> price = mid + offset
                multiplier = (sides[is_limit] * 2) - 1
                prices = mid_ticks[slot_mask][is_limit] + (self.config.price_offset_ticks * multiplier)
                prices = torch.clamp(prices, min=1)

                self.submit_limit_orders(
                    action_builder=action_builder,
                    env_indices=target_envs[is_limit],
                    action_slot=slot,
                    sides=sides[is_limit],
                    prices=prices,
                    qtys=qtys[is_limit],
                    tifs=torch.full_like(sides[is_limit], TimeInForce.GTC)
                )