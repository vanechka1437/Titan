from typing import Optional

import torch
from dataclasses import dataclass

from Titan.src.agents.base_agent import NetworkConfig
from Titan.src.agents.base_agent import BaseAgent 
from Titan.src.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView
from Titan.src.core.actions import ActionBuilder, Side 
from Titan.src.core.distributions import Distribution 


@dataclass
class MomentumConfig:
    """
    Configuration for the Momentum (Risk Taker) Trader.
    This agent follows trends by comparing the current price to an Exponential Moving Average (EMA).
    """
    ema_alpha: float               # Smoothing factor for the EMA (0.0 to 1.0). Higher = tracks current price faster.
    signal_threshold: float        # Minimum relative deviation from EMA to trigger a trade (e.g., 0.0005 for 5 bps).
    order_qty: Distribution        # Distribution for aggressive order sizes.


class MomentumTrader(BaseAgent):
    """
    Trend-following aggressive trader (Risk Taker).
    
    Econophysics role:
    Unlike Market Makers who provide liquidity, the Momentum Trader consumes it (Taker).
    They create heavy-tailed price distributions (fat tails) and market impact by 
    aggressively buying into uptrends and selling into downtrends using Market Orders.
    """
    def __init__(self, num_envs: int, config: MomentumConfig, device: torch.device = torch.device('cpu'), network: Optional[NetworkConfig] = None):
        """
        Args:
            num_envs: Total number of parallel simulation environments.
            config: Momentum strategy hyperparameters.
            device: Target execution device (CPU/CUDA).
            network: Optional network configuration.
        """
        super().__init__(network)
        self.config = config
        self.device = device
        
        # Vectorized Agent State: Tracks the EMA of the Mid-Price for each environment
        self.ema_prices = torch.zeros((num_envs,), dtype=torch.float32, device=device)

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
        # 1. PRICE EXTRACTION & EMA INITIALIZATION
        # =====================================================================
        
        # Extract current Mid-Price
        current_mids = lob.get_midprice(active_env_indices)[:, self.id]  
        
        # Guard against empty Order Books (cannot calculate momentum without prices)
        valid_mid_mask = current_mids > 0
        if not valid_mid_mask.any():
            return
            
        target_envs = active_env_indices[valid_mid_mask]
        mids_valid = current_mids[valid_mid_mask]
        
        # Initialize EMA for environments that are running for the very first time (EMA == 0)
        uninitialized_mask = (self.ema_prices[target_envs] == 0)
        if uninitialized_mask.any():
            envs_to_init = target_envs[uninitialized_mask]
            self.ema_prices[envs_to_init] = mids_valid[uninitialized_mask]

        # =====================================================================
        # 2. VECTORIZED SIGNAL CALCULATION
        # =====================================================================
        
        old_emas = self.ema_prices[target_envs]
        
        # Calculate Momentum Signal: (Current Price - EMA) / EMA
        # Positive signal means uptrend, Negative signal means downtrend
        signal = (mids_valid - old_emas) / old_emas
        
        # Update EMA state for the next step
        new_emas = (self.config.ema_alpha * mids_valid) + ((1.0 - self.config.ema_alpha) * old_emas)
        self.ema_prices[target_envs] = new_emas

        # =====================================================================
        # 3. TRADING LOGIC (AGGRESSIVE MARKET ORDERS)
        # =====================================================================
        
        # Determine which environments triggered a BUY or SELL signal based on the threshold
        buy_trigger_mask = signal > self.config.signal_threshold
        sell_trigger_mask = signal < -self.config.signal_threshold
        
        # Process BUYS
        if buy_trigger_mask.any():
            buy_envs = target_envs[buy_trigger_mask]
            n_buys = buy_envs.shape[0]
            
            # Risk takers buy aggressively using Market Orders (crossing the spread)
            sides = torch.full((n_buys,), Side.BID, dtype=torch.int64, device=self.device)
            qtys = self.config.order_qty.sample((n_buys,), self.device).to(torch.int64)
            qtys = torch.clamp(qtys, min=1)
            
            # Write to Action Slot 0
            self.submit_market_orders(
                action_builder=action_builder,
                env_indices=buy_envs,
                action_slot=0,
                sides=sides,
                qtys=qtys
            )
            
        # Process SELLS
        if sell_trigger_mask.any():
            sell_envs = target_envs[sell_trigger_mask]
            n_sells = sell_envs.shape[0]
            
            sides = torch.full((n_sells,), Side.ASK, dtype=torch.int64, device=self.device)
            qtys = self.config.order_qty.sample((n_sells,), self.device).to(torch.int64)
            qtys = torch.clamp(qtys, min=1)
            
            # Write to Action Slot 1 (to avoid overwriting Slot 0 if somehow both triggered, 
            # though mathematically impossible here, it's good practice for independent logic branches)
            self.submit_market_orders(
                action_builder=action_builder,
                env_indices=sell_envs,
                action_slot=1,
                sides=sides,
                qtys=qtys
            )