import torch
import math
from dataclasses import dataclass

from titan.agents.base_agent import BaseAgent # type: ignore
from titan.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView # type: ignore
from titan.core.actions import ActionBuilder, Side # type: ignore
from titan.core.distributions import Distribution # type: ignore


@dataclass
class FundamentalConfig:
    """
    Configuration for the Fundamental (Value) Trader.
    Models institutional investors who trade based on a hidden "Fair Value" of the asset.
    """
    # Ornstein-Uhlenbeck (OU) Process parameters for Fair Value drift
    ou_mean: float                 # The long-term macroeconomic anchor price (in integer ticks)
    ou_reversion_rate: float       # Kappa (k): How strongly the fair value reverts to the mean (0.0 to 1.0)
    ou_volatility: float           # Sigma (σ): Random walk noise applied to the fair value per step
    
    # Trading behavior
    signal_threshold_ticks: int    # Minimum deviation between LOB Mid-Price and Fair Value to trigger a trade
    order_qty: Distribution        # Distribution for the order size


class FundamentalTrader(BaseAgent):
    """
    Econophysics Value Trader.
    
    Role in the simulation:
    Acts as the "gravitational pull" of the market. While Momentum traders chase trends
    and Market Makers profit off the spread, the Fundamental trader ensures Mean Reversion.
    
    Logic:
    1. Maintains a hidden `fair_value` for each environment that drifts via an OU process.
    2. Compares the current LOB Mid-Price to the hidden `fair_value`.
    3. If the asset is deeply undervalued (Mid < Fair - Threshold), it buys aggressively.
    4. If the asset is deeply overvalued (Mid > Fair + Threshold), it sells aggressively.
    """
    def __init__(self, num_envs: int, config: FundamentalConfig, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.config = config
        self.device = device
        
        # Vectorized Agent State: The hidden "True" Fair Value for each environment.
        # Initialized exactly at the long-term mean.
        self.fair_values = torch.full((num_envs,), config.ou_mean, dtype=torch.float32, device=device)

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
        # 1. ORNSTEIN-UHLENBECK FAIR VALUE DRIFT (Macroeconomic hidden state)
        # =====================================================================
        # Formula: dV_t = k * (μ - V_t) + σ * dW_t
        # Where dW_t is standard Brownian motion (Normal distribution)
        
        old_fair_values = self.fair_values[active_env_indices]
        
        # Mean reversion component: k * (μ - V_t)
        reversion_force = self.config.ou_reversion_rate * (self.config.ou_mean - old_fair_values)
        
        # Brownian motion noise: N(0, σ)
        noise = torch.empty_like(old_fair_values).normal_(mean=0.0, std=self.config.ou_volatility)
        
        # Update the hidden states
        new_fair_values = old_fair_values + reversion_force + noise
        self.fair_values[active_env_indices] = new_fair_values

        # =====================================================================
        # 2. PRICE EXTRACTION & DEVIATION CALCULATION
        # =====================================================================
        
        current_mids_float = lob.get_midprice(active_env_indices)
        current_mids_ticks = (current_mids_float / lob.tick_size).to(torch.int64)
        
        # Guard against completely empty limit order books
        valid_mid_mask = current_mids_ticks > 0
        if not valid_mid_mask.any():
            return
            
        target_envs = active_env_indices[valid_mid_mask]
        valid_mids = current_mids_ticks[valid_mid_mask]
        valid_fairs = new_fair_values[valid_mid_mask].to(torch.int64)
        
        # Calculate Deviation: True Value - Market Price
        # Positive deviation -> Asset is Undervalued (Cheap) -> BUY
        # Negative deviation -> Asset is Overvalued (Expensive) -> SELL
        deviation = valid_fairs - valid_mids

        # =====================================================================
        # 3. TRADING LOGIC (ARBITRAGE CAPTURE)
        # =====================================================================
        
        # Fundamental traders use Market Orders to instantly capture the mispricing 
        # when the deviation exceeds their strict threshold.
        buy_trigger_mask = deviation > self.config.signal_threshold_ticks
        sell_trigger_mask = deviation < -self.config.signal_threshold_ticks
        
        # Process BUYS (Undervalued)
        if buy_trigger_mask.any():
            buy_envs = target_envs[buy_trigger_mask]
            n_buys = buy_envs.shape[0]
            
            sides = torch.full((n_buys,), Side.BID, dtype=torch.int64, device=self.device)
            qtys = self.config.order_qty.sample((n_buys,), self.device).to(torch.int64)
            qtys = torch.clamp(qtys, min=1)
            
            # Action Slot 0 for Buys
            self.submit_market_orders(
                action_builder=action_builder,
                env_indices=buy_envs,
                action_slot=0,
                sides=sides,
                qtys=qtys
            )
            
        # Process SELLS (Overvalued)
        if sell_trigger_mask.any():
            sell_envs = target_envs[sell_trigger_mask]
            n_sells = sell_envs.shape[0]
            
            sides = torch.full((n_sells,), Side.ASK, dtype=torch.int64, device=self.device)
            qtys = self.config.order_qty.sample((n_sells,), self.device).to(torch.int64)
            qtys = torch.clamp(qtys, min=1)
            
            # Action Slot 1 for Sells to prevent collision
            self.submit_market_orders(
                action_builder=action_builder,
                env_indices=sell_envs,
                action_slot=1,
                sides=sides,
                qtys=qtys
            )