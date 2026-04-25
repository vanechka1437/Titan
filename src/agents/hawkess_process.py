import torch
from dataclasses import dataclass

from titan.agents.base import BaseAgent # type: ignore
from titan.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView, EventType # type: ignore
from titan.core.actions import ActionBuilder, Side, TimeInForce # type: ignore
from titan.core.distributions import Distribution  # type: ignore


@dataclass
class HawkesZIConfig:
    """
    Configuration for the Hawkes-driven Zero-Intelligence (ZI) background trader.
    Models crowd behavior and volatility clustering via a self-exciting point process.
    """
    # Hawkes Process Dynamics
    mu: float           # Base order arrival intensity (orders per tick)
    alpha: float        # Excitation jump magnitude per observed market trade
    beta: float         # Exponential decay factor of the excitation (0.0 to 1.0)
    
    # Order Parameter Distributions
    order_qty: Distribution        # Distribution for order sizes (e.g., LogNormal or Uniform)
    price_offset: Distribution     # Distance from Mid-Price in ticks (e.g., Exponential)
    
    # Safety Defaults
    default_price_ticks: int = 10000  # Fallback price if the LOB is completely empty


class HawkesZITrader(BaseAgent):
    """
    Econophysics-based Market Maker simulating crowd liquidity.
    
    Order arrival rates are modeled as a Hawkes process: 
    λ(t) = μ + ∫ α * e^{-β(t-s)} dN(s)
    
    This creates realistic microstructural features:
    1. Volatility Clustering: Trades trigger more trades.
    2. Realistic Spreads: Limit orders are placed exponentially close to the Mid-Price.
    """
    def __init__(self, num_envs: int, config: HawkesZIConfig, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.config = config
        self.device = device
        
        # Vectorized Agent State: Current intensity λ for each of the parallel environments
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

        # =====================================================================
        # 1. HAWKES INTENSITY UPDATE (Self-Excitation via Market Flow)
        # =====================================================================
        
        # Accumulate the number of TRADE events that occurred since the last wakeup
        trade_counts = torch.zeros(num_active, dtype=torch.float32, device=self.device)
        
        # Note: Event decoding is done per-environment since the ring buffer cursors 
        # differ across independent simulations.
        for i, env_idx in enumerate(active_env_indices.tolist()):
            env_events = events.decode_events(env_idx)
            trades_mask = (env_events["type"] == EventType.TRADE)
            trade_counts[i] = trades_mask.sum().float()

        # Hawkes Update Rule: Decayed old intensity + Excitation from new trades + Base rate
        old_lambdas = self.lambdas[active_env_indices]
        new_lambdas = (old_lambdas * self.config.beta) + (trade_counts * self.config.alpha) + self.config.mu
        
        # Persist updated state
        self.lambdas[active_env_indices] = new_lambdas

        # =====================================================================
        # 2. ORDER ARRIVAL SAMPLING (Poisson Process)
        # =====================================================================
        
        # Sample the discrete number of orders to place in this exact time step
        # N ~ Poisson(λ)
        num_orders = torch.poisson(new_lambdas).to(torch.int64)
        
        # Hardware Limit: Cap the number of orders to the Engine's max_actions_per_step
        max_actions = action_builder.max_actions
        num_orders = torch.clamp(num_orders, max=max_actions)
        
        max_orders_in_batch = int(num_orders.max().item())
        if max_orders_in_batch == 0:
            return

        # =====================================================================
        # 3. VECTORIZED ORDER GENERATION (Iterating over Action Slots)
        # =====================================================================
        
        # Pre-fetch Mid-Prices for all active environments to avoid recalculating in the loop
        mid_prices_float = lob.get_midprice(active_env_indices)
        mid_prices_ticks = (mid_prices_float / lob.tick_size).to(torch.int64)
        
        # Fallback for completely empty Order Books (Mid-Price == 0)
        empty_lob_mask = (mid_prices_ticks == 0)
        mid_prices_ticks[empty_lob_mask] = self.config.default_price_ticks

        # Loop over available C++ action slots to submit multiple orders if Poisson > 1
        for slot in range(max_orders_in_batch):
            
            # Boolean mask for environments that need to place an order in THIS slot
            slot_mask = num_orders > slot
            target_envs = active_env_indices[slot_mask]
            n_targets = target_envs.shape[0]
            
            if n_targets == 0:
                continue
                
            # Randomly assign Side: 0 for BID, 1 for ASK (Bernoulli p=0.5)
            sides = torch.randint(0, 2, (n_targets,), device=self.device, dtype=torch.int64)
            
            # Sample Quantities
            qtys = self.config.order_qty.sample((n_targets,), self.device).to(torch.int64)
            qtys = torch.clamp(qtys, min=1)
            
            # Sample Price Offsets (Distance from Mid-Price)
            offsets = self.config.price_offset.sample((n_targets,), self.device).to(torch.int64)
            offsets = torch.clamp(offsets, min=1)
            
            # Calculate absolute Prices
            # If Side == BID (0): Multiplier is -1 (Mid - Offset)
            # If Side == ASK (1): Multiplier is +1 (Mid + Offset)
            direction_multiplier = (sides * 2) - 1
            
            target_mids = mid_prices_ticks[slot_mask]
            prices_ticks = target_mids + (offsets * direction_multiplier)
            prices_ticks = torch.clamp(prices_ticks, min=1)

            # Inject directly into C++ MemoryArena (Zero-Copy)
            self.submit_limit_orders(
                action_builder=action_builder,
                env_indices=target_envs,
                action_slot=slot,
                sides=sides,
                prices=prices_ticks,
                qtys=qtys,
                tifs=torch.full_like(sides, TimeInForce.GTC)
            )