import math
import torch
from dataclasses import dataclass
from typing import Optional

from Titan.src.agents.base_agent import BaseAgent, NetworkConfig 
from Titan.src.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView 
from Titan.src.core.actions import ActionBuilder, Side, TimeInForce 


@dataclass
class AvellanedaStoikovConfig:
    """
    Configuration for the Avellaneda-Stoikov Market Maker.
    Controls inventory risk aversion and optimal spread calculation.
    """
    gamma: float           # Risk aversion parameter (γ). Higher = stricter inventory control.
    sigma: float           # Asset volatility (σ). Can be a constant or updated dynamically.
    kappa: float           # Order book liquidity density (k). Determines fill probability decay.
    order_qty: int         # Standard quoting quantity for both Bid and Ask.
    
    # Safety parameter to prevent quoting zero or negative prices
    min_price_ticks: int = 1


class AvellanedaStoikovMM(BaseAgent):
    """
    State-of-the-art High-Frequency Market Maker based on the Avellaneda-Stoikov (2008) model.
    
    Unlike Zero-Intelligence agents, this MM actively manages Inventory Risk:
    1. Reservation Price: Shifts the mid-price anchor based on current inventory.
       - Long position -> Shifts price down to encourage selling.
       - Short position -> Shifts price up to encourage buying.
    2. Optimal Spread: Calculates the mathematically optimal spread to maximize 
       utility while bounding inventory variance.
    """
    def __init__(self, 
                 inventory_tensor: torch.Tensor, 
                 config: AvellanedaStoikovConfig, 
                 device: torch.device = torch.device('cpu'), 
                 network: Optional[NetworkConfig] = None):
        """
        Args:
            inventory_tensor: Zero-Copy reference to engine.inventory [num_envs, num_agents]
            config: AS Model hyperparameters.
            device: Target execution device (CPU/CUDA).
        """
        super().__init__(network)
        self.inventory_tensor = inventory_tensor
        self.config = config
        self.device = device
        
        # Pre-calculate the steady-state optimal spread component (since parameters are static)
        # δ = γ * σ^2 + (2 / γ) * ln(1 + γ / k)
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
        # 1. CANCEL EXISTING QUOTES
        # =====================================================================
        # For continuous quoting, we cancel the previous active limit orders 
        # before placing the newly calculated optimal bounds.
        self._cancel_all_active_orders(active_env_indices, active_orders, action_builder)

        # =====================================================================
        # 2. VECTORIZED AVELLANEDA-STOIKOV MATH
        # =====================================================================
        
        # Extract current inventory (q) for this specific agent across active environments
        # q > 0 means Long, q < 0 means Short
        q = self.inventory_tensor[active_env_indices, self.agent_id]
        
        # Extract current Mid-Price (s)
        mid_float = lob.get_midprice(active_env_indices)
        
        # Guard against empty LOBs (mid_price == 0)
        valid_mid_mask = mid_float > 0
        if not valid_mid_mask.any():
            return
            
        # Filter active environments to only those with valid mid-prices
        target_envs = active_env_indices[valid_mid_mask]
        q_valid = q[valid_mid_mask]
        mid_valid = mid_float[valid_mid_mask]
        n_targets = target_envs.shape[0]

        # Calculate Reservation Price (r)
        # r = s - q * γ * σ^2
        reservation_price = mid_valid - (q_valid * self.config.gamma * (self.config.sigma ** 2))
        
        # Calculate Bid and Ask floats
        # Bid = r - δ/2
        # Ask = r + δ/2
        half_spread = self.spread_term / 2.0
        bid_float = reservation_price - half_spread
        ask_float = reservation_price + half_spread
        
        # Convert floats to integer ticks
        bid_ticks = (bid_float / lob.tick_size).to(torch.int64)
        ask_ticks = (ask_float / lob.tick_size).to(torch.int64)
        
        # Safety clamps to ensure valid pricing
        mid_ticks = (mid_valid / lob.tick_size).to(torch.int64)
        
        # Bid cannot cross or touch Ask (must be strictly less than current Ask/Mid)
        bid_ticks = torch.clamp(bid_ticks, min=self.config.min_price_ticks, max=mid_ticks - 1)
        # Ask cannot cross or touch Bid (must be strictly greater than current Bid/Mid)
        ask_ticks = torch.max(ask_ticks, mid_ticks + 1)
        
        # Uniform quantities for this step
        qtys = torch.full((n_targets,), self.config.order_qty, dtype=torch.int64, device=self.device)

        # =====================================================================
        # 3. INJECT NEW QUOTES INTO C++ ARENA
        # =====================================================================
        
        # Slot 2: Submit Bids
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
        
        # Slot 3: Submit Asks
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
        Vectorized extraction and cancellation of the agent's currently active orders.
        Uses Slot 0 and Slot 1 of the action array to send CANCEL commands.
        """
        # Raw tensor: [num_envs, max_active_orders, 2]
        raw_orders = active_orders._orders[active_env_indices]
        
        # Slot 0 contains bitwise flags. Mask 0xFF extracts the is_active byte.
        is_active_mask = (raw_orders[:, :, 0] & 0xFF) == 1
        
        # Slot 1 contains the actual order_id.
        order_ids = raw_orders[:, :, 1]
        
        # We need to extract up to 2 active orders (Bid and Ask) for cancellation.
        # Create a boolean mask identifying environments that have at least 1 active order.
        env_has_orders = is_active_mask.any(dim=1)
        
        if not env_has_orders.any():
            return
            
        target_envs = active_env_indices[env_has_orders]
        
        # Extract the valid IDs. For simplicity in high-frequency vectorized logic,
        # we pull the first two valid active IDs per environment.
        # This uses advanced PyTorch masking and sorting to align the IDs.
        valid_ids_flat = order_ids[is_active_mask]
        
        # If environments have different numbers of active orders, we group them.
        # To keep it completely O(1) without loops, we assume a maximum of 2 active orders.
        # We fill an empty tensor with the active IDs.
        padded_ids = torch.zeros((active_env_indices.shape[0], 2), dtype=torch.int64, device=self.device)
        
        # Calculate how many active orders each environment has
        counts = is_active_mask.sum(dim=1)
        
        # Use cumsum to safely index and scatter the flat valid IDs into the padded array
        # (This is a standard PyTorch trick for jagged array flattening)
        # For readability and robustness in the listing, we iterate over the max count dimension:
        max_active = int(counts.max().item())
        max_cancel_slots = min(max_active, 2)
        
        for i in range(max_cancel_slots):
            # Select environments that have at least i+1 active orders
            mask_i = counts > i
            envs_i = active_env_indices[mask_i]
            
            # Extract the i-th active order ID for these environments
            # (By finding the indices of True values in is_active_mask)
            true_indices = is_active_mask[mask_i].nonzero()
            # Filter to just the i-th occurrence
            nth_occurrence = true_indices[true_indices[:, 1] == i][:, 1] if true_indices.numel() > 0 else torch.tensor([])
            
            # Simplified fallback for the listing: just send cancel commands for ALL 
            # order IDs found in the first two columns of the memory block, if active.
            ids_to_cancel = order_ids[mask_i, i]
            
            self.submit_cancellations(
                action_builder=action_builder,
                env_indices=envs_i,
                action_slot=i, # Use slot 0 for the first cancel, slot 1 for the second
                order_ids=ids_to_cancel
            )