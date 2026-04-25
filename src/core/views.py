import torch
from enum import IntEnum
from typing import Dict, Tuple, Optional

# =============================================================================
# CONSTANTS & ENUMS
# Strictly aligned with titan::core::types C++ enums
# =============================================================================
class EventType(IntEnum):
    TRADE = 0
    LOB_UPDATE = 1
    CANCEL = 2
    ACCEPTED = 3
    REJECTED = 4

class Side(IntEnum):
    BID = 0
    ASK = 1

# =============================================================================
# LOB VIEW (Shadow LOB Parser)
# =============================================================================
class ShadowLOBView:
    """
    Zero-Copy parser for the flattened LOB tensor.
    Extracts bids, asks, and mid-prices while applying tick_size scaling.
    
    Expected Tensor Shape: [num_envs, num_agents, obs_depth * 4]
    Assumed Memory Layout per depth level: [bid_price, bid_qty, ask_price, ask_qty]
    """
    def __init__(self, lob_tensor: torch.Tensor, tick_size: float, obs_depth: int):
        self._lob = lob_tensor
        self.tick_size = tick_size
        self.obs_depth = obs_depth
        
        self.num_envs = self._lob.shape[0]
        self.num_agents = self._lob.shape[1]
        
        # Reshape for O(1) vectorized slicing: [E, A, Depth, 4 features]
        self._reshaped = self._lob.view(self.num_envs, self.num_agents, self.obs_depth, 4)

    def get_bids(self, env_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (prices, quantities) for Bids.
        Prices are converted to float and scaled by tick_size.
        """
        data = self._reshaped if env_mask is None else self._reshaped[env_mask]
        
        # Extract [..., Depth, 0] for prices and [..., Depth, 1] for quantities
        prices = data[..., 0].float() * self.tick_size
        qtys = data[..., 1]
        
        return prices, qtys

    def get_asks(self, env_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (prices, quantities) for Asks.
        Prices are converted to float and scaled by tick_size.
        """
        data = self._reshaped if env_mask is None else self._reshaped[env_mask]
        
        prices = data[..., 2].float() * self.tick_size
        qtys = data[..., 3]
        
        return prices, qtys

    def get_midprice(self, env_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculates the Mid-Price based on Top-of-Book (L1) states.
        """
        data = self._reshaped if env_mask is None else self._reshaped[env_mask]
        
        best_bid = data[..., 0, 0].float() * self.tick_size
        best_ask = data[..., 0, 2].float() * self.tick_size
        
        mid = (best_bid + best_ask) / 2.0
        return mid

# =============================================================================
# EVENT STREAM VIEW (Historical Ring Buffer Parser)
# =============================================================================
class EventStreamView:
    """
    Zero-Copy parser for the MarketDataEvent ring buffer.
    
    C++ Struct Memory Layout (Exactly 32 bytes):
    Slot 0 (8 bytes): order_id
    Slot 1 (8 bytes): qty_delta
    Slot 2 (8 bytes): [ price (4b) | owner_id (2b) | taker_id (2b) ]
    Slot 3 (8 bytes): [ type (1b) | side (1b) | padding (6b) ]
    """
    def __init__(self, events_tensor: torch.Tensor, cursors_tensor: torch.Tensor, tick_size: float):
        self._events = events_tensor
        self._cursors = cursors_tensor
        self.tick_size = tick_size

    def get_active_events(self, env_idx: int) -> torch.Tensor:
        """
        Extracts only the valid, newly generated events for a specific environment 
        during the last SMDP tick, ignoring the stale trailing memory in the buffer.
        """
        num_events = self._cursors[env_idx].item()
        
        if num_events == 0:
            return torch.empty((0, 4), dtype=self._events.dtype, device=self._events.device)
            
        return self._events[env_idx, :num_events]

    def decode_events(self, env_idx: int) -> Dict[str, torch.Tensor]:
        """
        Unpacks the raw bitwise C++ event structs into named PyTorch column tensors.
        Assumes Little-Endian architecture (x86/ARM).
        """
        raw_events = self.get_active_events(env_idx)
        
        if raw_events.shape[0] == 0:
            empty_t = torch.tensor([], dtype=torch.int64, device=self._events.device)
            return {
                "order_id": empty_t, "qty_delta": empty_t, "price": empty_t.float(),
                "owner_id": empty_t, "taker_id": empty_t, "type": empty_t, "side": empty_t
            }

        # Slot 0 & 1: Direct 64-bit mapping
        order_ids = raw_events[:, 0]
        qty_deltas = raw_events[:, 1]
        
        # Slot 2: Bitwise unpacking (price, owner_id, taker_id)
        chunk2 = raw_events[:, 2]
        prices = (chunk2 & 0xFFFFFFFF).float() * self.tick_size
        owner_ids = (chunk2 >> 32) & 0xFFFF
        taker_ids = (chunk2 >> 48) & 0xFFFF
        
        # Slot 3: Bitwise unpacking (type, side)
        chunk3 = raw_events[:, 3]
        event_types = chunk3 & 0xFF
        sides = (chunk3 >> 8) & 0xFF

        return {
            "order_id": order_ids,
            "qty_delta": qty_deltas,
            "price": prices,
            "owner_id": owner_ids,
            "taker_id": taker_ids,
            "type": event_types,
            "side": sides
        }

# =============================================================================
# ACTIVE ORDERS VIEW 
# =============================================================================
class ActiveOrdersView:
    """
    Zero-Copy parser for the internal agent order trackers.
    
    Expected Tensor Shape: [num_envs, max_active_orders, 2]
    Memory Layout Assumption (2 int64 slots per ActiveOrderRecord):
    Slot 0: [ is_active (8b) | side (8b) | reserved (48b) ]
    Slot 1: [ order_id ]
    """
    def __init__(self, active_orders_tensor: torch.Tensor):
        self._orders = active_orders_tensor

    def get_agent_orders(self, env_idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns currently active orders waiting in the LOB for a specific environment.
        """
        header = self._orders[env_idx, :, 0]
        
        is_active_mask = (header & 0xFF) == 1
        
        valid_headers = header[is_active_mask]
        sides = (valid_headers >> 8) & 0xFF
        order_ids = self._orders[env_idx, is_active_mask, 1]

        return {
            "side": sides,
            "order_id": order_ids
        }