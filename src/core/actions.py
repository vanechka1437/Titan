import torch
from enum import IntEnum

# =============================================================================
# CONSTANTS & ENUMS
# Must perfectly align with titan::core::types::ActionType, Side, TimeInForce
# =============================================================================
class ActionType(IntEnum):
    LIMIT_ORDER = 1
    MARKET_ORDER = 2
    CANCEL_ORDER = 3

class Side(IntEnum):
    BID = 0
    ASK = 1

class TimeInForce(IntEnum):
    GTC = 0  # Good 'Til Canceled
    IOC = 1  # Immediate Or Cancel
    FOK = 2  # Fill Or Kill

# =============================================================================
# ACTION BUILDER
# Vectorized encapsulation for zero-copy C++ memory modification.
# =============================================================================
class ActionBuilder:
    """
    Writes vectorized trading actions directly into the C++ Arena memory.
    Expects self._actions to be a Zero-Copy Tensor of shape:
    [num_envs, max_actions_per_step, 4] (dtype=torch.int64)
    
    Memory Layout Assumption (4 int64 slots per ActionPayload):
    Slot 0: [ action_type (8b) | side (8b) | tif (8b) | reserved ]
    Slot 1: [ price (ticks) ]
    Slot 2: [ quantity ]
    Slot 3: [ order_id (for cancellations) ]
    """
    def __init__(self, actions_tensor: torch.Tensor):
        self._actions = actions_tensor
        self.max_actions = self._actions.shape[1]

    def clear(self) -> None:
        """
        Zeroes out the action buffer for the current SMDP step.
        Must be called before writing new actions.
        """
        self._actions.zero_()

    def make_limit_order(self, 
                         env_indices: torch.Tensor, 
                         action_indices: torch.Tensor,
                         sides: torch.Tensor, 
                         prices: torch.Tensor, 
                         qtys: torch.Tensor, 
                         tifs: torch.Tensor = None) -> None:
        """
        Vectorized injection of Limit Orders.
        """
        if tifs is None:
            tifs = torch.full_like(sides, TimeInForce.GTC)

        # Bitwise packing for Slot 0
        header = (ActionType.LIMIT_ORDER) | (sides << 8) | (tifs << 16)
        
        # Write directly to C++ RAM (Zero-Copy)
        self._actions[env_indices, action_indices, 0] = header.to(torch.int64)
        self._actions[env_indices, action_indices, 1] = prices.to(torch.int64)
        self._actions[env_indices, action_indices, 2] = qtys.to(torch.int64)
        self._actions[env_indices, action_indices, 3] = 0

    def make_market_order(self, 
                          env_indices: torch.Tensor, 
                          action_indices: torch.Tensor,
                          sides: torch.Tensor, 
                          qtys: torch.Tensor) -> None:
        """
        Vectorized injection of Market Orders (Aggressive Liquidity Taking).
        """
        header = (ActionType.MARKET_ORDER) | (sides << 8)
        
        self._actions[env_indices, action_indices, 0] = header.to(torch.int64)
        self._actions[env_indices, action_indices, 1] = 0  # Market orders ignore price
        self._actions[env_indices, action_indices, 2] = qtys.to(torch.int64)
        self._actions[env_indices, action_indices, 3] = 0

    def make_cancel_order(self, 
                          env_indices: torch.Tensor, 
                          action_indices: torch.Tensor,
                          order_ids: torch.Tensor) -> None:
        """
        Vectorized injection of Cancellation Orders.
        """
        header = torch.tensor(ActionType.CANCEL_ORDER, dtype=torch.int64, device=self._actions.device)
        
        self._actions[env_indices, action_indices, 0] = header
        self._actions[env_indices, action_indices, 1] = 0
        self._actions[env_indices, action_indices, 2] = 0
        self._actions[env_indices, action_indices, 3] = order_ids.to(torch.int64)