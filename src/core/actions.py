import torch
from enum import IntEnum

class ActionType(IntEnum):
    LIMIT_ORDER = 0
    CANCEL_ORDER = 1 
    MARKET_ORDER = 2  
    NO_OP = 3

class Side(IntEnum):
    BID = 0
    ASK = 1

class TimeInForce(IntEnum):
    GTC = 0  
    IOC = 1  
    FOK = 2  

class ActionBuilder:
    def __init__(self, actions_tensor: torch.Tensor):
        self._actions = actions_tensor
        self.max_actions = self._actions.shape[1]

    def clear(self) -> None:
        self._actions.zero_()

    def make_limit_order(self, 
                         env_indices: torch.Tensor, 
                         action_indices: torch.Tensor,
                         sides: torch.Tensor, 
                         prices: torch.Tensor, 
                         qtys: torch.Tensor, 
                         agent_id: int,
                         tifs: torch.Tensor = None) -> None:
        
        self._actions[env_indices, action_indices, 0] = 0
        
        self._actions[env_indices, action_indices, 1] = qtys.to(torch.int64)
        
        packed_slot2 = prices.to(torch.int64) | (env_indices.to(torch.int64) << 32)
        self._actions[env_indices, action_indices, 2] = packed_slot2
        
        header = agent_id | (ActionType.LIMIT_ORDER << 16) | (sides.to(torch.int64) << 24)
        self._actions[env_indices, action_indices, 3] = header

    def make_market_order(self, 
                          env_indices: torch.Tensor, 
                          action_indices: torch.Tensor,
                          sides: torch.Tensor, 
                          qtys: torch.Tensor,
                          agent_id: int) -> None:
        
        self._actions[env_indices, action_indices, 0] = 0
        self._actions[env_indices, action_indices, 1] = qtys.to(torch.int64)
        
        sentinel_price = torch.where(sides == Side.BID, 4294967295, 0)
        packed_slot2 = sentinel_price | (env_indices.to(torch.int64) << 32)
        self._actions[env_indices, action_indices, 2] = packed_slot2
        
        header = agent_id | (ActionType.MARKET_ORDER << 16) | (sides.to(torch.int64) << 24)
        self._actions[env_indices, action_indices, 3] = header

    def make_cancel_order(self, 
                          env_indices: torch.Tensor, 
                          action_indices: torch.Tensor,
                          order_ids: torch.Tensor,
                          agent_id: int) -> None:
        
        self._actions[env_indices, action_indices, 0] = order_ids.to(torch.int64)
        self._actions[env_indices, action_indices, 1] = 0
        self._actions[env_indices, action_indices, 2] = (env_indices.to(torch.int64) << 32)
        
        header = agent_id | (ActionType.CANCEL_ORDER << 16)
        self._actions[env_indices, action_indices, 3] = header