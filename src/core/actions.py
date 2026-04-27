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
        # New 4D Architecture: [num_envs, num_agents, max_actions_per_agent, 4]
        self._actions = actions_tensor
        self.max_actions_per_agent = self._actions.shape[2]

    def clear(self) -> None:
        # Rapidly zero out the tensor and set NO_OP flag in all headers (Slot 3).
        # The C++ engine will see (action_type == 3) and skip these slots in O(1).
        self._actions.fill_(0)
        self._actions[..., 3] = (ActionType.NO_OP << 16)

    def make_limit_order(self, 
                         env_indices: torch.Tensor, 
                         action_indices: torch.Tensor, # Agent's personal slot (0 to max_actions-1)
                         sides: torch.Tensor, 
                         prices: torch.Tensor, 
                         qtys: torch.Tensor, 
                         agent_id: int,
                         tifs: torch.Tensor = None) -> None:
        
        self._actions[env_indices, agent_id, action_indices, 0] = 0
        self._actions[env_indices, agent_id, action_indices, 1] = qtys.to(torch.int64)
        
        # Pack environment ID into the upper 32 bits and price into the lower 32 bits
        packed_slot2 = prices.to(torch.int64) | (env_indices.to(torch.int64) << 32)
        self._actions[env_indices, agent_id, action_indices, 2] = packed_slot2
        
        # Pack agent_id, action_type, and side into the header
        header = agent_id | (ActionType.LIMIT_ORDER << 16) | (sides.to(torch.int64) << 24)
        self._actions[env_indices, agent_id, action_indices, 3] = header

    def make_market_order(self, 
                          env_indices: torch.Tensor, 
                          action_indices: torch.Tensor,
                          sides: torch.Tensor, 
                          qtys: torch.Tensor,
                          agent_id: int) -> None:
        
        self._actions[env_indices, agent_id, action_indices, 0] = 0
        self._actions[env_indices, agent_id, action_indices, 1] = qtys.to(torch.int64)
        
        # Sentinel Price for Market Orders: BID=UINT32_MAX, ASK=0
        sentinel_price = torch.where(sides == Side.BID, 4294967295, 0)
        packed_slot2 = sentinel_price | (env_indices.to(torch.int64) << 32)
        self._actions[env_indices, agent_id, action_indices, 2] = packed_slot2
        
        header = agent_id | (ActionType.MARKET_ORDER << 16) | (sides.to(torch.int64) << 24)
        self._actions[env_indices, agent_id, action_indices, 3] = header

    def make_cancel_order(self, 
                          env_indices: torch.Tensor, 
                          action_indices: torch.Tensor,
                          order_ids: torch.Tensor,
                          agent_id: int) -> None:
        
        self._actions[env_indices, agent_id, action_indices, 0] = order_ids.to(torch.int64)
        self._actions[env_indices, agent_id, action_indices, 1] = 0
        self._actions[env_indices, agent_id, action_indices, 2] = (env_indices.to(torch.int64) << 32)
        
        header = agent_id | (ActionType.CANCEL_ORDER << 16)
        self._actions[env_indices, agent_id, action_indices, 3] = header