import torch
from dataclasses import dataclass
from typing import Optional

from titan.agents.base_agent import BaseAgent #type: ignore
from titan.core.views import ShadowLOBView, EventStreamView, ActiveOrdersView, EventType #type: ignore
from titan.core.actions import ActionBuilder, Side, TimeInForce #type: ignore
from titan.core.distributions import Distribution #type: ignore

@dataclass
class HawkesZIConfig:
    mu: float           
    alpha: float        
    beta: float         
    order_qty: Distribution
    price_offset: Distribution
    default_price_ticks: int = 10000

class HawkesZITrader(BaseAgent):
    def __init__(self, num_envs: int, config: HawkesZIConfig, device: torch.device = torch.device('cpu')):
        super().__init__()
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

        # =====================================================================
        # 1. СУПЕР-ВЕКТОРИЗОВАННОЕ ОБНОВЛЕНИЕ HAWKES (Без Python циклов!)
        # =====================================================================
        # Достаем сырую память событий сразу для всех активных сред
        raw_events = events._events[active_env_indices]
        cursors = events._cursors[active_env_indices].unsqueeze(1)
        max_events = raw_events.shape[1]
        
        # Создаем матричную маску валидности (отсекаем мусор после курсора кольцевого буфера)
        arange = torch.arange(max_events, device=self.device).unsqueeze(0).expand(num_active, max_events)
        valid_mask = arange < cursors
        
        # Тип события лежит в младшем байте слота 3. Распаковываем матрично:
        event_types = raw_events[:, :, 3] & 0xFF
        is_trade = (event_types == EventType.TRADE) & valid_mask
        
        # ОДНА операция sum заменяет 512 итераций цикла!
        trade_counts = is_trade.sum(dim=1).float()

        old_lambdas = self.lambdas[active_env_indices]
        new_lambdas = (old_lambdas * self.config.beta) + (trade_counts * self.config.alpha) + self.config.mu
        self.lambdas[active_env_indices] = new_lambdas

        # =====================================================================
        # 2. ГЕНЕРАЦИЯ ОРДЕРОВ
        # =====================================================================
        num_orders = torch.poisson(new_lambdas).to(torch.int64)
        num_orders = torch.clamp(num_orders, max=action_builder.max_actions)
        
        max_orders_in_batch = int(num_orders.max().item())
        if max_orders_in_batch == 0:
            return

        mid_prices_float = lob.get_midprice(active_env_indices)
        mid_prices_ticks = (mid_prices_float / lob.tick_size).to(torch.int64)
        
        empty_lob_mask = (mid_prices_ticks == 0)
        mid_prices_ticks[empty_lob_mask] = self.config.default_price_ticks

        # =====================================================================
        # 3. ОТПРАВКА БАТЧЕЙ (В цикле только по max_actions, т.е. максимум 16 раз)
        # =====================================================================
        for slot in range(max_orders_in_batch):
            slot_mask = num_orders > slot
            target_envs = active_env_indices[slot_mask]
            n_targets = target_envs.shape[0]
            
            if n_targets == 0:
                continue
                
            sides = torch.randint(0, 2, (n_targets,), device=self.device, dtype=torch.int64)
            qtys = self.config.order_qty.sample((n_targets,), self.device).to(torch.int64)
            qtys = torch.clamp(qtys, min=1)
            
            offsets = self.config.price_offset.sample((n_targets,), self.device).to(torch.int64)
            offsets = torch.clamp(offsets, min=1)
            
            direction_multiplier = (sides * 2) - 1
            prices_ticks = mid_prices_ticks[slot_mask] + (offsets * direction_multiplier)
            prices_ticks = torch.clamp(prices_ticks, min=1)

            self.submit_limit_orders(
                action_builder=action_builder,
                env_indices=target_envs,
                action_slot=slot,
                sides=sides,
                prices=prices_ticks,
                qtys=qtys,
                tifs=torch.full_like(sides, TimeInForce.GTC)
            )