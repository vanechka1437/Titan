import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from typing import Optional

class LOBAnimator:
    """
    Professional-grade Limit Order Book (LOB) Visualizer mimicking a High-Frequency 
    Broker Terminal (e.g., Interactive Brokers TWS / TradingView).

    Engineered for the Titan Digital Twin to resolve asynchronous latency artifacts
    and provide high-fidelity market depth surveillance.
    """
    def __init__(self, tick_size: float, num_levels: int = 15):
        self.tick_size = tick_size
        self.num_levels = num_levels
        self.frames_bids = []
        self.frames_asks = []
        self.frames_mids = []
        
        # Professional Broker Terminal Color Palette
        self.COLOR_BG   = '#0B0E11'  # Deep Slate / Dark mode
        self.COLOR_GRID = '#23272E'  # Subtle dark grid
        self.COLOR_BID  = '#00C076'  # Emerald Green (Success)
        self.COLOR_ASK  = '#FF3B30'  # High-Intensity Red (Danger)
        self.COLOR_MID  = '#F0B90B'  # Gold / Highlight
        self.COLOR_TEXT = '#EAECEF'  # High-readability white/gray

    def capture_frame(self, 
                      lob, 
                      env_idx: int = 0, 
                      agent_idx: int = 0, 
                      clean_book: bool = True) -> None:
        """
        Captures a synchronized snapshot of the LOB from a specific agent's perspective.
        
        Args:
            lob: ShadowLOBView instance.
            env_idx: Parallel environment index.
            agent_idx: Target agent ID (determines the network latency view).
            clean_book: If True, applies 'Broker Smoothing' to resolve latency-induced artifacts.
        """
        # ShadowLOB memory layout: [envs, agents, depth, 4] -> [price_bid, qty_bid, price_ask, qty_ask]
        raw_lob = lob._reshaped[env_idx, agent_idx]
        
        # Optimized vectorized extraction to CPU
        bids_price = (raw_lob[:self.num_levels, 0].float() * self.tick_size).cpu().numpy()
        bids_qty = raw_lob[:self.num_levels, 1].to(torch.int64).cpu().numpy()
        
        asks_price = (raw_lob[:self.num_levels, 2].float() * self.tick_size).cpu().numpy()
        asks_qty = raw_lob[:self.num_levels, 3].to(torch.int64).cpu().numpy()
        
        # Initialize data stacks
        bids = np.column_stack((bids_price[bids_qty > 0], bids_qty[bids_qty > 0]))
        asks = np.column_stack((asks_price[asks_qty > 0], asks_qty[asks_qty > 0]))
        
        # --- Advanced Broker Smoothing (Uncrossing) ---
        # Mitigates 'Phantom Crosses' occurring when network packets for cancellations 
        # arrive out-of-order relative to aggressive fills in the ShadowLOB.
        if clean_book and len(bids) > 0 and len(asks) > 0:
            best_bid = bids[0, 0]
            best_ask = asks[0, 0]
            if best_bid >= best_ask:
                # Calculate the fair equilibrium point for visualization
                mid_theoretical = (best_bid + best_ask) / 2.0
                # Filter out orders that have crossed the theoretical mid
                bids = bids[bids[:, 0] < mid_theoretical]
                asks = asks[asks[:, 0] > mid_theoretical]

        self.frames_bids.append(bids)
        self.frames_asks.append(asks)
        
        # Cache mid-price for line tracking
        if len(bids) > 0 and len(asks) > 0:
            self.frames_mids.append((bids[0, 0] + asks[0, 0]) / 2.0)
        else:
            self.frames_mids.append(0.0)

    def generate_html_animation(self, interval: int = 200, dpi: int = 120) -> HTML:
        """
        Compiles the captured depth history into an interactive H5 animation.
        """
        if not self.frames_bids:
            return HTML("<div style='color:#FF3B30'>System Error: Buffer Empty. No frames captured.</div>")

        fig, ax = plt.subplots(figsize=(11, 6), dpi=dpi)
        fig.patch.set_facecolor(self.COLOR_BG)
        
        def update(i):
            ax.clear()
            ax.set_facecolor(self.COLOR_BG)
            
            bids = self.frames_bids[i]
            asks = self.frames_asks[i]
            mid = self.frames_mids[i]
            
            # 1. Render Depth Histogram
            if len(bids) > 0:
                ax.bar(bids[:, 0], bids[:, 1], color=self.COLOR_BID, 
                       width=self.tick_size * 0.9, alpha=0.85, label='BID DEPTH')
            
            if len(asks) > 0:
                ax.bar(asks[:, 0], asks[:, 1], color=self.COLOR_ASK, 
                       width=self.tick_size * 0.9, alpha=0.85, label='ASK DEPTH')
            
            # 2. Render Vertical Price Tracking Line
            if mid > 0:
                ax.axvline(x=mid, color=self.COLOR_MID, linestyle='-', linewidth=1.5, 
                           alpha=0.5, label=f'MID: {mid:.4f}')
            
            # 3. Dynamic Adaptive Constraints
            # Prevents volume squashing during high-intensity liquidity events
            max_vol = 100.0
            if len(bids) > 0: max_vol = max(max_vol, bids[:, 1].max())
            if len(asks) > 0: max_vol = max(max_vol, asks[:, 1].max())
            ax.set_ylim(0, max_vol * 1.15)
            
            # Centers camera on the active spread
            if len(bids) > 0 and len(asks) > 0:
                x_min = bids[-1, 0] - (self.tick_size * 5)
                x_max = asks[-1, 0] + (self.tick_size * 5)
                ax.set_xlim(x_min, x_max)
            
            # 4. Professional Terminal Styling
            ax.set_title(f"TITAN | BROKER TERMINAL | STEP {i}", loc='left', 
                         fontsize=12, fontweight='bold', color=self.COLOR_TEXT, pad=15)
            ax.set_ylabel("CONTRACTS / VOLUME", color=self.COLOR_TEXT, fontsize=9, fontweight='semibold')
            ax.set_xlabel("PRICE (USD/TICKS)", color=self.COLOR_TEXT, fontsize=9, fontweight='semibold')
            
            ax.grid(True, which='major', axis='both', color=self.COLOR_GRID, linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', colors=self.COLOR_TEXT, labelsize=8)
            
            # Remove container box
            for spine in ax.spines.values():
                spine.set_color(self.COLOR_GRID)
                spine.set_alpha(0.3)
            
            # Professional Legend
            legend = ax.legend(loc='upper right', frameon=True, fontsize=8)
            legend.get_frame().set_facecolor(self.COLOR_BG)
            legend.get_frame().set_edgecolor(self.COLOR_GRID)
            legend.get_frame().set_alpha(0.8)
            for text in legend.get_texts():
                text.set_color(self.COLOR_TEXT)

        anim = animation.FuncAnimation(fig, update, frames=len(self.frames_bids), interval=interval)
        plt.close(fig)
        return HTML(anim.to_jshtml())