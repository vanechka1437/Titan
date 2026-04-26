import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from Titan.src.core.views import ShadowLOBView 

class LOBAnimator:
    """
    Offline Visualizer for Google Colab.
    Captures LOB snapshots during the high-frequency simulation loop with minimal overhead,
    and renders an interactive HTML5 video animation after the simulation ends.
    """
    def __init__(self, tick_size: float, num_levels: int = 10):
        self.tick_size = tick_size
        self.num_levels = num_levels
        
        # Memory buffers for captured frames
        self.frames_bids = []
        self.frames_asks = []
        self.frames_mids = []

    def capture_frame(self, lob: ShadowLOBView, env_idx: int = 0) -> None:
        """
        Takes a microsecond-fast snapshot of the LOB for a specific environment.
        Call this inside your main simulation loop (e.g., every 10 steps to save RAM).
        """
        # Extract data for a single environment, agent 0 (public LOB view)
        # lob._reshaped shape: [num_envs, num_agents, depth, 4]
        raw_lob = lob._reshaped[env_idx, 0]
        
        # Convert to CPU numpy arrays for Matplotlib (only slicing the needed levels)
        bids_price = (raw_lob[:self.num_levels, 0].float() * self.tick_size).cpu().numpy()
        bids_qty = raw_lob[:self.num_levels, 1].to(torch.int64).cpu().numpy()
        
        asks_price = (raw_lob[:self.num_levels, 2].float() * self.tick_size).cpu().numpy()
        asks_qty = raw_lob[:self.num_levels, 3].to(torch.int64).cpu().numpy()
        
        self.frames_bids.append(np.column_stack((bids_price, bids_qty)))
        self.frames_asks.append(np.column_stack((asks_price, asks_qty)))
        
        # Calculate Mid-Price
        best_bid = bids_price[0] if bids_qty[0] > 0 else 0.0
        best_ask = asks_price[0] if asks_qty[0] > 0 else 0.0
        mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0.0
        
        self.frames_mids.append(mid)

    def generate_html_animation(self, interval: int = 100, width_mult: float = 0.8) -> HTML:
        """
        Compiles the captured frames into an HTML5 interactive video.
        Returns an IPython HTML object that natively renders in Google Colab.
        
        Args:
            interval: Delay between frames in milliseconds.
            width_mult: Controls the visual width of the bars relative to tick_size.
        """
        if not self.frames_bids:
            return HTML("<p>No frames captured.</p>")

        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        def update(frame_idx):
            ax.clear()
            bids = self.frames_bids[frame_idx]
            asks = self.frames_asks[frame_idx]
            mid = self.frames_mids[frame_idx]
            
            # Filter out empty price levels (qty == 0)
            bids = bids[bids[:, 1] > 0]
            asks = asks[asks[:, 1] > 0]
            
            # Plot Bids (Green)
            if len(bids) > 0:
                ax.bar(bids[:, 0], bids[:, 1], color='#2ca02c', width=self.tick_size * width_mult, alpha=0.8, label='Bids')
            
            # Plot Asks (Red)
            if len(asks) > 0:
                ax.bar(asks[:, 0], asks[:, 1], color='#d62728', width=self.tick_size * width_mult, alpha=0.8, label='Asks')
            
            # Plot Mid-Price line
            if mid > 0:
                ax.axvline(x=mid, color='#1f77b4', linestyle='--', linewidth=2, label=f'Mid: {mid:.2f}')
            
            # Formatting
            ax.set_title(f'Titan LOB Dynamics | Step: {frame_idx}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Price', fontsize=12)
            ax.set_ylabel('Liquidity (Volume)', fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Dynamic X-axis scaling to follow the spread
            if len(bids) > 0 and len(asks) > 0:
                min_price = bids[-1, 0] - (self.tick_size * 5)
                max_price = asks[-1, 0] + (self.tick_size * 5)
                ax.set_xlim(min_price, max_price)

        # Create the animation
        anim = animation.FuncAnimation(
            fig, 
            update, 
            frames=len(self.frames_bids), 
            interval=interval,
            blit=False
        )
        
        # Close the static figure so it doesn't print a duplicate empty chart in Colab
        plt.close(fig)
        
        # Convert to Javascript HTML5 Video
        return HTML(anim.to_jshtml())