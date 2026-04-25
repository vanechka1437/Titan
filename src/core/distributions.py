import abc
import torch
from typing import Tuple

# =============================================================================
# VECTORIZED C++ DISTRIBUTIONS
# 
# This module provides a clean API for Agent-Based Modeling (ABM) hyperparameters.
# Under the hood, it strictly avoids Python loops. It uses PyTorch's native 
# in-place tensor operations (e.g., `.normal_()`, `.exponential_()`) which are 
# executed entirely in highly optimized C++ / CUDA kernels.
# 
# This guarantees O(1) latency in Python when sampling parameters for 4096+ 
# parallel simulation environments.
# =============================================================================

class Distribution(abc.ABC):
    """
    Abstract base class for all hardware-accelerated distributions.
    """
    @abc.abstractmethod
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """
        Samples a tensor of the given shape.
        
        Args:
            shape: Tuple representing the dimensions (e.g., (num_envs, num_agents)).
            device: Target memory device (CPU/CUDA) where the tensor should reside.
            
        Returns:
            torch.Tensor: Vectorized sample.
        """
        pass


class Constant(Distribution):
    """
    Deterministic constant value. Useful for control groups or turning off noise.
    """
    def __init__(self, value: float):
        self.value = float(value)
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.full(shape, self.value, device=device, dtype=torch.float32)


class Uniform(Distribution):
    """
    Continuous Uniform distribution U(low, high).
    Commonly used for spatial positioning or random baseline noise.
    """
    def __init__(self, low: float, high: float):
        self.low = float(low)
        self.high = float(high)
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        # In-place C++ operation: avoids intermediate tensor allocations
        return torch.empty(shape, device=device, dtype=torch.float32).uniform_(self.low, self.high)


class Normal(Distribution):
    """
    Gaussian Normal distribution N(mu, sigma).
    Used for order sizing, fundamental value drift, and latency jitter.
    """
    def __init__(self, mu: float, sigma: float):
        self.mu = float(mu)
        self.sigma = float(sigma)
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=torch.float32).normal_(mean=self.mu, std=self.sigma)


class LogNormal(Distribution):
    """
    Log-Normal distribution.
    Crucial for modeling network latencies (where latency cannot be < 0 and 
    has a heavy right tail) and financial asset returns.
    """
    def __init__(self, mu: float, sigma: float):
        self.mu = float(mu)
        self.sigma = float(sigma)
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=torch.float32).log_normal_(mean=self.mu, std=self.sigma)


class Exponential(Distribution):
    """
    Exponential distribution Exp(lambda).
    Used in Market Microstructure to model the distance of limit orders 
    from the Mid-Price (closer to mid = exponentially higher probability).
    """
    def __init__(self, rate: float):
        self.rate = float(rate) # Note: PyTorch expects lambda (rate), not scale
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=torch.float32).exponential_(lambd=self.rate)


class Poisson(Distribution):
    """
    Poisson distribution.
    Models the discrete number of events (e.g., order arrivals) occurring 
    in a fixed time interval with a known constant mean rate (lambda).
    """
    def __init__(self, rate: float):
        self.rate = float(rate)
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        # Create a tensor of rates, then apply Poisson sampling via C++ backend
        rates = torch.full(shape, self.rate, device=device, dtype=torch.float32)
        return torch.poisson(rates)


class Bernoulli(Distribution):
    """
    Bernoulli distribution (Coin Flip).
    Used to randomly assign Bid (0) or Ask (1) sides for Zero-Intelligence traders.
    """
    def __init__(self, p: float):
        assert 0.0 <= p <= 1.0, "Probability p must be between 0 and 1"
        self.p = float(p)
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=torch.float32).bernoulli_(self.p)


class Gamma(Distribution):
    """
    Gamma distribution.
    Used in Bayesian inference and for modeling time until N events occur 
    (Wait times in queuing theory for order book execution).
    """
    def __init__(self, concentration: float, rate: float):
        self.concentration = float(concentration) # alpha
        self.rate = float(rate)                   # beta
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        # Falls back to torch.distributions for complex sampling logic, 
        # still heavily optimized in C++
        dist = torch.distributions.Gamma(self.concentration, self.rate)
        return dist.sample(shape).to(device)


class Beta(Distribution):
    """
    Beta distribution.
    Used for modeling probabilities and proportions (values strictly between 0 and 1),
    such as the ratio of aggressive vs passive orders.
    """
    def __init__(self, alpha: float, beta: float):
        self.alpha = float(alpha)
        self.beta = float(beta)
        
    def sample(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        dist = torch.distributions.Beta(self.alpha, self.beta)
        return dist.sample(shape).to(device)