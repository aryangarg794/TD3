import numpy as np

from collections import deque
from typing import Self, List

class RollingAverage:
    def __init__(
        self: Self, 
        window_size: int, 
    ) -> None:
        
        self.window = deque(maxlen=window_size)
        self.averages = []
        self.all_rewards = []

    def update(
        self: Self, 
        values: List
    ) -> None:
        self.window.append(float(np.mean(values)))
        self.averages.append(self.get_average)
        self.all_rewards.append(values)

    @property
    def get_values(self: Self) -> np.ndarray:
        return np.array(self.all_rewards) 
    
    @property
    def get_average(self: Self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0