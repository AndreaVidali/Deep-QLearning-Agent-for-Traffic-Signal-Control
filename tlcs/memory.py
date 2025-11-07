import random
from collections import deque
from typing import Any, Deque


class Memory:
    """Replay memory with a bounded size and a minimum warmup threshold."""

    def __init__(self, size_max: int, size_min: int) -> None:
        if size_max <= 0:
            raise ValueError("size_max must be > 0")
        if not (0 <= size_min <= size_max):
            raise ValueError("size_min must be between 0 and size_max")

        self.samples: Deque = deque(maxlen=size_max)
        self.size_max = size_max
        self.size_min = size_min

    def add_sample(self, sample: Any) -> None:
        """Add a sample to memory."""
        self.samples.append(sample)

    def get_samples(self, n: int):
        """
        Get n samples randomly from memory.

        Returns an empty list if the buffer has fewer than size_min samples.
        """
        if len(self) < self.size_min or n <= 0:
            return []

        # check if we need to get all the samples, or a subset of them
        n = min(n, len(self.samples))

        return random.sample(self.samples, n)

    def __len__(self) -> int:
        return len(self.samples)
