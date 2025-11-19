import random
from collections import deque
from dataclasses import dataclass

from numpy.typing import NDArray


@dataclass
class Sample:
    """Single transition sample stored in replay memory."""

    state: NDArray
    action: int
    reward: float
    next_state: NDArray


class Memory:
    """Replay memory with a bounded size and a minimum warmup threshold."""

    def __init__(self, size_max: int, size_min: int) -> None:
        """Initialize the replay memory.

        Args:
            size_max: Maximum number of samples to store.
            size_min: Minimum number of samples required before sampling.
        """
        self.samples: deque[Sample] = deque(maxlen=size_max)
        self.size_max = size_max
        self.size_min = size_min

    def add_sample(self, sample: Sample) -> None:
        """Add a sample to memory.

        Args:
            sample: The sample to store.
        """
        self.samples.append(sample)

    def get_samples(self, n: int) -> list[Sample]:
        """Return up to n random samples from memory.

        If the buffer contains fewer than `size_min` samples or if `n` is not positive,
        an empty list is returned.

        Args:
            n: Number of samples to draw.

        Returns:
            A list of randomly drawn samples.
        """
        if n <= 0 or len(self) < self.size_min:
            return []

        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def __len__(self) -> int:
        """Return the current number of stored samples.

        Returns:
            The number of samples currently stored in memory.
        """
        return len(self.samples)
