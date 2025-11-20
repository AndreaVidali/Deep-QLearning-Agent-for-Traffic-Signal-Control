import random
from pathlib import Path

import numpy as np

from tlcs.constants import NUM_ACTIONS, STATE_SIZE
from tlcs.memory import Memory
from tlcs.model import Model
from tlcs.settings import TrainingSettings


class Agent:
    """Reinforcement-learning agent using an epsilon-greedy policy."""

    def __init__(
        self,
        settings: TrainingSettings,
        epsilon: float = 1.0,
        model_path: Path | None = None,
    ) -> None:
        """Initialize the agent and its underlying model.

        Args:
            settings: Training settings containing model hyperparameters.
            epsilon: Exploration probability in [0, 1].
            model_path: Optional path to load a pre-trained model.
        """
        self.epsilon = epsilon
        self.model = Model(
            num_layers=settings.num_layers,
            width=settings.width_layers,
            learning_rate=settings.learning_rate,
            input_dim=STATE_SIZE,
            output_dim=NUM_ACTIONS,
            model_path=model_path,
        )

    def set_epsilon(self, epsilon: float) -> None:
        """Set the epsilon value for epsilon-greedy exploration.

        Args:
            epsilon: Exploration probability in [0, 1].
        """
        if not 0 <= epsilon <= 1:
            msg = "Epsilon must be between 0 and 1."
            raise ValueError(msg)
        self.epsilon = epsilon

    def choose_action(self, state: np.ndarray) -> int:
        """Choose an action according to an epsilon-greedy policy.

        Args:
            state: Current state representation.

        Returns:
            Index of the selected action.
        """
        if random.random() < self.epsilon:
            # Explore: random action
            return random.randrange(self.model.output_dim)

        # Exploit: greedy action
        q_values = self.model.predict_one(state)
        return int(np.argmax(q_values))

    def replay(self, memory: Memory, gamma: float, batch_size: int) -> None:
        """Sample from replay memory and perform a Q-learning update.

        Args:
            memory: Experience replay buffer.
            gamma: Discount factor for future rewards.
            batch_size: Number of samples to draw from the memory.
        """
        batch = memory.get_samples(batch_size)

        if not batch:
            return

        # Extract states and next states
        states = np.array([sample.state for sample in batch])
        next_states = np.array([sample.next_state for sample in batch])

        # Predict Q-values for current and next states
        q_values = self.model.predict_batch(states)
        next_q_values = self.model.predict_batch(next_states)

        # Prepare training data
        x = states
        y = q_values.copy()

        for i, sample in enumerate(batch):
            # Q-learning target: r + gamma * max_a' Q(s', a')
            target = sample.reward + gamma * np.max(next_q_values[i])
            y[i, sample.action] = target

        # Train model on the updated Q-values
        self.model.train_batch(x, y)

    def save_model(self, out_path: Path) -> None:
        """Save the underlying model to disk.

        Args:
            out_path: Destination path for the saved model.
        """
        self.model.save_model(out_path)
