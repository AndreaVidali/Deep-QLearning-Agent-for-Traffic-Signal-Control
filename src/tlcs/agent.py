import random
from pathlib import Path

import numpy as np

from tlcs.memory import Memory
from tlcs.model import Model
from tlcs.settings import TrainingSettings


class Agent:
    def __init__(
        self,
        settings: TrainingSettings,
        epsilon: float = 1,
        model_path: Path | None = None,
    ):
        self.epsilon = epsilon
        self.model = Model(
            num_layers=settings.num_layers,
            width=settings.width_layers,
            batch_size=settings.batch_size,
            learning_rate=settings.learning_rate,
            input_dim=settings.state_size,
            output_dim=settings.num_actions,
            model_path=model_path,
        )

    def set_epsilon(self, epsilon):
        if epsilon < 0 or epsilon > 1:
            msg = "Epsilon out of bounds"
            raise ValueError(msg)
        self.epsilon = epsilon

    def choose_action(self, state: np.ndarray) -> int:
        """Choose exploration vs exploitation according to epsilon-greedy policy."""
        if random.random() < self.epsilon:
            # explore
            action_n = random.randint(0, self.model.output_dim - 1)
        else:
            # exploit
            action_n = int(np.argmax(self.model.predict_one(state)))

        return action_n

    def replay(self, memory: Memory, gamma):
        """Get samples from the memory and for each of them update the learning equation, then train"""
        batch = memory.get_samples(self.model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            # extract states from the batch
            states = np.array([val[0] for val in batch])

            # extract next states from the batch
            next_states = np.array([val[3] for val in batch])

            # predict Q(state), for every sample
            q_s_a = self.model.predict_batch(states)

            # predict Q(next_state), for every sample
            q_s_a_d = self.model.predict_batch(next_states)

            # setup training arrays
            x = np.zeros((len(batch), self.model.input_dim))
            y = np.zeros((len(batch), self.model.output_dim))

            for i, b in enumerate(batch):
                # extract data from one sample
                state, action, reward, _ = (b[0], b[1], b[2], b[3])
                # get the Q(state) predicted before
                current_q = q_s_a[i]
                # update Q(state, action)
                current_q[action] = reward + gamma * np.amax(q_s_a_d[i])
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self.model.train_batch(x, y)  # train the NN

    def save_model(self, out_path: Path):
        self.model.save_model(out_path)
