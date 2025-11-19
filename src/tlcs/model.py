from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim

from tlcs.constants import MODEL_FILE
from tlcs.logger import get_logger

logger = get_logger(__name__)


class MLP(nn.Module):
    """Simple multi-layer perceptron with configurable depth and width."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        width: int,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            num_layers: Number of hidden layers with width `width`.
            width: Number of units in each hidden layer.
        """
        super().__init__()

        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())

        for _ in range(num_layers):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Run a forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tensor: Output tensor of shape (batch_size, output_dim).
        """
        return self.net(x)


class Model:
    """Wrapper around the MLP model with training and inference utilities."""

    def __init__(  # noqa: PLR0913
        self,
        num_layers: int,
        width: int,
        learning_rate: float,
        input_dim: int,
        output_dim: int,
        model_path: Path | None = None,
    ) -> None:
        """Initialize the model, loading from disk if available.

        Args:
            num_layers: Number of hidden layers for the MLP.
            width: Number of units in each hidden layer.
            learning_rate: Learning rate for the optimizer.
            input_dim: Input feature dimension.
            output_dim: Output feature dimension.
            model_path: Optional directory containing a saved model file.
        """
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        if model_path and (model_path / MODEL_FILE).exists():
            model_file = model_path / MODEL_FILE
            logger.info(f"Loading trained model for the Agent from {model_file}")
            self.model = self.load_model(model_file)
        else:
            logger.info("Creating new model for the Agent")
            self.model = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                width=width,
            )

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _predict(self, states: NDArray) -> NDArray:
        """Internal helper to run prediction on a batch of states.

        Args:
            states: Input states as array of shape (batch_size, input_dim).

        Returns:
            NDArray: Model outputs as array of shape (batch_size, output_dim).
        """
        self.model.eval()
        with torch.no_grad():
            states = np.asarray(states, dtype=np.float32)
            outputs = self.model(torch.from_numpy(states))
            return outputs.cpu().numpy()

    def predict_one(self, state: NDArray) -> NDArray:
        """Predict for a single state.

        Args:
            state: Input state as 1D array of shape (input_dim,).

        Returns:
            NDArray: Predicted values as array of shape (1, output_dim).
        """
        state_2d = np.asarray(state, dtype=np.float32).reshape(1, self.input_dim)
        return self._predict(state_2d)

    def predict_batch(self, states: NDArray) -> NDArray:
        """Predict for a batch of states.

        Args:
            states: Input states as array of shape (batch_size, input_dim).

        Returns:
            NDArray: Predicted values as array of shape (batch_size, output_dim).
        """
        return self._predict(states)

    def train_batch(self, states: NDArray, q_sa: NDArray) -> None:
        """Train the model on a batch of states and target values.

        Args:
            states: Input states as array of shape (batch_size, input_dim).
            q_sa: Target values, e.g., Q(s, a), of shape (batch_size, output_dim).
        """
        self.model.train()

        states_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32))
        q_sa_tensor = torch.from_numpy(np.asarray(q_sa, dtype=np.float32))

        self.optimizer.zero_grad()
        predictions = self.model(states_tensor)
        loss = self.loss_fn(predictions, q_sa_tensor)
        loss.backward()
        self.optimizer.step()

    def save_model(self, out_path: Path) -> None:
        """Save the model to ``out_path / MODEL_FILE``."""
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, out_path / MODEL_FILE)

    def load_model(self, model_file: Path) -> nn.Module:
        """Load a model from disk.

        Args:
            model_file: Path to the saved model file.

        Returns:
            nn.Module: Loaded model instance.
        """
        return torch.load(model_file, weights_only=False, map_location="cpu")
