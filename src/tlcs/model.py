from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn, optim

from tlcs.constants import MODEL_FILE
from tlcs.logger import get_logger

logger = get_logger(__name__)


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int, width: int) -> None:
        super().__init__()
        layers = nn.ModuleList()
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())
        for _ in range(num_layers):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Model:
    def __init__(
        self,
        num_layers: int,
        width: int,
        batch_size: int,
        learning_rate: float,
        input_dim: int,
        output_dim: int,
        model_path: Path | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        if model_path and (model_path / MODEL_FILE).exists():
            logger.info("Loading new model for the Agent")
            model_file = model_path / MODEL_FILE
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

    def predict_one(self, state: NDArray) -> NDArray:
        self.model.eval()
        with torch.no_grad():
            state = np.reshape(state, [1, self.input_dim]).astype(np.float32)
            return self.model(torch.from_numpy(state)).cpu().numpy()

    def predict_batch(self, states: NDArray) -> NDArray:
        self.model.eval()
        with torch.no_grad():
            states = np.asarray(states, dtype=np.float32)
            return self.model(torch.from_numpy(states)).cpu().numpy()

    def train_batch(self, states: NDArray, q_sa: NDArray) -> None:
        self.model.train()
        states_tensor = torch.from_numpy(np.asarray(states, np.float32))
        q_sa_tensor = torch.from_numpy(np.asarray(q_sa, np.float32))

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(states_tensor), q_sa_tensor)
        loss.backward()
        self.optimizer.step()

    def save_model(self, out_path: Path) -> None:
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, out_path / MODEL_FILE)

    def load_model(self, model_file: Path) -> nn.Module:
        return torch.load(model_file, weights_only=False, map_location="cpu")
