from pathlib import Path

import numpy as np
import torch
from torch import nn, optim

from tlcs.constants import MODEL_FILE


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, width):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.ReLU())
        for _ in range(num_layers):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Model:
    def __init__(
        self,
        num_layers,
        width,
        batch_size,
        learning_rate,
        input_dim,
        output_dim,
        model_path: Path | None = None,
    ):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim

        if model_path and (model_path / MODEL_FILE).exists():
            print("Loading new model for the Agent")
            model_file = model_path / MODEL_FILE
            self.model = self.load_model(model_file)
        else:
            print("Creating new model for the Agent")
            self.model = MLP(
                input_dim=input_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                width=width,
            )

        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def predict_one(self, state):
        self.model.eval()
        with torch.no_grad():
            state = np.reshape(state, [1, self.input_dim]).astype(np.float32)
            return self.model(torch.from_numpy(state)).cpu().numpy()

    def predict_batch(self, states):
        self.model.eval()
        with torch.no_grad():
            states = np.asarray(states, dtype=np.float32)
            return self.model(torch.from_numpy(states)).cpu().numpy()

    def train_batch(self, states, q_sa):
        self.model.train()
        states = torch.from_numpy(np.asarray(states, np.float32))
        q_sa = torch.from_numpy(np.asarray(q_sa, np.float32))

        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(states), q_sa)
        loss.backward()
        self.optimizer.step()

    def save_model(self, out_path: Path):
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, out_path / MODEL_FILE)

    def load_model(self, model_file: Path):
        return torch.load(model_file, weights_only=False, map_location="cpu")
