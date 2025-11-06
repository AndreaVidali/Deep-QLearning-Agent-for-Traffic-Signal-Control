import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


class TrainModel:
    def __init__(
        self, num_layers, width, batch_size, learning_rate, input_dim, output_dim
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self._build_model(num_layers, width)
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def _build_model(self, num_layers, width):
        """
        Build a fully connected deep neural network
        """
        return MLP(self.input_dim, self.output_dim, num_layers, width)

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        self.model.eval()
        with torch.no_grad():
            state = np.reshape(state, [1, self.input_dim]).astype(np.float32)
            inp = torch.from_numpy(state)
            out = self.model(inp).cpu().numpy()
        return out

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        self.model.eval()
        with torch.no_grad():
            states = np.asarray(states, dtype=np.float32)
            inp = torch.from_numpy(states)
            out = self.model(inp).cpu().numpy()
        return out

    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values (one epoch over the given batch)
        """
        self.model.train()
        states = np.asarray(states, dtype=np.float32)
        q_sa = np.asarray(q_sa, dtype=np.float32)
        inp = torch.from_numpy(states)
        tgt = torch.from_numpy(q_sa)

        self.optimizer.zero_grad()
        pred = self.model(inp)
        loss = self.loss_fn(pred, tgt)
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        """
        Save the current model as a .pt file and try to export a model graph png (if torchviz is available)
        """
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "trained_model.pt")
        torch.save(self.model, model_path)


class TestModel:
    def __init__(self, input_dim, model_path):
        self.input_dim = input_dim
        self.model = self._load_my_model(model_path)

    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, "trained_model.pt")

        if os.path.isfile(model_file_path):
            loaded_model = torch.load(model_file_path, map_location="cpu")
            loaded_model.eval()
            return loaded_model
        else:
            sys.exit("Model number not found")

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        self.model.eval()
        with torch.no_grad():
            state = np.reshape(state, [1, self.input_dim]).astype(np.float32)
            inp = torch.from_numpy(state)
            out = self.model(inp).cpu().numpy()
        return out
