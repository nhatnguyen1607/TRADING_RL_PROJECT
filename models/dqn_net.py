import torch
import torch.nn as nn


class DQN(nn.Module):
    """Dueling DQN with a GRU encoder for windowed market states."""

    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()

        self.sequence_shape = input_dim if isinstance(input_dim, tuple) else None
        feature_dim = input_dim[-1] if isinstance(input_dim, tuple) else input_dim

        self.input_norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.feature_layer = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x):
        if x.dim() == 2:
            if self.sequence_shape is None:
                x = x.unsqueeze(1)
            else:
                x = x.view(x.size(0), self.sequence_shape[0], self.sequence_shape[1])

        x = self.input_norm(x)
        _, hidden = self.encoder(x)
        features = self.feature_layer(hidden[-1])
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
