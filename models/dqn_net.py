import torch
import torch.nn as nn

from models.research_layers import DynamicCorrelationGraphEncoder, TemporalTransformerEncoder, infer_multi_asset_layout


class DQN(nn.Module):
    """Dueling DQN with selectable sequence encoders for ablation studies."""

    def __init__(self, input_dim, action_dim, encoder_type="gru", n_assets=3, asset_feature_dim=None):
        super(DQN, self).__init__()

        self.sequence_shape = input_dim if isinstance(input_dim, tuple) else None
        feature_dim = input_dim[-1] if isinstance(input_dim, tuple) else input_dim
        self.encoder_type = encoder_type

        encoder_output_dim = 64
        if encoder_type == "gru":
            self.input_norm = nn.LayerNorm(feature_dim)
            self.encoder = nn.GRU(
                input_size=feature_dim,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
            )
        elif encoder_type == "transformer":
            self.encoder = TemporalTransformerEncoder(feature_dim, hidden_dim=96, n_heads=4, n_layers=2)
            encoder_output_dim = 96
        elif encoder_type == "graph_transformer":
            layout = infer_multi_asset_layout(input_dim, n_assets=n_assets, asset_feature_dim=asset_feature_dim)
            if layout is None:
                raise ValueError("graph_transformer requires multi-asset sequence input with account features.")
            inferred_assets, asset_feature_dim, account_dim = layout
            self.encoder = DynamicCorrelationGraphEncoder(
                input_dim=feature_dim,
                n_assets=inferred_assets,
                asset_feature_dim=asset_feature_dim,
                account_dim=account_dim,
                hidden_dim=96,
                n_heads=4,
                n_layers=2,
            )
            encoder_output_dim = 96
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.feature_layer = nn.Sequential(
            nn.LayerNorm(encoder_output_dim),
            nn.Linear(encoder_output_dim, 64),
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

        if self.encoder_type == "gru":
            x = self.input_norm(x)
            _, hidden = self.encoder(x)
            encoded = hidden[-1]
        else:
            encoded = self.encoder(x)
        features = self.feature_layer(encoded)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))
