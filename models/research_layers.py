import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TemporalTransformerEncoder(nn.Module):
    """Compact Transformer encoder with exposed temporal attention weights."""

    def __init__(self, input_dim, hidden_dim=96, n_heads=4, n_layers=2, dropout=0.10):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attn_probe = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.last_attention = None

    def forward(self, x):
        z = self.position(self.input_proj(self.input_norm(x)))
        z = self.encoder(z)
        pooled_query = z[:, -1:, :]
        context, attn = self.attn_probe(pooled_query, z, z, need_weights=True, average_attn_weights=False)
        self.last_attention = attn.detach()
        return self.output_norm(context.squeeze(1))


class DynamicCorrelationGraphEncoder(nn.Module):
    """
    Learns dynamic cross-asset relations before temporal encoding.

    The state is expected to contain asset-major feature blocks followed by account features.
    Correlation edges are computed from recent asset returns and used as a soft graph filter.
    """

    def __init__(
        self,
        input_dim,
        n_assets,
        asset_feature_dim,
        account_dim,
        hidden_dim=96,
        n_heads=4,
        n_layers=2,
        dropout=0.10,
    ):
        super().__init__()
        self.n_assets = int(n_assets)
        self.asset_feature_dim = int(asset_feature_dim)
        self.account_dim = int(account_dim)
        self.hidden_dim = hidden_dim
        self.asset_proj = nn.Linear(asset_feature_dim, hidden_dim)
        self.account_proj = nn.Linear(account_dim, hidden_dim)
        self.graph_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.temporal_encoder = TemporalTransformerEncoder(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.last_graph = None

        expected = self.n_assets * self.asset_feature_dim + self.account_dim
        if expected != input_dim:
            raise ValueError(
                f"DynamicCorrelationGraphEncoder expected input_dim={expected}, got {input_dim}. "
                "Check asset_feature_dim/account_dim inferred from env attrs."
            )

    def _dynamic_adjacency(self, asset_features):
        returns = asset_features[..., 0]
        returns = returns - returns.mean(dim=1, keepdim=True)
        cov = torch.einsum("bta,btm->bam", returns, returns) / max(returns.size(1) - 1, 1)
        std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8))
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-8)
        adj = torch.softmax(corr.clamp(-1.0, 1.0), dim=-1)
        self.last_graph = adj.detach()
        return adj

    def forward(self, x):
        bsz, seq_len, _ = x.shape
        asset_flat = x[:, :, : self.n_assets * self.asset_feature_dim]
        account = x[:, :, self.n_assets * self.asset_feature_dim :]
        asset_features = asset_flat.view(bsz, seq_len, self.n_assets, self.asset_feature_dim)

        asset_emb = self.asset_proj(asset_features)
        adj = self._dynamic_adjacency(asset_features)
        graph_emb = torch.einsum("bam,btmh->btah", adj, asset_emb)
        gate = self.graph_gate(torch.cat([asset_emb, graph_emb], dim=-1))
        mixed_assets = gate * graph_emb + (1.0 - gate) * asset_emb
        portfolio_context = mixed_assets.mean(dim=2)

        account_context = self.account_proj(account)
        temporal_input = torch.cat([portfolio_context, account_context], dim=-1)
        return self.temporal_encoder(temporal_input)


def infer_multi_asset_layout(sequence_shape, n_assets=None, asset_feature_dim=None):
    if not isinstance(sequence_shape, tuple) or len(sequence_shape) != 2:
        return None
    feature_dim = sequence_shape[-1]
    if n_assets is None:
        n_assets = 3
    if asset_feature_dim is not None:
        asset_width = int(n_assets) * int(asset_feature_dim)
        account_dim = feature_dim - asset_width
        if account_dim > 0:
            return int(n_assets), int(asset_feature_dim), int(account_dim)
        return None

    account_dim = int(n_assets) + 1
    asset_width = feature_dim - account_dim
    if asset_width <= 0 or asset_width % int(n_assets) != 0:
        return None
    return int(n_assets), asset_width // int(n_assets), account_dim
