import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class ActorCritic(nn.Module):
    """Actor-Critic with a GRU encoder and bounded allocation policy."""

    def __init__(self, input_dim, action_dim=1):
        super(ActorCritic, self).__init__()

        self.sequence_shape = input_dim if isinstance(input_dim, tuple) else None
        feature_dim = input_dim[-1] if isinstance(input_dim, tuple) else input_dim
        self.action_dim = action_dim
        self.is_discrete = action_dim > 1

        self.input_norm = nn.LayerNorm(feature_dim)
        self.encoder = nn.GRU(
            input_size=feature_dim,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.shared_layers = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )
        if not self.is_discrete:
            self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * -1.8)

        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            if self.sequence_shape is None:
                x = x.unsqueeze(1)
            else:
                x = x.view(x.size(0), self.sequence_shape[0], self.sequence_shape[1])

        x = self.input_norm(x)
        _, hidden = self.encoder(x)
        shared = self.shared_layers(hidden[-1])

        value = self.critic(shared)
        actor_output = self.actor_mean(shared)
        if self.is_discrete:
            return actor_output, None, value
        action_log_std = self.actor_log_std.expand_as(actor_output)
        action_std = torch.exp(action_log_std).clamp(min=0.03, max=0.30)
        return actor_output, action_std, value

    def get_action(self, x, deterministic=False):
        actor_output, std, _ = self.forward(x)
        if self.is_discrete:
            dist = Categorical(logits=actor_output)
            action = torch.argmax(actor_output, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob, dist
        dist = Normal(actor_output, std)
        raw_action = actor_output if deterministic else dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        action = torch.clamp(raw_action, -1.0, 1.0)
        return action, log_prob, dist

    def evaluate_actions(self, x, actions):
        actor_output, std, value = self.forward(x)
        if self.is_discrete:
            dist = Categorical(logits=actor_output)
            if actions.dim() > 1:
                actions = actions.squeeze(-1)
            log_probs = dist.log_prob(actions.long())
            entropy = dist.entropy()
            return log_probs, entropy, value.squeeze(-1)
        dist = Normal(actor_output, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy, value.squeeze(-1)
