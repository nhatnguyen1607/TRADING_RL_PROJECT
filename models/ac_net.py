import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    """
    Actor-Critic architecture dùng LSTM.
    Dựa trên lý thuyết Policy Gradient.
    """
    def __init__(self, input_dim, action_dim=1, hidden_dim=64, num_layers=1):
        super(ActorCritic, self).__init__()
        
        # Shared Feature Extractor (LSTM)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Actor Head (Outputs mean and std for continuous action)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh() # Đưa mean về khoảng [-1, 1]
        )
        # Log_std là tham số học được (độc lập với state hoặc phụ thuộc state)
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic Head (Outputs state-value V(s))
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        
        # Critic
        value = self.critic(last_out)
        
        # Actor
        action_mean = self.actor_mean(last_out)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        return action_mean, action_std, value
        
    def get_action(self, x):
        mean, std, _ = self.forward(x)
        dist = Normal(mean, std)
        action = dist.sample()
        # Tính log probability của action để update policy
        log_prob = dist.log_prob(action).sum(dim=-1)
        # Kẹp action vào khoảng [-1, 1]
        action = torch.clamp(action, -1.0, 1.0)
        return action, log_prob