import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim=1):
        super(ActorCritic, self).__init__()
        
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.actor_mean = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh() # Đảm bảo output Mean nằm trong [-1, 1]
        )
        # 🔧 FIX: Khởi tạo log_std âm để mô hình không đánh ALL IN / ALL OUT mù quáng
        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        shared = self.shared_layers(x)
        
        value = self.critic(shared)
        action_mean = self.actor_mean(shared)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        return action_mean, action_std, value
        
    def get_action(self, x):
        mean, std, _ = self.forward(x)
        dist = Normal(mean, std)
        action = dist.sample()
        # 🔧 FIXED: Lấy log_prob của raw action TRƯỚC KHI clamp để không làm vỡ Gradients
        log_prob = dist.log_prob(action).sum(dim=-1)
        action = torch.clamp(action, -1.0, 1.0) 
        
        return action, log_prob, dist