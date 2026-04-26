import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from models.ac_net import ActorCritic

class ACAgent:
    def __init__(self, state_dim, lr=5e-4, gamma=0.99): 
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.huber_loss = nn.SmoothL1Loss() # 🔧 FIXED

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, dist = self.model.get_action(state_tensor) 
        return action.detach().numpy()[0], log_prob, dist

    def train_step(self, state, log_prob, dist, reward, next_state, done):
        state_flat = state.reshape(-1) if isinstance(state, np.ndarray) else state
        next_state_flat = next_state.reshape(-1) if isinstance(next_state, np.ndarray) else next_state
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state_flat).unsqueeze(0)
        
        _, _, value = self.model(state_tensor)
        
        with torch.no_grad():
            _, _, next_value = self.model(next_state_tensor)
            
        td_target = reward + self.gamma * next_value * (1 - done)
        delta = td_target - value
        
        # 🔧 FIXED: Dùng Huber loss cho Critic để tránh gai nhiễu Reward
        critic_loss = self.huber_loss(value, td_target)
        
        actor_loss = -(log_prob * delta.detach()).mean()
        entropy = dist.entropy().mean()
        
        
        # 🔧 FIX: Hạ Entropy Bonus xuống mức vừa đủ để không bị "lì", nhưng không bị "tăng động"
        total_loss = actor_loss + critic_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return total_loss.item()