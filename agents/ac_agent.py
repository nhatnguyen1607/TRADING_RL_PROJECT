import torch
import torch.optim as optim
import numpy as np
from models.ac_net import ActorCritic

class ACAgent:
    def __init__(self, state_dim, lr=1e-4, gamma=0.99):
        self.gamma = gamma
        self.model = ActorCritic(state_dim, action_dim=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob = self.model.get_action(state_tensor)
        return action.detach().numpy()[0], log_prob

    def train_step(self, state, log_prob, reward, next_state, done):
        """
        1-step TD update cho Actor-Critic
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Forward pass để lấy Value hiện tại
        _, _, value = self.model(state_tensor)
        
        # Lấy Value của state tiếp theo
        with torch.no_grad():
            _, _, next_value = self.model(next_state_tensor)
            
        # Tính TD Target & TD Error (Advantage)
        td_target = reward + self.gamma * next_value * (1 - done)
        delta = td_target - value
        
        # Critic Loss: Giảm thiểu bình phương TD Error (MSE)
        critic_loss = delta.pow(2).mean()
        
        # Actor Loss: Policy Gradient theorem (-log_prob * Advantage)
        # Gradient ascent trên J(theta), nên loss phải có dấu âm
        actor_loss = -(log_prob * delta.detach()).mean()
        
        loss = actor_loss + critic_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()