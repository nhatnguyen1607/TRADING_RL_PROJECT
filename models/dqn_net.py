import torch
import torch.nn as nn

class DQN(nn.Module):
    """ Dueling DQN sử dụng MLP (Cách FinRL làm để ổn định Loss) """
    def __init__(self, input_dim, action_dim):
        super(DQN, self).__init__()
        
        # Mạng chung trích xuất đặc trưng
        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Luồng Value (V)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Luồng Advantage (A)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        # ÉP PHẲNG (Flatten) State: từ (batch, window, features) -> (batch, window * features)
        x = x.view(x.size(0), -1) 
        
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Dueling formula
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values