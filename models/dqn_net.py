import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Dueling Deep Q-Network sử dụng LSTM.
    Tách biệt luồng đánh giá Value (thị trường chung) và Advantage (lợi thế hành động).
    """
    def __init__(self, input_dim, action_dim, hidden_dim=64, num_layers=1):
        super(DQN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 1. Feature Extractor (Trích xuất đặc trưng chuỗi thời gian)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 2. Luồng Value (V) - Trạng thái thị trường này tốt hay xấu?
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Output 1 giá trị duy nhất
        )
        
        # 3. Luồng Advantage (A) - Hành động nào có lợi thế nhất?
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim) # Output cho từng hành động riêng biệt
        )

    def forward(self, x):
        # x shape: (batch_size, window_size, input_dim)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :] # Chỉ lấy output của time-step cuối cùng
        
        # Tính Value và Advantage
        value = self.value_stream(last_out)
        advantage = self.advantage_stream(last_out)
        
        # CÔNG THỨC DUELING: Q(s,a) = V(s) + ( A(s,a) - mean(A(s,a)) )
        # Việc trừ đi giá trị trung bình giúp mạng ổn định hơn (Identifiability)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values