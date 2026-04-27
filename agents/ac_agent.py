import torch
import torch.optim as optim
import numpy as np
from models.ac_net import ActorCritic

class ACAgent:
    def __init__(self, state_dim, lr=1e-3, gamma=0.99):  # 🔧 FIXED: lr=1e-3 (10x tăng)
        self.gamma = gamma
        # Khởi tạo mô hình Actor-Critic (Kiến trúc MLP Flatten)
        self.model = ActorCritic(state_dim, action_dim=1)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def act(self, state):
        """
        Trả về 3 giá trị: action, log_prob, và distribution (để tính entropy)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        # Gọi hàm get_action từ models/ac_net.py
        action, log_prob, dist = self.model.get_action(state_tensor) 
        
        # Chuyển action sang numpy để môi trường xử lý
        return action.detach().numpy()[0], log_prob, dist

    def train_step(self, state, log_prob, dist, reward, next_state, done):
        """
        Cập nhật trọng số mạng neural dựa trên TD Error
        """
        # 🔧 FIXED: Flatten state trước khi convert thành tensor
        state_flat = state.reshape(-1) if isinstance(state, np.ndarray) else state
        next_state_flat = next_state.reshape(-1) if isinstance(next_state, np.ndarray) else next_state
        state_tensor = torch.FloatTensor(state_flat).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state_flat).unsqueeze(0)
        
        # 1. Forward pass lấy giá trị Value hiện tại V(s)
        _, _, value = self.model(state_tensor)
        
        # 2. Lấy Value của trạng thái tiếp theo V(s') mà không tính gradient
        with torch.no_grad():
            _, _, next_value = self.model(next_state_tensor)
            
        # 3. Tính TD Target và Advantage (delta)
        # Target = r + gamma * V(s')
        td_target = reward + self.gamma * next_value * (1 - done)
        # Advantage = TD Error = Target - V(s)
        delta = td_target - value
        
        # 4. Tính toán các thành phần Loss
        
        # A. Critic Loss: Phải luôn DƯƠNG (MSE giữa V(s) và Target)
        critic_loss = delta.pow(2).mean()
        
        # B. Actor Loss: Policy Gradient (-log_prob * Advantage)
        # Chúng ta detach delta để không update ngược vào Critic thông qua luồng này
        actor_loss = -(log_prob * delta.detach()).mean()
        
        # C. Entropy Bonus: Khuyến khích khám phá, tránh hội tụ sớm vào 1 hành động
        # Trong các bài báo như SAC hay FinRL, entropy giúp mô hình không bị "lì"
        entropy = dist.entropy().mean()
        
        # 5. Tổng hợp Loss (CÂN BẰNG TRỌNG SỐ)
        # 🔧 FIXED: Tăng hệ số critic từ 0.5 → 1.0 để critic học tốt hơn
        total_loss = actor_loss + 1.0 * critic_loss - 0.001 * entropy
        
        # 6. Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient Clipping: Chống nổ gradient (Rất quan trọng cho tài chính)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return total_loss.item()