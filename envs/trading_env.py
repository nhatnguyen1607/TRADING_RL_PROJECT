import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    """
    Môi trường giao dịch tài chính chuẩn OpenAI Gym.
    Tích hợp Sharpe-like Reward (Phạt rủi ro biến động).
    """
    def __init__(self, df, window_size=20, initial_balance=10000.0, transaction_cost=0.001, is_discrete=True):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.is_discrete = is_discrete
        
        # Các cột features: Open, High, Low, Close, Volume, RSI, MACD, MA, VIX...
        self.feature_cols = df.attrs.get('feature_cols', [])
        self.state_shape = (self.window_size, len(self.feature_cols) + 2) # +2 cho balance và position
        
        # Action space
        if self.is_discrete:
            self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            
        # Observation space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.state_shape, dtype=np.float32)
        
        # Khởi tạo các tham số cho Risk-Adjusted Reward
        self.returns_history = []
        self.risk_penalty_coef = 0.5 # Hệ số phạt nếu tài khoản dao động mạnh
        
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.done = False
        self.returns_history = [] # Reset lại lịch sử return mỗi episode
        self.prev_action = 0 if self.is_discrete else 0.0
        return self._get_obs()

    def _get_obs(self):
        obs_df = self.df[self.feature_cols].iloc[self.current_step - self.window_size : self.current_step].values
        
        balance_scaled = self.balance / self.initial_balance
        shares_scaled = self.shares_held / (self.initial_balance / self.df['Close'].iloc[self.current_step])
        
        account_info = np.array([[balance_scaled, shares_scaled]] * self.window_size)
        obs = np.hstack((obs_df, account_info))
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        current_price = self.df['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth

        # 1. Xử lý Action
        if self.is_discrete:
            if action == 1: # Buy max
                shares_bought = self.balance / current_price * (1 - self.transaction_cost)
                self.shares_held += shares_bought
                self.balance = 0
            elif action == 2: # Sell all
                self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0
        else:
            action_val = action[0]
            if action_val > 0: # Buy
                invest_amount = self.balance * action_val
                shares_bought = invest_amount / current_price * (1 - self.transaction_cost)
                self.shares_held += shares_bought
                self.balance -= invest_amount
            elif action_val < 0: # Sell
                sell_ratio = abs(action_val)
                shares_sold = self.shares_held * sell_ratio
                self.balance += shares_sold * current_price * (1 - self.transaction_cost)
                self.shares_held -= shares_sold

        # 2. Tính Portfolio Value
        self.net_worth = self.balance + self.shares_held * current_price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # 3. Tính Sharpe-like Reward (Risk-Adjusted)
        return_pct = (self.net_worth - prev_net_worth) / prev_net_worth if prev_net_worth > 0 else 0
        self.returns_history.append(return_pct)

        if len(self.returns_history) > 20:
            self.returns_history.pop(0)
            
        variance = np.var(self.returns_history) if len(self.returns_history) >= 2 else 0

        # TÍNH TOÁN TURNOVER PENALTY (Phạt Lướt sóng)
        turnover_penalty = 0
        if self.is_discrete:
            # Nếu hnay khác hqua (VD: hqua BUY (1), hnay SELL (2)) -> Phạt
            if action != self.prev_action:
                turnover_penalty = 0.001 # Tinh chỉnh hệ số này
        else:
            # Phạt dựa trên độ lệch tỷ trọng (VD: Hqua 100% Long, Hnay -50% Short -> Lệch 1.5)
            action_val = action[0]
            turnover_penalty = abs(action_val - self.prev_action) * 0.001
            
        # Cập nhật prev_action cho bước tiếp theo
        self.prev_action = action if self.is_discrete else action[0]

        # REWARD CUỐI CÙNG: Scale x100 để gradient đủ lớn + Trừ đi phí lướt sóng
        reward = (return_pct - (self.risk_penalty_coef * variance) - turnover_penalty) * 10.0

        # 4. Kiểm tra episode kết thúc
        if self.current_step >= len(self.df) - 1 or self.net_worth <= self.initial_balance * 0.1:
            self.done = True

        return self._get_obs(), reward, self.done, {'net_worth': self.net_worth}

    def render(self, mode='human'):
        print(f'Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}')