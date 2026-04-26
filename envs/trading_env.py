import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df, window_size=20, initial_balance=10000.0, transaction_cost=0.001, is_discrete=True):
        super(TradingEnv, self).__init__()
        
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.is_discrete = is_discrete
        
        self.feature_cols = df.attrs.get('feature_cols', [])
        self.state_shape = (self.window_size, len(self.feature_cols) + 2)
        
        if self.is_discrete:
            self.action_space = spaces.Discrete(3) # 0: Hold, 1: Buy, 2: Sell
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
            
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.state_shape, dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        obs_df = self.df[self.feature_cols].iloc[self.current_step - self.window_size : self.current_step].values
        
        # 🔧 FIXED: Tránh việc balance/shares scale lên quá lớn gây nhiễu state
        balance_scaled = np.clip(self.balance / self.initial_balance, 0, 5)
        current_price = self.df['Close'].iloc[self.current_step]
        max_possible_shares = self.initial_balance / current_price
        shares_scaled = np.clip(self.shares_held / max_possible_shares, 0, 5)
        
        account_info = np.array([[balance_scaled, shares_scaled]] * self.window_size)
        obs = np.hstack((obs_df, account_info))
        return obs.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        current_price = self.df['Close'].iloc[self.current_step]
        prev_net_worth = self.net_worth

        # 1. THỰC THI ACTION
        if self.is_discrete:
            if action == 1 and self.balance > 0: # Buy max
                shares_bought = self.balance / current_price * (1 - self.transaction_cost)
                self.shares_held += shares_bought
                self.balance = 0
            elif action == 2 and self.shares_held > 0: # Sell all
                self.balance += self.shares_held * current_price * (1 - self.transaction_cost)
                self.shares_held = 0
        else:
            action_val = action[0]
            
            # 🚀 CẢI TIẾN 1: DEADZONE CHỐNG NHIỄU (Bảo vệ Actor-Critic khỏi phí giao dịch)
            if abs(action_val) < 0.15: # Nếu độ tự tin dưới 15%, ép chuyển thành HOLD
                action_val = 0.0 
                
            if action_val > 0 and self.balance > 0: # Buy
                invest_amount = self.balance * action_val
                shares_bought = invest_amount / current_price * (1 - self.transaction_cost)
                self.shares_held += shares_bought
                self.balance -= invest_amount
            elif action_val < 0 and self.shares_held > 0: # Sell
                sell_ratio = abs(action_val)
                shares_sold = self.shares_held * sell_ratio
                self.balance += shares_sold * current_price * (1 - self.transaction_cost)
                self.shares_held -= shares_sold

        # 2. CẬP NHẬT NET WORTH
        self.net_worth = self.balance + self.shares_held * current_price

        # 3. 🚀 CẢI TIẾN 2: ASYMMETRIC REWARD (Dạy AI biết sợ thị trường Gấu)
        step_profit = self.net_worth - prev_net_worth
        
        # Phạt nặng hơn 1.5 lần nếu bước vừa rồi gây lỗ
        if step_profit < 0:
            step_profit *= 1.5 
            
        # Giữ nguyên mức nhân x100.0 của phiên bản tốt nhất
        reward = (step_profit / self.initial_balance) * 100.0

        # 4. KIỂM TRA KẾT THÚC
        if self.current_step >= len(self.df) - 1 or self.net_worth <= self.initial_balance * 0.2:
            self.done = True

        return self._get_obs(), reward, self.done, {'net_worth': self.net_worth}