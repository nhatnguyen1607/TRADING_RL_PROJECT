import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.data_loader import load_and_preprocess_data
from envs.trading_env import TradingEnv
from agents.dqn_agent import DQNAgent
from agents.ac_agent import ACAgent

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.sqrt(252) * (np.mean(returns) - risk_free_rate) / np.std(returns)

def train_dqn(env, episodes=50):
    print("\n🚀 Bắt đầu Train DQN...")
    state_dim = env.observation_space.shape[-1]
    agent = DQNAgent(state_dim=state_dim, action_dim=env.action_space.n)
    
    # Tracking Metrics cho Deep Learning
    history = {'rewards': [], 'loss': [], 'epsilon': []}
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        ep_losses = []
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0: ep_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
        agent.update_target_network()
        agent.decay_epsilon()
        
        # Lưu metrics
        avg_loss = np.mean(ep_losses) if len(ep_losses) > 0 else 0
        history['rewards'].append(total_reward)
        history['loss'].append(avg_loss)
        history['epsilon'].append(agent.epsilon)
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"DQN Ep {ep+1}/{episodes} | Rwd: {total_reward:.2f} | Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f} | NetWorth: ${info['net_worth']:.0f}")
        
    return agent, history

def train_ac(env, episodes=50):
    print("\n🚀 Bắt đầu Train Actor-Critic...")
    state_dim = env.observation_space.shape[-1]
    agent = ACAgent(state_dim=state_dim)
    history = {'rewards': [], 'loss': []}
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        ep_losses = []
        done = False
        
        while not done:
            action, log_prob = agent.act(state)
            next_state, reward, done, info = env.step(action)
            
            loss = agent.train_step(state, log_prob, reward, next_state, done)
            ep_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
        avg_loss = np.mean(ep_losses) if len(ep_losses) > 0 else 0
        history['rewards'].append(total_reward)
        history['loss'].append(avg_loss)
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"AC Ep {ep+1}/{episodes} | Rwd: {total_reward:.2f} | Loss: {avg_loss:.4f} | NetWorth: ${info['net_worth']:.0f}")
        
    return agent, history

def evaluate_and_log_trades(env, agent, test_df, model_name, is_dqn=True):
    """Hàm chạy test và ghi lại từng bước mua bán"""
    state = env.reset()
    done = False
    net_worths = [env.initial_balance]
    returns = []
    trade_log = []
    step = 0
    
    if is_dqn: agent.epsilon = 0.0 # Tắt random khi test
        
    while not done:
        # Lấy giá đóng cửa ngày hôm đó để ghi log
        date = test_df.index[env.current_step] if type(test_df.index) == pd.DatetimeIndex else env.current_step
        price = test_df['Close'].iloc[env.current_step]
        
        if is_dqn:
            action = agent.act(state)
            act_str = "HOLD" if action == 0 else ("BUY MAX" if action == 1 else "SELL ALL")
        else:
            action, _ = agent.act(state)
            act_val = action[0]
            act_str = f"BUY {act_val:.2%}" if act_val > 0 else (f"SELL {abs(act_val):.2%}" if act_val < 0 else "HOLD")
            
        next_state, reward, done, info = env.step(action)
        
        # Ghi nhật ký
        trade_log.append({
            "Step": step,
            "Date": date,
            "Close_Price": price,
            "Action": act_str,
            "Net_Worth": info['net_worth'],
            "Reward": reward
        })
        
        net_worths.append(info['net_worth'])
        returns.append(reward)
        state = next_state
        step += 1
        
    sharpe = calculate_sharpe_ratio(returns)
    trade_df = pd.DataFrame(trade_log)
    
    # Xuất file CSV
    trade_df.to_csv(f'results/{model_name}_trade_log.csv', index=False)
    
    return net_worths, sharpe, trade_df

def plot_deep_learning_metrics(dqn_hist, ac_hist):
    """Vẽ biểu đồ Loss và Reward hội tụ của Deep Learning"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Deep Learning Training Metrics", fontsize=16)
    
    # DQN Rewards & Loss
    axs[0, 0].plot(dqn_hist['rewards'], color='blue')
    axs[0, 0].set_title('DQN Episode Rewards')
    axs[0, 1].plot(dqn_hist['loss'], color='red')
    axs[0, 1].set_title('DQN Training Loss (MSE)')
    
    # AC Rewards & Loss
    axs[1, 0].plot(ac_hist['rewards'], color='orange')
    axs[1, 0].set_title('Actor-Critic Episode Rewards')
    axs[1, 1].plot(ac_hist['loss'], color='purple')
    axs[1, 1].set_title('Actor-Critic Training Loss')
    
    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Episodes')
        
    plt.tight_layout()
    plt.savefig('results/dl_metrics.png')
    

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    df, _ = load_and_preprocess_data("SPY", start="2015-01-01", end="2023-01-01")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    env_train_dqn = TradingEnv(train_df, window_size=60,is_discrete=True)
    env_test_dqn = TradingEnv(test_df, window_size=60,is_discrete=True)
    
    env_train_ac = TradingEnv(train_df, window_size=60, is_discrete=False)
    env_test_ac = TradingEnv(test_df, window_size=60, is_discrete=False)
    
    # Đã tăng số episode lên 50 để mô hình có đủ thời gian "thoát" khỏi mode collapse
    trained_dqn, dqn_history = train_dqn(env_train_dqn, episodes=100)
    trained_ac, ac_history = train_ac(env_train_ac, episodes=100)
    
    # Vẽ và lưu biểu đồ Loss/Reward của Deep Learning
    plot_deep_learning_metrics(dqn_history, ac_history)
    
    print("\n📊 Đang Test và Xuất File Nhật ký giao dịch...")
    dqn_net_worths, dqn_sharpe, dqn_trades = evaluate_and_log_trades(env_test_dqn, trained_dqn, test_df, "DQN", is_dqn=True)
    ac_net_worths, ac_sharpe, ac_trades = evaluate_and_log_trades(env_test_ac, trained_ac, test_df, "AC", is_dqn=False)
    
    print(f"\n✅ Đã lưu file Trade Log: results/DQN_trade_log.csv")
    print(f"✅ Đã lưu file Trade Log: results/AC_trade_log.csv")
    print(f"✅ Đã lưu biểu đồ DL Metrics: results/dl_metrics.png")
    
    # Baseline
    initial_price = test_df['Close'].iloc[0]
    buy_hold_net_worths = [env_test_dqn.initial_balance]
    shares = env_test_dqn.initial_balance / initial_price
    for price in test_df['Close'].iloc[1:]:
        buy_hold_net_worths.append(shares * price)
        
    # Plot cuối cùng
    plt.figure(figsize=(14, 7))
    plt.plot(dqn_net_worths, label=f'DQN Agent (Sharpe: {dqn_sharpe:.2f})', color='blue')
    plt.plot(ac_net_worths, label=f'Actor-Critic Agent (Sharpe: {ac_sharpe:.2f})', color='orange')
    plt.plot(buy_hold_net_worths, label='Buy & Hold', color='gray', linestyle='--')
    
    plt.title('Backtesting Performance', fontsize=14)
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/equity_curve_comparison.png')
    # plt.show()