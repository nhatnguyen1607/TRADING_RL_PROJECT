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

def train_dqn(env, episodes=100):
    print("\n🚀 Bắt đầu Train DQN (Dueling + Soft Update)...")
    # 🚀 Ép phẳng (Flatten) State cho MLP của FinRL
    flat_state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    agent = DQNAgent(state_dim=flat_state_dim, action_dim=env.action_space.n)
    
    history = {'rewards': [], 'loss': [], 'epsilon': []}
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        ep_losses = []
        done = False
        
        while not done:
            # 🔧 FIXED: Flatten state trước khi gửi vào agent
            state_flat = state.reshape(-1)
            action = agent.act(state_flat)
            next_state, reward, done, info = env.step(action)
            
            # 🔧 FIXED: Store flattened state
            next_state_flat = next_state.reshape(-1)
            agent.store_transition(state_flat, action, reward, next_state_flat, done)
            loss = agent.train_step() # 🚀 Soft update đã nằm trong này
            if loss > 0: ep_losses.append(loss)
            
            state = next_state
            total_reward += reward
            
        # 🚀 Đã xóa Hard Update (agent.update_target_network) ở đây
        agent.decay_epsilon()
        
        avg_loss = np.mean(ep_losses) if len(ep_losses) > 0 else 0
        history['rewards'].append(total_reward)
        history['loss'].append(avg_loss)
        history['epsilon'].append(agent.epsilon)
        
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"DQN Ep {ep+1}/{episodes} | Rwd: {total_reward:.2f} | Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f} | NetWorth: ${info['net_worth']:.0f}")
        
    return agent, history

def train_ac(env, episodes=100):
    print("\n🚀 Bắt đầu Train Actor-Critic (Entropy Bonus)...")
    flat_state_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
    agent = ACAgent(state_dim=flat_state_dim)
    history = {'rewards': [], 'loss': []}
    
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        ep_losses = []
        done = False
        
        while not done:
            # 🔧 FIXED: Flatten state trước khi gửi vào agent
            state_flat = state.reshape(-1)
            action, log_prob, dist = agent.act(state_flat)
            next_state, reward, done, info = env.step(action)
            
            # 🔧 FIXED: Truyền state_flat đã flatten vào train_step
            next_state_flat = next_state.reshape(-1)
            loss = agent.train_step(state_flat, log_prob, dist, reward, next_state_flat, done)
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
    state = env.reset()
    done = False
    net_worths = [env.initial_balance]
    returns = []
    trade_log = []
    step = 0
    
    if is_dqn: agent.epsilon = 0.0 # Tắt ngẫu nhiên khi đi thi
        
    while not done:
        date = test_df.index[env.current_step] if type(test_df.index) == pd.DatetimeIndex else env.current_step
        price = test_df['Close'].iloc[env.current_step]
        
        if is_dqn:
            # 🔧 FIXED: Flatten state trước khi gửi vào DQN agent
            state_flat = state.reshape(-1)
            action = agent.act(state_flat) # DQN chỉ trả về 1 biến
            act_str = "HOLD" if action == 0 else ("BUY MAX" if action == 1 else "SELL ALL")
        else:
            # 🔧 FIXED: Flatten state trước khi gửi vào AC agent
            state_flat = state.reshape(-1)
            action, _, _ = agent.act(state_flat) # AC trả về 3 biến
            act_val = action[0]
            act_str = f"BUY {act_val:.2%}" if act_val > 0 else (f"SELL {abs(act_val):.2%}" if act_val < 0 else "HOLD")
            
        next_state, reward, done, info = env.step(action)
        
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
    trade_df.to_csv(f'results/{model_name}_trade_log.csv', index=False)
    
    return net_worths, sharpe, trade_df

def plot_deep_learning_metrics(dqn_hist, ac_hist):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Deep Learning Training Metrics", fontsize=16)
    
    axs[0, 0].plot(dqn_hist['rewards'], color='blue')
    axs[0, 0].set_title('DQN Episode Rewards')
    axs[0, 1].plot(dqn_hist['loss'], color='red')
    axs[0, 1].set_title('DQN Training Loss (MSE)')
    
    axs[1, 0].plot(ac_hist['rewards'], color='orange')
    axs[1, 0].set_title('Actor-Critic Episode Rewards')
    axs[1, 1].plot(ac_hist['loss'], color='purple')
    axs[1, 1].set_title('Actor-Critic Training Loss')
    
    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Episodes')
        
    plt.tight_layout()
    plt.savefig('results/dl_metrics.png')
    plt.close() # Giải phóng bộ nhớ

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    df, _ = load_and_preprocess_data("SPY", start="2015-01-01", end="2023-01-01")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # 🚀 Window_size = 60 để AI nhìn được xu hướng dài hạn
    env_train_dqn = TradingEnv(train_df, window_size=60, is_discrete=True)
    env_test_dqn = TradingEnv(test_df, window_size=60, is_discrete=True)
    
    env_train_ac = TradingEnv(train_df, window_size=60, is_discrete=False)
    env_test_ac = TradingEnv(test_df, window_size=60, is_discrete=False)
    
    # 🚀 Train 100 Episodes
    trained_dqn, dqn_history = train_dqn(env_train_dqn, episodes=100)
    trained_ac, ac_history = train_ac(env_train_ac, episodes=100)
    
    plot_deep_learning_metrics(dqn_history, ac_history)
    
    print("\n📊 Đang Test và Xuất File Nhật ký giao dịch...")
    dqn_net_worths, dqn_sharpe, dqn_trades = evaluate_and_log_trades(env_test_dqn, trained_dqn, test_df, "DQN", is_dqn=True)
    ac_net_worths, ac_sharpe, ac_trades = evaluate_and_log_trades(env_test_ac, trained_ac, test_df, "AC", is_dqn=False)
    
    print(f"\n✅ Đã lưu file Trade Log: results/DQN_trade_log.csv")
    print(f"✅ Đã lưu file Trade Log: results/AC_trade_log.csv")
    print(f"✅ Đã lưu biểu đồ DL Metrics: results/dl_metrics.png")
    
    # Ghi Báo Cáo
    with open('results/dqn_report.txt', 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO KẾT QUẢ: DEEP Q-NETWORK (DQN)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Số vốn ban đầu: ${env_test_dqn.initial_balance:.2f}\n")
        f.write(f"Giá trị danh mục (Final Net Worth): ${dqn_net_worths[-1]:.2f}\n")
        f.write(f"Tỷ suất lợi nhuận trên rủi ro (Sharpe Ratio): {dqn_sharpe:.4f}\n")

    with open('results/ac_report.txt', 'w', encoding='utf-8') as f:
        f.write("BÁO CÁO KẾT QUẢ: ACTOR-CRITIC\n")
        f.write("-" * 40 + "\n")
        f.write(f"Số vốn ban đầu: ${env_test_ac.initial_balance:.2f}\n")
        f.write(f"Giá trị danh mục (Final Net Worth): ${ac_net_worths[-1]:.2f}\n")
        f.write(f"Tỷ suất lợi nhuận trên rủi ro (Sharpe Ratio): {ac_sharpe:.4f}\n")

    # Baseline (Buy & Hold)
    initial_price = test_df['Close'].iloc[0]
    buy_hold_net_worths = [env_test_dqn.initial_balance]
    shares = env_test_dqn.initial_balance / initial_price
    for price in test_df['Close'].iloc[1:]:
        buy_hold_net_worths.append(shares * price)
        
    # Plot kết quả
    plt.figure(figsize=(14, 7))
    plt.plot(dqn_net_worths, label=f'DQN Agent (Sharpe: {dqn_sharpe:.2f})', color='blue')
    plt.plot(ac_net_worths, label=f'Actor-Critic Agent (Sharpe: {ac_sharpe:.2f})', color='orange')
    plt.plot(buy_hold_net_worths, label='Buy & Hold', color='gray', linestyle='--')
    
    plt.title('Backtesting Performance (FinRL Architecture)', fontsize=14)
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/equity_curve_comparison.png')
    plt.close() # Chỉ lưu, không hiển thị
    
    print("🚀 HOÀN TẤT TOÀN BỘ PIPELINE! Hãy mở thư mục results/ để xem kết quả.")