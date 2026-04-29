import os
import random
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from agents.ac_agent import ACAgent
from agents.dqn_agent import DQNAgent
from envs.trading_env import MultiAssetTradingEnv, TradingEnv
from utils.data_loader import load_and_preprocess_data, load_multi_asset_data

DQN_EPISODES = 100
AC_EPISODES = 100
WINDOW_SIZE = 30
AC_UPDATE_EVERY = 4
SUPERVISED_PRETRAIN_EPOCHS = 8


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.sqrt(252) * (np.mean(returns) - risk_free_rate) / np.std(returns)


def split_with_attrs(df, split_idx):
    left = df.iloc[:split_idx].copy()
    right = df.iloc[split_idx:].copy()
    left.attrs["feature_cols"] = df.attrs["feature_cols"]
    right.attrs["feature_cols"] = df.attrs["feature_cols"]
    if "asset_cols" in df.attrs:
        left.attrs["asset_cols"] = df.attrs["asset_cols"]
        right.attrs["asset_cols"] = df.attrs["asset_cols"]
    if "tickers" in df.attrs:
        left.attrs["tickers"] = df.attrs["tickers"]
        right.attrs["tickers"] = df.attrs["tickers"]
    return left, right


def train_test_scale(df, split_idx):
    feature_cols = df.attrs["feature_cols"]
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])
    train_df.attrs["feature_cols"] = feature_cols
    test_df.attrs["feature_cols"] = feature_cols
    if "asset_cols" in df.attrs:
        train_df.attrs["asset_cols"] = df.attrs["asset_cols"]
        test_df.attrs["asset_cols"] = df.attrs["asset_cols"]
    if "tickers" in df.attrs:
        train_df.attrs["tickers"] = df.attrs["tickers"]
        test_df.attrs["tickers"] = df.attrs["tickers"]
    return train_df, test_df, scaler


def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    returns = np.asarray(returns, dtype=np.float64)
    downside = returns[returns < risk_free_rate]
    if len(returns) < 2 or len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return np.sqrt(252) * (np.mean(returns) - risk_free_rate) / np.std(downside)


def validation_score(final_net_worth, sharpe, sortino, max_drawdown, avg_turnover, initial_balance=10000.0):
    total_return = final_net_worth / initial_balance - 1.0
    return 1.2 * sharpe + 0.4 * sortino + 1.5 * total_return - 1.5 * abs(max_drawdown) - 0.75 * avg_turnover


def select_trend_rule(df):
    close_col = "Close_SPY" if "Close_SPY" in df.columns else "Close"
    close = df[close_col].reset_index(drop=True)
    returns = close.pct_change().fillna(0.0).to_numpy()
    candidates = []

    for fast in range(3, 16):
        for slow in range(fast + 5, 81, 5):
            fast_ma = close.rolling(fast, min_periods=fast).mean()
            slow_ma = close.rolling(slow, min_periods=slow).mean()
            raw_signal = np.where(fast_ma > slow_ma, 1.0, -1.0)
            raw_signal[:slow] = 0.0
            signal = np.r_[0.0, raw_signal[:-1]]

            for long_exp in (0.2, 0.35, 0.5, 0.75):
                for short_exp in (-0.2, -0.35, -0.5, -0.75):
                    exposure = np.where(signal > 0, long_exp, np.where(signal < 0, short_exp, 0.0))
                    strat_returns = exposure * returns
                    if np.std(strat_returns) == 0:
                        continue
                    sharpe = calculate_sharpe_ratio(strat_returns)
                    equity = 10000.0 * np.cumprod(1.0 + strat_returns)
                    max_drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
                    score = sharpe + 0.5 * (equity[-1] / 10000.0 - 1.0) - 0.5 * abs(max_drawdown)
                    candidates.append((score, sharpe, fast, slow, long_exp, short_exp))

    if not candidates:
        return {"fast": 5, "slow": 20, "long": 0.35, "short": -0.35}

    _, sharpe, fast, slow, long_exp, short_exp = max(candidates, key=lambda item: item[0])
    print(
        f"Selected warm-start trend rule on validation: SMA {fast}/{slow}, "
        f"long={long_exp:.2f}, short={short_exp:.2f}, val Sharpe={sharpe:.2f}"
    )
    return {"fast": fast, "slow": slow, "long": long_exp, "short": short_exp}


def rule_exposure_for_step(df, step, rule):
    signal_idx = max(step - 1, 0)
    close_col = "Close_SPY" if "Close_SPY" in df.columns else "Close"
    close = df[close_col].iloc[: signal_idx + 1]
    if len(close) < rule["slow"]:
        return 0.0

    fast_ma = close.rolling(rule["fast"], min_periods=rule["fast"]).mean().iloc[-1]
    slow_ma = close.rolling(rule["slow"], min_periods=rule["slow"]).mean().iloc[-1]
    if not np.isfinite(fast_ma) or not np.isfinite(slow_ma):
        return 0.0
    return rule["long"] if fast_ma > slow_ma else rule["short"]


def supervised_dataset_from_env(env, rule):
    states = []
    exposures = []
    for step in range(env.window_size, len(env.df) - 1):
        obs_df = env.df[env.feature_cols].iloc[step - env.window_size : step].values
        account_info = np.array([[1.0, 0.0]] * env.window_size)
        states.append(np.hstack((obs_df, account_info)).astype(np.float32))
        exposures.append(rule_exposure_for_step(env.df, step, rule))
    return np.asarray(states), np.asarray(exposures, dtype=np.float32)


def pretrain_dqn(agent, env, rule, epochs=SUPERVISED_PRETRAIN_EPOCHS):
    states, exposures = supervised_dataset_from_env(env, rule)
    if len(states) == 0:
        return

    action_values = np.asarray(env.discrete_allocations, dtype=np.float32)
    labels = np.abs(exposures[:, None] - action_values[None, :]).argmin(axis=1)
    states_tensor = torch.FloatTensor(states)
    labels_tensor = torch.LongTensor(labels)
    optimizer = torch.optim.AdamW(agent.policy_net.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    indices = np.arange(len(states))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), 128):
            batch_idx = indices[start : start + 128]
            logits = agent.policy_net(states_tensor[batch_idx])
            loss = loss_fn(logits, labels_tensor[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
            optimizer.step()

    agent.target_net.load_state_dict(agent.policy_net.state_dict())


def pretrain_ac(agent, env, rule, epochs=SUPERVISED_PRETRAIN_EPOCHS):
    states, exposures = supervised_dataset_from_env(env, rule)
    if len(states) == 0:
        return

    states_tensor = torch.FloatTensor(states)
    targets_tensor = torch.FloatTensor(exposures).unsqueeze(1).clamp(-1.0, 1.0)
    optimizer = torch.optim.AdamW(agent.model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    indices = np.arange(len(states))
    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, len(indices), 128):
            batch_idx = indices[start : start + 128]
            mean, _, _ = agent.model(states_tensor[batch_idx])
            loss = loss_fn(mean, targets_tensor[batch_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), max_norm=1.0)
            optimizer.step()


def evaluate_policy(env, agent, is_dqn=True):
    state = env.reset()
    done = False
    returns = []
    turnovers = []
    net_worths = [env.initial_balance]
    net_worth = env.initial_balance

    old_epsilon = getattr(agent, "epsilon", None)
    if is_dqn:
        agent.epsilon = 0.0

    while not done:
        if is_dqn:
            action = agent.act(state)
        else:
            action, _, _ = agent.act(state, deterministic=True)
        state, reward, done, info = env.step(action)
        returns.append(info["portfolio_return"])
        turnovers.append(info["turnover"])
        net_worth = info["net_worth"]
        net_worths.append(net_worth)

    if is_dqn and old_epsilon is not None:
        agent.epsilon = old_epsilon

    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    equity = np.asarray(net_worths, dtype=np.float64)
    max_drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    return net_worth, sharpe, sortino, max_drawdown, avg_turnover


def train_dqn(env, episodes=DQN_EPISODES, val_env=None, warm_start_rule=None):
    print("\nStarting DQN training (GRU + Double DQN)...")
    agent = DQNAgent(state_dim=env.observation_space.shape, action_dim=env.action_space.n)
    if warm_start_rule is not None:
        print("Supervised warm-starting DQN from validation-selected trend rule...")
        pretrain_dqn(agent, env, warm_start_rule)
    history = {"rewards": [], "loss": [], "epsilon": []}
    best_state = copy.deepcopy(agent.policy_net.state_dict())
    best_score = -np.inf

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        ep_losses = []
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0:
                ep_losses.append(loss)

            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["rewards"].append(total_reward)
        history["loss"].append(avg_loss)
        history["epsilon"].append(agent.epsilon)

        if (ep + 1) % 5 == 0 or ep == 0:
            if val_env is not None:
                val_net, val_sharpe, val_sortino, val_dd, val_turnover = evaluate_policy(val_env, agent, is_dqn=True)
                score = validation_score(val_net, val_sharpe, val_sortino, val_dd, val_turnover, env.initial_balance)
                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(agent.policy_net.state_dict())
                val_msg = f" | ValNet: ${val_net:.0f} | ValSharpe: {val_sharpe:.2f}"
            else:
                val_msg = ""
            print(
                f"DQN Ep {ep + 1}/{episodes} | Reward: {total_reward:.2f} | "
                f"Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f} | "
                f"NetWorth: ${info['net_worth']:.0f}{val_msg}"
            )

    if val_env is not None:
        agent.policy_net.load_state_dict(best_state)
    return agent, history


def train_ac(env, episodes=AC_EPISODES, val_env=None, teacher_agent=None, warm_start_rule=None):
    print("\nStarting Actor-Critic training (online TD + DQN teacher regularization)...")
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, "shape") else 1
    agent = ACAgent(state_dim=env.observation_space.shape, action_dim=action_dim)
    if warm_start_rule is not None:
        print("Supervised warm-starting Actor-Critic from validation-selected trend rule...")
        pretrain_ac(agent, env, warm_start_rule)
    history = {"rewards": [], "loss": []}
    best_state = copy.deepcopy(agent.model.state_dict())
    best_score = -np.inf

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        ep_losses = []
        done = False

        step_idx = 0
        while not done:
            action, log_prob, dist = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if step_idx % AC_UPDATE_EVERY == 0 or done:
                imitation_target = None
                if teacher_agent is not None and hasattr(env, "portfolios"):
                    old_epsilon = teacher_agent.epsilon
                    teacher_agent.epsilon = 0.0
                    teacher_action = teacher_agent.act(state)
                    teacher_agent.epsilon = old_epsilon
                    asset_weights = env.portfolios[int(teacher_action)]
                    cash_weight = max(0.0, 1.0 - float(np.sum(asset_weights)))
                    imitation_target = np.asarray([cash_weight, *asset_weights], dtype=np.float32)
                elif teacher_agent is not None and hasattr(env, "discrete_allocations"):
                    old_epsilon = teacher_agent.epsilon
                    teacher_agent.epsilon = 0.0
                    teacher_action = teacher_agent.act(state)
                    teacher_agent.epsilon = old_epsilon
                    imitation_target = [-0.75, -0.35, 0.0, 0.35, 0.75][int(teacher_action)]
                loss = agent.train_step(
                    state,
                    log_prob,
                    dist,
                    reward,
                    next_state,
                    done,
                    imitation_target=imitation_target,
                )
                ep_losses.append(loss)

            state = next_state
            total_reward += reward
            step_idx += 1

        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["rewards"].append(total_reward)
        history["loss"].append(avg_loss)

        if (ep + 1) % 5 == 0 or ep == 0:
            if val_env is not None:
                val_net, val_sharpe, val_sortino, val_dd, val_turnover = evaluate_policy(val_env, agent, is_dqn=False)
                score = validation_score(val_net, val_sharpe, val_sortino, val_dd, val_turnover, env.initial_balance)
                if score > best_score:
                    best_score = score
                    best_state = copy.deepcopy(agent.model.state_dict())
                val_msg = f" | ValNet: ${val_net:.0f} | ValSharpe: {val_sharpe:.2f}"
            else:
                val_msg = ""
            print(
                f"AC Ep {ep + 1}/{episodes} | Reward: {total_reward:.2f} | "
                f"Loss: {avg_loss:.4f} | NetWorth: ${info['net_worth']:.0f}{val_msg}"
            )

    if val_env is not None:
        agent.model.load_state_dict(best_state)
    return agent, history


def evaluate_and_log_trades(env, agent, test_df, model_name, is_dqn=True):
    state = env.reset()
    done = False
    net_worths = [env.initial_balance]
    portfolio_returns = []
    trade_log = []
    step = 0

    if is_dqn:
        agent.epsilon = 0.0

    while not done:
        date = test_df.index[env.current_step] if isinstance(test_df.index, pd.DatetimeIndex) else env.current_step
        # print("Danh sách các cột trong test_df:", test_df.columns.tolist())
        price_SPY = test_df["Close_SPY"].iloc[env.current_step]
        price_SH  = test_df["Close_SH"].iloc[env.current_step]
        price_TLT = test_df["Close_TLT"].iloc[env.current_step]

# Sau đó cập nhật logic tính toán net_worth bên dưới cho phù hợp với số lượng cổ phiếu đang giữ của từng mã.

        if is_dqn:
            action = agent.act(state)
        else:
            action, _, _ = agent.act(state, deterministic=True)

        if hasattr(env, "_target_weights"):
            target_weights = env._smooth_target_weights(env._target_weights(action))
            target = float(np.sum(target_weights))
            target_repr = np.array2string(target_weights, precision=3, separator="|")
        else:
            target_weights = None
            target = env._target_allocation(action)
            target_repr = ""

        next_state, reward, done, info = env.step(action)
        action_str = f"TARGET {target:.0%}"

        trade_log.append(
            {
                "Step": step,
                "Date": date,
                "Close_Price": f"SPY: {price_SPY}, SH: {price_SH}, TLT: {price_TLT}",
                "Action": action_str,
                "Target_Allocation": target,
                "Realized_Allocation": info["allocation"],
                "Turnover": info["turnover"],
                "Net_Worth": info["net_worth"],
                "Portfolio_Return": info["portfolio_return"],
                "Reward": reward,
                "Weights": target_repr if target_weights is not None else info.get("weights", ""),
                "Cash_Weight": info.get("cash_weight", ""),
            }
        )

        net_worths.append(info["net_worth"])
        portfolio_returns.append(info["portfolio_return"])
        state = next_state
        step += 1

    sharpe = calculate_sharpe_ratio(portfolio_returns)
    trade_df = pd.DataFrame(trade_log)
    trade_df.to_csv(f"results/{model_name}_trade_log.csv", index=False)
    meaningful_trades = int((trade_df["Turnover"] > 0.005).sum())
    return net_worths, sharpe, trade_df, meaningful_trades


def buy_and_hold_curve(test_df, initial_balance, window_size):
    close_col = "Close_SPY" if "Close_SPY" in test_df.columns else "Close"
    start_price = test_df[close_col].iloc[window_size]
    shares = initial_balance / start_price
    prices = test_df[close_col].iloc[window_size : window_size + len(test_df) - window_size]
    curve = (shares * prices).tolist()
    return [initial_balance] + curve[1:]


def plot_deep_learning_metrics(dqn_hist, ac_hist):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Deep Learning Training Metrics", fontsize=16)

    axs[0, 0].plot(dqn_hist["rewards"], color="blue")
    axs[0, 0].set_title("DQN Episode Rewards")
    axs[0, 1].plot(dqn_hist["loss"], color="red")
    axs[0, 1].set_title("DQN Training Loss (Huber)")

    axs[1, 0].plot(ac_hist["rewards"], color="orange")
    axs[1, 0].set_title("Actor-Critic Episode Rewards")
    axs[1, 1].plot(ac_hist["loss"], color="purple")
    axs[1, 1].set_title("Actor-Critic Training Loss")

    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Episodes")

    plt.tight_layout()
    plt.savefig("results/dl_metrics.png")
    plt.close()


def summarize_trade_log(trade_df):
    returns = trade_df["Portfolio_Return"].astype(float).to_numpy()
    equity = trade_df["Net_Worth"].astype(float).to_numpy()
    max_drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0)) if len(equity) else 0.0
    avg_turnover = float(trade_df["Turnover"].astype(float).mean()) if len(trade_df) else 0.0
    avg_allocation = float(trade_df["Realized_Allocation"].astype(float).mean()) if len(trade_df) else 0.0
    sortino = calculate_sortino_ratio(returns)
    return {
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "avg_allocation": avg_allocation,
    }


def write_report(path, title, env, final_net_worth, sharpe, buy_hold_final, meaningful_trades, metrics):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write("-" * 40 + "\n")
        f.write(f"Initial Capital: ${env.initial_balance:.2f}\n")
        f.write(f"Final Net Worth: ${final_net_worth:.2f}\n")
        f.write(f"Buy & Hold Final Net Worth: ${buy_hold_final:.2f}\n")
        f.write(f"Annualized Sharpe Ratio: {sharpe:.4f}\n")
        f.write(f"Annualized Sortino Ratio: {metrics['sortino']:.4f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
        f.write(f"Average Daily Turnover: {metrics['avg_turnover']:.2%}\n")
        f.write(f"Average Realized Allocation: {metrics['avg_allocation']:.2%}\n")
        f.write(f"Meaningful Trades: {meaningful_trades}\n")


if __name__ == "__main__":
    set_seed(42)
    os.makedirs("results", exist_ok=True)

    df, _ = load_multi_asset_data(["SPY", "SH", "TLT"], start="2015-01-01", end="2023-01-01", scale=False)
    split_idx = int(len(df) * 0.8)
    train_df, test_df, _ = train_test_scale(df, split_idx)
    model_split_idx = int(len(train_df) * 0.85)
    model_train_df, val_df = split_with_attrs(train_df, model_split_idx)

    window_size = WINDOW_SIZE
    env_train_dqn = MultiAssetTradingEnv(model_train_df, window_size=window_size, is_discrete=True)
    env_val_dqn = MultiAssetTradingEnv(val_df, window_size=window_size, is_discrete=True)
    env_test_dqn = MultiAssetTradingEnv(test_df, window_size=window_size, is_discrete=True)

    env_train_ac = MultiAssetTradingEnv(model_train_df, window_size=window_size, is_discrete=False)
    env_val_ac = MultiAssetTradingEnv(val_df, window_size=window_size, is_discrete=False)
    env_test_ac = MultiAssetTradingEnv(test_df, window_size=window_size, is_discrete=False)

    trained_dqn, dqn_history = train_dqn(
        env_train_dqn,
        episodes=DQN_EPISODES,
        val_env=env_val_dqn,
        warm_start_rule=None,
    )
    trained_ac, ac_history = train_ac(
        env_train_ac,
        episodes=AC_EPISODES,
        val_env=env_val_ac,
        teacher_agent=None,
        warm_start_rule=None,
    )

    plot_deep_learning_metrics(dqn_history, ac_history)

    print("\nTesting agents and writing trade logs...")
    dqn_net_worths, dqn_sharpe, dqn_trades, dqn_trade_count = evaluate_and_log_trades(
        env_test_dqn, trained_dqn, test_df, "DQN", is_dqn=True
    )
    ac_net_worths, ac_sharpe, ac_trades, ac_trade_count = evaluate_and_log_trades(
        env_test_ac, trained_ac, test_df, "AC", is_dqn=False
    )

    buy_hold_net_worths = buy_and_hold_curve(test_df, env_test_dqn.initial_balance, window_size)
    buy_hold_final = buy_hold_net_worths[-1]
    dqn_metrics = summarize_trade_log(dqn_trades)
    ac_metrics = summarize_trade_log(ac_trades)

    write_report(
        "results/dqn_report.txt",
        "DEEP Q-NETWORK (DQN) RESULTS",
        env_test_dqn,
        dqn_net_worths[-1],
        dqn_sharpe,
        buy_hold_final,
        dqn_trade_count,
        dqn_metrics,
    )
    write_report(
        "results/ac_report.txt",
        "ACTOR-CRITIC RESULTS",
        env_test_ac,
        ac_net_worths[-1],
        ac_sharpe,
        buy_hold_final,
        ac_trade_count,
        ac_metrics,
    )

    plt.figure(figsize=(14, 7))
    plt.plot(dqn_net_worths, label=f"DQN Agent (Sharpe: {dqn_sharpe:.2f})", color="blue")
    plt.plot(ac_net_worths, label=f"Actor-Critic Agent (Sharpe: {ac_sharpe:.2f})", color="orange")
    plt.plot(buy_hold_net_worths[: len(dqn_net_worths)], label="Buy & Hold", color="gray", linestyle="--")
    plt.title("Backtesting Performance", fontsize=14)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/equity_curve_comparison.png")
    plt.close()

    print("Pipeline complete. Check results/ for reports, trade logs, and charts.")
