import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import calculate_sharpe_ratio, calculate_sortino_ratio


def evaluate_policy(env, agent, is_dqn=True):
    state = env.reset()
    done = False
    returns = []
    turnovers = []
    net_worths = [env.initial_balance]
    net_worth = env.initial_balance
    trade_count = 0
    actions_taken = []

    old_epsilon = getattr(agent, "epsilon", None)
    if is_dqn and old_epsilon is not None:
        agent.epsilon = 0.0

    while not done:
        if is_dqn:
            action = agent.act(state)
        else:
            action, _, _ = agent.act(state, deterministic=True)
        actions_taken.append(int(action) if np.isscalar(action) or np.asarray(action).size == 1 else tuple(np.asarray(action)))
        state, _, done, info = env.step(action)
        returns.append(info["portfolio_return"])
        turnovers.append(info["turnover"])
        net_worth = info["net_worth"]
        trade_count = int(info.get("trade_count", trade_count))
        net_worths.append(net_worth)

    if is_dqn and old_epsilon is not None:
        agent.epsilon = old_epsilon

    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    avg_turnover = float(np.mean(turnovers)) if turnovers else 0.0
    equity = np.asarray(net_worths, dtype=np.float64)
    max_drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0))
    action_changes = sum(1 for i in range(1, len(actions_taken)) if actions_taken[i] != actions_taken[i - 1])
    unique_actions = len(set(actions_taken))
    return net_worth, sharpe, sortino, max_drawdown, avg_turnover, trade_count, action_changes, unique_actions


def evaluate_validation_bundle(val_envs, agent, is_dqn=True):
    if not isinstance(val_envs, (list, tuple)):
        return evaluate_policy(val_envs, agent, is_dqn=is_dqn)

    metrics = [evaluate_policy(env, agent, is_dqn=is_dqn) for env in val_envs]
    return (
        float(np.mean([m[0] for m in metrics])),
        float(np.mean([m[1] for m in metrics])),
        float(np.mean([m[2] for m in metrics])),
        float(np.mean([m[3] for m in metrics])),
        float(np.mean([m[4] for m in metrics])),
        int(round(np.mean([m[5] for m in metrics]))),
        int(round(np.mean([m[6] for m in metrics]))),
        int(min(m[7] for m in metrics)),
    )


def evaluate_and_log_trades(env, agent, test_df, model_name, results_dir, is_dqn=True):
    state = env.reset()
    done = False
    net_worths = [env.initial_balance]
    portfolio_returns = []
    trade_log = []
    step = 0

    old_epsilon = getattr(agent, "epsilon", None)
    if is_dqn and old_epsilon is not None:
        agent.epsilon = 0.0

    while not done:
        date = test_df.index[env.current_step] if isinstance(test_df.index, pd.DatetimeIndex) else env.current_step
        prices = {col: test_df[col].iloc[env.current_step] for col in env.asset_cols}

        if is_dqn:
            action = agent.act(state)
        else:
            action, _, _ = agent.act(state, deterministic=True)

        next_state, reward, done, info = env.step(action)
        target_weights = info.get("weights")
        target_repr = (
            np.array2string(np.asarray(target_weights), precision=3, separator="|")
            if target_weights is not None
            else ""
        )
        trade_log.append(
            {
                "Step": step,
                "Date": date,
                "Close_Price": ", ".join(f"{k}: {v}" for k, v in prices.items()),
                "Action": f"TARGET {info.get('target_allocation', 0.0):.0%}",
                "Target_Allocation": info.get("target_allocation", 0.0),
                "Realized_Allocation": info["allocation"],
                "Turnover": info["turnover"],
                "Net_Worth": info["net_worth"],
                "Portfolio_Return": info["portfolio_return"],
                "Reward": reward,
                "Weights": target_repr,
                "Cash_Weight": info.get("cash_weight", ""),
            }
        )
        net_worths.append(info["net_worth"])
        portfolio_returns.append(info["portfolio_return"])
        state = next_state
        step += 1

    if is_dqn and old_epsilon is not None:
        agent.epsilon = old_epsilon

    trade_df = pd.DataFrame(trade_log)
    trade_df.to_csv(os.path.join(results_dir, f"{model_name}_trade_log.csv"), index=False)
    meaningful_trades = int((trade_df["Turnover"] > 0.005).sum())
    return net_worths, calculate_sharpe_ratio(portfolio_returns), trade_df, meaningful_trades


def buy_and_hold_curve(test_df, initial_balance, window_size):
    close_col = "Close_SPY" if "Close_SPY" in test_df.columns else "Close"
    start_price = test_df[close_col].iloc[window_size]
    shares = initial_balance / start_price
    prices = test_df[close_col].iloc[window_size : window_size + len(test_df) - window_size]
    curve = (shares * prices).tolist()
    return [initial_balance] + curve[1:]


def plot_deep_learning_metrics(dqn_hist, ac_hist, results_dir):
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Deep Learning Training Metrics", fontsize=16)
    axs[0, 0].plot(dqn_hist.get("rewards", []), color="blue")
    axs[0, 0].set_title("DQN Episode Rewards")
    axs[0, 1].plot(dqn_hist.get("loss", []), color="red")
    axs[0, 1].set_title("DQN Training Loss (Huber)")
    axs[1, 0].plot(ac_hist.get("rewards", []), color="orange")
    axs[1, 0].set_title("Actor-Critic Episode Rewards")
    axs[1, 1].plot(ac_hist.get("loss", []), color="purple")
    axs[1, 1].set_title("Actor-Critic Training Loss")
    for ax in axs.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Episodes")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "dl_metrics.png"))
    plt.close()


def plot_equity_curves(dqn_net_worths, ac_net_worths, buy_hold_net_worths, dqn_sharpe, ac_sharpe, results_dir):
    plt.figure(figsize=(14, 7))
    if dqn_net_worths:
        plt.plot(dqn_net_worths, label=f"DQN Agent (Sharpe: {dqn_sharpe:.2f})", color="blue")
    if ac_net_worths:
        plt.plot(ac_net_worths, label=f"Actor-Critic Agent (Sharpe: {ac_sharpe:.2f})", color="orange")
    plt.plot(buy_hold_net_worths[: max(len(dqn_net_worths), len(ac_net_worths))], label="Buy & Hold", color="gray", linestyle="--")
    plt.title("Backtesting Performance", fontsize=14)
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, "equity_curve_comparison.png"))
    plt.close()


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
