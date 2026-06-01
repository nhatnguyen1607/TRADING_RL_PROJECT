import os
import re

import numpy as np
import pandas as pd

from .metrics import calculate_sharpe_ratio, calculate_sortino_ratio


EXPERT_PATHS = {
    "DQN_BASE": "results/baseline_positive_reproduction/DQN_trade_log.csv",
    "AC_BASE": "results/baseline_positive_reproduction/AC_trade_log.csv",
    "MACRO_AC_V2": "results/macro_regime_ablation/raw_runs/macro_regime_v2/AC_trade_log.csv",
    "HEDGE_GRAPH": "results/hedge_graph_ablation/raw_runs/hedge_graph_templates_pilot/HEDGE_GRAPH_trade_log.csv",
}


def _required_experts(strategy):
    if strategy in {"ac_dqn_blend", "adaptive_blend"}:
        return ("DQN_BASE", "AC_BASE")
    if strategy == "regime_gate":
        return ("AC_BASE", "HEDGE_GRAPH")
    raise ValueError(f"Unknown ensemble strategy: {strategy}")


def _parse_close_prices(value):
    prices = {}
    for key in ("SPY", "SH", "TLT"):
        match = re.search(rf"(?:Close_)?{key}:\s*([-+0-9.eE]+)", str(value))
        if match:
            prices[key] = float(match.group(1))
    return prices


def _parse_weights(value, n_assets=3):
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    weights = np.asarray([float(x) for x in numbers[:n_assets]], dtype=np.float64)
    if len(weights) != n_assets:
        return np.zeros(n_assets, dtype=np.float64)
    return weights


def _load_expert(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing expert trade log: {path}")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Weights_Array"] = df["Weights"].map(_parse_weights)
    return df


def _prepare_panel(experts):
    base = experts["AC_BASE"][["Date", "Close_Price"]].copy()
    close = base["Close_Price"].map(_parse_close_prices)
    base["SPY_Close"] = close.map(lambda x: x.get("SPY", np.nan))
    base["SH_Close"] = close.map(lambda x: x.get("SH", np.nan))
    base["TLT_Close"] = close.map(lambda x: x.get("TLT", np.nan))
    base["SMA20_Prev"] = base["SPY_Close"].rolling(20).mean().shift(1)
    base["Momentum20_Prev"] = base["SPY_Close"].pct_change(20).shift(1)
    return base


def _expert_return_panel(panel, experts, transaction_cost=0.001):
    returns = pd.DataFrame({"Date": panel["Date"]})
    turnovers = pd.DataFrame({"Date": panel["Date"]})
    prices = panel[["SPY_Close", "SH_Close", "TLT_Close"]].to_numpy(dtype=np.float64)
    for name, df in experts.items():
        weights_by_date = df.set_index("Date")["Weights_Array"].to_dict()
        weights = np.vstack([weights_by_date.get(date, np.zeros(3)) for date in panel["Date"]])
        expert_returns = np.zeros(len(panel), dtype=np.float64)
        expert_turnovers = np.zeros(len(panel), dtype=np.float64)
        prev_weights = np.zeros(3, dtype=np.float64)
        prev_prices = None
        for idx, target_weights in enumerate(weights):
            if prev_prices is None or not np.all(np.isfinite(prev_prices)):
                gross_return = 0.0
            else:
                gross_return = float(prev_weights @ (prices[idx] / prev_prices - 1.0))
            turnover = float(np.sum(np.abs(target_weights - prev_weights)))
            expert_turnovers[idx] = turnover
            expert_returns[idx] = gross_return - transaction_cost * turnover
            prev_weights = target_weights
            prev_prices = prices[idx]
        returns[name] = expert_returns
        turnovers[name] = expert_turnovers
    return returns, turnovers


def _rolling_expert_score(returns, turnovers, idx, expert_name, window=20):
    start = max(0, idx - window)
    if idx <= start + 4:
        return 0.0
    recent = returns[expert_name].iloc[start:idx].to_numpy(dtype=np.float64)
    recent_turnover = turnovers[expert_name].iloc[start:idx].to_numpy(dtype=np.float64)
    vol = float(np.std(recent))
    sharpe = float(np.sqrt(252.0) * np.mean(recent) / (vol + 1e-8)) if vol > 0 else 0.0
    equity = np.cumprod(1.0 + recent)
    drawdown = abs(_max_drawdown(equity))
    turnover_penalty = float(np.mean(recent_turnover))
    return sharpe - 1.50 * drawdown - 0.35 * turnover_penalty


def _adaptive_ac_weight(row, returns, turnovers, window=20, temperature=1.0, min_weight=0.25, max_weight=0.75):
    idx = int(row.name)
    ac_score = _rolling_expert_score(returns, turnovers, idx, "AC_BASE", window=window)
    dqn_score = _rolling_expert_score(returns, turnovers, idx, "DQN_BASE", window=window)
    scores = np.asarray([ac_score, dqn_score], dtype=np.float64) / max(float(temperature), 1e-6)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    return float(np.clip(probs[0], min_weight, max_weight))


def _select_regime_expert(row):
    if np.isfinite(row["SMA20_Prev"]) and np.isfinite(row["Momentum20_Prev"]):
        if row["SPY_Close"] < row["SMA20_Prev"] and row["Momentum20_Prev"] < 0.0:
            return "HEDGE_GRAPH"
    return "AC_BASE"


def _max_drawdown(equity):
    equity = np.asarray(equity, dtype=np.float64)
    if len(equity) == 0:
        return 0.0
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def _format_weights(weights):
    return "[" + "|".join(f"{x:.3f}".rstrip("0").rstrip(".") for x in weights) + "]"


def _strategy_weights(
    strategy,
    row,
    expert_by_date,
    ac_weight=0.50,
    expert_returns=None,
    expert_turnovers=None,
    adaptive_window=20,
    adaptive_temperature=1.0,
    min_ac_weight=0.25,
    max_ac_weight=0.75,
):
    date = row["Date"]
    if strategy == "regime_gate":
        expert = _select_regime_expert(row)
        return expert, np.asarray(expert_by_date[expert].get(date, np.zeros(3)), dtype=np.float64)

    if strategy == "ac_dqn_blend":
        ac_weights = np.asarray(expert_by_date["AC_BASE"].get(date, np.zeros(3)), dtype=np.float64)
        dqn_weights = np.asarray(expert_by_date["DQN_BASE"].get(date, np.zeros(3)), dtype=np.float64)
        ac_weight = float(np.clip(ac_weight, 0.0, 1.0))
        dqn_weight = 1.0 - ac_weight
        label = f"AC{int(round(ac_weight * 100)):02d}_DQN{int(round(dqn_weight * 100)):02d}_BLEND"
        return label, ac_weight * ac_weights + dqn_weight * dqn_weights

    if strategy == "adaptive_blend":
        if expert_returns is None or expert_turnovers is None:
            raise ValueError("adaptive_blend requires expert return and turnover panels.")
        ac_weight = _adaptive_ac_weight(
            row,
            expert_returns,
            expert_turnovers,
            window=adaptive_window,
            temperature=adaptive_temperature,
            min_weight=min_ac_weight,
            max_weight=max_ac_weight,
        )
        dqn_weight = 1.0 - ac_weight
        ac_weights = np.asarray(expert_by_date["AC_BASE"].get(date, np.zeros(3)), dtype=np.float64)
        dqn_weights = np.asarray(expert_by_date["DQN_BASE"].get(date, np.zeros(3)), dtype=np.float64)
        label = f"ADAPTIVE_AC{int(round(ac_weight * 100)):02d}_DQN{int(round(dqn_weight * 100)):02d}"
        return label, ac_weight * ac_weights + dqn_weight * dqn_weights

    raise ValueError(f"Unknown ensemble strategy: {strategy}")


def run_offline_ensemble(
    results_dir="results/ensemble_current",
    initial_balance=10000.0,
    transaction_cost=0.001,
    strategy="ac_dqn_blend",
    ac_weight=0.50,
    adaptive_window=20,
    adaptive_temperature=1.0,
    min_ac_weight=0.25,
    max_ac_weight=0.75,
):
    if transaction_cost < 0.001:
        raise ValueError("Transaction cost must be at least 0.001 to preserve realistic market-friction assumptions.")
    os.makedirs(results_dir, exist_ok=True)
    experts = {name: _load_expert(EXPERT_PATHS[name]) for name in _required_experts(strategy)}
    panel = _prepare_panel(experts)
    expert_returns, expert_turnovers = _expert_return_panel(panel, experts, transaction_cost=transaction_cost)

    expert_by_date = {
        name: df.set_index("Date")["Weights_Array"].to_dict()
        for name, df in experts.items()
    }

    net_worth = float(initial_balance)
    max_net_worth = net_worth
    prev_weights = np.zeros(3, dtype=np.float64)
    prev_prices = None
    rows = []

    for _, row in panel.iterrows():
        date = row["Date"]
        prices = np.asarray([row["SPY_Close"], row["SH_Close"], row["TLT_Close"]], dtype=np.float64)
        expert, target_weights = _strategy_weights(
            strategy,
            row,
            expert_by_date,
            ac_weight=ac_weight,
            expert_returns=expert_returns,
            expert_turnovers=expert_turnovers,
            adaptive_window=adaptive_window,
            adaptive_temperature=adaptive_temperature,
            min_ac_weight=min_ac_weight,
            max_ac_weight=max_ac_weight,
        )

        if prev_prices is None or not np.all(np.isfinite(prev_prices)):
            gross_return = 0.0
        else:
            asset_returns = prices / prev_prices - 1.0
            gross_return = float(prev_weights @ asset_returns)

        net_worth *= 1.0 + gross_return
        turnover = float(np.sum(np.abs(target_weights - prev_weights)))
        cost = transaction_cost * turnover
        net_worth *= max(0.0, 1.0 - cost)
        portfolio_return = gross_return - cost

        max_net_worth = max(max_net_worth, net_worth)
        drawdown = net_worth / max_net_worth - 1.0
        rows.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "Selected_Expert": expert,
                "SPY_Close": prices[0],
                "SH_Close": prices[1],
                "TLT_Close": prices[2],
                "Target_Allocation": float(np.sum(target_weights)),
                "Weights": _format_weights(target_weights),
                "Turnover": turnover,
                "Portfolio_Return": portfolio_return,
                "Net_Worth": net_worth,
                "Drawdown": drawdown,
                "SPY_SMA20_Prev": row["SMA20_Prev"],
                "SPY_Momentum20_Prev": row["Momentum20_Prev"],
            }
        )
        prev_weights = target_weights
        prev_prices = prices

    trade_log = pd.DataFrame(rows)
    trade_log.to_csv(os.path.join(results_dir, "ENSEMBLE_trade_log.csv"), index=False)

    returns = trade_log["Portfolio_Return"].astype(float).to_numpy()
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = _max_drawdown(trade_log["Net_Worth"].to_numpy())
    avg_turnover = float(trade_log["Turnover"].mean())
    avg_allocation = float(trade_log["Target_Allocation"].mean())
    final_net = float(trade_log["Net_Worth"].iloc[-1])
    expert_shares = trade_log["Selected_Expert"].value_counts(normalize=True)

    report_path = os.path.join(results_dir, "ensemble_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REGIME-GATED OFFLINE ENSEMBLE RESULTS\n")
        f.write("----------------------------------------\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Transaction Cost: {transaction_cost:.4f}\n")
        if strategy == "ac_dqn_blend":
            f.write(f"AC Blend Weight: {ac_weight:.2f}\n")
        if strategy == "adaptive_blend":
            f.write(f"Adaptive Window: {adaptive_window}\n")
            f.write(f"Adaptive Temperature: {adaptive_temperature:.2f}\n")
            f.write(f"AC Weight Clamp: [{min_ac_weight:.2f}, {max_ac_weight:.2f}]\n")
        f.write(f"Initial Capital: ${initial_balance:.2f}\n")
        f.write(f"Final Net Worth: ${final_net:.2f}\n")
        f.write(f"Annualized Sharpe Ratio: {sharpe:.4f}\n")
        f.write(f"Annualized Sortino Ratio: {sortino:.4f}\n")
        f.write(f"Max Drawdown: {max_dd:.2%}\n")
        f.write(f"Average Daily Turnover: {avg_turnover:.2%}\n")
        f.write(f"Average Realized Allocation: {avg_allocation:.2%}\n")
        f.write("Expert Selection Shares: ")
        f.write(", ".join(f"{name}: {share:.2%}" for name, share in expert_shares.items()))
        f.write("\n")
        if strategy == "regime_gate":
            f.write("Gate: choose HEDGE_GRAPH when SPY is below prior SMA20 and prior 20-day momentum is negative; otherwise choose AC_BASE.\n")
        elif strategy == "adaptive_blend":
            f.write("Rule: dynamically blend AC_BASE and DQN_BASE using prior rolling Sharpe, drawdown, and turnover scores.\n")
        else:
            f.write("Rule: use a fixed 50/50 blend of AC_BASE and DQN_BASE target weights.\n")
        f.write("Note: PnL is recomputed from selected expert weights and asset prices, including expert-switch turnover cost.\n")

    return trade_log, {
        "final_net_worth": final_net,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_dd,
        "avg_turnover": avg_turnover,
        "avg_allocation": avg_allocation,
        "expert_shares": expert_shares.to_dict(),
        "report_path": report_path,
    }
