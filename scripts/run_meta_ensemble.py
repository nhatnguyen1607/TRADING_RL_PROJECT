import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.metrics import calculate_sharpe_ratio, calculate_sortino_ratio


def _parse_weights(value, n_assets=3):
    numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(value))
    weights = np.asarray([float(x) for x in numbers[:n_assets]], dtype=np.float64)
    if len(weights) != n_assets:
        return np.zeros(n_assets, dtype=np.float64)
    return weights


def _load_log(path, name):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df[f"{name}_Weights"] = df["Weights"].map(_parse_weights)
    return df


def _max_drawdown(equity):
    equity = np.asarray(equity, dtype=np.float64)
    if len(equity) == 0:
        return 0.0
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def _format_weights(weights):
    return "[" + "|".join(f"{x:.3f}".rstrip("0").rstrip(".") for x in weights) + "]"


def _rolling_score(returns, turnovers, idx, col, window):
    start = max(0, idx - window)
    if idx <= start + 4:
        return 0.0
    recent = returns[col].iloc[start:idx].to_numpy(dtype=np.float64)
    recent_turnover = turnovers[col].iloc[start:idx].to_numpy(dtype=np.float64)
    vol = float(np.std(recent))
    sharpe = float(np.sqrt(252.0) * np.mean(recent) / (vol + 1e-8)) if vol > 0 else 0.0
    equity = np.cumprod(1.0 + recent)
    drawdown = abs(_max_drawdown(equity))
    turnover_penalty = float(np.mean(recent_turnover))
    return sharpe - 1.5 * drawdown - 0.35 * turnover_penalty


def _expert_panels(panel, transaction_cost):
    prices = panel[["SPY_Close", "SH_Close", "TLT_Close"]].to_numpy(dtype=np.float64)
    returns = pd.DataFrame({"Date": panel["Date"]})
    turnovers = pd.DataFrame({"Date": panel["Date"]})
    for col in ["Baseline_Weights", "Options_Weights"]:
        prev_prices = None
        prev_weights = np.zeros(3, dtype=np.float64)
        expert_returns = []
        expert_turnovers = []
        for idx, weights in enumerate(panel[col]):
            if prev_prices is None or not np.all(np.isfinite(prev_prices)):
                gross_return = 0.0
            else:
                gross_return = float(prev_weights @ (prices[idx] / prev_prices - 1.0))
            turnover = float(np.sum(np.abs(weights - prev_weights)))
            expert_turnovers.append(turnover)
            expert_returns.append(gross_return - transaction_cost * turnover)
            prev_prices = prices[idx]
            prev_weights = weights
        prefix = col.replace("_Weights", "")
        returns[prefix] = expert_returns
        turnovers[prefix] = expert_turnovers
    return returns, turnovers


def _adaptive_options_weight(returns, turnovers, idx, window, temperature, min_weight, max_weight):
    base_score = _rolling_score(returns, turnovers, idx, "Baseline", window)
    opt_score = _rolling_score(returns, turnovers, idx, "Options", window)
    scores = np.asarray([base_score, opt_score], dtype=np.float64) / max(float(temperature), 1e-8)
    scores = scores - np.max(scores)
    probs = np.exp(scores)
    probs = probs / np.sum(probs)
    return float(np.clip(probs[1], min_weight, max_weight))


def run_meta_ensemble(
    baseline_log,
    options_log,
    results_dir,
    strategy="adaptive_blend",
    options_weight=0.20,
    transaction_cost=0.001,
    adaptive_window=20,
    adaptive_temperature=1.0,
    min_options_weight=0.0,
    max_options_weight=0.40,
    max_weight_delta=None,
    initial_balance=10000.0,
):
    if transaction_cost < 0.001:
        raise ValueError("Transaction cost must be at least 0.001.")
    os.makedirs(results_dir, exist_ok=True)

    baseline = _load_log(baseline_log, "Baseline")
    options = _load_log(options_log, "Options")
    panel = baseline[
        ["Date", "SPY_Close", "SH_Close", "TLT_Close", "Baseline_Weights"]
    ].merge(options[["Date", "Options_Weights"]], on="Date", how="inner")
    if panel.empty:
        raise ValueError("No overlapping dates between baseline and options logs.")

    expert_returns, expert_turnovers = _expert_panels(panel, transaction_cost)
    net_worth = float(initial_balance)
    max_net_worth = net_worth
    prev_weights = np.zeros(3, dtype=np.float64)
    prev_prices = None
    rows = []

    for idx, row in panel.iterrows():
        prices = row[["SPY_Close", "SH_Close", "TLT_Close"]].to_numpy(dtype=np.float64)
        if strategy == "fixed_blend":
            opt_w = float(np.clip(options_weight, 0.0, 1.0))
        elif strategy == "adaptive_blend":
            opt_w = _adaptive_options_weight(
                expert_returns,
                expert_turnovers,
                idx,
                adaptive_window,
                adaptive_temperature,
                min_options_weight,
                max_options_weight,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        target_weights = (1.0 - opt_w) * row["Baseline_Weights"] + opt_w * row["Options_Weights"]
        if max_weight_delta is not None:
            max_delta = float(max_weight_delta)
            target_weights = prev_weights + np.clip(target_weights - prev_weights, -max_delta, max_delta)
            target_weights = np.clip(target_weights, 0.0, 1.0)

        if prev_prices is None or not np.all(np.isfinite(prev_prices)):
            gross_return = 0.0
        else:
            gross_return = float(prev_weights @ (prices / prev_prices - 1.0))
        turnover = float(np.sum(np.abs(target_weights - prev_weights)))
        cost = transaction_cost * turnover
        net_worth *= 1.0 + gross_return
        net_worth *= max(0.0, 1.0 - cost)
        portfolio_return = gross_return - cost
        max_net_worth = max(max_net_worth, net_worth)

        rows.append(
            {
                "Date": row["Date"].strftime("%Y-%m-%d"),
                "Strategy": strategy,
                "Options_Weight": opt_w,
                "SPY_Close": prices[0],
                "SH_Close": prices[1],
                "TLT_Close": prices[2],
                "Weights": _format_weights(target_weights),
                "Target_Allocation": float(np.sum(target_weights)),
                "Turnover": turnover,
                "Portfolio_Return": portfolio_return,
                "Net_Worth": net_worth,
                "Drawdown": net_worth / max_net_worth - 1.0,
            }
        )
        prev_weights = target_weights
        prev_prices = prices

    trade_log = pd.DataFrame(rows)
    trade_path = os.path.join(results_dir, "META_ENSEMBLE_trade_log.csv")
    trade_log.to_csv(trade_path, index=False)

    returns = trade_log["Portfolio_Return"].to_numpy(dtype=np.float64)
    metrics = {
        "final_net_worth": float(trade_log["Net_Worth"].iloc[-1]),
        "sharpe": calculate_sharpe_ratio(returns),
        "sortino": calculate_sortino_ratio(returns),
        "max_drawdown": _max_drawdown(trade_log["Net_Worth"].to_numpy(dtype=np.float64)),
        "avg_turnover": float(trade_log["Turnover"].mean()),
        "avg_allocation": float(trade_log["Target_Allocation"].mean()),
        "avg_options_weight": float(trade_log["Options_Weight"].mean()),
    }
    report_path = os.path.join(results_dir, "meta_ensemble_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("META OFFLINE ENSEMBLE RESULTS\n")
        f.write("----------------------------------------\n")
        f.write(f"Strategy: {strategy}\n")
        f.write(f"Transaction Cost: {transaction_cost:.4f}\n")
        if strategy == "fixed_blend":
            f.write(f"Options Blend Weight: {options_weight:.2f}\n")
        else:
            f.write(f"Adaptive Window: {adaptive_window}\n")
            f.write(f"Adaptive Temperature: {adaptive_temperature:.2f}\n")
            f.write(f"Options Weight Clamp: [{min_options_weight:.2f}, {max_options_weight:.2f}]\n")
        if max_weight_delta is not None:
            f.write(f"Max Weight Delta: {float(max_weight_delta):.3f}\n")
        f.write(f"Initial Capital: ${initial_balance:.2f}\n")
        f.write(f"Final Net Worth: ${metrics['final_net_worth']:.2f}\n")
        f.write(f"Annualized Sharpe Ratio: {metrics['sharpe']:.4f}\n")
        f.write(f"Annualized Sortino Ratio: {metrics['sortino']:.4f}\n")
        f.write(f"Max Drawdown: {metrics['max_drawdown']:.2%}\n")
        f.write(f"Average Daily Turnover: {metrics['avg_turnover']:.2%}\n")
        f.write(f"Average Realized Allocation: {metrics['avg_allocation']:.2%}\n")
        f.write(f"Average Options Blend Weight: {metrics['avg_options_weight']:.2%}\n")
        f.write("Note: PnL is recomputed from blended weights and asset prices with transaction cost.\n")
    metrics["report_path"] = report_path
    metrics["trade_path"] = trade_path
    return trade_log, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Blend baseline and options ensemble trade logs.")
    parser.add_argument("--baseline-log", default="results/ensemble_current/ENSEMBLE_trade_log.csv")
    parser.add_argument("--options-log", default="results/options_ensemble_current/ENSEMBLE_trade_log.csv")
    parser.add_argument("--results-dir", default="results/meta_ensemble_current")
    parser.add_argument("--strategy", choices=["fixed_blend", "adaptive_blend"], default="adaptive_blend")
    parser.add_argument("--options-weight", type=float, default=0.20)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--adaptive-window", type=int, default=20)
    parser.add_argument("--adaptive-temperature", type=float, default=1.0)
    parser.add_argument("--min-options-weight", type=float, default=0.0)
    parser.add_argument("--max-options-weight", type=float, default=0.40)
    parser.add_argument("--max-weight-delta", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    _, metrics = run_meta_ensemble(
        baseline_log=args.baseline_log,
        options_log=args.options_log,
        results_dir=args.results_dir,
        strategy=args.strategy,
        options_weight=args.options_weight,
        transaction_cost=args.transaction_cost,
        adaptive_window=args.adaptive_window,
        adaptive_temperature=args.adaptive_temperature,
        min_options_weight=args.min_options_weight,
        max_options_weight=args.max_options_weight,
        max_weight_delta=args.max_weight_delta,
    )
    print(f"Meta ensemble report written to {metrics['report_path']}")
    print(
        f"Final=${metrics['final_net_worth']:.2f} | "
        f"Sharpe={metrics['sharpe']:.4f} | "
        f"Sortino={metrics['sortino']:.4f} | "
        f"MaxDD={metrics['max_drawdown']:.2%}"
    )


if __name__ == "__main__":
    main()
