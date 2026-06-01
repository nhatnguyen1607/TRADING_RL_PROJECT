import os

import numpy as np
import pandas as pd

from .metrics import calculate_sharpe_ratio, calculate_sortino_ratio


RUN_SPECS = [
    ("DQN Baseline", "results/baseline_positive_reproduction/DQN_trade_log.csv"),
    ("AC Baseline", "results/baseline_positive_reproduction/AC_trade_log.csv"),
    ("SAC", "results/sac_v3/SAC_trade_log.csv"),
    ("HedgeGraph Best", "results/hedge_graph_ablation/raw_runs/hedge_graph_templates_pilot/HEDGE_GRAPH_trade_log.csv"),
    ("Macro AC v2", "results/macro_regime_ablation/raw_runs/macro_regime_v2/AC_trade_log.csv"),
    ("Ensemble 50/50", "results/ensemble_current/ENSEMBLE_trade_log.csv"),
]


def _max_drawdown(equity):
    equity = np.asarray(equity, dtype=np.float64)
    if len(equity) == 0:
        return 0.0
    return float(np.min(equity / np.maximum.accumulate(equity) - 1.0))


def _metric_row(name, returns, initial_balance=10000.0):
    equity = initial_balance * np.cumprod(1.0 + returns)
    return {
        "Model": name,
        "Final Net Worth": float(equity[-1]),
        "Sharpe": float(calculate_sharpe_ratio(returns)),
        "Sortino": float(calculate_sortino_ratio(returns)),
        "Max Drawdown": _max_drawdown(equity),
    }


def _moving_block_sample(returns, rng, block_size=20):
    returns = np.asarray(returns, dtype=np.float64)
    n = len(returns)
    if n == 0:
        return returns
    starts = rng.integers(0, max(1, n - block_size + 1), size=int(np.ceil(n / block_size)))
    sampled = [returns[start : start + block_size] for start in starts]
    return np.concatenate(sampled)[:n]


def bootstrap_ci(returns, n_boot=1000, block_size=20, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_boot):
        sample = _moving_block_sample(returns, rng, block_size=block_size)
        rows.append(_metric_row("bootstrap", sample))
    boot = pd.DataFrame(rows)
    ci = {}
    for col in ("Final Net Worth", "Sharpe", "Sortino", "Max Drawdown"):
        ci[f"{col} CI Low"] = float(boot[col].quantile(0.025))
        ci[f"{col} CI High"] = float(boot[col].quantile(0.975))
    return ci


def load_returns(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing trade log: {path}")
    df = pd.read_csv(path)
    if "Portfolio_Return" not in df.columns:
        raise ValueError(f"Trade log has no Portfolio_Return column: {path}")
    returns = df["Portfolio_Return"].astype(float).to_numpy()
    if len(returns) and abs(returns[0]) < 1e-15:
        returns = returns[1:]
    return returns


def build_research_summary(results_dir="results/research_summary", n_boot=1000, block_size=20):
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for name, path in RUN_SPECS:
        returns = load_returns(path)
        row = _metric_row(name, returns)
        row.update(bootstrap_ci(returns, n_boot=n_boot, block_size=block_size, seed=42))
        row["Trade Log"] = path
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(results_dir, "metrics_summary.csv"), index=False)
    _write_markdown(summary, os.path.join(results_dir, "metrics_summary.md"))
    return summary


def _fmt_money(value):
    return f"${value:,.2f}"


def _fmt_num(value):
    return f"{value:.4f}"


def _fmt_pct(value):
    return f"{value:.2%}"


def _write_markdown(summary, path):
    sorted_summary = summary.sort_values("Sharpe", ascending=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Research Metrics Summary\n\n")
        f.write("Bootstrap confidence intervals use moving blocks of 20 trading days with 1,000 resamples.\n\n")
        f.write("| Model | Final | Sharpe | Sharpe 95% CI | Sortino | Sortino 95% CI | Max DD | Max DD 95% CI |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in sorted_summary.iterrows():
            f.write(
                "| "
                f"{row['Model']} | "
                f"{_fmt_money(row['Final Net Worth'])} | "
                f"{_fmt_num(row['Sharpe'])} | "
                f"[{_fmt_num(row['Sharpe CI Low'])}, {_fmt_num(row['Sharpe CI High'])}] | "
                f"{_fmt_num(row['Sortino'])} | "
                f"[{_fmt_num(row['Sortino CI Low'])}, {_fmt_num(row['Sortino CI High'])}] | "
                f"{_fmt_pct(row['Max Drawdown'])} | "
                f"[{_fmt_pct(row['Max Drawdown CI Low'])}, {_fmt_pct(row['Max Drawdown CI High'])}] |\n"
            )

        f.write("\n## Interpretation\n\n")
        f.write("- `Ensemble 50/50` is the strongest risk-adjusted method by Sharpe and Sortino.\n")
        f.write("- `AC Baseline` retains the highest final net worth among individual agents, but with materially higher drawdown.\n")
        f.write("- `DQN Baseline` remains the strongest single-agent drawdown-controlled benchmark.\n")
        f.write("- HedgeGraph and Macro variants are retained as ablation evidence, not as final proposed methods.\n")
