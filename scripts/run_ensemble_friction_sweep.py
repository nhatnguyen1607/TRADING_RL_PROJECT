import argparse
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.ensemble import run_offline_ensemble


def parse_args():
    parser = argparse.ArgumentParser(description="Run ensemble friction sensitivity sweep.")
    parser.add_argument("--results-dir", default="results/ensemble_friction_sweep")
    parser.add_argument("--strategy", choices=["ac_dqn_blend", "adaptive_blend", "regime_gate"], default="adaptive_blend")
    parser.add_argument("--costs", nargs="+", type=float, default=[0.001, 0.0015, 0.002])
    return parser.parse_args()


def _write_markdown(rows, path):
    df = pd.DataFrame(rows)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Ensemble Friction Sensitivity\n\n")
        f.write("PnL is recomputed from weights and asset prices for each transaction-cost assumption.\n\n")
        f.write("| Strategy | Transaction Cost | Final | Sharpe | Sortino | Max DD | Turnover | Allocation |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, row in df.iterrows():
            f.write(
                f"| {row['Strategy']} | {row['Transaction Cost']:.4f} | "
                f"${row['Final Net Worth']:,.2f} | {row['Sharpe']:.4f} | {row['Sortino']:.4f} | "
                f"{row['Max Drawdown']:.2%} | {row['Average Turnover']:.2%} | {row['Average Allocation']:.2%} |\n"
            )


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    rows = []
    for cost in args.costs:
        label = str(cost).replace(".", "p")
        run_dir = os.path.join(args.results_dir, "raw_runs", f"{args.strategy}_cost_{label}")
        _, metrics = run_offline_ensemble(
            results_dir=run_dir,
            strategy=args.strategy,
            transaction_cost=cost,
        )
        rows.append(
            {
                "Strategy": args.strategy,
                "Transaction Cost": cost,
                "Final Net Worth": metrics["final_net_worth"],
                "Sharpe": metrics["sharpe"],
                "Sortino": metrics["sortino"],
                "Max Drawdown": metrics["max_drawdown"],
                "Average Turnover": metrics["avg_turnover"],
                "Average Allocation": metrics["avg_allocation"],
                "Report": metrics["report_path"],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.results_dir, "friction_sweep.csv"), index=False)
    _write_markdown(rows, os.path.join(args.results_dir, "friction_sweep.md"))
    print(df[["Strategy", "Transaction Cost", "Final Net Worth", "Sharpe", "Sortino", "Max Drawdown"]])


if __name__ == "__main__":
    main()
