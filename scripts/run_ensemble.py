import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.ensemble import run_offline_ensemble


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline regime-gated expert ensemble evaluation.")
    parser.add_argument("--results-dir", default="results/ensemble_current")
    parser.add_argument("--strategy", choices=["ac_dqn_blend", "regime_gate", "adaptive_blend"], default="ac_dqn_blend")
    parser.add_argument("--ac-weight", type=float, default=0.50)
    parser.add_argument("--transaction-cost", type=float, default=0.001)
    parser.add_argument("--adaptive-window", type=int, default=20)
    parser.add_argument("--adaptive-temperature", type=float, default=1.0)
    parser.add_argument("--min-ac-weight", type=float, default=0.25)
    parser.add_argument("--max-ac-weight", type=float, default=0.75)
    parser.add_argument("--dqn-log", default=None)
    parser.add_argument("--ac-log", default=None)
    parser.add_argument("--max-weight-delta", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    _, metrics = run_offline_ensemble(
        results_dir=args.results_dir,
        strategy=args.strategy,
        ac_weight=args.ac_weight,
        transaction_cost=args.transaction_cost,
        adaptive_window=args.adaptive_window,
        adaptive_temperature=args.adaptive_temperature,
        min_ac_weight=args.min_ac_weight,
        max_ac_weight=args.max_ac_weight,
        max_weight_delta=args.max_weight_delta,
        expert_paths={
            key: value
            for key, value in {
                "DQN_BASE": args.dqn_log,
                "AC_BASE": args.ac_log,
            }.items()
            if value is not None
        },
    )
    print(f"Ensemble report written to {metrics['report_path']}")
    print(
        f"Final=${metrics['final_net_worth']:.2f} | "
        f"Sharpe={metrics['sharpe']:.4f} | "
        f"Sortino={metrics['sortino']:.4f} | "
        f"MaxDD={metrics['max_drawdown']:.2%}"
    )


if __name__ == "__main__":
    main()
