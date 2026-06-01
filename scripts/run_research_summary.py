import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.research_summary import build_research_summary


def parse_args():
    parser = argparse.ArgumentParser(description="Build research metrics table with bootstrap confidence intervals.")
    parser.add_argument("--results-dir", default="results/research_summary")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--block-size", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_research_summary(
        results_dir=args.results_dir,
        n_boot=args.n_boot,
        block_size=args.block_size,
    )
    print(f"Wrote {args.results_dir}/metrics_summary.md")
    print(summary.sort_values("Sharpe", ascending=False)[["Model", "Final Net Worth", "Sharpe", "Sortino", "Max Drawdown"]])


if __name__ == "__main__":
    main()
