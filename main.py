import argparse

from training.pipeline import run_experiment
from training.settings import config_for_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Trading RL research pipeline launcher.")
    parser.add_argument("--mode", choices=["baseline", "fast", "research", "macro_research", "options_research", "smoke"], default="baseline")
    parser.add_argument("--encoder", choices=["gru", "transformer", "graph_transformer"], default=None)
    parser.add_argument("--dqn-episodes", type=int, default=None)
    parser.add_argument("--ac-episodes", type=int, default=None)
    parser.add_argument("--sac-episodes", type=int, default=None)
    parser.add_argument("--sac-alpha", type=float, default=None)
    parser.add_argument("--hedge-graph-episodes", type=int, default=None)
    parser.add_argument("--hedge-prior-strength", type=float, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--sentiment-csv", default=None)
    parser.add_argument("--options-csv", default=None)
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--macro", action="store_true")
    parser.add_argument("--utility-cvar", action="store_true")
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument("--skip-ac", action="store_true")
    parser.add_argument("--skip-sac", action="store_true")
    parser.add_argument("--run-hedge-graph-dqn", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = config_for_mode(args.mode)
    if args.encoder is not None:
        cfg.encoder_type = args.encoder
    if args.dqn_episodes is not None:
        cfg.dqn_episodes = args.dqn_episodes
    if args.ac_episodes is not None:
        cfg.ac_episodes = args.ac_episodes
    if args.sac_episodes is not None:
        cfg.sac_episodes = args.sac_episodes
    if args.sac_alpha is not None:
        cfg.sac_alpha = args.sac_alpha
    if args.hedge_graph_episodes is not None:
        cfg.hedge_graph_episodes = args.hedge_graph_episodes
    if args.hedge_prior_strength is not None:
        cfg.hedge_prior_strength = args.hedge_prior_strength
    if args.validation_interval is not None:
        cfg.validation_interval = args.validation_interval
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir
    if args.sentiment_csv is not None:
        cfg.sentiment_path = args.sentiment_csv
    if args.options_csv is not None:
        cfg.options_path = args.options_csv
    if args.tickers is not None:
        cfg.tickers = tuple(args.tickers)
    if args.no_macro:
        cfg.include_macro = False
    if args.macro:
        cfg.include_macro = True
    if args.utility_cvar:
        cfg.reward_mode = "utility_cvar"
    if args.skip_dqn:
        cfg.run_dqn = False
    if args.skip_ac:
        cfg.run_ac = False
    if args.skip_sac:
        cfg.run_sac = False
    if args.run_hedge_graph_dqn:
        cfg.run_hedge_graph_dqn = True
    run_experiment(cfg)


if __name__ == "__main__":
    main()
