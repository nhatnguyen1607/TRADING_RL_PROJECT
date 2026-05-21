import argparse

from training.pipeline import run_experiment
from training.settings import config_for_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Trading RL research pipeline launcher.")
    parser.add_argument("--mode", choices=["fast", "research", "smoke"], default="fast")
    parser.add_argument("--encoder", choices=["gru", "transformer", "graph_transformer"], default=None)
    parser.add_argument("--dqn-episodes", type=int, default=None)
    parser.add_argument("--ac-episodes", type=int, default=None)
    parser.add_argument("--validation-interval", type=int, default=None)
    parser.add_argument("--no-macro", action="store_true")
    parser.add_argument("--skip-dqn", action="store_true")
    parser.add_argument("--skip-ac", action="store_true")
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
    if args.validation_interval is not None:
        cfg.validation_interval = args.validation_interval
    if args.no_macro:
        cfg.include_macro = False
    if args.skip_dqn:
        cfg.run_dqn = False
    if args.skip_ac:
        cfg.run_ac = False
    run_experiment(cfg)


if __name__ == "__main__":
    main()
