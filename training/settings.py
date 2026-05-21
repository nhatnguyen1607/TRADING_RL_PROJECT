from dataclasses import dataclass


@dataclass
class RunConfig:
    tickers: tuple = ("SPY", "SH", "TLT")
    start: str = "2015-01-01"
    end: str = "2023-01-01"
    window_size: int = 30
    dqn_episodes: int = 40
    ac_episodes: int = 40
    validation_interval: int = 10
    seed: int = 42
    encoder_type: str = "gru"
    asset_feature_dim: int = 18
    reward_mode: str = "utility_cvar"
    include_macro: bool = True
    results_dir: str = "results"
    dqn_batch_size: int = 64
    dqn_train_every: int = 4
    ac_update_every: int = 4
    teacher_pretrain_epochs: int = 4
    run_dqn: bool = True
    run_ac: bool = True


def config_for_mode(mode):
    if mode == "research":
        return RunConfig(
            dqn_episodes=100,
            ac_episodes=100,
            validation_interval=10,
            encoder_type="graph_transformer",
            teacher_pretrain_epochs=12,
        )
    if mode == "smoke":
        return RunConfig(
            dqn_episodes=3,
            ac_episodes=3,
            validation_interval=1,
            encoder_type="gru",
            include_macro=False,
            teacher_pretrain_epochs=1,
        )
    return RunConfig()
