from dataclasses import dataclass


@dataclass
class RunConfig:
    mode_name: str = "baseline"
    tickers: tuple = ("SPY", "SH", "TLT")
    start: str = "2015-01-01"
    end: str = "2023-01-01"
    window_size: int = 30
    dqn_episodes: int = 100
    ac_episodes: int = 100
    validation_interval: int = 5
    seed: int = 42
    encoder_type: str = "gru"
    asset_feature_dim: int = 18
    reward_mode: str = "heuristic"
    include_macro: bool = False
    sentiment_path: str = None
    options_path: str = None
    results_dir: str = "results"
    dqn_batch_size: int = 64
    dqn_train_every: int = 4
    dqn_concentration_penalty_coef: float = 0.0
    dqn_dominance_threshold: float = 0.35
    ac_update_every: int = 4
    ac_algorithm: str = "ppo"
    ppo_epochs: int = 4
    ppo_minibatch_size: int = 128
    teacher_pretrain_epochs: int = 6
    dqn_max_weight_delta: float = 0.08
    dqn_turnover_penalty_coef: float = 0.0050
    dqn_drawdown_penalty_coef: float = 0.0030
    dqn_target_portfolio_vol: float = None
    ac_max_weight_delta: float = 0.06
    ac_max_total_allocation: float = 0.60
    ac_cash_logit_bias: float = 0.90
    ac_turnover_penalty_coef: float = 0.0055
    ac_drawdown_penalty_coef: float = 0.0035
    ac_concentration_penalty_coef: float = 0.0050
    ac_target_portfolio_vol: float = None
    ac_dominance_threshold: float = 0.30
    critic_target_mode: str = "td"
    ez_risk_aversion: float = 6.0
    ez_eis: float = 1.5
    ac_imitation_coef: float = 0.0
    ac_entropy_coef: float = 0.0040
    ac_teacher_smoothing: float = 0.35
    ac_online_teacher: bool = False
    ac_regime_hedge_weight: float = 0.35
    ac_macro_hedge_weight: float = 0.0
    ac_regime_ma_window: int = 80
    run_dqn: bool = True
    run_ac: bool = True
    run_sac: bool = True
    sac_episodes: int = 120
    sac_alpha: float = 0.05
    sac_batch_size: int = 128
    sac_teacher_pretrain_epochs: int = 0
    sac_teacher_smoothing: float = 0.25
    sac_turnover_penalty_coef: float = 0.0050
    sac_drawdown_penalty_coef: float = 0.0030
    sac_regime_hedge_weight: float = 0.0
    sac_regime_ma_window: int = 80
    run_hedge_graph_dqn: bool = False
    hedge_graph_episodes: int = 120
    hedge_graph_max_total_allocation: float = 0.45
    hedge_graph_max_weight_delta: float = 0.06
    hedge_prior_strength: float = 1.25
    hedge_risk_off_threshold: float = 0.76
    hedge_risk_on_threshold: float = 0.38
    hedge_hard_mask_strength: float = 1.50
    hedge_transition_de_risk_strength: float = 0.80
    hedge_recovery_strength: float = 0.80
    hedge_use_regime_action_shield: bool = False
    hedge_shield_penalty_strength: float = 0.75
    hedge_graph_portfolio_templates: tuple = (
        (0.00, 0.00, 0.00),
        (0.20, 0.00, 0.10),
        (0.35, 0.00, 0.10),
        (0.25, 0.05, 0.10),
        (0.10, 0.12, 0.10),
        (0.05, 0.18, 0.10),
        (0.00, 0.22, 0.10),
        (0.00, 0.32, 0.08),
        (0.00, 0.20, 0.00),
        (0.00, 0.00, 0.25),
    )


def config_for_mode(mode):
    if mode == "fast":
        return RunConfig(
            mode_name="fast",
            dqn_episodes=40,
            ac_episodes=40,
            sac_episodes=40,
            hedge_graph_episodes=40,
            validation_interval=10,
            teacher_pretrain_epochs=4,
            dqn_target_portfolio_vol=None,
            ac_target_portfolio_vol=None,
            results_dir="results/fast",
        )
    if mode == "research":
        return RunConfig(
            mode_name="research",
            dqn_episodes=120,
            ac_episodes=120,
            sac_episodes=120,
            hedge_graph_episodes=120,
            validation_interval=5,
            encoder_type="gru",
            reward_mode="heuristic",
            include_macro=False,
            critic_target_mode="td",
            teacher_pretrain_epochs=6,
            results_dir="results/research",
        )
    if mode == "macro_research":
        return RunConfig(
            mode_name="macro_research",
            dqn_episodes=120,
            ac_episodes=120,
            sac_episodes=120,
            hedge_graph_episodes=120,
            validation_interval=5,
            encoder_type="gru",
            reward_mode="heuristic",
            include_macro=True,
            critic_target_mode="td",
            teacher_pretrain_epochs=6,
            dqn_concentration_penalty_coef=0.0060,
            dqn_dominance_threshold=0.30,
            ac_max_total_allocation=0.50,
            ac_cash_logit_bias=1.05,
            ac_macro_hedge_weight=0.0,
            results_dir="results/macro_regime_current",
        )
    if mode == "options_research":
        return RunConfig(
            mode_name="options_research",
            tickers=("SPY", "TLT"),
            dqn_episodes=120,
            ac_episodes=120,
            sac_episodes=120,
            hedge_graph_episodes=120,
            validation_interval=5,
            encoder_type="gru",
            reward_mode="heuristic",
            include_macro=False,
            options_path="data/external/options_features_daily.csv",
            critic_target_mode="td",
            teacher_pretrain_epochs=6,
            results_dir="results/options_current",
        )
    if mode == "smoke":
        return RunConfig(
            mode_name="smoke",
            dqn_episodes=3,
            ac_episodes=3,
            sac_episodes=3,
            hedge_graph_episodes=3,
            validation_interval=1,
            encoder_type="gru",
            include_macro=False,
            teacher_pretrain_epochs=1,
            dqn_target_portfolio_vol=None,
            ac_target_portfolio_vol=None,
            results_dir="results/smoke",
        )
    return RunConfig(results_dir="results/baseline")
