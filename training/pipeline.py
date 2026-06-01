import os

from envs.trading_env import MultiAssetTradingEnv
from utils.data_loader import load_multi_asset_data
from .data import set_seed, slice_with_attrs, split_with_attrs, train_test_scale
from .evaluation import (
    buy_and_hold_curve,
    evaluate_and_log_trades,
    plot_deep_learning_metrics,
    plot_equity_curves,
    write_report,
    write_run_metadata,
)
from .metrics import summarize_trade_log
from .trainers import train_ac, train_dqn, train_hedge_graph_dqn, train_sac


def make_multi_asset_env(df, cfg, is_discrete=True, ac_variant=False):
    kwargs = {
        "window_size": cfg.window_size,
        "is_discrete": is_discrete,
        "reward_mode": cfg.reward_mode,
        "max_weight_delta": cfg.dqn_max_weight_delta,
        "turnover_penalty_coef": cfg.dqn_turnover_penalty_coef,
        "drawdown_penalty_coef": cfg.dqn_drawdown_penalty_coef,
        "concentration_penalty_coef": cfg.dqn_concentration_penalty_coef,
        "target_portfolio_vol": cfg.dqn_target_portfolio_vol,
    }
    if ac_variant:
        kwargs.update(
            {
                "max_weight_delta": cfg.ac_max_weight_delta,
                "max_total_allocation": cfg.ac_max_total_allocation,
                "cash_logit_bias": cfg.ac_cash_logit_bias,
                "ac_temperature": 1.20,
                "ac_mix_power": 1.0,
                "concentration_penalty_coef": cfg.ac_concentration_penalty_coef,
                "turnover_penalty_coef": cfg.ac_turnover_penalty_coef,
                "drawdown_penalty_coef": cfg.ac_drawdown_penalty_coef,
                "risk_aversion": 4.0,
                "cvar_alpha": 0.95,
                "tail_risk_coef": 0.35,
                "target_portfolio_vol": cfg.ac_target_portfolio_vol,
                "regime_hedge_weight": cfg.ac_regime_hedge_weight,
                "macro_hedge_weight": cfg.ac_macro_hedge_weight,
                "regime_ma_window": cfg.ac_regime_ma_window,
            }
        )
    return MultiAssetTradingEnv(df, **kwargs)


def make_sac_env(df, cfg):
    return MultiAssetTradingEnv(
        df,
        window_size=cfg.window_size,
        is_discrete=True,
        reward_mode=cfg.reward_mode,
        max_weight_delta=cfg.dqn_max_weight_delta,
        turnover_penalty_coef=cfg.sac_turnover_penalty_coef,
        drawdown_penalty_coef=cfg.sac_drawdown_penalty_coef,
        target_portfolio_vol=cfg.dqn_target_portfolio_vol,
        regime_hedge_weight=cfg.sac_regime_hedge_weight,
        regime_ma_window=cfg.sac_regime_ma_window,
    )


def make_hedge_graph_env(df, cfg):
    return MultiAssetTradingEnv(
        df,
        window_size=cfg.window_size,
        is_discrete=True,
        reward_mode=cfg.reward_mode,
        max_weight_delta=cfg.hedge_graph_max_weight_delta,
        max_total_allocation=cfg.hedge_graph_max_total_allocation,
        turnover_penalty_coef=cfg.dqn_turnover_penalty_coef,
        drawdown_penalty_coef=cfg.dqn_drawdown_penalty_coef,
        target_portfolio_vol=cfg.dqn_target_portfolio_vol,
        portfolio_templates=cfg.hedge_graph_portfolio_templates,
    )


def build_datasets(cfg):
    df, _ = load_multi_asset_data(
        list(cfg.tickers),
        start=cfg.start,
        end=cfg.end,
        scale=False,
        include_macro=cfg.include_macro,
        sentiment_path=cfg.sentiment_path,
        options_path=cfg.options_path,
    )
    split_idx = int(len(df) * 0.8)
    train_df, test_df, _ = train_test_scale(df, split_idx)
    model_split_idx = int(len(train_df) * 0.85)
    model_train_df, val_df = split_with_attrs(train_df, model_split_idx)
    return model_train_df, val_df, test_df


def build_ac_validation_envs(val_df, cfg):
    val_len = len(val_df)
    val_starts = [0, max(0, val_len // 3), max(0, (2 * val_len) // 3)]
    val_ends = [max(cfg.window_size + 20, val_len // 3), max(2 * val_len // 3, cfg.window_size + 20), val_len]
    val_slices = []
    for start_idx, end_idx in zip(val_starts, val_ends):
        if end_idx - start_idx > cfg.window_size + 20:
            val_slices.append(slice_with_attrs(val_df, start_idx, end_idx))
    return [make_multi_asset_env(val_slice, cfg, is_discrete=True, ac_variant=True) for val_slice in val_slices]


def run_experiment(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.results_dir, exist_ok=True)
    write_run_metadata(os.path.join(cfg.results_dir, "run_config.txt"), cfg)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "latest_run.txt"), "w", encoding="utf-8") as f:
        f.write(f"{cfg.results_dir}\n")
    print(
        "Run config | "
        f"encoder={cfg.encoder_type} | reward={cfg.reward_mode} | "
        f"dqn_eps={cfg.dqn_episodes} | ac_eps={cfg.ac_episodes} | sac_eps={cfg.sac_episodes} | "
        f"hedge_graph_eps={cfg.hedge_graph_episodes} | "
        f"val_every={cfg.validation_interval} | macro={cfg.include_macro} | "
        f"options={bool(cfg.options_path)}"
    )

    model_train_df, val_df, test_df = build_datasets(cfg)
    env_train_dqn = make_multi_asset_env(model_train_df, cfg, is_discrete=True)
    env_val_dqn = make_multi_asset_env(val_df, cfg, is_discrete=True)
    env_test_dqn = make_multi_asset_env(test_df, cfg, is_discrete=True)

    env_train_ac = make_multi_asset_env(model_train_df, cfg, is_discrete=True, ac_variant=True)
    env_val_ac = build_ac_validation_envs(val_df, cfg)
    env_test_ac = make_multi_asset_env(test_df, cfg, is_discrete=True, ac_variant=True)

    env_train_sac = make_sac_env(model_train_df, cfg)
    env_val_sac = make_sac_env(val_df, cfg)
    env_test_sac = make_sac_env(test_df, cfg)

    trained_dqn = None
    dqn_history = {"rewards": [], "loss": []}
    if cfg.run_dqn:
        trained_dqn, dqn_history = train_dqn(env_train_dqn, cfg, val_env=env_val_dqn)

    trained_hedge_graph = None
    hedge_graph_history = {"rewards": [], "loss": []}
    if cfg.run_hedge_graph_dqn:
        env_train_hedge_graph = make_hedge_graph_env(model_train_df, cfg)
        env_val_hedge_graph = make_hedge_graph_env(val_df, cfg)
        env_test_hedge_graph = make_hedge_graph_env(test_df, cfg)
        trained_hedge_graph, hedge_graph_history = train_hedge_graph_dqn(
            env_train_hedge_graph,
            cfg,
            val_env=env_val_hedge_graph,
        )

    trained_ac = None
    ac_history = {"rewards": [], "loss": []}
    if cfg.run_ac:
        trained_ac, ac_history = train_ac(env_train_ac, cfg, val_env=env_val_ac, teacher_agent=trained_dqn)

    trained_sac = None
    sac_history = {"rewards": [], "loss": []}
    if cfg.run_sac:
        trained_sac, sac_history = train_sac(env_train_sac, cfg, val_env=env_val_sac, teacher_agent=trained_dqn)

    plot_deep_learning_metrics(
        dqn_history,
        ac_history,
        cfg.results_dir,
        sac_history=sac_history if cfg.run_sac else None,
        hedge_graph_history=hedge_graph_history if cfg.run_hedge_graph_dqn else None,
    )

    print("\nTesting agents and writing trade logs...")
    buy_hold_net_worths = buy_and_hold_curve(test_df, env_test_dqn.initial_balance, cfg.window_size)
    buy_hold_final = buy_hold_net_worths[-1]

    dqn_net_worths, dqn_sharpe = [], 0.0
    if trained_dqn is not None:
        dqn_net_worths, dqn_sharpe, dqn_trades, dqn_trade_count = evaluate_and_log_trades(
            env_test_dqn,
            trained_dqn,
            test_df,
            "DQN",
            cfg.results_dir,
            is_dqn=True,
        )
        write_report(
            os.path.join(cfg.results_dir, "dqn_report.txt"),
            "DEEP Q-NETWORK (DQN) RESULTS",
            env_test_dqn,
            dqn_net_worths[-1],
            dqn_sharpe,
            buy_hold_final,
            dqn_trade_count,
            summarize_trade_log(dqn_trades),
        )

    hedge_graph_net_worths, hedge_graph_sharpe = [], 0.0
    if trained_hedge_graph is not None:
        hedge_graph_net_worths, hedge_graph_sharpe, hedge_graph_trades, hedge_graph_trade_count = evaluate_and_log_trades(
            env_test_hedge_graph,
            trained_hedge_graph,
            test_df,
            "HEDGE_GRAPH",
            cfg.results_dir,
            is_dqn=True,
        )
        write_report(
            os.path.join(cfg.results_dir, "hedge_graph_report.txt"),
            "DYNAMIC HEDGE-GRAPH DQN RESULTS",
            env_test_hedge_graph,
            hedge_graph_net_worths[-1],
            hedge_graph_sharpe,
            buy_hold_final,
            hedge_graph_trade_count,
            summarize_trade_log(hedge_graph_trades),
        )

    ac_net_worths, ac_sharpe = [], 0.0
    if trained_ac is not None:
        ac_net_worths, ac_sharpe, ac_trades, ac_trade_count = evaluate_and_log_trades(
            env_test_ac,
            trained_ac,
            test_df,
            "AC",
            cfg.results_dir,
            is_dqn=False,
        )
        write_report(
            os.path.join(cfg.results_dir, "ac_report.txt"),
            "ACTOR-CRITIC RESULTS",
            env_test_ac,
            ac_net_worths[-1],
            ac_sharpe,
            buy_hold_final,
            ac_trade_count,
            summarize_trade_log(ac_trades),
        )

    sac_net_worths, sac_sharpe = [], 0.0
    if trained_sac is not None:
        sac_net_worths, sac_sharpe, sac_trades, sac_trade_count = evaluate_and_log_trades(
            env_test_sac,
            trained_sac,
            test_df,
            "SAC",
            cfg.results_dir,
            is_dqn=True,
        )
        write_report(
            os.path.join(cfg.results_dir, "sac_report.txt"),
            "DISCRETE SOFT ACTOR-CRITIC (SAC) RESULTS",
            env_test_sac,
            sac_net_worths[-1],
            sac_sharpe,
            buy_hold_final,
            sac_trade_count,
            summarize_trade_log(sac_trades),
        )

    plot_equity_curves(
        dqn_net_worths,
        ac_net_worths,
        buy_hold_net_worths,
        dqn_sharpe,
        ac_sharpe,
        cfg.results_dir,
        sac_net_worths=sac_net_worths,
        sac_sharpe=sac_sharpe,
        hedge_graph_net_worths=hedge_graph_net_worths,
        hedge_graph_sharpe=hedge_graph_sharpe,
    )
    print("Pipeline complete. Check results/ for reports, trade logs, and charts.")
