import numpy as np


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    returns = np.asarray(returns, dtype=np.float64)
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return np.sqrt(252) * (np.mean(returns) - risk_free_rate) / np.std(returns)


def calculate_sortino_ratio(returns, risk_free_rate=0.0):
    returns = np.asarray(returns, dtype=np.float64)
    downside = returns[returns < risk_free_rate]
    if len(returns) < 2 or len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return np.sqrt(252) * (np.mean(returns) - risk_free_rate) / np.std(downside)


def validation_score(
    final_net_worth,
    sharpe,
    sortino,
    max_drawdown,
    avg_turnover,
    initial_balance=10000.0,
    trade_count=None,
    action_changes=None,
    unique_actions=None,
):
    total_return = final_net_worth / initial_balance - 1.0
    score = 1.6 * sharpe + 0.6 * sortino + 1.2 * total_return - 2.0 * abs(max_drawdown) - 1.75 * avg_turnover
    if trade_count is not None and trade_count < 20:
        score -= 0.75
    if action_changes is not None and action_changes < 12:
        score -= 0.50
    if unique_actions is not None and unique_actions < 3:
        score -= 0.50
    return score


def summarize_trade_log(trade_df):
    returns = trade_df["Portfolio_Return"].astype(float).to_numpy()
    equity = trade_df["Net_Worth"].astype(float).to_numpy()
    max_drawdown = float(np.min(equity / np.maximum.accumulate(equity) - 1.0)) if len(equity) else 0.0
    avg_turnover = float(trade_df["Turnover"].astype(float).mean()) if len(trade_df) else 0.0
    avg_allocation = float(trade_df["Realized_Allocation"].astype(float).mean()) if len(trade_df) else 0.0
    dominant_weight = ""
    dominant_weight_share = 0.0
    action_changes = 0
    avg_policy_risk_weight = None
    avg_policy_stress_score = None
    avg_spy_tlt_correlation = None
    policy_regime_shares = None
    if "Weights" in trade_df.columns and len(trade_df):
        counts = trade_df["Weights"].astype(str).value_counts()
        dominant_weight = counts.index[0]
        dominant_weight_share = float(counts.iloc[0] / len(trade_df))
        action_changes = int((trade_df["Weights"].astype(str) != trade_df["Weights"].astype(str).shift()).sum())
    if "Policy_Risk_Weight" in trade_df.columns and len(trade_df):
        values = trade_df["Policy_Risk_Weight"].replace("", np.nan).astype(float).dropna()
        avg_policy_risk_weight = float(values.mean()) if len(values) else None
    if "Policy_Stress_Score" in trade_df.columns and len(trade_df):
        values = trade_df["Policy_Stress_Score"].replace("", np.nan).astype(float).dropna()
        avg_policy_stress_score = float(values.mean()) if len(values) else None
    if "Policy_SPY_TLT_Correlation" in trade_df.columns and len(trade_df):
        values = trade_df["Policy_SPY_TLT_Correlation"].replace("", np.nan).astype(float).dropna()
        avg_spy_tlt_correlation = float(values.mean()) if len(values) else None
    if "Policy_Regime_State" in trade_df.columns and len(trade_df):
        policy_regime_shares = (trade_df["Policy_Regime_State"].value_counts(normalize=True) * 100.0).to_dict()

    return {
        "sortino": calculate_sortino_ratio(returns),
        "max_drawdown": max_drawdown,
        "avg_turnover": avg_turnover,
        "avg_allocation": avg_allocation,
        "dominant_weight": dominant_weight,
        "dominant_weight_share": dominant_weight_share,
        "action_changes": action_changes,
        "avg_policy_risk_weight": avg_policy_risk_weight,
        "avg_policy_stress_score": avg_policy_stress_score,
        "avg_spy_tlt_correlation": avg_spy_tlt_correlation,
        "policy_regime_shares": policy_regime_shares,
    }
