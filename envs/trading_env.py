import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TradingEnv(gym.Env):
    """Signed portfolio allocation environment for daily equity trading."""

    def __init__(
        self,
        df,
        window_size=20,
        initial_balance=10000.0,
        transaction_cost=0.001,
        is_discrete=True,
        max_weight_delta=0.08,
        max_total_allocation=0.70,
        cash_logit_bias=0.75,
        ac_temperature=1.35,
        ac_mix_power=1.0,
        concentration_penalty_coef=0.0,
        downside_penalty_coef=0.10,
        turnover_penalty_coef=0.0050,
        drawdown_penalty_coef=0.0030,
        reward_mode="heuristic",
        risk_aversion=3.0,
        cvar_alpha=0.95,
        tail_risk_coef=0.25,
        tail_window=60,
        target_portfolio_vol=None,
        vol_window=20,
    ):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.is_discrete = is_discrete
        self.feature_cols = df.attrs.get("feature_cols", [])
        self.max_exposure_delta = 0.08
        self.target_daily_vol = None
        self.regime_blend = 0.0
        self.rule_fallback_weight = 0.0
        self.reward_mode = reward_mode
        self.risk_aversion = risk_aversion
        self.cvar_alpha = cvar_alpha
        self.tail_risk_coef = tail_risk_coef
        self.tail_window = tail_window

        if not self.feature_cols:
            raise ValueError("TradingEnv requires df.attrs['feature_cols'] to be populated.")

        self.state_shape = (self.window_size, len(self.feature_cols) + 2)

        if self.is_discrete:
            self.discrete_allocations = [-0.75, -0.35, 0.0, 0.35, 0.75]
            self.action_space = spaces.Discrete(len(self.discrete_allocations))
        else:
            self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.state_shape, dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.done = False
        self.prev_allocation = 0.0
        self.cash_idle_steps = 0
        self.trade_count = 0
        self.return_history = []
        return self._get_obs()

    def _get_obs(self):
        obs_df = self.df[self.feature_cols].iloc[
            self.current_step - self.window_size : self.current_step
        ].values

        price = self.df["Close"].iloc[self.current_step]
        position_value = self.shares_held * price
        net_worth = max(self.net_worth, 1e-8)
        cash_ratio = self.balance / net_worth
        allocation = np.clip(position_value / net_worth, -1.5, 1.5)

        account_info = np.array([[cash_ratio, allocation]] * self.window_size)
        obs = np.hstack((obs_df, account_info))
        return obs.astype(np.float32)

    def _target_allocation(self, action):
        if self.is_discrete:
            target = self.discrete_allocations[int(action)]
        else:
            raw_action = float(np.asarray(action).reshape(-1)[0])
            target = float(np.clip(raw_action, -1.0, 1.0))

        regime_target = self.regime_target_exposure()
        target = (1.0 - self.regime_blend) * target + self.regime_blend * regime_target
        if self.rule_fallback_weight > 0 and abs(target) < 0.20:
            target = (1.0 - self.rule_fallback_weight) * target + self.rule_fallback_weight * regime_target
        capped_target = self._apply_regime_cap(target)
        delta = np.clip(capped_target - self.prev_allocation, -self.max_exposure_delta, self.max_exposure_delta)
        return float(self.prev_allocation + delta)

    def _apply_regime_cap(self, target_allocation):
        price = self.df["Close"].iloc[self.current_step]
        sma20 = self.df["SMA_20"].iloc[self.current_step]
        sma50 = self.df["SMA_50"].iloc[self.current_step]

        if price < sma20 < sma50:
            min_exposure, max_exposure = -1.0, 0.20
        elif price < sma50:
            min_exposure, max_exposure = -0.50, 0.50
        elif price < sma20:
            min_exposure, max_exposure = -0.25, 0.75
        else:
            min_exposure, max_exposure = 0.0, 1.0

        target = float(np.clip(target_allocation, min_exposure, max_exposure))
        return self._apply_volatility_target(target)

    def _apply_volatility_target(self, target_allocation):
        if self.target_daily_vol is None or "Raw_Volatility_20" not in self.df.columns:
            return target_allocation

        realized_vol = self.df["Raw_Volatility_20"].iloc[self.current_step]
        if not np.isfinite(realized_vol) or realized_vol <= 1e-8:
            return target_allocation

        vol_scale = np.clip(self.target_daily_vol / realized_vol, 0.25, 1.0)
        return float(target_allocation * vol_scale)

    def regime_target_exposure(self):
        price = self.df["Close"].iloc[self.current_step]
        sma20 = self.df["SMA_20"].iloc[self.current_step]
        sma50 = self.df["SMA_50"].iloc[self.current_step]

        if price < sma20 < sma50:
            return -0.45
        if price < sma50:
            return -0.20
        if price < sma20:
            return 0.0
        return 0.55

    def _regime_alignment_penalty(self, allocation):
        price = self.df["Close"].iloc[self.current_step]
        sma20 = self.df["SMA_20"].iloc[self.current_step]
        sma50 = self.df["SMA_50"].iloc[self.current_step]

        if price < sma20 < sma50:
            return 0.0015 * max(allocation, 0.0)
        if price < sma50:
            return 0.0008 * max(allocation - 0.10, 0.0)
        if price > sma20 > sma50:
            return 0.0008 * max(-allocation, 0.0)
        return 0.0

    def _tail_risk_penalty(self):
        if len(self.return_history) < max(10, self.tail_window // 3):
            return 0.0
        recent = np.asarray(self.return_history[-self.tail_window :], dtype=np.float64)
        losses = -recent
        var = np.quantile(losses, self.cvar_alpha)
        tail_losses = losses[losses >= var]
        return float(np.mean(tail_losses)) if len(tail_losses) else float(var)

    def _utility_reward(self, portfolio_return, turnover, drawdown, allocation):
        gamma = max(float(self.risk_aversion), 1e-6)
        clipped_return = float(np.clip(portfolio_return, -0.20, 0.20))
        utility_gain = (1.0 - np.exp(-gamma * clipped_return)) / gamma
        tail_penalty = self.tail_risk_coef * self._tail_risk_penalty()
        turnover_penalty = 0.0030 * turnover
        drawdown_penalty = 0.0015 * drawdown
        borrow_penalty = 0.00015 * abs(min(allocation, 0.0))
        return utility_gain - tail_penalty - turnover_penalty - drawdown_penalty - borrow_penalty

    def _rebalance_to(self, target_allocation, price):
        portfolio_value = self.balance + self.shares_held * price
        target_position_value = portfolio_value * target_allocation
        current_position_value = self.shares_held * price
        trade_value = target_position_value - current_position_value

        if abs(trade_value) < abs(portfolio_value) * 0.02:
            return 0.0

        self.trade_count += 1
        cost = abs(trade_value) * self.transaction_cost

        shares_delta = trade_value / price
        self.shares_held += shares_delta
        self.balance -= trade_value + cost

        return abs(trade_value) / max(portfolio_value, 1e-8)

    def step(self, action):
        price = self.df["Close"].iloc[self.current_step]
        prev_net_worth = self.net_worth

        target_allocation = self._target_allocation(action)
        turnover = self._rebalance_to(target_allocation, price)

        self.net_worth = self.balance + self.shares_held * price
        self.max_net_worth = max(self.max_net_worth, self.net_worth)
        realized_allocation = (self.shares_held * price) / max(self.net_worth, 1e-8)

        raw_return = (self.net_worth - prev_net_worth) / max(prev_net_worth, 1e-8)
        log_return = np.log(max(self.net_worth, 1e-8) / max(prev_net_worth, 1e-8))
        drawdown = (self.max_net_worth - self.net_worth) / max(self.max_net_worth, 1e-8)

        borrow_penalty = 0.00015 * abs(min(realized_allocation, 0.0))
        self.return_history.append(raw_return)
        if self.reward_mode == "utility_cvar":
            reward = self._utility_reward(raw_return, turnover, drawdown, realized_allocation) * 100.0
        else:
            downside_penalty = 0.15 * abs(min(raw_return, 0.0))
            turnover_penalty = 0.0025 * turnover
            drawdown_penalty = 0.0025 * drawdown
            leverage_penalty = 0.0005 * max(abs(realized_allocation) - 1.0, 0.0)
            regime_penalty = self._regime_alignment_penalty(realized_allocation)
            reward = (
                log_return
                - downside_penalty
                - turnover_penalty
                - drawdown_penalty
                - borrow_penalty
                - leverage_penalty
                - regime_penalty
            ) * 100.0
        reward = float(np.clip(reward, -5.0, 5.0))

        self.prev_allocation = realized_allocation
        self.current_step += 1
        if self.current_step >= len(self.df) - 1 or self.net_worth <= self.initial_balance * 0.2:
            self.done = True

        info = {
            "net_worth": self.net_worth,
            "portfolio_return": raw_return,
            "allocation": realized_allocation,
            "target_allocation": target_allocation,
            "turnover": turnover,
            "trade_count": self.trade_count,
        }
        return self._get_obs(), reward, self.done, info

    def render(self, mode="human"):
        print(
            f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, "
            f"Cash: {self.balance:.2f}, Shares: {self.shares_held:.4f}"
        )


class MultiAssetTradingEnv(gym.Env):
    """Long-only multi-asset allocation env with cash, SPY, inverse hedge, and bond hedge."""

    def __init__(
        self,
        df,
        window_size=20,
        initial_balance=10000.0,
        transaction_cost=0.001,
        is_discrete=True,
        max_weight_delta=0.08,
        max_total_allocation=0.70,
        cash_logit_bias=0.75,
        ac_temperature=1.35,
        ac_mix_power=1.0,
        concentration_penalty_coef=0.0,
        downside_penalty_coef=0.10,
        turnover_penalty_coef=0.0050,
        drawdown_penalty_coef=0.0030,
        reward_mode="heuristic",
        risk_aversion=3.0,
        cvar_alpha=0.95,
        tail_risk_coef=0.25,
        tail_window=60,
        target_portfolio_vol=None,
        vol_window=20,
        sharpe_window=60,
        sortino_weight=0.50,
        sharpe_reward_scale=0.08,
        return_reward_weight=0.35,
        regime_hedge_weight=0.0,
        macro_hedge_weight=0.0,
        options_hedge_weight=0.0,
        options_hedge_trigger=0.45,
        regime_ma_window=80,
        portfolio_templates=None,
    ):
        super(MultiAssetTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.is_discrete = is_discrete
        self.feature_cols = df.attrs.get("feature_cols", [])
        self.asset_cols = df.attrs.get("asset_cols", [])
        self.tickers = df.attrs.get("tickers", [col.replace("Close_", "") for col in self.asset_cols])
        self.max_weight_delta = max_weight_delta
        self.max_total_allocation = max_total_allocation
        self.cash_logit_bias = cash_logit_bias
        self.ac_temperature = ac_temperature
        self.ac_mix_power = ac_mix_power
        self.concentration_penalty_coef = concentration_penalty_coef
        self.downside_penalty_coef = downside_penalty_coef
        self.turnover_penalty_coef = turnover_penalty_coef
        self.drawdown_penalty_coef = drawdown_penalty_coef
        self.reward_mode = reward_mode
        self.risk_aversion = risk_aversion
        self.cvar_alpha = cvar_alpha
        self.tail_risk_coef = tail_risk_coef
        self.tail_window = tail_window
        self.target_portfolio_vol = target_portfolio_vol
        self.vol_window = vol_window
        self.sharpe_window = sharpe_window
        self.sortino_weight = sortino_weight
        self.sharpe_reward_scale = sharpe_reward_scale
        self.return_reward_weight = return_reward_weight
        self.regime_hedge_weight = regime_hedge_weight
        self.macro_hedge_weight = macro_hedge_weight
        self.options_hedge_weight = options_hedge_weight
        self.options_hedge_trigger = options_hedge_trigger
        self.regime_ma_window = regime_ma_window
        self.portfolio_templates = portfolio_templates

        if not self.feature_cols or not self.asset_cols:
            raise ValueError("MultiAssetTradingEnv requires feature_cols and asset_cols attrs.")

        self.n_assets = len(self.asset_cols)
        self.state_shape = (self.window_size, len(self.feature_cols) + self.n_assets + 1)
        self.portfolios = self._build_portfolios()

        if self.is_discrete:
            self.action_space = spaces.Discrete(len(self.portfolios))
        else:
            self.action_space = spaces.Box(low=-5, high=5, shape=(len(self.portfolios),), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.state_shape, dtype=np.float32)
        self.reset()

    def _build_portfolios(self):
        if self.portfolio_templates is not None:
            portfolios = [np.asarray(weights, dtype=np.float32) for weights in self.portfolio_templates]
            if any(weights.shape != (self.n_assets,) for weights in portfolios):
                raise ValueError("Each portfolio template must match the number of assets.")
            if not np.allclose(portfolios[0], np.zeros(self.n_assets)):
                raise ValueError("The first portfolio template must represent cash.")
            return portfolios

        portfolios = []
        portfolios.append(np.zeros(self.n_assets))  # cash
        for i in range(self.n_assets):
            w = np.zeros(self.n_assets)
            w[i] = self.max_total_allocation
            portfolios.append(w)

        if self.n_assets >= 3:
            portfolios.extend(
                [
                    np.array([0.45, 0.00, 0.25]),
                    np.array([0.20, 0.25, 0.25]),
                    np.array([0.00, 0.40, 0.30]),
                    np.array([0.30, 0.20, 0.20]),
                ]
            )
        return [p.astype(np.float32) for p in portfolios]

    def reset(self):
        self.current_step = self.window_size
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.cash_weight = 1.0
        self.done = False
        self.trade_count = 0
        self.return_history = []
        return self._get_obs()

    def _get_obs(self):
        obs_df = self.df[self.feature_cols].iloc[
            self.current_step - self.window_size : self.current_step
        ].values
        account_info = np.array([[self.cash_weight, *self.weights]] * self.window_size)
        obs = np.hstack((obs_df, account_info))
        return obs.astype(np.float32)

    def _target_weights(self, action):
        if self.is_discrete:
            weights = self.portfolios[int(action)].copy()
        else:
            logits = np.asarray(action, dtype=np.float64).reshape(-1)
            logits = np.clip(logits, -5, 5)
            logits = logits / self.ac_temperature
            logits[0] += self.cash_logit_bias #
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / np.sum(probs)
            probs = np.power(probs, self.ac_mix_power)
            probs = probs / np.sum(probs)
            portfolio_matrix = np.stack(self.portfolios, axis=0).astype(np.float64)
            weights = probs @ portfolio_matrix

        weights = np.clip(weights, 0.0, 1.0)
        total = float(np.sum(weights))
        if total > self.max_total_allocation:
            weights = weights / total * self.max_total_allocation #
        weights = self._apply_regime_hedge(weights)
        weights = self._apply_portfolio_vol_target(weights)
        return weights.astype(np.float32)

    def _apply_regime_hedge(self, weights):
        if (
            self.regime_hedge_weight <= 0.0
            and self.macro_hedge_weight <= 0.0
            and self.options_hedge_weight <= 0.0
        ) or self.n_assets < 3:
            return weights

        hedge = np.zeros_like(weights)
        hedge[1] = min(0.35, self.max_total_allocation * 0.58)
        hedge[2] = min(0.25, self.max_total_allocation * 0.42)

        blend = 0.0
        if self.regime_hedge_weight > 0.0 and "Close_SPY" in self.df.columns:
            signal_idx = self.current_step - 1
            start = max(0, signal_idx - self.regime_ma_window + 1)
            close = self.df["Close_SPY"].iloc[start : signal_idx + 1].astype(float)
            if len(close) >= max(20, self.regime_ma_window // 2):
                price = float(close.iloc[-1])
                ma = float(close.mean())
                momentum_20 = price / float(close.iloc[max(0, len(close) - 21)]) - 1.0 if len(close) > 20 else 0.0
                if price < ma and momentum_20 < 0.0:
                    blend = max(blend, self.regime_hedge_weight)

        if self.macro_hedge_weight > 0.0 and "Macro_Risk_Off_Raw" in self.df.columns:
            macro_risk = float(self.df["Macro_Risk_Off_Raw"].iloc[self.current_step])
            if np.isfinite(macro_risk) and macro_risk >= 0.50:
                blend = max(blend, self.macro_hedge_weight * macro_risk)

        if self.options_hedge_weight > 0.0 and "Options_RiskOff_Raw" in self.df.columns:
            signal_idx = max(0, self.current_step - 1)
            options_risk = float(self.df["Options_RiskOff_Raw"].iloc[signal_idx])
            trigger = self.options_hedge_trigger
            if np.isfinite(options_risk) and options_risk >= trigger:
                stress = np.clip((options_risk - trigger) / max(1.0 - trigger, 1e-8), 0.0, 1.0)
                blend = max(blend, self.options_hedge_weight * stress)

        if blend <= 0.0:
            return weights

        blend = min(max(blend, 0.0), 1.0)
        mixed = (1.0 - blend) * weights + blend * hedge
        total = float(np.sum(mixed))
        if total > self.max_total_allocation:
            mixed = mixed / total * self.max_total_allocation
        return mixed

    def _apply_portfolio_vol_target(self, weights):
        if self.target_portfolio_vol is None:
            return weights
        start = max(0, self.current_step - self.vol_window)
        if self.current_step - start < max(5, min(self.vol_window, 10)):
            return weights

        prices = self.df[self.asset_cols].iloc[start : self.current_step].astype(float)
        returns = prices.pct_change().dropna()
        if len(returns) < 5:
            return weights

        cov = returns.cov().values
        port_var = float(weights @ cov @ weights.T)
        if not np.isfinite(port_var) or port_var <= 1e-10:
            return weights
        port_vol = np.sqrt(port_var)
        scale = min(1.0, float(self.target_portfolio_vol) / (port_vol + 1e-8))
        return weights * scale

    def _smooth_target_weights(self, target_weights):
        delta = np.clip(target_weights - self.weights, -self.max_weight_delta, self.max_weight_delta) #
        smoothed = self.weights + delta
        smoothed = np.clip(smoothed, 0.0, 1.0)
        total = float(np.sum(smoothed))
        if total > self.max_total_allocation:
            smoothed = smoothed / total * self.max_total_allocation
        return smoothed.astype(np.float32)

    def _tail_risk_penalty(self):
        if len(self.return_history) < max(10, self.tail_window // 3):
            return 0.0
        recent = np.asarray(self.return_history[-self.tail_window :], dtype=np.float64)
        losses = -recent
        var = np.quantile(losses, self.cvar_alpha)
        tail_losses = losses[losses >= var]
        return float(np.mean(tail_losses)) if len(tail_losses) else float(var)

    def _utility_reward(self, portfolio_return, turnover, drawdown, max_asset_weight):
        gamma = max(float(self.risk_aversion), 1e-6)
        clipped_return = float(np.clip(portfolio_return, -0.20, 0.20))
        utility_gain = (1.0 - np.exp(-gamma * clipped_return)) / gamma
        tail_penalty = self.tail_risk_coef * self._tail_risk_penalty()
        turnover_penalty = self.turnover_penalty_coef * turnover
        concentration_penalty = self.concentration_penalty_coef * max(max_asset_weight - 0.45, 0.0)
        drawdown_penalty = self.drawdown_penalty_coef * drawdown
        return utility_gain - tail_penalty - turnover_penalty - drawdown_penalty - concentration_penalty

    def _rolling_sharpe_sortino_reward(self, portfolio_return, turnover, drawdown, max_asset_weight):
        recent = np.asarray(self.return_history[-self.sharpe_window :], dtype=np.float64)
        if len(recent) < max(10, self.sharpe_window // 4):
            risk_score = 0.0
        else:
            mean_return = float(np.mean(recent))
            vol = float(np.std(recent) + 1e-8)
            downside = recent[recent < 0.0]
            downside_vol = float(np.std(downside) + 1e-8) if len(downside) else vol
            sharpe = np.sqrt(252.0) * mean_return / vol
            sortino = np.sqrt(252.0) * mean_return / downside_vol
            risk_score = float(np.clip(sharpe + self.sortino_weight * sortino, -5.0, 5.0))

        concentration_penalty = self.concentration_penalty_coef * max(max_asset_weight - 0.45, 0.0)
        penalty = (
            self.turnover_penalty_coef * turnover
            + self.drawdown_penalty_coef * drawdown
            + self.downside_penalty_coef * abs(min(portfolio_return, 0.0))
            + concentration_penalty
        ) * 100.0
        return self.return_reward_weight * portfolio_return * 100.0 + self.sharpe_reward_scale * risk_score - penalty

    def step(self, action):
        prev_net_worth = self.net_worth
        prev_prices = self.df[self.asset_cols].iloc[self.current_step - 1].values.astype(np.float64)
        current_prices = self.df[self.asset_cols].iloc[self.current_step].values.astype(np.float64)
        asset_returns = current_prices / prev_prices - 1.0

        gross_return = float(np.dot(self.weights, asset_returns))
        self.net_worth *= 1.0 + gross_return

        target_weights = self._smooth_target_weights(self._target_weights(action))
        turnover = float(np.sum(np.abs(target_weights - self.weights)))
        if turnover > 0.01:
            self.trade_count += 1
        cost = self.transaction_cost * turnover
        self.net_worth *= max(0.0, 1.0 - cost)

        self.weights = target_weights
        self.cash_weight = max(0.0, 1.0 - float(np.sum(self.weights)))
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        portfolio_return = self.net_worth / max(prev_net_worth, 1e-8) - 1.0
        log_return = np.log(max(self.net_worth, 1e-8) / max(prev_net_worth, 1e-8)) #    
        drawdown = (self.max_net_worth - self.net_worth) / max(self.max_net_worth, 1e-8)
        max_asset_weight = float(np.max(self.weights)) if len(self.weights) else 0.0
        self.return_history.append(portfolio_return)
        if self.reward_mode == "utility_cvar":
            reward = self._utility_reward(portfolio_return, turnover, drawdown, max_asset_weight) * 100.0
        elif self.reward_mode == "sharpe_sortino":
            reward = self._rolling_sharpe_sortino_reward(portfolio_return, turnover, drawdown, max_asset_weight)
        else:
            downside_penalty = self.downside_penalty_coef * abs(min(portfolio_return, 0.0))
            turnover_penalty = self.turnover_penalty_coef * turnover
            drawdown_penalty = self.drawdown_penalty_coef * drawdown
            concentration_penalty = self.concentration_penalty_coef * max(max_asset_weight - 0.45, 0.0) #
            reward = (
                log_return
                - downside_penalty
                - turnover_penalty
                - drawdown_penalty
                - concentration_penalty
            ) * 100.0
        reward = float(np.clip(reward, -5.0, 5.0))

        self.current_step += 1
        if self.current_step >= len(self.df) - 1 or self.net_worth <= self.initial_balance * 0.2:
            self.done = True

        info = {
            "net_worth": self.net_worth,
            "portfolio_return": portfolio_return,
            "allocation": float(np.sum(self.weights)),
            "target_allocation": float(np.sum(target_weights)),
            "weights": self.weights.copy(),
            "cash_weight": self.cash_weight,
            "turnover": turnover,
            "trade_count": self.trade_count,
            "max_asset_weight": max_asset_weight,
        }
        return self._get_obs(), reward, self.done, info
