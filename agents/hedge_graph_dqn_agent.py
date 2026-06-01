import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_net import DQN


class DynamicHedgeGraphDQNAgent:
    """Graph-encoded DQN with a transparent regime-sensitive hedge preference."""

    def __init__(
        self,
        state_dim,
        action_dim,
        portfolios,
        lr=3e-4,
        gamma=0.995,
        epsilon_start=1.0,
        epsilon_end=0.03,
        epsilon_decay=0.985,
        n_assets=3,
        asset_feature_dim=18,
        feature_indices=None,
        batch_size=64,
        train_every=4,
        prior_strength=1.25,
        risk_off_threshold=0.76,
        risk_on_threshold=0.38,
        hard_mask_strength=1.50,
        transition_de_risk_strength=0.80,
        recovery_strength=0.80,
        use_regime_action_shield=True,
        shield_penalty_strength=0.75,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.n_assets = n_assets
        self.asset_feature_dim = asset_feature_dim
        self.feature_indices = feature_indices or {}
        self.batch_size = batch_size
        self.train_every = train_every
        self.prior_strength = prior_strength
        self.risk_off_threshold = risk_off_threshold
        self.risk_on_threshold = risk_on_threshold
        self.hard_mask_strength = hard_mask_strength
        self.transition_de_risk_strength = transition_de_risk_strength
        self.recovery_strength = recovery_strength
        self.use_regime_action_shield = use_regime_action_shield
        self.shield_penalty_strength = shield_penalty_strength
        self.learn_step = 0
        self.portfolios = torch.FloatTensor(np.asarray(portfolios, dtype=np.float32))

        self.policy_net = DQN(
            state_dim,
            action_dim,
            encoder_type="graph_transformer",
            n_assets=n_assets,
            asset_feature_dim=asset_feature_dim,
        )
        self.target_net = DQN(
            state_dim,
            action_dim,
            encoder_type="graph_transformer",
            n_assets=n_assets,
            asset_feature_dim=asset_feature_dim,
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=20000)

    def _regime_signals(self, states):
        latest = states[:, -1, :]
        vix = latest[:, self.feature_indices["vix"]]
        momentum = latest[:, self.feature_indices["momentum"]]
        trend = latest[:, self.feature_indices["trend"]]

        return_indices = [i * self.asset_feature_dim for i in range(self.n_assets)]
        returns = states[:, :, return_indices]
        centered = returns - returns.mean(dim=1, keepdim=True)
        cov = torch.einsum("bta,btm->bam", centered, centered) / max(states.size(1) - 1, 1)
        std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8))
        corr = cov / (std.unsqueeze(-1) * std.unsqueeze(-2) + 1e-8)
        spy_tlt_corr = corr[:, 0, 2].clamp(-1.0, 1.0)

        stress_logit = 0.75 * vix - 1.10 * momentum - 0.55 * trend + 0.85 * spy_tlt_corr
        stress = torch.sigmoid(stress_logit)
        return stress, spy_tlt_corr, momentum, trend

    def _regime_state(self, states):
        stress, spy_tlt_corr, momentum, trend = self._regime_signals(states)
        bearish = (momentum < 0.0) | (trend < 0.0) | (spy_tlt_corr > 0.15)
        bullish = (momentum > 0.0) & (trend > 0.0)
        risk_off = (stress >= self.risk_off_threshold) & bearish
        risk_on = (stress <= self.risk_on_threshold) & bullish
        transition = ~(risk_off | risk_on)
        return stress, spy_tlt_corr, risk_on, transition, risk_off

    def _adjust_scores(self, q_values, states):
        stress, spy_tlt_corr, risk_on, transition, risk_off = self._regime_state(states)
        portfolios = self.portfolios.to(q_values.device)
        spy_weight = portfolios[:, 0]
        sh_weight = portfolios[:, 1]
        tlt_weight = portfolios[:, 2]
        cash_weight = 1.0 - portfolios.sum(dim=1)
        total_weight = portfolios.sum(dim=1)
        diversification = torch.where(
            total_weight > 0.01,
            1.0 - torch.max(portfolios, dim=1).values / total_weight.clamp_min(0.01),
            torch.zeros_like(total_weight),
        )
        lost_bond_hedge = torch.relu(spy_tlt_corr).unsqueeze(1) * tlt_weight.unsqueeze(0)
        risk_off_preference = (
            1.60 * sh_weight.unsqueeze(0)
            + 0.65 * cash_weight.unsqueeze(0)
            - 1.05 * spy_weight.unsqueeze(0)
            - 0.70 * lost_bond_hedge
        )
        transition_preference = (
            0.35 * cash_weight.unsqueeze(0)
            + 0.75 * diversification.unsqueeze(0)
            + 0.30 * sh_weight.unsqueeze(0)
            + 0.20 * tlt_weight.unsqueeze(0)
            - 0.45 * torch.relu(total_weight - 0.55).unsqueeze(0)
            - 0.45 * spy_weight.unsqueeze(0)
            - 0.40 * lost_bond_hedge
        )
        risk_on_preference = (
            0.50 * spy_weight.unsqueeze(0)
            + 0.40 * tlt_weight.unsqueeze(0)
            + 0.35 * diversification.unsqueeze(0)
            - 0.25 * cash_weight.unsqueeze(0)
            - 0.75 * sh_weight.unsqueeze(0)
        )
        q_scale = q_values.detach().std(dim=1, keepdim=True).clamp_min(0.10)
        adjusted = q_values
        adjusted = adjusted + risk_off.unsqueeze(1).float() * self.prior_strength * q_scale * risk_off_preference
        adjusted = adjusted + transition.unsqueeze(1).float() * self.transition_de_risk_strength * q_scale * transition_preference
        adjusted = adjusted + risk_on.unsqueeze(1).float() * self.recovery_strength * q_scale * risk_on_preference

        stressed = risk_off.unsqueeze(1)
        exposed = (spy_weight.unsqueeze(0) >= 0.60) & (sh_weight.unsqueeze(0) < 0.05)
        adjusted = adjusted - stressed.float() * exposed.float() * self.hard_mask_strength * q_scale
        if self.use_regime_action_shield:
            risk_on_allowed = (spy_weight >= sh_weight) & (sh_weight <= 0.12)
            transition_allowed = (spy_weight <= 0.25) & (sh_weight <= 0.22) & (total_weight <= 0.40)
            risk_off_allowed = ((spy_weight <= 0.05) & (sh_weight >= 0.18)) | (total_weight <= 0.01)
            allowed = (
                risk_on.unsqueeze(1) & risk_on_allowed.unsqueeze(0)
                | transition.unsqueeze(1) & transition_allowed.unsqueeze(0)
                | risk_off.unsqueeze(1) & risk_off_allowed.unsqueeze(0)
            )
            shield_penalty = self.shield_penalty_strength * q_scale
            adjusted = adjusted - (~allowed).float() * shield_penalty
        return adjusted

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        was_training = self.policy_net.training
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            scores = self._adjust_scores(self.policy_net(state_tensor), state_tensor)
        if was_training:
            self.policy_net.train()
        return int(torch.argmax(scores, dim=1).item())

    def diagnostics_for_state(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            stress, spy_tlt_corr, risk_on, transition, risk_off = self._regime_state(state_tensor)
        if bool(risk_off.item()):
            regime_state = "Risk-Off"
        elif bool(risk_on.item()):
            regime_state = "Risk-On"
        else:
            regime_state = "Transition"
        return {
            "Policy_Stress_Score": float(stress.item()),
            "Policy_SPY_TLT_Correlation": float(spy_tlt_corr.item()),
            "Policy_Regime_State": regime_state,
        }

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, int(action), float(reward), next_state, bool(done)))

    def train_step(self):
        self.learn_step += 1
        if self.learn_step % self.train_every != 0 or len(self.memory) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, self.batch_size))
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            self.policy_net.eval()
            next_policy_q = self.policy_net(next_states)
            next_actions = self._adjust_scores(next_policy_q, next_states).argmax(dim=1, keepdim=True)
            self.policy_net.train()
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.update_target_network()
        return float(loss.item())

    def update_target_network(self, tau=0.01):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
