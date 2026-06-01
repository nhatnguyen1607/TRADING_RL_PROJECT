import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.ac_net import ActorCritic


class ACAgent:
    def __init__(
        self,
        state_dim,
        action_dim=1,
        lr=3e-4,
        gamma=0.995,
        gae_lambda=0.95,
        clip_ratio=0.20,
        cash_logit_bias=0.75,
        ac_temperature=1.35,
        encoder_type="gru",
        n_assets=3,
        asset_feature_dim=None,
        critic_target_mode="td",
        ez_risk_aversion=6.0,
        ez_eis=1.5,
        imitation_coef=0.05,
        entropy_coef=0.0020,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.cash_logit_bias = cash_logit_bias
        self.ac_temperature = ac_temperature
        self.critic_target_mode = critic_target_mode
        self.ez_risk_aversion = ez_risk_aversion
        self.ez_eis = ez_eis
        self.imitation_coef = imitation_coef
        self.entropy_coef = entropy_coef
        self.model = ActorCritic(
            state_dim,
            action_dim=action_dim,
            encoder_type=encoder_type,
            n_assets=n_assets,
            asset_feature_dim=asset_feature_dim,
        )
        self.is_discrete = action_dim > 1
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def _critic_target(self, reward, next_value, done):
        reward_tensor = torch.as_tensor([[reward]], dtype=torch.float32)
        if self.critic_target_mode != "epstein_zin":
            return reward_tensor + self.gamma * next_value * (1 - int(done))

        gamma = max(float(self.ez_risk_aversion), 1e-6)
        psi = max(float(self.ez_eis), 1e-6)
        rho = 1.0 - 1.0 / psi
        alpha = 1.0 - gamma

        reward_scaled = torch.clamp(reward_tensor / 100.0, min=-0.95, max=0.95)
        consumption_utility = torch.sign(reward_scaled) * torch.log1p(torch.abs(reward_scaled))
        continuation = torch.sign(next_value) * torch.pow(torch.abs(next_value) + 1e-6, alpha)
        certainty_equivalent = torch.sign(continuation) * torch.pow(torch.abs(continuation) + 1e-6, rho / alpha)
        recursive_value = (1.0 - self.gamma) * consumption_utility + self.gamma * certainty_equivalent * (1 - int(done))
        return torch.sign(recursive_value) * torch.pow(torch.abs(recursive_value) + 1e-6, 1.0 / max(rho, 1e-6)) * 100.0

    def act(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, dist = self.model.get_action(state_tensor, deterministic=deterministic)
        action_np = action.detach().cpu().numpy()
        if self.is_discrete:
            return int(action_np.reshape(-1)[0]), log_prob, dist
        return action_np[0], log_prob, dist

    def train_step(self, state, log_prob, dist, reward, next_state, done, imitation_target=None):
        state_tensor = torch.FloatTensor(np.asarray(state)).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(np.asarray(next_state)).unsqueeze(0)

        actor_output, _, value = self.model(state_tensor)
        with torch.no_grad():
            _, _, next_value = self.model(next_state_tensor)

        td_target = self._critic_target(reward, next_value, done)
        advantage = td_target - value

        critic_loss = advantage.pow(2).mean()
        actor_loss = -(log_prob * advantage.detach()).mean()
        entropy = dist.entropy().mean()
        imitation_loss = torch.tensor(0.0)
        if imitation_target is not None:
            target_arr = np.asarray(imitation_target, dtype=np.float32).reshape(-1)
            if target_arr.size == actor_output.shape[-1]:
                target_tensor = torch.as_tensor(target_arr.reshape(1, -1), dtype=torch.float32)
                logits = actor_output
                log_probs = torch.log_softmax(logits, dim=-1)
                imitation_loss = -(target_tensor * log_probs).sum(dim=-1).mean()
            else:
                target_tensor = torch.as_tensor([[float(target_arr[0])]], dtype=torch.float32)
                imitation_loss = (actor_output - target_tensor).pow(2).mean()

        total_loss = actor_loss + 0.7 * critic_loss + self.imitation_coef * imitation_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()

    def pretrain_from_teacher(self, states, target_weight_vectors, epochs=6, batch_size=128):
        if len(states) == 0:
            return 0.0

        states_tensor = torch.FloatTensor(np.asarray(states))
        targets_tensor = torch.FloatTensor(np.asarray(target_weight_vectors))
        indices = np.arange(len(states))
        losses = []

        for _ in range(epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = torch.LongTensor(indices[start : start + batch_size])
                logits, _, _ = self.model(states_tensor[batch_idx])
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = -(targets_tensor[batch_idx] * log_probs).sum(dim=-1).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0

    def train_trajectory(self, states, actions, rewards, dones, old_log_probs, ppo_epochs=4, minibatch_size=128):
        states_tensor = torch.FloatTensor(np.asarray(states))
        actions_tensor = torch.FloatTensor(np.asarray(actions)).view(-1, 1)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards))
        dones_tensor = torch.FloatTensor(np.asarray(dones, dtype=np.float32))
        old_log_probs_tensor = torch.FloatTensor(np.asarray(old_log_probs))

        with torch.no_grad():
            _, _, values = self.model(states_tensor)
            values = values.squeeze(-1)

        advantages = torch.zeros_like(rewards_tensor)
        last_gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones_tensor[t]
            delta = rewards_tensor[t] + self.gamma * next_value * mask - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * mask * last_gae
            advantages[t] = last_gae
            next_value = values[t]

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = len(states)
        indices = np.arange(n)
        losses = []
        for _ in range(ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, minibatch_size):
                batch_idx = indices[start : start + minibatch_size]
                batch_idx = torch.LongTensor(batch_idx)

                log_probs, entropy, value_pred = self.model.evaluate_actions(
                    states_tensor[batch_idx], actions_tensor[batch_idx]
                )
                ratio = torch.exp(log_probs - old_log_probs_tensor[batch_idx])
                adv = advantages[batch_idx]
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
                actor_loss = -torch.min(unclipped, clipped).mean()
                critic_loss = (returns[batch_idx] - value_pred).pow(2).mean()
                entropy_bonus = entropy.mean()
                total_loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy_bonus

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                losses.append(total_loss.item())

        return float(np.mean(losses)) if losses else 0.0
