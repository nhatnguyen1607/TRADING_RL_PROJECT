import numpy as np
import torch
import torch.optim as optim

from models.ac_net import ActorCritic


class ACAgent:
    def __init__(self, state_dim, action_dim=1, lr=3e-4, gamma=0.995, gae_lambda=0.95, clip_ratio=0.20):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.cash_logit_bias = 0.75
        self.ac_temperature = 1.35
        self.model = ActorCritic(state_dim, action_dim=action_dim)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def act(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, dist = self.model.get_action(state_tensor, deterministic=deterministic)
        return action.detach().numpy()[0], log_prob, dist

    def train_step(self, state, log_prob, dist, reward, next_state, done, imitation_target=None):
        state_tensor = torch.FloatTensor(np.asarray(state)).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(np.asarray(next_state)).unsqueeze(0)

        mean, _, value = self.model(state_tensor)
        with torch.no_grad():
            _, _, next_value = self.model(next_state_tensor)

        td_target = torch.as_tensor([[reward]], dtype=torch.float32) + self.gamma * next_value * (1 - int(done))
        advantage = td_target - value

        critic_loss = advantage.pow(2).mean()
        actor_loss = -(log_prob * advantage.detach()).mean()
        entropy = dist.entropy().mean()
        imitation_loss = torch.tensor(0.0)
        if imitation_target is not None:
            target_arr = np.asarray(imitation_target, dtype=np.float32).reshape(-1)
            if target_arr.size == mean.shape[-1]:
                target_tensor = torch.as_tensor(target_arr.reshape(1, -1), dtype=torch.float32)
                logits = mean / self.ac_temperature
                logits = logits.clone()
                logits[:, 0] += self.cash_logit_bias
                pred_weights = torch.softmax(logits, dim=-1)
                imitation_loss = (pred_weights - target_tensor).pow(2).mean()
            else:
                target_tensor = torch.as_tensor([[float(target_arr[0])]], dtype=torch.float32)
                imitation_loss = (mean - target_tensor).pow(2).mean()

        total_loss = actor_loss + 0.7 * critic_loss + 0.05 * imitation_loss - 0.001 * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()

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
