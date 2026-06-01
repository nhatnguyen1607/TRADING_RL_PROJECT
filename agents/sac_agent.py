import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.ac_net import ActorCritic
from models.dqn_net import DQN


class DiscreteSACAgent:
    """Discrete Soft Actor-Critic over portfolio templates."""

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.995,
        alpha=0.05,
        batch_size=128,
        replay_size=50000,
        encoder_type="gru",
        n_assets=3,
        asset_feature_dim=None,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size

        self.actor = ActorCritic(
            state_dim,
            action_dim=action_dim,
            encoder_type=encoder_type,
            n_assets=n_assets,
            asset_feature_dim=asset_feature_dim,
        )
        self.q1 = DQN(state_dim, action_dim, encoder_type=encoder_type, n_assets=n_assets, asset_feature_dim=asset_feature_dim)
        self.q2 = DQN(state_dim, action_dim, encoder_type=encoder_type, n_assets=n_assets, asset_feature_dim=asset_feature_dim)
        self.target_q1 = DQN(state_dim, action_dim, encoder_type=encoder_type, n_assets=n_assets, asset_feature_dim=asset_feature_dim)
        self.target_q2 = DQN(state_dim, action_dim, encoder_type=encoder_type, n_assets=n_assets, asset_feature_dim=asset_feature_dim)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1e-4)
        self.q_optimizer = optim.AdamW(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr, weight_decay=1e-4)
        self.memory = deque(maxlen=replay_size)

    def act(self, state, deterministic=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            logits, _, _ = self.actor(state_tensor)
            probs = torch.softmax(logits, dim=-1)
            if deterministic:
                return int(torch.argmax(probs, dim=-1).item())
            return int(torch.distributions.Categorical(probs=probs).sample().item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, int(action), float(reward), next_state, bool(done)))

    def pretrain_from_teacher(self, states, target_distributions, epochs=3, batch_size=64):
        if states is None or len(states) == 0:
            return 0.0

        states = torch.FloatTensor(np.asarray(states, dtype=np.float32))
        targets = torch.FloatTensor(np.asarray(target_distributions, dtype=np.float32))
        dataset_size = states.size(0)
        losses = []

        for _ in range(epochs):
            permutation = torch.randperm(dataset_size)
            for start in range(0, dataset_size, batch_size):
                idx = permutation[start : start + batch_size]
                batch_states = states[idx]
                batch_targets = targets[idx]
                logits, _, _ = self.actor(batch_states)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = F.kl_div(log_probs, batch_targets, reduction="batchmean")

                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
                self.actor_optimizer.step()
                losses.append(float(loss.item()))

        return float(np.mean(losses)) if losses else 0.0

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)

        with torch.no_grad():
            next_logits, _, _ = self.actor(next_states)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log_softmax(next_logits, dim=-1)
            next_q = torch.min(self.target_q1(next_states), self.target_q2(next_states))
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = rewards + (1.0 - dones) * self.gamma * next_v

        q1_pred = self.q1(states).gather(1, actions)
        q2_pred = self.q2(states).gather(1, actions)
        q_loss = F.smooth_l1_loss(q1_pred, target_q) + F.smooth_l1_loss(q2_pred, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0)
        self.q_optimizer.step()

        logits, _, _ = self.actor(states)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        q_min = torch.min(self.q1(states), self.q2(states)).detach()
        actor_loss = (probs * (self.alpha * log_probs - q_min)).sum(dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        self.update_targets()
        return float((q_loss + actor_loss).item())

    def update_targets(self, tau=0.01):
        for target_param, local_param in zip(self.target_q1.parameters(), self.q1.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
        for target_param, local_param in zip(self.target_q2.parameters(), self.q2.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
