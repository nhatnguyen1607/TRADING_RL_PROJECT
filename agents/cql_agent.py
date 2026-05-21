import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.dqn_net import DQN


class CQLAgent:
    """
    Conservative Q-Learning for discrete portfolio templates.

    This is an offline-RL research baseline: it penalizes unseen/high-Q actions through
    logsumexp(Q) - Q(dataset action), reducing overestimation under distribution shift.
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.995,
        cql_alpha=0.50,
        encoder_type="graph_transformer",
        n_assets=3,
        asset_feature_dim=18,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.cql_alpha = cql_alpha
        self.policy_net = DQN(
            state_dim,
            action_dim,
            encoder_type=encoder_type,
            n_assets=n_assets,
            asset_feature_dim=asset_feature_dim,
        )
        self.target_net = DQN(
            state_dim,
            action_dim,
            encoder_type=encoder_type,
            n_assets=n_assets,
            asset_feature_dim=asset_feature_dim,
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.td_loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=100000)
        self.batch_size = 128

    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        return int(torch.argmax(q_values, dim=-1).item())

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, int(action), float(reward), next_state, bool(done)))

    def fit_offline(self, transitions, epochs=20, batch_size=128):
        self.memory.clear()
        self.memory.extend(transitions)
        self.batch_size = batch_size
        losses = []
        for _ in range(epochs):
            steps = max(1, len(self.memory) // batch_size)
            for _ in range(steps):
                loss = self.train_step()
                if loss > 0:
                    losses.append(loss)
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

        q_all = self.policy_net(states)
        q_dataset = q_all.gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * next_q

        td_loss = self.td_loss_fn(q_dataset, target_q)
        conservative_loss = torch.logsumexp(q_all, dim=1, keepdim=True).mean() - q_dataset.mean()
        loss = td_loss + self.cql_alpha * conservative_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.update_target_network()
        return float(loss.item())

    def update_target_network(self, tau=0.01):
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
