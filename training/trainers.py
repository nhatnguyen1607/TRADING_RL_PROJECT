import copy
import time

import numpy as np
import torch

from agents.ac_agent import ACAgent
from agents.dqn_agent import DQNAgent
from .evaluation import evaluate_policy, evaluate_validation_bundle
from .metrics import validation_score


def teacher_weight_dataset_from_env(env, teacher_agent):
    if teacher_agent is None or not hasattr(env, "portfolios"):
        return np.asarray([]), np.asarray([])

    states = []
    target_distributions = []
    state = env.reset()
    done = False
    old_epsilon = getattr(teacher_agent, "epsilon", None)
    if old_epsilon is not None:
        teacher_agent.epsilon = 0.0

    while not done:
        teacher_action = teacher_agent.act(state)
        target = np.zeros(len(env.portfolios), dtype=np.float32)
        target[int(teacher_action)] = 1.0
        states.append(np.asarray(state, dtype=np.float32))
        target_distributions.append(target)
        state, _, done, _ = env.step(teacher_action)

    if old_epsilon is not None:
        teacher_agent.epsilon = old_epsilon

    return np.asarray(states, dtype=np.float32), np.asarray(target_distributions, dtype=np.float32)


def train_dqn(env, cfg, val_env=None, warm_start_rule=None):
    print(f"\nStarting DQN training ({cfg.encoder_type} + Double DQN)...")
    agent = DQNAgent(
        state_dim=env.observation_space.shape,
        action_dim=env.action_space.n,
        encoder_type=cfg.encoder_type,
        n_assets=getattr(env, "n_assets", 3),
        asset_feature_dim=cfg.asset_feature_dim,
        batch_size=cfg.dqn_batch_size,
        train_every=cfg.dqn_train_every,
    )
    if warm_start_rule is not None:
        print("DQN warm-start was requested, but multi-asset warm-start is intentionally skipped in this pipeline.")

    history = {"rewards": [], "loss": [], "epsilon": [], "episode_seconds": [], "validation_seconds": []}
    best_state = copy.deepcopy(agent.policy_net.state_dict())
    best_score = -np.inf

    for ep in range(cfg.dqn_episodes):
        start_time = time.perf_counter()
        state = env.reset()
        total_reward = 0.0
        ep_losses = []
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss > 0:
                ep_losses.append(loss)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        train_seconds = time.perf_counter() - start_time
        validation_seconds = 0.0
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["rewards"].append(total_reward)
        history["loss"].append(avg_loss)
        history["epsilon"].append(agent.epsilon)
        history["episode_seconds"].append(train_seconds)

        should_validate = val_env is not None and ((ep + 1) % cfg.validation_interval == 0 or ep == 0)
        if should_validate:
            val_start = time.perf_counter()
            (
                val_net,
                val_sharpe,
                val_sortino,
                val_dd,
                val_turnover,
                val_trades,
                val_action_changes,
                val_unique_actions,
            ) = evaluate_policy(val_env, agent, is_dqn=True)
            validation_seconds = time.perf_counter() - val_start
            score = validation_score(
                val_net,
                val_sharpe,
                val_sortino,
                val_dd,
                val_turnover,
                env.initial_balance,
                trade_count=val_trades,
                action_changes=val_action_changes,
                unique_actions=val_unique_actions,
            )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(agent.policy_net.state_dict())
            val_msg = (
                f" | ValNet: ${val_net:.0f} | ValSharpe: {val_sharpe:.2f}"
                f" | ValTrades: {val_trades} | ValActs: {val_action_changes}/{val_unique_actions}"
                f" | ValSec: {validation_seconds:.1f}"
            )
        else:
            val_msg = ""
        history["validation_seconds"].append(validation_seconds)
        print(
            f"DQN Ep {ep + 1}/{cfg.dqn_episodes} | Reward: {total_reward:.2f} | "
            f"Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f} | "
            f"NetWorth: ${info['net_worth']:.0f} | TrainSec: {train_seconds:.1f}{val_msg}"
        )

    if val_env is not None:
        agent.policy_net.load_state_dict(best_state)
    return agent, history


def train_ac(env, cfg, val_env=None, teacher_agent=None, warm_start_rule=None):
    print(f"\nStarting Actor-Critic training ({cfg.encoder_type} + teacher regularization)...")
    action_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    agent = ACAgent(
        state_dim=env.observation_space.shape,
        action_dim=action_dim,
        cash_logit_bias=getattr(env, "cash_logit_bias", 0.75),
        ac_temperature=getattr(env, "ac_temperature", 1.35),
        encoder_type=cfg.encoder_type,
        n_assets=getattr(env, "n_assets", 3),
        asset_feature_dim=cfg.asset_feature_dim,
    )
    if warm_start_rule is not None:
        print("AC warm-start was requested, but multi-asset warm-start is intentionally skipped in this pipeline.")
    if teacher_agent is not None and hasattr(env, "portfolios") and cfg.teacher_pretrain_epochs > 0:
        print(f"Pretraining Actor-Critic on DQN teacher trajectory ({cfg.teacher_pretrain_epochs} epochs)...")
        teacher_states, teacher_targets = teacher_weight_dataset_from_env(env, teacher_agent)
        pretrain_loss = agent.pretrain_from_teacher(
            teacher_states,
            teacher_targets,
            epochs=cfg.teacher_pretrain_epochs,
            batch_size=64,
        )
        print(f"AC teacher pretrain loss: {pretrain_loss:.4f}")

    history = {"rewards": [], "loss": [], "episode_seconds": [], "validation_seconds": []}
    best_state = copy.deepcopy(agent.model.state_dict())
    best_score = -np.inf
    best_ep = 0
    best_val_snapshot = None

    for ep in range(cfg.ac_episodes):
        start_time = time.perf_counter()
        state = env.reset()
        total_reward = 0.0
        ep_losses = []
        done = False
        step_idx = 0

        while not done:
            action, log_prob, dist = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if step_idx % cfg.ac_update_every == 0 or done:
                imitation_target = None
                if teacher_agent is not None and hasattr(env, "portfolios"):
                    old_epsilon = getattr(teacher_agent, "epsilon", None)
                    if old_epsilon is not None:
                        teacher_agent.epsilon = 0.0
                    teacher_action = teacher_agent.act(state)
                    if old_epsilon is not None:
                        teacher_agent.epsilon = old_epsilon
                    imitation_target = np.zeros(len(env.portfolios), dtype=np.float32)
                    imitation_target[int(teacher_action)] = 1.0

                loss = agent.train_step(
                    state,
                    log_prob,
                    dist,
                    reward,
                    next_state,
                    done,
                    imitation_target=imitation_target,
                )
                ep_losses.append(loss)
            state = next_state
            total_reward += reward
            step_idx += 1

        train_seconds = time.perf_counter() - start_time
        validation_seconds = 0.0
        avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        history["rewards"].append(total_reward)
        history["loss"].append(avg_loss)
        history["episode_seconds"].append(train_seconds)

        should_validate = val_env is not None and ((ep + 1) % cfg.validation_interval == 0 or ep == 0)
        if should_validate:
            val_start = time.perf_counter()
            (
                val_net,
                val_sharpe,
                val_sortino,
                val_dd,
                val_turnover,
                val_trades,
                val_action_changes,
                val_unique_actions,
            ) = evaluate_validation_bundle(val_env, agent, is_dqn=False)
            validation_seconds = time.perf_counter() - val_start
            score = validation_score(
                val_net,
                val_sharpe,
                val_sortino,
                val_dd,
                val_turnover,
                env.initial_balance,
                trade_count=val_trades,
                action_changes=val_action_changes,
                unique_actions=val_unique_actions,
            )
            if val_action_changes < 40:
                score -= 0.75
            if val_unique_actions < 4:
                score -= 0.75
            if val_action_changes < 20:
                score -= 1.00
            if val_unique_actions < 3:
                score -= 2.00
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(agent.model.state_dict())
                best_ep = ep + 1
                best_val_snapshot = (
                    val_net,
                    val_sharpe,
                    val_sortino,
                    val_dd,
                    val_turnover,
                    val_trades,
                    val_action_changes,
                    val_unique_actions,
                )
            val_msg = (
                f" | ValNet: ${val_net:.0f} | ValSharpe: {val_sharpe:.2f}"
                f" | ValTrades: {val_trades} | ValActs: {val_action_changes}/{val_unique_actions}"
                f" | ValSec: {validation_seconds:.1f}"
            )
        else:
            val_msg = ""
        history["validation_seconds"].append(validation_seconds)
        print(
            f"AC Ep {ep + 1}/{cfg.ac_episodes} | Reward: {total_reward:.2f} | "
            f"Loss: {avg_loss:.4f} | NetWorth: ${info['net_worth']:.0f} | "
            f"TrainSec: {train_seconds:.1f}{val_msg}"
        )

    if val_env is not None:
        agent.model.load_state_dict(best_state)
        if best_val_snapshot is not None:
            print(
                "AC best checkpoint "
                f"ep {best_ep} | ValNet: ${best_val_snapshot[0]:.0f} | ValSharpe: {best_val_snapshot[1]:.2f} "
                f"| ValTrades: {best_val_snapshot[5]} | ValActs: {best_val_snapshot[6]}/{best_val_snapshot[7]}"
            )
    return agent, history
