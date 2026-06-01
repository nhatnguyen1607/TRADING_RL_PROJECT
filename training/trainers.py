import copy
import time

import numpy as np
import torch

from agents.ac_agent import ACAgent
from agents.dqn_agent import DQNAgent
from agents.hedge_graph_dqn_agent import DynamicHedgeGraphDQNAgent
from agents.sac_agent import DiscreteSACAgent
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
                val_dominant_action,
                val_dominant_share,
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
            if val_dominant_share > cfg.dqn_dominance_threshold:
                score -= 1.75 * (val_dominant_share - cfg.dqn_dominance_threshold) / max(
                    1.0 - cfg.dqn_dominance_threshold,
                    1e-8,
                )
            if score > best_score:
                best_score = score
                best_state = copy.deepcopy(agent.policy_net.state_dict())
            val_msg = (
                f" | ValNet: ${val_net:.0f} | ValSharpe: {val_sharpe:.2f}"
                f" | ValTrades: {val_trades} | ValActs: {val_action_changes}/{val_unique_actions}"
                f" | ValTop: {val_dominant_action}:{val_dominant_share:.0%}"
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


def train_hedge_graph_dqn(env, cfg, val_env=None):
    feature_cols = list(getattr(env, "feature_cols", []))
    required_features = {
        "vix": "SPY_VIX",
        "momentum": "SPY_Momentum_20",
        "trend": "SPY_Trend_Regime",
    }
    feature_indices = {}
    for signal, name in required_features.items():
        if name not in feature_cols:
            raise ValueError(f"Dynamic Hedge-Graph DQN requires state feature {name}.")
        feature_indices[signal] = feature_cols.index(name)

    print("\nStarting Dynamic Hedge-Graph DQN training (graph_transformer + hedge preference)...")
    agent = DynamicHedgeGraphDQNAgent(
        state_dim=env.observation_space.shape,
        action_dim=env.action_space.n,
        portfolios=env.portfolios,
        n_assets=getattr(env, "n_assets", 3),
        asset_feature_dim=cfg.asset_feature_dim,
        feature_indices=feature_indices,
        batch_size=cfg.dqn_batch_size,
        train_every=cfg.dqn_train_every,
        prior_strength=cfg.hedge_prior_strength,
        risk_off_threshold=cfg.hedge_risk_off_threshold,
        risk_on_threshold=cfg.hedge_risk_on_threshold,
        hard_mask_strength=cfg.hedge_hard_mask_strength,
        transition_de_risk_strength=cfg.hedge_transition_de_risk_strength,
        recovery_strength=cfg.hedge_recovery_strength,
        use_regime_action_shield=cfg.hedge_use_regime_action_shield,
        shield_penalty_strength=cfg.hedge_shield_penalty_strength,
    )
    history = {"rewards": [], "loss": [], "epsilon": [], "episode_seconds": [], "validation_seconds": []}
    best_state = copy.deepcopy(agent.policy_net.state_dict())
    best_score = -np.inf

    for ep in range(cfg.hedge_graph_episodes):
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
                val_dominant_action,
                val_dominant_share,
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
                f" | ValTop: {val_dominant_action}:{val_dominant_share:.0%}"
                f" | ValSec: {validation_seconds:.1f}"
            )
        else:
            val_msg = ""
        history["validation_seconds"].append(validation_seconds)
        print(
            f"HEDGE-GRAPH DQN Ep {ep + 1}/{cfg.hedge_graph_episodes} | Reward: {total_reward:.2f} | "
            f"Loss: {avg_loss:.4f} | Eps: {agent.epsilon:.2f} | "
            f"NetWorth: ${info['net_worth']:.0f} | TrainSec: {train_seconds:.1f}{val_msg}"
        )

    if val_env is not None:
        agent.policy_net.load_state_dict(best_state)
    return agent, history


def train_ac(env, cfg, val_env=None, teacher_agent=None, warm_start_rule=None):
    print(f"\nStarting Actor-Critic training ({cfg.encoder_type} + {cfg.ac_algorithm.upper()} + teacher pretrain)...")
    action_dim = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    agent = ACAgent(
        state_dim=env.observation_space.shape,
        action_dim=action_dim,
        cash_logit_bias=getattr(env, "cash_logit_bias", 0.75),
        ac_temperature=getattr(env, "ac_temperature", 1.35),
        encoder_type=cfg.encoder_type,
        n_assets=getattr(env, "n_assets", 3),
        asset_feature_dim=cfg.asset_feature_dim,
        critic_target_mode=cfg.critic_target_mode,
        ez_risk_aversion=cfg.ez_risk_aversion,
        ez_eis=cfg.ez_eis,
        imitation_coef=cfg.ac_imitation_coef,
        entropy_coef=cfg.ac_entropy_coef,
    )
    if warm_start_rule is not None:
        print("AC warm-start was requested, but multi-asset warm-start is intentionally skipped in this pipeline.")
    if teacher_agent is not None and hasattr(env, "portfolios") and cfg.teacher_pretrain_epochs > 0:
        print(f"Pretraining Actor-Critic on DQN teacher trajectory ({cfg.teacher_pretrain_epochs} epochs)...")
        teacher_states, teacher_targets = teacher_weight_dataset_from_env(env, teacher_agent)
        if len(teacher_targets):
            uniform = np.full_like(teacher_targets, 1.0 / teacher_targets.shape[1])
            teacher_targets = (1.0 - cfg.ac_teacher_smoothing) * teacher_targets + cfg.ac_teacher_smoothing * uniform
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
        states = []
        actions = []
        rewards = []
        dones = []
        old_log_probs = []

        while not done:
            action, log_prob, dist = agent.act(state)
            next_state, reward, done, info = env.step(action)
            if cfg.ac_algorithm == "ppo":
                states.append(np.asarray(state, dtype=np.float32))
                actions.append(int(action) if np.isscalar(action) or np.asarray(action).size == 1 else action)
                rewards.append(float(reward))
                dones.append(bool(done))
                old_log_probs.append(float(log_prob.detach().cpu().item()))
            elif step_idx % cfg.ac_update_every == 0 or done:
                imitation_target = None
                if cfg.ac_online_teacher and teacher_agent is not None and hasattr(env, "portfolios"):
                    old_epsilon = getattr(teacher_agent, "epsilon", None)
                    if old_epsilon is not None:
                        teacher_agent.epsilon = 0.0
                    teacher_action = teacher_agent.act(state)
                    if old_epsilon is not None:
                        teacher_agent.epsilon = old_epsilon
                    imitation_target = np.zeros(len(env.portfolios), dtype=np.float32)
                    imitation_target[int(teacher_action)] = 1.0
                    uniform = np.full_like(imitation_target, 1.0 / len(imitation_target))
                    imitation_target = (
                        (1.0 - cfg.ac_teacher_smoothing) * imitation_target
                        + cfg.ac_teacher_smoothing * uniform
                    )

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

        if cfg.ac_algorithm == "ppo" and states:
            loss = agent.train_trajectory(
                states,
                actions,
                rewards,
                dones,
                old_log_probs,
                ppo_epochs=cfg.ppo_epochs,
                minibatch_size=cfg.ppo_minibatch_size,
            )
            ep_losses.append(loss)

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
                val_dominant_action,
                val_dominant_share,
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
            if val_dominant_share > cfg.ac_dominance_threshold:
                score -= 2.50 * (val_dominant_share - cfg.ac_dominance_threshold) / max(1.0 - cfg.ac_dominance_threshold, 1e-8)
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
                    val_dominant_action,
                    val_dominant_share,
                )
            val_msg = (
                f" | ValNet: ${val_net:.0f} | ValSharpe: {val_sharpe:.2f}"
                f" | ValTrades: {val_trades} | ValActs: {val_action_changes}/{val_unique_actions}"
                f" | ValTop: {val_dominant_action}:{val_dominant_share:.0%}"
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
                f"| ValTrades: {best_val_snapshot[5]} | ValActs: {best_val_snapshot[6]}/{best_val_snapshot[7]} "
                f"| ValTop: {best_val_snapshot[8]}:{best_val_snapshot[9]:.0%}"
            )
    return agent, history


def train_sac(env, cfg, val_env=None, teacher_agent=None):
    print(f"\nStarting Discrete SAC training ({cfg.encoder_type} + categorical policy)...")
    agent = DiscreteSACAgent(
        state_dim=env.observation_space.shape,
        action_dim=env.action_space.n,
        alpha=cfg.sac_alpha,
        batch_size=cfg.sac_batch_size,
        encoder_type=cfg.encoder_type,
        n_assets=getattr(env, "n_assets", 3),
        asset_feature_dim=cfg.asset_feature_dim,
    )
    if teacher_agent is not None and hasattr(env, "portfolios") and cfg.sac_teacher_pretrain_epochs > 0:
        print(f"Pretraining Discrete SAC actor on DQN teacher trajectory ({cfg.sac_teacher_pretrain_epochs} epochs)...")
        teacher_states, teacher_targets = teacher_weight_dataset_from_env(env, teacher_agent)
        if len(teacher_targets):
            uniform = np.full_like(teacher_targets, 1.0 / teacher_targets.shape[1])
            teacher_targets = (1.0 - cfg.sac_teacher_smoothing) * teacher_targets + cfg.sac_teacher_smoothing * uniform
        pretrain_loss = agent.pretrain_from_teacher(
            teacher_states,
            teacher_targets,
            epochs=cfg.sac_teacher_pretrain_epochs,
            batch_size=64,
        )
        print(f"SAC teacher pretrain loss: {pretrain_loss:.4f}")

    history = {"rewards": [], "loss": [], "episode_seconds": [], "validation_seconds": []}
    best_actor_state = copy.deepcopy(agent.actor.state_dict())
    best_q1_state = copy.deepcopy(agent.q1.state_dict())
    best_q2_state = copy.deepcopy(agent.q2.state_dict())
    best_score = -np.inf

    for ep in range(cfg.sac_episodes):
        start_time = time.perf_counter()
        state = env.reset()
        total_reward = 0.0
        losses = []
        done = False

        while not done:
            action = agent.act(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss:
                losses.append(loss)
            state = next_state
            total_reward += reward

        train_seconds = time.perf_counter() - start_time
        validation_seconds = 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
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
                val_dominant_action,
                val_dominant_share,
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
            if val_dominant_share > 0.45:
                score -= 1.0 * (val_dominant_share - 0.45) / 0.55
            if score > best_score:
                best_score = score
                best_actor_state = copy.deepcopy(agent.actor.state_dict())
                best_q1_state = copy.deepcopy(agent.q1.state_dict())
                best_q2_state = copy.deepcopy(agent.q2.state_dict())
            val_msg = (
                f" | ValNet: ${val_net:.0f} | ValSharpe: {val_sharpe:.2f}"
                f" | ValTrades: {val_trades} | ValActs: {val_action_changes}/{val_unique_actions}"
                f" | ValTop: {val_dominant_action}:{val_dominant_share:.0%}"
                f" | ValSec: {validation_seconds:.1f}"
            )
        else:
            val_msg = ""
        history["validation_seconds"].append(validation_seconds)
        print(
            f"SAC Ep {ep + 1}/{cfg.sac_episodes} | Reward: {total_reward:.2f} | "
            f"Loss: {avg_loss:.4f} | NetWorth: ${info['net_worth']:.0f} | "
            f"TrainSec: {train_seconds:.1f}{val_msg}"
        )

    if val_env is not None:
        agent.actor.load_state_dict(best_actor_state)
        agent.q1.load_state_dict(best_q1_state)
        agent.q2.load_state_dict(best_q2_state)
    return agent, history
