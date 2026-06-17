"""Shared evaluation helpers for multi-agent rollouts."""

from __future__ import annotations

import numpy as np

from project06.config import ENV_CFG
from project06.envs import CollectCoinsParallelEnv


def evaluate_shared_team(model, n_episodes: int, seed: int, deterministic: bool = False) -> tuple[float, float]:
    """Both agents use the same policy (parameter sharing — same algorithm)."""
    penv = CollectCoinsParallelEnv(cfg=ENV_CFG)
    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        obs, _ = penv.reset(seed=seed + ep)
        ep_reward = 0.0
        while penv.agents:
            a0, _ = model.predict(obs["agent_0"], deterministic=deterministic)
            a1, _ = model.predict(obs["agent_1"], deterministic=deterministic)
            obs, rewards, terminated, truncated, _ = penv.step(
                {"agent_0": int(a0), "agent_1": int(a1)}
            )
            ep_reward += float(np.mean(list(rewards.values())))
            if any(terminated.values()) or any(truncated.values()):
                break
        episode_rewards.append(ep_reward)

    penv.close()
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


def evaluate_mixed_team(ppo_model, a2c_model, n_episodes: int, seed: int, deterministic: bool = False) -> tuple[float, float]:
    """agent_0 = PPO, agent_1 = A2C in the same episode."""
    penv = CollectCoinsParallelEnv(cfg=ENV_CFG)
    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        obs, _ = penv.reset(seed=seed + ep)
        ep_reward = 0.0
        while penv.agents:
            a0, _ = ppo_model.predict(obs["agent_0"], deterministic=deterministic)
            a1, _ = a2c_model.predict(obs["agent_1"], deterministic=deterministic)
            obs, rewards, terminated, truncated, _ = penv.step(
                {"agent_0": int(a0), "agent_1": int(a1)}
            )
            ep_reward += float(np.mean(list(rewards.values())))
            if any(terminated.values()) or any(truncated.values()):
                break
        episode_rewards.append(ep_reward)

    penv.close()
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))
