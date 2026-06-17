"""
wrappers.py — Convert multi-agent ParallelEnv into SB3-friendly Gymnasium envs.

We implement a simple "joint control" wrapper for the shared-policy experiment:
- single-agent observation = concat(obs_agent0, obs_agent1)
- single-agent action = MultiDiscrete([5, 5]) (one action per agent)
- reward = mean reward across agents (they're shared anyway)
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from project06.envs.collect_coins import CollectCoinsParallelEnv

PartnerPolicy = Callable[[str, np.ndarray], int]


class JointControlGymEnv(gym.Env):
    """
    Wrap a 2-agent PettingZoo ParallelEnv into a single-agent Gymnasium Env.

    This is the easiest way to train PPO/A2C with Stable-Baselines3 without
    multi-agent algorithms: one policy controls both agents simultaneously.
    """

    metadata = {"render_modes": []}

    def __init__(self, env: CollectCoinsParallelEnv):
        super().__init__()
        self.penv = env

        # assume fixed two agents
        self._agents = ["agent_0", "agent_1"]

        o0 = self.penv.observation_space("agent_0")
        o1 = self.penv.observation_space("agent_1")
        if o0.shape != o1.shape:
            raise ValueError("This wrapper assumes equal obs shapes for both agents.")

        self.single_obs_dim = int(np.prod(o0.shape))
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.single_obs_dim * 2,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete([self.penv.action_space(a).n for a in self._agents])

        self._last_obs: Dict[str, np.ndarray] | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.penv.reset(seed=seed, options=options)
        self._last_obs = obs
        joint_obs = self._concat_obs(obs)
        return joint_obs, {"per_agent": info}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(-1)
        if action.shape[0] != 2:
            raise ValueError("Expected 2 actions (agent_0, agent_1).")
        actions = {"agent_0": int(action[0]), "agent_1": int(action[1])}

        obs, rewards, terminated, truncated, infos = self.penv.step(actions)
        self._last_obs = obs

        joint_obs = self._concat_obs(obs) if obs else np.zeros(self.observation_space.shape, dtype=np.float32)

        # shared environment -> average rewards
        r = float(np.mean([rewards.get(a, 0.0) for a in self._agents]))
        term = bool(any(terminated.get(a, False) for a in self._agents))
        trunc = bool(any(truncated.get(a, False) for a in self._agents))

        info = {
            "per_agent": infos,
            "coins_remaining": infos.get("agent_0", {}).get("coins_remaining", None),
        }
        return joint_obs, r, term, trunc, info

    def _concat_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        o0 = np.asarray(obs["agent_0"], dtype=np.float32).reshape(-1)
        o1 = np.asarray(obs["agent_1"], dtype=np.float32).reshape(-1)
        return np.concatenate([o0, o1], dtype=np.float32)

    def close(self):
        self.penv.close()


class RandomPartnerPolicy:
    def __init__(self, n_actions: int, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def __call__(self, agent: str, obs: np.ndarray) -> int:
        return int(self._rng.integers(0, self.n_actions))


class SingleAgentGymEnv(gym.Env):
    """
    Train one agent while the partner follows a fixed/random policy.
    Used for the mixed-algorithm experiment (PPO vs A2C in one episode).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env: CollectCoinsParallelEnv,
        controlled_agent: str,
        partner_policy: Optional[PartnerPolicy] = None,
        partner_seed: int = 0,
    ):
        super().__init__()
        if controlled_agent not in {"agent_0", "agent_1"}:
            raise ValueError("controlled_agent must be agent_0 or agent_1")

        self.penv = env
        self.controlled = controlled_agent
        self.partner = "agent_1" if controlled_agent == "agent_0" else "agent_0"
        self._agents = ["agent_0", "agent_1"]

        obs_space = self.penv.observation_space(self.controlled)
        act_space = self.penv.action_space(self.controlled)
        self.observation_space = spaces.Box(
            low=obs_space.low,
            high=obs_space.high,
            shape=obs_space.shape,
            dtype=np.float32,
        )
        self.action_space = act_space

        if partner_policy is None:
            partner_policy = RandomPartnerPolicy(act_space.n, seed=partner_seed)
        self.partner_policy = partner_policy
        self._last_obs: Dict[str, np.ndarray] | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.penv.reset(seed=seed, options=options)
        self._last_obs = obs
        return np.asarray(obs[self.controlled], dtype=np.float32), {"per_agent": info}

    def step(self, action):
        if self._last_obs is None:
            raise RuntimeError("Environment must be reset before step().")

        actions = {
            self.controlled: int(action),
            self.partner: int(self.partner_policy(self.partner, self._last_obs[self.partner])),
        }
        obs, rewards, terminated, truncated, infos = self.penv.step(actions)
        self._last_obs = obs if obs else self._last_obs

        o = np.asarray(obs.get(self.controlled, np.zeros(self.observation_space.shape)), dtype=np.float32)
        r = float(np.mean([rewards.get(a, 0.0) for a in self._agents]))
        term = bool(any(terminated.get(a, False) for a in self._agents))
        trunc = bool(any(truncated.get(a, False) for a in self._agents))
        return o, r, term, trunc, {"per_agent": infos}

    def close(self):
        self.penv.close()


class ModelPartnerPolicy:
    """Use a trained SB3 model as the partner policy."""

    def __init__(self, model, agent_name: str):
        self.model = model
        self.agent_name = agent_name

    def __call__(self, agent: str, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

