"""
collect_coins.py — Custom 2-agent PettingZoo ParallelEnv.

Environment idea (own multi-agent environment for project06):
- 2 agents move on a small grid to collect coins before time runs out.
- Discrete actions: 0 stay, 1 up, 2 down, 3 left, 4 right
- Observation: small continuous vector encoding positions + coin locations.

Designed to be fast for PPO/A2C training on GPU (small MLP, no rendering).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from gymnasium import spaces
from pettingzoo.utils.env import ParallelEnv


@dataclass(frozen=True)
class CollectCoinsConfig:
    grid_size: int = 9
    n_coins: int = 4
    max_steps: int = 65
    shared_reward: bool = True
    individual_bonus: float = 0.0
    step_penalty: float = -0.01
    distance_shaping: float = 0.02  # small bonus when agent moves closer to a coin
    fixed_walls: Tuple[Tuple[int, int], ...] = (
        (3, 3), (3, 4), (3, 5),
        (6, 3), (6, 4), (6, 5),
    )
    n_random_walls: int = 0
    seed: Optional[int] = None


class CollectCoinsParallelEnv(ParallelEnv):
    """
    Two-agent cooperative gridworld with coins.

    Agents:
      - agent_0
      - agent_1

    Rewards:
      - Collecting a coin yields +1.0 (shared between agents if shared_reward=True).
      - Optional individual bonus for the agent that collected the coin.
      - Optional per-step penalty.

    Episode ends when:
      - all coins collected, OR
      - max_steps reached (truncation).
    """

    metadata = {"name": "collect_coins_v0"}

    def __init__(self, cfg: CollectCoinsConfig = CollectCoinsConfig()):
        super().__init__()
        self.cfg = cfg

        self.possible_agents = ["agent_0", "agent_1"]
        self.agents: List[str] = []

        self._rng = np.random.default_rng(cfg.seed)

        # Spaces
        # Observation vector:
        #  - self pos (x,y) normalized
        #  - other pos (x,y) normalized
        #  - coin positions (n_coins * 2) normalized, padded with -1 for missing coins
        #  - step fraction (t / max_steps)
        #  - wall sensors: up/down/left/right (1 = free, -1 = blocked)
        obs_dim = 2 + 2 + (cfg.n_coins * 2) + 1 + 4
        self._obs_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        self._act_space = spaces.Discrete(5)

        self._step = 0
        self._positions: Dict[str, Tuple[int, int]] = {}
        self._coins: List[Tuple[int, int]] = []
        self._walls: FrozenSet[Tuple[int, int]] = frozenset()

    # PettingZoo API ---------------------------------------------------------

    def observation_space(self, agent: str):
        return self._obs_space

    def action_space(self, agent: str):
        return self._act_space

    def reset(self, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.agents = self.possible_agents[:]
        self._step = 0
        self._walls = self._build_walls()

        # Spawn agents at distinct positions.
        a0 = self._sample_empty_cell(exclude=set(self._walls))
        a1 = self._sample_empty_cell(exclude={a0, *self._walls})
        self._positions = {"agent_0": a0, "agent_1": a1}

        # Spawn coins (not on agents / walls, unique).
        exclude = {a0, a1, *self._walls}
        self._coins = []
        for _ in range(self.cfg.n_coins):
            c = self._sample_empty_cell(exclude=exclude)
            self._coins.append(c)
            exclude.add(c)

        obs = {a: self._obs(a) for a in self.agents}
        infos = {a: {} for a in self.agents}
        return obs, infos

    def step(self, actions: Dict[str, int]):
        self._step += 1

        # Apply actions (simultaneous move)
        intended = {}
        positions_before = dict(self._positions)
        for a in self.agents:
            act = int(actions.get(a, 0))
            intended[a] = self._move(self._positions[a], act)

        # Resolve collisions: if both want same cell, both stay.
        if intended["agent_0"] == intended["agent_1"]:
            intended["agent_0"] = self._positions["agent_0"]
            intended["agent_1"] = self._positions["agent_1"]

        self._positions = {"agent_0": intended["agent_0"], "agent_1": intended["agent_1"]}

        # Rewards
        rewards = {a: 0.0 for a in self.agents}

        if self.cfg.distance_shaping > 0:
            for a in self.agents:
                old_d = self._nearest_coin_dist(positions_before[a])
                new_d = self._nearest_coin_dist(self._positions[a])
                if new_d < old_d:
                    rewards[a] += float(self.cfg.distance_shaping) * float(old_d - new_d)

        # Coin collection
        collected_by: Dict[str, int] = {}
        remaining = []
        for c in self._coins:
            owner = None
            for a in self.agents:
                if self._positions[a] == c:
                    owner = a
                    break
            if owner is None:
                remaining.append(c)
            else:
                collected_by[owner] = collected_by.get(owner, 0) + 1
        self._coins = remaining

        total_collected = sum(collected_by.values())
        if total_collected > 0:
            if self.cfg.shared_reward:
                for a in self.agents:
                    rewards[a] += float(total_collected)
            else:
                for a, k in collected_by.items():
                    rewards[a] += float(k)
            if self.cfg.individual_bonus > 0:
                for a, k in collected_by.items():
                    rewards[a] += float(k) * float(self.cfg.individual_bonus)

        if self.cfg.step_penalty != 0.0:
            for a in self.agents:
                rewards[a] += float(self.cfg.step_penalty)

        terminated = {a: False for a in self.agents}
        truncation = {a: False for a in self.agents}

        all_coins_collected = len(self._coins) == 0
        if all_coins_collected:
            terminated = {a: True for a in self.agents}

        if self._step >= self.cfg.max_steps:
            truncation = {a: True for a in self.agents}

        infos = {a: {"coins_remaining": len(self._coins), "step": self._step} for a in self.agents}
        obs = {a: self._obs(a) for a in self.agents}

        # PettingZoo ParallelEnv expects agents list unchanged until done; keep it.
        if all_coins_collected or self._step >= self.cfg.max_steps:
            self.agents = []

        return obs, rewards, terminated, truncation, infos

    def close(self):
        return

    def get_state(self) -> dict:
        """Public snapshot for rendering / debugging."""
        return {
            "positions": dict(self._positions),
            "coins": list(self._coins),
            "step": self._step,
            "max_steps": self.cfg.max_steps,
            "grid_size": self.cfg.grid_size,
            "agents_active": list(self.agents),
            "walls": list(self._walls),
        }

    # Internals --------------------------------------------------------------

    def _build_walls(self) -> FrozenSet[Tuple[int, int]]:
        n = self.cfg.grid_size
        walls = {tuple(w) for w in self.cfg.fixed_walls if 0 <= w[0] < n and 0 <= w[1] < n}
        attempts = 0
        while len(walls) < len(self.cfg.fixed_walls) + self.cfg.n_random_walls and attempts < 500:
            attempts += 1
            x = int(self._rng.integers(0, n))
            y = int(self._rng.integers(0, n))
            # keep corners relatively open so agents can spawn
            if (x, y) in {(0, 0), (n - 1, 0), (0, n - 1), (n - 1, n - 1)}:
                continue
            walls.add((x, y))
        return frozenset(walls)

    def _sample_empty_cell(self, exclude: set) -> Tuple[int, int]:
        n = self.cfg.grid_size
        blocked = set(exclude) | set(self._walls)
        attempts = 0
        while attempts < 500:
            attempts += 1
            x = int(self._rng.integers(0, n))
            y = int(self._rng.integers(0, n))
            if (x, y) not in blocked:
                return (x, y)
        raise RuntimeError("Could not sample empty cell — grid may be over-constrained.")

    def _move(self, pos: Tuple[int, int], act: int) -> Tuple[int, int]:
        x, y = pos
        if act == 1:  # up
            y -= 1
        elif act == 2:  # down
            y += 1
        elif act == 3:  # left
            x -= 1
        elif act == 4:  # right
            x += 1
        n = self.cfg.grid_size
        x = int(np.clip(x, 0, n - 1))
        y = int(np.clip(y, 0, n - 1))
        if (x, y) in self._walls:
            return pos
        return (x, y)

    def _obs(self, agent: str) -> np.ndarray:
        n = float(self.cfg.grid_size - 1)
        ax, ay = self._positions[agent]
        other = "agent_1" if agent == "agent_0" else "agent_0"
        bx, by = self._positions[other]

        def norm_xy(x: int, y: int) -> Tuple[float, float]:
            if n <= 0:
                return 0.0, 0.0
            return (x / n) * 2.0 - 1.0, (y / n) * 2.0 - 1.0

        axy = norm_xy(ax, ay)
        bxy = norm_xy(bx, by)

        coin_vec: List[float] = []
        for (cx, cy) in self._coins[: self.cfg.n_coins]:
            nx, ny = norm_xy(cx, cy)
            coin_vec.extend([nx, ny])

        # Pad missing coins with -1
        while len(coin_vec) < self.cfg.n_coins * 2:
            coin_vec.extend([-1.0, -1.0])

        step_frac = float(self._step) / float(max(1, self.cfg.max_steps))
        step_norm = step_frac * 2.0 - 1.0
        sensors = self._wall_sensors(ax, ay)

        obs = np.array([*axy, *bxy, *coin_vec, step_norm, *sensors], dtype=np.float32)
        return obs

    def _nearest_coin_dist(self, pos: Tuple[int, int]) -> int:
        if not self._coins:
            return 0
        return min(abs(pos[0] - c[0]) + abs(pos[1] - c[1]) for c in self._coins)

    def _wall_sensors(self, x: int, y: int) -> List[float]:
        """1.0 if movement in direction is possible, -1.0 if blocked by wall or border."""
        n = self.cfg.grid_size
        dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right
        out: List[float] = []
        for dx, dy in dirs:
            nx, ny = x + dx, y + dy
            blocked = nx < 0 or ny < 0 or nx >= n or ny >= n or (nx, ny) in self._walls
            out.append(-1.0 if blocked else 1.0)
        return out

