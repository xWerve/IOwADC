"""
train_mixed.py — Mixed-algorithm experiment.

Trains:
  - PPO policy for agent_0 (partner = random)
  - A2C policy for agent_1 (partner = random)

Then logs learning curve for the mixed team (PPO controls agent_0, A2C controls agent_1)
in the same episode.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback

from project06.config import (
    A2C_VARIANTS,
    DEVICE,
    ENV_CFG,
    EVAL_EPISODES,
    EVAL_FREQ,
    MODELS_DIR,
    N_SEEDS,
    PPO_VARIANTS,
    RESULTS_DIR,
    TOTAL_TIMESTEPS,
)
from project06.envs import CollectCoinsParallelEnv
from project06.utils import (
    Timer,
    make_dirs,
    meta_path,
    model_path,
    results_path,
    run_id,
    save_eval_row,
    save_metadata,
)
from project06.wrappers import SingleAgentGymEnv


from project06.eval_helpers import evaluate_mixed_team
class MixedEvalLogger(BaseCallback):
    def __init__(
        self,
        csv_path: str,
        ppo_model,
        eval_freq: int,
        n_eval_episodes: int,
        timer: Timer,
        seed: int,
    ):
        super().__init__(verbose=1)
        self.csv_path = csv_path
        self.ppo_model = ppo_model
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.timer = timer
        self.seed = seed
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True

        self._last_eval = self.num_timesteps
        mean_r, std_r = evaluate_mixed_team(
            self.ppo_model, self.model, self.n_eval_episodes, self.seed + self.num_timesteps
        )
        save_eval_row(self.csv_path, self.num_timesteps, mean_r, std_r, self.timer.elapsed())
        print(
            f"  [mixed] step {self.num_timesteps:>8,} | mean={mean_r:+.3f} std={std_r:.3f} "
            f"| elapsed={self.timer.elapsed():.0f}s"
        )
        return True


def make_single_env(controlled: str, seed: int):
    penv = CollectCoinsParallelEnv(cfg=ENV_CFG)
    env = SingleAgentGymEnv(penv, controlled_agent=controlled, partner_seed=seed + 1)
    env.reset(seed=seed)
    return env


def train_mixed_one(seed: int, total_timesteps: int, ppo_variant: str = "v0", a2c_variant: str = "v0"):
    make_dirs(RESULTS_DIR, MODELS_DIR)
    timer = Timer()

    ppo_rid = run_id("mixed", "ppo", ppo_variant, "agent_0", seed)
    a2c_rid = run_id("mixed", "a2c", a2c_variant, "agent_1", seed)
    joint_rid = run_id("mixed", "ppo+a2c", ppo_variant, "joint", seed)

    ppo_path = model_path(MODELS_DIR, ppo_rid)
    a2c_path = model_path(MODELS_DIR, a2c_rid)
    joint_csv = results_path(RESULTS_DIR, joint_rid)

    # --- Train PPO for agent_0 ---
    print(f"\n=== {ppo_rid} ===")
    ppo_env = make_single_env("agent_0", seed)
    ppo_model = PPO(
        policy="MlpPolicy",
        env=ppo_env,
        device=DEVICE,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
        **PPO_VARIANTS[ppo_variant],
    )
    ppo_steps = max(total_timesteps // 2, EVAL_FREQ)
    ppo_model.learn(total_timesteps=ppo_steps, progress_bar=True)
    tmp = ppo_path + ".tmp"
    ppo_model.save(tmp)
    os.replace(tmp, ppo_path)
    save_metadata(meta_path(MODELS_DIR, ppo_rid), {
        "run_id": ppo_rid,
        "experiment": "mixed",
        "algo": "ppo",
        "controlled_agent": "agent_0",
        "seed": seed,
        "device": DEVICE,
        "env_cfg": asdict(ENV_CFG),
        "timesteps_trained": ppo_steps,
        "hyperparams": PPO_VARIANTS[ppo_variant],
    })
    ppo_env.close()

    # --- Train A2C for agent_1 with mixed eval logging ---
    print(f"\n=== {a2c_rid} ===")
    a2c_env = make_single_env("agent_1", seed + 100)
    a2c_model = A2C(
        policy="MlpPolicy",
        env=a2c_env,
        device=DEVICE,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [128, 128]},
        **A2C_VARIANTS[a2c_variant],
    )
    a2c_steps = max(total_timesteps // 2, EVAL_FREQ)
    mixed_cb = MixedEvalLogger(
        csv_path=joint_csv,
        ppo_model=ppo_model,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        timer=timer,
        seed=seed,
    )
    a2c_model.learn(total_timesteps=a2c_steps, callback=mixed_cb, progress_bar=True)
    tmp = a2c_path + ".tmp"
    a2c_model.save(tmp)
    os.replace(tmp, a2c_path)
    save_metadata(meta_path(MODELS_DIR, a2c_rid), {
        "run_id": a2c_rid,
        "experiment": "mixed",
        "algo": "a2c",
        "controlled_agent": "agent_1",
        "seed": seed,
        "device": DEVICE,
        "env_cfg": asdict(ENV_CFG),
        "timesteps_trained": a2c_steps,
        "hyperparams": A2C_VARIANTS[a2c_variant],
    })
    a2c_env.close()

    # Final mixed evaluation point (reuse in-memory models)
    mean_r, std_r = evaluate_mixed_team(ppo_model, a2c_model, EVAL_EPISODES, seed)
    save_eval_row(joint_csv, a2c_steps, mean_r, std_r, timer.elapsed())
    print(f"  [mixed final] mean={mean_r:+.3f} std={std_r:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS, dest="total_timesteps")
    p.add_argument("--ppo-variant", default="v0")
    p.add_argument("--a2c-variant", default="v0")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))
    for seed in seeds:
        train_mixed_one(seed, args.total_timesteps, args.ppo_variant, args.a2c_variant)
