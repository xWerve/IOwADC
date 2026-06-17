"""
train_shared.py — Same-algorithm experiment with parameter sharing.

Train: single-agent env (agent_0, random partner) — easier than joint MultiDiscrete.
Eval:  both agents use the same policy on their own observations.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict

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
from project06.eval_helpers import evaluate_shared_team
from project06.utils import (
    Timer,
    load_model,
    make_dirs,
    meta_path,
    model_path,
    results_path,
    run_id,
    save_eval_row,
    save_metadata,
)
from project06.wrappers import SingleAgentGymEnv


class TeamEvalLogger(BaseCallback):
    def __init__(self, csv_path: str, eval_freq: int, n_eval_episodes: int, timer: Timer, ckpt_path: str, seed: int):
        super().__init__(verbose=1)
        self.csv_path = csv_path
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.timer = timer
        self.ckpt_path = ckpt_path
        self.seed = seed
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval >= self.eval_freq:
            self._last_eval = self.num_timesteps
            mean_r, std_r = evaluate_shared_team(
                self.model, self.n_eval_episodes, self.seed + self.num_timesteps
            )
            save_eval_row(self.csv_path, self.num_timesteps, mean_r, std_r, self.timer.elapsed())
            print(f"  step {self.num_timesteps:>8,} | team mean={mean_r:+.3f} std={std_r:.3f} | elapsed={self.timer.elapsed():.0f}s")

        if self.num_timesteps % 50_000 == 0 and self.num_timesteps > 0:
            tmp = self.ckpt_path + ".tmp"
            self.model.save(tmp)
            os.replace(tmp, self.ckpt_path)
        return True


def make_train_env(seed: int):
    penv = CollectCoinsParallelEnv(cfg=ENV_CFG)
    env = SingleAgentGymEnv(penv, controlled_agent="agent_0", partner_seed=seed + 1)
    env.reset(seed=seed)
    return env


def train_one(algo: str, variant: str, seed: int, total_timesteps: int, resume: bool = False):
    make_dirs(RESULTS_DIR, MODELS_DIR)

    rid = run_id("shared", algo, variant, "joint", seed)
    csv = results_path(RESULTS_DIR, rid)
    mpath = model_path(MODELS_DIR, rid)
    jmeta = meta_path(MODELS_DIR, rid)

    env = make_train_env(seed)
    timer = Timer()

    if algo == "ppo":
        hp = PPO_VARIANTS[variant].copy()
        cls = PPO
    elif algo == "a2c":
        hp = A2C_VARIANTS[variant].copy()
        cls = A2C
    else:
        raise ValueError("algo must be ppo or a2c")

    if resume and os.path.exists(mpath):
        model = load_model(cls, mpath, env=env, device=DEVICE)
        reset_num_timesteps = False
    else:
        model = cls(
            policy="MlpPolicy",
            env=env,
            device=DEVICE,
            seed=seed,
            verbose=0,
            policy_kwargs={"net_arch": [128, 128]},
            **hp,
        )
        reset_num_timesteps = True

    cb = TeamEvalLogger(csv, EVAL_FREQ, EVAL_EPISODES, timer, mpath, seed)
    print(f"\n=== {rid} ===")
    print(f"device={DEVICE} timesteps={total_timesteps:,} env={ENV_CFG}")
    model.learn(total_timesteps=total_timesteps, callback=cb, progress_bar=True, reset_num_timesteps=reset_num_timesteps)

    tmp = mpath + ".tmp"
    model.save(tmp)
    os.replace(tmp, mpath)

    mean_r, std_r = evaluate_shared_team(model, EVAL_EPISODES, seed, deterministic=True)
    save_eval_row(csv, total_timesteps, mean_r, std_r, timer.elapsed())

    save_metadata(jmeta, {
        "run_id": rid,
        "experiment": "shared",
        "algo": algo,
        "variant": variant,
        "seed": seed,
        "device": DEVICE,
        "env_cfg": asdict(ENV_CFG),
        "timesteps_trained": int(total_timesteps),
        "hyperparams": hp,
        "final_deterministic_mean": mean_r,
        "elapsed_sec": timer.elapsed(),
    })

    env.close()
    print(f"  final deterministic team mean={mean_r:+.3f} std={std_r:.3f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "a2c"], default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS, dest="total_timesteps")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    algos = [args.algo] if args.algo else ["ppo", "a2c"]
    for algo in algos:
        variants = (
            [args.variant] if args.variant else
            (list(PPO_VARIANTS.keys()) if algo == "ppo" else list(A2C_VARIANTS.keys()))
        )
        for variant in variants:
            seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))
            for seed in seeds:
                train_one(algo, variant, seed, args.total_timesteps, resume=args.resume)
