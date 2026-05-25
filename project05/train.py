"""
train.py — Train SAC on BipedalWalker-v3.

Usage:
    # Train all configs from scratch:
    python train.py

    # Train a single config (useful for debugging or reruns):
    python train.py --hp hp_default --arch arch_large

    # Resume training a specific run:
    python train.py --hp hp_default --arch arch_large --seed 3 --resume

    # Override total timesteps:
    python train.py --total-timesteps 300000
"""

import argparse
import os
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config import (
    ARCHITECTURES,
    BEST_MODEL_THRESHOLD,
    DEVICE,
    ENV_ID,
    EVAL_EPISODES,
    EVAL_FREQ,
    HYPERPARAMS,
    MODELS_DIR,
    N_SEEDS,
    RESULTS_DIR,
    TOTAL_TIMESTEPS,
)
from utils import (
    Timer,
    config_id,
    load_metadata,
    make_dirs,
    results_path,
    run_id,
    save_eval_row,
    save_metadata,
)


# ── Callback ──────────────────────────────────────────────────────────────────

class TrainingLogger(BaseCallback):
    """
    Saves evaluation results to CSV and prints progress.
    Wraps SB3's EvalCallback output via on_step polling.
    """

    def __init__(
        self,
        csv_path: str,
        eval_env: gym.Env,
        eval_freq: int,
        n_eval_episodes: int,
        timer: Timer,
        checkpoint_path: str,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.timer = timer
        self.checkpoint_path = checkpoint_path
        self._last_eval_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_step >= self.eval_freq:
            self._last_eval_step = self.num_timesteps
            self._run_eval()
        # Checkpoint every 50k steps — atomic save
        if self.num_timesteps % 50_000 == 0 and self.num_timesteps > 0:
            tmp = self.checkpoint_path + ".tmp"
            self.model.save(tmp)
            os.replace(tmp, self.checkpoint_path)
        return True

    def _run_eval(self):
        from stable_baselines3.common.evaluation import evaluate_policy

        mean_r, std_r = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=False,   # stochastic during training eval
            warn=False,
        )
        elapsed = self.timer.elapsed()
        save_eval_row(self.csv_path, self.num_timesteps, mean_r, std_r, elapsed)

        if self.verbose:
            print(
                f"  step {self.num_timesteps:>8,}  |  "
                f"mean_reward={mean_r:+.2f}  std={std_r:.2f}  |  "
                f"elapsed={elapsed:.0f}s"
            )


# ── Benchmark step/episode timing ─────────────────────────────────────────────

def benchmark_env(n_steps: int = 500) -> dict:
    """Measure time per step and per episode."""
    env = gym.make(ENV_ID)
    obs, _ = env.reset()
    t0 = time.perf_counter()
    steps = 0
    episodes = 0
    episode_steps = 0

    while steps < n_steps:
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1
        episode_steps += 1
        if terminated or truncated:
            obs, _ = env.reset()
            episodes += 1
            episode_steps = 0

    elapsed = time.perf_counter() - t0
    env.close()

    return {
        "n_steps": n_steps,
        "elapsed_sec": elapsed,
        "ms_per_step": elapsed / n_steps * 1000,
        "steps_per_sec": n_steps / elapsed,
        "episodes_completed": episodes,
    }


# ── Single run ────────────────────────────────────────────────────────────────

def train_single(
    hp_name: str,
    arch_name: str,
    seed: int,
    total_timesteps: int,
    resume: bool = False,
) -> None:
    make_dirs(RESULTS_DIR, MODELS_DIR)

    rid = run_id(hp_name, arch_name, seed)
    csv_path = results_path(RESULTS_DIR, hp_name, arch_name, seed)
    model_path = os.path.join(MODELS_DIR, f"{rid}.zip")
    meta_path = os.path.join(MODELS_DIR, f"{rid}_meta.json")

    # ── Resume guard ─────────────────────────────────────────────────────────
    steps_done = 0
    if resume and os.path.exists(model_path):
        if os.path.exists(meta_path):
            meta = load_metadata(meta_path)
            steps_done = meta.get("timesteps_trained", 0)
        remaining = total_timesteps - steps_done
        if remaining <= 0:
            print(f"[{rid}] Already complete ({steps_done} steps). Skipping.")
            return
        print(f"[{rid}] Resuming from {steps_done} steps, {remaining} remaining.")
    elif not resume and os.path.exists(csv_path):
        print(f"[{rid}] Results already exist. Use --resume to continue. Skipping.")
        return

    # ── Environments ─────────────────────────────────────────────────────────
    train_env = Monitor(gym.make(ENV_ID))
    eval_env  = Monitor(gym.make(ENV_ID))

    # ── Model ─────────────────────────────────────────────────────────────────
    hp = HYPERPARAMS[hp_name].copy()
    arch = ARCHITECTURES[arch_name].copy()
    timer = Timer()

    if resume and os.path.exists(model_path):
        print(f"[{rid}] Loading model from {model_path}")
        model = SAC.load(
            model_path,
            env=train_env,
            device=DEVICE,
        )
        remaining_steps = total_timesteps - steps_done
    else:
        model = SAC(
            policy="MlpPolicy",
            env=train_env,
            policy_kwargs=arch,
            device=DEVICE,
            seed=seed,
            verbose=0,
            **hp,
        )
        remaining_steps = total_timesteps

    # ── Callback ──────────────────────────────────────────────────────────────
    callback = TrainingLogger(
        csv_path=csv_path,
        eval_env=eval_env,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
        timer=timer,
        checkpoint_path=model_path,
        verbose=1,
    )

    print(f"\n{'='*60}")
    print(f"  Run : {rid}")
    print(f"  Device : {DEVICE}")
    print(f"  Steps  : {remaining_steps:,}")
    print(f"{'='*60}")

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=remaining_steps,
        callback=callback,
        reset_num_timesteps=not resume,
        progress_bar=True,
    )

    # ── Save (atomic: write to tmp then rename to avoid corruption on interrupt)
    tmp_path = model_path + ".tmp"
    model.save(tmp_path)
    os.replace(tmp_path, model_path)   # atomic on Linux
    save_metadata(meta_path, {
        "run_id": rid,
        "hp_name": hp_name,
        "arch_name": arch_name,
        "seed": seed,
        "timesteps_trained": steps_done + remaining_steps,
        "hyperparams": hp,
        "architecture": {k: str(v) for k, v in arch.items()},
        "device": DEVICE,
        "env_id": ENV_ID,
        "elapsed_sec": timer.elapsed(),
    })

    train_env.close()
    eval_env.close()
    print(f"[{rid}] Done. Saved to {model_path}")


# ── Full sweep ────────────────────────────────────────────────────────────────

def train_all(total_timesteps: int, resume: bool = False) -> None:
    """Train all hyperparameter × architecture × seed combinations."""
    combos = [
        (hp_name, arch_name, seed)
        for hp_name in HYPERPARAMS
        for arch_name in ARCHITECTURES
        for seed in range(N_SEEDS)
    ]
    total = len(combos)
    print(f"Total runs: {total}  ({len(HYPERPARAMS)} hp × {len(ARCHITECTURES)} arch × {N_SEEDS} seeds)")

    for i, (hp_name, arch_name, seed) in enumerate(combos, 1):
        print(f"\n[{i}/{total}] {hp_name} / {arch_name} / seed={seed}")
        train_single(hp_name, arch_name, seed, total_timesteps, resume=resume)


# ── Benchmark ─────────────────────────────────────────────────────────────────

def run_benchmark() -> None:
    print("Benchmarking environment step time...")
    stats = benchmark_env(n_steps=2000)
    print(f"  Steps/sec    : {stats['steps_per_sec']:.1f}")
    print(f"  ms/step      : {stats['ms_per_step']:.3f}")
    print(f"  Episodes done: {stats['episodes_completed']}")
    make_dirs(MODELS_DIR)
    save_metadata(os.path.join(MODELS_DIR, "benchmark.json"), stats)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train SAC on BipedalWalker-v3")
    p.add_argument("--hp",   default=None, choices=list(HYPERPARAMS.keys()),
                   help="Single hyperparameter set (default: all)")
    p.add_argument("--arch", default=None, choices=list(ARCHITECTURES.keys()),
                   help="Single architecture (default: all)")
    p.add_argument("--seed", type=int, default=None,
                   help="Single seed (default: all 0..N_SEEDS-1)")
    p.add_argument("--resume", action="store_true",
                   help="Continue training from saved checkpoint")
    p.add_argument("--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
                   dest="total_timesteps")
    p.add_argument("--benchmark", action="store_true",
                   help="Only run environment benchmark and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.benchmark:
        run_benchmark()
    elif args.hp and args.arch and args.seed is not None:
        train_single(args.hp, args.arch, args.seed, args.total_timesteps, resume=args.resume)
    elif args.hp and args.arch:
        for seed in range(N_SEEDS):
            train_single(args.hp, args.arch, seed, args.total_timesteps, resume=args.resume)
    else:
        train_all(args.total_timesteps, resume=args.resume)
