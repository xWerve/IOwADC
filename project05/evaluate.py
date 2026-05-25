"""
evaluate.py — Deterministic evaluation of saved SAC agents.

Usage:
    # Evaluate all saved models:
    python evaluate.py

    # Evaluate a specific run:
    python evaluate.py --hp hp_default --arch arch_large --seed 3

    # Render one episode (requires display):
    python evaluate.py --hp hp_default --arch arch_large --seed 3 --render

    # Override number of eval episodes:
    python evaluate.py --episodes 20
"""

import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from config import (
    ARCHITECTURES,
    DEVICE,
    ENV_ID,
    HYPERPARAMS,
    MODELS_DIR,
    N_SEEDS,
    RESULTS_DIR,
)
from utils import load_metadata, make_dirs, results_path, run_id, save_metadata


# ── Single evaluation ─────────────────────────────────────────────────────────

def evaluate_single(
    hp_name: str,
    arch_name: str,
    seed: int,
    n_episodes: int = 20,
    render: bool = False,
) -> dict | None:
    rid = run_id(hp_name, arch_name, seed)
    model_path = os.path.join(MODELS_DIR, f"{rid}.zip")

    if not os.path.exists(model_path):
        print(f"[{rid}] Model not found: {model_path}")
        return None

    render_mode = "human" if render else None
    env = Monitor(gym.make(ENV_ID, render_mode=render_mode))

    model = SAC.load(model_path, env=env, device=DEVICE)

    mean_r, std_r = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,   # exploitation only — no entropy exploration
        warn=False,
    )

    # Collect per-episode rewards for distribution
    episode_rewards = []
    obs, _ = env.reset()
    current_reward = 0.0
    episodes_done = 0

    while episodes_done < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        current_reward += reward
        if terminated or truncated:
            episode_rewards.append(current_reward)
            current_reward = 0.0
            episodes_done += 1
            obs, _ = env.reset()

    env.close()

    result = {
        "run_id": rid,
        "hp_name": hp_name,
        "arch_name": arch_name,
        "seed": seed,
        "n_episodes": n_episodes,
        "deterministic": True,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "episode_rewards": episode_rewards,
    }

    out_path = os.path.join(MODELS_DIR, f"{rid}_eval.json")
    save_metadata(out_path, result)
    print(
        f"[{rid}]  mean={result['mean_reward']:+.2f}  "
        f"std={result['std_reward']:.2f}  "
        f"min={result['min_reward']:+.2f}  "
        f"max={result['max_reward']:+.2f}"
    )
    return result


# ── Summary across all seeds for a config ────────────────────────────────────

def evaluate_config(hp_name: str, arch_name: str, n_episodes: int = 20) -> dict:
    """Evaluate all seeds for a config and aggregate."""
    results = []
    for seed in range(N_SEEDS):
        r = evaluate_single(hp_name, arch_name, seed, n_episodes)
        if r:
            results.append(r)

    if not results:
        return {}

    all_means = [r["mean_reward"] for r in results]
    summary = {
        "config": f"{hp_name}__{arch_name}",
        "n_seeds_evaluated": len(results),
        "mean_of_means": float(np.mean(all_means)),
        "std_of_means": float(np.std(all_means)),
        "best_seed_mean": float(np.max(all_means)),
        "worst_seed_mean": float(np.min(all_means)),
        "per_seed": results,
    }

    out_path = os.path.join(MODELS_DIR, f"{hp_name}__{arch_name}_summary.json")
    save_metadata(out_path, summary)
    print(
        f"\nConfig {hp_name}/{arch_name}: "
        f"mean={summary['mean_of_means']:+.2f} ± {summary['std_of_means']:.2f} "
        f"(best seed: {summary['best_seed_mean']:+.2f})"
    )
    return summary


# ── Find best overall model ───────────────────────────────────────────────────

def find_best_model(n_episodes: int = 20) -> dict | None:
    """Evaluate all configs and return info about the best-performing one."""
    best = None
    best_mean = -np.inf

    summaries = []
    for hp_name in HYPERPARAMS:
        for arch_name in ARCHITECTURES:
            print(f"\n── {hp_name} / {arch_name} ──────────────────────────────────")
            s = evaluate_config(hp_name, arch_name, n_episodes)
            if s and s["mean_of_means"] > best_mean:
                best_mean = s["mean_of_means"]
                best = s
            summaries.append(s)

    if best:
        print(f"\n{'='*60}")
        print(f"  Best config : {best['config']}")
        print(f"  Mean reward : {best['mean_of_means']:+.2f} ± {best['std_of_means']:.2f}")
        print(f"{'='*60}")
        save_metadata(os.path.join(MODELS_DIR, "best_config.json"), best)

    return best


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate saved SAC agents (deterministic)")
    p.add_argument("--hp",   default=None, choices=list(HYPERPARAMS.keys()))
    p.add_argument("--arch", default=None, choices=list(ARCHITECTURES.keys()))
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--render", action="store_true",
                   help="Render one seed visually (requires display)")
    p.add_argument("--best", action="store_true",
                   help="Find and report the best config across all runs")
    return p.parse_args()


if __name__ == "__main__":
    make_dirs(MODELS_DIR)
    args = parse_args()

    if args.best:
        find_best_model(n_episodes=args.episodes)
    elif args.hp and args.arch and args.seed is not None:
        evaluate_single(args.hp, args.arch, args.seed, args.episodes, args.render)
    elif args.hp and args.arch:
        evaluate_config(args.hp, args.arch, args.episodes)
    else:
        find_best_model(n_episodes=args.episodes)