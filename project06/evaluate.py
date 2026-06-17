"""
evaluate.py — Deterministic evaluation of saved agents.
"""

from __future__ import annotations

import argparse
import os

from stable_baselines3 import A2C, PPO

from project06.config import (
    A2C_VARIANTS,
    DEVICE,
    MODELS_DIR,
    N_SEEDS,
    PPO_VARIANTS,
)
from project06.eval_helpers import evaluate_mixed_team, evaluate_shared_team
from project06.utils import load_model, make_dirs, model_path, run_id, save_metadata


ALGO_CLS = {"ppo": PPO, "a2c": A2C}


def evaluate_shared(algo: str, variant: str, seed: int, n_episodes: int = 20) -> dict | None:
    rid = run_id("shared", algo, variant, "joint", seed)
    mpath = model_path(MODELS_DIR, rid)
    if not os.path.exists(mpath):
        print(f"[{rid}] model not found")
        return None

    model = load_model(ALGO_CLS[algo], mpath, device=DEVICE)
    mean_r, std_r = evaluate_shared_team(model, n_episodes, seed, deterministic=True)

    result = {
        "run_id": rid,
        "experiment": "shared",
        "algo": algo,
        "variant": variant,
        "seed": seed,
        "n_episodes": n_episodes,
        "mean_reward": float(mean_r),
        "std_reward": float(std_r),
    }
    save_metadata(os.path.join(MODELS_DIR, f"{rid}_eval.json"), result)
    print(f"[{rid}] mean={mean_r:+.3f} std={std_r:.3f}")
    return result


def evaluate_mixed_seed(seed: int, ppo_variant: str = "v0", a2c_variant: str = "v0", n_episodes: int = 20):
    ppo_rid = run_id("mixed", "ppo", ppo_variant, "agent_0", seed)
    a2c_rid = run_id("mixed", "a2c", a2c_variant, "agent_1", seed)
    joint_rid = run_id("mixed", "ppo+a2c", ppo_variant, "joint", seed)

    ppo_path = model_path(MODELS_DIR, ppo_rid)
    a2c_path = model_path(MODELS_DIR, a2c_rid)
    if not os.path.exists(ppo_path) or not os.path.exists(a2c_path):
        print(f"[{joint_rid}] missing PPO or A2C model")
        return None

    ppo_model = load_model(PPO, ppo_path, device=DEVICE)
    a2c_model = load_model(A2C, a2c_path, device=DEVICE)
    mean_r, std_r = evaluate_mixed_team(ppo_model, a2c_model, n_episodes, seed, deterministic=True)

    result = {
        "run_id": joint_rid,
        "experiment": "mixed",
        "seed": seed,
        "n_episodes": n_episodes,
        "mean_reward": mean_r,
        "std_reward": std_r,
    }
    save_metadata(os.path.join(MODELS_DIR, f"{joint_rid}_eval.json"), result)
    print(f"[{joint_rid}] mean={mean_r:+.3f} std={std_r:.3f}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["shared", "mixed", "all"], default="all")
    p.add_argument("--algo", choices=["ppo", "a2c"], default=None)
    p.add_argument("--variant", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    make_dirs(MODELS_DIR)
    args = parse_args()
    seeds = [args.seed] if args.seed is not None else list(range(N_SEEDS))

    if args.mode in ("shared", "all"):
        algos = [args.algo] if args.algo else ["ppo", "a2c"]
        for algo in algos:
            variants = list(PPO_VARIANTS.keys()) if algo == "ppo" else list(A2C_VARIANTS.keys())
            if args.variant:
                variants = [args.variant] if args.variant in variants else variants
            for variant in variants:
                for seed in seeds:
                    evaluate_shared(algo, variant, seed, args.episodes)

    if args.mode in ("mixed", "all"):
        for seed in seeds:
            evaluate_mixed_seed(seed, n_episodes=args.episodes)
