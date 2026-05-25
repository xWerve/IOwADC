"""
utils.py — Helper functions: directory setup, metadata I/O, results loading.
"""

import csv
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def make_dirs(*dirs: str) -> None:
    """Create output directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_id(hp_name: str, arch_name: str, seed: int) -> str:
    """Canonical identifier for a single training run."""
    return f"{hp_name}__{arch_name}__seed{seed:02d}"


def config_id(hp_name: str, arch_name: str) -> str:
    """Identifier for a hyperparameter × architecture combination."""
    return f"{hp_name}__{arch_name}"


# ── CSV results ───────────────────────────────────────────────────────────────

def results_path(results_dir: str, hp_name: str, arch_name: str, seed: int) -> str:
    return os.path.join(results_dir, f"{run_id(hp_name, arch_name, seed)}.csv")


def save_eval_row(
    path: str,
    timestep: int,
    mean_reward: float,
    std_reward: float,
    elapsed_sec: float,
) -> None:
    """Append one evaluation row to a CSV file."""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestep", "mean_reward", "std_reward", "elapsed_sec"])
        writer.writerow([timestep, mean_reward, std_reward, elapsed_sec])


def load_results(results_dir: str, hp_name: str, arch_name: str) -> dict[str, Any]:
    """
    Load all seed CSVs for a config and return arrays:
        timesteps : (T,)
        rewards   : (N_seeds, T)  — NaN where a seed has fewer points
    """
    from config import N_SEEDS

    all_rewards = []
    timesteps = None

    for seed in range(N_SEEDS):
        path = results_path(results_dir, hp_name, arch_name, seed)
        if not os.path.exists(path):
            continue
        data = np.genfromtxt(path, delimiter=",", names=True)
        if timesteps is None:
            timesteps = data["timestep"].astype(int)
        all_rewards.append(data["mean_reward"])

    if not all_rewards:
        return {}

    # Pad shorter arrays with NaN
    max_len = max(len(r) for r in all_rewards)
    padded = np.full((len(all_rewards), max_len), np.nan)
    for i, r in enumerate(all_rewards):
        padded[i, : len(r)] = r

    return {
        "timesteps": timesteps[:max_len],
        "rewards": padded,
        "mean": np.nanmean(padded, axis=0),
        "std": np.nanstd(padded, axis=0),
        "n_seeds": len(all_rewards),
    }


# ── Metadata ──────────────────────────────────────────────────────────────────

def save_metadata(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_metadata(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Timing ────────────────────────────────────────────────────────────────────

class Timer:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self):
        self._start = time.perf_counter()