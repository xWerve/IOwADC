"""
plot.py — Generate learning curve plots for project06.
"""

from __future__ import annotations

import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

from project06.config import A2C_VARIANTS, N_SEEDS, PLOTS_DIR, PPO_VARIANTS, RESULTS_DIR
from project06.utils import load_group, make_dirs, run_id

COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706", "#7C3AED"]
ALPHA_FILL = 0.18


def style_axes(ax, title, xlabel="Timestep", ylabel="Mean episode reward"):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_curve(ax, timesteps, mean, std, label, color):
    ax.plot(timesteps, mean, color=color, linewidth=2, label=label)
    ax.fill_between(timesteps, mean - std, mean + std, color=color, alpha=ALPHA_FILL)


def _save(fig, filename: str):
    make_dirs(PLOTS_DIR)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_algo_comparison():
    """PPO vs A2C (shared policy, variant v0)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    specs = [
        ("ppo", "v0", "PPO (shared)"),
        ("a2c", "v0", "A2C (shared)"),
    ]
    for i, (algo, variant, label) in enumerate(specs):
        rids = [run_id("shared", algo, variant, "joint", s) for s in range(N_SEEDS)]
        data = load_group(RESULTS_DIR, rids)
        if not data:
            continue
        plot_curve(ax, data["timesteps"], data["mean"], data["std"],
                   f"{label} (n={data['n_runs']})", COLORS[i])
    style_axes(ax, "Algorithm comparison — shared policy (CollectCoins)")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "algo_comparison.png")


def plot_ppo_variants():
    """PPO v0 vs v1 (shared policy)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, variant in enumerate(PPO_VARIANTS):
        rids = [run_id("shared", "ppo", variant, "joint", s) for s in range(N_SEEDS)]
        data = load_group(RESULTS_DIR, rids)
        if not data:
            continue
        plot_curve(ax, data["timesteps"], data["mean"], data["std"],
                   f"PPO {variant} (n={data['n_runs']})", COLORS[i])
    style_axes(ax, "PPO hyperparameter variants — shared policy")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "ppo_variants.png")


def plot_same_vs_mixed():
    """Shared PPO vs mixed PPO+A2C team."""
    fig, ax = plt.subplots(figsize=(9, 5))
    specs = [
        ("shared", "ppo", "v0", "joint", "Same algo: PPO (both agents)"),
        ("mixed", "ppo+a2c", "v0", "joint", "Mixed: PPO + A2C"),
    ]
    for i, (exp, algo, variant, role, label) in enumerate(specs):
        rids = [run_id(exp, algo, variant, role, s) for s in range(N_SEEDS)]
        data = load_group(RESULTS_DIR, rids)
        if not data:
            continue
        plot_curve(ax, data["timesteps"], data["mean"], data["std"],
                   f"{label} (n={data['n_runs']})", COLORS[i])
    style_axes(ax, "Same algorithm vs mixed algorithms in episode")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "same_vs_mixed.png")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", default=True)
    return p.parse_args()


if __name__ == "__main__":
    plot_algo_comparison()
    plot_ppo_variants()
    plot_same_vs_mixed()
