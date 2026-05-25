"""
plot.py — Generate learning curve plots.

Produces:
    1. hyperparams_comparison.png  — 3 HP sets vs each other (per architecture)
    2. arch_comparison.png         — 2 architectures vs each other (per HP set)
    3. eval_overlay.png            — best config training curve + deterministic eval line

Usage:
    python plot.py
    python plot.py --show        # also open plots interactively
"""

import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")   # headless by default; overridden with --show

from config import ARCHITECTURES, HYPERPARAMS, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from utils import config_id, load_results, make_dirs

# ── Style ─────────────────────────────────────────────────────────────────────

COLORS = [
    "#2563EB",   # blue
    "#DC2626",   # red
    "#16A34A",   # green
    "#D97706",   # amber
    "#7C3AED",   # violet
    "#0891B2",   # cyan
]
ALPHA_FILL = 0.18


def style_axes(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_curve(ax, timesteps, mean, std, label, color, linestyle="-"):
    ax.plot(timesteps, mean, color=color, linewidth=2, linestyle=linestyle, label=label)
    ax.fill_between(
        timesteps,
        mean - std,
        mean + std,
        color=color,
        alpha=ALPHA_FILL,
    )


# ── Plot 1: hyperparameter comparison ────────────────────────────────────────

def plot_hyperparams_comparison(show: bool = False):
    """One subplot per architecture, lines = HP sets."""
    n_arch = len(ARCHITECTURES)
    fig, axes = plt.subplots(1, n_arch, figsize=(7 * n_arch, 5), sharey=False)
    if n_arch == 1:
        axes = [axes]

    for ax, arch_name in zip(axes, ARCHITECTURES):
        for i, hp_name in enumerate(HYPERPARAMS):
            data = load_results(RESULTS_DIR, hp_name, arch_name)
            if not data:
                continue
            label = f"{hp_name}  (n={data['n_seeds']})"
            plot_curve(ax, data["timesteps"], data["mean"], data["std"],
                       label=label, color=COLORS[i % len(COLORS)])

        style_axes(ax,
                   title=f"Hyperparameter comparison\n({arch_name})",
                   xlabel="Timestep",
                   ylabel="Mean episode reward")
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(f"BipedalWalker-v3 — SAC", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "hyperparams_comparison.png", show)


# ── Plot 2: architecture comparison ──────────────────────────────────────────

def plot_arch_comparison(show: bool = False):
    """One subplot per HP set, lines = architectures."""
    n_hp = len(HYPERPARAMS)
    fig, axes = plt.subplots(1, n_hp, figsize=(7 * n_hp, 5), sharey=False)
    if n_hp == 1:
        axes = [axes]

    for ax, hp_name in zip(axes, HYPERPARAMS):
        for i, arch_name in enumerate(ARCHITECTURES):
            data = load_results(RESULTS_DIR, hp_name, arch_name)
            if not data:
                continue
            label = f"{arch_name}  (n={data['n_seeds']})"
            plot_curve(ax, data["timesteps"], data["mean"], data["std"],
                       label=label, color=COLORS[i % len(COLORS)])

        style_axes(ax,
                   title=f"Architecture comparison\n({hp_name})",
                   xlabel="Timestep",
                   ylabel="Mean episode reward")
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(f"BipedalWalker-v3 — SAC", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "arch_comparison.png", show)


# ── Plot 3: deterministic eval overlay ───────────────────────────────────────

def plot_eval_overlay(show: bool = False):
    """
    Training curve of the best config (mean ± std across seeds)
    with a horizontal band showing deterministic evaluation results.
    """
    best_meta_path = os.path.join(MODELS_DIR, "best_config.json")
    if not os.path.exists(best_meta_path):
        print("[plot_eval_overlay] Run evaluate.py --best first to generate best_config.json")
        return

    with open(best_meta_path) as f:
        best = json.load(f)

    config = best["config"]                    # "hp_name__arch_name"
    hp_name, arch_name = config.split("__", 1)
    det_mean = best["mean_of_means"]
    det_std  = best["std_of_means"]

    data = load_results(RESULTS_DIR, hp_name, arch_name)
    if not data:
        print(f"[plot_eval_overlay] No training results found for {config}")
        return

    fig, ax = plt.subplots(figsize=(9, 5))

    # Training curve
    plot_curve(ax, data["timesteps"], data["mean"], data["std"],
               label=f"Training (stochastic, n={data['n_seeds']})",
               color=COLORS[0])

    # Deterministic eval band
    ax.axhline(det_mean, color=COLORS[1], linewidth=2, linestyle="--",
               label=f"Deterministic eval  mean={det_mean:.1f}")
    ax.axhspan(det_mean - det_std, det_mean + det_std,
               color=COLORS[1], alpha=ALPHA_FILL,
               label=f"Deterministic eval  ±std={det_std:.1f}")

    style_axes(ax,
               title=f"Training vs Deterministic Evaluation\n{config}",
               xlabel="Timestep",
               ylabel="Mean episode reward")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    _save(fig, "eval_overlay.png", show)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(fig, filename: str, show: bool):
    make_dirs(PLOTS_DIR)
    path = os.path.join(PLOTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    if show:
        plt.show()
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate learning curve plots")
    p.add_argument("--show", action="store_true", help="Open plots interactively")
    p.add_argument("--hp-only",   action="store_true", help="Only hyperparams plot")
    p.add_argument("--arch-only", action="store_true", help="Only architecture plot")
    p.add_argument("--eval-only", action="store_true", help="Only eval overlay plot")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    any_specific = args.hp_only or args.arch_only or args.eval_only

    if not any_specific or args.hp_only:
        plot_hyperparams_comparison(show=args.show)
    if not any_specific or args.arch_only:
        plot_arch_comparison(show=args.show)
    if not any_specific or args.eval_only:
        plot_eval_overlay(show=args.show)