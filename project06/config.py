"""
config.py — Project06 experiment configuration.
"""

from __future__ import annotations

from pathlib import Path

import torch

from project06.envs.collect_coins import CollectCoinsConfig

ROOT = Path(__file__).resolve().parent

# --- Environment (single harder setup) --------------------------------------

ENV_CFG = CollectCoinsConfig()


# --- Experiment -------------------------------------------------------------

TOTAL_TIMESTEPS = 200_000
EVAL_FREQ = 5_000
EVAL_EPISODES = 20
N_SEEDS = 3

# MLP policies train more reliably on CPU in SB3
DEVICE = "cpu"

RESULTS_DIR = str(ROOT / "results")
MODELS_DIR = str(ROOT / "models")
PLOTS_DIR = str(ROOT / "plots")


# --- Hyperparameters --------------------------------------------------------

PPO_VARIANTS = {
    "v0": {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "n_epochs": 10,
    },
    "v1": {  # variant for the “two HP settings” requirement
        "learning_rate": 1e-3,
        "n_steps": 1024,
        "batch_size": 256,
        "gamma": 0.995,
        "gae_lambda": 0.9,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "n_epochs": 10,
    },
}

A2C_VARIANTS = {
    "v0": {
        "learning_rate": 7e-4,
        "n_steps": 32,
        "gamma": 0.99,
        "gae_lambda": 1.0,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }
}

