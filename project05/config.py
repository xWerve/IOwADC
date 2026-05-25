"""
config.py — Hyperparameter sets and network architectures for BipedalWalker-v3 + SAC
"""

import torch

# ── Environment ───────────────────────────────────────────────────────────────
ENV_ID = "BipedalWalker-v3"
TOTAL_TIMESTEPS = 400_000      # per single run
EVAL_FREQ = 5_000              # evaluate every N timesteps
EVAL_EPISODES = 10             # episodes per evaluation point
N_SEEDS = 5                    # independent runs per config
BEST_MODEL_THRESHOLD = 300.0   # reward considered "solved"

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Hyperparameter sets ───────────────────────────────────────────────────────
# Each dict is passed directly to SAC(...) as kwargs (minus policy/env).
# Architecture is separate (policy_kwargs) — see ARCHITECTURES below.
HYPERPARAMS = {
    "hp_default": {
        "learning_rate": 3e-4,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "learning_starts": 10_000,
        "buffer_size": 300_000,
        "ent_coef": "auto",
    },
    "hp_high_lr": {
        "learning_rate": 1e-3,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "learning_starts": 10_000,
        "buffer_size": 300_000,
        "ent_coef": "auto",
    },
    "hp_large_batch": {
        "learning_rate": 3e-4,
        "batch_size": 512,
        "tau": 0.02,
        "gamma": 0.995,
        "learning_starts": 10_000,
        "buffer_size": 300_000,
        "ent_coef": "auto",
    },
}

# ── Network architectures ─────────────────────────────────────────────────────
# policy_kwargs passed to SAC. Both use MlpPolicy (default for continuous envs).
# Input:  observation vector (24,)
# Output: action vector (4,) — joint torques, tanh-squashed to [-1, 1]
ARCHITECTURES = {
    "arch_small": {
        "net_arch": [64, 64],
        "activation_fn": torch.nn.ReLU,
    },
    "arch_large": {
        "net_arch": [256, 256],
        "activation_fn": torch.nn.ReLU,
    },
}

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = "results"       # CSV files with eval rewards per run
MODELS_DIR  = "models"        # saved .zip agents
PLOTS_DIR   = "plots"         # generated figures