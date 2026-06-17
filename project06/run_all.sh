#!/usr/bin/env bash
# Run all project06 experiments using venv2
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv2/bin/python"

echo "=== Shared training (PPO v0/v1 + A2C) ==="
$PY -m project06.train_shared

echo "=== Mixed training (PPO + A2C) ==="
$PY -m project06.train_mixed

echo "=== Evaluation ==="
$PY -m project06.evaluate

echo "=== Plots ==="
$PY -m project06.plot

echo "Done."
