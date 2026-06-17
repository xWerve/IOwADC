"""
utils.py — Lightweight helpers (mirrors project05 patterns).
"""

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np


def make_dirs(*dirs: str) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def run_id(
    experiment: str,
    algo: str,
    variant: str,
    role: str,
    seed: int,
) -> str:
    """
    Canonical run identifier.
      experiment: shared|mixed
      algo: ppo|a2c|ppo+a2c
      variant: v0|v1|...
      role: joint|agent_0|agent_1
    """
    return f"{experiment}__{algo}__{variant}__{role}__seed{seed:02d}"


def results_path(results_dir: str, rid: str) -> str:
    return os.path.join(results_dir, f"{rid}.csv")


def model_path(models_dir: str, rid: str) -> str:
    return os.path.join(models_dir, f"{rid}.zip")


def meta_path(models_dir: str, rid: str) -> str:
    return os.path.join(models_dir, f"{rid}_meta.json")


def save_eval_row(
    path: str,
    timestep: int,
    mean_reward: float,
    std_reward: float,
    elapsed_sec: float,
) -> None:
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestep", "mean_reward", "std_reward", "elapsed_sec"])
        w.writerow([timestep, mean_reward, std_reward, elapsed_sec])


def load_results_csv(path: str) -> dict[str, Any]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    return {
        "timestep": data["timestep"].astype(int),
        "mean_reward": data["mean_reward"].astype(float),
        "std_reward": data["std_reward"].astype(float),
        "elapsed_sec": data["elapsed_sec"].astype(float),
    }


def load_group(
    results_dir: str,
    run_ids: Iterable[str],
) -> dict[str, Any]:
    """
    Load multiple CSVs and compute mean±std over seeds at each eval point.
    Pads shorter runs with NaN.
    """
    curves = []
    timesteps = None
    for rid in run_ids:
        path = results_path(results_dir, rid)
        if not os.path.exists(path):
            continue
        r = load_results_csv(path)
        if timesteps is None:
            timesteps = r["timestep"]
        curves.append(r["mean_reward"])

    if not curves:
        return {}

    max_len = max(len(c) for c in curves)
    padded = np.full((len(curves), max_len), np.nan)
    for i, c in enumerate(curves):
        padded[i, : len(c)] = c

    if timesteps is None:
        timesteps = np.arange(max_len, dtype=int)
    elif len(timesteps) < max_len:
        step = int(timesteps[-1] - timesteps[-2]) if len(timesteps) > 1 else 1000
        extra = np.arange(1, max_len - len(timesteps) + 1) * step + timesteps[-1]
        timesteps = np.concatenate([timesteps, extra])

    return {
        "timesteps": timesteps[:max_len],
        "rewards": padded,
        "mean": np.nanmean(padded, axis=0),
        "std": np.nanstd(padded, axis=0),
        "n_runs": len(curves),
    }


def save_metadata(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_metadata(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _patch_sb3_zip_load() -> None:
    """Work around torch>=2.12 failing to load .pth directly from zip streams."""
    import io

    import torch as th
    import stable_baselines3.common.save_util as save_util

    if getattr(save_util, "_project06_patched", False):
        return

    original = save_util.load_from_zip_file

    def load_from_zip_file(
        load_path,
        load_data=True,
        custom_objects=None,
        device="auto",
        verbose=0,
        print_system_info=False,
    ):
        import os
        import pathlib
        import warnings
        import zipfile

        from stable_baselines3.common.save_util import json_to_data, open_path
        from stable_baselines3.common.utils import get_device

        file = open_path(load_path, "r", verbose=verbose, suffix="zip")
        device = get_device(device=device)
        try:
            with zipfile.ZipFile(file) as archive:
                namelist = archive.namelist()
                data = None
                pytorch_variables = None
                params = {}

                if print_system_info and "system_info.txt" in namelist:
                    print("== SAVED MODEL SYSTEM INFO ==")
                    print(archive.read("system_info.txt").decode())

                if "data" in namelist and load_data:
                    data = json_to_data(archive.read("data").decode(), custom_objects=custom_objects)

                for file_path in [n for n in namelist if os.path.splitext(n)[1] == ".pth"]:
                    th_object = th.load(
                        io.BytesIO(archive.read(file_path)),
                        map_location=device,
                        weights_only=False,
                    )
                    if file_path in {"pytorch_variables.pth", "tensors.pth"}:
                        pytorch_variables = th_object
                    else:
                        params[os.path.splitext(file_path)[0]] = th_object
        except zipfile.BadZipFile as e:
            raise ValueError(f"Error: the file {load_path} wasn't a zip-file") from e
        finally:
            if isinstance(load_path, (str, pathlib.Path)):
                file.close()
        return data, params, pytorch_variables

    save_util.load_from_zip_file = load_from_zip_file
    import stable_baselines3.common.base_class as base_class

    base_class.load_from_zip_file = load_from_zip_file
    save_util._project06_patched = True


def load_model(cls, path: str, env=None, device: str | None = None):
    _patch_sb3_zip_load()
    return cls.load(path, env=env, device=device)


class Timer:
    def __init__(self):
        self._start = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def reset(self):
        self._start = time.perf_counter()

