"""
visualize.py — Pygame viewer for CollectCoins multi-agent episodes.

Usage (from repo root, venv2):
    python -m project06.visualize --mode shared --algo ppo --seed 0
    python -m project06.visualize --mode mixed --seed 0
    python -m project06.visualize --mode shared --algo ppo --seed 0 --export project06/plots/demo_frame.png
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import pygame
from stable_baselines3 import A2C, PPO

from project06.config import DEVICE, ENV_CFG, MODELS_DIR
from project06.envs import CollectCoinsParallelEnv
from project06.utils import load_model, model_path, run_id

# --- Layout -----------------------------------------------------------------

CELL = 56
MARGIN = 40
GRID_PX = CELL * ENV_CFG.grid_size
BOARD_W = GRID_PX + MARGIN * 2
PANEL_W = 280
STATS_H = 72
WIN_W = BOARD_W + PANEL_W
WIN_H = GRID_PX + MARGIN * 2 + STATS_H

BG = (22, 24, 32)
GRID_LINE = (55, 60, 78)
GRID_FILL = (30, 33, 44)
TEXT = (230, 232, 240)
TEXT_DIM = (140, 145, 165)
AGENT0 = (59, 130, 246)
AGENT1 = (239, 68, 68)
COIN = (250, 204, 21)
COIN_RING = (180, 140, 10)
WALL = (75, 78, 95)
REWARD_GOOD = (74, 222, 128)
REWARD_BAD = (248, 113, 113)

ACTION_NAMES = ["stay", "up", "down", "left", "right"]


def cell_rect(x: int, y: int) -> pygame.Rect:
    return pygame.Rect(MARGIN + x * CELL, MARGIN + y * CELL, CELL, CELL)


def draw_board(surf: pygame.Surface, env: CollectCoinsParallelEnv):
    state = env.get_state()
    n = state["grid_size"]
    font = pygame.font.SysFont("monospace", 14, bold=True)

    board_rect = pygame.Rect(MARGIN, MARGIN, GRID_PX, GRID_PX)
    pygame.draw.rect(surf, GRID_FILL, board_rect, border_radius=8)
    pygame.draw.rect(surf, GRID_LINE, board_rect, 2, border_radius=8)

    for x in range(n + 1):
        px = MARGIN + x * CELL
        pygame.draw.line(surf, GRID_LINE, (px, MARGIN), (px, MARGIN + GRID_PX))
    for y in range(n + 1):
        py = MARGIN + y * CELL
        pygame.draw.line(surf, GRID_LINE, (MARGIN, py), (MARGIN + GRID_PX, py))

    for (cx, cy) in state["coins"]:
        r = cell_rect(cx, cy)
        center = r.center
        pygame.draw.circle(surf, COIN_RING, center, CELL // 3 + 2)
        pygame.draw.circle(surf, COIN, center, CELL // 3)
        lbl = font.render("$", True, (80, 60, 0))
        surf.blit(lbl, lbl.get_rect(center=center))

    for (wx, wy) in state.get("walls", []):
        r = cell_rect(wx, wy)
        inner = r.inflate(-4, -4)
        pygame.draw.rect(surf, WALL, inner, border_radius=4)
        pygame.draw.rect(surf, GRID_LINE, inner, 1, border_radius=4)

    agent_styles = {
        "agent_0": (AGENT0, "A0"),
        "agent_1": (AGENT1, "A1"),
    }
    for agent, (color, label) in agent_styles.items():
        if agent not in state["positions"]:
            continue
        x, y = state["positions"][agent]
        r = cell_rect(x, y)
        inner = r.inflate(-10, -10)
        pygame.draw.rect(surf, color, inner, border_radius=10)
        font_sm = pygame.font.SysFont("monospace", 13, bold=True)
        txt = font_sm.render(label, True, (255, 255, 255))
        surf.blit(txt, txt.get_rect(center=inner.center))


def draw_chart(surf, rect, values, title, color):
    pygame.draw.rect(surf, (28, 30, 40), rect, border_radius=6)
    pygame.draw.rect(surf, GRID_LINE, rect, 1, border_radius=6)
    font = pygame.font.SysFont("monospace", 12)
    surf.blit(font.render(title, True, TEXT_DIM), (rect.x + 8, rect.y + 6))

    if len(values) < 2:
        return

    inner = pygame.Rect(rect.x + 8, rect.y + 24, rect.w - 16, rect.h - 32)
    arr = np.array(values, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-6:
        hi = lo + 1.0

    points = []
    for i, v in enumerate(arr):
        px = inner.x + int(i / (len(arr) - 1) * inner.w)
        py = inner.bottom - int((v - lo) / (hi - lo) * inner.h)
        points.append((px, py))
    pygame.draw.lines(surf, color, False, points, 2)


def draw_stats(surf, rect, lines: list[tuple[str, str, tuple]]):
    pygame.draw.rect(surf, (28, 30, 40), rect)
    pygame.draw.rect(surf, GRID_LINE, rect, 1)
    x = rect.x + 12
    for label, value, color in lines:
        font_sm = pygame.font.SysFont("monospace", 11)
        font_md = pygame.font.SysFont("monospace", 13, bold=True)
        surf.blit(font_sm.render(label, True, TEXT_DIM), (x, rect.y + 8))
        surf.blit(font_md.render(value, True, color), (x, rect.y + 22))
        x += 130


def load_shared_model(algo: str, variant: str, seed: int):
    rid = run_id("shared", algo, variant, "joint", seed)
    path = model_path(MODELS_DIR, rid)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nRun train_shared first.")
    cls = PPO if algo == "ppo" else A2C
    return cls, load_model(cls, path, device=DEVICE), rid


def load_mixed_models(variant: str, seed: int):
    ppo_rid = run_id("mixed", "ppo", variant, "agent_0", seed)
    a2c_rid = run_id("mixed", "a2c", "v0", "agent_1", seed)
    ppo_path = model_path(MODELS_DIR, ppo_rid)
    a2c_path = model_path(MODELS_DIR, a2c_rid)
    if not os.path.exists(ppo_path) or not os.path.exists(a2c_path):
        raise FileNotFoundError("Mixed models not found. Run train_mixed first.")
    ppo = load_model(PPO, ppo_path, device=DEVICE)
    a2c = load_model(A2C, a2c_path, device=DEVICE)
    return ppo, a2c, f"{ppo_rid} + {a2c_rid}"


def predict_shared(model, obs: dict) -> dict[str, int]:
    a0, _ = model.predict(obs["agent_0"], deterministic=True)
    a1, _ = model.predict(obs["agent_1"], deterministic=True)
    return {"agent_0": int(a0), "agent_1": int(a1)}


def predict_mixed(ppo_model, a2c_model, obs: dict) -> dict[str, int]:
    a0, _ = ppo_model.predict(obs["agent_0"], deterministic=True)
    a1, _ = a2c_model.predict(obs["agent_1"], deterministic=True)
    return {"agent_0": int(a0), "agent_1": int(a1)}


def run_viewer(
    mode: str,
    algo: str,
    variant: str,
    seed: int,
    n_episodes: int,
    speed: float,
    export_path: str | None,
):
    env = CollectCoinsParallelEnv(cfg=ENV_CFG)

    if mode == "shared":
        cls, model, rid = load_shared_model(algo, variant, seed)
        predict = lambda obs: predict_shared(model, obs)
        title = f"CollectCoins — shared {algo.upper()} ({variant})"
    else:
        ppo, a2c, rid = load_mixed_models(variant, seed)
        predict = lambda obs: predict_mixed(ppo, a2c, obs)
        title = "CollectCoins — mixed PPO + A2C"

    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"CollectCoins — {rid}")
    clock = pygame.time.Clock()
    font_title = pygame.font.SysFont("monospace", 14, bold=True)

    episode_rewards: list[float] = []
    step_rewards: deque[float] = deque(maxlen=200)
    ep = 0
    step = 0
    ep_reward = 0.0
    last_actions = {"agent_0": 0, "agent_1": 0}
    episode_done = False
    running = True
    step_delay = 1.0 / (8.0 * max(speed, 0.01)) if speed > 0 else 0
    last_step_time = time.perf_counter()

    obs, _ = env.reset(seed=seed)
    stats_rect = pygame.Rect(0, WIN_H - STATS_H, WIN_W, STATS_H)
    chart_rect = pygame.Rect(BOARD_W + 10, MARGIN, PANEL_W - 20, GRID_PX // 2 - 10)
    chart2_rect = pygame.Rect(BOARD_W + 10, MARGIN + GRID_PX // 2 + 10, PANEL_W - 20, GRID_PX // 2 - 10)

    exported = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    episode_done = False
                elif event.key == pygame.K_r:
                    obs, _ = env.reset(seed=seed + ep)
                    ep_reward = 0.0
                    step = 0
                    step_rewards.clear()
                    episode_done = False

        now = time.perf_counter()
        if not episode_done and env.agents and (now - last_step_time >= step_delay):
            last_step_time = now
            last_actions = predict(obs)
            obs, rewards, terminated, truncated, _ = env.step(last_actions)
            r = float(np.mean(list(rewards.values())))
            ep_reward += r
            step += 1
            step_rewards.append(ep_reward)

            if any(terminated.values()) or any(truncated.values()) or not env.agents:
                episode_rewards.append(ep_reward)
                ep += 1
                episode_done = True
                if ep >= n_episodes:
                    running = False
                    continue
                obs, _ = env.reset(seed=seed + ep)
                ep_reward = 0.0
                step = 0
                step_rewards.clear()
                episode_done = False

        screen.fill(BG)
        draw_board(screen, env)

        # Side panel
        screen.blit(font_title.render(title, True, TEXT), (BOARD_W + 10, 12))
        draw_chart(
            screen, chart_rect, episode_rewards if episode_rewards else [0],
            f"Episode rewards (n={len(episode_rewards)})", AGENT0,
        )
        draw_chart(
            screen, chart2_rect, list(step_rewards) if step_rewards else [0],
            f"Current ep. reward (step {step})", AGENT1,
        )

        reward_color = REWARD_GOOD if ep_reward >= 0 else REWARD_BAD
        state = env.get_state()
        draw_stats(screen, stats_rect, [
            ("MODE", mode.upper(), AGENT0),
            ("EPISODE", f"{min(ep + 1, n_episodes)}/{n_episodes}", TEXT),
            ("STEP", f"{step}/{state['max_steps']}", TEXT),
            ("COINS LEFT", str(len(state["coins"])), COIN),
            ("EP REWARD", f"{ep_reward:+.2f}", reward_color),
            ("A0 / A1", f"{ACTION_NAMES[last_actions['agent_0']][:1]} / {ACTION_NAMES[last_actions['agent_1']][:1]}", TEXT_DIM),
        ])

        pygame.display.flip()
        clock.tick(60)

        if export_path and not exported:
            pygame.image.save(screen, export_path)
            print(f"Saved frame: {export_path}")
            exported = True
            if speed <= 0:
                running = False

    pygame.quit()
    env.close()

    if episode_rewards:
        print(f"\n── {rid} ──")
        print(f"  Episodes : {len(episode_rewards)}")
        print(f"  Mean     : {np.mean(episode_rewards):+.3f}")
        print(f"  Std      : {np.std(episode_rewards):.3f}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualize CollectCoins episodes")
    p.add_argument("--mode", choices=["shared", "mixed"], default="shared")
    p.add_argument("--algo", choices=["ppo", "a2c"], default="ppo")
    p.add_argument("--variant", default="v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--speed", type=float, default=2.0, help="Playback speed (steps/sec multiplier)")
    p.add_argument("--export", default=None, help="Save one frame to PNG and exit")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_viewer(
        mode=args.mode,
        algo=args.algo,
        variant=args.variant,
        seed=args.seed,
        n_episodes=args.episodes,
        speed=0 if args.export else args.speed,
        export_path=args.export,
    )
