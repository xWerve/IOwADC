"""
visualize.py — Pygame dashboard for deterministic agent evaluation.

Layout:
  ┌─────────────────────────┬──────────────────────┐
  │                         │  Episode reward curve │
  │   BipedalWalker-v3      │  (all episodes so far)│
  │   environment render    ├──────────────────────┤
  │                         │  Current episode      │
  │                         │  step reward curve    │
  ├─────────────────────────┴──────────────────────┤
  │  Stats bar: episode | step | reward | actions  │
  └────────────────────────────────────────────────┘

Usage:
    # Best config (reads models/best_config.json):
    python visualize.py

    # Specific run:
    python visualize.py --hp hp_default --arch arch_large --seed 3

    # Number of episodes to run:
    python visualize.py --episodes 10

    # Speed (1.0 = real-time, 2.0 = 2x, 0 = unlimited):
    python visualize.py --speed 1.5
"""

import argparse
import json
import os
import time
from collections import deque

import gymnasium as gym
import numpy as np
import pygame
from stable_baselines3 import SAC

from config import DEVICE, HYPERPARAMS, MODELS_DIR
from utils import run_id

# ── Layout constants ──────────────────────────────────────────────────────────

WIN_W, WIN_H     = 1200, 700
ENV_W, ENV_H     = 700, 540          # left panel: env render
PANEL_W          = WIN_W - ENV_W     # right panel width
STATS_H          = WIN_H - ENV_H     # bottom bar height
CHART_H          = (WIN_H - STATS_H) // 2   # each chart height

# ── Colours ───────────────────────────────────────────────────────────────────

BG          = (18,  18,  24)
PANEL_BG    = (26,  26,  34)
BORDER      = (50,  50,  65)
TEXT_MAIN   = (220, 220, 230)
TEXT_DIM    = (120, 120, 140)
TEXT_GOOD   = ( 74, 222, 128)   # green
TEXT_BAD    = (248,  113, 113)  # red
TEXT_ACCENT = ( 96, 165, 250)   # blue
CHART_LINE  = ( 96, 165, 250)
CHART_LINE2 = (251, 191,  36)   # yellow — current episode
CHART_GRID  = ( 40,  40,  52)
ZERO_LINE   = ( 80,  80, 100)
ACTION_COLS = [
    (129, 140, 248),   # indigo
    ( 52, 211, 153),   # emerald
    (251, 146,  60),   # orange
    (232,  90, 143),   # pink
]


# ── Mini chart helper ─────────────────────────────────────────────────────────

def draw_chart(
    surf: pygame.Surface,
    rect: pygame.Rect,
    values: list[float],
    title: str,
    color,
    y_min: float | None = None,
    y_max: float | None = None,
    zero_line: bool = True,
    fill_alpha: int = 40,
):
    """Draw a line chart inside rect with grid and title."""
    pygame.draw.rect(surf, PANEL_BG, rect)
    pygame.draw.rect(surf, BORDER, rect, 1)

    pad = 10
    inner = pygame.Rect(rect.x + pad + 20, rect.y + 24, rect.w - pad*2 - 20, rect.h - 40)

    # Title
    font_sm = pygame.font.SysFont("monospace", 12)
    surf.blit(font_sm.render(title, True, TEXT_DIM), (rect.x + pad, rect.y + 6))

    if len(values) < 2:
        return

    arr = np.array(values, dtype=float)
    lo = y_min if y_min is not None else arr.min()
    hi = y_max if y_max is not None else arr.max()
    if hi - lo < 1e-3:
        hi = lo + 1.0

    def to_px(v):
        nx = inner.x + int((i / (len(arr) - 1)) * inner.w)
        ny = inner.bottom - int((v - lo) / (hi - lo) * inner.h)
        return nx, ny

    # Grid lines (4 horizontal)
    for k in range(5):
        gy = inner.top + int(k / 4 * inner.h)
        pygame.draw.line(surf, CHART_GRID, (inner.left, gy), (inner.right, gy))
        label_v = hi - k / 4 * (hi - lo)
        lbl = font_sm.render(f"{label_v:+.0f}", True, TEXT_DIM)
        surf.blit(lbl, (rect.x + 2, gy - 6))

    # Zero line
    if zero_line and lo < 0 < hi:
        zy = inner.bottom - int((0 - lo) / (hi - lo) * inner.h)
        pygame.draw.line(surf, ZERO_LINE, (inner.left, zy), (inner.right, zy), 1)

    # Fill polygon
    points = []
    for i, v in enumerate(arr):
        points.append(to_px(v))

    if fill_alpha > 0 and len(points) >= 2:
        base_y = inner.bottom
        poly = [(inner.left, base_y)] + points + [(inner.right, base_y)]
        fill_surf = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
        offset = (rect.x, rect.y)
        adj = [(p[0] - offset[0], p[1] - offset[1]) for p in poly]
        r, g, b = color
        pygame.draw.polygon(fill_surf, (r, g, b, fill_alpha), adj)
        surf.blit(fill_surf, offset)

    # Line
    px_points = [to_px(v) for i, v in enumerate(arr)]
    pygame.draw.lines(surf, color, False, px_points, 2)

    # Last value label
    last_px = to_px(arr[-1])
    lv_lbl = font_sm.render(f"{arr[-1]:+.1f}", True, color)
    surf.blit(lv_lbl, (min(last_px[0] + 4, inner.right - 30), last_px[1] - 8))


# ── Action bar helper ─────────────────────────────────────────────────────────

def draw_action_bars(surf, rect, actions: np.ndarray):
    """Horizontal bars showing each action value in [-1, 1]."""
    pygame.draw.rect(surf, PANEL_BG, rect)
    pygame.draw.rect(surf, BORDER, rect, 1)

    font_sm = pygame.font.SysFont("monospace", 12)
    surf.blit(font_sm.render("Actions  [-1 … +1]", True, TEXT_DIM), (rect.x + 8, rect.y + 6))

    n = len(actions)
    bar_h = 14
    spacing = (rect.h - 28) // n

    for i, a in enumerate(actions):
        label = ["hip L", "knee L", "hip R", "knee R"][i] if n == 4 else f"a{i}"
        y = rect.y + 24 + i * spacing
        bar_w = rect.w - 80

        # Background track
        track = pygame.Rect(rect.x + 50, y, bar_w, bar_h)
        pygame.draw.rect(surf, CHART_GRID, track, border_radius=3)

        # Center marker
        cx = rect.x + 50 + bar_w // 2
        pygame.draw.line(surf, ZERO_LINE, (cx, y), (cx, y + bar_h))

        # Filled bar from center
        fill_w = int(abs(a) * bar_w / 2)
        if a >= 0:
            fill_rect = pygame.Rect(cx, y, fill_w, bar_h)
        else:
            fill_rect = pygame.Rect(cx - fill_w, y, fill_w, bar_h)
        col = ACTION_COLS[i % len(ACTION_COLS)]
        pygame.draw.rect(surf, col, fill_rect, border_radius=3)

        # Labels
        surf.blit(font_sm.render(label, True, TEXT_DIM),  (rect.x + 4, y + 1))
        surf.blit(font_sm.render(f"{a:+.2f}", True, col), (rect.x + 50 + bar_w + 4, y + 1))


# ── Stats bar ─────────────────────────────────────────────────────────────────

def draw_stats(surf, rect, ep: int, total_ep: int, step: int,
               ep_reward: float, total_reward: float, done: bool, config_name: str):
    pygame.draw.rect(surf, PANEL_BG, rect)
    pygame.draw.rect(surf, BORDER, rect, 1)

    font_md = pygame.font.SysFont("monospace", 14, bold=True)
    font_sm = pygame.font.SysFont("monospace", 12)

    items = [
        ("CONFIG",   config_name,           TEXT_ACCENT),
        ("EPISODE",  f"{ep}/{total_ep}",     TEXT_MAIN),
        ("STEP",     f"{step}",              TEXT_MAIN),
        ("EP REWARD", f"{ep_reward:+.2f}",  TEXT_GOOD if ep_reward >= 0 else TEXT_BAD),
        ("TOTAL",    f"{total_reward:+.2f}", TEXT_GOOD if total_reward >= 0 else TEXT_BAD),
        ("MODE",     "DETERMINISTIC",        TEXT_GOOD),
        ("STATUS",   "DONE ✓" if done else "RUNNING", TEXT_GOOD if done else TEXT_ACCENT),
    ]

    x = rect.x + 12
    for label, value, color in items:
        surf.blit(font_sm.render(label, True, TEXT_DIM),  (x, rect.y + 6))
        surf.blit(font_md.render(value, True, color),     (x, rect.y + 20))
        x += max(100, len(value) * 9 + 20)
        if x > rect.right - 120:
            break


# ── Main dashboard ────────────────────────────────────────────────────────────

def run_dashboard(
    hp_name: str,
    arch_name: str,
    seed: int,
    n_episodes: int = 10,
    speed: float = 1.5,
):
    rid = run_id(hp_name, arch_name, seed)
    model_path = os.path.join(MODELS_DIR, f"{rid}.zip")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}\nRun train.py first.")

    # ── Load model ────────────────────────────────────────────────────────────
    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    model = SAC.load(model_path, env=env, device=DEVICE)
    config_name = f"{hp_name} / {arch_name} / seed {seed}"

    # ── Pygame init ───────────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(f"BipedalWalker-v3 — SAC Evaluation — {rid}")
    clock = pygame.time.Clock()

    # ── State ─────────────────────────────────────────────────────────────────
    episode_rewards: list[float]   = []
    step_rewards:    deque[float]  = deque(maxlen=500)
    current_action:  np.ndarray    = np.zeros(env.action_space.shape[0])

    ep = 0
    step = 0
    ep_reward = 0.0
    total_reward = 0.0
    episode_done = False

    obs, _ = env.reset()
    frame = env.render()

    # ── Rects ─────────────────────────────────────────────────────────────────
    env_rect    = pygame.Rect(0, 0, ENV_W, ENV_H)
    chart1_rect = pygame.Rect(ENV_W, 0, PANEL_W, CHART_H)
    chart2_rect = pygame.Rect(ENV_W, CHART_H, PANEL_W, CHART_H)
    action_rect = pygame.Rect(ENV_W, CHART_H * 2, PANEL_W, WIN_H - CHART_H * 2 - STATS_H)
    stats_rect  = pygame.Rect(0, ENV_H, WIN_W, STATS_H)

    running = True
    step_delay = 1.0 / (60.0 * max(speed, 0.01)) if speed > 0 else 0
    last_step_time = time.perf_counter()

    while running:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r:  # restart
                    obs, _ = env.reset()
                    ep_reward = 0.0
                    step = 0
                    step_rewards.clear()
                    episode_done = False

        # ── Step ──────────────────────────────────────────────────────────────
        now = time.perf_counter()
        if not episode_done and (now - last_step_time >= step_delay):
            last_step_time = now

            action, _ = model.predict(obs, deterministic=True)
            current_action = action
            obs, reward, terminated, truncated, _ = env.step(action)
            frame = env.render()

            ep_reward    += float(reward)
            total_reward += float(reward)
            step         += 1
            step_rewards.append(ep_reward)

            if terminated or truncated:
                episode_rewards.append(ep_reward)
                ep += 1
                episode_done = True

                if ep >= n_episodes:
                    running = False
                    continue

                obs, _ = env.reset()
                ep_reward = 0.0
                step = 0
                step_rewards.clear()
                episode_done = False

        # ── Draw ──────────────────────────────────────────────────────────────
        screen.fill(BG)

        # Env frame
        if frame is not None:
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            scaled = pygame.transform.scale(surf, (ENV_W, ENV_H))
            screen.blit(scaled, env_rect)
        pygame.draw.rect(screen, BORDER, env_rect, 1)

        # Chart 1 — episode rewards history
        draw_chart(
            screen, chart1_rect,
            episode_rewards if episode_rewards else [0],
            f"Episode Rewards  (episodes: {len(episode_rewards)})",
            CHART_LINE,
            zero_line=True,
        )

        # Chart 2 — current episode cumulative reward
        draw_chart(
            screen, chart2_rect,
            list(step_rewards) if step_rewards else [0],
            f"Cumulative Reward — episode {ep+1}  (step {step})",
            CHART_LINE2,
            zero_line=True,
        )

        # Action bars
        draw_action_bars(screen, action_rect, current_action)

        # Stats bar
        draw_stats(
            screen, stats_rect,
            ep + 1, n_episodes, step,
            ep_reward, total_reward,
            episode_done and ep >= n_episodes,
            config_name,
        )

        pygame.display.flip()
        clock.tick(120)

    # ── Final summary ─────────────────────────────────────────────────────────
    pygame.quit()
    env.close()

    if episode_rewards:
        print(f"\n── Evaluation complete ──────────────────────────────")
        print(f"  Config   : {config_name}")
        print(f"  Episodes : {len(episode_rewards)}")
        print(f"  Mean     : {np.mean(episode_rewards):+.2f}")
        print(f"  Std      : {np.std(episode_rewards):.2f}")
        print(f"  Min/Max  : {np.min(episode_rewards):+.2f} / {np.max(episode_rewards):+.2f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Pygame evaluation dashboard for SAC agent")
    p.add_argument("--hp",       default=None, choices=list(HYPERPARAMS.keys()))
    p.add_argument("--arch",     default=None)
    p.add_argument("--seed",     type=int, default=None)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--speed",    type=float, default=1.5,
                   help="Playback speed multiplier (1.0=real-time, 0=unlimited)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Auto-load best config if not specified
    hp_name   = args.hp
    arch_name = args.arch
    seed      = args.seed

    if not all([hp_name, arch_name, seed is not None]):
        best_path = os.path.join(MODELS_DIR, "best_config.json")
        if os.path.exists(best_path):
            with open(best_path) as f:
                meta = json.load(f)
            hp_name, arch_name = meta["config"].split("__", 1)
            seed = max(meta["per_seed"], key=lambda x: x["mean_reward"])["seed"]
            print(f"Auto-selected best config: {hp_name} / {arch_name} / seed {seed}")
        else:
            print("No best_config.json found. Run: python evaluate.py --best")
            print("Or specify manually: python visualize.py --hp hp_default --arch arch_large --seed 0")
            exit(1)

    run_dashboard(hp_name, arch_name, seed, n_episodes=args.episodes, speed=args.speed)