"""
Uruchomienie:
  python run.py --mode human      # graj ręcznie
  python run.py --mode train      # trenuj agenta Q-learning
  python run.py --mode agent      # oglądaj wytrenowanego agenta
"""

import argparse
import pickle
import sys
import numpy as np
import pygame

from maze_env import MazeEnv

# ─── Q-learning ───────────────────────────────────────────────────────────────

GRID   = 20
N_KEYS = 6
N_OBS  = GRID * GRID * (2 ** N_KEYS)
N_ACT  = 4

def obs_to_idx(obs):
    x, y, keys = obs
    return y * GRID * (2**N_KEYS) + x * (2**N_KEYS) + keys

def train(episodes=5000, render_every=0, save_path="qtable.pkl"):
    env = MazeEnv(render_mode="human" if render_every else None)
    Q   = np.zeros((N_OBS, N_ACT))

    alpha   = 0.3    # learning rate
    gamma   = 0.95   # discount
    eps     = 1.0    # eksploracja
    eps_min = 0.05
    eps_dec = (eps - eps_min) / (episodes * 0.8)

    best_reward = -np.inf
    stats = []

    for ep in range(episodes):
        obs, _ = env.reset()
        total_r = 0.0
        done    = False

        while not done:
            if render_every and (ep % render_every == 0):
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close(); sys.exit()

            s = obs_to_idx(obs)

            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[s]))

            obs2, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            s2 = obs_to_idx(obs2)
            Q[s, action] += alpha * (reward + gamma * np.max(Q[s2]) - Q[s, action])

            obs      = obs2
            total_r += reward

        eps = max(eps_min, eps - eps_dec)
        stats.append(total_r)

        if total_r > best_reward:
            best_reward = total_r

        if (ep + 1) % 200 == 0:
            avg = np.mean(stats[-200:])
            print(f"Ep {ep+1:5d} | avg_reward={avg:7.2f} | best={best_reward:7.2f} | eps={eps:.3f}")

    env.close()
    with open(save_path, "wb") as f:
        pickle.dump(Q, f)
    print(f"\nZapisano Q-tablicę → {save_path}")
    return Q


def run_agent(qtable_path="qtable.pkl", episodes=3):
    with open(qtable_path, "rb") as f:
        Q = pickle.load(f)

    env = MazeEnv(render_mode="human")
    for ep in range(episodes):
        obs, _ = env.reset()
        done   = False
        total  = 0.0
        while not done:
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    sys.exit()
            s = obs_to_idx(obs)
            action = int(np.argmax(Q[s]))
            obs, r, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total += r
        print(f"Epizod {ep+1} zakończony | reward={total:.2f} | kroki={env._steps}")
    env.close()


def play_human():
    env = MazeEnv(render_mode="human")
    obs, _ = env.reset()
    env.render()
    running = True
    ACTION_MAP = {
        pygame.K_UP:    0,
        pygame.K_RIGHT: 1,
        pygame.K_DOWN:  2,
        pygame.K_LEFT:  3,
        pygame.K_w:     0,
        pygame.K_d:     1,
        pygame.K_s:     2,
        pygame.K_a:     3,
    }
    print("Sterowanie: strzałki / WASD | Q = wyjście")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                running = False
            elif event.type == pygame.KEYDOWN and event.key in ACTION_MAP:
                action = ACTION_MAP[event.key]
                obs, reward, terminated, truncated, _ = env.step(action)
                env.render()
                if terminated:
                    print("Wygrałeś!")
                    pygame.time.wait(2000)
                    running = False
                elif truncated:
                    print("Przekroczono limit kroków.")
                    running = False
    env.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["human", "train", "agent"], default="human")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--render_every", type=int, default=500,
                        help="Renderuj podczas treningu co N epizodów (0=wyłącz)")
    parser.add_argument("--qtable", default="qtable.pkl")
    args = parser.parse_args()

    if args.mode == "human":
        play_human()
    elif args.mode == "train":
        train(episodes=args.episodes, render_every=args.render_every, save_path=args.qtable)
    elif args.mode == "agent":
        run_agent(qtable_path=args.qtable)
