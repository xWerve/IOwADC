import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pygame.font
import sys

# ─── Kolory ───────────────────────────────────────────────────────────────────
BLACK   = (15,  15,  20)
WHITE   = (240, 238, 230)
GRAY    = (50,  52,  60)
WALL    = (35,  38,  50)
FLOOR   = (22,  24,  32)
PLAYER  = (80, 160, 255)
GOAL    = (120, 220, 100)
KEY_COLORS = [
    (240, 180, 50), (100, 210, 180), (210, 110, 230),
    (230, 130, 100), (150, 200, 255), (180, 240, 100)
]
DOOR_COLORS = [
    (200, 140, 30), (60, 170, 140), (170, 70, 200),
    (190, 90, 60), (100, 150, 200), (130, 190, 70)
]
TP_IN   = (140, 130, 240)
TP_OUT  = (90,  200, 240)
HUD_BG  = (18,  20,  28)

# ─── Rozmiar siatki i okna ────────────────────────────────────────────────────
GRID    = 20
CELL    = 32
HUD_W   = 220
WIN_W   = GRID * CELL + HUD_W
WIN_H   = GRID * CELL
# ─── Kierunki ────────────────────────────────────────────────────────────────
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
N, E, S, W = 1, 2, 4, 8
OPPOSITE = {N: S, S: N, E: W, W: E}
DIR_WALL = {(0,-1): N, (1,0): E, (0,1): S, (-1,0): W}

# ─── Generowanie labiryntu DFS ────────────────────────────────────────────────
def generate_maze(rows, cols, seed=42):
    rng = np.random.default_rng(seed)
    walls = np.full((rows, cols), N | E | S | W, dtype=np.int16)

    def carve(x, y, visited):
        visited.add((x, y))
        dirs = [(0, -1, N, S), (1, 0, E, W), (0, 1, S, N), (-1, 0, W, E)]
        order = rng.permutation(4)
        for i in order:
            dx, dy, w, ow = dirs[i]
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited:
                walls[y][x] &= ~w
                walls[ny][nx] &= ~ow
                carve(nx, ny, visited)

    carve(0, 0, set())

    for _ in range(100):
        x, y = rng.integers(0, cols), rng.integers(0, rows)
        dirs = [(0, -1, N, S), (1, 0, E, W), (0, 1, S, N), (-1, 0, W, E)]
        dx, dy, w, ow = dirs[rng.integers(0, 4)]
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            walls[y][x] &= ~w
            walls[ny][nx] &= ~ow

    for x in range(13, 20):
        walls[19][x] |= N
        if x > 0:
            walls[19][x] &= ~W
            walls[19][x - 1] &= ~E

    walls[19][13] &= ~N
    if 18 < rows: walls[18][13] &= ~S

    return walls

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    N_KEYS = 6
    KEYS = [
        {"pos": (1, 1), "id": 0},  # Lewa góra
        {"pos": (18, 1), "id": 1},  # Prawa góra
        {"pos": (1, 15), "id": 2},  # Lewy dół
        {"pos": (10, 10), "id": 3},  # Środek
        {"pos": (18, 8), "id": 4},  # Prawa strona
        {"pos": (5, 18), "id": 5},  # Dół
    ]
    DOORS = [
        {"pos": (14, 19), "key": 0},
        {"pos": (15, 19), "key": 1},
        {"pos": (16, 19), "key": 2},
        {"pos": (17, 19), "key": 3},
        {"pos": (18, 19), "key": 4},
        {"pos": (19, 19), "key": 5},
    ]
    TELEPORTS = [
        {"p1": (0, 0), "p2": (10, 11)},
        {"p1": (19, 0), "p2": (1, 19)},
    ]
    START = (0, 0)
    GOAL = (19, 19)

    def __init__(self, render_mode=None, maze_seed=42):
        super().__init__()
        self.render_mode = render_mode
        self.maze_seed   = maze_seed
        self.walls       = generate_maze(GRID, GRID, seed=maze_seed)

        for door in self.DOORS:
            dx, dy = door["pos"]

        self.observation_space = spaces.MultiDiscrete(
            [GRID, GRID, 2 ** self.N_KEYS]
        )
        self.action_space = spaces.Discrete(4)

        # pygame
        self._screen = None
        self._clock  = None
        self._font_big   = None
        self._font_small = None

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _get_obs(self):
        return np.array([self._px, self._py, self._keys], dtype=np.int64)

    def _key_at(self, x, y):
        for k in self.KEYS:
            if k["pos"] == (x, y) and not (self._keys & (1 << k["id"])):
                return k
        return None

    def _door_at(self, x, y):
        for d in self.DOORS:
            if d["pos"] == (x, y):
                return d
        return None

    def _tp_at(self, x, y):
        for t in self.TELEPORTS:
            if t["p1"] == (x, y):
                return t["p2"]
            elif t["p2"] == (x, y):
                return t["p1"]
        return None

    def _wall_between(self, x, y, dx, dy):
        """Czy między (x,y) a (x+dx, y+dy) jest ściana labiryntu?"""
        bit = DIR_WALL[(dx, dy)]
        return bool(self.walls[y][x] & bit)

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._px, self._py = self.START
        self._keys    = 0
        self._steps   = 0
        self._message = ""
        self._collected = []
        return self._get_obs(), {}

    def step(self, action):
        dx, dy = ACTIONS[action]
        nx, ny = self._px + dx, self._py + dy
        self._steps += 1
        self._message = ""

        reward      = -0.01   # kara za każdy krok
        terminated  = False
        truncated   = self._steps >= 2000

        # Sprawdź granicę
        if not (0 <= nx < GRID and 0 <= ny < GRID):
            return self._get_obs(), -0.05, False, truncated, {}

        # Sprawdź ścianę labiryntu
        if self._wall_between(self._px, self._py, dx, dy):
            return self._get_obs(), -0.05, False, truncated, {}

        # Sprawdź drzwi
        door = self._door_at(nx, ny)
        if door:
            if not (self._keys & (1 << door["key"])):
                self._message = f"Brak klucza K{door['key']+1}!"
                return self._get_obs(), -0.1, False, truncated, {}

        # Ruch zaakceptowany
        self._px, self._py = nx, ny

        # Zbierz klucz
        key = self._key_at(self._px, self._py)
        if key:
            self._keys |= (1 << key["id"])
            self._collected.append(key["id"])
            reward += 1.0
            self._message = f"Zebrałeś klucz K{key['id']+1}!"

        # Teleport
        tp_dest = self._tp_at(self._px, self._py)
        if tp_dest:
            self._px, self._py = tp_dest
            self._message = f"Teleport → ({self._px},{self._py})"

        # Cel
        if (self._px, self._py) == self.GOAL:
            reward     += 20.0
            terminated  = True
            self._message = "Dotarłeś do celu!"

        return self._get_obs(), reward, terminated, truncated, {}

    # ── Render ────────────────────────────────────────────────────────────────
    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return

        if self._screen is None:
            pygame.init()
            pygame.font.init()
            pygame.display.set_caption("Maze Gymnasium — PW4")
            if self.render_mode == "human":
                self._screen = pygame.display.set_mode((WIN_W, WIN_H))
            else:
                self._screen = pygame.Surface((WIN_W, WIN_H))
            self._clock      = pygame.time.Clock()
            self._font_big   = pygame.font.SysFont("monospace", 18, bold=True)
            self._font_small = pygame.font.SysFont("monospace", 13)

        surf = self._screen
        surf.fill(BLACK)

        # ── Siatka ──
        for gy in range(GRID):
            for gx in range(GRID):
                rx, ry = gx * CELL, gy * CELL
                pygame.draw.rect(surf, FLOOR, (rx+1, ry+1, CELL-2, CELL-2))

                w = self.walls[gy][gx]
                thick = 4
                if w & N: pygame.draw.line(surf, WALL, (rx, ry),          (rx+CELL, ry),          thick)
                if w & E: pygame.draw.line(surf, WALL, (rx+CELL, ry),     (rx+CELL, ry+CELL),     thick)
                if w & S: pygame.draw.line(surf, WALL, (rx, ry+CELL),     (rx+CELL, ry+CELL),     thick)
                if w & W: pygame.draw.line(surf, WALL, (rx, ry),          (rx, ry+CELL),           thick)

        # ── Cel ──
        gx, gy = self.GOAL
        star_surf = pygame.font.SysFont("segoe ui emoji", 34).render("★", True, GOAL)
        surf.blit(star_surf, (gx*CELL + 14, gy*CELL + 12))

        # ── Teleporty ──
        for tp in self.TELEPORTS:
            for pos in (tp["p1"], tp["p2"]):
                tx, ty = pos
                pygame.draw.circle(surf, TP_IN, (tx * CELL + CELL // 2, ty * CELL + CELL // 2), CELL // 2 - 6, 3)
                lbl = self._font_small.render("⭕"[0], True, TP_IN)
                surf.blit(lbl, (tx * CELL + CELL // 2 - 6, ty * CELL + CELL // 2 - 8))

        # ── Klucze ──
        for k in self.KEYS:
            if not (self._keys & (1 << k["id"])):
                kx, ky = k["pos"]
                col = KEY_COLORS[k["id"]]
                pygame.draw.polygon(surf, col, [
                    (kx*CELL+CELL//2, ky*CELL+12),
                    (kx*CELL+CELL-12, ky*CELL+CELL//2),
                    (kx*CELL+CELL//2, ky*CELL+CELL-12),
                    (kx*CELL+12,      ky*CELL+CELL//2),
                ])
                lbl = self._font_small.render(f"K{k['id']+1}", True, BLACK)
                surf.blit(lbl, (kx*CELL+CELL//2-10, ky*CELL+CELL//2-7))

        # ── Drzwi ──
        for d in self.DOORS:
            dx, dy = d["pos"]
            col  = DOOR_COLORS[d["key"]]
            unlocked = bool(self._keys & (1 << d["key"]))
            if not unlocked:
                pygame.draw.rect(surf, col, (dx*CELL+4, dy*CELL+4, CELL-8, CELL-8), 4, border_radius=6)
                lbl = self._font_small.render(f"K{d['key']+1}", True, col)
                surf.blit(lbl, (dx*CELL+CELL//2-8, dy*CELL+CELL//2-7))
            else:
                pygame.draw.rect(surf, col, (dx*CELL+4, dy*CELL+4, CELL-8, CELL-8), 1, border_radius=6)
                lbl = self._font_small.render("OPEN", True, col)
                surf.blit(lbl, (dx*CELL+CELL//2-14, dy*CELL+CELL//2-7))

        # ── Gracz ──
        px, py = self._px, self._py
        pygame.draw.circle(surf, PLAYER, (px*CELL+CELL//2, py*CELL+CELL//2), CELL//2-8)
        pygame.draw.circle(surf, WHITE,  (px*CELL+CELL//2, py*CELL+CELL//2), CELL//2-8, 2)

        # ── HUD ───────────────────────────────────────────────────────────────
        hx = GRID * CELL
        pygame.draw.rect(surf, HUD_BG, (hx, 0, HUD_W, WIN_H))
        pygame.draw.line(surf, GRAY, (hx, 0), (hx, WIN_H), 1)

        def hud_text(txt, x, y, col=WHITE, font=None):
            f = font or self._font_small
            surf.blit(f.render(txt, True, col), (x, y))

        hud_text("MAZE ENV", hx+10, 14, WHITE, self._font_big)
        hud_text(f"Projekt 4 — Gymnasium", hx+10, 36, GRAY)

        pygame.draw.line(surf, GRAY, (hx+10, 56), (hx+HUD_W-10, 56), 1)

        hud_text("POZYCJA", hx+10, 66, GRAY)
        hud_text(f"({self._px}, {self._py})", hx+10, 84, WHITE, self._font_big)

        hud_text("KROKI", hx+10, 112, GRAY)
        hud_text(str(self._steps), hx+10, 130, WHITE, self._font_big)

        hud_text("STAN (obs)", hx+10, 158, GRAY)
        bitmask = format(self._keys, f"0{self.N_KEYS}b")
        hud_text(f"x={self._px} y={self._py}", hx+10, 176, WHITE)
        hud_text(f"keys={bitmask}", hx+10, 194, WHITE)
        total = GRID * GRID * (2**self.N_KEYS)
        state_id = self._py * GRID * (2**self.N_KEYS) + self._px * (2**self.N_KEYS) + self._keys
        hud_text(f"id: {state_id}/{total}", hx+10, 212, (120, 120, 140))

        pygame.draw.line(surf, GRAY, (hx + 10, 232), (hx + HUD_W - 10, 232), 1)
        hud_text("KLUCZE", hx + 10, 242, GRAY)
        for i in range(self.N_KEYS):
            row = i // 3
            col_idx = i % 3
            has = bool(self._keys & (1 << i))
            color = KEY_COLORS[i] if has else (50, 52, 60)

            kx = hx + 10 + col_idx * 65
            ky = 260 + row * 35

            pygame.draw.rect(surf, color, (kx, ky, 55, 28), border_radius=4)
            if not has:
                pygame.draw.rect(surf, (70, 72, 82), (kx, ky, 55, 28), 1, border_radius=4)

            lbl = self._font_small.render(f"K{i + 1}", True, BLACK if has else (100, 102, 112))
            surf.blit(lbl, (kx + 15, ky + 7))

        pygame.draw.line(surf, GRAY, (hx+10, 340), (hx+HUD_W-10, 340), 1)
        hud_text("LEGENDA", hx+10, 350, GRAY)
        legends = [
            (KEY_COLORS[0], "◆ klucz"),
            (DOOR_COLORS[0], "▣ drzwi"),
            (TP_IN,   "○ teleport"),
            (GOAL,    "★ cel"),
            (PLAYER,  "● gracz"),
        ]
        for i, (col, txt) in enumerate(legends):
            hud_text(txt, hx+10, 370 + i*18, col)

        if self._message:
            pygame.draw.line(surf, GRAY, (hx+10, WIN_H-70), (hx+HUD_W-10, WIN_H-70), 1)
            lines = [self._message[j:j+22] for j in range(0, len(self._message), 22)]
            for i, line in enumerate(lines[:3]):
                hud_text(line, hx+10, WIN_H-55 + i*16, (200, 230, 170))

        if self.render_mode == "human":
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None
