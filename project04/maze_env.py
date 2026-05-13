import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import pygame.font

# ─── Kolory ───────────────────────────────────────────────────────────────────
BLACK   = (15,  15,  20)
WHITE   = (240, 238, 230)
GRAY    = (50,  52,  60)
WALL    = (100, 105, 120)
FLOOR   = (22,  24,  32)
PLAYER  = (80, 160, 255)
GOAL    = (120, 220, 100)
TP_IN   = (140, 130, 240)
HUD_BG  = (18,  20,  28)

KEY_COLORS =[
    (240, 180, 50), (100, 210, 180), (210, 110, 230), (230, 130, 100),
    (150, 200, 255), (180, 240, 100), (255, 120, 150), (50,  255, 200),
    (200, 150, 50), (150, 50,  255)
]
DOOR_COLORS =[
    (200, 140, 30), (60,  170, 140), (170, 70,  200), (190, 90,  60),
    (100, 150, 200), (130, 190, 70),  (200, 90,  120), (40,  200, 160),
    (160, 120, 40), (120, 40,  200)
]

# ─── Rozmiar siatki i okna ────────────────────────────────────────────────────
GRID    = 40
CELL    = 24
HUD_W   = 280
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

    for _ in range(400):
        x, y = rng.integers(0, cols), rng.integers(0, rows)
        dirs = [(0, -1, N, S), (1, 0, E, W), (0, 1, S, N), (-1, 0, W, E)]
        dx, dy, w, ow = dirs[rng.integers(0, 4)]
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            walls[y][x] &= ~w
            walls[ny][nx] &= ~ow

    for x in range(29, 40):
        walls[39][x] |= N
        walls[38][x] |= S

        if x > 0:
            walls[39][x] &= ~W
            walls[39][x - 1] &= ~E

    walls[39][29] &= ~N
    walls[38][29] &= ~S

    return walls

class MazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    N_KEYS = 10
    KEYS = [
        {"pos": (2, 2), "id": 0}, {"pos": (37, 2), "id": 1},
        {"pos": (2, 37), "id": 2}, {"pos": (20, 20), "id": 3},
        {"pos": (10, 30), "id": 4}, {"pos": (30, 10), "id": 5},
        {"pos": (15, 5), "id": 6}, {"pos": (5, 15), "id": 7},
        {"pos": (35, 25), "id": 8}, {"pos": (25, 35), "id": 9},
    ]
    DOORS = [
        {"pos": (30, 39), "key": 0}, {"pos": (31, 39), "key": 1},
        {"pos": (32, 39), "key": 2}, {"pos": (33, 39), "key": 3},
        {"pos": (34, 39), "key": 4}, {"pos": (35, 39), "key": 5},
        {"pos": (36, 39), "key": 6}, {"pos": (37, 39), "key": 7},
        {"pos": (38, 39), "key": 8}, {"pos": (39, 39), "key": 9},
    ]
    TELEPORTS = [
        {"p1": (5, 5), "p2": (35, 35)},
        {"p1": (5, 35), "p2": (35, 5)},
        {"p1": (20, 2), "p2": (20, 37)},
        {"p1": (2, 20), "p2": (37, 20)},
    ]
    START = (0, 0)
    GOAL = (39, 39)

    def __init__(self, render_mode=None, maze_seed=41):
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
                pygame.draw.rect(surf, FLOOR, (rx, ry, CELL, CELL))

                w = self.walls[gy][gx]
                thick = 2

                if w & N: pygame.draw.line(surf, WALL, (rx, ry), (rx + CELL, ry), thick)
                if w & E: pygame.draw.line(surf, WALL, (rx + CELL - 1, ry), (rx + CELL - 1, ry + CELL), thick)
                if w & S: pygame.draw.line(surf, WALL, (rx, ry + CELL - 1), (rx + CELL, ry + CELL - 1), thick)
                if w & W: pygame.draw.line(surf, WALL, (rx, ry), (rx, ry + CELL), thick)


        # ── Cel ──
        gx, gy = self.GOAL
        margin_g = max(2, int(CELL * 0.15))
        pygame.draw.rect(surf, GOAL,
                         (gx * CELL + margin_g, gy * CELL + margin_g, CELL - 2 * margin_g, CELL - 2 * margin_g))
        pygame.draw.rect(surf, WHITE,
                         (gx * CELL + margin_g, gy * CELL + margin_g, CELL - 2 * margin_g, CELL - 2 * margin_g), 2)

        # ── Teleporty ──
        for tp in self.TELEPORTS:
            for pos in (tp["p1"], tp["p2"]):
                tx, ty = pos
                cx, cy = tx * CELL + CELL // 2, ty * CELL + CELL // 2
                radius = int(CELL * 0.35)
                thickness = max(1, int(CELL * 0.1))
                pygame.draw.circle(surf, TP_IN, (cx, cy), radius, thickness)

        # ── Klucze ──
        for k in self.KEYS:
            if not (self._keys & (1 << k["id"])):
                kx, ky = k["pos"]
                col = KEY_COLORS[k["id"]]
                cx, cy = kx * CELL + CELL // 2, ky * CELL + CELL // 2
                margin = int(CELL * 0.15)
                points = [
                    (cx, ky * CELL + margin),  # góra
                    (kx * CELL + CELL - margin, cy),  # prawo
                    (cx, ky * CELL + CELL - margin),  # dół
                    (kx * CELL + margin, cy)  # lewo
                ]
                pygame.draw.polygon(surf, col, points)

        # ── Drzwi ──
        for d in self.DOORS:
            dx, dy = d["pos"]
            col = DOOR_COLORS[d["key"]]
            unlocked = bool(self._keys & (1 << d["key"]))
            margin = int(CELL * 0.1)
            rect = (dx * CELL + margin, dy * CELL + margin, CELL - 2 * margin, CELL - 2 * margin)

            if not unlocked:
                thickness = max(2, int(CELL * 0.2))
                pygame.draw.rect(surf, col, rect, thickness, border_radius=4)
            else:
                pygame.draw.rect(surf, col, rect, 1, border_radius=4)

        # ── Gracz ──
        px, py = self._px, self._py
        cx, cy = px * CELL + CELL // 2, py * CELL + CELL // 2
        p_radius = int(CELL * 0.4)
        pygame.draw.circle(surf, PLAYER, (cx, cy), p_radius)
        pygame.draw.circle(surf, WHITE, (cx, cy), p_radius, max(1, int(CELL * 0.05)))

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
            row = i // 5
            col_idx = i % 5
            has = bool(self._keys & (1 << i))
            color = KEY_COLORS[i] if has else (50, 52, 60)

            kx = hx + 10 + col_idx * 52
            ky = 260 + row * 35

            pygame.draw.rect(surf, color, (kx, ky, 45, 28), border_radius=4)
            if not has:
                pygame.draw.rect(surf, (70, 72, 82), (kx, ky, 45, 28), 1, border_radius=4)

            lbl = self._font_small.render(f"K{i + 1}", True, BLACK if has else (100, 102, 112))
            surf.blit(lbl, (kx + 10, ky + 7))

        pygame.draw.line(surf, GRAY, (hx + 10, 340), (hx + HUD_W - 10, 340), 1)
        hud_text("LEGENDA", hx + 10, 350, GRAY)

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
