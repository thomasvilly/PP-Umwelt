from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import List

import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np


MAX_SIZE = 13
OBS_DIM = 3 * MAX_SIZE * MAX_SIZE + 1  # 507 spatial + 1 BFS boolean hint

CURRICULUM_CONFIGS = {
    0: {"size": 5,  "n_walls": 1, "dynamic": False},
    1: {"size": 7,  "n_walls": 1, "dynamic": False},
    2: {"size": 11,  "n_walls": 1, "dynamic": False},
    3: {"size": 13, "n_walls": 1, "dynamic": False},
    4: {"size": 13, "n_walls": 6, "dynamic": True},
}


@dataclass
class Wall:
    fixed_pos: int    # row (H) or col (V) that stays constant
    start: int        # first index along the wall's axis (inclusive)
    end: int          # last index along the wall's axis (inclusive)
    direction: str    # "H" (horizontal) or "V" (vertical)
    is_dynamic: bool = False

    def cells(self):
        """Returns list of (x, y) = (col, row) tuples blocked by this wall."""
        if self.direction == "H":
            # fixed_pos is the row (y); wall spans columns (x)
            return [(x, self.fixed_pos) for x in range(self.start, self.end + 1)]
        else:
            # fixed_pos is the col (x); wall spans rows (y)
            return [(self.fixed_pos, y) for y in range(self.start, self.end + 1)]


class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, level=0):
        cfg = CURRICULUM_CONFIGS[level]
        self.level = level
        self.size = cfg["size"]
        self._n_walls = cfg["n_walls"]
        self._dynamic = cfg["dynamic"]
        self.window_size = 512

        # Fixed obs space for all curriculum levels: 3 channels × 12×12, flattened
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            Actions.right.value: np.array([1, 0]),
            Actions.up.value: np.array([0, 1]),
            Actions.left.value: np.array([-1, 0]),
            Actions.down.value: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        self._walls: List[Wall] = []
        self._wall_mask = np.zeros((MAX_SIZE, MAX_SIZE), dtype=np.float32)
        self._step_count = 0
        self._optimal_path_length = -1
        self._current_bfs_dist: int = 0

    def _get_obs(self):
        agent_ch = np.zeros((MAX_SIZE, MAX_SIZE), dtype=np.float32)
        goal_ch  = np.zeros((MAX_SIZE, MAX_SIZE), dtype=np.float32)
        ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
        tx, ty = int(self._target_location[0]), int(self._target_location[1])
        agent_ch[ay, ax] = 1.0
        goal_ch[ty, tx]  = 1.0
        manhattan = abs(ax - tx) + abs(ay - ty)
        wall_between = np.array([float(self._current_bfs_dist > manhattan)], dtype=np.float32)
        return np.concatenate([agent_ch.ravel(), goal_ch.ravel(), self._wall_mask.ravel(), wall_between])

    def _get_info(self):
        return {
            "distance": float(np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )),
            "optimal_path_length": self._optimal_path_length,
        }

    def _build_wall_mask(self) -> np.ndarray:
        mask = np.zeros((MAX_SIZE, MAX_SIZE), dtype=np.float32)
        for wall in self._walls:
            for (x, y) in wall.cells():
                if 0 <= x < self.size and 0 <= y < self.size:
                    mask[y, x] = 1.0
        return mask

    def _bfs_solvable(self, wall_mask: np.ndarray) -> bool:
        sx, sy = int(self._agent_location[0]), int(self._agent_location[1])
        gx, gy = int(self._target_location[0]), int(self._target_location[1])
        visited = np.zeros((self.size, self.size), dtype=bool)
        queue = deque([(sx, sy)])
        visited[sy, sx] = True
        while queue:
            cx, cy = queue.popleft()
            if cx == gx and cy == gy:
                return True
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if not visited[ny, nx] and wall_mask[ny, nx] == 0:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
        return False

    def _bfs_path_length(self) -> int:
        """Returns shortest path length from agent to goal, or -1 if unreachable."""
        sx, sy = int(self._agent_location[0]), int(self._agent_location[1])
        gx, gy = int(self._target_location[0]), int(self._target_location[1])
        visited = np.zeros((self.size, self.size), dtype=bool)
        queue = deque([(sx, sy, 0)])
        visited[sy, sx] = True
        while queue:
            cx, cy, dist = queue.popleft()
            if cx == gx and cy == gy:
                return dist
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if not visited[ny, nx] and self._wall_mask[ny, nx] == 0:
                        visited[ny, nx] = True
                        queue.append((nx, ny, dist + 1))
        return -1

    def _generate_walls(self) -> List[Wall]:
        if self._n_walls == 0:
            return []
        max_len = max(1, self.size // 3)
        for _ in range(200):
            walls = []
            for _ in range(self._n_walls):
                direction = "H" if self.np_random.integers(0, 2) == 0 else "V"
                fixed_pos = int(self.np_random.integers(0, self.size))
                start = int(self.np_random.integers(0, self.size))
                length = int(self.np_random.integers(1, max_len + 1))
                end = min(start + length - 1, self.size - 1)
                walls.append(Wall(fixed_pos, start, end, direction, self._dynamic))

            temp_mask = np.zeros((MAX_SIZE, MAX_SIZE), dtype=np.float32)
            for w in walls:
                for (x, y) in w.cells():
                    if 0 <= x < self.size and 0 <= y < self.size:
                        temp_mask[y, x] = 1.0

            ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
            tx, ty = int(self._target_location[0]), int(self._target_location[1])
            if temp_mask[ay, ax] or temp_mask[ty, tx]:
                continue

            saved = self._wall_mask
            self._wall_mask = temp_mask
            solvable = self._bfs_solvable(temp_mask)
            self._wall_mask = saved
            if solvable:
                return walls

        return []  # fallback: no walls if no valid config found

    def _move_dynamic_walls(self):
        for wall in self._walls:
            if not wall.is_dynamic:
                continue
            roll = int(self.np_random.integers(0, 3))  # 0=stay, 1=+1, 2=-1
            if roll == 0:
                continue
            delta = 1 if roll == 1 else -1
            old_fp = wall.fixed_pos
            new_fp = wall.fixed_pos + delta
            if new_fp < 0 or new_fp >= self.size:
                continue
            wall.fixed_pos = new_fp
            new_mask = self._build_wall_mask()
            ax, ay = int(self._agent_location[0]), int(self._agent_location[1])
            tx, ty = int(self._target_location[0]), int(self._target_location[1])
            if new_mask[ay, ax] or new_mask[ty, tx]:
                wall.fixed_pos = old_fp
                continue
            saved = self._wall_mask
            self._wall_mask = new_mask
            if not self._bfs_solvable(new_mask):
                self._wall_mask = saved
                wall.fixed_pos = old_fp

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Sample target until it differs from agent
        self._target_location = self._agent_location.copy()
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # Generate walls (BFS uses current agent/goal positions)
        self._walls = self._generate_walls()
        self._wall_mask = self._build_wall_mask()
        self._optimal_path_length = self._bfs_path_length()
        self._current_bfs_dist = self._optimal_path_length

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        if self._dynamic:
            self._move_dynamic_walls()

        direction = self._action_to_direction[action]
        new_loc = np.clip(self._agent_location + direction, 0, self.size - 1)
        nx, ny = int(new_loc[0]), int(new_loc[1])
        # Only move if the destination is not a wall cell
        dist_before = float(np.linalg.norm(self._agent_location - self._target_location, ord=1))
        if not self._wall_mask[ny, nx]:
            self._agent_location = new_loc
        dist_after = float(np.linalg.norm(self._agent_location - self._target_location, ord=1))

        self._step_count += 1
        terminated = bool(np.array_equal(self._agent_location, self._target_location))
        shaping = (dist_before - dist_after) * 0.10
        reward = 1.0 if terminated else -0.01 + shaping
        truncated = self._step_count >= 4 * self.size * self.size
        self._current_bfs_dist = self._bfs_path_length()
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw walls
        for y in range(self.size):
            for x in range(self.size):
                if self._wall_mask[y, x]:
                    pygame.draw.rect(
                        canvas, (80, 80, 80),
                        pygame.Rect(pix_square_size * x, pix_square_size * y,
                                    pix_square_size, pix_square_size),
                    )

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to
            # keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
