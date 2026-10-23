"""
ObstacleReversiEnv — Reversi / Othello variant with obstacles.

State matrix (N×N NumPy int8):
    -1 = Player 1 (Black)      0 = Empty
     1 = Player 2 (White)      2 = Obstacle
"""

import os
import warnings
import numpy as np
from typing import Tuple, Optional

import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame

    _HAS_PYGAME = True
except ImportError:
    _HAS_PYGAME = False

# ---------------------------------------------------------------------------
# 8 compass directions as (dr, dc) offsets
# ---------------------------------------------------------------------------
_DIRECTIONS = np.array(
    [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]],
    dtype=np.int8,
)


class ObstacleReversiEnv(gym.Env):
    """Reversi with obstacles, driven by a NumPy integer board matrix.

    Compatible with the Gymnasium API: ``reset()`` returns ``(obs, info)``,
    ``step()`` returns ``(obs, reward, terminated, truncated, info)``.
    Internal attributes (``.board``, ``.current_player``, ``.done``, etc.)
    are kept public for MCTS / self-play direct access.
    """

    EMPTY = 0
    PLAYER1 = -1
    PLAYER2 = 1
    OBSTACLE = 2

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, N: int = 8, obstacle_ratio: float = 0.1):
        if N < 4:
            raise ValueError("Board size N must be >= 4")
        if not 0.0 <= obstacle_ratio < 1.0:
            raise ValueError("obstacle_ratio must be in the range [0.0, 1.0)")
        self.N = N
        self.obstacle_ratio = obstacle_ratio

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=-1, high=2, shape=(N, N), dtype=np.int8
        )
        self.action_space = spaces.Discrete(N * N + 1)  # 0..N²-1 board, N² = pass

        self.board: np.ndarray = None  # (N, N) int8
        self.current_player: int = None
        self.done: bool = False
        self.last_move: Optional[Tuple[int, int]] = None
        self.consecutive_passes: int = 0

        # Pygame state (lazy-init in render())
        self._screen = None
        self._cell_size: int = 0
        self._margin: int = 0
        self._info_h: int = 0

        self.reset()

    # ------------------------------------------------------------------
    # Board reset & obstacle generation
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        """Reset to initial board. Returns ``(observation, info)`` per Gymnasium API."""
        super().reset(seed=seed)
        self.board = np.zeros((self.N, self.N), dtype=np.int8)

        # Classic centre-4 cross
        m = self.N // 2
        self.board[m - 1, m - 1] = self.PLAYER2
        self.board[m - 1, m] = self.PLAYER1
        self.board[m, m - 1] = self.PLAYER1
        self.board[m, m] = self.PLAYER2

        # Random obstacles on ~obstacle_ratio of the remaining empty cells
        empty_idx = np.argwhere(self.board == self.EMPTY)
        n_obs = int(len(empty_idx) * self.obstacle_ratio)
        if n_obs > 0:
            chosen_idx = self.np_random.choice(len(empty_idx), n_obs, replace=False)
            chosen = empty_idx[chosen_idx]
            self.board[chosen[:, 0], chosen[:, 1]] = self.OBSTACLE

        self.current_player = self.PLAYER1
        self.done = False
        self.last_move = None
        self.consecutive_passes = 0
        return self.board.copy(), {}

    # ------------------------------------------------------------------
    # Step — action → ray-flip → turn advance
    # ------------------------------------------------------------------

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute a move or pass.

        Parameters
        ----------
        action : int
            -1              → pass (no legal move for current player).
            [0, N²-1]       → drop a piece at ``divmod(action, N)``.

        Returns ``(observation, reward, terminated, truncated, info)``.
        Raises ``ValueError`` on illegal moves.

        Terminal condition
        ------------------
        The game ends once neither player has a legal move. Consecutive passes
        are still tracked for callers that model pass turns explicitly.
        """
        if self.done:
            raise ValueError("Cannot step after the game is over; call reset() first")

        # Gymnasium action space: N² maps to pass, 0..N²-1 to board cells
        if action == self.N * self.N:
            action = -1

        # ── Pass ─────────────────────────────────────────────
        if action == -1:
            if self.get_legal_actions(self.current_player).any():
                raise ValueError("Cannot pass while legal moves are available")
            self.consecutive_passes += 1
            self.current_player *= -1
            if self.consecutive_passes >= 2 or self.is_terminal_position():
                self.done = True
            reward = self._compute_reward()
            return self.board.copy(), reward, self.done, False, {}

        # ── Normal move ──────────────────────────────────────
        r, c = divmod(action, self.N)

        if not (0 <= r < self.N and 0 <= c < self.N):
            raise ValueError(
                f"Action {action} is out of bounds for {self.N}×{self.N} board"
            )
        if self.board[r, c] != self.EMPTY:
            raise ValueError(
                f"Cell ({r},{c}) is occupied (value={self.board[r, c]})"
            )
        if not self._ray_check(r, c, self.current_player):
            raise ValueError(
                f"Illegal move at ({r},{c}); move must flip at least one disc"
            )

        self.consecutive_passes = 0
        player = self.current_player
        self.board[r, c] = player
        self._flip_rays(r, c, player)
        self.last_move = (r, c)

        # Always hand control to opponent; caller must detect no-legal-moves
        # and call step(-1) for the opponent's pass.
        self.current_player = -player
        if self.is_terminal_position():
            self.done = True

        reward = self._compute_reward()
        return self.board.copy(), reward, self.done, False, {}

    def is_terminal_position(self) -> bool:
        """True when neither player has a legal move."""
        return (
            not self.get_legal_actions(self.PLAYER1).any()
            and not self.get_legal_actions(self.PLAYER2).any()
        )

    def _compute_reward(self) -> float:
        """Compute terminal reward from the perspective of the player who just acted.

        Returns +1 for win, -1 for loss, 0 otherwise.
        """
        if not self.done:
            return 0.0
        p1, p2 = self.get_score()
        # After step(), current_player has already switched.
        # The player who just acted is -current_player.
        last_actor = -self.current_player
        if p1 == p2:
            return 0.0
        winner = self.PLAYER1 if p1 > p2 else self.PLAYER2
        return 1.0 if last_actor == winner else -1.0

    def _flip_rays(self, r: int, c: int, player: int):
        """Flip opponent discs along every direction that is bracketed."""
        for dr, dc in _DIRECTIONS:
            bracket = []
            nr, nc = r + int(dr), c + int(dc)
            while 0 <= nr < self.N and 0 <= nc < self.N:
                cell = self.board[nr, nc]
                if cell == self.EMPTY or cell == self.OBSTACLE:
                    break
                if cell == player:
                    for br, bc in bracket:
                        self.board[br, bc] = player
                    break
                if cell == -player:
                    bracket.append((nr, nc))
                nr += int(dr)
                nc += int(dc)

    # ------------------------------------------------------------------
    # Legal-action mask
    # ------------------------------------------------------------------

    def get_legal_actions(self, player: int) -> np.ndarray:
        """Boolean mask (N×N), True where *player* can legally drop."""
        legal = np.zeros((self.N, self.N), dtype=bool)
        for r, c in np.argwhere(self.board == self.EMPTY):
            if self._ray_check(r, c, player):
                legal[r, c] = True
        return legal

    def _ray_check(self, r: int, c: int, player: int) -> bool:
        """True if dropping at (r,c) flips at least one opponent disc."""
        opp = -player
        for dr, dc in _DIRECTIONS:
            nr, nc = r + int(dr), c + int(dc)
            saw_opp = False
            while 0 <= nr < self.N and 0 <= nc < self.N:
                cell = self.board[nr, nc]
                if cell == self.EMPTY or cell == self.OBSTACLE:
                    break
                if cell == opp:
                    saw_opp = True
                elif cell == player:
                    if saw_opp:
                        return True
                    break
                nr += int(dr)
                nc += int(dc)
        return False

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def get_score(self) -> Tuple[int, int]:
        """Return (player1_discs, player2_discs)."""
        p1 = int(np.sum(self.board == self.PLAYER1))
        p2 = int(np.sum(self.board == self.PLAYER2))
        return p1, p2

    # ==================================================================
    #  Pygame rendering (independent, self-contained)
    # ==================================================================

    # Visual constants
    _BG_COLOR = (34, 100, 34)
    _CELL_LIGHT = (46, 139, 87)
    _CELL_DARK = (38, 125, 78)
    _GRID_COLOR = (0, 0, 0)
    _HIGHLIGHT = (255, 215, 0)  # last-move dot

    def render(self, cell_size: int = 80, show_legal: bool = True, flip: bool = True):
        """Draw the board in a Pygame window.

        Creates the window on first call; subsequent calls redraw in-place.
        Set *flip=False* to defer ``pygame.display.flip()`` (e.g. to draw
        an overlay on top before flipping).
        """
        if not _HAS_PYGAME:
            raise ImportError(
                "Pygame is required for rendering.  Install:  pip install pygame"
            )

        if self._screen is None:
            self._cell_size = cell_size
            self._margin = 40
            self._info_h = 60
            w = self.N * cell_size + 2 * self._margin
            h = self.N * cell_size + 2 * self._margin + self._info_h
            pygame.init()
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Obstacle Reversi")

        self._draw_frame(cell_size, show_legal, flip)

    # ------------------------------------------------------------------
    # Frame drawing helpers
    # ------------------------------------------------------------------

    def _draw_frame(self, cell_size: int, show_legal: bool, flip: bool = True):
        self._screen.fill(self._BG_COLOR)
        self._draw_info_bar()
        legal = self.get_legal_actions(self.current_player) if show_legal else None
        for r in range(self.N):
            for c in range(self.N):
                x = self._margin + c * cell_size
                y = self._info_h + r * cell_size
                self._draw_cell(r, c, x, y, cell_size, legal)
        if flip:
            pygame.display.flip()

    def _draw_info_bar(self):
        font = pygame.font.Font(None, 36)
        p1, p2 = self.get_score()
        if self.done:
            if p1 > p2:
                msg = f"Game Over — Black wins!     Black: {p1}   White: {p2}"
            elif p2 > p1:
                msg = f"Game Over — White wins!     Black: {p1}   White: {p2}"
            else:
                msg = f"Game Over — Draw.           Black: {p1}   White: {p2}"
        else:
            turn = "Black" if self.current_player == self.PLAYER1 else "White"
            msg = f"{turn}'s turn     Black: {p1}   White: {p2}"
        surf = font.render(msg, True, (255, 255, 255))
        self._screen.blit(surf, (self._margin, 15))

    def _draw_cell(
        self,
        row: int,
        col: int,
        x: int,
        y: int,
        size: int,
        legal_mask: Optional[np.ndarray],
    ):
        """Draw a single board cell and its contents."""
        # Checkerboard background
        bg = self._CELL_LIGHT if (row + col) % 2 == 0 else self._CELL_DARK
        pygame.draw.rect(self._screen, bg, (x, y, size, size))
        pygame.draw.rect(self._screen, self._GRID_COLOR, (x, y, size, size), 1)

        cx, cy = x + size // 2, y + size // 2
        val = self.board[row, col]
        r = size // 2 - 5

        # --- Pieces / obstacles (geometric + colour distinction) ---
        if val == self.PLAYER1:  # Black disc
            pygame.draw.circle(self._screen, (30, 30, 30), (cx, cy), r)
            pygame.draw.circle(self._screen, (0, 0, 0), (cx, cy), r, 2)
        elif val == self.PLAYER2:  # White disc with black rim
            pygame.draw.circle(self._screen, (240, 240, 240), (cx, cy), r)
            pygame.draw.circle(self._screen, (0, 0, 0), (cx, cy), r, 2)
        elif val == self.OBSTACLE:  # Brown square with diagonal cross
            pad = 4
            pygame.draw.rect(
                self._screen,
                (101, 67, 33),
                (x + pad, y + pad, size - 2 * pad, size - 2 * pad),
            )
            off = 8
            pygame.draw.line(
                self._screen,
                (60, 40, 20),
                (x + off, y + off),
                (x + size - off, y + size - off),
                3,
            )
            pygame.draw.line(
                self._screen,
                (60, 40, 20),
                (x + size - off, y + off),
                (x + off, y + size - off),
                3,
            )

        # Last-move highlight
        if self.last_move is not None and self.last_move == (row, col):
            pygame.draw.circle(self._screen, self._HIGHLIGHT, (cx, cy), 5)

        # Legal-move indicator (semi-transparent green dot)
        if legal_mask is not None and not self.done and val == self.EMPTY:
            if legal_mask[row, col]:
                dot_r = max(size // 7, 5)
                dot = pygame.Surface((dot_r * 2, dot_r * 2), pygame.SRCALPHA)
                pygame.draw.circle(dot, (0, 200, 0, 160), (dot_r, dot_r), dot_r)
                self._screen.blit(dot, (cx - dot_r, cy - dot_r))

    # ------------------------------------------------------------------
    # Mouse → action mapping
    # ------------------------------------------------------------------

    def pixel_to_action(self, mx: int, my: int, cell_size: int = None) -> Optional[int]:
        """Map a pixel coordinate to an action index, or *None* if outside the board."""
        cs = cell_size or self._cell_size or 80
        col = (mx - self._margin) // cs
        row = (my - self._info_h) // cs
        if 0 <= row < self.N and 0 <= col < self.N:
            return int(row * self.N + col)
        return None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the Pygame window."""
        if _HAS_PYGAME and self._screen is not None:
            pygame.quit()
            self._screen = None


# ===========================================================================
#  Demo — human vs. human (click to place, R to reset)
# ===========================================================================
if __name__ == "__main__":
    if not _HAS_PYGAME:
        print("Pygame not installed.  Run:  pip install pygame")
        raise SystemExit(1)

    env = ObstacleReversiEnv(N=8, obstacle_ratio=0.1)
    CS = 80
    env.render(CS)

    clock = pygame.time.Clock()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and not env.done:
                action = env.pixel_to_action(*event.pos, CS)
                if action is not None:
                    r, c = divmod(action, env.N)
                    if env.get_legal_actions(env.current_player)[r, c]:
                        env.step(action)
                        env.render(CS)

            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                env.reset()
                env.render(CS)

        clock.tick(30)

    env.close()


# Register with gymnasium so users can `gym.make("ObstacleReversi-v0")`
gym.register(
    id="ObstacleReversi-v0",
    entry_point="obstacle_reversi_env:ObstacleReversiEnv",
)
