"""
main.py — Human-vs-AI ObstacleReversi desktop application.

Threading model
──────────────
- **Main thread**  — 60-FPS Pygame event loop + rendering.
- **AI thread**    — daemon thread running MCTS ``get_action_with_time_budget``.
- **Queue**        — thread-safe hand-off of the chosen action.
- **Lock**         — protects ``env.board`` / ``env.current_player`` during
                     the brief moments MCTS mutates them inside ``_expand``;
                     the network forward pass runs *outside* the lock.

Controls
────────
- Click a highlighted cell to place a disc.
- **R**  — reset the game (disabled while AI is thinking).
- **Esc** — quit.
"""

from __future__ import annotations

import os
import sys
import time
import threading
import queue
import warnings

import numpy as np

# Suppress pkg_resources deprecation warning from pygame internals
warnings.filterwarnings("ignore", message=".*pkg_resources.*")

import torch
import pygame

from obstacle_reversi_env import ObstacleReversiEnv
from alpha_reversi_net import AlphaReversiNet
from mcts import MCTS

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

BOARD_SIZE = 8
CELL_SIZE = 80
OBSTACLE_RATIO = 0.1

AI_PLAYER = 1          # White = AI;  set to -1 for Black AI
AI_TIME_LIMIT = 3.0    # seconds per move
AI_NET_CHANNELS = 128
AI_C_PUCT = 1.4
AI_NUM_RES_BLOCKS = 4
MODEL_PATH = os.environ.get("MODEL_PATH", "")

FPS = 60

# Layout (must match the hard-coded values in ObstacleReversiEnv.render)
MARGIN = 40
INFO_H = 60

# ═══════════════════════════════════════════════════════════════════════════════
# AI worker (runs in background thread)
# ═══════════════════════════════════════════════════════════════════════════════


def _ai_worker(
    mcts: MCTS,
    board: np.ndarray,
    player: int,
    consecutive_passes: int,
    time_limit: float,
    result_queue: queue.Queue,
) -> None:
    """Run MCTS search and post ``("ok", action, iters)`` or ``("error", msg)``."""
    try:
        action, _root, iters = mcts.get_action_with_time_budget(
            board, player, time_limit, consecutive_passes
        )
        result_queue.put(("ok", action, iters))
    except Exception as exc:
        result_queue.put(("error", str(exc)))


# ═══════════════════════════════════════════════════════════════════════════════
# UI helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _draw_thinking_overlay(
    screen: pygame.Surface,
    board_px: int,
    start_time: float,
    last_iters: int,
) -> None:
    """Semi-transparent overlay with animated 'AI thinking …' text."""
    elapsed = time.time() - start_time
    dots = "." * (int(elapsed * 1.5) % 4)

    # Dark scrim over the board area
    scrim = pygame.Surface((board_px, board_px), pygame.SRCALPHA)
    scrim.fill((0, 0, 0, 140))
    screen.blit(scrim, (MARGIN, INFO_H))

    # Main label
    font = pygame.font.Font(None, 44)
    text = font.render(f"AI thinking{dots}   ", True, (255, 255, 255))
    rect = text.get_rect(
        center=(MARGIN + board_px // 2, INFO_H + board_px // 2)
    )
    pill = rect.inflate(36, 20)
    pygame.draw.rect(screen, (25, 25, 35), pill, border_radius=12)
    screen.blit(text, rect)

    # Iteration count from the *previous* search (informational)
    if last_iters > 0:
        small = pygame.font.Font(None, 22)
        info = small.render(
            f"Last search: {last_iters} iterations", True, (180, 180, 180)
        )
        screen.blit(info, (MARGIN + 6, INFO_H + board_px + 8))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # ── detect board size from checkpoint (if available) ───────────────────
    board_size = BOARD_SIZE
    net_channels = AI_NET_CHANNELS
    net_res_blocks = AI_NUM_RES_BLOCKS

    if MODEL_PATH and os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            board_size = checkpoint.get("N", BOARD_SIZE)
            net_channels = checkpoint.get("num_channels", AI_NET_CHANNELS)
            net_res_blocks = checkpoint.get("num_res_blocks", AI_NUM_RES_BLOCKS)
        else:
            state_dict = checkpoint
        if board_size != BOARD_SIZE:
            print(f"[INFO] Checkpoint N={board_size} overrides default N={BOARD_SIZE}")
        print(f"[OK] Loaded trained model: {MODEL_PATH}  "
              f"(N={board_size}, ch={net_channels}, res={net_res_blocks})")
    else:
        state_dict = None
        print("[WARNING] No trained model — AI will play with random weights")

    # ── init env ───────────────────────────────────────────────────────────
    env = ObstacleReversiEnv(N=board_size, obstacle_ratio=OBSTACLE_RATIO)

    # ── init network ───────────────────────────────────────────────────────
    net = AlphaReversiNet(
        N=board_size, num_channels=net_channels,
        num_res_blocks=net_res_blocks,
    )
    if state_dict is not None:
        net.load_state_dict(state_dict)
    net.eval()

    # ── init MCTS (with a lock shared with the main thread) ────────────────
    env_lock = threading.Lock()
    mcts = MCTS(env, net, c_puct=AI_C_PUCT, device="cpu", env_lock=env_lock)

    # ── init Pygame window ─────────────────────────────────────────────────
    env.render(CELL_SIZE, show_legal=True, flip=True)
    clock = pygame.time.Clock()

    # ── mutable state ──────────────────────────────────────────────────────
    ai_thinking = False
    result_queue: queue.Queue = queue.Queue()
    thinking_start = 0.0
    last_ai_iters = 0

    human_player = -AI_PLAYER
    board_px = board_size * CELL_SIZE

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # ==================================================================
        # 1 — Event processing
        # ==================================================================
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # ---- mouse click ---------------------------------------------
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Ignore when game is over, AI is computing, or not human's turn
                if env.done or ai_thinking:
                    continue
                if env.current_player != human_player:
                    continue

                action = env.pixel_to_action(*event.pos, CELL_SIZE)
                if action is None:
                    continue

                r, c = divmod(action, env.N)
                legal = env.get_legal_actions(env.current_player)
                if not legal[r, c]:
                    continue  # illegal click → silently ignored

                env.step(action)
                mcts.advance_root(action)

            # ---- keyboard ------------------------------------------------
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    if not ai_thinking:              # block reset during AI turn
                        env.reset()
                        while not result_queue.empty():
                            result_queue.get_nowait()
                        last_ai_iters = 0

                elif event.key == pygame.K_ESCAPE:
                    running = False

        # ==================================================================
        # 2 — Auto-pass when current player has no legal moves
        # ==================================================================
        if not env.done and not ai_thinking:
            if not env.get_legal_actions(env.current_player).any():
                env.step(-1)
                mcts.advance_root(-1)
                continue  # skip rest of loop — next iteration will render & re-check

        # ==================================================================
        # 3 — Trigger AI thread when it's the AI's turn
        # ==================================================================
        if (
            not env.done
            and not ai_thinking
            and env.current_player == AI_PLAYER
        ):
            ai_thinking = True
            thinking_start = time.time()
            while not result_queue.empty():         # drain stale results
                result_queue.get_nowait()

            t = threading.Thread(
                target=_ai_worker,
                args=(
                    mcts,
                    env.board.copy(),
                    env.current_player,
                    env.consecutive_passes,
                    AI_TIME_LIMIT,
                    result_queue,
                ),
                daemon=True,
            )
            t.start()

        # ==================================================================
        # 4 — Collect AI result (non-blocking poll)
        # ==================================================================
        if ai_thinking:
            try:
                status, *data = result_queue.get_nowait()
                if status == "ok":
                    action, iters = data
                    last_ai_iters = iters
                    env.step(action)                # apply on main thread
                    mcts.advance_root(action)
                else:
                    print(f"[AI error] {data[0]}")
                ai_thinking = False
            except queue.Empty:
                pass

        # ==================================================================
        # 5 — Render (lock protects against concurrent MCTS env writes)
        # ==================================================================
        show_legal = (
            not env.done
            and not ai_thinking
            and env.current_player == human_player
        )

        with env_lock:
            env.render(CELL_SIZE, show_legal=show_legal, flip=False)

        # ==================================================================
        # 6 — AI-thinking overlay
        # ==================================================================
        if ai_thinking:
            _draw_thinking_overlay(
                env._screen, board_px, thinking_start, last_ai_iters
            )

        # ==================================================================
        # 7 — Single flip per frame
        # ==================================================================
        pygame.display.flip()

    env.close()
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()
