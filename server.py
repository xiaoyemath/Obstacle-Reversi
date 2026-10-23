"""
server.py — FastAPI backend for ObstacleReversi with persistent MCTS tree.

Split-turn design
─────────────────
- ``POST /api/human_move``  — fast  (O(N) matrix update, ~ms).
- ``POST /api/ai_move``     — slow  (MCTS in thread pool, ~seconds).

Each session owns an ``ObstacleReversiEnv``, an ``MCTS`` searcher, and a
``tree_lock`` that serialises env-step + tree-advance operations.

Start:   uvicorn server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import uuid
import time
import threading
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from obstacle_reversi_env import ObstacleReversiEnv
from alpha_reversi_net import AlphaReversiNet
from mcts import MCTS

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

BOARD_SIZE = 8
OBSTACLE_RATIO = 0.1
AI_PLAYER = 1                # 1 = White (AI),  -1 = Black (AI)
AI_TIME_LIMIT = 2.0          # seconds per MCTS search
AI_C_PUCT = 1.4
NET_CHANNELS = 128
NET_RES_BLOCKS = 4
MODEL_PATH = os.environ.get("MODEL_PATH", "")

# ═══════════════════════════════════════════════════════════════════════════════
# Global state (initialised during lifespan)
# ═══════════════════════════════════════════════════════════════════════════════

model: AlphaReversiNet | None = None
active_sessions: dict[str, dict] = {}
sessions_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# Lifespan
# ═══════════════════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, BOARD_SIZE, NET_CHANNELS, NET_RES_BLOCKS
    # ── startup ──────────────────────────────────────────────────────────
    if MODEL_PATH and os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            BOARD_SIZE = checkpoint.get("N", BOARD_SIZE)
            NET_CHANNELS = checkpoint.get("num_channels", NET_CHANNELS)
            NET_RES_BLOCKS = checkpoint.get("num_res_blocks", NET_RES_BLOCKS)
        else:
            state_dict = checkpoint
        model = AlphaReversiNet(
            N=BOARD_SIZE, num_channels=NET_CHANNELS,
            num_res_blocks=NET_RES_BLOCKS,
        )
        model.load_state_dict(state_dict)
        print(f"[startup] Loaded weights from {MODEL_PATH}  "
              f"(N={BOARD_SIZE}, ch={NET_CHANNELS}, res={NET_RES_BLOCKS})")
    else:
        print("[startup] No checkpoint found — using random weights")
        model = AlphaReversiNet(
            N=BOARD_SIZE,
            num_channels=NET_CHANNELS,
            num_res_blocks=NET_RES_BLOCKS,
        )

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[startup] Model ready  ({n_params:,} parameters)")
    yield
    # ── shutdown ──────────────────────────────────────────────────────────
    with sessions_lock:
        active_sessions.clear()
    print("[shutdown] Server stopped")


# ═══════════════════════════════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="ObstacleReversi API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic schemas
# ═══════════════════════════════════════════════════════════════════════════════


class HumanMoveRequest(BaseModel):
    session_id: str
    x: int  # row
    y: int  # column


class AIMoveRequest(BaseModel):
    session_id: str


class HumanPassRequest(BaseModel):
    session_id: str


class GameStateResponse(BaseModel):
    session_id: str
    board: list[list[int]]
    current_player: int
    legal_moves: list[list[int]]
    game_over: bool
    winner: int | None
    score: dict[str, int]
    ai_player: int


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _get_session(session_id: str) -> dict:
    """Return session dict or raise 404."""
    with sessions_lock:
        session = active_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, f"Session '{session_id}' not found")
    return session


def _serialize(session_id: str, session: dict) -> dict:
    """Convert session state → JSON-serialisable dict."""
    env: ObstacleReversiEnv = session["env"]
    if not env.done and env.is_terminal_position():
        env.done = True

    legal = (
        np.zeros((env.N, env.N), dtype=bool)
        if env.done
        else env.get_legal_actions(env.current_player)
    )
    legal_coords = [[int(r), int(c)] for r, c in np.argwhere(legal)]

    p1, p2 = env.get_score()
    winner = None
    if env.done:
        if p1 > p2:
            winner = -1
        elif p2 > p1:
            winner = 1
        else:
            winner = 0

    return {
        "session_id": session_id,
        "board": env.board.astype(int).tolist(),
        "current_player": int(env.current_player),
        "legal_moves": legal_coords,
        "game_over": env.done,
        "winner": winner,
        "score": {"black": p1, "white": p2},
        "ai_player": session["ai_player"],
    }


def _mcts_search(session: dict, board: np.ndarray, player: int, passes: int) -> int:
    """Run MCTS using the session's persistent searcher.

    A throw-away env is swapped in so the session env is never
    touched by the background thread.  The entire call is guarded
    by *tree_lock* to serialise access to the MCTS tree.
    """
    mcts: MCTS = session["mcts"]
    tree_lock: threading.Lock = session["tree_lock"]

    with tree_lock:
        # Swap in a disposable env for simulation
        temp_env = ObstacleReversiEnv(
            N=board.shape[0], obstacle_ratio=OBSTACLE_RATIO
        )
        real_env = mcts.env
        mcts.env = temp_env

        temp_env.board = board.copy()
        temp_env.current_player = player
        temp_env.done = False
        temp_env.consecutive_passes = passes

        try:
            action, _root, _iters = mcts.get_action_with_time_budget(
                board, player, AI_TIME_LIMIT, passes
            )
        finally:
            mcts.env = real_env

    return action


def _step_and_advance(session: dict, action: int) -> None:
    """Execute *action* on the env and advance the MCTS tree.

    Must be called inside ``session["tree_lock"]``.
    """
    env: ObstacleReversiEnv = session["env"]
    mcts: MCTS = session["mcts"]
    env.step(action)
    mcts.advance_root(action)


# ═══════════════════════════════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════════════════════════════


SESSION_TTL = 3600  # auto-clean sessions older than 1 hour


def _cleanup_stale_sessions() -> int:
    """Drop sessions older than SESSION_TTL. Returns count removed."""
    now = time.time()
    stale = [
        sid for sid, s in active_sessions.items()
        if now - s["created_at"] > SESSION_TTL
    ]
    for sid in stale:
        del active_sessions[sid]
    return len(stale)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the single-page frontend."""
    import os
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/api/new_game", response_model=GameStateResponse)
async def new_game():
    """Create a fresh game session.

    Returns the initial board.  If the AI is configured to move first
    the AI's opening move is executed synchronously inside this call.
    """
    session_id = str(uuid.uuid4())
    env = ObstacleReversiEnv(N=BOARD_SIZE, obstacle_ratio=OBSTACLE_RATIO)
    env.reset()

    # Create a persistent MCTS searcher for this session
    mcts = MCTS(env, model, c_puct=AI_C_PUCT, device="cpu")

    session = {
        "env": env,
        "mcts": mcts,
        "ai_player": AI_PLAYER,
        "tree_lock": threading.Lock(),
        "created_at": time.time(),
    }

    with sessions_lock:
        _cleanup_stale_sessions()
        active_sessions[session_id] = session

    # AI opens?
    if env.current_player == AI_PLAYER:
        # Expand root so advance_root has a tree to work with later
        with session["tree_lock"]:
            mcts.get_action_with_time_budget(
                env.board.copy(), env.current_player, AI_TIME_LIMIT
            )
            # best action by visit count
            ai_action = max(
                mcts.root.children.keys(),
                key=lambda a: mcts.root.children[a].N,
            )
            _step_and_advance(session, ai_action)

    return _serialize(session_id, session)


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/human_move  (fast — O(N), ~ms)
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/api/human_move", response_model=GameStateResponse)
async def human_move(req: HumanMoveRequest):
    """Execute the human's move and return the resulting state immediately.

    The action is validated, applied to the env, and the MCTS tree is
    advanced — all under *tree_lock*.  No neural-network inference
    happens here, so the response is built in microseconds.
    """
    session = _get_session(req.session_id)
    env: ObstacleReversiEnv = session["env"]

    # ── validation ────────────────────────────────────────────────────
    if env.done:
        raise HTTPException(400, "Game is already over")

    if env.current_player == session["ai_player"]:
        raise HTTPException(400, "Not your turn — the AI is to move")

    r, c = req.x, req.y
    if not (0 <= r < env.N and 0 <= c < env.N):
        raise HTTPException(400, f"Coordinates ({r},{c}) out of bounds")

    legal = env.get_legal_actions(env.current_player)
    if not legal[r, c]:
        raise HTTPException(400, f"Illegal move at ({r},{c})")

    action = r * env.N + c

    # ── execute + advance (under lock) ─────────────────────────────────
    with session["tree_lock"]:
        _step_and_advance(session, action)

    return _serialize(req.session_id, session)


@app.post("/api/human_pass", response_model=GameStateResponse)
async def human_pass(req: HumanPassRequest):
    """Human passes their turn (no legal moves available).

    Calls ``env.step(-1)`` to record the pass.  If the AI also passed
    immediately before, the game ends.
    """
    session = _get_session(req.session_id)
    env: ObstacleReversiEnv = session["env"]

    if env.done:
        raise HTTPException(400, "Game is already over")

    if env.current_player == session["ai_player"]:
        raise HTTPException(400, "Not your turn — the AI is to move")

    # Verify the human actually has no legal moves
    legal = env.get_legal_actions(env.current_player)
    if legal.any():
        raise HTTPException(400, "You have legal moves — cannot pass")

    with session["tree_lock"]:
        _step_and_advance(session, -1)

    return _serialize(req.session_id, session)


# ──────────────────────────────────────────────────────────────────────────────
# POST /api/ai_move  (slow — CPU-bound MCTS offloaded to thread pool)
# ──────────────────────────────────────────────────────────────────────────────


@app.post("/api/ai_move", response_model=GameStateResponse)
async def ai_move(req: AIMoveRequest):
    """Run the AI's turn.

    MCTS search is dispatched to a background thread via
    ``run_in_threadpool`` so the event loop stays responsive for
    other sessions.  Once the search finishes, the chosen action is
    applied under *tree_lock*.
    """
    session = _get_session(req.session_id)
    env: ObstacleReversiEnv = session["env"]

    if env.done:
        raise HTTPException(400, "Game is already over")

    if env.current_player != session["ai_player"]:
        raise HTTPException(400, "Not the AI's turn")

    # Guard: AI has no legal moves → pass
    legal = env.get_legal_actions(env.current_player)
    if not legal.any():
        with session["tree_lock"]:
            _step_and_advance(session, -1)
        return _serialize(req.session_id, session)

    # ── offload MCTS to thread pool ────────────────────────────────────
    ai_action = await run_in_threadpool(
        _mcts_search,
        session,
        env.board.copy(),
        env.current_player,
        env.consecutive_passes,
    )

    # ── apply AI action + advance tree (under lock) ────────────────────
    if ai_action >= 0:
        with session["tree_lock"]:
            _step_and_advance(session, ai_action)

    return _serialize(req.session_id, session)


# ──────────────────────────────────────────────────────────────────────────────
# Read / delete / health
# ──────────────────────────────────────────────────────────────────────────────


@app.get("/api/game/{session_id}", response_model=GameStateResponse)
async def get_game(session_id: str):
    """Return the current state for an existing session."""
    session = _get_session(session_id)
    return _serialize(session_id, session)


@app.delete("/api/game/{session_id}")
async def delete_game(session_id: str):
    """Remove a session (free memory)."""
    with sessions_lock:
        if session_id not in active_sessions:
            raise HTTPException(404, f"Session '{session_id}' not found")
        del active_sessions[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "active_sessions": len(active_sessions),
        "model_loaded": model is not None,
    }

# ═══════════════════════════════════════════════════════════════════════════════
# Entry-point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
