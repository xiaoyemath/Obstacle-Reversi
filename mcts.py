"""
MCTS — Monte Carlo Tree Search engine for ObstacleReversi.

Combines ObstacleReversiEnv (state simulation) with AlphaReversiNet
(prior policy + value) to run time-budgeted tree search.
"""

from __future__ import annotations

import time
import numpy as np
import torch

from alpha_reversi_net import AlphaReversiNet


# ──────────────────────────────────────────────────────────────────────────────
# Dummy lock (no-op when threading is not needed)
# ──────────────────────────────────────────────────────────────────────────────

class _DummyLock:
    """Drop-in no-op replacement for ``threading.Lock``."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Tree node
# ──────────────────────────────────────────────────────────────────────────────

class _Node:
    """A single node in the MCTS tree.

    Stores the game state (board + player to move), visit statistics,
    and prior probability assigned by the network.
    """

    __slots__ = (
        "parent",
        "children",
        "N",
        "W",
        "P",
        "prior",
        "board",
        "player",
        "action_from_parent",
        "is_terminal",
        "consecutive_passes",
    )

    def __init__(
        self,
        board: np.ndarray | None,
        player: int | None,
        parent: _Node | None = None,
        action_from_parent: int | None = None,
        prior: float = 0.0,
        consecutive_passes: int = 0,
    ):
        self.parent = parent
        self.children: dict[int, _Node] = {}  # action → child node
        self.N = 0            # visit count
        self.W = 0.0          # total value (from this node's player's perspective)
        self.P = None         # prior probability vector (N²,) — set after expansion
        self.prior = prior    # P(s, parent_action) for the edge above this node
        self.board = board    # (N, N) int8 board matrix
        self.player = player  # whose turn at this node
        self.action_from_parent = action_from_parent
        self.is_terminal = False
        self.consecutive_passes = consecutive_passes

    @property
    def Q(self) -> float:
        """Mean action value Q = W / N."""
        return self.W / self.N if self.N > 0 else 0.0

    @property
    def is_expanded(self) -> bool:
        """A node is expanded once the network has produced its children or it is terminal."""
        return len(self.children) > 0 or self.is_terminal


# ──────────────────────────────────────────────────────────────────────────────
# MCTS engine
# ──────────────────────────────────────────────────────────────────────────────

class MCTS:
    """Monte-Carlo Tree Search for ObstacleReversi.

    Parameters
    ----------
    env : ObstacleReversiEnv
        Game environment used for state transitions and legal-action queries.
    net : AlphaReversiNet
        Dual-head network providing prior policy **p** and value **v**.
    c_puct : float
        Exploration constant in the PUCT formula (default 1.4).
    device : str
        Torch device for network inference (default ``'cpu'``).
    """

    def __init__(
        self,
        env,            # ObstacleReversiEnv
        net,            # AlphaReversiNet
        c_puct: float = 1.4,
        device: str = "cpu",
        env_lock=None,  # threading.Lock | None — protects env state during expand
    ):
        self.env = env
        self.net = net.to(device)
        self.c_puct = c_puct
        self.device = device
        self.N = env.N
        self._env_lock = env_lock or _DummyLock()
        self.net.eval()  # use running BN statistics for single-sample inference

        # Set after a search completes
        self.root: _Node | None = None
        self.iterations: int = 0

    # ── public API ───────────────────────────────────────────────────────

    def get_action_with_time_budget(
        self,
        root_board: np.ndarray,
        root_player: int,
        time_limit_seconds: float,
        consecutive_passes: int = 0,
    ) -> tuple[int, _Node, int]:
        """Run time-budgeted MCTS and return the most-visited root action.

        If ``self.root`` already matches *root_board* and *root_player*,
        the persisted subtree is reused (saving the initial network
        forward pass and preserving accumulated statistics).

        Args:
            root_board:  (N, N) board matrix.
            root_player: whose turn it is (-1 or 1).
            time_limit_seconds: wall-clock budget in seconds.
            consecutive_passes: current consecutive pass count.

        Returns:
            (action, root_node, iterations)
        """
        # ── try to reuse the persisted root ──────────────────────────
        root = self.root
        if (
            root is None
            or root.board is None
            or not np.array_equal(root.board, root_board)
            or root.player != root_player
            or root.consecutive_passes != consecutive_passes
            or root.is_terminal
            or len(root.children) == 0
        ):
            root = _Node(board=root_board.copy(), player=root_player,
                        consecutive_passes=consecutive_passes)
            self._expand(root)

        if root.is_terminal:
            self.root = root
            self.iterations = 0
            return -1, root, 0  # no legal moves

        deadline = time.time() + time_limit_seconds
        iters = 0

        while time.time() < deadline:
            leaf, path = self._select(root)

            if leaf.is_terminal:
                v = self._terminal_value(leaf)
            else:
                v = self._expand(leaf)

            self._backpropagate(path, v)
            iters += 1

        self.root = root
        self.iterations = iters

        # Action with largest visit count (τ → 0).  If the caller gives a
        # zero-second budget, no child has visits yet, so fall back to priors.
        if any(child.N > 0 for child in root.children.values()):
            best_action = max(root.children.keys(),
                              key=lambda a: root.children[a].N)
        else:
            best_action = max(root.children.keys(),
                              key=lambda a: root.children[a].prior)
        return best_action, root, iters

    # ── tree advance ────────────────────────────────────────────────────

    def advance_root(self, action: int) -> None:
        """Promote the child corresponding to *action* to be the new root.

        After a move is played in the real environment, call this so
        the next ``get_action_with_time_budget`` can reuse the subtree.
        If the action was never explored, the root is cleared and the
        next search starts fresh.
        """
        if self.root is None:
            return
        child = self.root.children.get(action)
        if child is not None and not child.is_terminal:
            child.parent = None          # detach from old tree
            self.root = child
        else:
            self.root = None             # action unexplored → fresh start

    def get_action_probs(
        self, root: _Node | None = None, temperature: float = 1.0
    ) -> np.ndarray:
        """Convert root visit counts into an action probability vector.

        Args:
            root: The root node (uses ``self.root`` if None).
            temperature: τ > 0 smooths the distribution; τ = 0 → argmax.

        Returns:
            Float32 array of shape (N²,) summing to 1.
        """
        if root is None:
            root = self.root
        if root is None:
            raise ValueError("No root node available; run a search first.")

        visits = np.zeros(self.N * self.N, dtype=np.float64)
        for a, child in root.children.items():
            if a >= 0:                # pass action has no slot in the policy vector
                visits[a] = child.N

        if visits.sum() == 0:
            return np.ones(self.N * self.N, dtype=np.float32) / (self.N * self.N)

        if temperature == 0.0:
            probs = np.zeros_like(visits)
            probs[int(np.argmax(visits))] = 1.0
            return probs.astype(np.float32)

        visits = visits ** (1.0 / temperature)
        return (visits / visits.sum()).astype(np.float32)

    # ── selection ────────────────────────────────────────────────────────

    def _select(self, root: _Node) -> tuple[_Node, list[_Node]]:
        """Traverse from *root* to a leaf using PUCT, return (leaf, path)."""
        path = [root]
        node = root

        while node.is_expanded and not node.is_terminal:
            sqrt_N = np.sqrt(node.N)
            best_a = None
            best_score = -float("inf")

            for a, child in node.children.items():
                # child.Q is from the child player's perspective, so negate it
                # when selecting the best action for the current parent node.
                score = -child.Q + self.c_puct * child.prior * sqrt_N / (1.0 + child.N)
                if score > best_score:
                    best_score = score
                    best_a = a

            if best_a is None:     # safety — shouldn't happen
                break

            node = node.children[best_a]
            path.append(node)

        return node, path

    # ── expansion ────────────────────────────────────────────────────────

    def _expand(self, node: _Node) -> float:
        """Run network on *node*, create children for all legal actions.

        Returns the scalar value *v* (from *node*'s player's perspective).

        The env-lock is held only during brief env-state mutations;
        network inference runs outside the lock so the main-thread
        renderer is never blocked for more than a few microseconds.
        """
        # ── locked: snapshot env & query legal actions ──────────────────────
        with self._env_lock:
            saved_board = self.env.board
            saved_player = self.env.current_player
            saved_done = self.env.done
            saved_passes = self.env.consecutive_passes

            try:
                self.env.board = node.board
                self.env.current_player = node.player
                self.env.done = False
                self.env.consecutive_passes = node.consecutive_passes

                legal = self.env.get_legal_actions(node.player)

                # ── pass-only node (no legal board moves) ───────────────
                _pass_only = not legal.any()
                _board = node.board.copy()
                _player = node.player
                if not _pass_only:
                    _legal = legal.copy()
            finally:
                self.env.board = saved_board
                self.env.current_player = saved_player
                self.env.done = saved_done
                self.env.consecutive_passes = saved_passes

        if _pass_only:
            # unlocked: network inference for value only.
            # Use a dummy all-true mask — the policy output is ignored
            # because the only action is pass (prior = 1.0).
            x = AlphaReversiNet.build_input(_board, _player).to(self.device)
            mask_dummy = torch.ones(1, self.N * self.N, dtype=torch.bool, device=self.device)
            with torch.no_grad():
                _, v = self.net(x, mask_dummy)
            v_val = float(v[0, 0].item())
            node.P = np.zeros(self.N * self.N, dtype=np.float64)

            # locked: simulate pass child
            with self._env_lock:
                self.env.board = node.board.copy()
                self.env.current_player = node.player
                self.env.done = False
                self.env.consecutive_passes = node.consecutive_passes

                new_board, _, done, _, _ = self.env.step(-1)

                child = _Node(
                    board=new_board.copy(),
                    player=self.env.current_player,
                    parent=node,
                    action_from_parent=-1,
                    prior=1.0,
                    consecutive_passes=self.env.consecutive_passes,
                )
                child.is_terminal = done
                node.children[-1] = child

                # restore
                self.env.board = saved_board
                self.env.current_player = saved_player
                self.env.done = saved_done
                self.env.consecutive_passes = saved_passes

            return v_val
        # ── normal node (has legal board moves) ─────────────────────────

        # ── unlocked: network inference (the slow part) ─────────────────────
        x = AlphaReversiNet.build_input(_board, _player).to(self.device)
        mask = torch.from_numpy(_legal.flatten().copy()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            p, v = self.net(x, mask)

        p_arr = p[0].cpu().numpy().astype(np.float64)
        v_val = float(v[0, 0].item())
        node.P = p_arr

        # ── locked: simulate children & restore ─────────────────────────────
        with self._env_lock:
            legal_flat = _legal.flatten()
            for a in range(self.N * self.N):
                if not legal_flat[a]:
                    continue

                self.env.board = node.board.copy()
                self.env.current_player = node.player
                self.env.done = False
                self.env.consecutive_passes = node.consecutive_passes

                new_board, _, done, _, _ = self.env.step(a)

                child = _Node(
                    board=new_board.copy(),
                    player=self.env.current_player,
                    parent=node,
                    action_from_parent=a,
                    prior=float(p_arr[a]),
                    consecutive_passes=self.env.consecutive_passes,
                )
                child.is_terminal = done
                node.children[a] = child

            # restore original env state
            self.env.board = saved_board
            self.env.current_player = saved_player
            self.env.done = saved_done
            self.env.consecutive_passes = saved_passes

        return v_val

    # ── backpropagation ──────────────────────────────────────────────────

    def _backpropagate(self, path: list[_Node], v: float):
        """Propagate *v* up the path, flipping sign at each level (zero-sum)."""
        for node in reversed(path):
            node.N += 1
            node.W += v
            v = -v

    # ── terminal evaluation ──────────────────────────────────────────────

    def _terminal_value(self, node: _Node) -> float:
        """Game-outcome value from the perspective of *node.player*.

        Returns +1 (win), -1 (loss), or 0 (draw).
        """
        if node.board is None or node.player is None:
            raise ValueError("Terminal node is missing board/player state")

        p1 = int(np.sum(node.board == -1))
        p2 = int(np.sum(node.board == 1))
        if p1 > p2:
            return 1.0 if node.player == -1 else -1.0
        elif p2 > p1:
            return 1.0 if node.player == 1 else -1.0
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from obstacle_reversi_env import ObstacleReversiEnv
    from alpha_reversi_net import AlphaReversiNet

    env = ObstacleReversiEnv(N=6, obstacle_ratio=0.08)
    env.reset()
    print("Initial board:")
    print(env.board)
    print(f"Current player: {env.current_player}  (-1=Black, 1=White)")
    print(f"Obstacles: {np.sum(env.board == 2)}")

    net = AlphaReversiNet(N=6, num_channels=64, num_res_blocks=4)
    net.eval()

    mcts = MCTS(env, net, c_puct=1.4, device="cpu")

    print("\nRunning MCTS with 3-second budget ...")
    action, root, iters = mcts.get_action_with_time_budget(
        env.board, env.current_player, time_limit_seconds=3.0
    )

    r, c = divmod(action, 6)
    print(f"\nBest action: {action}  →  board[{r},{c}]")
    print(f"Iterations completed: {iters}")
    print(f"Root node visits:    {root.N}")
    print(f"Root children:       {len(root.children)}")

    # Show top actions by visit count
    print("\nTop actions (τ→0 ranking):")
    for a, child in sorted(root.children.items(),
                           key=lambda kv: -kv[1].N)[:6]:
        r, c = divmod(a, 6)
        print(
            f"  a={a:2d}  ({r},{c})  "
            f"N={child.N:4d}  Q={child.Q:+.4f}  P={child.prior:.4f}"
        )

    # Verify visit-count proportions sum to 1
    probs = mcts.get_action_probs(temperature=0.0)
    print(f"\nτ=0 action probs sum: {probs.sum():.4f}  (should be 1.0)")
    print(f"Argmax action: {int(np.argmax(probs))}")
