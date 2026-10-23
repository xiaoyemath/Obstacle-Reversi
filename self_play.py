"""
self_play.py — Self-play data generation and experience replay for AlphaReversi.

Provides:
- ExperienceReplayBuffer  — fixed-capacity deque with D₄ symmetry augmentation.
- SelfPlayWorker          — plays one game end-to-end, pushes (S, π, z) triples.
"""

from __future__ import annotations

import collections
import numpy as np
import torch

from obstacle_reversi_env import ObstacleReversiEnv
from alpha_reversi_net import AlphaReversiNet
from mcts import MCTS


# ═══════════════════════════════════════════════════════════════════════════════
# Experience Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════════

class ExperienceReplayBuffer:
    """Fixed-capacity replay buffer with on-the-fly D₄ data augmentation.

    Each stored entry is a ``(state, pi, z)`` tuple:

    - **state** : ``FloatTensor (4, N, N)`` — 4-channel board representation
      from the *current player's* perspective (output of
      ``AlphaReversiNet.build_input``, batch-dim squeezed).
    - **pi**    : ``FloatTensor (N²,)`` — MCTS visit-count policy.
    - **z**     : ``float`` — terminal value ∈ {−1, 0, +1} from the same
      player's perspective.

    ``sample(batch_size)`` applies a *random* dihedral symmetry
    (rotation × flip) to each sampled position, which effectively
    multiplies the dataset size by 8 over the course of training.
    """

    def __init__(self, maxlen: int = 10000):
        self.buffer: collections.deque = collections.deque(maxlen=maxlen)

    # ── mutate ──────────────────────────────────────────────────────────

    def push(
        self, state: torch.Tensor, pi: torch.Tensor, z: float
    ) -> None:
        """Push a single experience triple into the buffer."""
        self.buffer.append((state, pi, z))

    def push_batch(
        self, triples: list[tuple[torch.Tensor, torch.Tensor, float]]
    ) -> None:
        """Push many ``(state, pi, z)`` triples at once."""
        for t in triples:
            self.buffer.append(t)

    # ── sample ──────────────────────────────────────────────────────────

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch with random symmetry augmentation.

        Returns:
            ``(states, pis, zs)`` where

            - **states** : ``(B, 4, N, N)``
            - **pis**    : ``(B, N²)``
            - **zs**     : ``(B,)``
        """
        n = len(self.buffer)
        if n == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        replace = n < batch_size
        indices = np.random.choice(n, batch_size, replace=replace)

        states, pis, zs = [], [], []
        for idx in indices:
            s, pi, z = self.buffer[idx]
            s_aug, pi_aug = self._random_symmetry(s, pi)
            states.append(s_aug)
            pis.append(pi_aug)
            zs.append(z)

        return (
            torch.stack(states),                     # (B, 4, N, N)
            torch.stack(pis),                         # (B, N²)
            torch.tensor(zs, dtype=torch.float32),    # (B,)
        )

    # ── augmentation (D₄ — 8 symmetries) ────────────────────────────────

    @staticmethod
    def _random_symmetry(
        state: torch.Tensor, pi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a random element of the dihedral group D₄.

        Rotates by 0° / 90° / 180° / 270° (random), then optionally
        applies a horizontal flip.
        """
        k = np.random.randint(0, 4)          # rotation index
        do_flip = np.random.rand() < 0.5

        N = int(np.sqrt(pi.shape[-1]))
        pi_2d = pi.view(N, N)

        # Rotate spatial dims of both state and policy
        state = torch.rot90(state, k, dims=(-2, -1))
        pi_2d = torch.rot90(pi_2d, k, dims=(-2, -1))

        if do_flip:
            state = torch.flip(state, dims=(-1,))
            pi_2d = torch.flip(pi_2d, dims=(-1,))

        return state, pi_2d.flatten()

    def __len__(self) -> int:
        return len(self.buffer)


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Play Worker
# ═══════════════════════════════════════════════════════════════════════════════

class SelfPlayWorker:
    """Play one full game of self-play and push experiences into the buffer.

    Parameters
    ----------
    net :
        The current ``AlphaReversiNet`` used for MCTS inference.
    replay_buffer :
        An ``ExperienceReplayBuffer`` instance.
    N :
        Board size.
    obstacle_ratio :
        Fraction of cells turned into obstacles.
    c_puct :
        PUCT exploration constant for MCTS.
    time_limit :
        Seconds of search per move.
    temp_threshold :
        Number of opening moves to play with τ = 1.0 before switching
        to τ → 0 (greedy).
    device :
        Torch device for network inference.
    """

    def __init__(
        self,
        net: AlphaReversiNet,
        replay_buffer: ExperienceReplayBuffer,
        N: int = 8,
        obstacle_ratio: float = 0.1,
        c_puct: float = 1.4,
        time_limit: float = 2.0,
        temp_threshold: int = 15,
        device: str = "cpu",
    ):
        self.net = net
        self.replay_buffer = replay_buffer
        self.N = N
        self.obstacle_ratio = obstacle_ratio
        self.c_puct = c_puct
        self.time_limit = time_limit
        self.temp_threshold = temp_threshold
        self.device = device

    # ── play one game ───────────────────────────────────────────────────

    def play_one_game(self) -> tuple[int, int, int]:
        """Run one full self-play episode.

        Returns:
            ``(winner, move_count, history_length)`` where *winner* is
            −1 (Black), 1 (White), or 0 (draw).
        """
        # Fresh environment
        env = ObstacleReversiEnv(N=self.N, obstacle_ratio=self.obstacle_ratio)
        env.reset()

        # Fresh MCTS searcher (no lock needed — single-threaded self-play)
        mcts = MCTS(env, self.net, c_puct=self.c_puct, device=self.device)

        # Per-step records: (state, pi, player)
        history: list[tuple[torch.Tensor, torch.Tensor, int]] = []

        move_count = 0
        while not env.done:
            # ── temperature annealing ────────────────────────────────
            temperature = 1.0 if move_count < self.temp_threshold else 0.0

            # ── build network input ───────────────────────────────────
            state = AlphaReversiNet.build_input(
                env.board, env.current_player
            ).squeeze(0)  # (4, N, N)

            # ── MCTS search ───────────────────────────────────────────
            action, root, _iters = mcts.get_action_with_time_budget(
                env.board.copy(), env.current_player, self.time_limit,
                env.consecutive_passes,
            )

            if action == -1:
                # No legal board moves — forced pass
                env.step(-1)
                if env.done:
                    break
                continue

            # ── policy from visit counts ──────────────────────────────
            pi_np = mcts.get_action_probs(root, temperature=temperature)
            pi = torch.from_numpy(pi_np)          # (N²,)

            # ── record & step ─────────────────────────────────────────
            history.append((state, pi, env.current_player))
            env.step(action)
            move_count += 1

        # ── terminal outcome ──────────────────────────────────────────
        p1, p2 = env.get_score()
        if p1 > p2:
            winner = -1          # Black
        elif p2 > p1:
            winner = 1           # White
        else:
            winner = 0           # draw

        # ── assign z per move & push to buffer ────────────────────────
        for state, pi, player in history:
            if winner == 0:
                z = 0.0
            elif player == winner:
                z = 1.0
            else:
                z = -1.0
            self.replay_buffer.push(state, pi, z)

        return winner, move_count, len(history)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("Self-play smoke test")
    print("=" * 60)

    # ---- instantiate net & buffer ----
    net = AlphaReversiNet(N=6, num_channels=64, num_res_blocks=4)
    net.eval()

    buf = ExperienceReplayBuffer(maxlen=5000)
    worker = SelfPlayWorker(
        net, buf,
        N=6, obstacle_ratio=0.08,
        c_puct=1.4, time_limit=1.5,
        temp_threshold=8,
        device="cpu",
    )

    # ---- play one game ----
    t0 = time.time()
    winner, moves, hist_len = worker.play_one_game()
    elapsed = time.time() - t0

    print(f"\nGame finished in {elapsed:.1f} s")
    print(f"Winner: {winner}  (−1=Black, 1=White, 0=Draw)")
    print(f"Moves:  {moves}")
    print(f"History length: {hist_len}")
    print(f"Buffer size:    {len(buf)}")

    # ---- inspect buffer entries ----
    s, pi, z = buf.buffer[0]
    print(f"\nFirst entry — state: {s.shape}, pi: {pi.shape}, z: {z}")

    # ---- sample ----
    batch_s, batch_pi, batch_z = buf.sample(16)
    print(f"\nSample(16) → states: {batch_s.shape}, pis: {batch_pi.shape}, zs: {batch_z.shape}")
    print(f"z distribution: +1={int((batch_z>0.5).sum())}  "
          f"0={int((batch_z.abs()<0.5).sum())}  "
          f"−1={int((batch_z<-0.5).sum())}")

    # ---- verify symmetry invariance: z unchanged after augmentation ----
    s_orig, pi_orig, z_orig = buf.buffer[0]
    s_aug, pi_aug = ExperienceReplayBuffer._random_symmetry(s_orig, pi_orig)
    assert s_aug.shape == s_orig.shape, "State shape changed after symmetry!"
    assert pi_aug.shape == pi_orig.shape, "Pi shape changed after symmetry!"
    assert torch.allclose(pi_aug.sum(), pi_orig.sum(), atol=1e-5), \
        "Pi sum changed after symmetry!"
    print("\nSymmetry invariance check passed.")
    print("All checks passed.")
