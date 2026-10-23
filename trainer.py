"""
trainer.py — AlphaZero training loop with self-play data generation,
           experience replay, and gradient-descent updates.
"""

from __future__ import annotations

import os
import time
import concurrent.futures
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_reversi_net import AlphaReversiNet
from self_play import ExperienceReplayBuffer, SelfPlayWorker


# ═══════════════════════════════════════════════════════════════════════════════
# AlphaZero Trainer
# ═══════════════════════════════════════════════════════════════════════════════

class AlphaZeroTrainer:
    """End-to-end AlphaZero training loop for ObstacleReversi.

    Alternates between parallel self-play data generation and
    mini-batch gradient-descent updates, saving checkpoints each iteration.

    Parameters
    ----------
    net : AlphaReversiNet
        The network to train.
    replay_buffer : ExperienceReplayBuffer
        Shared replay buffer (pre-allocated).
    lr : float
        Initial learning rate for Adam.
    weight_decay : float
        L₂ regularisation coefficient.
    lr_step_size : int
        Number of iterations between LR decays.
    lr_gamma : float
        Multiplicative decay factor for the LR scheduler.
    gradient_clip : float or None
        Max gradient norm (None disables clipping).
    """

    def __init__(
        self,
        net: AlphaReversiNet,
        replay_buffer: ExperienceReplayBuffer,
        *,
        device: str | torch.device | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_step_size: int = 30,
        lr_gamma: float = 0.1,
        gradient_clip: float | None = 1.0,
    ):
        if device is None:
            try:
                self.device = next(net.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.net = net.to(self.device)
        self.replay_buffer = replay_buffer

        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=lr_step_size, gamma=lr_gamma
        )
        self.gradient_clip = gradient_clip

    # ── Loss function ───────────────────────────────────────────────────

    def compute_loss(
        self,
        states: torch.Tensor,
        pis: torch.Tensor,
        zs: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float]:
        """Joint policy + value loss.

        - **Policy loss**: cross-entropy  ``H(π, p)``, computed via
          ``log_softmax`` on *unmasked* raw logits — numerically stable
          even when logits contain ``-1e9`` stubs.
        - **Value loss**:  ``MSE(z, v)``.

        Returns:
            ``(total_loss, policy_loss_val, value_loss_val)``.
        """
        # Forward — request raw logits for stable CE computation
        p, v, logits = self.net(states, return_logits=True)

        # Policy loss:  − Σ π · log_softmax(logits)
        log_probs = F.log_softmax(logits, dim=-1)         # (B, N²)
        policy_loss = -(pis * log_probs).sum(dim=-1).mean()

        # Value loss
        value_loss = F.mse_loss(v.squeeze(-1), zs)

        return (
            policy_loss + value_loss,
            float(policy_loss.item()),
            float(value_loss.item()),
        )

    # ── Data generation (parallel self-play) ────────────────────────────

    @staticmethod
    def _play_one(net, buffer, **worker_kwargs):
        """Worker entry-point for a single self-play game."""
        worker = SelfPlayWorker(net, buffer, **worker_kwargs)
        return worker.play_one_game()

    def generate_data(
        self,
        num_games: int,
        num_workers: int = 4,
        **worker_kwargs,
    ) -> None:
        """Run *num_games* self-play episodes in parallel.

        Each worker creates its own environment + MCTS instance; the
        network is shared read-only in ``eval()`` mode.
        """
        self.net.eval()

        # Merge in any caller overrides (time_limit, N, …)
        cfg = dict(
            N=8, obstacle_ratio=0.1,
            c_puct=1.4, time_limit=2.0,
            temp_threshold=15, device=str(self.device),
        )
        cfg.update(worker_kwargs)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_workers
        ) as executor:
            futures = [
                executor.submit(
                    AlphaZeroTrainer._play_one,
                    self.net, self.replay_buffer, **cfg,
                )
                for _ in range(num_games)
            ]
            # Wait for completion, surfacing any exception
            for f in concurrent.futures.as_completed(futures):
                f.result()          # re-raises if a worker crashed

    # ── Single training step ────────────────────────────────────────────

    def train_step(
        self, batch_size: int, batches: int
    ) -> tuple[float, float, float]:
        """Run *batches* gradient updates, return average losses."""
        self.net.train()

        total, total_p, total_v = 0.0, 0.0, 0.0
        for _ in range(batches):
            states, pis, zs = self.replay_buffer.sample(batch_size)
            states = states.to(self.device)
            pis = pis.to(self.device)
            zs = zs.to(self.device)

            self.optimizer.zero_grad()
            loss, p_loss, v_loss = self.compute_loss(states, pis, zs)
            loss.backward()

            if self.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.net.parameters(), self.gradient_clip
                )

            self.optimizer.step()

            total += loss.item()
            total_p += p_loss
            total_v += v_loss

        n = max(batches, 1)
        return total / n, total_p / n, total_v / n

    # ── Macro training loop ─────────────────────────────────────────────

    def train_loop(
        self,
        num_iterations: int = 100,
        games_per_iter: int = 100,
        num_workers: int = 4,
        batch_size: int = 64,
        batches_per_iter: int = 10,
        save_dir: str = "checkpoints",
        **worker_kwargs,
    ) -> None:
        """Run the full AlphaZero training loop.

        Each iteration:

        1. Generate *games_per_iter* self-play episodes (parallel).
        2. Sample *batches_per_iter* mini-batches and update weights.
        3. Step the LR scheduler.
        4. Save ``iter_NNN.pth`` to *save_dir*.

        Parameters
        ----------
        num_iterations : int
            Number of outer-loop iterations.
        games_per_iter : int
            Self-play games to generate per iteration.
        num_workers : int
            Number of parallel self-play threads.
        batch_size : int
            Mini-batch size for gradient updates.
        batches_per_iter : int
            Number of mini-batches per iteration.
        save_dir : str
            Directory for checkpoint files.
        """
        # Detect network architecture from state_dict
        sd = self.net.state_dict()
        num_channels = sd["conv_init.0.weight"].shape[0]
        num_res_blocks = len([k for k in sd if k.startswith("res_blocks.") and k.endswith(".conv1.weight")])
        os.makedirs(save_dir, exist_ok=True)

        for iteration in range(num_iterations):
            t0 = time.time()

            # ---- 1 — self-play data generation --------------------------
            self.generate_data(
                games_per_iter, num_workers=num_workers, **worker_kwargs
            )
            t1 = time.time()

            # ---- 2 — network updates ------------------------------------
            avg_loss, avg_p, avg_v = self.train_step(
                batch_size, batches_per_iter
            )
            self.scheduler.step()
            t2 = time.time()

            # ---- 3 — checkpoint -----------------------------------------
            ckpt_path = os.path.join(save_dir, f"iter_{iteration:03d}.pth")
            checkpoint = {
                "model_state_dict": self.net.state_dict(),
                "N": self.net.N,
                "num_channels": num_channels,
                "num_res_blocks": num_res_blocks,
            }
            torch.save(checkpoint, ckpt_path)

            print(
                f"[iter {iteration:3d}] "
                f"loss={avg_loss:.4f}  "
                f"(π={avg_p:.4f}  v={avg_v:.4f})  "
                f"lr={self.scheduler.get_last_lr()[0]:.2e}  "
                f"gen={t1 - t0:.0f}s  train={t2 - t1:.1f}s  "
                f"buf={len(self.replay_buffer):,}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Mini training run (smoke test)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("AlphaZero Trainer — mini smoke test")
    print("=" * 60)

    # Small-scale configuration for a quick test
    N = 6
    net = AlphaReversiNet(N=N, num_channels=64, num_res_blocks=4)
    buf = ExperienceReplayBuffer(maxlen=5000)

    trainer = AlphaZeroTrainer(
        net, buf,
        lr=1e-3, weight_decay=1e-4,
        lr_step_size=10, lr_gamma=0.5,
        gradient_clip=1.0,
    )

    # Pre-populate buffer with a few games so we have something to sample
    print("\nSeeding buffer with 2 games …")
    trainer.generate_data(
        num_games=2, num_workers=2,
        N=N, obstacle_ratio=0.08, time_limit=1.0,
        temp_threshold=6, device="cpu",
    )
    print(f"Buffer size after seeding: {len(buf)}")

    # Quick training loop: 3 iterations × 3 games × 4 batches
    print("\nStarting mini training loop (3 iterations) …\n")
    trainer.train_loop(
        num_iterations=3,
        games_per_iter=3,
        num_workers=2,
        batch_size=32,
        batches_per_iter=4,
        save_dir="checkpoints_test",
        N=N, obstacle_ratio=0.08, time_limit=1.0,
        temp_threshold=6, device="cpu",
    )

    # Verify checkpoint files
    import glob
    ckpts = sorted(glob.glob("checkpoints_test/iter_*.pth"))
    print(f"\nSaved checkpoints: {len(ckpts)}")
    for c in ckpts:
        print(f"  {c}")

    # Quick validation — load checkpoint and run inference
    ck = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
    ck_N = ck.get("N", N) if isinstance(ck, dict) else N
    loaded = AlphaReversiNet(N=ck_N, num_channels=64, num_res_blocks=4)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    loaded.load_state_dict(sd)
    loaded.eval()
    x = torch.randn(1, 4, N, N)
    with torch.no_grad():
        p, v = loaded(x)
    print(f"\nLoaded checkpoint inference: p={p.shape}, v={v.shape}")
    print("All checks passed.")
