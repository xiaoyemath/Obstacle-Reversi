"""
train.py — Launch AlphaZero self-play training for ObstacleReversi.

Usage:
    python train.py                          # defaults: N=8, 128ch, cpu
    python train.py --N 6 --iter 50 --gpu   # small board, GPU
"""

from __future__ import annotations

import argparse
import os
import torch
from alpha_reversi_net import AlphaReversiNet
from self_play import ExperienceReplayBuffer
from trainer import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(
        description="Train AlphaReversiNet via self-play reinforcement learning"
    )
    parser.add_argument("--N", type=int, default=8, help="Board size (default 8)")
    parser.add_argument("--obstacle-ratio", type=float, default=0.1)
    parser.add_argument("--channels", type=int, default=128, help="Feature channels")
    parser.add_argument("--res-blocks", type=int, default=4, help="Residual blocks")
    parser.add_argument("--c-puct", type=float, default=1.4, help="PUCT constant")
    parser.add_argument("--time-limit", type=float, default=2.0, help="MCTS seconds per move")
    parser.add_argument("--temp-threshold", type=int, default=15, help="Opening moves with tau=1")
    parser.add_argument("--iter", dest="num_iterations", type=int, default=100)
    parser.add_argument("--games-per-iter", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4, help="Parallel self-play threads")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-iter", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=30, help="LR decay interval (iterations)")
    parser.add_argument("--lr-gamma", type=float, default=0.1, help="LR decay factor")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--buffer-size", type=int, default=50000)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--gpu", action="store_true", help="Use CUDA if available")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    args = parser.parse_args()

    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"

    state_dict = None
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            args.N = int(checkpoint.get("N", args.N))
            args.channels = int(checkpoint.get("num_channels", args.channels))
            args.res_blocks = int(checkpoint.get("num_res_blocks", args.res_blocks))
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

    print(f"Device: {device}")
    print(f"Board:  {args.N}×{args.N}, channels={args.channels}, "
          f"res_blocks={args.res_blocks}")
    print(f"Training: {args.num_iterations} iterations × "
          f"{args.games_per_iter} games × {args.batches_per_iter} batches")

    # ── Network ──────────────────────────────────────────────────────────
    net = AlphaReversiNet(
        N=args.N,
        num_channels=args.channels,
        num_res_blocks=args.res_blocks,
    )
    if state_dict is not None:
        net.load_state_dict(state_dict)
        print(f"Resumed from: {args.resume}")

    # ── Buffer & Trainer ──────────────────────────────────────────────────
    buf = ExperienceReplayBuffer(maxlen=args.buffer_size)
    trainer = AlphaZeroTrainer(
        net, buf,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lr_step_size=args.lr_step,
        lr_gamma=args.lr_gamma,
        gradient_clip=args.grad_clip,
        device=device,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    trainer.train_loop(
        num_iterations=args.num_iterations,
        games_per_iter=args.games_per_iter,
        num_workers=args.workers,
        batch_size=args.batch_size,
        batches_per_iter=args.batches_per_iter,
        save_dir=args.save_dir,
        N=args.N,
        obstacle_ratio=args.obstacle_ratio,
        c_puct=args.c_puct,
        time_limit=args.time_limit,
        temp_threshold=args.temp_threshold,
        device=device,
    )


if __name__ == "__main__":
    main()
