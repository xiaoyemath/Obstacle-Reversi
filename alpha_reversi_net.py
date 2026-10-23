"""
AlphaReversiNet — AlphaGo-style dual-head CNN for ObstacleReversi.

Input:  (B, 4, N, N) — my pieces, opponent pieces, obstacles, turn indicator
Output: p (B, N²) action probabilities,  v (B, 1) state value in [-1, 1]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Residual block
# ──────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Standard residual block: 2 × (Conv3x3 → BN → ReLU) + skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity, inplace=True)


# ──────────────────────────────────────────────────────────────────────────────
# AlphaReversiNet
# ──────────────────────────────────────────────────────────────────────────────

class AlphaReversiNet(nn.Module):
    """Dual-head CNN for the obstacle-Reversi game.

    Parameters
    ----------
    N : int
        Board size (default 8).
    num_channels : int
        Feature channels throughout the residual tower (default 128).
    num_res_blocks : int
        Number of residual blocks (default 4).
    """

    def __init__(
        self, N: int = 8, num_channels: int = 128, num_res_blocks: int = 4
    ):
        super().__init__()
        self.N = N

        # ── initial conv:  4 ch  →  num_channels ──────────────────────
        self.conv_init = nn.Sequential(
            nn.Conv2d(4, num_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # ── residual tower ─────────────────────────────────────────────
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # ── policy (actor) head ────────────────────────────────────────
        self.policy_conv = nn.Sequential(
            nn.Conv2d(num_channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(2 * N * N, N * N)

        # ── value (critic) head ────────────────────────────────────────
        self.value_conv = nn.Sequential(
            nn.Conv2d(num_channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
        )
        self.value_fc1 = nn.Linear(N * N, 64)
        self.value_fc2 = nn.Linear(64, 1)

    # ──────────────────────────────────────────────────────────────────
    # Forward pass
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor = None,
        return_logits: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Forward pass.

        Args:
            x:          (B, 4, N, N)  — 4-channel board representation.
            legal_mask: (B, N*N) bool  — True for legal actions; if None,
                        softmax is taken over all actions.
            return_logits: If True, also return the raw (unmasked) policy
                        logits as a third output — useful for numerically
                        stable loss computation.

        Returns:
            p:      (B, N*N)  — action probability distribution.
            v:      (B, 1)    — scalar state value in [-1, 1].
            logits: (B, N*N)  — raw policy logits (only if *return_logits*).
        """
        # Shared backbone
        features = self.conv_init(x)
        features = self.res_blocks(features)

        # ── Policy head ───────────────────────────────────────────────
        p = self.policy_conv(features)      # (B, 2, N, N)
        p = p.flatten(1)                    # (B, 2·N·N)
        logits = self.policy_fc(p)          # (B, N·N) — raw logits
        p = logits

        if legal_mask is not None:
            p = p.masked_fill(~legal_mask, -1e9)

        p = F.softmax(p, dim=-1)

        # ── Value head ────────────────────────────────────────────────
        v = self.value_conv(features)       # (B, 1, N, N)
        v = v.flatten(1)                    # (B, N·N)
        v = F.relu(self.value_fc1(v))       # (B, 64)
        v = torch.tanh(self.value_fc2(v))   # (B, 1)

        if return_logits:
            return p, v, logits
        return p, v

    # ──────────────────────────────────────────────────────────────────
    # Input builder (NumPy → torch tensor)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def build_input(
        board: np.ndarray, current_player: int
    ) -> torch.Tensor:
        """Convert a game-state matrix into a 4-channel network input.

        Args:
            board:           (N, N) int8  — {-1, 0, 1, 2} as in ObstacleReversiEnv.
            current_player:  -1 or 1      — whose turn it is.

        Returns:
            Tensor of shape (1, 4, N, N), float32.
        """
        N = board.shape[0]
        ch0 = (board == current_player).astype(np.float32)
        ch1 = (board == -current_player).astype(np.float32)
        ch2 = (board == 2).astype(np.float32)
        # Turn indicator: all-1 for player -1, all-0 for player 1
        ch3 = np.full((N, N), 1.0 if current_player == -1 else 0.0,
                      dtype=np.float32)

        stacked = np.stack([ch0, ch1, ch2, ch3], axis=0)  # (4, N, N)
        return torch.from_numpy(stacked).unsqueeze(0)       # (1, 4, N, N)


# ──────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    net = AlphaReversiNet(N=8, num_channels=64, num_res_blocks=4)
    print(net)
    total = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total:,}")

    # Dummy forward pass
    x = torch.randn(2, 4, 8, 8)
    mask = torch.ones(2, 64, dtype=torch.bool)
    mask[0, 0] = False  # make action 0 illegal for batch item 0

    with torch.no_grad():
        p, v = net(x, mask)

    print(f"p shape: {p.shape}  (expected [2, 64])")
    print(f"v shape: {v.shape}  (expected [2, 1])")
    print(f"p[0] sum: {p[0].sum().item():.4f}  (should be 1.0)")
    print(f"p[0, 0]:  {p[0, 0].item():.6f}   (should be 0 — masked)")
    print(f"v range:  [{v.min().item():.3f}, {v.max().item():.3f}]")

    # Test build_input with a mock board
    from obstacle_reversi_env import ObstacleReversiEnv
    env = ObstacleReversiEnv(N=8)
    env.reset()
    inp = AlphaReversiNet.build_input(env.board, env.current_player)
    print(f"\nbuild_input shape: {inp.shape}  (expected [1, 4, 8, 8])")
    p2, v2 = net(inp)
    print(f"Single-board forward: p={p2.shape}, v={v2.shape}")
    print("All checks passed.")
