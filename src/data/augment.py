"""ECG augmentations for PTB-XL centralized + federated training.

All ops expect (B, C, L) float tensors. Only active during .train().
In .eval() mode: deterministic left-crop to `crop_length` for shape
compatibility with the trained model.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def random_crop_1d(x: torch.Tensor, out_length: int) -> torch.Tensor:
    """Random crop along the time axis (last dim).

    Input shape (..., L) with L >= out_length. Picks one start offset
    for the whole batch so the conv stack sees a consistent window size.
    """
    L = x.size(-1)
    if L == out_length:
        return x
    if L < out_length:
        raise ValueError(f"input length {L} < crop length {out_length}")
    max_start = L - out_length
    start = int(torch.randint(0, max_start + 1, (1,)).item())
    return x[..., start : start + out_length]


def gaussian_noise(x: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Add N(0, std) noise. Signals are z-scored so std is in normalized units."""
    if std <= 0.0:
        return x
    return x + std * torch.randn_like(x)


def baseline_wander(x: torch.Tensor, max_shift: float = 0.05) -> torch.Tensor:
    """Add a slow sinusoidal baseline drift.

    Amplitude uniform in [0, max_shift], frequency 0.5–1.5 cycles across
    the window, random phase. Broadcasts over batch + channel dims.
    """
    if max_shift <= 0.0:
        return x
    L = x.size(-1)
    n_periods = 0.5 + float(torch.rand(1).item())  # 0.5..1.5 cycles
    phase = float(torch.rand(1).item()) * 2.0 * math.pi
    amp = float(torch.rand(1).item()) * max_shift
    t = torch.linspace(
        0.0, n_periods * 2.0 * math.pi, L, device=x.device, dtype=x.dtype
    )
    drift = amp * torch.sin(t + phase)
    shape = [1] * (x.dim() - 1) + [L]
    return x + drift.view(*shape)


class PTBXLAugment(nn.Module):
    """Training-time augmentation wrapper for PTB-XL.

    In train mode: random crop to `crop_length`, then stochastically apply
    gaussian noise and baseline wander with probability `prob` each.
    In eval mode: deterministic left-crop to `crop_length`.
    """

    def __init__(
        self,
        crop_length: int = 900,
        noise_std: float = 0.02,
        baseline_max: float = 0.05,
        prob: float = 0.5,
    ):
        super().__init__()
        self.crop_length = crop_length
        self.noise_std = noise_std
        self.baseline_max = baseline_max
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            if x.size(-1) > self.crop_length:
                return x[..., : self.crop_length]
            return x

        if x.size(-1) > self.crop_length:
            x = random_crop_1d(x, self.crop_length)
        if torch.rand(1).item() < self.prob:
            x = gaussian_noise(x, self.noise_std)
        if torch.rand(1).item() < self.prob:
            x = baseline_wander(x, self.baseline_max)
        return x
