"""Optional 1-D ResNet ablation.

Not used for the main Phase 2 experiments; kept for later ablations.
Four residual blocks, channels [64, 128, 256, 512], BN after every conv,
stride-2 downsample between blocks. Head: GAP → Linear(512, num_classes).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock1D(nn.Module):
    def __init__(self, in_c: int, out_c: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_c)
        if stride != 1 or in_c != out_c:
            self.skip = nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_c),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNet1D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        channels = [64, 128, 256, 512]
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        blocks: list[nn.Module] = [_BasicBlock1D(channels[0], channels[0], stride=1)]
        for i in range(1, len(channels)):
            blocks.append(_BasicBlock1D(channels[i - 1], channels[i], stride=2))
        self.body = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) != 1 and x.size(2) == 1:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.stem(x)
        x = self.body(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"ResNet1D(params={total:,})"
