"""Primary 1-D CNN for ECG classification.

BatchNorm-heavy by design: every conv and the fc layer are followed by a BN
module so downstream experiments (FedBN, DP-FedBN) can keep/share the right
subset of parameters. Works on both MIT-BIH (250-sample beats) and PTB-XL
(1000-sample Lead-II traces) via AdaptiveAvgPool1d(1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1D(nn.Module):
    def __init__(self, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.pool = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(256, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(1) != 1 and x.size(2) == 1:
            x = x.transpose(1, 2)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

    def __repr__(self) -> str:
        total = sum(p.numel() for p in self.parameters())
        return f"CNN1D(params={total:,})"
