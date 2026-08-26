import torch
import torch.nn as nn


class BEVTemporalPredictor(nn.Module):
    """History-only BEV predictor. Predict is performed only at BEV level."""
    def __init__(self, channels=96, hidden=64, age_cap=8.0):
        super().__init__()
        self.age_cap = float(age_cap)
        in_ch = channels * 2 + 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )

    def forward(self, prev1, prev2, age_map):
        delta = prev1 - prev2
        age = (age_map / self.age_cap).clamp(0, 1).to(prev1.dtype)
        residual = self.net(torch.cat([prev1, delta, age], dim=1))
        return prev1 + residual


def linear_extrapolate(prev1, prev2, alpha=1.0):
    return prev1 + float(alpha) * (prev1 - prev2)
