import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyStereoContext(nn.Module):
    """
    Very small current-frame stereo encoder.
    It intentionally produces a global context vector rather than pretending
    that image pixels and BEV cells are geometrically aligned.
    Spatial localization is mainly carried by previous BEV + age.
    """
    def __init__(self, context_dim=32, input_hw=(80, 312)):
        super().__init__()
        self.input_hw = tuple(input_hw)
        self.net = nn.Sequential(
            nn.Conv2d(6, 16, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, context_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(context_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, left, right):
        x = torch.cat([left, right], dim=1)
        x = F.interpolate(x, size=self.input_hw, mode='bilinear', align_corners=False)
        return self.net(x).flatten(1)


class BEVCenterPredictor(nn.Module):
    """
    Predict one BEV refresh-center heatmap.

    Inputs:
      left/right: current stereo images [B,3,Himg,Wimg]
      prev_bev:   previous adaptive BEV [B,C,H,W]
      age_map:    true-observation age [B,1,H,W]

    Output:
      logits: [B,1,H,W]
    """
    def __init__(self, bev_channels=96, hidden=32, context_dim=32, age_cap=8.0):
        super().__init__()
        self.age_cap = float(age_cap)
        self.stereo_context = TinyStereoContext(context_dim=context_dim)

        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.age_encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden + hidden + 8, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 1),
        )

    def forward(self, left, right, prev_bev, age_map):
        bev = self.bev_encoder(prev_bev)
        age = self.age_encoder((age_map / self.age_cap).clamp(0, 1))
        ctx = self.stereo_context(left, right)
        ctx = self.context_proj(ctx)[:, :, None, None].expand(
            -1, -1, prev_bev.shape[-2], prev_bev.shape[-1]
        )
        return self.head(torch.cat([bev, age, ctx], dim=1))

    @torch.no_grad()
    def predict_center(self, left, right, prev_bev, age_map):
        logits = self(left, right, prev_bev, age_map)
        B, _, H, W = logits.shape
        idx = logits.flatten(1).argmax(dim=1)
        y = torch.div(idx, W, rounding_mode='floor')
        x = idx % W
        return torch.stack([x, y], dim=1), logits
