import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int):
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, groups=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, groups=groups, bias=False),
            _gn(out_ch),
            nn.SiLU(inplace=True),
        )


class StageImportancePredictor(nn.Module):
    """
    End-to-end stage-wise importance predictor with NO explicit importance labels.

    Inputs
    ------
    cur_left/cur_right:
        Current A_s (= stem + layer1) stereo features.
    prev_left/prev_right:
        Previous-frame A_s stereo features. A_s is always full, therefore this
        history is cheap and does not suffer from selective-cache ambiguity.
    prev_bev:
        Previous emitted pre-fusion BEV feature (spatial_features).
    age_a/age_b/age_f:
        Stage-A, stage-B and final-BEV age maps. Age is state, not importance.
    a_hw/b_hw/cd_hw:
        Native output H/W for Q_A, Q_B and Q_CD.

    Outputs
    -------
    q_a, q_b, q_cd:
        Positive, spatially normalized importance maps. Each map sums to one
        per sample. They receive supervision only through downstream BEV
        reconstruction and detection losses.
    """

    def __init__(
        self,
        shallow_channels=64,
        bev_channels=96,
        hidden=32,
        age_cap=8.0,
    ):
        super().__init__()
        self.age_cap = float(age_cap)

        # [cur, prev, |cur-prev|] for left and right: 6 * shallow_channels.
        shallow_in = shallow_channels * 6
        self.shallow_encoder = nn.Sequential(
            ConvGNAct(shallow_in, hidden, kernel_size=1, padding=0),
            ConvGNAct(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            ConvGNAct(hidden, hidden, kernel_size=1, padding=0),
        )

        self.bev_encoder = nn.Sequential(
            ConvGNAct(bev_channels + 1, hidden, kernel_size=1, padding=0),
            ConvGNAct(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
            ConvGNAct(hidden, hidden, kernel_size=1, padding=0),
        )

        self.shared_encoder = nn.Sequential(
            ConvGNAct(hidden * 2, hidden, kernel_size=3, padding=1),
            ConvGNAct(hidden, hidden, kernel_size=3, padding=1, groups=hidden),
        )

        self.head_a = nn.Sequential(
            ConvGNAct(hidden + 1, hidden, kernel_size=3, padding=1),
            nn.Conv2d(hidden, 1, 1),
        )
        self.head_b = nn.Sequential(
            ConvGNAct(hidden + 2, hidden, kernel_size=3, padding=1),
            nn.Conv2d(hidden, 1, 1),
        )
        self.head_cd = nn.Sequential(
            ConvGNAct(hidden + 2, hidden, kernel_size=3, padding=1),
            nn.Conv2d(hidden, 1, 1),
        )

    def _age(self, age, hw, dtype):
        age = F.interpolate(age.float(), size=hw, mode='nearest')
        return (age / self.age_cap).clamp(0.0, 1.0).to(dtype)

    @staticmethod
    def _normalize(logits):
        # softplus guarantees non-negative refresh value while keeping gradients.
        q = F.softplus(logits)
        return q / (q.sum(dim=(-2, -1), keepdim=True) + 1e-6)

    def forward(
        self,
        cur_left,
        cur_right,
        prev_left,
        prev_right,
        prev_bev,
        age_a,
        age_b,
        age_f,
        a_hw,
        b_hw,
        cd_hw,
    ):
        # A and B currently use the common 80x312 native grid. Use A as the
        # shared-context grid and let each head resize to its own native space.
        base_hw = tuple(int(v) for v in a_hw)

        def rs(x):
            return F.interpolate(x, size=base_hw, mode='bilinear', align_corners=False)

        cur_left_b = rs(cur_left)
        cur_right_b = rs(cur_right)
        prev_left_b = rs(prev_left)
        prev_right_b = rs(prev_right)

        shallow = torch.cat(
            [
                cur_left_b,
                prev_left_b,
                (cur_left_b - prev_left_b).abs(),
                cur_right_b,
                prev_right_b,
                (cur_right_b - prev_right_b).abs(),
            ],
            dim=1,
        )
        z_s = self.shallow_encoder(shallow)

        prev_bev_b = F.interpolate(prev_bev, size=base_hw, mode='bilinear', align_corners=False)
        age_f_b = self._age(age_f, base_hw, prev_bev_b.dtype)
        z_f = self.bev_encoder(torch.cat([prev_bev_b, age_f_b], dim=1))
        z = self.shared_encoder(torch.cat([z_s, z_f], dim=1))

        age_a_b = self._age(age_a, base_hw, z.dtype)
        age_b_b = self._age(age_b, base_hw, z.dtype)
        age_f_b = self._age(age_f, base_hw, z.dtype)

        logit_a = self.head_a(torch.cat([z, age_a_b], dim=1))
        logit_b = self.head_b(torch.cat([z, age_a_b, age_b_b], dim=1))
        logit_cd = self.head_cd(torch.cat([z, age_b_b, age_f_b], dim=1))

        if tuple(logit_a.shape[-2:]) != tuple(a_hw):
            logit_a = F.interpolate(logit_a, size=a_hw, mode='bilinear', align_corners=False)
        if tuple(logit_b.shape[-2:]) != tuple(b_hw):
            logit_b = F.interpolate(logit_b, size=b_hw, mode='bilinear', align_corners=False)
        if tuple(logit_cd.shape[-2:]) != tuple(cd_hw):
            logit_cd = F.interpolate(logit_cd, size=cd_hw, mode='bilinear', align_corners=False)

        return {
            'q_a': self._normalize(logit_a),
            'q_b': self._normalize(logit_b),
            'q_cd': self._normalize(logit_cd),
            'logit_a': logit_a,
            'logit_b': logit_b,
            'logit_cd': logit_cd,
        }
