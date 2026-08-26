#!/usr/bin/env python3
"""CPU-only unit tests for geometry/mask utilities. Run before GPU experiments."""
import random
import torch

from pcdet.models.adaptive_single_roi.roi_utils import (
    center_to_box,
    make_rpu_masks,
    oracle_center_from_error,
    rectangular_sum_map,
)


def brute_box_sum(x, center, roi_hw):
    H, W = x.shape[-2:]
    y0, y1, x0, x1 = center_to_box(center, roi_hw, H, W)
    return float(x[..., y0:y1, x0:x1].sum().item())


def test_oracle_exact():
    torch.manual_seed(1)
    H, W = 17, 19
    roi = (8, 7)
    x = torch.rand(2, 1, H, W)
    score = rectangular_sum_map(x, roi)
    for b in range(2):
        for _ in range(100):
            cx = random.randrange(W)
            cy = random.randrange(H)
            a = float(score[b, 0, cy, cx].item())
            e = brute_box_sum(x[b:b+1], (cx, cy), roi)
            assert abs(a - e) < 1e-4, (a, e, cx, cy)

    centers, score = oracle_center_from_error(x, roi)
    for b in range(2):
        got = brute_box_sum(x[b:b+1], centers[b], roi)
        best = max(brute_box_sum(x[b:b+1], (cx, cy), roi)
                   for cy in range(H) for cx in range(W))
        assert abs(got - best) < 1e-4, (got, best)


def test_masks():
    H, W = 18, 20
    roi = (8, 7)
    for center in [(0, 0), (W-1, H-1), (W//2, H//2)]:
        r, p, u, box = make_rpu_masks(center, roi, 3, H, W, 'cpu', torch.float32)
        assert torch.allclose(r + p + u, torch.ones_like(r))
        assert int(r.sum().item()) == roi[0] * roi[1]
        y0, y1, x0, x1 = box
        assert y1-y0 == roi[0] and x1-x0 == roi[1]


if __name__ == '__main__':
    test_oracle_exact()
    test_masks()
    print('single_roi unit tests: PASS')
