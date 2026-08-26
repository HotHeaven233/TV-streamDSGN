import math
from typing import Tuple

import torch
import torch.nn.functional as F


Box = Tuple[int, int, int, int]  # y0, y1, x0, x1, half-open


def _as_int(v):
    if isinstance(v, torch.Tensor):
        return int(v.item())
    return int(v)


def center_to_box(center_xy, roi_hw, H: int, W: int) -> Box:
    """Convert center (x, y) to a fixed-size half-open box inside HxW."""
    rh, rw = int(roi_hw[0]), int(roi_hw[1])
    rh = min(max(rh, 1), H)
    rw = min(max(rw, 1), W)

    cx, cy = _as_int(center_xy[0]), _as_int(center_xy[1])
    cx = max(0, min(W - 1, cx))
    cy = max(0, min(H - 1, cy))

    y0 = cy - rh // 2
    x0 = cx - rw // 2
    y0 = max(0, min(H - rh, y0))
    x0 = max(0, min(W - rw, x0))
    y1 = y0 + rh
    x1 = x0 + rw
    return y0, y1, x0, x1


def box_to_mask(box: Box, H: int, W: int, device=None, dtype=torch.float32):
    y0, y1, x0, x1 = box
    assert 0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W, (box, H, W)
    m = torch.zeros((1, 1, H, W), device=device, dtype=dtype)
    m[..., y0:y1, x0:x1] = 1
    return m


def expand_align_box(box: Box, halo: int, align: int, H: int, W: int) -> Box:
    """Expand box by halo then align start/end to the global lattice."""
    y0, y1, x0, x1 = box
    y0 = max(0, y0 - int(halo))
    y1 = min(H, y1 + int(halo))
    x0 = max(0, x0 - int(halo))
    x1 = min(W, x1 + int(halo))

    if align > 1:
        y0 = (y0 // align) * align
        x0 = (x0 // align) * align
        y1 = min(H, int(math.ceil(y1 / align) * align))
        x1 = min(W, int(math.ceil(x1 / align) * align))
    assert y0 < y1 and x0 < x1
    return y0, y1, x0, x1


def make_rpu_masks(center_xy, roi_hw, pred_margin: int, H: int, W: int, device, dtype):
    """
    R: one fixed-size rectangular ROI.
    P: a rectangular ring around R with pred_margin.
    U: everything else.
    """
    r_box = center_to_box(center_xy, roi_hw, H, W)
    r = box_to_mask(r_box, H, W, device=device, dtype=dtype)
    if pred_margin <= 0:
        p = torch.zeros_like(r)
        u = 1 - r
        return r, p, u, r_box

    y0, y1, x0, x1 = r_box
    p_box = (
        max(0, y0 - pred_margin),
        min(H, y1 + pred_margin),
        max(0, x0 - pred_margin),
        min(W, x1 + pred_margin),
    )
    outer = box_to_mask(p_box, H, W, device=device, dtype=dtype)
    p = (outer - r).clamp_(0, 1)
    u = (1 - outer).clamp_(0, 1)
    return r, p, u, r_box


def _valid_window_sum(x: torch.Tensor, roi_hw):
    """Return exact sums for every valid fixed-size top-left ROI."""
    assert x.ndim == 4 and x.shape[1] == 1
    h, w = int(roi_hw[0]), int(roi_hw[1])
    H, W = x.shape[-2:]
    h = min(max(h, 1), H)
    w = min(max(w, 1), W)

    integ = x.cumsum(dim=-2).cumsum(dim=-1)
    integ = F.pad(integ, (1, 0, 1, 0), mode='constant', value=0)
    sums = (
        integ[..., h:, w:]
        - integ[..., :-h, w:]
        - integ[..., h:, :-w]
        + integ[..., :-h, :-w]
    )
    assert sums.shape[-2:] == (H - h + 1, W - w + 1)
    return sums, h, w


def rectangular_sum_map(x: torch.Tensor, roi_hw):
    """
    Exact score map consistent with center_to_box().

    For centers near a boundary, center_to_box() shifts the fixed-size ROI inward
    instead of truncating it. This function follows exactly that rule, so the
    oracle score at every center is the sum of the box that will actually run.
    """
    sums, h, w = _valid_window_sum(x, roi_hw)
    H, W = x.shape[-2:]

    ys = torch.arange(H, device=x.device)
    xs = torch.arange(W, device=x.device)
    y0 = (ys - h // 2).clamp(0, H - h)
    x0 = (xs - w // 2).clamp(0, W - w)

    # Advanced indexing produces [B,1,H,W].
    out = sums[..., y0[:, None], x0[None, :]]
    assert out.shape[-2:] == (H, W)
    return out


def oracle_center_from_error(error_map: torch.Tensor, roi_hw):
    """
    error_map: [B,1,H,W], larger means more value from recomputation.
    returns centers [B,2] in (x,y), and a score map exactly matching center_to_box.
    """
    score = rectangular_sum_map(error_map, roi_hw)
    B, _, H, W = score.shape
    idx = score.flatten(1).argmax(dim=1)
    y = torch.div(idx, W, rounding_mode='floor')
    x = idx % W
    return torch.stack([x, y], dim=1), score


def gaussian_heatmap(centers_xy: torch.Tensor, H: int, W: int, sigma: float, device=None, dtype=None):
    """Build one Gaussian target heatmap per batch item."""
    B = centers_xy.shape[0]
    device = device if device is not None else centers_xy.device
    dtype = dtype if dtype is not None else torch.float32
    yy = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    xx = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    cx = centers_xy[:, 0].to(dtype).view(B, 1, 1)
    cy = centers_xy[:, 1].to(dtype).view(B, 1, 1)
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    heat = torch.exp(-d2 / max(2.0 * sigma * sigma, 1e-6))
    return heat.unsqueeze(1)
