import math
import torch
import torch.nn.functional as F


def roi_hw_from_ratio(H, W, ratio, align=4):
    """Aspect-ratio-preserving rectangular ROI size for a target area ratio."""
    ratio = float(ratio)
    if ratio <= 0.0:
        return 1, 1
    if ratio >= 1.0:
        return int(H), int(W)
    scale = math.sqrt(ratio)
    h = int(round(H * scale / align)) * align
    w = int(round(W * scale / align)) * align
    h = min(int(H), max(int(align), h))
    w = min(int(W), max(int(align), w))
    return h, w


def straight_through_rect_mask(importance, roi_hw, temperature=0.5):
    """
    Hard rectangle in forward, soft differentiable window selection in backward.

    importance: [B,1,H,W], positive map (normally normalized to sum=1).
    roi_hw: (h,w) on the same native grid.

    Returns
    -------
    st_mask: hard rectangle numerically, gradient from soft rectangle mixture.
    hard_mask: detached exact 0/1 rectangle, used for cache/age state updates.
    score_map: valid top-left window scores [B,1,H-h+1,W-w+1].
    hard_xy: selected top-left (x,y), [B,2].
    """
    assert importance.ndim == 4 and importance.shape[1] == 1
    B, _, H, W = importance.shape
    h = min(max(int(roi_hw[0]), 1), H)
    w = min(max(int(roi_hw[1]), 1), W)

    # Exact rectangular integral, expressed as avg-pool * area. This is fully
    # differentiable with respect to the importance map.
    score = F.avg_pool2d(importance, kernel_size=(h, w), stride=1) * float(h * w)
    flat = score.flatten(1)

    temp = max(float(temperature), 1e-4)
    soft_prob = F.softmax(flat / temp, dim=1)
    hard_idx = flat.argmax(dim=1)

    hard_prob = torch.zeros_like(soft_prob)
    hard_prob.scatter_(1, hard_idx[:, None], 1.0)

    Hv, Wv = score.shape[-2:]
    soft_top_left = soft_prob.view(B, 1, Hv, Wv)
    hard_top_left = hard_prob.view(B, 1, Hv, Wv)

    kernel = torch.ones((1, 1, h, w), device=importance.device, dtype=importance.dtype)
    soft_mask = F.conv_transpose2d(soft_top_left.to(importance.dtype), kernel)
    hard_mask = F.conv_transpose2d(hard_top_left.to(importance.dtype), kernel)

    # Forward value == hard_mask. Backward derivative == derivative(soft_mask).
    st_mask = hard_mask.detach() - soft_mask.detach() + soft_mask

    y = torch.div(hard_idx, Wv, rounding_mode='floor')
    x = hard_idx % Wv
    hard_xy = torch.stack([x, y], dim=1)
    return st_mask, hard_mask.detach(), score, hard_xy


def rpu_masks_from_recompute(recompute_st, recompute_hard, pred_margin):
    """
    Build BEV Recompute/Predict/Reuse masks.

    P = Dilate(R, margin) \\ R
    U = complement(Dilate(R, margin))

    The ST masks are used in the differentiable training composition; hard masks
    are used for cache/age state updates.
    """
    margin = int(pred_margin)
    if margin <= 0:
        p_st = torch.zeros_like(recompute_st)
        u_st = 1.0 - recompute_st
        p_hard = torch.zeros_like(recompute_hard)
        u_hard = 1.0 - recompute_hard
        return recompute_st, p_st, u_st, recompute_hard, p_hard, u_hard

    k = margin * 2 + 1
    outer_st = F.max_pool2d(recompute_st, kernel_size=k, stride=1, padding=margin)
    outer_hard = F.max_pool2d(recompute_hard, kernel_size=k, stride=1, padding=margin)

    p_st = (outer_st - recompute_st).clamp(0.0, 1.0)
    u_st = (1.0 - outer_st).clamp(0.0, 1.0)
    p_hard = (outer_hard - recompute_hard).clamp(0.0, 1.0)
    u_hard = (1.0 - outer_hard).clamp(0.0, 1.0)
    return recompute_st, p_st, u_st, recompute_hard, p_hard, u_hard
