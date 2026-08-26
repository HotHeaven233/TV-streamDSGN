#!/usr/bin/env python3
"""
Physical B->C->D experiment for one BEV ROI.

It directly answers whether dependency-aware upstream support is necessary:
  dep_bbox    : B ROI is the exact rectangular bbox of Omega_B(CD-exec ROI)
  shared      : same B ROI size, but center is naively mapped from BEV center
  full_b      : B is Full, C+D remain one ROI
  full_bcd    : full B+C+D baseline

The timed scope starts AFTER the expensive 2D feature frontend. A full teacher
forward is used outside the timed region only to capture A->B inputs, mapping
coordinates, and the numerical reference.

Prerequisite for dep_bbox/shared:
  your local BuildCostVolume.forward_roi(...) DPS ROI implementation.
The public GitHub main branch does not contain that API.
"""
import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcdet.models import load_data_to_gpu
from pcdet.models.adaptive_single_roi.roi_utils import center_to_box, expand_align_box

from single_roi_common import add_base_args, build_env, cuda_autocast, find_module, percentile, save_json


def parse_args():
    p = argparse.ArgumentParser('Single-ROI dependency physical oracle')
    add_base_args(p, training=False)
    p.add_argument('--roi-h', type=int, default=96)
    p.add_argument('--roi-w', type=int, default=96)
    p.add_argument('--centers', nargs='*', default=['0.50,0.50', '0.35,0.50', '0.65,0.50'])
    p.add_argument('--modes', nargs='*', choices=['dep_bbox', 'shared', 'full_b', 'full_bcd'], default=['dep_bbox', 'shared', 'full_b', 'full_bcd'])
    p.add_argument('--cd-halo', type=int, default=12)
    p.add_argument('--cd-align', type=int, default=4)
    p.add_argument('--b-halo', type=int, default=1)
    p.add_argument('--b-align', type=int, default=4)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--repeat', type=int, default=50)
    p.add_argument('--numeric-tol', type=float, default=0.005)
    p.add_argument('--output', default='outputs/single_roi/dependency_physical.json')
    return p.parse_args()


def parse_center(s, H, W):
    a, b = s.split(',')
    nx, ny = float(a), float(b)
    x = int(round(nx * (W - 1)))
    y = int(round(ny * (H - 1)))
    return x, y


def bbox_from_grid_support(norm_grid, valid_mask, H_b, W_b):
    gx = (norm_grid[..., 0].float() + 1.0) * 0.5 * (W_b - 1)
    gy = (norm_grid[..., 1].float() + 1.0) * 0.5 * (H_b - 1)
    m = valid_mask.bool()
    if m.any():
        gx = gx[m]
        gy = gy[m]
    else:
        gx = gx.reshape(-1)
        gy = gy.reshape(-1)
    x0 = max(0, int(torch.floor(gx.min()).item()))
    x1 = min(W_b, int(torch.ceil(gx.max()).item()) + 1)
    y0 = max(0, int(torch.floor(gy.min()).item()))
    y1 = min(H_b, int(torch.ceil(gy.max()).item()) + 1)
    return y0, y1, x0, x1


def same_size_centered_box(center_xy, hw, H, W):
    h, w = hw
    return center_to_box(center_xy, (h, w), H, W)


def global_grid_to_local(norm_grid, y0, x0, H_full, W_full, H_roi, W_roi):
    g = norm_grid.clone().float()
    xg = (g[..., 0] + 1.0) * 0.5 * (W_full - 1)
    yg = (g[..., 1] + 1.0) * 0.5 * (H_full - 1)
    xl = xg - float(x0)
    yl = yg - float(y0)
    if W_roi > 1:
        g[..., 0] = 2.0 * xl / float(W_roi - 1) - 1.0
    else:
        g[..., 0] = 0
    if H_roi > 1:
        g[..., 1] = 2.0 * yl / float(H_roi - 1) - 1.0
    else:
        g[..., 1] = 0
    return g.to(norm_grid.dtype)


def run_d(backbone, voxel):
    x = backbone.rpn3d_convs(voxel)
    if backbone.num_3dconvs_hg > 0:
        if backbone.num_3dconvs_hg == 1:
            pre, post = True, True
            for hg in backbone.rpn3d_hgs:
                x, pre, post = hg(x, pre, post)
        else:
            pre, post = None, None
            for hg in backbone.rpn3d_hgs:
                x = hg(x, pre, post)
    x = backbone.rpn3d_pool(x)
    B, C, D, H, W = x.shape
    return x.view(B, C * D, H, W)


def capture_teacher(backbone, map_to_bev, frame, amp_enabled):
    cap = {}

    def pre_build_cost(_m, args):
        cap['b_args'] = tuple(a.detach() if torch.is_tensor(a) else a for a in args)

    def hook_dres1(_m, args, out):
        cap['b_full'] = (out + args[0]).detach()

    h1 = backbone.build_cost.register_forward_pre_hook(pre_build_cost)
    h2 = backbone.dres1.register_forward_hook(hook_dres1)
    try:
        x = dict(frame)
        with torch.no_grad(), cuda_autocast(amp_enabled):
            x = backbone(x)
            x = map_to_bev(x)
    finally:
        h1.remove()
        h2.remove()
    cap['norm'] = x['norm_coord_imgs'].detach()
    cap['valids'] = x['valids'].detach()
    cap['full_bev'] = x['spatial_features'].detach()
    return cap


def run_b_full(backbone, b_args):
    cost = backbone.build_cost(*b_args)
    c0 = backbone.dres0(cost)
    return backbone.dres1(c0) + c0


def run_b_roi(backbone, b_args, box_exec):
    if not hasattr(backbone.build_cost, 'forward_roi'):
        raise RuntimeError(
            'BuildCostVolume.forward_roi is missing. Public GitHub main does not contain the ROI DPS API. '
            'Apply/use your existing local ROI DPS patch before running dep_bbox/shared.'
        )
    y0, y1, x0, x1 = box_exec
    left, right, left_raw, right_raw, shift = b_args[:5]
    psv = b_args[5] if len(b_args) > 5 else None
    cost = backbone.build_cost.forward_roi(
        left, right, left_raw, right_raw, shift,
        psv_disps_channels=psv,
        ph0=y0, ph1=y1, pw0=x0, pw1=x1,
    )
    c0 = backbone.dres0(cost)
    return backbone.dres1(c0) + c0


def run_cd_from_b(backbone, b, norm, valids, cd_exec, b_origin=(0, 0), b_full_hw=None):
    cy0, cy1, cx0, cx1 = cd_exec
    grid = norm[:, :, cy0:cy1, cx0:cx1, :]
    valid = valids[:, :, cy0:cy1, cx0:cx1]

    if b_origin != (0, 0):
        assert b_full_hw is not None
        Hf, Wf = b_full_hw
        grid = global_grid_to_local(
            grid, b_origin[0], b_origin[1], Hf, Wf, b.shape[-2], b.shape[-1]
        )
    voxel = F.grid_sample(b, grid, align_corners=True)
    voxel = voxel * valid[:, None].to(voxel.dtype)
    return run_d(backbone, voxel)


def timed_cuda(fn, warmup, repeat):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(repeat):
        st = torch.cuda.Event(enable_timing=True)
        ed = torch.cuda.Event(enable_timing=True)
        st.record()
        fn()
        ed.record()
        ed.synchronize()
        vals.append(float(st.elapsed_time(ed)))
    return {
        'mean_ms': float(np.mean(vals)),
        'p50_ms': percentile(vals, 50),
        'p95_ms': percentile(vals, 95),
        'p99_ms': percentile(vals, 99),
    }


def main():
    args = parse_args()
    cfg, dataset, loader, model, logger = build_env(args, training=False)
    assert args.batch_size == 1
    backbone = find_module(model, 'StreamDSGN2Backbone')
    map_to_bev = find_module(model, 'HeightCompression')
    assert backbone.num_hg == 0, 'script currently targets the provided num_hg=0 config'
    assert not backbone.cat_img_feature and not backbone.cat_right_img_feature, 'script targets the provided config without image-feature concatenation'

    amp_enabled = bool(model.use_amp_dict.get('TEST', False))
    results = []

    for fi, batch in enumerate(loader):
        if args.max_batches > 0 and fi >= args.max_batches:
            break
        load_data_to_gpu(batch)
        frame = batch['token']
        cap = capture_teacher(backbone, map_to_bev, frame, amp_enabled)
        full_bev = cap['full_bev']
        full_b_ref = cap['b_full']
        norm = cap['norm']
        valids = cap['valids']
        b_args = cap['b_args']
        H_bev, W_bev = full_bev.shape[-2:]
        H_b, W_b = full_b_ref.shape[-2:]
        assert torch.isfinite(full_bev).all() and torch.isfinite(full_b_ref).all()

        for center_str in args.centers:
            center = parse_center(center_str, H_bev, W_bev)
            core = center_to_box(center, (args.roi_h, args.roi_w), H_bev, W_bev)
            cd_exec = expand_align_box(core, args.cd_halo, args.cd_align, H_bev, W_bev)
            cy0, cy1, cx0, cx1 = cd_exec
            support_grid = norm[:, :, cy0:cy1, cx0:cx1, :]
            support_valid = valids[:, :, cy0:cy1, cx0:cx1]
            dep_core = bbox_from_grid_support(support_grid, support_valid, H_b, W_b)
            dep_h = dep_core[1] - dep_core[0]
            dep_w = dep_core[3] - dep_core[2]
            mapped_center = (
                int(round(center[0] / max(W_bev - 1, 1) * (W_b - 1))),
                int(round(center[1] / max(H_bev - 1, 1) * (H_b - 1))),
            )
            shared_core = same_size_centered_box(mapped_center, (dep_h, dep_w), H_b, W_b)

            # Full BCD numerical reference for this core, same direct physical boundary.
            def full_bcd_fn():
                with cuda_autocast(amp_enabled):
                    b = run_b_full(backbone, b_args)
                    voxel = F.grid_sample(b, norm, align_corners=True)
                    voxel = voxel * valids[:, None].to(voxel.dtype)
                    return run_d(backbone, voxel)

            with torch.no_grad():
                full_recon = full_bcd_fn()
            full_core_ref = full_recon[..., core[0]:core[1], core[2]:core[3]].detach()

            for mode in args.modes:
                if mode == 'full_bcd':
                    fn = full_bcd_fn
                    b_exec = (0, H_b, 0, W_b)
                    cd_box = (0, H_bev, 0, W_bev)

                    def get_core(out):
                        return out[..., core[0]:core[1], core[2]:core[3]]
                else:
                    cd_box = cd_exec
                    if mode == 'full_b':
                        b_exec = (0, H_b, 0, W_b)

                        def fn():
                            with cuda_autocast(amp_enabled):
                                b = run_b_full(backbone, b_args)
                                return run_cd_from_b(backbone, b, norm, valids, cd_box)
                    else:
                        b_core = dep_core if mode == 'dep_bbox' else shared_core
                        b_exec = expand_align_box(b_core, args.b_halo, args.b_align, H_b, W_b)

                        def fn(b_exec=b_exec):
                            with cuda_autocast(amp_enabled):
                                b = run_b_roi(backbone, b_args, b_exec)
                                return run_cd_from_b(
                                    backbone, b, norm, valids, cd_box,
                                    b_origin=(b_exec[0], b_exec[2]),
                                    b_full_hw=(H_b, W_b),
                                )

                    ry0 = core[0] - cd_box[0]
                    ry1 = ry0 + (core[1] - core[0])
                    rx0 = core[2] - cd_box[2]
                    rx1 = rx0 + (core[3] - core[2])

                    def get_core(out, ry0=ry0, ry1=ry1, rx0=rx0, rx1=rx1):
                        return out[..., ry0:ry1, rx0:rx1]

                with torch.no_grad():
                    out = fn()
                    core_out = get_core(out)
                    maxdiff = float((core_out.float() - full_core_ref.float()).abs().max().item())
                    meandiff = float((core_out.float() - full_core_ref.float()).abs().mean().item())
                    assert core_out.shape == full_core_ref.shape, (core_out.shape, full_core_ref.shape)
                    assert torch.isfinite(core_out).all(), 'selective core contains NaN/Inf'
                    timing = timed_cuda(fn, args.warmup, args.repeat)

                rec = {
                    'frame': fi,
                    'center_norm': center_str,
                    'center_xy': list(center),
                    'mode': mode,
                    'bev_core': list(core),
                    'cd_exec': list(cd_box),
                    'b_dep_core': list(dep_core),
                    'b_shared_core': list(shared_core),
                    'b_exec': list(b_exec),
                    'cd_exec_ratio': (cd_box[1]-cd_box[0])*(cd_box[3]-cd_box[2])/(H_bev*W_bev),
                    'b_exec_ratio': (b_exec[1]-b_exec[0])*(b_exec[3]-b_exec[2])/(H_b*W_b),
                    'max_diff': maxdiff,
                    'mean_diff': meandiff,
                    'correct_under_tol': bool(maxdiff <= args.numeric_tol),
                    **timing,
                }
                results.append(rec)
                print(
                    f"frame={fi} center={center_str} mode={mode:9s} "
                    f"B={100*rec['b_exec_ratio']:.2f}% CD={100*rec['cd_exec_ratio']:.2f}% "
                    f"p99={rec['p99_ms']:.3f} ms maxdiff={maxdiff:.6g}"
                )
        # One frame is enough by default unless user asks more.
        if args.max_batches == 0:
            break

    summary = {
        'scope': 'direct CUDA-event B->C->D physical execution after the 2D frontend; not end-to-end model-forward latency',
        'roi_hw': [args.roi_h, args.roi_w],
        'cd_halo_align': [args.cd_halo, args.cd_align],
        'b_halo_align': [args.b_halo, args.b_align],
        'numeric_tol': args.numeric_tol,
        'results': results,
    }
    save_json(args.output, summary)
    print(f'Saved: {args.output}')


if __name__ == '__main__':
    main()
