#!/usr/bin/env python3
"""
Joint end-to-end training for:
  1) StageImportancePredictor -> Q_A, Q_B, Q_CD
  2) BEVTemporalPredictor     -> predicted current BEV

There is NO explicit importance-map supervision.

Loss:
    L = L_det + lambda_l1 * L1(F_mix, F_full)
              + lambda_cos * CosLoss(F_mix, F_full)

Training uses full current stage computation plus differentiable hard/soft masks
as a surrogate for physical ROI execution. Physical ROI kernels are used only in
profiling/inference after training.
"""

import argparse
import copy
import random
from collections import deque
from pathlib import Path

import torch
import torch.nn.functional as F

from pcdet.models import load_data_to_gpu
from pcdet.models.adaptive_single_roi import (
    StageImportancePredictor,
    BEVTemporalPredictor,
    roi_hw_from_ratio,
    straight_through_rect_mask,
    rpu_masks_from_recompute,
)

from single_roi_common import add_base_args, build_env, cuda_autocast, find_module


def parse_float_list(text):
    vals = [float(x.strip()) for x in text.split(',') if x.strip()]
    if not vals:
        raise ValueError('empty budget list')
    for v in vals:
        if not (0.0 < v <= 1.0):
            raise ValueError(f'budget ratio must be in (0,1], got {v}')
    return vals


def parse_args():
    p = argparse.ArgumentParser('Joint train stage importance + BEV predictor')
    add_base_args(p, training=True)
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--pred-hidden', type=int, default=64)
    p.add_argument('--age-cap', type=int, default=8)
    p.add_argument('--pred-margin', type=int, default=12,
                   help='BEV cells around CD recompute ROI assigned to Predict')
    p.add_argument('--a-budgets', default='0.15,0.25,0.40,0.60,0.80')
    p.add_argument('--b-budgets', default='0.15,0.25,0.40,0.60,0.80')
    p.add_argument('--cd-budgets', default='0.15,0.25,0.40,0.60,0.80')
    p.add_argument('--roi-align', type=int, default=4)
    p.add_argument('--tau-start', type=float, default=1.0)
    p.add_argument('--tau-end', type=float, default=0.20)
    p.add_argument('--lambda-l1', type=float, default=1.0)
    p.add_argument('--lambda-cos', type=float, default=1.0)
    p.add_argument('--lambda-det', type=float, default=1.0)
    p.add_argument('--grad-clip', type=float, default=5.0)
    p.add_argument('--grad-log-every', type=int, default=20,
                   help='log per-head raw gradient norms every N optimizer steps; <=0 disables')
    p.add_argument('--grad-eps', type=float, default=1e-12,
                   help='gradient norm above this value counts as non-zero')
    p.add_argument('--amp-init-scale', type=float, default=8192.0,
                   help='initial AMP GradScaler scale')
    p.add_argument('--max-nonfinite-skips', type=int, default=20,
                   help='abort only after too many AMP non-finite-gradient skips in one epoch')
    p.add_argument('--out', default='outputs/adaptive_joint/adaptive_joint.pth')
    p.add_argument('--resume-adaptive', default='')
    return p.parse_args()


def cosine_feature_loss(x, y):
    # Cosine over channel dimension, then average over BEV cells and batch.
    return (1.0 - F.cosine_similarity(x.float(), y.float(), dim=1, eps=1e-6)).mean()


def update_age(age, hard_refresh_mask, age_cap):
    # State update is intentionally detached / non-differentiable.
    with torch.no_grad():
        m = F.interpolate(hard_refresh_mask.float(), size=age.shape[-2:], mode='nearest')
        return torch.where(
            m > 0.5,
            torch.zeros_like(age),
            torch.clamp(age + 1.0, max=float(age_cap)),
        )


def _frame_idx(frame):
    return frame.get('this_sample_idx', '')


def extract_full_stage(model, backbone, frame, amp_enabled):
    """Frozen Full teacher feature extraction, including A/B taps from backbone patch."""
    if frame is None:
        return None
    backbone.clear_adaptive_training_state()
    backbone.clear_adaptive_a_profile()
    x = dict(frame)
    with torch.no_grad(), cuda_autocast(amp_enabled):
        for module in model.feature_extractor:
            x = module(x)
    required = [
        'left_shallow_feature', 'right_shallow_feature',
        'adaptive_stage_a_left', 'adaptive_stage_a_right',
        'adaptive_stage_b', 'spatial_features',
    ]
    missing = [k for k in required if k not in x]
    if missing:
        raise KeyError(
            f'Missing backbone training taps {missing}. '
            'Apply stream_dsgn2_backbone_adaptive_training.patch first.'
        )
    return {
        'shallow_l': x['left_shallow_feature'].detach(),
        'shallow_r': x['right_shallow_feature'].detach(),
        'a_l': x['adaptive_stage_a_left'].detach(),
        'a_r': x['adaptive_stage_a_right'].detach(),
        'b': x['adaptive_stage_b'].detach(),
        'bev': x['spatial_features'].detach(),
        'sample_idx': _frame_idx(frame),
    }


def init_state(full0, age_cap):
    B = full0['bev'].shape[0]
    # Stage A physical ROI is on the 80x312 grid; A stereo output is 4x larger
    # in the current full-resolution feature-neck configuration.
    a_hw = (full0['a_l'].shape[-2] // 4, full0['a_l'].shape[-1] // 4)
    b_hw = full0['b'].shape[-2:]
    f_hw = full0['bev'].shape[-2:]

    device = full0['bev'].device
    dtype = full0['bev'].dtype
    return {
        'a_l': full0['a_l'].clone(),
        'a_r': full0['a_r'].clone(),
        'b': full0['b'].clone(),
        # True BEV cache: only actual R locations will overwrite this state.
        'true_bev': full0['bev'].clone(),
        # Emitted BEV history is allowed to contain R/P/U composition and is what
        # the temporal predictor / original StreamDSGN history consumes.
        'emit1': full0['bev'].clone(),
        'emit2': full0['bev'].clone(),
        'prev_shallow_l': full0['shallow_l'].clone(),
        'prev_shallow_r': full0['shallow_r'].clone(),
        'age_a': torch.zeros((B, 1, *a_hw), device=device, dtype=dtype),
        'age_b': torch.zeros((B, 1, *b_hw), device=device, dtype=dtype),
        'age_f': torch.zeros((B, 1, *f_hw), device=device, dtype=dtype),
        'last_sample_idx': full0['sample_idx'],
        'age_cap': int(age_cap),
    }


def run_feature_extractor_adaptive(model, backbone, frame, state, a_mask, b_mask, amp_enabled):
    backbone.clear_adaptive_a_profile()
    backbone.set_adaptive_training_state(
        a_mask=a_mask,
        b_mask=b_mask,
        a_cache_left=state['a_l'],
        a_cache_right=state['a_r'],
        b_cache=state['b'],
    )
    x = dict(frame)
    try:
        with cuda_autocast(amp_enabled):
            for module in model.feature_extractor:
                x = module(x)
    finally:
        backbone.clear_adaptive_training_state()
    return x


def sample_mask(q, budgets, tau, align):
    ratio = random.choice(budgets)
    H, W = q.shape[-2:]
    roi_hw = roi_hw_from_ratio(H, W, ratio, align=align)
    st, hard, score, xy = straight_through_rect_mask(q, roi_hw, temperature=tau)
    return st, hard, ratio, roi_hw, score, xy


def run_adaptive_frame(
    model,
    backbone,
    importance_net,
    pred_net,
    frame,
    full_teacher,
    state,
    a_budgets,
    b_budgets,
    cd_budgets,
    tau,
    pred_margin,
    align,
    amp_enabled,
):
    a_hw = state['age_a'].shape[-2:]
    b_hw = state['age_b'].shape[-2:]
    cd_hw = state['age_f'].shape[-2:]

    # History BEFORE this frame is needed by the original StreamDSGN fusion.
    history_idx = state['last_sample_idx']
    history_bev = state['emit1']

    with cuda_autocast(amp_enabled):
        imp = importance_net(
            cur_left=full_teacher['shallow_l'],
            cur_right=full_teacher['shallow_r'],
            prev_left=state['prev_shallow_l'],
            prev_right=state['prev_shallow_r'],
            prev_bev=state['emit1'],
            age_a=state['age_a'],
            age_b=state['age_b'],
            age_f=state['age_f'],
            a_hw=a_hw,
            b_hw=b_hw,
            cd_hw=cd_hw,
        )

        a_st, a_hard, a_ratio, a_roi_hw, _, a_xy = sample_mask(
            imp['q_a'], a_budgets, tau, align
        )
        b_st, b_hard, b_ratio, b_roi_hw, _, b_xy = sample_mask(
            imp['q_b'], b_budgets, tau, align
        )
        cd_st, cd_hard, cd_ratio, cd_roi_hw, _, cd_xy = sample_mask(
            imp['q_cd'], cd_budgets, tau, align
        )

        # A and B use full current computation during training, but the patched
        # backbone composes current/cache at their stage interfaces with ST masks.
        cur = run_feature_extractor_adaptive(
            model, backbone, frame, state, a_st, b_st, amp_enabled
        )
        bev_current = cur['spatial_features']

        pred_bev = pred_net(state['emit1'], state['emit2'], state['age_f'])
        r_st, p_st, u_st, r_hard, _, _ = rpu_masks_from_recompute(
            cd_st, cd_hard, pred_margin
        )

        # CD/BEV training surrogate:
        #   R -> current CD result
        #   P -> temporal predictor
        #   U -> true recompute cache
        bev_mix = (
            r_st * bev_current
            + p_st * pred_bev
            + u_st * state['true_bev']
        )

        teacher_bev = full_teacher['bev'].to(bev_mix.dtype)
        loss_l1 = F.l1_loss(bev_mix, teacher_bev)
        loss_cos = cosine_feature_loss(bev_mix, teacher_bev)

    # State transitions are detached. This keeps training memory bounded while
    # still exposing the next frame to the model's own adaptive history.
    with torch.no_grad():
        state['a_l'] = cur['adaptive_stage_a_left'].detach()
        state['a_r'] = cur['adaptive_stage_a_right'].detach()
        state['b'] = cur['adaptive_stage_b'].detach()

        r_cache = F.interpolate(r_hard, size=bev_current.shape[-2:], mode='nearest')
        state['true_bev'] = (
            r_cache * bev_current.detach()
            + (1.0 - r_cache) * state['true_bev']
        )

        state['emit2'] = state['emit1']
        state['emit1'] = bev_mix.detach()
        state['prev_shallow_l'] = full_teacher['shallow_l'].detach()
        state['prev_shallow_r'] = full_teacher['shallow_r'].detach()
        state['age_a'] = update_age(state['age_a'], a_hard, state['age_cap'])
        state['age_b'] = update_age(state['age_b'], b_hard, state['age_cap'])
        state['age_f'] = update_age(state['age_f'], r_hard, state['age_cap'])
        state['last_sample_idx'] = full_teacher['sample_idx']

    # Replace the pre-fusion BEV with the differentiable R/P/U composition.
    cur['spatial_features'] = bev_mix

    stats = {
        'a_ratio': a_ratio,
        'b_ratio': b_ratio,
        'cd_ratio': cd_ratio,
        'a_roi_hw': a_roi_hw,
        'b_roi_hw': b_roi_hw,
        'cd_roi_hw': cd_roi_hw,
        'a_xy': a_xy.detach(),
        'b_xy': b_xy.detach(),
        'cd_xy': cd_xy.detach(),
    }
    return cur, loss_l1, loss_cos, state, history_idx, history_bev, stats


def detection_loss_from_adaptive_token(model, batch, token_data, history_idx, history_bev, amp_enabled):
    # Match STREAM.forward_train() history queue format exactly:
    #   (sample_idx, {'spatial_features': tensor})
    history = deque(maxlen=max(1, len(model.history_tag or [])))
    history.append((history_idx, {'spatial_features': history_bev}))
    token_data['history_features'] = history

    with cuda_autocast(amp_enabled):
        for module in model.fusion_module:
            token_data = module(token_data)
        for module in model.after_fusion_blocks:
            token_data = module(token_data)

        loss_batch = dict(batch)
        loss_batch['token'] = token_data

        tb = {}
        loss_det = token_data['spatial_features'].sum() * 0.0
        if getattr(model, 'dense_head_2d', None):
            loss_2d, tb = model.dense_head_2d.get_loss(loss_batch, tb)
            loss_det = loss_det + loss_2d
        if getattr(model, 'dense_head', None):
            loss_3d, tb = model.dense_head.get_loss(loss_batch, tb)
            loss_det = loss_det + loss_3d

    return loss_det, tb, token_data



def module_grad_stats(module):
    """Return raw (unscaled, pre-clipping) gradient statistics for a module."""
    total_sq = 0.0
    max_abs = 0.0
    num_params = 0
    num_with_grad = 0
    num_none = 0
    num_nonfinite = 0

    for p in module.parameters():
        if not p.requires_grad:
            continue
        num_params += 1
        if p.grad is None:
            num_none += 1
            continue

        num_with_grad += 1
        g = p.grad.detach().float()
        finite = torch.isfinite(g)
        num_nonfinite += int((~finite).sum().item())
        if finite.any():
            gf = g[finite]
            total_sq += float((gf * gf).sum().item())
            max_abs = max(max_abs, float(gf.abs().max().item()))

    return {
        'norm': total_sq ** 0.5,
        'max_abs': max_abs,
        'num_params': num_params,
        'num_with_grad': num_with_grad,
        'num_none': num_none,
        'num_nonfinite': num_nonfinite,
    }


def find_nonfinite_grad_params(importance_net, pred_net, max_names=16):
    """Find exact trainable parameters whose gradients contain NaN/Inf."""
    bad = []

    for prefix, module in (
        ('importance', importance_net),
        ('predictor', pred_net),
    ):
        for name, p in module.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue

            g = p.grad.detach()
            bad_count = int((~torch.isfinite(g)).sum().item())

            if bad_count > 0:
                bad.append((f'{prefix}.{name}', bad_count))

                if len(bad) >= max_names:
                    return bad

    return bad


def update_grad_summary(summary, name, stats, eps):
    rec = summary[name]
    rec['checks'] += 1
    rec['finite'] += int(stats['num_nonfinite'] == 0)
    rec['none_free'] += int(stats['num_none'] == 0)
    rec['positive'] += int(stats['norm'] > eps)
    rec['sum_norm'] += stats['norm']
    rec['min_norm'] = min(rec['min_norm'], stats['norm'])
    rec['max_norm'] = max(rec['max_norm'], stats['norm'])


def print_grad_summary(summary, eps):
    print('\n===== gradient summary (raw / unscaled / pre-clip) =====')
    for name, rec in summary.items():
        checks = rec['checks']
        if checks == 0:
            print(f'[GRAD-SUMMARY] {name}: no checks')
            continue
        mean_norm = rec['sum_norm'] / checks
        strict_ok = (
            rec['positive'] == checks
            and rec['finite'] == checks
            and rec['none_free'] == checks
        )
        if strict_ok:
            verdict = 'PASS'
        elif rec['positive'] > 0 and rec['finite'] == checks:
            verdict = 'WARN'
        else:
            verdict = 'FAIL'
        print(
            f'[GRAD-SUMMARY] {name:<9s} verdict={verdict} '
            f'nonzero={rec["positive"]}/{checks} finite={rec["finite"]}/{checks} '
            f'none_free={rec["none_free"]}/{checks} '
            f'norm[min/mean/max]={rec["min_norm"]:.3e}/'
            f'{mean_norm:.3e}/{rec["max_norm"]:.3e} eps={eps:.1e}'
        )


def anneal_tau(epoch, epochs, start, end):
    if epochs <= 1:
        return float(end)
    r = float(epoch) / float(epochs - 1)
    return float(start) * (1.0 - r) + float(end) * r


def main():
    args = parse_args()
    args.a_budgets = parse_float_list(args.a_budgets)
    args.b_budgets = parse_float_list(args.b_budgets)
    args.cd_budgets = parse_float_list(args.cd_budgets)

    cfg, dataset, loader, model, logger = build_env(args, training=True)
    # Keep the pretrained detector in eval mode: frozen BN/statistics, but DO NOT
    # use no_grad on the adaptive/detection path because input gradients must flow
    # back to the two adaptive networks.
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    backbone = find_module(model, 'StreamDSGN2Backbone')
    if not hasattr(backbone, 'set_adaptive_training_state'):
        raise RuntimeError(
            'Backbone training interface not found. Apply '
            'stream_dsgn2_backbone_adaptive_training.patch first.'
        )

    shallow_channels = int(cfg.MODEL.BACKBONE_3D.feature_backbone.get('base_channels', 64))
    bev_channels = int(cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)

    importance_net = StageImportancePredictor(
        shallow_channels=shallow_channels,
        bev_channels=bev_channels,
        hidden=args.hidden,
        age_cap=args.age_cap,
    ).cuda()
    pred_net = BEVTemporalPredictor(
        channels=bev_channels,
        hidden=args.pred_hidden,
        age_cap=args.age_cap,
    ).cuda()

    params = list(importance_net.parameters()) + list(pred_net.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)

    start_epoch = 0
    if args.resume_adaptive:
        ck = torch.load(args.resume_adaptive, map_location='cpu')
        importance_net.load_state_dict(ck['importance_net'], strict=True)
        pred_net.load_state_dict(ck['pred_net'], strict=True)
        if 'optimizer' in ck:
            optimizer.load_state_dict(ck['optimizer'])
        start_epoch = int(ck.get('epoch', -1)) + 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    amp_enabled = bool(model.use_amp_dict.get('TRAIN', False))

    try:
        scaler = torch.amp.GradScaler(
            'cuda',
            enabled=amp_enabled,
            init_scale=float(args.amp_init_scale),
        )
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(
            enabled=amp_enabled,
            init_scale=float(args.amp_init_scale),
        )

    for epoch in range(start_epoch, args.epochs):
        importance_net.train()
        pred_net.train()
        tau = anneal_tau(epoch, args.epochs, args.tau_start, args.tau_end)

        sum_loss = 0.0
        sum_det = 0.0
        sum_l1 = 0.0
        sum_cos = 0.0
        used = 0
        nonfinite_skips = 0

        grad_summary = {
            name: {
                'checks': 0, 'positive': 0, 'finite': 0, 'none_free': 0,
                'sum_norm': 0.0, 'min_norm': float('inf'), 'max_norm': 0.0,
            }
            for name in ('A', 'B', 'CD', 'Predictor')
        }

        for bi, batch in enumerate(loader):
            if args.max_batches > 0 and bi >= args.max_batches:
                break
            load_data_to_gpu(batch)

            token = batch.get('token')
            prev = batch.get('prev')
            prev2 = batch.get('prev2')
            if token is None or prev is None:
                continue
            if prev2 is None:
                # Degenerate warm start. The first adaptive step sees zero motion,
                # while the token step still receives a genuine adaptive prev.
                prev2 = prev

            # Frozen Full teachers. These calls also provide current A_s, A-stage,
            # B-stage and final BEV states used to initialize/compare the rollout.
            full_prev2 = extract_full_stage(model, backbone, prev2, amp_enabled)
            full_prev = extract_full_stage(model, backbone, prev, amp_enabled)
            full_token = extract_full_stage(model, backbone, token, amp_enabled)
            state = init_state(full_prev2, args.age_cap)

            optimizer.zero_grad(set_to_none=True)

            # Local two-step rollout: prev warms the model with its own adaptive
            # state, token supplies the original StreamDSGN detection objective.
            prev_cur, prev_l1, prev_cos, state, _, _, _ = run_adaptive_frame(
                model, backbone, importance_net, pred_net,
                prev, full_prev, state,
                args.a_budgets, args.b_budgets, args.cd_budgets,
                tau, args.pred_margin, args.roi_align, amp_enabled,
            )

            token_cur, tok_l1, tok_cos, state, hist_idx, hist_bev, stats = run_adaptive_frame(
                model, backbone, importance_net, pred_net,
                token, full_token, state,
                args.a_budgets, args.b_budgets, args.cd_budgets,
                tau, args.pred_margin, args.roi_align, amp_enabled,
            )

            loss_det, tb, _ = detection_loss_from_adaptive_token(
                model, batch, token_cur, hist_idx, hist_bev, amp_enabled
            )

            loss_l1 = 0.5 * (prev_l1 + tok_l1)
            loss_cos = 0.5 * (prev_cos + tok_cos)
            loss = (
                args.lambda_det * loss_det
                + args.lambda_l1 * loss_l1
                + args.lambda_cos * loss_cos
            )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f'NaN/Inf loss: total={loss.detach()} det={loss_det.detach()} '
                    f'l1={loss_l1.detach()} cos={loss_cos.detach()}'
                )

            scaler.scale(loss).backward()

            # AMP unscale first. GradScaler records whether NaN/Inf occurred.
            scaler.unscale_(optimizer)

            # ------------------------------------------------------------
            # AMP overflow handling
            #
            # Do not send NaN/Inf gradients into clip_grad_norm_.
            # GradScaler has already recorded found_inf during unscale_().
            # scaler.step() will therefore SKIP the AdamW update, while
            # scaler.update() lowers the loss scale automatically.
            # ------------------------------------------------------------
            bad_grad_params = find_nonfinite_grad_params(
                importance_net,
                pred_net,
                max_names=16,
            )

            if bad_grad_params:
                nonfinite_skips += 1
                scale_before = float(scaler.get_scale())

                bad_text = ', '.join(
                    f'{name}[bad={count}]'
                    for name, count in bad_grad_params
                )

                print(
                    f'[AMP-SKIP] '
                    f'epoch={epoch:02d} '
                    f'data_iter={bi:05d} '
                    f'opt_steps={used:05d} '
                    f'scale_before={scale_before:.1f} '
                    f'bad_params={bad_text}'
                )

                # found_inf was recorded in scaler.unscale_().
                # Thus this call does NOT update model parameters.
                scaler.step(optimizer)
                scaler.update()

                scale_after = float(scaler.get_scale())

                print(
                    f'[AMP-SKIP] '
                    f'scale_after={scale_after:.1f} '
                    f'skips_this_epoch={nonfinite_skips}'
                )

                optimizer.zero_grad(set_to_none=True)

                if nonfinite_skips > args.max_nonfinite_skips:
                    raise FloatingPointError(
                        f'too many non-finite AMP gradient skips '
                        f'in epoch {epoch}: '
                        f'{nonfinite_skips} > {args.max_nonfinite_skips}'
                    )

                continue

            # ------------------------------------------------------------
            # Normal finite-gradient path
            # ------------------------------------------------------------
            iter_id = used + 1

            do_grad_log = (
                args.grad_log_every > 0
                and (
                    iter_id <= 3
                    or iter_id % args.grad_log_every == 0
                )
            )

            if do_grad_log:
                grad_groups = {
                    'A': module_grad_stats(importance_net.head_a),
                    'B': module_grad_stats(importance_net.head_b),
                    'CD': module_grad_stats(importance_net.head_cd),
                    'Predictor': module_grad_stats(pred_net),
                }

                for name, gs in grad_groups.items():
                    update_grad_summary(
                        grad_summary,
                        name,
                        gs,
                        args.grad_eps,
                    )

                print(
                    '[GRAD] '
                    + ' '.join(
                        f'{name}={gs["norm"]:.3e}'
                        f'(max={gs["max_abs"]:.2e},'
                        f'none={gs["num_none"]},'
                        f'bad={gs["num_nonfinite"]})'
                        for name, gs in grad_groups.items()
                    )
                )

            # Only finite gradients reach clipping.
            global_preclip = torch.nn.utils.clip_grad_norm_(
                params,
                args.grad_clip,
            )

            if not torch.isfinite(global_preclip):
                raise FloatingPointError(
                    'Unexpected non-finite global gradient norm '
                    'after explicit finite-gradient check: '
                    f'{global_preclip}'
                )

            scaler.step(optimizer)
            scaler.update()

            cur_loss = float(loss.detach().item())
            cur_det = float(loss_det.detach().item())
            cur_l1 = float(loss_l1.detach().item())
            cur_cos = float(loss_cos.detach().item())

            used += 1
            sum_loss += cur_loss
            sum_det += cur_det
            sum_l1 += cur_l1
            sum_cos += cur_cos

            if used <= 3 or used % 20 == 0:
                print(
                    f'epoch={epoch:02d} iter={used:05d} tau={tau:.3f} '
                    f'loss_cur={cur_loss:.5f} loss_avg={sum_loss/used:.5f} '
                    f'det_cur={cur_det:.5f} det_avg={sum_det/used:.5f} '
                    f'l1_cur={cur_l1:.5f} l1_avg={sum_l1/used:.5f} '
                    f'cos_cur={cur_cos:.5f} cos_avg={sum_cos/used:.5f} '
                    f'grad_global_preclip={float(global_preclip):.3e} '
                    f'budget=({stats["a_ratio"]:.2f},'
                    f'{stats["b_ratio"]:.2f},{stats["cd_ratio"]:.2f})'
                )

        if args.grad_log_every > 0:
            print_grad_summary(grad_summary, args.grad_eps)

        ckpt = {
            'importance_net': importance_net.state_dict(),
            'pred_net': pred_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'meta': {
                'base_cfg': args.cfg,
                'base_ckpt': args.ckpt,
                'age_cap': args.age_cap,
                'pred_margin': args.pred_margin,
                'a_budgets': args.a_budgets,
                'b_budgets': args.b_budgets,
                'cd_budgets': args.cd_budgets,
                'tau': tau,
                'loss': 'L_det + lambda_l1*BEV_L1 + lambda_cos*BEV_cos',
                'explicit_importance_supervision': False,
            },
        }
        torch.save(ckpt, out)
        print(
            f'[epoch {epoch}] '
            f'used={used} '
            f'nonfinite_skips={nonfinite_skips} '
            f'mean_loss={sum_loss/max(used,1):.6f} '
            f'saved={out}'
        )


if __name__ == '__main__':
    main()
