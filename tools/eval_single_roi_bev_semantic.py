#!/usr/bin/env python3
"""
Semantic / accuracy validation for the agreed design:
  - one BEV ROI center per frame
  - Recompute only inside that ROI (full current BEV teacher is used here)
  - Predict / Reuse only at BEV spatial_features
  - original FeatureAlignment / VAN / StreamDetHead are preserved

IMPORTANT:
This script computes the full current BEV feature before composition, so its
wall-clock time is NOT a valid selective-execution latency measurement.
"""
import argparse
import pickle
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

from pcdet.models import load_data_to_gpu
from pcdet.models.adaptive_single_roi import (
    BEVCenterPredictor,
    BEVTemporalPredictor,
    linear_extrapolate,
    oracle_center_from_error,
    make_rpu_masks,
)
from eval_utils.eval_utils import format_paper_metrics

from single_roi_common import (
    add_base_args,
    build_env,
    cuda_autocast,
    extract_full_bev,
    save_json,
    scalar_string,
)


def parse_args():
    p = argparse.ArgumentParser('Semantic single-ROI BEV evaluation')
    add_base_args(p, training=False)
    p.add_argument('--adaptive-ckpt', default='outputs/single_roi/single_roi_modules.pth')
    p.add_argument('--center-strategy', choices=['learned', 'oracle', 'fixed', 'random', 'max_age', 'full'], default='oracle')
    p.add_argument('--outside-mode', choices=['reuse', 'predict', 'ring'], default='ring')
    p.add_argument('--predictor', choices=['learned', 'linear'], default='linear')
    p.add_argument('--roi-h', type=int, default=96)
    p.add_argument('--roi-w', type=int, default=96)
    p.add_argument('--pred-margin', type=int, default=48)
    p.add_argument('--age-cap', type=int, default=8)
    p.add_argument(
        '--collect-staleness',
        action='store_true',
        help=(
            'Collect Observation-2 stale-age statistics on reused BEV cells. '
            'Diagnostic only; requires --outside-mode reuse.'
        ),
    )
    p.add_argument('--print-every', type=int, default=20)
    p.add_argument('--output', default='outputs/single_roi/semantic_eval')
    return p.parse_args()


def choose_center(args, center_net, token, full_bev, true_cache, pred_bev, age):
    B, _, H, W = full_bev.shape
    assert B == 1, 'streaming semantic evaluator currently expects batch_size=1'

    if args.center_strategy == 'fixed':
        return torch.tensor([[W // 2, H // 2]], device=full_bev.device)
    if args.center_strategy == 'random':
        return torch.tensor([[random.randrange(W), random.randrange(H)]], device=full_bev.device)
    if args.center_strategy in ('age', 'max_age'):
        centers, _ = oracle_center_from_error(
            age.float(),
            (args.roi_h, args.roi_w)
        )
        return centers
    if args.center_strategy == 'learned':
        centers, _ = center_net.predict_center(
            token['left_img'], token['right_img'], true_cache, age
        )
        return centers
    if args.center_strategy == 'oracle':
        reuse_err = (full_bev - true_cache).abs().mean(1, keepdim=True)
        if pred_bev is not None:
            pred_err = (full_bev - pred_bev).abs().mean(1, keepdim=True)
            # Oracle refreshes where even the better temporal compensation is bad.
            base_err = torch.minimum(reuse_err, pred_err)
        else:
            base_err = reuse_err
        centers, _ = oracle_center_from_error(base_err.float(), (args.roi_h, args.roi_w))
        return centers
    raise ValueError(args.center_strategy)


def validate_masks(r, p, u, roi_h, roi_w, H, W, first):
    assert r.shape == p.shape == u.shape == (1, 1, H, W)
    total = r + p + u
    max_partition_err = float((total.float() - 1.0).abs().max().item())
    assert max_partition_err <= 1e-6, f'R/P/U masks do not partition the BEV: {max_partition_err}'
    if not first:
        expected = min(max(int(roi_h), 1), H) * min(max(int(roi_w), 1), W) / float(H * W)
        actual = float(r.float().mean().item())
        assert abs(actual - expected) <= 1e-6, (actual, expected)


def main():
    args = parse_args()
    cfg, dataset, loader, model, logger = build_env(args, training=False)
    assert args.batch_size == 1, 'use --batch-size 1 for streaming state'
    assert args.roi_h > 0 and args.roi_w > 0
    assert args.age_cap > 0

    if args.collect_staleness and args.outside_mode != 'reuse':
        raise ValueError(
            '--collect-staleness requires --outside-mode reuse'
        )

    bev_channels = int(cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES)
    center_net = BEVCenterPredictor(bev_channels=bev_channels, age_cap=args.age_cap).cuda().eval()
    pred_net = BEVTemporalPredictor(channels=bev_channels, age_cap=args.age_cap).cuda().eval()

    ckpt_path = Path(args.adaptive_ckpt)
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location='cpu')
        if 'center_net' in ck:
            center_net.load_state_dict(ck['center_net'], strict=True)
        if 'pred_net' in ck:
            pred_net.load_state_dict(ck['pred_net'], strict=True)
    if args.center_strategy == 'learned' and not ckpt_path.exists():
        raise FileNotFoundError(f'learned center requested but checkpoint is missing: {ckpt_path}')
    if args.predictor == 'learned' and args.outside_mode != 'reuse' and not ckpt_path.exists():
        raise FileNotFoundError(f'learned predictor requested but checkpoint is missing: {ckpt_path}')

    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    det_annos = []
    diagnostics = []

    history_queue = deque(maxlen=len(model.history_tag) if model.history_tag is not None else 1)
    adaptive_history = deque(maxlen=2)
    true_cache = None
    age = None
    last_scene = None
    amp_enabled = bool(model.use_amp_dict.get('TEST', False))

    # Observation 2 statistics.
    #
    # Index 1..age_cap is used.
    # The final age_cap bucket means >= age_cap because the runtime
    # age map itself is clamped at age_cap.
    #
    # Accumulate on GPU and synchronize only once at the end.
    staleness_count = None
    staleness_error_sum = None
    staleness_error_sq_sum = None

    for bi, batch in enumerate(loader):
        if args.max_batches > 0 and bi >= args.max_batches:
            break

        load_data_to_gpu(batch)
        token = batch['token']
        scene = scalar_string(token.get('scene', None))
        prev_sample_idx = scalar_string(token.get('prev_sample_idx', None))
        first = (last_scene is None or scene != last_scene or prev_sample_idx == '')

        if first:
            history_queue.clear()
            adaptive_history.clear()
            true_cache = None
            age = None
        last_scene = scene

        cur_data, full_bev = extract_full_bev(model, token)
        full_bev = full_bev.detach()
        B, C, H, W = full_bev.shape
        assert B == 1 and C == bev_channels, (full_bev.shape, bev_channels)
        assert torch.isfinite(full_bev).all(), 'full BEV contains NaN/Inf'

        if first or args.center_strategy == 'full':
            adaptive = full_bev.clone()
            true_cache = full_bev.clone()
            age = torch.zeros((B, 1, H, W), device=full_bev.device, dtype=full_bev.dtype)
            center = torch.tensor([[W // 2, H // 2]], device=full_bev.device)
            r = torch.ones((B, 1, H, W), device=full_bev.device, dtype=full_bev.dtype)
            p = torch.zeros_like(r)
            u = torch.zeros_like(r)
        else:
            assert true_cache is not None and age is not None and len(adaptive_history) >= 1
            prev1 = adaptive_history[-1]
            prev2 = adaptive_history[-2] if len(adaptive_history) >= 2 else prev1

            if args.outside_mode == 'reuse':
                pred_bev = None
            elif args.predictor == 'learned':
                with torch.no_grad(), cuda_autocast(amp_enabled):
                    pred_bev = pred_net(prev1, prev2, age)
            else:
                pred_bev = linear_extrapolate(prev1, prev2, alpha=1.0)

            center = choose_center(args, center_net, token, full_bev, true_cache, pred_bev, age)
            assert center.shape == (1, 2)
            assert 0 <= int(center[0, 0]) < W and 0 <= int(center[0, 1]) < H

            r, p, u, _ = make_rpu_masks(
                center[0], (args.roi_h, args.roi_w),
                args.pred_margin if args.outside_mode == 'ring' else 0,
                H, W, full_bev.device, full_bev.dtype,
            )
            if args.outside_mode == 'predict':
                p = 1 - r
                u = torch.zeros_like(r)
            elif args.outside_mode == 'reuse':
                p = torch.zeros_like(r)
                u = 1 - r

            validate_masks(r, p, u, args.roi_h, args.roi_w, H, W, first=False)

            old_cache = true_cache
            if pred_bev is None:
                pred_bev = old_cache

            adaptive = r * full_bev + p * pred_bev + u * old_cache

            # ----------------------------------------------------
            # Observation 2:
            # measure the error of cells that are actually REUSED
            # on the current frame.
            #
            # Before processing frame t:
            #     age == A_{t-1}
            #
            # If a cell is reused at frame t, its effective age is:
            #     A_t = A_{t-1} + 1
            #
            # The cache error is measured against the full current
            # BEV teacher:
            #
            #     e_t(i) = mean_c |F_t^full(i,c) - F_t^cache(i,c)|
            #
            # Only U/reused cells are included. Refreshed R cells
            # are excluded.
            # ----------------------------------------------------
            if args.collect_staleness:
                reuse_error_map = (
                    full_bev.float() - old_cache.float()
                ).abs().mean(
                    dim=1,
                    keepdim=True,
                )

                effective_age = (
                    age.float() + 1.0
                ).clamp(
                    max=args.age_cap
                ).long()

                reuse_mask = (u > 0.5)

                age_index = effective_age[
                    reuse_mask
                ].reshape(-1)

                error_values = reuse_error_map[
                    reuse_mask
                ].reshape(-1)

                if age_index.numel() > 0:
                    if staleness_count is None:
                        stats_size = args.age_cap + 1

                        staleness_count = torch.zeros(
                            stats_size,
                            dtype=torch.float64,
                            device=full_bev.device,
                        )

                        staleness_error_sum = torch.zeros_like(
                            staleness_count
                        )

                        staleness_error_sq_sum = torch.zeros_like(
                            staleness_count
                        )

                    staleness_count += torch.bincount(
                        age_index,
                        minlength=args.age_cap + 1,
                    ).to(torch.float64)

                    staleness_error_sum += torch.bincount(
                        age_index,
                        weights=error_values,
                        minlength=args.age_cap + 1,
                    ).to(torch.float64)

                    staleness_error_sq_sum += torch.bincount(
                        age_index,
                        weights=error_values.square(),
                        minlength=args.age_cap + 1,
                    ).to(torch.float64)

            true_cache = r * full_bev + (1 - r) * old_cache

            age = torch.where(
                r > 0.5,
                torch.zeros_like(age),
                (age + 1).clamp(max=args.age_cap),
            )

        if first or args.center_strategy == 'full':
            validate_masks(r, p, u, args.roi_h, args.roi_w, H, W, first=True)

        assert torch.isfinite(adaptive).all(), 'adaptive BEV contains NaN/Inf'
        assert torch.isfinite(true_cache).all(), 'true cache contains NaN/Inf'

        # Match STREAM.forward_test(): history is the pre-fusion spatial_features.
        cur_data['spatial_features'] = adaptive
        cur_data['history_features'] = history_queue
        current_for_history = adaptive.detach().clone()

        # The repository's STREAM.forward() wraps the whole inference path in AMP.
        # Since this script manually splits the forward, restore the same boundary.
        with torch.no_grad(), cuda_autocast(amp_enabled):
            for m in model.fusion_module:
                cur_data = m(cur_data)
            for m in model.after_fusion_blocks:
                cur_data = m(cur_data)
            pred_dicts, _ = model.post_processing(cur_data)

        if model.history_tag is not None:
            history_queue.append((cur_data['this_sample_idx'], {'spatial_features': current_for_history}))
        adaptive_history.append(current_for_history)

        batch['token'] = cur_data
        annos = dataset.generate_prediction_dicts(batch, pred_dicts, dataset.class_names)
        assert len(annos) == 1, 'batch_size=1 evaluator expects exactly one annotation per batch'
        det_annos += annos

        diag = {
            'index': bi,
            'sample_idx': scalar_string(cur_data.get('this_sample_idx', '')),
            'scene': scene,
            'first_frame': bool(first),
            'center_x': int(center[0, 0].item()),
            'center_y': int(center[0, 1].item()),
            'r_ratio': float(r.float().mean().item()),
            'p_ratio': float(p.float().mean().item()),
            'u_ratio': float(u.float().mean().item()),
            'feature_l1': float((adaptive.float() - full_bev.float()).abs().mean().item()),
            'age_mean': float(age.float().mean().item()),
            'age_max': float(age.float().max().item()),
        }
        diagnostics.append(diag)

        if bi < 3 or (args.print_every > 0 and bi % args.print_every == 0):
            print(
                f"[{bi}] scene={scene} first={int(first)} "
                f"center=({diag['center_x']},{diag['center_y']}) "
                f"R/P/U={diag['r_ratio']:.3f}/{diag['p_ratio']:.3f}/{diag['u_ratio']:.3f} "
                f"featL1={diag['feature_l1']:.5f} age={diag['age_mean']:.2f}/{diag['age_max']:.0f}"
            )

    # Always save predictions/diagnostics, even for a truncated smoke run.
    with open(outdir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    processed = len(diagnostics)
    complete_predictions = (
        processed == len(dataset)
        and len(det_annos) == len(dataset)
    )

    # IMPORTANT:
    # This semantic/oracle evaluator computes the FULL current-frame BEV
    # feature first, then performs R/P/U composition. Therefore the wall
    # clock time of this script is NOT a valid selective-execution latency.
    #
    # The original StreamDSGN config contains:
    #   offline_3d / stream_copy / stream_kf
    #
    # stream_copy and stream_kf require dataset.empirical loaded from a
    # real INFER_TIME_PATH. They are also scientifically invalid for this
    # semantic/oracle experiment because we still compute the full teacher
    # BEV every frame.
    #
    # Therefore this experiment intentionally evaluates OFFLINE 3D only.
    configured_metrics = list(
        cfg.MODEL.POST_PROCESSING.EVAL_METRIC
    )
    semantic_eval_metrics = ['offline_3d']

    if any('stream' in str(m) for m in configured_metrics):
        print(
            '[INFO] semantic/oracle evaluation ignores configured '
            f'streaming metrics {configured_metrics}.'
        )
        print(
            '[INFO] using offline_3d only; streaming sAP will be '
            'evaluated after real physical ROI execution and real '
            'model-forward latency are integrated.'
        )

    if complete_predictions:
        result_str, result_dict = dataset.evaluation(
            det_annos,
            dataset.class_names,
            eval_metric=semantic_eval_metrics,
            output_path=outdir,
        )

        if result_str is not None:
            for name, metric_text in result_str.items():
                print(f'===== {name} =====')
                print(
                    format_paper_metrics(metric_text)
                    if getattr(cfg, 'PAPER_METRICS_ONLY', False)
                    else metric_text
                )
    else:
        result_dict = {}
        print(
            f'[SMOKE] processed {processed}/{len(dataset)} samples; '
            'skipping official KITTI AP because official evaluation '
            'requires one prediction annotation per dataset frame.'
        )

    # ------------------------------------------------------------------
    # Observation 1 export:
    # Spatial redundancy and task sensitivity under partial BEV recomputation.
    #
    # Important:
    #   - scene-first frames are intentionally Full;
    #   - therefore partial-recompute statistics are computed on non-first frames;
    #   - this evaluator computes the full current BEV teacher first, so the
    #     following statistics are diagnostic only and MUST NOT be used for
    #     latency claims.
    # ------------------------------------------------------------------
    nonfirst_diagnostics = [
        x for x in diagnostics
        if not x['first_frame']
    ]
    obs_frames = (
        nonfirst_diagnostics
        if nonfirst_diagnostics
        else diagnostics
    )

    def _flatten_numeric_metrics(obj, prefix=''):
        out = {}

        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f'{prefix}/{k}' if prefix else str(k)
                out.update(_flatten_numeric_metrics(v, key))

        elif (
            isinstance(obj, (int, float, np.number))
            and not isinstance(obj, (bool, np.bool_))
        ):
            out[prefix] = float(obj)

        return out

    flat_metrics = _flatten_numeric_metrics(result_dict)

    def _lookup_metric(suffix):
        # Direct OpenPCDet-style key.
        if suffix in flat_metrics:
            return flat_metrics[suffix]

        # Also support an outer metric namespace such as:
        # offline_3d/Car_3d/moderate_R40
        matches = [
            v for k, v in flat_metrics.items()
            if k.endswith(suffix)
        ]

        if len(matches) == 1:
            return matches[0]

        return None

    obs_l1 = [
        float(x['feature_l1'])
        for x in obs_frames
    ]
    obs_r = [
        float(x['r_ratio'])
        for x in obs_frames
    ]

    observation1 = {
        'diagnostic_only': True,
        'latency_valid': False,

        # "oracle" here is only a diagnostic feature-discrepancy selector.
        # It is NOT the deployable center-selection method.
        'selector': (
            'feature_discrepancy'
            if args.center_strategy == 'oracle'
            else args.center_strategy
        ),

        'outside_policy': args.outside_mode,
        'roi_hw': [
            int(args.roi_h),
            int(args.roi_w),
        ],

        'num_first_frames': int(
            sum(bool(x['first_frame']) for x in diagnostics)
        ),
        'num_nonfirst_frames': int(
            len(nonfirst_diagnostics)
        ),

        # Excluding scene-first Full frames.
        'recompute_ratio_nonfirst': (
            float(np.mean(obs_r))
            if obs_r else None
        ),

        'feature_l1_mean_nonfirst': (
            float(np.mean(obs_l1))
            if obs_l1 else None
        ),

        'feature_l1_p50_nonfirst': (
            float(np.percentile(obs_l1, 50))
            if obs_l1 else None
        ),

        'feature_l1_p95_nonfirst': (
            float(np.percentile(obs_l1, 95))
            if obs_l1 else None
        ),

        # Official KITTI strict class-specific thresholds:
        # Car: 0.70; Pedestrian/Cyclist: 0.50.
        'ap_r40_3d_moderate': {
            cls_name: _lookup_metric(
                f'{cls_name}_3d/moderate_R40'
            )
            for cls_name in (
                'Car',
                'Pedestrian',
                'Cyclist',
            )
        },

        'metric_note': (
            'Official KITTI class-specific strict IoU thresholds are used: '
            'Car=0.70, Pedestrian=0.50, Cyclist=0.50.'
        ),
    }

    # ================================================================
    # Observation 2 export
    # ================================================================
    observation2 = None

    if args.collect_staleness:
        if staleness_count is None:
            raise RuntimeError(
                'No staleness statistics were collected.'
            )

        counts = (
            staleness_count
            .detach()
            .cpu()
            .numpy()
        )

        error_sums = (
            staleness_error_sum
            .detach()
            .cpu()
            .numpy()
        )

        error_sq_sums = (
            staleness_error_sq_sum
            .detach()
            .cpu()
            .numpy()
        )

        total_count = float(
            counts[1:].sum()
        )

        age_rows = []

        for age_value in range(
            1,
            args.age_cap + 1,
        ):
            count = float(
                counts[age_value]
            )

            error_sum = float(
                error_sums[age_value]
            )

            error_sq_sum = float(
                error_sq_sums[age_value]
            )

            if count > 0:
                mean_error = (
                    error_sum / count
                )

                variance = max(
                    error_sq_sum / count
                    - mean_error * mean_error,
                    0.0,
                )

                std_error = (
                    variance ** 0.5
                )
            else:
                mean_error = None
                std_error = None

            age_label = (
                f'{args.age_cap}+'
                if age_value == args.age_cap
                else str(age_value)
            )

            age_rows.append({
                'age': int(age_value),
                'age_label': age_label,
                'is_capped_bucket': bool(
                    age_value == args.age_cap
                ),
                'count': int(count),
                'fraction': (
                    count / total_count
                    if total_count > 0
                    else None
                ),
                'mean_feature_l1': mean_error,
                'std_feature_l1': std_error,

                # Kept so the plotting script can form
                # statistically correct grouped buckets.
                'feature_l1_sum': error_sum,
                'feature_l1_sq_sum': error_sq_sum,
            })

        nonfirst_frames = [
            x for x in diagnostics
            if not x['first_frame']
        ]

        observation2 = {
            'name': (
                'feature_staleness_under_temporal_reuse'
            ),

            'diagnostic_only': True,
            'latency_valid': False,

            'selector': (
                'feature_discrepancy'
                if args.center_strategy == 'oracle'
                else args.center_strategy
            ),

            'outside_policy': args.outside_mode,

            'roi_hw': [
                int(args.roi_h),
                int(args.roi_w),
            ],

            'age_cap': int(args.age_cap),

            'age_cap_semantics': (
                f'The final bucket {args.age_cap}+ contains '
                f'all ages >= {args.age_cap}, because the '
                'runtime age map is clamped.'
            ),

            'error_definition': (
                'Per-cell channel-mean L1 between the full '
                'current BEV feature and the cached BEV '
                'feature, evaluated only on cells actually '
                'reused on the current frame.'
            ),

            'num_nonfirst_frames': int(
                len(nonfirst_frames)
            ),

            'mean_recompute_ratio_nonfirst': (
                float(np.mean([
                    x['r_ratio']
                    for x in nonfirst_frames
                ]))
                if nonfirst_frames
                else None
            ),

            'total_reused_cell_observations': int(
                total_count
            ),

            'age_statistics': age_rows,
        }

        save_json(
            outdir / 'observation2_staleness.json',
            observation2,
        )

        print(
            'Saved: '
            f'{outdir / "observation2_staleness.json"}'
        )

        print()
        print(
            '[Observation 2] '
            'stale-age vs cached-feature error'
        )

        print(
            f"{'Age':>6} "
            f"{'Cells%':>10} "
            f"{'MeanL1':>12} "
            f"{'StdL1':>12}"
        )

        print('-' * 46)

        for row in age_rows:
            fraction_pct = (
                100.0 * row['fraction']
                if row['fraction'] is not None
                else float('nan')
            )

            mean_l1 = (
                row['mean_feature_l1']
                if row['mean_feature_l1'] is not None
                else float('nan')
            )

            std_l1 = (
                row['std_feature_l1']
                if row['std_feature_l1'] is not None
                else float('nan')
            )

            print(
                f"{row['age_label']:>6} "
                f"{fraction_pct:10.4f} "
                f"{mean_l1:12.6f} "
                f"{std_l1:12.6f}"
            )

        print()

    summary = {
        'center_strategy': args.center_strategy,
        'outside_mode': args.outside_mode,
        'predictor': args.predictor,
        'roi_hw': [args.roi_h, args.roi_w],
        'pred_margin': args.pred_margin,
        'num_samples': processed,
        'dataset_size': len(dataset),
        'official_evaluation_performed': bool(complete_predictions),
        'configured_eval_metrics': configured_metrics,
        'semantic_eval_metrics': semantic_eval_metrics,
        'mean_feature_l1': float(np.mean([x['feature_l1'] for x in diagnostics])) if diagnostics else None,
        'mean_r_ratio': float(np.mean([x['r_ratio'] for x in diagnostics])) if diagnostics else None,
        'mean_p_ratio': float(np.mean([x['p_ratio'] for x in diagnostics])) if diagnostics else None,
        'mean_u_ratio': float(np.mean([x['u_ratio'] for x in diagnostics])) if diagnostics else None,
        'dataset_metrics': result_dict,
        'observation2': observation2,
        'observation1': observation1,
        'note': 'Semantic/oracle validation only: full current BEV teacher is computed before composition. Do not use this script for latency claims.',
        'frames': diagnostics,
    }
    save_json(outdir / 'summary.json', summary)
    print(f'Saved: {outdir / "summary.json"}')


if __name__ == '__main__':
    main()
