#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--cfg_file',
        default='configs/stream/kitti_models/'
                'stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-'
                'lka_7-mcl_5090_eval.yaml',
    )

    p.add_argument('--ckpt', required=True)

    p.add_argument(
        '--ratios',
        nargs='+',
        type=float,
        default=[
            0.0,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],
    )

    p.add_argument(
        '--positions',
        nargs='+',
        default=['center', 'random', 'boundary'],
        choices=['center', 'random', 'boundary'],
    )

    p.add_argument('--sample-index', type=int, default=10)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--repeat', type=int, default=300)

    p.add_argument(
        '--output',
        default='outputs/adaptive_profile/a_stage_roi_dense.json',
    )

    return p.parse_args()


def get_backbone(model):
    for m in model.modules():
        if type(m).__name__ == 'StreamDSGN2Backbone':
            return m

    raise RuntimeError('StreamDSGN2Backbone not found')


def percentile(xs, q):
    return float(np.percentile(np.asarray(xs, dtype=np.float64), q))


def summarize(xs):
    xs = np.asarray(xs, dtype=np.float64)

    return {
        'n': int(xs.size),
        'mean_ms': float(xs.mean()),
        'std_ms': float(xs.std()),
        'p50_ms': percentile(xs, 50),
        'p95_ms': percentile(xs, 95),
        'p99_ms': percentile(xs, 99),
        'min_ms': float(xs.min()),
        'max_ms': float(xs.max()),
    }


@torch.no_grad()
def run_pair(bb, left, right, amp):
    with torch.cuda.amp.autocast(enabled=amp):
        bb.forward_2d_adaptive(left, side='left')
        bb.forward_2d_adaptive(right, side='right')


@torch.no_grad()
def run_reference_pair(bb, left, right, amp):
    with torch.cuda.amp.autocast(enabled=amp):
        bb.forward_2d(left)
        bb.forward_2d(right)


@torch.no_grad()
def run_shallow_pair(bb, left, right, amp):
    with torch.cuda.amp.autocast(enabled=amp):
        ls = bb.forward_2d_shallow(left)
        rs = bb.forward_2d_shallow(right)
    return ls, rs


@torch.no_grad()
def run_deep_pair(bb, left, right, ls, rs, amp):
    with torch.cuda.amp.autocast(enabled=amp):
        bb.forward_2d_deep_full_from_shallow(left, ls)
        bb.forward_2d_deep_full_from_shallow(right, rs)


def event_time(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    fn()
    end.record()

    end.synchronize()

    return float(start.elapsed_time(end))


def measure(fn, warmup, repeat):
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()

    xs = []

    for _ in range(repeat):
        xs.append(event_time(fn))

    return summarize(xs)


def main():
    args = parse_args()

    for r in args.ratios:
        assert 0.0 <= r <= 1.0

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=True,
    )

    model.cuda().eval()

    bb = get_backbone(model)

    sample = dataset[args.sample_index]
    batch = dataset.collate_batch([sample])

    load_data_to_gpu(batch)

    token = batch['token'] if 'token' in batch else batch

    left = token['left_img']
    right = token['right_img']

    amp = bool(cfg.MODEL.USE_AMP['TEST'])

    print()
    print('=' * 80)
    print('Stage-A profiling')
    print('A_s  = stem + layer1 FULL')
    print('A_d  = layer2 + layer3 + layer4 selective/cache')
    print('neck = full-context SPP + ROI FPN/upconv/lastconv + stereo cache')
    print('=' * 80)

    # ------------------------------------------------------------
    # Original A reference
    # ------------------------------------------------------------
    bb.clear_adaptive_a_profile()

    ref_stat = measure(
        lambda: run_reference_pair(
            bb,
            left,
            right,
            amp,
        ),
        args.warmup,
        args.repeat,
    )

    print(
        f"[Full original A] "
        f"mean={ref_stat['mean_ms']:.3f} "
        f"p50={ref_stat['p50_ms']:.3f} "
        f"p95={ref_stat['p95_ms']:.3f} "
        f"p99={ref_stat['p99_ms']:.3f}"
    )

    # ------------------------------------------------------------
    # A_s only
    # ------------------------------------------------------------
    shallow_stat = measure(
        lambda: run_shallow_pair(
            bb,
            left,
            right,
            amp,
        ),
        args.warmup,
        args.repeat,
    )

    print(
        f"[A_s Full] "
        f"mean={shallow_stat['mean_ms']:.3f} "
        f"p50={shallow_stat['p50_ms']:.3f} "
        f"p95={shallow_stat['p95_ms']:.3f} "
        f"p99={shallow_stat['p99_ms']:.3f}"
    )

    # ------------------------------------------------------------
    # A_d full + neck, given precomputed shallow
    # ------------------------------------------------------------
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        ls = bb.forward_2d_shallow(left)
        rs = bb.forward_2d_shallow(right)

    deep_stat = measure(
        lambda: run_deep_pair(
            bb,
            left,
            right,
            ls,
            rs,
            amp,
        ),
        args.warmup,
        args.repeat,
    )

    print(
        f"[A_d Full + Neck] "
        f"mean={deep_stat['mean_ms']:.3f} "
        f"p50={deep_stat['p50_ms']:.3f} "
        f"p95={deep_stat['p95_ms']:.3f} "
        f"p99={deep_stat['p99_ms']:.3f}"
    )

    records = []

    # ------------------------------------------------------------
    # ROI curve
    # ------------------------------------------------------------
    for ratio in args.ratios:
        for position in args.positions:

            # Each configuration starts from a valid FULL stage cache.
            bb.reset_adaptive_a_cache()

            bb.set_adaptive_a_profile(
                ratio=1.0,
                position_mode='center',
            )

            # Full initialization is outside formal measurement.
            run_pair(
                bb,
                left,
                right,
                amp,
            )

            torch.cuda.synchronize()

            bb.set_adaptive_a_profile(
                ratio=ratio,
                position_mode=position,
            )

            stat = measure(
                lambda: run_pair(
                    bb,
                    left,
                    right,
                    amp,
                ),
                args.warmup,
                args.repeat,
            )

            roi = bb._adaptive_a_last_roi

            if roi is None:
                roi_hw = (0, 0)
                actual_ratio = 0.0
            else:
                y0, y1, x0, x1 = roi
                roi_hw = (y1 - y0, x1 - x0)

                H = bb._adaptive_a_cache['left'][0].shape[-2]
                W = bb._adaptive_a_cache['left'][0].shape[-1]

                actual_ratio = (
                    roi_hw[0] * roi_hw[1]
                    /
                    float(H * W)
                )

            rec = {
                'requested_ratio': float(ratio),
                'actual_ratio': float(actual_ratio),
                'position': position,
                'roi_hw': list(roi_hw),
                **stat,
            }

            records.append(rec)

            print(
                f"ratio={ratio:5.2f} "
                f"actual={actual_ratio:6.3f} "
                f"pos={position:8s} "
                f"ROI={roi_hw[0]:3d}x{roi_hw[1]:3d} "
                f"mean={stat['mean_ms']:7.3f} "
                f"p50={stat['p50_ms']:7.3f} "
                f"p95={stat['p95_ms']:7.3f} "
                f"p99={stat['p99_ms']:7.3f}"
            )

    bb.clear_adaptive_a_profile()

    result = {
        'definition': {
            'A_s': 'ResNet stem + layer1, full',
            'A_d':
                'ResNet layer2 + layer3 + layer4 + ROI neck, selective/cache',
            'feature_neck':
                'full-context SPP + ROI FPN/upconv/lastconv + stereo cache',
            'timing_boundary':
                'left+right full A_s + selective layer2-4 + '
                'ROI neck + stage-A stereo cache update',
        },
        'reference_full_a': ref_stat,
        'a_s_full': shallow_stat,
        'a_d_full_plus_neck': deep_stat,
        'records': records,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    out.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print()
    print('Saved:', out)


if __name__ == '__main__':
    main()
