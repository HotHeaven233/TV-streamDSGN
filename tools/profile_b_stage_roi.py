#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import profile_b2_roi_fragmentation as base


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = (
    "extra_data/checkpoint_epoch_20.pth"
)


def parse_args():

    p = argparse.ArgumentParser(
        "Profile selective DPS CostVolume + B2"
    )

    p.add_argument(
        "--cfg_file",
        default=DEFAULT_CFG,
    )

    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
    )

    p.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[
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
        "--fragments",
        nargs="+",
        type=int,
        default=[
            1, 2, 4, 8, 16
        ],
    )

    p.add_argument(
        "--halo",
        type=int,
        default=1,
    )

    p.add_argument(
        "--align",
        type=int,
        default=4,
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=30,
    )

    p.add_argument(
        "--repeat",
        type=int,
        default=200,
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "b_stage_roi_profile.json"
        ),
    )

    return p.parse_args()


def main():

    args = parse_args()

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    cfg.LOCAL_RANK = 0
    cfg.MODEL.SAVE_TIME = False

    logger = common_utils.create_logger(
        rank=0
    )

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
        num_class=len(
            cfg.CLASS_NAMES
        ),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=False,
    )

    model.cuda().eval()

    backbone = base.find_backbone(
        model
    )

    # ========================================================
    # Capture REAL BuildCostVolume inputs.
    # ========================================================

    captured = {}

    def hook(
        module,
        inputs,
    ):

        captured["left"] = (
            inputs[0]
            .detach()
            .clone()
        )

        captured["right"] = (
            inputs[1]
            .detach()
            .clone()
        )

        captured["shift"] = (
            inputs[4]
            .detach()
            .clone()
        )

        if len(inputs) >= 6:
            captured["psv"] = (
                inputs[5]
                .detach()
                .clone()
            )
        else:
            captured["psv"] = None

    handle = (
        backbone.build_cost
        .register_forward_pre_hook(
            hook
        )
    )

    sample = dataset[0]

    batch = dataset.collate_batch(
        [sample]
    )

    load_data_to_gpu(
        batch
    )

    with torch.no_grad():
        model(batch)

    torch.cuda.synchronize()

    handle.remove()

    left = (
        captured["left"]
        .contiguous()
    )

    right = (
        captured["right"]
        .contiguous()
    )

    shift = (
        captured["shift"]
        .contiguous()
    )

    psv = captured["psv"]

    if psv is None:
        raise RuntimeError(
            "Current forward did not use DPS "
            "psv_disps_channels."
        )

    psv = psv.contiguous()

    print()
    print(
        "left:",
        tuple(left.shape),
        left.dtype,
    )

    print(
        "right:",
        tuple(right.shape),
        right.dtype,
    )

    print(
        "shift:",
        tuple(shift.shape),
        shift.dtype,
    )

    print(
        "psv:",
        tuple(psv.shape),
        psv.dtype,
    )

    if left.shape[1] <= backbone.cv_dim:
        raise RuntimeError(
            "Expected DPS path, but "
            f"left channels={left.shape[1]} "
            f"<= cv_dim={backbone.cv_dim}"
        )

    # ========================================================
    # Full reference.
    # ========================================================

    def run_full_cost():

        return backbone.build_cost(
            left,
            right,
            None,
            None,
            shift,
            psv,
        )

    with torch.no_grad():

        full_cost = (
            run_full_cost()
        )

        full_b2 = (
            base.run_b2(
                backbone,
                full_cost,
            )
        )

    torch.cuda.synchronize()

    print()
    print(
        "full cost:",
        tuple(full_cost.shape),
        full_cost.dtype,
    )

    print(
        "full B2:",
        tuple(full_b2.shape),
        full_b2.dtype,
    )

    _, _, _, H, W = (
        full_cost.shape
    )

    # ========================================================
    # Full cost-only latency.
    # ========================================================

    full_cost_stats = (
        base.measure(
            run_full_cost,
            args.warmup,
            args.repeat,
        )
    )

    # ========================================================
    # Full B-stage:
    # cost-volume + dres0 + dres1.
    # ========================================================

    def run_full_b():

        c = run_full_cost()

        return base.run_b2(
            backbone,
            c,
        )

    full_b_stats = (
        base.measure(
            run_full_b,
            args.warmup,
            args.repeat,
        )
    )

    print()
    print("=" * 80)

    print(
        "FULL cost volume:",
        f"mean={full_cost_stats['mean_ms']:.3f}",
        f"p95={full_cost_stats['p95_ms']:.3f}",
        f"p99={full_cost_stats['p99_ms']:.3f}",
    )

    print(
        "FULL B stage    :",
        f"mean={full_b_stats['mean_ms']:.3f}",
        f"p95={full_b_stats['p95_ms']:.3f}",
        f"p99={full_b_stats['p99_ms']:.3f}",
    )

    print("=" * 80)

    canvas = torch.empty_like(
        full_b2
    )

    records = []

    print()
    print("=" * 132)

    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'Exec%':>8} "
        f"{'Cost':>9} "
        f"{'B-total':>9} "
        f"{'P99':>9} "
        f"{'Speedup':>9} "
        f"{'CostDiff':>12} "
        f"{'B2Diff':>12}"
    )

    print("-" * 132)

    for ratio in args.ratios:

        for frag in args.fragments:

            semantic_rois = (
                base.make_fragmented_rois(
                    H,
                    W,
                    ratio,
                    frag,
                )
            )

            exec_rois = [
                base.expand_align(
                    r,
                    args.halo,
                    H,
                    W,
                    args.align,
                )
                for r in semantic_rois
            ]

            exec_sum_area = sum(
                base.roi_area(r)
                for r in exec_rois
            )

            exec_ratio = (
                exec_sum_area
                / float(H * W)
            )

            # ------------------------------------------------
            # Cost-volume ROI only.
            # ------------------------------------------------

            def run_cost_plan():

                last = None

                for e in exec_rois:

                    (
                        ey0,
                        ey1,
                        ex0,
                        ex1,
                    ) = e

                    last = (
                        backbone.build_cost
                        .forward_roi(
                            left,
                            right,
                            None,
                            None,
                            shift,
                            ey0,
                            ey1,
                            ex0,
                            ex1,
                            psv,
                        )
                    )

                return last

            cost_stats = (
                base.measure(
                    run_cost_plan,
                    args.warmup,
                    args.repeat,
                )
            )

            # ------------------------------------------------
            # Combined:
            #
            # DPS cost ROI
            #   -> B2 ROI
            #   -> semantic-core scatter
            # ------------------------------------------------

            def run_b_plan():

                for core, e in zip(
                    semantic_rois,
                    exec_rois,
                ):

                    (
                        cy0,
                        cy1,
                        cx0,
                        cx1,
                    ) = core

                    (
                        ey0,
                        ey1,
                        ex0,
                        ex1,
                    ) = e

                    cost_roi = (
                        backbone.build_cost
                        .forward_roi(
                            left,
                            right,
                            None,
                            None,
                            shift,
                            ey0,
                            ey1,
                            ex0,
                            ex1,
                            psv,
                        )
                    )

                    y = (
                        base.run_b2(
                            backbone,
                            cost_roi,
                        )
                    )

                    ry0 = (
                        cy0 - ey0
                    )

                    rx0 = (
                        cx0 - ex0
                    )

                    rh = (
                        cy1 - cy0
                    )

                    rw = (
                        cx1 - cx0
                    )

                    canvas[
                        ...,
                        cy0:cy1,
                        cx0:cx1
                    ].copy_(
                        y[
                            ...,
                            ry0:ry0 + rh,
                            rx0:rx0 + rw
                        ]
                    )

                return canvas

            b_stats = (
                base.measure(
                    run_b_plan,
                    args.warmup,
                    args.repeat,
                )
            )

            # ------------------------------------------------
            # Correctness.
            # ------------------------------------------------

            max_cost_diff = 0.0
            max_b2_diff = 0.0

            with torch.no_grad():

                for core, e in zip(
                    semantic_rois,
                    exec_rois,
                ):

                    (
                        cy0,
                        cy1,
                        cx0,
                        cx1,
                    ) = core

                    (
                        ey0,
                        ey1,
                        ex0,
                        ex1,
                    ) = e

                    cost_roi = (
                        backbone.build_cost
                        .forward_roi(
                            left,
                            right,
                            None,
                            None,
                            shift,
                            ey0,
                            ey1,
                            ex0,
                            ex1,
                            psv,
                        )
                    )

                    cost_ref = (
                        full_cost[
                            ...,
                            ey0:ey1,
                            ex0:ex1
                        ]
                    )

                    cost_diff = (
                        cost_roi.float()
                        - cost_ref.float()
                    ).abs()

                    max_cost_diff = max(
                        max_cost_diff,
                        float(
                            cost_diff
                            .max()
                            .item()
                        ),
                    )

                    y = (
                        base.run_b2(
                            backbone,
                            cost_roi,
                        )
                    )

                    ry0 = (
                        cy0 - ey0
                    )

                    rx0 = (
                        cx0 - ex0
                    )

                    rh = (
                        cy1 - cy0
                    )

                    rw = (
                        cx1 - cx0
                    )

                    pred = (
                        y[
                            ...,
                            ry0:ry0 + rh,
                            rx0:rx0 + rw
                        ]
                        .float()
                    )

                    ref = (
                        full_b2[
                            ...,
                            cy0:cy1,
                            cx0:cx1
                        ]
                        .float()
                    )

                    bdiff = (
                        pred - ref
                    ).abs()

                    max_b2_diff = max(
                        max_b2_diff,
                        float(
                            bdiff
                            .max()
                            .item()
                        ),
                    )

            speedup = (
                full_b_stats[
                    "mean_ms"
                ]
                /
                b_stats[
                    "mean_ms"
                ]
            )

            rec = {
                "ratio":
                    ratio,

                "fragments":
                    frag,

                "exec_sum_area_ratio":
                    exec_ratio,

                "cost_latency":
                    cost_stats,

                "b_stage_latency":
                    b_stats,

                "speedup_vs_full":
                    speedup,

                "max_cost_abs_diff":
                    max_cost_diff,

                "max_b2_abs_diff":
                    max_b2_diff,

                "semantic_rois": [
                    list(r)
                    for r in semantic_rois
                ],

                "exec_rois": [
                    list(r)
                    for r in exec_rois
                ],
            }

            records.append(
                rec
            )

            print(
                f"{ratio*100:6.1f}% "
                f"{frag:5d} "
                f"{exec_ratio*100:7.2f}% "
                f"{cost_stats['mean_ms']:9.3f} "
                f"{b_stats['mean_ms']:9.3f} "
                f"{b_stats['p99_ms']:9.3f} "
                f"{speedup:9.3f} "
                f"{max_cost_diff:12.6g} "
                f"{max_b2_diff:12.6g}"
            )

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            {
                "full_cost":
                    full_cost_stats,

                "full_b_stage":
                    full_b_stats,

                "input_left_shape":
                    list(left.shape),

                "cost_shape":
                    list(full_cost.shape),

                "halo":
                    args.halo,

                "align":
                    args.align,

                "records":
                    records,
            },
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        out,
    )


if __name__ == "__main__":
    main()
