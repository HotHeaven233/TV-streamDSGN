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
import eval_b2_packing_oracle as packing
import eval_b2_bucketed_oracle as bucketed


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():

    p = argparse.ArgumentParser(
        "Joint DPS CostVolume + B2 packing oracle"
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
        "--b2-lut",
        required=True,
    )

    p.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[
            0.10,
            0.30,
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
        "--metric",
        choices=[
            "mean_ms",
            "p95_ms",
            "p99_ms",
        ],
        default="p99_ms",
    )

    p.add_argument(
        "--guard-ms",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    p.add_argument(
        "--repeat",
        type=int,
        default=100,
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "b_stage_packing_oracle_p99.json"
        ),
    )

    return p.parse_args()


def make_physical_rois(
    semantic_rois,
    strategy,
    halo,
    H,
    W,
    align,
):

    if strategy == "naive":

        return list(
            semantic_rois
        )

    if strategy == "cap4":

        return (
            packing.area_greedy_to_cap(
                semantic_rois,
                min(
                    4,
                    len(semantic_rois),
                ),
                halo,
                H,
                W,
                align,
            )
        )

    if strategy == "cap2":

        return (
            packing.area_greedy_to_cap(
                semantic_rois,
                min(
                    2,
                    len(semantic_rois),
                ),
                halo,
                H,
                W,
                align,
            )
        )

    if strategy == "bbox1":

        return [
            packing.union_bbox(
                semantic_rois
            )
        ]

    raise ValueError(
        strategy
    )


def main():

    args = parse_args()

    # ========================================================
    # Load B2 physical-shape buckets
    # ========================================================

    with open(
        args.b2_lut,
        "r",
    ) as f:
        lut = json.load(f)

    halo = int(
        lut["halo"]
    )

    align = int(
        lut["align"]
    )

    h_buckets, w_buckets = (
        bucketed.load_buckets(
            lut
        )
    )

    print(
        "H buckets:",
        h_buckets,
    )

    print(
        "W buckets:",
        w_buckets,
    )

    # ========================================================
    # Build model
    # ========================================================

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
    # Capture true DPS inputs
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

        captured["psv"] = (
            inputs[5]
            .detach()
            .clone()
        )

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

    psv = (
        captured["psv"]
        .contiguous()
    )

    # ========================================================
    # Full reference
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

    _, _, _, H, W = (
        full_cost.shape
    )

    def run_full_b():

        c = (
            run_full_cost()
        )

        return base.run_b2(
            backbone,
            c,
        )

    full_stats = base.measure(
        run_full_b,
        args.warmup,
        args.repeat,
    )

    full_metric = float(
        full_stats[
            args.metric
        ]
    )

    print()
    print(
        "Full B:",
        f"mean={full_stats['mean_ms']:.3f}",
        f"p95={full_stats['p95_ms']:.3f}",
        f"p99={full_stats['p99_ms']:.3f}",
    )

    # Full-size composition destination.
    canvas = (
        torch.empty_like(
            full_b2
        )
    )

    strategies = [
        "naive",
        "cap4",
        "cap2",
        "bbox1",
    ]

    records = []

    print()
    print("=" * 146)

    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'Strategy':>10} "
        f"{'ROI':>4} "
        f"{'ExecSum%':>9} "
        f"{'Mean':>9} "
        f"{'P95':>9} "
        f"{'P99':>9} "
        f"{'Save':>9} "
        f"{'MaxDiff':>11}"
    )

    print("-" * 146)

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

            group = []

            for strategy in strategies:

                physical_rois = (
                    make_physical_rois(
                        semantic_rois,
                        strategy,
                        halo,
                        H,
                        W,
                        align,
                    )
                )

                # --------------------------------------------
                # Map each physical semantic ROI to an actual
                # profiled B2 execution bucket.
                # --------------------------------------------

                pairs = []

                exec_sum_area = 0

                for core in physical_rois:

                    e = (
                        bucketed
                        .make_bucketed_exec_roi(
                            core,
                            H,
                            W,
                            halo,
                            align,
                            h_buckets,
                            w_buckets,
                        )
                    )

                    pairs.append(
                        (
                            core,
                            e,
                        )
                    )

                    exec_sum_area += (
                        base.roi_area(
                            e
                        )
                    )

                exec_ratio = (
                    exec_sum_area
                    / float(
                        H * W
                    )
                )

                # --------------------------------------------
                # Actual physical execution.
                # --------------------------------------------

                def run_plan():

                    for core, e in pairs:

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
                            backbone
                            .build_cost
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

                        y = base.run_b2(
                            backbone,
                            cost_roi,
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

                s = base.measure(
                    run_plan,
                    args.warmup,
                    args.repeat,
                )

                # --------------------------------------------
                # Correctness on the PHYSICAL core.
                # --------------------------------------------

                max_diff = 0.0

                with torch.no_grad():

                    for core, e in pairs:

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
                            backbone
                            .build_cost
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

                        y = base.run_b2(
                            backbone,
                            cost_roi,
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

                        diff = (
                            pred - ref
                        ).abs()

                        max_diff = max(
                            max_diff,
                            float(
                                diff
                                .max()
                                .item()
                            ),
                        )

                metric_ms = float(
                    s[
                        args.metric
                    ]
                )

                saving = (
                    full_metric
                    - metric_ms
                )

                rec = {
                    "ratio":
                        ratio,

                    "fragments":
                        frag,

                    "strategy":
                        strategy,

                    "num_rois":
                        len(
                            physical_rois
                        ),

                    "exec_sum_area_ratio":
                        exec_ratio,

                    "latency":
                        s,

                    "metric_ms":
                        metric_ms,

                    "saving_ms":
                        saving,

                    "max_abs_diff":
                        max_diff,

                    "physical_rois": [
                        list(r)
                        for r
                        in physical_rois
                    ],

                    "exec_rois": [
                        list(e)
                        for _, e
                        in pairs
                    ],
                }

                records.append(
                    rec
                )

                group.append(
                    rec
                )

                print(
                    f"{ratio*100:6.1f}% "
                    f"{frag:5d} "
                    f"{strategy:>10} "
                    f"{len(physical_rois):4d} "
                    f"{exec_ratio*100:8.2f}% "
                    f"{s['mean_ms']:9.3f} "
                    f"{s['p95_ms']:9.3f} "
                    f"{s['p99_ms']:9.3f} "
                    f"{saving:9.3f} "
                    f"{max_diff:11.6g}"
                )

            # --------------------------------------------
            # Measured joint B-stage oracle.
            # --------------------------------------------

            best = min(
                group,
                key=lambda r:
                    r[
                        "metric_ms"
                    ],
            )

            if (
                best[
                    "metric_ms"
                ]
                + args.guard_ms
                < full_metric
            ):
                decision = (
                    best[
                        "strategy"
                    ]
                )

                final_ms = (
                    best[
                        "metric_ms"
                    ]
                )

            else:
                decision = "FULL"
                final_ms = full_metric

            print(
                "    -> "
                f"best={best['strategy']} "
                f"{best['metric_ms']:.3f} ms, "
                f"no-regret={decision}, "
                f"final={final_ms:.3f} ms"
            )

    # ========================================================
    # Save
    # ========================================================

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
                "metric":
                    args.metric,

                "guard_ms":
                    args.guard_ms,

                "full_b_stage":
                    full_stats,

                "halo":
                    halo,

                "align":
                    align,

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
