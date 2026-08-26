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


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():
    p = argparse.ArgumentParser(
        "Validate LUT-aligned B2 execution buckets"
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
        "--lut",
        required=True,
    )

    p.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[
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
        default=30,
    )

    p.add_argument(
        "--repeat",
        type=int,
        default=300,
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "b2_bucketed_oracle.json"
        ),
    )

    return p.parse_args()


def load_buckets(lut):
    """
    Physical execution dimensions that were ACTUALLY profiled.
    """

    hs = sorted({
        int(r["exec_h"])
        for r in lut["records"]
    })

    ws = sorted({
        int(r["exec_w"])
        for r in lut["records"]
    })

    return hs, ws


def next_bucket(
    value,
    buckets,
):
    for v in buckets:
        if v >= value:
            return v

    return buckets[-1]


def fit_interval(
    req0,
    req1,
    target_size,
    limit,
):
    """
    Enlarge [req0, req1) to exactly target_size,
    preserving the required interval.
    """

    req_size = (
        req1 - req0
    )

    if target_size < req_size:
        raise ValueError(
            "target_size smaller than required interval"
        )

    extra = (
        target_size
        - req_size
    )

    before = (
        extra // 2
    )

    out0 = (
        req0
        - before
    )

    out1 = (
        out0
        + target_size
    )

    if out0 < 0:
        out1 -= out0
        out0 = 0

    if out1 > limit:
        shift = (
            out1 - limit
        )

        out0 -= shift
        out1 = limit

    if out0 < 0:
        out0 = 0
        out1 = target_size

    assert (
        out1 - out0
        == target_size
    )

    assert (
        out0 <= req0
        and
        out1 >= req1
    )

    return (
        out0,
        out1,
    )


def make_bucketed_exec_roi(
    core_roi,
    H,
    W,
    halo,
    align,
    h_buckets,
    w_buckets,
):
    """
    1. Build minimum correct halo ROI.
    2. Round PHYSICAL execution size upward to a measured LUT bucket.
    3. Preserve global coordinates.
    """

    required = (
        base.expand_align(
            core_roi,
            halo,
            H,
            W,
            align,
        )
    )

    (
        ry0,
        ry1,
        rx0,
        rx1,
    ) = required

    req_h = (
        ry1 - ry0
    )

    req_w = (
        rx1 - rx0
    )

    bucket_h = next_bucket(
        req_h,
        h_buckets,
    )

    bucket_w = next_bucket(
        req_w,
        w_buckets,
    )

    ey0, ey1 = fit_interval(
        ry0,
        ry1,
        bucket_h,
        H,
    )

    ex0, ex1 = fit_interval(
        rx0,
        rx1,
        bucket_w,
        W,
    )

    return (
        ey0,
        ey1,
        ex0,
        ex1,
    )


def build_exact_lut(
    lut,
    metric,
):
    """
    Conservative exact-shape lookup.

    Multiple semantic core shapes can map to the same physical
    ExecH x ExecW.  Take the maximum observed latency for that
    physical shape.
    """

    table = {}

    for r in lut["records"]:

        key = (
            int(r["exec_h"]),
            int(r["exec_w"]),
        )

        value = float(
            r[metric]
        )

        if key not in table:
            table[key] = value
        else:
            table[key] = max(
                table[key],
                value,
            )

    return table


def predict_plan(
    physical_rois,
    H,
    W,
    halo,
    align,
    h_buckets,
    w_buckets,
    table,
):
    total = 0.0
    exec_rois = []

    for core in physical_rois:

        e = make_bucketed_exec_roi(
            core,
            H,
            W,
            halo,
            align,
            h_buckets,
            w_buckets,
        )

        eh = (
            e[1] - e[0]
        )

        ew = (
            e[3] - e[2]
        )

        key = (
            eh,
            ew,
        )

        if key not in table:
            raise RuntimeError(
                f"Missing exact LUT bucket {key}"
            )

        total += table[key]

        exec_rois.append(
            e
        )

    return (
        total,
        exec_rois,
    )


def make_exec_fn(
    backbone,
    full_input,
    physical_rois,
    canvas,
    H,
    W,
    halo,
    align,
    h_buckets,
    w_buckets,
):
    """
    crop + contiguous + B2 + core scatter are timed.
    """

    pairs = []

    for core in physical_rois:

        e = make_bucketed_exec_roi(
            core,
            H,
            W,
            halo,
            align,
            h_buckets,
            w_buckets,
        )

        pairs.append(
            (
                core,
                e,
            )
        )

    def run():

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

            crop = (
                full_input[
                    ...,
                    ey0:ey1,
                    ex0:ex1
                ]
                .contiguous()
            )

            y = base.run_b2(
                backbone,
                crop,
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

    return run


@torch.no_grad()
def validate_plan(
    backbone,
    full_input,
    full_output,
    physical_rois,
    H,
    W,
    halo,
    align,
    h_buckets,
    w_buckets,
):
    max_abs = 0.0

    for core in physical_rois:

        e = make_bucketed_exec_roi(
            core,
            H,
            W,
            halo,
            align,
            h_buckets,
            w_buckets,
        )

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

        crop = (
            full_input[
                ...,
                ey0:ey1,
                ex0:ex1
            ]
            .contiguous()
        )

        y = base.run_b2(
            backbone,
            crop,
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

        pred = y[
            ...,
            ry0:ry0 + rh,
            rx0:rx0 + rw
        ].float()

        ref = full_output[
            ...,
            cy0:cy1,
            cx0:cx1
        ].float()

        diff = (
            pred - ref
        ).abs()

        max_abs = max(
            max_abs,
            float(
                diff.max().item()
            ),
        )

    return max_abs


def main():
    args = parse_args()

    with open(
        args.lut,
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
        load_buckets(
            lut
        )
    )

    table = build_exact_lut(
        lut,
        args.metric,
    )

    print(
        "H buckets:",
        h_buckets,
    )

    print(
        "W buckets:",
        w_buckets,
    )

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    cfg.LOCAL_RANK = 0
    cfg.MODEL.SAVE_TIME = False

    logger = (
        common_utils.create_logger(
            rank=0
        )
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

    backbone = (
        base.find_backbone(
            model
        )
    )

    captured = {}

    def hook(
        module,
        inputs,
    ):
        captured["x"] = (
            inputs[0]
            .detach()
            .clone()
        )

    handle = (
        backbone.dres0
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

    x = (
        captured["x"]
        .contiguous()
    )

    _, _, _, H, W = x.shape

    with torch.no_grad():
        full_output = (
            base.run_b2(
                backbone,
                x,
            )
        )

    torch.cuda.synchronize()

    canvas = (
        torch.empty_like(
            full_output
        )
    )

    full_stats = (
        base.measure(
            lambda:
                base.run_b2(
                    backbone,
                    x,
                ),
            args.warmup,
            args.repeat,
        )
    )

    full_real = float(
        full_stats[
            args.metric
        ]
    )

    full_pred = float(
        lut[
            "full_b2"
        ][args.metric]
    )

    print()
    print(
        "Full measured:",
        full_real,
        "ms",
    )

    print(
        "Full LUT:",
        full_pred,
        "ms",
    )

    print()
    print("=" * 140)

    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'Strategy':>10} "
        f"{'ROI':>4} "
        f"{'Pred':>9} "
        f"{'Real':>9} "
        f"{'Err':>9} "
        f"{'MaxDiff':>11}"
    )

    print("-" * 140)

    records = []

    strategies = [
        "naive",
        "cap4",
        "cap2",
        "bbox1",
    ]

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

                if strategy == "naive":

                    physical_rois = list(
                        semantic_rois
                    )

                elif strategy == "cap4":

                    physical_rois = (
                        packing.area_greedy_to_cap(
                            semantic_rois,
                            min(
                                4,
                                len(
                                    semantic_rois
                                ),
                            ),
                            halo,
                            H,
                            W,
                            align,
                        )
                    )

                elif strategy == "cap2":

                    physical_rois = (
                        packing.area_greedy_to_cap(
                            semantic_rois,
                            min(
                                2,
                                len(
                                    semantic_rois
                                ),
                            ),
                            halo,
                            H,
                            W,
                            align,
                        )
                    )

                elif strategy == "bbox1":

                    physical_rois = [
                        packing.union_bbox(
                            semantic_rois
                        )
                    ]

                pred_ms, exec_rois = (
                    predict_plan(
                        physical_rois,
                        H,
                        W,
                        halo,
                        align,
                        h_buckets,
                        w_buckets,
                        table,
                    )
                )

                fn = make_exec_fn(
                    backbone,
                    x,
                    physical_rois,
                    canvas,
                    H,
                    W,
                    halo,
                    align,
                    h_buckets,
                    w_buckets,
                )

                s = base.measure(
                    fn,
                    args.warmup,
                    args.repeat,
                )

                real_ms = float(
                    s[
                        args.metric
                    ]
                )

                max_diff = (
                    validate_plan(
                        backbone,
                        x,
                        full_output,
                        physical_rois,
                        H,
                        W,
                        halo,
                        align,
                        h_buckets,
                        w_buckets,
                    )
                )

                err = (
                    pred_ms
                    - real_ms
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

                    "pred_ms":
                        pred_ms,

                    "real_ms":
                        real_ms,

                    "error_ms":
                        err,

                    "max_abs_diff":
                        max_diff,

                    "physical_rois": [
                        list(r)
                        for r in physical_rois
                    ],

                    "exec_rois": [
                        list(r)
                        for r in exec_rois
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
                    f"{pred_ms:9.3f} "
                    f"{real_ms:9.3f} "
                    f"{err:9.3f} "
                    f"{max_diff:11.6g}"
                )

            pred_best = min(
                group,
                key=lambda r:
                    r["pred_ms"],
            )

            real_best = min(
                group,
                key=lambda r:
                    r["real_ms"],
            )

            if (
                pred_best["pred_ms"]
                + args.guard_ms
                < full_pred
            ):
                pred_decision = (
                    pred_best[
                        "strategy"
                    ]
                )
                chosen_real = (
                    pred_best[
                        "real_ms"
                    ]
                )
            else:
                pred_decision = "FULL"
                chosen_real = full_real

            if (
                real_best["real_ms"]
                + args.guard_ms
                < full_real
            ):
                oracle_decision = (
                    real_best[
                        "strategy"
                    ]
                )
                oracle_real = (
                    real_best[
                        "real_ms"
                    ]
                )
            else:
                oracle_decision = "FULL"
                oracle_real = full_real

            regret = max(
                0.0,
                chosen_real
                - oracle_real,
            )

            print(
                "    -> "
                f"pred={pred_decision}, "
                f"oracle={oracle_decision}, "
                f"regret={regret:.3f} ms"
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
                "metric":
                    args.metric,

                "guard_ms":
                    args.guard_ms,

                "full_real":
                    full_stats,

                "full_lut":
                    lut["full_b2"],

                "h_buckets":
                    h_buckets,

                "w_buckets":
                    w_buckets,

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
