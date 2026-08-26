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
import profile_c_roi_fragmentation as cprof
import profile_roi_fragmentation as dprof
import eval_b2_packing_oracle as packing


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():

    p = argparse.ArgumentParser(
        "Joint C + D physical packing oracle"
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
        default=[1, 2, 4, 8, 16],
    )

    p.add_argument(
        "--halo",
        type=int,
        default=12,
    )

    p.add_argument(
        "--align",
        type=int,
        default=4,
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
        "--numeric-tol",
        type=float,
        default=0.005,
        help=(
            "Maximum absolute feature difference allowed for "
            "FP16-equivalent selective execution."
        ),
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=30,
    )

    p.add_argument(
        "--repeat",
        type=int,
        default=150,
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "cd_packing_oracle_p99.json"
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
        return packing.area_greedy_to_cap(
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

    if strategy == "cap2":
        return packing.area_greedy_to_cap(
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
    # Capture exact B output (= C input), exact C output
    # (= D input), and exact full D output.
    # ========================================================

    captured = {}

    def dres0_hook(
        module,
        inputs,
        output,
    ):
        captured["dres0"] = (
            output
            .detach()
            .clone()
        )

    def dres1_hook(
        module,
        inputs,
        output,
    ):
        captured["dres1"] = (
            output
            .detach()
            .clone()
        )

    def d_input_hook(
        module,
        inputs,
    ):
        captured["d_input"] = (
            inputs[0]
            .detach()
            .clone()
        )

    def d_output_hook(
        module,
        inputs,
        output,
    ):
        captured["d_output"] = (
            output
            .detach()
            .clone()
        )

    h0 = (
        backbone.dres0
        .register_forward_hook(
            dres0_hook
        )
    )

    h1 = (
        backbone.dres1
        .register_forward_hook(
            dres1_hook
        )
    )

    hd0 = (
        backbone.rpn3d_convs
        .register_forward_pre_hook(
            d_input_hook
        )
    )

    hd1 = (
        backbone.rpn3d_pool
        .register_forward_hook(
            d_output_hook
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

    h0.remove()
    h1.remove()
    hd0.remove()
    hd1.remove()

    token_data = batch["token"]

    if token_data["batch_size"] != 1:
        raise RuntimeError(
            "Profiler assumes batch_size=1"
        )

    stereo_out = (
        captured["dres0"]
        + captured["dres1"]
    ).contiguous()

    d_input_ref = (
        captured["d_input"]
        .contiguous()
    )

    d_output_ref = (
        captured["d_output"]
        .contiguous()
    )

    left = token_data[
        "left_img"
    ]

    calib = token_data[
        "calib"
    ][0]

    image_shape = token_data[
        "image_shape"
    ][0]

    random_T = (
        token_data["random_T"][0]
        if "random_T" in token_data
        else None
    )

    tensor_dtype = (
        torch.float16
        if backbone.use_amp
        else torch.float32
    )

    _, _, _, H, W = (
        d_input_ref.shape
    )

    print()
    print(
        "Stereo/B output:",
        tuple(
            stereo_out.shape
        ),
        stereo_out.dtype,
    )

    print(
        "C/D boundary:",
        tuple(
            d_input_ref.shape
        ),
        d_input_ref.dtype,
    )

    print(
        "D/full output:",
        tuple(
            d_output_ref.shape
        ),
        d_output_ref.dtype,
    )

    print(
        "AMP:",
        backbone.use_amp,
    )

    # ========================================================
    # One-time C shared setup PER FORWARD.
    #
    # This is deliberately executed inside every measured plan,
    # but only once regardless of NROI.
    # ========================================================

    def prepare_c_shared():

        coords = (
            backbone.coordinates_3d
            .cuda()
        )

        if backbone.use_amp:
            coords = coords.half()

        P2 = torch.as_tensor(
            calib.P2,
            device="cuda",
            dtype=tensor_dtype,
        )

        return (
            coords,
            P2,
        )

    # ========================================================
    # Full C+D baseline using the same physical implementation
    # boundary as selective execution:
    #
    #   shared setup
    #   -> full C query
    #   -> exact D stage
    #
    # Compare it against outputs captured from the original
    # model before using timings.
    # ========================================================

    full_roi = (
        0,
        H,
        0,
        W,
    )

    def run_full_cd():

        coords, P2 = (
            prepare_c_shared()
        )

        c = cprof.run_c_roi(
            backbone,
            stereo_out,
            coords,
            full_roi,
            left.shape[2:],
            image_shape,
            P2,
            random_T=random_T,
        )

        y = dprof.run_d_stage(
            backbone,
            c,
        )

        return y

    with torch.no_grad():
        full_cd_test = (
            run_full_cd()
        )

    torch.cuda.synchronize()

    full_d_input_test = None

    with torch.no_grad():

        coords_test, P2_test = (
            prepare_c_shared()
        )

        full_d_input_test = (
            cprof.run_c_roi(
                backbone,
                stereo_out,
                coords_test,
                full_roi,
                left.shape[2:],
                image_shape,
                P2_test,
                random_T=random_T,
            )
        )

    torch.cuda.synchronize()

    c_diff = (
        full_d_input_test.float()
        - d_input_ref.float()
    ).abs()

    d_diff = (
        full_cd_test.float()
        - d_output_ref.float()
    ).abs()

    print()
    print(
        "Full C max diff:",
        float(
            c_diff.max().item()
        ),
    )

    print(
        "Full C+D max diff:",
        float(
            d_diff.max().item()
        ),
    )

    if float(
        c_diff.max().item()
    ) > 1e-4:
        raise RuntimeError(
            "Full C reconstruction mismatch"
        )

    if float(
        d_diff.max().item()
    ) > 1e-4:
        raise RuntimeError(
            "Full C+D reconstruction mismatch"
        )

    full_stats = (
        base.measure(
            run_full_cd,
            args.warmup,
            args.repeat,
        )
    )

    full_metric = float(
        full_stats[
            args.metric
        ]
    )

    print()
    print("=" * 90)

    print(
        "FULL C+D:",
        f"mean={full_stats['mean_ms']:.3f}",
        f"p50={full_stats['p50_ms']:.3f}",
        f"p95={full_stats['p95_ms']:.3f}",
        f"p99={full_stats['p99_ms']:.3f}",
    )

    print("=" * 90)

    # ========================================================
    # Joint physical packing sweep.
    #
    # physical core ROI
    #       ↓
    # D dependency halo + alignment
    #       ↓
    # C computes exactly the expanded ROI
    #       ↓
    # D executes on that ROI
    #       ↓
    # scatter only the physical core
    # ========================================================

    strategies = [
        "naive",
        "cap4",
        "cap2",
        "bbox1",
    ]

    canvas = torch.empty_like(
        d_output_ref
    )

    records = []

    print()
    print("=" * 154)

    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'Strategy':>10} "
        f"{'ROI':>4} "
        f"{'Core%':>8} "
        f"{'ExecSum%':>9} "
        f"{'Mean':>9} "
        f"{'P95':>9} "
        f"{'P99':>9} "
        f"{'Save':>9} "
        f"{'MaxDiff':>11}"
    )

    print("-" * 154)

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
                        args.halo,
                        H,
                        W,
                        args.align,
                    )
                )

                exec_pairs = []

                core_sum_area = 0
                exec_sum_area = 0

                for core in physical_rois:

                    e = base.expand_align(
                        core,
                        args.halo,
                        H,
                        W,
                        args.align,
                    )

                    exec_pairs.append(
                        (
                            core,
                            e,
                        )
                    )

                    core_sum_area += (
                        base.roi_area(
                            core
                        )
                    )

                    exec_sum_area += (
                        base.roi_area(
                            e
                        )
                    )

                core_ratio = (
                    core_sum_area
                    / float(
                        H * W
                    )
                )

                exec_ratio = (
                    exec_sum_area
                    / float(
                        H * W
                    )
                )

                # Offline-only geometry bookkeeping:
                # unique physical execution coverage.
                exec_union_mask = torch.zeros(
                    (H, W),
                    dtype=torch.bool,
                    device="cpu",
                )

                for _, e in exec_pairs:
                    ey0, ey1, ex0, ex1 = e

                    exec_union_mask[
                        ey0:ey1,
                        ex0:ex1
                    ] = True

                exec_union_ratio = (
                    float(
                        exec_union_mask.sum().item()
                    )
                    / float(H * W)
                )

                def run_plan():

                    # One shared setup for all ROIs.
                    coords, P2 = (
                        prepare_c_shared()
                    )

                    for core, e in exec_pairs:

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

                        c_roi = (
                            cprof.run_c_roi(
                                backbone,
                                stereo_out,
                                coords,
                                e,
                                left.shape[2:],
                                image_shape,
                                P2,
                                random_T=random_T,
                            )
                        )

                        d_roi = (
                            dprof.run_d_stage(
                                backbone,
                                c_roi,
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
                            d_roi[
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
                # Numerical equivalence on the physical cores.
                # --------------------------------------------

                max_diff = 0.0

                with torch.no_grad():

                    coords_check, P2_check = (
                        prepare_c_shared()
                    )

                    for core, e in exec_pairs:

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

                        c_roi = (
                            cprof.run_c_roi(
                                backbone,
                                stereo_out,
                                coords_check,
                                e,
                                left.shape[2:],
                                image_shape,
                                P2_check,
                                random_T=random_T,
                            )
                        )

                        d_roi = (
                            dprof.run_d_stage(
                                backbone,
                                c_roi,
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
                            d_roi[
                                ...,
                                ry0:ry0 + rh,
                                rx0:rx0 + rw
                            ]
                            .float()
                        )

                        ref = (
                            d_output_ref[
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
                                diff.max().item()
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

                    "core_sum_area_ratio":
                        core_ratio,

                    "exec_sum_area_ratio":
                        exec_ratio,

                    "exec_union_area_ratio":
                        exec_union_ratio,

                    "force_full_by_coverage":
                        (
                            exec_union_ratio
                            >= 1.0 - 1e-12
                        ),

                    "latency":
                        s,

                    "metric_ms":
                        metric_ms,

                    "saving_ms":
                        saving,

                    "max_abs_diff":
                        max_diff,

                    "semantic_rois": [
                        list(r)
                        for r
                        in semantic_rois
                    ],

                    "physical_rois": [
                        list(r)
                        for r
                        in physical_rois
                    ],

                    "exec_rois": [
                        list(e)
                        for _, e
                        in exec_pairs
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
                    f"{core_ratio*100:7.2f}% "
                    f"{exec_ratio*100:8.2f}% "
                    f"{s['mean_ms']:9.3f} "
                    f"{s['p95_ms']:9.3f} "
                    f"{s['p99_ms']:9.3f} "
                    f"{saving:9.3f} "
                    f"{max_diff:11.6g}"
                )

            # --------------------------------------------
            # Experimental measured oracle.
            # --------------------------------------------

            numerically_valid = [
                r
                for r in group
                if (
                    r["max_abs_diff"]
                    <= args.numeric_tol
                )
            ]

            # A selective plan whose completed dependency
            # support already covers the whole feature map
            # has no structural selective-compute benefit.
            selective_candidates = [
                r
                for r in numerically_valid
                if (
                    r["exec_union_area_ratio"]
                    < 1.0 - 1e-12
                )
            ]

            if len(selective_candidates) == 0:

                decision = "FULL"
                final_ms = full_metric

                if len(numerically_valid) == 0:
                    reason = (
                        "no numerically valid selective plan"
                    )
                else:
                    reason = (
                        "dependency-completed coverage is FULL"
                    )

                print(
                    "    -> "
                    f"{reason}, "
                    f"no-regret=FULL, "
                    f"final={final_ms:.3f} ms"
                )

            else:

                best = min(
                    selective_candidates,
                    key=lambda r:
                        r["metric_ms"],
                )

                if (
                    best["metric_ms"]
                    + args.guard_ms
                    < full_metric
                ):
                    decision = (
                        best["strategy"]
                    )

                    final_ms = (
                        best["metric_ms"]
                    )

                else:
                    decision = "FULL"
                    final_ms = full_metric

                print(
                    "    -> "
                    f"best={best['strategy']} "
                    f"{best['metric_ms']:.3f} ms, "
                    f"save="
                    f"{full_metric-best['metric_ms']:.3f} ms, "
                    f"ExecUniq="
                    f"{best['exec_union_area_ratio']*100:.2f}%, "
                    f"MaxDiff="
                    f"{best['max_abs_diff']:.6g}, "
                    f"no-regret={decision}, "
                    f"final={final_ms:.3f} ms"
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

                "numeric_tol":
                    args.numeric_tol,

                "halo":
                    args.halo,

                "align":
                    args.align,

                "full_cd":
                    full_stats,

                "full_c_max_diff":
                    float(
                        c_diff.max().item()
                    ),

                "full_cd_max_diff":
                    float(
                        d_diff.max().item()
                    ),

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
