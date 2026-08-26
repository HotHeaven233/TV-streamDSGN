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

import profile_bev_to_stereo_support as omega_prof
import profile_c_roi_fragmentation as cprof
import profile_roi_fragmentation as dprof


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():

    p = argparse.ArgumentParser(
        "Direct measured joint B->C->D physical oracle"
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
        default=[0.10, 0.50],
    )

    p.add_argument(
        "--fragments",
        nargs="+",
        type=int,
        default=[1, 4, 16],
    )

    p.add_argument(
        "--cd-halo",
        type=int,
        default=12,
    )

    p.add_argument(
        "--cd-align",
        type=int,
        default=4,
    )

    p.add_argument(
        "--b-halo",
        type=int,
        default=1,
    )

    p.add_argument(
        "--b-align",
        type=int,
        default=4,
    )

    p.add_argument(
        "--numeric-tol",
        type=float,
        default=0.005,
    )

    p.add_argument(
        "--guard-ms",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=3,
    )

    p.add_argument(
        "--repeat",
        type=int,
        default=20,
    )

    p.add_argument(
        "--max-b-rois-screen",
        type=int,
        default=0,
        help=(
            "Profiling-only candidate screen. "
            "0 disables screening. "
            "This is NOT a runtime K_max constraint."
        ),
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "joint_bcd_physical_oracle.json"
        ),
    )

    return p.parse_args()


def percentile_stats(xs):

    a = np.asarray(
        xs,
        dtype=np.float64,
    )

    return {
        "mean_ms":
            float(a.mean()),

        "p50_ms":
            float(
                np.percentile(
                    a,
                    50,
                )
            ),

        "p95_ms":
            float(
                np.percentile(
                    a,
                    95,
                )
            ),

        "p99_ms":
            float(
                np.percentile(
                    a,
                    99,
                )
            ),
    }


def measure(
    fn,
    warmup,
    repeat,
):

    with torch.no_grad():

        for _ in range(
            warmup
        ):
            fn()

        torch.cuda.synchronize()

        times = []

        for _ in range(
            repeat
        ):

            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            fn()

            end.record()

            end.synchronize()

            times.append(
                start.elapsed_time(
                    end
                )
            )

    return percentile_stats(
        times
    )


def roi_union_ratio(
    rois,
    H,
    W,
):

    if len(rois) == 0:
        return 0.0

    mask = np.zeros(
        (H, W),
        dtype=np.bool_,
    )

    for y0, y1, x0, x1 in rois:
        mask[
            y0:y1,
            x0:x1
        ] = True

    return float(
        mask.mean()
    )


def canonical_rois(
    rois,
):
    return tuple(
        sorted(
            tuple(
                int(v)
                for v in r
            )
            for r in rois
        )
    )


def make_b_cores(
    rects,
    strategy,
    halo,
    H,
    W,
    align,
):

    if len(rects) == 0:
        return []

    if strategy == "keep":
        return list(
            rects
        )

    if strategy == "cap4":

        return (
            omega_prof.packing
            .area_greedy_to_cap(
                rects,
                min(
                    4,
                    len(rects),
                ),
                halo,
                H,
                W,
                align,
            )
        )

    if strategy == "cap2":

        return (
            omega_prof.packing
            .area_greedy_to_cap(
                rects,
                min(
                    2,
                    len(rects),
                ),
                halo,
                H,
                W,
                align,
            )
        )

    if strategy == "bbox1":

        return [
            omega_prof.packing
            .union_bbox(
                rects
            )
        ]

    raise ValueError(
        strategy
    )


def expand_rois(
    rois,
    halo,
    H,
    W,
    align,
):

    return [
        omega_prof.base.expand_align(
            r,
            halo,
            H,
            W,
            align,
        )
        for r in rois
    ]


def max_core_diff_5d(
    x,
    ref,
    cores,
):

    d = 0.0

    for y0, y1, x0, x1 in cores:

        cur = (
            x[
                ...,
                y0:y1,
                x0:x1,
            ]
            .float()
        )

        tgt = (
            ref[
                ...,
                y0:y1,
                x0:x1,
            ]
            .float()
        )

        d = max(
            d,
            float(
                (
                    cur - tgt
                )
                .abs()
                .max()
                .item()
            ),
        )

    return d


def build_omega(
    backbone,
    coordinates_3d,
    cd_exec_rois,
    left_hw,
    image_shape,
    P2,
    random_T,
    Hs,
    Ws,
):

    omega = torch.zeros(
        (Hs, Ws),
        dtype=torch.bool,
        device="cuda",
    )

    for e in cd_exec_rois:

        grid, valid = (
            omega_prof
            .build_mapping_roi(
                backbone,
                coordinates_3d,
                e,
                left_hw,
                image_shape,
                P2,
                random_T=random_T,
            )
        )

        cur = (
            omega_prof
            .omega_from_grid(
                grid,
                valid,
                Hs,
                Ws,
            )
        )

        omega |= cur

    return omega


def main():

    args = parse_args()

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

    dataset, _, _ = (
        build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=0,
            logger=logger,
            training=False,
        )
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
        omega_prof.base
        .find_backbone(
            model
        )
    )

    captured = {}

    # --------------------------------------------------------
    # Capture exact B inputs used by the original model.
    # --------------------------------------------------------

    def cost_pre_hook(
        module,
        hook_args,
        hook_kwargs,
    ):

        captured[
            "cost_args"
        ] = tuple(
            x.detach()
            if torch.is_tensor(x)
            else x
            for x in hook_args
        )

        captured[
            "cost_kwargs"
        ] = dict(
            hook_kwargs
        )

    # Full B2 reference.
    def dres0_hook(
        module,
        inputs,
        output,
    ):
        captured[
            "dres0"
        ] = output.detach()

    def dres1_hook(
        module,
        inputs,
        output,
    ):
        captured[
            "dres1"
        ] = output.detach()

    # Exact C/D boundary.
    def d_pre_hook(
        module,
        inputs,
    ):
        captured[
            "cd_ref"
        ] = (
            inputs[0]
            .detach()
            .clone()
        )

    # Exact D output.
    def d_out_hook(
        module,
        inputs,
        output,
    ):
        captured[
            "d_ref"
        ] = (
            output
            .detach()
            .clone()
        )

    h_cost = (
        backbone.build_cost
        .register_forward_pre_hook(
            cost_pre_hook,
            with_kwargs=True,
        )
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
            d_pre_hook
        )
    )

    hd1 = (
        backbone.rpn3d_pool
        .register_forward_hook(
            d_out_hook
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

    h_cost.remove()
    h0.remove()
    h1.remove()
    hd0.remove()
    hd1.remove()

    # --------------------------------------------------------
    # Decode captured B inputs.
    # Original BuildCostVolume.forward:
    #
    # left, right, left_raw, right_raw,
    # shift, psv_disps_channels
    # --------------------------------------------------------

    ca = captured[
        "cost_args"
    ]

    ck = captured[
        "cost_kwargs"
    ]

    if len(ca) < 5:
        raise RuntimeError(
            "Unexpected build_cost inputs"
        )

    b_left = ca[0]
    b_right = ca[1]
    left_raw = ca[2]
    right_raw = ca[3]
    shift = ca[4]

    if len(ca) >= 6:
        psv = ca[5]
    else:
        psv = ck.get(
            "psv_disps_channels",
            None,
        )

    stereo_ref = (
        captured["dres0"]
        +
        captured["dres1"]
    ).contiguous()

    cd_ref = captured[
        "cd_ref"
    ]

    d_ref = captured[
        "d_ref"
    ]

    token = batch[
        "token"
    ]

    if token[
        "batch_size"
    ] != 1:
        raise RuntimeError(
            "Profiler assumes batch_size=1"
        )

    c_left = token[
        "left_img"
    ]

    calib = token[
        "calib"
    ][0]

    image_shape = token[
        "image_shape"
    ][0]

    random_T = (
        token[
            "random_T"
        ][0]
        if "random_T" in token
        else None
    )

    _, _, Ds, Hs, Ws = (
        stereo_ref.shape
    )

    _, _, Zb, Hb, Wb = (
        cd_ref.shape
    )

    tensor_dtype = (
        torch.float16
        if backbone.use_amp
        else torch.float32
    )

    # --------------------------------------------------------
    # C shared setup.
    #
    # Keep this inside every timed execution so Full and
    # selective use the same BCD timing boundary.
    # --------------------------------------------------------

    def prepare_c_shared():

        coordinates_3d = (
            backbone.coordinates_3d
            .cuda()
        )

        if backbone.use_amp:
            coordinates_3d = (
                coordinates_3d.half()
            )

        P2 = torch.as_tensor(
            calib.P2,
            device="cuda",
            dtype=tensor_dtype,
        )

        return (
            coordinates_3d,
            P2,
        )

    # Offline copy only for dependency-plan generation.
    coordinates_off, P2_off = (
        prepare_c_shared()
    )

    # --------------------------------------------------------
    # Exact B execution functions.
    # --------------------------------------------------------

    def run_full_b():

        with torch.amp.autocast(
            "cuda",
            enabled=backbone.use_amp,
        ):

            cost = (
                backbone.build_cost(
                    b_left,
                    b_right,
                    left_raw,
                    right_raw,
                    shift,
                    psv,
                )
            )

            x0 = (
                backbone.dres0(
                    cost
                )
            )

            out = (
                backbone.dres1(
                    x0
                )
                + x0
            )

        return out

    def run_b_roi(
        e,
    ):

        y0, y1, x0, x1 = e

        with torch.amp.autocast(
            "cuda",
            enabled=backbone.use_amp,
        ):

            cost_roi = (
                backbone.build_cost
                .forward_roi(
                    b_left,
                    b_right,
                    left_raw,
                    right_raw,
                    shift,
                    psv_disps_channels=psv,
                    ph0=y0,
                    ph1=y1,
                    pw0=x0,
                    pw1=x1,
                )
            )

            x0v = (
                backbone.dres0(
                    cost_roi
                )
            )

            out = (
                backbone.dres1(
                    x0v
                )
                + x0v
            )

        return out

    # --------------------------------------------------------
    # Full BCD baseline.
    # --------------------------------------------------------

    full_cd_roi = (
        0,
        Hb,
        0,
        Wb,
    )

    def run_full_bcd():

        coordinates_3d, P2 = (
            prepare_c_shared()
        )

        stereo = run_full_b()

        c = cprof.run_c_roi(
            backbone,
            stereo,
            coordinates_3d,
            full_cd_roi,
            c_left.shape[2:],
            image_shape,
            P2,
            random_T=random_T,
        )

        d = dprof.run_d_stage(
            backbone,
            c,
        )

        return d

    with torch.no_grad():

        full_test = (
            run_full_bcd()
        )

        full_diff = float(
            (
                full_test.float()
                -
                d_ref.float()
            )
            .abs()
            .max()
            .item()
        )

    print()
    print(
        "B input left:",
        tuple(
            b_left.shape
        ),
        b_left.dtype,
    )

    print(
        "B input right:",
        tuple(
            b_right.shape
        ),
        b_right.dtype,
    )

    print(
        "Stereo/B ref:",
        tuple(
            stereo_ref.shape
        ),
        stereo_ref.dtype,
    )

    print(
        "C/D ref:",
        tuple(
            cd_ref.shape
        ),
        cd_ref.dtype,
    )

    print(
        "D ref:",
        tuple(
            d_ref.shape
        ),
        d_ref.dtype,
    )

    print(
        "AMP:",
        backbone.use_amp,
    )

    print(
        "Full BCD max diff:",
        full_diff,
    )

    if (
        full_diff
        > args.numeric_tol
    ):
        raise RuntimeError(
            "Full BCD reconstruction failed"
        )

    full_stats = measure(
        run_full_bcd,
        args.warmup,
        args.repeat,
    )

    full_p99 = (
        full_stats[
            "p99_ms"
        ]
    )

    print()
    print("=" * 100)

    print(
        "FULL BCD: "
        f"mean={full_stats['mean_ms']:.3f} "
        f"p50={full_stats['p50_ms']:.3f} "
        f"p95={full_stats['p95_ms']:.3f} "
        f"p99={full_stats['p99_ms']:.3f}"
    )

    print("=" * 100)

    # Reusable physical canvases.
    #
    # Untouched regions are intentionally unspecified:
    # dependency completion guarantees that current C queries
    # only read freshly recomputed Omega_B cells.
    stereo_canvas = torch.empty_like(
        stereo_ref
    )

    d_canvas = torch.empty_like(
        d_ref
    )

    cd_strategies = [
        "naive",
        "cap4",
        "cap2",
        "bbox1",
    ]

    b_strategies = [
        "keep",
        "cap4",
        "cap2",
        "bbox1",
    ]

    # Keep representative cover families for the first
    # measured oracle. More can be added later.
    tile_shapes = [
        (4, 8),
        (4, 16),
        (8, 16),
        (16, 32),
    ]

    all_records = []
    group_results = []

    print()
    print("=" * 190)

    print(
        f"{'R%':>6} "
        f"{'Frag':>5} "
        f"{'CD':>12} "
        f"{'CDR':>4} "
        f"{'CDExec%':>8} "
        f"{'Omega%':>7} "
        f"{'BPlan':>18} "
        f"{'BR':>4} "
        f"{'BExec%':>8} "
        f"{'Mean':>8} "
        f"{'P95':>8} "
        f"{'P99':>8} "
        f"{'Save':>8} "
        f"{'BDiff':>10} "
        f"{'DDiff':>10}"
    )

    print("-" * 190)

    for ratio in args.ratios:

        for frag in args.fragments:

            semantic_rois = (
                omega_prof.base
                .make_fragmented_rois(
                    Hb,
                    Wb,
                    ratio,
                    frag,
                )
            )

            semantic_ratio = (
                roi_union_ratio(
                    semantic_rois,
                    Hb,
                    Wb,
                )
            )

            group = []

            # --------------------------------------------
            # Canonicalize identical CD physical plans.
            # --------------------------------------------

            cd_plan_dict = {}

            for cd_strategy in cd_strategies:

                cd_cores = (
                    omega_prof
                    .make_physical_rois(
                        semantic_rois,
                        cd_strategy,
                        args.cd_halo,
                        Hb,
                        Wb,
                        args.cd_align,
                    )
                )

                key = (
                    canonical_rois(
                        cd_cores
                    )
                )

                if key not in cd_plan_dict:

                    cd_plan_dict[
                        key
                    ] = {
                        "labels": [],
                        "cores":
                            cd_cores,
                    }

                cd_plan_dict[
                    key
                ][
                    "labels"
                ].append(
                    cd_strategy
                )

            for cd_info in (
                cd_plan_dict.values()
            ):

                cd_cores = (
                    cd_info[
                        "cores"
                    ]
                )

                cd_label = "/".join(
                    cd_info[
                        "labels"
                    ]
                )

                cd_exec = expand_rois(
                    cd_cores,
                    args.cd_halo,
                    Hb,
                    Wb,
                    args.cd_align,
                )

                cd_exec_ratio = (
                    roi_union_ratio(
                        cd_exec,
                        Hb,
                        Wb,
                    )
                )

                # If dependency completion already covers
                # full C+D spatial support, selective CD
                # has no structural benefit.
                if (
                    cd_exec_ratio
                    >= 1.0 - 1e-12
                ):
                    continue

                omega = build_omega(
                    backbone,
                    coordinates_off,
                    cd_exec,
                    c_left.shape[2:],
                    image_shape,
                    P2_off,
                    random_T,
                    Hs,
                    Ws,
                )

                omega_ratio = (
                    float(
                        omega.sum()
                        .item()
                    )
                    /
                    float(
                        Hs * Ws
                    )
                )

                omega_cpu = (
                    omega
                    .detach()
                    .cpu()
                    .numpy()
                )

                # ----------------------------------------
                # Generate multiple rectangular
                # representations of Omega_B.
                # ----------------------------------------

                representations = {}

                cc = (
                    omega_prof
                    .component_bboxes(
                        omega_cpu
                    )
                )

                representations[
                    "cc"
                ] = cc

                for th, tw in tile_shapes:

                    tr = (
                        omega_prof
                        .tile_rect_cover(
                            omega_cpu,
                            th,
                            tw,
                        )
                    )

                    representations[
                        f"tile{th}x{tw}"
                    ] = tr

                # ----------------------------------------
                # Canonicalize B physical plans across
                # representations and packing strategies.
                # ----------------------------------------

                b_plan_dict = {}

                for rep_name, rects in (
                    representations.items()
                ):

                    for bs in b_strategies:

                        b_cores = (
                            make_b_cores(
                                rects,
                                bs,
                                args.b_halo,
                                Hs,
                                Ws,
                                args.b_align,
                            )
                        )

                        if len(
                            b_cores
                        ) == 0:
                            continue

                        key = (
                            canonical_rois(
                                b_cores
                            )
                        )

                        label = (
                            f"{rep_name}:{bs}"
                        )

                        if key not in (
                            b_plan_dict
                        ):

                            b_plan_dict[
                                key
                            ] = {
                                "labels":
                                    [],

                                "cores":
                                    b_cores,
                            }

                        b_plan_dict[
                            key
                        ][
                            "labels"
                        ].append(
                            label
                        )

                # ----------------------------------------
                # Function factory:
                # B selective -> stereo canvas
                # -> selective C -> selective D.
                # ----------------------------------------

                def make_joint_fn(
                    b_cores,
                    b_exec,
                    use_full_b,
                ):

                    def fn():

                        coordinates_3d, P2 = (
                            prepare_c_shared()
                        )

                        if use_full_b:

                            stereo = (
                                run_full_b()
                            )

                        else:

                            for (
                                core,
                                e,
                            ) in zip(
                                b_cores,
                                b_exec,
                            ):

                                broi = (
                                    run_b_roi(
                                        e
                                    )
                                )

                                cy0, cy1, cx0, cx1 = (
                                    core
                                )

                                ey0, ey1, ex0, ex1 = (
                                    e
                                )

                                ly0 = (
                                    cy0 - ey0
                                )

                                ly1 = (
                                    ly0
                                    +
                                    (
                                        cy1
                                        -
                                        cy0
                                    )
                                )

                                lx0 = (
                                    cx0 - ex0
                                )

                                lx1 = (
                                    lx0
                                    +
                                    (
                                        cx1
                                        -
                                        cx0
                                    )
                                )

                                stereo_canvas[
                                    ...,
                                    cy0:cy1,
                                    cx0:cx1,
                                ].copy_(
                                    broi[
                                        ...,
                                        ly0:ly1,
                                        lx0:lx1,
                                    ]
                                )

                            stereo = (
                                stereo_canvas
                            )

                        for (
                            core,
                            e,
                        ) in zip(
                            cd_cores,
                            cd_exec,
                        ):

                            c_roi = (
                                cprof.run_c_roi(
                                    backbone,
                                    stereo,
                                    coordinates_3d,
                                    e,
                                    c_left.shape[2:],
                                    image_shape,
                                    P2,
                                    random_T=
                                        random_T,
                                )
                            )

                            d_roi = (
                                dprof.run_d_stage(
                                    backbone,
                                    c_roi,
                                )
                            )

                            cy0, cy1, cx0, cx1 = (
                                core
                            )

                            ey0, ey1, ex0, ex1 = (
                                e
                            )

                            ly0 = (
                                cy0 - ey0
                            )

                            ly1 = (
                                ly0
                                +
                                (
                                    cy1
                                    -
                                    cy0
                                )
                            )

                            lx0 = (
                                cx0 - ex0
                            )

                            lx1 = (
                                lx0
                                +
                                (
                                    cx1
                                    -
                                    cx0
                                )
                            )

                            d_canvas[
                                ...,
                                cy0:cy1,
                                cx0:cx1,
                            ].copy_(
                                d_roi[
                                    ...,
                                    ly0:ly1,
                                    lx0:lx1,
                                ]
                            )

                        return d_canvas

                    return fn

                # ----------------------------------------
                # B FULL + CD selective is always a
                # legitimate stage-coupled candidate.
                # ----------------------------------------

                candidate_specs = [
                    {
                        "label":
                            "B_FULL",

                        "cores":
                            [],

                        "exec":
                            [],

                        "exec_ratio":
                            1.0,

                        "use_full_b":
                            True,
                    }
                ]

                for b_info in (
                    b_plan_dict.values()
                ):

                    b_cores = (
                        b_info[
                            "cores"
                        ]
                    )

                    # Profiling acceleration only.
                    #
                    # Do NOT interpret this as a fixed
                    # runtime K_max. The final execution
                    # policy remains latency-driven.
                    if (
                        args.max_b_rois_screen > 0
                        and
                        len(b_cores)
                        >
                        args.max_b_rois_screen
                    ):
                        continue

                    b_exec = expand_rois(
                        b_cores,
                        args.b_halo,
                        Hs,
                        Ws,
                        args.b_align,
                    )

                    b_exec_ratio = (
                        roi_union_ratio(
                            b_exec,
                            Hs,
                            Ws,
                        )
                    )

                    # If selective B dependency completion
                    # covers Full, use the actual B_FULL
                    # candidate instead.
                    if (
                        b_exec_ratio
                        >= 1.0 - 1e-12
                    ):
                        continue

                    candidate_specs.append(
                        {
                            "label":
                                "/".join(
                                    b_info[
                                        "labels"
                                    ]
                                ),

                            "cores":
                                b_cores,

                            "exec":
                                b_exec,

                            "exec_ratio":
                                b_exec_ratio,

                            "use_full_b":
                                False,
                        }
                    )

                # ----------------------------------------
                # Direct measured candidates.
                # ----------------------------------------

                for spec in candidate_specs:

                    fn = make_joint_fn(
                        spec[
                            "cores"
                        ],
                        spec[
                            "exec"
                        ],
                        spec[
                            "use_full_b"
                        ],
                    )

                    with torch.no_grad():
                        fn()

                    if spec[
                        "use_full_b"
                    ]:

                        b_diff = 0.0

                    else:

                        b_diff = (
                            max_core_diff_5d(
                                stereo_canvas,
                                stereo_ref,
                                spec[
                                    "cores"
                                ],
                            )
                        )

                    d_diff = (
                        max_core_diff_5d(
                            d_canvas,
                            d_ref,
                            cd_cores,
                        )
                    )

                    max_diff = max(
                        b_diff,
                        d_diff,
                    )

                    if (
                        max_diff
                        >
                        args.numeric_tol
                    ):

                        stats = {
                            "mean_ms":
                                float("inf"),

                            "p50_ms":
                                float("inf"),

                            "p95_ms":
                                float("inf"),

                            "p99_ms":
                                float("inf"),
                        }

                    else:

                        stats = measure(
                            fn,
                            args.warmup,
                            args.repeat,
                        )

                    p99 = (
                        stats[
                            "p99_ms"
                        ]
                    )

                    save = (
                        full_p99
                        -
                        p99
                    )

                    rec = {
                        "target_ratio":
                            ratio,

                        "semantic_ratio":
                            semantic_ratio,

                        "fragments":
                            frag,

                        "cd_label":
                            cd_label,

                        "cd_num_rois":
                            len(
                                cd_cores
                            ),

                        "cd_cores": [
                            list(r)
                            for r in cd_cores
                        ],

                        "cd_exec": [
                            list(r)
                            for r in cd_exec
                        ],

                        "cd_exec_ratio":
                            cd_exec_ratio,

                        "omega_ratio":
                            omega_ratio,

                        "b_label":
                            spec[
                                "label"
                            ],

                        "b_full":
                            spec[
                                "use_full_b"
                            ],

                        "b_num_rois":
                            (
                                1
                                if spec[
                                    "use_full_b"
                                ]
                                else len(
                                    spec[
                                        "cores"
                                    ]
                                )
                            ),

                        "b_cores": [
                            list(r)
                            for r in spec[
                                "cores"
                            ]
                        ],

                        "b_exec": [
                            list(r)
                            for r in spec[
                                "exec"
                            ]
                        ],

                        "b_exec_ratio":
                            spec[
                                "exec_ratio"
                            ],

                        "b_max_diff":
                            b_diff,

                        "d_max_diff":
                            d_diff,

                        "max_diff":
                            max_diff,

                        "latency":
                            stats,

                        "saving_ms":
                            save,
                    }

                    all_records.append(
                        rec
                    )

                    if (
                        max_diff
                        <= args.numeric_tol
                    ):
                        group.append(
                            rec
                        )

                    print(
                        f"{semantic_ratio*100:5.1f}% "
                        f"{frag:5d} "
                        f"{cd_label:>12.12} "
                        f"{len(cd_cores):4d} "
                        f"{cd_exec_ratio*100:7.2f}% "
                        f"{omega_ratio*100:6.2f}% "
                        f"{spec['label']:>18.18} "
                        f"{rec['b_num_rois']:4d} "
                        f"{spec['exec_ratio']*100:7.2f}% "
                        f"{stats['mean_ms']:8.3f} "
                        f"{stats['p95_ms']:8.3f} "
                        f"{stats['p99_ms']:8.3f} "
                        f"{save:8.3f} "
                        f"{b_diff:10.6g} "
                        f"{d_diff:10.6g}"
                    )

            # --------------------------------------------
            # Global stage-coupled oracle for this
            # semantic (ratio, fragmentation) pair.
            # --------------------------------------------

            # --------------------------------------------
            # Stage-local B no-regret gate.
            #
            # Compare B selective against B_FULL under the
            # SAME CD physical plan. This prevents a nearly
            # full B selective path from being chosen for a
            # sub-guard timing fluctuation.
            # --------------------------------------------

            if len(group) > 0:

                by_cd = {}

                for r in group:

                    key = tuple(
                        tuple(
                            int(v)
                            for v in roi
                        )
                        for roi in r[
                            "cd_cores"
                        ]
                    )

                    by_cd.setdefault(
                        key,
                        [],
                    ).append(
                        r
                    )

                stage_valid = []

                for cd_group in (
                    by_cd.values()
                ):

                    b_full_group = [
                        r
                        for r in cd_group
                        if r[
                            "b_full"
                        ]
                    ]

                    if len(
                        b_full_group
                    ) == 0:
                        continue

                    b_full_ref = min(
                        b_full_group,
                        key=lambda r:
                            r[
                                "latency"
                            ][
                                "p99_ms"
                            ],
                    )

                    b_full_p99 = (
                        b_full_ref[
                            "latency"
                        ][
                            "p99_ms"
                        ]
                    )

                    # B_FULL is always a valid stage-local
                    # candidate.
                    stage_valid.append(
                        b_full_ref
                    )

                    for r in cd_group:

                        if r[
                            "b_full"
                        ]:
                            continue

                        p99 = (
                            r[
                                "latency"
                            ][
                                "p99_ms"
                            ]
                        )

                        if (
                            p99
                            +
                            args.guard_ms
                            <
                            b_full_p99
                        ):
                            stage_valid.append(
                                r
                            )

                group = stage_valid

            if len(group) == 0:

                decision = (
                    "FULL_BCD"
                )

                best = None

                print(
                    "    -> no valid selective "
                    "BCD candidate, "
                    "no-regret=FULL_BCD"
                )

            else:

                best = min(
                    group,
                    key=lambda r:
                        r[
                            "latency"
                        ][
                            "p99_ms"
                        ],
                )

                best_p99 = (
                    best[
                        "latency"
                    ][
                        "p99_ms"
                    ]
                )

                if (
                    best_p99
                    +
                    args.guard_ms
                    <
                    full_p99
                ):

                    decision = (
                        "SELECTIVE"
                    )

                else:

                    decision = (
                        "FULL_BCD"
                    )

                print(
                    "    -> GLOBAL BEST: "
                    f"CD={best['cd_label']} "
                    f"B={best['b_label']} "
                    f"P99={best_p99:.3f} ms "
                    f"save={full_p99-best_p99:.3f} ms "
                    f"CDExec={best['cd_exec_ratio']*100:.2f}% "
                    f"Omega={best['omega_ratio']*100:.2f}% "
                    f"BExec={best['b_exec_ratio']*100:.2f}% "
                    f"no-regret={decision}"
                )

            group_results.append(
                {
                    "target_ratio":
                        ratio,

                    "semantic_ratio":
                        semantic_ratio,

                    "fragments":
                        frag,

                    "decision":
                        decision,

                    "best":
                        best,
                }
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
                "full_bcd":
                    full_stats,

                "full_bcd_max_diff":
                    full_diff,

                "numeric_tol":
                    args.numeric_tol,

                "guard_ms":
                    args.guard_ms,

                "cd_halo":
                    args.cd_halo,

                "cd_align":
                    args.cd_align,

                "b_halo":
                    args.b_halo,

                "b_align":
                    args.b_align,

                "max_b_rois_screen":
                    args.max_b_rois_screen,

                "tile_shapes": [
                    list(x)
                    for x in tile_shapes
                ],

                "records":
                    all_records,

                "groups":
                    group_results,
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
