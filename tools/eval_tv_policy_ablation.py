#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    ElasticBEVBranch,
)
from pcdet.datasets.kitti.kitti_object_eval_python import (
    eval as kitti_eval,
)

from test_stream_buffer_timestamp import (
    load_one,
)

import test_tv_stream3d_online_forward as tv_online

from test_tv_stream3d_online_forward import (
    make_cfg,
    choose_scene_indices,
    configure_contender,
    execute_one,
    prewarm_all_causal_prefixes,
)

from tv_stream3d_controller import (
    TVStream3DController,
)

from tv_stream3d_causal_fused_runtime import (
    enable_causal_prefix_fused_cache,
    fused_cache_stats,
    set_forbid_cache_miss,
)

from smooth_cuda_contention import (
    make_high_priority_detector_stream,
)

from eval_tv_stream3d_30hz_random50 import (
    build_balanced_trace,
    frame_meta,
    scene_groups,
    copy_det,
    disable_recall,
    make_prediction,
    stats,
    builtin,
    reset_history,
)


POLICIES = (
    "full",
    "best_static",
    "tv_dynamic",
    "oracle_level",
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--full_cfg",
        required=True,
    )

    p.add_argument(
        "--full_ckpt",
        required=True,
    )

    p.add_argument(
        "--elastic_ckpt",
        required=True,
    )

    p.add_argument(
        "--prefix_bn_bank",
        required=True,
    )

    p.add_argument(
        "--controller_csv",
        required=True,
    )

    p.add_argument(
        "--levels_json",
        required=True,
    )

    p.add_argument(
        "--policy",
        required=True,
        choices=POLICIES,
    )

    p.add_argument(
        "--pressure_level",
        required=True,
        choices=(
            "L1",
            "L2",
            "L3",
            "L4",
        ),
    )

    p.add_argument(
        "--pressure_fraction",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--trace_seed",
        type=int,
        default=20260903,
    )

    p.add_argument(
        "--input_hz",
        type=float,
        default=35.0,
    )

    p.add_argument(
        "--runtime_warmup_frames",
        type=int,
        default=80,
    )

    p.add_argument(
        "--max_frames",
        type=int,
        default=0,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    p.add_argument(
        "--control_guard_per_boundary_ms",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    return p.parse_args()


def total_profile_p99(
    controller,
    level,
    pid,
):
    probe = float(
        controller.level_rows[
            level
        ][
            "probe_p99_ms"
        ]
    )

    suffix = float(
        controller.rows[
            (
                level,
                "after_prefix",
                int(pid),
            )
        ][
            "_bound"
        ]
    )

    return (
        probe
        +
        suffix
    )


def find_full_profile(
    controller,
):
    target = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )

    ids = [
        pid
        for pid
        in controller.profile_ids
        if tuple(
            controller.profile_schedule[
                pid
            ]
        ) == target
    ]

    if len(ids) != 1:
        raise RuntimeError(
            f"full schedule profile "
            f"count={len(ids)}"
        )

    return int(
        ids[0]
    )


def select_best_static(
    controller,
    pressure_level,
    period_ms,
):
    rows = []

    for pid in (
        controller.profile_ids
    ):
        worst_p99 = max(
            total_profile_p99(
                controller,
                "L0",
                pid,
            ),
            total_profile_p99(
                controller,
                pressure_level,
                pid,
            ),
        )

        rows.append(
            (
                int(pid),
                float(
                    controller
                    .profile_quality[
                        pid
                    ]
                ),
                float(
                    worst_p99
                ),
            )
        )

    feasible = [
        x
        for x
        in rows
        if x[2] <= period_ms
    ]

    if feasible:
        chosen = sorted(
            feasible,
            key=lambda x: (
                -x[1],
                x[2],
                x[0],
            ),
        )[0]

        was_feasible = True

    else:
        chosen = sorted(
            rows,
            key=lambda x: (
                x[2],
                -x[1],
                x[0],
            ),
        )[0]

        was_feasible = False

    pid, quality, bound = (
        chosen
    )

    return {
        "profile_id":
            pid,

        "schedule":
            tuple(
                controller
                .profile_schedule[
                    pid
                ]
            ),

        "quality":
            quality,

        "predicted_worst_total_p99_ms":
            bound,

        "feasible":
            was_feasible,
    }


def select_oracle_level_profile(
    controller,
    true_level,
    budget_ms,
):
    rows = []

    for pid in (
        controller.profile_ids
    ):
        bound = (
            total_profile_p99(
                controller,
                true_level,
                pid,
            )
        )

        rows.append(
            (
                int(pid),
                float(
                    controller
                    .profile_quality[
                        pid
                    ]
                ),
                float(
                    bound
                ),
            )
        )

    feasible = [
        x
        for x
        in rows
        if x[2] <= budget_ms
    ]

    if feasible:
        chosen = sorted(
            feasible,
            key=lambda x: (
                -x[1],
                x[2],
                x[0],
            ),
        )[0]

        is_feasible = True

    else:
        chosen = sorted(
            rows,
            key=lambda x: (
                x[2],
                -x[1],
                x[0],
            ),
        )[0]

        is_feasible = False

    pid, quality, bound = (
        chosen
    )

    return {
        "profile_id":
            pid,

        "schedule":
            tuple(
                controller
                .profile_schedule[
                    pid
                ]
            ),

        "quality":
            quality,

        "predicted_total_p99_ms":
            bound,

        "feasible":
            is_feasible,
    }


def execute_fixed_schedule(
    base_model,
    branch,
    bank,
    batch,
    model_stream,
    contender,
    schedule,
):
    """
    Execute one complete fixed schedule WITHOUT rolling
    checkpoint synchronization/decisions.

    Neural operations are identical to execute_one() and
    prewarm_all_causal_prefixes(); only runtime replanning
    boundaries are removed.
    """

    schedule = tuple(
        float(x)
        for x
        in schedule
    )

    if len(schedule) != 6:
        raise ValueError(
            f"schedule={schedule}"
        )

    if any(
        schedule[i + 1]
        >
        schedule[i]
        for i in range(5)
    ):
        raise RuntimeError(
            "non-monotonic schedule: "
            f"{schedule}"
        )

    frame = batch[
        "token"
    ]

    backbone = (
        base_model.backbone_3d
    )

    amp_enabled = bool(
        base_model.use_amp_dict[
            "TEST"
        ]
    )

    if (
        base_model
        .history_feature_queue
        is not None
        and (
            "prev_sample_idx"
            not in frame
            or
            frame[
                "prev_sample_idx"
            ] == ""
        )
    ):
        base_model\
            .history_feature_queue\
            .clear()

    input_ready = torch.cuda.Event(
        enable_timing=False
    )

    input_ready.record(
        torch.cuda.current_stream()
    )

    launched = False

    try:
        if contender is not None:
            contender.launch()
            launched = True

        with torch.cuda.stream(
            model_stream
        ):
            model_stream.wait_event(
                input_ready
            )

            start = (
                tv_online.record_event()
            )

            with (
                torch.no_grad(),
                torch.amp.autocast(
                    "cuda",
                    enabled=amp_enabled,
                ),
            ):
                prefix = (
                    tv_online
                    .extract_fixed_layer1_prefix(
                        backbone,
                        frame,
                    )
                )

                state = {
                    "left_l1":
                        prefix[
                            "left_l1"
                        ],

                    "right_l1":
                        prefix[
                            "right_l1"
                        ],
                }

                elastic_started = False

                # ----------------------------------------------
                # Res2
                # ----------------------------------------------

                w = schedule[0]

                if (
                    not elastic_started
                    and
                    w >= 1.0
                ):
                    (
                        state[
                            "left_l2"
                        ],
                        state[
                            "right_l2"
                        ],
                    ) = (
                        tv_online
                        ._native_res_stage(
                            backbone
                            .feature_backbone,
                            "layer2",
                            state[
                                "left_l1"
                            ],
                            state[
                                "right_l1"
                            ],
                        )
                    )

                else:
                    elastic_started = True

                    ctx = (
                        tv_online
                        .begin_stage_prefix(
                            branch,
                            bank,
                            0,
                            schedule[:1],
                            allow_prepare=False,
                        )
                    )

                    state = (
                        branch.stage_res2(
                            prefix,
                            w,
                        )
                    )

                    tv_online\
                        .finish_stage_prefix(
                            branch,
                            ctx,
                            allow_prepare=False,
                        )

                # ----------------------------------------------
                # Res3
                # ----------------------------------------------

                w = schedule[1]

                if (
                    not elastic_started
                    and
                    w >= 1.0
                ):
                    (
                        state[
                            "left_l3"
                        ],
                        state[
                            "right_l3"
                        ],
                    ) = (
                        tv_online
                        ._native_res_stage(
                            backbone
                            .feature_backbone,
                            "layer3",
                            state[
                                "left_l2"
                            ],
                            state[
                                "right_l2"
                            ],
                        )
                    )

                else:
                    elastic_started = True

                    ctx = (
                        tv_online
                        .begin_stage_prefix(
                            branch,
                            bank,
                            1,
                            schedule[:2],
                            allow_prepare=False,
                        )
                    )

                    state = (
                        branch.stage_res3(
                            state,
                            w,
                        )
                    )

                    tv_online\
                        .finish_stage_prefix(
                            branch,
                            ctx,
                            allow_prepare=False,
                        )

                # ----------------------------------------------
                # Res4
                # ----------------------------------------------

                w = schedule[2]

                if (
                    not elastic_started
                    and
                    w >= 1.0
                ):
                    (
                        state[
                            "left_l4"
                        ],
                        state[
                            "right_l4"
                        ],
                    ) = (
                        tv_online
                        ._native_res_stage(
                            backbone
                            .feature_backbone,
                            "layer4",
                            state[
                                "left_l3"
                            ],
                            state[
                                "right_l3"
                            ],
                        )
                    )

                else:
                    elastic_started = True

                    ctx = (
                        tv_online
                        .begin_stage_prefix(
                            branch,
                            bank,
                            2,
                            schedule[:3],
                            allow_prepare=False,
                        )
                    )

                    state = (
                        branch.stage_res4(
                            state,
                            w,
                        )
                    )

                    tv_online\
                        .finish_stage_prefix(
                            branch,
                            ctx,
                            allow_prepare=False,
                        )

                # ----------------------------------------------
                # FPN
                # ----------------------------------------------

                w = schedule[3]

                if (
                    not elastic_started
                    and
                    w >= 1.0
                ):
                    state = (
                        tv_online
                        ._native_fpn(
                            backbone,
                            frame,
                            state,
                        )
                    )

                else:
                    elastic_started = True

                    ctx = (
                        tv_online
                        .begin_stage_prefix(
                            branch,
                            bank,
                            3,
                            schedule[:4],
                            allow_prepare=False,
                        )
                    )

                    state = (
                        branch.stage_fpn(
                            frame,
                            state,
                            w,
                        )
                    )

                    tv_online\
                        .finish_stage_prefix(
                            branch,
                            ctx,
                            allow_prepare=False,
                        )

                # ----------------------------------------------
                # Stereo
                # ----------------------------------------------

                w = schedule[4]

                if (
                    not elastic_started
                    and
                    w >= 1.0
                ):
                    stereo = (
                        tv_online
                        ._native_stereo(
                            backbone,
                            frame,
                            state,
                        )
                    )

                else:
                    elastic_started = True

                    ctx = (
                        tv_online
                        .begin_stage_prefix(
                            branch,
                            bank,
                            4,
                            schedule[:5],
                            allow_prepare=False,
                        )
                    )

                    stereo = (
                        branch.stage_stereo(
                            frame,
                            state,
                            backbone,
                            w,
                        )
                    )

                    tv_online\
                        .finish_stage_prefix(
                            branch,
                            ctx,
                            allow_prepare=False,
                        )

                # ----------------------------------------------
                # RPN
                # ----------------------------------------------

                w = schedule[5]

                if (
                    not elastic_started
                    and
                    w >= 1.0
                ):
                    (
                        bev,
                        valids,
                    ) = (
                        tv_online
                        .native_full_rpn(
                            branch,
                            frame,
                            stereo,
                            backbone,
                        )
                    )

                else:
                    elastic_started = True

                    ctx = (
                        tv_online
                        .begin_stage_prefix(
                            branch,
                            bank,
                            5,
                            schedule[:6],
                            allow_prepare=False,
                        )
                    )

                    (
                        bev,
                        valids,
                    ) = (
                        branch.stage_rpn(
                            frame,
                            stereo,
                            backbone,
                            w,
                        )
                    )

                    tv_online\
                        .finish_stage_prefix(
                            branch,
                            ctx,
                            allow_prepare=False,
                        )

                tv_online\
                    .run_forward_tail_only(
                        base_model,
                        batch,
                        bev,
                        valids,
                    )

                end = (
                    tv_online.record_event()
                )

        end.synchronize()

        forward_ms = float(
            tv_online.elapsed_ms(
                start,
                end,
            )
        )

    finally:
        if launched:
            contender.finish()

    return {
        "forward_ms":
            forward_ms,

        "schedule":
            schedule,
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    if args.input_hz <= 0:
        raise ValueError(
            "input_hz must be > 0"
        )

    if not (
        0.0
        <
        args.pressure_fraction
        <
        1.0
    ):
        raise ValueError(
            "invalid pressure fraction"
        )

    np.random.seed(
        args.seed
    )

    torch.manual_seed(
        args.seed
    )

    torch.cuda.manual_seed_all(
        args.seed
    )

    period_ms = (
        1000.0
        /
        args.input_hz
    )

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cfg = make_cfg(
        args.full_cfg
    )

    logger = (
        common_utils.create_logger()
    )

    dataset, _, _ = (
        build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=args.workers,
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
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )

    model.cuda().eval()

    branch = ElasticBEVBranch(
        model.backbone_3d,
        output_bev_channels=int(
            cfg.MODEL
            .MAP_TO_BEV
            .NUM_BEV_FEATURES
        ),
    ).cuda().eval()

    elastic = torch.load(
        args.elastic_ckpt,
        map_location="cpu",
    )

    branch.load_state_dict(
        elastic["branch"],
        strict=True,
    )

    bank = torch.load(
        args.prefix_bn_bank,
        map_location="cpu",
    )

    if (
        bank.get("version")
        !=
        "elastic_v4_bn_causal_prefix_bank_v1"
    ):
        raise RuntimeError(
            "wrong BN bank"
        )

    enable_causal_prefix_fused_cache(
        branch
    )

    controller = (
        TVStream3DController(
            controller_csv=(
                args.controller_csv
            ),
            contention_levels_json=(
                args.levels_json
            ),
            bound_column=(
                "remaining_p99_ms"
            ),
            classifier=(
                "conservative_gap"
            ),
            control_guard_per_boundary_ms=(
                args
                .control_guard_per_boundary_ms
            ),
        )
    )

    levels_data = json.loads(
        Path(
            args.levels_json
        ).read_text()
    )

    contenders = {
        "L0":
            configure_contender(
                levels_data,
                "L0",
            ),

        args.pressure_level:
            configure_contender(
                levels_data,
                args.pressure_level,
            ),
    }

    model_stream = (
        make_high_priority_detector_stream(
            torch.cuda.current_device()
        )
    )

    # ============================================================
    # Deterministic cache prewarm
    # ============================================================

    _, warm_indices = (
        choose_scene_indices(
            dataset,
            max(
                1,
                args.runtime_warmup_frames
                +
                1,
            ),
        )
    )

    warm_batch = load_one(
        dataset,
        warm_indices[0],
    )

    reset_history(
        model
    )

    # ============================================================
    # Exact formal-evaluator causal-prefix initialization
    #
    # PASS 1: deterministic FP32 prewarm
    # ============================================================

    with torch.no_grad():
        pw_fp32 = (
            prewarm_all_causal_prefixes(
                base_model=model,
                branch=branch,
                bank=bank,
                controller=controller,
                batch=warm_batch,
                model_stream=model_stream,
            )
        )

    if (
        pw_fp32.get("expected_prefixes") != 203
        or
        pw_fp32.get("ready_prefixes") != 203
    ):
        raise RuntimeError(
            f"FP32 prewarm incomplete: {pw_fp32}"
        )

    set_forbid_cache_miss(
        branch,
        True,
    )

    cache0 = fused_cache_stats(
        branch
    )

    # ============================================================
    # PASS 2: exact runtime AMP-path prewarm
    #
    # This must NOT create any additional fused cache state.
    # It mirrors eval_tv_stream3d_30hz_random50.py.
    # ============================================================

    with (
        torch.no_grad(),
        torch.amp.autocast(
            "cuda",
            enabled=bool(
                model.use_amp_dict["TEST"]
            ),
        ),
    ):
        pw_amp = (
            prewarm_all_causal_prefixes(
                base_model=model,
                branch=branch,
                bank=bank,
                controller=controller,
                batch=warm_batch,
                model_stream=model_stream,
            )
        )

    cache1 = fused_cache_stats(
        branch
    )

    if cache1 != cache0:
        raise RuntimeError(
            "AMP prewarm changed fused cache: "
            f"{cache0} -> {cache1}"
        )

    if (
        pw_amp.get("expected_prefixes") != 203
        or
        pw_amp.get("ready_prefixes") != 203
    ):
        raise RuntimeError(
            f"AMP prewarm incomplete: {pw_amp}"
        )

    # ============================================================
    # Fixed policy selection
    # ============================================================

    full_pid = (
        find_full_profile(
            controller
        )
    )

    full_schedule = tuple(
        controller.profile_schedule[
            full_pid
        ]
    )

    best_static = (
        select_best_static(
            controller=controller,
            pressure_level=(
                args.pressure_level
            ),
            period_ms=period_ms,
        )
    )

    if args.policy == "full":
        fixed_pid = (
            full_pid
        )

        fixed_schedule = (
            full_schedule
        )

    elif (
        args.policy
        ==
        "best_static"
    ):
        fixed_pid = int(
            best_static[
                "profile_id"
            ]
        )

        fixed_schedule = tuple(
            best_static[
                "schedule"
            ]
        )

    else:
        fixed_pid = None
        fixed_schedule = None

    selection = {
        "policy":
            args.policy,

        "full_profile_id":
            full_pid,

        "full_schedule":
            list(
                full_schedule
            ),

        "best_static":
            builtin(
                best_static
            ),
    }

    (
        out_dir
        /
        "policy_selection.json"
    ).write_text(
        json.dumps(
            selection,
            indent=2,
        )
        +
        "\n"
    )

    print(
        "=" * 100
    )

    print(
        "TV-Stream3D POLICY ABLATION"
    )

    print(
        f"policy          : "
        f"{args.policy}"
    )

    print(
        f"pressure        : "
        f"L0 + "
        f"{args.pressure_level}"
    )

    print(
        f"input Hz        : "
        f"{args.input_hz}"
    )

    print(
        f"period          : "
        f"{period_ms:.6f} ms"
    )

    if fixed_schedule is not None:
        print(
            f"fixed profile   : "
            f"{fixed_pid}"
        )

        print(
            "fixed schedule  : "
            +
            ",".join(
                f"{x:g}"
                for x
                in fixed_schedule
            )
        )

    print(
        "=" * 100
    )

    # ============================================================
    # Match formal evaluator: disable recall before runtime warmup
    # ============================================================

    original_recall = (
        disable_recall(
            model
        )
    )

    # ============================================================
    # Runtime warmup
    # ============================================================

    reset_history(
        model
    )

    last_batch = None

    for wi in range(
        args.runtime_warmup_frames
    ):
        idx = warm_indices[
            wi
            %
            len(
                warm_indices
            )
        ]

        batch = load_one(
            dataset,
            idx,
        )

        true_level = (
            args.pressure_level
            if wi % 2
            else
            "L0"
        )

        contender = contenders[
            true_level
        ]

        if (
            args.policy
            ==
            "tv_dynamic"
        ):
            execute_one(
                base_model=model,
                branch=branch,
                bank=bank,
                controller=controller,
                batch=batch,
                model_stream=model_stream,
                contender=contender,
                deadline_ms=period_ms,
                allow_prepare=False,
            )

        elif (
            args.policy
            ==
            "oracle_level"
        ):
            o = (
                select_oracle_level_profile(
                    controller=controller,
                    true_level=(
                        true_level
                    ),
                    budget_ms=period_ms,
                )
            )

            execute_fixed_schedule(
                base_model=model,
                branch=branch,
                bank=bank,
                batch=batch,
                model_stream=model_stream,
                contender=contender,
                schedule=o[
                    "schedule"
                ],
            )

        else:
            execute_fixed_schedule(
                base_model=model,
                branch=branch,
                bank=bank,
                batch=batch,
                model_stream=model_stream,
                contender=contender,
                schedule=(
                    fixed_schedule
                ),
            )

        last_batch = batch

    if last_batch is not None:
        make_prediction(
            model,
            dataset,
            last_batch,
        )

    reset_history(
        model
    )

    torch.cuda.synchronize()

    cache2 = fused_cache_stats(
        branch
    )

    if cache2 != cache1:
        raise RuntimeError(
            "runtime warmup changed fused cache: "
            f"{cache1} -> {cache2}"
        )

    # ============================================================
    # Formal Random50 trace
    # ============================================================

    total = len(
        dataset
    )

    eval_count = (
        total
        if args.max_frames == 0
        else min(
            total,
            args.max_frames,
        )
    )

    eval_indices = list(
        range(
            eval_count
        )
    )

    groups = scene_groups(
        dataset,
        eval_indices,
    )

    (
        trace_by_idx,
        trace_scene_summary,
    ) = build_balanced_trace(
        groups=groups,
        pressure_level=(
            args.pressure_level
        ),
        fraction=(
            args.pressure_fraction
        ),
        seed=(
            args.trace_seed
        ),
    )

    trace_csv = (
        out_dir
        /
        "contention_trace.csv"
    )

    with trace_csv.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "global_index",
                "scene",
                "local_pos",
                "frame_id",
                "true_level",
            ],
        )

        w.writeheader()

        for scene, indices in (
            groups.items()
        ):
            for local_pos, idx in (
                enumerate(indices)
            ):
                (
                    _,
                    frame_id,
                    _,
                ) = frame_meta(
                    dataset,
                    idx,
                )

                w.writerow({
                    "global_index":
                        idx,

                    "scene":
                        scene,

                    "local_pos":
                        local_pos,

                    "frame_id":
                        frame_id,

                    "true_level":
                        trace_by_idx[
                            idx
                        ],
                })

    rows = {}
    events = []

    fwd_times = []
    wait_times = []
    response_times = []
    controller_times = []

    schedule_hist = Counter()
    profile_hist = Counter()
    observed_hist = Counter()

    processed = 0
    dropped = 0
    misses = 0

    try:
        for scene_i, (
            scene,
            indices,
        ) in enumerate(
            groups.items(),
            start=1,
        ):
            reset_history(
                model
            )

            pos = 0
            gpu_free_ms = 0.0
            n = len(
                indices
            )

            print(
                f"[scene "
                f"{scene_i:02d}/"
                f"{len(groups):02d}] "
                f"{scene}: {n}"
            )

            while pos < n:
                idx = indices[
                    pos
                ]

                (
                    meta_scene,
                    frame_id,
                    next_frame_id,
                ) = frame_meta(
                    dataset,
                    idx,
                )

                if (
                    meta_scene
                    !=
                    scene
                ):
                    raise RuntimeError(
                        "scene mismatch"
                    )

                arrival_ms = (
                    pos
                    *
                    period_ms
                )

                deadline_ms = (
                    arrival_ms
                    +
                    period_ms
                )

                start_ms = max(
                    gpu_free_ms,
                    arrival_ms,
                )

                budget_ms = (
                    deadline_ms
                    -
                    start_ms
                )

                runtime_budget = max(
                    budget_ms,
                    1e-6,
                )

                true_level = (
                    trace_by_idx[
                        idx
                    ]
                )

                contender = (
                    contenders[
                        true_level
                    ]
                )

                batch = load_one(
                    dataset,
                    idx,
                )

                observed = ""
                controller_ms = 0.0
                selected_pid = None

                if (
                    args.policy
                    ==
                    "tv_dynamic"
                ):
                    result = execute_one(
                        base_model=model,
                        branch=branch,
                        bank=bank,
                        controller=controller,
                        batch=batch,
                        model_stream=model_stream,
                        contender=contender,
                        deadline_ms=(
                            runtime_budget
                        ),
                        allow_prepare=False,
                    )

                    forward_ms = float(
                        result[
                            "forward_ms"
                        ]
                    )

                    schedule = tuple(
                        float(x)
                        for x
                        in result[
                            "schedule"
                        ]
                    )

                    observed = str(
                        result[
                            "observed_level"
                        ]
                    )

                    controller_ms = float(
                        result[
                            "controller_ms"
                        ]
                    )

                    # Final realized schedule must be one
                    # of the legal 84 profiles.
                    matches = [
                        pid
                        for pid
                        in controller.profile_ids
                        if tuple(
                            controller
                            .profile_schedule[
                                pid
                            ]
                        ) == schedule
                    ]

                    if len(matches) != 1:
                        raise RuntimeError(
                            "dynamic schedule does "
                            "not map uniquely to "
                            "one legal profile: "
                            f"{schedule}"
                        )

                    selected_pid = int(
                        matches[0]
                    )

                elif (
                    args.policy
                    ==
                    "oracle_level"
                ):
                    o = (
                        select_oracle_level_profile(
                            controller=controller,
                            true_level=(
                                true_level
                            ),
                            budget_ms=(
                                runtime_budget
                            ),
                        )
                    )

                    selected_pid = int(
                        o[
                            "profile_id"
                        ]
                    )

                    schedule = tuple(
                        o[
                            "schedule"
                        ]
                    )

                    result = (
                        execute_fixed_schedule(
                            base_model=model,
                            branch=branch,
                            bank=bank,
                            batch=batch,
                            model_stream=model_stream,
                            contender=contender,
                            schedule=schedule,
                        )
                    )

                    forward_ms = float(
                        result[
                            "forward_ms"
                        ]
                    )

                    observed = (
                        true_level
                    )

                else:
                    selected_pid = int(
                        fixed_pid
                    )

                    schedule = tuple(
                        fixed_schedule
                    )

                    result = (
                        execute_fixed_schedule(
                            base_model=model,
                            branch=branch,
                            bank=bank,
                            batch=batch,
                            model_stream=model_stream,
                            contender=contender,
                            schedule=schedule,
                        )
                    )

                    forward_ms = float(
                        result[
                            "forward_ms"
                        ]
                    )

                finish_ms = (
                    start_ms
                    +
                    forward_ms
                )

                wait_ms = (
                    start_ms
                    -
                    arrival_ms
                )

                response_ms = (
                    finish_ms
                    -
                    arrival_ms
                )

                slack_ms = (
                    deadline_ms
                    -
                    finish_ms
                )

                miss = (
                    finish_ms
                    >
                    deadline_ms
                    +
                    1e-9
                )

                pred_anno = (
                    make_prediction(
                        model,
                        dataset,
                        batch,
                    )
                )

                schedule_s = ",".join(
                    f"{x:g}"
                    for x
                    in schedule
                )

                rows[
                    idx
                ] = {
                    "global_index":
                        idx,

                    "scene":
                        scene,

                    "local_pos":
                        pos,

                    "frame_id":
                        frame_id,

                    "arrival_ms":
                        arrival_ms,

                    "deadline_ms":
                        deadline_ms,

                    "status":
                        "processed",

                    "forward_start_ms":
                        start_ms,

                    "forward_finish_ms":
                        finish_ms,

                    "initial_budget_ms":
                        budget_ms,

                    "forward_ms":
                        forward_ms,

                    "queue_wait_ms":
                        wait_ms,

                    "response_ms":
                        response_ms,

                    "deadline_slack_ms":
                        slack_ms,

                    "deadline_miss":
                        int(
                            miss
                        ),

                    "true_level":
                        true_level,

                    "observed_level":
                        observed,

                    "profile_id":
                        selected_pid,

                    "schedule":
                        schedule_s,

                    "controller_ms":
                        controller_ms,
                }

                events.append({
                    "scene":
                        scene,

                    "source_index":
                        idx,

                    "source_frame_id":
                        frame_id,

                    "finish_ms":
                        finish_ms,

                    "anno":
                        pred_anno,
                })

                processed += 1

                misses += int(
                    miss
                )

                fwd_times.append(
                    forward_ms
                )

                wait_times.append(
                    wait_ms
                )

                response_times.append(
                    response_ms
                )

                controller_times.append(
                    controller_ms
                )

                schedule_hist[
                    schedule_s
                ] += 1

                profile_hist[
                    str(
                        selected_pid
                    )
                ] += 1

                if observed:
                    observed_hist[
                        observed
                    ] += 1

                # ================================================
                # Capacity-1 latest-frame mailbox
                # ================================================

                latest_arrived = min(
                    n - 1,
                    int(
                        np.floor(
                            (
                                finish_ms
                                +
                                1e-9
                            )
                            /
                            period_ms
                        )
                    ),
                )

                if (
                    latest_arrived
                    >=
                    pos + 1
                ):
                    next_pos = (
                        latest_arrived
                    )

                    for dp in range(
                        pos + 1,
                        next_pos,
                    ):
                        didx = (
                            indices[
                                dp
                            ]
                        )

                        (
                            _,
                            dfid,
                            _,
                        ) = frame_meta(
                            dataset,
                            didx,
                        )

                        rows[
                            didx
                        ] = {
                            "global_index":
                                didx,

                            "scene":
                                scene,

                            "local_pos":
                                dp,

                            "frame_id":
                                dfid,

                            "arrival_ms":
                                dp
                                *
                                period_ms,

                            "deadline_ms":
                                (
                                    dp
                                    +
                                    1
                                )
                                *
                                period_ms,

                            "status":
                                "dropped",

                            "forward_start_ms":
                                "",

                            "forward_finish_ms":
                                "",

                            "initial_budget_ms":
                                "",

                            "forward_ms":
                                "",

                            "queue_wait_ms":
                                "",

                            "response_ms":
                                "",

                            "deadline_slack_ms":
                                "",

                            "deadline_miss":
                                "",

                            "true_level":
                                trace_by_idx[
                                    didx
                                ],

                            "observed_level":
                                "",

                            "profile_id":
                                "",

                            "schedule":
                                "",

                            "controller_ms":
                                "",
                        }

                        dropped += 1

                    pos = (
                        next_pos
                    )

                    gpu_free_ms = (
                        finish_ms
                    )

                else:
                    pos += 1
                    gpu_free_ms = (
                        finish_ms
                    )

                if (
                    processed <= 3
                    or
                    processed % 100 == 0
                ):
                    print(
                        f"[{args.policy}] "
                        f"processed="
                        f"{processed:04d} "
                        f"true={true_level} "
                        f"fwd="
                        f"{forward_ms:.3f} "
                        f"miss="
                        f"{int(miss)} "
                        f"pid="
                        f"{selected_pid}"
                    )

        if (
            processed
            +
            dropped
            !=
            eval_count
        ):
            raise RuntimeError(
                "frame accounting mismatch"
            )

        # ========================================================
        # Streaming AP
        # ========================================================

        events_by_scene = (
            OrderedDict(
                (
                    scene,
                    [],
                )
                for scene
                in groups
            )
        )

        for e in events:
            events_by_scene[
                e[
                    "scene"
                ]
            ].append(
                e
            )

        for scene in (
            events_by_scene
        ):
            events_by_scene[
                scene
            ].sort(
                key=lambda e:
                    e[
                        "finish_ms"
                    ]
            )

        aligned = {}

        for scene, indices in (
            groups.items()
        ):
            es = (
                events_by_scene[
                    scene
                ]
            )

            ptr = 0
            latest = None

            for pos, idx in (
                enumerate(
                    indices
                )
            ):
                query_ms = (
                    pos
                    *
                    period_ms
                )

                while (
                    ptr
                    <
                    len(es)
                    and
                    es[
                        ptr
                    ][
                        "finish_ms"
                    ]
                    <=
                    query_ms
                    +
                    1e-9
                ):
                    latest = (
                        es[
                            ptr
                        ][
                            "anno"
                        ]
                    )

                    ptr += 1

                (
                    _,
                    fid,
                    next_fid,
                ) = frame_meta(
                    dataset,
                    idx,
                )

                aligned[
                    idx
                ] = copy_det(
                    latest,
                    scene,
                    fid,
                    next_fid,
                )

        gt_annos = [
            copy.deepcopy(
                dataset.kitti_infos[
                    idx
                ][
                    "infos"
                ][
                    "token"
                ][
                    "annos"
                ]
            )
            for idx
            in eval_indices
        ]

        det_annos = [
            aligned[
                idx
            ]
            for idx
            in eval_indices
        ]

        (
            result_str,
            ap_dict,
        ) = (
            kitti_eval
            .get_official_eval_result(
                gt_annos,
                det_annos,
                dataset.class_names,
            )
        )

        car = float(
            ap_dict.get(
                "Car_3d/moderate_R40",
                np.nan,
            )
        )

        ped = float(
            ap_dict.get(
                "Pedestrian_3d/moderate_R40",
                np.nan,
            )
        )

        cyc = float(
            ap_dict.get(
                "Cyclist_3d/moderate_R40",
                np.nan,
            )
        )

        macro = float(
            np.nanmean(
                [
                    car,
                    ped,
                    cyc,
                ]
            )
        )

        # ========================================================
        # Save
        # ========================================================

        timeline = (
            out_dir
            /
            "frame_timeline.csv"
        )

        with timeline.open(
            "w",
            newline="",
        ) as f:
            fieldnames = list(
                next(
                    iter(
                        rows.values()
                    )
                ).keys()
            )

            w = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            w.writeheader()

            for idx in (
                eval_indices
            ):
                w.writerow(
                    rows[
                        idx
                    ]
                )

        with (
            out_dir
            /
            "stream_det_annos.pkl"
        ).open(
            "wb"
        ) as f:
            pickle.dump(
                det_annos,
                f,
                protocol=(
                    pickle
                    .HIGHEST_PROTOCOL
                ),
            )

        (
            out_dir
            /
            "stream_sap_result.txt"
        ).write_text(
            result_str
        )

        (
            out_dir
            /
            "stream_sap_dict.json"
        ).write_text(
            json.dumps(
                builtin(
                    ap_dict
                ),
                indent=2,
            )
            +
            "\n"
        )

        summary = {
            "version":
                "tv_static_dynamic_ablation_v1",

            "policy":
                args.policy,

            "pressure_level":
                args.pressure_level,

            "pressure_fraction":
                args.pressure_fraction,

            "trace_seed":
                args.trace_seed,

            "input_hz":
                args.input_hz,

            "period_ms":
                period_ms,

            "timing_scope":
                "forward_only",

            "buffer_policy":
                "latest_frame_only",

            "history_policy":
                "processed_frames_only",

            "sensor_frames":
                eval_count,

            "processed_frames":
                processed,

            "dropped_frames":
                dropped,

            "drop_rate":
                (
                    dropped
                    /
                    eval_count
                ),

            "deadline_miss_count":
                misses,

            "deadline_miss_rate":
                (
                    misses
                    /
                    processed
                ),

            "forward_latency":
                stats(
                    fwd_times
                ),

            "queue_wait":
                stats(
                    wait_times
                ),

            "arrival_to_finish":
                stats(
                    response_times
                ),

            "controller_cpu_ms":
                stats(
                    controller_times
                ),

            "schedule_histogram":
                dict(
                    schedule_hist
                ),

            "profile_histogram":
                dict(
                    profile_hist
                ),

            "observed_levels":
                dict(
                    observed_hist
                ),

            "policy_selection":
                selection,

            "stream_sap_3d_moderate_R40": {
                "Car":
                    car,

                "Pedestrian":
                    ped,

                "Cyclist":
                    cyc,

                "Macro":
                    macro,
            },

            "contention_trace_csv":
                str(
                    trace_csv
                ),

            "cache":
                builtin(
                    fused_cache_stats(
                        branch
                    )
                ),
        }

        summary_path = (
            out_dir
            /
            "summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            )
            +
            "\n"
        )

        print()
        print(
            "=" * 100
        )

        print(
            f"{args.policy} | "
            f"L0+"
            f"{args.pressure_level}"
        )

        print(
            f"Macro     : "
            f"{macro:.4f}"
        )

        print(
            f"Miss      : "
            f"{100*misses/processed:.3f}%"
        )

        print(
            f"Drop      : "
            f"{100*dropped/eval_count:.3f}%"
        )

        fs = stats(
            fwd_times
        )

        print(
            "p50/p90/p99: "
            f"{fs['p50_ms']:.3f}/"
            f"{fs['p90_ms']:.3f}/"
            f"{fs['p99_ms']:.3f} ms"
        )

        print(
            f"summary   : "
            f"{summary_path}"
        )

        print(
            "=" * 100
        )

    finally:
        model.generate_recall_record = (
            original_recall
        )


if __name__ == "__main__":
    main()
