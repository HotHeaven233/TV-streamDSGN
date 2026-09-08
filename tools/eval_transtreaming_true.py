#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from collections import Counter, OrderedDict, deque
from pathlib import Path

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from pcdet.datasets.kitti.kitti_object_eval_python import (
    eval as kitti_eval,
)

from test_stream_buffer_timestamp import (
    load_one,
)

from test_tv_stream3d_online_forward import (
    make_cfg,
    choose_scene_indices,
    configure_contender,
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
)

from transtreaming_adaptive_runtime import (
    TranstreamingAdaptivePlanner,
    reset_model_history,
    transtreaming_forward_no_post,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cfg",
        required=True,
    )

    p.add_argument(
        "--ckpt",
        required=True,
    )

    p.add_argument(
        "--levels_json",
        required=True,
    )

    p.add_argument(
        "--pressure_level",
        default="L0",
        choices=[
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
        ],
    )

    p.add_argument(
        "--pressure_fraction",
        type=float,
        default=0.0,
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
        "--past_length",
        type=int,
        default=3,
    )

    p.add_argument(
        "--future_length",
        type=int,
        default=4,
    )

    p.add_argument(
        "--max_future",
        type=int,
        default=8,
    )

    p.add_argument(
        "--planner_window",
        type=int,
        default=5,
    )

    p.add_argument(
        "--disable_late_skip",
        action="store_true",
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    return p.parse_args()


def branch_prediction(
    model,
    dataset,
    batch,
    branch_data,
):
    original = (
        batch["token"]
    )

    batch["token"] = (
        branch_data
    )

    try:
        pred = make_prediction(
            model,
            dataset,
            batch,
        )
    finally:
        batch[
            "token"
        ] = original

    return pred


def build_trace(
    groups,
    pressure_level,
    pressure_fraction,
    trace_seed,
):
    if (
        pressure_level == "L0"
        or
        pressure_fraction <= 0.0
    ):
        trace = {}
        summary = {}

        for scene, indices in (
            groups.items()
        ):
            levels = []

            for idx in indices:
                trace[idx] = "L0"
                levels.append(
                    "L0"
                )

            summary[
                scene
            ] = {
                "sensor_frames":
                    int(
                        len(indices)
                    ),

                "pressure_frames":
                    0,

                "l0_frames":
                    int(
                        len(indices)
                    ),

                "pressure_fraction_realized":
                    0.0,

                "switches":
                    0,

                "scene_seed":
                    None,
            }

        return trace, summary

    return build_balanced_trace(
        groups=groups,
        pressure_level=(
            pressure_level
        ),
        fraction=(
            pressure_fraction
        ),
        seed=trace_seed,
    )


def write_trace(
    path,
    dataset,
    groups,
    trace_by_idx,
):
    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_index",
                "scene",
                "local_pos",
                "frame_id",
                "true_level",
            ],
        )

        writer.writeheader()

        for scene, indices in (
            groups.items()
        ):
            for local_pos, idx in (
                enumerate(indices)
            ):
                _, frame_id, _ = (
                    frame_meta(
                        dataset,
                        idx,
                    )
                )

                writer.writerow({
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


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    if args.input_hz <= 0:
        raise ValueError(
            "input_hz must be >0"
        )

    if args.past_length <= 0:
        raise ValueError(
            "past_length must be >0"
        )

    if args.future_length <= 0:
        raise ValueError(
            "future_length must be >0"
        )

    if args.max_future <= 0:
        raise ValueError(
            "max_future must be >0"
        )

    no_load = (
        args.pressure_level
        ==
        "L0"
    )

    if no_load:
        if (
            abs(
                args.pressure_fraction
            )
            >
            1e-12
        ):
            raise ValueError(
                "L0 requires "
                "pressure_fraction=0"
            )
    else:
        if not (
            0.0
            <
            args.pressure_fraction
            <
            1.0
        ):
            raise ValueError(
                "random contention requires "
                "0<f<1"
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

    # ==================================================================
    # Dataset / model
    # ==================================================================

    cfg = make_cfg(
        args.cfg
    )

    logger = (
        common_utils
        .create_logger()
    )

    dataset, _, _ = (
        build_dataloader(
            dataset_cfg=(
                cfg.DATA_CONFIG
            ),
            class_names=(
                cfg.CLASS_NAMES
            ),
            batch_size=1,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=False,
        )
    )

    model = build_network(
        model_cfg=(
            cfg.MODEL
        ),
        num_class=len(
            cfg.CLASS_NAMES
        ),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=True,
    )

    model.cuda().eval()

    if (
        type(model).__name__
        !=
        "TRANSTREAMING_STREAM"
    ):
        raise RuntimeError(
            "wrong detector: "
            f"{type(model).__name__}"
        )

    fusion_names = [
        type(x).__name__
        for x in (
            model.fusion_module
        )
    ]

    if (
        "TranstreamingBEVTAT"
        not in
        fusion_names
    ):
        raise RuntimeError(
            "TranstreamingBEVTAT "
            f"missing: {fusion_names}"
        )

    logger.info(
        "=" * 80
    )

    logger.info(
        "TRUE Transtreaming-style "
        "StreamDSGN"
    )

    logger.info(
        f"checkpoint = {args.ckpt}"
    )

    logger.info(
        "dynamic actual P^P"
    )

    logger.info(
        "dynamic adaptive P^F"
    )

    logger.info(
        "ONE TAT + ONE shared detection path"
    )

    logger.info(
        "=" * 80
    )

    # ==================================================================
    # Contention
    # ==================================================================

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
            )
    }

    if not no_load:
        contenders[
            args.pressure_level
        ] = (
            configure_contender(
                levels_data,
                args.pressure_level,
            )
        )

    model_stream = (
        make_high_priority_detector_stream(
            torch.cuda.current_device()
        )
    )

    original_recall = (
        disable_recall(
            model
        )
    )

    # ==================================================================
    # CUDA warmup
    #
    # Warm:
    #   TAT with four future queries
    #   shared detection path four times
    #
    # Warmup runtime never updates formal planner.
    # ==================================================================

    _, warm_indices = (
        choose_scene_indices(
            dataset,
            max(
                2,
                args.runtime_warmup_frames
                + 1,
            ),
        )
    )

    reset_model_history(
        model
    )

    warm_positions = deque(
        maxlen=(
            args.past_length
        )
    )

    last_batch = None
    last_result = None

    for wi in range(
        args.runtime_warmup_frames
    ):
        local_pos = (
            wi
            %
            len(warm_indices)
        )

        if (
            local_pos == 0
            and
            wi > 0
        ):
            reset_model_history(
                model
            )

            warm_positions.clear()

        idx = warm_indices[
            local_pos
        ]

        pp = tuple(
            int(x - local_pos)
            for x in (
                warm_positions
            )
        )

        batch = load_one(
            dataset,
            idx,
        )

        if no_load:
            warm_level = "L0"
        else:
            warm_level = (
                args.pressure_level
                if wi % 2
                else
                "L0"
            )

        result = (
            transtreaming_forward_no_post(
                model=model,
                batch=batch,

                past_offsets=pp,

                # Max-shape warmup.
                future_offsets=(
                    1,
                    2,
                    4,
                    8,
                ),

                model_stream=(
                    model_stream
                ),

                contender=(
                    contenders[
                        warm_level
                    ]
                ),

                source_local_pos=(
                    local_pos
                ),

                job_start_ms=(
                    local_pos
                    *
                    period_ms
                ),

                period_ms=(
                    period_ms
                ),

                estimated_bn_ms=(
                    0.5
                    *
                    period_ms
                ),

                estimated_head_ms=(
                    0.5
                    *
                    period_ms
                ),

                enable_late_skip=False,
            )
        )

        warm_positions.append(
            local_pos
        )

        last_batch = batch
        last_result = result

    # Warm post/NMS outside timed path.
    if (
        last_batch is not None
        and
        last_result is not None
    ):
        for horizon in (
            last_result
            .executed_pf
        ):
            branch_prediction(
                model,
                dataset,
                last_batch,
                last_result
                .branch_data[
                    horizon
                ],
            )

    reset_model_history(
        model
    )

    planner = (
        TranstreamingAdaptivePlanner(
            period_ms=(
                period_ms
            ),
            future_length=(
                args.future_length
            ),
            max_future=(
                args.max_future
            ),
            window_size=(
                args.planner_window
            ),
        )
    )

    # Warmup timing must not leak.
    planner.reset()

    torch.cuda.synchronize()

    # ==================================================================
    # Evaluation indices / frozen trace
    # ==================================================================

    total_dataset = len(
        dataset
    )

    eval_count = (
        total_dataset
        if args.max_frames == 0
        else min(
            total_dataset,
            args.max_frames,
        )
    )

    eval_indices = list(
        range(eval_count)
    )

    groups = scene_groups(
        dataset,
        eval_indices,
    )

    (
        trace_by_idx,
        trace_scene_summary,
    ) = build_trace(
        groups=groups,
        pressure_level=(
            args.pressure_level
        ),
        pressure_fraction=(
            args.pressure_fraction
        ),
        trace_seed=(
            args.trace_seed
        ),
    )

    trace_csv = (
        out_dir
        /
        "contention_trace.csv"
    )

    write_trace(
        trace_csv,
        dataset,
        groups,
        trace_by_idx,
    )

    sensor_level_hist = Counter(
        trace_by_idx[idx]
        for idx in eval_indices
    )

    print()
    print("=" * 110)
    print(
        "TRUE TRANSTREAMING-STYLE "
        "STREAMDSGN"
    )
    print("=" * 110)

    print(
        f"Hz                  : "
        f"{args.input_hz}"
    )

    print(
        f"period              : "
        f"{period_ms:.6f} ms"
    )

    print(
        "timing              : "
        "backbone + TAT + "
        "executed future detection paths"
    )

    print(
        "history             : "
        "actual processed-frame P^P"
    )

    print(
        "future              : "
        f"dynamic P^F, max={args.max_future}"
    )

    print(
        "future outputs/job  : "
        f"max={args.future_length}"
    )

    print(
        "late-output skip    : "
        f"{not args.disable_late_skip}"
    )

    print(
        "output dispatch     : "
        "closest temporal target "
        "among released outputs"
    )

    print(
        f"sensor frames       : "
        f"{eval_count}"
    )

    print(
        f"sensor levels       : "
        f"{dict(sensor_level_hist)}"
    )

    print("=" * 110)

    # ==================================================================
    # Formal stream
    # ==================================================================

    timeline = {}

    prediction_events = []

    decision_rows = []
    output_event_rows = []

    forward_times = []
    bn_times = []
    head_times_all = []
    queue_wait_times = []
    response_times = []
    first_output_response = []

    processed_level_hist = Counter()
    proposal_hist = Counter()
    past_hist = Counter()
    dispatch_horizon_hist = Counter()

    processed = 0
    dropped = 0
    misses = 0

    total_skipped_outputs = 0
    total_executed_outputs = 0

    try:
        for scene_i, (
            scene,
            indices,
        ) in enumerate(
            groups.items(),
            start=1,
        ):
            reset_model_history(
                model
            )

            # Actual processed sensor positions corresponding
            # to the most recent historical features used by TAT.
            history_positions = deque(
                maxlen=(
                    args.past_length
                )
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
                idx = (
                    indices[
                        pos
                    ]
                )

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

                queue_wait_ms = (
                    start_ms
                    -
                    arrival_ms
                )

                true_level = (
                    trace_by_idx[
                        idx
                    ]
                )

                # ==================================================
                # TRUE actual historical temporal positions.
                #
                # Example:
                #
                # processed:
                #   0, 1, 3, 6
                #
                # current=6
                #
                # P^P:
                #   [-6, -5, -3]
                # ==================================================

                pp = tuple(
                    int(
                        history_pos
                        -
                        pos
                    )
                    for history_pos in (
                        history_positions
                    )
                )

                plan = planner.plan(
                    queue_wait_ms=(
                        queue_wait_ms
                    ),
                    past_offsets=(
                        pp
                    ),
                )

                pf = (
                    plan.future_offsets
                )

                batch = load_one(
                    dataset,
                    idx,
                )

                result = (
                    transtreaming_forward_no_post(
                        model=model,
                        batch=batch,
                        past_offsets=(
                            pp
                        ),
                        future_offsets=(
                            pf
                        ),
                        model_stream=(
                            model_stream
                        ),
                        contender=(
                            contenders[
                                true_level
                            ]
                        ),

                        source_local_pos=(
                            pos
                        ),

                        job_start_ms=(
                            start_ms
                        ),

                        period_ms=(
                            period_ms
                        ),

                        estimated_bn_ms=(
                            plan
                            .estimated_bn_ms
                        ),

                        estimated_head_ms=(
                            plan
                            .estimated_head_ms
                        ),

                        enable_late_skip=(
                            not
                            args
                            .disable_late_skip
                        ),
                    )
                )

                # Current runtime becomes planner history
                # only after current routing/execution.
                planner.observe(
                    bn_ms=(
                        result.bn_ms
                    ),
                    head_times_ms=(
                        result
                        .head_ms
                        .values()
                    ),
                )

                finish_ms = (
                    start_ms
                    +
                    result.total_ms
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

                miss = bool(
                    finish_ms
                    >
                    deadline_ms
                    +
                    1e-9
                )

                # ==================================================
                # Postprocessing outside timing.
                # ==================================================

                pred_by_horizon = {}

                for horizon in (
                    result
                    .executed_pf
                ):
                    pred_by_horizon[
                        horizon
                    ] = (
                        branch_prediction(
                            model,
                            dataset,
                            batch,
                            result
                            .branch_data[
                                horizon
                            ],
                        )
                    )

                first_horizon = (
                    result
                    .executed_pf[
                        0
                    ]
                )

                first_ready_ms = (
                    start_ms
                    +
                    result
                    .ready_offset_ms[
                        first_horizon
                    ]
                )

                # ==================================================
                # Output Buffer
                #
                # Official source:
                #
                # first result -> quick first
                # later results -> held until target time if early
                #
                # ==================================================

                for order, horizon in enumerate(
                    result
                    .executed_pf
                ):
                    target_pos = (
                        pos
                        +
                        horizon
                    )

                    if target_pos >= n:
                        continue

                    target_idx = (
                        indices[
                            target_pos
                        ]
                    )

                    (
                        _,
                        target_frame_id,
                        _,
                    ) = frame_meta(
                        dataset,
                        target_idx,
                    )

                    ready_ms = (
                        start_ms
                        +
                        result
                        .ready_offset_ms[
                            horizon
                        ]
                    )

                    target_sensor_ms = (
                        target_pos
                        *
                        period_ms
                    )

                    if order == 0:
                        # Official "quick first".
                        release_ms = (
                            ready_ms
                        )
                    else:
                        release_ms = max(
                            ready_ms,
                            target_sensor_ms,
                        )

                    event = {
                        "scene":
                            scene,

                        "source_local_pos":
                            pos,

                        "source_index":
                            idx,

                        "source_frame_id":
                            frame_id,

                        "horizon":
                            horizon,

                        "target_local_pos":
                            target_pos,

                        "target_index":
                            target_idx,

                        "target_frame_id":
                            target_frame_id,

                        "compute_ready_ms":
                            ready_ms,

                        "target_sensor_ms":
                            target_sensor_ms,

                        "release_ms":
                            release_ms,

                        "quick_first":
                            int(
                                order == 0
                            ),

                        "anno":
                            pred_by_horizon[
                                horizon
                            ],
                    }

                    prediction_events.append(
                        event
                    )

                    output_event_rows.append({
                        key: value
                        for key, value
                        in event.items()
                        if key != "anno"
                    })

                history_positions.append(
                    pos
                )

                pp_text = "|".join(
                    str(x)
                    for x in pp
                )

                pf_text = "|".join(
                    str(x)
                    for x in pf
                )

                exec_text = "|".join(
                    str(x)
                    for x in (
                        result
                        .executed_pf
                    )
                )

                skip_text = "|".join(
                    str(x)
                    for x in (
                        result
                        .skipped_pf
                    )
                )

                proposal_hist[
                    pf_text
                ] += 1

                past_hist[
                    pp_text
                ] += 1

                total_skipped_outputs += (
                    len(
                        result
                        .skipped_pf
                    )
                )

                total_executed_outputs += (
                    len(
                        result
                        .executed_pf
                    )
                )

                decision_rows.append({
                    "global_index":
                        idx,

                    "scene":
                        scene,

                    "local_pos":
                        pos,

                    "frame_id":
                        frame_id,

                    "true_level":
                        true_level,

                    "arrival_ms":
                        arrival_ms,

                    "queue_wait_ms":
                        queue_wait_ms,

                    "past_offsets":
                        pp_text,

                    "planned_pf":
                        pf_text,

                    "executed_pf":
                        exec_text,

                    "skipped_pf":
                        skip_text,

                    "estimated_bn_ms":
                        plan
                        .estimated_bn_ms,

                    "estimated_head_ms":
                        plan
                        .estimated_head_ms,

                    "estimated_first_ms":
                        plan
                        .estimated_first_ms,

                    "estimated_delta_ms":
                        plan
                        .estimated_delta_ms,

                    "actual_bn_ms":
                        result.bn_ms,

                    "H1_ms":
                        result
                        .head_ms
                        .get(
                            1,
                            "",
                        ),

                    "H2_ms":
                        result
                        .head_ms
                        .get(
                            2,
                            "",
                        ),

                    "H3_ms":
                        result
                        .head_ms
                        .get(
                            3,
                            "",
                        ),

                    "H4_ms":
                        result
                        .head_ms
                        .get(
                            4,
                            "",
                        ),

                    "H5_ms":
                        result
                        .head_ms
                        .get(
                            5,
                            "",
                        ),

                    "H6_ms":
                        result
                        .head_ms
                        .get(
                            6,
                            "",
                        ),

                    "H7_ms":
                        result
                        .head_ms
                        .get(
                            7,
                            "",
                        ),

                    "H8_ms":
                        result
                        .head_ms
                        .get(
                            8,
                            "",
                        ),

                    "total_forward_ms":
                        result.total_ms,

                    "forward_finish_ms":
                        finish_ms,

                    "deadline_miss":
                        int(
                            miss
                        ),
                })

                timeline[
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

                    "absolute_deadline_ms":
                        deadline_ms,

                    "status":
                        "processed",

                    "drop_reason":
                        "",

                    "forward_start_ms":
                        start_ms,

                    "forward_finish_ms":
                        finish_ms,

                    "prediction_timestamp_ms":
                        first_ready_ms,

                    "queue_wait_ms":
                        queue_wait_ms,

                    "forward_ms":
                        result.total_ms,

                    "arrival_to_finish_ms":
                        response_ms,

                    "deadline_slack_ms":
                        slack_ms,

                    "deadline_miss":
                        int(
                            miss
                        ),

                    "true_level":
                        true_level,

                    "past_offsets":
                        pp_text,

                    "planned_pf":
                        pf_text,

                    "executed_pf":
                        exec_text,

                    "skipped_pf":
                        skip_text,
                }

                processed += 1

                misses += int(
                    miss
                )

                forward_times.append(
                    result.total_ms
                )

                bn_times.append(
                    result.bn_ms
                )

                head_times_all.extend(
                    result
                    .head_ms
                    .values()
                )

                queue_wait_times.append(
                    queue_wait_ms
                )

                response_times.append(
                    response_ms
                )

                first_output_response.append(
                    first_ready_ms
                    -
                    arrival_ms
                )

                processed_level_hist[
                    true_level
                ] += 1

                # ==================================================
                # Frozen latest-frame mailbox.
                #
                # Worker remains occupied until all executed
                # future outputs in this job have finished.
                # ==================================================

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
                            dropped_fid,
                            _,
                        ) = frame_meta(
                            dataset,
                            didx,
                        )

                        dropped_arrival = (
                            dp
                            *
                            period_ms
                        )

                        timeline[
                            didx
                        ] = {
                            "global_index":
                                didx,

                            "scene":
                                scene,

                            "local_pos":
                                dp,

                            "frame_id":
                                dropped_fid,

                            "arrival_ms":
                                dropped_arrival,

                            "absolute_deadline_ms":
                                (
                                    dropped_arrival
                                    +
                                    period_ms
                                ),

                            "status":
                                "dropped",

                            "drop_reason":
                                "stale_replaced_by_latest",

                            "forward_start_ms":
                                "",

                            "forward_finish_ms":
                                "",

                            "prediction_timestamp_ms":
                                "",

                            "queue_wait_ms":
                                "",

                            "forward_ms":
                                "",

                            "arrival_to_finish_ms":
                                "",

                            "deadline_slack_ms":
                                "",

                            "deadline_miss":
                                "",

                            "true_level":
                                trace_by_idx[
                                    didx
                                ],

                            "past_offsets":
                                "",

                            "planned_pf":
                                "",

                            "executed_pf":
                                "",

                            "skipped_pf":
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
                    processed <= 5
                    or
                    processed % 100
                    ==
                    0
                ):
                    print(
                        f"[true={true_level}] "
                        f"processed={processed:04d} "
                        f"{scene}/{frame_id} "
                        f"wait={queue_wait_ms:.3f} "
                        f"PP=[{pp_text}] "
                        f"PF=[{pf_text}] "
                        f"exec=[{exec_text}] "
                        f"skip=[{skip_text}] "
                        f"bn={result.bn_ms:.3f} "
                        f"total={result.total_ms:.3f} "
                        f"miss={int(miss)}"
                    )

        # ==================================================================
        # Accounting checks
        # ==================================================================

        missing = [
            idx
            for idx in eval_indices
            if idx not in timeline
        ]

        if missing:
            raise RuntimeError(
                "missing frame statuses: "
                f"{missing[:20]}"
            )

        if (
            processed
            +
            dropped
            !=
            eval_count
        ):
            raise RuntimeError(
                "accounting mismatch: "
                f"{processed}+{dropped}"
                f"!={eval_count}"
            )

        # ==================================================================
        # Output-buffer dispatch
        #
        # At each sensor query, among outputs already released,
        # choose the prediction whose intended temporal target is
        # closest to the current query.
        # ==================================================================

        events_by_scene = (
            OrderedDict(
                (
                    scene,
                    [],
                )
                for scene in groups
            )
        )

        for event in (
            prediction_events
        ):
            events_by_scene[
                event["scene"]
            ].append(
                event
            )

        for scene in (
            events_by_scene
        ):
            events_by_scene[
                scene
            ].sort(
                key=lambda e: (
                    e[
                        "release_ms"
                    ],
                    e[
                        "source_local_pos"
                    ],
                    e[
                        "horizon"
                    ],
                )
            )

        aligned = {}
        dispatch_rows = []

        for scene, indices in (
            groups.items()
        ):
            scene_events = (
                events_by_scene[
                    scene
                ]
            )

            active = []
            ptr = 0

            for query_pos, idx in (
                enumerate(indices)
            ):
                query_ms = (
                    query_pos
                    *
                    period_ms
                )

                while (
                    ptr
                    <
                    len(
                        scene_events
                    )
                    and
                    scene_events[
                        ptr
                    ][
                        "release_ms"
                    ]
                    <=
                    query_ms
                    +
                    1e-9
                ):
                    active.append(
                        scene_events[
                            ptr
                        ]
                    )

                    ptr += 1

                (
                    _,
                    query_fid,
                    query_next_fid,
                ) = frame_meta(
                    dataset,
                    idx,
                )

                if len(active) == 0:
                    chosen = None
                    pred = None

                else:
                    chosen = min(
                        active,
                        key=lambda e: (
                            abs(
                                e[
                                    "target_local_pos"
                                ]
                                -
                                query_pos
                            ),

                            -
                            e[
                                "release_ms"
                            ],

                            -
                            e[
                                "source_local_pos"
                            ],
                        ),
                    )

                    pred = (
                        chosen[
                            "anno"
                        ]
                    )

                aligned[
                    idx
                ] = copy_det(
                    pred,
                    scene,
                    query_fid,
                    query_next_fid,
                )

                if chosen is None:
                    dispatch_rows.append({
                        "scene":
                            scene,

                        "query_local_pos":
                            query_pos,

                        "query_index":
                            idx,

                        "query_frame_id":
                            query_fid,

                        "query_ms":
                            query_ms,

                        "has_prediction":
                            0,

                        "source_local_pos":
                            "",

                        "target_local_pos":
                            "",

                        "horizon":
                            "",

                        "target_error_frames":
                            "",

                        "release_ms":
                            "",
                    })

                else:
                    h = int(
                        chosen[
                            "horizon"
                        ]
                    )

                    dispatch_horizon_hist[
                        h
                    ] += 1

                    dispatch_rows.append({
                        "scene":
                            scene,

                        "query_local_pos":
                            query_pos,

                        "query_index":
                            idx,

                        "query_frame_id":
                            query_fid,

                        "query_ms":
                            query_ms,

                        "has_prediction":
                            1,

                        "source_local_pos":
                            chosen[
                                "source_local_pos"
                            ],

                        "target_local_pos":
                            chosen[
                                "target_local_pos"
                            ],

                        "horizon":
                            h,

                        "target_error_frames":
                            (
                                chosen[
                                    "target_local_pos"
                                ]
                                -
                                query_pos
                            ),

                        "release_ms":
                            chosen[
                                "release_ms"
                            ],
                    })

        # ==================================================================
        # KITTI streaming AP
        # ==================================================================

        gt_annos = [
            copy.deepcopy(
                dataset
                .kitti_infos[
                    idx
                ][
                    "infos"
                ][
                    "token"
                ][
                    "annos"
                ]
            )
            for idx in eval_indices
        ]

        det_annos = [
            aligned[
                idx
            ]
            for idx in eval_indices
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

        # ==================================================================
        # Save audits
        # ==================================================================

        timeline_fields = [
            "global_index",
            "scene",
            "local_pos",
            "frame_id",
            "arrival_ms",
            "absolute_deadline_ms",
            "status",
            "drop_reason",
            "forward_start_ms",
            "forward_finish_ms",
            "prediction_timestamp_ms",
            "queue_wait_ms",
            "forward_ms",
            "arrival_to_finish_ms",
            "deadline_slack_ms",
            "deadline_miss",
            "true_level",
            "past_offsets",
            "planned_pf",
            "executed_pf",
            "skipped_pf",
        ]

        with (
            out_dir
            /
            "frame_timeline.csv"
        ).open(
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(
                f,
                fieldnames=(
                    timeline_fields
                ),
            )

            writer.writeheader()

            for idx in eval_indices:
                writer.writerow(
                    timeline[
                        idx
                    ]
                )

        if decision_rows:
            with (
                out_dir
                /
                "transtreaming_decisions.csv"
            ).open(
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        decision_rows[
                            0
                        ].keys()
                    ),
                )

                writer.writeheader()

                writer.writerows(
                    decision_rows
                )

        if output_event_rows:
            with (
                out_dir
                /
                "transtreaming_output_events.csv"
            ).open(
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        output_event_rows[
                            0
                        ].keys()
                    ),
                )

                writer.writeheader()

                writer.writerows(
                    output_event_rows
                )

        if dispatch_rows:
            with (
                out_dir
                /
                "transtreaming_dispatch.csv"
            ).open(
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        dispatch_rows[
                            0
                        ].keys()
                    ),
                )

                writer.writeheader()

                writer.writerows(
                    dispatch_rows
                )

        with (
            out_dir
            /
            "prediction_events.pkl"
        ).open("wb") as f:
            pickle.dump(
                prediction_events,
                f,
                protocol=(
                    pickle
                    .HIGHEST_PROTOCOL
                ),
            )

        with (
            out_dir
            /
            "stream_det_annos.pkl"
        ).open("wb") as f:
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
                "transtreaming_true_"
                "streamdsgn_v1",

            "method":
                "Transtreaming-style "
                "StreamDSGN",

            "architecture":
                "shared_stereo_backbone"
                "_tat_rtpe_shared_head",

            "model_cfg":
                str(
                    args.cfg
                ),

            "checkpoint":
                str(
                    args.ckpt
                ),

            "input_hz":
                args.input_hz,

            "period_ms":
                period_ms,

            "timing_scope":
                "backbone_tat_rtpe_"
                "executed_shared_heads_"
                "forward_only",

            "past_length":
                args.past_length,

            "future_length":
                args.future_length,

            "max_future":
                args.max_future,

            "late_skip_enabled":
                bool(
                    not
                    args.disable_late_skip
                ),

            "planner": {
                "type":
                    "Transtreaming AdaptiveStrategy",

                "runtime_decay":
                    0.5,

                "t_other_ms":
                    0.0,

                "causal":
                    True,

                "formula":
                    "t_first=wait+t_bn+t_h; "
                    "t_delta=((M+1)t_h+t_bn)/M",

                "warmup_runtime_leak":
                    False,
            },

            "history_policy":
                "actual_processed_frame_offsets",

            "input_buffer_policy":
                "latest_frame_only",

            "output_buffer_policy":
                "quick_first_then_target_release_"
                "closest_temporal_dispatch",

            "contention_trace": {
                "pressure_level":
                    args
                    .pressure_level,

                "pressure_fraction":
                    args
                    .pressure_fraction,

                "trace_seed":
                    args
                    .trace_seed,

                "scene_summary":
                    trace_scene_summary,
            },

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
                    max(
                        processed,
                        1,
                    )
                ),

            "forward_latency":
                stats(
                    forward_times
                ),

            "backbone_neck_latency":
                stats(
                    bn_times
                ),

            "shared_head_latency":
                stats(
                    head_times_all
                ),

            "queue_wait":
                stats(
                    queue_wait_times
                ),

            "arrival_to_finish":
                stats(
                    response_times
                ),

            "arrival_to_first_output":
                stats(
                    first_output_response
                ),

            "proposal_histogram":
                dict(
                    proposal_hist
                ),

            "past_offset_histogram":
                dict(
                    past_hist
                ),

            "dispatch_horizon_histogram":
                {
                    str(k):
                        int(v)
                    for k, v
                    in sorted(
                        dispatch_horizon_hist
                        .items()
                    )
                },

            "executed_future_outputs":
                int(
                    total_executed_outputs
                ),

            "skipped_future_outputs":
                int(
                    total_skipped_outputs
                ),

            "mean_executed_outputs_per_job":
                (
                    total_executed_outputs
                    /
                    max(
                        processed,
                        1,
                    )
                ),

            "skip_rate_over_planned_outputs":
                (
                    total_skipped_outputs
                    /
                    max(
                        total_skipped_outputs
                        +
                        total_executed_outputs,
                        1,
                    )
                ),

            "true_sensor_levels":
                dict(
                    sensor_level_hist
                ),

            "true_processed_levels":
                dict(
                    processed_level_hist
                ),

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

        fs = stats(
            forward_times
        )

        print()
        print("=" * 110)

        print(
            "TRUE TRANSTREAMING-STYLE "
            "RESULT"
        )

        print(
            f"sensor              : "
            f"{eval_count}"
        )

        print(
            "processed / dropped : "
            f"{processed} / {dropped}"
        )

        print(
            f"drop rate           : "
            f"{100*dropped/eval_count:.3f}%"
        )

        print(
            "deadline miss       : "
            f"{misses}/{processed} "
            f"("
            f"{100*misses/max(processed,1):.3f}%"
            f")"
        )

        print(
            "forward p50/p90/p99 : "
            f"{fs['p50_ms']:.4f} / "
            f"{fs['p90_ms']:.4f} / "
            f"{fs['p99_ms']:.4f} ms"
        )

        print(
            "sAP Moderate R40    : "
            f"Car={car:.4f} "
            f"Ped={ped:.4f} "
            f"Cyc={cyc:.4f} "
            f"Macro={macro:.4f}"
        )

        print(
            f"PF histogram        : "
            f"{dict(proposal_hist)}"
        )

        print(
            f"dispatch horizons   : "
            f"{dict(dispatch_horizon_hist)}"
        )

        print(
            "executed outputs    : "
            f"{total_executed_outputs}"
        )

        print(
            "skipped outputs     : "
            f"{total_skipped_outputs}"
        )

        print(
            "outputs/job         : "
            f"{summary['mean_executed_outputs_per_job']:.4f}"
        )

        print(
            f"summary             : "
            f"{summary_path}"
        )

        print("=" * 110)

    finally:
        model.generate_recall_record = (
            original_recall
        )


if __name__ == "__main__":
    main()
