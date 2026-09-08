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
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

from test_stream_buffer_timestamp import load_one

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

from transtreaming_three_head_runtime import (
    TranstreamingPlanner,
    TranstreamingThreeHeadBank,
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
        "--h2_ckpt",
        required=True,
    )

    p.add_argument(
        "--h3_ckpt",
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
        "--planner_window",
        type=int,
        default=5,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    return p.parse_args()


def reset_history(
    model,
):
    q = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if q is not None:
        q.clear()


def make_head_prediction(
    model,
    dataset,
    batch,
    head_data,
):
    original_token = (
        batch["token"]
    )

    batch["token"] = (
        head_data
    )

    try:
        pred = make_prediction(
            model,
            dataset,
            batch,
        )
    finally:
        batch["token"] = (
            original_token
        )

    return pred


def make_trace(
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
        scene_summary = {}

        for (
            scene,
            indices,
        ) in groups.items():
            for idx in indices:
                trace[idx] = "L0"

            scene_summary[
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

        return (
            trace,
            scene_summary,
        )

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

        for (
            scene,
            indices,
        ) in groups.items():
            for (
                local_pos,
                idx,
            ) in enumerate(
                indices
            ):
                (
                    _,
                    frame_id,
                    _,
                ) = frame_meta(
                    dataset,
                    idx,
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
            "CUDA is required"
        )

    if args.input_hz <= 0:
        raise ValueError(
            "input_hz must be > 0"
        )

    if args.max_frames < 0:
        raise ValueError(
            "max_frames must be >= 0"
        )

    no_load = (
        args.pressure_level
        ==
        "L0"
    )

    if no_load:
        if (
            abs(
                args
                .pressure_fraction
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
                "random contention "
                "requires 0 < f < 1"
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

    # ================================================================
    # Dataset/model
    # ================================================================

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

    bank = (
        TranstreamingThreeHeadBank(
            model=model,
            h2_ckpt=(
                args.h2_ckpt
            ),
            h3_ckpt=(
                args.h3_ckpt
            ),
        )
    )

    logger.info(
        "=" * 70
    )

    logger.info(
        "Transtreaming-style "
        "three-head bank loaded"
    )

    logger.info(
        "H1 -> next"
    )

    logger.info(
        f"H2 -> next2: "
        f"{args.h2_ckpt}"
    )

    logger.info(
        f"H3 -> next3: "
        f"{args.h3_ckpt}"
    )

    logger.info(
        "Shared K3 trunk executes "
        "once per processed job."
    )

    logger.info(
        "=" * 70
    )

    # ================================================================
    # Contention
    # ================================================================

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

    # ================================================================
    # Warm H1/H2/H3 CUDA kernels.
    #
    # IMPORTANT:
    # warmup timing does not enter planner.
    # ================================================================

    _, warm_indices = (
        choose_scene_indices(
            dataset,
            max(
                1,
                args.runtime_warmup_frames
                + 1,
            ),
        )
    )

    reset_history(
        model
    )

    last_batch = None
    last_result = None

    for wi in range(
        args.runtime_warmup_frames
    ):
        idx = (
            warm_indices[
                wi
                %
                len(warm_indices)
            ]
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
                if (
                    wi % 2
                )
                else
                "L0"
            )

        result = (
            transtreaming_forward_no_post(
                model=model,
                bank=bank,
                batch=batch,
                planned_horizons=(
                    1,
                    2,
                    3,
                ),
                model_stream=(
                    model_stream
                ),
                contender=(
                    contenders[
                        warm_level
                    ]
                ),
            )
        )

        last_batch = batch
        last_result = result

    # Warm post/NMS paths.
    if (
        last_batch is not None
        and
        last_result is not None
    ):
        for horizon in (
            last_result
            .executed_horizons
        ):
            make_head_prediction(
                model,
                dataset,
                last_batch,
                last_result
                .head_data[
                    horizon
                ],
            )

    reset_history(
        model
    )

    planner = (
        TranstreamingPlanner(
            future_length=3,
            max_horizon=3,
            window_size=(
                args
                .planner_window
            ),
        )
    )

    # Explicit no runtime leakage.
    planner.reset()

    torch.cuda.synchronize()

    # ================================================================
    # Evaluation frame set
    # ================================================================

    total_dataset = len(
        dataset
    )

    eval_count = (
        total_dataset
        if (
            args.max_frames
            ==
            0
        )
        else
        min(
            total_dataset,
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
    ) = make_trace(
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

    sensor_level_hist = Counter(
        trace_by_idx[idx]
        for idx in eval_indices
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

    print()
    print("=" * 110)

    print(
        "Transtreaming-style "
        "StreamDSGN streaming evaluation"
    )

    print("=" * 110)

    print(
        f"input Hz          : "
        f"{args.input_hz}"
    )

    print(
        f"period            : "
        f"{period_ms:.6f} ms"
    )

    print(
        "timing scope      : "
        "shared trunk + selected "
        "future heads"
    )

    if no_load:
        print(
            "contention        : L0"
        )
    else:
        print(
            "contention        : "
            f"L0 + "
            f"{args.pressure_level}"
        )

        print(
            f"pressure fraction : "
            f"{100 * args.pressure_fraction:.2f}%"
        )

    print(
        f"sensor levels      : "
        f"{dict(sensor_level_hist)}"
    )

    print(
        "input buffer       : "
        "latest-frame only"
    )

    print(
        "history            : "
        "processed-frame only"
    )

    print(
        "output buffer      : "
        "future-target release"
    )

    print(
        f"sensor frames      : "
        f"{eval_count}"
    )

    print("=" * 110)

    # ================================================================
    # Streaming state
    # ================================================================

    rows = {}

    prediction_events = []

    decision_rows = []

    output_event_rows = []

    forward_times = []
    shared_times = []
    wait_times = []
    response_times = []
    first_output_times = []
    heads_per_job = []

    processed_level_hist = Counter()
    proposal_hist = Counter()
    head_exec_hist = Counter()

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
                f"{scene}: "
                f"{n} frames"
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

                wait_ms = (
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
                # Causal future proposal
                # ==================================================

                plan = planner.plan(
                    queue_wait_ms=(
                        wait_ms
                    ),
                    period_ms=(
                        period_ms
                    ),
                )

                planned_pf = (
                    plan.horizons
                )

                batch = load_one(
                    dataset,
                    idx,
                )

                result = (
                    transtreaming_forward_no_post(
                        model=model,
                        bank=bank,
                        batch=batch,
                        planned_horizons=(
                            planned_pf
                        ),
                        model_stream=(
                            model_stream
                        ),
                        contender=(
                            contenders[
                                true_level
                            ]
                        ),
                    )
                )

                # ==================================================
                # Update estimator only AFTER current routing.
                # ==================================================

                planner.observe(
                    shared_ms=(
                        result
                        .shared_ms
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

                miss = (
                    finish_ms
                    >
                    deadline_ms
                    +
                    1e-9
                )

                # ==================================================
                # Post/NMS outside logical forward time.
                # ==================================================

                pred_by_horizon = {}

                for horizon in (
                    result
                    .executed_horizons
                ):
                    pred_by_horizon[
                        horizon
                    ] = (
                        make_head_prediction(
                            model,
                            dataset,
                            batch,
                            result
                            .head_data[
                                horizon
                            ],
                        )
                    )

                first_ready_ms = min(
                    start_ms
                    +
                    result
                    .ready_offset_ms[
                        horizon
                    ]
                    for horizon in (
                        result
                        .executed_horizons
                    )
                )

                # ==================================================
                # Output buffer.
                #
                # Hk predicts source+k.
                #
                # release =
                #   max(compute_ready,
                #       target_sensor_time)
                #
                # This prevents a future prediction from
                # becoming visible before its target time.
                # ==================================================

                for horizon in (
                    result
                    .executed_horizons
                ):
                    head_exec_hist[
                        horizon
                    ] += 1

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
                        target_fid,
                        _,
                    ) = frame_meta(
                        dataset,
                        target_idx,
                    )

                    compute_ready_ms = (
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

                    release_ms = max(
                        compute_ready_ms,
                        target_sensor_ms,
                    )

                    pred_event = {
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
                            target_fid,

                        "compute_ready_ms":
                            compute_ready_ms,

                        "target_sensor_ms":
                            target_sensor_ms,

                        "release_ms":
                            release_ms,

                        "anno":
                            pred_by_horizon[
                                horizon
                            ],
                    }

                    prediction_events.append(
                        pred_event
                    )

                    output_event_rows.append({
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
                            target_fid,

                        "compute_ready_ms":
                            compute_ready_ms,

                        "target_sensor_ms":
                            target_sensor_ms,

                        "release_ms":
                            release_ms,

                        "release_lateness_ms":
                            (
                                release_ms
                                -
                                target_sensor_ms
                            ),
                    })

                pf_text = "|".join(
                    str(x)
                    for x in (
                        planned_pf
                    )
                )

                proposal_hist[
                    pf_text
                ] += 1

                heads_per_job.append(
                    len(
                        result
                        .executed_horizons
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
                        wait_ms,

                    "estimated_shared_ms":
                        (
                            ""
                            if (
                                plan
                                .estimated_shared_ms
                                is None
                            )
                            else
                            plan
                            .estimated_shared_ms
                        ),

                    "estimated_head_ms":
                        (
                            ""
                            if (
                                plan
                                .estimated_head_ms
                                is None
                            )
                            else
                            plan
                            .estimated_head_ms
                        ),

                    "estimated_first_ready_ms":
                        (
                            ""
                            if (
                                plan
                                .estimated_first_ready_ms
                                is None
                            )
                            else
                            plan
                            .estimated_first_ready_ms
                        ),

                    "estimated_delta_ms":
                        (
                            ""
                            if (
                                plan
                                .estimated_delta_ms
                                is None
                            )
                            else
                            plan
                            .estimated_delta_ms
                        ),

                    "planned_pf":
                        pf_text,

                    "executed_pf":
                        "|".join(
                            str(x)
                            for x in (
                                result
                                .executed_horizons
                            )
                        ),

                    "shared_ms":
                        result.shared_ms,

                    "H1_ms":
                        (
                            result
                            .head_ms
                            .get(
                                1,
                                "",
                            )
                        ),

                    "H2_ms":
                        (
                            result
                            .head_ms
                            .get(
                                2,
                                "",
                            )
                        ),

                    "H3_ms":
                        (
                            result
                            .head_ms
                            .get(
                                3,
                                "",
                            )
                        ),

                    "H1_ready_offset_ms":
                        (
                            result
                            .ready_offset_ms
                            .get(
                                1,
                                "",
                            )
                        ),

                    "H2_ready_offset_ms":
                        (
                            result
                            .ready_offset_ms
                            .get(
                                2,
                                "",
                            )
                        ),

                    "H3_ready_offset_ms":
                        (
                            result
                            .ready_offset_ms
                            .get(
                                3,
                                "",
                            )
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

                rows[idx] = {
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
                        wait_ms,

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

                    "planned_pf":
                        pf_text,
                }

                processed += 1

                misses += int(
                    miss
                )

                forward_times.append(
                    result.total_ms
                )

                shared_times.append(
                    result.shared_ms
                )

                wait_times.append(
                    wait_ms
                )

                response_times.append(
                    response_ms
                )

                first_output_times.append(
                    first_ready_ms
                    -
                    arrival_ms
                )

                processed_level_hist[
                    true_level
                ] += 1

                # ==================================================
                # Latest-frame input mailbox.
                #
                # GPU busy until all planned future heads finish.
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
                            dfid,
                            _,
                        ) = frame_meta(
                            dataset,
                            didx,
                        )

                        da = (
                            dp
                            *
                            period_ms
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
                                da,

                            "absolute_deadline_ms":
                                (
                                    da
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

                            "planned_pf":
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
                    processed % 100
                    ==
                    0
                ):
                    print(
                        f"[true={true_level}] "
                        f"processed="
                        f"{processed:04d} "
                        f"{scene}/{frame_id} "
                        f"wait="
                        f"{wait_ms:.3f} "
                        f"PF="
                        f"{pf_text} "
                        f"shared="
                        f"{result.shared_ms:.3f} "
                        f"total="
                        f"{result.total_ms:.3f} "
                        f"finish="
                        f"{finish_ms:.3f} "
                        f"miss="
                        f"{int(miss)}"
                    )

        # ==============================================================
        # Structural checks
        # ==============================================================

        missing = [
            idx
            for idx in (
                eval_indices
            )
            if idx not in rows
        ]

        if missing:
            raise RuntimeError(
                f"missing statuses: "
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
                "frame accounting "
                "mismatch: "
                f"{processed}+"
                f"{dropped}!="
                f"{eval_count}"
            )

        # ==============================================================
        # Transtreaming output-buffer alignment
        # ==============================================================

        events_by_scene = (
            OrderedDict(
                (
                    scene,
                    [],
                )
                for scene in (
                    groups
                )
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
                key=lambda event: (
                    event[
                        "release_ms"
                    ],
                    event[
                        "source_local_pos"
                    ],
                    event[
                        "horizon"
                    ],
                )
            )

        aligned = {}

        for (
            scene,
            indices,
        ) in groups.items():
            scene_events = (
                events_by_scene[
                    scene
                ]
            )

            ptr = 0
            latest_event = None

            for (
                pos,
                idx,
            ) in enumerate(
                indices
            ):
                query_ms = (
                    pos
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
                    latest_event = (
                        scene_events[
                            ptr
                        ]
                    )

                    ptr += 1

                (
                    _,
                    frame_id,
                    next_frame_id,
                ) = frame_meta(
                    dataset,
                    idx,
                )

                pred = (
                    None
                    if (
                        latest_event
                        is None
                    )
                    else
                    latest_event[
                        "anno"
                    ]
                )

                aligned[
                    idx
                ] = copy_det(
                    pred,
                    scene,
                    frame_id,
                    next_frame_id,
                )

        # ==============================================================
        # KITTI sAP
        # ==============================================================

        gt_annos = [
            copy.deepcopy(
                dataset
                .kitti_infos[idx]
                ["infos"]
                ["token"]
                ["annos"]
            )
            for idx in (
                eval_indices
            )
        ]

        det_annos = [
            aligned[
                idx
            ]
            for idx in (
                eval_indices
            )
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

        # ==============================================================
        # Save outputs
        # ==============================================================

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
            "planned_pf",
        ]

        with (
            out_dir
            /
            "frame_timeline.csv"
        ).open(
            "w",
            newline="",
        ) as f:
            writer = (
                csv.DictWriter(
                    f,
                    fieldnames=(
                        timeline_fields
                    ),
                )
            )

            writer.writeheader()

            for idx in (
                eval_indices
            ):
                writer.writerow(
                    rows[idx]
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
                writer = (
                    csv.DictWriter(
                        f,
                        fieldnames=list(
                            decision_rows[
                                0
                            ].keys()
                        ),
                    )
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
                writer = (
                    csv.DictWriter(
                        f,
                        fieldnames=list(
                            output_event_rows[
                                0
                            ].keys()
                        ),
                    )
                )

                writer.writeheader()

                writer.writerows(
                    output_event_rows
                )

        with (
            out_dir
            /
            "prediction_events.pkl"
        ).open(
            "wb"
        ) as f:
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
                "transtreaming_style_"
                "streamdsgn_h123_v1",

            "method":
                "Transtreaming-style "
                "StreamDSGN",

            "base_detector":
                "k3_streamdsgn_"
                "shared_trunk",

            "model_cfg":
                str(
                    args.cfg
                ),

            "h1_ckpt":
                str(
                    args.ckpt
                ),

            "h2_ckpt":
                str(
                    args.h2_ckpt
                ),

            "h3_ckpt":
                str(
                    args.h3_ckpt
                ),

            "head_targets": {
                "H1": "next",
                "H2": "next2",
                "H3": "next3",
            },

            "input_hz":
                args.input_hz,

            "period_ms":
                period_ms,

            "timing_scope":
                "shared_trunk_plus_"
                "selected_future_heads_"
                "forward_only",

            "no_load":
                bool(
                    no_load
                ),

            "planner": {
                "type":
                    "transtreaming_style_"
                    "adaptive_future_"
                    "temporal_proposals",

                "future_length":
                    3,

                "max_horizon":
                    3,

                "window_size":
                    args
                    .planner_window,

                "causal":
                    True,

                "warmup_runtime_leak":
                    False,

                "release_rule":
                    "max(compute_ready,"
                    "target_sensor_time)",
            },

            "contention_trace": {
                "type":
                    (
                        "L0_only"
                        if no_load
                        else
                        "balanced_random_"
                        "per_sensor_frame"
                    ),

                "base_level":
                    "L0",

                "pressure_level":
                    args
                    .pressure_level,

                "pressure_fraction_target":
                    args
                    .pressure_fraction,

                "trace_seed":
                    args.trace_seed,

                "trace_csv":
                    str(
                        trace_csv
                    ),

                "scene_summary":
                    trace_scene_summary,
            },

            "true_sensor_levels":
                dict(
                    sensor_level_hist
                ),

            "true_processed_levels":
                dict(
                    processed_level_hist
                ),

            "buffer_policy":
                "latest_frame_only",

            "history_policy":
                "processed_frames_only",

            "output_buffer_policy":
                "future_target_release",

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

            "shared_latency":
                stats(
                    shared_times
                ),

            "queue_wait":
                stats(
                    wait_times
                ),

            "arrival_to_finish":
                stats(
                    response_times
                ),

            "arrival_to_first_output":
                stats(
                    first_output_times
                ),

            "proposal_histogram":
                dict(
                    proposal_hist
                ),

            "head_execution_counts": {
                "H1":
                    int(
                        head_exec_hist[
                            1
                        ]
                    ),

                "H2":
                    int(
                        head_exec_hist[
                            2
                        ]
                    ),

                "H3":
                    int(
                        head_exec_hist[
                            3
                        ]
                    ),
            },

            "mean_future_heads_per_job":
                (
                    float(
                        np.mean(
                            heads_per_job
                        )
                    )
                    if (
                        heads_per_job
                    )
                    else
                    0.0
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

        if no_load:
            print(
                "Transtreaming-style "
                "StreamDSGN "
                f"@ {args.input_hz:g} Hz "
                "/ L0"
            )
        else:
            print(
                "Transtreaming-style "
                "StreamDSGN "
                f"@ {args.input_hz:g} Hz "
                f"/ L0+"
                f"{args.pressure_level}"
            )

        print(
            f"sensor              : "
            f"{eval_count}"
        )

        print(
            "processed / dropped : "
            f"{processed} / "
            f"{dropped}"
        )

        print(
            f"drop rate           : "
            f"{100*dropped/eval_count:.3f}%"
        )

        print(
            "deadline miss       : "
            f"{misses}/"
            f"{processed} "
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
            "sAP 3D Moderate R40 : "
            f"Car={car:.4f} "
            f"Ped={ped:.4f} "
            f"Cyc={cyc:.4f} "
            f"Macro={macro:.4f}"
        )

        print(
            f"proposal histogram  : "
            f"{dict(proposal_hist)}"
        )

        print(
            "head executions     : "
            f"H1="
            f"{head_exec_hist[1]} "
            f"H2="
            f"{head_exec_hist[2]} "
            f"H3="
            f"{head_exec_hist[3]}"
        )

        print(
            "mean heads/job      : "
            f"{summary['mean_future_heads_per_job']:.4f}"
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
