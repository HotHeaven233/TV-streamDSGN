#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from collections import Counter, OrderedDict, defaultdict
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
from smooth_cuda_contention import make_high_priority_detector_stream
from eval_tv_stream3d_30hz_random50 import (
    build_balanced_trace,
    frame_meta,
    scene_groups,
    copy_det,
    disable_recall,
    stats,
    builtin,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "LASP-style StreamDSGN forward-only "
            "streaming evaluator"
        )
    )

    p.add_argument(
        "--cfg",
        required=True,
    )

    p.add_argument(
        "--ckpt",
        required=True,
    )

    p.add_argument(
        "--pressure_level",
        choices=[
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
        ],
        default="L0",
    )

    p.add_argument(
        "--levels_json",
        default=None,
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
        required=True,
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
        "--output_dir",
        required=True,
    )

    return p.parse_args()


def reset_lasp(model):
    memory = getattr(
        model,
        "lasp_memory",
        None,
    )

    if memory is None:
        raise RuntimeError(
            "LASP model has no lasp_memory"
        )

    memory.clear()

    q = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if q is not None:
        q.clear()


def validate_model_api(model):
    for name in (
        "forward_stream_core",
        "postprocess_last",
        "get_compensated_prediction",
    ):
        if not callable(
            getattr(
                model,
                name,
                None,
            )
        ):
            raise RuntimeError(
                f"LASP runtime API missing: {name}"
            )

    if not hasattr(
        model,
        "lasp_memory",
    ):
        raise RuntimeError(
            "LASP runtime API missing: lasp_memory"
        )


def cpu_prediction(pred):
    required = (
        "pred_boxes",
        "pred_scores",
        "pred_labels",
    )

    missing = [
        k
        for k in required
        if k not in pred
    ]

    if missing:
        raise RuntimeError(
            f"prediction missing keys: {missing}"
        )

    out = {
        k: (
            v.detach().cpu().clone()
            if torch.is_tensor(v)
            else copy.deepcopy(v)
        )
        for k, v in pred.items()
    }

    boxes = out[
        "pred_boxes"
    ]

    scores = out[
        "pred_scores"
    ]

    labels = out[
        "pred_labels"
    ]

    if (
        boxes.ndim != 2
        or
        boxes.shape[-1] < 7
    ):
        raise RuntimeError(
            "invalid pred_boxes shape: "
            f"{tuple(boxes.shape)}"
        )

    if (
        scores.ndim != 1
        or
        labels.ndim != 1
    ):
        raise RuntimeError(
            "invalid score/label shapes: "
            f"{tuple(scores.shape)}, "
            f"{tuple(labels.shape)}"
        )

    if not (
        boxes.shape[0]
        ==
        scores.shape[0]
        ==
        labels.shape[0]
    ):
        raise RuntimeError(
            "prediction length mismatch"
        )

    if (
        not torch.isfinite(
            boxes
        ).all()
        or
        not torch.isfinite(
            scores
        ).all()
    ):
        raise RuntimeError(
            "non-finite prediction"
        )

    return out


def timed_lasp_forward(
    model,
    batch,
    model_stream,
    contender,
):
    #
    # H2D is explicitly excluded.
    #
    torch.cuda.synchronize()

    launched = False

    try:
        if contender is not None:
            contender.launch()
            launched = True

        with (
            torch.cuda.stream(
                model_stream
            ),
            torch.no_grad(),
            torch.amp.autocast(
                "cuda",
                enabled=bool(
                    model.use_amp_dict[
                        "TEST"
                    ]
                ),
            ),
        ):
            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            #
            # ALL LASP neural/model operators belong here.
            #
            meta = (
                model.forward_stream_core(
                    batch
                )
            )

            end.record()

        end.synchronize()

        forward_ms = float(
            start.elapsed_time(
                end
            )
        )

    finally:
        if launched:
            contender.finish()

    if (
        not np.isfinite(
            forward_ms
        )
        or
        forward_ms <= 0
    ):
        raise RuntimeError(
            "invalid forward latency: "
            f"{forward_ms}"
        )

    return (
        meta,
        forward_ms,
    )


def replay_lasp_forward(
    model,
    batch,
):
    with (
        torch.no_grad(),
        torch.amp.autocast(
            "cuda",
            enabled=bool(
                model.use_amp_dict[
                    "TEST"
                ]
            ),
        ),
    ):
        meta = (
            model.forward_stream_core(
                batch
            )
        )

    torch.cuda.synchronize()

    return meta


def untimed_h0_postprocess(
    model,
):
    """
    One post/NMS per processed frame.

    It is outside:
        CUDA-event timing
        logical service time

    This matches the existing Original/TV formal evaluators,
    which also execute post/NMS between processed forwards.
    """

    with torch.no_grad():
        pred_dicts, _ = (
            model.postprocess_last(
                delta_frames=0.0
            )
        )

    torch.cuda.synchronize()

    if len(
        pred_dicts
    ) != 1:
        raise RuntimeError(
            "H0 maintenance postprocess "
            "expected one prediction, "
            f"got {len(pred_dicts)}"
        )


def build_l0_trace(
    groups,
):
    trace = {}
    summary = {}

    for scene, indices in (
        groups.items()
    ):
        for idx in indices:
            trace[
                idx
            ] = "L0"

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

    return (
        trace,
        summary,
    )


def cpu_query_batch(
    dataset,
    idx,
):
    """
    Used only after formal timing.

    No H2D is needed to convert a cached raw prediction
    to KITTI annotation.
    """

    return dataset.collate_batch(
        [
            dataset[
                idx
            ]
        ]
    )


def raw_to_query_anno(
    dataset,
    query_batch,
    raw_pred,
    scene,
    frame_id,
    next_frame_id,
):
    #
    # IMPORTANT:
    #
    # Use QUERY-frame metadata/calibration.
    #
    annos = (
        dataset
        .generate_prediction_dicts(
            query_batch,
            [
                raw_pred
            ],
            dataset.class_names,
            output_path=None,
        )
    )

    if len(
        annos
    ) != 1:
        raise RuntimeError(
            "expected one generated "
            f"annotation, got {len(annos)}"
        )

    return copy_det(
        annos[0],
        scene,
        frame_id,
        next_frame_id,
    )


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

    if (
        args.runtime_warmup_frames < 0
        or
        args.max_frames < 0
    ):
        raise ValueError(
            "warmup/max_frames "
            "must be >= 0"
        )

    random50 = (
        args.pressure_level
        !=
        "L0"
    )

    if random50:
        if not (
            0.0
            <
            args.pressure_fraction
            <
            1.0
        ):
            raise ValueError(
                "Random50 requires "
                "0 < pressure_fraction < 1"
            )

        if args.levels_json is None:
            raise ValueError(
                "--levels_json is required "
                "when pressure_level != L0"
            )

        if not Path(
            args.levels_json
        ).is_file():
            raise FileNotFoundError(
                args.levels_json
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
        args.cfg
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

    if not hasattr(
        dataset,
        "kitti_infos",
    ):
        raise RuntimeError(
            "dataset has no kitti_infos"
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
        to_cpu=True,
    )

    model.cuda().eval()

    validate_model_api(
        model
    )

    max_horizon = int(
        max(
            int(x)
            for x
            in cfg.MODEL.LASP.FUTURE_STEPS
        )
    )

    if max_horizon <= 0:
        raise RuntimeError(
            "invalid LASP max horizon: "
            f"{max_horizon}"
        )

    if random50:
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

    else:
        #
        # Real no-load:
        # do not launch a contender at all.
        #
        contenders = {
            "L0":
                None
        }

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

    try:

        # =========================================================
        # Runtime warmup
        # =========================================================

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

        reset_lasp(
            model
        )

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

            level = (
                args.pressure_level
                if (
                    random50
                    and
                    wi % 2
                )
                else
                "L0"
            )

            timed_lasp_forward(
                model,
                batch,
                model_stream,
                contenders[
                    level
                ],
            )

        #
        # First NMS/CUDA-extension initialization
        # stays outside formal measurement.
        #
        if (
            args.runtime_warmup_frames
            >
            0
        ):
            untimed_h0_postprocess(
                model
            )

        reset_lasp(
            model
        )

        torch.cuda.synchronize()

        # =========================================================
        # Evaluation set
        # =========================================================

        eval_count = (
            len(
                dataset
            )
            if args.max_frames == 0
            else min(
                len(
                    dataset
                ),
                args.max_frames,
            )
        )

        if eval_count <= 0:
            raise RuntimeError(
                "evaluation set is empty"
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

        # =========================================================
        # Exact sensor-frame contention trace
        # =========================================================

        if random50:
            (
                trace_by_idx,
                trace_scene_summary,
            ) = (
                build_balanced_trace(
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
            )

        else:
            (
                trace_by_idx,
                trace_scene_summary,
            ) = build_l0_trace(
                groups
            )

        sensor_level_hist = Counter(
            trace_by_idx[
                idx
            ]
            for idx
            in eval_indices
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
                    enumerate(
                        indices
                    )
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

        print(
            "=" * 100
        )

        print(
            "LASP-style StreamDSGN "
            "(3D adaptation)"
        )

        print(
            f"input Hz          : "
            f"{args.input_hz:g}"
        )

        print(
            f"period            : "
            f"{period_ms:.6f} ms"
        )

        print(
            "timing scope      : "
            "forward_stream_core only"
        )

        print(
            "contention        : "
            +
            (
                f"Random50 "
                f"L0+{args.pressure_level}"
                if random50
                else
                "none / L0"
            )
        )

        print(
            f"sensor frames      : "
            f"{eval_count}"
        )

        print(
            f"LASP max horizon   : "
            f"H{max_horizon}"
        )

        print(
            "=" * 100
        )

        # =========================================================
        # PASS 1
        #
        # FORMAL TIMING ONLY
        # =========================================================

        rows = {}

        timing_events = []

        processed_by_scene = (
            OrderedDict(
                (
                    scene,
                    [],
                )
                for scene
                in groups
            )
        )

        forward_times = []
        wait_times = []
        response_times = []

        processed_level_hist = (
            Counter()
        )

        processed = 0
        dropped = 0
        misses = 0

        for scene_i, (
            scene,
            indices,
        ) in enumerate(
            groups.items(),
            start=1,
        ):

            reset_lasp(
                model
            )

            pos = 0
            gpu_free_ms = 0.0
            n = len(
                indices
            )

            print(
                f"[timing scene "
                f"{scene_i:02d}/"
                f"{len(groups):02d}] "
                f"{scene}: {n} frames"
            )

            while pos < n:
                idx = indices[
                    pos
                ]

                (
                    meta_scene,
                    frame_id,
                    _,
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

                true_level = (
                    trace_by_idx[
                        idx
                    ]
                )

                batch = load_one(
                    dataset,
                    idx,
                )

                (
                    meta,
                    forward_ms,
                ) = timed_lasp_forward(
                    model,
                    batch,
                    model_stream,
                    contenders[
                        true_level
                    ],
                )

                #
                # Formal protocol excludes post/NMS from logical time,
                # but existing baselines execute one such operation
                # between processed forwards.
                #
                untimed_h0_postprocess(
                    model
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

                event = {
                    "scene":
                        scene,

                    "source_index":
                        idx,

                    "source_pos":
                        pos,

                    "source_frame_id":
                        frame_id,

                    "arrival_ms":
                        arrival_ms,

                    "start_ms":
                        start_ms,

                    "finish_ms":
                        finish_ms,

                    "forward_ms":
                        forward_ms,

                    "true_level":
                        true_level,

                    "lasp_meta":
                        builtin(
                            meta
                        ),
                }

                timing_events.append(
                    event
                )

                processed_by_scene[
                    scene
                ].append(
                    event
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
                        finish_ms,

                    "queue_wait_ms":
                        wait_ms,

                    "forward_ms":
                        forward_ms,

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
                }

                processed += 1

                misses += int(
                    miss
                )

                forward_times.append(
                    forward_ms
                )

                wait_times.append(
                    wait_ms
                )

                response_times.append(
                    response_ms
                )

                processed_level_hist[
                    true_level
                ] += 1

                # ================================================
                # capacity-1 latest-frame mailbox
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
                                da
                                +
                                period_ms,

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
                        f"[timing true={true_level}] "
                        f"processed={processed:04d} "
                        f"{scene}/{frame_id} "
                        f"arrival={arrival_ms:.3f} "
                        f"start={start_ms:.3f} "
                        f"finish={finish_ms:.3f} "
                        f"fwd={forward_ms:.3f} "
                        f"miss={int(miss)}"
                    )

        # =========================================================
        # Structural accounting
        # =========================================================

        missing = [
            idx
            for idx
            in eval_indices
            if idx not in rows
        ]

        if missing:
            raise RuntimeError(
                "missing frame statuses: "
                f"{missing[:10]}"
            )

        if (
            processed
            +
            dropped
            !=
            eval_count
        ):
            raise RuntimeError(
                "frame accounting mismatch: "
                f"{processed}+"
                f"{dropped}!="
                f"{eval_count}"
            )

        # =========================================================
        # Determine which completed source is visible at each
        # sensor query time.
        #
        # Also determine the exact H0...H8 entries needed.
        # =========================================================

        assignments = {}

        needed_horizons = (
            defaultdict(
                set
            )
        )

        for scene, indices in (
            groups.items()
        ):
            es = (
                processed_by_scene[
                    scene
                ]
            )

            ptr = 0
            latest = None

            for qpos, idx in (
                enumerate(
                    indices
                )
            ):
                query_ms = (
                    qpos
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
                        ]
                    )

                    ptr += 1

                (
                    _,
                    qfid,
                    qnext,
                ) = frame_meta(
                    dataset,
                    idx,
                )

                if latest is None:
                    assignments[
                        idx
                    ] = {
                        "scene":
                            scene,

                        "query_pos":
                            qpos,

                        "query_frame_id":
                            qfid,

                        "query_next_frame_id":
                            qnext,

                        "query_ms":
                            query_ms,

                        "source_index":
                            None,

                        "source_pos":
                            None,

                        "source_frame_id":
                            None,

                        "source_finish_ms":
                            None,

                        "delta_frames":
                            None,

                        "used_horizon":
                            None,
                    }

                    continue

                delta_frames = (
                    qpos
                    -
                    latest[
                        "source_pos"
                    ]
                )

                if delta_frames < 0:
                    raise RuntimeError(
                        "negative LASP query delta: "
                        f"{delta_frames}"
                    )

                used_horizon = min(
                    int(
                        delta_frames
                    ),
                    max_horizon,
                )

                needed_horizons[
                    latest[
                        "source_index"
                    ]
                ].add(
                    used_horizon
                )

                assignments[
                    idx
                ] = {
                    "scene":
                        scene,

                    "query_pos":
                        qpos,

                    "query_frame_id":
                        qfid,

                    "query_next_frame_id":
                        qnext,

                    "query_ms":
                        query_ms,

                    "source_index":
                        latest[
                            "source_index"
                        ],

                    "source_pos":
                        latest[
                            "source_pos"
                        ],

                    "source_frame_id":
                        latest[
                            "source_frame_id"
                        ],

                    "source_finish_ms":
                        latest[
                            "finish_ms"
                        ],

                    "delta_frames":
                        int(
                            delta_frames
                        ),

                    "used_horizon":
                        int(
                            used_horizon
                        ),
                }

        # =========================================================
        # PASS 2
        #
        # EXACT processed-frame replay
        #
        # No contention.
        # No formal latency measurement.
        #
        # Only requested horizons are materialized.
        # =========================================================

        raw_cache = {}

        replay_processed = 0

        replay_postprocess_calls = 0

        for scene_i, (
            scene,
            _,
        ) in enumerate(
            groups.items(),
            start=1,
        ):

            reset_lasp(
                model
            )

            es = (
                processed_by_scene[
                    scene
                ]
            )

            print(
                f"[replay scene "
                f"{scene_i:02d}/"
                f"{len(groups):02d}] "
                f"{scene}: "
                f"{len(es)} processed frames"
            )

            for event in es:
                idx = event[
                    "source_index"
                ]

                batch = load_one(
                    dataset,
                    idx,
                )

                replay_lasp_forward(
                    model,
                    batch,
                )

                horizons = sorted(
                    needed_horizons.get(
                        idx,
                        set(),
                    )
                )

                for h in horizons:
                    with torch.no_grad():
                        pred_dicts, _ = (
                            model.postprocess_last(
                                delta_frames=float(
                                    h
                                )
                            )
                        )

                    if len(
                        pred_dicts
                    ) != 1:
                        raise RuntimeError(
                            f"source={idx}, H{h}: "
                            "expected one prediction, "
                            f"got {len(pred_dicts)}"
                        )

                    raw_cache[
                        (
                            idx,
                            int(
                                h
                            ),
                        )
                    ] = cpu_prediction(
                        pred_dicts[
                            0
                        ]
                    )

                    replay_postprocess_calls += 1

                replay_processed += 1

                if (
                    replay_processed <= 3
                    or
                    replay_processed % 100 == 0
                ):
                    print(
                        f"[replay] "
                        f"processed="
                        f"{replay_processed:04d} "
                        f"source={idx} "
                        f"horizons={horizons}"
                    )

        if (
            replay_processed
            !=
            processed
        ):
            raise RuntimeError(
                "replay processed mismatch: "
                f"{replay_processed}!="
                f"{processed}"
            )

        for idx, a in (
            assignments.items()
        ):
            if (
                a[
                    "source_index"
                ]
                is None
            ):
                continue

            key = (
                a[
                    "source_index"
                ],
                a[
                    "used_horizon"
                ],
            )

            if key not in raw_cache:
                raise RuntimeError(
                    "missing replay cache "
                    f"for query={idx}: "
                    f"{key}"
                )

        # =========================================================
        # Convert compensated raw boxes in QUERY-frame context
        # =========================================================

        aligned = {}

        query_rows = []

        for qn, idx in enumerate(
            eval_indices,
            start=1,
        ):
            a = assignments[
                idx
            ]

            scene = a[
                "scene"
            ]

            qfid = a[
                "query_frame_id"
            ]

            qnext = a[
                "query_next_frame_id"
            ]

            if (
                a[
                    "source_index"
                ]
                is None
            ):
                anno = copy_det(
                    None,
                    scene,
                    qfid,
                    qnext,
                )

            else:
                raw = raw_cache[
                    (
                        a[
                            "source_index"
                        ],
                        a[
                            "used_horizon"
                        ],
                    )
                ]

                qbatch = (
                    cpu_query_batch(
                        dataset,
                        idx,
                    )
                )

                anno = (
                    raw_to_query_anno(
                        dataset,
                        qbatch,
                        raw,
                        scene,
                        qfid,
                        qnext,
                    )
                )

            aligned[
                idx
            ] = anno

            query_rows.append({
                "global_index":
                    idx,

                "scene":
                    scene,

                "query_pos":
                    a[
                        "query_pos"
                    ],

                "frame_id":
                    qfid,

                "query_ms":
                    a[
                        "query_ms"
                    ],

                "source_index":
                    (
                        ""
                        if
                        a[
                            "source_index"
                        ]
                        is None
                        else
                        a[
                            "source_index"
                        ]
                    ),

                "source_frame_id":
                    (
                        ""
                        if
                        a[
                            "source_frame_id"
                        ]
                        is None
                        else
                        a[
                            "source_frame_id"
                        ]
                    ),

                "source_pos":
                    (
                        ""
                        if
                        a[
                            "source_pos"
                        ]
                        is None
                        else
                        a[
                            "source_pos"
                        ]
                    ),

                "source_finish_ms":
                    (
                        ""
                        if
                        a[
                            "source_finish_ms"
                        ]
                        is None
                        else
                        a[
                            "source_finish_ms"
                        ]
                    ),

                "delta_frames":
                    (
                        ""
                        if
                        a[
                            "delta_frames"
                        ]
                        is None
                        else
                        a[
                            "delta_frames"
                        ]
                    ),

                "used_horizon":
                    (
                        ""
                        if
                        a[
                            "used_horizon"
                        ]
                        is None
                        else
                        a[
                            "used_horizon"
                        ]
                    ),
            })

            if (
                qn
                %
                500
                ==
                0
            ):
                print(
                    "[query conversion] "
                    f"{qn}/{eval_count}"
                )

        # =========================================================
        # Streaming KITTI AP
        # =========================================================

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

        ap_values = np.asarray(
            [
                car,
                ped,
                cyc,
            ],
            dtype=np.float64,
        )

        #
        # Full formal set must have all three.
        # A small smoke subset may lack one class.
        #
        if (
            args.max_frames == 0
            and
            not np.isfinite(
                ap_values
            ).all()
        ):
            raise RuntimeError(
                "non-finite full-set AP: "
                f"Car={car}, "
                f"Ped={ped}, "
                f"Cyc={cyc}"
            )

        macro = (
            float(
                np.nanmean(
                    ap_values
                )
            )
            if np.isfinite(
                ap_values
            ).any()
            else
            float(
                "nan"
            )
        )

        # =========================================================
        # Outputs
        # =========================================================

        timeline_path = (
            out_dir
            /
            "frame_timeline.csv"
        )

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
        ]

        with timeline_path.open(
            "w",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=(
                    timeline_fields
                ),
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

        query_path = (
            out_dir
            /
            "query_alignment.csv"
        )

        query_fields = [
            "global_index",
            "scene",
            "query_pos",
            "frame_id",
            "query_ms",
            "source_index",
            "source_frame_id",
            "source_pos",
            "source_finish_ms",
            "delta_frames",
            "used_horizon",
        ]

        with query_path.open(
            "w",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=(
                    query_fields
                ),
            )

            w.writeheader()

            w.writerows(
                query_rows
            )

        (
            out_dir
            /
            "timing_events.json"
        ).write_text(
            json.dumps(
                builtin(
                    timing_events
                ),
                indent=2,
            )
            +
            "\n"
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

        fs = stats(
            forward_times
        )

        ws = stats(
            wait_times
        )

        rs = stats(
            response_times
        )

        summary = {
            "version":
                "lasp_style_3d_adaptation_"
                "forward_only_v1",

            "method":
                "LASP-style StreamDSGN "
                "(3D adaptation)",

            "model_cfg":
                str(
                    args.cfg
                ),

            "model_ckpt":
                str(
                    args.ckpt
                ),

            "input_hz":
                args.input_hz,

            "period_ms":
                period_ms,

            "timing_scope":
                "forward_stream_core_only",

            "timing_excludes": [
                "dataloader",
                "H2D",
                "post_processing",
                "NMS",
                "trajectory_query_replay",
                "KITTI_annotation_conversion",
                "evaluation",
            ],

            "contention": {
                "mode":
                    (
                        "random50"
                        if random50
                        else
                        "no_load"
                    ),

                "base_level":
                    "L0",

                "pressure_level":
                    (
                        args.pressure_level
                        if random50
                        else
                        None
                    ),

                "pressure_fraction_target":
                    (
                        args.pressure_fraction
                        if random50
                        else
                        0.0
                    ),

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

            "query_policy": {
                "query_times":
                    "sensor_arrival_times",

                "source":
                    "latest_prediction_"
                    "with_finish_le_query",

                "delta_frames":
                    "query_local_pos-"
                    "source_local_pos",

                "max_horizon":
                    max_horizon,

                "beyond_horizon":
                    f"clamp_to_H"
                    f"{max_horizon}",

                "annotation_coordinate_frame":
                    "query_frame",

                "prediction_replay_is_timed":
                    False,
            },

            "sensor_frames":
                eval_count,

            "processed_frames":
                processed,

            "dropped_frames":
                dropped,

            "drop_rate":
                dropped
                /
                eval_count,

            "deadline_miss_count":
                misses,

            "deadline_miss_rate":
                misses
                /
                processed,

            "forward_latency":
                fs,

            "queue_wait":
                ws,

            "arrival_to_finish":
                rs,

            "replay": {
                "processed_frames":
                    replay_processed,

                "postprocess_calls":
                    replay_postprocess_calls,

                "cached_source_horizon_pairs":
                    len(
                        raw_cache
                    ),
            },

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
                builtin(
                    summary
                ),
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
            f"LASP-style @ "
            f"{args.input_hz:g} Hz / "
            +
            (
                f"Random50 "
                f"L0+{args.pressure_level}"
                if random50
                else
                "L0 no-load"
            )
        )

        print(
            "processed / dropped : "
            f"{processed} / "
            f"{dropped}"
        )

        print(
            f"drop rate           : "
            f"{100.0*dropped/eval_count:.3f}%"
        )

        print(
            f"deadline miss       : "
            f"{misses}/{processed} "
            f"("
            f"{100.0*misses/processed:.3f}%"
            f")"
        )

        print(
            f"forward p50/p90/p99 : "
            f"{fs['p50_ms']:.4f} / "
            f"{fs['p90_ms']:.4f} / "
            f"{fs['p99_ms']:.4f} ms"
        )

        print(
            f"sAP 3D Moderate R40 : "
            f"Car={car:.4f} "
            f"Ped={ped:.4f} "
            f"Cyc={cyc:.4f} "
            f"Macro={macro:.4f}"
        )

        print(
            f"replay postprocess  : "
            f"{replay_postprocess_calls}"
        )

        print(
            f"summary             : "
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
