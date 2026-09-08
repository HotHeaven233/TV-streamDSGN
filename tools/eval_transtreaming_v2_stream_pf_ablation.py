#!/usr/bin/env python3

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
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


# The TAT was explicitly supervised only at these future offsets.
# Runtime must not silently query H3/H5/H6/H7.
TRAINED_FUTURE_HORIZONS = (1, 2, 4, 8)


# =============================================================================
# Runtime estimator
# =============================================================================

class DecayEstimator:

    def __init__(
        self,
        initial_ms,
        window=5,
    ):
        self.initial_ms = float(
            initial_ms
        )
        self.window = int(window)
        self.reset()

    def reset(self):
        self.values = [
            self.initial_ms
            for _ in range(
                self.window
            )
        ]

    def observe(
        self,
        value,
    ):
        value = float(value)

        if (
            not math.isfinite(value)
            or value <= 0
        ):
            return

        self.values = [
            value,
            *self.values[
                : self.window - 1
            ],
        ]

    def estimate(self):
        # Transtreaming-style decay:
        # 0.5, 0.25, 0.125, ...
        weights = []

        for i in range(
            self.window
        ):
            if i < self.window - 1:
                w = 0.5 ** (
                    i + 1
                )
            else:
                w = 0.5 ** (
                    self.window - 1
                )

            weights.append(w)

        return float(
            sum(
                x * w
                for x, w
                in zip(
                    self.values,
                    weights,
                )
            )
        )


class AdaptivePlanner:

    def __init__(
        self,
        period_ms,
        future_length=4,
        max_future=8,
        window=5,
    ):
        self.period_ms = float(
            period_ms
        )

        self.future_length = int(
            future_length
        )

        self.max_future = int(
            max_future
        )

        # Original Transtreaming uses an initially conservative
        # half-frame runtime estimate.
        init = (
            0.5
            *
            self.period_ms
        )

        self.bn = DecayEstimator(
            init,
            window,
        )

        self.head = DecayEstimator(
            init,
            window,
        )

    def reset(self):
        self.bn.reset()
        self.head.reset()

    def plan(
        self,
        queue_wait_ms,
    ):
        t_bn = (
            self.bn.estimate()
        )

        t_h = (
            self.head.estimate()
        )

        # Our frozen forward-only protocol excludes
        # loader / H2D / post / evaluation.
        t_other = 0.0

        t_first = (
            float(queue_wait_ms)
            +
            t_bn
            +
            t_h
        )

        t_delta = (
            (
                (
                    self.future_length
                    +
                    1
                )
                *
                t_h
                +
                t_bn
                +
                t_other
            )
            /
            self.future_length
        )

        raw_pf = []

        for i in range(
            self.future_length
        ):
            t = (
                t_first
                +
                i * t_delta
            )

            h = int(
                math.ceil(
                    t
                    /
                    self.period_ms
                )
            )

            h = max(
                1,
                min(
                    self.max_future,
                    h,
                ),
            )

            if h not in raw_pf:
                raw_pf.append(h)

        if not raw_pf:
            raw_pf = [1]

        # ------------------------------------------------------------
        # CRITICAL:
        # TAT was trained only on {1,2,4,8}.
        #
        # Quantize each raw causal proposal upward to the nearest
        # TRAINED horizon:
        #
        #   1 -> 1
        #   2 -> 2
        #   3 -> 4
        #   4 -> 4
        #   5/6/7/8 -> 8
        #
        # Then de-duplicate while preserving order.
        # ------------------------------------------------------------

        trained = [
            h
            for h in TRAINED_FUTURE_HORIZONS
            if h <= self.max_future
        ]

        if not trained:
            raise RuntimeError(
                "no trained future horizon is compatible "
                f"with max_future={self.max_future}"
            )

        pf = []

        for raw_h in raw_pf:
            q = next(
                (
                    h
                    for h in trained
                    if h >= raw_h
                ),
                trained[-1],
            )

            if q not in pf:
                pf.append(q)

        return {
            "pf":
                tuple(pf),

            "raw_pf":
                tuple(raw_pf),

            "estimated_bn_ms":
                float(t_bn),

            "estimated_head_ms":
                float(t_h),

            "estimated_first_ms":
                float(t_first),

            "estimated_delta_ms":
                float(t_delta),
        }

    def observe(
        self,
        bn_ms,
        head_ms_values,
    ):
        self.bn.observe(
            bn_ms
        )

        values = [
            float(x)
            for x in head_ms_values
            if (
                math.isfinite(float(x))
                and float(x) > 0
            )
        ]

        if values:
            self.head.observe(
                sum(values)
                /
                len(values)
            )


# =============================================================================
# Model runtime
# =============================================================================

def reset_history(model):
    model.history_feature_queue.clear()


def run_forward(
    model,
    batch,
    *,
    past_offsets,
    future_offsets,
    source_pos,
    start_ms,
    period_ms,
    estimated_bn_ms,
    estimated_head_ms,
    model_stream,
    contender,
    late_skip=True,
):
    """
    Timed region:

      current stereo feature extractor
        + TAT / RTPE
        + actually executed shared VAN + StreamDetHead paths

    Excluded:

      dataloader
      H2D
      post / NMS
      KITTI evaluation

    History semantics:

      only a completed processed current frame enters
      model.history_feature_queue.
    """

    cur = batch["token"]

    launched = False

    current_history = None

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
                    model.use_amp_dict.get(
                        "TEST",
                        False,
                    )
                ),
            ),
        ):
            ev_start = torch.cuda.Event(
                enable_timing=True
            )

            ev_bn = torch.cuda.Event(
                enable_timing=True
            )

            ev_start.record()

            # ---------------------------------------------------------
            # Current stereo backbone exactly once.
            # ---------------------------------------------------------

            cur = model._run_feature_extractor(
                cur
            )

            current_history = (
                model._snapshot_history_feature(
                    cur,
                    clone=True,
                )
            )

            # ---------------------------------------------------------
            # Historical Feature Buffer
            # ---------------------------------------------------------

            actual_history_count = len(
                model.history_feature_queue
            )

            if (
                actual_history_count
                !=
                len(past_offsets)
            ):
                raise RuntimeError(
                    "runtime P^P/history mismatch: "
                    f"P^P={past_offsets}, "
                    f"queue={actual_history_count}"
                )

            # Startup:
            # no real completed history yet.
            #
            # Use the current pre-TAT BEV as one zero-delta
            # causal pseudo-history, matching V2 offline startup.
            if actual_history_count == 0:
                startup_queue = deque(
                    maxlen=model.ts_past_length
                )

                startup_queue.append(
                    (
                        cur.get(
                            "this_sample_idx",
                            "",
                        ),
                        current_history,
                    )
                )

                cur[
                    "history_features"
                ] = startup_queue

                pp_for_tat = [0]

            else:
                cur[
                    "history_features"
                ] = (
                    model
                    .history_feature_queue
                )

                pp_for_tat = list(
                    past_offsets
                )

            # ---------------------------------------------------------
            # Dynamic P^P / P^F
            # ---------------------------------------------------------

            model._set_temporal_proposals(
                cur_data=cur,
                past_offsets=pp_for_tat,
                future_offsets=list(
                    future_offsets
                ),
            )

            cur, future_bev = (
                model._run_tat(
                    cur
                )
            )

            if (
                future_bev.shape[1]
                !=
                len(future_offsets)
            ):
                raise RuntimeError(
                    "future proposal count mismatch: "
                    f"{future_bev.shape[1]} vs "
                    f"{future_offsets}"
                )

            ev_bn.record()

            # ---------------------------------------------------------
            # Same shared VAN + StreamDetHead for every P^F.
            # ---------------------------------------------------------

            branch_data = {}
            head_events = {}

            executed = []
            skipped = []

            previous_event = (
                ev_bn
            )

            estimated_elapsed = float(
                estimated_bn_ms
            )

            for j, horizon in enumerate(
                future_offsets
            ):
                horizon = int(
                    horizon
                )

                # Always execute first future output.
                #
                # Later outputs may be skipped if causally
                # predicted to be stale already.
                if (
                    j > 0
                    and late_skip
                ):
                    estimated_now = (
                        float(start_ms)
                        +
                        estimated_elapsed
                    )

                    target_ms = (
                        (
                            int(source_pos)
                            +
                            horizon
                        )
                        *
                        float(period_ms)
                    )

                    if (
                        estimated_now
                        >
                        target_ms
                        +
                        float(
                            estimated_head_ms
                        )
                    ):
                        skipped.append(
                            horizon
                        )

                        continue

                bd = (
                    model
                    ._run_shared_detection_path(
                        fused_data=cur,
                        future_feature=(
                            future_bev[
                                :,
                                j,
                            ]
                        ),
                    )
                )

                ev = torch.cuda.Event(
                    enable_timing=True
                )

                ev.record()

                branch_data[
                    horizon
                ] = bd

                head_events[
                    horizon
                ] = (
                    previous_event,
                    ev,
                )

                previous_event = ev

                executed.append(
                    horizon
                )

                estimated_elapsed += float(
                    estimated_head_ms
                )

            if not executed:
                raise RuntimeError(
                    "no future head executed"
                )

            ev_final = (
                head_events[
                    executed[-1]
                ][1]
            )

        # Final operator boundary.
        ev_final.synchronize()

        bn_ms = float(
            ev_start.elapsed_time(
                ev_bn
            )
        )

        head_ms = {}
        ready_ms = {}

        for horizon in executed:
            e0, e1 = (
                head_events[
                    horizon
                ]
            )

            head_ms[
                horizon
            ] = float(
                e0.elapsed_time(e1)
            )

            ready_ms[
                horizon
            ] = float(
                ev_start.elapsed_time(
                    e1
                )
            )

        total_ms = float(
            ev_start.elapsed_time(
                ev_final
            )
        )

    finally:
        if launched:
            contender.finish()

    # -------------------------------------------------------------
    # Current feature becomes historical only after job completion.
    # -------------------------------------------------------------

    model.history_feature_queue.append(
        (
            cur.get(
                "this_sample_idx",
                "",
            ),
            current_history,
        )
    )

    batch["token"] = cur

    return {
        "bn_ms":
            bn_ms,

        "head_ms":
            head_ms,

        "ready_offset_ms":
            ready_ms,

        "total_ms":
            total_ms,

        "executed_pf":
            tuple(executed),

        "skipped_pf":
            tuple(skipped),

        "branch_data":
            branch_data,
    }


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
        result = make_prediction(
            model,
            dataset,
            batch,
        )
    finally:
        batch["token"] = (
            original
        )

    return result


# =============================================================================
# CLI
# =============================================================================

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
        "--fixed_pf",
        type=str,
        default="",
        help=(
            "optional fixed trained future proposals, "
            "e.g. 1 or 1,2 or 1,2,4 or 1,2,4,8"
        ),
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


def make_trace(
    groups,
    pressure_level,
    pressure_fraction,
    trace_seed,
):
    if (
        pressure_level == "L0"
        or pressure_fraction <= 0
    ):
        trace = {}

        summary = {}

        for scene, indices in (
            groups.items()
        ):
            for idx in indices:
                trace[idx] = "L0"

            summary[scene] = {
                "sensor_frames":
                    len(indices),

                "pressure_frames":
                    0,

                "l0_frames":
                    len(indices),

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
    trace,
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
            for pos, idx in enumerate(
                indices
            ):
                _, fid, _ = frame_meta(
                    dataset,
                    idx,
                )

                writer.writerow({
                    "global_index":
                        idx,

                    "scene":
                        scene,

                    "local_pos":
                        pos,

                    "frame_id":
                        fid,

                    "true_level":
                        trace[idx],
                })


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(
        args.seed
    )

    period_ms = (
        1000.0
        /
        float(args.input_hz)
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

    model.cuda()
    model.eval()

    if (
        type(model).__name__
        !=
        "TRANSTREAMING_STREAM_V2"
    ):
        raise RuntimeError(
            "expected TRANSTREAMING_STREAM_V2, got "
            f"{type(model).__name__}"
        )

    if (
        model
        .history_feature_queue
        .maxlen
        !=
        args.past_length
    ):
        raise RuntimeError(
            "history capacity mismatch"
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
            )
    }

    if (
        args.pressure_level
        !=
        "L0"
    ):
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

    planner = AdaptivePlanner(
        period_ms=period_ms,
        future_length=(
            args.future_length
        ),
        max_future=(
            args.max_future
        ),
        window=5,
    )

    fixed_pf = None

    if args.fixed_pf.strip():
        values = []

        for token in args.fixed_pf.split(","):
            token = token.strip()

            if not token:
                continue

            h = int(token)

            if h not in TRAINED_FUTURE_HORIZONS:
                raise ValueError(
                    f"--fixed_pf contains untrained horizon H{h}; "
                    f"allowed={TRAINED_FUTURE_HORIZONS}"
                )

            if h > args.max_future:
                raise ValueError(
                    f"H{h} exceeds max_future={args.max_future}"
                )

            if h not in values:
                values.append(h)

        if not values:
            raise ValueError(
                "--fixed_pf was provided but no horizon was parsed"
            )

        if values != sorted(values):
            raise ValueError(
                f"--fixed_pf must be ascending, got {values}"
            )

        fixed_pf = tuple(values)

        print(
            "[PF POLICY] FIXED TRAINED PF =",
            fixed_pf,
        )
    else:
        print(
            "[PF POLICY] DYNAMIC + QUANTIZED "
            "TO TRAINED {1,2,4,8}"
        )

    # =================================================================
    # Runtime warmup
    # =================================================================

    _, warm_indices = (
        choose_scene_indices(
            dataset,
            max(
                args.runtime_warmup_frames
                + 1,
                4,
            ),
        )
    )

    reset_history(model)

    warm_positions = deque(
        maxlen=args.past_length
    )

    for wi in range(
        args.runtime_warmup_frames
    ):
        pos = (
            wi
            %
            len(warm_indices)
        )

        if (
            pos == 0
            and wi > 0
        ):
            reset_history(model)
            warm_positions.clear()

        idx = warm_indices[pos]

        pp = tuple(
            int(x - pos)
            for x in warm_positions
        )

        batch = load_one(
            dataset,
            idx,
        )

        level = (
            "L0"
            if (
                args.pressure_level
                ==
                "L0"
            )
            else (
                args.pressure_level
                if wi % 2
                else "L0"
            )
        )

        run_forward(
            model,
            batch,
            past_offsets=pp,
            future_offsets=(
                1,
                2,
                4,
                8,
            ),
            source_pos=pos,
            start_ms=(
                pos * period_ms
            ),
            period_ms=period_ms,
            estimated_bn_ms=(
                0.5 * period_ms
            ),
            estimated_head_ms=(
                0.5 * period_ms
            ),
            model_stream=model_stream,
            contender=(
                contenders[level]
            ),
            late_skip=False,
        )

        warm_positions.append(pos)

    reset_history(model)

    # Warmup must not leak to planner.
    planner.reset()

    torch.cuda.synchronize()

    # =================================================================
    # Evaluation range + frozen contention trace
    # =================================================================

    n_dataset = len(dataset)

    n_eval = (
        n_dataset
        if args.max_frames == 0
        else min(
            args.max_frames,
            n_dataset,
        )
    )

    eval_indices = list(
        range(n_eval)
    )

    groups = scene_groups(
        dataset,
        eval_indices,
    )

    trace, trace_summary = (
        make_trace(
            groups,
            args.pressure_level,
            args.pressure_fraction,
            args.trace_seed,
        )
    )

    write_trace(
        out_dir
        /
        "contention_trace.csv",
        dataset,
        groups,
        trace,
    )

    timeline = {}

    decision_rows = []

    output_events = []

    processed = 0
    dropped = 0
    misses = 0

    forward_ms_all = []
    bn_ms_all = []
    head_ms_all = []
    queue_wait_all = []
    response_ms_all = []

    pf_hist = Counter()
    pp_hist = Counter()
    level_hist = Counter()

    total_exec = 0
    total_skip = 0

    try:
        for scene_i, (
            scene,
            indices,
        ) in enumerate(
            groups.items(),
            start=1,
        ):
            print(
                f"[scene "
                f"{scene_i}/"
                f"{len(groups)}] "
                f"{scene}"
            )

            reset_history(model)

            history_positions = deque(
                maxlen=args.past_length
            )

            pos = 0

            gpu_free_ms = 0.0

            n = len(indices)

            while pos < n:
                idx = indices[pos]

                _, frame_id, _ = (
                    frame_meta(
                        dataset,
                        idx,
                    )
                )

                arrival_ms = (
                    pos * period_ms
                )

                deadline_ms = (
                    arrival_ms
                    +
                    period_ms
                )

                start_ms = max(
                    arrival_ms,
                    gpu_free_ms,
                )

                queue_wait_ms = (
                    start_ms
                    -
                    arrival_ms
                )

                pp = tuple(
                    int(
                        hp
                        -
                        pos
                    )
                    for hp
                    in history_positions
                )

                if (
                    len(pp)
                    !=
                    len(
                        model
                        .history_feature_queue
                    )
                ):
                    raise RuntimeError(
                        "P^P/history mismatch before job"
                    )

                plan = planner.plan(
                    queue_wait_ms
                )

                if fixed_pf is None:
                    pf = plan["pf"]
                else:
                    pf = fixed_pf

                level = trace[idx]

                batch = load_one(
                    dataset,
                    idx,
                )

                result = run_forward(
                    model,
                    batch,
                    past_offsets=pp,
                    future_offsets=pf,
                    source_pos=pos,
                    start_ms=start_ms,
                    period_ms=period_ms,
                    estimated_bn_ms=(
                        plan[
                            "estimated_bn_ms"
                        ]
                    ),
                    estimated_head_ms=(
                        plan[
                            "estimated_head_ms"
                        ]
                    ),
                    model_stream=(
                        model_stream
                    ),
                    contender=(
                        contenders[level]
                    ),
                    late_skip=(
                        not
                        args.disable_late_skip
                    ),
                )

                planner.observe(
                    result["bn_ms"],
                    result[
                        "head_ms"
                    ].values(),
                )

                finish_ms = (
                    start_ms
                    +
                    result[
                        "total_ms"
                    ]
                )

                response_ms = (
                    finish_ms
                    -
                    arrival_ms
                )

                miss = bool(
                    finish_ms
                    >
                    deadline_ms
                    +
                    1e-9
                )

                # -----------------------------------------------------
                # Post/NMS excluded from timed region.
                # -----------------------------------------------------

                preds = {}

                for horizon in (
                    result[
                        "executed_pf"
                    ]
                ):
                    preds[horizon] = (
                        branch_prediction(
                            model,
                            dataset,
                            batch,
                            result[
                                "branch_data"
                            ][horizon],
                        )
                    )

                # -----------------------------------------------------
                # Output Buffer.
                #
                # First forecast may be made available immediately
                # once computed.
                #
                # Later forecasts are held until their temporal
                # target when they finish early.
                # -----------------------------------------------------

                for j, horizon in enumerate(
                    result[
                        "executed_pf"
                    ]
                ):
                    target_pos = (
                        pos
                        +
                        int(horizon)
                    )

                    if target_pos >= n:
                        continue

                    ready_ms = (
                        start_ms
                        +
                        result[
                            "ready_offset_ms"
                        ][horizon]
                    )

                    target_ms = (
                        target_pos
                        *
                        period_ms
                    )

                    if j == 0:
                        release_ms = ready_ms
                    else:
                        release_ms = max(
                            ready_ms,
                            target_ms,
                        )

                    output_events.append({
                        "scene":
                            scene,

                        "source_pos":
                            pos,

                        "source_index":
                            idx,

                        "horizon":
                            int(horizon),

                        "target_pos":
                            target_pos,

                        "ready_ms":
                            ready_ms,

                        "target_ms":
                            target_ms,

                        "release_ms":
                            release_ms,

                        "anno":
                            preds[horizon],
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
                    for x in result[
                        "executed_pf"
                    ]
                )

                skip_text = "|".join(
                    str(x)
                    for x in result[
                        "skipped_pf"
                    ]
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
                        level,

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
                        plan[
                            "estimated_bn_ms"
                        ],

                    "estimated_head_ms":
                        plan[
                            "estimated_head_ms"
                        ],

                    "actual_bn_ms":
                        result[
                            "bn_ms"
                        ],

                    "total_forward_ms":
                        result[
                            "total_ms"
                        ],

                    "forward_finish_ms":
                        finish_ms,

                    "deadline_miss":
                        int(miss),
                })

                timeline[idx] = {
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

                    "queue_wait_ms":
                        queue_wait_ms,

                    "forward_ms":
                        result[
                            "total_ms"
                        ],

                    "arrival_to_finish_ms":
                        response_ms,

                    "deadline_miss":
                        int(miss),

                    "true_level":
                        level,

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

                misses += int(miss)

                forward_ms_all.append(
                    result[
                        "total_ms"
                    ]
                )

                bn_ms_all.append(
                    result[
                        "bn_ms"
                    ]
                )

                head_ms_all.extend(
                    result[
                        "head_ms"
                    ].values()
                )

                queue_wait_all.append(
                    queue_wait_ms
                )

                response_ms_all.append(
                    response_ms
                )

                pf_hist[pf_text] += 1
                pp_hist[pp_text] += 1
                level_hist[level] += 1

                total_exec += len(
                    result[
                        "executed_pf"
                    ]
                )

                total_skip += len(
                    result[
                        "skipped_pf"
                    ]
                )

                # -----------------------------------------------------
                # Latest-frame-only input mailbox.
                # -----------------------------------------------------

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
                            indices[dp]
                        )

                        _, dfid, _ = (
                            frame_meta(
                                dataset,
                                didx,
                            )
                        )

                        darrival = (
                            dp
                            *
                            period_ms
                        )

                        timeline[didx] = {
                            "global_index":
                                didx,

                            "scene":
                                scene,

                            "local_pos":
                                dp,

                            "frame_id":
                                dfid,

                            "arrival_ms":
                                darrival,

                            "absolute_deadline_ms":
                                (
                                    darrival
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

                            "queue_wait_ms":
                                "",

                            "forward_ms":
                                "",

                            "arrival_to_finish_ms":
                                "",

                            "deadline_miss":
                                "",

                            "true_level":
                                trace[didx],

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

                    pos = next_pos
                else:
                    pos += 1

                gpu_free_ms = finish_ms

                if (
                    processed <= 10
                    or
                    processed % 100
                    ==
                    0
                ):
                    print(
                        f"[{level}] "
                        f"job={processed:04d} "
                        f"{scene}/{frame_id} "
                        f"wait={queue_wait_ms:.3f} "
                        f"PP=[{pp_text}] "
                        f"PF=[{pf_text}] "
                        f"exec=[{exec_text}] "
                        f"skip=[{skip_text}] "
                        f"bn={result['bn_ms']:.3f} "
                        f"total={result['total_ms']:.3f} "
                        f"miss={int(miss)}"
                    )

        # =============================================================
        # Accounting
        # =============================================================

        missing = [
            idx
            for idx in eval_indices
            if idx not in timeline
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
            n_eval
        ):
            raise RuntimeError(
                "sensor accounting mismatch: "
                f"{processed}+{dropped}"
                f"!={n_eval}"
            )

        # =============================================================
        # Output-buffer sensor-time dispatch
        # =============================================================

        scene_events = OrderedDict(
            (
                scene,
                [],
            )
            for scene in groups
        )

        for event in output_events:
            scene_events[
                event["scene"]
            ].append(
                event
            )

        for scene in scene_events:
            scene_events[
                scene
            ].sort(
                key=lambda x: (
                    x["release_ms"],
                    x["source_pos"],
                    x["horizon"],
                )
            )

        aligned = {}

        dispatch_rows = []

        dispatch_hist = Counter()

        for scene, indices in (
            groups.items()
        ):
            events = (
                scene_events[
                    scene
                ]
            )

            active = []

            ptr = 0

            for query_pos, idx in enumerate(
                indices
            ):
                query_ms = (
                    query_pos
                    *
                    period_ms
                )

                while (
                    ptr < len(events)
                    and
                    events[ptr][
                        "release_ms"
                    ]
                    <=
                    query_ms
                    +
                    1e-9
                ):
                    active.append(
                        events[ptr]
                    )
                    ptr += 1

                _, fid, next_fid = (
                    frame_meta(
                        dataset,
                        idx,
                    )
                )

                if active:
                    chosen = min(
                        active,
                        key=lambda x: (
                            abs(
                                x[
                                    "target_pos"
                                ]
                                -
                                query_pos
                            ),
                            -
                            x[
                                "release_ms"
                            ],
                            -
                            x[
                                "source_pos"
                            ],
                        ),
                    )

                    pred = chosen[
                        "anno"
                    ]

                    dispatch_hist[
                        int(
                            chosen[
                                "horizon"
                            ]
                        )
                    ] += 1

                else:
                    chosen = None
                    pred = None

                aligned[idx] = copy_det(
                    pred,
                    scene,
                    fid,
                    next_fid,
                )

                dispatch_rows.append({
                    "scene":
                        scene,

                    "query_pos":
                        query_pos,

                    "global_index":
                        idx,

                    "frame_id":
                        fid,

                    "query_ms":
                        query_ms,

                    "has_prediction":
                        int(
                            chosen
                            is not None
                        ),

                    "source_pos":
                        (
                            ""
                            if chosen is None
                            else chosen[
                                "source_pos"
                            ]
                        ),

                    "target_pos":
                        (
                            ""
                            if chosen is None
                            else chosen[
                                "target_pos"
                            ]
                        ),

                    "horizon":
                        (
                            ""
                            if chosen is None
                            else chosen[
                                "horizon"
                            ]
                        ),

                    "release_ms":
                        (
                            ""
                            if chosen is None
                            else chosen[
                                "release_ms"
                            ]
                        ),
                })

        # =============================================================
        # Streaming KITTI AP
        # =============================================================

        gt_annos = [
            copy.deepcopy(
                dataset
                .kitti_infos[idx]
                ["infos"]
                ["token"]
                ["annos"]
            )
            for idx
            in eval_indices
        ]

        det_annos = [
            aligned[idx]
            for idx in eval_indices
        ]

        result_str, ap_dict = (
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

        # =============================================================
        # Save
        # =============================================================

        if decision_rows:
            with (
                out_dir
                /
                "transtreaming_decisions.csv"
            ).open(
                "w",
                newline="",
            ) as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=list(
                        decision_rows[
                            0
                        ].keys()
                    ),
                )

                w.writeheader()
                w.writerows(
                    decision_rows
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
                w = csv.DictWriter(
                    f,
                    fieldnames=list(
                        dispatch_rows[
                            0
                        ].keys()
                    ),
                )

                w.writeheader()
                w.writerows(
                    dispatch_rows
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
            "queue_wait_ms",
            "forward_ms",
            "arrival_to_finish_ms",
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
            w = csv.DictWriter(
                f,
                fieldnames=timeline_fields,
            )

            w.writeheader()

            for idx in eval_indices:
                w.writerow(
                    timeline[idx]
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
            "method":
                "Transtreaming-style StreamDSGN V2",

            "selected_epoch":
                13,

            "checkpoint":
                args.ckpt,

            "input_hz":
                args.input_hz,

            "period_ms":
                period_ms,

            "timing_scope":
                "current_backbone_plus_TAT_plus_"
                "executed_shared_detection_paths",

            "history_policy":
                "completed_processed_frames_only",

            "past_offsets":
                "actual_processed_sensor_offsets",

            "input_mailbox":
                "latest_frame_only",

            "future_planner":
                (
                    "fixed_trained_proposals"
                    if fixed_pf is not None
                    else
                    "causal_Transtreaming_style_"
                    "quantized_to_trained_horizons"
                ),

            "fixed_pf":
                (
                    None
                    if fixed_pf is None
                    else list(fixed_pf)
                ),

            "trained_future_horizons":
                list(
                    TRAINED_FUTURE_HORIZONS
                ),

            "late_skip":
                bool(
                    not
                    args.disable_late_skip
                ),

            "output_buffer":
                "quick_first_then_target_release",

            "sensor_frames":
                n_eval,

            "processed_frames":
                processed,

            "dropped_frames":
                dropped,

            "drop_rate":
                (
                    dropped
                    /
                    max(
                        n_eval,
                        1,
                    )
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
                    forward_ms_all
                ),

            "backbone_tat_latency":
                stats(
                    bn_ms_all
                ),

            "shared_head_latency":
                stats(
                    head_ms_all
                ),

            "queue_wait":
                stats(
                    queue_wait_all
                ),

            "arrival_to_finish":
                stats(
                    response_ms_all
                ),

            "proposal_histogram":
                dict(pf_hist),

            "past_offset_histogram":
                dict(pp_hist),

            "dispatch_horizon_histogram":
                {
                    str(k):
                        int(v)
                    for k, v
                    in sorted(
                        dispatch_hist.items()
                    )
                },

            "executed_future_outputs":
                int(total_exec),

            "skipped_future_outputs":
                int(total_skip),

            "mean_outputs_per_processed_job":
                (
                    total_exec
                    /
                    max(
                        processed,
                        1,
                    )
                ),

            "processed_level_histogram":
                dict(level_hist),

            "trace_summary":
                trace_summary,

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

        (
            out_dir
            /
            "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            )
            +
            "\n"
        )

        latency = stats(
            forward_ms_all
        )

        print()
        print("=" * 110)
        print(
            "TRANSTREAMING V2 STREAMING RESULT"
        )
        print("=" * 110)

        print(
            f"checkpoint epoch     : 13"
        )

        print(
            f"sensor frames        : {n_eval}"
        )

        print(
            f"processed / dropped  : "
            f"{processed} / {dropped}"
        )

        print(
            f"drop rate            : "
            f"{100*summary['drop_rate']:.3f}%"
        )

        print(
            f"deadline miss rate   : "
            f"{100*summary['deadline_miss_rate']:.3f}%"
        )

        print(
            "forward p50/p90/p99  : "
            f"{latency['p50_ms']:.4f} / "
            f"{latency['p90_ms']:.4f} / "
            f"{latency['p99_ms']:.4f} ms"
        )

        print(
            "sAP Moderate R40     : "
            f"Car={car:.4f} "
            f"Ped={ped:.4f} "
            f"Cyc={cyc:.4f} "
            f"Macro={macro:.4f}"
        )

        print(
            f"PF histogram         : "
            f"{dict(pf_hist)}"
        )

        print(
            f"dispatch horizons    : "
            f"{dict(dispatch_hist)}"
        )

        print(
            f"executed outputs     : "
            f"{total_exec}"
        )

        print(
            f"skipped outputs      : "
            f"{total_skip}"
        )

        print(
            f"summary              : "
            f"{out_dir/'summary.json'}"
        )

        print("=" * 110)

    finally:
        model.generate_recall_record = (
            original_recall
        )


if __name__ == "__main__":
    main()
