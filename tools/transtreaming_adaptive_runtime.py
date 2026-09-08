#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch


# ============================================================================
# Transtreaming runtime estimator
# ============================================================================

class DecayRuntimeEstimator:
    """
    Same exponential weighting structure as official Transtreaming:

        [0.5, 0.25, 0.125, 0.0625, 0.0625]

    New observation occupies index 0.
    """

    def __init__(
        self,
        initial_ms: float,
        window_size: int = 5,
    ):
        if window_size < 2:
            raise ValueError(
                "window_size must be >= 2"
            )

        self.initial_ms = float(
            initial_ms
        )

        self.window_size = int(
            window_size
        )

        self.reset()

    def reset(self):
        self.values = [
            self.initial_ms
            for _ in range(
                self.window_size
            )
        ]

    def observe(
        self,
        value_ms: float,
    ):
        value_ms = float(
            value_ms
        )

        if not math.isfinite(
            value_ms
        ):
            return

        if value_ms <= 0:
            return

        self.values = [
            value_ms,
            *self.values[
                :self.window_size - 1
            ],
        ]

    def estimate(self) -> float:
        weights = []

        for i in range(
            self.window_size
        ):
            if (
                i
                <
                self.window_size - 1
            ):
                weight = (
                    0.5
                    **
                    (i + 1)
                )
            else:
                weight = (
                    0.5
                    **
                    (
                        self.window_size
                        - 1
                    )
                )

            weights.append(
                weight
            )

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


@dataclass(frozen=True)
class AdaptivePlan:
    past_offsets: Tuple[int, ...]
    future_offsets: Tuple[int, ...]

    estimated_bn_ms: float
    estimated_head_ms: float
    estimated_other_ms: float

    estimated_first_ms: float
    estimated_delta_ms: float


class TranstreamingAdaptivePlanner:
    """
    Discrete-frame adaptation of official Transtreaming AdaptiveStrategy.

    Official structure:

        t_first =
            startup_delay
            + t_bn
            + t_h

        t_delta =
            (
                (future_length + 1) * t_h
                + t_bn
                + t_other
            )
            / future_length

        TF_i =
            ceil(
                t_first
                + i * t_delta
            )

    Here all times are milliseconds and are converted
    to sensor-frame offsets using period_ms.

    Forward-only protocol:
        t_other = 0

    because data loading, H2D, NMS and evaluation
    are explicitly excluded.
    """

    def __init__(
        self,
        period_ms: float,
        future_length: int = 4,
        max_future: int = 8,
        window_size: int = 5,
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

        if self.period_ms <= 0:
            raise ValueError(
                "period_ms must be > 0"
            )

        if self.future_length <= 0:
            raise ValueError(
                "future_length must be > 0"
            )

        if self.max_future <= 0:
            raise ValueError(
                "max_future must be > 0"
            )

        # Official implementation initializes
        # backbone/neck and head estimators at
        # 0.5 frame-time units.
        self.bn = DecayRuntimeEstimator(
            initial_ms=(
                0.5
                *
                self.period_ms
            ),
            window_size=window_size,
        )

        self.head = DecayRuntimeEstimator(
            initial_ms=(
                0.5
                *
                self.period_ms
            ),
            window_size=window_size,
        )

    def reset(self):
        self.bn.reset()
        self.head.reset()

    def observe(
        self,
        bn_ms: float,
        head_times_ms: Iterable[float],
    ):
        self.bn.observe(
            bn_ms
        )

        for value in (
            head_times_ms
        ):
            self.head.observe(
                float(value)
            )

    def plan(
        self,
        queue_wait_ms: float,
        past_offsets: Sequence[int],
    ) -> AdaptivePlan:
        queue_wait_ms = max(
            0.0,
            float(queue_wait_ms),
        )

        t_bn = self.bn.estimate()
        t_h = self.head.estimate()

        # Forward-only protocol.
        t_other = 0.0

        t_first = (
            queue_wait_ms
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

        future = []

        for i in range(
            self.future_length
        ):
            estimated_ms = (
                t_first
                +
                i
                *
                t_delta
            )

            horizon = int(
                math.ceil(
                    estimated_ms
                    /
                    self.period_ms
                )
            )

            horizon = max(
                1,
                min(
                    self.max_future,
                    horizon,
                ),
            )

            if (
                horizon
                not in future
            ):
                future.append(
                    horizon
                )

        if len(future) == 0:
            future = [1]

        return AdaptivePlan(
            past_offsets=tuple(
                int(x)
                for x in past_offsets
            ),

            future_offsets=tuple(
                future
            ),

            estimated_bn_ms=float(
                t_bn
            ),

            estimated_head_ms=float(
                t_h
            ),

            estimated_other_ms=0.0,

            estimated_first_ms=float(
                t_first
            ),

            estimated_delta_ms=float(
                t_delta
            ),
        )


# ============================================================================
# Forward runtime
# ============================================================================

@dataclass
class TranstreamingRuntimeResult:
    bn_ms: float

    head_ms: Dict[
        int,
        float,
    ]

    ready_offset_ms: Dict[
        int,
        float,
    ]

    total_ms: float

    executed_pf: Tuple[
        int,
        ...
    ]

    skipped_pf: Tuple[
        int,
        ...
    ]

    branch_data: Dict[
        int,
        dict,
    ]


def reset_model_history(
    model,
):
    q = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if q is not None:
        q.clear()


def _amp_enabled(
    model,
) -> bool:
    d = getattr(
        model,
        "use_amp_dict",
        {},
    )

    if isinstance(
        d,
        dict,
    ):
        return bool(
            d.get(
                "TEST",
                False,
            )
        )

    return False


def transtreaming_forward_no_post(
    model,
    batch,
    past_offsets: Sequence[int],
    future_offsets: Sequence[int],
    model_stream,
    contender=None,

    *,
    source_local_pos: int,
    job_start_ms: float,
    period_ms: float,

    estimated_bn_ms: float,
    estimated_head_ms: float,

    enable_late_skip: bool = True,
):
    """
    Timed forward-only Transtreaming inference.

    Scope:
        current stereo feature extractor
        + TAT/RTPE
        + selected future VANBackbone/StreamDetHead paths

    Excludes:
        dataloader
        H2D
        post/NMS
        evaluation

    Historical feature is appended exactly once,
    after the processed job completes.
    """

    future_offsets = tuple(
        int(x)
        for x in future_offsets
    )

    past_offsets = tuple(
        int(x)
        for x in past_offsets
    )

    if not future_offsets:
        raise ValueError(
            "future_offsets cannot be empty"
        )

    if (
        len(set(future_offsets))
        !=
        len(future_offsets)
    ):
        raise ValueError(
            f"duplicate PF: "
            f"{future_offsets}"
        )

    if any(
        x <= 0
        for x in future_offsets
    ):
        raise ValueError(
            f"future offsets must be >0: "
            f"{future_offsets}"
        )

    if any(
        x >= 0
        for x in past_offsets
    ):
        raise ValueError(
            f"past offsets must be <0: "
            f"{past_offsets}"
        )

    cur_data = batch[
        "token"
    ]

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
                enabled=(
                    _amp_enabled(
                        model
                    )
                ),
            ),
        ):
            start_event = (
                torch.cuda.Event(
                    enable_timing=True
                )
            )

            bn_end_event = (
                torch.cuda.Event(
                    enable_timing=True
                )
            )

            start_event.record()

            # ========================================================
            # Backbone / current BEV extraction
            # ========================================================

            for module in (
                model.feature_extractor
            ):
                cur_data = module(
                    cur_data
                )

            cur_data[
                "history_features"
            ] = (
                model
                .history_feature_queue
            )

            history_features = None

            if (
                model.history_tag
                is not None
            ):
                history_features = {}

                for feature_name in (
                    model
                    .history_features_name
                ):
                    history_features[
                        feature_name
                    ] = (
                        cur_data[
                            feature_name
                        ].clone()
                    )

            # Exact actual temporal positions.
            cur_data[
                "transtreaming_past_offsets"
            ] = list(
                past_offsets
            )

            cur_data[
                "transtreaming_future_offsets"
            ] = list(
                future_offsets
            )

            # ========================================================
            # TAT / RTPE neck
            # ========================================================

            for module in (
                model.fusion_module
            ):
                cur_data = module(
                    cur_data
                )

            future_bev = cur_data[
                "transtreaming_future_spatial_features"
            ]

            if future_bev.ndim != 5:
                raise RuntimeError(
                    "expected future BEV "
                    "[B,T,C,H,W], got "
                    f"{future_bev.shape}"
                )

            if (
                future_bev.shape[1]
                !=
                len(future_offsets)
            ):
                raise RuntimeError(
                    "PF/future-BEV count mismatch: "
                    f"{len(future_offsets)} vs "
                    f"{future_bev.shape[1]}"
                )

            bn_end_event.record()

            # ========================================================
            # One shared post-TAT detection path.
            #
            # Same VANBackbone + same StreamDetHead is reused for
            # every selected future feature.
            # ========================================================

            branch_data = {}
            head_events = {}

            executed = []
            skipped = []

            previous_event = (
                bn_end_event
            )

            # Causal estimate of how far this job has progressed.
            #
            # We deliberately do NOT cuda.synchronize() between
            # future heads because that would inject host stalls into
            # the frozen forward-only critical path.
            estimated_elapsed_ms = max(
                0.0,
                float(
                    estimated_bn_ms
                ),
            )

            for order, horizon in enumerate(
                future_offsets
            ):
                # Official strategy always executes first future output.
                if (
                    order > 0
                    and
                    enable_late_skip
                ):
                    estimated_now_ms = (
                        float(
                            job_start_ms
                        )
                        +
                        estimated_elapsed_ms
                    )

                    target_ms = (
                        (
                            int(
                                source_local_pos
                            )
                            +
                            int(
                                horizon
                            )
                        )
                        *
                        float(
                            period_ms
                        )
                    )

                    # Source-code-faithful analogue of:
                    #
                    #   if now > target + t_h:
                    #       skip
                    #
                    if (
                        estimated_now_ms
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

                bd = dict(
                    cur_data
                )

                bd[
                    "spatial_features"
                ] = future_bev[
                    :,
                    order,
                ]

                # Prevent outputs from one future branch
                # leaking into another branch.
                for key in (
                    "spatial_features_2d",
                    "batch_cls_preds",
                    "batch_box_preds",
                    "cls_preds_normalized",
                    "reg_features",
                ):
                    bd.pop(
                        key,
                        None,
                    )

                for module in (
                    model.after_fusion_blocks
                ):
                    bd = module(
                        bd
                    )

                ev = torch.cuda.Event(
                    enable_timing=True
                )

                ev.record()

                executed.append(
                    horizon
                )

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

                estimated_elapsed_ms += max(
                    0.0,
                    float(
                        estimated_head_ms
                    ),
                )

            if not executed:
                raise RuntimeError(
                    "no future output executed"
                )

            final_event = (
                head_events[
                    executed[-1]
                ][1]
            )

        final_event.synchronize()

        bn_ms = float(
            start_event.elapsed_time(
                bn_end_event
            )
        )

        head_ms = {}
        ready_offsets = {}

        for horizon in executed:
            prev_ev, end_ev = (
                head_events[
                    horizon
                ]
            )

            head_ms[
                horizon
            ] = float(
                prev_ev.elapsed_time(
                    end_ev
                )
            )

            ready_offsets[
                horizon
            ] = float(
                start_event.elapsed_time(
                    end_ev
                )
            )

        total_ms = float(
            start_event.elapsed_time(
                final_event
            )
        )

    finally:
        if launched:
            contender.finish()

    # ================================================================
    # Historical feature buffer:
    #
    # current backbone feature becomes available to future jobs
    # only once this processed job finishes.
    # ================================================================

    if (
        model.history_tag
        is not None
        and
        history_features
        is not None
    ):
        model.history_feature_queue.append(
            (
                cur_data[
                    "this_sample_idx"
                ],
                history_features,
            )
        )

    batch[
        "token"
    ] = cur_data

    return TranstreamingRuntimeResult(
        bn_ms=bn_ms,
        head_ms=head_ms,
        ready_offset_ms=(
            ready_offsets
        ),
        total_ms=total_ms,
        executed_pf=tuple(
            executed
        ),
        skipped_pf=tuple(
            skipped
        ),
        branch_data=branch_data,
    )
