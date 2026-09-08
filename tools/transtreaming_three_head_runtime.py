#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch


def _load_checkpoint_model_state(path: str):
    ckpt = torch.load(
        str(path),
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(ckpt, dict):
        raise RuntimeError(
            f"checkpoint is not dict: {path}"
        )

    for key in (
        "model_state",
        "state_dict",
        "model",
    ):
        value = ckpt.get(key)

        if isinstance(value, dict):
            return value

    if all(
        torch.is_tensor(v)
        for v in ckpt.values()
    ):
        return ckpt

    raise RuntimeError(
        f"cannot find model state: {path}"
    )


def extract_dense_head_state(
    path: str,
):
    state = (
        _load_checkpoint_model_state(
            path
        )
    )

    dense = {}

    for key, value in state.items():
        k = str(key)

        if k.startswith("module."):
            k = k[len("module."):]

        if not k.startswith(
            "dense_head."
        ):
            continue

        k = k[len("dense_head."):]

        dense[k] = value

    if not dense:
        raise RuntimeError(
            f"checkpoint has no "
            f"dense_head.* tensors: {path}"
        )

    return dense


class DecayWindow:
    """
    Transtreaming-style exponentially decayed
    runtime estimator.

    window=5:
        0.5
        0.25
        0.125
        0.0625
        0.0625
    """

    def __init__(
        self,
        window_size: int = 5,
    ):
        if window_size < 2:
            raise ValueError(
                "window_size must be >= 2"
            )

        self.window_size = int(
            window_size
        )

        self.values = deque(
            maxlen=self.window_size
        )

    def reset(self):
        self.values.clear()

    def observe(
        self,
        value_ms: float,
    ):
        value_ms = float(
            value_ms
        )

        if value_ms <= 0:
            return

        # Initialize the complete history using
        # the first real formal observation.
        if not self.values:
            for _ in range(
                self.window_size
            ):
                self.values.append(
                    value_ms
                )

            return

        self.values.appendleft(
            value_ms
        )

    def estimate(
        self,
    ) -> Optional[float]:
        if not self.values:
            return None

        vals = list(
            self.values
        )

        weights = []

        for i in range(
            self.window_size
        ):
            if i < (
                self.window_size - 1
            ):
                weight = (
                    0.5 ** (i + 1)
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
                v * w
                for v, w
                in zip(
                    vals,
                    weights,
                )
            )
        )


@dataclass(frozen=True)
class TranstreamingPlan:
    horizons: Tuple[int, ...]

    estimated_shared_ms: (
        Optional[float]
    )

    estimated_head_ms: (
        Optional[float]
    )

    estimated_first_ready_ms: (
        Optional[float]
    )

    estimated_delta_ms: (
        Optional[float]
    )


class TranstreamingPlanner:
    """
    Discrete H1/H2/H3 adaptation of
    Transtreaming AdaptiveStrategy.

    Current proposal uses ONLY:

        current known queue wait
        previous shared runtimes
        previous head runtimes

    Current actual runtime is observed only
    after current forward has completed.
    """

    def __init__(
        self,
        future_length: int = 3,
        max_horizon: int = 3,
        window_size: int = 5,
    ):
        if future_length <= 0:
            raise ValueError(
                "future_length must be > 0"
            )

        if max_horizon != 3:
            raise ValueError(
                "this baseline expects "
                "H1/H2/H3"
            )

        self.future_length = int(
            future_length
        )

        self.max_horizon = int(
            max_horizon
        )

        self.shared_runtime = (
            DecayWindow(
                window_size
            )
        )

        self.head_runtime = (
            DecayWindow(
                window_size
            )
        )

    def reset(self):
        self.shared_runtime.reset()
        self.head_runtime.reset()

    def observe(
        self,
        shared_ms: float,
        head_times_ms: Iterable[float],
    ):
        self.shared_runtime.observe(
            shared_ms
        )

        for value in (
            head_times_ms
        ):
            self.head_runtime.observe(
                float(value)
            )

    def plan(
        self,
        queue_wait_ms: float,
        period_ms: float,
    ) -> TranstreamingPlan:
        if period_ms <= 0:
            raise ValueError(
                "period_ms must be > 0"
            )

        queue_wait_ms = max(
            0.0,
            float(queue_wait_ms),
        )

        shared = (
            self.shared_runtime
            .estimate()
        )

        head = (
            self.head_runtime
            .estimate()
        )

        # Formal cold start.
        # CUDA warmup has already happened, but its
        # timing never leaks into this planner.
        if (
            shared is None
            or
            head is None
        ):
            return TranstreamingPlan(
                horizons=(1,),
                estimated_shared_ms=(
                    shared
                ),
                estimated_head_ms=(
                    head
                ),
                estimated_first_ready_ms=None,
                estimated_delta_ms=None,
            )

        # Official Transtreaming-style structure:
        #
        # t_first =
        #   current_delay + backbone/neck + head
        #
        # t_delta =
        #   ((future_length + 1) * head
        #    + backbone/neck)
        #   / future_length
        #
        # In our forward-only protocol:
        #
        # current_delay = queue_wait
        # backbone/neck = shared K3 trunk
        # head = one future dense head

        t_first = (
            queue_wait_ms
            +
            shared
            +
            head
        )

        t_delta = (
            (
                (
                    self.future_length
                    + 1
                )
                *
                head
                +
                shared
            )
            /
            self.future_length
        )

        proposals = []

        for i in range(
            self.future_length
        ):
            estimated_ready = (
                t_first
                +
                i * t_delta
            )

            horizon = int(
                math.ceil(
                    estimated_ready
                    /
                    period_ms
                )
            )

            horizon = max(
                1,
                min(
                    self.max_horizon,
                    horizon,
                ),
            )

            if (
                horizon
                not in proposals
            ):
                proposals.append(
                    horizon
                )

        if not proposals:
            proposals = [1]

        return TranstreamingPlan(
            horizons=tuple(
                proposals
            ),
            estimated_shared_ms=float(
                shared
            ),
            estimated_head_ms=float(
                head
            ),
            estimated_first_ready_ms=float(
                t_first
            ),
            estimated_delta_ms=float(
                t_delta
            ),
        )


class TranstreamingThreeHeadBank:
    """
    Shared K3 StreamDSGN trunk.

      H1 = existing FULL dense_head
      H2 = trained next2 dense_head
      H3 = trained next3 dense_head

    Shared network runs once.
    Selected heads run sequentially.
    """

    def __init__(
        self,
        model,
        h2_ckpt: str,
        h3_ckpt: str,
    ):
        self.model = model

        if not hasattr(
            model,
            "dense_head",
        ):
            raise RuntimeError(
                "model has no dense_head"
            )

        if not hasattr(
            model,
            "after_fusion_blocks",
        ):
            raise RuntimeError(
                "model has no "
                "after_fusion_blocks"
            )

        positions = [
            i
            for i, module
            in enumerate(
                model
                .after_fusion_blocks
            )
            if (
                module
                is model.dense_head
            )
        ]

        if len(positions) != 1:
            raise RuntimeError(
                "dense_head must appear "
                "exactly once in "
                "after_fusion_blocks; "
                f"positions={positions}"
            )

        self.dense_index = int(
            positions[0]
        )

        self.prefix_modules = list(
            model
            .after_fusion_blocks[
                :self.dense_index
            ]
        )

        self.suffix_modules = list(
            model
            .after_fusion_blocks[
                self.dense_index + 1:
            ]
        )

        h1 = model.dense_head
        h2 = copy.deepcopy(h1)
        h3 = copy.deepcopy(h1)

        state_h2 = (
            extract_dense_head_state(
                h2_ckpt
            )
        )

        state_h3 = (
            extract_dense_head_state(
                h3_ckpt
            )
        )

        self._validate(
            reference=h1,
            candidate=state_h2,
            label="H2",
            ckpt=h2_ckpt,
        )

        self._validate(
            reference=h1,
            candidate=state_h3,
            label="H3",
            ckpt=h3_ckpt,
        )

        h2.load_state_dict(
            state_h2,
            strict=True,
        )

        h3.load_state_dict(
            state_h3,
            strict=True,
        )

        h2.eval()
        h3.eval()

        model.add_module(
            "transtreaming_head_h2",
            h2,
        )

        model.add_module(
            "transtreaming_head_h3",
            h3,
        )

        self.heads = {
            1: h1,
            2: h2,
            3: h3,
        }

    @staticmethod
    def _validate(
        reference,
        candidate,
        label,
        ckpt,
    ):
        ref = (
            reference
            .state_dict()
        )

        missing = sorted(
            set(ref)
            -
            set(candidate)
        )

        unexpected = sorted(
            set(candidate)
            -
            set(ref)
        )

        if missing or unexpected:
            raise RuntimeError(
                f"{label} key mismatch\n"
                f"ckpt={ckpt}\n"
                f"missing={missing}\n"
                f"unexpected={unexpected}"
            )

        bad_shapes = []

        for key in ref:
            if (
                tuple(
                    ref[key].shape
                )
                !=
                tuple(
                    candidate[
                        key
                    ].shape
                )
            ):
                bad_shapes.append(
                    (
                        key,
                        tuple(
                            ref[key]
                            .shape
                        ),
                        tuple(
                            candidate[
                                key
                            ].shape
                        ),
                    )
                )

        if bad_shapes:
            raise RuntimeError(
                f"{label} shape mismatch: "
                f"{bad_shapes}"
            )


@dataclass
class TranstreamingForwardResult:
    shared_data: dict

    head_data: Dict[
        int,
        dict,
    ]

    shared_ms: float

    head_ms: Dict[
        int,
        float,
    ]

    ready_offset_ms: Dict[
        int,
        float,
    ]

    total_ms: float

    executed_horizons: (
        Tuple[int, ...]
    )


def _test_amp_enabled(
    model,
) -> bool:
    value = getattr(
        model,
        "use_amp_dict",
        None,
    )

    if isinstance(
        value,
        dict,
    ):
        return bool(
            value.get(
                "TEST",
                False,
            )
        )

    return False


def transtreaming_forward_no_post(
    model,
    bank: TranstreamingThreeHeadBank,
    batch,
    planned_horizons: Sequence[int],
    model_stream,
    contender=None,
):
    horizons = tuple(
        int(x)
        for x in planned_horizons
    )

    if not horizons:
        raise ValueError(
            "planned_horizons empty"
        )

    if (
        len(set(horizons))
        !=
        len(horizons)
    ):
        raise ValueError(
            f"duplicate horizons: "
            f"{horizons}"
        )

    for horizon in horizons:
        if horizon not in (
            1,
            2,
            3,
        ):
            raise ValueError(
                f"invalid horizon: "
                f"{horizon}"
            )

    cur_data = batch["token"]

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
                    _test_amp_enabled(
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

            shared_end_event = (
                torch.cuda.Event(
                    enable_timing=True
                )
            )

            start_event.record()

            # =====================================================
            # Shared StreamDSGN feature extractor
            # =====================================================

            for module in (
                model.feature_extractor
            ):
                cur_data = module(
                    cur_data
                )

            # =====================================================
            # Processed-frame history only
            # =====================================================

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

            # =====================================================
            # Shared temporal fusion
            # =====================================================

            for module in (
                model.fusion_module
            ):
                cur_data = module(
                    cur_data
                )

            # =====================================================
            # Shared modules before dense head
            # =====================================================

            for module in (
                bank.prefix_modules
            ):
                cur_data = module(
                    cur_data
                )

            shared_end_event.record()

            head_data = {}
            head_events = {}

            previous_event = (
                shared_end_event
            )

            # =====================================================
            # Multiple selected future heads
            # =====================================================

            for horizon in horizons:
                # Dense head mutates batch dict keys.
                # Tensor storage remains shared.
                branch_data = dict(
                    cur_data
                )

                branch_data = (
                    bank.heads[
                        horizon
                    ](
                        branch_data
                    )
                )

                for module in (
                    bank.suffix_modules
                ):
                    branch_data = module(
                        branch_data
                    )

                end_event = (
                    torch.cuda.Event(
                        enable_timing=True
                    )
                )

                end_event.record()

                head_data[
                    horizon
                ] = branch_data

                head_events[
                    horizon
                ] = (
                    previous_event,
                    end_event,
                )

                previous_event = (
                    end_event
                )

            final_event = (
                head_events[
                    horizons[-1]
                ][1]
            )

        # Exactly one synchronization after
        # selected forward operators.
        final_event.synchronize()

        shared_ms = float(
            start_event.elapsed_time(
                shared_end_event
            )
        )

        head_ms = {}
        ready_offsets = {}

        for horizon in horizons:
            prev_event, end_event = (
                head_events[
                    horizon
                ]
            )

            head_ms[
                horizon
            ] = float(
                prev_event.elapsed_time(
                    end_event
                )
            )

            ready_offsets[
                horizon
            ] = float(
                start_event.elapsed_time(
                    end_event
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

    # History only becomes available after
    # the complete processed job.
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

    batch["token"] = cur_data

    return TranstreamingForwardResult(
        shared_data=cur_data,
        head_data=head_data,
        shared_ms=shared_ms,
        head_ms=head_ms,
        ready_offset_ms=(
            ready_offsets
        ),
        total_ms=total_ms,
        executed_horizons=(
            horizons
        ),
    )
