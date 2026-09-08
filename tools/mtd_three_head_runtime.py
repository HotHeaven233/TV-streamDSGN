#!/usr/bin/env python3
from __future__ import annotations

import copy
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch


@dataclass(frozen=True)
class MTDDecision:
    branch_step: int
    estimated_forward_ms: float
    queue_wait_ms: float
    estimated_response_ms: float
    delay_slots: int
    runtime_history_count: int


class MTDDelayAnalyzer:
    """
    Causal MTD-style Delay Analysis Module.

    IMPORTANT:
    - decision happens BEFORE current forward
    - never uses current-frame true runtime
    - only uses:
          1) current known queue wait
          2) previous one/two measured forward runtimes

    Following the official MTD implementation idea, when two previous
    runtimes exist, use the smaller one as the estimate of the next
    inference runtime.

        estimated_response
            = current_queue_wait
            + min(last_forward, prev_forward)

        delay_slots
            = floor(estimated_response / frame_period)

        H = delay_slots + 1

    StreamDSGN itself predicts next frame, therefore:
        delay 0 -> H1 -> next
        delay 1 -> H2 -> next2
        delay >=2 -> H3 -> next3
    """

    def __init__(
        self,
        period_ms: float,
        max_horizon: int = 3,
    ):
        self.period_ms = float(period_ms)
        self.max_horizon = int(max_horizon)

        if self.period_ms <= 0:
            raise ValueError("period_ms must be > 0")

        if self.max_horizon != 3:
            raise ValueError(
                "This baseline is defined for exactly three heads"
            )

        self.forward_history = deque(maxlen=2)

    def reset(self):
        self.forward_history.clear()

    def observe(self, forward_ms: float):
        x = float(forward_ms)

        if not math.isfinite(x) or x <= 0:
            raise ValueError(
                f"invalid observed forward runtime: {forward_ms}"
            )

        self.forward_history.append(x)

    def estimate_forward_ms(self) -> float:
        if len(self.forward_history) == 0:
            # Cold start:
            # no future/runtime information is available.
            # Select H1 for the first formal frame.
            return 0.0

        if len(self.forward_history) == 1:
            return float(self.forward_history[-1])

        return float(
            min(self.forward_history[-1], self.forward_history[-2])
        )

    def branch_for_response_ms(
        self,
        response_ms: float,
    ) -> int:
        response_ms = max(0.0, float(response_ms))

        delay_slots = int(
            math.floor(
                (response_ms + 1e-9)
                /
                self.period_ms
            )
        )

        return max(
            1,
            min(
                self.max_horizon,
                delay_slots + 1,
            ),
        )

    def select(
        self,
        queue_wait_ms: float,
    ) -> MTDDecision:
        queue_wait_ms = max(
            0.0,
            float(queue_wait_ms),
        )

        estimated_forward_ms = (
            self.estimate_forward_ms()
        )

        estimated_response_ms = (
            queue_wait_ms
            +
            estimated_forward_ms
        )

        delay_slots = int(
            math.floor(
                (estimated_response_ms + 1e-9)
                /
                self.period_ms
            )
        )

        branch_step = max(
            1,
            min(
                self.max_horizon,
                delay_slots + 1,
            ),
        )

        return MTDDecision(
            branch_step=branch_step,
            estimated_forward_ms=estimated_forward_ms,
            queue_wait_ms=queue_wait_ms,
            estimated_response_ms=estimated_response_ms,
            delay_slots=delay_slots,
            runtime_history_count=len(
                self.forward_history
            ),
        )


def _load_checkpoint_model_state(
    filename,
):
    filename = Path(filename)

    if not filename.exists():
        raise FileNotFoundError(filename)

    try:
        ckpt = torch.load(
            filename,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        ckpt = torch.load(
            filename,
            map_location="cpu",
        )

    if isinstance(ckpt, dict):
        if "model_state" in ckpt:
            state = ckpt["model_state"]
        elif "state_dict" in ckpt:
            state = ckpt["state_dict"]
        else:
            state = ckpt
    else:
        raise RuntimeError(
            f"unsupported checkpoint type: {type(ckpt)}"
        )

    if not isinstance(state, dict):
        raise RuntimeError(
            f"checkpoint has no model state: {filename}"
        )

    return state


def extract_dense_head_state(
    filename,
):
    state = _load_checkpoint_model_state(
        filename
    )

    result = {}

    for name, value in state.items():
        key = str(name)

        if key.startswith("module."):
            key = key[len("module."):]

        if not key.startswith(
            "dense_head."
        ):
            continue

        result[
            key[len("dense_head."):]
        ] = value

    if len(result) == 0:
        raise RuntimeError(
            "No dense_head.* parameters found in "
            f"{filename}"
        )

    return result


class MTDThreeHeadBank:
    """
    Three real neural prediction heads sharing one StreamDSGN trunk.

        H1 = existing K3 StreamDSGN dense_head (next)
        H2 = separately trained dense_head (next2)
        H3 = separately trained dense_head (next3)

    Only ONE head is executed for each processed frame.
    """

    def __init__(
        self,
        model,
        h2_ckpt,
        h3_ckpt,
        logger=None,
    ):
        if getattr(
            model,
            "dense_head",
            None,
        ) is None:
            raise RuntimeError(
                "MTD requires model.dense_head"
            )

        if not hasattr(
            model,
            "after_fusion_blocks",
        ):
            raise RuntimeError(
                "MTD requires after_fusion_blocks"
            )

        dense_head_hits = sum(
            int(module is model.dense_head)
            for module
            in model.after_fusion_blocks
        )

        if dense_head_hits != 1:
            raise RuntimeError(
                "Expected model.dense_head to appear "
                "exactly once in after_fusion_blocks, "
                f"got {dense_head_hits}"
            )

        # H1 is the existing K3 head.
        h1 = model.dense_head

        # H2 / H3 have exactly the same architecture.
        h2 = copy.deepcopy(h1)
        h3 = copy.deepcopy(h1)

        h2_state = extract_dense_head_state(
            h2_ckpt
        )
        h3_state = extract_dense_head_state(
            h3_ckpt
        )

        try:
            h2.load_state_dict(
                h2_state,
                strict=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load H2 dense head from "
                f"{h2_ckpt}"
            ) from e

        try:
            h3.load_state_dict(
                h3_state,
                strict=True,
            )
        except Exception as e:
            raise RuntimeError(
                "Failed to load H3 dense head from "
                f"{h3_ckpt}"
            ) from e

        # Supervision metadata is not needed in inference,
        # but keeping it explicit makes checkpoint auditing easier.
        if hasattr(
            h2,
            "box3d_supervision",
        ):
            h2.box3d_supervision = "next2"

        if hasattr(
            h3,
            "box3d_supervision",
        ):
            h3.box3d_supervision = "next3"

        if hasattr(h2, "history_tag"):
            h2.history_tag = []

        if hasattr(h3, "history_tag"):
            h3.history_tag = []

        # Register the two additional heads on the model.
        # This makes .eval(), state inspection, etc. behave normally.
        model.add_module(
            "mtd_head_h2",
            h2,
        )

        model.add_module(
            "mtd_head_h3",
            h3,
        )

        model.mtd_head_h2.eval()
        model.mtd_head_h3.eval()

        self.model = model

        self.heads: Dict[int, torch.nn.Module] = {
            1: model.dense_head,
            2: model.mtd_head_h2,
            3: model.mtd_head_h3,
        }

        self.h2_ckpt = str(h2_ckpt)
        self.h3_ckpt = str(h3_ckpt)

        if logger is not None:
            logger.info(
                "============================================================"
            )
            logger.info(
                "TRUE MTD three-head bank loaded"
            )
            logger.info(
                "H1: K3 FULL dense_head -> next"
            )
            logger.info(
                f"H2: {h2_ckpt} -> next2"
            )
            logger.info(
                f"H3: {h3_ckpt} -> next3"
            )
            logger.info(
                "Only the dynamically selected head "
                "will execute per processed frame."
            )
            logger.info(
                "============================================================"
            )

    def get_head(
        self,
        branch_step: int,
    ):
        branch_step = int(
            branch_step
        )

        if branch_step not in self.heads:
            raise ValueError(
                f"invalid MTD branch: {branch_step}"
            )

        return self.heads[
            branch_step
        ]


def _autocast_context(
    model,
):
    enabled = bool(
        model.use_amp_dict["TEST"]
    )

    if (
        hasattr(torch, "amp")
        and
        hasattr(torch.amp, "autocast")
    ):
        return torch.amp.autocast(
            "cuda",
            enabled=enabled,
        )

    return torch.cuda.amp.autocast(
        enabled=enabled,
    )


def mtd_forward_no_post(
    model,
    batch,
    model_stream,
    contender,
    head_bank: MTDThreeHeadBank,
    branch_step: int,
):
    """
    Forward-only MTD execution.

    Timing scope is deliberately identical to the existing
    Original StreamDSGN evaluator:

        feature extractor
        + processed-frame history preparation
        + temporal fusion
        + shared after-fusion trunk
        + ONE selected MTD dense head

    Excluded:
        dataloader
        H2D
        post-processing / NMS
        recall evaluation

    History is appended only after the actual forward finishes.
    """

    branch_step = int(
        branch_step
    )

    selected_head = (
        head_bank.get_head(
            branch_step
        )
    )

    cur_data = batch[
        "token"
    ]

    launched = False

    history_features = None

    try:
        if contender is not None:
            contender.launch()
            launched = True

        with (
            torch.cuda.stream(
                model_stream
            ),
            torch.no_grad(),
            _autocast_context(model),
        ):
            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            # ----------------------------------------------------
            # 1. Shared K3 StreamDSGN feature extractor
            # ----------------------------------------------------

            for module in (
                model.feature_extractor
            ):
                cur_data = module(
                    cur_data
                )

            # ----------------------------------------------------
            # 2. Processed-frame-only history
            # ----------------------------------------------------

            cur_data[
                "history_features"
            ] = (
                model.history_feature_queue
            )

            if (
                model.history_tag
                is not None
            ):
                history_features = {}

                for feature_name in (
                    model.history_features_name
                ):
                    history_features[
                        feature_name
                    ] = (
                        cur_data[
                            feature_name
                        ].clone()
                    )

            # ----------------------------------------------------
            # 3. Shared temporal fusion
            # ----------------------------------------------------

            for module in (
                model.fusion_module
            ):
                cur_data = module(
                    cur_data
                )

            # ----------------------------------------------------
            # 4. Shared post-fusion trunk + ONE selected head
            # ----------------------------------------------------

            selected_count = 0

            for module in (
                model.after_fusion_blocks
            ):
                if (
                    module
                    is
                    model.dense_head
                ):
                    cur_data = (
                        selected_head(
                            cur_data
                        )
                    )

                    selected_count += 1

                else:
                    cur_data = module(
                        cur_data
                    )

            if selected_count != 1:
                raise RuntimeError(
                    "selected MTD head was not "
                    "executed exactly once"
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

    # Current-frame feature becomes history only after
    # the actual detector forward completed.
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

    return (
        cur_data,
        forward_ms,
    )
