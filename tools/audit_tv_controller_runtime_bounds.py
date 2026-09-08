#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    ElasticBEVBranch,
)

from test_stream_buffer_timestamp import load_one

from test_tv_stream3d_online_forward import (
    make_cfg,
    configure_contender,
    execute_one,
    prewarm_all_causal_prefixes,
)

from tv_stream3d_controller import (
    TVStream3DController,
    Decision,
    NEXT_STAGE_INDEX,
    REMAINING_CONTROL_BOUNDARIES,
)

from tv_stream3d_causal_fused_runtime import (
    enable_causal_prefix_fused_cache,
    fused_cache_stats,
    set_forbid_cache_miss,
)

from smooth_cuda_contention import (
    make_high_priority_detector_stream,
)


DECISION_CHECKPOINTS = (
    "after_prefix",
    "after_res2",
    "after_res3",
    "after_res4",
    "after_fpn",
    "after_stereo",
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--prefix_bn_bank", required=True)
    p.add_argument("--controller_csv", required=True)
    p.add_argument("--levels_json", required=True)

    p.add_argument(
        "--levels",
        default="L0,L1,L2,L3,L4",
    )

    p.add_argument(
        "--input_hz",
        type=float,
        default=35.0,
    )

    p.add_argument(
        "--control_guard_per_boundary_ms",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--runtime_warmup_frames",
        type=int,
        default=20,
    )

    p.add_argument(
        "--overhead_frames",
        type=int,
        default=200,
    )

    p.add_argument(
        "--bound_warmup_frames",
        type=int,
        default=10,
    )

    p.add_argument(
        "--bound_frames",
        type=int,
        default=100,
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


def reset_history(model):
    q = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if q is not None:
        q.clear()


def stats(xs):
    x = np.asarray(
        xs,
        dtype=np.float64,
    )

    if x.size == 0:
        return {"n": 0}

    return {
        "n":
            int(x.size),

        "mean":
            float(x.mean()),

        "p50":
            float(
                np.percentile(
                    x,
                    50,
                )
            ),

        "p90":
            float(
                np.percentile(
                    x,
                    90,
                )
            ),

        "p99":
            float(
                np.percentile(
                    x,
                    99,
                )
            ),

        "min":
            float(x.min()),

        "max":
            float(x.max()),
    }


class TimedControllerProxy:
    """
    Measure classification in addition to execute_one()'s
    existing per-decision controller_ms measurement.
    """

    def __init__(self, base):
        self.base = base
        self.classify_ms = 0.0
        self.classify_calls = 0

    def __getattr__(self, name):
        return getattr(
            self.base,
            name,
        )

    def classify_probe(
        self,
        probe_ms,
    ):
        t0 = time.perf_counter_ns()

        out = self.base.classify_probe(
            probe_ms
        )

        self.classify_ms += (
            time.perf_counter_ns()
            -
            t0
        ) / 1e6

        self.classify_calls += 1

        return out

    def decide(
        self,
        *args,
        **kwargs,
    ):
        return self.base.decide(
            *args,
            **kwargs,
        )


class ForcedLevelController:
    """
    Dynamic rolling controller, but contention classification is
    forced to the known calibrated level.

    Used only for bound validation, so classifier error is not
    mixed with p99-bound calibration error.
    """

    def __init__(
        self,
        base,
        level,
    ):
        self.base = base
        self.level = str(level)

    def classify_probe(
        self,
        probe_ms,
    ):
        return self.level

    def decide(
        self,
        checkpoint,
        elapsed_ms,
        deadline_ms,
        observed_level,
        executed_prefix,
    ):
        return self.base.decide(
            checkpoint=checkpoint,
            elapsed_ms=elapsed_ms,
            deadline_ms=deadline_ms,
            observed_level=self.level,
            executed_prefix=executed_prefix,
        )


class FreezeAtCheckpointController:
    """
    Before target checkpoint:
        normal rolling controller with true calibrated level.

    At target checkpoint:
        choose normally, then freeze that complete profile.

    After target checkpoint:
        force all remaining widths from the frozen profile.

    Therefore:
        actual_remaining_ms
    can be compared directly with the target checkpoint's
        remaining_p99_ms
    for the SAME complete schedule.
    """

    def __init__(
        self,
        base,
        level,
        target_checkpoint,
    ):
        if (
            target_checkpoint
            not in DECISION_CHECKPOINTS
        ):
            raise ValueError(
                target_checkpoint
            )

        self.base = base
        self.level = str(level)
        self.target_checkpoint = (
            str(target_checkpoint)
        )

        self.frozen_pid = None
        self.target_decision = None

    def classify_probe(
        self,
        probe_ms,
    ):
        return self.level

    def _decision_for_pid(
        self,
        pid,
        checkpoint,
        elapsed_ms,
        deadline_ms,
        executed_prefix,
    ):
        pid = int(pid)

        schedule = tuple(
            float(x)
            for x
            in self.base.profile_schedule[
                pid
            ]
        )

        prefix = tuple(
            float(x)
            for x
            in executed_prefix
        )

        if (
            schedule[
                :len(prefix)
            ]
            !=
            prefix
        ):
            raise RuntimeError(
                "frozen profile no longer "
                "matches executed prefix: "
                f"pid={pid}, "
                f"schedule={schedule}, "
                f"executed={prefix}"
            )

        row = self.base.rows[
            (
                self.level,
                checkpoint,
                pid,
            )
        ]

        bound_ms = float(
            row["_bound"]
        )

        guard_ms = (
            self.base
            .control_guard_per_boundary_ms
            *
            REMAINING_CONTROL_BOUNDARIES[
                checkpoint
            ]
        )

        required_ms = (
            bound_ms
            +
            guard_ms
        )

        elapsed_ms = float(
            elapsed_ms
        )

        deadline_ms = float(
            deadline_ms
        )

        remaining_budget = (
            deadline_ms
            -
            elapsed_ms
        )

        next_idx = (
            NEXT_STAGE_INDEX[
                checkpoint
            ]
        )

        next_width = (
            None
            if next_idx is None
            else float(
                schedule[
                    next_idx
                ]
            )
        )

        return Decision(
            checkpoint=checkpoint,
            observed_level=self.level,
            elapsed_ms=elapsed_ms,
            deadline_ms=deadline_ms,
            remaining_budget_ms=(
                remaining_budget
            ),
            feasible=(
                required_ms
                <=
                remaining_budget
            ),
            profile_id=pid,
            schedule=schedule,
            quality=float(
                self.base.profile_quality[
                    pid
                ]
            ),
            remaining_bound_ms=(
                bound_ms
            ),
            control_guard_ms=(
                guard_ms
            ),
            required_remaining_ms=(
                required_ms
            ),
            next_width=next_width,
        )

    def decide(
        self,
        checkpoint,
        elapsed_ms,
        deadline_ms,
        observed_level,
        executed_prefix,
    ):
        if self.frozen_pid is None:
            d = self.base.decide(
                checkpoint=checkpoint,
                elapsed_ms=elapsed_ms,
                deadline_ms=deadline_ms,
                observed_level=self.level,
                executed_prefix=executed_prefix,
            )

            if (
                checkpoint
                ==
                self.target_checkpoint
            ):
                self.frozen_pid = int(
                    d.profile_id
                )

                self.target_decision = d

            return d

        return self._decision_for_pid(
            pid=self.frozen_pid,
            checkpoint=checkpoint,
            elapsed_ms=elapsed_ms,
            deadline_ms=deadline_ms,
            executed_prefix=executed_prefix,
        )


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

    levels = [
        x.strip()
        for x
        in args.levels.split(",")
        if x.strip()
    ]

    valid_levels = {
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    }

    if (
        not levels
        or
        any(
            x not in valid_levels
            for x in levels
        )
    ):
        raise ValueError(
            f"invalid levels: {levels}"
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
            "wrong causal-prefix BN bank"
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
        level:
            configure_contender(
                levels_data,
                level,
            )
        for level
        in levels
    }

    model_stream = (
        make_high_priority_detector_stream(
            torch.cuda.current_device()
        )
    )

    # ============================================================
    # Causal-prefix cache prewarm: identical to formal evaluator
    # ============================================================

    prewarm_batch = load_one(
        dataset,
        0,
    )

    reset_history(
        model
    )

    with torch.no_grad():
        pw_fp32 = (
            prewarm_all_causal_prefixes(
                base_model=model,
                branch=branch,
                bank=bank,
                controller=controller,
                batch=prewarm_batch,
                model_stream=model_stream,
            )
        )

    if (
        pw_fp32.get(
            "expected_prefixes"
        )
        != 203
        or
        pw_fp32.get(
            "ready_prefixes"
        )
        != 203
    ):
        raise RuntimeError(
            f"FP32 prewarm failed: "
            f"{pw_fp32}"
        )

    set_forbid_cache_miss(
        branch,
        True,
    )

    cache0 = (
        fused_cache_stats(
            branch
        )
    )

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
        pw_amp = (
            prewarm_all_causal_prefixes(
                base_model=model,
                branch=branch,
                bank=bank,
                controller=controller,
                batch=prewarm_batch,
                model_stream=model_stream,
            )
        )

    cache1 = (
        fused_cache_stats(
            branch
        )
    )

    if (
        cache1 != cache0
        or
        pw_amp.get(
            "ready_prefixes"
        )
        != 203
    ):
        raise RuntimeError(
            "AMP prewarm/cache check failed"
        )

    # We intentionally use ordinary sequential validation indices.
    # Defaults fit comfortably inside the 3673-frame validation set.
    max_needed = max(
        args.runtime_warmup_frames
        +
        args.overhead_frames,

        args.bound_warmup_frames
        +
        args.bound_frames,
    )

    if max_needed > len(
        dataset
    ):
        raise RuntimeError(
            "requested frames exceed dataset: "
            f"{max_needed} > {len(dataset)}"
        )

    # ============================================================
    # A1. Runtime controller CPU overhead
    # ============================================================

    overhead_rows = []

    for level in levels:
        print()
        print(
            "=" * 90
        )
        print(
            f"CONTROLLER OVERHEAD | "
            f"{level}"
        )
        print(
            "=" * 90
        )

        reset_history(
            model
        )

        contender = contenders[
            level
        ]

        for i in range(
            args.runtime_warmup_frames
        ):
            batch = load_one(
                dataset,
                i,
            )

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

        torch.cuda.synchronize()

        start_idx = (
            args.runtime_warmup_frames
        )

        for j in range(
            args.overhead_frames
        ):
            idx = (
                start_idx
                +
                j
            )

            batch = load_one(
                dataset,
                idx,
            )

            proxy = (
                TimedControllerProxy(
                    controller
                )
            )

            result = execute_one(
                base_model=model,
                branch=branch,
                bank=bank,
                controller=proxy,
                batch=batch,
                model_stream=model_stream,
                contender=contender,
                deadline_ms=period_ms,
                allow_prepare=False,
            )

            decision_ms = float(
                result[
                    "controller_ms"
                ]
            )

            classify_ms = float(
                proxy.classify_ms
            )

            total_cpu_ms = (
                decision_ms
                +
                classify_ms
            )

            calls = int(
                result[
                    "controller_calls"
                ]
            )

            overhead_rows.append({
                "level":
                    level,

                "dataset_index":
                    idx,

                "forward_ms":
                    float(
                        result[
                            "forward_ms"
                        ]
                    ),

                "observed_level":
                    str(
                        result[
                            "observed_level"
                        ]
                    ),

                "controller_decision_ms":
                    decision_ms,

                "classifier_ms":
                    classify_ms,

                "controller_cpu_total_ms":
                    total_cpu_ms,

                "controller_calls":
                    calls,

                "decision_us_per_call":
                    (
                        1000.0
                        *
                        decision_ms
                        /
                        max(
                            1,
                            calls,
                        )
                    ),

                "cpu_overhead_pct_period":
                    (
                        100.0
                        *
                        total_cpu_ms
                        /
                        period_ms
                    ),

                "cpu_overhead_pct_forward":
                    (
                        100.0
                        *
                        total_cpu_ms
                        /
                        float(
                            result[
                                "forward_ms"
                            ]
                        )
                    ),
            })

            if (
                j < 3
                or
                (j + 1) % 50 == 0
            ):
                print(
                    f"[{level}] "
                    f"{j+1:04d}/"
                    f"{args.overhead_frames} "
                    f"fwd="
                    f"{result['forward_ms']:.3f} "
                    f"ctrl="
                    f"{1000*total_cpu_ms:.2f} us"
                )

    overhead_csv = (
        out_dir
        /
        "controller_overhead.csv"
    )

    with overhead_csv.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                overhead_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            overhead_rows
        )

    # ============================================================
    # A2. Runtime p99-bound validation
    # ============================================================

    bound_rows = []

    for level in levels:
        contender = contenders[
            level
        ]

        for target_cp in (
            DECISION_CHECKPOINTS
        ):
            print()
            print(
                "=" * 90
            )
            print(
                "BOUND VALIDATION | "
                f"{level} | "
                f"{target_cp}"
            )
            print(
                "=" * 90
            )

            reset_history(
                model
            )

            # Warmup with true-level dynamic controller.
            forced = (
                ForcedLevelController(
                    controller,
                    level,
                )
            )

            for i in range(
                args.bound_warmup_frames
            ):
                batch = load_one(
                    dataset,
                    i,
                )

                execute_one(
                    base_model=model,
                    branch=branch,
                    bank=bank,
                    controller=forced,
                    batch=batch,
                    model_stream=model_stream,
                    contender=contender,
                    deadline_ms=period_ms,
                    allow_prepare=False,
                )

            torch.cuda.synchronize()

            start_idx = (
                args.bound_warmup_frames
            )

            for j in range(
                args.bound_frames
            ):
                idx = (
                    start_idx
                    +
                    j
                )

                batch = load_one(
                    dataset,
                    idx,
                )

                frozen = (
                    FreezeAtCheckpointController(
                        base=controller,
                        level=level,
                        target_checkpoint=(
                            target_cp
                        ),
                    )
                )

                result = execute_one(
                    base_model=model,
                    branch=branch,
                    bank=bank,
                    controller=frozen,
                    batch=batch,
                    model_stream=model_stream,
                    contender=contender,
                    deadline_ms=period_ms,
                    allow_prepare=False,
                )

                d = (
                    frozen.target_decision
                )

                if d is None:
                    raise RuntimeError(
                        "target decision was "
                        "never reached: "
                        f"{target_cp}"
                    )

                final_forward_ms = float(
                    result[
                        "forward_ms"
                    ]
                )

                actual_remaining_ms = (
                    final_forward_ms
                    -
                    float(
                        d.elapsed_ms
                    )
                )

                raw_bound_ms = float(
                    d.remaining_bound_ms
                )

                guarded_bound_ms = float(
                    d.required_remaining_ms
                )

                raw_margin_ms = (
                    raw_bound_ms
                    -
                    actual_remaining_ms
                )

                guarded_margin_ms = (
                    guarded_bound_ms
                    -
                    actual_remaining_ms
                )

                bound_rows.append({
                    "level":
                        level,

                    "checkpoint":
                        target_cp,

                    "dataset_index":
                        idx,

                    "profile_id":
                        int(
                            d.profile_id
                        ),

                    "schedule":
                        ",".join(
                            f"{x:g}"
                            for x
                            in d.schedule
                        ),

                    "decision_feasible":
                        int(
                            d.feasible
                        ),

                    "checkpoint_elapsed_ms":
                        float(
                            d.elapsed_ms
                        ),

                    "final_forward_ms":
                        final_forward_ms,

                    "actual_remaining_ms":
                        actual_remaining_ms,

                    "predicted_remaining_p99_ms":
                        raw_bound_ms,

                    "control_guard_ms":
                        float(
                            d.control_guard_ms
                        ),

                    "guarded_required_ms":
                        guarded_bound_ms,

                    "raw_margin_ms":
                        raw_margin_ms,

                    "guarded_margin_ms":
                        guarded_margin_ms,

                    "raw_p99_violation":
                        int(
                            actual_remaining_ms
                            >
                            raw_bound_ms
                            +
                            1e-9
                        ),

                    "guarded_violation":
                        int(
                            actual_remaining_ms
                            >
                            guarded_bound_ms
                            +
                            1e-9
                        ),
                })

                if (
                    j < 2
                    or
                    (j + 1) % 50 == 0
                ):
                    print(
                        f"[{level}/"
                        f"{target_cp}] "
                        f"{j+1:04d}/"
                        f"{args.bound_frames} "
                        f"actual="
                        f"{actual_remaining_ms:.3f} "
                        f"p99="
                        f"{raw_bound_ms:.3f} "
                        f"+guard="
                        f"{guarded_bound_ms:.3f}"
                    )

    bound_csv = (
        out_dir
        /
        "bound_validation.csv"
    )

    with bound_csv.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                bound_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            bound_rows
        )

    # ============================================================
    # Summary
    # ============================================================

    overhead_summary = {}

    for level in levels:
        rs = [
            r
            for r
            in overhead_rows
            if r[
                "level"
            ] == level
        ]

        overhead_summary[
            level
        ] = {
            "frames":
                len(rs),

            "controller_cpu_total_us":
                stats([
                    1000.0
                    *
                    r[
                        "controller_cpu_total_ms"
                    ]
                    for r
                    in rs
                ]),

            "decision_us_per_call":
                stats([
                    r[
                        "decision_us_per_call"
                    ]
                    for r
                    in rs
                ]),

            "cpu_overhead_pct_period":
                stats([
                    r[
                        "cpu_overhead_pct_period"
                    ]
                    for r
                    in rs
                ]),

            "cpu_overhead_pct_forward":
                stats([
                    r[
                        "cpu_overhead_pct_forward"
                    ]
                    for r
                    in rs
                ]),
        }

    bound_summary = {}

    for level in levels:
        bound_summary[
            level
        ] = {}

        for cp in (
            DECISION_CHECKPOINTS
        ):
            rs = [
                r
                for r
                in bound_rows
                if (
                    r[
                        "level"
                    ] == level
                    and
                    r[
                        "checkpoint"
                    ] == cp
                )
            ]

            n = len(rs)

            raw_v = sum(
                r[
                    "raw_p99_violation"
                ]
                for r
                in rs
            )

            guard_v = sum(
                r[
                    "guarded_violation"
                ]
                for r
                in rs
            )

            feasible = sum(
                r[
                    "decision_feasible"
                ]
                for r
                in rs
            )

            bound_summary[
                level
            ][
                cp
            ] = {
                "n":
                    n,

                "feasible_fraction":
                    (
                        feasible
                        /
                        max(
                            1,
                            n,
                        )
                    ),

                "raw_p99_violation_rate":
                    (
                        raw_v
                        /
                        max(
                            1,
                            n,
                        )
                    ),

                "guarded_violation_rate":
                    (
                        guard_v
                        /
                        max(
                            1,
                            n,
                        )
                    ),

                "actual_remaining_ms":
                    stats([
                        r[
                            "actual_remaining_ms"
                        ]
                        for r
                        in rs
                    ]),

                "raw_margin_ms":
                    stats([
                        r[
                            "raw_margin_ms"
                        ]
                        for r
                        in rs
                    ]),

                "guarded_margin_ms":
                    stats([
                        r[
                            "guarded_margin_ms"
                        ]
                        for r
                        in rs
                    ]),

                "profile_histogram":
                    dict(
                        Counter(
                            str(
                                r[
                                    "profile_id"
                                ]
                            )
                            for r
                            in rs
                        )
                    ),
            }

    summary = {
        "version":
            "tv_controller_runtime_bound_audit_v1",

        "input_hz":
            args.input_hz,

        "period_ms":
            period_ms,

        "control_guard_per_boundary_ms":
            args.control_guard_per_boundary_ms,

        "levels":
            levels,

        "overhead":
            overhead_summary,

        "bound_validation":
            bound_summary,

        "files": {
            "controller_overhead_csv":
                str(
                    overhead_csv
                ),

            "bound_validation_csv":
                str(
                    bound_csv
                ),
        },

        "cache":
            fused_cache_stats(
                branch
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
        "=" * 110
    )
    print(
        "CONTROLLER OVERHEAD SUMMARY"
    )
    print(
        "=" * 110
    )

    for level in levels:
        s = overhead_summary[
            level
        ]

        x = s[
            "controller_cpu_total_us"
        ]

        print(
            f"{level}: "
            f"frame-controller CPU "
            f"p50/p90/p99="
            f"{x['p50']:.2f}/"
            f"{x['p90']:.2f}/"
            f"{x['p99']:.2f} us"
        )

    print()
    print(
        "=" * 110
    )
    print(
        "P99 BOUND VALIDATION SUMMARY"
    )
    print(
        "=" * 110
    )

    for level in levels:
        for cp in (
            DECISION_CHECKPOINTS
        ):
            s = (
                bound_summary[
                    level
                ][
                    cp
                ]
            )

            print(
                f"{level:>2s} "
                f"{cp:>13s} | "
                f"raw violation="
                f"{100*s['raw_p99_violation_rate']:.2f}% | "
                f"guarded="
                f"{100*s['guarded_violation_rate']:.2f}% | "
                f"guard margin p50="
                f"{s['guarded_margin_ms']['p50']:.3f} ms"
            )

    print()
    print(
        f"summary = {summary_path}"
    )


if __name__ == "__main__":
    main()
