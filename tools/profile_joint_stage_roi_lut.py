#!/usr/bin/env python3
"""
Profile the joint A_d / B / CD ROI action space for the deadline scheduler.

IMPORTANT
---------
This script measures REAL adaptive model-forward latency.

The adaptive model must support a profiling override:

    model.set_profile_action(
        a_level: int,
        b_level: int,
        cd_level: int,
        position_mode: str,
    )

and:

    model.clear_profile_action()

During set_profile_action(), the normal learned ROI/budget scheduler is bypassed
ONLY for the final action selection. The real adaptive execution path must still
run, including:
    - full shallow A_s
    - stage importance network (optional; controlled by implementation)
    - A_d selective/full-spatial execution
    - A cache composition
    - B selective/full-spatial execution
    - B cache composition
    - CD selective/full-spatial execution
    - BEV predictor
    - R/P/U composition
    - FeatureAlignment
    - VAN
    - Head
    - post-processing

Offline profiling is allowed to synchronize CUDA. Online inference is not.
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from pcdet.models import load_data_to_gpu

from single_roi_common import (
    add_base_args,
    build_env,
    cuda_autocast,
)


# ----------------------------------------------------------------------
# Utility
# ----------------------------------------------------------------------

def pct(xs, q):
    xs = np.asarray(xs, dtype=np.float64)
    if xs.size == 0:
        return float("nan")
    return float(np.percentile(xs, q))


def stats(xs):
    xs = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(xs.size),
        "mean_ms": float(xs.mean()),
        "std_ms": float(xs.std()),
        "min_ms": float(xs.min()),
        "p50_ms": pct(xs, 50),
        "p90_ms": pct(xs, 90),
        "p95_ms": pct(xs, 95),
        "p99_ms": pct(xs, 99),
        "max_ms": float(xs.max()),
    }


def to_jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    return x


def clone_batch_shallow(batch):
    """
    We do not deep-copy CUDA tensors.

    model.forward() may mutate batch_dict keys, therefore each call receives
    a fresh Python dict while reusing the same already-H2D tensors.
    """
    return dict(batch)


def cuda_event_time_ms(fn):
    """
    Measure one model call with CUDA events.

    Event timing is appropriate for the model-forward GPU execution boundary.
    Synchronization here is intentional because this is OFFLINE profiling.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    out = fn()
    end.record()

    end.synchronize()
    ms = float(start.elapsed_time(end))
    return ms, out


def wall_time_ms(fn):
    """
    Optional sanity check.

    This is NOT the number used to populate the GPU scheduler LUT by default.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


# ----------------------------------------------------------------------
# Model profiling interface
# ----------------------------------------------------------------------

def require_profile_api(model):
    missing = []

    for name in (
        "set_profile_action",
        "clear_profile_action",
    ):
        if not hasattr(model, name):
            missing.append(name)

    if missing:
        raise RuntimeError(
            "\nAdaptive profiling API is not installed on the detector.\n"
            "Missing: %s\n\n"
            "Do NOT fake the 125-action LUT with the original Full forward.\n"
            "The final adaptive detector must expose:\n\n"
            "  model.set_profile_action(a_level, b_level, cd_level, position_mode)\n"
            "  model.clear_profile_action()\n\n"
            "and the forced action must route through the REAL selective/cache "
            "forward path.\n" % ", ".join(missing)
        )


def set_action(model, a, b, cd, position_mode):
    model.set_profile_action(
        a_level=int(a),
        b_level=int(b),
        cd_level=int(cd),
        position_mode=str(position_mode),
    )


def clear_action(model):
    model.clear_profile_action()


# ----------------------------------------------------------------------
# Full / adaptive forward
# ----------------------------------------------------------------------

@torch.no_grad()
def run_model(model, batch):
    enabled = bool(model.use_amp_dict.get("TEST", False))

    # Keep the timing boundary identical to the detector's normal model.forward.
    with cuda_autocast(enabled):
        return model.forward(clone_batch_shallow(batch))


# ----------------------------------------------------------------------
# State handling
# ----------------------------------------------------------------------

def reset_stream_state_if_available(model):
    """
    Profiling different actions must not contaminate one another with arbitrary
    stage-cache histories.

    Prefer a dedicated adaptive reset API if implemented.
    """
    if hasattr(model, "reset_adaptive_stream_state"):
        model.reset_adaptive_stream_state()
        return

    # Existing StreamDSGN implementations typically maintain history internally.
    # We intentionally do NOT guess private field names here.
    #
    # For final profiling, implement reset_adaptive_stream_state() in the model.
    raise RuntimeError(
        "model.reset_adaptive_stream_state() is required for reliable joint "
        "profiling. It must clear StreamDSGN history, A/B/BEV stage caches, "
        "stage age maps, and shallow-history state."
    )


def initialize_with_full_frame(model, batch):
    """
    Each adaptive profiling sequence starts from a valid Full frame so every
    cache has a real initialized state.

    The model should expose a one-shot force-full profiling mode.
    """
    if not hasattr(model, "set_profile_global_full"):
        raise RuntimeError(
            "model.set_profile_global_full(enabled: bool) is required. "
            "The initialization frame must execute the exact original "
            "StreamDSGN Full path."
        )

    model.set_profile_global_full(True)
    try:
        _ = run_model(model, batch)
        torch.cuda.synchronize()
    finally:
        model.set_profile_global_full(False)


# ----------------------------------------------------------------------
# Dataset helpers
# ----------------------------------------------------------------------

def get_profile_batches(loader, max_batches):
    batches = []

    for batch_idx, batch in enumerate(loader):
        load_data_to_gpu(batch)
        batches.append(batch)

        if max_batches > 0 and len(batches) >= max_batches:
            break

    if not batches:
        raise RuntimeError("No profiling batches were loaded.")

    return batches


# ----------------------------------------------------------------------
# One action
# ----------------------------------------------------------------------

def profile_one_action(
    model,
    init_batch,
    profile_batches,
    action,
    position_mode,
    warmup,
    repeat,
    timing,
):
    a, b, cd = action

    # Every action starts from the same semantic state definition:
    # Full initialization -> repeated streaming adaptive frames.
    reset_stream_state_if_available(model)
    initialize_with_full_frame(model, init_batch)

    set_action(model, a, b, cd, position_mode)

    try:
        # Warmup.
        for n in range(warmup):
            batch = profile_batches[n % len(profile_batches)]
            _ = run_model(model, batch)

        torch.cuda.synchronize()

        samples = []

        for n in range(repeat):
            batch = profile_batches[n % len(profile_batches)]

            if timing == "cuda":
                ms, _ = cuda_event_time_ms(
                    lambda: run_model(model, batch)
                )
            else:
                ms, _ = wall_time_ms(
                    lambda: run_model(model, batch)
                )

            samples.append(ms)

        return samples

    finally:
        clear_action(model)


# ----------------------------------------------------------------------
# Global Full
# ----------------------------------------------------------------------

def profile_global_full(
    model,
    init_batch,
    profile_batches,
    warmup,
    repeat,
    timing,
):
    reset_stream_state_if_available(model)

    if not hasattr(model, "set_profile_global_full"):
        raise RuntimeError(
            "model.set_profile_global_full(enabled) is required."
        )

    model.set_profile_global_full(True)

    try:
        for n in range(warmup):
            batch = profile_batches[n % len(profile_batches)]
            _ = run_model(model, batch)

        torch.cuda.synchronize()

        samples = []

        for n in range(repeat):
            batch = profile_batches[n % len(profile_batches)]

            if timing == "cuda":
                ms, _ = cuda_event_time_ms(
                    lambda: run_model(model, batch)
                )
            else:
                ms, _ = wall_time_ms(
                    lambda: run_model(model, batch)
                )

            samples.append(ms)

        return samples

    finally:
        model.set_profile_global_full(False)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Profile 5x5x5 joint stage-ROI latency LUT"
    )

    add_base_args(p, training=False)

    p.add_argument(
        "--levels",
        type=int,
        default=5,
        help="Number of discrete levels per stage; current method uses 5.",
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=30,
    )

    p.add_argument(
        "--repeat",
        type=int,
        default=200,
    )

    p.add_argument(
        "--profile-batches",
        type=int,
        default=16,
        help="Number of already-H2D frames cycled during profiling.",
    )

    p.add_argument(
        "--position-modes",
        nargs="+",
        default=["center", "random", "boundary"],
        choices=["center", "random", "boundary"],
        help=(
            "Profile ROI location sensitivity. "
            "Final LUT takes worst P99 across modes."
        ),
    )

    p.add_argument(
        "--timing",
        default="cuda",
        choices=["cuda", "wall"],
        help="Use CUDA-event timing for formal GPU LUT.",
    )

    p.add_argument(
        "--margin-ms",
        type=float,
        default=0.20,
        help="Additional fixed safety margin added to worst-case P99.",
    )

    p.add_argument(
        "--margin-ratio",
        type=float,
        default=0.02,
        help="Additional multiplicative safety margin.",
    )

    p.add_argument(
        "--output",
        default="outputs/adaptive_profile/joint_stage_roi_lut.json",
    )

    p.add_argument(
        "--csv",
        default="outputs/adaptive_profile/joint_stage_roi_lut.csv",
    )

    p.add_argument(
        "--only-action",
        nargs=3,
        type=int,
        default=None,
        metavar=("A", "B", "CD"),
        help="Smoke-test only one action, e.g. --only-action 1 1 1",
    )

    return p.parse_args()


def main():
    args = parse_args()

    assert args.batch_size == 1, "Formal streaming profiling requires batch size 1."
    assert args.levels > 0
    assert args.warmup >= 0
    assert args.repeat > 0

    cfg, dataset, loader, model, logger = build_env(args, training=False)

    require_profile_api(model)

    batches = get_profile_batches(
        loader,
        max_batches=args.profile_batches,
    )

    init_batch = batches[0]

    if args.only_action is None:
        actions = [
            (a, b, cd)
            for a in range(args.levels)
            for b in range(args.levels)
            for cd in range(args.levels)
        ]
    else:
        action = tuple(args.only_action)
        for x in action:
            assert 0 <= x < args.levels
        actions = [action]

    logger.info("=" * 80)
    logger.info("Joint stage ROI profiling")
    logger.info(f"actions       : {len(actions)}")
    logger.info(f"levels/stage  : {args.levels}")
    logger.info(f"warmup        : {args.warmup}")
    logger.info(f"repeat        : {args.repeat}")
    logger.info(f"profile frames: {len(batches)}")
    logger.info(f"positions     : {args.position_modes}")
    logger.info(f"timing        : {args.timing}")
    logger.info("=" * 80)

    # --------------------------------------------------------------
    # Original StreamDSGN Full reference.
    # --------------------------------------------------------------
    logger.info("Profiling GLOBAL FULL reference...")

    full_samples = profile_global_full(
        model=model,
        init_batch=init_batch,
        profile_batches=batches,
        warmup=args.warmup,
        repeat=args.repeat,
        timing=args.timing,
    )

    full_stat = stats(full_samples)

    logger.info(
        "GLOBAL FULL "
        f"mean={full_stat['mean_ms']:.3f} "
        f"p50={full_stat['p50_ms']:.3f} "
        f"p95={full_stat['p95_ms']:.3f} "
        f"p99={full_stat['p99_ms']:.3f}"
    )

    # --------------------------------------------------------------
    # Adaptive actions.
    # --------------------------------------------------------------
    records = []

    for action_idx, action in enumerate(actions):
        a, b, cd = action

        logger.info(
            f"[{action_idx + 1:03d}/{len(actions):03d}] "
            f"A={a} B={b} CD={cd}"
        )

        per_position = {}

        for position_mode in args.position_modes:
            samples = profile_one_action(
                model=model,
                init_batch=init_batch,
                profile_batches=batches,
                action=action,
                position_mode=position_mode,
                warmup=args.warmup,
                repeat=args.repeat,
                timing=args.timing,
            )

            st = stats(samples)
            per_position[position_mode] = st

            logger.info(
                f"  {position_mode:>8s}: "
                f"mean={st['mean_ms']:.3f} "
                f"p50={st['p50_ms']:.3f} "
                f"p95={st['p95_ms']:.3f} "
                f"p99={st['p99_ms']:.3f}"
            )

        # Conservative LUT entry:
        # use the WORST P99 over tested ROI positions.
        worst_position = max(
            per_position,
            key=lambda k: per_position[k]["p99_ms"],
        )

        worst_p99 = per_position[worst_position]["p99_ms"]

        safe_ms = (
            worst_p99 * (1.0 + args.margin_ratio)
            + args.margin_ms
        )

        record = {
            "a_level": int(a),
            "b_level": int(b),
            "cd_level": int(cd),
            "positions": per_position,
            "worst_position": worst_position,
            "worst_p99_ms": float(worst_p99),
            "safe_ms": float(safe_ms),
            "speedup_vs_full_p99": (
                float(full_stat["p99_ms"] / worst_p99)
                if worst_p99 > 0 else float("nan")
            ),
        }

        records.append(record)

        logger.info(
            f"  -> worst={worst_position} "
            f"p99={worst_p99:.3f} ms "
            f"safe={safe_ms:.3f} ms"
        )

    # --------------------------------------------------------------
    # Dense LUT.
    # --------------------------------------------------------------
    lut = np.full(
        (args.levels, args.levels, args.levels),
        np.nan,
        dtype=np.float64,
    )

    for r in records:
        lut[
            r["a_level"],
            r["b_level"],
            r["cd_level"],
        ] = r["safe_ms"]

    result = {
        "schema_version": 1,
        "cfg": args.cfg,
        "ckpt": args.ckpt,
        "gpu": torch.cuda.get_device_name(torch.cuda.current_device()),
        "timing": args.timing,
        "levels": int(args.levels),
        "level_semantics": {
            "0": "reuse_only",
            "1": "small",
            "2": "medium",
            "3": "large",
            "4": "stage_full",
        },
        "position_modes": args.position_modes,
        "warmup": int(args.warmup),
        "repeat": int(args.repeat),
        "num_profile_batches": int(len(batches)),
        "margin_ms": float(args.margin_ms),
        "margin_ratio": float(args.margin_ratio),
        "global_full": full_stat,
        "records": records,
        "safe_lut_ms": lut,
        "note": (
            "safe_lut_ms[a,b,cd] = worst-position P99 * "
            "(1+margin_ratio) + margin_ms. "
            "Global Full is the exact original StreamDSGN path and is not "
            "equivalent to adaptive action (4,4,4)."
        ),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            to_jsonable(result),
            indent=2,
            ensure_ascii=False,
        )
    )

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "a_level",
            "b_level",
            "cd_level",
            "worst_position",
            "mean_ms_center",
            "p95_ms_center",
            "p99_ms_center",
            "mean_ms_random",
            "p95_ms_random",
            "p99_ms_random",
            "mean_ms_boundary",
            "p95_ms_boundary",
            "p99_ms_boundary",
            "worst_p99_ms",
            "safe_ms",
            "speedup_vs_full_p99",
        ])

        for r in records:
            def get(pos, key):
                if pos not in r["positions"]:
                    return ""
                return r["positions"][pos][key]

            writer.writerow([
                r["a_level"],
                r["b_level"],
                r["cd_level"],
                r["worst_position"],
                get("center", "mean_ms"),
                get("center", "p95_ms"),
                get("center", "p99_ms"),
                get("random", "mean_ms"),
                get("random", "p95_ms"),
                get("random", "p99_ms"),
                get("boundary", "mean_ms"),
                get("boundary", "p95_ms"),
                get("boundary", "p99_ms"),
                r["worst_p99_ms"],
                r["safe_ms"],
                r["speedup_vs_full_p99"],
            ])

    logger.info("=" * 80)
    logger.info(f"Saved JSON: {out_path}")
    logger.info(f"Saved CSV : {csv_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
