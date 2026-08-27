#!/usr/bin/env python3
"""
Profile the REAL adaptive runtime and isolate the latency problem.

This script intentionally reuses tools/eval_adaptive_true_stream.py instead of
reimplementing the adaptive detector. It answers two questions:

1) Where is wall/GPU time spent in the current adaptive forward?
2) How much of the latency is caused by the current 15x avg_pool2d ROI planner?

For each forced joint action, e.g. (0,0,0), (1,3,4), (2,2,4), it profiles the
same physical adaptive runtime twice:

  legacy   : current stage_plan_gpu() based on 15 large-window avg_pool2d calls
  integral : one 2-D prefix sum per stage + O(1) rectangle-sum lookup per window

The 125-action CPU scheduler is deliberately still executed in both cases, then
its selected action is overridden by --actions. Thus legacy-vs-integral isolates
the online ROI-planning implementation while keeping the rest of the runtime
unchanged.

Timing semantics
----------------
H2D/load_data_to_gpu is OUTSIDE the measured forward, matching the user's
true-stream timing definition.

The reported total_forward_ms is:
    cuda synchronize
    t0
    real engine.forward(...)
    cuda synchronize
    t1

Internal GPU component times use CUDA events and therefore do not insert extra
synchronizations inside the measured forward.

The CPU timings for finalize_stage_plans and scheduler.choose use perf_counter.
In particular finalize_stage_plans includes the current .cpu().numpy() barrier.

This is a diagnostic profiler. It does not compute AP/sAP.
"""

import argparse
import json
import math
import time
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcdet.models import load_data_to_gpu

# Run as `python tools/profile_adaptive_runtime_breakdown.py`.
import eval_adaptive_true_stream as adaptive


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-"
    "lka_7-mcl_5090_eval.yaml"
)
DEFAULT_BASE_CKPT = "extra_data/checkpoint_epoch_20.pth"
DEFAULT_ADAPTIVE_CKPT = "outputs/adaptive_joint/adaptive_joint_full_10ep.pth"
DEFAULT_LUT = (
    "outputs/adaptive_profile/latency_lut_full/"
    "adaptive_latency_lut_provisional.json"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Profile current adaptive runtime latency breakdown"
    )
    p.add_argument("--cfg_file", default=DEFAULT_CFG)
    p.add_argument("--ckpt", default=DEFAULT_BASE_CKPT)
    p.add_argument("--adaptive_ckpt", default=DEFAULT_ADAPTIVE_CKPT)
    p.add_argument("--latency_lut", default=DEFAULT_LUT)

    p.add_argument("--levels", default="0.15,0.25,0.40,0.60,0.80")
    p.add_argument(
        "--actions",
        default="0,0,0;1,3,4;2,2,4",
        help="Semicolon-separated forced joint actions a,b,cd",
    )
    p.add_argument(
        "--planners",
        default="legacy,integral",
        help="legacy,integral or both",
    )

    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--warmup_adaptive", type=int, default=10)
    p.add_argument("--measure_frames", type=int, default=100)

    p.add_argument("--roi_align", type=int, default=4)
    p.add_argument("--b_halo", type=int, default=1)
    p.add_argument("--cd_halo", type=int, default=12)
    p.add_argument("--pred_margin", type=int, default=12)
    p.add_argument("--age_cap", type=float, default=8.0)

    p.add_argument("--lambda_a", type=float, default=1.0)
    p.add_argument("--lambda_b", type=float, default=1.0)
    p.add_argument("--lambda_cd", type=float, default=1.0)
    p.add_argument("--importance_hidden", type=int, default=32)
    p.add_argument("--predictor_hidden", type=int, default=64)

    p.add_argument("--scheduler_extra_ms", type=float, default=2.0)
    p.add_argument("--deadline_ms", type=float, default=25.0)

    p.add_argument("--self_check_tol", type=float, default=0.02)
    p.add_argument("--skip_self_check", action="store_true")
    p.add_argument("--planner_check_tol", type=float, default=1e-5)

    p.add_argument(
        "--out_dir",
        default="outputs/adaptive_profile/runtime_breakdown",
    )
    return p.parse_args()


def parse_levels(s):
    xs = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if len(xs) != 5:
        raise ValueError(f"Expected 5 levels, got {xs}")
    return xs


def parse_actions(s):
    out = []
    for item in str(s).split(";"):
        item = item.strip()
        if not item:
            continue
        vals = tuple(int(x.strip()) for x in item.split(","))
        if len(vals) != 3:
            raise ValueError(f"Bad action: {item}")
        if any(x < 0 or x > 4 for x in vals):
            raise ValueError(f"Action level outside [0,4]: {vals}")
        out.append(vals)
    if not out:
        raise ValueError("No actions")
    return out


def parse_planners(s):
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    for x in out:
        if x not in ("legacy", "integral"):
            raise ValueError(f"Unknown planner: {x}")
    return out


def pct(xs, q):
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return None
    return float(np.percentile(a, q))


def summarize(xs):
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {
            "n": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "p50_ms": pct(a, 50),
        "p90_ms": pct(a, 90),
        "p99_ms": pct(a, 99),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


# =============================================================================
# Integral-image planner
# =============================================================================

def integral_stage_plan_gpu(q, levels, align):
    """
    Same return contract as eval_adaptive_true_stream.stage_plan_gpu().

    Build ONE integral image for the Q map. Each rectangular-window sum is then:
        S(y0,y1,x0,x1)
          = I(y1,x1) - I(y0,x1) - I(y1,x0) + I(y0,x0)

    Complexity no longer scales with the ROI kernel area.
    """
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 1:
        raise RuntimeError(f"Expected Q [1,1,H,W], got {tuple(q.shape)}")

    H, W = q.shape[-2:]

    # Float32 prefix sums avoid FP16 accumulation drift over 80x312 / 288x256.
    qf = q.float()
    integ = F.pad(qf, (1, 0, 1, 0), mode="constant", value=0.0)
    integ = integ.cumsum(dim=-2).cumsum(dim=-1)

    plans = []
    for level, ratio in enumerate(levels):
        h, w = adaptive.roi_hw_from_ratio(H, W, ratio, align=align)

        # [B,1,H-h+1,W-w+1]
        score = (
            integ[..., h:, w:]
            - integ[..., :-h, w:]
            - integ[..., h:, :-w]
            + integ[..., :-h, :-w]
        )

        flat = score.flatten(1)
        value, idx = flat.max(dim=1)

        plans.append({
            "level": level,
            "ratio": float(ratio),
            "h": int(h),
            "w": int(w),
            "Wv": int(score.shape[-1]),
            "value_gpu": value[0],
            "idx_gpu": idx[0],
        })

    return plans


@torch.no_grad()
def check_integral_equivalence(levels, align, tol, logger):
    """
    Mathematical/numerical guard before profiling.
    Random positive maps avoid pathological exact ties.
    """
    torch.manual_seed(12345)
    shapes = [(80, 312), (288, 256)]

    logger.info("=" * 100)
    logger.info("Checking legacy avg_pool planner vs integral-image planner")

    for H, W in shapes:
        q = torch.rand((1, 1, H, W), device="cuda", dtype=torch.float32)
        q = q / q.sum(dim=(-2, -1), keepdim=True)

        legacy = adaptive.stage_plan_gpu(q, levels, align)
        integ = integral_stage_plan_gpu(q, levels, align)

        for i, (a, b) in enumerate(zip(legacy, integ)):
            va = float(a["value_gpu"].detach().cpu())
            vb = float(b["value_gpu"].detach().cpu())
            ia = int(a["idx_gpu"].detach().cpu())
            ib = int(b["idx_gpu"].detach().cpu())
            diff = abs(va - vb)

            logger.info(
                "planner-check HW=%sx%s level=%d value_diff=%.9g idx_equal=%s",
                H, W, i, diff, ia == ib,
            )

            # If argmax differs but the maximum value is numerically tied, this
            # is harmless. A materially different maximum is not.
            if diff > tol:
                raise RuntimeError(
                    f"Integral planner mismatch HW={(H,W)} level={i}: "
                    f"legacy={va}, integral={vb}, diff={diff}"
                )

    logger.info("Integral planner numerical check: PASS")
    logger.info("=" * 100)


# =============================================================================
# Per-frame instrumentation without extra CUDA synchronizations
# =============================================================================

class FrameRecorder:
    def __init__(self):
        self.active = False
        self.events = []
        self.cpu = defaultdict(float)
        self.frames = []

    def begin(self):
        self.active = True
        self.events = []
        self.cpu = defaultdict(float)

    def end(self, total_ms):
        # Caller has already torch.cuda.synchronize()'d.
        gpu = defaultdict(float)
        for name, start, end in self.events:
            gpu[name] += float(start.elapsed_time(end))

        rec = {
            "total_forward_ms": float(total_ms),
            "gpu_ms": dict(gpu),
            "cpu_wall_ms": dict(self.cpu),
        }
        self.frames.append(rec)
        self.active = False
        self.events = []
        self.cpu = defaultdict(float)
        return rec

    def cuda_call(self, name, fn):
        if not self.active:
            return fn()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        self.events.append((name, start, end))
        return out

    def cpu_call(self, name, fn):
        if not self.active:
            return fn()
        t0 = time.perf_counter()
        out = fn()
        self.cpu[name] += (time.perf_counter() - t0) * 1000.0
        return out


class RuntimePatcher:
    def __init__(
        self,
        engine,
        planner_impl,
        forced_action,
        recorder,
        deadline_ms,
    ):
        self.engine = engine
        self.planner_impl = planner_impl
        self.forced_action = tuple(forced_action)
        self.rec = recorder
        self.deadline_ms = float(deadline_ms)
        self.saved = []

    def _save_attr(self, obj, name):
        self.saved.append((obj, name, getattr(obj, name)))

    def install(self):
        engine = self.engine
        bb = engine.backbone
        imp = engine.importance_net
        pred = engine.pred_net
        sched = engine.scheduler
        rec = self.rec

        # ------------------------------------------------------------------
        # A_s: two calls per adaptive frame.
        # ------------------------------------------------------------------
        self._save_attr(bb, "forward_2d_shallow")
        orig = bb.forward_2d_shallow

        def shallow_wrapped(_self, *args, **kwargs):
            return rec.cuda_call(
                "A_s",
                lambda: orig(*args, **kwargs),
            )

        bb.forward_2d_shallow = types.MethodType(shallow_wrapped, bb)

        # Importance network.
        self._save_attr(imp, "forward")
        orig_imp = imp.forward

        def imp_wrapped(_self, *args, **kwargs):
            return rec.cuda_call(
                "importance",
                lambda: orig_imp(*args, **kwargs),
            )

        imp.forward = types.MethodType(imp_wrapped, imp)

        # Predictor.
        self._save_attr(pred, "forward")
        orig_pred = pred.forward

        def pred_wrapped(_self, *args, **kwargs):
            return rec.cuda_call(
                "predictor",
                lambda: orig_pred(*args, **kwargs),
            )

        pred.forward = types.MethodType(pred_wrapped, pred)

        # Engine physical stages.
        for meth_name, label in (
            ("_run_a_explicit", "A_physical"),
            ("_prepare_b_shared", "B_prepare"),
            ("_run_b_roi", "B_physical"),
            ("_run_cd_roi_to_bev_patch", "CD_physical"),
            ("_run_downstream", "downstream"),
        ):
            self._save_attr(engine, meth_name)
            original_bound = getattr(engine, meth_name)

            def make_wrapped(original_bound, label):
                def wrapped(_self, *args, **kwargs):
                    return rec.cuda_call(
                        label,
                        lambda: original_bound(*args, **kwargs),
                    )
                return wrapped

            setattr(
                engine,
                meth_name,
                types.MethodType(make_wrapped(original_bound, label), engine),
            )

        # ------------------------------------------------------------------
        # Planner GPU work.
        # stage_plan_gpu is called once for A, B, CD. Each call evaluates 5 levels.
        # ------------------------------------------------------------------
        self.saved.append(
            (adaptive, "stage_plan_gpu", adaptive.stage_plan_gpu)
        )

        def planner_wrapped(q, levels, align):
            return rec.cuda_call(
                "roi_planner_gpu",
                lambda: self.planner_impl(q, levels, align),
            )

        adaptive.stage_plan_gpu = planner_wrapped

        # Current finalize contains .cpu().numpy(): measure its blocking wall cost.
        self.saved.append(
            (adaptive, "finalize_stage_plans", adaptive.finalize_stage_plans)
        )
        orig_finalize = adaptive.finalize_stage_plans

        def finalize_wrapped(groups):
            return rec.cpu_call(
                "planner_finalize_cpu_sync",
                lambda: orig_finalize(groups),
            )

        adaptive.finalize_stage_plans = finalize_wrapped

        # Execute the real 125-action scheduler, measure it, then override only
        # the selected action. This preserves scheduler overhead.
        self._save_attr(sched, "choose")
        self._save_attr(sched, "full_fits")
        orig_choose = sched.choose

        def choose_wrapped(_self, plans, deadline):
            original_result = rec.cpu_call(
                "scheduler_125_cpu",
                lambda: orig_choose(plans, deadline),
            )

            a, b, cd = self.forced_action
            key = f"{a},{b},{cd}"
            lut_rec = sched.actions[key]

            utility = (
                sched.lambda_a * plans["A"][a]["utility"]
                + sched.lambda_b * plans["B"][b]["utility"]
                + sched.lambda_cd * plans["CD"][cd]["utility"]
            )

            return {
                "key": key,
                "a": a,
                "b": b,
                "cd": cd,
                "estimated_safe_ms": (
                    float(lut_rec["safe_ms"]) + sched.extra_ms
                ),
                "utility": float(utility),
                "feasible_count": original_result.get("feasible_count", 0),
                "no_feasible_action": False,
            }

        sched.choose = types.MethodType(choose_wrapped, sched)

        # Never allow the stale Full-P99 shortcut during action profiling.
        def full_fits_wrapped(_self, deadline):
            return False

        sched.full_fits = types.MethodType(full_fits_wrapped, sched)

    def restore(self):
        for obj, name, old in reversed(self.saved):
            setattr(obj, name, old)
        self.saved = []


def aggregate_breakdown(frames):
    total = [x["total_forward_ms"] for x in frames]

    gpu_keys = sorted({
        k
        for x in frames
        for k in x["gpu_ms"].keys()
    })
    cpu_keys = sorted({
        k
        for x in frames
        for k in x["cpu_wall_ms"].keys()
    })

    gpu = {}
    for k in gpu_keys:
        gpu[k] = summarize([
            x["gpu_ms"].get(k, 0.0)
            for x in frames
        ])

    cpu = {}
    for k in cpu_keys:
        cpu[k] = summarize([
            x["cpu_wall_ms"].get(k, 0.0)
            for x in frames
        ])

    # This is diagnostic, not an exact algebraic decomposition:
    # CUDA-event stage sums exclude some tiny tensor/state ops and CPU portions.
    gpu_sum_per_frame = [
        sum(x["gpu_ms"].values())
        for x in frames
    ]
    cpu_sum_per_frame = [
        sum(x["cpu_wall_ms"].values())
        for x in frames
    ]
    residual = [
        x["total_forward_ms"]
        - sum(x["gpu_ms"].values())
        - sum(x["cpu_wall_ms"].values())
        for x in frames
    ]

    return {
        "total_forward_ms": summarize(total),
        "gpu_components_ms": gpu,
        "cpu_critical_path_ms": cpu,
        "gpu_component_sum_ms": summarize(gpu_sum_per_frame),
        "cpu_component_sum_ms": summarize(cpu_sum_per_frame),
        "unattributed_residual_ms": summarize(residual),
    }


@torch.no_grad()
def profile_pass(
    loader,
    engine,
    planner_name,
    planner_impl,
    action,
    warmup_adaptive,
    measure_frames,
    deadline_ms,
    logger,
):
    recorder = FrameRecorder()
    patcher = RuntimePatcher(
        engine=engine,
        planner_impl=planner_impl,
        forced_action=action,
        recorder=recorder,
        deadline_ms=deadline_ms,
    )
    patcher.install()

    try:
        engine.reset()
        current_scene = None
        warm = 0
        measured = 0

        for batch_idx, batch in enumerate(loader):
            scene, frame = adaptive.get_scene_frame(batch)
            token = batch["token"]

            if scene != current_scene:
                current_scene = scene
                engine.reset()

            # Explicitly excluded from latency.
            load_data_to_gpu(batch)

            is_init = engine.state is None
            if is_init:
                # Scene-init Full is required to create valid A/B/BEV caches.
                _ = engine.forward(
                    token,
                    deadline_ms,
                    force_full=True,
                )
                torch.cuda.synchronize()
                continue

            if warm < warmup_adaptive:
                _ = engine.forward(
                    token,
                    deadline_ms,
                    force_full=False,
                )
                torch.cuda.synchronize()
                warm += 1
                continue

            recorder.begin()
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            _ = engine.forward(
                token,
                deadline_ms,
                force_full=False,
            )

            torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t0) * 1000.0
            frame_rec = recorder.end(total_ms)
            frame_rec["scene"] = scene
            frame_rec["frame"] = frame

            measured += 1

            if measured % 20 == 0:
                logger.info(
                    "[%s action=%s] %d/%d last=%.3f ms",
                    planner_name,
                    ",".join(map(str, action)),
                    measured,
                    measure_frames,
                    total_ms,
                )

            if measured >= measure_frames:
                break

        if measured == 0:
            raise RuntimeError(
                "No adaptive frames were measured. "
                "Increase dataset span or inspect scene handling."
            )

        summary = aggregate_breakdown(recorder.frames)
        summary.update({
            "planner": planner_name,
            "forced_action": list(action),
            "warmup_adaptive": int(warmup_adaptive),
            "measure_frames": int(measured),
            "deadline_ms_argument": float(deadline_ms),
        })
        return summary, recorder.frames

    finally:
        patcher.restore()
        engine.reset()


def format_summary(summary):
    lines = []
    lines.append(
        f"planner={summary['planner']} "
        f"action={','.join(map(str, summary['forced_action']))}"
    )
    lines.append(
        "TOTAL  mean={mean_ms:.3f} p50={p50_ms:.3f} "
        "p90={p90_ms:.3f} p99={p99_ms:.3f}".format(
            **summary["total_forward_ms"]
        )
    )

    lines.append("GPU components (mean ms):")
    for k, v in summary["gpu_components_ms"].items():
        lines.append(
            f"  {k:24s} {v['mean_ms']:.3f}"
        )

    lines.append("CPU critical-path pieces (mean ms):")
    for k, v in summary["cpu_critical_path_ms"].items():
        lines.append(
            f"  {k:24s} {v['mean_ms']:.3f}"
        )

    lines.append(
        "  {:24s} {:.3f}".format(
            "unattributed_residual",
            summary["unattributed_residual_ms"]["mean_ms"],
        )
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    levels = parse_levels(args.levels)
    actions = parse_actions(args.actions)
    planners = parse_planners(args.planners)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        dataset,
        loader,
        model,
        backbone,
        importance_net,
        pred_net,
        logger,
        _,
    ) = adaptive.build_env(args)

    scheduler = adaptive.DeadlineScheduler(
        args.latency_lut,
        levels,
        args.lambda_a,
        args.lambda_b,
        args.lambda_cd,
        args.scheduler_extra_ms,
    )

    engine = adaptive.AdaptiveEngine(
        model=model,
        backbone=backbone,
        importance_net=importance_net,
        pred_net=pred_net,
        scheduler=scheduler,
        levels=levels,
        roi_align=args.roi_align,
        b_halo=args.b_halo,
        cd_halo=args.cd_halo,
        pred_margin=args.pred_margin,
        age_cap=args.age_cap,
    )

    logger.info("=" * 100)
    logger.info("Adaptive runtime latency breakdown")
    logger.info("dataset frames      : %d", len(dataset))
    logger.info("actions             : %s", actions)
    logger.info("planners            : %s", planners)
    logger.info("warmup adaptive     : %d", args.warmup_adaptive)
    logger.info("measure frames      : %d", args.measure_frames)
    logger.info("H2D/data prep       : EXCLUDED")
    logger.info("current Full LUT P99: %.3f ms", scheduler.full_p99_ms)
    logger.info("=" * 100)

    # Verify the current physical B/CD path before latency diagnosis.
    if not args.skip_self_check:
        first_batch = next(iter(loader))
        load_data_to_gpu(first_batch)
        adaptive.self_check_physical_bcd(
            engine,
            first_batch["token"],
            args.self_check_tol,
            logger,
        )

    # Verify that prefix-sum planner computes the same maximum rectangle value.
    if "integral" in planners:
        check_integral_equivalence(
            levels,
            args.roi_align,
            args.planner_check_tol,
            logger,
        )

    legacy_impl = adaptive.stage_plan_gpu
    planner_impls = {
        "legacy": legacy_impl,
        "integral": integral_stage_plan_gpu,
    }

    results = []
    raw = {}

    for action in actions:
        for planner_name in planners:
            logger.info("")
            logger.info(
                "Profiling planner=%s action=%s",
                planner_name,
                action,
            )

            summary, frames = profile_pass(
                loader=loader,
                engine=engine,
                planner_name=planner_name,
                planner_impl=planner_impls[planner_name],
                action=action,
                warmup_adaptive=args.warmup_adaptive,
                measure_frames=args.measure_frames,
                deadline_ms=args.deadline_ms,
                logger=logger,
            )

            results.append(summary)
            raw[f"{planner_name}:{','.join(map(str, action))}"] = frames

            logger.info("\n%s", format_summary(summary))

    # Compare integral vs legacy total latency per action.
    comparison = {}
    for action in actions:
        key = ",".join(map(str, action))
        recs = {
            x["planner"]: x
            for x in results
            if x["forced_action"] == list(action)
        }
        if "legacy" in recs and "integral" in recs:
            a = recs["legacy"]["total_forward_ms"]["mean_ms"]
            b = recs["integral"]["total_forward_ms"]["mean_ms"]
            comparison[key] = {
                "legacy_mean_ms": a,
                "integral_mean_ms": b,
                "saved_ms": a - b,
                "speedup": a / b if b > 0 else None,
            }

    payload = {
        "schema_version": 1,
        "purpose": "diagnose current physical adaptive runtime latency",
        "latency_semantics": {
            "total": "forward-only wall time with CUDA sync before/after",
            "excluded": [
                "dataset IO",
                "CPU preprocessing",
                "load_data_to_gpu",
                "H2D",
            ],
            "internal_gpu": "CUDA events; no extra in-forward synchronization",
        },
        "config": {
            "cfg_file": args.cfg_file,
            "base_ckpt": args.ckpt,
            "adaptive_ckpt": args.adaptive_ckpt,
            "latency_lut": args.latency_lut,
            "levels": levels,
            "actions": [list(x) for x in actions],
            "planners": planners,
            "warmup_adaptive": args.warmup_adaptive,
            "measure_frames": args.measure_frames,
            "deadline_ms": args.deadline_ms,
        },
        "results": results,
        "legacy_vs_integral": comparison,
        "note": (
            "The forced action override happens after the real 125-action CPU "
            "scheduler executes, so scheduler overhead remains measured."
        ),
    }

    json_path = out_dir / "runtime_breakdown.json"
    raw_path = out_dir / "runtime_breakdown_raw.json"
    txt_path = out_dir / "runtime_breakdown.txt"

    json_path.write_text(json.dumps(payload, indent=2))
    raw_path.write_text(json.dumps(raw, indent=2))

    text = []
    for x in results:
        text.append("=" * 100)
        text.append(format_summary(x))
    if comparison:
        text.append("=" * 100)
        text.append("LEGACY -> INTEGRAL TOTAL LATENCY")
        for k, v in comparison.items():
            text.append(
                f"action {k}: "
                f"{v['legacy_mean_ms']:.3f} -> {v['integral_mean_ms']:.3f} ms, "
                f"saved {v['saved_ms']:.3f} ms, speedup {v['speedup']:.3f}x"
            )

    txt_path.write_text("\n".join(text) + "\n")

    logger.info("")
    logger.info("Saved:")
    logger.info("  %s", json_path)
    logger.info("  %s", txt_path)
    logger.info("  %s", raw_path)


if __name__ == "__main__":
    main()