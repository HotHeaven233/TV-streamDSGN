#!/usr/bin/env python3
"""Same-process, same-frame Full-vs-Adaptive latency comparison.

Run AFTER apply_streamdsgn_latency_optimizations.py and CUDA rebuild.

Timing boundary:
    load_data_to_gpu(batch)   # excluded
    cuda synchronize
    t0
    actual forward
    cuda synchronize
    t1

For each forced action, the adaptive pass still executes:
  A_s -> ImportanceNet -> integral ROI planning -> GPU 5x5x5 scheduling work
  -> physical A/B/CD -> predictor/RPU -> downstream/post-processing.
Only the final selected level triplet is forced so we can compare specific
physical actions under identical control overhead.
"""

import argparse
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch

from pcdet.models import load_data_to_gpu
import eval_adaptive_true_stream as adaptive


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg_file", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--adaptive_ckpt", required=True)
    p.add_argument("--latency_lut", required=True)
    p.add_argument("--levels", default="0.15,0.25,0.40,0.60,0.80")
    p.add_argument("--actions", default="0,0,0;1,3,4;2,2,4")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--warmup_frames", type=int, default=10)
    p.add_argument("--measure_frames", type=int, default=100)
    p.add_argument("--deadline_ms", type=float, default=25.0)
    p.add_argument("--roi_align", type=int, default=4)
    p.add_argument("--b_halo", type=int, default=2)
    p.add_argument("--cd_halo", type=int, default=2)
    p.add_argument("--pred_margin", type=int, default=12)
    p.add_argument("--age_cap", type=float, default=8.0)
    p.add_argument("--lambda_a", type=float, default=1.0)
    p.add_argument("--lambda_b", type=float, default=1.0)
    p.add_argument("--lambda_cd", type=float, default=1.0)
    p.add_argument("--importance_hidden", type=int, default=32)
    p.add_argument("--predictor_hidden", type=int, default=64)
    p.add_argument("--scheduler_extra_ms", type=float, default=2.0)
    p.add_argument("--self_check_tol", type=float, default=0.02)
    p.add_argument("--skip_self_check", action="store_true")
    p.add_argument("--out_dir", default="outputs/adaptive_profile/full_vs_adaptive_opt")
    return p.parse_args()


def parse_levels(s):
    xs = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(xs) != 5:
        raise ValueError(xs)
    return xs


def parse_actions(s):
    out = []
    for item in s.split(";"):
        if not item.strip():
            continue
        a = tuple(int(x) for x in item.split(","))
        if len(a) != 3 or any(x < 0 or x >= 5 for x in a):
            raise ValueError(item)
        out.append(a)
    return out


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "mean_ms": float(a.mean()),
        "p50_ms": float(np.percentile(a, 50)),
        "p90_ms": float(np.percentile(a, 90)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
    }


def clear_model_history(model):
    q = getattr(model, "history_feature_queue", None)
    if q is not None:
        q.clear()


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, out


def make_forced_choose_gpu(scheduler, forced_action):
    """Same GPU scheduling work as optimized evaluator, but forced level triplet."""
    a_force, b_force, cd_force = forced_action

    def choose_gpu(_self, gpu_plans, deadline_ms):
        device = gpu_plans["A"][0]["value_gpu"].device
        K = len(scheduler.levels)

        if not hasattr(scheduler, "_latency_lut_cpu"):
            vals = []
            for a in range(K):
                for b in range(K):
                    for cd in range(K):
                        vals.append(
                            float(scheduler.actions[f"{a},{b},{cd}"]["safe_ms"])
                            + scheduler.extra_ms
                        )
            scheduler._latency_lut_cpu = torch.tensor(
                vals, dtype=torch.float32
            ).view(K, K, K)
            scheduler._latency_lut_gpu = None
            scheduler._latency_lut_device = None

        if (
            scheduler._latency_lut_gpu is None
            or scheduler._latency_lut_device != device
        ):
            scheduler._latency_lut_gpu = scheduler._latency_lut_cpu.to(device)
            scheduler._latency_lut_device = device

        ua = torch.stack([x["value_gpu"].float() for x in gpu_plans["A"]])
        ub = torch.stack([x["value_gpu"].float() for x in gpu_plans["B"]])
        uc = torch.stack([x["value_gpu"].float() for x in gpu_plans["CD"]])
        util = (
            scheduler.lambda_a * ua[:, None, None]
            + scheduler.lambda_b * ub[None, :, None]
            + scheduler.lambda_cd * uc[None, None, :]
        )
        lat = scheduler._latency_lut_gpu

        # Execute the actual vectorized 125-action feasibility/argmax work even
        # though the diagnostic forces the final triplet afterwards.
        feasible = lat <= float(deadline_ms)
        masked = util.masked_fill(~feasible, -float("inf"))
        _ = masked.reshape(-1).argmax()

        a = a_force
        b = b_force
        cd = cd_force
        a_gpu = torch.tensor(a, device=device, dtype=torch.long)
        b_gpu = torch.tensor(b, device=device, dtype=torch.long)
        c_gpu = torch.tensor(cd, device=device, dtype=torch.long)

        idx_a = torch.stack([x["idx_gpu"] for x in gpu_plans["A"]])[a_gpu]
        idx_b = torch.stack([x["idx_gpu"] for x in gpu_plans["B"]])[b_gpu]
        idx_c = torch.stack([x["idx_gpu"] for x in gpu_plans["CD"]])[c_gpu]
        feasible_count = feasible.sum()

        selected_lat = lat[a_gpu, b_gpu, c_gpu]
        selected_util = util[a_gpu, b_gpu, c_gpu]
        packed = torch.stack([
            idx_a.float(), idx_b.float(), idx_c.float(), feasible_count.float(),
            selected_lat.float(), selected_util.float(),
        ]).detach().cpu().tolist()
        ia, ib, ic, nfeas = [int(x) for x in packed[:4]]
        selected_lat_cpu = float(packed[4])
        selected_util_cpu = float(packed[5])

        def plan(stage, level, idx):
            rec = gpu_plans[stage][level]
            y0 = idx // rec["Wv"]
            x0 = idx % rec["Wv"]
            return {
                "rect": (y0, y0 + rec["h"], x0, x0 + rec["w"]),
                "ratio": rec["ratio"],
            }

        choice = {
            "key": f"{a},{b},{cd}",
            "a": a,
            "b": b,
            "cd": cd,
            "estimated_safe_ms": selected_lat_cpu,
            "utility": selected_util_cpu,
            "feasible_count": nfeas,
            "no_feasible_action": False,
        }
        selected = {
            "A": plan("A", a, ia),
            "B": plan("B", b, ib),
            "CD": plan("CD", cd, ic),
        }
        return choice, selected

    return types.MethodType(choose_gpu, scheduler)


@torch.no_grad()
def run_full_pass(loader, model, warmup, measure):
    clear_model_history(model)
    current_scene = None
    warmed = 0
    measured = 0
    times = []
    keys = []

    for batch in loader:
        scene, frame = adaptive.get_scene_frame(batch)
        scene_init = scene != current_scene
        if scene_init:
            current_scene = scene
            clear_model_history(model)

        load_data_to_gpu(batch)  # excluded

        if scene_init:
            _ = model(batch)
            torch.cuda.synchronize()
            continue

        if warmed < warmup:
            _ = model(batch)
            torch.cuda.synchronize()
            warmed += 1
            continue

        ms, _ = timed(lambda: model(batch))
        times.append(ms)
        keys.append((scene, frame))
        measured += 1
        if measured >= measure:
            break

    return times, keys


@torch.no_grad()
def run_adaptive_pass(loader, engine, scheduler, action, warmup, measure, deadline):
    engine.reset()
    original_choose = scheduler.choose_gpu
    original_full_fits = scheduler.full_fits
    scheduler.choose_gpu = make_forced_choose_gpu(scheduler, action)
    scheduler.full_fits = types.MethodType(lambda self, d: False, scheduler)

    current_scene = None
    warmed = 0
    measured = 0
    times = []
    keys = []

    try:
        for batch in loader:
            scene, frame = adaptive.get_scene_frame(batch)
            scene_init = scene != current_scene
            if scene_init:
                current_scene = scene
                engine.reset()

            load_data_to_gpu(batch)  # excluded
            token = batch["token"]

            if scene_init or engine.state is None:
                _ = engine.forward(token, deadline, force_full=True)
                torch.cuda.synchronize()
                continue

            if warmed < warmup:
                _ = engine.forward(token, deadline, force_full=False)
                torch.cuda.synchronize()
                warmed += 1
                continue

            ms, _ = timed(
                lambda: engine.forward(token, deadline, force_full=False)
            )
            times.append(ms)
            keys.append((scene, frame))
            measured += 1
            if measured >= measure:
                break
    finally:
        scheduler.choose_gpu = original_choose
        scheduler.full_fits = original_full_fits
        engine.reset()

    return times, keys


def main():
    args = parse_args()
    levels = parse_levels(args.levels)
    actions = parse_actions(args.actions)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    (
        dataset, loader, model, backbone, importance_net, pred_net, logger, _
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

    if not hasattr(scheduler, "choose_gpu"):
        raise RuntimeError(
            "Optimized evaluator not installed: DeadlineScheduler.choose_gpu missing"
        )

    if not args.skip_self_check:
        first = next(iter(loader))
        load_data_to_gpu(first)
        adaptive.self_check_physical_bcd(
            engine, first["token"], args.self_check_tol, logger
        )
        engine.reset()

    results = []
    text = []

    for action in actions:
        # Both passes restart from the same beginning of the validation loader.
        full_times, full_keys = run_full_pass(
            loader, model, args.warmup_frames, args.measure_frames
        )
        adaptive_times, adaptive_keys = run_adaptive_pass(
            loader,
            engine,
            scheduler,
            action,
            args.warmup_frames,
            args.measure_frames,
            args.deadline_ms,
        )

        if full_keys != adaptive_keys:
            raise RuntimeError(
                "Full/adaptive measured frame lists differ. "
                f"first full={full_keys[:3]}, adaptive={adaptive_keys[:3]}"
            )

        fs = stats(full_times)
        ads = stats(adaptive_times)
        saved = fs["mean_ms"] - ads["mean_ms"]
        speedup = fs["mean_ms"] / ads["mean_ms"]
        rec = {
            "action": list(action),
            "frames": len(full_keys),
            "full": fs,
            "adaptive": ads,
            "mean_saved_ms": saved,
            "mean_speedup": speedup,
            "adaptive_under_25ms_mean": ads["mean_ms"] < 25.0,
            "adaptive_under_25ms_p50": ads["p50_ms"] < 25.0,
        }
        results.append(rec)

        line = (
            f"action={action}: Full mean {fs['mean_ms']:.3f} ms -> "
            f"Adaptive mean {ads['mean_ms']:.3f} ms | "
            f"saved {saved:.3f} ms | speedup {speedup:.3f}x | "
            f"Adaptive p50/p90/p99={ads['p50_ms']:.3f}/"
            f"{ads['p90_ms']:.3f}/{ads['p99_ms']:.3f} ms"
        )
        print(line)
        text.append(line)

    payload = {
        "schema_version": 1,
        "timing": "same process, same frames, forward-only, H2D excluded",
        "halos": {"A_layer2": 2, "A_layer3": 2, "A_layer4": 2, "A_neck": 2,
                  "B": args.b_halo, "CD": args.cd_halo},
        "results": results,
    }
    (out / "full_vs_adaptive_optimized.json").write_text(
        json.dumps(payload, indent=2)
    )
    (out / "full_vs_adaptive_optimized.txt").write_text(
        "\n".join(text) + "\n"
    )


if __name__ == "__main__":
    main()