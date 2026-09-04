#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics

import torch

from smooth_cuda_contention import (
    SmoothCudaContention,
    make_high_priority_detector_stream,
    stream_priority_range,
    stream_priority_value,
)


def percentile(xs, q):
    ys = sorted(float(x) for x in xs)
    if not ys:
        return float("nan")
    if len(ys) == 1:
        return ys[0]
    pos = (len(ys) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    frac = pos - lo
    return ys[lo] * (1.0 - frac) + ys[hi] * frac


def stats(xs):
    return {
        "n": len(xs),
        "mean_ms": statistics.fmean(xs),
        "p50_ms": percentile(xs, 0.50),
        "p90_ms": percentile(xs, 0.90),
        "p99_ms": percentile(xs, 0.99),
        "min_ms": min(xs),
        "max_ms": max(xs),
    }


def timed_matmul(stream, a, b):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(stream), torch.no_grad():
        start.record()
        _ = torch.mm(a, b)
        end.record()

    end.synchronize()
    return float(start.elapsed_time(end))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strength", type=float, default=0.125)
    p.add_argument("--window-ms", type=float, default=100.0)
    p.add_argument("--slice-ms", type=float, default=0.05)
    p.add_argument("--start-delay-ms", type=float, default=2.0)
    p.add_argument("--threads", type=int, default=256)
    p.add_argument("--matrix-size", type=int, default=2048)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--repeats", type=int, default=30)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device_index = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_index}")

    detector = make_high_priority_detector_stream(device_index)

    # IMPORTANT:
    # Allocate matrices BEFORE timing so allocator / first-use costs cannot
    # contaminate the baseline measurement.
    n = int(args.matrix_size)
    a = torch.randn((n, n), device=device, dtype=torch.float16)
    b = torch.randn((n, n), device=device, dtype=torch.float16)

    # Force CUDA context, cuBLAS and selected GEMM algorithm to warm up.
    with torch.cuda.stream(detector), torch.no_grad():
        for _ in range(int(args.warmup)):
            _ = torch.mm(a, b)
    detector.synchronize()

    # Clean baseline repetitions with the exact same matrices and stream.
    baseline = [
        timed_matmul(detector, a, b)
        for _ in range(int(args.repeats))
    ]

    contender = SmoothCudaContention(
        strength=args.strength,
        window_ms=args.window_ms,
        slice_ms=args.slice_ms,
        start_delay_ms=args.start_delay_ms,
        threads=args.threads,
        device=device_index,
    )

    stressed = []
    chain_actual = []

    # Each repetition receives a fresh finite contention window.
    for _ in range(int(args.repeats)):
        contender.launch()
        ms = timed_matmul(detector, a, b)
        chain_ms = contender.finish()
        stressed.append(ms)
        chain_actual.append(chain_ms)

    base_stats = stats(baseline)
    stress_stats = stats(stressed)
    chain_stats = stats(chain_actual)

    result = {
        "priority_requests": {
            "background_detector": list(stream_priority_range()),
            "detector_actual": stream_priority_value(detector),
            "background_actual": contender.background_priority_actual,
        },
        "configuration": {
            "matrix_size": n,
            "dtype": "fp16",
            "strength": args.strength,
            "slice_ms": args.slice_ms,
            "window_ms": args.window_ms,
            "start_delay_ms": args.start_delay_ms,
            "threads": args.threads,
            "warmup": args.warmup,
            "repeats": args.repeats,
        },
        "baseline": base_stats,
        "stressed": stress_stats,
        "slowdown_p50": (
            stress_stats["p50_ms"] / base_stats["p50_ms"]
        ),
        "contention_chain_actual": chain_stats,
        "raw": {
            "baseline_ms": baseline,
            "stressed_ms": stressed,
            "chain_ms": chain_actual,
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
