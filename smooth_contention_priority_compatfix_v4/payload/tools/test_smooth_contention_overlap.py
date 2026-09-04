#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import torch

from smooth_cuda_contention import (
    SmoothCudaContention,
    make_high_priority_detector_stream,
    stream_priority_range,
    stream_priority_value,
)


def one_matmul(stream, n=2048):
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        s.record()
        _ = a @ b
        e.record()
    e.synchronize()
    return float(s.elapsed_time(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strength", type=float, default=0.125)
    p.add_argument("--window-ms", type=float, default=100.0)
    p.add_argument("--slice-ms", type=float, default=0.05)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    device = torch.cuda.current_device()
    detector = make_high_priority_detector_stream(device)

    baseline = one_matmul(detector)

    cont = SmoothCudaContention(
        strength=args.strength,
        window_ms=args.window_ms,
        slice_ms=args.slice_ms,
        start_delay_ms=2.0,
        threads=256,
        device=device,
    )
    cont.launch()
    stressed = one_matmul(detector)
    chain_actual = cont.finish()

    result = {
        "priority_range": stream_priority_range(),
        "detector_priority": stream_priority_value(detector),
        "background_priority": cont.background_priority_actual,
        "strength": args.strength,
        "slice_ms": args.slice_ms,
        "baseline_matmul_ms": baseline,
        "stressed_matmul_ms": stressed,
        "slowdown": stressed / baseline,
        "contention_chain_actual_ms": chain_actual,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
