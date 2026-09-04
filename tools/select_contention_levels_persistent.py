#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate-dir", required=True)
    p.add_argument("--duration-ms", type=float, required=True)
    p.add_argument("--start-delay-ms", type=float, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--targets", default="1.20,1.45,1.80,2.30")
    p.add_argument("--min-p50-gap-ms", type=float, default=0.08)
    p.add_argument("--output", required=True)
    return p.parse_args()


def classify(x, centroids):
    return min(
        range(len(centroids)),
        key=lambda i: abs(float(x) - float(centroids[i])),
    )


def main():
    args = parse_args()

    base = json.loads(Path(args.baseline).read_text())
    base_p50 = float(base["probe"]["p50_ms"])

    rows = []
    pat = re.compile(r"probe_s([0-9p]+)\.json$")
    for p in sorted(Path(args.candidate_dir).glob("probe_s*.json")):
        m = pat.search(p.name)
        if not m:
            continue

        strength = float(m.group(1).replace("p", "."))
        d = json.loads(p.read_text())

        rows.append({
            "strength": strength,
            "probe_p10_ms": float(d["probe"]["p10_ms"]),
            "probe_p50_ms": float(d["probe"]["p50_ms"]),
            "probe_p90_ms": float(d["probe"]["p90_ms"]),
            "probe_p99_ms": float(d["probe"]["p99_ms"]),
            "probe_std_ms": float(d["probe"]["std_ms"]),
            "slowdown": float(d["probe"]["p50_ms"]) / base_p50,
            "raw_ms": [float(x) for x in d["raw_ms"]],
            "contention": d["contention"],
            "probe_json": str(p),
        })

    rows.sort(key=lambda x: x["strength"])
    if len(rows) < 4:
        raise RuntimeError("Need at least four contention-strength candidates")

    # Keep a measured monotonic envelope rather than assuming nominal strength
    # translates monotonically to runtime pressure.
    envelope = []
    best_p50 = base_p50
    for r in rows:
        if r["probe_p50_ms"] >= best_p50 + args.min_p50_gap_ms:
            envelope.append(r)
            best_p50 = r["probe_p50_ms"]

    if len(envelope) < 4:
        compact = [
            {
                "strength": r["strength"],
                "p50_ms": round(r["probe_p50_ms"], 4),
                "p90_ms": round(r["probe_p90_ms"], 4),
                "slowdown": round(r["slowdown"], 3),
            }
            for r in rows
        ]
        raise RuntimeError(
            "Fewer than 4 clearly separated persistent-contention points. "
            f"Measured candidates: {compact}. "
            "Increase STRENGTHS or CONTENTION_THREADS."
        )

    targets = [float(x) for x in args.targets.split(",")]
    if len(targets) != 4:
        raise ValueError("Exactly four target slowdowns are required")

    best_combo = None
    best_cost = float("inf")

    for combo in itertools.combinations(envelope, 4):
        cost = sum(
            (
                math.log(max(c["slowdown"], 1e-6))
                - math.log(target)
            ) ** 2
            for c, target in zip(combo, targets)
        )
        if cost < best_cost:
            best_combo = combo
            best_cost = cost

    selected = list(best_combo)

    levels = [{
        "level": "L0",
        "strength": 0.0,
        "probe_p10_ms": float(base["probe"]["p10_ms"]),
        "probe_p50_ms": float(base["probe"]["p50_ms"]),
        "probe_p90_ms": float(base["probe"]["p90_ms"]),
        "probe_p99_ms": float(base["probe"]["p99_ms"]),
        "probe_std_ms": float(base["probe"]["std_ms"]),
        "slowdown": 1.0,
        "raw_ms": [float(x) for x in base["raw_ms"]],
        "contention": None,
    }]

    for i, r in enumerate(selected, 1):
        levels.append({
            "level": f"L{i}",
            "strength": r["strength"],
            "probe_p10_ms": r["probe_p10_ms"],
            "probe_p50_ms": r["probe_p50_ms"],
            "probe_p90_ms": r["probe_p90_ms"],
            "probe_p99_ms": r["probe_p99_ms"],
            "probe_std_ms": r["probe_std_ms"],
            "slowdown": r["slowdown"],
            "raw_ms": r["raw_ms"],
            "contention": r["contention"],
        })

    centroids = [x["probe_p50_ms"] for x in levels]
    thresholds = [
        (centroids[i] + centroids[i + 1]) / 2.0
        for i in range(4)
    ]

    confusion = [[0 for _ in range(5)] for _ in range(5)]
    total = 0
    correct = 0
    per_level_accuracy = {}

    for true_idx, level in enumerate(levels):
        for x in level["raw_ms"]:
            pred = classify(x, centroids)
            confusion[true_idx][pred] += 1
            total += 1
            if pred == true_idx:
                correct += 1

        per_level_accuracy[level["level"]] = (
            confusion[true_idx][true_idx]
            / max(len(level["raw_ms"]), 1)
        )

    compact_levels = []
    for x in levels:
        y = dict(x)
        y.pop("raw_ms")
        compact_levels.append(y)

    result = {
        "workload": {
            "type": "finite_persistent_cuda_arithmetic_kernel",
            "vary_only": "nominal strength = blocks / GPU SM count",
            "duration_ms": args.duration_ms,
            "start_delay_ms": args.start_delay_ms,
            "threads": args.threads,
            "note": (
                "nominal strength is only the load-generator setting; "
                "actual level is defined by measured fixed-prefix latency"
            ),
        },
        "targets": targets,
        "levels": compact_levels,
        "nearest_centroid_thresholds_ms": thresholds,
        "classifier_validation": {
            "centroids_ms": centroids,
            "confusion_matrix_rows_true_cols_pred": confusion,
            "per_level_accuracy": per_level_accuracy,
            "overall_accuracy": correct / max(total, 1),
        },
        "all_candidates": [
            {k: v for k, v in r.items() if k != "raw_ms"}
            for r in rows
        ],
        "monotonic_envelope": [
            {k: v for k, v in r.items() if k != "raw_ms"}
            for r in envelope
        ],
    }

    p = Path(args.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

