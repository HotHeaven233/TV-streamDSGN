#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from tv_stream3d_controller import (
    CHECKPOINTS,
    TVStream3DController,
    is_monotonic_schedule,
    parse_schedule,
)


LEVELS = ("L0", "L1", "L2", "L3", "L4")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--controller-csv", required=True)
    p.add_argument("--levels-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--decision-csv", required=True)
    p.add_argument(
        "--frequencies",
        default="20,25,30,35,40",
    )
    return p.parse_args()


def main():
    args = parse_args()

    controller = TVStream3DController(
        controller_csv=args.controller_csv,
        contention_levels_json=args.levels_json,
        bound_column="remaining_p99_ms",
        classifier="conservative_gap",
    )

    rows = []
    with open(args.controller_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    checks = {}

    checks["rows"] = len(rows)
    checks["expected_rows"] = 5 * 84 * 7

    ids_by_level_checkpoint = defaultdict(set)
    q_by_pid = defaultdict(set)
    schedule_by_pid = {}

    values = {}

    for r in rows:
        level = r["level"]
        cp = r["checkpoint"]
        pid = int(r["profile_id"])
        s = parse_schedule(r["schedule"])

        ids_by_level_checkpoint[(level, cp)].add(pid)
        q_by_pid[pid].add(float(r["reference_quality"]))
        schedule_by_pid[pid] = s

        values[(level, cp, pid)] = {
            "p50": float(r["remaining_p50_ms"]),
            "p90": float(r["remaining_p90_ms"]),
            "p99": float(r["remaining_p99_ms"]),
        }

    checks["all_cells_have_84_profiles"] = all(
        len(ids_by_level_checkpoint[(L, cp)]) == 84
        for L in LEVELS
        for cp in CHECKPOINTS
    )

    checks["quality_invariant_across_levels"] = all(
        len(v) == 1
        for v in q_by_pid.values()
    )

    checks["all_schedules_monotonic"] = all(
        is_monotonic_schedule(s)
        for s in schedule_by_pid.values()
    )

    # For each profile/checkpoint, stronger contention must not lower the
    # measured suffix quantile.
    violations_contention = []
    for cp in CHECKPOINTS:
        for pid in range(1, 85):
            for stat in ("p50", "p90", "p99"):
                xs = [
                    values[(L, cp, pid)][stat]
                    for L in LEVELS
                ]
                if any(xs[i + 1] < xs[i] for i in range(4)):
                    violations_contention.append(
                        {
                            "checkpoint": cp,
                            "profile_id": pid,
                            "stat": stat,
                            "values": xs,
                        }
                    )

    checks["contention_monotonic_violations"] = len(
        violations_contention
    )

    # Within one level/profile, the remaining suffix must shrink as forward
    # execution advances.
    violations_checkpoint = []
    for L in LEVELS:
        for pid in range(1, 85):
            xs = [
                values[(L, cp, pid)]["p99"]
                for cp in CHECKPOINTS
            ]
            if any(xs[i + 1] > xs[i] for i in range(6)):
                violations_checkpoint.append(
                    {
                        "level": L,
                        "profile_id": pid,
                        "values": xs,
                    }
                )

    checks["suffix_checkpoint_violations"] = len(
        violations_checkpoint
    )

    frequencies = [
        float(x)
        for x in args.frequencies.split(",")
    ]

    decision_rows = []

    # Prefix-level policy audit using each calibrated level's probe p50 as
    # elapsed time at the end of the fixed probe.
    for L in LEVELS:
        probe_p50 = float(
            controller.level_rows[L]["probe_p50_ms"]
        )
        classified = controller.classify_probe(probe_p50)

        for hz in frequencies:
            deadline = 1000.0 / hz
            d = controller.decide(
                checkpoint="after_prefix",
                elapsed_ms=probe_p50,
                deadline_ms=deadline,
                observed_level=classified,
                executed_prefix=(),
            )

            decision_rows.append(
                {
                    "true_level": L,
                    "probe_p50_ms": probe_p50,
                    "classified_level": classified,
                    "frequency_hz": hz,
                    "deadline_ms": deadline,
                    "feasible": int(d.feasible),
                    "profile_id": d.profile_id,
                    "schedule": ",".join(
                        str(x) for x in d.schedule
                    ),
                    "quality": d.quality,
                    "remaining_budget_ms": (
                        d.remaining_budget_ms
                    ),
                    "remaining_p99_bound_ms": (
                        d.remaining_bound_ms
                    ),
                    "predicted_total_bound_ms": (
                        d.elapsed_ms
                        + d.remaining_bound_ms
                    ),
                    "next_width": d.next_width,
                }
            )

    decision_path = Path(args.decision_csv)
    decision_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with decision_path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                decision_rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(decision_rows)

    result = {
        "checks": checks,
        "classifier": {
            "mode": controller.classifier,
            "centroid_thresholds_ms": (
                controller.centroid_thresholds
            ),
            "conservative_thresholds_ms": (
                controller.conservative_thresholds
            ),
        },
        "decision_audit_csv": str(
            decision_path
        ),
        "pass": (
            checks["rows"] == checks["expected_rows"]
            and checks["all_cells_have_84_profiles"]
            and checks["quality_invariant_across_levels"]
            and checks["all_schedules_monotonic"]
            and checks["contention_monotonic_violations"] == 0
            and checks["suffix_checkpoint_violations"] == 0
        ),
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nPREFIX DECISION AUDIT")
    for r in decision_rows:
        print(
            f'{r["true_level"]} '
            f'{r["frequency_hz"]:>4.0f}Hz '
            f'cls={r["classified_level"]} '
            f'feasible={r["feasible"]} '
            f'id={r["profile_id"]:02d} '
            f'Q={r["quality"]:.3f} '
            f'bound={r["predicted_total_bound_ms"]:.3f}/'
            f'{r["deadline_ms"]:.3f} ms '
            f'{r["schedule"]}'
        )

    if not result["pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
