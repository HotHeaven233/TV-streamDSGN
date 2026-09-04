#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


STAGE_METRICS = (
    "fixed_prefix_ms",
    "res2_ms",
    "res3_ms",
    "res4_ms",
    "fpn_ms",
    "stereo_ms",
    "rpn_ms",
    "fixed_tail_ms",
    "forward_total_ms",
    "remain_after_prefix_ms",
    "remain_after_res2_ms",
    "remain_after_res3_ms",
    "remain_after_res4_ms",
    "remain_after_fpn_ms",
    "remain_after_stereo_ms",
    "remain_after_rpn_ms",
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--schedule_tsv", required=True)
    p.add_argument("--profile_root", required=True)
    p.add_argument("--quality_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_json", required=True)
    p.add_argument("--allow_partial", action="store_true")
    args = p.parse_args()

    with open(args.schedule_tsv, newline="") as f:
        schedules = list(csv.DictReader(f, delimiter="\t"))

    with open(args.quality_csv, newline="") as f:
        qrows = list(csv.DictReader(f))
    qmap = {int(r["id"]): r for r in qrows}

    root = Path(args.profile_root)
    merged = []
    missing = []

    for s in schedules:
        pid = int(s["id"])
        tag = s["tag"]
        path = root / f"{pid:03d}_{tag}" / "forward_profile_summary.json"
        if not path.exists():
            missing.append(pid)
            continue

        timing = json.loads(path.read_text())
        q = qmap.get(pid, {})

        row = {
            "id": pid,
            "schedule": s["schedule"],
            "tag": tag,
            "res2": s["res2"],
            "res3": s["res3"],
            "res4": s["res4"],
            "fpn": s["fpn"],
            "stereo": s["stereo"],
            "rpn": s["rpn"],
            "reference_mean3d_moderate": q.get(
                "reference_mean3d_moderate", ""
            ),
            "car_3d_iou07_moderate": q.get(
                "car_3d_iou07_moderate", ""
            ),
            "pedestrian_3d_iou05_moderate": q.get(
                "pedestrian_3d_iou05_moderate", ""
            ),
            "cyclist_3d_iou05_moderate": q.get(
                "cyclist_3d_iou05_moderate", ""
            ),
        }

        for metric in STAGE_METRICS:
            st = timing["metrics"][metric]
            for stat in ("mean_ms", "p50_ms", "p90_ms", "p99_ms"):
                row[f"{metric}_{stat.replace('_ms','')}"] = st[stat]

        row["profile_summary_path"] = str(path)
        merged.append(row)

    # Mark p50 Pareto points using the reference quality score only.
    # This is an analysis convenience, not the final controller objective.
    for row in merged:
        row["pareto_reference_p50"] = 0

    valid = [
        r for r in merged
        if r["reference_mean3d_moderate"] not in ("", None)
    ]
    for a in valid:
        qa = float(a["reference_mean3d_moderate"])
        ca = float(a["forward_total_ms_p50"])
        dominated = False
        for b in valid:
            if a is b:
                continue
            qb = float(b["reference_mean3d_moderate"])
            cb = float(b["forward_total_ms_p50"])
            if (qb >= qa and cb <= ca) and (qb > qa or cb < ca):
                dominated = True
                break
        if not dominated:
            a["pareto_reference_p50"] = 1

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if merged:
        columns = list(merged[0].keys())
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=columns)
            w.writeheader()
            w.writerows(merged)

    pareto = sorted(
        [
            {
                "id": int(r["id"]),
                "schedule": r["schedule"],
                "reference_quality": float(r["reference_mean3d_moderate"]),
                "forward_p50_ms": float(r["forward_total_ms_p50"]),
                "forward_p99_ms": float(r["forward_total_ms_p99"]),
            }
            for r in valid
            if int(r["pareto_reference_p50"]) == 1
        ],
        key=lambda x: x["forward_p50_ms"],
    )

    result = {
        "expected_profiles": 84,
        "completed_profiles": len(merged),
        "missing_profile_ids": missing,
        "merged_csv": str(out_csv),
        "reference_p50_pareto": pareto,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if missing and not args.allow_partial:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

