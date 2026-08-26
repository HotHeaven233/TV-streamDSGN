#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        "Analyze D-stage no-regret selective/full crossover"
    )

    p.add_argument(
        "--input",
        required=True,
        help="JSON produced by eval_latency_aware_packing.py",
    )

    p.add_argument(
        "--output",
        required=True,
        help="Output CSV path",
    )

    p.add_argument(
        "--guard-ms",
        type=float,
        default=0.10,
        help=(
            "Minimum latency saving required before "
            "accepting selective execution"
        ),
    )

    p.add_argument(
        "--metric",
        choices=[
            "mean_ms",
            "p95_ms",
            "p99_ms",
        ],
        default="mean_ms",
        help="Latency statistic used for no-regret decision",
    )

    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    # --------------------------------------------------------
    # Validate exact schema produced by eval_latency_aware_packing.py
    # --------------------------------------------------------

    if "full_d" not in data:
        raise RuntimeError(
            "Input JSON has no 'full_d' field."
        )

    if "records" not in data:
        raise RuntimeError(
            "Input JSON has no 'records' field."
        )

    if args.metric not in data["full_d"]:
        raise RuntimeError(
            f"full_d has no metric '{args.metric}'. "
            f"Available keys: {list(data['full_d'].keys())}"
        )

    full_ms = float(
        data["full_d"][args.metric]
    )

    raw_records = data["records"]

    records = []

    for r in raw_records:

        required = [
            "ratio",
            "fragments",
            "strategy",
            "mode",
            "num_rois",
            "predicted_ms",
            "real",
        ]

        missing = [
            k for k in required
            if k not in r
        ]

        if missing:
            raise RuntimeError(
                f"Record is missing fields: {missing}\n"
                f"Record keys: {list(r.keys())}"
            )

        if args.metric not in r["real"]:
            raise RuntimeError(
                f"real has no metric '{args.metric}'. "
                f"Available keys: {list(r['real'].keys())}"
            )

        records.append(
            {
                "ratio": float(r["ratio"]),
                "frag": int(r["fragments"]),
                "strategy": str(r["strategy"]),
                "mode": str(r["mode"]),
                "num_rois": int(r["num_rois"]),
                "predicted_ms": float(
                    r["predicted_ms"]
                ),
                "real_ms": float(
                    r["real"][args.metric]
                ),
                "mean_ms": float(
                    r["real"]["mean_ms"]
                ),
                "p95_ms": float(
                    r["real"]["p95_ms"]
                ),
                "p99_ms": float(
                    r["real"]["p99_ms"]
                ),
            }
        )

    # --------------------------------------------------------
    # Group by semantic recompute ratio + fragmentation
    # --------------------------------------------------------

    groups = {}

    for r in records:
        key = (
            round(r["ratio"], 8),
            r["frag"],
        )

        groups.setdefault(
            key,
            []
        ).append(r)

    rows = []

    print()
    print("=" * 124)
    print(
        f"No-regret analysis"
        f" | metric={args.metric}"
        f" | Full D={full_ms:.3f} ms"
        f" | guard={args.guard_ms:.3f} ms"
    )
    print("=" * 124)

    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'Best selective':>16} "
        f"{'ROI':>4} "
        f"{'Sel(ms)':>9} "
        f"{'Saving':>9} "
        f"{'Decision':>16} "
        f"{'Final(ms)':>10} "
        f"{'Speedup':>8}"
    )

    print("-" * 124)

    for (
        ratio,
        frag
    ) in sorted(groups.keys()):

        candidates = groups[
            (ratio, frag)
        ]

        # IMPORTANT:
        # Only physically selective executions are candidates here.
        #
        # latency_v2 may already return mode="full".
        # That record must NOT be counted as a selective candidate.
        selective_candidates = [
            r
            for r in candidates
            if r["mode"] == "selective"
        ]

        if selective_candidates:

            best = min(
                selective_candidates,
                key=lambda x: x["real_ms"]
            )

            saving_ms = (
                full_ms
                - best["real_ms"]
            )

            # Require positive saving greater than guard.
            use_selective = (
                saving_ms
                > args.guard_ms
            )

        else:

            best = None
            saving_ms = 0.0
            use_selective = False

        if use_selective:

            decision = best["strategy"]

            final_ms = best[
                "real_ms"
            ]

            best_strategy = best[
                "strategy"
            ]

            best_roi = best[
                "num_rois"
            ]

            best_ms = best[
                "real_ms"
            ]

        else:

            decision = "FULL"

            final_ms = full_ms

            if best is None:
                best_strategy = "N/A"
                best_roi = 0
                best_ms = float("nan")
            else:
                best_strategy = best[
                    "strategy"
                ]
                best_roi = best[
                    "num_rois"
                ]
                best_ms = best[
                    "real_ms"
                ]

        speedup = (
            full_ms
            / final_ms
        )

        row = {
            "ratio": ratio,
            "ratio_percent": ratio * 100.0,
            "fragments": frag,
            "metric": args.metric,
            "best_selective_strategy": best_strategy,
            "best_selective_num_rois": best_roi,
            "best_selective_ms": best_ms,
            "full_ms": full_ms,
            "saving_ms": saving_ms,
            "guard_ms": args.guard_ms,
            "decision": decision,
            "no_regret_ms": final_ms,
            "speedup_vs_full": speedup,
        }

        rows.append(row)

        print(
            f"{ratio * 100:6.1f}% "
            f"{frag:5d} "
            f"{best_strategy:>16} "
            f"{best_roi:4d} "
            f"{best_ms:9.3f} "
            f"{saving_ms:9.3f} "
            f"{decision:>16} "
            f"{final_ms:10.3f} "
            f"{speedup:8.3f}"
        )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        out,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        "Saved:",
        out
    )

    # --------------------------------------------------------
    # Crossover summary
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print("No-regret crossover summary")
    print("=" * 90)

    all_frags = sorted(
        {
            r["fragments"]
            for r in rows
        }
    )

    for frag in all_frags:

        frag_rows = [
            r
            for r in rows
            if r["fragments"] == frag
        ]

        selective_rows = [
            r
            for r in frag_rows
            if r["decision"] != "FULL"
        ]

        full_rows = [
            r
            for r in frag_rows
            if r["decision"] == "FULL"
        ]

        if selective_rows:

            max_sel = max(
                r["ratio_percent"]
                for r in selective_rows
            )

            msg = (
                f"frag={frag:2d}: "
                f"selective survives through "
                f"{max_sel:.1f}%"
            )

        else:

            msg = (
                f"frag={frag:2d}: "
                f"no selective setting clears guard"
            )

        if full_rows:

            min_full = min(
                r["ratio_percent"]
                for r in full_rows
            )

            msg += (
                f", first FULL at "
                f"{min_full:.1f}%"
            )

        print(msg)


if __name__ == "__main__":
    main()
