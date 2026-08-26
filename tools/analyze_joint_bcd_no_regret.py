#!/usr/bin/env python3

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        required=True,
    )

    p.add_argument(
        "--guard-ms",
        type=float,
        default=None,
    )

    p.add_argument(
        "--output",
        default=None,
    )

    return p.parse_args()


def cd_key(r):
    return tuple(
        tuple(int(v) for v in roi)
        for roi in r["cd_cores"]
    )


def main():

    args = parse_args()

    data = json.load(
        open(args.input, "r")
    )

    guard = (
        data["guard_ms"]
        if args.guard_ms is None
        else args.guard_ms
    )

    tol = data[
        "numeric_tol"
    ]

    full_p99 = data[
        "full_bcd"
    ][
        "p99_ms"
    ]

    groups = defaultdict(
        list
    )

    for r in data[
        "records"
    ]:

        p99 = r[
            "latency"
        ][
            "p99_ms"
        ]

        if not math.isfinite(
            p99
        ):
            continue

        if (
            r["max_diff"]
            > tol
        ):
            continue

        key = (
            float(
                r[
                    "target_ratio"
                ]
            ),
            int(
                r[
                    "fragments"
                ]
            ),
        )

        groups[
            key
        ].append(
            r
        )

    summaries = []

    print()
    print("=" * 150)

    print(
        f"{'R%':>6} "
        f"{'Frag':>5} "
        f"{'CD':>18} "
        f"{'B':>28} "
        f"{'BExec%':>8} "
        f"{'P99':>8} "
        f"{'SaveFull':>9} "
        f"{'Decision':>12}"
    )

    print("-" * 150)

    all_keys = sorted(
        {
            (
                float(g["target_ratio"]),
                int(g["fragments"]),
            )
            for g in data[
                "groups"
            ]
        }
    )

    for key in all_keys:

        ratio, frag = key

        records = groups.get(
            key,
            [],
        )

        if len(
            records
        ) == 0:

            print(
                f"{ratio*100:5.1f}% "
                f"{frag:5d} "
                f"{'FULL':>18} "
                f"{'FULL':>28} "
                f"{100.0:7.2f}% "
                f"{full_p99:8.3f} "
                f"{0.0:9.3f} "
                f"{'FULL_BCD':>12}"
            )

            summaries.append(
                {
                    "target_ratio":
                        ratio,
                    "fragments":
                        frag,
                    "decision":
                        "FULL_BCD",
                    "best":
                        None,
                }
            )

            continue

        # --------------------------------------------
        # Stage-local B no-regret gate.
        #
        # For every fixed CD physical plan:
        #
        #   select B selective only if
        #
        #   T(B_sel, CD) + guard
        #       < T(B_full, CD)
        #
        # Otherwise B_FULL dominates the selective
        # B plan from a no-regret perspective.
        # --------------------------------------------

        by_cd = defaultdict(
            list
        )

        for r in records:
            by_cd[
                cd_key(r)
            ].append(
                r
            )

        eligible = []

        for _, cd_records in (
            by_cd.items()
        ):

            b_full_records = [
                r
                for r in cd_records
                if r[
                    "b_full"
                ]
            ]

            if len(
                b_full_records
            ) == 0:
                continue

            b_full = min(
                b_full_records,
                key=lambda r:
                    r[
                        "latency"
                    ][
                        "p99_ms"
                    ],
            )

            b_full_p99 = (
                b_full[
                    "latency"
                ][
                    "p99_ms"
                ]
            )

            for r in cd_records:

                p99 = (
                    r[
                        "latency"
                    ][
                        "p99_ms"
                    ]
                )

                if r[
                    "b_full"
                ]:

                    b_valid = True
                    b_save = 0.0

                else:

                    b_save = (
                        b_full_p99
                        -
                        p99
                    )

                    b_valid = (
                        p99
                        +
                        guard
                        <
                        b_full_p99
                    )

                r2 = dict(
                    r
                )

                r2[
                    "b_full_reference_p99_ms"
                ] = b_full_p99

                r2[
                    "b_saving_vs_full_ms"
                ] = b_save

                r2[
                    "b_no_regret_valid"
                ] = b_valid

                if b_valid:
                    eligible.append(
                        r2
                    )

        if len(
            eligible
        ) == 0:

            best = None
            decision = (
                "FULL_BCD"
            )

        else:

            best = min(
                eligible,
                key=lambda r:
                    r[
                        "latency"
                    ][
                        "p99_ms"
                    ],
            )

            best_p99 = (
                best[
                    "latency"
                ][
                    "p99_ms"
                ]
            )

            if (
                best_p99
                +
                guard
                <
                full_p99
            ):
                decision = (
                    "SELECTIVE"
                )
            else:
                decision = (
                    "FULL_BCD"
                )

        if best is None:

            print(
                f"{ratio*100:5.1f}% "
                f"{frag:5d} "
                f"{'FULL':>18} "
                f"{'FULL':>28} "
                f"{100.0:7.2f}% "
                f"{full_p99:8.3f} "
                f"{0.0:9.3f} "
                f"{decision:>12}"
            )

        elif (
            decision
            ==
            "FULL_BCD"
        ):

            print(
                f"{ratio*100:5.1f}% "
                f"{frag:5d} "
                f"{best['cd_label']:>18.18} "
                f"{best['b_label']:>28.28} "
                f"{best['b_exec_ratio']*100:7.2f}% "
                f"{best['latency']['p99_ms']:8.3f} "
                f"{full_p99-best['latency']['p99_ms']:9.3f} "
                f"{decision:>12}"
            )

        else:

            print(
                f"{ratio*100:5.1f}% "
                f"{frag:5d} "
                f"{best['cd_label']:>18.18} "
                f"{best['b_label']:>28.28} "
                f"{best['b_exec_ratio']*100:7.2f}% "
                f"{best['latency']['p99_ms']:8.3f} "
                f"{full_p99-best['latency']['p99_ms']:9.3f} "
                f"{decision:>12}"
            )

        summaries.append(
            {
                "target_ratio":
                    ratio,
                "fragments":
                    frag,
                "decision":
                    decision,
                "best":
                    best,
            }
        )

    print("=" * 150)

    print(
        f"FULL BCD P99 = "
        f"{full_p99:.3f} ms"
    )

    print(
        f"Guard        = "
        f"{guard:.3f} ms"
    )

    if args.output is not None:

        out = Path(
            args.output
        )

        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        out.write_text(
            json.dumps(
                {
                    "source":
                        args.input,
                    "full_bcd_p99_ms":
                        full_p99,
                    "guard_ms":
                        guard,
                    "numeric_tol":
                        tol,
                    "groups":
                        summaries,
                },
                indent=2,
            )
        )

        print(
            "Saved:",
            out,
        )


if __name__ == "__main__":
    main()
