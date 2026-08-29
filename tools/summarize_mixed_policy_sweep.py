#!/usr/bin/env python3

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        required=True,
    )

    parser.add_argument(
        "--output_csv",
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(
        args.root
    )

    files = sorted(
        root.glob(
            "*hz/*/summary.json"
        )
    )

    if not files:
        raise RuntimeError(
            f"No summary.json found under {root}"
        )

    rows = []

    for path in files:

        with open(
            path,
            "r",
        ) as f:
            x = json.load(f)

        rows.append(
            {
                "hz":
                    float(
                        x["input_hz"]
                    ),

                "policy":
                    x["policy"],

                "nominal_F":
                    float(
                        x[
                            "nominal_full_fraction"
                        ]
                    ),

                "realized_F":
                    float(
                        x[
                            "realized_full_fraction"
                        ]
                    ),

                "sAP3D07_mod_mean":
                    float(
                        x[
                            "mean_sAP3D_moderate_0.7"
                        ]
                    ),

                "sAP3D05_mod_mean":
                    float(
                        x[
                            "mean_sAP3D_moderate_0.5"
                        ]
                    ),

                "processed":
                    int(
                        x["processed_frames"]
                    ),

                "dropped":
                    int(
                        x["dropped_frames"]
                    ),

                "drop_ratio":
                    float(
                        x["drop_ratio"]
                    ),

                "mean_service":
                    x[
                        "service_ms"
                    ][
                        "mean_all"
                    ],

                "mean_F_service":
                    x[
                        "service_ms"
                    ][
                        "mean_full"
                    ],

                "mean_L_service":
                    x[
                        "service_ms"
                    ][
                        "mean_light"
                    ],

                "mean_output_latency":
                    x[
                        "output_latency_ms"
                    ][
                        "mean"
                    ],

                "path":
                    str(
                        path.parent
                    ),
            }
        )

    rows.sort(
        key=lambda x: (
            x["hz"],
            -x[
                "sAP3D07_mod_mean"
            ],
        )
    )

    output_csv = (
        Path(
            args.output_csv
        )
        if args.output_csv
        else
        root
        /
        "mixed_policy_summary.csv"
    )

    with open(
        output_csv,
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
        writer.writerows(
            rows
        )

    grouped = defaultdict(
        list
    )

    for row in rows:
        grouped[
            row["hz"]
        ].append(
            row
        )

    print()
    print("=" * 120)
    print(
        "FIXED MIXED-POLICY VERDICT"
    )
    print("=" * 120)

    for hz in sorted(
        grouped
    ):

        group = grouped[
            hz
        ]

        by_policy = {
            x["policy"]: x
            for x in group
        }

        print()
        print(
            f"{hz:g} Hz"
        )

        print(
            "-" * 120
        )

        print(
            f"{'policy':>12} "
            f"{'F ratio':>10} "
            f"{'sAP07':>10} "
            f"{'drop':>10} "
            f"{'lat(ms)':>10}"
        )

        for x in group:

            print(
                f"{x['policy']:>12} "
                f"{x['realized_F']:>10.3f} "
                f"{x['sAP3D07_mod_mean']:>10.4f} "
                f"{x['drop_ratio']:>10.4f} "
                f"{x['mean_output_latency']:>10.3f}"
            )

        if (
            "F" not in by_policy
            or
            "L" not in by_policy
        ):
            print(
                "[WARN] F/L endpoint missing."
            )
            continue

        endpoint_best = max(
            by_policy["F"][
                "sAP3D07_mod_mean"
            ],
            by_policy["L"][
                "sAP3D07_mod_mean"
            ],
        )

        mixed = [
            x
            for x in group
            if x["policy"]
            not in (
                "F",
                "L",
            )
        ]

        if not mixed:
            continue

        best_mixed = max(
            mixed,
            key=lambda x:
                x[
                    "sAP3D07_mod_mean"
                ],
        )

        gain = (
            best_mixed[
                "sAP3D07_mod_mean"
            ]
            -
            endpoint_best
        )

        print()

        print(
            f"Best endpoint : "
            f"{endpoint_best:.4f}"
        )

        print(
            f"Best mixed    : "
            f"{best_mixed['policy']} "
            f"= "
            f"{best_mixed['sAP3D07_mod_mean']:.4f}"
        )

        print(
            f"Mixed gain    : "
            f"{gain:+.4f} AP"
        )

        if gain > 0:
            print(
                "VERDICT       : "
                "MIXED BEATS BOTH ENDPOINTS"
            )
        else:
            print(
                "VERDICT       : "
                "NO MIXED POLICY BEATS BOTH ENDPOINTS"
            )

    print()
    print("=" * 120)

    print(
        f"CSV saved: {output_csv}"
    )


if __name__ == "__main__":
    main()
