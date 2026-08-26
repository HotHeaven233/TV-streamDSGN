#!/usr/bin/env python3

import argparse
import json


def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument(
        "--input",
        required=True,
    )

    return p.parse_args()


def main():

    args = parse_args()

    data = json.load(
        open(
            args.input,
            "r",
        )
    )

    print()
    print(
        "=" * 130
    )

    print(
        f"{'R%':>6} "
        f"{'Frag':>5} "
        f"{'CD':>7} "
        f"{'Omega%':>8} "
        f"{'CCBox%':>8} "
        f"{'Tile':>8} "
        f"{'Rects':>6} "
        f"{'Cover%':>8} "
        f"{'Keep%':>8} "
        f"{'Cap4%':>8} "
        f"{'Cap2%':>8} "
        f"{'BBox%':>8}"
    )

    print(
        "-" * 130
    )

    for r in data[
        "records"
    ]:

        # Select the tile shape whose KEEP execution
        # has minimum geometric coverage.
        #
        # This is geometry-only, NOT a latency oracle.

        best = min(
            r[
                "tile_cover_stats"
            ],
            key=lambda x:
                x[
                    "plans"
                ][
                    "keep"
                ][
                    "exec_ratio"
                ],
        )

        plans = best[
            "plans"
        ]

        print(
            f"{r['semantic_ratio']*100:5.1f}% "
            f"{r['fragments']:5d} "
            f"{r['cd_strategy']:>7} "
            f"{r['omega_ratio']*100:7.2f}% "
            f"{r['component_bbox_union_ratio']*100:7.2f}% "
            f"{best['tile_h']}x{best['tile_w']:<4} "
            f"{best['num_cover_rects']:6d} "
            f"{best['cover_ratio']*100:7.2f}% "
            f"{plans['keep']['exec_ratio']*100:7.2f}% "
            f"{plans['cap4']['exec_ratio']*100:7.2f}% "
            f"{plans['cap2']['exec_ratio']*100:7.2f}% "
            f"{plans['bbox1']['exec_ratio']*100:7.2f}%"
        )


if __name__ == "__main__":
    main()
