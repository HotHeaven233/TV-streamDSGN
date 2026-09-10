#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table


METHODS = [
    ("StreamDSGN", "original"),
    ("Streamer-style", "streamer"),
    ("MTD", "mtd"),
    ("Transtreaming-style", "transtreaming"),
    ("TV-streamDSGN", "tv"),
]

LEVELS = ["L1", "L2", "L3", "L4"]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--repo_root",
        default="/data/jhb/workspace/streamDSGN",
    )
    p.add_argument(
        "--hz",
        type=int,
        default=35,
    )
    p.add_argument(
        "--trace_seed",
        type=int,
        default=20260903,
    )
    p.add_argument(
        "--pressure_fraction",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--tv_exp_name",
        default="elastic_bev_v4_bn_from_k3",
    )
    p.add_argument(
        "--output_dir",
        default=(
            "outputs/paper_figures/"
            "random50_35hz_table"
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return p.parse_args()


def fraction_tag(x):
    return f"{x:g}"


def summary_path(
    repo,
    kind,
    hz,
    level,
    seed,
    fraction,
    tv_exp_name,
):
    run_tag = (
        f"{hz}Hz_forward_only_seed{seed}"
    )

    level_tag = (
        f"L0_{level}_p{fraction_tag(fraction)}"
    )

    if kind == "original":
        return (
            repo
            / "outputs"
            / "original_streamdsgn"
            / "formal_streaming_random50"
            / run_tag
            / level_tag
            / "summary.json"
        )

    if kind == "streamer":
        return (
            repo
            / "outputs"
            / "streamer_style_streamdsgn"
            / "formal_streaming_random50"
            / run_tag
            / level_tag
            / "summary.json"
        )

    if kind == "mtd":
        return (
            repo
            / "outputs"
            / "mtd_three_head"
            / "formal_streaming_random50"
            / run_tag
            / level_tag
            / "summary.json"
        )

    if kind == "transtreaming":
        return (
            repo
            / "outputs"
            / "transtreaming_v2_best"
            / "formal_streaming_random50"
            / run_tag
            / level_tag
            / "summary.json"
        )

    if kind == "tv":
        return (
            repo
            / "outputs"
            / "elastic_bev"
            / tv_exp_name
            / "formal_streaming_random50"
            / run_tag
            / level_tag
            / "summary.json"
        )

    raise ValueError(
        f"Unknown method kind: {kind}"
    )


def read_summary(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing summary:\n  {path}"
        )

    with path.open() as f:
        s = json.load(f)

    q = s.get(
        "stream_sap_3d_moderate_R40"
    )

    if q is None:
        raise KeyError(
            f"{path}: missing "
            "stream_sap_3d_moderate_R40"
        )

    if "Macro" not in q:
        raise KeyError(
            f"{path}: missing Macro"
        )

    if "deadline_miss_rate" not in s:
        raise KeyError(
            f"{path}: missing "
            "deadline_miss_rate"
        )

    macro = float(
        q["Macro"]
    )

    # Stored as fraction: misses / processed.
    miss_pct = (
        float(
            s["deadline_miss_rate"]
        )
        * 100.0
    )

    drop_pct = (
        float(
            s.get(
                "drop_rate",
                0.0,
            )
        )
        * 100.0
    )

    return {
        "macro": macro,
        "miss_pct": miss_pct,
        "drop_pct": drop_pct,
    }


def collect(args, repo):
    rows = []

    for method_name, kind in METHODS:

        row = {
            "Method": method_name,
        }

        for level in LEVELS:

            path = summary_path(
                repo=repo,
                kind=kind,
                hz=args.hz,
                level=level,
                seed=args.trace_seed,
                fraction=args.pressure_fraction,
                tv_exp_name=args.tv_exp_name,
            )

            result = read_summary(
                path
            )

            row[
                f"{level}_Macro"
            ] = result["macro"]

            row[
                f"{level}_MissPct"
            ] = result["miss_pct"]

            # Keep Drop in CSV for later use,
            # but do not draw it in the main table.
            row[
                f"{level}_DropPct"
            ] = result["drop_pct"]

            print(
                f"[READ] "
                f"{method_name:<20s} "
                f"L0/{level} | "
                f"Macro={result['macro']:8.4f} | "
                f"Miss={result['miss_pct']:8.3f}% | "
                f"Drop={result['drop_pct']:8.3f}%"
            )

        rows.append(
            row
        )

    return rows


def write_csv(rows, path):
    fields = [
        "Method"
    ]

    for level in LEVELS:
        fields.extend([
            f"{level}_Macro",
            f"{level}_MissPct",
            f"{level}_DropPct",
        ])

    with path.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()
        w.writerows(
            rows
        )


def find_best(rows):
    best = {}

    for level in LEVELS:

        macros = [
            float(
                r[f"{level}_Macro"]
            )
            for r in rows
        ]

        misses = [
            float(
                r[f"{level}_MissPct"]
            )
            for r in rows
        ]

        max_macro = max(
            macros
        )

        min_miss = min(
            misses
        )

        best[level] = {
            "macro": {
                i
                for i, x in enumerate(macros)
                if abs(
                    x - max_macro
                ) < 1e-8
            },
            "miss": {
                i
                for i, x in enumerate(misses)
                if abs(
                    x - min_miss
                ) < 1e-8
            },
        }

    return best


def configure_style():
    plt.rcParams.update({
        "font.family":
            "sans-serif",

        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],

        "font.size": 8,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def render_table(
    rows,
    output_path,
    args,
):
    configure_style()

    best = find_best(
        rows
    )

    fig, ax = plt.subplots(
        figsize=(13.2, 3.45)
    )

    ax.set_axis_off()

    table = Table(
        ax,
        bbox=[
            0.015,
            0.08,
            0.97,
            0.76,
        ],
    )

    total_rows = (
        len(rows)
        + 2
    )

    row_h = (
        1.0
        / total_rows
    )

    method_w = 0.19

    metric_w = (
        1.0
        - method_w
    ) / 8.0

    edge = "#555555"
    header_face = "#eeeeee"
    subheader_face = "#f7f7f7"
    tv_face = "#fafafa"

    # ============================================================
    # Header: Method
    # ============================================================

    c = table.add_cell(
        0,
        0,
        width=method_w,
        height=row_h * 2,
        text="Method",
        loc="center",
        facecolor=header_face,
        edgecolor=edge,
    )

    c.get_text().set_weight(
        "bold"
    )

    # ============================================================
    # Header: L1-L4 groups
    # ============================================================

    for j, level in enumerate(
        LEVELS
    ):
        col = (
            1
            + 2 * j
        )

        c = table.add_cell(
            0,
            col,
            width=metric_w * 2,
            height=row_h,
            text=f"L0 / {level}",
            loc="center",
            facecolor=header_face,
            edgecolor=edge,
        )

        c.get_text().set_weight(
            "bold"
        )

        c = table.add_cell(
            1,
            col,
            width=metric_w,
            height=row_h,
            text="Macro ↑",
            loc="center",
            facecolor=subheader_face,
            edgecolor=edge,
        )

        c.get_text().set_weight(
            "bold"
        )

        c = table.add_cell(
            1,
            col + 1,
            width=metric_w,
            height=row_h,
            text="Miss (%) ↓",
            loc="center",
            facecolor=subheader_face,
            edgecolor=edge,
        )

        c.get_text().set_weight(
            "bold"
        )

    # ============================================================
    # Body
    # ============================================================

    for body_i, row in enumerate(
        rows
    ):
        table_row = (
            body_i + 2
        )

        method = row[
            "Method"
        ]

        is_tv = (
            method
            == "TV-streamDSGN"
        )

        face = (
            tv_face
            if is_tv
            else "white"
        )

        c = table.add_cell(
            table_row,
            0,
            width=method_w,
            height=row_h,
            text=method,
            loc="left",
            facecolor=face,
            edgecolor=edge,
        )

        c.PAD = 0.10

        if is_tv:
            c.get_text().set_weight(
                "bold"
            )

        for j, level in enumerate(
            LEVELS
        ):
            col = (
                1
                + 2 * j
            )

            macro = float(
                row[
                    f"{level}_Macro"
                ]
            )

            miss = float(
                row[
                    f"{level}_MissPct"
                ]
            )

            cm = table.add_cell(
                table_row,
                col,
                width=metric_w,
                height=row_h,
                text=f"{macro:.2f}",
                loc="center",
                facecolor=face,
                edgecolor=edge,
            )

            cd = table.add_cell(
                table_row,
                col + 1,
                width=metric_w,
                height=row_h,
                text=f"{miss:.2f}",
                loc="center",
                facecolor=face,
                edgecolor=edge,
            )

            if (
                body_i
                in best[level]["macro"]
            ):
                cm.get_text().set_weight(
                    "bold"
                )

            if (
                body_i
                in best[level]["miss"]
            ):
                cd.get_text().set_weight(
                    "bold"
                )

    # ============================================================
    # Styling
    # ============================================================

    for cell in table.get_celld().values():

        cell.set_linewidth(
            0.65
        )

        cell.get_text().set_fontsize(
            8.2
        )

    ax.add_table(
        table
    )

    ax.text(
        0.5,
        0.93,
        (
            "Random50 streaming performance "
            f"at {args.hz} Hz"
        ),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.025,
        (
            "Random50 alternates L0 and the indicated "
            f"pressure level with "
            f"{100*args.pressure_fraction:.0f}% pressured frames. "
            "Macro: mean streaming 3D AP$_{R40}$ over "
            "Car, Pedestrian, and Cyclist (Moderate). "
            "Miss: deadline-miss rate over processed frames."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
        transform=ax.transAxes,
    )

    fig.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )


def main():
    args = parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    if not repo.is_dir():
        raise FileNotFoundError(
            repo
        )

    out_dir = (
        repo
        / args.output_dir
    ).resolve()

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 110)
    print(
        "RANDOM50 @ 35 Hz MAIN-RESULT TABLE"
    )
    print("=" * 110)

    rows = collect(
        args,
        repo,
    )

    tag = (
        f"{args.hz}Hz_"
        f"p{fraction_tag(args.pressure_fraction)}_"
        f"seed{args.trace_seed}"
    )

    csv_path = (
        out_dir
        / f"random50_macro_miss_{tag}.csv"
    )

    png_path = (
        out_dir
        / f"random50_macro_miss_{tag}.png"
    )

    write_csv(
        rows,
        csv_path,
    )

    render_table(
        rows,
        png_path,
        args,
    )

    print()
    print("=" * 110)
    print("[DONE]")
    print(f"CSV : {csv_path}")
    print(f"PNG : {png_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
