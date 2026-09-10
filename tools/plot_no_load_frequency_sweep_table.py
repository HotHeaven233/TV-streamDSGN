#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table


METHODS = [
    {
        "name": "StreamDSGN",
        "kind": "original",
    },
    {
        "name": "Streamer-style",
        "kind": "streamer",
    },
    {
        "name": "MTD",
        "kind": "mtd",
    },
    {
        "name": "Transtreaming-style",
        "kind": "transtreaming",
    },
    {
        "name": "TV-streamDSGN",
        "kind": "tv",
    },
]

FREQUENCIES = [35, 40, 45, 50]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Read formal no-load frequency-sweep summaries and "
            "render Macro AP + deadline-miss table."
        )
    )

    p.add_argument(
        "--repo_root",
        default="/data/jhb/workspace/streamDSGN",
    )

    p.add_argument(
        "--tv_exp_name",
        default="elastic_bev_v4_bn_from_k3",
    )

    p.add_argument(
        "--output_dir",
        default=(
            "outputs/paper_figures/"
            "no_load_frequency_sweep_table"
        ),
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return p.parse_args()


def summary_path(
    repo: Path,
    kind: str,
    hz: int,
    tv_exp_name: str,
) -> Path:

    if kind == "original":
        return (
            repo
            / "outputs"
            / "original_streamdsgn"
            / "formal_streaming_no_load"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    if kind == "streamer":
        return (
            repo
            / "outputs"
            / "streamer_style_streamdsgn"
            / "formal_streaming_no_load"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    if kind == "mtd":
        return (
            repo
            / "outputs"
            / "mtd_three_head"
            / "formal_streaming_no_load"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    if kind == "transtreaming":
        return (
            repo
            / "outputs"
            / "transtreaming_v2_best"
            / "formal_streaming_no_load"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    if kind == "tv":
        return (
            repo
            / "outputs"
            / "elastic_bev"
            / tv_exp_name
            / "formal_streaming"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    raise ValueError(
        f"unknown method kind: {kind}"
    )


def read_summary(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing summary:\n  {path}"
        )

    with path.open() as f:
        data = json.load(f)

    if "stream_sap_3d_moderate_R40" not in data:
        raise KeyError(
            f"{path}: missing "
            "'stream_sap_3d_moderate_R40'"
        )

    sap = data[
        "stream_sap_3d_moderate_R40"
    ]

    if "Macro" not in sap:
        raise KeyError(
            f"{path}: missing Macro"
        )

    if "deadline_miss_rate" not in data:
        raise KeyError(
            f"{path}: missing deadline_miss_rate"
        )

    macro = float(
        sap["Macro"]
    )

    # summary.json stores misses / processed.
    # Convert fraction to percentage for the paper table.
    miss_pct = (
        100.0
        * float(
            data["deadline_miss_rate"]
        )
    )

    return macro, miss_pct


def collect_results(
    repo: Path,
    tv_exp_name: str,
):
    rows = []

    for method in METHODS:
        row = {
            "Method": method["name"],
        }

        for hz in FREQUENCIES:
            path = summary_path(
                repo=repo,
                kind=method["kind"],
                hz=hz,
                tv_exp_name=tv_exp_name,
            )

            macro, miss = read_summary(
                path
            )

            row[f"{hz}Hz_Macro"] = macro
            row[f"{hz}Hz_MissPct"] = miss

            print(
                f"[READ] "
                f"{method['name']:<20s} "
                f"{hz:>2d} Hz | "
                f"Macro={macro:8.4f} | "
                f"Miss={miss:8.4f}% | "
                f"{path}"
            )

        rows.append(row)

    return rows


def write_csv(
    rows,
    output_path: Path,
):
    fields = ["Method"]

    for hz in FREQUENCIES:
        fields.extend([
            f"{hz}Hz_Macro",
            f"{hz}Hz_MissPct",
        ])

    with output_path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def best_indices(
    rows,
):
    """
    For every Hz:
      Macro: larger is better
      Miss:  smaller is better

    Return all ties within numerical tolerance.
    """
    result = {}

    for hz in FREQUENCIES:
        macros = [
            float(
                r[f"{hz}Hz_Macro"]
            )
            for r in rows
        ]

        misses = [
            float(
                r[f"{hz}Hz_MissPct"]
            )
            for r in rows
        ]

        best_macro = max(macros)
        best_miss = min(misses)

        macro_idx = {
            i
            for i, value
            in enumerate(macros)
            if abs(
                value - best_macro
            ) < 1e-8
        }

        miss_idx = {
            i
            for i, value
            in enumerate(misses)
            if abs(
                value - best_miss
            ) < 1e-8
        }

        result[hz] = {
            "macro": macro_idx,
            "miss": miss_idx,
        }

    return result


def render_table(
    rows,
    output_path: Path,
    dpi: int,
):
    configure_style()

    best = best_indices(
        rows
    )

    # Wide enough for a double-column paper table.
    fig, ax = plt.subplots(
        figsize=(13.2, 3.45)
    )

    ax.set_axis_off()

    # ------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------

    left = 0.015
    bottom = 0.08
    width = 0.97
    height = 0.76

    table = Table(
        ax,
        bbox=[
            left,
            bottom,
            width,
            height,
        ],
    )

    n_method_rows = len(rows)

    # 2 header rows + 5 method rows
    total_rows = 2 + n_method_rows

    row_h = 1.0 / total_rows

    method_w = 0.19
    metric_w = (
        1.0 - method_w
    ) / 8.0

    header_face = "#eeeeee"
    subheader_face = "#f7f7f7"
    tv_face = "#fafafa"
    white = "white"
    edge = "#555555"

    # ------------------------------------------------------------
    # Header row 0
    # ------------------------------------------------------------

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

    for group_i, hz in enumerate(
        FREQUENCIES
    ):
        col = (
            1
            + group_i * 2
        )

        c = table.add_cell(
            0,
            col,
            width=metric_w * 2,
            height=row_h,
            text=f"{hz} Hz",
            loc="center",
            facecolor=header_face,
            edgecolor=edge,
        )

        c.get_text().set_weight(
            "bold"
        )

    # ------------------------------------------------------------
    # Header row 1
    # ------------------------------------------------------------

    for group_i, hz in enumerate(
        FREQUENCIES
    ):
        col = (
            1
            + group_i * 2
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

    # ------------------------------------------------------------
    # Body
    # ------------------------------------------------------------

    for row_i, row in enumerate(
        rows,
        start=2,
    ):
        method_idx = (
            row_i - 2
        )

        method_name = row[
            "Method"
        ]

        is_tv = (
            method_name
            == "TV-streamDSGN"
        )

        face = (
            tv_face
            if is_tv
            else white
        )

        c = table.add_cell(
            row_i,
            0,
            width=method_w,
            height=row_h,
            text=method_name,
            loc="left",
            facecolor=face,
            edgecolor=edge,
        )

        c.PAD = 0.10

        if is_tv:
            c.get_text().set_weight(
                "bold"
            )

        for group_i, hz in enumerate(
            FREQUENCIES
        ):
            col = (
                1
                + group_i * 2
            )

            macro = float(
                row[
                    f"{hz}Hz_Macro"
                ]
            )

            miss = float(
                row[
                    f"{hz}Hz_MissPct"
                ]
            )

            c_macro = table.add_cell(
                row_i,
                col,
                width=metric_w,
                height=row_h,
                text=f"{macro:.2f}",
                loc="center",
                facecolor=face,
                edgecolor=edge,
            )

            c_miss = table.add_cell(
                row_i,
                col + 1,
                width=metric_w,
                height=row_h,
                text=f"{miss:.2f}",
                loc="center",
                facecolor=face,
                edgecolor=edge,
            )

            if (
                method_idx
                in best[hz]["macro"]
            ):
                c_macro.get_text().set_weight(
                    "bold"
                )

            if (
                method_idx
                in best[hz]["miss"]
            ):
                c_miss.get_text().set_weight(
                    "bold"
                )

    # ------------------------------------------------------------
    # General formatting
    # ------------------------------------------------------------

    for (
        _,
        cell,
    ) in table.get_celld().items():

        cell.set_linewidth(
            0.65
        )

        cell.get_text().set_fontsize(
            8.2
        )

    ax.add_table(
        table
    )

    # ------------------------------------------------------------
    # Title and note
    # ------------------------------------------------------------

    ax.text(
        0.5,
        0.93,
        "No-load streaming performance under increasing input rates",
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
            "Macro: mean streaming 3D AP$_{R40}$ over "
            "Car, Pedestrian, and Cyclist (Moderate). "
            "Miss: deadline-miss rate over processed frames."
        ),
        ha="center",
        va="bottom",
        fontsize=7.4,
        transform=ax.transAxes,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
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
            f"repo root does not exist: "
            f"{repo}"
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
        "NO-LOAD FREQUENCY SWEEP TABLE"
    )
    print("=" * 110)

    rows = collect_results(
        repo=repo,
        tv_exp_name=args.tv_exp_name,
    )

    csv_path = (
        out_dir
        / "no_load_frequency_sweep_macro_miss.csv"
    )

    png_path = (
        out_dir
        / "no_load_frequency_sweep_macro_miss.png"
    )

    write_csv(
        rows,
        csv_path,
    )

    render_table(
        rows,
        png_path,
        args.dpi,
    )

    print()
    print("=" * 110)
    print("[DONE]")
    print(
        f"CSV : {csv_path}"
    )
    print(
        f"PNG : {png_path}"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
