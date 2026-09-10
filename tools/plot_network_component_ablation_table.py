#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table


TARGET_SCHEDULES = {
    "full": "1.0,1.0,1.0,1.0,1.0,1.0",
    "w075": "0.75,0.75,0.75,0.75,0.75,0.75",
    "w050": "0.5,0.5,0.5,0.5,0.5,0.5",
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot network-level ablation and elastic execution table "
            "for TV-streamDSGN."
        )
    )

    p.add_argument(
        "--repo_root",
        default="/data/jhb/workspace/streamDSGN",
    )

    p.add_argument(
        "--original_root",
        default=(
            "outputs/stream_buffer_timestamp/original_10hz"
        ),
    )

    p.add_argument(
        "--all84_csv",
        default=(
            "outputs/elastic_bev/"
            "elastic_bev_v4_bn_from_k3/"
            "all84_sap/e20_n100_10Hz/"
            "all84_sap_summary.csv"
        ),
    )

    p.add_argument(
        "--output_dir",
        default=(
            "outputs/paper_figures/"
            "network_component_ablation_table"
        ),
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return p.parse_args()


def locate_unique_or_deepest(
    root: Path,
    filename: str,
):
    hits = list(
        root.rglob(filename)
    )

    if not hits:
        raise FileNotFoundError(
            f"Cannot find {filename} under:\n  {root}"
        )

    # Original evaluation result is normally nested as:
    # original_10hz/<cfg-tag>/10Hz/<filename>
    hits.sort(
        key=lambda p: len(p.parts),
        reverse=True,
    )

    if len(hits) > 1:
        print(
            f"[WARN] Multiple {filename} files found; "
            f"using deepest path:\n  {hits[0]}"
        )

    return hits[0]


def parse_original_macro(
    paper_path: Path,
):
    """
    Parse:
      Car        : 3D, IoU=0.7, Moderate
      Pedestrian : 3D, IoU=0.5, Moderate
      Cyclist    : 3D, IoU=0.5, Moderate

    Then Macro = mean of the three.
    """

    text = paper_path.read_text(
        errors="replace"
    ).splitlines()

    current_class = None
    current_iou = None

    values = {}

    class_re = re.compile(
        r"^(Car|Pedestrian|Cyclist)\s*$"
    )

    iou_re = re.compile(
        r"^\s*IoU=(0\.[57])\s*$"
    )

    sap3d_re = re.compile(
        r"^\s*sAP3D\s*:\s*"
        r"([-+0-9.eE]+)\s*,\s*"
        r"([-+0-9.eE]+)\s*,\s*"
        r"([-+0-9.eE]+)\s*$"
    )

    for line in text:
        m = class_re.match(line)

        if m:
            current_class = m.group(1)
            current_iou = None
            continue

        m = iou_re.match(line)

        if m:
            current_iou = m.group(1)
            continue

        m = sap3d_re.match(line)

        if (
            m
            and current_class is not None
            and current_iou is not None
        ):
            # Easy / Moderate / Hard
            moderate = float(
                m.group(2)
            )

            values[
                (
                    current_class,
                    current_iou,
                )
            ] = moderate

    needed = {
        "Car": ("Car", "0.7"),
        "Pedestrian": (
            "Pedestrian",
            "0.5",
        ),
        "Cyclist": (
            "Cyclist",
            "0.5",
        ),
    }

    missing = [
        name
        for name, key in needed.items()
        if key not in values
    ]

    if missing:
        raise RuntimeError(
            "Failed to parse required Original metrics: "
            + ", ".join(missing)
        )

    car = values[
        needed["Car"]
    ]

    ped = values[
        needed["Pedestrian"]
    ]

    cyc = values[
        needed["Cyclist"]
    ]

    macro = (
        car
        + ped
        + cyc
    ) / 3.0

    return {
        "Car": car,
        "Pedestrian": ped,
        "Cyclist": cyc,
        "Macro": macro,
    }


def read_original(
    original_root: Path,
):
    paper = locate_unique_or_deepest(
        original_root,
        "paper_sap.txt",
    )

    summary_path = locate_unique_or_deepest(
        original_root,
        "summary.json",
    )

    metrics = parse_original_macro(
        paper
    )

    summary = json.loads(
        summary_path.read_text()
    )

    # test_stream_buffer_timestamp.py uses p99_service_ms.
    if "p99_service_ms" not in summary:
        raise KeyError(
            f"{summary_path}: missing p99_service_ms"
        )

    p99_ms = float(
        summary[
            "p99_service_ms"
        ]
    )

    print(
        "[ORIGINAL]"
    )
    print(
        f"  paper   : {paper}"
    )
    print(
        f"  summary : {summary_path}"
    )
    print(
        f"  Car     : {metrics['Car']:.4f}"
    )
    print(
        f"  Ped     : {metrics['Pedestrian']:.4f}"
    )
    print(
        f"  Cyc     : {metrics['Cyclist']:.4f}"
    )
    print(
        f"  Macro   : {metrics['Macro']:.4f}"
    )
    print(
        f"  p99     : {p99_ms:.4f} ms"
    )

    return {
        "macro": metrics["Macro"],
        "p99_ms": p99_ms,
    }


def get_schedule_row(
    df: pd.DataFrame,
    schedule: str,
):
    x = df[
        df["schedule"].astype(str)
        == schedule
    ]

    if len(x) != 1:
        raise RuntimeError(
            f"Expected exactly one row for schedule "
            f"{schedule}, found {len(x)}"
        )

    row = x.iloc[0]

    if str(
        row["status"]
    ) != "OK":
        raise RuntimeError(
            f"Schedule {schedule} status is "
            f"{row['status']}"
        )

    return row


def read_elastic(
    all84_csv: Path,
):
    if not all84_csv.is_file():
        raise FileNotFoundError(
            all84_csv
        )

    df = pd.read_csv(
        all84_csv
    )

    required = {
        "schedule",
        "status",
        "reference_mean3d_moderate",
        "p99_service_ms",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise RuntimeError(
            "all84 CSV missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    out = {}

    for name, schedule in (
        TARGET_SCHEDULES.items()
    ):
        row = get_schedule_row(
            df,
            schedule,
        )

        out[name] = {
            "schedule": schedule,
            "macro": float(
                row[
                    "reference_mean3d_moderate"
                ]
            ),
            "p99_ms": float(
                row[
                    "p99_service_ms"
                ]
            ),
        }

        print(
            f"[ALL84/{name}] "
            f"schedule={schedule} | "
            f"Macro={out[name]['macro']:.4f} | "
            f"p99={out[name]['p99_ms']:.4f} ms"
        )

    return out


def build_rows(
    original,
    elastic,
):
    return [
        {
            "Configuration":
                "Original StreamDSGN",

            "Elastic schedule":
                "–",

            "Macro AP":
                original["macro"],

            "p99 latency (ms)":
                original["p99_ms"],
        },

        {
            "Configuration":
                "+ K3 History Residual Adapter",

            "Elastic schedule":
                "Full",

            "Macro AP":
                elastic[
                    "full"
                ][
                    "macro"
                ],

            "p99 latency (ms)":
                elastic[
                    "full"
                ][
                    "p99_ms"
                ],
        },

        {
            "Configuration":
                "+ Elastic execution",

            "Elastic schedule":
                "0.75 × 6",

            "Macro AP":
                elastic[
                    "w075"
                ][
                    "macro"
                ],

            "p99 latency (ms)":
                elastic[
                    "w075"
                ][
                    "p99_ms"
                ],
        },

        {
            "Configuration":
                "+ Elastic execution",

            "Elastic schedule":
                "0.50 × 6",

            "Macro AP":
                elastic[
                    "w050"
                ][
                    "macro"
                ],

            "p99 latency (ms)":
                elastic[
                    "w050"
                ][
                    "p99_ms"
                ],
        },
    ]


def write_csv(
    rows,
    path: Path,
):
    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Configuration",
                "Elastic schedule",
                "Macro AP",
                "p99 latency (ms)",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


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


def render_table(
    rows,
    output_path: Path,
    dpi: int,
):
    configure_style()

    fig, ax = plt.subplots(
        figsize=(8.4, 2.75)
    )

    ax.set_axis_off()

    table = Table(
        ax,
        bbox=[
            0.015,
            0.13,
            0.97,
            0.69,
        ],
    )

    headers = [
        "Configuration",
        "Elastic schedule",
        "Macro AP ↑",
        "p99 latency (ms) ↓",
    ]

    col_widths = [
        0.42,
        0.21,
        0.17,
        0.20,
    ]

    total_rows = (
        1
        + len(rows)
    )

    row_h = (
        1.0
        / total_rows
    )

    edge = "#555555"
    header_face = "#eeeeee"

    # ------------------------------------------------------------
    # Header
    # ------------------------------------------------------------

    for j, (
        header,
        width,
    ) in enumerate(
        zip(
            headers,
            col_widths,
        )
    ):
        c = table.add_cell(
            0,
            j,
            width=width,
            height=row_h,
            text=header,
            loc="center",
            facecolor=header_face,
            edgecolor=edge,
        )

        c.get_text().set_weight(
            "bold"
        )

    # ------------------------------------------------------------
    # Body
    # ------------------------------------------------------------

    macro_values = [
        float(
            r["Macro AP"]
        )
        for r in rows
    ]

    latency_values = [
        float(
            r["p99 latency (ms)"]
        )
        for r in rows
    ]

    best_macro = max(
        macro_values
    )

    best_latency = min(
        latency_values
    )

    for i, row in enumerate(
        rows,
        start=1,
    ):
        values = [
            row["Configuration"],
            row["Elastic schedule"],
            f"{row['Macro AP']:.2f}",
            f"{row['p99 latency (ms)']:.2f}",
        ]

        for j, (
            value,
            width,
        ) in enumerate(
            zip(
                values,
                col_widths,
            )
        ):
            loc = (
                "left"
                if j == 0
                else "center"
            )

            c = table.add_cell(
                i,
                j,
                width=width,
                height=row_h,
                text=value,
                loc=loc,
                facecolor="white",
                edgecolor=edge,
            )

            if j == 0:
                c.PAD = 0.07

            # Highlight K3 method name.
            if (
                i == 2
                and j == 0
            ):
                c.get_text().set_weight(
                    "bold"
                )

            # Best Macro.
            if (
                j == 2
                and abs(
                    float(
                        row["Macro AP"]
                    )
                    - best_macro
                ) < 1e-8
            ):
                c.get_text().set_weight(
                    "bold"
                )

            # Lowest p99.
            if (
                j == 3
                and abs(
                    float(
                        row[
                            "p99 latency (ms)"
                        ]
                    )
                    - best_latency
                ) < 1e-8
            ):
                c.get_text().set_weight(
                    "bold"
                )

    for cell in (
        table
        .get_celld()
        .values()
    ):
        cell.set_linewidth(
            0.65
        )

        cell.get_text().set_fontsize(
            8.3
        )

    ax.add_table(
        table
    )

    ax.text(
        0.5,
        0.93,
        (
            "Effect of temporal enhancement "
            "and elastic execution"
        ),
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        0.5,
        0.035,
        (
            "Macro AP: mean Moderate 3D AP$_{R40}$ over "
            "Car, Pedestrian, and Cyclist. "
            "Latency is measured at 10 Hz."
        ),
        ha="center",
        va="bottom",
        fontsize=7.2,
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

    original_root = (
        repo
        / args.original_root
    ).resolve()

    all84_csv = (
        repo
        / args.all84_csv
    ).resolve()

    output_dir = (
        repo
        / args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)
    print(
        "NETWORK COMPONENT / ELASTIC EXECUTION TABLE"
    )
    print("=" * 100)

    original = read_original(
        original_root
    )

    elastic = read_elastic(
        all84_csv
    )

    rows = build_rows(
        original,
        elastic,
    )

    csv_path = (
        output_dir
        / "network_component_ablation_table.csv"
    )

    png_path = (
        output_dir
        / "network_component_ablation_table.png"
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
    print("-" * 100)

    for row in rows:
        print(
            f"{row['Configuration']:<34s} | "
            f"{row['Elastic schedule']:<10s} | "
            f"Macro={row['Macro AP']:.4f} | "
            f"p99={row['p99 latency (ms)']:.4f} ms"
        )

    print("-" * 100)
    print(f"CSV : {csv_path}")
    print(f"PNG : {png_path}")
    print("=" * 100)


if __name__ == "__main__":
    main()
