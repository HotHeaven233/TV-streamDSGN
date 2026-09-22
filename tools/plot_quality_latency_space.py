#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WIDTH_COLS = [
    "res2",
    "res3",
    "res4",
    "fpn",
    "stereo",
    "rpn",
]

QUALITY_COL = "reference_mean3d_moderate"
LATENCY_COL = "forward_total_ms_p99"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the quality-latency space of the 84 legal "
            "monotonic TV-streamDSGN schedules."
        )
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="all84_quality_forward_latency.csv",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for paper figure outputs.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return parser.parse_args()


def validate_input(df: pd.DataFrame):
    required = [
        "id",
        "schedule",
        *WIDTH_COLS,
        QUALITY_COL,
        LATENCY_COL,
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(missing)
        )

    if len(df) != 84:
        raise RuntimeError(
            f"Expected exactly 84 schedule rows, got {len(df)}. "
            "Please run the complete all-84 profiling first."
        )


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in [
        *WIDTH_COLS,
        QUALITY_COL,
        LATENCY_COL,
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    bad = df[
        df[
            [
                QUALITY_COL,
                LATENCY_COL,
            ]
        ].isna().any(axis=1)
    ]

    if not bad.empty:
        raise RuntimeError(
            "Found schedules with missing quality/latency values:\n"
            + bad[
                [
                    "id",
                    "schedule",
                ]
            ].to_string(index=False)
        )

    df["is_uniform"] = df[WIDTH_COLS].apply(
        lambda row: np.allclose(
            row.to_numpy(dtype=float),
            float(row.iloc[0]),
        ),
        axis=1,
    )

    return df


def compute_pareto(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    A schedule is Pareto optimal if no other schedule has:

      * no larger p99 forward latency, and
      * no lower reference Macro AP,

    with at least one strict improvement.
    """

    rows = []

    for idx, a in df.iterrows():

        quality_a = float(
            a[QUALITY_COL]
        )

        latency_a = float(
            a[LATENCY_COL]
        )

        dominated = False

        for jdx, b in df.iterrows():

            if idx == jdx:
                continue

            quality_b = float(
                b[QUALITY_COL]
            )

            latency_b = float(
                b[LATENCY_COL]
            )

            if (
                quality_b >= quality_a
                and latency_b <= latency_a
                and (
                    quality_b > quality_a
                    or latency_b < latency_a
                )
            ):
                dominated = True
                break

        if not dominated:
            rows.append(a)

    pareto = pd.DataFrame(rows)

    pareto = pareto.sort_values(
        by=[
            LATENCY_COL,
            QUALITY_COL,
        ],
        ascending=[
            True,
            True,
        ],
    )

    return pareto


def uniform_label(
    width: float,
) -> str:

    if np.isclose(width, 1.0):
        return r"$1.00\times6$"

    if np.isclose(width, 0.75):
        return r"$0.75\times6$"

    if np.isclose(width, 0.50):
        return r"$0.50\times6$"

    if np.isclose(width, 0.25):
        return r"$0.25\times6$"

    return rf"${width:.2f}\times6$"


def annotate_uniform(
    ax,
    row,
):
    """
    Place each uniform-width annotation independently.

    Data-coordinate positions are used rather than a common pixel offset
    because the four reference points occupy very different regions of
    the quality-latency space.
    """

    x = float(
        row[LATENCY_COL]
    )

    y = float(
        row[QUALITY_COL]
    )

    width = float(
        row["res2"]
    )

    # ---------------------------------------------------------
    # 1.00 x 6
    # ---------------------------------------------------------
    if np.isclose(
        width,
        1.0,
    ):
        xytext = (
            x - 0.45,
            y + 0.65,
        )
        ha = "right"
        va = "bottom"

    # ---------------------------------------------------------
    # 0.75 x 6
    # ---------------------------------------------------------
    elif np.isclose(
        width,
        0.75,
    ):
        xytext = (
            x + 0.48,
            y + 2.2,
        )
        ha = "left"
        va = "bottom"

    # ---------------------------------------------------------
    # 0.50 x 6
    # ---------------------------------------------------------
    elif np.isclose(
        width,
        0.50,
    ):
        xytext = (
            x + 0.55,
            y - 4.0,
        )
        ha = "left"
        va = "top"

    # ---------------------------------------------------------
    # 0.25 x 6
    # ---------------------------------------------------------
    elif np.isclose(
        width,
        0.25,
    ):
        xytext = (
            x + 0.55,
            y - 2.6,
        )
        ha = "left"
        va = "top"

    else:
        return

    ax.annotate(
        uniform_label(width),
        xy=(x, y),
        xytext=xytext,
        textcoords="data",
        fontsize=7.4,
        ha=ha,
        va=va,
        annotation_clip=False,
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="white",
            edgecolor="none",
            alpha=0.92,
        ),
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.55,
            color="0.40",
            shrinkA=2,
            shrinkB=4,
        ),
        zorder=8,
    )


def main():

    args = parse_args()

    input_csv = Path(
        args.input_csv
    )

    output_dir = Path(
        args.output_dir
    )

    if not input_csv.is_file():
        raise FileNotFoundError(
            input_csv
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = pd.read_csv(
        input_csv
    )

    validate_input(
        df
    )

    df = prepare_dataframe(
        df
    )

    # ---------------------------------------------------------
    # Pareto frontier
    # ---------------------------------------------------------

    pareto = compute_pareto(
        df
    )

    uniform = df[
        df["is_uniform"]
    ].copy()

    # ---------------------------------------------------------
    # Save Pareto schedules
    # ---------------------------------------------------------

    pareto_out = (
        output_dir
        / "quality_latency_pareto.csv"
    )

    pareto[
        [
            "id",
            "schedule",
            *WIDTH_COLS,
            QUALITY_COL,
            LATENCY_COL,
        ]
    ].to_csv(
        pareto_out,
        index=False,
    )

    # ---------------------------------------------------------
    # Figure style
    # ---------------------------------------------------------

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "Helvetica",
                "DejaVu Sans",
            ],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 6.7,
        }
    )

    fig, ax = plt.subplots(
        figsize=(
            3.55,
            2.75,
        )
    )

    # ---------------------------------------------------------
    # All 84 legal schedules
    # ---------------------------------------------------------

    ax.scatter(
        df[LATENCY_COL],
        df[QUALITY_COL],
        s=18,
        c="#BDBDBD",
        alpha=0.72,
        edgecolors="none",
        label="Legal schedules",
        zorder=1,
    )

    # ---------------------------------------------------------
    # Pareto frontier
    # ---------------------------------------------------------

    ax.plot(
        pareto[LATENCY_COL],
        pareto[QUALITY_COL],
        color="#C62828",
        linewidth=1.15,
        marker="o",
        markersize=3.7,
        markerfacecolor="#C62828",
        markeredgewidth=0,
        label="Pareto frontier",
        zorder=3,
    )

    # ---------------------------------------------------------
    # Uniform-width schedules
    # ---------------------------------------------------------

    ax.scatter(
        uniform[LATENCY_COL],
        uniform[QUALITY_COL],
        s=36,
        marker="s",
        facecolors="white",
        edgecolors="#2457A6",
        linewidths=1.15,
        label="Uniform width",
        zorder=5,
    )

    # ---------------------------------------------------------
    # Reference labels
    # ---------------------------------------------------------

    for _, row in uniform.sort_values(
        "res2"
    ).iterrows():

        annotate_uniform(
            ax,
            row,
        )

    # ---------------------------------------------------------
    # Axes
    # ---------------------------------------------------------

    ax.set_xlabel(
        "p99 forward latency (ms)"
    )

    ax.set_ylabel(
        "Macro AP"
    )

    ax.grid(
        True,
        linestyle="--",
        linewidth=0.45,
        alpha=0.32,
        zorder=0,
    )

    # ---------------------------------------------------------
    # Legend order
    # ---------------------------------------------------------

    handles, labels = (
        ax.get_legend_handles_labels()
    )

    desired_order = [
        "Legal schedules",
        "Uniform width",
        "Pareto frontier",
    ]

    ordered_handles = []
    ordered_labels = []

    for name in desired_order:

        if name not in labels:
            continue

        idx = labels.index(
            name
        )

        ordered_handles.append(
            handles[idx]
        )

        ordered_labels.append(
            labels[idx]
        )

    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        borderpad=0.35,
        handletextpad=0.45,
    )

    # ---------------------------------------------------------
    # Axis padding
    # ---------------------------------------------------------

    x = df[
        LATENCY_COL
    ].to_numpy(
        dtype=float
    )

    y = df[
        QUALITY_COL
    ].to_numpy(
        dtype=float
    )

    x_range = float(
        x.max() - x.min()
    )

    y_range = float(
        y.max() - y.min()
    )

    x_pad_left = max(
        0.55,
        0.04 * x_range,
    )

    x_pad_right = max(
        0.65,
        0.05 * x_range,
    )

    y_pad_bottom = max(
        1.2,
        0.04 * y_range,
    )

    y_pad_top = max(
        2.8,
        0.06 * y_range,
    )

    ax.set_xlim(
        x.min() - x_pad_left,
        x.max() + x_pad_right,
    )

    ax.set_ylim(
        y.min() - y_pad_bottom,
        y.max() + y_pad_top,
    )

    fig.tight_layout(
        pad=0.45
    )

    # ---------------------------------------------------------
    # Save figure
    # ---------------------------------------------------------

    png = (
        output_dir
        / "quality_latency_space.png"
    )

    fig.savefig(
        png,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    nonuniform_pareto = int(
        (
            ~pareto[
                "is_uniform"
            ]
        ).sum()
    )

    print(
        "============================================================"
    )
    print(
        "QUALITY--LATENCY SPACE"
    )
    print(
        "============================================================"
    )
    print(
        "input              :",
        input_csv,
    )
    print(
        "legal schedules    :",
        len(df),
    )
    print(
        "uniform schedules  :",
        len(uniform),
    )
    print(
        "Pareto schedules   :",
        len(pareto),
    )
    print(
        "non-uniform Pareto :",
        nonuniform_pareto,
    )
    print(
        "quality column     :",
        QUALITY_COL,
    )
    print(
        "latency column     :",
        LATENCY_COL,
    )
    print(
        "PNG                :",
        png,
    )
    print(
        "Pareto CSV         :",
        pareto_out,
    )
    print(
        "============================================================"
    )

    print(
        "\nPareto frontier:"
    )

    print(
        pareto[
            [
                "id",
                "schedule",
                QUALITY_COL,
                LATENCY_COL,
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
