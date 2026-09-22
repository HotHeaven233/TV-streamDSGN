#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


STAGE_NAMES = [
    "Res2",
    "Res3",
    "Res4",
    "FPN",
    "Stereo",
    "3D Voxel",
]

WIDTH_VALUES = [
    0.25,
    0.50,
    0.75,
    1.00,
]

LEVEL_TO_INT = {
    "L0": 0,
    "L1": 1,
    "L2": 2,
    "L3": 3,
    "L4": 4,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot representative TV-streamDSGN runtime behavior "
            "under L0/L1--L4 random GPU contention."
        )
    )

    p.add_argument(
        "--input-root",
        required=True,
        help=(
            "Root directory containing L0_L1_pX, ..., "
            "L0_L4_pX experiment directories."
        ),
    )

    p.add_argument(
        "--output-dir",
        required=True,
    )

    p.add_argument(
        "--hz",
        type=float,
        required=True,
    )

    p.add_argument(
        "--trace-seed",
        type=int,
        required=True,
    )

    p.add_argument(
        "--pressure-fraction",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--window",
        type=int,
        default=50,
    )

    p.add_argument(
        "--force-reselect",
        action="store_true",
        help=(
            "Ignore existing selected_window_L*.csv files "
            "and select new representative windows."
        ),
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return p.parse_args()


def require_columns(
    df: pd.DataFrame,
    columns,
    path: Path,
):
    missing = [
        c for c in columns
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            f"{path} is missing columns: {missing}"
        )


def normalize_dataframe(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    numeric_cols = [
        "global_index",
        "local_pos",
        "arrival_ms",
        "absolute_deadline_ms",
        "forward_start_ms",
        "forward_finish_ms",
        "initial_budget_ms",
        "queue_wait_ms",
        "forward_ms",
        "arrival_to_finish_ms",
        "deadline_slack_ms",
        "deadline_miss",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(
                out[col],
                errors="coerce",
            )

    return out


def parse_schedule(
    text,
):
    if pd.isna(text):
        return None

    s = str(text).strip()

    if not s:
        return None

    values = [
        float(x.strip())
        for x in s.split(",")
        if x.strip()
    ]

    if len(values) != 6:
        raise ValueError(
            f"Expected six widths, got {text!r}"
        )

    return values


def pressure_fraction_in_window(
    df: pd.DataFrame,
    pressure_level: str,
) -> float:

    if len(df) == 0:
        return 0.0

    return float(
        (
            df["true_level"].astype(str)
            == pressure_level
        ).mean()
    )


def contention_switch_rate(
    df: pd.DataFrame,
) -> float:

    levels = df[
        "true_level"
    ].astype(str).tolist()

    if len(levels) <= 1:
        return 0.0

    switches = sum(
        levels[i] != levels[i - 1]
        for i in range(1, len(levels))
    )

    return switches / (len(levels) - 1)


def select_representative_window(
    df: pd.DataFrame,
    pressure_level: str,
    window: int,
    pressure_fraction: float,
) -> pd.DataFrame:
    """
    Select a scene-contiguous representative window.

    Selection does NOT explicitly maximize deadline misses.
    It primarily matches the requested contention fraction and
    prefers windows containing visible contention-state changes.
    """

    if window <= 0:
        raise ValueError(
            "window must be > 0"
        )

    candidates = []

    for scene, group in df.groupby(
        "scene",
        sort=False,
    ):
        g = group.sort_values(
            "local_pos"
        ).reset_index(drop=True)

        if len(g) < window:
            continue

        for start in range(
            0,
            len(g) - window + 1,
        ):
            w = g.iloc[
                start:start + window
            ].copy()

            # Require contiguous sensor-frame positions.
            positions = w[
                "local_pos"
            ].to_numpy(dtype=float)

            if not np.allclose(
                np.diff(positions),
                1.0,
            ):
                continue

            frac = pressure_fraction_in_window(
                w,
                pressure_level,
            )

            switch_rate = contention_switch_rate(
                w
            )

            # Primary criterion:
            # match requested pressure fraction.
            frac_error = abs(
                frac - pressure_fraction
            )

            # Secondary criterion:
            # prefer windows with runtime state variation.
            score = (
                frac_error
                - 0.08 * switch_rate
            )

            candidates.append(
                (
                    score,
                    str(scene),
                    int(start),
                    w,
                    frac,
                    switch_rate,
                )
            )

    if not candidates:
        raise RuntimeError(
            f"No scene contains a contiguous "
            f"{window}-frame window."
        )

    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
            x[2],
        )
    )

    return candidates[0][3].copy()


def load_or_select_window(
    timeline_path: Path,
    selected_path: Path,
    pressure_level: str,
    window: int,
    pressure_fraction: float,
    force_reselect: bool,
) -> pd.DataFrame:

    if (
        selected_path.is_file()
        and not force_reselect
    ):
        print(
            f"[REUSE] {selected_path}"
        )

        df = pd.read_csv(
            selected_path
        )

        return normalize_dataframe(
            df
        )

    if not timeline_path.is_file():
        raise FileNotFoundError(
            timeline_path
        )

    print(
        f"[SELECT] {timeline_path}"
    )

    df = pd.read_csv(
        timeline_path
    )

    require_columns(
        df,
        [
            "global_index",
            "scene",
            "local_pos",
            "status",
            "true_level",
            "schedule",
            "arrival_to_finish_ms",
            "deadline_miss",
        ],
        timeline_path,
    )

    df = normalize_dataframe(
        df
    )

    selected = select_representative_window(
        df=df,
        pressure_level=pressure_level,
        window=window,
        pressure_fraction=pressure_fraction,
    )

    selected.to_csv(
        selected_path,
        index=False,
    )

    print(
        f"[SAVED] {selected_path}"
    )

    return selected


def build_width_matrix(
    df: pd.DataFrame,
):
    """
    Return:
      width_matrix : [6, N], values are widths for processed frames
      dropped_mask : [N], True for dropped frames
    """

    n = len(df)

    matrix = np.full(
        (6, n),
        np.nan,
        dtype=float,
    )

    dropped = np.zeros(
        n,
        dtype=bool,
    )

    for col, (_, row) in enumerate(
        df.iterrows()
    ):
        status = str(
            row.get(
                "status",
                "",
            )
        ).strip().lower()

        if status != "processed":
            dropped[col] = True
            continue

        schedule = parse_schedule(
            row.get(
                "schedule",
                "",
            )
        )

        if schedule is None:
            continue

        matrix[:, col] = schedule

    return matrix, dropped


def width_to_index_matrix(
    matrix,
):
    out = np.full_like(
        matrix,
        np.nan,
        dtype=float,
    )

    for idx, width in enumerate(
        WIDTH_VALUES
    ):
        out[
            np.isclose(
                matrix,
                width,
                equal_nan=False,
            )
        ] = idx

    return out


def add_dropped_spans(
    ax,
    dropped_mask,
):
    for i, dropped in enumerate(
        dropped_mask
    ):
        if dropped:
            ax.axvspan(
                i - 0.5,
                i + 0.5,
                facecolor="#D0D0D0",
                edgecolor="none",
                alpha=1.0,
                zorder=0,
            )


def plot_one_panel(
    fig,
    sub_spec,
    df: pd.DataFrame,
    pressure_level: str,
    panel_label: str,
    period_ms: float,
    show_left_labels: bool = True,
):
    """
    One panel contains:
      1) runtime contention
      2) controller-selected stage widths
      3) response latency and deadline
    """

    inner = sub_spec.subgridspec(
        3,
        1,
        height_ratios=[
            0.72,
            1.25,
            1.05,
        ],
        hspace=0.28,
    )

    ax_cont = fig.add_subplot(
        inner[0]
    )

    ax_width = fig.add_subplot(
        inner[1],
        sharex=ax_cont,
    )

    ax_lat = fig.add_subplot(
        inner[2],
        sharex=ax_cont,
    )

    n = len(df)
    x = np.arange(n)

    # ---------------------------------------------------------
    # Runtime contention
    # ---------------------------------------------------------

    contention = np.asarray(
        [
            LEVEL_TO_INT.get(
                str(v),
                np.nan,
            )
            for v in df["true_level"]
        ],
        dtype=float,
    )

    ax_cont.step(
        x,
        contention,
        where="mid",
        linewidth=1.1,
        color="black",
    )

    ax_cont.set_ylim(
        -0.35,
        4.35,
    )

    ax_cont.set_yticks(
        [0, 1, 2, 3, 4]
    )

    ax_cont.set_yticklabels(
        ["L0", "L1", "L2", "L3", "L4"]
    )

    ax_cont.set_title(
        "Runtime contention",
        fontsize=8,
        fontweight="bold",
        pad=3,
    )

    if show_left_labels:
        ax_cont.set_ylabel(
            "Contention",
            fontsize=7,
        )

    ax_cont.tick_params(
        axis="both",
        labelsize=6.5,
        length=2.5,
    )

    ax_cont.tick_params(
        axis="x",
        labelbottom=False,
    )

    ax_cont.grid(
        axis="y",
        linestyle=":",
        linewidth=0.45,
        alpha=0.45,
    )

    for side in [
        "top",
        "right",
    ]:
        ax_cont.spines[
            side
        ].set_visible(False)

    # ---------------------------------------------------------
    # Width heatmap
    # ---------------------------------------------------------

    matrix, dropped_mask = (
        build_width_matrix(df)
    )

    indexed = width_to_index_matrix(
        matrix
    )

    # Same yellow/gold family as the existing figure.
    cmap = ListedColormap(
        [
            "#FFF6C7",  # 0.25
            "#FFE28A",  # 0.50
            "#F5B934",  # 0.75
            "#D39A00",  # 1.00
        ]
    )

    cmap.set_bad(
        "#D0D0D0"
    )

    norm = BoundaryNorm(
        [
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
        ],
        cmap.N,
    )

    ax_width.imshow(
        indexed,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
        origin="upper",
        extent=[
            -0.5,
            n - 0.5,
            5.5,
            -0.5,
        ],
    )

    # Thin boundaries between sensor frames.
    for xx in np.arange(
        0.5,
        n - 0.5,
        1.0,
    ):
        ax_width.axvline(
            xx,
            linewidth=0.23,
            color="white",
            alpha=0.78,
        )

    ax_width.set_yticks(
        np.arange(6)
    )

    ax_width.set_yticklabels(
        STAGE_NAMES
    )

    ax_width.set_title(
        "Controller-selected stage widths",
        fontsize=8,
        fontweight="bold",
        pad=3,
    )

    if show_left_labels:
        ax_width.set_ylabel(
            "Elastic stage",
            fontsize=7,
        )

    ax_width.tick_params(
        axis="y",
        labelsize=6.5,
        length=0,
    )

    ax_width.tick_params(
        axis="x",
        labelbottom=False,
        length=2.5,
    )

    for side in [
        "top",
        "right",
        "bottom",
    ]:
        ax_width.spines[
            side
        ].set_visible(False)

    # ---------------------------------------------------------
    # Response latency
    #
    # IMPORTANT:
    # This deliberately uses arrival_to_finish_ms rather than
    # forward_ms so that the plotted latency uses the same
    # arrival-based deadline semantics as deadline_miss.
    # ---------------------------------------------------------

    response = pd.to_numeric(
        df["arrival_to_finish_ms"],
        errors="coerce",
    ).to_numpy(dtype=float)

    misses = (
        pd.to_numeric(
            df["deadline_miss"],
            errors="coerce",
        )
        .fillna(0)
        .to_numpy(dtype=int)
        > 0
    )

    processed = (
        df["status"]
        .astype(str)
        .str.lower()
        .eq("processed")
        .to_numpy()
    )

    response[
        ~processed
    ] = np.nan

    ax_lat.plot(
        x,
        response,
        color="red",
        marker="o",
        markersize=2.5,
        linewidth=1.15,
        label="Response latency",
        zorder=3,
    )

    ax_lat.axhline(
        period_ms,
        color="#555555",
        linewidth=1.0,
        label=f"Deadline ({period_ms:.2f} ms)",
        zorder=2,
    )

    # Keep dropped-frame terminology exactly as requested.
    add_dropped_spans(
        ax_lat,
        dropped_mask,
    )

    miss_x = x[
        misses
        & processed
        & np.isfinite(response)
    ]

    miss_y = response[
        misses
        & processed
        & np.isfinite(response)
    ]

    if len(miss_x) > 0:
        ax_lat.scatter(
            miss_x,
            miss_y,
            marker="x",
            s=31,
            linewidths=1.1,
            color="black",
            zorder=5,
        )

    valid_response = response[
        np.isfinite(response)
    ]

    ymax = period_ms * 1.10

    if len(valid_response):
        ymax = max(
            ymax,
            float(
                np.nanmax(
                    valid_response
                )
            ) * 1.08,
        )

    ax_lat.set_ylim(
        0,
        ymax,
    )

    ax_lat.set_title(
        "Response latency and deadline",
        fontsize=8,
        fontweight="bold",
        pad=3,
    )

    if show_left_labels:
        ax_lat.set_ylabel(
            "Time (ms)",
            fontsize=7,
        )

    ax_lat.set_xlabel(
        "Sensor frame index within selected window",
        fontsize=7,
    )

    xticks = sorted(
        set(
            [
                0,
                min(10, n - 1),
                min(20, n - 1),
                min(30, n - 1),
                min(40, n - 1),
                n - 1,
            ]
        )
    )

    ax_lat.set_xticks(
        xticks
    )

    ax_lat.tick_params(
        axis="both",
        labelsize=6.5,
        length=2.5,
    )

    ax_lat.grid(
        axis="y",
        linestyle=":",
        linewidth=0.45,
        alpha=0.45,
    )

    for side in [
        "top",
        "right",
    ]:
        ax_lat.spines[
            side
        ].set_visible(False)

    # ---------------------------------------------------------
    # Panel label
    # ---------------------------------------------------------

    ax_cont.text(
        -0.12,
        1.12,
        panel_label,
        transform=ax_cont.transAxes,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="bottom",
        clip_on=False,
    )

    return {
        "n": n,
        "dropped": int(
            dropped_mask.sum()
        ),
        "misses": int(
            (
                misses
                & processed
            ).sum()
        ),
        "pressure_fraction": (
            pressure_fraction_in_window(
                df,
                pressure_level,
            )
        ),
        "switch_rate": (
            contention_switch_rate(
                df
            )
        ),
        "response_mean_ms": (
            float(
                np.nanmean(
                    valid_response
                )
            )
            if len(valid_response)
            else np.nan
        ),
        "response_max_ms": (
            float(
                np.nanmax(
                    valid_response
                )
            )
            if len(valid_response)
            else np.nan
        ),
    }


def save_individual_panel(
    df,
    pressure_level,
    panel_label,
    period_ms,
    output_path,
    dpi,
):
    fig = plt.figure(
        figsize=(
            7.0,
            4.35,
        )
    )

    outer = fig.add_gridspec(
        1,
        1,
    )

    plot_one_panel(
        fig=fig,
        sub_spec=outer[0],
        df=df,
        pressure_level=pressure_level,
        panel_label=panel_label,
        period_ms=period_ms,
        show_left_labels=True,
    )

    fig.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.94,
        bottom=0.11,
    )

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def main():
    args = parse_args()

    input_root = Path(
        args.input_root
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    period_ms = (
        1000.0
        / float(args.hz)
    )

    levels = [
        "L1",
        "L2",
        "L3",
        "L4",
    ]

    panel_labels = [
        "(a)",
        "(b)",
        "(c)",
        "(d)",
    ]

    plt.rcParams.update(
        {
            "font.family":
                "sans-serif",

            "font.sans-serif":
                [
                    "Arial",
                    "Helvetica",
                    "DejaVu Sans",
                ],

            "font.size":
                8,

            "axes.linewidth":
                0.65,
        }
    )

    windows = {}
    summary_rows = []

    fraction_text = (
        f"{args.pressure_fraction:g}"
    )

    # ---------------------------------------------------------
    # Load / select representative windows
    # ---------------------------------------------------------

    for level in levels:

        experiment_dir = (
            input_root
            /
            f"L0_{level}_p{fraction_text}"
        )

        timeline_path = (
            experiment_dir
            /
            "frame_timeline.csv"
        )

        selected_path = (
            output_dir
            /
            f"selected_window_{level}.csv"
        )

        selected = load_or_select_window(
            timeline_path=timeline_path,
            selected_path=selected_path,
            pressure_level=level,
            window=args.window,
            pressure_fraction=args.pressure_fraction,
            force_reselect=args.force_reselect,
        )

        require_columns(
            selected,
            [
                "status",
                "true_level",
                "schedule",
                "arrival_to_finish_ms",
                "deadline_miss",
            ],
            selected_path,
        )

        if len(selected) != args.window:
            raise RuntimeError(
                f"{selected_path}: expected "
                f"{args.window} rows, got "
                f"{len(selected)}"
            )

        windows[
            level
        ] = selected.reset_index(
            drop=True
        )

    # ---------------------------------------------------------
    # Combined 2 x 2 figure
    # ---------------------------------------------------------

    fig = plt.figure(
        figsize=(
            11.6,
            6.75,
        )
    )

    outer = fig.add_gridspec(
        2,
        2,
        left=0.065,
        right=0.99,
        bottom=0.08,
        top=0.87,
        wspace=0.15,
        hspace=0.32,
    )

    for idx, level in enumerate(
        levels
    ):
        row = idx // 2
        col = idx % 2

        stats = plot_one_panel(
            fig=fig,
            sub_spec=outer[
                row,
                col,
            ],
            df=windows[level],
            pressure_level=level,
            panel_label=panel_labels[idx],
            period_ms=period_ms,
            show_left_labels=True,
        )

        first = windows[level].iloc[0]
        last = windows[level].iloc[-1]

        summary_rows.append(
            {
                "pressure_level":
                    level,

                "panel":
                    panel_labels[idx],

                "scene":
                    first.get(
                        "scene",
                        "",
                    ),

                "start_global_index":
                    first.get(
                        "global_index",
                        "",
                    ),

                "end_global_index":
                    last.get(
                        "global_index",
                        "",
                    ),

                "frames":
                    stats["n"],

                "pressure_fraction":
                    stats[
                        "pressure_fraction"
                    ],

                "switch_rate":
                    stats[
                        "switch_rate"
                    ],

                "dropped_frames":
                    stats[
                        "dropped"
                    ],

                "deadline_misses":
                    stats[
                        "misses"
                    ],

                "response_mean_ms":
                    stats[
                        "response_mean_ms"
                    ],

                "response_max_ms":
                    stats[
                        "response_max_ms"
                    ],
            }
        )

    # ---------------------------------------------------------
    # Global legend
    # ---------------------------------------------------------

    width_colors = [
        "#FFF6C7",
        "#FFE28A",
        "#F5B934",
        "#D39A00",
    ]

    legend_handles = [
        Patch(
            facecolor=width_colors[0],
            edgecolor="none",
            label="Width 0.25",
        ),

        Patch(
            facecolor=width_colors[1],
            edgecolor="none",
            label="Width 0.50",
        ),

        Patch(
            facecolor=width_colors[2],
            edgecolor="none",
            label="Width 0.75",
        ),

        Patch(
            facecolor=width_colors[3],
            edgecolor="none",
            label="Width 1.00",
        ),

        # Keep this wording unchanged.
        Patch(
            facecolor="#D0D0D0",
            edgecolor="none",
            label="Dropped frame",
        ),

        Line2D(
            [0],
            [0],
            color="red",
            marker="o",
            linewidth=1.4,
            markersize=4,
            label="Response latency",
        ),

        Line2D(
            [0],
            [0],
            color="#555555",
            linewidth=1.1,
            label=(
                f"Deadline "
                f"({period_ms:.2f} ms)"
            ),
        ),

        Line2D(
            [0],
            [0],
            color="black",
            marker="x",
            linewidth=0,
            markersize=6,
            label="Deadline miss",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.52,
            0.985,
        ),
        ncol=4,
        frameon=False,
        fontsize=8,
        columnspacing=1.15,
        handlelength=1.9,
        handletextpad=0.5,
    )

    combined_png = (
        output_dir
        /
        "runtime_behavior_analysis.png"
    )

    combined_pdf = (
        output_dir
        /
        "runtime_behavior_analysis.pdf"
    )

    fig.savefig(
        combined_png,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        combined_pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    # ---------------------------------------------------------
    # Individual panel PNGs
    # ---------------------------------------------------------

    for idx, level in enumerate(
        levels
    ):
        individual = (
            output_dir
            /
            (
                "controller_runtime_case_study_"
                f"{args.hz:g}Hz_"
                f"L0_{level}_"
                f"p{fraction_text}_"
                f"seed{args.trace_seed}.png"
            )
        )

        save_individual_panel(
            df=windows[level],
            pressure_level=level,
            panel_label=panel_labels[idx],
            period_ms=period_ms,
            output_path=individual,
            dpi=args.dpi,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary_path = (
        output_dir
        /
        "selected_windows_summary.csv"
    )

    pd.DataFrame(
        summary_rows
    ).to_csv(
        summary_path,
        index=False,
    )

    print()
    print(
        "=" * 86
    )
    print(
        "CONTROLLER RUNTIME CASE STUDY"
    )
    print(
        "=" * 86
    )
    print(
        f"input root       : {input_root}"
    )
    print(
        f"output dir       : {output_dir}"
    )
    print(
        f"input frequency  : {args.hz:g} Hz"
    )
    print(
        f"deadline         : {period_ms:.6f} ms"
    )
    print(
        "latency field    : arrival_to_finish_ms"
    )
    print(
        "latency label    : Response latency"
    )
    print(
        "drop label       : Dropped frame"
    )
    print(
        "last stage label : 3D Voxel"
    )
    print(
        f"combined PNG     : {combined_png}"
    )
    print(
        f"combined PDF     : {combined_pdf}"
    )
    print(
        f"summary          : {summary_path}"
    )
    print(
        "=" * 86
    )

    print()

    for row in summary_rows:
        print(
            f"{row['pressure_level']} | "
            f"scene={row['scene']} | "
            f"range="
            f"{row['start_global_index']}--"
            f"{row['end_global_index']} | "
            f"pressure="
            f"{100.0 * row['pressure_fraction']:.1f}% | "
            f"drop={row['dropped_frames']} | "
            f"miss={row['deadline_misses']} | "
            f"response_mean="
            f"{row['response_mean_ms']:.3f} ms | "
            f"response_max="
            f"{row['response_max_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
