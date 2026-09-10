#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LEVELS = ["L0", "L1", "L2", "L3", "L4"]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot controller CPU overhead and runtime "
            "p99-bound calibration for TV-streamDSGN."
        )
    )

    p.add_argument(
        "--audit_dir",
        default=(
            "outputs/elastic_bev/"
            "elastic_bev_v4_bn_from_k3/"
            "system_audit/"
            "controller_runtime_bounds_35hz"
        ),
    )

    p.add_argument(
        "--output_dir",
        default=(
            "outputs/paper_figures/"
            "controller_overhead_bound_calibration"
        ),
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return p.parse_args()


def configure_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
        ],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def percentile99(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.size == 0:
        raise RuntimeError(
            "Cannot compute p99 on empty data."
        )

    return float(
        np.percentile(
            x,
            99,
        )
    )


def validate_inputs(
    overhead_df,
    bound_df,
):
    required_overhead = {
        "level",
        "controller_cpu_total_ms",
        "cpu_overhead_pct_period",
    }

    required_bound = {
        "level",
        "checkpoint",
        "raw_p99_violation",
        "guarded_violation",
        "control_guard_ms",
    }

    miss1 = (
        required_overhead
        - set(overhead_df.columns)
    )

    miss2 = (
        required_bound
        - set(bound_df.columns)
    )

    if miss1:
        raise RuntimeError(
            "controller_overhead.csv missing columns: "
            + ", ".join(sorted(miss1))
        )

    if miss2:
        raise RuntimeError(
            "bound_validation.csv missing columns: "
            + ", ".join(sorted(miss2))
        )

    present_levels = set(
        overhead_df["level"]
        .astype(str)
        .unique()
    )

    for level in LEVELS:
        if level not in present_levels:
            raise RuntimeError(
                f"Missing overhead samples for {level}"
            )


def summarize_overhead(
    overhead_df,
):
    rows = []

    for level in LEVELS:
        x = overhead_df[
            overhead_df["level"] == level
        ]

        if len(x) == 0:
            raise RuntimeError(
                f"No overhead rows for {level}"
            )

        cpu_us = (
            x[
                "controller_cpu_total_ms"
            ].astype(float)
            * 1000.0
        )

        pct = (
            x[
                "cpu_overhead_pct_period"
            ].astype(float)
        )

        rows.append({
            "level": level,
            "n": len(x),
            "p99_us": percentile99(
                cpu_us
            ),
            "p99_pct_period": percentile99(
                pct
            ),
        })

    return pd.DataFrame(
        rows
    )


def summarize_bounds(
    bound_df,
):
    raw = (
        bound_df[
            "raw_p99_violation"
        ]
        .astype(int)
    )

    guarded = (
        bound_df[
            "guarded_violation"
        ]
        .astype(int)
    )

    n = len(
        bound_df
    )

    if n == 0:
        raise RuntimeError(
            "bound_validation.csv is empty"
        )

    raw_count = int(
        raw.sum()
    )

    guarded_count = int(
        guarded.sum()
    )

    raw_rate = (
        100.0
        * raw_count
        / n
    )

    guarded_rate = (
        100.0
        * guarded_count
        / n
    )

    return {
        "n": n,
        "raw_count": raw_count,
        "guarded_count": guarded_count,
        "raw_rate": raw_rate,
        "guarded_rate": guarded_rate,
    }


def load_summary(
    summary_path,
):
    if not summary_path.is_file():
        return {}

    with summary_path.open() as f:
        return json.load(
            f
        )


def plot_figure(
    overhead_summary,
    bound_summary,
    audit_summary,
    output_dir,
    dpi,
):
    configure_style()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.05),
    )

    ax1, ax2 = axes

    # ============================================================
    # (a) Controller CPU overhead
    # ============================================================

    x = np.arange(
        len(LEVELS)
    )

    p99_us = (
        overhead_summary[
            "p99_us"
        ].to_numpy()
    )

    p99_pct = (
        overhead_summary[
            "p99_pct_period"
        ].to_numpy()
    )

    bars = ax1.bar(
        x,
        p99_us,
        width=0.62,
        edgecolor="black",
        linewidth=0.7,
    )

    ax1.set_xticks(
        x
    )

    ax1.set_xticklabels(
        LEVELS
    )

    ax1.set_xlabel(
        "GPU contention level"
    )

    ax1.set_ylabel(
        "Controller CPU p99 overhead (μs)"
    )

    ax1.set_title(
        "(a) Controller overhead",
        fontweight="bold",
        pad=6,
    )

    ax1.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )

    ax1.set_axisbelow(
        True
    )

    ymax = max(
        p99_us
    )

    # More headroom for labels.
    ax1.set_ylim(
        0,
        ymax * 1.38,
    )

    # Fixed point offset above each bar.
    for bar, us, pct in zip(
        bars,
        p99_us,
        p99_pct,
    ):
        ax1.annotate(
            f"{us:.1f}\n({pct:.2f}%)",
            xy=(
                bar.get_x()
                + bar.get_width() / 2.0,
                bar.get_height(),
            ),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.2,
            linespacing=1.15,
            clip_on=False,
        )

    input_hz = float(
        audit_summary.get(
            "input_hz",
            35.0,
        )
    )

    period_ms = float(
        audit_summary.get(
            "period_ms",
            1000.0 / input_hz,
        )
    )

    ax1.text(
        0.5,
        0.96,
        f"Frame period = {period_ms:.2f} ms",
        transform=ax1.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
    )

    # ============================================================
    # (b) Bound calibration
    # ============================================================

    labels = [
        "Raw p99\nprofile",
        "+ Runtime\nguard",
    ]

    rates = [
        bound_summary[
            "raw_rate"
        ],
        bound_summary[
            "guarded_rate"
        ],
    ]

    counts = [
        bound_summary[
            "raw_count"
        ],
        bound_summary[
            "guarded_count"
        ],
    ]

    xb = np.arange(
        2
    )

    bars2 = ax2.bar(
        xb,
        rates,
        width=0.56,
        edgecolor="black",
        linewidth=0.7,
    )

    ax2.set_xticks(
        xb
    )

    ax2.set_xticklabels(
        labels
    )

    ax2.set_ylabel(
        "Bound violation rate (%)"
    )

    ax2.set_title(
        "(b) Runtime bound calibration",
        fontweight="bold",
        pad=6,
    )

    ax2.grid(
        axis="y",
        linestyle="--",
        linewidth=0.5,
        alpha=0.35,
    )

    ax2.set_axisbelow(
        True
    )

    ymax2 = max(
        rates
    )

    # More vertical room above the tall bar.
    ax2.set_ylim(
        0,
        max(
            10.0,
            ymax2 * 1.35,
        ),
    )

    n = bound_summary[
        "n"
    ]

    # Fixed point offset, so labels never overlap bars.
    for bar, rate, count in zip(
        bars2,
        rates,
        counts,
    ):
        ax2.annotate(
            (
                f"{rate:.1f}%\n"
                f"{count}/{n}"
            ),
            xy=(
                bar.get_x()
                + bar.get_width() / 2.0,
                bar.get_height(),
            ),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7.5,
            linespacing=1.15,
            clip_on=False,
        )

    guard_per_boundary = float(
        audit_summary.get(
            "control_guard_per_boundary_ms",
            0.25,
        )
    )

    # Move the note to the upper-right corner to avoid bar labels.
    ax2.text(
        0.98,
        0.97,
        (
            f"Guard = {guard_per_boundary:.2f} ms "
            "per remaining boundary"
        ),
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=7.0,
    )

    # ============================================================
    # Final layout
    # ============================================================

    fig.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.19,
        top=0.88,
        wspace=0.31,
    )

    png = (
        output_dir
        / "controller_overhead_bound_calibration.png"
    )

    pdf = (
        output_dir
        / "controller_overhead_bound_calibration.pdf"
    )

    svg = (
        output_dir
        / "controller_overhead_bound_calibration.svg"
    )

    fig.savefig(
        png,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    return png, pdf, svg


def write_summary_csv(
    overhead_summary,
    bound_summary,
    output_dir,
):
    overhead_path = (
        output_dir
        / "controller_overhead_p99_summary.csv"
    )

    overhead_summary.to_csv(
        overhead_path,
        index=False,
    )

    bound_path = (
        output_dir
        / "bound_calibration_summary.csv"
    )

    pd.DataFrame([
        {
            "setting":
                "Raw p99 profile",

            "violation_count":
                bound_summary[
                    "raw_count"
                ],

            "total":
                bound_summary[
                    "n"
                ],

            "violation_rate_pct":
                bound_summary[
                    "raw_rate"
                ],
        },
        {
            "setting":
                "+ Runtime guard",

            "violation_count":
                bound_summary[
                    "guarded_count"
                ],

            "total":
                bound_summary[
                    "n"
                ],

            "violation_rate_pct":
                bound_summary[
                    "guarded_rate"
                ],
        },
    ]).to_csv(
        bound_path,
        index=False,
    )

    return (
        overhead_path,
        bound_path,
    )


def main():
    args = parse_args()

    audit_dir = Path(
        args.audit_dir
    ).resolve()

    output_dir = Path(
        args.output_dir
    ).resolve()

    overhead_csv = (
        audit_dir
        / "controller_overhead.csv"
    )

    bound_csv = (
        audit_dir
        / "bound_validation.csv"
    )

    summary_json = (
        audit_dir
        / "summary.json"
    )

    for p in [
        overhead_csv,
        bound_csv,
    ]:
        if not p.is_file():
            raise FileNotFoundError(
                f"Missing required file: {p}"
            )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 100
    )

    print(
        "TV-streamDSGN CONTROLLER OVERHEAD "
        "AND BOUND CALIBRATION"
    )

    print(
        "=" * 100
    )

    print(
        f"audit dir : {audit_dir}"
    )

    overhead_df = pd.read_csv(
        overhead_csv
    )

    bound_df = pd.read_csv(
        bound_csv
    )

    validate_inputs(
        overhead_df,
        bound_df,
    )

    audit_summary = load_summary(
        summary_json
    )

    overhead_summary = (
        summarize_overhead(
            overhead_df
        )
    )

    bound_summary = (
        summarize_bounds(
            bound_df
        )
    )

    print()
    print(
        "Controller p99 overhead:"
    )

    for _, r in (
        overhead_summary.iterrows()
    ):
        print(
            f"  {r['level']}: "
            f"{r['p99_us']:.2f} us | "
            f"{r['p99_pct_period']:.3f}% "
            "of frame period"
        )

    print()
    print(
        "Aggregate bound validation:"
    )

    print(
        "  Raw     : "
        f"{bound_summary['raw_count']}/"
        f"{bound_summary['n']} = "
        f"{bound_summary['raw_rate']:.3f}%"
    )

    print(
        "  Guarded : "
        f"{bound_summary['guarded_count']}/"
        f"{bound_summary['n']} = "
        f"{bound_summary['guarded_rate']:.3f}%"
    )

    (
        overhead_summary_csv,
        bound_summary_csv,
    ) = write_summary_csv(
        overhead_summary,
        bound_summary,
        output_dir,
    )

    png, pdf, svg = plot_figure(
        overhead_summary=overhead_summary,
        bound_summary=bound_summary,
        audit_summary=audit_summary,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    print()
    print(
        "=" * 100
    )

    print(
        "[DONE]"
    )

    print(
        f"PNG : {png}"
    )

    print(
        f"PDF : {pdf}"
    )

    print(
        f"SVG : {svg}"
    )

    print(
        f"CSV : {overhead_summary_csv}"
    )

    print(
        f"CSV : {bound_summary_csv}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
