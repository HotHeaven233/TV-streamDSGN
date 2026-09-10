#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


LEVELS = ["L1", "L2", "L3", "L4"]

STAGES = [
    "Res2",
    "Res3",
    "Res4",
    "FPN",
    "Stereo",
    "3D RPN",
]

LEVEL_TO_Y = {
    "L0": 0,
    "L1": 1,
    "L2": 2,
    "L3": 3,
    "L4": 4,
}

WIDTH_TO_IDX = {
    0.25: 0,
    0.50: 1,
    0.75: 2,
    1.00: 3,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot four separate TV-streamDSGN controller/runtime "
            "case-study figures for Random50 L1-L4."
        )
    )

    p.add_argument(
        "--repo_root",
        default="/data/jhb/workspace/streamDSGN",
    )
    p.add_argument(
        "--exp_name",
        default="elastic_bev_v4_bn_from_k3",
    )
    p.add_argument(
        "--hz",
        type=int,
        default=35,
    )
    p.add_argument(
        "--pressure_fraction",
        type=float,
        default=0.5,
    )
    p.add_argument(
        "--trace_seed",
        type=int,
        default=20260903,
    )
    p.add_argument(
        "--window",
        type=int,
        default=50,
    )
    p.add_argument(
        "--output_dir",
        default=(
            "outputs/paper_figures/"
            "controller_runtime_case_study_all_levels"
        ),
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return p.parse_args()


def frac_tag(x):
    return f"{x:g}"


def timeline_path(
    repo,
    exp_name,
    hz,
    level,
    frac,
    seed,
):
    return (
        repo
        / "outputs"
        / "elastic_bev"
        / exp_name
        / "formal_streaming_random50"
        / f"{hz}Hz_forward_only_seed{seed}"
        / f"L0_{level}_p{frac_tag(frac)}"
        / "frame_timeline.csv"
    )


def load_timeline(path):
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise RuntimeError(
            f"timeline is empty: {path}"
        )

    required = {
        "global_index",
        "scene",
        "local_pos",
        "status",
        "forward_ms",
        "deadline_miss",
        "true_level",
        "schedule",
    }

    missing = required - set(
        rows[0].keys()
    )

    if missing:
        raise RuntimeError(
            f"{path}: missing columns: "
            f"{sorted(missing)}"
        )

    for r in rows:
        r["global_index"] = int(
            r["global_index"]
        )
        r["local_pos"] = int(
            r["local_pos"]
        )

    return rows


def parse_schedule(s):
    s = (s or "").strip()

    if not s:
        return None

    vals = [
        float(x.strip())
        for x in s.split(",")
    ]

    if len(vals) != 6:
        raise RuntimeError(
            f"Expected 6 widths, got "
            f"{len(vals)}: {s}"
        )

    for v in vals:
        if v not in WIDTH_TO_IDX:
            raise RuntimeError(
                f"Invalid width {v}: {s}"
            )

    return vals


def group_by_scene(rows):
    out = {}

    for r in rows:
        out.setdefault(
            r["scene"],
            [],
        ).append(r)

    for scene in out:
        out[scene].sort(
            key=lambda r: r["local_pos"]
        )

    return out


def processed_rows(rows):
    return [
        r
        for r in rows
        if r["status"] == "processed"
    ]


def count_misses(rows):
    n = 0

    for r in processed_rows(rows):
        value = (
            r.get(
                "deadline_miss",
                "",
            )
            or ""
        ).strip()

        if value:
            n += int(value)

    return n


def count_drops(rows):
    return sum(
        r["status"] == "dropped"
        for r in rows
    )


def count_true_switches(rows):
    return sum(
        rows[i]["true_level"]
        != rows[i - 1]["true_level"]
        for i in range(
            1,
            len(rows),
        )
    )


def count_schedule_changes(rows):
    schedules = [
        r["schedule"]
        for r in processed_rows(rows)
        if (r.get("schedule") or "").strip()
    ]

    return sum(
        schedules[i]
        != schedules[i - 1]
        for i in range(
            1,
            len(schedules),
        )
    )


def choose_best_window(
    rows,
    window,
):
    scenes = group_by_scene(
        rows
    )

    best_rows = None
    best_meta = None
    best_score = None

    for scene, sr in scenes.items():

        if len(sr) < window:
            continue

        for start in range(
            0,
            len(sr) - window + 1,
        ):
            chunk = sr[
                start:start + window
            ]

            misses = count_misses(
                chunk
            )
            drops = count_drops(
                chunk
            )
            switches = (
                count_true_switches(
                    chunk
                )
            )
            sched_changes = (
                count_schedule_changes(
                    chunk
                )
            )

            processed = len(
                processed_rows(
                    chunk
                )
            )

            # Selection priority:
            # 1. zero/few deadline misses
            # 2. few drops
            # 3. more contention switches
            # 4. more schedule adaptation
            # 5. more processed frames
            score = (
                misses,
                drops,
                -switches,
                -sched_changes,
                -processed,
                scene,
                chunk[0]["local_pos"],
            )

            if (
                best_score is None
                or score < best_score
            ):
                best_score = score
                best_rows = chunk

                best_meta = {
                    "scene": scene,
                    "start_local_pos":
                        chunk[0]["local_pos"],
                    "end_local_pos":
                        chunk[-1]["local_pos"],
                    "misses": misses,
                    "drops": drops,
                    "true_switches":
                        switches,
                    "schedule_changes":
                        sched_changes,
                    "processed":
                        processed,
                }

    if best_rows is None:
        raise RuntimeError(
            f"No valid window "
            f"with length={window}"
        )

    return best_rows, best_meta


def prepare_arrays(rows):
    n = len(rows)

    x = np.arange(
        n
    )

    true_y = np.full(
        n,
        np.nan,
    )

    forward = np.full(
        n,
        np.nan,
    )

    miss = np.zeros(
        n,
        dtype=bool,
    )

    dropped = np.zeros(
        n,
        dtype=bool,
    )

    schedule_idx = np.full(
        (6, n),
        np.nan,
    )

    for i, r in enumerate(rows):

        level = (
            r.get(
                "true_level",
                "",
            )
            or ""
        ).strip()

        if level:
            true_y[i] = (
                LEVEL_TO_Y[level]
            )

        if r["status"] == "dropped":
            dropped[i] = True
            continue

        fwd = (
            r.get(
                "forward_ms",
                "",
            )
            or ""
        ).strip()

        if fwd:
            value = float(fwd)

            if math.isfinite(value):
                forward[i] = value

        dm = (
            r.get(
                "deadline_miss",
                "",
            )
            or ""
        ).strip()

        if dm:
            miss[i] = bool(
                int(dm)
            )

        schedule = parse_schedule(
            r.get(
                "schedule",
                "",
            )
        )

        if schedule is not None:
            for stage_i, width in enumerate(
                schedule
            ):
                schedule_idx[
                    stage_i,
                    i,
                ] = WIDTH_TO_IDX[
                    width
                ]

    return {
        "x": x,
        "true_y": true_y,
        "forward": forward,
        "miss": miss,
        "dropped": dropped,
        "schedule_idx":
            schedule_idx,
    }


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
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,

        "axes.linewidth": 0.8,

        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax):
    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    ax.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.55,
    )


def add_dropped_background(
    ax,
    dropped,
):
    for i, flag in enumerate(
        dropped
    ):
        if flag:
            ax.axvspan(
                i - 0.5,
                i + 0.5,
                color="#d9d9d9",
                linewidth=0,
                zorder=0,
            )


def write_rows(
    path,
    rows,
):
    with path.open(
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


def plot_one_level(
    level,
    rows,
    meta,
    args,
    out_dir,
    cmap,
    norm,
    width_colors,
):
    data = prepare_arrays(
        rows
    )

    deadline_ms = (
        1000.0
        / float(args.hz)
    )

    n = len(rows)

    fig = plt.figure(
        figsize=(7.2, 4.8)
    )

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[
            0.85,
            1.55,
            1.15,
        ],
        hspace=0.28,
    )

    ax1 = fig.add_subplot(
        gs[0]
    )

    ax2 = fig.add_subplot(
        gs[1],
        sharex=ax1,
    )

    ax3 = fig.add_subplot(
        gs[2],
        sharex=ax1,
    )

    # ============================================================
    # (a) True contention only
    # ============================================================

    add_dropped_background(
        ax1,
        data["dropped"],
    )

    ax1.step(
        data["x"],
        data["true_y"],
        where="mid",
        color="black",
        linewidth=1.5,
        zorder=3,
    )

    ax1.set_ylim(
        -0.35,
        4.35,
    )

    ax1.set_yticks(
        [0, 1, 2, 3, 4]
    )

    ax1.set_yticklabels(
        ["L0", "L1", "L2", "L3", "L4"]
    )

    ax1.set_ylabel(
        "Contention"
    )
    ax1.yaxis.set_label_coords(
        -0.095,
        0.5,
    )

    ax1.set_title(
        "Runtime contention",
        loc="center",
        fontweight="bold",
        pad=5,
    )

    style_axis(
        ax1
    )

    # ============================================================
    # (b) Discrete stage widths
    # ============================================================

    add_dropped_background(
        ax2,
        data["dropped"],
    )

    # Draw the 6 x N schedule as discrete cells.
    # Thin white borders create small gaps both:
    #   - between elastic stages
    #   - between consecutive sensor frames
    #
    # This keeps the heatmap compact while making each cell legible.
    masked = np.ma.masked_invalid(
        data["schedule_idx"]
    )

    x_edges = np.arange(
        n + 1,
        dtype=float,
    ) - 0.5

    y_edges = np.arange(
        7,
        dtype=float,
    ) - 0.5

    ax2.pcolormesh(
        x_edges,
        y_edges,
        masked,
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="white",
        linewidth=0.35,
        antialiased=True,
        zorder=2,
    )

    ax2.set_ylim(
        5.5,
        -0.5,
    )

    ax2.set_yticks(
        np.arange(6)
    )

    ax2.set_yticklabels(
        STAGES
    )

    ax2.set_ylabel(
        "Elastic stage"
    )
    ax2.yaxis.set_label_coords(
        -0.095,
        0.5,
    )

    ax2.set_title(
        "Controller-selected stage widths",
        loc="center",
        fontweight="bold",
        pad=5,
    )

    ax2.spines[
        "top"
    ].set_visible(False)

    ax2.spines[
        "right"
    ].set_visible(False)

    # ============================================================
    # (c) Forward latency vs constant deadline
    # ============================================================

    add_dropped_background(
        ax3,
        data["dropped"],
    )

    proc = np.isfinite(
        data["forward"]
    )

    ax3.axhline(
        deadline_ms,
        color="#555555",
        linewidth=1.3,
        linestyle="-",
        zorder=2,
    )

    ax3.plot(
        data["x"][proc],
        data["forward"][proc],
        color="red",
        linewidth=1.8,
        marker="o",
        markersize=3.4,
        zorder=4,
    )

    miss_mask = (
        data["miss"]
        & proc
    )

    if np.any(
        miss_mask
    ):
        ax3.scatter(
            data["x"][miss_mask],
            data["forward"][miss_mask],
            marker="x",
            s=36,
            color="black",
            linewidths=1.2,
            zorder=6,
        )

    # ------------------------------------------------------------
    # IMPORTANT:
    # time y-axis always starts from zero
    # ------------------------------------------------------------

    finite_forward = (
        data["forward"][proc]
    )

    if finite_forward.size:
        max_forward = float(
            np.max(
                finite_forward
            )
        )
    else:
        max_forward = 0.0

    upper_time = max(
        deadline_ms,
        max_forward,
    )

    upper_time = (
        math.ceil(
            upper_time
            * 1.08
        )
    )

    ax3.set_ylim(
        0,
        upper_time,
    )

    ax3.set_ylabel(
        "Time (ms)"
    )
    ax3.yaxis.set_label_coords(
        -0.095,
        0.5,
    )

    ax3.set_xlabel(
        "Sensor frame index within selected window"
    )

    ax3.set_title(
        "Forward latency and deadline",
        loc="center",
        fontweight="bold",
        pad=5,
    )

    style_axis(
        ax3
    )

    # ============================================================
    # Shared x-axis
    # ============================================================

    for ax in (
        ax1,
        ax2,
        ax3,
    ):
        ax.set_xlim(
            -0.5,
            n - 0.5,
        )

    plt.setp(
        ax1.get_xticklabels(),
        visible=False,
    )

    plt.setp(
        ax2.get_xticklabels(),
        visible=False,
    )

    step = (
        10
        if n >= 40
        else 5
    )

    ticks = list(
        range(
            0,
            n,
            step,
        )
    )

    if (n - 1) not in ticks:
        ticks.append(
            n - 1
        )

    ax3.set_xticks(
        ticks
    )

    ax3.set_xticklabels(
        [
            str(x)
            for x in ticks
        ]
    )

    # ============================================================
    # Legend
    # ============================================================

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
        Patch(
            facecolor="#d9d9d9",
            edgecolor="none",
            label="Dropped frame",
        ),
        Line2D(
            [0], [0],
            color="red",
            marker="o",
            markersize=4,
            linewidth=1.8,
            label="Forward latency",
        ),
        Line2D(
            [0], [0],
            color="#555555",
            linewidth=1.3,
            label=(
                f"Deadline "
                f"({deadline_ms:.2f} ms)"
            ),
        ),
    ]

    if np.any(
        miss_mask
    ):
        legend_handles.append(
            Line2D(
                [0], [0],
                color="black",
                marker="x",
                linestyle="none",
                markersize=6,
                label="Deadline miss",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(
            0.5,
            0.99,
        ),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.0,
    )

    fig.suptitle(
        (
            f"TV-streamDSGN runtime behavior under "
            f"Random50: L0/{level}"
        ),
        fontsize=9,
        y=0.998,
    )

    fig.subplots_adjust(
        left=0.125,
        right=0.985,
        bottom=0.10,
        top=0.885,
    )

    tag = (
        f"{args.hz}Hz_"
        f"L0_{level}_"
        f"p{frac_tag(args.pressure_fraction)}_"
        f"seed{args.trace_seed}"
    )

    stem = (
        out_dir
        / (
            "controller_runtime_case_study_"
            f"{tag}"
        )
    )

    fig.savefig(
        stem.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(
            ".svg"
        ),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(
            ".png"
        ),
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"[{level}] "
        f"scene={meta['scene']} "
        f"frames="
        f"{meta['start_local_pos']}-"
        f"{meta['end_local_pos']} "
        f"miss={meta['misses']} "
        f"drop={meta['drops']} "
        f"switch={meta['true_switches']} "
        f"sched_change="
        f"{meta['schedule_changes']}"
    )

    print(
        f"      PNG={stem.with_suffix('.png')}"
    )

    return stem


def main():
    args = parse_args()

    if args.window < 5:
        raise ValueError(
            "--window must be >= 5"
        )

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

    configure_style()

    # ============================================================
    # Four discrete width colors, same blue family, light -> dark
    # ============================================================

    # Four discrete width levels in one yellow/gold family.
    # Light -> dark as the execution width increases.
    width_colors = [
        "#fff8c6",  # 0.25
        "#ffe58a",  # 0.50
        "#f5c242",  # 0.75
        "#c99700",  # 1.00
    ]

    cmap = ListedColormap(
        width_colors
    )

    cmap.set_bad(
        color="white"
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

    summary_rows = []

    print()
    print("=" * 110)
    print(
        "TV-streamDSGN CONTROLLER/RUNTIME "
        "CASE STUDY — FOUR SEPARATE FIGURES"
    )
    print("=" * 110)

    for level in LEVELS:

        path = timeline_path(
            repo=repo,
            exp_name=args.exp_name,
            hz=args.hz,
            level=level,
            frac=args.pressure_fraction,
            seed=args.trace_seed,
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing timeline:\n"
                f"  {path}"
            )

        rows = load_timeline(
            path
        )

        chosen, meta = (
            choose_best_window(
                rows,
                args.window,
            )
        )

        selected_csv = (
            out_dir
            / f"selected_window_{level}.csv"
        )

        write_rows(
            selected_csv,
            chosen,
        )

        plot_one_level(
            level=level,
            rows=chosen,
            meta=meta,
            args=args,
            out_dir=out_dir,
            cmap=cmap,
            norm=norm,
            width_colors=width_colors,
        )

        summary_rows.append({
            "level": level,
            "scene": meta["scene"],
            "start_local_pos":
                meta["start_local_pos"],
            "end_local_pos":
                meta["end_local_pos"],
            "window":
                args.window,
            "processed":
                meta["processed"],
            "drops":
                meta["drops"],
            "misses":
                meta["misses"],
            "true_switches":
                meta["true_switches"],
            "schedule_changes":
                meta["schedule_changes"],
            "source":
                str(path),
        })

    summary_path = (
        out_dir
        / "selected_windows_summary.csv"
    )

    with summary_path.open(
        "w",
        newline="",
    ) as f:

        fields = list(
            summary_rows[0].keys()
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    print("=" * 110)
    print(
        f"[SUMMARY] {summary_path}"
    )
    print("=" * 110)


if __name__ == "__main__":
    main()
