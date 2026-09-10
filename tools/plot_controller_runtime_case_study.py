#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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

Y_TO_LEVEL = {
    v: k for k, v in LEVEL_TO_Y.items()
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot TV-streamDSGN controller/runtime behavior "
            "from a formal Random50 frame_timeline.csv."
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
        "--pressure_level",
        choices=["L1", "L2", "L3", "L4"],
        default="L4",
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
        default=60,
        help="Number of sensor frames shown.",
    )

    p.add_argument(
        "--scene",
        default=None,
        help=(
            "Optional exact scene ID. "
            "Default: first scene with enough frames."
        ),
    )

    p.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start local_pos inside the selected scene.",
    )

    p.add_argument(
        "--output_dir",
        default="outputs/paper_figures/controller_runtime_case_study",
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return p.parse_args()


def frac_tag(x):
    return f"{x:g}"


def timeline_path(args, repo):
    return (
        repo
        / "outputs"
        / "elastic_bev"
        / args.exp_name
        / "formal_streaming_random50"
        / (
            f"{args.hz}Hz_forward_only_"
            f"seed{args.trace_seed}"
        )
        / (
            f"L0_{args.pressure_level}_"
            f"p{frac_tag(args.pressure_fraction)}"
        )
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
        "initial_budget_ms",
        "deadline_miss",
        "true_level",
        "observed_level",
        "schedule",
    }

    missing = required - set(rows[0].keys())

    if missing:
        raise RuntimeError(
            f"{path}: missing columns: "
            f"{sorted(missing)}"
        )

    for r in rows:
        r["local_pos"] = int(r["local_pos"])
        r["global_index"] = int(r["global_index"])

    return rows


def select_window(rows, scene, start, window):
    scenes = {}

    for r in rows:
        scenes.setdefault(
            r["scene"],
            []
        ).append(r)

    for s in scenes:
        scenes[s].sort(
            key=lambda r: r["local_pos"]
        )

    if scene is not None:
        if scene not in scenes:
            raise RuntimeError(
                f"scene {scene!r} not found. "
                f"Available scenes: {list(scenes)[:20]}"
            )
        selected_scene = scene

    else:
        # Deterministic selection:
        # first scene containing at least `window` frames
        # after the requested start position.
        selected_scene = None

        for s, sr in scenes.items():
            eligible = [
                r for r in sr
                if r["local_pos"] >= start
            ]

            if len(eligible) >= window:
                selected_scene = s
                break

        if selected_scene is None:
            # Fall back to the longest scene.
            selected_scene = max(
                scenes,
                key=lambda s: len(scenes[s]),
            )

    sr = scenes[selected_scene]

    selected = [
        r
        for r in sr
        if r["local_pos"] >= start
    ][:window]

    if not selected:
        raise RuntimeError(
            f"No rows for scene={selected_scene}, "
            f"start={start}"
        )

    return selected_scene, selected


def parse_float(s):
    if s is None or s == "":
        return np.nan

    x = float(s)

    if not math.isfinite(x):
        return np.nan

    return x


def parse_schedule(s):
    if s is None:
        return None

    s = s.strip()

    if not s:
        return None

    values = [
        float(x.strip())
        for x in s.split(",")
    ]

    if len(values) != 6:
        raise RuntimeError(
            f"Expected 6-stage schedule, got "
            f"{len(values)} values: {s!r}"
        )

    valid = {
        1.0,
        0.75,
        0.5,
        0.25,
    }

    for x in values:
        if x not in valid:
            raise RuntimeError(
                f"Invalid elastic width {x} "
                f"in schedule {s!r}"
            )

    return values


def prepare_arrays(rows):
    n = len(rows)

    x = np.arange(n)

    true_y = np.full(
        n,
        np.nan,
        dtype=float,
    )

    observed_y = np.full(
        n,
        np.nan,
        dtype=float,
    )

    schedule_matrix = np.full(
        (6, n),
        np.nan,
        dtype=float,
    )

    forward = np.full(
        n,
        np.nan,
        dtype=float,
    )

    budget = np.full(
        n,
        np.nan,
        dtype=float,
    )

    miss = np.zeros(
        n,
        dtype=bool,
    )

    dropped = np.zeros(
        n,
        dtype=bool,
    )

    for i, r in enumerate(rows):
        true = r["true_level"].strip()

        if true:
            if true not in LEVEL_TO_Y:
                raise RuntimeError(
                    f"Unknown true_level: {true}"
                )
            true_y[i] = LEVEL_TO_Y[true]

        if r["status"] == "dropped":
            dropped[i] = True
            continue

        observed = r[
            "observed_level"
        ].strip()

        if observed:
            if observed not in LEVEL_TO_Y:
                raise RuntimeError(
                    f"Unknown observed_level: "
                    f"{observed}"
                )
            observed_y[i] = (
                LEVEL_TO_Y[observed]
            )

        schedule = parse_schedule(
            r["schedule"]
        )

        if schedule is not None:
            schedule_matrix[:, i] = (
                np.asarray(
                    schedule,
                    dtype=float,
                )
            )

        forward[i] = parse_float(
            r["forward_ms"]
        )

        budget[i] = parse_float(
            r["initial_budget_ms"]
        )

        dm = r["deadline_miss"].strip()

        if dm:
            miss[i] = bool(int(dm))

    return {
        "x": x,
        "true_y": true_y,
        "observed_y": observed_y,
        "schedule": schedule_matrix,
        "forward": forward,
        "budget": budget,
        "miss": miss,
        "dropped": dropped,
    }


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
        "legend.fontsize": 7.2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.55,
    )


def add_dropped_background(ax, dropped):
    # Extremely light gray vertical band:
    # a sensor frame existed but was not processed.
    for i, flag in enumerate(dropped):
        if flag:
            ax.axvspan(
                i - 0.5,
                i + 0.5,
                color="0.93",
                linewidth=0,
                zorder=0,
            )


def plot_case_study(
    data,
    selected_rows,
    selected_scene,
    args,
    output_dir,
):
    configure_style()

    x = data["x"]
    n = len(x)

    fig = plt.figure(
        figsize=(7.2, 5.5)
    )

    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[
            1.05,
            1.75,
            1.35,
        ],
        hspace=0.27,
    )

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(
        gs[1],
        sharex=ax1,
    )
    ax3 = fig.add_subplot(
        gs[2],
        sharex=ax1,
    )

    # ============================================================
    # (a) True vs observed contention
    # ============================================================

    add_dropped_background(
        ax1,
        data["dropped"],
    )

    ax1.step(
        x,
        data["true_y"],
        where="mid",
        color="0.25",
        linewidth=1.4,
        label="True contention",
        zorder=2,
    )

    obs_mask = np.isfinite(
        data["observed_y"]
    )

    ax1.scatter(
        x[obs_mask],
        data["observed_y"][obs_mask],
        s=20,
        facecolors="none",
        edgecolors="red",
        linewidths=1.0,
        label="Observed contention",
        zorder=4,
    )

    ax1.set_yticks(
        [0, 1, 2, 3, 4]
    )

    ax1.set_yticklabels(
        ["L0", "L1", "L2", "L3", "L4"]
    )

    ax1.set_ylim(
        -0.4,
        4.4,
    )

    ax1.set_ylabel(
        "Contention"
    )

    ax1.set_title(
        "(a) Runtime contention observation",
        loc="left",
        fontweight="bold",
        pad=3,
    )

    ax1.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
    )

    style_axis(ax1)

    # ============================================================
    # (b) Stage-wise selected widths
    # ============================================================

    # White indicates an unprocessed/dropped sensor frame.
    masked = np.ma.masked_invalid(
        data["schedule"]
    )

    cmap = plt.get_cmap(
        "Blues"
    ).copy()

    cmap.set_bad(
        color="white"
    )

    im = ax2.imshow(
        masked,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=[
            -0.5,
            n - 0.5,
            5.5,
            -0.5,
        ],
        vmin=0.25,
        vmax=1.0,
        cmap=cmap,
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

    ax2.set_title(
        "(b) Controller-selected stage widths",
        loc="left",
        fontweight="bold",
        pad=3,
    )

    cbar = fig.colorbar(
        im,
        ax=ax2,
        fraction=0.022,
        pad=0.018,
    )

    cbar.set_label(
        "Width"
    )

    cbar.set_ticks(
        [0.25, 0.50, 0.75, 1.00]
    )

    # ============================================================
    # (c) Forward runtime / available budget
    # ============================================================

    add_dropped_background(
        ax3,
        data["dropped"],
    )

    proc = np.isfinite(
        data["forward"]
    )

    budget_mask = np.isfinite(
        data["budget"]
    )

    ax3.plot(
        x[budget_mask],
        data["budget"][budget_mask],
        color="0.35",
        linewidth=1.3,
        marker=None,
        label="Available forward budget",
        zorder=2,
    )

    ax3.plot(
        x[proc],
        data["forward"][proc],
        color="red",
        linewidth=1.7,
        marker="o",
        markersize=3.2,
        label="TV-streamDSGN forward latency",
        zorder=4,
    )

    miss_mask = (
        data["miss"]
        & proc
    )

    if np.any(miss_mask):
        ax3.scatter(
            x[miss_mask],
            data["forward"][miss_mask],
            s=36,
            marker="x",
            color="black",
            linewidths=1.2,
            label="Deadline miss",
            zorder=6,
        )

    ax3.set_ylabel(
        "Time (ms)"
    )

    ax3.set_xlabel(
        "Sensor frame within selected window"
    )

    ax3.set_title(
        "(c) Forward latency and available deadline budget",
        loc="left",
        fontweight="bold",
        pad=3,
    )

    ax3.legend(
        loc="best",
        frameon=False,
        ncol=2,
    )

    style_axis(ax3)

    # ------------------------------------------------------------
    # Shared x-axis
    # ------------------------------------------------------------

    for ax in [ax1, ax2, ax3]:
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

    # Use relative positions to keep the paper figure compact.
    tick_step = (
        10
        if n >= 40
        else 5
    )

    ticks = list(
        range(
            0,
            n,
            tick_step,
        )
    )

    if (n - 1) not in ticks:
        ticks.append(n - 1)

    ax3.set_xticks(
        ticks
    )

    ax3.set_xticklabels(
        [str(t) for t in ticks]
    )

    # ------------------------------------------------------------
    # Figure title
    # ------------------------------------------------------------

    local_start = (
        selected_rows[0]["local_pos"]
    )

    local_end = (
        selected_rows[-1]["local_pos"]
    )

    fig.suptitle(
        (
            "TV-streamDSGN runtime behavior under Random50 "
            f"(35 Hz, L0/{args.pressure_level}, "
            f"scene {selected_scene}, "
            f"frames {local_start}-{local_end})"
        ),
        fontsize=9,
        y=0.995,
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.95,
        bottom=0.085,
        top=0.945,
    )

    tag = (
        f"{args.hz}Hz_"
        f"L0_{args.pressure_level}_"
        f"p{frac_tag(args.pressure_fraction)}_"
        f"seed{args.trace_seed}"
    )

    stem = (
        output_dir
        / f"controller_runtime_case_study_{tag}"
    )

    fig.savefig(
        stem.with_suffix(".pdf"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".svg"),
        bbox_inches="tight",
    )

    fig.savefig(
        stem.with_suffix(".png"),
        dpi=args.dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    return stem


def write_selected_csv(
    rows,
    path,
):
    fields = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        w.writeheader()
        w.writerows(rows)


def print_summary(
    rows,
    selected_scene,
):
    processed = [
        r for r in rows
        if r["status"] == "processed"
    ]

    dropped = [
        r for r in rows
        if r["status"] == "dropped"
    ]

    misses = sum(
        int(r["deadline_miss"])
        for r in processed
        if r["deadline_miss"] != ""
    )

    schedules = [
        r["schedule"]
        for r in processed
        if r["schedule"]
    ]

    schedule_changes = sum(
        schedules[i] != schedules[i - 1]
        for i in range(
            1,
            len(schedules),
        )
    )

    true_switches = sum(
        rows[i]["true_level"]
        != rows[i - 1]["true_level"]
        for i in range(
            1,
            len(rows),
        )
    )

    correct_obs = 0
    observed_n = 0

    for r in processed:
        obs = r["observed_level"]

        if not obs:
            continue

        observed_n += 1

        if obs == r["true_level"]:
            correct_obs += 1

    print()
    print("=" * 90)
    print("CONTROLLER / RUNTIME CASE STUDY")
    print("=" * 90)
    print(f"scene              : {selected_scene}")
    print(
        f"local_pos          : "
        f"{rows[0]['local_pos']} .. "
        f"{rows[-1]['local_pos']}"
    )
    print(f"sensor frames      : {len(rows)}")
    print(f"processed          : {len(processed)}")
    print(f"dropped            : {len(dropped)}")
    print(f"deadline misses    : {misses}")
    print(f"true-level switches: {true_switches}")
    print(f"schedule changes   : {schedule_changes}")

    if observed_n:
        print(
            "observation accuracy: "
            f"{100 * correct_obs / observed_n:.2f}% "
            f"({correct_obs}/{observed_n})"
        )

    print("=" * 90)


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
            f"repo root does not exist: {repo}"
        )

    timeline = timeline_path(
        args,
        repo,
    )

    if not timeline.is_file():
        raise FileNotFoundError(
            "Missing formal TV-streamDSGN timeline:\n"
            f"  {timeline}"
        )

    print(
        f"[INPUT] {timeline}"
    )

    rows = load_timeline(
        timeline
    )

    selected_scene, selected_rows = (
        select_window(
            rows=rows,
            scene=args.scene,
            start=args.start,
            window=args.window,
        )
    )

    print_summary(
        selected_rows,
        selected_scene,
    )

    data = prepare_arrays(
        selected_rows
    )

    output_dir = (
        repo
        / args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected_csv = (
        output_dir
        / (
            "controller_runtime_case_study_"
            "selected_window.csv"
        )
    )

    write_selected_csv(
        selected_rows,
        selected_csv,
    )

    stem = plot_case_study(
        data=data,
        selected_rows=selected_rows,
        selected_scene=selected_scene,
        args=args,
        output_dir=output_dir,
    )

    print()
    print("[DONE]")
    print(f"WINDOW CSV : {selected_csv}")
    print(
        f"PDF        : "
        f"{stem.with_suffix('.pdf')}"
    )
    print(
        f"SVG        : "
        f"{stem.with_suffix('.svg')}"
    )
    print(
        f"PNG        : "
        f"{stem.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
