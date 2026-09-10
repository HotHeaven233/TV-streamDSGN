#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


PRESSURE_LEVELS = ["L1", "L2", "L3", "L4"]

# 固定颜色，确保不同 figure 中方法颜色一致。
METHODS = [
    {
        "name": "Original StreamDSGN",
        "short": "StreamDSGN",
        "marker": "o",
        "color": "#1f77b4",
    },
    {
        "name": "Streamer-style StreamDSGN",
        "short": "Streamer-style",
        "marker": "s",
        "color": "#ff7f0e",
    },
    {
        "name": "MTD Three-Head",
        "short": "MTD",
        "marker": "^",
        "color": "#2ca02c",
    },
    {
        "name": "Transtreaming-style",
        "short": "Transtreaming-style",
        "marker": "D",
        "color": "purple",
    },
    {
        "name": "TV-streamDSGN",
        "short": "TV-streamDSGN",
        "marker": "*",
        "color": "#d62728",
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot formal Random50 pressure robustness accuracy "
            "(Car / Pedestrian / Cyclist / Macro)."
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
        "--output_dir",
        default="outputs/paper_figures/random50_pressure_robustness_accuracy",
    )

    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )

    return p.parse_args()


def fraction_tag(value: float) -> str:
    # 0.5 -> "0.5"，与现有输出目录命名一致。
    return f"{value:g}"


def summary_path(
    repo: Path,
    exp_name: str,
    method: str,
    hz: int,
    seed: int,
    fraction: float,
    pressure: str,
) -> Path:

    frac = fraction_tag(fraction)

    suffix = (
        f"{hz}Hz_forward_only_seed{seed}"
    )

    leaf = f"L0_{pressure}_p{frac}"

    if method == "Original StreamDSGN":
        return (
            repo
            / "outputs/original_streamdsgn"
            / "formal_streaming_random50"
            / suffix
            / leaf
            / "summary.json"
        )

    if method == "Streamer-style StreamDSGN":
        return (
            repo
            / "outputs/streamer_style_streamdsgn"
            / "formal_streaming_random50"
            / suffix
            / leaf
            / "summary.json"
        )

    if method == "MTD Three-Head":
        return (
            repo
            / "outputs/mtd_three_head"
            / "formal_streaming_random50"
            / suffix
            / leaf
            / "summary.json"
        )

    if method == "Transtreaming-style":
        return (
            repo
            / "outputs/transtreaming_v2_best"
            / "formal_streaming_random50"
            / suffix
            / leaf
            / "summary.json"
        )

    if method == "TV-streamDSGN":
        return (
            repo
            / "outputs/elastic_bev"
            / exp_name
            / "formal_streaming_random50"
            / suffix
            / leaf
            / "summary.json"
        )

    raise KeyError(method)


def require_number(obj, key, source):
    if key not in obj:
        raise KeyError(
            f"{source}: missing required key {key!r}"
        )

    value = obj[key]

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"{source}: {key!r} is not numeric: {value!r}"
        )

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(
            f"{source}: {key!r} is not finite: {value}"
        )

    return value


def validate_random50_metadata(
    data,
    path,
    expected_hz,
    expected_pressure,
    expected_fraction,
    expected_seed,
):
    if "input_hz" in data:
        actual = float(data["input_hz"])
        if abs(actual - expected_hz) > 1e-6:
            raise RuntimeError(
                f"{path}: input_hz={actual}, "
                f"expected {expected_hz}"
            )

    if data.get("no_load") is True:
        raise RuntimeError(
            f"{path}: this is marked as no_load=True"
        )

    trace = data.get("contention_trace")

    # 不同 evaluator 的 metadata 可能略有差异，
    # 因此只有字段存在时才严格核验。
    if isinstance(trace, dict):

        if "pressure_level" in trace:
            if trace["pressure_level"] != expected_pressure:
                raise RuntimeError(
                    f"{path}: pressure_level="
                    f"{trace['pressure_level']}, "
                    f"expected {expected_pressure}"
                )

        if "pressure_fraction_target" in trace:
            actual = float(
                trace["pressure_fraction_target"]
            )

            if abs(actual - expected_fraction) > 1e-9:
                raise RuntimeError(
                    f"{path}: pressure_fraction_target="
                    f"{actual}, expected {expected_fraction}"
                )

        if "trace_seed" in trace:
            actual = int(trace["trace_seed"])

            if actual != expected_seed:
                raise RuntimeError(
                    f"{path}: trace_seed={actual}, "
                    f"expected {expected_seed}"
                )


def load_one(
    path,
    method,
    pressure,
    hz,
    fraction,
    seed,
):
    data = json.loads(
        path.read_text()
    )

    validate_random50_metadata(
        data=data,
        path=path,
        expected_hz=hz,
        expected_pressure=pressure,
        expected_fraction=fraction,
        expected_seed=seed,
    )

    q = data.get(
        "stream_sap_3d_moderate_R40"
    )

    if not isinstance(q, dict):
        raise KeyError(
            f"{path}: missing "
            "'stream_sap_3d_moderate_R40'"
        )

    return {
        "method": method,
        "pressure_level": pressure,
        "Hz": hz,
        "pressure_fraction": fraction,
        "trace_seed": seed,

        "Car_3d_mod_R40":
            require_number(q, "Car", path),

        "Pedestrian_3d_mod_R40":
            require_number(q, "Pedestrian", path),

        "Cyclist_3d_mod_R40":
            require_number(q, "Cyclist", path),

        "Macro_3d_mod_R40":
            require_number(q, "Macro", path),

        "source": str(path),
    }


def collect(
    repo,
    exp_name,
    hz,
    fraction,
    seed,
):
    rows = []

    for spec in METHODS:
        method = spec["name"]

        for pressure in PRESSURE_LEVELS:

            path = summary_path(
                repo=repo,
                exp_name=exp_name,
                method=method,
                hz=hz,
                seed=seed,
                fraction=fraction,
                pressure=pressure,
            )

            if not path.is_file():
                raise FileNotFoundError(
                    "Missing formal Random50 result:\n"
                    f"  method   = {method}\n"
                    f"  pressure = {pressure}\n"
                    f"  Hz       = {hz}\n"
                    f"  fraction = {fraction}\n"
                    f"  seed     = {seed}\n"
                    f"  path     = {path}"
                )

            rows.append(
                load_one(
                    path=path,
                    method=method,
                    pressure=pressure,
                    hz=hz,
                    fraction=fraction,
                    seed=seed,
                )
            )

    return rows


def write_csv(rows, path):
    fields = [
        "method",
        "pressure_level",
        "Hz",
        "pressure_fraction",
        "trace_seed",
        "Car_3d_mod_R40",
        "Pedestrian_3d_mod_R40",
        "Cyclist_3d_mod_R40",
        "Macro_3d_mod_R40",
        "source",
    ]

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib():
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

        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,

        # 尽量保留矢量文字
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def rows_for_method(rows, method):
    order = {
        level: i
        for i, level
        in enumerate(PRESSURE_LEVELS)
    }

    selected = [
        r
        for r in rows
        if r["method"] == method
    ]

    selected.sort(
        key=lambda r: order[
            r["pressure_level"]
        ]
    )

    return selected


def add_common_axis_style(ax):
    x = list(
        range(len(PRESSURE_LEVELS))
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        PRESSURE_LEVELS
    )

    ax.set_xlim(
        -0.15,
        len(PRESSURE_LEVELS) - 0.85,
    )

    ax.set_xlabel(
        "Pressure level"
    )

    ax.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)


def plot_metric(
    ax,
    rows,
    key,
    title,
    panel,
):
    x = list(
        range(len(PRESSURE_LEVELS))
    )

    for spec in METHODS:

        mr = rows_for_method(
            rows,
            spec["name"],
        )

        if len(mr) != len(PRESSURE_LEVELS):
            raise RuntimeError(
                f"{spec['name']}: expected "
                f"{len(PRESSURE_LEVELS)} rows, "
                f"got {len(mr)}"
            )

        y = [
            r[key]
            for r in mr
        ]

        is_ours = (
            spec["name"]
            == "TV-streamDSGN"
        )

        ax.plot(
            x,
            y,

            # 全部实线
            linestyle="-",

            marker=spec["marker"],
            color=spec["color"],

            linewidth=(
                2.4
                if is_ours
                else 1.6
            ),

            markersize=(
                8.0
                if is_ours
                else 4.8
            ),

            label=spec["short"],

            zorder=(
                10
                if is_ours
                else 3
            ),
        )

    ax.set_title(
        f"({panel}) {title}",
        loc="left",
        fontweight="bold",
        pad=4,
    )

    ax.set_ylabel(
        r"Streaming AP$_{R40}$"
    )

    add_common_axis_style(ax)


def print_table(rows):
    print()
    print("=" * 122)
    print(
        "FORMAL RANDOM50 PRESSURE ROBUSTNESS ACCURACY"
    )
    print("=" * 122)

    for pressure in PRESSURE_LEVELS:

        print(
            f"\nL0/{pressure} Random50"
        )

        for spec in METHODS:

            hit = [
                r
                for r in rows
                if (
                    r["method"]
                    == spec["name"]
                    and
                    r["pressure_level"]
                    == pressure
                )
            ]

            if len(hit) != 1:
                raise RuntimeError(
                    f"unexpected row count: "
                    f"{spec['name']} "
                    f"{pressure}: {len(hit)}"
                )

            r = hit[0]

            print(
                f"  {spec['short']:20s} | "
                f"Car={r['Car_3d_mod_R40']:7.4f} | "
                f"Ped={r['Pedestrian_3d_mod_R40']:7.4f} | "
                f"Cyc={r['Cyclist_3d_mod_R40']:7.4f} | "
                f"Macro={r['Macro_3d_mod_R40']:7.4f}"
            )

    print()
    print("=" * 122)


def plot_figure(
    rows,
    output_dir,
    hz,
    fraction,
    seed,
    dpi,
):
    configure_matplotlib()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.35),
        constrained_layout=False,
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    plot_metric(
        ax_a,
        rows,
        "Car_3d_mod_R40",
        "Car",
        "a",
    )

    plot_metric(
        ax_b,
        rows,
        "Pedestrian_3d_mod_R40",
        "Pedestrian",
        "b",
    )

    plot_metric(
        ax_c,
        rows,
        "Cyclist_3d_mod_R40",
        "Cyclist",
        "c",
    )

    plot_metric(
        ax_d,
        rows,
        "Macro_3d_mod_R40",
        "Macro",
        "d",
    )

    handles, labels = (
        ax_a.get_legend_handles_labels()
    )

    fig.legend(
        handles,
        labels,

        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),

        ncol=5,
        frameon=False,

        columnspacing=1.1,
        handlelength=2.4,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.09,
        top=0.88,
        wspace=0.28,
        hspace=0.36,
    )

    frac = fraction_tag(
        fraction
    )

    stem = (
        output_dir
        /
        (
            "random50_pressure_robustness_accuracy_"
            f"{hz}Hz_p{frac}_seed{seed}"
        )
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
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    return stem


def main():
    args = parse_args()

    repo = Path(
        args.repo_root
    ).resolve()

    if not repo.is_dir():
        raise FileNotFoundError(
            f"repo root does not exist: {repo}"
        )

    output_dir = (
        repo
        / args.output_dir
    ).resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = collect(
        repo=repo,
        exp_name=args.exp_name,
        hz=args.hz,
        fraction=args.pressure_fraction,
        seed=args.trace_seed,
    )

    print_table(
        rows
    )

    frac = fraction_tag(
        args.pressure_fraction
    )

    csv_path = (
        output_dir
        /
        (
            "random50_pressure_robustness_accuracy_data_"
            f"{args.hz}Hz_p{frac}_seed{args.trace_seed}.csv"
        )
    )

    write_csv(
        rows,
        csv_path,
    )

    stem = plot_figure(
        rows=rows,
        output_dir=output_dir,
        hz=args.hz,
        fraction=args.pressure_fraction,
        seed=args.trace_seed,
        dpi=args.dpi,
    )

    print()
    print("[DONE]")
    print(f"CSV : {csv_path}")
    print(
        f"PDF : {stem.with_suffix('.pdf')}"
    )
    print(
        f"SVG : {stem.with_suffix('.svg')}"
    )
    print(
        f"PNG : {stem.with_suffix('.png')}"
    )


if __name__ == "__main__":
    main()
