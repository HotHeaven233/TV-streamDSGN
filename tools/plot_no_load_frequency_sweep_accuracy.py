#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


FREQUENCIES = [35, 40, 45, 50]

METHODS = [
    {
        "name": "Original StreamDSGN",
        "short": "StreamDSGN",
        "marker": "o",
        "color": None,
    },
    {
        "name": "Streamer-style StreamDSGN",
        "short": "Streamer-style",
        "marker": "s",
        "color": None,
    },
    {
        "name": "MTD Three-Head",
        "short": "MTD",
        "marker": "^",
        "color": None,
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
        "color": "red",
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot formal no-load 35/40/45/50-Hz frequency sweep "
            "accuracy curves (Car / Pedestrian / Cyclist / Macro)."
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
        "--output_dir",
        default="outputs/paper_figures/frequency_sweep_accuracy",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )
    return p.parse_args()


def summary_path(repo, exp_name, method, hz):
    if method == "Original StreamDSGN":
        return (
            repo
            / "outputs/original_streamdsgn"
            / "formal_streaming_no_load"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    if method == "Streamer-style StreamDSGN":
        return (
            repo
            / "outputs/streamer_style_streamdsgn"
            / "formal_streaming_no_load"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    if method == "MTD Three-Head":
        return (
            repo
            / "outputs/mtd_three_head"
            / "formal_streaming_no_load"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    if method == "Transtreaming-style":
        return (
            repo
            / "outputs/transtreaming_v2_best"
            / "formal_streaming_no_load"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    if method == "TV-streamDSGN":
        return (
            repo
            / "outputs/elastic_bev"
            / exp_name
            / "formal_streaming"
            / f"{hz}Hz_forward_only"
            / "L0"
            / "summary.json"
        )

    raise KeyError(method)


def require_number(obj, key, source):
    if key not in obj:
        raise KeyError(f"{source}: missing required key {key!r}")

    value = obj[key]

    if not isinstance(value, (int, float)):
        raise TypeError(f"{source}: {key!r} is not numeric: {value!r}")

    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{source}: {key!r} is not finite: {value}")

    return value


def load_one(path, expected_hz, method):
    data = json.loads(path.read_text())

    if "input_hz" in data:
        actual_hz = float(data["input_hz"])
        if abs(actual_hz - expected_hz) > 1e-6:
            raise RuntimeError(
                f"{path}: input_hz={actual_hz}, expected {expected_hz}"
            )

    if "no_load" in data and not bool(data["no_load"]):
        raise RuntimeError(
            f"{path}: expected no-load result, but no_load={data['no_load']!r}"
        )

    q = data.get("stream_sap_3d_moderate_R40")
    if not isinstance(q, dict):
        raise KeyError(f"{path}: missing 'stream_sap_3d_moderate_R40'")

    return {
        "method": method,
        "Hz": float(expected_hz),
        "Car_3d_mod_R40": require_number(q, "Car", path),
        "Pedestrian_3d_mod_R40": require_number(q, "Pedestrian", path),
        "Cyclist_3d_mod_R40": require_number(q, "Cyclist", path),
        "Macro_3d_mod_R40": require_number(q, "Macro", path),
        "source": str(path),
    }


def collect(repo, exp_name):
    rows = []

    for spec in METHODS:
        method = spec["name"]

        for hz in FREQUENCIES:
            path = summary_path(repo, exp_name, method, hz)

            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing formal result:\n"
                    f"  method = {method}\n"
                    f"  Hz     = {hz}\n"
                    f"  path   = {path}"
                )

            rows.append(load_one(path, hz, method))

    return rows


def rows_for_method(rows, method):
    out = [r for r in rows if r["method"] == method]
    out.sort(key=lambda x: x["Hz"])
    return out


def write_csv(rows, path):
    fields = [
        "method",
        "Hz",
        "Car_3d_mod_R40",
        "Pedestrian_3d_mod_R40",
        "Cyclist_3d_mod_R40",
        "Macro_3d_mod_R40",
        "source",
    ]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def configure_matplotlib():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.8,
        "lines.markersize": 5.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_common_axis_style(ax):
    ax.set_xticks(FREQUENCIES)
    ax.set_xlim(min(FREQUENCIES) - 1, max(FREQUENCIES) + 1)
    ax.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Input frequency (Hz)")


def plot_metric(ax, rows, key, title_text, panel):
    for spec in METHODS:
        mr = rows_for_method(rows, spec["name"])
        x = [r["Hz"] for r in mr]
        y = [r[key] for r in mr]

        lw = 2.4 if spec["name"] == "TV-streamDSGN" else 1.6
        ms = 8.0 if spec["name"] == "TV-streamDSGN" else 4.8

        plot_kwargs = dict(
            linestyle="-",
            marker=spec["marker"],
            linewidth=lw,
            markersize=ms,
            label=spec["short"],
            zorder=10 if spec["name"] == "TV-streamDSGN" else 3,
        )

        if spec["color"] is not None:
            plot_kwargs["color"] = spec["color"]

        ax.plot(
            x,
            y,
            **plot_kwargs,
        )

    ax.set_title(
        f"({panel}) {title_text}",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    ax.set_ylabel(r"Streaming AP$_{R40}$")
    add_common_axis_style(ax)


def plot_accuracy_figure(rows, output_dir, dpi):
    configure_matplotlib()

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.15, 5.35),
        constrained_layout=False,
    )

    ax_a, ax_b = axes[0]
    ax_c, ax_d = axes[1]

    plot_metric(ax_a, rows, "Car_3d_mod_R40", "Car", "a")
    plot_metric(ax_b, rows, "Pedestrian_3d_mod_R40", "Pedestrian", "b")
    plot_metric(ax_c, rows, "Cyclist_3d_mod_R40", "Cyclist", "c")
    plot_metric(ax_d, rows, "Macro_3d_mod_R40", "Macro", "d")

    handles, labels = ax_a.get_legend_handles_labels()
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

    stem = output_dir / "frequency_sweep_accuracy_no_load_35_40_45_50"

    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return stem


def print_table(rows):
    print()
    print("=" * 120)
    print("FORMAL NO-LOAD FREQUENCY SWEEP ACCURACY")
    print("=" * 120)

    for hz in FREQUENCIES:
        print(f"\n{hz} Hz")
        for spec in METHODS:
            hit = [
                r for r in rows
                if r["method"] == spec["name"] and abs(r["Hz"] - hz) < 1e-9
            ]
            r = hit[0]
            print(
                f"  {spec['short']:20s} | "
                f"Car={r['Car_3d_mod_R40']:7.4f} | "
                f"Ped={r['Pedestrian_3d_mod_R40']:7.4f} | "
                f"Cyc={r['Cyclist_3d_mod_R40']:7.4f} | "
                f"Macro={r['Macro_3d_mod_R40']:7.4f}"
            )

    print()
    print("=" * 120)


def main():
    args = parse_args()

    repo = Path(args.repo_root).resolve()
    if not repo.is_dir():
        raise FileNotFoundError(f"repo root does not exist: {repo}")

    output_dir = (repo / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect(repo, args.exp_name)
    print_table(rows)

    csv_path = output_dir / "frequency_sweep_accuracy_no_load_data.csv"
    write_csv(rows, csv_path)

    stem = plot_accuracy_figure(rows, output_dir, args.dpi)

    print()
    print("[DONE]")
    print(f"CSV : {csv_path}")
    print(f"PDF : {stem.with_suffix('.pdf')}")
    print(f"SVG : {stem.with_suffix('.svg')}")
    print(f"PNG : {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
