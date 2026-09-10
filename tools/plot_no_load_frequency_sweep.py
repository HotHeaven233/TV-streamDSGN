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
        "linestyle": "--",
    },
    {
        "name": "Streamer-style StreamDSGN",
        "short": "Streamer-style",
        "marker": "s",
        "linestyle": "-.",
    },
    {
        "name": "MTD Three-Head",
        "short": "MTD",
        "marker": "^",
        "linestyle": ":",
    },
    {
        "name": "Transtreaming-style",
        "short": "Transtreaming-style",
        "marker": "D",
        "linestyle": "--",
    },
    {
        "name": "TV-Stream3D",
        "short": "TV-Stream3D",
        "marker": "*",
        "linestyle": "-",
    },
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot formal no-load 35/40/45/50-Hz frequency sweep "
            "for StreamDSGN baselines and TV-Stream3D."
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
        default="outputs/paper_figures/frequency_sweep",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=400,
    )
    p.add_argument(
        "--allow_missing",
        action="store_true",
        help=(
            "Allow missing method/frequency summaries. "
            "Formal paper plotting should normally NOT use this."
        ),
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

    if method == "TV-Stream3D":
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


def load_one(path, expected_hz, method):
    data = json.loads(path.read_text())

    if "input_hz" in data:
        actual_hz = float(data["input_hz"])
        if abs(actual_hz - expected_hz) > 1e-6:
            raise RuntimeError(
                f"{path}: input_hz={actual_hz}, "
                f"expected {expected_hz}"
            )

    if "no_load" in data and not bool(data["no_load"]):
        raise RuntimeError(
            f"{path}: expected formal no-load result, "
            f"but no_load={data['no_load']!r}"
        )

    q = data.get(
        "stream_sap_3d_moderate_R40"
    )
    if not isinstance(q, dict):
        raise KeyError(
            f"{path}: missing "
            "'stream_sap_3d_moderate_R40'"
        )

    latency = data.get("forward_latency")
    if not isinstance(latency, dict):
        raise KeyError(
            f"{path}: missing 'forward_latency'"
        )

    macro = require_number(
        q,
        "Macro",
        path,
    )

    car = require_number(
        q,
        "Car",
        path,
    )

    ped = require_number(
        q,
        "Pedestrian",
        path,
    )

    cyc = require_number(
        q,
        "Cyclist",
        path,
    )

    miss_rate = require_number(
        data,
        "deadline_miss_rate",
        path,
    )

    drop_rate = require_number(
        data,
        "drop_rate",
        path,
    )

    p50 = require_number(
        latency,
        "p50_ms",
        path,
    )

    p90 = require_number(
        latency,
        "p90_ms",
        path,
    )

    p99 = require_number(
        latency,
        "p99_ms",
        path,
    )

    sensor_frames = int(
        data.get("sensor_frames", -1)
    )
    processed_frames = int(
        data.get("processed_frames", -1)
    )
    dropped_frames = int(
        data.get("dropped_frames", -1)
    )

    if sensor_frames >= 0:
        if processed_frames < 0 or dropped_frames < 0:
            raise RuntimeError(
                f"{path}: incomplete frame accounting"
            )

        if processed_frames + dropped_frames != sensor_frames:
            raise RuntimeError(
                f"{path}: frame accounting mismatch: "
                f"processed={processed_frames}, "
                f"dropped={dropped_frames}, "
                f"sensor={sensor_frames}"
            )

    return {
        "method": method,
        "Hz": float(expected_hz),
        "period_ms": 1000.0 / float(expected_hz),
        "sensor_frames": sensor_frames,
        "processed_frames": processed_frames,
        "dropped_frames": dropped_frames,
        "drop_rate": drop_rate,
        "deadline_miss_rate": miss_rate,
        "forward_p50_ms": p50,
        "forward_p90_ms": p90,
        "forward_p99_ms": p99,
        "Car_3d_mod_R40": car,
        "Pedestrian_3d_mod_R40": ped,
        "Cyclist_3d_mod_R40": cyc,
        "Macro_3d_mod_R40": macro,
        "source": str(path),
    }


def collect(repo, exp_name, allow_missing):
    rows = []

    missing = []

    for m in METHODS:
        method = m["name"]

        for hz in FREQUENCIES:
            path = summary_path(
                repo,
                exp_name,
                method,
                hz,
            )

            if not path.is_file():
                missing.append(
                    (method, hz, path)
                )

                if allow_missing:
                    continue

                raise FileNotFoundError(
                    f"Missing formal result:\n"
                    f"  method = {method}\n"
                    f"  Hz     = {hz}\n"
                    f"  path   = {path}"
                )

            rows.append(
                load_one(
                    path,
                    hz,
                    method,
                )
            )

    if missing:
        print()
        print("WARNING: missing results")
        for method, hz, path in missing:
            print(
                f"  {method:26s} "
                f"{hz:2d} Hz  {path}"
            )

    return rows


def write_csv(rows, path):
    if not rows:
        raise RuntimeError("No rows to write")

    fields = [
        "method",
        "Hz",
        "period_ms",
        "sensor_frames",
        "processed_frames",
        "dropped_frames",
        "drop_rate",
        "deadline_miss_rate",
        "forward_p50_ms",
        "forward_p90_ms",
        "forward_p99_ms",
        "Car_3d_mod_R40",
        "Pedestrian_3d_mod_R40",
        "Cyclist_3d_mod_R40",
        "Macro_3d_mod_R40",
        "source",
    ]

    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(rows)


def rows_for_method(rows, method):
    out = [
        r for r in rows
        if r["method"] == method
    ]
    out.sort(key=lambda x: x["Hz"])
    return out


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
        "legend.fontsize": 7.4,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def add_common_axis_style(ax):
    ax.set_xticks(FREQUENCIES)
    ax.set_xlim(
        min(FREQUENCIES) - 1,
        max(FREQUENCIES) + 1,
    )
    ax.grid(
        True,
        axis="y",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric(
    ax,
    rows,
    key,
    ylabel,
    panel,
    percent=False,
):
    for i, spec in enumerate(METHODS):
        mr = rows_for_method(
            rows,
            spec["name"],
        )

        if not mr:
            continue

        x = [r["Hz"] for r in mr]
        y = [r[key] for r in mr]

        if percent:
            y = [100.0 * v for v in y]

        lw = (
            2.2
            if spec["name"] == "TV-Stream3D"
            else 1.4
        )

        ms = (
            7.0
            if spec["name"] == "TV-Stream3D"
            else 4.8
        )

        ax.plot(
            x,
            y,
            marker=spec["marker"],
            linestyle=spec["linestyle"],
            linewidth=lw,
            markersize=ms,
            label=spec["short"],
            zorder=10 if spec["name"] == "TV-Stream3D" else 3,
        )

    ax.set_title(
        f"({panel}) {ylabel}",
        loc="left",
        fontweight="bold",
        pad=4,
    )
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Input frequency (Hz)")
    add_common_axis_style(ax)


def plot_frequency_sweep(rows, output_dir, dpi):
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
        "Macro_3d_mod_R40",
        r"Streaming Macro AP$_{R40}$",
        "a",
        percent=False,
    )

    plot_metric(
        ax_b,
        rows,
        "deadline_miss_rate",
        "Deadline miss rate (%)",
        "b",
        percent=True,
    )

    plot_metric(
        ax_c,
        rows,
        "drop_rate",
        "Drop rate (%)",
        "c",
        percent=True,
    )

    plot_metric(
        ax_d,
        rows,
        "forward_p99_ms",
        "Forward p99 latency (ms)",
        "d",
        percent=False,
    )

    # Period/deadline line in the p99 panel.
    periods = [
        1000.0 / hz
        for hz in FREQUENCIES
    ]

    ax_d.plot(
        FREQUENCIES,
        periods,
        linestyle=(0, (3, 2)),
        linewidth=1.2,
        marker=None,
        label=r"Deadline $T=1/f$",
        zorder=2,
    )

    # Rates are percentages by definition.
    ax_b.set_ylim(0, 105)
    ax_c.set_ylim(0, 105)

    handles, labels = ax_a.get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
        columnspacing=1.2,
        handlelength=2.5,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.09,
        top=0.88,
        wspace=0.28,
        hspace=0.38,
    )

    stem = output_dir / "frequency_sweep_no_load_35_40_45_50"

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


def print_table(rows):
    print()
    print("=" * 120)
    print("FORMAL NO-LOAD FREQUENCY SWEEP")
    print("=" * 120)

    for hz in FREQUENCIES:
        print(
            f"\n{hz} Hz "
            f"(deadline={1000.0/hz:.3f} ms)"
        )

        for spec in METHODS:
            hit = [
                r for r in rows
                if (
                    r["method"] == spec["name"]
                    and
                    abs(r["Hz"] - hz) < 1e-9
                )
            ]

            if not hit:
                print(
                    f"  {spec['short']:20s} | MISSING"
                )
                continue

            r = hit[0]

            print(
                f"  {spec['short']:20s} | "
                f"Macro={r['Macro_3d_mod_R40']:7.4f} | "
                f"miss={100*r['deadline_miss_rate']:7.3f}% | "
                f"drop={100*r['drop_rate']:7.3f}% | "
                f"p99={r['forward_p99_ms']:7.3f} ms"
            )

    print()
    print("=" * 120)


def main():
    args = parse_args()

    repo = Path(args.repo_root).resolve()

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
        repo,
        args.exp_name,
        args.allow_missing,
    )

    if not rows:
        raise RuntimeError(
            "No frequency-sweep results found"
        )

    print_table(rows)

    csv_path = (
        output_dir
        / "frequency_sweep_no_load_data.csv"
    )

    write_csv(
        rows,
        csv_path,
    )

    stem = plot_frequency_sweep(
        rows,
        output_dir,
        args.dpi,
    )

    print()
    print("[DONE]")
    print(f"CSV : {csv_path}")
    print(f"PDF : {stem.with_suffix('.pdf')}")
    print(f"SVG : {stem.with_suffix('.svg')}")
    print(f"PNG : {stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
