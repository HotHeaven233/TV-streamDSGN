#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.table import Table


LEVELS = ["L2", "L3", "L4"]

POLICIES = [
    ("Full-width", "full"),
    ("Best Static", "best_static"),
    ("TV Dynamic", "tv_dynamic"),
    ("Oracle-Level (Upper Bound)", "oracle_level"),
]

DEPLOYABLE_POLICIES = {
    "full",
    "best_static",
    "tv_dynamic",
}


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--repo_root",
        default="/data/jhb/workspace/streamDSGN",
    )
    p.add_argument(
        "--tv_exp_name",
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
        default=(
            "outputs/paper_figures/"
            "tv_policy_ablation_table"
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
    exp_name: str,
    hz: int,
    seed: int,
    level: str,
    policy: str,
):
    return (
        repo
        / "outputs"
        / "elastic_bev"
        / exp_name
        / "system_ablation"
        / "static_vs_dynamic"
        / f"{hz}Hz_seed{seed}"
        / level
        / policy
        / "summary.json"
    )


def read_summary(
    path: Path,
    expected_policy: str,
    expected_level: str,
    expected_hz: int,
    expected_seed: int,
    expected_fraction: float,
):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing summary:\n  {path}"
        )

    with path.open() as f:
        s = json.load(f)

    # ------------------------------------------------------------
    # Validate formal experiment metadata.
    # ------------------------------------------------------------

    if s.get("policy") != expected_policy:
        raise RuntimeError(
            f"{path}: policy mismatch: "
            f"{s.get('policy')} != {expected_policy}"
        )

    if s.get("pressure_level") != expected_level:
        raise RuntimeError(
            f"{path}: level mismatch: "
            f"{s.get('pressure_level')} != {expected_level}"
        )

    if abs(
        float(s.get("input_hz"))
        - float(expected_hz)
    ) > 1e-8:
        raise RuntimeError(
            f"{path}: input_hz mismatch"
        )

    if int(
        s.get("trace_seed")
    ) != int(expected_seed):
        raise RuntimeError(
            f"{path}: trace_seed mismatch"
        )

    if abs(
        float(s.get("pressure_fraction"))
        - float(expected_fraction)
    ) > 1e-8:
        raise RuntimeError(
            f"{path}: pressure_fraction mismatch"
        )

    q = s.get(
        "stream_sap_3d_moderate_R40"
    )

    if q is None or "Macro" not in q:
        raise KeyError(
            f"{path}: missing Macro AP"
        )

    if "deadline_miss_rate" not in s:
        raise KeyError(
            f"{path}: missing deadline_miss_rate"
        )

    return {
        "macro": float(
            q["Macro"]
        ),

        # summary stores misses / processed.
        "miss_pct": (
            100.0
            * float(
                s["deadline_miss_rate"]
            )
        ),
    }


def collect(
    args,
    repo: Path,
):
    rows = []

    for display_name, policy in POLICIES:
        row = {
            "Policy": display_name,
            "policy_key": policy,
        }

        for level in LEVELS:
            path = summary_path(
                repo=repo,
                exp_name=args.tv_exp_name,
                hz=args.hz,
                seed=args.trace_seed,
                level=level,
                policy=policy,
            )

            result = read_summary(
                path=path,
                expected_policy=policy,
                expected_level=level,
                expected_hz=args.hz,
                expected_seed=args.trace_seed,
                expected_fraction=args.pressure_fraction,
            )

            row[
                f"{level}_Macro"
            ] = result["macro"]

            row[
                f"{level}_MissPct"
            ] = result["miss_pct"]

            print(
                f"[READ] "
                f"{display_name:<28s} "
                f"L0/{level} | "
                f"Macro={result['macro']:8.4f} | "
                f"Miss={result['miss_pct']:8.3f}% | "
                f"{path}"
            )

        rows.append(row)

    return rows


def write_csv(
    rows,
    path: Path,
):
    fields = [
        "Policy",
    ]

    for level in LEVELS:
        fields.extend([
            f"{level}_Macro",
            f"{level}_MissPct",
        ])

    with path.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        w.writeheader()
        w.writerows(rows)


def deployable_best(rows):
    """
    Oracle-Level is an upper bound and is deliberately excluded
    from 'best deployable policy' highlighting.
    """
    best = {}

    for level in LEVELS:
        candidates = [
            (i, r)
            for i, r in enumerate(rows)
            if r["policy_key"]
            in DEPLOYABLE_POLICIES
        ]

        best_macro = max(
            float(
                r[f"{level}_Macro"]
            )
            for _, r in candidates
        )

        best_miss = min(
            float(
                r[f"{level}_MissPct"]
            )
            for _, r in candidates
        )

        best[level] = {
            "macro": {
                i
                for i, r in candidates
                if abs(
                    float(
                        r[f"{level}_Macro"]
                    )
                    - best_macro
                ) < 1e-8
            },

            "miss": {
                i
                for i, r in candidates
                if abs(
                    float(
                        r[f"{level}_MissPct"]
                    )
                    - best_miss
                ) < 1e-8
            },
        }

    return best


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
    args,
):
    configure_style()

    best = deployable_best(
        rows
    )

    fig, ax = plt.subplots(
        figsize=(10.8, 3.35)
    )

    ax.set_axis_off()

    table = Table(
        ax,
        bbox=[
            0.02,
            0.09,
            0.96,
            0.73,
        ],
    )

    total_rows = (
        2 + len(rows)
    )

    row_h = (
        1.0 / total_rows
    )

    policy_w = 0.28

    metric_w = (
        1.0 - policy_w
    ) / 6.0

    edge = "#555555"
    header_face = "#eeeeee"
    subheader_face = "#f7f7f7"
    tv_face = "#fafafa"
    oracle_face = "#eeeeee"

    # ============================================================
    # Main header
    # ============================================================

    c = table.add_cell(
        0,
        0,
        width=policy_w,
        height=row_h * 2,
        text="Policy",
        loc="center",
        facecolor=header_face,
        edgecolor=edge,
    )
    c.get_text().set_weight(
        "bold"
    )

    for j, level in enumerate(
        LEVELS
    ):
        col = (
            1 + 2 * j
        )

        c = table.add_cell(
            0,
            col,
            width=metric_w * 2,
            height=row_h,
            text=f"L0 / {level}",
            loc="center",
            facecolor=header_face,
            edgecolor=edge,
        )

        c.get_text().set_weight(
            "bold"
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

    # ============================================================
    # Body
    # ============================================================

    for body_i, row in enumerate(
        rows
    ):
        tr = body_i + 2

        policy_key = row[
            "policy_key"
        ]

        is_tv = (
            policy_key
            == "tv_dynamic"
        )

        is_oracle = (
            policy_key
            == "oracle_level"
        )

        if is_oracle:
            face = oracle_face
        elif is_tv:
            face = tv_face
        else:
            face = "white"

        c = table.add_cell(
            tr,
            0,
            width=policy_w,
            height=row_h,
            text=row["Policy"],
            loc="left",
            facecolor=face,
            edgecolor=edge,
        )

        c.PAD = 0.08

        if is_tv:
            c.get_text().set_weight(
                "bold"
            )

        if is_oracle:
            c.get_text().set_style(
                "italic"
            )

        for j, level in enumerate(
            LEVELS
        ):
            col = 1 + 2 * j

            macro = float(
                row[
                    f"{level}_Macro"
                ]
            )

            miss = float(
                row[
                    f"{level}_MissPct"
                ]
            )

            cm = table.add_cell(
                tr,
                col,
                width=metric_w,
                height=row_h,
                text=f"{macro:.2f}",
                loc="center",
                facecolor=face,
                edgecolor=edge,
            )

            cd = table.add_cell(
                tr,
                col + 1,
                width=metric_w,
                height=row_h,
                text=f"{miss:.2f}",
                loc="center",
                facecolor=face,
                edgecolor=edge,
            )

            # Bold best deployable policy only.
            if (
                not is_oracle
                and body_i
                in best[level]["macro"]
            ):
                cm.get_text().set_weight(
                    "bold"
                )

            if (
                not is_oracle
                and body_i
                in best[level]["miss"]
            ):
                cd.get_text().set_weight(
                    "bold"
                )

            # Upper-bound values are italic rather than bold.
            if is_oracle:
                cm.get_text().set_style(
                    "italic"
                )
                cd.get_text().set_style(
                    "italic"
                )

    for cell in (
        table.get_celld().values()
    ):
        cell.set_linewidth(
            0.65
        )
        cell.get_text().set_fontsize(
            8.4
        )

    ax.add_table(
        table
    )

    ax.text(
        0.5,
        0.92,
        (
            "Runtime-policy ablation under "
            f"Random50 contention at {args.hz} Hz"
        ),
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
            "Macro: mean streaming 3D AP$_{R40}$ "
            "over Car, Pedestrian, and Cyclist (Moderate). "
            "Oracle-Level uses the true contention level "
            "and is reported as an upper bound."
        ),
        ha="center",
        va="bottom",
        fontsize=7.3,
        transform=ax.transAxes,
    )

    fig.savefig(
        output_path,
        dpi=args.dpi,
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

    print()
    print("=" * 110)
    print("TV POLICY ABLATION TABLE")
    print("=" * 110)

    rows = collect(
        args,
        repo,
    )

    tag = (
        f"{args.hz}Hz_"
        f"seed{args.trace_seed}"
    )

    csv_path = (
        out_dir
        / f"tv_policy_ablation_macro_miss_{tag}.csv"
    )

    png_path = (
        out_dir
        / f"tv_policy_ablation_macro_miss_{tag}.png"
    )

    write_csv(
        rows,
        csv_path,
    )

    render_table(
        rows,
        png_path,
        args,
    )

    print()
    print("=" * 110)
    print("[DONE]")
    print(f"CSV : {csv_path}")
    print(f"PNG : {png_path}")
    print("=" * 110)


if __name__ == "__main__":
    main()
