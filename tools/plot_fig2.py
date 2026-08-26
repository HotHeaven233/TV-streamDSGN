#!/usr/bin/env python3

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            'Plot Observation 2: cached BEV feature '
            'staleness versus feature age.'
        )
    )

    p.add_argument(
        '--input',
        default=(
            'outputs/single_roi/'
            'observation2_staleness_160/'
            'observation2_staleness.json'
        ),
    )

    p.add_argument(
        '--outdir',
        default=(
            'outputs/single_roi/'
            'observation2_fig2'
        ),
    )

    return p.parse_args()


def load_observation(path):
    with path.open('r') as f:
        obj = json.load(f)

    # Allow either the standalone observation2 JSON
    # or the complete evaluator summary.json.
    if (
        'age_statistics' not in obj
        and obj.get('observation2') is not None
    ):
        obj = obj['observation2']

    if 'age_statistics' not in obj:
        raise KeyError(
            'Cannot find age_statistics in input JSON'
        )

    return obj


def exact_rows(obs):
    rows = []

    for row in obs['age_statistics']:
        rows.append({
            'age': int(row['age']),
            'age_label': str(row['age_label']),
            'count': int(row['count']),
            'fraction': float(row['fraction']),
            'fraction_pct': (
                100.0 * float(row['fraction'])
            ),
            'mean_feature_l1': float(
                row['mean_feature_l1']
            ),
            'std_feature_l1': float(
                row['std_feature_l1']
            ),
            'feature_l1_sum': float(
                row['feature_l1_sum']
            ),
            'feature_l1_sq_sum': float(
                row['feature_l1_sq_sum']
            ),
        })

    return rows


def aggregate_group(rows, label, lo, hi):
    selected = [
        row for row in rows
        if lo <= row['age'] <= hi
    ]

    count = sum(
        row['count']
        for row in selected
    )

    error_sum = sum(
        row['feature_l1_sum']
        for row in selected
    )

    error_sq_sum = sum(
        row['feature_l1_sq_sum']
        for row in selected
    )

    total = sum(
        row['count']
        for row in rows
    )

    if count > 0:
        mean = (
            error_sum / count
        )

        variance = max(
            error_sq_sum / count
            - mean * mean,
            0.0,
        )

        std = math.sqrt(
            variance
        )

    else:
        mean = float('nan')
        std = float('nan')

    return {
        'age_group': label,
        'age_lo': lo,
        'age_hi': hi,
        'count': count,
        'fraction': (
            count / total
            if total > 0
            else float('nan')
        ),
        'fraction_pct': (
            100.0 * count / total
            if total > 0
            else float('nan')
        ),
        'mean_feature_l1': mean,
        'std_feature_l1': std,
    }


def grouped_rows(rows, age_cap):
    groups = []

    if age_cap >= 1:
        groups.append(
            aggregate_group(
                rows,
                '1',
                1,
                1,
            )
        )

    if age_cap >= 2:
        groups.append(
            aggregate_group(
                rows,
                '2',
                2,
                2,
            )
        )

    if age_cap >= 3:
        upper = min(
            4,
            age_cap,
        )

        groups.append(
            aggregate_group(
                rows,
                (
                    '3'
                    if upper == 3
                    else '3-4'
                ),
                3,
                upper,
            )
        )

    if age_cap >= 5:
        groups.append(
            aggregate_group(
                rows,
                f'5-{age_cap}+',
                5,
                age_cap,
            )
        )

    return groups


def write_csv(path, rows, fields):
    with path.open(
        'w',
        newline='',
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row[key]
                for key in fields
            })


def write_latex_table(path, groups):
    lines = [
        r'\begin{tabular}{lrr}',
        r'\toprule',
        (
            r'Feature age & Reused cells (\%) '
            r'& Mean feature L1 \\'
        ),
        r'\midrule',
    ]

    for row in groups:
        lines.append(
            (
                f"{row['age_group']} & "
                f"{row['fraction_pct']:.2f} & "
                f"{row['mean_feature_l1']:.5f} "
                r'\\'
            )
        )

    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
    ])

    path.write_text(
        '\n'.join(lines) + '\n'
    )


def plot_error(rows, outdir):
    x = [
        row['age']
        for row in rows
    ]

    y = [
        row['mean_feature_l1']
        for row in rows
    ]

    labels = [
        row['age_label']
        for row in rows
    ]

    fig = plt.figure(
        figsize=(4.8, 3.4)
    )

    ax = fig.add_axes([
        0.16,
        0.17,
        0.80,
        0.78,
    ])

    ax.plot(
        x,
        y,
        marker='o',
        linewidth=1.8,
        markersize=5,
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_xlabel(
        'Cached feature age (frames)'
    )

    ax.set_ylabel(
        'Mean BEV feature L1'
    )

    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.3,
    )

    fig.savefig(
        outdir / 'fig2a_staleness_error.pdf',
        bbox_inches='tight',
    )

    fig.savefig(
        outdir / 'fig2a_staleness_error.png',
        dpi=300,
        bbox_inches='tight',
    )

    plt.close(fig)


def plot_distribution(rows, outdir):
    x = [
        row['age']
        for row in rows
    ]

    y = [
        row['fraction_pct']
        for row in rows
    ]

    labels = [
        row['age_label']
        for row in rows
    ]

    fig = plt.figure(
        figsize=(4.8, 3.4)
    )

    ax = fig.add_axes([
        0.16,
        0.17,
        0.80,
        0.78,
    ])

    ax.bar(
        x,
        y,
        width=0.7,
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_xlabel(
        'Cached feature age (frames)'
    )

    ax.set_ylabel(
        'Reused BEV cells (%)'
    )

    ax.set_ylim(
        bottom=0,
    )

    ax.grid(
        True,
        axis='y',
        linewidth=0.5,
        alpha=0.3,
    )

    fig.savefig(
        outdir / 'fig2b_staleness_distribution.pdf',
        bbox_inches='tight',
    )

    fig.savefig(
        outdir / 'fig2b_staleness_distribution.png',
        dpi=300,
        bbox_inches='tight',
    )

    plt.close(fig)


def print_exact_table(rows):
    print()
    print(
        f"{'Age':>8} "
        f"{'Cells%':>10} "
        f"{'MeanL1':>12} "
        f"{'StdL1':>12}"
    )

    print(
        '-' * 48
    )

    for row in rows:
        print(
            f"{row['age_label']:>8} "
            f"{row['fraction_pct']:10.3f} "
            f"{row['mean_feature_l1']:12.6f} "
            f"{row['std_feature_l1']:12.6f}"
        )

    print()


def print_grouped_table(groups):
    print(
        '[Paper grouped table]'
    )

    print(
        f"{'Age group':>12} "
        f"{'Cells%':>10} "
        f"{'MeanL1':>12}"
    )

    print(
        '-' * 38
    )

    for row in groups:
        print(
            f"{row['age_group']:>12} "
            f"{row['fraction_pct']:10.3f} "
            f"{row['mean_feature_l1']:12.6f}"
        )

    print()


def main():
    args = parse_args()

    input_path = Path(
        args.input
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    obs = load_observation(
        input_path
    )

    rows = exact_rows(
        obs
    )

    age_cap = int(
        obs['age_cap']
    )

    groups = grouped_rows(
        rows,
        age_cap,
    )

    print_exact_table(
        rows
    )

    print_grouped_table(
        groups
    )

    exact_csv = (
        outdir
        / 'observation2_exact.csv'
    )

    grouped_csv = (
        outdir
        / 'observation2_grouped.csv'
    )

    latex_path = (
        outdir
        / 'observation2_grouped_table.tex'
    )

    write_csv(
        exact_csv,
        rows,
        [
            'age',
            'age_label',
            'count',
            'fraction_pct',
            'mean_feature_l1',
            'std_feature_l1',
        ],
    )

    write_csv(
        grouped_csv,
        groups,
        [
            'age_group',
            'count',
            'fraction_pct',
            'mean_feature_l1',
            'std_feature_l1',
        ],
    )

    write_latex_table(
        latex_path,
        groups,
    )

    plt.rcParams.update({
        'font.size': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    plot_error(
        rows,
        outdir,
    )

    plot_distribution(
        rows,
        outdir,
    )

    print(
        f'Saved: {exact_csv}'
    )

    print(
        f'Saved: {grouped_csv}'
    )

    print(
        f'Saved: {latex_path}'
    )

    print(
        'Saved: '
        f'{outdir / "fig2a_staleness_error.pdf"}'
    )

    print(
        'Saved: '
        f'{outdir / "fig2b_staleness_distribution.pdf"}'
    )


if __name__ == '__main__':
    main()
