#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CLASSES = (
    'Car',
    'Pedestrian',
    'Cyclist',
)

METRIC_SUFFIX = {
    cls_name: f'{cls_name}_3d/moderate_R40'
    for cls_name in CLASSES
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Plot Observation 1: spatial redundancy and '
            'task sensitivity under partial BEV recomputation.'
        )
    )

    parser.add_argument(
        '--root',
        default='outputs/single_roi',
    )

    parser.add_argument(
        '--sizes',
        type=int,
        nargs='+',
        default=[96, 128, 160, 192, 224],
    )

    parser.add_argument(
        '--roi-pattern',
        default='oracle_reuse_{size}',
    )

    parser.add_argument(
        '--full-dir',
        default='full_baseline',
    )

    parser.add_argument(
        '--outdir',
        default='outputs/single_roi/observation1_fig1',
    )

    return parser.parse_args()


def flatten_numeric_dict(obj, prefix=''):
    """
    Flatten nested metric dictionaries.

    Supports both:

        Car_3d/moderate_R40

    and:

        offline_3d/Car_3d/moderate_R40
    """
    out = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = (
                f'{prefix}/{key}'
                if prefix
                else str(key)
            )
            out.update(
                flatten_numeric_dict(
                    value,
                    new_key,
                )
            )

    elif (
        isinstance(obj, (int, float))
        and not isinstance(obj, bool)
    ):
        out[prefix] = float(obj)

    return out


def lookup_metric(summary, suffix):
    """
    First use the new observation1 export.

    If the result was produced before the Observation-1 patch,
    fall back to dataset_metrics.
    """
    obs = summary.get('observation1') or {}

    explicit = (
        obs.get('ap_r40_3d_moderate')
        or {}
    )

    cls_name = suffix.split('_3d/', 1)[0]

    if explicit.get(cls_name) is not None:
        return float(explicit[cls_name])

    flat = flatten_numeric_dict(
        summary.get('dataset_metrics', {})
    )

    if suffix in flat:
        return flat[suffix]

    matches = [
        (key, value)
        for key, value in flat.items()
        if key.endswith(suffix)
    ]

    if len(matches) == 1:
        return matches[0][1]

    if len(matches) == 0:
        available = '\n'.join(
            sorted(flat.keys())
        )

        raise KeyError(
            f'Cannot find metric {suffix!r}.\n'
            f'Available metric keys:\n{available}'
        )

    raise KeyError(
        f'Ambiguous metric {suffix!r}: '
        + ', '.join(
            key for key, _ in matches
        )
    )


def get_nonfirst_frames(summary):
    """
    Scene-first frames are intentionally Full.

    Observation 1 studies partial recomputation, so use
    non-first frames whenever possible.
    """
    frames = summary.get('frames', [])

    nonfirst = [
        frame for frame in frames
        if not bool(
            frame.get('first_frame', False)
        )
    ]

    return (
        nonfirst
        if nonfirst
        else frames
    )


def read_run(summary_path, label):
    with summary_path.open('r') as f:
        summary = json.load(f)

    obs = summary.get('observation1') or {}
    frames = get_nonfirst_frames(summary)

    # ------------------------------------------------------------
    # Recompute ratio
    # ------------------------------------------------------------
    if (
        obs.get(
            'recompute_ratio_nonfirst'
        )
        is not None
    ):
        recompute_ratio = float(
            obs['recompute_ratio_nonfirst']
        )

    elif frames:
        recompute_ratio = float(
            np.mean([
                float(frame['r_ratio'])
                for frame in frames
            ])
        )

    else:
        recompute_ratio = float(
            summary['mean_r_ratio']
        )

    # ------------------------------------------------------------
    # Feature L1
    # ------------------------------------------------------------
    if (
        obs.get(
            'feature_l1_mean_nonfirst'
        )
        is not None
    ):
        feature_l1 = float(
            obs['feature_l1_mean_nonfirst']
        )

    elif frames:
        feature_l1 = float(
            np.mean([
                float(frame['feature_l1'])
                for frame in frames
            ])
        )

    else:
        feature_l1 = float(
            summary['mean_feature_l1']
        )

    # ------------------------------------------------------------
    # Official KITTI AP_R40, Moderate, strict class threshold.
    # Car: 0.70
    # Pedestrian: 0.50
    # Cyclist: 0.50
    # ------------------------------------------------------------
    aps = {
        cls_name: lookup_metric(
            summary,
            METRIC_SUFFIX[cls_name],
        )
        for cls_name in CLASSES
    }

    return {
        'label': label,
        'summary': str(summary_path),

        'recompute_ratio': recompute_ratio,
        'recompute_pct': (
            recompute_ratio * 100.0
        ),

        'feature_l1': feature_l1,

        'Car_ap3d_r40_moderate':
            aps['Car'],

        'Pedestrian_ap3d_r40_moderate':
            aps['Pedestrian'],

        'Cyclist_ap3d_r40_moderate':
            aps['Cyclist'],
    }


def save_csv(rows, output_path):
    fields = [
        'label',
        'recompute_ratio',
        'recompute_pct',
        'feature_l1',
        'Car_ap3d_r40_moderate',
        'Pedestrian_ap3d_r40_moderate',
        'Cyclist_ap3d_r40_moderate',
        'summary',
    ]

    with output_path.open(
        'w',
        newline='',
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def plot_feature_l1(rows, outdir):
    """
    Figure 1(a):
        recomputed BEV ratio vs feature reconstruction error.
    """
    x = [
        row['recompute_pct']
        for row in rows
    ]

    y = [
        row['feature_l1']
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

    ax.set_xlabel(
        'Recomputed BEV ratio (%)'
    )

    ax.set_ylabel(
        'Mean BEV feature L1'
    )

    ax.set_xlim(
        0,
        104,
    )

    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.3,
    )

    fig.savefig(
        outdir / 'fig1a_feature_l1.pdf',
        bbox_inches='tight',
    )

    fig.savefig(
        outdir / 'fig1a_feature_l1.png',
        dpi=300,
        bbox_inches='tight',
    )

    plt.close(fig)


def plot_ap(rows, outdir):
    """
    Figure 1(b):
        recomputed BEV ratio vs detection AP.

    Official class-specific strict KITTI thresholds:
        Car        : IoU 0.70
        Pedestrian : IoU 0.50
        Cyclist    : IoU 0.50
    """
    x = [
        row['recompute_pct']
        for row in rows
    ]

    markers = {
        'Car': 'o',
        'Pedestrian': 's',
        'Cyclist': '^',
    }

    fig = plt.figure(
        figsize=(4.8, 3.4)
    )

    ax = fig.add_axes([
        0.16,
        0.17,
        0.80,
        0.78,
    ])

    for cls_name in CLASSES:
        y = [
            row[
                f'{cls_name}_ap3d_r40_moderate'
            ]
            for row in rows
        ]

        ax.plot(
            x,
            y,
            marker=markers[cls_name],
            linewidth=1.8,
            markersize=5,
            label=cls_name,
        )

    ax.set_xlabel(
        'Recomputed BEV ratio (%)'
    )

    ax.set_ylabel(
        r'3D AP$_{R40}$ (Moderate)'
    )

    ax.set_xlim(
        0,
        104,
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.3,
    )

    fig.savefig(
        outdir / 'fig1b_ap3d.pdf',
        bbox_inches='tight',
    )

    fig.savefig(
        outdir / 'fig1b_ap3d.png',
        dpi=300,
        bbox_inches='tight',
    )

    plt.close(fig)


def print_table(rows):
    print()
    print(
        f"{'Run':>10} "
        f"{'BEV%':>8} "
        f"{'FeatL1':>10} "
        f"{'Car':>9} "
        f"{'Ped':>9} "
        f"{'Cyc':>9}"
    )

    print(
        '-' * 68
    )

    for row in rows:
        print(
            f"{row['label']:>10} "
            f"{row['recompute_pct']:8.3f} "
            f"{row['feature_l1']:10.6f} "
            f"{row['Car_ap3d_r40_moderate']:9.4f} "
            f"{row['Pedestrian_ap3d_r40_moderate']:9.4f} "
            f"{row['Cyclist_ap3d_r40_moderate']:9.4f}"
        )

    print()


def main():
    args = parse_args()

    root = Path(
        args.root
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    # ------------------------------------------------------------
    # Partial-recompute diagnostic runs.
    # ------------------------------------------------------------
    for size in args.sizes:
        run_dir = root / (
            args.roi_pattern.format(
                size=size
            )
        )

        summary_path = (
            run_dir / 'summary.json'
        )

        if not summary_path.exists():
            raise FileNotFoundError(
                summary_path
            )

        rows.append(
            read_run(
                summary_path,
                f'{size}x{size}',
            )
        )

    # ------------------------------------------------------------
    # Full-computation reference.
    # ------------------------------------------------------------
    full_summary = (
        root
        / args.full_dir
        / 'summary.json'
    )

    if not full_summary.exists():
        raise FileNotFoundError(
            full_summary
        )

    rows.append(
        read_run(
            full_summary,
            'Full',
        )
    )

    # Sort by actual non-first recompute ratio.
    rows.sort(
        key=lambda x:
            x['recompute_ratio']
    )

    csv_path = (
        outdir
        / 'fig1_data.csv'
    )

    save_csv(
        rows,
        csv_path,
    )

    print_table(rows)

    # Paper-friendly vector output.
    plt.rcParams.update({
        'font.size': 10,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

    plot_feature_l1(
        rows,
        outdir,
    )

    plot_ap(
        rows,
        outdir,
    )

    print(
        f'Saved: {csv_path}'
    )

    print(
        'Saved: '
        f'{outdir / "fig1a_feature_l1.pdf"}'
    )

    print(
        'Saved: '
        f'{outdir / "fig1b_ap3d.pdf"}'
    )


if __name__ == '__main__':
    main()
