#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--candidate-dir', required=True)
    p.add_argument('--matrix-size', type=int, required=True)
    p.add_argument('--idle-ms', type=float, required=True)
    p.add_argument('--dtype', required=True)
    p.add_argument('--targets', default='1.25,1.50,2.00,2.50')
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    args = parse_args()
    baseline = json.loads(Path(args.baseline).read_text())
    base = float(baseline['probe']['p50_ms'])

    rows = []
    pat = re.compile(r'probe_r(\d+)\.json$')
    for p in sorted(Path(args.candidate_dir).glob('probe_r*.json')):
        m = pat.search(p.name)
        if not m:
            continue
        repeats = int(m.group(1))
        data = json.loads(p.read_text())
        p50 = float(data['probe']['p50_ms'])
        rows.append({
            'busy_repeats': repeats,
            'probe_p50_ms': p50,
            'slowdown': p50 / base,
            'probe_json': str(p),
        })

    if len(rows) < 4:
        raise RuntimeError('Need at least four contention candidates')

    # Build a monotonic envelope: both compute amount (repeats) and measured
    # probe latency must increase.  This removes noisy/non-informative points.
    rows.sort(key=lambda x: x['busy_repeats'])
    envelope = []
    best_p50 = base * 1.03
    for r in rows:
        if r['probe_p50_ms'] > best_p50:
            envelope.append(r)
            best_p50 = r['probe_p50_ms']

    if len(envelope) < 4:
        raise RuntimeError(
            'Fewer than 4 clearly separated contention points. '
            'Increase CANDIDATES, increase GEMM_N, or reduce IDLE_MS.'
        )

    targets = [float(x) for x in args.targets.split(',')]
    if len(targets) != 4:
        raise ValueError('Exactly four target slowdown values are required')

    selected = []
    start = 0
    for i, target in enumerate(targets):
        remaining_needed = len(targets) - i - 1
        end = len(envelope) - remaining_needed
        pool = envelope[start:end]
        choice_rel = min(
            range(len(pool)),
            key=lambda j: abs(pool[j]['slowdown'] - target),
        )
        choice_idx = start + choice_rel
        selected.append(envelope[choice_idx])
        start = choice_idx + 1

    levels = [{
        'level': 'L0',
        'busy_repeats': 0,
        'probe_p50_ms': base,
        'slowdown': 1.0,
    }]
    for i, r in enumerate(selected, 1):
        levels.append({
            'level': f'L{i}',
            'busy_repeats': int(r['busy_repeats']),
            'probe_p50_ms': float(r['probe_p50_ms']),
            'slowdown': float(r['slowdown']),
        })

    centroids = [float(x['probe_p50_ms']) for x in levels]
    thresholds = [
        (centroids[i] + centroids[i + 1]) / 2.0
        for i in range(len(centroids) - 1)
    ]

    result = {
        'workload': {
            'operation': 'torch.mm fixed square GEMM',
            'matrix_size': args.matrix_size,
            'idle_ms': args.idle_ms,
            'dtype': args.dtype,
            'vary_only': 'busy_repeats per fixed burst',
        },
        'targets': targets,
        'levels': levels,
        'nearest_centroid_thresholds_ms': thresholds,
        'all_candidates': rows,
        'monotonic_envelope': envelope,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

