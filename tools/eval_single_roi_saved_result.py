#!/usr/bin/env python3

"""
Evaluate a result.pkl already produced by
eval_single_roi_bev_semantic.py.

IMPORTANT:
The semantic/oracle experiment computes the full current-frame BEV teacher
before R/P/U composition. Therefore it has no valid selective-execution
latency.

Only offline_3d is evaluated here.
"""

import argparse
import pickle
from pathlib import Path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils

from eval_utils.eval_utils import format_paper_metrics
from single_roi_common import DEFAULT_CFG, save_json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate saved single-ROI semantic predictions'
    )

    parser.add_argument(
        '--cfg',
        default=DEFAULT_CFG,
        help='StreamDSGN config file'
    )

    parser.add_argument(
        '--result',
        required=True,
        help='result.pkl produced by eval_single_roi_bev_semantic.py'
    )

    parser.add_argument(
        '--output',
        default='',
        help='output directory; default=<result parent>/offline_eval'
    )

    parser.add_argument(
        '--workers',
        type=int,
        default=2
    )

    parser.add_argument(
        '--paper-metrics-only',
        action='store_true'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg_from_yaml_file(args.cfg, cfg)

    logger = common_utils.create_logger(rank=0)

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    result_path = Path(args.result)

    if not result_path.exists():
        raise FileNotFoundError(
            f'result file does not exist: {result_path}'
        )

    print(f'[INFO] loading: {result_path}')

    with result_path.open('rb') as f:
        det_annos = pickle.load(f)

    if not isinstance(det_annos, list):
        raise TypeError(
            f'expected result.pkl to contain list, '
            f'got {type(det_annos)}'
        )

    print(f'[INFO] predictions = {len(det_annos)}')
    print(f'[INFO] dataset     = {len(dataset)}')

    if len(det_annos) != len(dataset):
        raise RuntimeError(
            'prediction count mismatch: '
            f'det={len(det_annos)}, dataset={len(dataset)}. '
            'Official KITTI AP requires one prediction '
            'annotation for every dataset frame.'
        )

    outdir = (
        Path(args.output)
        if args.output
        else result_path.parent / 'offline_eval'
    )
    outdir.mkdir(parents=True, exist_ok=True)

    configured_metrics = list(
        cfg.MODEL.POST_PROCESSING.EVAL_METRIC
    )

    print(
        '[INFO] config EVAL_METRIC =',
        configured_metrics
    )

    print(
        '[INFO] semantic/oracle result has no valid '
        'selective-execution timing.'
    )

    print(
        '[INFO] evaluating offline_3d only.'
    )

    result_str, result_dict = dataset.evaluation(
        det_annos,
        dataset.class_names,
        eval_metric=['offline_3d'],
        output_path=outdir,
    )

    if result_str is not None:
        for name, text in result_str.items():
            print()
            print('=' * 80)
            print(name)
            print('=' * 80)

            if args.paper_metrics_only:
                print(format_paper_metrics(text))
            else:
                print(text)

    summary = {
        'result_file': str(result_path),
        'num_predictions': len(det_annos),
        'dataset_size': len(dataset),
        'configured_eval_metrics': configured_metrics,
        'used_eval_metrics': ['offline_3d'],
        'metrics': result_dict,
        'note': (
            'Semantic/oracle validation only. '
            'Streaming metrics are intentionally not computed '
            'because this experiment computes the full current '
            'BEV teacher and therefore has no valid selective '
            'execution latency.'
        ),
    }

    save_json(
        outdir / 'offline_metrics.json',
        summary
    )

    print()
    print(
        '[DONE] saved:',
        outdir / 'offline_metrics.json'
    )


if __name__ == '__main__':
    main()
