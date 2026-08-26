#!/usr/bin/env python3

import argparse
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--cfg_file',
        default='configs/stream/kitti_models/'
                'stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-'
                'lka_7-mcl_5090_eval.yaml',
    )

    p.add_argument('--ckpt', required=True)
    p.add_argument('--ratio', type=float, default=0.5)
    p.add_argument('--sample-index', type=int, default=10)

    args = p.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    logger = common_utils.create_logger()

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=True,
    )

    model.cuda().eval()

    bb = None

    for m in model.modules():
        if type(m).__name__ == 'StreamDSGN2Backbone':
            bb = m
            break

    assert bb is not None

    sample = dataset[args.sample_index]
    batch = dataset.collate_batch([sample])
    load_data_to_gpu(batch)

    # ------------------------------------------------------------
    # First pass: exact Full, initializes stage-A cache.
    # ------------------------------------------------------------
    bb.clear_adaptive_a_profile()

    with torch.no_grad():
        pred, ret = model.forward(batch)

    torch.cuda.synchronize()

    print('Full initialization: OK')

    # ------------------------------------------------------------
    # Second pass: adaptive A, downstream remains original Full.
    # ------------------------------------------------------------
    bb.set_adaptive_a_profile(
        ratio=args.ratio,
        position_mode='center',
    )

    with torch.no_grad():
        pred, ret = model.forward(batch)

    torch.cuda.synchronize()

    print('Adaptive A forward: OK')
    print('ratio:', args.ratio)
    print('ROI:', bb._adaptive_a_last_roi)

    bb.clear_adaptive_a_profile()


if __name__ == '__main__':
    main()
