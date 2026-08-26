#!/usr/bin/env python3

import argparse
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        '--cfg_file',
        default='configs/stream/kitti_models/'
                'stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-'
                'lka_7-mcl_5090_eval.yaml',
    )

    p.add_argument('--ckpt', required=True)
    p.add_argument('--sample-index', type=int, default=10)

    return p.parse_args()


def get_backbone(model):
    for m in model.modules():
        if type(m).__name__ == 'StreamDSGN2Backbone':
            return m

    raise RuntimeError('StreamDSGN2Backbone not found')


def maxdiff(a, b):
    if a is None and b is None:
        return 0.0

    if (a is None) != (b is None):
        return float('inf')

    return float((a.float() - b.float()).abs().max().item())


def main():
    args = parse_args()

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
    bb = get_backbone(model)

    sample = dataset[args.sample_index]
    batch = dataset.collate_batch([sample])
    load_data_to_gpu(batch)

    token = batch['token'] if 'token' in batch else batch
    left = token['left_img']
    right = token['right_img']

    amp = bool(cfg.MODEL.USE_AMP['TEST'])

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):

        # ------------------------------------------------------------
        # LEFT reference
        # ------------------------------------------------------------
        ref_left_stereo, ref_left_sem = bb.forward_2d(left)

        left_shallow = bb.forward_2d_shallow(left)

        (
            split_left_stereo,
            split_left_sem,
            split_left_feats,
        ) = bb.forward_2d_deep_full_from_shallow(
            left,
            left_shallow,
            return_stage_features=True,
        )

        # ------------------------------------------------------------
        # RIGHT reference
        # ------------------------------------------------------------
        ref_right_stereo, ref_right_sem = bb.forward_2d(right)

        right_shallow = bb.forward_2d_shallow(right)

        (
            split_right_stereo,
            split_right_sem,
            split_right_feats,
        ) = bb.forward_2d_deep_full_from_shallow(
            right,
            right_shallow,
            return_stage_features=True,
        )

    print()
    print('=' * 72)
    print('A_s / A_d split equivalence')
    print('=' * 72)

    print('left shallow :', tuple(left_shallow.shape))
    print('left layer2  :', tuple(split_left_feats[1].shape))
    print('left layer3  :', tuple(split_left_feats[2].shape))
    print('left layer4  :', tuple(split_left_feats[3].shape))
    print('left stereo  :', tuple(split_left_stereo.shape))

    print()

    print(
        'Left stereo MaxDiff :',
        f'{maxdiff(ref_left_stereo, split_left_stereo):.8f}'
    )

    print(
        'Left sem MaxDiff    :',
        f'{maxdiff(ref_left_sem, split_left_sem):.8f}'
    )

    print(
        'Right stereo MaxDiff:',
        f'{maxdiff(ref_right_stereo, split_right_stereo):.8f}'
    )

    print(
        'Right sem MaxDiff   :',
        f'{maxdiff(ref_right_sem, split_right_sem):.8f}'
    )

    print()
    print('Expected:')
    print('  shallow  ~= [1, 64, 160, 624]')
    print('  layer2   ~= [1,128,  80, 312]')
    print('  layer3   ~= [1,128,  80, 312]')
    print('  layer4   ~= [1,128,  80, 312]')
    print()
    print('Full split path should be numerically identical or at FP16 noise level.')
    print('=' * 72)


if __name__ == '__main__':
    main()
