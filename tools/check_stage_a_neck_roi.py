#!/usr/bin/env python3

import argparse
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def get_bb(model):
    for m in model.modules():
        if type(m).__name__ == 'StreamDSGN2Backbone':
            return m
    raise RuntimeError('StreamDSGN2Backbone not found')


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
    p.add_argument('--ratio', type=float, default=0.5)

    p.add_argument(
        '--halos',
        nargs='+',
        type=int,
        default=[1, 2, 3, 4, 6, 8],
    )

    p.add_argument(
        '--positions',
        nargs='+',
        default=['center', 'boundary'],
        choices=['center', 'random', 'boundary'],
    )

    return p.parse_args()


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
    bb = get_bb(model)

    sample = dataset[args.sample_index]
    batch = dataset.collate_batch([sample])
    load_data_to_gpu(batch)

    token = batch['token'] if 'token' in batch else batch
    left = token['left_img']

    amp = bool(cfg.MODEL.USE_AMP['TEST'])

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
        backbone_feats = bb.feature_backbone(left)

        feats = [
            left,
            *list(backbone_feats),
        ]

        full_stereo, _ = bb.feature_neck(feats)

    H, W = backbone_feats[1].shape[-2:]

    print()
    print('=' * 92)
    print('Local neck equivalence')
    print('Full stereo:', tuple(full_stereo.shape))
    print('Base grid  :', (H, W))
    print('=' * 92)

    for position in args.positions:

        rect = bb._adaptive_a_rect_from_ratio(
            H,
            W,
            args.ratio,
            position_mode=position,
            align=4,
        )

        print()
        print('position:', position)
        print('base ROI :', rect)

        for halo in args.halos:

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp):
                patch, out_rect = \
                    bb.feature_neck.forward_stereo_roi(
                        feats,
                        base_rect=rect,
                        base_halo=halo,
                    )

            y0, y1, x0, x1 = out_rect

            ref = full_stereo[
                ...,
                y0:y1,
                x0:x1,
            ]

            diff = (
                patch.float()
                -
                ref.float()
            ).abs()

            max_diff = float(diff.max().item())
            mean_diff = float(diff.mean().item())

            print(
                f'halo={halo:2d} '
                f'outROI={y1-y0:3d}x{x1-x0:4d} '
                f'MeanDiff={mean_diff:.8f} '
                f'MaxDiff={max_diff:.8f}'
            )


if __name__ == '__main__':
    main()
