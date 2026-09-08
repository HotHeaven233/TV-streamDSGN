#!/usr/bin/env python3

import argparse
import math

import torch

from pcdet.config import (
    cfg,
    cfg_from_yaml_file,
)
from pcdet.datasets import (
    build_dataloader,
)
from pcdet.models import (
    build_network,
    load_data_to_gpu,
)
from pcdet.utils import (
    common_utils,
)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--cfg_file',
        required=True,
    )

    ap.add_argument(
        '--pretrained_model',
        required=True,
    )

    ap.add_argument(
        '--workers',
        type=int,
        default=0,
    )

    args = ap.parse_args()

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    logger = (
        common_utils.create_logger(
            rank=0
        )
    )

    (
        dataset,
        loader,
        _,
    ) = build_dataloader(
        dataset_cfg=(
            cfg.DATA_CONFIG
        ),
        class_names=(
            cfg.CLASS_NAMES
        ),
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True,
    )

    model = build_network(
        cfg.MODEL,
        len(
            cfg.CLASS_NAMES
        ),
        dataset,
    )

    model.cuda()

    model.load_params_from_file(
        args.pretrained_model,
        logger=logger,
        to_cpu=True,
    )

    model.train()

    trainable = [
        (
            n,
            p.numel(),
        )
        for (
            n,
            p,
        )
        in model.named_parameters()
        if p.requires_grad
    ]

    print(
        'trainable tensors='
        f'{len(trainable)} '
        'params='
        f'{sum(x[1] for x in trainable):,}'
    )

    trainable_name_set = [
        n
        for n, _
        in model.named_parameters()
        if _.requires_grad
    ]

    bad_frozen = [
        n
        for n
        in trainable_name_set
        if n.startswith(
            'backbone_3d.'
        )
    ]

    if bad_frozen:
        raise RuntimeError(
            'LASP frozen stereo extractor has trainable parameters:\n'
            + '\n'.join(
                bad_frozen[:30]
            )
        )

    required_groups = {
        'backbone_2d':
            any(
                n.startswith(
                    'backbone_2d.'
                )
                for n
                in trainable_name_set
            ),

        'dense_head':
            any(
                n.startswith(
                    'dense_head.'
                )
                for n
                in trainable_name_set
            ),

        'lasp':
            any(
                n.startswith(
                    'lasp.'
                )
                for n
                in trainable_name_set
            ),
    }

    if not all(
        required_groups.values()
    ):
        raise RuntimeError(
            'unexpected LASP trainable groups: '
            f'{required_groups}'
        )

    print(
        '[PASS] trainable scope = '
        'VANBackbone + StreamDetHead + LASP'
    )

    print(
        'first trainable names:'
    )

    for (
        n,
        k,
    ) in trainable[
        :30
    ]:

        print(
            f'  {n}: {k:,}'
        )

    batch = next(
        iter(
            loader
        )
    )

    load_data_to_gpu(
        batch
    )

    (
        ret,
        tb,
        _,
    ) = model(
        batch
    )

    loss = ret[
        'loss'
    ]

    if not torch.isfinite(
        loss
    ):
        raise RuntimeError(
            f'non-finite loss: {loss}'
        )

    if not loss.requires_grad:
        raise RuntimeError(
            'loss does not require grad'
        )

    loss.backward()

    lasp_grad = 0.0
    lasp_count = 0

    for (
        n,
        p,
    ) in model.named_parameters():

        if (
            n.startswith(
                'lasp.'
            )
            and
            p.grad is not None
        ):

            lasp_grad += float(
                p.grad
                .detach()
                .float()
                .norm()
                .item()
            )

            lasp_count += 1

    if (
        lasp_count == 0
        or
        not math.isfinite(
            lasp_grad
        )
        or
        lasp_grad <= 0
    ):
        raise RuntimeError(
            'bad LASP gradients: '
            f'tensors={lasp_count}, '
            f'norm_sum={lasp_grad}'
        )

    print(
        '=== LASP AUDIT PASS ==='
    )

    print(
        'loss='
        f'{float(loss.detach().item()):.6f}'
    )

    print(
        'lasp_grad_tensors='
        f'{lasp_count} '
        'grad_norm_sum='
        f'{lasp_grad:.6f}'
    )

    for key in sorted(
        tb
    ):

        if (
            str(
                key
            ).startswith(
                'lasp_'
            )
            or
            key
            in (
                'base_loss',
                'total_loss',
            )
        ):

            print(
                f'{key}: '
                f'{tb[key]}'
            )


if __name__ == '__main__':
    main()
