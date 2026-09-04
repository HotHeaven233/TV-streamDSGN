#!/usr/bin/env python3

import argparse
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
    model_fn_decorator,
)

from pcdet.utils import (
    common_utils,
)


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cfg",
        required=True,
    )

    p.add_argument(
        "--pretrained",
        required=True,
    )

    a = p.parse_args()

    cfg.clear()

    cfg_from_yaml_file(
        a.cfg,
        cfg,
    )

    cfg.LOCAL_RANK = 0

    cfg.OPTIMIZATION.USE_AMP = (
        cfg.MODEL.get(
            "USE_AMP",
            {
                "TRAIN": False,
                "TEST": False,
            },
        )
    )

    logger = (
        common_utils.create_logger()
    )

    dataset, loader, _ = (
        build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=0,
            logger=logger,
            training=True,
            total_epochs=1,
        )
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(
            cfg.CLASS_NAMES
        ),
        dataset=dataset,
    )

    model.load_params_from_file(
        a.pretrained,
        logger=logger,
        to_cpu=True,
    )

    model.cuda()

    for name, param in (
        model.named_parameters()
    ):
        param.requires_grad = (
            name.startswith(
                "dense_head."
            )
        )

    model._train_mtd_head_only = True

    model.train()

    for name, module in (
        model.named_modules()
    ):
        if (
            name == "dense_head"
            or
            name.startswith(
                "dense_head."
            )
        ):
            continue

        module.eval()

    model.dense_head.train()

    batch = next(
        iter(loader)
    )

    loss, tb, _ = (
        model_fn_decorator()(
            model,
            batch,
        )
    )

    loss.backward()

    dense_grad = 0
    bad_grad = []

    for name, param in (
        model.named_parameters()
    ):
        if param.grad is None:
            continue

        if name.startswith(
            "dense_head."
        ):
            dense_grad += 1
        else:
            bad_grad.append(
                name
            )

    if dense_grad == 0:
        raise RuntimeError(
            "dense_head received no gradient"
        )

    if bad_grad:
        raise RuntimeError(
            "frozen parameters have gradients: "
            f"{bad_grad[:10]}"
        )

    print(
        "=" * 80
    )

    print(
        "[PASS] MTD head-training smoke"
    )

    print(
        "target:",
        cfg.MODEL.DENSE_HEAD
        .BOX3D_SUPERVISION,
    )

    print(
        "loss:",
        float(
            loss.item()
        ),
    )

    print(
        "dense grad vars:",
        dense_grad,
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()

# SMOKE_MTD_HEAD_TRAINING_EOF
