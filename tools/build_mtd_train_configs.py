#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml
from easydict import EasyDict

from pcdet.config import (
    cfg_from_yaml_file,
)


def plain(x):
    if isinstance(
        x,
        dict,
    ):
        return {
            k: plain(v)
            for k, v in x.items()
        }

    if isinstance(
        x,
        (list, tuple),
    ):
        return [
            plain(v)
            for v in x
        ]

    return x


def build_one(
    base_cfg,
    horizon,
    out_path,
    epochs,
    lr,
):
    cfg = EasyDict()

    cfg_from_yaml_file(
        base_cfg,
        cfg,
    )

    if horizon not in (
        2,
        3,
    ):
        raise ValueError(
            "horizon must be 2 or 3"
        )

    target = (
        f"next{horizon}"
    )

    future_steps = list(
        range(
            2,
            horizon + 1,
        )
    )

    cfg.DATA_CONFIG.MTD_FUTURE_STEPS = (
        future_steps
    )

    cfg.DATA_CONFIG.BOX3D_SUPERVISION = (
        target
    )

    found_sampler = False

    for aug in (
        cfg.DATA_CONFIG
        .TRAIN_DATA_AUGMENTOR
    ):
        if (
            aug["NAME"]
            !=
            "gt_sampling"
        ):
            continue

        found_sampler = True

        tags = copy.deepcopy(
            aug[
                "ALL_SAMPLE_TAG"
            ]
        )

        for step in future_steps:
            tags[
                f"next{step}"
            ] = step

        aug[
            "ALL_SAMPLE_TAG"
        ] = tags

        aug[
            "BOX3D_SUPERVISION"
        ] = target

    if not found_sampler:
        raise RuntimeError(
            "gt_sampling not found"
        )

    cfg.MODEL.DENSE_HEAD.BOX3D_SUPERVISION = (
        target
    )

    # Original trend loss is defined
    # around token / prev / prev2.
    # Future MTD heads use direct future detection loss.
    cfg.MODEL.DENSE_HEAD.HISTORY_TAG = []

    cfg.OPTIMIZATION.NUM_EPOCHS = int(
        epochs
    )

    cfg.OPTIMIZATION.LR = float(
        lr
    )

    out = Path(
        out_path
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        yaml.safe_dump(
            plain(cfg),
            sort_keys=False,
            default_flow_style=False,
        )
    )

    print(
        f"[WRITE] {out}"
    )

    print(
        f"        target={target}"
    )

    print(
        "        future_steps="
        f"{future_steps}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base_cfg",
        required=True,
    )

    p.add_argument(
        "--h2_out",
        required=True,
    )

    p.add_argument(
        "--h3_out",
        required=True,
    )

    p.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=2e-4,
    )

    a = p.parse_args()

    build_one(
        a.base_cfg,
        2,
        a.h2_out,
        a.epochs,
        a.lr,
    )

    build_one(
        a.base_cfg,
        3,
        a.h3_out,
        a.epochs,
        a.lr,
    )


if __name__ == "__main__":
    main()

# BUILD_MTD_CONFIGS_EOF
