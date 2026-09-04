#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    ElasticBEVBranch,
    MONOTONIC_SCHEDULES,
    extract_fixed_layer1_prefix,
)
from pcdet.models.backbones_3d_stream.elastic_v4_bn_hybrid import (
    FULL_SCHEDULE,
    hybrid_forward,
)


TRAIN_SCHEDULES = tuple(
    s for s in MONOTONIC_SCHEDULES if tuple(s) != FULL_SCHEDULE
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Final BatchNorm running-statistics calibration for Elastic-v4-BN. "
            "No trainable weight is updated."
        )
    )
    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--output_ckpt", required=True)
    p.add_argument("--batches", type=int, default=1660)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


def make_cfg(path):
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = Path(path).stem
    cfg.DATA_CONFIG.INFER_TIME_PATH = None
    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None
    return cfg


def bn_modules(module):
    return [
        m
        for m in module.modules()
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    ]


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    c = make_cfg(args.full_cfg)
    logger = common_utils.create_logger()

    dataset, loader, _ = build_dataloader(
        dataset_cfg=c.DATA_CONFIG,
        class_names=c.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    base_model = build_network(
        model_cfg=c.MODEL,
        num_class=len(c.CLASS_NAMES),
        dataset=dataset,
    )
    base_model.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )
    base_model.cuda().eval()
    for p in base_model.parameters():
        p.requires_grad = False

    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(c.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES),
    ).cuda().eval()

    checkpoint = torch.load(args.elastic_ckpt, map_location="cpu")
    branch.load_state_dict(checkpoint["branch"], strict=True)

    bns = bn_modules(branch)
    if not bns:
        raise RuntimeError("No BatchNorm modules found")

    # Only BN modules enter train mode. Everything else remains eval.
    old_momentum = {}
    for bn in bns:
        bn.reset_running_stats()
        old_momentum[id(bn)] = bn.momentum
        bn.momentum = None
        bn.train()

    amp_enabled = bool(c.MODEL.USE_AMP.get("TEST", True))
    requested = max(int(args.batches), 1)
    seen = 0
    loader_iter = iter(loader)

    while seen < requested:
        try:
            batch_dict = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_dict = next(loader_iter)

        load_data_to_gpu(batch_dict)
        schedule = TRAIN_SCHEDULES[seen % len(TRAIN_SCHEDULES)]

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            prefix = extract_fixed_layer1_prefix(
                base_model.backbone_3d,
                batch_dict["token"],
            )
            hybrid_forward(
                branch,
                batch_dict["token"],
                prefix,
                base_model.backbone_3d,
                schedule,
            )

        seen += 1
        if seen <= 5 or seen % 100 == 0:
            print(
                f"[BN-CAL] {seen:04d}/{requested:04d} "
                f"schedule={schedule}"
            )

    for bn in bns:
        bn.momentum = old_momentum[id(bn)]
        bn.eval()
    branch.eval()

    out = dict(checkpoint)
    out["branch"] = branch.state_dict()
    out["version"] = "elastic_v4_bn_profile_robust_calibrated"
    out["bn_calibration"] = {
        "batches": seen,
        "schedule_policy": "uniform cycle over 83 non-Full hybrid profiles",
        "dataset_split": "test/val",
        "momentum": "cumulative_average",
        "native_full_prefix": True,
    }

    output_path = Path(args.output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)

    print("=" * 72)
    print(f"Saved calibrated checkpoint: {output_path}")
    print(f"BN modules: {len(bns)}")
    print(f"Calibration batches: {seen}")
    print(
        "Each of the 83 trainable runtime profiles was visited "
        f"about {seen / len(TRAIN_SCHEDULES):.1f} times."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()

