#!/usr/bin/env python3
"""
Fixed-profile BatchNorm calibration diagnostic for Elastic-v4-BN.

This script does NOT update convolution weights. It recalibrates only the BN
running statistics actually exercised by one specified hybrid profile, using
the TRAIN split only.

Purpose:
  Test whether low-width collapse is caused by width-only BN statistics being
  contaminated by many mixed profiles.

Example:
  --schedule 0.25,0.25,0.25,0.25,0.25,0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    ElasticBEVBranch,
    extract_fixed_layer1_prefix,
    validate_schedule,
)
from pcdet.models.backbones_3d_stream.elastic_v4_bn_hybrid import (
    FULL_SCHEDULE,
    hybrid_forward,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Recompute BN statistics for exactly one Elastic-v4-BN profile "
            "on the training split; no trainable weight is updated."
        )
    )
    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--output_ckpt", required=True)
    p.add_argument(
        "--schedule",
        required=True,
        help="Six ratios: Res2,Res3,Res4,FPN,Stereo,RPN",
    )
    p.add_argument("--batches", type=int, default=1000)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1024)
    return p.parse_args()


def make_cfg(path):
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = Path(path).stem
    cfg.DATA_CONFIG.INFER_TIME_PATH = None
    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None
    return cfg


def parse_schedule(text):
    return validate_schedule(tuple(float(x.strip()) for x in text.split(",")))


def all_bn_modules(module):
    return [
        m
        for m in module.modules()
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d))
    ]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    set_seed(args.seed)
    schedule = parse_schedule(args.schedule)
    if tuple(schedule) == FULL_SCHEDULE:
        raise RuntimeError("All-1.0 uses native Full; no elastic BN to calibrate.")

    c = make_cfg(args.full_cfg)
    logger = common_utils.create_logger()

    # IMPORTANT: training=True -> train split, not validation/test.
    dataset, loader, _ = build_dataloader(
        dataset_cfg=c.DATA_CONFIG,
        class_names=c.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("Empty training dataset")

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
    for p in branch.parameters():
        p.requires_grad = False

    bns = all_bn_modules(branch)
    if not bns:
        raise RuntimeError("No elastic BatchNorm modules found")

    # Reset all elastic BN statistics. Only BNs actually reached by `schedule`
    # will be updated. Unused BN states are irrelevant for this diagnostic.
    old_momentum = {}
    for bn in bns:
        bn.reset_running_stats()
        old_momentum[id(bn)] = bn.momentum
        bn.momentum = None  # cumulative moving average
        bn.train()

    amp_enabled = bool(c.MODEL.USE_AMP.get("TEST", True))
    requested = max(1, int(args.batches))
    seen = 0
    loader_iter = iter(loader)

    print("=" * 76)
    print("Fixed-profile BN calibration diagnostic")
    print(f"schedule       : {tuple(schedule)}")
    print(f"train samples  : {len(dataset)}")
    print(f"calib batches  : {requested}")
    print("weights        : frozen")
    print("BN statistics  : reset, cumulative average")
    print("data split     : TRAIN")
    print("=" * 76)

    while seen < requested:
        try:
            batch_dict = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch_dict = next(loader_iter)

        load_data_to_gpu(batch_dict)

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
            print(f"[FIXED-BN-CAL] {seen:04d}/{requested:04d}")

    # Freeze calibrated statistics.
    active = 0
    inactive = 0
    min_batches = None
    max_batches = 0
    for bn in bns:
        bn.momentum = old_momentum[id(bn)]
        bn.eval()
        n = int(bn.num_batches_tracked.item())
        if n > 0:
            active += 1
            min_batches = n if min_batches is None else min(min_batches, n)
            max_batches = max(max_batches, n)
        else:
            inactive += 1
    branch.eval()

    out = dict(checkpoint)
    out["branch"] = branch.state_dict()
    out["version"] = "elastic_v4_bn_fixed_profile_calibrated_diagnostic"
    out["bn_calibration"] = {
        "type": "fixed_profile_diagnostic",
        "schedule": [float(x) for x in schedule],
        "batches": seen,
        "dataset_split": "train",
        "momentum": "cumulative_average",
        "weights_updated": False,
        "native_full_prefix": True,
        "active_bn_modules": active,
        "inactive_bn_modules": inactive,
        "min_active_bn_batches": min_batches,
        "max_active_bn_batches": max_batches,
    }

    output_path = Path(args.output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output_path)

    print("=" * 76)
    print(f"Saved: {output_path}")
    print(f"active BN modules   : {active}")
    print(f"inactive BN modules : {inactive}")
    print(f"active BN batches   : min={min_batches}, max={max_batches}")
    print("=" * 76)


if __name__ == "__main__":
    main()

