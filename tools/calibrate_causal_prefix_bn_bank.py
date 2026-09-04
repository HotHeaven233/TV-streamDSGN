#!/usr/bin/env python3
"""
Calibrate all 203 causal-prefix BN states from one raw Elastic-v4-BN checkpoint.

Important:
  * train split only;
  * convolution weights and BN affine gamma/beta are frozen;
  * calibration is stage-by-stage;
  * when calibrating stage j, all earlier elastic stages use their already
    calibrated causal-prefix statistics;
  * only the current stage's active BN statistics update;
  * future stages are not executed, avoiding unnecessary compute.

This produces a BN BANK, not a network checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from test_stream_buffer_timestamp import load_one

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    ElasticBEVBranch,
    extract_fixed_layer1_prefix,
)
from pcdet.models.backbones_3d_stream.elastic_v4_bn_hybrid import (
    _native_fpn,
    _native_res_stage,
    _native_stereo,
)
from pcdet.models.backbones_3d_stream.causal_prefix_bn_bank import (
    STAGE_NAMES,
    apply_earlier_prefix_stats,
    capture_stage_stats,
    causal_prefixes_by_stage,
    discovered_stage_bn_counts,
    prefix_key,
    reset_stage_active_bn,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--output_bank", required=True)
    p.add_argument("--batches_per_prefix", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing partially calibrated bank.",
    )
    p.add_argument("--save_every", type=int, default=5)
    return p.parse_args()


def make_cfg(path):
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = Path(path).stem
    cfg.DATA_CONFIG.INFER_TIME_PATH = None
    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None
    return cfg


@torch.no_grad()
def forward_to_stage(branch, frame, prefix_cache, full_backbone, prefix):
    """
    Execute exactly the causal prefix and stop after its final stage.
    Leading 1.0 stages are native Full.
    """
    target = len(prefix) - 1
    first = next(i for i, r in enumerate(prefix) if float(r) < 1.0)

    state = {
        "left_l1": prefix_cache["left_l1"],
        "right_l1": prefix_cache["right_l1"],
    }

    # Res2
    if target >= 0:
        if first > 0:
            state["left_l2"], state["right_l2"] = _native_res_stage(
                full_backbone.feature_backbone,
                "layer2",
                state["left_l1"],
                state["right_l1"],
            )
        else:
            state = branch.stage_res2(prefix_cache, prefix[0])
        if target == 0:
            return state

    # Res3
    if target >= 1:
        if first > 1:
            state["left_l3"], state["right_l3"] = _native_res_stage(
                full_backbone.feature_backbone,
                "layer3",
                state["left_l2"],
                state["right_l2"],
            )
        else:
            state = branch.stage_res3(state, prefix[1])
        if target == 1:
            return state

    # Res4
    if target >= 2:
        if first > 2:
            state["left_l4"], state["right_l4"] = _native_res_stage(
                full_backbone.feature_backbone,
                "layer4",
                state["left_l3"],
                state["right_l3"],
            )
        else:
            state = branch.stage_res4(state, prefix[2])
        if target == 2:
            return state

    # FPN
    if target >= 3:
        if first > 3:
            state = _native_fpn(full_backbone, frame, state)
        else:
            state = branch.stage_fpn(frame, state, prefix[3])
        if target == 3:
            return state

    # Stereo
    if target >= 4:
        if first > 4:
            stereo = _native_stereo(full_backbone, frame, state)
        else:
            stereo = branch.stage_stereo(
                frame, state, full_backbone, prefix[4]
            )
        if target == 4:
            return stereo

    # RPN
    return branch.stage_rpn(
        frame,
        stereo,
        full_backbone,
        prefix[5],
    )


def new_bank(args, source_checkpoint):
    return {
        "version": "elastic_v4_bn_causal_prefix_bank_v1",
        "full_cfg": args.full_cfg,
        "full_ckpt": args.full_ckpt,
        "elastic_ckpt": args.elastic_ckpt,
        "source_epoch": source_checkpoint.get("epoch"),
        "batches_per_prefix": int(args.batches_per_prefix),
        "dataset_split": "train (eval preprocessing; no augmentation)",
        "affine_policy": "shared per width; bank stores running stats only",
        "stage_counts": {
            "res2": 3, "res3": 9, "res4": 19,
            "fpn": 34, "stereo": 55, "rpn": 83,
        },
        "stages": {name: {} for name in STAGE_NAMES},
    }


def save_bank(bank, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, path)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    c = make_cfg(args.full_cfg)
    logger = common_utils.create_logger()

    # BN calibration must use TRAIN frames but inference/eval preprocessing.
    # `training=True` would enable GT sampling / random flip / crop / rotation /
    # scaling, which biases running statistics away from deployment inputs.
    dataset_cfg = copy.deepcopy(c.DATA_CONFIG)
    dataset_cfg.DATA_SPLIT["test"] = dataset_cfg.DATA_SPLIT["train"]
    dataset_cfg.INFO_PATH["test"] = copy.deepcopy(
        dataset_cfg.INFO_PATH["train"]
    )
    dataset, _, _ = build_dataloader(
        dataset_cfg=dataset_cfg,
        class_names=c.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("Empty train-split eval dataset")

    base_model = build_network(
        model_cfg=c.MODEL,
        num_class=len(c.CLASS_NAMES),
        dataset=dataset,
    )
    base_model.load_params_from_file(
        filename=args.full_ckpt, logger=logger, to_cpu=True
    )
    base_model.cuda().eval()
    for p in base_model.parameters():
        p.requires_grad = False

    source = torch.load(args.elastic_ckpt, map_location="cpu")
    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(c.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES),
    ).cuda().eval()
    branch.load_state_dict(source["branch"], strict=True)
    for p in branch.parameters():
        p.requires_grad = False

    stage_bn_counts, unknown_bns = discovered_stage_bn_counts(branch)
    print(f"[BN-MAP] discovered switchable-BN modules: {stage_bn_counts}")
    if unknown_bns:
        raise RuntimeError(
            "Unmapped switchable BN modules: " + ", ".join(unknown_bns)
        )
    if any(stage_bn_counts[name] <= 0 for name in STAGE_NAMES):
        raise RuntimeError(
            f"Missing switchable BN stage(s): {stage_bn_counts}"
        )

    output_path = Path(args.output_bank)
    if args.resume and output_path.exists():
        bank = torch.load(output_path, map_location="cpu")
        if bank.get("version") != "elastic_v4_bn_causal_prefix_bank_v1":
            raise RuntimeError("Existing bank version mismatch")
        print(f"[RESUME] {output_path}")
    else:
        bank = new_bank(args, source)

    prefixes = causal_prefixes_by_stage()
    total = sum(len(v) for v in prefixes.values())
    already = sum(
        len(bank["stages"].get(name, {})) for name in STAGE_NAMES
    )
    print("=" * 80)
    print("Elastic-v4 causal-prefix BN-bank calibration")
    print(f"train samples       : {len(dataset)} (eval preprocessing)")
    print(f"prefix states       : {total} (expected 203)")
    print(f"already calibrated  : {already}")
    print(f"frames/prefix       : {args.batches_per_prefix} (evenly spaced, shared across prefixes)")
    print(f"output              : {output_path}")
    print("=" * 80)

    # Every causal prefix sees the same deterministic, evenly-spaced training
    # frames. This avoids giving different prefixes different scene/content
    # distributions merely because calibration is sequential.
    calib_n = int(args.batches_per_prefix)
    calib_indices = np.linspace(
        0,
        len(dataset) - 1,
        num=calib_n,
        dtype=np.int64,
    ).tolist()

    amp_enabled = bool(c.MODEL.USE_AMP.get("TEST", True))
    completed_since_save = 0
    global_done = already
    start_all = time.time()

    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        stage_prefixes = prefixes[stage_name]
        print(
            f"\n[STAGE {stage_idx+1}/6] {stage_name}: "
            f"{len(stage_prefixes)} causal prefixes"
        )
        for prefix in stage_prefixes:
            pkey = prefix_key(prefix)
            if pkey in bank["stages"][stage_name]:
                continue

            # Every prefix is calibrated independently at the current stage.
            # Earlier stages use their exact already-calibrated prefix stats.
            branch.eval()
            apply_earlier_prefix_stats(
                branch, bank, prefix, current_stage_idx=stage_idx
            )
            active = reset_stage_active_bn(
                branch,
                stage_idx,
                prefix[-1],
                cumulative=True,
            )

            t0 = time.time()
            for seen, dataset_index in enumerate(calib_indices, start=1):
                batch = load_one(dataset, int(dataset_index))
                with torch.no_grad(), torch.cuda.amp.autocast(
                    enabled=amp_enabled
                ):
                    prefix_cache = extract_fixed_layer1_prefix(
                        base_model.backbone_3d,
                        batch["token"],
                    )
                    forward_to_stage(
                        branch,
                        batch["token"],
                        prefix_cache,
                        base_model.backbone_3d,
                        prefix,
                    )

            stats = capture_stage_stats(branch, stage_idx, prefix)
            batch_counts = [
                int(bn.num_batches_tracked.item()) for _, bn in active
            ]
            bank["stages"][stage_name][pkey] = {
                "prefix": [float(x) for x in prefix],
                "stage_index": stage_idx,
                "frames": int(args.batches_per_prefix),
                "active_bn_num_batches_min": min(batch_counts),
                "active_bn_num_batches_max": max(batch_counts),
                "bn": stats,
            }

            global_done += 1
            completed_since_save += 1
            elapsed = time.time() - t0
            print(
                f"[{global_done:03d}/{total:03d}] "
                f"{stage_name}:{pkey:<23s} "
                f"{elapsed:6.1f}s "
                f"BNbatches={min(batch_counts)}..{max(batch_counts)}"
            )

            if completed_since_save >= max(1, args.save_every):
                save_bank(bank, output_path)
                completed_since_save = 0

        save_bank(bank, output_path)
        completed_since_save = 0

    bank["completed"] = True
    bank["completed_prefix_states"] = total
    bank["calibration_wall_seconds"] = float(time.time() - start_all)
    save_bank(bank, output_path)

    print("=" * 80)
    print(f"[DONE] saved causal-prefix BN bank: {output_path}")
    print(f"states: {total}/203")
    print(
        f"wall time: {bank['calibration_wall_seconds']/60.0:.1f} min "
        "(this run, excluding any resumed work)"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()

