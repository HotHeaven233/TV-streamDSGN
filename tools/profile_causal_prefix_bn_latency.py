#!/usr/bin/env python3
"""
Short service-time benchmark for a materialized causal-prefix BN profile.

Timing scope matches the streaming evaluator:
    model forward + post-processing; data loading/H2D excluded.

mode=fused:
    Conv+selected prefix-BN statistics are fused into lazy static contiguous
    kernels.  This is the intended deployment path.

mode=bn_eval:
    Keep eval-mode BatchNorm kernels to quantify the fusion benefit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    ElasticBEVBranch,
    extract_fixed_layer1_prefix,
    validate_schedule,
)
from pcdet.models.backbones_3d_stream.elastic_v4_bn_hybrid import (
    FULL_SCHEDULE,
    enable_fused_bn_static_cache,
    hybrid_forward,
)
from test_elastic_stream_v4_bn import run_elastic_downstream
from test_stream_buffer_timestamp import build_scene_index, load_one


torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--mode", choices=("fused", "bn_eval"), default="fused")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--frames", type=int, default=160)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--output", default=None)
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


class RuntimeModel(nn.Module):
    def __init__(self, base_model, branch, schedule):
        super().__init__()
        self.base_model = base_model
        self.branch = branch
        self.schedule = tuple(schedule)

    def forward(self, batch_dict):
        cur = batch_dict["token"]
        if (
            self.base_model.history_feature_queue is not None
            and ("prev_sample_idx" not in cur or cur["prev_sample_idx"] == "")
        ):
            self.base_model.history_feature_queue.clear()

        if tuple(self.schedule) == FULL_SCHEDULE:
            return self.base_model(batch_dict)

        backbone = self.base_model.backbone_3d
        amp_enabled = bool(self.base_model.use_amp_dict["TEST"])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            prefix = extract_fixed_layer1_prefix(backbone, cur)
            bev, valids = hybrid_forward(
                self.branch,
                cur,
                prefix,
                backbone,
                self.schedule,
            )
            return run_elastic_downstream(
                self.base_model, batch_dict, bev, valids
            )


def pct(x, q):
    return float(np.percentile(np.asarray(x, dtype=np.float64), q))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    c = make_cfg(args.full_cfg)
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(
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
        filename=args.full_ckpt, logger=logger, to_cpu=True
    )
    base_model.cuda().eval()

    schedule = parse_schedule(args.schedule)
    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(c.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES),
    ).cuda().eval()
    ckpt = torch.load(args.elastic_ckpt, map_location="cpu")
    branch.load_state_dict(ckpt["branch"], strict=True)

    fused_counts = None
    if args.mode == "fused" and tuple(schedule) != FULL_SCHEDULE:
        fused_counts = enable_fused_bn_static_cache(branch)

    runtime = RuntimeModel(base_model, branch, schedule).cuda().eval()
    scene_to_indices = build_scene_index(dataset)
    ordered = [idx for indices in scene_to_indices.values() for idx in indices]
    if not ordered:
        raise RuntimeError("Empty dataset")

    warm_idx = ordered[min(5, len(ordered)-1)]
    for _ in range(max(0, args.warmup)):
        if base_model.history_feature_queue is not None:
            base_model.history_feature_queue.clear()
        batch = load_one(dataset, warm_idx)
        with torch.no_grad():
            runtime(batch)
    torch.cuda.synchronize()

    values = []
    total = min(int(args.frames), len(ordered))
    for i, dataset_index in enumerate(ordered[:total]):
        batch = load_one(dataset, dataset_index)
        torch.cuda.synchronize()
        t0 = time.perf_counter_ns()
        with torch.no_grad():
            runtime(batch)
        torch.cuda.synchronize()
        ms = (time.perf_counter_ns() - t0) / 1e6
        values.append(ms)
        if i < 10 or (i + 1) % 20 == 0:
            print(
                f"[{i+1:04d}/{total:04d}] service={ms:7.3f}ms "
                f"mode={args.mode} schedule={tuple(schedule)}"
            )

    stable = values[2:] if len(values) > 4 else values
    result = {
        "mode": args.mode,
        "schedule": [float(x) for x in schedule],
        "checkpoint": args.elastic_ckpt,
        "frames": len(values),
        "stable_frames": len(stable),
        "mean_ms": float(np.mean(stable)),
        "p50_ms": pct(stable, 50),
        "p90_ms": pct(stable, 90),
        "p99_ms": pct(stable, 99),
        "min_ms": float(np.min(stable)),
        "max_ms": float(np.max(stable)),
        "fused_modules": fused_counts,
    }
    print("\n" + "=" * 76)
    print(json.dumps(result, indent=2))
    print("=" * 76)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

