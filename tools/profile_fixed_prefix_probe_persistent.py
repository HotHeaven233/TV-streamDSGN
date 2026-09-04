#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    extract_fixed_layer1_prefix,
)
from test_stream_buffer_timestamp import build_scene_index, load_one
from persistent_cuda_contention import PersistentCudaContention


torch.backends.cudnn.benchmark = True


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Profile fixed Stem+Layer1 under one finite persistent contention "
            "window per sample."
        )
    )
    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--workers", type=int, default=0)

    p.add_argument("--contention-strength", type=float, default=0.0)
    p.add_argument("--contention-duration-ms", type=float, default=100.0)
    p.add_argument("--contention-start-delay-ms", type=float, default=5.0)
    p.add_argument("--contention-threads", type=int, default=256)

    p.add_argument("--output_json", required=True)
    return p.parse_args()


def make_cfg(path):
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = Path(path).stem
    cfg.DATA_CONFIG.INFER_TIME_PATH = None
    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None
    return cfg


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean_ms": float(np.mean(x)),
        "std_ms": float(np.std(x)),
        "p10_ms": float(np.percentile(x, 10)),
        "p50_ms": float(np.percentile(x, 50)),
        "p90_ms": float(np.percentile(x, 90)),
        "p99_ms": float(np.percentile(x, 99)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def choose_indices(dataset, needed):
    scene_to_indices = build_scene_index(dataset)
    scenes = sorted(
        scene_to_indices.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    if not scenes:
        raise RuntimeError("No scenes found")
    scene, idx = scenes[0]
    out = []
    while len(out) < needed:
        out.extend(idx)
    return scene, out[:needed]


def one_probe(backbone, frame, amp_enabled, model_stream, input_ready_event):
    """
    Run the probe on a dedicated NON-DEFAULT CUDA stream.

    This is essential when a persistent contender is active.  Using the legacy
    default stream can introduce implicit stream ordering and make the probe
    wait for the entire contention kernel instead of overlapping with it.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.cuda.stream(model_stream):
        model_stream.wait_event(input_ready_event)
        start.record()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            _ = extract_fixed_layer1_prefix(backbone, frame)
        end.record()

    # Wait only for the detector stream endpoint.
    end.synchronize()
    return float(start.elapsed_time(end))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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

    model = build_network(
        model_cfg=c.MODEL,
        num_class=len(c.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )
    model.cuda().eval()

    amp_enabled = bool(model.use_amp_dict["TEST"])
    needed = args.warmup + args.frames
    scene, indices = choose_indices(dataset, needed)

    # Never run the measured detector path on the legacy default stream while
    # the persistent contender is active.  Both detector and contender use
    # independent non-default streams.
    model_stream = torch.cuda.Stream(
        device=torch.cuda.current_device()
    )

    contender = None
    if args.contention_strength > 0:
        contender = PersistentCudaContention(
            strength=args.contention_strength,
            duration_ms=args.contention_duration_ms,
            start_delay_ms=args.contention_start_delay_ms,
            threads=args.contention_threads,
            device=torch.cuda.current_device(),
        )

    def measured_probe(dataset_index):
        batch = load_one(dataset, dataset_index)

        # load_one/H2D may have used the current/default stream.  Record an
        # explicit readiness event before entering the detector stream.
        input_ready = torch.cuda.Event(enable_timing=False)
        input_ready.record(torch.cuda.current_stream())

        if contender is not None:
            contender.launch()

        value = one_probe(
            model.backbone_3d,
            batch["token"],
            amp_enabled,
            model_stream,
            input_ready,
        )

        if contender is not None:
            contender.assert_covers_forward(
                value,
                margin_ms=1.0,
            )
            contender.finish()

        return value

    for i in range(args.warmup):
        _ = measured_probe(indices[i])

    vals = []
    for i in range(args.frames):
        vals.append(measured_probe(indices[args.warmup + i]))

    result = {
        "scope": "fixed Stem+Layer1 only, CUDA-event GPU time",
        "scene": scene,
        "warmup": args.warmup,
        "frames": args.frames,
        "amp_enabled": amp_enabled,
        "contention": None if contender is None else contender.config(),
        "probe": stats(vals),
        "raw_ms": vals,
    }

    p = Path(args.output_json)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    compact = dict(result)
    compact.pop("raw_ms")
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

