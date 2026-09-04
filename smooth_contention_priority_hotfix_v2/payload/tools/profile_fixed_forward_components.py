#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

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
    first_elastic_stage,
    _native_res_stage,
    _native_fpn,
    _native_stereo,
)
from test_stream_buffer_timestamp import build_scene_index, load_one
from smooth_cuda_contention import SmoothCudaContention, make_high_priority_detector_stream, stream_priority_range


torch.backends.cudnn.benchmark = True

STAGES = (
    "fixed_prefix",
    "res2",
    "res3",
    "res4",
    "fpn",
    "stereo",
    "rpn",
    "fixed_tail",
)

BOUNDARIES = (
    "start",
    "after_prefix",
    "after_res2",
    "after_res3",
    "after_res4",
    "after_fpn",
    "after_stereo",
    "after_rpn",
    "forward_end",
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Forward-only CUDA-event profiler for one materialized Elastic-v4-BN "
            "fixed schedule. Data loading/H2D and post-processing/NMS are excluded."
        )
    )
    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--raw_csv", required=True)
    p.add_argument("--summary_json", required=True)
    p.add_argument("--contention-strength", type=float, default=0.0)
    p.add_argument("--contention-window-ms", type=float, default=100.0)
    p.add_argument("--contention-slice-ms", type=float, default=0.25)
    p.add_argument("--contention-start-delay-ms", type=float, default=2.0)
    p.add_argument("--contention-threads", type=int, default=256)
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


def pct(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(x.size),
        "mean_ms": float(np.mean(x)),
        "p50_ms": pct(x, 50),
        "p90_ms": pct(x, 90),
        "p99_ms": pct(x, 99),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def new_event():
    return torch.cuda.Event(enable_timing=True)


def record():
    e = new_event()
    e.record()
    return e


def fill_history(queue, bev):
    if queue is None:
        return
    # Keep the history semantics simple and deterministic for timing:
    # three representative processed BEV states.
    queue.clear()
    for i in range(3):
        queue.append((f"profile_history_{i}", {"spatial_features": bev.detach().clone()}))


def run_forward_tail_only(base_model, batch_dict, bev, valids):
    """
    K3 temporal fusion + VAN + detection head only.

    IMPORTANT:
      * This deliberately stops BEFORE base_model.post_processing().
      * Thus NMS / post-processing are excluded from every timing number.
    """
    cur_data = batch_dict["token"]
    cur_data["spatial_features"] = bev
    cur_data["spatial_features_stride"] = 1
    cur_data["valids"] = valids
    cur_data["history_features"] = base_model.history_feature_queue

    history_feature = {"spatial_features": bev.detach().clone()}

    for module in base_model.fusion_module:
        cur_data = module(cur_data)
    for module in base_model.after_fusion_blocks:
        cur_data = module(cur_data)

    if base_model.history_feature_queue is not None:
        base_model.history_feature_queue.append(
            (cur_data["this_sample_idx"], history_feature)
        )
    return cur_data


def native_full_rpn(branch, frame, stereo, full_backbone):
    """
    Native Full RPN/geometry path for the current K3 configuration.

    This mirrors StreamDSGN2Backbone's inference path under the assumptions
    already enforced by ElasticBEVBranch:
      cat_img_feature=False
      cat_right_img_feature=False
      one RPN3D hourglass
      stereo output type == feature in the current K3 config

    HeightCompression is only a reshape for the current dense volume, so the
    BEV tensor is produced directly without invoking post-processing.
    """
    norm_coord_imgs, valids = branch._geometry_grid(
        frame, full_backbone, stereo.dtype
    )
    voxel = F.grid_sample(stereo, norm_coord_imgs, align_corners=True)
    voxel = voxel * valids.float()[:, None].to(voxel.dtype)

    voxel = full_backbone.rpn3d_convs(voxel)

    if int(full_backbone.num_3dconvs_hg) > 0:
        if int(full_backbone.num_3dconvs_hg) != 1:
            raise RuntimeError(
                "Profiler currently expects the K3 config with one RPN3D hourglass"
            )
        pre, post = True, True
        for hg in full_backbone.rpn3d_hgs:
            voxel, pre, post = hg(voxel, pre, post)

    voxel = full_backbone.rpn3d_pool(voxel)
    n, c, d, h, w = voxel.shape
    bev = voxel.view(n, c * d, h, w)
    return bev, valids


def stagewise_forward(base_model, branch, batch_dict, schedule):
    """
    Execute one schedule and place CUDA events at logical forward boundaries.

    There is NO cuda.synchronize() between stages.  All events are recorded on
    the default stream and one synchronize is issued only after forward_end.
    Therefore the profiler does not inject six per-stage synchronization
    barriers into the measured forward path.
    """
    frame = batch_dict["token"]
    backbone = base_model.backbone_3d
    amp_enabled = bool(base_model.use_amp_dict["TEST"])

    events = {}

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=amp_enabled):
        events["start"] = record()

        prefix = extract_fixed_layer1_prefix(backbone, frame)
        events["after_prefix"] = record()

        first = first_elastic_stage(schedule)

        state = {
            "left_l1": prefix["left_l1"],
            "right_l1": prefix["right_l1"],
        }

        # Res2
        if first is None or first > 0:
            state["left_l2"], state["right_l2"] = _native_res_stage(
                backbone.feature_backbone,
                "layer2",
                state["left_l1"],
                state["right_l1"],
            )
        else:
            state = branch.stage_res2(prefix, schedule[0])
        events["after_res2"] = record()

        # Res3
        if first is None or first > 1:
            state["left_l3"], state["right_l3"] = _native_res_stage(
                backbone.feature_backbone,
                "layer3",
                state["left_l2"],
                state["right_l2"],
            )
        else:
            state = branch.stage_res3(state, schedule[1])
        events["after_res3"] = record()

        # Res4
        if first is None or first > 2:
            state["left_l4"], state["right_l4"] = _native_res_stage(
                backbone.feature_backbone,
                "layer4",
                state["left_l3"],
                state["right_l3"],
            )
        else:
            state = branch.stage_res4(state, schedule[2])
        events["after_res4"] = record()

        # FPN
        if first is None or first > 3:
            state = _native_fpn(backbone, frame, state)
        else:
            state = branch.stage_fpn(frame, state, schedule[3])
        events["after_fpn"] = record()

        # Stereo
        if first is None or first > 4:
            stereo = _native_stereo(backbone, frame, state)
        else:
            stereo = branch.stage_stereo(
                frame, state, backbone, schedule[4]
            )
        events["after_stereo"] = record()

        # RPN / geometry
        if first is None:
            bev, valids = native_full_rpn(
                branch, frame, stereo, backbone
            )
        else:
            # Any legal non-Full monotonic profile has elastic RPN < 1.0.
            bev, valids = branch.stage_rpn(
                frame, stereo, backbone, schedule[5]
            )
        events["after_rpn"] = record()

        run_forward_tail_only(base_model, batch_dict, bev, valids)
        events["forward_end"] = record()

    events["forward_end"].synchronize()
    return events, bev


def event_ms(a, b):
    return float(a.elapsed_time(b))


def row_from_events(frame_index, dataset_index, schedule, events):
    b = events
    row = {
        "frame_index": int(frame_index),
        "dataset_index": int(dataset_index),
        "schedule": ",".join(str(float(x)) for x in schedule),
        "fixed_prefix_ms": event_ms(b["start"], b["after_prefix"]),
        "res2_ms": event_ms(b["after_prefix"], b["after_res2"]),
        "res3_ms": event_ms(b["after_res2"], b["after_res3"]),
        "res4_ms": event_ms(b["after_res3"], b["after_res4"]),
        "fpn_ms": event_ms(b["after_res4"], b["after_fpn"]),
        "stereo_ms": event_ms(b["after_fpn"], b["after_stereo"]),
        "rpn_ms": event_ms(b["after_stereo"], b["after_rpn"]),
        "fixed_tail_ms": event_ms(b["after_rpn"], b["forward_end"]),
        "forward_total_ms": event_ms(b["start"], b["forward_end"]),
        # Direct remaining-path measurements from each checkpoint.
        "remain_after_prefix_ms": event_ms(b["after_prefix"], b["forward_end"]),
        "remain_after_res2_ms": event_ms(b["after_res2"], b["forward_end"]),
        "remain_after_res3_ms": event_ms(b["after_res3"], b["forward_end"]),
        "remain_after_res4_ms": event_ms(b["after_res4"], b["forward_end"]),
        "remain_after_fpn_ms": event_ms(b["after_fpn"], b["forward_end"]),
        "remain_after_stereo_ms": event_ms(b["after_stereo"], b["forward_end"]),
        "remain_after_rpn_ms": event_ms(b["after_rpn"], b["forward_end"]),
    }
    return row


def choose_scene_indices(dataset, needed):
    scene_to_indices = build_scene_index(dataset)
    if not scene_to_indices:
        raise RuntimeError("Empty scene index")

    # Prefer one long scene so temporal-history timing is stable and no scene
    # reset occurs inside the main measurement block.
    scenes = sorted(
        scene_to_indices.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    scene_name, indices = scenes[0]
    if not indices:
        raise RuntimeError("Longest scene is empty")

    # If requested samples exceed the longest scene, repeat it.  Data loading
    # is outside timing; history remains representative of processed frames.
    out = []
    while len(out) < needed:
        out.extend(indices)
    return scene_name, out[:needed]


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    schedule = parse_schedule(args.schedule)
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
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )
    base_model.cuda().eval()

    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(c.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES),
    ).cuda().eval()

    ckpt = torch.load(args.elastic_ckpt, map_location="cpu")
    branch.load_state_dict(ckpt["branch"], strict=True)

    fused_counts = None
    if tuple(schedule) != FULL_SCHEDULE:
        fused_counts = enable_fused_bn_static_cache(branch)

    needed = int(args.warmup) + int(args.frames)
    scene_name, indices = choose_scene_indices(dataset, needed)

    if base_model.history_feature_queue is not None:
        base_model.history_feature_queue.clear()

    contender = None
    if float(args.contention_strength) > 0:
        contender = SmoothCudaContention(strength=float(args.contention_strength),window_ms=float(args.contention_window_ms),slice_ms=float(args.contention_slice_ms),start_delay_ms=float(args.contention_start_delay_ms),threads=int(args.contention_threads),device=torch.cuda.current_device())
    model_stream = make_high_priority_detector_stream(
        torch.cuda.current_device()
    )
    print(
        "[CUDA] stream priority range (least, greatest):",
        stream_priority_range(),
    )
    print(
        "[CUDA] detector stream priority:",
        int(model_stream.priority),
    )
    def run_one(batch, check_window):
        ready=torch.cuda.Event(enable_timing=False); ready.record(torch.cuda.current_stream())
        if contender is not None: contender.launch()
        with torch.cuda.stream(model_stream):
            model_stream.wait_event(ready); events,bev=stagewise_forward(base_model,branch,batch,schedule)
        fwd=event_ms(events["start"],events["forward_end"])
        if contender is not None:
            if check_window: contender.assert_covers_forward(fwd,margin_ms=2.0)
            contender.finish()
        return events,bev
    last_bev=None
    for k in range(int(args.warmup)):
        batch=load_one(dataset,indices[k]); _,last_bev=run_one(batch,False)
    if last_bev is not None: fill_history(base_model.history_feature_queue,last_bev)
    rows=[]; offset=int(args.warmup)
    for i in range(int(args.frames)):
        dataset_index=indices[offset+i]; batch=load_one(dataset,dataset_index); events,_=run_one(batch,True); row=row_from_events(i,dataset_index,schedule,events); rows.append(row)
        if i<5 or (i+1)%20==0: print(f"[{i+1:04d}/{args.frames:04d}] forward={row['forward_total_ms']:.3f} ms prefix={row['fixed_prefix_ms']:.3f} ms schedule={tuple(schedule)}")

    raw_path = Path(args.raw_csv)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "fixed_prefix_ms",
        "res2_ms",
        "res3_ms",
        "res4_ms",
        "fpn_ms",
        "stereo_ms",
        "rpn_ms",
        "fixed_tail_ms",
        "forward_total_ms",
        "remain_after_prefix_ms",
        "remain_after_res2_ms",
        "remain_after_res3_ms",
        "remain_after_res4_ms",
        "remain_after_fpn_ms",
        "remain_after_stereo_ms",
        "remain_after_rpn_ms",
    ]

    summary = {
        "timing_scope": (
            "forward only: fixed stem+layer1 + Res2/3/4 + FPN + Stereo + RPN "
            "+ K3 fusion + VAN + detection head; data loading/H2D and "
            "post_processing/NMS excluded"
        ),
        "measurement": (
            "CUDA events at stage boundaries; no per-stage cuda.synchronize; "
            "one synchronization after forward_end"
        ),
        "schedule": [float(x) for x in schedule],
        "checkpoint": args.elastic_ckpt,
        "scene": scene_name,
        "warmup": int(args.warmup),
        "frames": int(args.frames),
        "fused_modules": fused_counts,
        "contention": None if contender is None else contender.config(),
        "metrics": {
            name: stats([r[name] for r in rows])
            for name in metric_names
        },
        "raw_csv": str(raw_path),
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )

    print("\n" + "=" * 80)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
