#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
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
)
from pcdet.models.backbones_3d_stream.elastic_v4_bn_hybrid import (
    _native_res_stage,
    _native_fpn,
    _native_stereo,
)
from test_stream_buffer_timestamp import (
    build_scene_index,
    load_one,
)

from tv_stream3d_controller import TVStream3DController
from tv_stream3d_causal_fused_runtime import (
    begin_stage_prefix,
    enable_causal_prefix_fused_cache,
    finish_stage_prefix,
    fused_cache_stats,
    set_forbid_cache_miss,
)
from smooth_cuda_contention import (
    SmoothCudaContention,
    make_high_priority_detector_stream,
)


STAGE_NAMES = (
    "res2",
    "res3",
    "res4",
    "fpn",
    "stereo",
    "rpn",
)

CHECKPOINT_AFTER_STAGE = (
    "after_res2",
    "after_res3",
    "after_res4",
    "after_fpn",
    "after_stereo",
    "after_rpn",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--prefix_bn_bank", required=True)
    p.add_argument("--controller_csv", required=True)
    p.add_argument("--levels_json", required=True)

    p.add_argument("--expected_level", default="L0")
    p.add_argument("--input_hz", type=float, default=35.0)

    p.add_argument("--warmup_frames", type=int, default=80)
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument(
        "--control_guard_per_boundary_ms",
        type=float,
        default=0.25,
        help=(
            "Empirical reserve added for each future rolling-control "
            "boundary. Default 0.25 ms is a provisional guard derived "
            "from the L0/L4 online smoke overhead."
        ),
    )

    p.add_argument("--output_csv", required=True)
    p.add_argument("--output_json", required=True)
    return p.parse_args()


def make_cfg(path):
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = Path(path).stem
    cfg.DATA_CONFIG.INFER_TIME_PATH = None
    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None
    return cfg


def choose_scene_indices(dataset, needed):
    scene_to_indices = build_scene_index(dataset)
    scenes = sorted(
        scene_to_indices.items(),
        key=lambda kv: len(kv[1]),
        reverse=True,
    )
    if not scenes:
        raise RuntimeError("No stream scenes found")

    scene, indices = scenes[0]
    out = []
    while len(out) < needed:
        out.extend(indices)

    return scene, out[:needed]


def record_event():
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def elapsed_ms(start, end):
    return float(start.elapsed_time(end))


def native_full_rpn(branch, frame, stereo, full_backbone):
    norm_coord_imgs, valids = branch._geometry_grid(
        frame,
        full_backbone,
        stereo.dtype,
    )
    voxel = F.grid_sample(
        stereo,
        norm_coord_imgs,
        align_corners=True,
    )
    voxel = voxel * valids.float()[:, None].to(voxel.dtype)

    voxel = full_backbone.rpn3d_convs(voxel)

    if int(full_backbone.num_3dconvs_hg) > 0:
        if int(full_backbone.num_3dconvs_hg) != 1:
            raise RuntimeError(
                "Expected current K3 config with one RPN3D hourglass"
            )

        pre, post = True, True
        for hg in full_backbone.rpn3d_hgs:
            voxel, pre, post = hg(
                voxel,
                pre,
                post,
            )

    voxel = full_backbone.rpn3d_pool(voxel)
    n, c, d, h, w = voxel.shape
    bev = voxel.view(n, c * d, h, w)
    return bev, valids


def run_forward_tail_only(base_model, batch_dict, bev, valids):
    """
    Exact timing scope used by the offline forward/suffix profiler:
    K3 fusion + VAN + detection head, but NO post_processing/NMS.
    """
    cur_data = batch_dict["token"]
    cur_data["spatial_features"] = bev
    cur_data["spatial_features_stride"] = 1
    cur_data["valids"] = valids
    cur_data["history_features"] = base_model.history_feature_queue

    history_feature = {
        "spatial_features": bev.detach().clone()
    }

    for module in base_model.fusion_module:
        cur_data = module(cur_data)

    for module in base_model.after_fusion_blocks:
        cur_data = module(cur_data)

    if base_model.history_feature_queue is not None:
        base_model.history_feature_queue.append(
            (
                cur_data["this_sample_idx"],
                history_feature,
            )
        )

    return cur_data


def configure_contender(levels_data, level):
    if level == "L0":
        return None

    by_name = {
        x["level"]: x
        for x in levels_data["levels"]
    }
    if level not in by_name:
        raise KeyError(level)

    lev = by_name[level]
    w = levels_data["workload"]

    return SmoothCudaContention(
        strength=float(lev["strength"]),
        window_ms=float(w["window_ms"]),
        slice_ms=float(w["slice_ms"]),
        start_delay_ms=float(w["start_delay_ms"]),
        threads=int(w["threads"]),
        device=torch.cuda.current_device(),
    )


def stage_decision(
    controller,
    checkpoint,
    start_event,
    checkpoint_event,
    observed_level,
    deadline_ms,
    executed,
):
    checkpoint_event.synchronize()
    elapsed = elapsed_ms(
        start_event,
        checkpoint_event,
    )

    t0 = time.perf_counter_ns()
    decision = controller.decide(
        checkpoint=checkpoint,
        elapsed_ms=elapsed,
        deadline_ms=deadline_ms,
        observed_level=observed_level,
        executed_prefix=tuple(executed),
    )
    controller_ms = (
        time.perf_counter_ns() - t0
    ) / 1e6

    return decision, elapsed, controller_ms



def prewarm_all_causal_prefixes(
    base_model,
    branch,
    bank,
    controller,
    batch,
    model_stream,
):
    """
    确定性遍历全部 84 条合法单调 schedule，
    materialize 所有非 Full 因果前缀对应的 fused Conv-BN cache。

    当前设计应恰好得到 203 个非 Full causal-prefix states。
    """
    frame = batch["token"]
    backbone = base_model.backbone_3d
    amp_enabled = bool(
        base_model.use_amp_dict["TEST"]
    )

    input_ready = torch.cuda.Event(
        enable_timing=False
    )
    input_ready.record(
        torch.cuda.current_stream()
    )

    expected_contexts = set()

    with torch.cuda.stream(model_stream):
        model_stream.wait_event(input_ready)

        # Deterministic cache prewarm is outside measured forward.
        # Run the entire prewarm path in FP32 so native FP32 modules and
        # their inputs always have matching dtypes.
        #
        # IMPORTANT: execute_one() keeps its original AMP behavior.
        with torch.no_grad():
            prefix = extract_fixed_layer1_prefix(
                backbone,
                frame,
            )

        for pid in controller.profile_ids:
            schedule = controller.profile_schedule[pid]

            state = {
                "left_l1": prefix["left_l1"],
                "right_l1": prefix["right_l1"],
            }

            elastic_started = False

            # --------------------------------------------------
            # Res2
            # --------------------------------------------------
            w = float(schedule[0])

            if not elastic_started and w >= 1.0:
                (
                    state["left_l2"],
                    state["right_l2"],
                ) = _native_res_stage(
                    backbone.feature_backbone,
                    "layer2",
                    state["left_l1"],
                    state["right_l1"],
                )
            else:
                elastic_started = True

                ctx = begin_stage_prefix(
                    branch,
                    bank,
                    0,
                    schedule[:1],
                    allow_prepare=True,
                )

                expected_contexts.add(ctx)

                state = branch.stage_res2(
                    prefix,
                    w,
                )

                finish_stage_prefix(
                    branch,
                    ctx,
                    allow_prepare=True,
                )

            # --------------------------------------------------
            # Res3
            # --------------------------------------------------
            w = float(schedule[1])

            if not elastic_started and w >= 1.0:
                (
                    state["left_l3"],
                    state["right_l3"],
                ) = _native_res_stage(
                    backbone.feature_backbone,
                    "layer3",
                    state["left_l2"],
                    state["right_l2"],
                )
            else:
                elastic_started = True

                ctx = begin_stage_prefix(
                    branch,
                    bank,
                    1,
                    schedule[:2],
                    allow_prepare=True,
                )

                expected_contexts.add(ctx)

                state = branch.stage_res3(
                    state,
                    w,
                )

                finish_stage_prefix(
                    branch,
                    ctx,
                    allow_prepare=True,
                )

            # --------------------------------------------------
            # Res4
            # --------------------------------------------------
            w = float(schedule[2])

            if not elastic_started and w >= 1.0:
                (
                    state["left_l4"],
                    state["right_l4"],
                ) = _native_res_stage(
                    backbone.feature_backbone,
                    "layer4",
                    state["left_l3"],
                    state["right_l3"],
                )
            else:
                elastic_started = True

                ctx = begin_stage_prefix(
                    branch,
                    bank,
                    2,
                    schedule[:3],
                    allow_prepare=True,
                )

                expected_contexts.add(ctx)

                state = branch.stage_res4(
                    state,
                    w,
                )

                finish_stage_prefix(
                    branch,
                    ctx,
                    allow_prepare=True,
                )

            # --------------------------------------------------
            # FPN
            # --------------------------------------------------
            w = float(schedule[3])

            if not elastic_started and w >= 1.0:
                state = _native_fpn(
                    backbone,
                    frame,
                    state,
                )
            else:
                elastic_started = True

                ctx = begin_stage_prefix(
                    branch,
                    bank,
                    3,
                    schedule[:4],
                    allow_prepare=True,
                )

                expected_contexts.add(ctx)

                state = branch.stage_fpn(
                    frame,
                    state,
                    w,
                )

                finish_stage_prefix(
                    branch,
                    ctx,
                    allow_prepare=True,
                )

            # --------------------------------------------------
            # Stereo
            # --------------------------------------------------
            w = float(schedule[4])

            if not elastic_started and w >= 1.0:
                stereo = _native_stereo(
                    backbone,
                    frame,
                    state,
                )
            else:
                elastic_started = True

                ctx = begin_stage_prefix(
                    branch,
                    bank,
                    4,
                    schedule[:5],
                    allow_prepare=True,
                )

                expected_contexts.add(ctx)

                stereo = branch.stage_stereo(
                    frame,
                    state,
                    backbone,
                    w,
                )

                finish_stage_prefix(
                    branch,
                    ctx,
                    allow_prepare=True,
                )

            # --------------------------------------------------
            # RPN
            # --------------------------------------------------
            w = float(schedule[5])

            if not elastic_started and w >= 1.0:
                bev, valids = native_full_rpn(
                    branch,
                    frame,
                    stereo,
                    backbone,
                )
            else:
                elastic_started = True

                ctx = begin_stage_prefix(
                    branch,
                    bank,
                    5,
                    schedule[:6],
                    allow_prepare=True,
                )

                expected_contexts.add(ctx)

                bev, valids = branch.stage_rpn(
                    frame,
                    stereo,
                    backbone,
                    w,
                )

                finish_stage_prefix(
                    branch,
                    ctx,
                    allow_prepare=True,
                )

            del state
            del stereo
            del bev
            del valids

    # 84 条 schedule 全部提交完成后只同步一次。
    model_stream.synchronize()

    actual_contexts = set(
        getattr(
            branch,
            "_tv_ready_prefixes",
            set(),
        )
    )

    missing = sorted(
        expected_contexts - actual_contexts
    )

    extra = sorted(
        actual_contexts - expected_contexts
    )

    if missing or extra:
        raise RuntimeError(
            "Deterministic causal-prefix prewarm "
            "coverage mismatch: "
            f"expected={len(expected_contexts)}, "
            f"actual={len(actual_contexts)}, "
            f"missing={missing[:5]}, "
            f"extra={extra[:5]}"
        )

    # 6 stages / 4 widths / 单调不增 schedule：
    # 非 Full causal-prefix 数必须正好为 203。
    if len(expected_contexts) != 203:
        raise RuntimeError(
            "Unexpected causal-prefix state count: "
            f"{len(expected_contexts)} "
            "(expected 203)"
        )

    return {
        "expected_prefixes": len(
            expected_contexts
        ),
        "ready_prefixes": len(
            actual_contexts
        ),
        "cache": fused_cache_stats(
            branch
        ),
    }


def execute_one(
    base_model,
    branch,
    bank,
    controller,
    batch,
    model_stream,
    contender,
    deadline_ms,
    allow_prepare,
):
    frame = batch["token"]
    backbone = base_model.backbone_3d
    amp_enabled = bool(base_model.use_amp_dict["TEST"])

    if (
        base_model.history_feature_queue is not None
        and (
            "prev_sample_idx" not in frame
            or frame["prev_sample_idx"] == ""
        )
    ):
        base_model.history_feature_queue.clear()

    input_ready = torch.cuda.Event(enable_timing=False)
    input_ready.record(torch.cuda.current_stream())

    if contender is not None:
        contender.launch()

    executed = []
    decisions = []
    controller_times = []
    elastic_started = False

    with torch.cuda.stream(model_stream):
        model_stream.wait_event(input_ready)

        start = record_event()

        with torch.no_grad(), torch.amp.autocast(
            "cuda",
            enabled=amp_enabled,
        ):
            prefix = extract_fixed_layer1_prefix(
                backbone,
                frame,
            )

        after_prefix = record_event()

    after_prefix.synchronize()
    prefix_ms = elapsed_ms(start, after_prefix)
    observed_level = controller.classify_probe(prefix_ms)

    d, elapsed, ctrl_ms = stage_decision(
        controller,
        "after_prefix",
        start,
        after_prefix,
        observed_level,
        deadline_ms,
        executed,
    )
    decisions.append(d)
    controller_times.append(ctrl_ms)

    chosen = float(d.next_width)

    state = {
        "left_l1": prefix["left_l1"],
        "right_l1": prefix["right_l1"],
    }

    # ------------------------------------------------------------------
    # Res2
    # ------------------------------------------------------------------
    executed.append(chosen)
    with torch.cuda.stream(model_stream), torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=amp_enabled,
    ):
        if not elastic_started and chosen >= 1.0:
            state["left_l2"], state["right_l2"] = _native_res_stage(
                backbone.feature_backbone,
                "layer2",
                state["left_l1"],
                state["right_l1"],
            )
        else:
            elastic_started = True
            ctx = begin_stage_prefix(
                branch,
                bank,
                0,
                tuple(executed),
                allow_prepare=allow_prepare,
            )
            state = branch.stage_res2(
                prefix,
                chosen,
            )
            finish_stage_prefix(
                branch,
                ctx,
                allow_prepare=allow_prepare,
            )

        after_res2 = record_event()

    d, elapsed, ctrl_ms = stage_decision(
        controller,
        "after_res2",
        start,
        after_res2,
        observed_level,
        deadline_ms,
        executed,
    )
    decisions.append(d)
    controller_times.append(ctrl_ms)
    chosen = float(d.next_width)

    # ------------------------------------------------------------------
    # Res3
    # ------------------------------------------------------------------
    executed.append(chosen)
    with torch.cuda.stream(model_stream), torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=amp_enabled,
    ):
        if not elastic_started and chosen >= 1.0:
            state["left_l3"], state["right_l3"] = _native_res_stage(
                backbone.feature_backbone,
                "layer3",
                state["left_l2"],
                state["right_l2"],
            )
        else:
            elastic_started = True
            ctx = begin_stage_prefix(
                branch,
                bank,
                1,
                tuple(executed),
                allow_prepare=allow_prepare,
            )
            state = branch.stage_res3(
                state,
                chosen,
            )
            finish_stage_prefix(
                branch,
                ctx,
                allow_prepare=allow_prepare,
            )

        after_res3 = record_event()

    d, elapsed, ctrl_ms = stage_decision(
        controller,
        "after_res3",
        start,
        after_res3,
        observed_level,
        deadline_ms,
        executed,
    )
    decisions.append(d)
    controller_times.append(ctrl_ms)
    chosen = float(d.next_width)

    # ------------------------------------------------------------------
    # Res4
    # ------------------------------------------------------------------
    executed.append(chosen)
    with torch.cuda.stream(model_stream), torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=amp_enabled,
    ):
        if not elastic_started and chosen >= 1.0:
            state["left_l4"], state["right_l4"] = _native_res_stage(
                backbone.feature_backbone,
                "layer4",
                state["left_l3"],
                state["right_l3"],
            )
        else:
            elastic_started = True
            ctx = begin_stage_prefix(
                branch,
                bank,
                2,
                tuple(executed),
                allow_prepare=allow_prepare,
            )
            state = branch.stage_res4(
                state,
                chosen,
            )
            finish_stage_prefix(
                branch,
                ctx,
                allow_prepare=allow_prepare,
            )

        after_res4 = record_event()

    d, elapsed, ctrl_ms = stage_decision(
        controller,
        "after_res4",
        start,
        after_res4,
        observed_level,
        deadline_ms,
        executed,
    )
    decisions.append(d)
    controller_times.append(ctrl_ms)
    chosen = float(d.next_width)

    # ------------------------------------------------------------------
    # FPN
    # ------------------------------------------------------------------
    executed.append(chosen)
    with torch.cuda.stream(model_stream), torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=amp_enabled,
    ):
        if not elastic_started and chosen >= 1.0:
            state = _native_fpn(
                backbone,
                frame,
                state,
            )
        else:
            elastic_started = True
            ctx = begin_stage_prefix(
                branch,
                bank,
                3,
                tuple(executed),
                allow_prepare=allow_prepare,
            )
            state = branch.stage_fpn(
                frame,
                state,
                chosen,
            )
            finish_stage_prefix(
                branch,
                ctx,
                allow_prepare=allow_prepare,
            )

        after_fpn = record_event()

    d, elapsed, ctrl_ms = stage_decision(
        controller,
        "after_fpn",
        start,
        after_fpn,
        observed_level,
        deadline_ms,
        executed,
    )
    decisions.append(d)
    controller_times.append(ctrl_ms)
    chosen = float(d.next_width)

    # ------------------------------------------------------------------
    # Stereo
    # ------------------------------------------------------------------
    executed.append(chosen)
    with torch.cuda.stream(model_stream), torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=amp_enabled,
    ):
        if not elastic_started and chosen >= 1.0:
            stereo = _native_stereo(
                backbone,
                frame,
                state,
            )
        else:
            elastic_started = True
            ctx = begin_stage_prefix(
                branch,
                bank,
                4,
                tuple(executed),
                allow_prepare=allow_prepare,
            )
            stereo = branch.stage_stereo(
                frame,
                state,
                backbone,
                chosen,
            )
            finish_stage_prefix(
                branch,
                ctx,
                allow_prepare=allow_prepare,
            )

        after_stereo = record_event()

    d, elapsed, ctrl_ms = stage_decision(
        controller,
        "after_stereo",
        start,
        after_stereo,
        observed_level,
        deadline_ms,
        executed,
    )
    decisions.append(d)
    controller_times.append(ctrl_ms)
    chosen = float(d.next_width)

    # ------------------------------------------------------------------
    # RPN
    # ------------------------------------------------------------------
    executed.append(chosen)
    with torch.cuda.stream(model_stream), torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=amp_enabled,
    ):
        if not elastic_started and chosen >= 1.0:
            bev, valids = native_full_rpn(
                branch,
                frame,
                stereo,
                backbone,
            )
        else:
            elastic_started = True
            ctx = begin_stage_prefix(
                branch,
                bank,
                5,
                tuple(executed),
                allow_prepare=allow_prepare,
            )
            bev, valids = branch.stage_rpn(
                frame,
                stereo,
                backbone,
                chosen,
            )
            finish_stage_prefix(
                branch,
                ctx,
                allow_prepare=allow_prepare,
            )

        after_rpn = record_event()

    # No runtime decision exists after RPN. Do NOT synchronize here: doing so
    # creates a pure host-side bubble before the fixed tail and is outside the
    # rolling-control semantics. The elapsed value can be read after the final
    # forward event is synchronized.
    with torch.cuda.stream(model_stream), torch.no_grad(), torch.amp.autocast(
        "cuda",
        enabled=amp_enabled,
    ):
        run_forward_tail_only(
            base_model,
            batch,
            bev,
            valids,
        )
        forward_end = record_event()

    forward_end.synchronize()
    forward_ms = elapsed_ms(
        start,
        forward_end,
    )
    after_rpn_ms = elapsed_ms(
        start,
        after_rpn,
    )

    if contender is not None:
        contender.finish()

    return {
        "forward_ms": float(forward_ms),
        "prefix_ms": float(prefix_ms),
        "after_rpn_ms": float(after_rpn_ms),
        "observed_level": observed_level,
        "schedule": tuple(executed),
        "controller_ms": float(
            sum(controller_times)
        ),
        "controller_calls": len(
            controller_times
        ),
        "all_decisions_feasible": bool(
            all(x.feasible for x in decisions)
        ),
        "first_profile_id": int(
            decisions[0].profile_id
        ),
        "last_profile_id": int(
            decisions[-1].profile_id
        ),
        "min_decision_slack_ms": float(
            min(
                d.remaining_budget_ms
                - d.required_remaining_ms
                for d in decisions
            )
        ),
        "max_control_guard_ms": float(
            max(d.control_guard_ms for d in decisions)
        ),
    }


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "p50": float(np.percentile(x, 50)),
        "p90": float(np.percentile(x, 90)),
        "p99": float(np.percentile(x, 99)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )
    base_model.cuda().eval()

    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(
            c.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
        ),
    ).cuda().eval()

    checkpoint = torch.load(
        args.elastic_ckpt,
        map_location="cpu",
    )
    branch.load_state_dict(
        checkpoint["branch"],
        strict=True,
    )

    bank = torch.load(
        args.prefix_bn_bank,
        map_location="cpu",
    )
    if (
        bank.get("version")
        != "elastic_v4_bn_causal_prefix_bank_v1"
    ):
        raise RuntimeError(
            "Unsupported causal-prefix BN bank"
        )

    fused_counts = enable_causal_prefix_fused_cache(
        branch
    )

    controller = TVStream3DController(
        controller_csv=args.controller_csv,
        contention_levels_json=args.levels_json,
        bound_column="remaining_p99_ms",
        classifier="conservative_gap",
        control_guard_per_boundary_ms=(
            args.control_guard_per_boundary_ms
        ),
    )

    levels_data = json.loads(
        Path(args.levels_json).read_text()
    )
    contender = configure_contender(
        levels_data,
        args.expected_level,
    )

    model_stream = make_high_priority_detector_stream(
        torch.cuda.current_device()
    )

    deadline_ms = 1000.0 / float(args.input_hz)

    needed = (
        int(args.warmup_frames)
        + int(args.frames)
    )
    scene, indices = choose_scene_indices(
        dataset,
        needed,
    )

    if base_model.history_feature_queue is not None:
        base_model.history_feature_queue.clear()

    print("=" * 80)
    print("TV-Stream3D online forward smoke")
    print(f"scene           : {scene}")
    print(f"expected level  : {args.expected_level}")
    print(f"input Hz        : {args.input_hz}")
    print(f"deadline        : {deadline_ms:.6f} ms")
    print(f"warmup frames   : {args.warmup_frames}")
    print(f"measured frames : {args.frames}")
    print(f"fused modules   : {fused_counts}")
    print("timing scope    : forward only; excludes H2D and post_processing/NMS")
    print("=" * 80)

    # ============================================================
    # 第一步：
    # 确定性遍历全部 84 条合法单调 schedule，
    # materialize 全部 203 个非 Full causal-prefix fused states。
    # ============================================================

    prewarm_batch = load_one(
        dataset,
        indices[0],
    )

    deterministic_prewarm = (
        prewarm_all_causal_prefixes(
            base_model=base_model,
            branch=branch,
            bank=bank,
            controller=controller,
            batch=prewarm_batch,
            model_stream=model_stream,
        )
    )

    print(
        "[PREWARM] deterministic causal-prefix "
        "coverage:",
        json.dumps(
            deterministic_prewarm,
            ensure_ascii=False,
        ),
    )

    # ============================================================
    # 从这里开始禁止生成任何新的 fused cache。
    #
    # 因此：
    #   runtime warmup 也不允许 cache miss；
    #   measured frames 更不允许 cache miss。
    # ============================================================

    set_forbid_cache_miss(
        branch,
        True,
    )

    # ============================================================
    # 第二阶段确定性预热：
    #
    # 第一阶段已经在 FP32 下 materialize 全部 203 个 causal-prefix
    # fused cache 状态。
    #
    # 现在再把全部 84 条合法 schedule 在真实 runtime AMP 模式下
    # 执行一遍，用于预热 CUDA/cuDNN 的实际 FP16/AMP kernel、算法和
    # shape-specific runtime state。
    #
    # fused cache 已经锁死，因此这里不允许产生任何新 cache entry。
    # ============================================================

    cache_before_amp_prewarm = (
        fused_cache_stats(branch)
    )

    with torch.amp.autocast(
        "cuda",
        enabled=bool(
            base_model.use_amp_dict["TEST"]
        ),
    ):
        deterministic_amp_prewarm = (
            prewarm_all_causal_prefixes(
                base_model=base_model,
                branch=branch,
                bank=bank,
                controller=controller,
                batch=prewarm_batch,
                model_stream=model_stream,
            )
        )

    cache_after_amp_prewarm = (
        fused_cache_stats(branch)
    )

    if (
        cache_after_amp_prewarm
        != cache_before_amp_prewarm
    ):
        raise RuntimeError(
            "AMP deterministic path prewarm changed "
            "the fused cache: "
            f"before={cache_before_amp_prewarm}, "
            f"after={cache_after_amp_prewarm}"
        )

    print(
        "[PREWARM] deterministic AMP path coverage:",
        json.dumps(
            deterministic_amp_prewarm,
            ensure_ascii=False,
        ),
    )

    cache_before_runtime_warmup = (
        fused_cache_stats(branch)
    )

    for i in range(
        int(args.warmup_frames)
    ):
        batch = load_one(
            dataset,
            indices[i],
        )

        r = execute_one(
            base_model=base_model,
            branch=branch,
            bank=bank,
            controller=controller,
            batch=batch,
            model_stream=model_stream,
            contender=contender,
            deadline_ms=deadline_ms,
            allow_prepare=False,
        )

        if (
            i < 3
            or (i + 1) % 20 == 0
        ):
            print(
                f"[warmup "
                f"{i+1:03d}/"
                f"{args.warmup_frames:03d}] "
                f"forward="
                f"{r['forward_ms']:.3f} "
                f"level="
                f"{r['observed_level']} "
                f"schedule="
                f"{r['schedule']}"
            )

    cache_after_warmup = (
        fused_cache_stats(branch)
    )

    if (
        cache_after_warmup
        != cache_before_runtime_warmup
    ):
        raise RuntimeError(
            "Runtime warmup created new fused "
            "cache state after deterministic "
            "prewarm: "
            f"before="
            f"{cache_before_runtime_warmup}, "
            f"after="
            f"{cache_after_warmup}"
        )

    rows = []
    offset = int(args.warmup_frames)

    for i in range(int(args.frames)):
        batch = load_one(
            dataset,
            indices[offset + i],
        )

        r = execute_one(
            base_model=base_model,
            branch=branch,
            bank=bank,
            controller=controller,
            batch=batch,
            model_stream=model_stream,
            contender=contender,
            deadline_ms=deadline_ms,
            allow_prepare=False,
        )

        row = {
            "frame": i,
            "expected_level": args.expected_level,
            "observed_level": r["observed_level"],
            "input_hz": float(args.input_hz),
            "deadline_ms": float(deadline_ms),
            "forward_ms": r["forward_ms"],
            "deadline_miss": int(
                r["forward_ms"] > deadline_ms
            ),
            "prefix_ms": r["prefix_ms"],
            "after_rpn_ms": r["after_rpn_ms"],
            "schedule": ",".join(
                str(float(x))
                for x in r["schedule"]
            ),
            "first_profile_id": r["first_profile_id"],
            "last_profile_id": r["last_profile_id"],
            "all_decisions_feasible": int(
                r["all_decisions_feasible"]
            ),
            "controller_ms": r["controller_ms"],
            "controller_calls": r["controller_calls"],
            "min_decision_slack_ms": r["min_decision_slack_ms"],
            "max_control_guard_ms": r["max_control_guard_ms"],
        }
        rows.append(row)

        if i < 5 or (i + 1) % 20 == 0:
            print(
                f"[{i+1:03d}/{args.frames:03d}] "
                f"forward={row['forward_ms']:.3f}/"
                f"{deadline_ms:.3f} ms "
                f"miss={row['deadline_miss']} "
                f"level={row['observed_level']} "
                f"schedule={row['schedule']}"
            )

    cache_after_measure = fused_cache_stats(
        branch
    )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with output_csv.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    forward = [
        x["forward_ms"]
        for x in rows
    ]
    prefix = [
        x["prefix_ms"]
        for x in rows
    ]
    ctrl = [
        x["controller_ms"]
        for x in rows
    ]

    observed_counts = Counter(
        x["observed_level"]
        for x in rows
    )
    schedule_counts = Counter(
        x["schedule"]
        for x in rows
    )

    summary = {
        "timing_scope": (
            "forward only: fixed stem+layer1 + rolling controller + "
            "Res2/3/4 + FPN + Stereo + RPN + K3 fusion + VAN + "
            "detection head; excludes data loading/H2D and "
            "post_processing/NMS"
        ),
        "expected_level": args.expected_level,
        "input_hz": float(args.input_hz),
        "deadline_ms": float(deadline_ms),
        "control_guard_per_boundary_ms": float(
            args.control_guard_per_boundary_ms
        ),
        "warmup_frames": int(
            args.warmup_frames
        ),
        "frames": int(args.frames),
        "forward_ms": stats(forward),
        "prefix_ms": stats(prefix),
        "controller_ms_total_per_forward": stats(
            ctrl
        ),
        "deadline_miss_count": int(
            sum(x["deadline_miss"] for x in rows)
        ),
        "deadline_miss_rate": float(
            np.mean(
                [
                    x["deadline_miss"]
                    for x in rows
                ]
            )
        ),
        "observed_level_counts": dict(
            observed_counts
        ),
        "schedule_counts": dict(
            schedule_counts
        ),
        "decision_infeasible_frames": int(
            sum(
                1
                for x in rows
                if not x["all_decisions_feasible"]
            )
        ),
        "min_decision_slack_ms": stats(
            [
                x["min_decision_slack_ms"]
                for x in rows
            ]
        ),
        "deterministic_prefix_prewarm": (
            deterministic_prewarm
        ),
        "deterministic_amp_path_prewarm": (
            deterministic_amp_prewarm
        ),
        "fused_cache_before_amp_prewarm": (
            cache_before_amp_prewarm
        ),
        "fused_cache_after_amp_prewarm": (
            cache_after_amp_prewarm
        ),
        "fused_cache_before_runtime_warmup": (
            cache_before_runtime_warmup
        ),
        "fused_cache_after_warmup": (
            cache_after_warmup
        ),
        "fused_cache_after_measure": (
            cache_after_measure
        ),
        "new_fused_entries_during_measure": int(
            cache_after_measure["entries"]
            - cache_after_warmup["entries"]
        ),
        "raw_csv": str(output_csv),
    }

    output_json = Path(
        args.output_json
    )
    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    print("\n" + "=" * 80)
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )
    print("=" * 80)

    if (
        summary[
            "new_fused_entries_during_measure"
        ]
        != 0
    ):
        raise SystemExit(
            "Measured run created new fused cache entries"
        )


if __name__ == "__main__":
    main()
