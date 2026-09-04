#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    ElasticBEVBranch,
)

from test_stream_buffer_timestamp import load_one

from test_tv_stream3d_online_forward import (
    make_cfg,
    choose_scene_indices,
    configure_contender,
    execute_one,
    prewarm_all_causal_prefixes,
)

from tv_stream3d_controller import (
    TVStream3DController,
)

from tv_stream3d_causal_fused_runtime import (
    enable_causal_prefix_fused_cache,
    fused_cache_stats,
    set_forbid_cache_miss,
)

from smooth_cuda_contention import (
    make_high_priority_detector_stream,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--full_cfg",
        required=True,
    )
    p.add_argument(
        "--full_ckpt",
        required=True,
    )
    p.add_argument(
        "--elastic_ckpt",
        required=True,
    )
    p.add_argument(
        "--prefix_bn_bank",
        required=True,
    )
    p.add_argument(
        "--controller_csv",
        required=True,
    )
    p.add_argument(
        "--levels_json",
        required=True,
    )

    p.add_argument(
        "--input_hz",
        type=float,
        default=30.0,
    )
    p.add_argument(
        "--warmup_frames",
        type=int,
        default=30,
    )
    p.add_argument(
        "--frames",
        type=int,
        default=300,
    )
    p.add_argument(
        "--workers",
        type=int,
        default=0,
    )
    p.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    p.add_argument(
        "--control_guard_per_boundary_ms",
        type=float,
        default=0.25,
    )

    p.add_argument(
        "--levels",
        default="L0,L1,L2,L3,L4",
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    return p.parse_args()


def stats(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if x.size == 0:
        raise RuntimeError(
            "empty timing list"
        )

    return {
        "n": int(x.size),
        "mean_ms": float(
            np.mean(x)
        ),
        "p50_ms": float(
            np.percentile(x, 50)
        ),
        "p90_ms": float(
            np.percentile(x, 90)
        ),
        "p99_ms": float(
            np.percentile(x, 99)
        ),
        "min_ms": float(
            np.min(x)
        ),
        "max_ms": float(
            np.max(x)
        ),
    }


def disable_evaluation_recall(model):
    """
    保留本地 model.post_processing() 的真实 score filtering + NMS，
    只去掉 generate_recall_record()。

    generate_recall_record() 使用 GT 做 IoU / recall 统计，
    属于 evaluator bookkeeping，不属于部署时的 post/NMS。
    """

    if not hasattr(
        model,
        "generate_recall_record",
    ):
        raise RuntimeError(
            "model has no generate_recall_record"
        )

    original = (
        model.generate_recall_record
    )

    def no_recall_record(
        box_preds,
        recall_dict,
        batch_index,
        data_dict=None,
        thresh_list=None,
    ):
        return (
            recall_dict,
            None,
            None,
        )

    model.generate_recall_record = (
        no_recall_record
    )

    return original


def profile_post_once(
    model,
    batch_dict,
    model_stream,
    contender,
):
    """
    只计 model.post_processing() 的运行时间。

    若 level != L0，则 post/NMS 自身也运行在与之前 profile
    相同的低优先级背景 GPU contention 下。
    """

    amp_enabled = bool(
        model.use_amp_dict["TEST"]
    )

    if contender is not None:
        contender.launch()

    with (
        torch.cuda.stream(model_stream),
        torch.no_grad(),
        torch.amp.autocast(
            "cuda",
            enabled=amp_enabled,
        ),
    ):
        start = torch.cuda.Event(
            enable_timing=True
        )
        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()

        pred_dicts, _ = (
            model.post_processing(
                batch_dict
            )
        )

        end.record()

    end.synchronize()

    post_ms = float(
        start.elapsed_time(end)
    )

    if contender is not None:
        contender.finish()

    num_boxes = int(
        sum(
            x["pred_boxes"].shape[0]
            for x in pred_dicts
        )
    )

    return (
        post_ms,
        num_boxes,
    )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    np.random.seed(
        args.seed
    )
    torch.manual_seed(
        args.seed
    )
    torch.cuda.manual_seed_all(
        args.seed
    )

    levels = [
        x.strip()
        for x
        in args.levels.split(",")
        if x.strip()
    ]

    valid_levels = {
        "L0",
        "L1",
        "L2",
        "L3",
        "L4",
    }

    if (
        not levels
        or any(
            x not in valid_levels
            for x in levels
        )
    ):
        raise ValueError(
            f"invalid --levels: {levels}"
        )

    out_dir = Path(
        args.output_dir
    )
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Model / dataset
    # ============================================================

    c = make_cfg(
        args.full_cfg
    )

    logger = (
        common_utils.create_logger()
    )

    dataset, _, _ = (
        build_dataloader(
            dataset_cfg=c.DATA_CONFIG,
            class_names=c.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=False,
        )
    )

    model = build_network(
        model_cfg=c.MODEL,
        num_class=len(
            c.CLASS_NAMES
        ),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )

    model.cuda().eval()

    # ============================================================
    # Elastic branch
    # ============================================================

    branch = ElasticBEVBranch(
        model.backbone_3d,
        output_bev_channels=int(
            c.MODEL.MAP_TO_BEV
            .NUM_BEV_FEATURES
        ),
    ).cuda().eval()

    elastic = torch.load(
        args.elastic_ckpt,
        map_location="cpu",
    )

    branch.load_state_dict(
        elastic["branch"],
        strict=True,
    )

    # ============================================================
    # Causal-prefix BN bank
    # ============================================================

    bank = torch.load(
        args.prefix_bn_bank,
        map_location="cpu",
    )

    if (
        bank.get("version")
        !=
        "elastic_v4_bn_causal_prefix_bank_v1"
    ):
        raise RuntimeError(
            "unsupported causal-prefix BN bank"
        )

    enable_causal_prefix_fused_cache(
        branch
    )

    # ============================================================
    # Controller
    # ============================================================

    controller = TVStream3DController(
        controller_csv=(
            args.controller_csv
        ),
        contention_levels_json=(
            args.levels_json
        ),
        bound_column=(
            "remaining_p99_ms"
        ),
        classifier=(
            "conservative_gap"
        ),
        control_guard_per_boundary_ms=(
            args
            .control_guard_per_boundary_ms
        ),
    )

    levels_data = json.loads(
        Path(
            args.levels_json
        ).read_text()
    )

    model_stream = (
        make_high_priority_detector_stream(
            torch.cuda.current_device()
        )
    )

    # 注意：
    # 这里只用于产生真实 TV-Stream3D head outputs。
    # P 此时还未知，所以还没有减去 P。
    deadline_ms = (
        1000.0
        / float(args.input_hz)
    )

    needed = max(
        1,
        args.warmup_frames
        + args.frames,
    )

    scene, indices = (
        choose_scene_indices(
            dataset,
            needed,
        )
    )

    if (
        model.history_feature_queue
        is not None
    ):
        model.history_feature_queue.clear()

    # ============================================================
    # Phase 1:
    # deterministic 203-prefix fused-cache prewarm
    # ============================================================

    prewarm_batch = load_one(
        dataset,
        indices[0],
    )

    with torch.no_grad():
        prewarm = (
            prewarm_all_causal_prefixes(
                base_model=model,
                branch=branch,
                bank=bank,
                controller=controller,
                batch=prewarm_batch,
                model_stream=model_stream,
            )
        )

    if (
        prewarm["expected_prefixes"]
        != 203
        or
        prewarm["ready_prefixes"]
        != 203
    ):
        raise RuntimeError(
            "causal-prefix prewarm "
            f"incomplete: {prewarm}"
        )

    set_forbid_cache_miss(
        branch,
        True,
    )

    cache_before_amp = (
        fused_cache_stats(branch)
    )

    # ============================================================
    # Phase 2:
    # 84 legal paths under real TEST AMP
    # ============================================================

    with (
        torch.no_grad(),
        torch.amp.autocast(
            "cuda",
            enabled=bool(
                model
                .use_amp_dict["TEST"]
            ),
        ),
    ):
        amp_prewarm = (
            prewarm_all_causal_prefixes(
                base_model=model,
                branch=branch,
                bank=bank,
                controller=controller,
                batch=prewarm_batch,
                model_stream=model_stream,
            )
        )

    cache_after_amp = (
        fused_cache_stats(branch)
    )

    if (
        cache_after_amp
        != cache_before_amp
    ):
        raise RuntimeError(
            "AMP prewarm changed "
            "fused cache: "
            f"before={cache_before_amp}, "
            f"after={cache_after_amp}"
        )

    if (
        amp_prewarm[
            "expected_prefixes"
        ] != 203
        or
        amp_prewarm[
            "ready_prefixes"
        ] != 203
    ):
        raise RuntimeError(
            "AMP path prewarm "
            f"incomplete: {amp_prewarm}"
        )

    # ============================================================
    # 去掉 evaluation-only recall / GT IoU
    # ============================================================

    original_recall = (
        disable_evaluation_recall(
            model
        )
    )

    print(
        "=" * 96
    )
    print(
        "TV-Stream3D "
        "post-processing/NMS "
        "p99 profiler"
    )
    print(
        f"scene             : {scene}"
    )
    print(
        f"input Hz          : "
        f"{args.input_hz}"
    )
    print(
        f"controller period : "
        f"{deadline_ms:.6f} ms "
        "(P is not subtracted yet)"
    )
    print(
        f"warmup/level      : "
        f"{args.warmup_frames}"
    )
    print(
        f"measure/level     : "
        f"{args.frames}"
    )
    print(
        f"levels            : "
        f"{levels}"
    )
    print(
        "scope             : "
        "local model.post_processing; "
        "recall/GT IoU excluded"
    )
    print(
        f"causal prefixes   : "
        f"{prewarm['ready_prefixes']}"
    )
    print(
        "=" * 96
    )

    summary = {
        "version":
            "tv_stream3d_post_nms_p99_v1",

        "scope": (
            "repository "
            "model.post_processing "
            "score filtering + NMS; "
            "generate_recall_record/"
            "GT IoU disabled as "
            "evaluator-only work"
        ),

        "input_hz_for_output_generation":
            float(
                args.input_hz
            ),

        "warmup_frames_per_level":
            int(
                args.warmup_frames
            ),

        "measured_frames_per_level":
            int(
                args.frames
            ),

        "levels": {},
    }

    pooled = []

    try:
        for level in levels:

            print()
            print(
                "-" * 96
            )
            print(
                f"[{level}] start"
            )

            if (
                model.history_feature_queue
                is not None
            ):
                model.history_feature_queue.clear()

            # 一个用于生成真实 head output，
            # 一个专门用于 profile post/NMS。
            forward_contender = (
                configure_contender(
                    levels_data,
                    level,
                )
            )

            post_contender = (
                configure_contender(
                    levels_data,
                    level,
                )
            )

            # ====================================================
            # runtime warmup
            # ====================================================

            for i in range(
                args.warmup_frames
            ):
                batch = load_one(
                    dataset,
                    indices[i],
                )

                execute_one(
                    base_model=model,
                    branch=branch,
                    bank=bank,
                    controller=controller,
                    batch=batch,
                    model_stream=model_stream,
                    contender=(
                        forward_contender
                    ),
                    deadline_ms=(
                        deadline_ms
                    ),
                    allow_prepare=False,
                )

                profile_post_once(
                    model,
                    batch["token"],
                    model_stream,
                    post_contender,
                )

            # ====================================================
            # measured
            # ====================================================

            rows = []
            times = []
            box_counts = []

            for m in range(
                args.frames
            ):
                idx = indices[
                    args.warmup_frames
                    + m
                ]

                batch = load_one(
                    dataset,
                    idx,
                )

                # 先执行真实 TV-Stream3D forward，
                # 只为了生成真实 head outputs。
                # 这里的 forward 时间绝不进入 P。
                fwd = execute_one(
                    base_model=model,
                    branch=branch,
                    bank=bank,
                    controller=controller,
                    batch=batch,
                    model_stream=model_stream,
                    contender=(
                        forward_contender
                    ),
                    deadline_ms=(
                        deadline_ms
                    ),
                    allow_prepare=False,
                )

                post_ms, num_boxes = (
                    profile_post_once(
                        model,
                        batch["token"],
                        model_stream,
                        post_contender,
                    )
                )

                times.append(
                    post_ms
                )
                box_counts.append(
                    num_boxes
                )
                pooled.append(
                    post_ms
                )

                rows.append(
                    {
                        "sample":
                            m,

                        "dataset_index":
                            int(idx),

                        "level":
                            level,

                        "post_nms_ms":
                            post_ms,

                        "num_output_boxes":
                            num_boxes,

                        "forward_ms_not_in_P":
                            float(
                                fwd[
                                    "forward_ms"
                                ]
                            ),

                        "observed_level":
                            fwd[
                                "observed_level"
                            ],

                        "schedule":
                            ",".join(
                                str(x)
                                for x
                                in fwd[
                                    "schedule"
                                ]
                            ),
                    }
                )

                if (
                    m < 3
                    or
                    (m + 1) % 50 == 0
                ):
                    print(
                        f"[{level} "
                        f"{m+1:03d}/"
                        f"{args.frames:03d}] "
                        f"post/NMS="
                        f"{post_ms:.4f} ms "
                        f"boxes={num_boxes} "
                        f"schedule="
                        f"{fwd['schedule']}"
                    )

            csv_path = (
                out_dir
                /
                f"{level}_"
                "post_nms_raw.csv"
            )

            with csv_path.open(
                "w",
                newline="",
            ) as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=list(
                        rows[0].keys()
                    ),
                )

                writer.writeheader()
                writer.writerows(
                    rows
                )

            s = stats(
                times
            )

            s[
                "num_output_boxes"
            ] = {
                "mean":
                    float(
                        np.mean(
                            box_counts
                        )
                    ),

                "p99":
                    float(
                        np.percentile(
                            box_counts,
                            99,
                        )
                    ),

                "max":
                    int(
                        np.max(
                            box_counts
                        )
                    ),
            }

            s["raw_csv"] = str(
                csv_path
            )

            summary[
                "levels"
            ][level] = s

            print(
                f"[{level}] "
                f"p50="
                f"{s['p50_ms']:.4f} ms "
                f"p90="
                f"{s['p90_ms']:.4f} ms "
                f"p99="
                f"{s['p99_ms']:.4f} ms "
                f"max="
                f"{s['max_ms']:.4f} ms"
            )

    finally:
        model.generate_recall_record = (
            original_recall
        )

    # ============================================================
    # 固定 P：
    #
    # P = max_L p99(post/NMS | L)
    #
    # 不使用 pooled p99，因为我们希望 P 对所有 L0-L4 条件固定。
    # ============================================================

    per_level_p99 = {
        level:
            summary[
                "levels"
            ][level][
                "p99_ms"
            ]
        for level
        in levels
    }

    source_level = max(
        per_level_p99,
        key=per_level_p99.get,
    )

    P_ms = float(
        per_level_p99[
            source_level
        ]
    )

    summary[
        "pooled_stats"
    ] = stats(
        pooled
    )

    summary[
        "P_definition"
    ] = (
        "max_L "
        "p99(post_processing/NMS | L)"
    )

    summary[
        "P_ms"
    ] = P_ms

    summary[
        "P_source_level"
    ] = source_level

    summary_path = (
        out_dir
        /
        "post_nms_p99_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n"
    )

    print()
    print(
        "=" * 96
    )
    print(
        "FINAL FIXED "
        "POST/NMS RESERVE"
    )

    for level in levels:
        print(
            f"{level}: "
            f"p99="
            f"{per_level_p99[level]:.6f} ms"
        )

    print(
        f"P = max per-level p99 "
        f"= {P_ms:.6f} ms "
        f"(from {source_level})"
    )

    print(
        f"summary: "
        f"{summary_path}"
    )

    print(
        "=" * 96
    )


if __name__ == "__main__":
    main()
