#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

from test_stream_buffer_timestamp import load_one

from test_tv_stream3d_online_forward import (
    make_cfg,
    choose_scene_indices,
    configure_contender,
)

from smooth_cuda_contention import (
    make_high_priority_detector_stream,
)

# 直接复用 TV random50 evaluator 的 trace / sAP 辅助逻辑，
# 从根本上保证两种方法使用相同的 sensor-frame contention trace。
from eval_tv_stream3d_30hz_random50 import (
    build_balanced_trace,
    frame_meta,
    scene_groups,
    copy_det,
    disable_recall,
    make_prediction,
    stats,
    builtin,
)

from mtd_three_head_runtime import (
    MTDDelayAnalyzer,
    MTDThreeHeadBank,
    mtd_forward_no_post,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--cfg", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--h2_ckpt", required=True)
    p.add_argument("--h3_ckpt", required=True)
    p.add_argument("--levels_json", required=True)

    p.add_argument(
        "--pressure_level",
        required=True,
        choices=["L0", "L1", "L2", "L3", "L4"],
    )

    p.add_argument(
        "--pressure_fraction",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--trace_seed",
        type=int,
        default=20260903,
    )

    p.add_argument(
        "--input_hz",
        type=float,
        default=30.0,
    )

    p.add_argument(
        "--runtime_warmup_frames",
        type=int,
        default=80,
    )

    p.add_argument(
        "--max_frames",
        type=int,
        default=0,
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
        "--output_dir",
        required=True,
    )

    return p.parse_args()


def reset_history(model):
    q = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if q is not None:
        q.clear()


def original_forward_no_post(
    model,
    batch,
    model_stream,
    contender,
):
    """
    原始 StreamDSGN 的 forward-only 执行。

    计时范围：
        feature_extractor
        + history feature preparation
        + fusion_module
        + after_fusion_blocks

    不计：
        post_processing / NMS
        evaluation recall
        dataloader / H2D
    """

    cur_data = batch["token"]

    launched = False

    try:
        if contender is not None:
            contender.launch()
            launched = True

        with (
            torch.cuda.stream(model_stream),
            torch.no_grad(),
            torch.amp.autocast(
                "cuda",
                enabled=bool(
                    model.use_amp_dict["TEST"]
                ),
            ),
        ):
            start = torch.cuda.Event(
                enable_timing=True
            )

            end = torch.cuda.Event(
                enable_timing=True
            )

            start.record()

            # ----------------------------------------------------
            # 1. Original feature extractor
            # ----------------------------------------------------

            for module in model.feature_extractor:
                cur_data = module(cur_data)

            # ----------------------------------------------------
            # 2. Processed-frame history only
            #
            # 不使用 sensor-adjacent dropped frame。
            # queue 中只可能存在真正执行过的历史帧。
            # ----------------------------------------------------

            cur_data["history_features"] = (
                model.history_feature_queue
            )

            history_features = None

            if model.history_tag is not None:
                history_features = {}

                for feature_name in (
                    model.history_features_name
                ):
                    history_features[
                        feature_name
                    ] = cur_data[
                        feature_name
                    ].clone()

            # ----------------------------------------------------
            # 3. Original temporal fusion
            # ----------------------------------------------------

            for module in model.fusion_module:
                cur_data = module(cur_data)

            # ----------------------------------------------------
            # 4. Original post-fusion network
            # ----------------------------------------------------

            for module in model.after_fusion_blocks:
                cur_data = module(cur_data)

            end.record()

        end.synchronize()

        forward_ms = float(
            start.elapsed_time(end)
        )

    finally:
        if launched:
            contender.finish()

    # ------------------------------------------------------------
    # History becomes available at forward completion f_i.
    #
    # Dropped frames never execute this code.
    # ------------------------------------------------------------

    if (
        model.history_tag is not None
        and history_features is not None
    ):
        model.history_feature_queue.append(
            (
                cur_data["this_sample_idx"],
                history_features,
            )
        )

    batch["token"] = cur_data

    return (
        cur_data,
        forward_ms,
    )


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.input_hz <= 0:
        raise ValueError(
            "input_hz must be > 0"
        )

    if not (
        0.0
        <=
        args.pressure_fraction
        <
        1.0
    ):
        raise ValueError(
            "pressure_fraction must "
            "satisfy 0 <= f < 1"
        )

    if args.max_frames < 0:
        raise ValueError(
            "max_frames must be >= 0"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(
        args.seed
    )

    period_ms = (
        1000.0
        /
        args.input_hz
    )

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # Original StreamDSGN
    # ============================================================

    cfg = make_cfg(
        args.cfg
    )

    logger = (
        common_utils.create_logger()
    )

    dataset, _, _ = (
        build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=False,
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
        filename=args.ckpt,
        logger=logger,
        to_cpu=True,
    )

    model.cuda().eval()

    # ============================================================
    # TRUE MTD:
    # shared trunk + H1/H2/H3 neural prediction heads
    # ============================================================

    mtd_heads = MTDThreeHeadBank(
        model=model,
        h2_ckpt=args.h2_ckpt,
        h3_ckpt=args.h3_ckpt,
        logger=logger,
    )

    dam = MTDDelayAnalyzer(
        period_ms=period_ms,
        max_horizon=3,
    )

    mtd_decisions = []
    mtd_branch_hist = Counter()

    if not hasattr(
        model,
        "feature_extractor",
    ):
        raise RuntimeError(
            "model has no feature_extractor; "
            "this does not look like the "
            "original STREAM detector"
        )

    if not hasattr(
        model,
        "fusion_module",
    ):
        raise RuntimeError(
            "model has no fusion_module"
        )

    if not hasattr(
        model,
        "after_fusion_blocks",
    ):
        raise RuntimeError(
            "model has no after_fusion_blocks"
        )

    # ============================================================
    # Same contention mechanism as TV-Stream3D
    # ============================================================

    levels_data = json.loads(
        Path(
            args.levels_json
        ).read_text()
    )

    contenders = {
        "L0":
            configure_contender(
                levels_data,
                "L0",
            ),

        args.pressure_level:
            configure_contender(
                levels_data,
                args.pressure_level,
            ),
    }

    model_stream = (
        make_high_priority_detector_stream(
            torch.cuda.current_device()
        )
    )

    # ============================================================
    # Runtime warmup
    # ============================================================

    _, warm_indices = (
        choose_scene_indices(
            dataset,
            max(
                1,
                args.runtime_warmup_frames
                + 1,
            ),
        )
    )

    original_recall = (
        disable_recall(model)
    )

    reset_history(model)

    last_batch = None

    for wi in range(
        args.runtime_warmup_frames
    ):
        idx = warm_indices[
            wi
            %
            len(warm_indices)
        ]

        batch = load_one(
            dataset,
            idx,
        )

        warm_level = (
            args.pressure_level
            if wi % 2
            else "L0"
        )

        # Warm H1/H2/H3 CUDA kernels evenly.
        # These runtimes are deliberately NOT passed to DAM.
        warm_branch = (
            wi % 3
        ) + 1

        mtd_forward_no_post(
            model=model,
            batch=batch,
            model_stream=model_stream,
            contender=(
                contenders[
                    warm_level
                ]
            ),
            head_bank=mtd_heads,
            branch_step=warm_branch,
        )

        last_batch = batch

    # 首次 NMS/CUDA extension 初始化
    # 放在正式 streaming 测量之前。
    if last_batch is not None:
        make_prediction(
            model,
            dataset,
            last_batch,
        )

    reset_history(model)

    torch.cuda.synchronize()

    # ============================================================
    # Evaluation set + exact same random contention trace
    # ============================================================

    # Engineering warmup is outside the formal stream.
    # Do not leak its timing into DAM.
    dam.reset()

    total_dataset = len(
        dataset
    )

    eval_count = (
        total_dataset
        if args.max_frames == 0
        else min(
            total_dataset,
            args.max_frames,
        )
    )

    eval_indices = list(
        range(eval_count)
    )

    groups = scene_groups(
        dataset,
        eval_indices,
    )

    trace_by_idx, trace_scene_summary = (
        build_balanced_trace(
            groups=groups,
            pressure_level=(
                args.pressure_level
            ),
            fraction=(
                args.pressure_fraction
            ),
            seed=args.trace_seed,
        )
    )

    sensor_level_hist = Counter(
        trace_by_idx[idx]
        for idx
        in eval_indices
    )

    # ============================================================
    # Save exact external trace
    # ============================================================

    trace_csv = (
        out_dir
        /
        "contention_trace.csv"
    )

    with trace_csv.open(
        "w",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "global_index",
                "scene",
                "local_pos",
                "frame_id",
                "true_level",
            ],
        )

        w.writeheader()

        for scene, indices in (
            groups.items()
        ):
            for local_pos, idx in (
                enumerate(indices)
            ):
                _, frame_id, _ = (
                    frame_meta(
                        dataset,
                        idx,
                    )
                )

                w.writerow({
                    "global_index":
                        idx,

                    "scene":
                        scene,

                    "local_pos":
                        local_pos,

                    "frame_id":
                        frame_id,

                    "true_level":
                        trace_by_idx[idx],
                })

    print("=" * 100)
    print(
        "MTD Three-Head StreamDSGN "
        "streaming evaluation"
    )
    print(
        f"input Hz          : "
        f"{args.input_hz}"
    )
    print(
        f"period            : "
        f"{period_ms:.6f} ms"
    )
    print(
        "timing scope      : "
        "forward only"
    )
    print(
        "contention        : "
        f"L0 + {args.pressure_level}"
    )
    print(
        "pressure fraction : "
        f"{100*args.pressure_fraction:.2f}%"
    )
    print(
        f"trace seed         : "
        f"{args.trace_seed}"
    )
    print(
        f"sensor levels      : "
        f"{dict(sensor_level_hist)}"
    )
    print(
        "buffer             : "
        "latest-frame only"
    )
    print(
        "history            : "
        "processed-frame only"
    )
    print(
        f"sensor frames      : "
        f"{eval_count}"
    )
    print("=" * 100)

    rows = {}
    events = []

    forward_times = []
    wait_times = []
    response_times = []

    processed_level_hist = Counter()

    processed = 0
    dropped = 0
    misses = 0

    try:
        for scene_i, (
            scene,
            indices,
        ) in enumerate(
            groups.items(),
            start=1,
        ):
            # scene 之间绝对不能共享历史
            reset_history(model)

            pos = 0
            gpu_free_ms = 0.0
            n = len(indices)

            print(
                f"[scene "
                f"{scene_i:02d}/"
                f"{len(groups):02d}] "
                f"{scene}: {n} frames"
            )

            while pos < n:
                idx = indices[pos]

                (
                    meta_scene,
                    frame_id,
                    next_frame_id,
                ) = frame_meta(
                    dataset,
                    idx,
                )

                if meta_scene != scene:
                    raise RuntimeError(
                        "scene mismatch"
                    )

                # ------------------------------------------------
                # Absolute 30-Hz sensor timeline
                # ------------------------------------------------

                arrival_ms = (
                    pos
                    *
                    period_ms
                )

                deadline_ms = (
                    arrival_ms
                    +
                    period_ms
                )

                start_ms = max(
                    gpu_free_ms,
                    arrival_ms,
                )

                true_level = (
                    trace_by_idx[
                        idx
                    ]
                )

                # ------------------------------------------------
                # Original StreamDSGN forward
                # ------------------------------------------------

                batch = load_one(
                    dataset,
                    idx,
                )

                # --------------------------------------------
                # MTD Delay Analysis Module
                #
                # CAUSAL: decision is made before current forward.
                # --------------------------------------------

                dam_queue_wait_ms = (
                    start_ms
                    -
                    arrival_ms
                )

                mtd_decision = (
                    dam.select(
                        queue_wait_ms=(
                            dam_queue_wait_ms
                        )
                    )
                )

                branch_step = int(
                    mtd_decision.branch_step
                )

                mtd_branch_hist[
                    branch_step
                ] += 1

                _, forward_ms = (
                    mtd_forward_no_post(
                        model=model,
                        batch=batch,
                        model_stream=(
                            model_stream
                        ),
                        contender=(
                            contenders[
                                true_level
                            ]
                        ),
                        head_bank=mtd_heads,
                        branch_step=(
                            branch_step
                        ),
                    )
                )

                # Current runtime becomes available ONLY now.
                dam.observe(
                    forward_ms
                )

                actual_response_for_branch_ms = (
                    dam_queue_wait_ms
                    +
                    forward_ms
                )

                # Oracle branch is diagnostic only.
                # It is NEVER used by the detector.
                oracle_branch_step = (
                    dam.branch_for_response_ms(
                        actual_response_for_branch_ms
                    )
                )

                mtd_decisions.append({
                    "global_index":
                        idx,

                    "scene":
                        scene,

                    "local_pos":
                        pos,

                    "frame_id":
                        frame_id,

                    "true_level":
                        true_level,

                    "queue_wait_ms":
                        dam_queue_wait_ms,

                    "estimated_forward_ms":
                        mtd_decision.estimated_forward_ms,

                    "estimated_response_ms":
                        mtd_decision.estimated_response_ms,

                    "estimated_delay_slots":
                        mtd_decision.delay_slots,

                    "runtime_history_count":
                        mtd_decision.runtime_history_count,

                    "branch_step":
                        branch_step,

                    "actual_forward_ms":
                        forward_ms,

                    "actual_response_ms":
                        actual_response_for_branch_ms,

                    "oracle_branch_step":
                        oracle_branch_step,

                    "branch_match_oracle":
                        int(
                            branch_step
                            ==
                            oracle_branch_step
                        ),
                })

                finish_ms = (
                    start_ms
                    +
                    forward_ms
                )

                wait_ms = (
                    start_ms
                    -
                    arrival_ms
                )

                response_ms = (
                    finish_ms
                    -
                    arrival_ms
                )

                slack_ms = (
                    deadline_ms
                    -
                    finish_ms
                )

                miss = (
                    finish_ms
                    >
                    deadline_ms
                    + 1e-9
                )

                # ------------------------------------------------
                # post/NMS 仅生成检测结果，不进入 logical time
                # ------------------------------------------------

                pred_anno = (
                    make_prediction(
                        model,
                        dataset,
                        batch,
                    )
                )

                rows[idx] = {
                    "global_index":
                        idx,

                    "scene":
                        scene,

                    "local_pos":
                        pos,

                    "frame_id":
                        frame_id,

                    "arrival_ms":
                        arrival_ms,

                    "absolute_deadline_ms":
                        deadline_ms,

                    "status":
                        "processed",

                    "drop_reason":
                        "",

                    "forward_start_ms":
                        start_ms,

                    "forward_finish_ms":
                        finish_ms,

                    "prediction_timestamp_ms":
                        finish_ms,

                    "queue_wait_ms":
                        wait_ms,

                    "forward_ms":
                        forward_ms,

                    "arrival_to_finish_ms":
                        response_ms,

                    "deadline_slack_ms":
                        slack_ms,

                    "deadline_miss":
                        int(miss),

                    "true_level":
                        true_level,
                }

                events.append({
                    "scene":
                        scene,

                    "local_pos":
                        pos,

                    "source_index":
                        idx,

                    "source_frame_id":
                        frame_id,

                    "finish_ms":
                        finish_ms,

                    "anno":
                        pred_anno,
                })

                processed += 1

                misses += int(
                    miss
                )

                forward_times.append(
                    forward_ms
                )

                wait_times.append(
                    wait_ms
                )

                response_times.append(
                    response_ms
                )

                processed_level_hist[
                    true_level
                ] += 1

                # ------------------------------------------------
                # Latest-frame single-slot mailbox
                # ------------------------------------------------

                latest_arrived = min(
                    n - 1,
                    int(
                        np.floor(
                            (
                                finish_ms
                                +
                                1e-9
                            )
                            /
                            period_ms
                        )
                    ),
                )

                if (
                    latest_arrived
                    >=
                    pos + 1
                ):
                    next_pos = (
                        latest_arrived
                    )

                    for dp in range(
                        pos + 1,
                        next_pos,
                    ):
                        didx = (
                            indices[dp]
                        )

                        (
                            _,
                            dfid,
                            _,
                        ) = frame_meta(
                            dataset,
                            didx,
                        )

                        da = (
                            dp
                            *
                            period_ms
                        )

                        rows[didx] = {
                            "global_index":
                                didx,

                            "scene":
                                scene,

                            "local_pos":
                                dp,

                            "frame_id":
                                dfid,

                            "arrival_ms":
                                da,

                            "absolute_deadline_ms":
                                da
                                +
                                period_ms,

                            "status":
                                "dropped",

                            "drop_reason":
                                "stale_replaced_by_latest",

                            "forward_start_ms":
                                "",

                            "forward_finish_ms":
                                "",

                            "prediction_timestamp_ms":
                                "",

                            "queue_wait_ms":
                                "",

                            "forward_ms":
                                "",

                            "arrival_to_finish_ms":
                                "",

                            "deadline_slack_ms":
                                "",

                            "deadline_miss":
                                "",

                            # 保留被 drop sensor frame
                            # 原本对应的外部压力状态。
                            "true_level":
                                trace_by_idx[
                                    didx
                                ],
                        }

                        dropped += 1

                    pos = (
                        next_pos
                    )

                    gpu_free_ms = (
                        finish_ms
                    )

                else:
                    pos += 1

                    gpu_free_ms = (
                        finish_ms
                    )

                if (
                    processed <= 3
                    or
                    processed % 100 == 0
                ):
                    print(
                        f"[true={true_level}] "
                        f"processed="
                        f"{processed:04d} "
                        f"{scene}/{frame_id} "
                        f"arrival="
                        f"{arrival_ms:.3f} "
                        f"start="
                        f"{start_ms:.3f} "
                        f"finish="
                        f"{finish_ms:.3f} "
                        f"fwd="
                        f"{forward_ms:.3f} "
                        f"miss="
                        f"{int(miss)}"
                    )

        # ========================================================
        # Structural checks
        # ========================================================

        missing = [
            idx
            for idx
            in eval_indices
            if idx not in rows
        ]

        if missing:
            raise RuntimeError(
                f"missing statuses: "
                f"{missing[:10]}"
            )

        if (
            processed
            +
            dropped
            !=
            eval_count
        ):
            raise RuntimeError(
                "frame accounting "
                "mismatch: "
                f"{processed}+"
                f"{dropped}!="
                f"{eval_count}"
            )

        # ========================================================
        # Streaming sAP
        #
        # At query time a_i:
        # use latest prediction with finish <= a_i.
        # ========================================================

        events_by_scene = (
            OrderedDict(
                (
                    scene,
                    [],
                )
                for scene
                in groups
            )
        )

        for event in events:
            events_by_scene[
                event["scene"]
            ].append(
                event
            )

        for scene in (
            events_by_scene
        ):
            events_by_scene[
                scene
            ].sort(
                key=lambda e:
                    e["finish_ms"]
            )

        aligned = {}

        for (
            scene,
            indices,
        ) in groups.items():
            es = (
                events_by_scene[
                    scene
                ]
            )

            ptr = 0
            latest = None

            for pos, idx in (
                enumerate(indices)
            ):
                query_ms = (
                    pos
                    *
                    period_ms
                )

                while (
                    ptr
                    <
                    len(es)
                    and
                    es[ptr][
                        "finish_ms"
                    ]
                    <=
                    query_ms
                    +
                    1e-9
                ):
                    latest = (
                        es[ptr][
                            "anno"
                        ]
                    )

                    ptr += 1

                (
                    _,
                    fid,
                    next_fid,
                ) = frame_meta(
                    dataset,
                    idx,
                )

                aligned[idx] = (
                    copy_det(
                        latest,
                        scene,
                        fid,
                        next_fid,
                    )
                )

        gt_annos = [
            copy.deepcopy(
                dataset.kitti_infos[
                    idx
                ][
                    "infos"
                ][
                    "token"
                ][
                    "annos"
                ]
            )
            for idx
            in eval_indices
        ]

        det_annos = [
            aligned[idx]
            for idx
            in eval_indices
        ]

        (
            result_str,
            ap_dict,
        ) = (
            kitti_eval
            .get_official_eval_result(
                gt_annos,
                det_annos,
                dataset.class_names,
            )
        )

        car = float(
            ap_dict.get(
                "Car_3d/moderate_R40",
                np.nan,
            )
        )

        ped = float(
            ap_dict.get(
                "Pedestrian_3d/moderate_R40",
                np.nan,
            )
        )

        cyc = float(
            ap_dict.get(
                "Cyclist_3d/moderate_R40",
                np.nan,
            )
        )

        macro = float(
            np.nanmean(
                [
                    car,
                    ped,
                    cyc,
                ]
            )
        )

        # ========================================================
        # Outputs
        # ========================================================

        # --------------------------------------------------------
        # MTD routing audit
        # --------------------------------------------------------

        mtd_decision_path = (
            out_dir
            /
            "mtd_decisions.csv"
        )

        if len(mtd_decisions) > 0:
            with mtd_decision_path.open(
                "w",
                newline="",
            ) as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=list(
                        mtd_decisions[0].keys()
                    ),
                )

                w.writeheader()
                w.writerows(
                    mtd_decisions
                )

        timeline_path = (
            out_dir
            /
            "frame_timeline.csv"
        )

        fieldnames = [
            "global_index",
            "scene",
            "local_pos",
            "frame_id",
            "arrival_ms",
            "absolute_deadline_ms",
            "status",
            "drop_reason",
            "forward_start_ms",
            "forward_finish_ms",
            "prediction_timestamp_ms",
            "queue_wait_ms",
            "forward_ms",
            "arrival_to_finish_ms",
            "deadline_slack_ms",
            "deadline_miss",
            "true_level",
        ]

        with timeline_path.open(
            "w",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )

            w.writeheader()

            for idx in eval_indices:
                w.writerow(
                    rows[idx]
                )

        with (
            out_dir
            /
            "prediction_events.pkl"
        ).open("wb") as f:
            pickle.dump(
                events,
                f,
                protocol=(
                    pickle
                    .HIGHEST_PROTOCOL
                ),
            )

        with (
            out_dir
            /
            "stream_det_annos.pkl"
        ).open("wb") as f:
            pickle.dump(
                det_annos,
                f,
                protocol=(
                    pickle
                    .HIGHEST_PROTOCOL
                ),
            )

        (
            out_dir
            /
            "stream_sap_result.txt"
        ).write_text(
            result_str
        )

        (
            out_dir
            /
            "stream_sap_dict.json"
        ).write_text(
            json.dumps(
                builtin(
                    ap_dict
                ),
                indent=2,
            )
            +
            "\n"
        )

        summary = {
            "version":
                "mtd_three_head_streamdsgn_"
                "fixed_level_forward_only_v1",

            "method":
                "MTD Three-Head StreamDSGN",

            "base_detector":
                "k3_streamdsgn_shared_trunk",

            "model_cfg":
                str(args.cfg),

            "model_ckpt":
                str(args.ckpt),

            "mtd_heads": {
                "H1": {
                    "target": "next",
                    "checkpoint": str(
                        args.ckpt
                    ),
                },

                "H2": {
                    "target": "next2",
                    "checkpoint": str(
                        args.h2_ckpt
                    ),
                },

                "H3": {
                    "target": "next3",
                    "checkpoint": str(
                        args.h3_ckpt
                    ),
                },
            },

            "mtd_routing": {
                "decision_time":
                    "before_current_forward",

                "runtime_estimator":
                    "min_last_two_forward_ms",

                "estimated_response":
                    "queue_wait_plus_estimated_forward",

                "branch_rule":
                    "clip(floor(response/period)+1,1,3)",

                "branch_histogram":
                    dict(
                        mtd_branch_hist
                    ),

                "decision_csv":
                    str(
                        mtd_decision_path
                    ),

                "oracle_match_rate":
                    (
                        sum(
                            x[
                                "branch_match_oracle"
                            ]
                            for x
                            in mtd_decisions
                        )
                        /
                        len(mtd_decisions)
                        if len(
                            mtd_decisions
                        ) > 0
                        else 0.0
                    ),
            },

            "input_hz":
                args.input_hz,

            "period_ms":
                period_ms,

            "timing_scope":
                "forward_only",

            "no_load":
                bool(
                    args.pressure_level == "L0"
                    and
                    args.pressure_fraction == 0.0
                ),

            "contention_trace": {
                "type":
                    "balanced_random_"
                    "per_sensor_frame",

                "base_level":
                    "L0",

                "pressure_level":
                    args.pressure_level,

                "pressure_fraction_target":
                    args.pressure_fraction,

                "trace_seed":
                    args.trace_seed,

                "trace_csv":
                    str(
                        trace_csv
                    ),

                "scene_summary":
                    trace_scene_summary,
            },

            "true_sensor_levels":
                dict(
                    sensor_level_hist
                ),

            "true_processed_levels":
                dict(
                    processed_level_hist
                ),

            "buffer_policy":
                "latest_frame_only",

            "history_policy":
                "processed_frames_only",

            "sensor_frames":
                eval_count,

            "processed_frames":
                processed,

            "dropped_frames":
                dropped,

            "drop_rate":
                dropped
                /
                eval_count,

            "deadline_miss_count":
                misses,

            "deadline_miss_rate":
                misses
                /
                processed,

            "forward_latency":
                stats(
                    forward_times
                ),

            "queue_wait":
                stats(
                    wait_times
                ),

            "arrival_to_finish":
                stats(
                    response_times
                ),

            "stream_sap_3d_moderate_R40": {
                "Car":
                    car,

                "Pedestrian":
                    ped,

                "Cyclist":
                    cyc,

                "Macro":
                    macro,
            },
        }

        summary_path = (
            out_dir
            /
            "summary.json"
        )

        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            )
            +
            "\n"
        )

        fs = stats(
            forward_times
        )

        print()
        print("=" * 100)
        print(
            "Original StreamDSGN "
            f"@ {args.input_hz:g} Hz / "
            f"L0+{args.pressure_level}"
        )
        print(
            f"sensor              : "
            f"{eval_count}"
        )
        print(
            "processed / dropped : "
            f"{processed} / "
            f"{dropped}"
        )
        print(
            f"drop rate           : "
            f"{100*dropped/eval_count:.3f}%"
        )
        print(
            "deadline miss       : "
            f"{misses}/{processed} "
            f"("
            f"{100*misses/processed:.3f}%"
            f")"
        )
        print(
            "forward p50/p90/p99 : "
            f"{fs['p50_ms']:.4f} / "
            f"{fs['p90_ms']:.4f} / "
            f"{fs['p99_ms']:.4f} ms"
        )
        print(
            "sAP 3D Moderate R40 : "
            f"Car={car:.4f} "
            f"Ped={ped:.4f} "
            f"Cyc={cyc:.4f} "
            f"Macro={macro:.4f}"
        )
        print(
            f"true sensor levels  : "
            f"{dict(sensor_level_hist)}"
        )
        print(
            f"true processed      : "
            f"{dict(processed_level_hist)}"
        )
        print(
            f"summary             : "
            f"{summary_path}"
        )
        print("=" * 100)

    finally:
        model.generate_recall_record = (
            original_recall
        )


if __name__ == "__main__":
    main()

# ORIGINAL_STREAMDSGN_RANDOM50_EOF
