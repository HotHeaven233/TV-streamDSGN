#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import zlib
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import ElasticBEVBranch
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

from test_stream_buffer_timestamp import load_one
from test_tv_stream3d_online_forward import (
    make_cfg,
    choose_scene_indices,
    configure_contender,
    execute_one,
    prewarm_all_causal_prefixes,
)
from tv_stream3d_controller import TVStream3DController
from tv_stream3d_causal_fused_runtime import (
    enable_causal_prefix_fused_cache,
    fused_cache_stats,
    set_forbid_cache_miss,
)
from smooth_cuda_contention import make_high_priority_detector_stream


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--prefix_bn_bank", required=True)
    p.add_argument("--controller_csv", required=True)
    p.add_argument("--levels_json", required=True)
    p.add_argument(
        "--pressure_level",
        required=True,
        choices=["L1", "L2", "L3", "L4"],
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
    p.add_argument("--input_hz", type=float, default=30.0)
    p.add_argument("--runtime_warmup_frames", type=int, default=80)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--control_guard_per_boundary_ms", type=float, default=0.25)
    p.add_argument("--output_dir", required=True)
    return p.parse_args()


def stats(xs):
    x = np.asarray(xs, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean_ms": float(x.mean()),
        "p50_ms": float(np.percentile(x, 50)),
        "p90_ms": float(np.percentile(x, 90)),
        "p99_ms": float(np.percentile(x, 99)),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
    }


def builtin(x):
    if isinstance(x, dict):
        return {str(k): builtin(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [builtin(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.item()
        return x.detach().cpu().tolist()
    return x


def reset_history(model):
    q = getattr(model, "history_feature_queue", None)
    if q is not None:
        q.clear()


def frame_meta(dataset, idx):
    info = dataset.kitti_infos[idx]["sample_idx"]
    scene = str(info["scene"])
    tag = info["frame_tag"]
    return scene, str(tag["token"]), str(tag.get("next", ""))


def scene_groups(dataset, indices):
    out = OrderedDict()
    closed = set()
    last = None

    for idx in indices:
        scene, _, _ = frame_meta(dataset, idx)

        if scene != last:
            if scene in closed:
                raise RuntimeError(
                    f"dataset is not scene-contiguous: {scene}"
                )
            if last is not None:
                closed.add(last)
            last = scene

        out.setdefault(scene, []).append(idx)

    return out


def build_balanced_trace(
    groups,
    pressure_level,
    fraction,
    seed,
):
    """
    Contention is attached to SENSOR FRAME ID, not processed-job order.

    For each scene independently:
      - choose approximately fraction*N sensor frames,
      - assign them pressure_level,
      - assign all remaining frames L0,
      - use a deterministic scene-specific seed.

    Therefore if a sensor frame is later dropped by the latest-frame
    mailbox, its preassigned contention state is simply skipped.
    """

    trace = {}
    scene_summary = {}

    for scene, indices in groups.items():
        n = len(indices)

        # nearest integer; at fraction=0.5 this is as close to exact half
        # as possible for odd-length scenes.
        k = int(
            np.floor(
                n * fraction
                + 0.5
            )
        )

        k = max(
            0,
            min(n, k),
        )

        # Do NOT use Python hash(scene), because hash randomization would
        # change traces between processes.
        scene_crc = zlib.crc32(
            scene.encode("utf-8")
        )

        scene_seed = (
            int(seed)
            + int(scene_crc)
        ) & 0xFFFFFFFF

        rng = np.random.default_rng(
            scene_seed
        )

        if k > 0:
            pressured_positions = set(
                int(x)
                for x
                in rng.choice(
                    n,
                    size=k,
                    replace=False,
                ).tolist()
            )
        else:
            pressured_positions = set()

        levels = []

        for pos, idx in enumerate(indices):
            level = (
                pressure_level
                if pos in pressured_positions
                else "L0"
            )

            trace[idx] = level
            levels.append(level)

        switches = sum(
            levels[j] != levels[j - 1]
            for j in range(1, len(levels))
        )

        scene_summary[scene] = {
            "sensor_frames": int(n),
            "pressure_frames": int(k),
            "l0_frames": int(n - k),
            "pressure_fraction_realized": (
                float(k / n)
                if n > 0
                else 0.0
            ),
            "switches": int(switches),
            "scene_seed": int(scene_seed),
        }

    return trace, scene_summary


def empty_det(scene, frame_id, next_frame_id=""):
    return {
        "name": np.array([], dtype="<U1"),
        "truncated": np.array([], dtype=np.float32),
        "occluded": np.array([], dtype=np.float32),
        "alpha": np.array([], dtype=np.float32),
        "bbox": np.zeros((0, 4), dtype=np.float32),
        "dimensions": np.zeros((0, 3), dtype=np.float32),
        "location": np.zeros((0, 3), dtype=np.float32),
        "rotation_y": np.array([], dtype=np.float32),
        "score": np.array([], dtype=np.float32),
        "boxes_lidar": np.zeros((0, 7), dtype=np.float32),
        "scene": scene,
        "frame_id": frame_id,
        "next_frame_id": next_frame_id,
    }


def copy_det(pred, scene, frame_id, next_frame_id):
    if pred is None:
        return empty_det(scene, frame_id, next_frame_id)

    out = copy.deepcopy(pred)
    out["scene"] = scene
    out["frame_id"] = frame_id
    out["next_frame_id"] = next_frame_id
    return out


def disable_recall(model):
    if not hasattr(model, "generate_recall_record"):
        raise RuntimeError("model has no generate_recall_record")

    original = model.generate_recall_record

    def no_recall(box_preds, recall_dict, batch_index,
                  data_dict=None, thresh_list=None):
        return recall_dict, None, None

    model.generate_recall_record = no_recall
    return original


def make_prediction(model, dataset, batch):
    # post-processing 只生成检测结果，不进入逻辑时间。
    with torch.no_grad():
        pred_dicts, _ = model.post_processing(batch["token"])

    # 防止 NMS CUDA 工作残留到下一次被计时的 forward。
    torch.cuda.synchronize()

    annos = dataset.generate_prediction_dicts(
        batch,
        pred_dicts,
        dataset.class_names,
        output_path=None,
    )

    if len(annos) != 1:
        raise RuntimeError(f"expected batch size 1, got {len(annos)}")

    return annos[0]


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.input_hz <= 0:
        raise ValueError("input_hz must be > 0")
    if args.max_frames < 0:
        raise ValueError("max_frames must be >= 0")

    if not (0.0 < args.pressure_fraction < 1.0):
        raise ValueError(
            "pressure_fraction must satisfy 0 < fraction < 1"
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    period_ms = 1000.0 / args.input_hz

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(args.full_cfg)
    logger = common_utils.create_logger()

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    if not hasattr(dataset, "kitti_infos"):
        raise RuntimeError("dataset has no kitti_infos")

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )

    model.cuda().eval()

    branch = ElasticBEVBranch(
        model.backbone_3d,
        output_bev_channels=int(
            cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
        ),
    ).cuda().eval()

    elastic = torch.load(
        args.elastic_ckpt,
        map_location="cpu",
    )

    if "branch" not in elastic:
        raise RuntimeError("elastic checkpoint has no 'branch'")

    branch.load_state_dict(
        elastic["branch"],
        strict=True,
    )

    bank = torch.load(
        args.prefix_bn_bank,
        map_location="cpu",
    )

    if bank.get("version") != "elastic_v4_bn_causal_prefix_bank_v1":
        raise RuntimeError(
            f"wrong BN bank version: {bank.get('version')}"
        )

    enable_causal_prefix_fused_cache(branch)

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

    contenders = {
        "L0": configure_contender(
            levels_data,
            "L0",
        ),
        args.pressure_level: configure_contender(
            levels_data,
            args.pressure_level,
        ),
    }

    model_stream = make_high_priority_detector_stream(
        torch.cuda.current_device()
    )

    # ------------------------------------------------------------
    # deterministic FP32 causal-prefix prewarm
    # ------------------------------------------------------------

    _, warm_indices = choose_scene_indices(
        dataset,
        max(1, args.runtime_warmup_frames + 1),
    )

    warm_batch = load_one(
        dataset,
        warm_indices[0],
    )

    reset_history(model)

    with torch.no_grad():
        pw_fp32 = prewarm_all_causal_prefixes(
            base_model=model,
            branch=branch,
            bank=bank,
            controller=controller,
            batch=warm_batch,
            model_stream=model_stream,
        )

    if (
        pw_fp32.get("expected_prefixes") != 203
        or pw_fp32.get("ready_prefixes") != 203
    ):
        raise RuntimeError(
            f"FP32 prewarm incomplete: {pw_fp32}"
        )

    set_forbid_cache_miss(branch, True)

    cache0 = fused_cache_stats(branch)

    # ------------------------------------------------------------
    # deterministic runtime AMP path prewarm
    # ------------------------------------------------------------

    with (
        torch.no_grad(),
        torch.amp.autocast(
            "cuda",
            enabled=bool(model.use_amp_dict["TEST"]),
        ),
    ):
        pw_amp = prewarm_all_causal_prefixes(
            base_model=model,
            branch=branch,
            bank=bank,
            controller=controller,
            batch=warm_batch,
            model_stream=model_stream,
        )

    cache1 = fused_cache_stats(branch)

    if cache1 != cache0:
        raise RuntimeError(
            f"AMP prewarm changed cache: {cache0} -> {cache1}"
        )

    if (
        pw_amp.get("expected_prefixes") != 203
        or pw_amp.get("ready_prefixes") != 203
    ):
        raise RuntimeError(
            f"AMP prewarm incomplete: {pw_amp}"
        )

    original_recall = disable_recall(model)

    # ------------------------------------------------------------
    # runtime warmup under the requested true contention level
    # ------------------------------------------------------------

    reset_history(model)
    last_batch = None

    for wi in range(args.runtime_warmup_frames):
        idx = warm_indices[wi % len(warm_indices)]
        batch = load_one(dataset, idx)

        warm_level = (
            args.pressure_level
            if wi % 2 == 1
            else "L0"
        )

        execute_one(
            base_model=model,
            branch=branch,
            bank=bank,
            controller=controller,
            batch=batch,
            model_stream=model_stream,
            contender=contenders[warm_level],
            deadline_ms=period_ms,
            allow_prepare=False,
        )

        last_batch = batch

    # NMS/extension 的首次初始化也放到正式测量前。
    if last_batch is not None:
        make_prediction(
            model,
            dataset,
            last_batch,
        )

    reset_history(model)
    torch.cuda.synchronize()

    cache2 = fused_cache_stats(branch)

    if cache2 != cache1:
        raise RuntimeError(
            f"runtime warmup changed cache: {cache1} -> {cache2}"
        )

    total = len(dataset)

    eval_count = (
        total
        if args.max_frames == 0
        else min(total, args.max_frames)
    )

    eval_indices = list(range(eval_count))
    groups = scene_groups(dataset, eval_indices)

    trace_by_idx, trace_scene_summary = build_balanced_trace(
        groups=groups,
        pressure_level=args.pressure_level,
        fraction=args.pressure_fraction,
        seed=args.trace_seed,
    )

    true_sensor_hist = Counter(
        trace_by_idx[idx]
        for idx in eval_indices
    )

    trace_switches = 0
    trace_pairs = 0

    for _, indices in groups.items():
        for j in range(1, len(indices)):
            trace_pairs += 1

            if (
                trace_by_idx[indices[j]]
                !=
                trace_by_idx[indices[j - 1]]
            ):
                trace_switches += 1

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

        for scene, indices in groups.items():
            for local_pos, idx in enumerate(indices):
                _, frame_id, _ = frame_meta(
                    dataset,
                    idx,
                )

                w.writerow({
                    "global_index": idx,
                    "scene": scene,
                    "local_pos": local_pos,
                    "frame_id": frame_id,
                    "true_level": trace_by_idx[idx],
                })

    print("=" * 96)
    print("TV-Stream3D formal streaming evaluation")
    print(
        "contention      : "
        f"L0 + {args.pressure_level}"
    )
    print(
        "pressure target : "
        f"{100.0 * args.pressure_fraction:.2f}%"
    )
    print(
        f"trace seed      : {args.trace_seed}"
    )
    print(
        f"sensor levels   : {dict(true_sensor_hist)}"
    )
    print(
        "trace switches  : "
        f"{trace_switches}/{trace_pairs} "
        f"({100.0 * trace_switches / max(1, trace_pairs):.3f}%)"
    )
    print(f"input Hz       : {args.input_hz}")
    print(f"period         : {period_ms:.6f} ms")
    print("timing scope   : forward only")
    print("buffer         : latest frame only")
    print("history        : processed frames only")
    print(f"sensor frames  : {eval_count}")
    print(f"scenes         : {len(groups)}")
    print(f"cache          : {cache2}")
    print("=" * 96)

    rows = {}
    events = []
    fwd_times = []
    wait_times = []
    response_times = []
    observed_hist = Counter()
    true_processed_hist = Counter()
    true_to_observed = Counter()
    schedule_hist = Counter()

    processed = 0
    dropped = 0
    misses = 0

    try:
        for scene_i, (scene, indices) in enumerate(
            groups.items(),
            start=1,
        ):
            reset_history(model)

            pos = 0
            gpu_free_ms = 0.0
            n = len(indices)

            print(
                f"[scene {scene_i:02d}/{len(groups):02d}] "
                f"{scene}: {n} frames"
            )

            while pos < n:
                idx = indices[pos]

                meta_scene, frame_id, next_frame_id = frame_meta(
                    dataset,
                    idx,
                )

                if meta_scene != scene:
                    raise RuntimeError("scene mismatch")

                arrival_ms = pos * period_ms
                deadline_ms = arrival_ms + period_ms

                start_ms = max(
                    gpu_free_ms,
                    arrival_ms,
                )

                # 这是当前 frame 真正开始时剩下的总 forward deadline。
                budget_ms = deadline_ms - start_ms

                # 即使已经 infeasible，也按方案 A 继续处理。
                # 只防止现有 execute_one 对非正 deadline 的断言。
                runtime_budget_ms = max(
                    budget_ms,
                    1e-6,
                )

                true_level = trace_by_idx[idx]

                batch = load_one(
                    dataset,
                    idx,
                )

                result = execute_one(
                    base_model=model,
                    branch=branch,
                    bank=bank,
                    controller=controller,
                    batch=batch,
                    model_stream=model_stream,
                    contender=contenders[true_level],
                    deadline_ms=runtime_budget_ms,
                    allow_prepare=False,
                )

                forward_ms = float(
                    result["forward_ms"]
                )

                finish_ms = (
                    start_ms
                    + forward_ms
                )

                wait_ms = (
                    start_ms
                    - arrival_ms
                )

                response_ms = (
                    finish_ms
                    - arrival_ms
                )

                slack_ms = (
                    deadline_ms
                    - finish_ms
                )

                miss = (
                    finish_ms
                    >
                    deadline_ms
                    + 1e-9
                )

                # 检测框生成不推进 streaming logical clock。
                pred_anno = make_prediction(
                    model,
                    dataset,
                    batch,
                )

                observed = str(
                    result.get(
                        "observed_level",
                        "UNKNOWN",
                    )
                )

                schedule = tuple(
                    float(x)
                    for x
                    in result["schedule"]
                )

                schedule_s = ",".join(
                    f"{x:g}"
                    for x
                    in schedule
                )

                rows[idx] = {
                    "global_index": idx,
                    "scene": scene,
                    "local_pos": pos,
                    "frame_id": frame_id,
                    "arrival_ms": arrival_ms,
                    "absolute_deadline_ms": deadline_ms,
                    "status": "processed",
                    "drop_reason": "",
                    "forward_start_ms": start_ms,
                    "forward_finish_ms": finish_ms,
                    "prediction_timestamp_ms": finish_ms,
                    "initial_budget_ms": budget_ms,
                    "queue_wait_ms": wait_ms,
                    "forward_ms": forward_ms,
                    "arrival_to_finish_ms": response_ms,
                    "deadline_slack_ms": slack_ms,
                    "deadline_miss": int(miss),
                    "true_level": true_level,
                    "observed_level": observed,
                    "schedule": schedule_s,
                }

                events.append(
                    {
                        "scene": scene,
                        "local_pos": pos,
                        "source_index": idx,
                        "source_frame_id": frame_id,
                        "finish_ms": finish_ms,
                        "anno": pred_anno,
                    }
                )

                processed += 1
                misses += int(miss)

                fwd_times.append(forward_ms)
                wait_times.append(wait_ms)
                response_times.append(response_ms)

                observed_hist[observed] += 1
                true_processed_hist[true_level] += 1
                true_to_observed[
                    f"{true_level}->{observed}"
                ] += 1
                schedule_hist[schedule_s] += 1

                # ------------------------------------------------
                # latest-frame mailbox
                #
                # finish 时刻已经到达的帧中，只取最新的一张。
                # 中间 pending frames 全部 stale drop。
                # ------------------------------------------------

                latest_arrived = min(
                    n - 1,
                    int(
                        np.floor(
                            (
                                finish_ms
                                + 1e-9
                            )
                            / period_ms
                        )
                    ),
                )

                if latest_arrived >= pos + 1:
                    next_pos = latest_arrived

                    for dp in range(
                        pos + 1,
                        next_pos,
                    ):
                        didx = indices[dp]
                        _, dfid, _ = frame_meta(
                            dataset,
                            didx,
                        )

                        da = dp * period_ms

                        rows[didx] = {
                            "global_index": didx,
                            "scene": scene,
                            "local_pos": dp,
                            "frame_id": dfid,
                            "arrival_ms": da,
                            "absolute_deadline_ms": da + period_ms,
                            "status": "dropped",
                            "drop_reason": "stale_replaced_by_latest",
                            "forward_start_ms": "",
                            "forward_finish_ms": "",
                            "prediction_timestamp_ms": "",
                            "initial_budget_ms": "",
                            "queue_wait_ms": "",
                            "forward_ms": "",
                            "arrival_to_finish_ms": "",
                            "deadline_slack_ms": "",
                            "deadline_miss": "",
                            "true_level": trace_by_idx[didx],
                            "observed_level": "",
                            "schedule": "",
                        }

                        dropped += 1

                    pos = next_pos
                    gpu_free_ms = finish_ms

                else:
                    pos += 1
                    gpu_free_ms = finish_ms

                if (
                    processed <= 3
                    or processed % 100 == 0
                ):
                    print(
                        f"[true={true_level}] "
                        f"processed={processed:04d} "
                        f"{scene}/{frame_id} "
                        f"arrival={arrival_ms:.3f} "
                        f"start={start_ms:.3f} "
                        f"finish={finish_ms:.3f} "
                        f"budget={budget_ms:.3f} "
                        f"fwd={forward_ms:.3f} "
                        f"miss={int(miss)} "
                        f"obs={observed} "
                        f"s={schedule_s}"
                    )

        missing = [
            i for i in eval_indices
            if i not in rows
        ]

        if missing:
            raise RuntimeError(
                f"missing statuses: {missing[:10]}"
            )

        if processed + dropped != eval_count:
            raise RuntimeError(
                "frame accounting mismatch: "
                f"{processed}+{dropped}!={eval_count}"
            )

        # --------------------------------------------------------
        # streaming sAP:
        #
        # 每个 sensor query 时间点，只能使用该时间点之前
        # 已经完成 forward 的最近 prediction。
        # --------------------------------------------------------

        events_by_scene = OrderedDict(
            (scene, [])
            for scene in groups
        )

        for e in events:
            events_by_scene[
                e["scene"]
            ].append(e)

        for scene in events_by_scene:
            events_by_scene[scene].sort(
                key=lambda e: e["finish_ms"]
            )

        aligned = {}

        for scene, indices in groups.items():
            es = events_by_scene[scene]
            ptr = 0
            latest = None

            for pos, idx in enumerate(indices):
                query_ms = pos * period_ms

                while (
                    ptr < len(es)
                    and es[ptr]["finish_ms"]
                    <= query_ms + 1e-9
                ):
                    latest = es[ptr]["anno"]
                    ptr += 1

                _, fid, next_fid = frame_meta(
                    dataset,
                    idx,
                )

                aligned[idx] = copy_det(
                    latest,
                    scene,
                    fid,
                    next_fid,
                )

        gt_annos = [
            copy.deepcopy(
                dataset.kitti_infos[
                    idx
                ]["infos"]["token"]["annos"]
            )
            for idx in eval_indices
        ]

        det_annos = [
            aligned[idx]
            for idx in eval_indices
        ]

        result_str, ap_dict = (
            kitti_eval.get_official_eval_result(
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
                [car, ped, cyc]
            )
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
            "initial_budget_ms",
            "queue_wait_ms",
            "forward_ms",
            "arrival_to_finish_ms",
            "deadline_slack_ms",
            "deadline_miss",
            "true_level",
            "observed_level",
            "schedule",
        ]

        timeline = (
            out_dir
            /
            "frame_timeline.csv"
        )

        with timeline.open(
            "w",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=fieldnames,
            )
            w.writeheader()

            for idx in eval_indices:
                w.writerow(rows[idx])

        with (
            out_dir
            /
            "prediction_events.pkl"
        ).open("wb") as f:
            pickle.dump(
                events,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        with (
            out_dir
            /
            "stream_det_annos.pkl"
        ).open("wb") as f:
            pickle.dump(
                det_annos,
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        (
            out_dir
            /
            "stream_sap_result.txt"
        ).write_text(result_str)

        (
            out_dir
            /
            "stream_sap_dict.json"
        ).write_text(
            json.dumps(
                builtin(ap_dict),
                indent=2,
            )
            + "\n"
        )

        summary = {
            "version":
                "tv_stream3d_30hz_forward_only_v1",

            "method":
                "TV-Stream3D",

            "contention_trace":
                {
                    "type":
                        "balanced_random_per_sensor_frame",

                    "base_level":
                        "L0",

                    "pressure_level":
                        args.pressure_level,

                    "pressure_fraction_target":
                        args.pressure_fraction,

                    "trace_seed":
                        args.trace_seed,

                    "trace_switches":
                        trace_switches,

                    "trace_adjacent_pairs":
                        trace_pairs,

                    "trace_switch_rate":
                        trace_switches
                        /
                        max(1, trace_pairs),

                    "scene_summary":
                        trace_scene_summary,

                    "trace_csv":
                        str(trace_csv),
                },

            "true_sensor_levels":
                dict(true_sensor_hist),

            "true_processed_levels":
                dict(true_processed_hist),

            "input_hz":
                args.input_hz,

            "period_ms":
                period_ms,

            "timing_scope":
                "forward_only",

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
                dropped / eval_count,

            "deadline_miss_count":
                misses,

            "deadline_miss_rate":
                misses / processed,

            "forward_latency":
                stats(fwd_times),

            "queue_wait":
                stats(wait_times),

            "arrival_to_finish":
                stats(response_times),

            "observed_levels":
                dict(observed_hist),

            "true_to_observed":
                dict(true_to_observed),

            "classifier_accuracy":
                (
                    sum(
                        count
                        for key, count
                        in true_to_observed.items()
                        if (
                            key.split("->", 1)[0]
                            ==
                            key.split("->", 1)[1]
                        )
                    )
                    /
                    max(1, processed)
                ),

            "schedule_histogram":
                dict(schedule_hist),

            "stream_sap_3d_moderate_R40":
                {
                    "Car": car,
                    "Pedestrian": ped,
                    "Cyclist": cyc,
                    "Macro": macro,
                },

            "cache":
                builtin(cache2),
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
            + "\n"
        )

        print()
        print("=" * 96)
        print(
            f"TV-Stream3D @ {args.input_hz:g} Hz / "
            f"50%-style L0+{args.pressure_level}"
        )
        print(
            f"sensor              : {eval_count}"
        )
        print(
            f"processed / dropped : {processed} / {dropped}"
        )
        print(
            f"drop rate           : {100*dropped/eval_count:.3f}%"
        )
        print(
            f"deadline miss       : {misses}/{processed} "
            f"({100*misses/processed:.3f}%)"
        )

        fs = stats(fwd_times)

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
            f"true sensor levels  : {dict(true_sensor_hist)}"
        )

        print(
            f"true processed      : {dict(true_processed_hist)}"
        )

        print(
            f"observed levels     : {dict(observed_hist)}"
        )

        cls_correct = sum(
            count
            for key, count
            in true_to_observed.items()
            if (
                key.split("->", 1)[0]
                ==
                key.split("->", 1)[1]
            )
        )

        print(
            "classifier accuracy : "
            f"{100.0 * cls_correct / max(1, processed):.3f}%"
        )

        print(
            f"true->observed      : {dict(true_to_observed)}"
        )

        print(
            f"trace CSV           : {trace_csv}"
        )

        print(
            f"summary             : {summary_path}"
        )

        print("=" * 96)

    finally:
        model.generate_recall_record = original_recall


if __name__ == "__main__":
    main()

# TV30_EVAL_EOF
