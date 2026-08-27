#!/usr/bin/env python3

import argparse
import copy
import json
import pickle
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


torch.backends.cudnn.benchmark = True


# ============================================================
# Argument
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Original StreamDSGN with a capacity-1 latest-frame buffer, "
            "real measured inference latency, output timestamps, "
            "and timestamp-aligned streaming AP."
        )
    )

    parser.add_argument(
        "--cfg_file",
        type=str,
        default=(
            "configs/stream/kitti_models/"
            "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl.yaml"
        ),
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default="extra_data/checkpoint_epoch_20.pth",
    )

    parser.add_argument(
        "--input_hz",
        type=float,
        default=None,
        help=(
            "Sensor input frequency. "
            "Default: DATA_CONFIG.ANNOS_FREQUENCY from config."
        ),
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of model-only warmup forwards before timing.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/stream_buffer_timestamp",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    return parser.parse_args()


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_history(model):
    """
    StreamDSGN history must contain only actually processed frames.

    Dropped frames are never passed to model(), therefore they never enter
    the history queue.
    """
    queue = getattr(model, "history_feature_queue", None)

    if queue is not None:
        queue.clear()


def frame_token(dataset, dataset_index):
    info = dataset.kitti_infos[dataset_index]

    return str(
        info["sample_idx"]["frame_tag"]["token"]
    )


def frame_sort_key(dataset, dataset_index):
    token = frame_token(
        dataset,
        dataset_index,
    )

    try:
        return (0, int(token))
    except ValueError:
        return (1, token)


def build_scene_index(dataset):
    """
    dataset index -> scene -> chronological frame sequence.
    """

    scenes = OrderedDict()

    for dataset_index, info in enumerate(dataset.kitti_infos):
        scene = str(
            info["sample_idx"]["scene"]
        )

        scenes.setdefault(
            scene,
            [],
        ).append(dataset_index)

    for scene in scenes:
        scenes[scene].sort(
            key=lambda x: frame_sort_key(
                dataset,
                x,
            )
        )

    return scenes


def make_empty_prediction():
    """
    Empty detection used before the first prediction has been published.
    """

    return {
        "name": np.array([], dtype=str),
        "truncated": np.array([], dtype=np.float32),
        "occluded": np.array([], dtype=np.float32),
        "alpha": np.array([], dtype=np.float32),
        "bbox": np.empty(
            (0, 4),
            dtype=np.float32,
        ),
        "dimensions": np.empty(
            (0, 3),
            dtype=np.float32,
        ),
        "location": np.empty(
            (0, 3),
            dtype=np.float32,
        ),
        "rotation_y": np.array(
            [],
            dtype=np.float32,
        ),
        "score": np.array(
            [],
            dtype=np.float32,
        ),
        "boxes_lidar": np.empty(
            (0, 7),
            dtype=np.float32,
        ),
    }


# ============================================================
# Data + model
# ============================================================

def load_one(dataset, dataset_index):
    """
    Data loading and H2D are deliberately outside the model timer.

    This follows the timing boundary of tools/gen_infer_time.py.
    """

    data_dict = dataset[dataset_index]

    batch_dict = dataset.collate_batch(
        [data_dict]
    )

    load_data_to_gpu(batch_dict)

    return batch_dict


def warmup_model(
    model,
    dataset,
    scene_to_indices,
    warmup,
):
    if warmup <= 0:
        return

    first_scene = next(
        iter(scene_to_indices)
    )

    first_index = scene_to_indices[
        first_scene
    ][0]

    print(
        f"[warmup] {warmup} forwards, "
        f"dataset index={first_index}"
    )

    model.eval()

    with torch.no_grad():

        for i in range(warmup):

            # Every warmup behaves as a fresh scene.
            clear_history(model)

            batch_dict = load_one(
                dataset,
                first_index,
            )

            torch.cuda.synchronize()

            model(batch_dict)

            torch.cuda.synchronize()

            if (
                (i + 1) % 5 == 0
                or
                i + 1 == warmup
            ):
                print(
                    f"[warmup] "
                    f"{i + 1}/{warmup}"
                )

    clear_history(model)


def run_one_prediction(
    model,
    dataset,
    dataset_index,
):
    """
    Run exactly one original StreamDSGN forward.

    service_ms includes:
        model forward
        temporal fusion
        detection head
        post-processing / NMS

    It excludes:
        disk loading
        preprocessing
        H2D
    """

    batch_dict = load_one(
        dataset,
        dataset_index,
    )

    # Make all previous CUDA work complete.
    torch.cuda.synchronize()

    start_ns = time.perf_counter_ns()

    with torch.no_grad():
        pred_dicts, _ = model(
            batch_dict
        )

    # Prediction is considered published only after all CUDA work finishes.
    torch.cuda.synchronize()

    finish_ns = time.perf_counter_ns()

    service_ms = (
        finish_ns - start_ns
    ) / 1e6

    # Real host wall-clock timestamp.
    wall_finish_ns = time.time_ns()

    annos = dataset.generate_prediction_dicts(
        batch_dict,
        pred_dicts,
        dataset.class_names,
        output_path=None,
    )

    if len(annos) != 1:
        raise RuntimeError(
            "This test requires batch_size=1, "
            f"but got {len(annos)}"
        )

    return (
        annos[0],
        service_ms,
        wall_finish_ns,
    )


# ============================================================
# Timestamp
# ============================================================

def attach_timestamp(
    anno,
    scene,
    input_frame_id,
    arrival_ms,
    start_ms,
    finish_ms,
    service_ms,
    wall_finish_ns,
):
    """
    Attach all streaming timestamps to the actual prediction.
    """

    out = copy.deepcopy(anno)

    out["_stream_scene"] = scene

    out["_stream_input_frame_id"] = str(
        input_frame_id
    )

    out["_stream_input_ts_ms"] = float(
        arrival_ms
    )

    out["_stream_start_ts_ms"] = float(
        start_ms
    )

    # THIS is the timestamp used by streaming AP.
    out["_stream_output_ts_ms"] = float(
        finish_ms
    )

    out["_stream_service_ms"] = float(
        service_ms
    )

    # Only for audit/debug.
    out["_wall_output_time_ns"] = int(
        wall_finish_ns
    )

    return out


# ============================================================
# Single-slot latest-frame buffer
# ============================================================

def run_scene(
    model,
    dataset,
    scene,
    indices,
    period_ms,
    trace,
    processed_predictions,
):
    """
    Real streaming protocol.

    Buffer capacity = 1.

    If model is busy:

        frame t arrives
            -> enters buffer

        if model finishes before t+1 arrives
            -> t starts immediately

        otherwise t+1 arrives first
            -> buffered t is discarded
            -> buffer becomes t+1

    Therefore, when an inference finishes, the next frame to execute is
    always the newest sensor frame that has already arrived.
    """

    clear_history(model)

    n = len(indices)

    if n == 0:
        return [], 0, 0

    # Position in this scene's sensor-frame sequence.
    pos = 0

    # Virtual sensor/compute timeline.
    virtual_now_ms = 0.0

    processed_count = 0
    dropped_count = 0

    scene_outputs = []

    while pos < n:

        dataset_index = indices[pos]

        input_frame_id = frame_token(
            dataset,
            dataset_index,
        )

        # Sensor timestamp.
        arrival_ms = (
            pos * period_ms
        )

        # If model is idle, wait until sensor frame arrives.
        # If frame has already been waiting in buffer,
        # start immediately when GPU becomes free.
        start_ms = max(
            virtual_now_ms,
            arrival_ms,
        )

        (
            anno,
            service_ms,
            wall_finish_ns,
        ) = run_one_prediction(
            model,
            dataset,
            dataset_index,
        )

        # Actual prediction publication timestamp on virtual sensor clock.
        finish_ms = (
            start_ms
            +
            service_ms
        )

        stamped = attach_timestamp(
            anno=anno,
            scene=scene,
            input_frame_id=input_frame_id,
            arrival_ms=arrival_ms,
            start_ms=start_ms,
            finish_ms=finish_ms,
            service_ms=service_ms,
            wall_finish_ns=wall_finish_ns,
        )

        processed_predictions.append(
            stamped
        )

        scene_outputs.append(
            stamped
        )

        processed_count += 1

        # ------------------------------------------------------------
        # Buffer update.
        #
        # Find every frame that arrived while current inference was busy.
        #
        # Since buffer capacity is exactly 1, only the LAST of those frames
        # survives.
        # ------------------------------------------------------------

        latest_arrived_pos = pos

        probe = pos + 1

        while (
            probe < n
            and
            probe * period_ms <= finish_ms
        ):
            latest_arrived_pos = probe
            probe += 1

        if latest_arrived_pos > pos:

            # At least one later frame arrived while model was busy.
            #
            # The latest one survives in buffer.
            next_pos = latest_arrived_pos

            # Frames before the latest buffered frame were overwritten.
            dropped_now = max(
                0,
                next_pos - pos - 1,
            )

            next_reason = "buffer_latest"

        else:

            # No new frame arrived before model finished.
            # GPU will idle until the next sensor frame.
            next_pos = pos + 1

            dropped_now = 0

            next_reason = "wait_next_arrival"

        dropped_count += dropped_now

        trace.append(
            {
                "scene": scene,
                "dataset_index": int(
                    dataset_index
                ),
                "frame_id": input_frame_id,
                "scene_pos": int(pos),

                "arrival_ms": float(
                    arrival_ms
                ),
                "start_ms": float(
                    start_ms
                ),
                "finish_ms": float(
                    finish_ms
                ),
                "service_ms": float(
                    service_ms
                ),

                "buffer_wait_ms": float(
                    start_ms - arrival_ms
                ),

                "dropped_waiting_frames_after_this_output":
                    int(dropped_now),

                "next_scene_pos":
                    int(next_pos)
                    if next_pos < n
                    else None,

                "next_reason":
                    next_reason
                    if next_pos < n
                    else "scene_end",

                "wall_output_time_ns":
                    int(wall_finish_ns),
            }
        )

        print(
            f"[{scene}] "
            f"frame={input_frame_id} "
            f"pos={pos:03d}/{n - 1:03d} "
            f"arrival={arrival_ms:9.3f} "
            f"start={start_ms:9.3f} "
            f"finish={finish_ms:9.3f} "
            f"service={service_ms:7.3f} ms "
            f"drop+={dropped_now}"
        )

        virtual_now_ms = finish_ms

        pos = next_pos

    return (
        scene_outputs,
        processed_count,
        dropped_count,
    )


# ============================================================
# Timestamp -> sAP alignment
# ============================================================

def strip_private_fields(anno):
    """
    Remove runtime-only timestamp fields before KITTI evaluation.
    """

    return {
        key: value
        for key, value in anno.items()
        if (
            not key.startswith("_stream_")
            and
            not key.startswith("_wall_")
        )
    }


def timestamp_align_scene(
    dataset,
    scene,
    indices,
    scene_outputs,
    period_ms,
):
    """
    Streaming AP rule:

    At every annotation timestamp t:

        choose the newest prediction satisfying

            output_timestamp <= t

    No Kalman filter.
    No tracker.
    No empirical latency sampling.

    This is timestamp-based COPY streaming evaluation.
    """

    gt_annos = []
    det_annos = []

    output_ptr = 0

    latest_output = None

    for (
        target_pos,
        dataset_index
    ) in enumerate(indices):

        target_time_ms = (
            target_pos
            *
            period_ms
        )

        # Publish every prediction that has completed by this timestamp.
        # Keep only the most recent completed one.
        while (
            output_ptr
            <
            len(scene_outputs)
            and
            scene_outputs[
                output_ptr
            ][
                "_stream_output_ts_ms"
            ]
            <=
            target_time_ms
        ):

            latest_output = scene_outputs[
                output_ptr
            ]

            output_ptr += 1

        gt = copy.deepcopy(
            dataset.kitti_infos[
                dataset_index
            ][
                "infos"
            ][
                "token"
            ][
                "annos"
            ]
        )

        gt_annos.append(gt)

        target_frame_id = frame_token(
            dataset,
            dataset_index,
        )

        if latest_output is None:

            # Before first prediction completes there is no published result.
            det = make_empty_prediction()

            source_frame_id = None

            source_output_ts_ms = None

        else:

            det = copy.deepcopy(
                strip_private_fields(
                    latest_output
                )
            )

            source_frame_id = (
                latest_output[
                    "_stream_input_frame_id"
                ]
            )

            source_output_ts_ms = (
                latest_output[
                    "_stream_output_ts_ms"
                ]
            )

        # Metadata for inspecting alignment.
        # KITTI metric does not use these fields.
        det["scene"] = scene

        det["frame_id"] = (
            target_frame_id
        )

        det["sample_token"] = (
            f"{scene}_{target_frame_id}"
        )

        det["_sap_target_ts_ms"] = float(
            target_time_ms
        )

        det["_sap_source_frame_id"] = (
            source_frame_id
        )

        det["_sap_source_output_ts_ms"] = (
            source_output_ts_ms
        )

        det_annos.append(det)

    return (
        gt_annos,
        det_annos,
    )


def strip_eval_metadata(anno):
    return {
        key: value
        for key, value in anno.items()
        if not key.startswith("_sap_")
    }


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    set_seed(args.seed)

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    # ============================================================
    # Evaluation compatibility:
    #
    # The original StreamDSGN config uses:
    #
    #   feature_backbone_pretrained: torchvision://resnet18
    #
    # Old MMCV resolves this through torchvision's legacy model_urls.
    # Modern torchvision removed that mapping, which causes:
    #
    #   KeyError: 'resnet18'
    #
    # For evaluation this ImageNet initialization is unnecessary:
    # immediately after model construction we load the complete trained
    # StreamDSGN checkpoint. Therefore disable only the construction-time
    # ImageNet initialization while preserving the original architecture.
    # ============================================================

    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None

    cfg.TAG = Path(
        args.cfg_file
    ).stem

    # We do not use Empirical.draw().
    #
    # Every processed frame is timed directly.
    cfg.DATA_CONFIG.INFER_TIME_PATH = None

    logger = common_utils.create_logger()

    logger.info(
        f"cfg_file: {args.cfg_file}"
    )

    logger.info(
        f"ckpt: {args.ckpt}"
    )

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Empty test dataset"
        )

    # ------------------------------------------------------------
    # Original StreamDSGN
    # ------------------------------------------------------------

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

    model.cuda()

    model.eval()

    # ------------------------------------------------------------
    # Sensor timing
    # ------------------------------------------------------------

    if args.input_hz is not None:

        input_hz = float(
            args.input_hz
        )

    else:

        input_hz = float(
            cfg.DATA_CONFIG.get(
                "ANNOS_FREQUENCY",
                10,
            )
        )

    if input_hz <= 0:
        raise ValueError(
            f"input_hz must be positive: "
            f"{input_hz}"
        )

    period_ms = (
        1000.0
        /
        input_hz
    )

    scene_to_indices = build_scene_index(
        dataset
    )

    print()
    print("=" * 80)
    print(
        "Original StreamDSGN "
        "timestamped single-slot-buffer streaming test"
    )
    print("=" * 80)

    print(
        f"input_hz       : "
        f"{input_hz:.3f} Hz"
    )

    print(
        f"frame_period   : "
        f"{period_ms:.3f} ms"
    )

    print(
        f"scenes         : "
        f"{len(scene_to_indices)}"
    )

    print(
        f"sensor_frames  : "
        f"{len(dataset)}"
    )

    print(
        "timing scope   : "
        "model forward + post-processing; "
        "H2D excluded"
    )

    print(
        "buffer         : "
        "capacity=1; newest waiting frame "
        "replaces older waiting frame"
    )

    print(
        "publish rule   : "
        "latest prediction with "
        "output_ts <= annotation_ts"
    )

    print(
        "stream method  : "
        "timestamp COPY only"
    )

    print(
        "metrics        : "
        "paper Table-1 AP_R40 BEV/3D only"
    )

    print(
        "KF / tracker   : disabled"
    )

    print("=" * 80)
    print()

    # ------------------------------------------------------------
    # Warmup
    # ------------------------------------------------------------

    warmup_model(
        model,
        dataset,
        scene_to_indices,
        args.warmup,
    )

    # ------------------------------------------------------------
    # Streaming inference
    # ------------------------------------------------------------

    trace = []

    processed_predictions = []

    outputs_by_scene = {}

    total_processed = 0

    total_dropped = 0

    for (
        scene,
        indices,
    ) in scene_to_indices.items():

        print()

        print(
            f"===== scene {scene}: "
            f"{len(indices)} sensor frames ====="
        )

        (
            scene_outputs,
            processed_count,
            dropped_count,
        ) = run_scene(
            model=model,
            dataset=dataset,
            scene=scene,
            indices=indices,
            period_ms=period_ms,
            trace=trace,
            processed_predictions=
                processed_predictions,
        )

        outputs_by_scene[
            scene
        ] = scene_outputs

        total_processed += (
            processed_count
        )

        total_dropped += (
            dropped_count
        )

    # ------------------------------------------------------------
    # Timestamp-based streaming alignment
    # ------------------------------------------------------------

    all_gt_annos = []

    all_stream_det_annos = []

    for (
        scene,
        indices,
    ) in scene_to_indices.items():

        gt, det = timestamp_align_scene(
            dataset=dataset,
            scene=scene,
            indices=indices,
            scene_outputs=
                outputs_by_scene[scene],
            period_ms=period_ms,
        )

        all_gt_annos.extend(gt)

        all_stream_det_annos.extend(det)

    # ------------------------------------------------------------
    # KITTI AP evaluation on timestamp-aligned predictions
    # ------------------------------------------------------------

    eval_det_annos = [
        strip_eval_metadata(
            copy.deepcopy(x)
        )
        for x in all_stream_det_annos
    ]

    # IMPORTANT:
    #
    # evaluation_offline() here is ONLY the KITTI AP calculator.
    #
    # The predictions themselves have already been transformed into
    # timestamp-aligned streaming predictions above.
    #
    # Therefore the output is streaming AP, not offline AP.
    full_result_str, _ = (
        dataset.evaluation_offline(
            all_gt_annos,
            eval_det_annos,
            dataset.class_names,
            "3d",
        )
    )

    # Keep only the AP_R40 BEV/3D entries reported in StreamDSGN Table 1.
    paper_result_str = (
        eval_utils.format_paper_metrics(
            full_result_str
        )
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    output_dir = (
        Path(args.output_dir)
        /
        cfg.TAG
        /
        f"{input_hz:g}Hz"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_dir
        /
        "timeline.json",
        "w",
    ) as f:

        json.dump(
            trace,
            f,
            indent=2,
        )

    with open(
        output_dir
        /
        "processed_predictions_timestamped.pkl",
        "wb",
    ) as f:

        pickle.dump(
            processed_predictions,
            f,
        )

    with open(
        output_dir
        /
        "sap_timestamp_aligned_predictions.pkl",
        "wb",
    ) as f:

        pickle.dump(
            all_stream_det_annos,
            f,
        )

    with open(
        output_dir
        /
        "paper_sap.txt",
        "w",
    ) as f:

        f.write(
            paper_result_str
        )

        f.write("\n")

    # ------------------------------------------------------------
    # Runtime summary
    # ------------------------------------------------------------

    service = np.asarray(
        [
            x["service_ms"]
            for x in trace
        ],
        dtype=np.float64,
    )

    waits = np.asarray(
        [
            x["buffer_wait_ms"]
            for x in trace
        ],
        dtype=np.float64,
    )

    total_sensor_frames = sum(
        len(x)
        for x
        in scene_to_indices.values()
    )

    process_rate = (
        total_processed
        /
        max(
            total_sensor_frames,
            1,
        )
    )

    drop_rate = (
        total_dropped
        /
        max(
            total_sensor_frames,
            1,
        )
    )

    summary = {
        "cfg_file":
            args.cfg_file,

        "ckpt":
            args.ckpt,

        "input_hz":
            input_hz,

        "period_ms":
            period_ms,

        "total_sensor_frames":
            total_sensor_frames,

        "processed_frames":
            total_processed,

        "dropped_buffer_frames":
            total_dropped,

        "process_rate":
            process_rate,

        "drop_rate":
            drop_rate,

        "mean_service_ms":
            float(service.mean())
            if service.size
            else None,

        "p50_service_ms":
            float(
                np.percentile(
                    service,
                    50,
                )
            )
            if service.size
            else None,

        "p90_service_ms":
            float(
                np.percentile(
                    service,
                    90,
                )
            )
            if service.size
            else None,

        "p99_service_ms":
            float(
                np.percentile(
                    service,
                    99,
                )
            )
            if service.size
            else None,

        "mean_buffer_wait_ms":
            float(waits.mean())
            if waits.size
            else None,
    }

    with open(
        output_dir
        /
        "summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Print final result
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("STREAM SUMMARY")
    print("=" * 80)

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print()
    print("=" * 80)
    print(
        "TIMESTAMP-ALIGNED STREAMING AP "
        "— PAPER METRICS ONLY"
    )
    print("=" * 80)

    print(
        paper_result_str
    )

    print()

    print(
        f"Saved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()
