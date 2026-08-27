#!/usr/bin/env python3

import argparse
import copy
import csv
import json
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

try:
    from eval_utils.eval_utils import format_paper_metrics
except Exception:
    format_paper_metrics = None


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="True-stream StreamDSGN baseline evaluator"
    )

    parser.add_argument(
        "--mode",
        choices=["profile", "eval"],
        required=True,
    )

    parser.add_argument(
        "--cfg_file",
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--latency_trace",
        required=True,
        help="forward-only latency JSON",
    )

    parser.add_argument(
        "--speedups",
        default="1,2,3,4",
        help="stream-rate multipliers: 1x=10Hz, 2x=20Hz, 3x=30Hz, 4x=40Hz",
    )

    parser.add_argument(
        "--warmup_frames",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="fallback stream FPS; default DATA_CONFIG.ANNOS_FREQUENCY",
    )

    parser.add_argument(
        "--timestamp_json",
        default=None,
        help="optional real sensor timestamps",
    )

    parser.add_argument(
        "--timestamp_unit",
        choices=["ms", "s"],
        default="ms",
    )

    parser.add_argument(
        "--require_timestamp_json",
        action="store_true",
    )

    parser.add_argument(
        "--out_dir",
        default="outputs/true_stream_baseline",
    )

    parser.add_argument(
        "--paper_metrics_only",
        action="store_true",
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="debug only; -1 = all",
    )

    return parser.parse_args()


# ============================================================
# Utility
# ============================================================

def first_scalar(x):
    if isinstance(x, (list, tuple)):
        return first_scalar(x[0])

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return first_scalar(x[0])

    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        return first_scalar(x[0])

    return x


def get_scene_frame(batch_dict):
    token = batch_dict["token"]

    scene = str(
        first_scalar(token["scene"])
    )

    frame = str(
        first_scalar(token["this_sample_idx"])
    )

    return scene, frame


def frame_key(scene, frame):
    return f"{scene}/{frame}"


def reset_stream_state(model):
    """
    Clear all temporal state.

    Especially important at scene boundaries.
    """

    if (
        hasattr(model, "history_feature_queue")
        and model.history_feature_queue is not None
    ):
        model.history_feature_queue.clear()

    # Make sure the baseline is not accidentally using adaptive mode.
    for module in model.modules():
        for name in (
            "clear_adaptive_training_state",
            "clear_adaptive_a_profile",
            "clear_adaptive_profile",
        ):
            fn = getattr(module, name, None)

            if callable(fn):
                try:
                    fn()
                except TypeError:
                    pass


def history_ids(model):
    q = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if q is None:
        return []

    result = []

    for item in list(q):
        try:
            result.append(
                str(first_scalar(item[0]))
            )
        except Exception:
            result.append("?")

    return result


# ============================================================
# Build dataset + model
# ============================================================

def build_env(args):
    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    # Disable repository post-hoc streaming simulator.
    # We perform true streaming ourselves.
    cfg.DATA_CONFIG.INFER_TIME_PATH = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = common_utils.create_logger(
        out_dir / f"log_{args.mode}.txt",
        rank=0,
    )

    dataset, loader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    model.load_params_from_file(
        args.ckpt,
        logger=logger,
        to_cpu=True,
    )

    model.cuda()
    model.eval()

    return (
        dataset,
        loader,
        model,
        logger,
        out_dir,
    )


# ============================================================
# Warmup
# ============================================================

def warmup(loader, model, num_frames):
    if num_frames <= 0:
        return

    reset_stream_state(model)

    count = 0

    with torch.no_grad():

        for batch in loader:

            # NOT timed
            load_data_to_gpu(batch)

            torch.cuda.synchronize()

            model(batch)

            torch.cuda.synchronize()

            count += 1

            if count >= num_frames:
                break

    reset_stream_state(model)


# ============================================================
# Forward-only latency profiling
# ============================================================

def profile_latency(
    args,
    dataset,
    loader,
    model,
    logger,
    out_dir,
):

    warmup(
        loader,
        model,
        args.warmup_frames,
    )

    reset_stream_state(model)

    rows = []

    last_scene = None

    with torch.no_grad():

        for i, batch in enumerate(loader):

            if (
                args.max_frames > 0
                and i >= args.max_frames
            ):
                break

            scene, frame = get_scene_frame(
                batch
            )

            if scene != last_scene:
                reset_stream_state(model)
                last_scene = scene

            # ==================================================
            # IMPORTANT:
            #
            # data preparation / CPU->GPU transfer is excluded
            # from latency.
            # ==================================================

            load_data_to_gpu(batch)

            # ==================================================
            # Only model forward is timed.
            # ==================================================

            torch.cuda.synchronize()

            t0 = time.perf_counter()

            pred_dicts, ret_dict = model(batch)

            torch.cuda.synchronize()

            t1 = time.perf_counter()

            forward_ms = (
                t1 - t0
            ) * 1000.0

            rows.append(
                {
                    "scene": scene,
                    "frame_id": frame,
                    "forward_ms": forward_ms,
                }
            )

            if (i + 1) % 100 == 0:

                logger.info(
                    "[PROFILE] %d/%d "
                    "frame=%s "
                    "forward=%.3f ms",
                    i + 1,
                    len(loader),
                    frame_key(scene, frame),
                    forward_ms,
                )

    if not rows:
        raise RuntimeError(
            "No frames profiled"
        )

    forward_values = [
        x["forward_ms"]
        for x in rows
    ]

    payload = {

        "cfg_file": args.cfg_file,

        "ckpt": args.ckpt,

        "latency_semantics":
            "forward_only",

        "latency_scope": {
            "forward_ms":
                "model(batch) with CUDA sync before/after; "
                "data loading/preprocessing/H2D excluded"
        },

        "num_frames": len(rows),

        "stats": {

            "forward_mean_ms":
                float(np.mean(
                    forward_values
                )),

            "forward_p50_ms":
                float(np.percentile(
                    forward_values,
                    50,
                )),

            "forward_p90_ms":
                float(np.percentile(
                    forward_values,
                    90,
                )),

            "forward_p99_ms":
                float(np.percentile(
                    forward_values,
                    99,
                )),

            "forward_min_ms":
                float(np.min(
                    forward_values
                )),

            "forward_max_ms":
                float(np.max(
                    forward_values
                )),
        },

        "entries": rows,
    }

    path = Path(
        args.latency_trace
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )

    logger.info(
        "Latency trace saved: %s",
        path,
    )

    logger.info(
        "Forward latency stats: %s",
        payload["stats"],
    )


# ============================================================
# Timestamp
# ============================================================

def parse_timestamp_value(
    value,
    unit,
):

    if isinstance(
        value,
        (int, float),
    ):

        value = float(value)

        if unit == "s":
            return value * 1000.0

        return value

    value = str(value).strip()

    try:
        value = float(value)

        if unit == "s":
            return value * 1000.0

        return value

    except ValueError:
        pass

    # ISO datetime
    value = value.replace(
        "Z",
        "+00:00",
    )

    dt = datetime.fromisoformat(
        value
    )

    return (
        dt.timestamp()
        * 1000.0
    )


def load_timestamps(args):

    if args.timestamp_json is None:

        if args.require_timestamp_json:

            raise RuntimeError(
                "--require_timestamp_json "
                "but no --timestamp_json"
            )

        return None

    raw = json.loads(
        Path(
            args.timestamp_json
        ).read_text()
    )

    output = {}

    # Flat:
    #
    # {
    #   "0000/000001": 100.0
    # }

    if all(
        not isinstance(v, dict)
        for v in raw.values()
    ):

        for key, value in raw.items():

            output[str(key)] = (
                parse_timestamp_value(
                    value,
                    args.timestamp_unit,
                )
            )

        return output

    # Nested:
    #
    # {
    #   "0000": {
    #       "000001": 100.0
    #   }
    # }

    for scene, frames in raw.items():

        for frame, value in frames.items():

            output[
                frame_key(
                    str(scene),
                    str(frame),
                )
            ] = parse_timestamp_value(
                value,
                args.timestamp_unit,
            )

    return output


class StreamClock:

    def __init__(
        self,
        timestamp_map,
        fps,
    ):

        self.timestamp_map = (
            timestamp_map
        )

        self.fps = float(fps)

        self.scene_first_frame = {}

        self.scene_first_timestamp = {}


    def arrival_ms(
        self,
        scene,
        frame,
    ):

        key = frame_key(
            scene,
            frame,
        )

        # Physical timestamps
        if self.timestamp_map is not None:

            if key not in self.timestamp_map:

                raise KeyError(
                    f"Missing timestamp: {key}"
                )

            timestamp = float(
                self.timestamp_map[key]
            )

            if (
                scene
                not in
                self.scene_first_timestamp
            ):

                self.scene_first_timestamp[
                    scene
                ] = timestamp

            return (
                timestamp
                -
                self.scene_first_timestamp[
                    scene
                ]
            )

        # Fixed-rate KITTI clock
        frame_int = int(frame)

        if (
            scene
            not in
            self.scene_first_frame
        ):

            self.scene_first_frame[
                scene
            ] = frame_int

        return (
            frame_int
            -
            self.scene_first_frame[
                scene
            ]
        ) * (
            1000.0
            /
            self.fps
        )


# ============================================================
# Prediction helpers
# ============================================================

def empty_prediction(
    scene,
    frame,
    next_frame="",
):

    return {

        "name":
            np.zeros(
                (0,),
                dtype="<U1",
            ),

        "truncated":
            np.zeros(
                (0,),
                dtype=np.float64,
            ),

        "occluded":
            np.zeros(
                (0,),
                dtype=np.float64,
            ),

        "alpha":
            np.zeros(
                (0,),
                dtype=np.float64,
            ),

        "bbox":
            np.zeros(
                (0, 4),
                dtype=np.float64,
            ),

        "dimensions":
            np.zeros(
                (0, 3),
                dtype=np.float64,
            ),

        "location":
            np.zeros(
                (0, 3),
                dtype=np.float64,
            ),

        "rotation_y":
            np.zeros(
                (0,),
                dtype=np.float64,
            ),

        "score":
            np.zeros(
                (0,),
                dtype=np.float64,
            ),

        "boxes_lidar":
            np.zeros(
                (0, 7),
                dtype=np.float64,
            ),

        "scene": scene,

        "frame_id": frame,

        "next_frame_id": next_frame,
    }


def retag_copy_prediction(
    pred,
    scene,
    frame,
    next_frame="",
):

    if pred is None:

        return empty_prediction(
            scene,
            frame,
            next_frame,
        )

    output = copy.deepcopy(
        pred
    )

    output["scene"] = scene

    output["frame_id"] = frame

    output["next_frame_id"] = (
        next_frame
    )

    return output


# ============================================================
# Latency trace
# ============================================================

def load_latency_trace(path):

    payload = json.loads(
        Path(path).read_text()
    )

    table = {}

    for item in payload["entries"]:

        key = frame_key(
            str(item["scene"]),
            str(item["frame_id"]),
        )

        table[key] = item

    return payload, table


def effective_service_ms(
    latency,
    rate_multiplier,
):
    """
    Stream-rate stress test.

    IMPORTANT:

        T_service = T_forward

    The model itself is NOT artificially accelerated.

    rate_multiplier only changes frame arrival rate:

        1x = 10 Hz
        2x = 20 Hz
        3x = 30 Hz
        4x = 40 Hz
    """

    if "forward_ms" in latency:

        forward_ms = float(
            latency["forward_ms"]
        )

    elif "model_ms" in latency:

        forward_ms = float(
            latency["model_ms"]
        )

    else:

        raise KeyError(
            "Latency entry does not "
            "contain forward_ms"
        )

    return forward_ms


def to_jsonable(x):

    if isinstance(x, dict):

        return {
            str(k): to_jsonable(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):

        return [
            to_jsonable(v)
            for v in x
        ]

    if isinstance(x, np.ndarray):

        return x.tolist()

    if isinstance(
        x,
        (np.floating, np.integer),
    ):

        return x.item()

    if torch.is_tensor(x):

        if x.numel() == 1:
            return x.item()

        return (
            x.detach()
            .cpu()
            .tolist()
        )

    return x


# ============================================================
# True streaming evaluation
# ============================================================

def evaluate_speedup(
    args,
    dataset,
    loader,
    model,
    logger,
    out_dir,
    latency_table,
    speedup,
    clock,
):

    reset_stream_state(model)

    stream_annos = []

    trace_rows = []

    current_scene = None

    # Simulated accelerator state
    busy_until_ms = -math.inf

    # Completed jobs not yet visible at a sensor timestamp
    pending = deque()

    latest_completed = None

    processed = 0
    dropped = 0

    stale_frames = 0
    no_output_frames = 0

    ages_ms = []

    with torch.no_grad():

        for i, batch in enumerate(loader):

            if (
                args.max_frames > 0
                and i >= args.max_frames
            ):
                break

            scene, frame = (
                get_scene_frame(batch)
            )

            token = batch["token"]

            next_frame = ""

            if "next_sample_idx" in token:

                next_frame = str(
                    first_scalar(
                        token[
                            "next_sample_idx"
                        ]
                    )
                )

            # =================================================
            # New scene
            # =================================================

            if scene != current_scene:

                current_scene = scene

                reset_stream_state(
                    model
                )

                busy_until_ms = (
                    -math.inf
                )

                pending.clear()

                latest_completed = None

            # =================================================
            # Real sensor/frame arrival time
            # =================================================

            arrival_ms = (
                clock.arrival_ms(
                    scene,
                    frame,
                )
            )

            # =================================================
            # Publish all inference jobs that have completed
            # before this timestamp.
            # =================================================

            while (
                pending
                and
                pending[0][
                    "completion_ms"
                ]
                <=
                arrival_ms + 1e-9
            ):

                latest_completed = (
                    pending.popleft()
                )

            # =================================================
            # What prediction is available NOW?
            # =================================================

            if latest_completed is None:

                eval_pred = (
                    empty_prediction(
                        scene,
                        frame,
                        next_frame,
                    )
                )

                source_frame = ""

                source_arrival_ms = (
                    math.nan
                )

                source_completion_ms = (
                    math.nan
                )

                prediction_age_ms = (
                    math.nan
                )

                no_output_frames += 1

            else:

                eval_pred = (
                    retag_copy_prediction(
                        latest_completed[
                            "anno"
                        ],
                        scene,
                        frame,
                        next_frame,
                    )
                )

                source_frame = (
                    latest_completed[
                        "source_frame"
                    ]
                )

                source_arrival_ms = (
                    latest_completed[
                        "source_arrival_ms"
                    ]
                )

                source_completion_ms = (
                    latest_completed[
                        "completion_ms"
                    ]
                )

                prediction_age_ms = (
                    arrival_ms
                    -
                    source_arrival_ms
                )

                ages_ms.append(
                    prediction_age_ms
                )

                if source_frame != frame:
                    stale_frames += 1

            # One prediction per sensor timestamp
            stream_annos.append(
                eval_pred
            )

            # =================================================
            # Frame admission:
            #
            # no queue / no buffering.
            #
            # If accelerator is still busy when this frame
            # arrives, this frame is physically dropped.
            # =================================================

            accepted = (
                arrival_ms
                >=
                busy_until_ms - 1e-9
            )

            hist_before = (
                history_ids(model)
            )

            forward_wall_ms = math.nan
            service_ms = math.nan
            completion_ms = math.nan

            if accepted:

                key = frame_key(
                    scene,
                    frame,
                )

                if key not in latency_table:

                    raise KeyError(
                        "No latency trace "
                        f"for {key}"
                    )

                # =============================================
                # IMPORTANT:
                #
                # The accepted frame is really executed.
                #
                # Therefore StreamDSGN history_feature_queue
                # changes exactly as in real inference.
                #
                # load_data_to_gpu is NOT part of simulated
                # time/deadline.
                # =============================================

                load_data_to_gpu(batch)

                torch.cuda.synchronize()

                wall_t0 = (
                    time.perf_counter()
                )

                pred_dicts, ret_dict = (
                    model(batch)
                )

                torch.cuda.synchronize()

                wall_t1 = (
                    time.perf_counter()
                )

                forward_wall_ms = (
                    wall_t1 - wall_t0
                ) * 1000.0

                annos = (
                    dataset.generate_prediction_dicts(
                        batch,
                        pred_dicts,
                        dataset.class_names,
                    )
                )

                assert len(annos) == 1

                source_anno = annos[0]

                # =============================================
                # Simulated accelerated forward time
                # =============================================

                service_ms = (
                    effective_service_ms(
                        latency_table[key],
                        speedup,
                    )
                )

                completion_ms = (
                    arrival_ms
                    +
                    service_ms
                )

                busy_until_ms = (
                    completion_ms
                )

                pending.append(
                    {
                        "completion_ms":
                            completion_ms,

                        "source_arrival_ms":
                            arrival_ms,

                        "source_frame":
                            frame,

                        "anno":
                            source_anno,
                    }
                )

                processed += 1

            else:

                # =============================================
                # DROP:
                #
                # absolutely NO model(batch)
                #
                # therefore:
                #   - no current BEV
                #   - no history update
                # =============================================

                dropped += 1

            hist_after = (
                history_ids(model)
            )

            trace_rows.append(
                {
                    "scene":
                        scene,

                    "frame_id":
                        frame,

                    "arrival_ms":
                        arrival_ms,

                    "accepted":
                        int(accepted),

                    "dropped":
                        int(
                            not accepted
                        ),

                    "busy_until_ms":
                        busy_until_ms,

                    "effective_service_ms":
                        service_ms,

                    "physical_wall_forward_ms":
                        forward_wall_ms,

                    "history_before":
                        "|".join(
                            hist_before
                        ),

                    "history_after":
                        "|".join(
                            hist_after
                        ),

                    "sap_source_frame":
                        source_frame,

                    "sap_source_arrival_ms":
                        source_arrival_ms,

                    "sap_source_completion_ms":
                        source_completion_ms,

                    "sap_prediction_age_ms":
                        prediction_age_ms,
                }
            )

            if (i + 1) % 200 == 0:

                total = (
                    processed
                    +
                    dropped
                )

                logger.info(
                    "[RATE=%.2fx] "
                    "%d/%d "
                    "processed=%d "
                    "dropped=%d "
                    "drop=%.2f%% "
                    "arrival=%.1f "
                    "busy_until=%.1f "
                    "history=%s "
                    "sap_src=%s",
                    speedup,
                    i + 1,
                    len(loader),
                    processed,
                    dropped,
                    100.0
                    * dropped
                    / max(total, 1),
                    arrival_ms,
                    busy_until_ms,
                    hist_before,
                    source_frame,
                )

    num_frames = (
        processed
        +
        dropped
    )

    if (
        len(stream_annos)
        !=
        num_frames
    ):

        raise RuntimeError(
            "stream annotation "
            "count mismatch"
        )

    # ========================================================
    # Streaming simulation has ALREADY been done.
    #
    # offline_3d here is only used as the KITTI AP calculator.
    # ========================================================

    result_str_dict, result_dict = (
        dataset.evaluation(
            stream_annos,
            dataset.class_names,
            eval_metric=[
                "offline_3d"
            ],
        )
    )

    result_text = (
        result_str_dict[
            "offline_3d"
        ]
    )

    if (
        args.paper_metrics_only
        and
        format_paper_metrics
        is not None
    ):

        output_text = (
            format_paper_metrics(
                result_text
            )
        )

    else:

        output_text = result_text

    tag = str(
        speedup
    ).replace(
        ".",
        "p",
    )

    # ========================================================
    # Save AP / sAP text
    # ========================================================

    result_path = (
        out_dir
        /
        f"stream_baseline_{tag}x.txt"
    )

    result_path.write_text(
        output_text
    )

    # ========================================================
    # Save detailed per-frame streaming trace
    # ========================================================

    trace_path = (
        out_dir
        /
        f"stream_trace_{tag}x.csv"
    )

    if trace_rows:

        with trace_path.open(
            "w",
            newline="",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=list(
                    trace_rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                trace_rows
            )

    processed_ratio = (
        processed
        /
        max(num_frames, 1)
    )

    dropped_ratio = (
        dropped
        /
        max(num_frames, 1)
    )

    if ages_ms:

        age_stats = {

            "mean":
                float(np.mean(
                    ages_ms
                )),

            "p50":
                float(np.percentile(
                    ages_ms,
                    50,
                )),

            "p90":
                float(np.percentile(
                    ages_ms,
                    90,
                )),

            "p99":
                float(np.percentile(
                    ages_ms,
                    99,
                )),
        }

    else:

        age_stats = {

            "mean": None,
            "p50": None,
            "p90": None,
            "p99": None,
        }

    summary = {

        "speedup":
            speedup,

        "num_frames":
            num_frames,

        "processed_frames":
            processed,

        "dropped_frames":
            dropped,

        "processed_ratio":
            processed_ratio,

        "drop_ratio":
            dropped_ratio,

        "frames_without_any_completed_output":
            no_output_frames,

        "stale_output_frames":
            stale_frames,

        "prediction_age_ms":
            age_stats,

        "latency_semantics":
            "forward_only",

        "service_time_formula":
            "forward_ms",

        "rate_semantics":
            "input stream replay rate multiplier",

        "metric_name":
            "true_stream_copy_3d",

        "metric_impl":
            "timestamp-aligned predictions "
            "then KITTI offline_3d AP calculator",

        "result_dict":
            to_jsonable(
                result_dict
            ),

        "result_text_file":
            str(result_path),

        "frame_trace_csv":
            str(trace_path),
    }

    summary_path = (
        out_dir
        /
        f"summary_{tag}x.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    logger.info(
        "=================================================="
    )

    logger.info(
        "TRUE STREAM BASELINE RATE %.2fx",
        speedup,
    )

    logger.info(
        "processed=%d/%d (%.2f%%) "
        "dropped=%d (%.2f%%)",
        processed,
        num_frames,
        processed_ratio * 100.0,
        dropped,
        dropped_ratio * 100.0,
    )

    logger.info(
        "prediction age ms: %s",
        age_stats,
    )

    logger.info(
        "\n%s",
        output_text,
    )

    logger.info(
        "summary: %s",
        summary_path,
    )

    logger.info(
        "trace: %s",
        trace_path,
    )

    return summary


# ============================================================
# Eval multiple speedups
# ============================================================

def run_eval(
    args,
    dataset,
    loader,
    model,
    logger,
    out_dir,
):

    latency_payload, latency_table = (
        load_latency_trace(
            args.latency_trace
        )
    )

    fps = args.fps

    if fps is None:

        fps = float(
            cfg.DATA_CONFIG.get(
                "ANNOS_FREQUENCY",
                10,
            )
        )

    timestamp_map = (
        load_timestamps(args)
    )

    if timestamp_map is None:

        clock_source = (
            f"fixed_rate:{fps}Hz"
        )

        logger.info(
            "Using fixed-rate sensor clock: %.3f ms/frame",
            1000.0 / fps,
        )

    else:

        clock_source = (
            "timestamp_json:"
            f"{args.timestamp_json}"
        )

        logger.info(
            "Using physical timestamp JSON: %s",
            args.timestamp_json,
        )

    speedups = [
        float(x)
        for x
        in args.speedups.split(",")
        if x.strip()
    ]

    warmup(
        loader,
        model,
        args.warmup_frames,
    )

    summaries = []

    for speedup in speedups:

        if speedup <= 0:

            raise ValueError(
                f"Invalid speedup: {speedup}"
            )

        effective_fps = (
            fps
            *
            speedup
        )

        # Same KITTI frame sequence is replayed speedup-times faster.
        #
        # 1x -> 10 Hz
        # 2x -> 20 Hz
        # 3x -> 30 Hz
        # 4x -> 40 Hz
        #
        # Therefore:
        #
        # deadline = 1000 / effective_fps
        #
        if timestamp_map is None:

            clock = StreamClock(
                None,
                effective_fps,
            )

        else:

            # For physical timestamps, replay the same timestamp sequence
            # speedup-times faster by compressing relative time.
            scaled_timestamp_map = {}

            scene_first = {}

            for key, raw_t in timestamp_map.items():

                scene = key.split("/", 1)[0]

                if scene not in scene_first:
                    scene_first[scene] = float(raw_t)

                scaled_timestamp_map[key] = (
                    scene_first[scene]
                    +
                    (
                        float(raw_t)
                        -
                        scene_first[scene]
                    )
                    /
                    speedup
                )

            clock = StreamClock(
                scaled_timestamp_map,
                effective_fps,
            )

        logger.info(
            "[RATE] %.2fx -> %.3f Hz, frame interval/deadline = %.3f ms",
            speedup,
            effective_fps,
            1000.0 / effective_fps,
        )

        summary = (
            evaluate_speedup(
                args,
                dataset,
                loader,
                model,
                logger,
                out_dir,
                latency_table,
                speedup,
                clock,
            )
        )

        summary[
            "clock_source"
        ] = clock_source

        summary[
            "base_fps"
        ] = fps

        summary[
            "effective_fps"
        ] = effective_fps

        summary[
            "frame_interval_ms"
        ] = (
            1000.0
            /
            effective_fps
        )

        summaries.append(
            summary
        )

    (
        out_dir
        /
        "summary_all_speedups.json"
    ).write_text(
        json.dumps(
            to_jsonable(
                summaries
            ),
            indent=2,
        )
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    (
        dataset,
        loader,
        model,
        logger,
        out_dir,
    ) = build_env(args)

    if args.mode == "profile":

        profile_latency(
            args,
            dataset,
            loader,
            model,
            logger,
            out_dir,
        )

    else:

        run_eval(
            args,
            dataset,
            loader,
            model,
            logger,
            out_dir,
        )


if __name__ == "__main__":
    main()
