#!/usr/bin/env python3

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from eval_utils import eval_utils

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

# Reuse utilities already verified in the previous experiments.
from test_counterfactual_state_value import (
    make_cfg,
    clone_history_queue,
    install_history_queue,
    parse_paper_metrics,
    mean_3d_moderate,
    set_seed,
)

from test_stream_buffer_timestamp import (
    attach_timestamp,
    build_scene_index,
    frame_token,
    load_one,
    strip_eval_metadata,
    timestamp_align_scene,
    warmup_model,
)


torch.backends.cudnn.benchmark = True


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Real capacity-1 streaming evaluation for a fixed "
            "repeating Full/Light policy."
        )
    )

    parser.add_argument(
        "--full_cfg",
        required=True,
    )

    parser.add_argument(
        "--full_ckpt",
        required=True,
    )

    parser.add_argument(
        "--light_cfg",
        required=True,
    )

    parser.add_argument(
        "--light_ckpt",
        required=True,
    )

    parser.add_argument(
        "--policy",
        required=True,
        type=str,
        help=(
            "Repeating action pattern over PROCESSED frames. "
            "Examples: F, L, FL, FLL, FLLL."
        ),
    )

    parser.add_argument(
        "--input_hz",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    return parser.parse_args()


# ============================================================
# Policy
# ============================================================

def normalize_policy(policy):
    p = policy.upper().strip()

    if not p:
        raise ValueError(
            "Policy cannot be empty"
        )

    for c in p:
        if c not in ("F", "L"):
            raise ValueError(
                f"Invalid policy '{policy}'. "
                "Only F and L are allowed."
            )

    return p


# ============================================================
# One mixed-mode prediction
# ============================================================

def run_one_mixed_prediction(
    model,
    shared_queue,
    dataset,
    dataset_index,
):
    """
    Execute one selected model using the explicit shared history.

    Timing scope intentionally matches the existing
    test_stream_buffer_timestamp.py:

        INCLUDED:
            model forward
            temporal fusion
            detection head
            post-processing / NMS

        EXCLUDED:
            dataset loading
            preprocessing
            H2D
            queue copying / policy selection

    After inference, the selected model's updated queue becomes
    the global shared queue.

    Dropped frames are never passed here, so they never enter history.
    """

    install_history_queue(
        model,
        shared_queue,
    )

    batch_dict = load_one(
        dataset,
        dataset_index,
    )

    torch.cuda.synchronize()

    start_ns = time.perf_counter_ns()

    with torch.no_grad():
        pred_dicts, _ = model(
            batch_dict
        )

    torch.cuda.synchronize()

    finish_ns = time.perf_counter_ns()

    service_ms = (
        finish_ns - start_ns
    ) / 1e6

    wall_finish_ns = time.time_ns()

    annos = dataset.generate_prediction_dicts(
        batch_dict,
        pred_dicts,
        dataset.class_names,
        output_path=None,
    )

    if len(annos) != 1:
        raise RuntimeError(
            "Mixed streaming evaluation requires batch_size=1, "
            f"but got {len(annos)}"
        )

    updated_queue = clone_history_queue(
        model.history_feature_queue,
        maxlen=model.history_feature_queue.maxlen,
    )

    return (
        annos[0],
        service_ms,
        wall_finish_ns,
        updated_queue,
    )


# ============================================================
# One scene
# ============================================================

def run_scene_mixed(
    full_model,
    light_model,
    dataset,
    scene,
    indices,
    period_ms,
    policy,
    trace,
):
    """
    Capacity-1 latest-frame buffer.

    Policy index advances ONLY when an inference is actually executed.

    Example:
        policy = FLLL

    means:

        processed inference #0 -> F
        processed inference #1 -> L
        processed inference #2 -> L
        processed inference #3 -> L
        processed inference #4 -> F
        ...

    Dropped sensor frames do NOT consume policy actions.
    """

    n = len(indices)

    if n == 0:
        return [], 0, 0

    history_len = (
        light_model
        .history_feature_queue
        .maxlen
    )

    shared_queue = deque(
        maxlen=history_len
    )

    # Clear model-local leftovers.
    full_model.history_feature_queue.clear()
    light_model.history_feature_queue.clear()

    pos = 0

    virtual_now_ms = 0.0

    decision_index = 0

    processed_count = 0
    dropped_count = 0

    scene_outputs = []

    while pos < n:

        dataset_index = indices[pos]

        action = policy[
            decision_index
            %
            len(policy)
        ]

        if action == "F":
            model = full_model
        else:
            model = light_model

        input_frame_id = frame_token(
            dataset,
            dataset_index,
        )

        arrival_ms = (
            pos * period_ms
        )

        start_ms = max(
            virtual_now_ms,
            arrival_ms,
        )

        (
            anno,
            service_ms,
            wall_finish_ns,
            updated_queue,
        ) = run_one_mixed_prediction(
            model=model,
            shared_queue=shared_queue,
            dataset=dataset,
            dataset_index=dataset_index,
        )

        # Only processed frames update temporal memory.
        shared_queue = updated_queue

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

        # Extra metadata for mixed-mode audit.
        stamped[
            "_stream_action"
        ] = action

        stamped[
            "_stream_decision_index"
        ] = int(
            decision_index
        )

        scene_outputs.append(
            stamped
        )

        processed_count += 1

        # ----------------------------------------------------
        # Capacity-1 latest-frame buffer
        # ----------------------------------------------------

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

            next_pos = (
                latest_arrived_pos
            )

            dropped_now = max(
                0,
                next_pos
                -
                pos
                -
                1,
            )

            next_reason = (
                "buffer_latest"
            )

        else:

            next_pos = pos + 1

            dropped_now = 0

            next_reason = (
                "wait_next_arrival"
            )

        dropped_count += (
            dropped_now
        )

        trace.append(
            {
                "scene":
                    scene,

                "dataset_index":
                    int(
                        dataset_index
                    ),

                "frame_id":
                    input_frame_id,

                "scene_pos":
                    int(pos),

                "decision_index":
                    int(
                        decision_index
                    ),

                "policy_index":
                    int(
                        decision_index
                        %
                        len(policy)
                    ),

                "action":
                    action,

                "arrival_ms":
                    float(
                        arrival_ms
                    ),

                "start_ms":
                    float(
                        start_ms
                    ),

                "finish_ms":
                    float(
                        finish_ms
                    ),

                "service_ms":
                    float(
                        service_ms
                    ),

                "output_latency_ms":
                    float(
                        finish_ms
                        -
                        arrival_ms
                    ),

                "buffer_wait_ms":
                    float(
                        start_ms
                        -
                        arrival_ms
                    ),

                "history_size_after":
                    int(
                        len(
                            shared_queue
                        )
                    ),

                "dropped_waiting_frames_after_this_output":
                    int(
                        dropped_now
                    ),

                "next_scene_pos":
                    (
                        int(next_pos)
                        if next_pos < n
                        else None
                    ),

                "next_reason":
                    (
                        next_reason
                        if next_pos < n
                        else "scene_end"
                    ),

                "wall_output_time_ns":
                    int(
                        wall_finish_ns
                    ),
            }
        )

        print(
            f"[{scene}] "
            f"decision={decision_index:03d} "
            f"action={action} "
            f"frame={input_frame_id} "
            f"pos={pos:03d}/{n - 1:03d} "
            f"arrival={arrival_ms:9.3f} "
            f"start={start_ms:9.3f} "
            f"finish={finish_ms:9.3f} "
            f"service={service_ms:7.3f} ms "
            f"drop+={dropped_now}"
        )

        virtual_now_ms = (
            finish_ms
        )

        pos = next_pos

        decision_index += 1

    return (
        scene_outputs,
        processed_count,
        dropped_count,
    )


# ============================================================
# Statistics
# ============================================================

def safe_mean(values):
    if not values:
        return None

    return float(
        np.mean(values)
    )


def safe_percentile(
    values,
    q,
):
    if not values:
        return None

    return float(
        np.percentile(
            values,
            q,
        )
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    policy = normalize_policy(
        args.policy
    )

    if args.input_hz <= 0:
        raise ValueError(
            "--input_hz must be > 0"
        )

    set_seed(
        args.seed
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = common_utils.create_logger(
        log_file=str(
            output_dir
            /
            "run.log"
        )
    )

    # ========================================================
    # Independent configs
    # ========================================================

    full_cfg = make_cfg(
        args.full_cfg
    )

    light_cfg = make_cfg(
        args.light_cfg
    )

    if (
        list(full_cfg.CLASS_NAMES)
        !=
        list(light_cfg.CLASS_NAMES)
    ):
        raise RuntimeError(
            "Full / Light CLASS_NAMES differ"
        )

    # ========================================================
    # Dataset
    # ========================================================

    dataset, _, _ = build_dataloader(
        dataset_cfg=
            light_cfg.DATA_CONFIG,

        class_names=
            light_cfg.CLASS_NAMES,

        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Empty dataset"
        )

    scene_to_indices = (
        build_scene_index(
            dataset
        )
    )

    # ========================================================
    # Full / Light
    # ========================================================

    full_model = build_network(
        model_cfg=
            full_cfg.MODEL,

        num_class=
            len(
                full_cfg.CLASS_NAMES
            ),

        dataset=dataset,
    )

    light_model = build_network(
        model_cfg=
            light_cfg.MODEL,

        num_class=
            len(
                light_cfg.CLASS_NAMES
            ),

        dataset=dataset,
    )

    full_model.load_params_from_file(
        filename=
            args.full_ckpt,

        logger=logger,
        to_cpu=True,
    )

    light_model.load_params_from_file(
        filename=
            args.light_ckpt,

        logger=logger,
        to_cpu=True,
    )

    full_model.cuda()
    light_model.cuda()

    full_model.eval()
    light_model.eval()

    # ========================================================
    # Mixed-history interface checks
    # ========================================================

    if (
        full_model.history_feature_queue
        is None
        or
        light_model.history_feature_queue
        is None
    ):
        raise RuntimeError(
            "Both models must have temporal history queues"
        )

    full_k = (
        full_model
        .history_feature_queue
        .maxlen
    )

    light_k = (
        light_model
        .history_feature_queue
        .maxlen
    )

    if full_k != light_k:
        raise RuntimeError(
            f"History length mismatch: "
            f"Full={full_k}, Light={light_k}"
        )

    if (
        full_model.history_features_name
        !=
        light_model.history_features_name
    ):
        raise RuntimeError(
            "Full / Light history feature names differ"
        )

    period_ms = (
        1000.0
        /
        args.input_hz
    )

    nominal_full_fraction = (
        policy.count("F")
        /
        len(policy)
    )

    print()
    print("=" * 96)
    print(
        "Fixed Mixed-Policy Real Streaming Evaluation"
    )
    print("=" * 96)

    print(
        f"policy             : {policy}"
    )

    print(
        f"nominal Full ratio : "
        f"{nominal_full_fraction:.4f}"
    )

    print(
        f"input_hz           : "
        f"{args.input_hz:.3f}"
    )

    print(
        f"frame period       : "
        f"{period_ms:.3f} ms"
    )

    print(
        f"history K          : "
        f"{full_k}"
    )

    print(
        f"scenes             : "
        f"{len(scene_to_indices)}"
    )

    print(
        f"sensor frames      : "
        f"{len(dataset)}"
    )

    print()
    print(
        "Policy advances on PROCESSED inference decisions."
    )

    print(
        "Dropped frames do NOT consume an action "
        "and do NOT enter temporal history."
    )

    print()
    print(
        "Timing scope:"
    )

    print(
        "  model forward + fusion + head + post-processing"
    )

    print(
        "  H2D and queue copying excluded"
    )

    print()
    print(
        "Buffer:"
    )

    print(
        "  capacity = 1, latest waiting frame wins"
    )

    print("=" * 96)
    print()

    # ========================================================
    # Warmup
    # ========================================================

    print(
        "[warmup] Full model"
    )

    warmup_model(
        full_model,
        dataset,
        scene_to_indices,
        args.warmup,
    )

    print(
        "[warmup] Light model"
    )

    warmup_model(
        light_model,
        dataset,
        scene_to_indices,
        args.warmup,
    )

    full_model.history_feature_queue.clear()
    light_model.history_feature_queue.clear()

    # ========================================================
    # Streaming rollout
    # ========================================================

    trace = []

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
        ) = run_scene_mixed(
            full_model=
                full_model,

            light_model=
                light_model,

            dataset=
                dataset,

            scene=
                scene,

            indices=
                indices,

            period_ms=
                period_ms,

            policy=
                policy,

            trace=
                trace,
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

    # ========================================================
    # Timestamp alignment
    # ========================================================

    all_gt_annos = []
    all_stream_det_annos = []

    for (
        scene,
        indices,
    ) in scene_to_indices.items():

        gt, det = (
            timestamp_align_scene(
                dataset=
                    dataset,

                scene=
                    scene,

                indices=
                    indices,

                scene_outputs=
                    outputs_by_scene[
                        scene
                    ],

                period_ms=
                    period_ms,
            )
        )

        all_gt_annos.extend(
            gt
        )

        all_stream_det_annos.extend(
            det
        )

    eval_det_annos = [
        strip_eval_metadata(
            copy.deepcopy(x)
        )
        for x in (
            all_stream_det_annos
        )
    ]

    # ========================================================
    # Streaming AP
    # ========================================================

    full_result_str, _ = (
        dataset.evaluation_offline(
            all_gt_annos,
            eval_det_annos,
            dataset.class_names,
            "3d",
        )
    )

    paper_result_str = (
        eval_utils.format_paper_metrics(
            full_result_str
        )
    )

    metrics = (
        parse_paper_metrics(
            paper_result_str
        )
    )

    mean_07 = (
        mean_3d_moderate(
            metrics,
            "0.7",
        )
    )

    mean_05 = (
        mean_3d_moderate(
            metrics,
            "0.5",
        )
    )

    # ========================================================
    # Runtime statistics
    # ========================================================

    full_service = [
        x["service_ms"]
        for x in trace
        if x["action"] == "F"
    ]

    light_service = [
        x["service_ms"]
        for x in trace
        if x["action"] == "L"
    ]

    all_service = [
        x["service_ms"]
        for x in trace
    ]

    output_latency = [
        x["output_latency_ms"]
        for x in trace
    ]

    buffer_wait = [
        x["buffer_wait_ms"]
        for x in trace
    ]

    full_actions = len(
        full_service
    )

    light_actions = len(
        light_service
    )

    realized_full_fraction = (
        full_actions
        /
        total_processed
        if total_processed > 0
        else 0.0
    )

    sensor_frames = len(
        dataset
    )

    drop_ratio = (
        total_dropped
        /
        sensor_frames
    )

    summary = {
        "policy":
            policy,

        "input_hz":
            float(
                args.input_hz
            ),

        "frame_period_ms":
            float(
                period_ms
            ),

        "nominal_full_fraction":
            float(
                nominal_full_fraction
            ),

        "realized_full_fraction":
            float(
                realized_full_fraction
            ),

        "full_cfg":
            args.full_cfg,

        "full_ckpt":
            args.full_ckpt,

        "light_cfg":
            args.light_cfg,

        "light_ckpt":
            args.light_ckpt,

        "history_len":
            int(
                full_k
            ),

        "sensor_frames":
            int(
                sensor_frames
            ),

        "processed_frames":
            int(
                total_processed
            ),

        "dropped_frames":
            int(
                total_dropped
            ),

        "drop_ratio":
            float(
                drop_ratio
            ),

        "full_actions":
            int(
                full_actions
            ),

        "light_actions":
            int(
                light_actions
            ),

        "service_ms": {
            "mean_all":
                safe_mean(
                    all_service
                ),

            "p95_all":
                safe_percentile(
                    all_service,
                    95,
                ),

            "mean_full":
                safe_mean(
                    full_service
                ),

            "p95_full":
                safe_percentile(
                    full_service,
                    95,
                ),

            "mean_light":
                safe_mean(
                    light_service
                ),

            "p95_light":
                safe_percentile(
                    light_service,
                    95,
                ),
        },

        "output_latency_ms": {
            "mean":
                safe_mean(
                    output_latency
                ),

            "p95":
                safe_percentile(
                    output_latency,
                    95,
                ),
        },

        "buffer_wait_ms": {
            "mean":
                safe_mean(
                    buffer_wait
                ),

            "p95":
                safe_percentile(
                    buffer_wait,
                    95,
                ),
        },

        "mean_sAP3D_moderate_0.7":
            float(
                mean_07
            ),

        "mean_sAP3D_moderate_0.5":
            float(
                mean_05
            ),

        "metrics":
            metrics,
    }

    # ========================================================
    # Save
    # ========================================================

    with open(
        output_dir
        /
        "trace.json",
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
        "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    (
        output_dir
        /
        "paper_metrics.txt"
    ).write_text(
        paper_result_str
    )

    (
        output_dir
        /
        "full_eval.txt"
    ).write_text(
        full_result_str
    )

    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 96)
    print(
        "STREAMING RESULT"
    )
    print("=" * 96)

    print(
        paper_result_str
    )

    print()
    print(
        f"Policy                  : {policy}"
    )

    print(
        f"Input Hz                : {args.input_hz:g}"
    )

    print(
        f"Nominal Full fraction   : "
        f"{nominal_full_fraction:.4f}"
    )

    print(
        f"Realized Full fraction  : "
        f"{realized_full_fraction:.4f}"
    )

    print(
        f"Processed               : "
        f"{total_processed}/{sensor_frames}"
    )

    print(
        f"Dropped                 : "
        f"{total_dropped}"
    )

    print(
        f"Drop ratio              : "
        f"{drop_ratio:.4f}"
    )

    print(
        f"Mean Full service       : "
        f"{safe_mean(full_service)} ms"
    )

    print(
        f"Mean Light service      : "
        f"{safe_mean(light_service)} ms"
    )

    print(
        f"Mean output latency     : "
        f"{safe_mean(output_latency):.3f} ms"
    )

    print(
        f"Mean sAP3D@0.7 Moderate : "
        f"{mean_07:.4f}"
    )

    print(
        f"Mean sAP3D@0.5 Moderate : "
        f"{mean_05:.4f}"
    )

    print()
    print(
        f"Saved to: {output_dir}"
    )

    print("=" * 96)


if __name__ == "__main__":
    main()
