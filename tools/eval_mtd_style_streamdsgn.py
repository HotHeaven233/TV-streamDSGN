#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import shutil
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np

from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

from test_tv_stream3d_online_forward import make_cfg

from eval_tv_stream3d_30hz_random50 import (
    frame_meta,
    scene_groups,
    copy_det,
    builtin,
)

from mtd_style_runtime import (
    MTDDelayAnalyzer,
    MTDTimeStepBranchBank,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "MTD-style StreamDSGN "
            "causal replay evaluator"
        )
    )

    p.add_argument(
        "--cfg",
        required=True,
    )

    p.add_argument(
        "--source_dir",
        required=True,
    )

    p.add_argument(
        "--calibration_summary",
        required=True,
    )

    p.add_argument(
        "--output_dir",
        required=True,
    )

    p.add_argument(
        "--match_threshold_m",
        type=float,
        default=10.0,
    )

    p.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    p.add_argument(
        "--require_full",
        action="store_true",
    )

    return p.parse_args()


def load_json(
    path,
):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return json.loads(
        path.read_text()
    )


def validate_original_summary(
    summary,
    where,
):
    if (
        summary.get("method")
        !=
        "Original StreamDSGN"
    ):
        raise RuntimeError(
            f"{where}: expected "
            "Original StreamDSGN, got "
            f"{summary.get('method')}"
        )

    base = summary.get(
        "base_detector"
    )

    if (
        base is not None
        and
        base != "vanilla_streamdsgn"
    ):
        raise RuntimeError(
            f"{where}: expected "
            "vanilla_streamdsgn, got "
            f"{base}"
        )


def main():
    args = parse_args()

    if args.match_threshold_m <= 0:
        raise ValueError(
            "match_threshold_m must be > 0"
        )

    source_dir = Path(
        args.source_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_summary_path = (
        source_dir
        /
        "summary.json"
    )

    source_events_path = (
        source_dir
        /
        "prediction_events.pkl"
    )

    source_timeline_path = (
        source_dir
        /
        "frame_timeline.csv"
    )

    source_summary = load_json(
        source_summary_path
    )

    validate_original_summary(
        source_summary,
        source_summary_path,
    )

    calibration_path = Path(
        args.calibration_summary
    )

    calibration_summary = load_json(
        calibration_path
    )

    validate_original_summary(
        calibration_summary,
        calibration_path,
    )

    if not calibration_summary.get(
        "no_load",
        False,
    ):
        raise RuntimeError(
            "calibration summary must "
            "be an Original StreamDSGN "
            "no-load run"
        )

    initial_runtime_ms = float(
        calibration_summary[
            "forward_latency"
        ][
            "p50_ms"
        ]
    )

    if initial_runtime_ms <= 0:
        raise RuntimeError(
            "invalid calibration p50"
        )

    if not source_events_path.exists():
        raise FileNotFoundError(
            source_events_path
        )

    if not source_timeline_path.exists():
        raise FileNotFoundError(
            source_timeline_path
        )

    with source_events_path.open(
        "rb"
    ) as f:
        source_events = (
            pickle.load(f)
        )

    timeline = {}

    with source_timeline_path.open(
        newline=""
    ) as f:
        for row in csv.DictReader(f):
            idx = int(
                row["global_index"]
            )

            if idx in timeline:
                raise RuntimeError(
                    f"duplicate timeline "
                    f"index {idx}"
                )

            timeline[idx] = row

    input_hz = float(
        source_summary[
            "input_hz"
        ]
    )

    period_ms = float(
        source_summary[
            "period_ms"
        ]
    )

    sensor_frames = int(
        source_summary[
            "sensor_frames"
        ]
    )

    processed_frames = int(
        source_summary[
            "processed_frames"
        ]
    )

    if (
        len(source_events)
        !=
        processed_frames
    ):
        raise RuntimeError(
            "event count mismatch: "
            f"{len(source_events)} != "
            f"{processed_frames}"
        )

    cfg = make_cfg(
        args.cfg
    )

    logger = (
        common_utils.create_logger()
    )

    dataset, _, _ = (
        build_dataloader(
            dataset_cfg=(
                cfg.DATA_CONFIG
            ),
            class_names=(
                cfg.CLASS_NAMES
            ),
            batch_size=1,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=False,
        )
    )

    if sensor_frames > len(dataset):
        raise RuntimeError(
            f"source has {sensor_frames} "
            f"frames but dataset has "
            f"{len(dataset)}"
        )

    if (
        args.require_full
        and
        sensor_frames != len(dataset)
    ):
        raise RuntimeError(
            "formal run required: "
            f"source={sensor_frames}, "
            f"dataset={len(dataset)}"
        )

    eval_indices = list(
        range(sensor_frames)
    )

    groups = scene_groups(
        dataset,
        eval_indices,
    )

    source_by_scene = (
        OrderedDict(
            (
                scene,
                [],
            )
            for scene
            in groups
        )
    )

    for event in source_events:
        scene = event[
            "scene"
        ]

        if scene not in source_by_scene:
            raise RuntimeError(
                "unknown event scene: "
                f"{scene}"
            )

        source_by_scene[
            scene
        ].append(
            event
        )

    for scene in source_by_scene:
        source_by_scene[
            scene
        ].sort(
            key=lambda e:
                float(
                    e["finish_ms"]
                )
        )

    mtd_events = []
    event_rows = []

    branch_counts = Counter()
    raw_delay_counts = Counter()

    total_boxes = 0
    total_matches = 0

    estimates_ms = []

    for (
        scene,
        _indices,
    ) in groups.items():

        # MTD timing state resets at a scene boundary.
        dam = MTDDelayAnalyzer(
            period_ms=period_ms,
            initial_runtime_ms=(
                initial_runtime_ms
            ),
            num_branches=3,
        )

        tbm = (
            MTDTimeStepBranchBank(
                match_threshold_m=(
                    args.match_threshold_m
                )
            )
        )

        for event in (
            source_by_scene[
                scene
            ]
        ):
            source_index = int(
                event[
                    "source_index"
                ]
            )

            local_pos = int(
                event[
                    "local_pos"
                ]
            )

            row = timeline.get(
                source_index
            )

            if row is None:
                raise RuntimeError(
                    "missing timeline row "
                    "for processed frame "
                    f"{source_index}"
                )

            if (
                row["status"]
                !=
                "processed"
            ):
                raise RuntimeError(
                    "event points to "
                    "non-processed frame "
                    f"{source_index}"
                )

            observed_ms = float(
                row[
                    "forward_ms"
                ]
            )

            (
                branch,
                raw_delay,
                estimate_ms,
            ) = (
                dam.select_branch()
            )

            routed_anno = (
                tbm.route(
                    base_anno=(
                        event["anno"]
                    ),
                    current_local_pos=(
                        local_pos
                    ),
                    branch_index=(
                        branch
                    ),
                )
            )

            new_event = dict(
                event
            )

            new_event[
                "anno"
            ] = routed_anno

            new_event[
                "mtd_branch"
            ] = int(branch)

            new_event[
                "mtd_horizon_steps"
            ] = int(
                branch + 1
            )

            new_event[
                "mtd_raw_delay"
            ] = int(
                raw_delay
            )

            new_event[
                "mtd_estimated_runtime_ms"
            ] = float(
                estimate_ms
            )

            mtd_events.append(
                new_event
            )

            event_rows.append({
                "source_index":
                    source_index,

                "scene":
                    scene,

                "local_pos":
                    local_pos,

                "source_frame_id":
                    event[
                        "source_frame_id"
                    ],

                "finish_ms":
                    float(
                        event[
                            "finish_ms"
                        ]
                    ),

                "forward_ms":
                    observed_ms,

                "estimated_runtime_ms":
                    float(
                        estimate_ms
                    ),

                "raw_delay":
                    int(
                        raw_delay
                    ),

                "branch_index":
                    int(
                        branch
                    ),

                "horizon_steps":
                    int(
                        branch + 1
                    ),
            })

            branch_counts[
                int(branch)
            ] += 1

            raw_delay_counts[
                int(raw_delay)
            ] += 1

            estimates_ms.append(
                float(
                    estimate_ms
                )
            )

            # Current runtime becomes visible only
            # AFTER the current job finishes.
            dam.update(
                observed_ms
            )

        total_boxes += (
            tbm.total_current_boxes
        )

        total_matches += (
            tbm.total_matched_boxes
        )

    if (
        len(mtd_events)
        !=
        processed_frames
    ):
        raise RuntimeError(
            "MTD event count "
            "changed unexpectedly"
        )

    # ============================================================
    # Causal sAP alignment
    # ============================================================

    mtd_by_scene = (
        OrderedDict(
            (
                scene,
                [],
            )
            for scene
            in groups
        )
    )

    for event in mtd_events:
        mtd_by_scene[
            event["scene"]
        ].append(
            event
        )

    for scene in mtd_by_scene:
        mtd_by_scene[
            scene
        ].sort(
            key=lambda e:
                float(
                    e["finish_ms"]
                )
        )

    aligned = {}

    for (
        scene,
        indices,
    ) in groups.items():

        events = (
            mtd_by_scene[
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
                len(events)
                and
                float(
                    events[ptr][
                        "finish_ms"
                    ]
                )
                <=
                query_ms
                +
                1e-9
            ):
                latest = (
                    events[ptr][
                        "anno"
                    ]
                )

                ptr += 1

            (
                _,
                frame_id,
                next_frame_id,
            ) = frame_meta(
                dataset,
                idx,
            )

            aligned[idx] = (
                copy_det(
                    latest,
                    scene,
                    frame_id,
                    next_frame_id,
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

    event_csv = (
        output_dir
        /
        "mtd_event_timeline.csv"
    )

    with event_csv.open(
        "w",
        newline="",
    ) as f:
        fieldnames = [
            "source_index",
            "scene",
            "local_pos",
            "source_frame_id",
            "finish_ms",
            "forward_ms",
            "estimated_runtime_ms",
            "raw_delay",
            "branch_index",
            "horizon_steps",
        ]

        w = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        w.writeheader()

        w.writerows(
            event_rows
        )

    with (
        output_dir
        /
        "prediction_events.pkl"
    ).open("wb") as f:
        pickle.dump(
            mtd_events,
            f,
            protocol=(
                pickle
                .HIGHEST_PROTOCOL
            ),
        )

    with (
        output_dir
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
        output_dir
        /
        "stream_sap_result.txt"
    ).write_text(
        result_str
    )

    (
        output_dir
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

    source_trace = (
        source_dir
        /
        "contention_trace.csv"
    )

    if source_trace.exists():
        shutil.copy2(
            source_trace,
            output_dir
            /
            "contention_trace.csv",
        )

    match_ratio = (
        total_matches
        /
        total_boxes
        if total_boxes > 0
        else 0.0
    )

    estimate_stats = {
        "count":
            len(
                estimates_ms
            ),

        "mean_ms":
            float(
                np.mean(
                    estimates_ms
                )
            )
            if estimates_ms
            else None,

        "p50_ms":
            float(
                np.percentile(
                    estimates_ms,
                    50,
                )
            )
            if estimates_ms
            else None,

        "p90_ms":
            float(
                np.percentile(
                    estimates_ms,
                    90,
                )
            )
            if estimates_ms
            else None,

        "p99_ms":
            float(
                np.percentile(
                    estimates_ms,
                    99,
                )
            )
            if estimates_ms
            else None,
    }

    summary = {
        "version":
            "mtd_style_streamdsgn_v1",

        "method":
            "MTD-style StreamDSGN",

        "base_detector":
            "vanilla_streamdsgn",

        "source_original_summary":
            str(
                source_summary_path
            ),

        "model_cfg":
            str(
                args.cfg
            ),

        "input_hz":
            input_hz,

        "period_ms":
            period_ms,

        "timing_scope":
            "forward_only",

        "dam": {
            "rule":
                "floor(min(C_prev,C_prev2)/T), "
                "clipped to [0,2]",

            "initial_runtime_source":
                str(
                    calibration_path
                ),

            "initial_runtime_p50_ms":
                initial_runtime_ms,

            "branch_counts": {
                str(k):
                    int(v)
                for k, v
                in sorted(
                    branch_counts.items()
                )
            },

            "raw_delay_counts": {
                str(k):
                    int(v)
                for k, v
                in sorted(
                    raw_delay_counts.items()
                )
            },

            "estimated_runtime":
                estimate_stats,
        },

        "tbm": {
            "num_branches":
                3,

            "branch_semantics": {
                "0":
                    "h=1 native StreamDSGN "
                    "next prediction",

                "1":
                    "h=2 causal one-step "
                    "motion propagation",

                "2":
                    "h=3 causal two-step "
                    "motion propagation",
            },

            "match_threshold_m":
                args.match_threshold_m,

            "matched_boxes":
                total_matches,

            "current_boxes":
                total_boxes,

            "match_ratio":
                match_ratio,
        },

        "sensor_frames":
            sensor_frames,

        "processed_frames":
            processed_frames,

        "dropped_frames":
            int(
                source_summary[
                    "dropped_frames"
                ]
            ),

        "drop_rate":
            float(
                source_summary[
                    "drop_rate"
                ]
            ),

        "deadline_miss_count":
            int(
                source_summary[
                    "deadline_miss_count"
                ]
            ),

        "deadline_miss_rate":
            float(
                source_summary[
                    "deadline_miss_rate"
                ]
            ),

        "forward_latency":
            source_summary[
                "forward_latency"
            ],

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

    for key in [
        "contention_trace",
        "true_sensor_levels",
        "true_processed_levels",
        "no_load",
    ]:
        if key in source_summary:
            summary[key] = (
                source_summary[key]
            )

    summary_path = (
        output_dir
        /
        "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            builtin(
                summary
            ),
            indent=2,
        )
        +
        "\n"
    )

    print(
        "=" * 100
    )

    print(
        "MTD-style StreamDSGN "
        f"@ {input_hz:g} Hz"
    )

    print(
        "source Original     : "
        f"{source_dir}"
    )

    print(
        "initial runtime p50 : "
        f"{initial_runtime_ms:.4f} ms"
    )

    print(
        "branch counts       : "
        f"{dict(sorted(branch_counts.items()))}"
    )

    print(
        "raw delay counts    : "
        f"{dict(sorted(raw_delay_counts.items()))}"
    )

    print(
        "TBM match ratio     : "
        f"{100.0 * match_ratio:.2f}%"
    )

    print(
        "sAP 3D Moderate R40: "
        f"Car={car:.4f} "
        f"Ped={ped:.4f} "
        f"Cyc={cyc:.4f} "
        f"Macro={macro:.4f}"
    )

    print(
        "summary             : "
        f"{summary_path}"
    )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()


# MTD_STYLE_STREAMDSGN_EOF
