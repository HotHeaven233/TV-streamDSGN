#!/usr/bin/env python3

import argparse
import copy
import json
import pickle
import random
import re
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from torch.cuda.amp import autocast

from eval_utils import eval_utils
from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


CLASSES = ("Car", "Pedestrian", "Cyclist")


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Counterfactual temporal-state-value experiment. "
            "Same future frames and same Light compute; "
            "only H_t source differs: Full vs Light."
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
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--history_len",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max_horizon",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--anchor_stride",
        type=int,
        default=1,
        help=(
            "Evaluate every N-th valid anchor. "
            "Use >1 only for a quick sanity run."
        ),
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
        "--save_predictions",
        action="store_true",
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


def make_cfg(cfg_file):
    """
    Load Full / Light cfg independently.

    Do not reuse pcdet.config.cfg because Full and Light are
    two different model configurations in the same process.
    """

    c = EasyDict()

    c.ROOT_DIR = (
        Path(__file__).resolve().parent / ".."
    ).resolve()

    c.LOCAL_RANK = 0

    cfg_from_yaml_file(
        cfg_file,
        c,
    )

    # Evaluation does not need construction-time ImageNet init.
    if hasattr(c.MODEL, "BACKBONE_3D"):
        c.MODEL.BACKBONE_3D.feature_backbone_pretrained = None

    c.TAG = Path(cfg_file).stem

    # This experiment is deliberately fixed-frame.
    # No empirical timing / streaming alignment.
    c.DATA_CONFIG.INFER_TIME_PATH = None

    return c


# ============================================================
# Dataset chronology
# ============================================================

def frame_token(
    dataset,
    dataset_index,
):
    info = dataset.kitti_infos[
        dataset_index
    ]

    return str(
        info["sample_idx"]
        ["frame_tag"]
        ["token"]
    )


def frame_sort_key(
    dataset,
    dataset_index,
):
    token = frame_token(
        dataset,
        dataset_index,
    )

    try:
        return (0, int(token))
    except ValueError:
        return (1, token)


def build_scene_index(dataset):
    scenes = OrderedDict()

    for dataset_index, info in enumerate(
        dataset.kitti_infos
    ):
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


# ============================================================
# Data
# ============================================================

def load_one(
    dataset,
    dataset_index,
):
    data_dict = dataset[
        dataset_index
    ]

    batch_dict = dataset.collate_batch(
        [data_dict]
    )

    load_data_to_gpu(
        batch_dict
    )

    return batch_dict


def get_gt(
    dataset,
    dataset_index,
):
    return copy.deepcopy(
        dataset.kitti_infos[
            dataset_index
        ]["infos"]["token"]["annos"]
    )


# ============================================================
# History queue utilities
# ============================================================

def clone_value(x):
    if torch.is_tensor(x):
        return x.detach().clone()

    return copy.deepcopy(x)


def clone_feature_dict(
    feature_dict,
):
    return {
        key: clone_value(value)
        for key, value
        in feature_dict.items()
    }


def clone_history_queue(
    queue,
    maxlen,
):
    out = deque(
        maxlen=maxlen
    )

    for (
        sample_idx,
        feature_dict,
    ) in queue:

        out.append(
            (
                copy.deepcopy(
                    sample_idx
                ),
                clone_feature_dict(
                    feature_dict
                ),
            )
        )

    return out


def install_history_queue(
    model,
    queue,
):
    model.history_feature_queue.clear()

    for (
        sample_idx,
        feature_dict,
    ) in queue:

        model.history_feature_queue.append(
            (
                copy.deepcopy(
                    sample_idx
                ),
                clone_feature_dict(
                    feature_dict
                ),
            )
        )


# ============================================================
# State generation
# ============================================================

def extract_state(
    model,
    dataset,
    dataset_index,
):
    """
    Extract exactly the state that StreamDSGN would write
    into its temporal history.

    Only:
        feature_extractor

    No:
        temporal fusion
        VAN
        dense head
        NMS

    Therefore this is the pre-fusion BEV state H_t.
    """

    batch_dict = load_one(
        dataset,
        dataset_index,
    )

    cur_data = batch_dict[
        "token"
    ]

    with torch.no_grad():

        with autocast(
            enabled=model.use_amp_dict[
                "TEST"
            ]
        ):

            for module in (
                model.feature_extractor
            ):
                cur_data = module(
                    cur_data
                )

    state = {}

    for feature_name in (
        model.history_features_name
    ):

        state[feature_name] = (
            cur_data[
                feature_name
            ].detach().clone()
        )

    return (
        copy.deepcopy(
            cur_data[
                "this_sample_idx"
            ]
        ),
        state,
    )


# ============================================================
# Light future inference
# ============================================================

def run_light_prediction(
    model,
    dataset,
    dataset_index,
    input_queue,
):
    """
    Run current frame using Light, starting from an explicitly
    supplied temporal history queue.

    Returns:
        annotation
        history queue after current Light state has been appended
    """

    install_history_queue(
        model,
        input_queue,
    )

    batch_dict = load_one(
        dataset,
        dataset_index,
    )

    with torch.no_grad():
        pred_dicts, _ = model(
            batch_dict
        )

    annos = (
        dataset.generate_prediction_dicts(
            batch_dict,
            pred_dicts,
            dataset.class_names,
            output_path=None,
        )
    )

    if len(annos) != 1:
        raise RuntimeError(
            "Counterfactual test requires "
            f"batch_size=1, got {len(annos)}"
        )

    updated_queue = (
        clone_history_queue(
            model.history_feature_queue,
            maxlen=model
            .history_feature_queue
            .maxlen,
        )
    )

    return (
        annos[0],
        updated_queue,
    )


# ============================================================
# Diagnostics
# ============================================================

def state_shapes(state):
    result = {}

    for key, value in state.items():

        if torch.is_tensor(
            value
        ):
            result[key] = tuple(
                value.shape
            )
        else:
            result[key] = None

    return result


def queue_max_abs_diff(
    queue_a,
    queue_b,
):
    """
    Diagnostic only.

    At k=4 with K=3, the anchor H_t should already have
    left the FIFO queue. Because future stored Light states
    are pre-fusion, the two queues should then be identical.
    """

    if len(queue_a) != len(
        queue_b
    ):
        return float("inf")

    max_diff = 0.0

    for (
        (_, feat_a),
        (_, feat_b),
    ) in zip(
        queue_a,
        queue_b,
    ):

        if (
            feat_a.keys()
            !=
            feat_b.keys()
        ):
            return float("inf")

        for key in feat_a:

            a = feat_a[key]
            b = feat_b[key]

            if (
                torch.is_tensor(a)
                and
                torch.is_tensor(b)
            ):

                if a.shape != b.shape:
                    return float(
                        "inf"
                    )

                diff = (
                    (
                        a.float()
                        -
                        b.float()
                    )
                    .abs()
                    .max()
                    .item()
                )

                max_diff = max(
                    max_diff,
                    diff,
                )

    return max_diff


# ============================================================
# KITTI metric parsing
# ============================================================

def format_fixed_frame_metrics(
    full_result_text,
):
    """
    Reuse StreamDSGN's AP_R40 formatter.

    Its display labels say sAP because the same formatter is
    also used by the streaming runner. Here there is no timing,
    so relabel them as ordinary AP.
    """

    text = (
        eval_utils.format_paper_metrics(
            full_result_text
        )
    )

    text = text.replace(
        "sAPBEV",
        "APBEV",
    )

    text = text.replace(
        "sAP3D",
        "AP3D",
    )

    return text


def parse_paper_metrics(text):
    result = {}

    current_class = None
    current_iou = None

    for raw_line in (
        text.splitlines()
    ):
        line = raw_line.strip()

        if line in CLASSES:
            current_class = line

            result.setdefault(
                current_class,
                {},
            )

            continue

        if line.startswith(
            "IoU="
        ):
            current_iou = (
                line.split(
                    "=",
                    1,
                )[1]
            )

            if current_class:
                result[
                    current_class
                ].setdefault(
                    current_iou,
                    {},
                )

            continue

        if (
            current_class is None
            or
            current_iou is None
        ):
            continue

        match = re.match(
            r"(?:s?AP)?(BEV|3D)"
            r"\s*:\s*"
            r"([-+0-9.eE]+)"
            r"\s*,\s*"
            r"([-+0-9.eE]+)"
            r"\s*,\s*"
            r"([-+0-9.eE]+)",
            line,
        )

        if match is None:
            continue

        metric_name = (
            match.group(1)
        )

        values = [
            float(
                match.group(i)
            )
            for i in range(
                2,
                5,
            )
        ]

        result[
            current_class
        ][
            current_iou
        ][
            metric_name
        ] = {
            "easy": values[0],
            "moderate": values[1],
            "hard": values[2],
        }

    return result


def mean_3d_moderate(
    metrics,
    iou,
):
    values = []

    for class_name in CLASSES:

        values.append(
            metrics[
                class_name
            ][
                iou
            ][
                "3D"
            ][
                "moderate"
            ]
        )

    return float(
        np.mean(values)
    )


def evaluate_predictions(
    dataset,
    gt_annos,
    det_annos,
):
    full_text, _ = (
        dataset.evaluation_offline(
            gt_annos,
            det_annos,
            dataset.class_names,
            "3d",
        )
    )

    paper_text = (
        format_fixed_frame_metrics(
            full_text
        )
    )

    metrics = (
        parse_paper_metrics(
            paper_text
        )
    )

    return (
        full_text,
        paper_text,
        metrics,
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if args.history_len <= 0:
        raise ValueError(
            "--history_len must be > 0"
        )

    if args.max_horizon <= 0:
        raise ValueError(
            "--max_horizon must be > 0"
        )

    if args.anchor_stride <= 0:
        raise ValueError(
            "--anchor_stride must be > 0"
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

    logger = (
        common_utils.create_logger(
            log_file=str(
                output_dir
                /
                "run.log"
            )
        )
    )

    # --------------------------------------------------------
    # Independent Full / Light configs
    # --------------------------------------------------------

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
            "Full and Light CLASS_NAMES differ"
        )

    # --------------------------------------------------------
    # Shared dataset
    # --------------------------------------------------------

    dataset, _, _ = (
        build_dataloader(
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
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Empty dataset"
        )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    full_model = build_network(
        model_cfg=full_cfg.MODEL,
        num_class=len(
            full_cfg.CLASS_NAMES
        ),
        dataset=dataset,
    )

    light_model = build_network(
        model_cfg=light_cfg.MODEL,
        num_class=len(
            light_cfg.CLASS_NAMES
        ),
        dataset=dataset,
    )

    full_model.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )

    light_model.load_params_from_file(
        filename=args.light_ckpt,
        logger=logger,
        to_cpu=True,
    )

    full_model.cuda()
    light_model.cuda()

    full_model.eval()
    light_model.eval()

    # --------------------------------------------------------
    # Interface assertions
    # --------------------------------------------------------

    if (
        full_model
        .history_feature_queue
        is None
    ):
        raise RuntimeError(
            "Full model has no history queue"
        )

    if (
        light_model
        .history_feature_queue
        is None
    ):
        raise RuntimeError(
            "Light model has no history queue"
        )

    if (
        light_model
        .history_feature_queue
        .maxlen
        !=
        args.history_len
    ):
        raise RuntimeError(
            "Light history length mismatch: "
            f"model="
            f"{light_model.history_feature_queue.maxlen}, "
            f"arg={args.history_len}"
        )

    if (
        full_model.history_features_name
        !=
        light_model.history_features_name
    ):
        raise RuntimeError(
            "Full / Light history feature "
            "names differ"
        )

    scene_to_indices = (
        build_scene_index(
            dataset
        )
    )

    # --------------------------------------------------------
    # Valid anchors
    # --------------------------------------------------------

    valid_anchors = []

    for (
        scene,
        indices,
    ) in scene_to_indices.items():

        # Need K frames before t and max_horizon after t.
        stop = (
            len(indices)
            -
            args.max_horizon
        )

        for pos in range(
            args.history_len,
            stop,
            args.anchor_stride,
        ):
            valid_anchors.append(
                (
                    scene,
                    pos,
                )
            )

    if not valid_anchors:
        raise RuntimeError(
            "No valid anchors"
        )

    print()
    print("=" * 92)
    print(
        "Counterfactual Temporal State Value"
    )
    print("=" * 92)

    print(
        f"Full cfg      : "
        f"{args.full_cfg}"
    )
    print(
        f"Full ckpt     : "
        f"{args.full_ckpt}"
    )
    print(
        f"Light cfg     : "
        f"{args.light_cfg}"
    )
    print(
        f"Light ckpt    : "
        f"{args.light_ckpt}"
    )
    print(
        f"history_len   : "
        f"{args.history_len}"
    )
    print(
        f"max_horizon   : "
        f"{args.max_horizon}"
    )
    print(
        f"anchor_stride : "
        f"{args.anchor_stride}"
    )
    print(
        f"valid anchors : "
        f"{len(valid_anchors)}"
    )

    print()
    print(
        "Branch F: common L history "
        "-> H_t^F -> L -> L -> L -> L"
    )
    print(
        "Branch L: common L history "
        "-> H_t^L -> L -> L -> L -> L"
    )

    print()
    print(
        "No latency / no buffer / "
        "no frame dropping."
    )
    print("=" * 92)
    print()

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    records = {
        k: {
            "gt": [],
            "F": [],
            "L": [],
        }
        for k in range(
            1,
            args.max_horizon + 1,
        )
    }

    queue_diffs = {
        k: []
        for k in range(
            1,
            args.max_horizon + 1,
        )
    }

    anchor_state_diffs = []

    # --------------------------------------------------------
    # Counterfactual rollouts
    # --------------------------------------------------------

    for (
        anchor_no,
        (
            scene,
            pos,
        ),
    ) in enumerate(
        valid_anchors,
        start=1,
    ):

        indices = (
            scene_to_indices[
                scene
            ]
        )

        anchor_index = (
            indices[pos]
        )

        # ====================================================
        # Common memory before t:
        #
        # H_{t-3}^L,
        # H_{t-2}^L,
        # H_{t-1}^L
        # ====================================================

        common_queue = deque(
            maxlen=args.history_len
        )

        for hist_pos in range(
            pos - args.history_len,
            pos,
        ):

            hist_index = (
                indices[
                    hist_pos
                ]
            )

            (
                sample_idx,
                hist_state,
            ) = extract_state(
                light_model,
                dataset,
                hist_index,
            )

            common_queue.append(
                (
                    sample_idx,
                    hist_state,
                )
            )

        # ====================================================
        # Counterfactual intervention at frame t
        # ====================================================

        (
            sample_idx_l,
            state_l,
        ) = extract_state(
            light_model,
            dataset,
            anchor_index,
        )

        (
            sample_idx_f,
            state_f,
        ) = extract_state(
            full_model,
            dataset,
            anchor_index,
        )

        if (
            state_shapes(
                state_f
            )
            !=
            state_shapes(
                state_l
            )
        ):
            raise RuntimeError(
                "Unified state interface violated "
                f"at scene={scene}, pos={pos}: "
                f"F={state_shapes(state_f)}, "
                f"L={state_shapes(state_l)}"
            )

        # Measure how different H_t^F and H_t^L actually are.
        for feature_name in (
            state_f
        ):

            f = state_f[
                feature_name
            ]

            l = state_l[
                feature_name
            ]

            if (
                torch.is_tensor(f)
                and
                torch.is_tensor(l)
            ):
                anchor_state_diffs.append(
                    (
                        f.float()
                        -
                        l.float()
                    )
                    .abs()
                    .mean()
                    .item()
                )

        # ====================================================
        # Branch initialization
        # ====================================================

        f_queue = (
            clone_history_queue(
                common_queue,
                args.history_len,
            )
        )

        l_queue = (
            clone_history_queue(
                common_queue,
                args.history_len,
            )
        )

        # FIFO append H_t^F
        f_queue.append(
            (
                sample_idx_f,
                clone_feature_dict(
                    state_f
                ),
            )
        )

        # FIFO append H_t^L
        l_queue.append(
            (
                sample_idx_l,
                clone_feature_dict(
                    state_l
                ),
            )
        )

        # ====================================================
        # Future:
        #
        # t+1 ... t+4 are ALL Light.
        # ====================================================

        for k in range(
            1,
            args.max_horizon + 1,
        ):

            target_index = (
                indices[
                    pos + k
                ]
            )

            # Difference in queue BEFORE evaluating t+k.
            queue_diffs[k].append(
                queue_max_abs_diff(
                    f_queue,
                    l_queue,
                )
            )

            # Branch with H_t^F
            (
                anno_f,
                f_queue,
            ) = run_light_prediction(
                light_model,
                dataset,
                target_index,
                f_queue,
            )

            # Branch with H_t^L
            (
                anno_l,
                l_queue,
            ) = run_light_prediction(
                light_model,
                dataset,
                target_index,
                l_queue,
            )

            records[
                k
            ]["gt"].append(
                get_gt(
                    dataset,
                    target_index,
                )
            )

            records[
                k
            ]["F"].append(
                anno_f
            )

            records[
                k
            ]["L"].append(
                anno_l
            )

        if (
            anchor_no % 25 == 0
            or
            anchor_no
            ==
            len(valid_anchors)
        ):

            eviction_k = min(
                args.max_horizon,
                args.history_len + 1,
            )

            recent = (
                queue_diffs[
                    eviction_k
                ][-25:]
            )

            recent_max = (
                max(recent)
                if recent
                else float("nan")
            )

            print(
                f"[progress] "
                f"{anchor_no}/"
                f"{len(valid_anchors)} "
                f"| scene={scene} "
                f"| pos={pos} "
                f"| pre-k{eviction_k} "
                f"queue_maxdiff="
                f"{recent_max:.6g}"
            )

    # --------------------------------------------------------
    # Evaluate each horizon
    # --------------------------------------------------------

    summary = {
        "full_cfg":
            args.full_cfg,
        "full_ckpt":
            args.full_ckpt,
        "light_cfg":
            args.light_cfg,
        "light_ckpt":
            args.light_ckpt,
        "history_len":
            args.history_len,
        "max_horizon":
            args.max_horizon,
        "anchor_stride":
            args.anchor_stride,
        "num_valid_anchors":
            len(valid_anchors),
        "anchor_state_mean_abs_diff":
            (
                float(
                    np.mean(
                        anchor_state_diffs
                    )
                )
                if anchor_state_diffs
                else None
            ),
        "horizons": {},
    }

    print()
    print("=" * 118)
    print(
        "MAIN RESULT | fixed-frame state value "
        "| AP_R40 3D@0.7 Moderate"
    )
    print("=" * 118)

    print(
        f"{'k':>3} "
        f"{'Car F':>10} "
        f"{'Car L':>10} "
        f"{'Ped F':>10} "
        f"{'Ped L':>10} "
        f"{'Cyc F':>10} "
        f"{'Cyc L':>10} "
        f"{'Mean F':>10} "
        f"{'Mean L':>10} "
        f"{'G_mem':>10} "
        f"{'Qdiff':>12}"
    )

    print("-" * 118)

    for k in range(
        1,
        args.max_horizon + 1,
    ):

        gt_annos = (
            records[
                k
            ]["gt"]
        )

        (
            full_text_f,
            paper_text_f,
            metrics_f,
        ) = evaluate_predictions(
            dataset,
            gt_annos,
            records[k]["F"],
        )

        (
            full_text_l,
            paper_text_l,
            metrics_l,
        ) = evaluate_predictions(
            dataset,
            gt_annos,
            records[k]["L"],
        )

        mean_f_07 = (
            mean_3d_moderate(
                metrics_f,
                "0.7",
            )
        )

        mean_l_07 = (
            mean_3d_moderate(
                metrics_l,
                "0.7",
            )
        )

        mean_f_05 = (
            mean_3d_moderate(
                metrics_f,
                "0.5",
            )
        )

        mean_l_05 = (
            mean_3d_moderate(
                metrics_l,
                "0.5",
            )
        )

        g_07 = (
            mean_f_07
            -
            mean_l_07
        )

        g_05 = (
            mean_f_05
            -
            mean_l_05
        )

        qdiff_max = float(
            np.max(
                queue_diffs[k]
            )
        )

        qdiff_mean = float(
            np.mean(
                queue_diffs[k]
            )
        )

        f07 = {
            cls:
            metrics_f[
                cls
            ][
                "0.7"
            ][
                "3D"
            ][
                "moderate"
            ]
            for cls in CLASSES
        }

        l07 = {
            cls:
            metrics_l[
                cls
            ][
                "0.7"
            ][
                "3D"
            ][
                "moderate"
            ]
            for cls in CLASSES
        }

        print(
            f"{k:>3d} "
            f"{f07['Car']:>10.4f} "
            f"{l07['Car']:>10.4f} "
            f"{f07['Pedestrian']:>10.4f} "
            f"{l07['Pedestrian']:>10.4f} "
            f"{f07['Cyclist']:>10.4f} "
            f"{l07['Cyclist']:>10.4f} "
            f"{mean_f_07:>10.4f} "
            f"{mean_l_07:>10.4f} "
            f"{g_07:>10.4f} "
            f"{qdiff_max:>12.6g}"
        )

        summary[
            "horizons"
        ][
            str(k)
        ] = {
            "num_samples":
                len(gt_annos),

            "full_history_branch":
                metrics_f,

            "light_history_branch":
                metrics_l,

            "mean_3d_moderate_0.7_F":
                mean_f_07,

            "mean_3d_moderate_0.7_L":
                mean_l_07,

            "G_mem_0.7":
                g_07,

            "mean_3d_moderate_0.5_F":
                mean_f_05,

            "mean_3d_moderate_0.5_L":
                mean_l_05,

            "G_mem_0.5":
                g_05,

            "queue_max_abs_diff_mean":
                qdiff_mean,

            "queue_max_abs_diff_max":
                qdiff_max,
        }

        # Detailed KITTI metrics
        (
            output_dir
            /
            f"k{k}_F_full.txt"
        ).write_text(
            full_text_f
        )

        (
            output_dir
            /
            f"k{k}_F_paper.txt"
        ).write_text(
            paper_text_f
        )

        (
            output_dir
            /
            f"k{k}_L_full.txt"
        ).write_text(
            full_text_l
        )

        (
            output_dir
            /
            f"k{k}_L_paper.txt"
        ).write_text(
            paper_text_l
        )

        if args.save_predictions:

            with open(
                output_dir
                /
                f"k{k}_predictions.pkl",
                "wb",
            ) as f:

                pickle.dump(
                    {
                        "gt":
                            gt_annos,

                        "F_history":
                            records[
                                k
                            ]["F"],

                        "L_history":
                            records[
                                k
                            ]["L"],
                    },
                    f,
                    protocol=
                        pickle.HIGHEST_PROTOCOL,
                )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

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

    print("-" * 118)

    print()
    print(
        "G_mem(k) = "
        "Q(L_{t+k} | H_t^F) "
        "- "
        "Q(L_{t+k} | H_t^L)"
    )

    print()

    print(
        "Mean |H_t^F - H_t^L| = "
        f"{summary['anchor_state_mean_abs_diff']:.6g}"
    )

    # K=3 => H_t should be evicted before evaluating t+4.
    eviction_k = (
        args.history_len
        +
        1
    )

    if (
        eviction_k
        <=
        args.max_horizon
    ):

        result = (
            summary[
                "horizons"
            ][
                str(
                    eviction_k
                )
            ]
        )

        print()

        print(
            f"Eviction sanity k={eviction_k}:"
        )

        print(
            "  G_mem(0.7)   = "
            f"{result['G_mem_0.7']:.6f}"
        )

        print(
            "  queue maxdiff= "
            f"{result['queue_max_abs_diff_max']:.6g}"
        )

        print(
            "Expected for K=3 pre-fusion FIFO:"
        )

        print(
            "  queue maxdiff ~= 0"
        )

        print(
            "  G_mem(4)     ~= 0"
        )

    print()

    print(
        f"Saved results: "
        f"{output_dir}"
    )

    print("=" * 118)


if __name__ == "__main__":
    main()
