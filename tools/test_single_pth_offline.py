#!/usr/bin/env python3

import argparse
import copy
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import tqdm

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


torch.backends.cudnn.benchmark = True


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate exactly one StreamDSGN checkpoint "
            "with chronological offline KITTI 3D evaluation."
        )
    )

    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="Model yaml config."
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Checkpoint .pth file."
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
        type=str,
        default="outputs/single_pth_offline",
    )

    return parser.parse_args()


# ============================================================
# Utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_history(model):
    """
    Clear temporal history explicitly when a scene starts.
    """
    queue = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if queue is not None:
        queue.clear()


def frame_token(dataset, dataset_index):
    info = dataset.kitti_infos[
        dataset_index
    ]

    return str(
        info[
            "sample_idx"
        ][
            "frame_tag"
        ][
            "token"
        ]
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
    """
    Explicitly group all validation samples by scene
    and sort them chronologically.

    This is important for stateful StreamDSGN:
        t-3 -> t-2 -> t-1 -> t
    must be evaluated in temporal order.
    """

    scenes = OrderedDict()

    for dataset_index, info in enumerate(
        dataset.kitti_infos
    ):
        scene = str(
            info[
                "sample_idx"
            ][
                "scene"
            ]
        )

        scenes.setdefault(
            scene,
            [],
        ).append(
            dataset_index
        )

    for scene in scenes:
        scenes[scene].sort(
            key=lambda idx:
                frame_sort_key(
                    dataset,
                    idx,
                )
        )

    return scenes


def load_one(
    dataset,
    dataset_index,
):
    """
    Load one frame and transfer it to GPU.
    """

    data_dict = dataset[
        dataset_index
    ]

    batch_dict = (
        dataset.collate_batch(
            [data_dict]
        )
    )

    load_data_to_gpu(
        batch_dict
    )

    return batch_dict


# ============================================================
# Single frame inference
# ============================================================

def run_one_prediction(
    model,
    dataset,
    dataset_index,
):

    batch_dict = load_one(
        dataset,
        dataset_index,
    )

    with torch.no_grad():
        pred_dicts, ret_dict = model(
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
            "This script requires "
            "batch_size=1, but got "
            f"{len(annos)} predictions."
        )

    return (
        annos[0],
        ret_dict,
    )


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    set_seed(
        args.seed
    )

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    cfg.TAG = Path(
        args.cfg_file
    ).stem

    # --------------------------------------------------------
    # Disable old torchvision://resnet18 initialization.
    #
    # We immediately load the trained checkpoint below,
    # therefore construction-time ImageNet initialization
    # is unnecessary.
    # --------------------------------------------------------

    if hasattr(
        cfg.MODEL,
        "BACKBONE_3D"
    ):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None

    # --------------------------------------------------------
    # We explicitly perform offline evaluation below.
    #
    # Do NOT use old empirical-latency streaming simulator.
    # --------------------------------------------------------

    cfg.DATA_CONFIG.INFER_TIME_PATH = None

    logger = common_utils.create_logger()

    logger.info(
        "=" * 80
    )
    logger.info(
        "Single-checkpoint offline evaluation"
    )
    logger.info(
        "=" * 80
    )
    logger.info(
        f"cfg_file: {args.cfg_file}"
    )
    logger.info(
        f"ckpt: {args.ckpt}"
    )

    # ========================================================
    # Dataset
    # ========================================================

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Validation dataset is empty."
        )

    scene_to_indices = (
        build_scene_index(
            dataset
        )
    )

    logger.info(
        f"validation frames: {len(dataset)}"
    )
    logger.info(
        f"scenes: {len(scene_to_indices)}"
    )

    # ========================================================
    # Model
    # ========================================================

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

    logger.info(
        "Model loaded successfully."
    )

    # ========================================================
    # Chronological inference
    # ========================================================

    det_annos = []
    gt_annos = []

    total_frames = sum(
        len(indices)
        for indices
        in scene_to_indices.values()
    )

    pbar = tqdm.tqdm(
        total=total_frames,
        desc="offline eval",
        dynamic_ncols=True,
    )

    for scene, indices in (
        scene_to_indices.items()
    ):

        # Always reset memory at scene boundary.
        clear_history(
            model
        )

        for dataset_index in indices:

            anno, ret_dict = (
                run_one_prediction(
                    model=model,
                    dataset=dataset,
                    dataset_index=dataset_index,
                )
            )

            det_annos.append(
                copy.deepcopy(
                    anno
                )
            )

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

            gt_annos.append(
                gt
            )

            pbar.update(1)

    pbar.close()

    clear_history(
        model
    )

    if len(det_annos) != len(gt_annos):
        raise RuntimeError(
            f"Prediction/GT count mismatch: "
            f"{len(det_annos)} vs "
            f"{len(gt_annos)}"
        )

    logger.info(
        f"Finished inference on "
        f"{len(det_annos)} frames."
    )

    # ========================================================
    # KITTI offline 3D evaluation
    # ========================================================

    full_result_str, result_dict = (
        dataset.evaluation_offline(
            gt_annos,
            det_annos,
            dataset.class_names,
            "3d",
        )
    )

    # Same paper-metric formatter already used by
    # test_stream_buffer_timestamp.py.
    paper_result_str = (
        eval_utils.format_paper_metrics(
            full_result_str
        )
    )

    # ========================================================
    # Save results
    # ========================================================

    ckpt_stem = Path(
        args.ckpt
    ).stem

    output_dir = (
        Path(args.output_dir)
        /
        cfg.TAG
        /
        ckpt_stem
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    full_result_path = (
        output_dir
        /
        "offline_3d_full.txt"
    )

    paper_result_path = (
        output_dir
        /
        "offline_3d_paper_metrics.txt"
    )

    full_result_path.write_text(
        full_result_str
    )

    paper_result_path.write_text(
        paper_result_str
        + "\n"
    )

    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 80)
    print("FULL KITTI OFFLINE 3D RESULT")
    print("=" * 80)
    print()
    print(
        full_result_str
    )

    print()
    print("=" * 80)
    print(
        "STREAMDSGN PAPER METRICS "
        "(AP_R40 BEV / 3D)"
    )
    print("=" * 80)
    print()
    print(
        paper_result_str
    )

    print()
    print("=" * 80)
    print("RESULT FILES")
    print("=" * 80)
    print(
        f"full  : {full_result_path}"
    )
    print(
        f"paper : {paper_result_path}"
    )
    print()


if __name__ == "__main__":
    main()
