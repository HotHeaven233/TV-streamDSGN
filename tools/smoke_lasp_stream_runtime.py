#!/usr/bin/env python3

import argparse
import math

import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from test_stream_buffer_timestamp import load_one


def check_prediction(pred_dicts, delta):
    if not isinstance(pred_dicts, list):
        raise RuntimeError(
            f"delta={delta}: pred_dicts must be list, "
            f"got {type(pred_dicts)}"
        )

    if len(pred_dicts) != 1:
        raise RuntimeError(
            f"delta={delta}: expected batch size 1, "
            f"got {len(pred_dicts)}"
        )

    pred = pred_dicts[0]

    required = (
        "pred_boxes",
        "pred_scores",
        "pred_labels",
    )

    missing = [
        k for k in required
        if k not in pred
    ]

    if missing:
        raise RuntimeError(
            f"delta={delta}: missing prediction keys: "
            f"{missing}"
        )

    boxes = pred["pred_boxes"]
    scores = pred["pred_scores"]
    labels = pred["pred_labels"]

    if boxes.ndim != 2:
        raise RuntimeError(
            f"delta={delta}: pred_boxes shape={tuple(boxes.shape)}"
        )

    if boxes.shape[-1] < 7:
        raise RuntimeError(
            f"delta={delta}: pred_boxes last dim "
            f"must be >=7, got {boxes.shape[-1]}"
        )

    n = boxes.shape[0]

    if scores.ndim != 1 or labels.ndim != 1:
        raise RuntimeError(
            f"delta={delta}: invalid score/label shapes: "
            f"{tuple(scores.shape)}, {tuple(labels.shape)}"
        )

    if scores.shape[0] != n or labels.shape[0] != n:
        raise RuntimeError(
            f"delta={delta}: prediction length mismatch: "
            f"boxes={n}, scores={scores.shape[0]}, "
            f"labels={labels.shape[0]}"
        )

    if not torch.isfinite(boxes).all():
        raise RuntimeError(
            f"delta={delta}: non-finite pred_boxes"
        )

    if not torch.isfinite(scores).all():
        raise RuntimeError(
            f"delta={delta}: non-finite pred_scores"
        )

    return {
        "n": int(n),
        "boxes": boxes.detach().clone(),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cfg",
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--index",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required"
        )

    cfg_from_yaml_file(
        args.cfg,
        cfg,
    )

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

    if not (
        0 <= args.index < len(dataset)
    ):
        raise ValueError(
            f"--index={args.index} outside "
            f"[0, {len(dataset)})"
        )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
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
    # Runtime API audit
    # ------------------------------------------------------------

    required_methods = (
        "forward_stream_core",
        "postprocess_last",
        "get_compensated_prediction",
    )

    missing_methods = [
        name
        for name in required_methods
        if not callable(
            getattr(model, name, None)
        )
    ]

    if missing_methods:
        raise RuntimeError(
            "LASP runtime API missing: "
            + ", ".join(missing_methods)
        )

    if not hasattr(
        model,
        "lasp_memory",
    ):
        raise RuntimeError(
            "LASP model has no lasp_memory"
        )

    if not hasattr(
        model,
        "lasp",
    ):
        raise RuntimeError(
            "LASP model has no lasp module"
        )

    model.lasp_memory.clear()

    if len(model.lasp_memory) != 0:
        raise RuntimeError(
            "lasp_memory did not clear"
        )

    # ------------------------------------------------------------
    # H2D deliberately occurs before forward timing/smoke.
    # ------------------------------------------------------------

    batch = load_one(
        dataset,
        args.index,
    )

    if "token" not in batch:
        raise RuntimeError(
            "loaded batch has no token frame"
        )

    # ------------------------------------------------------------
    # One real LASP neural-network forward.
    # ------------------------------------------------------------

    with (
        torch.no_grad(),
        torch.amp.autocast(
            "cuda",
            enabled=bool(
                model.use_amp_dict["TEST"]
            ),
        ),
    ):
        meta = model.forward_stream_core(
            batch
        )

    torch.cuda.synchronize()

    if not isinstance(meta, dict):
        raise RuntimeError(
            "forward_stream_core must return dict metadata"
        )

    if "lasp_sensor_position" not in meta:
        raise RuntimeError(
            "missing lasp_sensor_position metadata"
        )

    if "lasp_memory_frames" not in meta:
        raise RuntimeError(
            "missing lasp_memory_frames metadata"
        )

    pos = float(
        meta["lasp_sensor_position"]
    )

    memory_frames = int(
        meta["lasp_memory_frames"]
    )

    if not math.isfinite(pos):
        raise RuntimeError(
            f"invalid sensor position: {pos}"
        )

    if memory_frames != 1:
        raise RuntimeError(
            "first completed forward should create "
            f"exactly one memory entry, got {memory_frames}"
        )

    if len(model.lasp_memory) != 1:
        raise RuntimeError(
            "lasp_memory length mismatch after forward: "
            f"{len(model.lasp_memory)}"
        )

    print()
    print("[PASS] forward_stream_core")
    print("sensor_position =", pos)
    print("memory_frames   =", memory_frames)

    # ------------------------------------------------------------
    # Re-query the SAME completed forward at multiple horizons.
    #
    # No neural-network forward is executed below.
    # ------------------------------------------------------------

    horizons = (
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
    )

    results = {}

    for delta in horizons:
        with torch.no_grad():
            pred_dicts, ret_dicts = (
                model.postprocess_last(
                    delta_frames=delta
                )
            )

        checked = check_prediction(
            pred_dicts,
            delta,
        )

        results[delta] = checked

        print(
            f"[PASS] delta_frames={delta:4.1f} "
            f"num_boxes={checked['n']}"
        )

    # ------------------------------------------------------------
    # postprocess/trajectory lookup must NOT mutate model history.
    # ------------------------------------------------------------

    if len(model.lasp_memory) != 1:
        raise RuntimeError(
            "postprocess_last unexpectedly modified "
            f"LASP memory: len={len(model.lasp_memory)}"
        )

    # ------------------------------------------------------------
    # Alias API must also work.
    # ------------------------------------------------------------

    with torch.no_grad():
        alias_pred, _ = (
            model.get_compensated_prediction(
                1.0
            )
        )

    alias_checked = check_prediction(
        alias_pred,
        1.0,
    )

    if (
        alias_checked["n"]
        != results[1.0]["n"]
    ):
        raise RuntimeError(
            "get_compensated_prediction and "
            "postprocess_last disagree"
        )

    print(
        "[PASS] get_compensated_prediction alias"
    )

    # ------------------------------------------------------------
    # Important:
    # Do NOT require boxes at delta=0 and delta=1 to differ.
    #
    # A valid frame may contain static objects, or NMS may remove
    # moving candidates. We only test API/finite/runtime semantics.
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("LASP STREAM RUNTIME SMOKE PASS")
    print("=" * 72)
    print(
        "one neural forward completed; "
        "trajectory re-query works without adding history"
    )


if __name__ == "__main__":
    main()
