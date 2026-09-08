#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval

from test_stream_buffer_timestamp import load_one
from test_tv_stream3d_online_forward import make_cfg

from eval_tv_stream3d_30hz_random50 import (
    frame_meta,
    scene_groups,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--cfg",
        required=True,
    )

    p.add_argument(
        "--ckpt",
        required=True,
    )

    p.add_argument(
        "--horizons",
        default="0,1,2,3",
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
        "--output",
        required=True,
    )

    return p.parse_args()


def reset_lasp(model):
    memory = getattr(
        model,
        "lasp_memory",
        None,
    )

    if memory is None:
        raise RuntimeError(
            "model has no lasp_memory"
        )

    memory.clear()

    q = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if q is not None:
        q.clear()


def cpu_pred(pred):
    out = {}

    for k, v in pred.items():
        if torch.is_tensor(v):
            out[k] = (
                v.detach()
                .cpu()
                .clone()
            )
        else:
            out[k] = copy.deepcopy(v)

    return out


def metric(ap_dict):
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

    vals = np.asarray(
        [car, ped, cyc],
        dtype=np.float64,
    )

    macro = float(
        np.nanmean(vals)
    )

    return {
        "Car": car,
        "Pedestrian": ped,
        "Cyclist": cyc,
        "Macro": macro,
    }


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    horizons = sorted(
        set(
            int(x.strip())
            for x
            in args.horizons.split(",")
            if x.strip()
        )
    )

    if not horizons:
        raise RuntimeError(
            "no horizons"
        )

    if min(horizons) < 0:
        raise RuntimeError(
            "negative horizon"
        )

    cfg = make_cfg(
        args.cfg
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

    for name in (
        "forward_stream_core",
        "postprocess_last",
    ):
        if not callable(
            getattr(model, name, None)
        ):
            raise RuntimeError(
                f"missing runtime API: {name}"
            )

    count = (
        len(dataset)
        if args.max_frames == 0
        else min(
            len(dataset),
            args.max_frames,
        )
    )

    indices = list(
        range(count)
    )

    groups = scene_groups(
        dataset,
        indices,
    )

    #
    # Per horizon:
    #
    # target GT + compensated prediction
    #
    gt_by_h = {
        h: []
        for h in horizons
    }

    det_by_h = {
        h: []
        for h in horizons
    }

    counts = {
        h: 0
        for h in horizons
    }

    processed = 0

    for scene_i, (
        scene,
        scene_indices,
    ) in enumerate(
        groups.items(),
        start=1,
    ):
        reset_lasp(
            model
        )

        print(
            f"[scene "
            f"{scene_i:02d}/"
            f"{len(groups):02d}] "
            f"{scene}: "
            f"{len(scene_indices)} frames"
        )

        n = len(
            scene_indices
        )

        for pos, idx in enumerate(
            scene_indices
        ):
            batch = load_one(
                dataset,
                idx,
            )

            with (
                torch.no_grad(),
                torch.amp.autocast(
                    "cuda",
                    enabled=bool(
                        model.use_amp_dict[
                            "TEST"
                        ]
                    ),
                ),
            ):
                model.forward_stream_core(
                    batch
                )

            torch.cuda.synchronize()

            #
            # All compared targets remain in the same KITTI
            # tracking scene, whose calibration is shared.
            #
            # generate_prediction_dicts only needs current
            # calibration/image geometry to convert the raw
            # pseudo-lidar prediction.
            #
            for h in horizons:
                target_pos = (
                    pos + h
                )

                if (
                    target_pos
                    >=
                    n
                ):
                    continue

                target_idx = (
                    scene_indices[
                        target_pos
                    ]
                )

                with torch.no_grad():
                    pred_dicts, _ = (
                        model.postprocess_last(
                            delta_frames=float(
                                h
                            )
                        )
                    )

                torch.cuda.synchronize()

                if len(
                    pred_dicts
                ) != 1:
                    raise RuntimeError(
                        f"H{h}: expected one "
                        "prediction"
                    )

                pred = cpu_pred(
                    pred_dicts[
                        0
                    ]
                )

                annos = (
                    dataset
                    .generate_prediction_dicts(
                        batch,
                        [pred],
                        dataset.class_names,
                        output_path=None,
                    )
                )

                if len(annos) != 1:
                    raise RuntimeError(
                        "prediction conversion "
                        "failed"
                    )

                anno = annos[
                    0
                ]

                #
                # Evaluation is positional, but keep metadata
                # truthful for inspection.
                #
                (
                    _,
                    target_fid,
                    target_next,
                ) = frame_meta(
                    dataset,
                    target_idx,
                )

                anno[
                    "scene"
                ] = scene

                anno[
                    "frame_id"
                ] = target_fid

                anno[
                    "next_frame_id"
                ] = target_next

                det_by_h[
                    h
                ].append(
                    anno
                )

                gt = copy.deepcopy(
                    dataset.kitti_infos[
                        target_idx
                    ][
                        "infos"
                    ][
                        "token"
                    ][
                        "annos"
                    ]
                )

                gt_by_h[
                    h
                ].append(
                    gt
                )

                counts[
                    h
                ] += 1

            processed += 1

            if (
                processed <= 3
                or
                processed % 200 == 0
            ):
                print(
                    f"[processed] "
                    f"{processed}"
                )

    results = {}

    print()
    print("=" * 90)
    print("LASP HORIZON AP AUDIT")
    print("=" * 90)

    for h in horizons:
        if (
            len(
                gt_by_h[
                    h
                ]
            )
            !=
            len(
                det_by_h[
                    h
                ]
            )
        ):
            raise RuntimeError(
                f"H{h}: GT/det length mismatch"
            )

        if not gt_by_h[h]:
            raise RuntimeError(
                f"H{h}: no samples"
            )

        (
            _,
            ap_dict,
        ) = (
            kitti_eval
            .get_official_eval_result(
                gt_by_h[
                    h
                ],
                det_by_h[
                    h
                ],
                dataset.class_names,
            )
        )

        m = metric(
            ap_dict
        )

        results[
            str(h)
        ] = {
            "samples":
                counts[
                    h
                ],

            **m,
        }

        print(
            f"H{h} | "
            f"N={counts[h]} | "
            f"Car={m['Car']:.4f} "
            f"Ped={m['Pedestrian']:.4f} "
            f"Cyc={m['Cyclist']:.4f} "
            f"Macro={m['Macro']:.4f}"
        )

    print("=" * 90)

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            {
                "cfg":
                    args.cfg,

                "ckpt":
                    args.ckpt,

                "horizons":
                    horizons,

                "results":
                    results,
            },
            indent=2,
        )
        +
        "\n"
    )

    print(
        f"[SAVE] {out}"
    )


if __name__ == "__main__":
    main()
