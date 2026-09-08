#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import numpy as np
from easydict import EasyDict

from pcdet.config import cfg_from_yaml_file
from pcdet.datasets import __all__ as DATASETS
from pcdet.utils import box_utils


def kmeans(
    x,
    k,
    seed=1024,
    iters=100,
):

    if len(
        x
    ) < k:
        raise RuntimeError(
            f'need at least {k} '
            f'samples, got {len(x)}'
        )

    rng = np.random.default_rng(
        seed
    )

    centers = x[
        rng.choice(
            len(
                x
            ),
            size=k,
            replace=False,
        )
    ].copy()

    for _ in range(
        iters
    ):

        dist = (
            (
                x[
                    :,
                    None,
                    :
                ]
                -
                centers[
                    None,
                    :,
                    :
                ]
            )
            ** 2
        ).sum(
            axis=-1
        )

        label = dist.argmin(
            axis=1
        )

        new = centers.copy()

        for j in range(
            k
        ):

            pts = x[
                label == j
            ]

            if len(
                pts
            ):
                new[
                    j
                ] = pts.mean(
                    axis=0
                )

        if (
            np.max(
                np.abs(
                    new
                    - centers
                )
            )
            < 1e-6
        ):
            centers = new
            break

        centers = new

    #
    # deterministic cluster ordering
    #
    norm = np.linalg.norm(
        centers,
        axis=1,
    )

    angle = np.arctan2(
        centers[
            :,
            1
        ],
        centers[
            :,
            0
        ],
    )

    order = np.lexsort(
        (
            angle,
            norm,
        )
    )

    return centers[
        order
    ]


def anno_boxes(
    dataset,
    scene,
    annos,
):

    if (
        annos is None
        or len(
            annos.get(
                'name',
                [],
            )
        ) == 0
    ):

        return np.zeros(
            (
                0,
                7,
            ),
            dtype=np.float32,
        )

    loc = annos[
        'location'
    ]

    dims = annos[
        'dimensions'
    ]

    rots = annos[
        'rotation_y'
    ]

    boxes_camera = np.concatenate(
        [
            loc,
            dims,
            rots[
                :,
                None
            ],
        ],
        axis=1,
    ).astype(
        np.float32
    )

    calib = dataset.get_calib(
        scene
    )

    return (
        box_utils
        .boxes3d_kitti_camera_to_lidar(
            boxes_camera,
            calib,
            pseudo_lidar=True,
            pseudo_cam2_view=(
                getattr(
                    dataset,
                    'boxes_gt_in_cam2_view',
                    False,
                )
            ),
        )[
            :,
            :7
        ]
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--cfg',
        required=True,
    )

    ap.add_argument(
        '--horizon',
        type=int,
        default=8,
    )

    ap.add_argument(
        '--k',
        type=int,
        default=6,
    )

    ap.add_argument(
        '--seed',
        type=int,
        default=1024,
    )

    ap.add_argument(
        '--output',
        required=True,
    )

    args = ap.parse_args()

    cfg = EasyDict()

    cfg_from_yaml_file(
        args.cfg,
        cfg,
    )

    if args.horizon < 2:
        raise ValueError(
            'horizon must be >=2'
        )

    cfg.DATA_CONFIG.MTD_FUTURE_STEPS = (
        sorted(
            set(
                list(
                    cfg.DATA_CONFIG.get(
                        'MTD_FUTURE_STEPS',
                        [],
                    )
                )
                +
                [
                    args.horizon
                ]
            )
        )
    )

    logger = logging.getLogger(
        'lasp-intentions'
    )

    logger.setLevel(
        logging.INFO
    )

    logger.addHandler(
        logging.StreamHandler()
    )

    dataset = DATASETS[
        cfg.DATA_CONFIG.DATASET
    ](
        dataset_cfg=(
            cfg.DATA_CONFIG
        ),
        class_names=(
            cfg.CLASS_NAMES
        ),
        training=True,
        root_path=None,
        logger=logger,
    )

    tag = (
        f'next{args.horizon}'
    )

    displacements = {
        name: []
        for name
        in cfg.CLASS_NAMES
    }

    valid_pairs = 0

    for item in (
        dataset.kitti_infos
    ):

        scene = str(
            item[
                'sample_idx'
            ][
                'scene'
            ]
        )

        token_ann = (
            item[
                'infos'
            ][
                'token'
            ][
                'annos'
            ]
        )

        future_info = (
            item[
                'infos'
            ].get(
                tag
            )
        )

        if future_info is None:
            continue

        future_ann = (
            future_info[
                'annos'
            ]
        )

        token_boxes = anno_boxes(
            dataset,
            scene,
            token_ann,
        )

        future_boxes = anno_boxes(
            dataset,
            scene,
            future_ann,
        )

        token_ids = [
            str(
                x
            )
            for x
            in token_ann[
                'object_id'
            ]
        ]

        future_ids = [
            str(
                x
            )
            for x
            in future_ann[
                'object_id'
            ]
        ]

        future_map = {
            oid: i
            for i, oid
            in enumerate(
                future_ids
            )
        }

        for (
            i,
            oid,
        ) in enumerate(
            token_ids
        ):

            j = future_map.get(
                oid
            )

            if j is None:
                continue

            name = str(
                token_ann[
                    'name'
                ][
                    i
                ]
            )

            if (
                name
                not in displacements
            ):
                continue

            displacements[
                name
            ].append(
                future_boxes[
                    j,
                    :2
                ]
                -
                token_boxes[
                    i,
                    :2
                ]
            )

            valid_pairs += 1

    centers = []

    for (
        ci,
        name,
    ) in enumerate(
        cfg.CLASS_NAMES
    ):

        x = np.asarray(
            displacements[
                name
            ],
            dtype=np.float32,
        )

        print(
            f'{name}: '
            f'endpoint pairs={len(x)}'
        )

        c = kmeans(
            x,
            args.k,
            seed=(
                args.seed
                + ci
            ),
        )

        centers.append(
            c
        )

        print(
            c
        )

    centers = np.stack(
        centers,
        axis=0,
    ).astype(
        np.float32
    )

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        output,
        centers,
    )

    print(
        f'saved {output} '
        f'shape={centers.shape} '
        f'valid_pairs={valid_pairs}'
    )


if __name__ == '__main__':
    main()
