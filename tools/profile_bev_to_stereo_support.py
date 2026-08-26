#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.torch_utils import (
    project_pseudo_lidar_to_rectcam,
)

import profile_b2_roi_fragmentation as base
import eval_b2_packing_oracle as packing


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():

    p = argparse.ArgumentParser(
        "Profile BEV -> stereo dependency completion"
    )

    p.add_argument(
        "--cfg_file",
        default=DEFAULT_CFG,
    )

    p.add_argument(
        "--ckpt",
        default=DEFAULT_CKPT,
    )

    p.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[
            0.10,
            0.30,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.00,
        ],
    )

    p.add_argument(
        "--fragments",
        nargs="+",
        type=int,
        default=[
            1,
            2,
            4,
            8,
            16,
        ],
    )

    # Verified C+D dependency halo.
    p.add_argument(
        "--cd-halo",
        type=int,
        default=12,
    )

    p.add_argument(
        "--cd-align",
        type=int,
        default=4,
    )

    # Verified B2 spatial dependency.
    p.add_argument(
        "--b-halo",
        type=int,
        default=1,
    )

    p.add_argument(
        "--b-align",
        type=int,
        default=4,
    )

    p.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify that keeping only Omega_B support "
            "reconstructs the requested C output."
        ),
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "bev_to_stereo_support.json"
        ),
    )

    return p.parse_args()


def roi_union_ratio(
    rois,
    H,
    W,
):
    if len(rois) == 0:
        return 0.0

    mask = np.zeros(
        (H, W),
        dtype=np.bool_,
    )

    for y0, y1, x0, x1 in rois:
        mask[
            y0:y1,
            x0:x1
        ] = True

    return float(mask.mean())


def make_physical_rois(
    semantic_rois,
    strategy,
    halo,
    H,
    W,
    align,
):

    if strategy == "naive":
        return list(
            semantic_rois
        )

    if strategy == "cap4":
        return packing.area_greedy_to_cap(
            semantic_rois,
            min(
                4,
                len(semantic_rois),
            ),
            halo,
            H,
            W,
            align,
        )

    if strategy == "cap2":
        return packing.area_greedy_to_cap(
            semantic_rois,
            min(
                2,
                len(semantic_rois),
            ),
            halo,
            H,
            W,
            align,
        )

    if strategy == "bbox1":

        if len(semantic_rois) == 0:
            return []

        return [
            packing.union_bbox(
                semantic_rois
            )
        ]

    raise ValueError(
        strategy
    )


def component_bboxes(
    mask,
):
    """
    Offline geometry oracle only.

    8-connected components on the small
    stereo spatial mask (80 x 312).

    This CPU implementation is NOT intended
    for the final online runtime path.
    """

    H, W = mask.shape

    visited = np.zeros_like(
        mask,
        dtype=np.bool_,
    )

    boxes = []

    neigh = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    ys, xs = np.nonzero(
        mask
    )

    for sy, sx in zip(
        ys.tolist(),
        xs.tolist(),
    ):

        if visited[sy, sx]:
            continue

        visited[sy, sx] = True

        stack = [
            (sy, sx)
        ]

        ymin = sy
        ymax = sy
        xmin = sx
        xmax = sx

        while stack:

            y, x = stack.pop()

            ymin = min(
                ymin,
                y,
            )
            ymax = max(
                ymax,
                y,
            )
            xmin = min(
                xmin,
                x,
            )
            xmax = max(
                xmax,
                x,
            )

            for dy, dx in neigh:

                yy = y + dy
                xx = x + dx

                if (
                    yy < 0
                    or yy >= H
                    or xx < 0
                    or xx >= W
                ):
                    continue

                if (
                    mask[yy, xx]
                    and not visited[
                        yy,
                        xx
                    ]
                ):
                    visited[
                        yy,
                        xx
                    ] = True

                    stack.append(
                        (
                            yy,
                            xx,
                        )
                    )

        boxes.append(
            (
                ymin,
                ymax + 1,
                xmin,
                xmax + 1,
            )
        )

    return boxes


def build_mapping_roi(
    backbone,
    coordinates_3d,
    roi,
    left_hw,
    image_shape,
    P2,
    random_T=None,
):
    """
    Reproduce the original C mapping for one
    BEV ROI.

    roi is half-open:
        [y0:y1, x0:x1]
    """

    y0, y1, x0, x1 = roi

    coords = coordinates_3d[
        :,
        y0:y1,
        x0:x1,
        :,
    ]

    Z, h, w, _ = (
        coords.shape
    )

    with torch.amp.autocast(
        "cuda",
        enabled=backbone.use_amp,
    ):

        c3d = coords.reshape(
            -1,
            3,
        )

        if random_T is not None:

            c3d = (
                torch.matmul(
                    c3d,
                    random_T[
                        :3,
                        :3,
                    ].T,
                )
                +
                random_T[
                    :3,
                    3,
                ]
            )

        c3d = (
            project_pseudo_lidar_to_rectcam(
                c3d
            )
        )

        coord_img, grid = (
            backbone.compute_mapping(
                c3d,
                left_hw,
                P2,
                [
                    backbone.CV_DEPTH_MIN,
                    backbone.CV_DEPTH_MAX,
                ],
                use_amp=backbone.use_amp,
            )
        )

        coord_img = (
            coord_img.view(
                Z,
                h,
                w,
                3,
            )
        )

        grid = (
            grid.view(
                1,
                Z,
                h,
                w,
                3,
            )
        )

        valid2d = (
            (coord_img[..., 0] >= 0)
            &
            (
                coord_img[..., 0]
                <= image_shape[1]
            )
            &
            (coord_img[..., 1] >= 0)
            &
            (
                coord_img[..., 1]
                <= image_shape[0]
            )
        )

        valid = (
            valid2d
            &
            (
                grid[
                    0,
                    ...,
                    2,
                ] >= -1.0
            )
            &
            (
                grid[
                    0,
                    ...,
                    2,
                ] <= 1.0
            )
        )

    return (
        grid,
        valid,
    )


def omega_from_grid(
    grid,
    valid,
    Hs,
    Ws,
):
    """
    Exact/conservative spatial support for
    align_corners=True trilinear grid_sample.

    B computes the complete disparity/depth
    dimension, therefore Omega only tracks
    the H/W support required from stereo_out.
    """

    gx = (
        grid[
            0,
            ...,
            0,
        ]
        .float()
    )

    gy = (
        grid[
            0,
            ...,
            1,
        ]
        .float()
    )

    # align_corners=True:
    #
    # x = ((gx + 1) / 2) * (W - 1)
    # y = ((gy + 1) / 2) * (H - 1)

    x = (
        (gx + 1.0)
        * 0.5
        * float(
            Ws - 1
        )
    )

    y = (
        (gy + 1.0)
        * 0.5
        * float(
            Hs - 1
        )
    )

    x0 = torch.floor(
        x
    ).long()

    y0 = torch.floor(
        y
    ).long()

    x1 = x0 + 1
    y1 = y0 + 1

    omega = torch.zeros(
        (
            Hs,
            Ws,
        ),
        dtype=torch.bool,
        device=grid.device,
    )

    flat = omega.view(
        -1
    )

    # Trilinear interpolation in D/H/W.
    #
    # D is always fully computed by the B ROI
    # executor, so only enumerate the 2x2 H/W
    # spatial interpolation neighbors here.

    for yy in (
        y0,
        y1,
    ):

        for xx in (
            x0,
            x1,
        ):

            m = (
                valid
                &
                (yy >= 0)
                &
                (yy < Hs)
                &
                (xx >= 0)
                &
                (xx < Ws)
            )

            if m.any():

                idx = (
                    yy[m]
                    * Ws
                    + xx[m]
                )

                flat[idx] = True

    return omega


def make_b_plan(
    component_boxes_list,
    strategy,
    halo,
    H,
    W,
    align,
):

    if len(
        component_boxes_list
    ) == 0:
        return []

    if strategy == "keep":

        physical = list(
            component_boxes_list
        )

    elif strategy == "cap4":

        physical = (
            packing.area_greedy_to_cap(
                component_boxes_list,
                min(
                    4,
                    len(
                        component_boxes_list
                    ),
                ),
                halo,
                H,
                W,
                align,
            )
        )

    elif strategy == "cap2":

        physical = (
            packing.area_greedy_to_cap(
                component_boxes_list,
                min(
                    2,
                    len(
                        component_boxes_list
                    ),
                ),
                halo,
                H,
                W,
                align,
            )
        )

    elif strategy == "bbox1":

        physical = [
            packing.union_bbox(
                component_boxes_list
            )
        ]

    else:
        raise ValueError(
            strategy
        )

    exec_rois = [
        base.expand_align(
            r,
            halo,
            H,
            W,
            align,
        )
        for r in physical
    ]

    return exec_rois



def tile_rect_cover(
    mask,
    tile_h,
    tile_w,
):
    """
    Offline rectangular cover of an irregular stereo support mask.

    1. Mark a coarse tile active if it contains at least one
       required Omega cell.
    2. Merge horizontally adjacent active tiles into runs.
    3. Merge vertically adjacent runs when their x interval
       is identical.

    Returned rectangles are half-open pixel coordinates:
        (y0, y1, x0, x1)

    This is an offline geometry oracle, not the final online
    implementation.
    """

    H, W = mask.shape

    nty = (
        H + tile_h - 1
    ) // tile_h

    ntx = (
        W + tile_w - 1
    ) // tile_w

    active = np.zeros(
        (nty, ntx),
        dtype=np.bool_,
    )

    for ty in range(nty):

        y0 = ty * tile_h
        y1 = min(
            H,
            (ty + 1) * tile_h,
        )

        for tx in range(ntx):

            x0 = tx * tile_w
            x1 = min(
                W,
                (tx + 1) * tile_w,
            )

            if mask[
                y0:y1,
                x0:x1
            ].any():

                active[
                    ty,
                    tx
                ] = True

    row_runs = []

    for ty in range(nty):

        runs = []

        tx = 0

        while tx < ntx:

            if not active[
                ty,
                tx
            ]:
                tx += 1
                continue

            x_begin = tx

            while (
                tx + 1 < ntx
                and active[
                    ty,
                    tx + 1
                ]
            ):
                tx += 1

            x_end = tx + 1

            runs.append(
                (
                    x_begin,
                    x_end,
                )
            )

            tx += 1

        row_runs.append(
            runs
        )

    completed = []
    open_rects = {}

    for ty, runs in enumerate(
        row_runs
    ):

        next_open = {}

        y0 = ty * tile_h

        y1 = min(
            H,
            (ty + 1) * tile_h,
        )

        for x_begin, x_end in runs:

            key = (
                x_begin,
                x_end,
            )

            if key in open_rects:

                rect = list(
                    open_rects.pop(
                        key
                    )
                )

                rect[1] = y1

            else:

                rect = [
                    y0,
                    y1,
                    x_begin * tile_w,
                    min(
                        W,
                        x_end * tile_w,
                    ),
                ]

            next_open[
                key
            ] = tuple(
                rect
            )

        completed.extend(
            open_rects.values()
        )

        open_rects = (
            next_open
        )

    completed.extend(
        open_rects.values()
    )

    return list(
        completed
    )



def main():

    args = parse_args()

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    cfg.LOCAL_RANK = 0
    cfg.MODEL.SAVE_TIME = False

    logger = (
        common_utils.create_logger(
            rank=0
        )
    )

    dataset, _, _ = (
        build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=1,
            dist=False,
            workers=0,
            logger=logger,
            training=False,
        )
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
        to_cpu=False,
    )

    model.cuda().eval()

    backbone = (
        base.find_backbone(
            model
        )
    )

    captured = {}

    def hook_dres0(
        module,
        inputs,
        output,
    ):
        captured["dres0"] = (
            output
            .detach()
            .clone()
        )

    def hook_dres1(
        module,
        inputs,
        output,
    ):
        captured["dres1"] = (
            output
            .detach()
            .clone()
        )

    h0 = (
        backbone.dres0
        .register_forward_hook(
            hook_dres0
        )
    )

    h1 = (
        backbone.dres1
        .register_forward_hook(
            hook_dres1
        )
    )

    sample = dataset[0]

    batch = dataset.collate_batch(
        [sample]
    )

    load_data_to_gpu(
        batch
    )

    with torch.no_grad():
        model(batch)

    torch.cuda.synchronize()

    h0.remove()
    h1.remove()

    stereo_out = (
        captured["dres0"]
        +
        captured["dres1"]
    ).contiguous()

    token = batch["token"]

    if token["batch_size"] != 1:
        raise RuntimeError(
            "Profiler assumes batch_size=1"
        )

    left = token[
        "left_img"
    ]

    calib = token[
        "calib"
    ][0]

    image_shape = token[
        "image_shape"
    ][0]

    random_T = (
        token["random_T"][0]
        if "random_T" in token
        else None
    )

    coordinates_3d = (
        backbone.coordinates_3d
        .cuda()
    )

    if backbone.use_amp:
        coordinates_3d = (
            coordinates_3d.half()
        )

    dtype = (
        torch.float16
        if backbone.use_amp
        else torch.float32
    )

    P2 = torch.as_tensor(
        calib.P2,
        device="cuda",
        dtype=dtype,
    )

    _, _, Ds, Hs, Ws = (
        stereo_out.shape
    )

    Z, Hb, Wb, _ = (
        coordinates_3d.shape
    )

    print()
    print(
        "Stereo volume:",
        tuple(
            stereo_out.shape
        ),
        stereo_out.dtype,
    )

    print(
        "BEV/C coordinates:",
        tuple(
            coordinates_3d.shape
        ),
        coordinates_3d.dtype,
    )

    print(
        "Stereo spatial:",
        f"H={Hs}",
        f"W={Ws}",
        f"D={Ds}",
    )

    print(
        "BEV spatial:",
        f"H={Hb}",
        f"W={Wb}",
        f"Z={Z}",
    )

    print(
        "C+D halo/align:",
        args.cd_halo,
        args.cd_align,
    )

    print(
        "B halo/align:",
        args.b_halo,
        args.b_align,
    )

    strategies_cd = [
        "naive",
        "cap4",
        "cap2",
        "bbox1",
    ]

    records = []

    print()
    print("=" * 190)

    print(
        f"{'R%':>6} "
        f"{'Frag':>5} "
        f"{'CD':>7} "
        f"{'CDROI':>5} "
        f"{'CDExec%':>9} "
        f"{'Omega%':>8} "
        f"{'NComp':>6} "
        f"{'CompBox%':>9} "
        f"{'Bkeep%':>8} "
        f"{'Bcap4%':>8} "
        f"{'Bcap2%':>8} "
        f"{'Bbbox%':>8} "
        f"{'SuppDiff':>10}"
    )

    print("-" * 190)

    for ratio in args.ratios:

        for frag in args.fragments:

            semantic_rois = (
                base.make_fragmented_rois(
                    Hb,
                    Wb,
                    ratio,
                    frag,
                )
            )

            semantic_ratio = (
                roi_union_ratio(
                    semantic_rois,
                    Hb,
                    Wb,
                )
            )

            for cd_strategy in strategies_cd:

                physical_rois = (
                    make_physical_rois(
                        semantic_rois,
                        cd_strategy,
                        args.cd_halo,
                        Hb,
                        Wb,
                        args.cd_align,
                    )
                )

                cd_exec_rois = [
                    base.expand_align(
                        r,
                        args.cd_halo,
                        Hb,
                        Wb,
                        args.cd_align,
                    )
                    for r in physical_rois
                ]

                cd_exec_ratio = (
                    roi_union_ratio(
                        cd_exec_rois,
                        Hb,
                        Wb,
                    )
                )

                omega = torch.zeros(
                    (
                        Hs,
                        Ws,
                    ),
                    dtype=torch.bool,
                    device="cuda",
                )

                mapping_items = []

                for exec_roi in cd_exec_rois:

                    grid, valid = (
                        build_mapping_roi(
                            backbone,
                            coordinates_3d,
                            exec_roi,
                            left.shape[2:],
                            image_shape,
                            P2,
                            random_T=random_T,
                        )
                    )

                    cur_omega = (
                        omega_from_grid(
                            grid,
                            valid,
                            Hs,
                            Ws,
                        )
                    )

                    omega |= cur_omega

                    if args.check:
                        mapping_items.append(
                            (
                                grid,
                                valid,
                            )
                        )

                omega_ratio = (
                    float(
                        omega.sum().item()
                    )
                    /
                    float(
                        Hs * Ws
                    )
                )

                omega_cpu = (
                    omega
                    .detach()
                    .cpu()
                    .numpy()
                )

                comp_boxes = (
                    component_bboxes(
                        omega_cpu
                    )
                )

                # ------------------------------------------------
                # Alternative rectangular covers for irregular
                # Omega_B.  Component bounding boxes can cause
                # severe over-computation for perspective-shaped
                # or fragmented stereo support.
                # ------------------------------------------------

                tile_shapes = [
                    (4, 8),
                    (4, 16),
                    (8, 8),
                    (8, 16),
                    (8, 24),
                    (12, 24),
                    (16, 32),
                ]

                tile_cover_stats = []

                for tile_h, tile_w in tile_shapes:

                    tile_rects = (
                        tile_rect_cover(
                            omega_cpu,
                            tile_h,
                            tile_w,
                        )
                    )

                    tile_cover_ratio = (
                        roi_union_ratio(
                            tile_rects,
                            Hs,
                            Ws,
                        )
                    )

                    plan_stats = {}

                    for tile_strategy in [
                        "keep",
                        "cap4",
                        "cap2",
                        "bbox1",
                    ]:

                        tile_exec = (
                            make_b_plan(
                                tile_rects,
                                tile_strategy,
                                args.b_halo,
                                Hs,
                                Ws,
                                args.b_align,
                            )
                        )

                        tile_exec_ratio = (
                            roi_union_ratio(
                                tile_exec,
                                Hs,
                                Ws,
                            )
                        )

                        plan_stats[
                            tile_strategy
                        ] = {
                            "num_exec_rois":
                                len(
                                    tile_exec
                                ),

                            "exec_ratio":
                                tile_exec_ratio,

                            "exec_rois": [
                                list(r)
                                for r in tile_exec
                            ],
                        }

                    tile_cover_stats.append(
                        {
                            "tile_h":
                                tile_h,

                            "tile_w":
                                tile_w,

                            "num_cover_rects":
                                len(
                                    tile_rects
                                ),

                            "cover_ratio":
                                tile_cover_ratio,

                            "cover_rois": [
                                list(r)
                                for r in tile_rects
                            ],

                            "plans":
                                plan_stats,
                        }
                    )

                comp_box_ratio = (
                    roi_union_ratio(
                        comp_boxes,
                        Hs,
                        Ws,
                    )
                )

                b_exec_ratios = {}

                for b_strategy in [
                    "keep",
                    "cap4",
                    "cap2",
                    "bbox1",
                ]:

                    b_exec = (
                        make_b_plan(
                            comp_boxes,
                            b_strategy,
                            args.b_halo,
                            Hs,
                            Ws,
                            args.b_align,
                        )
                    )

                    b_exec_ratios[
                        b_strategy
                    ] = (
                        roi_union_ratio(
                            b_exec,
                            Hs,
                            Ws,
                        )
                    )

                support_diff = 0.0

                if (
                    args.check
                    and len(
                        mapping_items
                    ) > 0
                ):

                    with torch.no_grad():

                        omega_float = (
                            omega[
                                None,
                                None,
                                None,
                                :,
                                :,
                            ]
                            .to(
                                dtype=stereo_out.dtype
                            )
                        )

                        masked_stereo = (
                            stereo_out
                            *
                            omega_float
                        )

                        for (
                            grid,
                            valid,
                        ) in mapping_items:

                            with torch.amp.autocast(
                                "cuda",
                                enabled=backbone.use_amp,
                            ):

                                full_c = (
                                    F.grid_sample(
                                        stereo_out,
                                        grid,
                                        align_corners=True,
                                    )
                                    *
                                    valid[
                                        None,
                                        None,
                                        ...,
                                    ].float()
                                )

                                omega_c = (
                                    F.grid_sample(
                                        masked_stereo,
                                        grid,
                                        align_corners=True,
                                    )
                                    *
                                    valid[
                                        None,
                                        None,
                                        ...,
                                    ].float()
                                )

                            diff = (
                                full_c.float()
                                -
                                omega_c.float()
                            ).abs()

                            support_diff = max(
                                support_diff,
                                float(
                                    diff.max().item()
                                ),
                            )

                rec = {
                    "target_ratio":
                        ratio,

                    "semantic_ratio":
                        semantic_ratio,

                    "fragments":
                        frag,

                    "cd_strategy":
                        cd_strategy,

                    "num_cd_rois":
                        len(
                            physical_rois
                        ),

                    "semantic_rois": [
                        list(r)
                        for r in semantic_rois
                    ],

                    "cd_physical_rois": [
                        list(r)
                        for r in physical_rois
                    ],

                    "cd_exec_rois": [
                        list(r)
                        for r in cd_exec_rois
                    ],

                    "cd_exec_ratio":
                        cd_exec_ratio,

                    "omega_ratio":
                        omega_ratio,

                    "omega_num_components":
                        len(
                            comp_boxes
                        ),

                    "omega_component_boxes": [
                        list(r)
                        for r in comp_boxes
                    ],

                    "component_bbox_union_ratio":
                        comp_box_ratio,

                    "b_exec_keep_ratio":
                        b_exec_ratios[
                            "keep"
                        ],

                    "b_exec_cap4_ratio":
                        b_exec_ratios[
                            "cap4"
                        ],

                    "b_exec_cap2_ratio":
                        b_exec_ratios[
                            "cap2"
                        ],

                    "b_exec_bbox1_ratio":
                        b_exec_ratios[
                            "bbox1"
                        ],

                    "support_max_diff":
                        support_diff,

                    "tile_cover_stats":
                        tile_cover_stats,
                }

                records.append(
                    rec
                )

                print(
                    f"{semantic_ratio*100:5.1f}% "
                    f"{frag:5d} "
                    f"{cd_strategy:>7} "
                    f"{len(physical_rois):5d} "
                    f"{cd_exec_ratio*100:8.2f}% "
                    f"{omega_ratio*100:7.2f}% "
                    f"{len(comp_boxes):6d} "
                    f"{comp_box_ratio*100:8.2f}% "
                    f"{b_exec_ratios['keep']*100:7.2f}% "
                    f"{b_exec_ratios['cap4']*100:7.2f}% "
                    f"{b_exec_ratios['cap2']*100:7.2f}% "
                    f"{b_exec_ratios['bbox1']*100:7.2f}% "
                    f"{support_diff:10.6g}"
                )

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
                "cd_halo":
                    args.cd_halo,

                "cd_align":
                    args.cd_align,

                "b_halo":
                    args.b_halo,

                "b_align":
                    args.b_align,

                "stereo_shape":
                    list(
                        stereo_out.shape
                    ),

                "coordinates_shape":
                    list(
                        coordinates_3d.shape
                    ),

                "records":
                    records,
            },
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        out,
    )


if __name__ == "__main__":
    main()
