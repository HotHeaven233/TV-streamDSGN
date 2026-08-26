#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.utils.torch_utils import project_pseudo_lidar_to_rectcam

import profile_b2_roi_fragmentation as base


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():
    p = argparse.ArgumentParser(
        "Profile selective C-stage voxel projection"
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
            0.10, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.90, 1.00,
        ],
    )

    p.add_argument(
        "--fragments",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )

    # Use D-stage dependency halo by default.
    p.add_argument(
        "--halo",
        type=int,
        default=4,
    )

    p.add_argument(
        "--align",
        type=int,
        default=4,
    )

    p.add_argument(
        "--warmup",
        type=int,
        default=30,
    )

    p.add_argument(
        "--repeat",
        type=int,
        default=300,
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "c_roi_fragmentation.json"
        ),
    )

    return p.parse_args()


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
    Compute mapping only for one voxel BEV ROI.

    coordinates_3d:
        [Z, H, W, 3]

    roi:
        (y0, y1, x0, x1) in voxel/BEV H,W.
    """

    y0, y1, x0, x1 = roi

    coords_roi = (
        coordinates_3d[
            :,
            y0:y1,
            x0:x1,
            :
        ]
        .contiguous()
    )

    Z = coords_roi.shape[0]
    H = coords_roi.shape[1]
    W = coords_roi.shape[2]

    c3d = coords_roi.reshape(
        -1,
        3,
    )

    if random_T is not None:
        c3d = (
            torch.matmul(
                c3d,
                random_T[:3, :3].T,
            )
            + random_T[:3, 3]
        )

    c3d = (
        project_pseudo_lidar_to_rectcam(
            c3d
        )
    )

    coord_img, norm_coord_img = (
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

    coord_img = coord_img.view(
        Z,
        H,
        W,
        3,
    )

    norm_coord_img = (
        norm_coord_img.view(
            Z,
            H,
            W,
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
            norm_coord_img[..., 2]
            >= -1.0
        )
        &
        (
            norm_coord_img[..., 2]
            <= 1.0
        )
    )

    return (
        norm_coord_img.unsqueeze(0),
        valid.float().unsqueeze(0),
    )


def run_c_roi(
    backbone,
    stereo_out,
    coordinates_3d,
    roi,
    left_hw,
    image_shape,
    P2,
    random_T=None,
):
    """
    Reproduce the original C-stage AMP execution context:

        mapping
        -> grid_sample
        -> valid masking

    The original detector wraps the whole model forward in autocast,
    so compute_mapping() must also execute under autocast.
    """

    with torch.amp.autocast(
        "cuda",
        enabled=backbone.use_amp,
    ):

        grid, valid = build_mapping_roi(
            backbone,
            coordinates_3d,
            roi,
            left_hw,
            image_shape,
            P2,
            random_T=random_T,
        )

        voxel = F.grid_sample(
            stereo_out,
            grid,
            align_corners=True,
        )

        voxel = (
            voxel
            *
            valid[:, None, :, :, :]
        )

    return voxel

def main():
    args = parse_args()

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    cfg.LOCAL_RANK = 0
    cfg.MODEL.SAVE_TIME = False

    logger = common_utils.create_logger(
        rank=0
    )

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=0,
        logger=logger,
        training=False,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )

    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=False,
    )

    model.cuda().eval()

    backbone = base.find_backbone(
        model
    )

    if backbone.cat_img_feature:
        raise RuntimeError(
            "Profiler assumes cat_img_feature=False"
        )

    if backbone.cat_right_img_feature:
        raise RuntimeError(
            "Profiler assumes cat_right_img_feature=False"
        )

    # ========================================================
    # Capture:
    #   - exact stereo B output
    #   - exact C output (= D input)
    # ========================================================

    captured = {}

    def dres0_hook(
        module,
        inputs,
        output,
    ):
        captured["dres0"] = (
            output
            .detach()
            .clone()
        )

    def dres1_hook(
        module,
        inputs,
        output,
    ):
        captured["dres1"] = (
            output
            .detach()
            .clone()
        )

    def d_input_hook(
        module,
        inputs,
    ):
        captured["voxel_ref"] = (
            inputs[0]
            .detach()
            .clone()
        )

    h0 = backbone.dres0.register_forward_hook(
        dres0_hook
    )

    h1 = backbone.dres1.register_forward_hook(
        dres1_hook
    )

    hd = (
        backbone.rpn3d_convs
        .register_forward_pre_hook(
            d_input_hook
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
    hd.remove()

    stereo_out = (
        captured["dres0"]
        + captured["dres1"]
    ).contiguous()

    voxel_ref = (
        captured["voxel_ref"]
        .contiguous()
    )

    token_data = batch["token"]

    if token_data["batch_size"] != 1:
        raise RuntimeError(
            "Current profiler assumes batch_size=1"
        )

    left = token_data["left_img"]
    calib = token_data["calib"][0]

    tensor_dtype = (
        torch.float16
        if backbone.use_amp
        else torch.float32
    )

    coordinates_3d = (
        backbone.coordinates_3d
        .cuda()
    )

    if backbone.use_amp:
        coordinates_3d = (
            coordinates_3d.half()
        )

    coordinates_3d = (
        coordinates_3d
        .contiguous()
    )

    if coordinates_3d.ndim != 4:
        raise RuntimeError(
            "Unexpected coordinates_3d shape: "
            f"{tuple(coordinates_3d.shape)}"
        )

    Z, H, W, XYZ = (
        coordinates_3d.shape
    )

    if XYZ != 3:
        raise RuntimeError(
            "Last coordinates dimension must be 3"
        )

    P2 = torch.as_tensor(
        calib.P2,
        device="cuda",
        dtype=tensor_dtype,
    )

    image_shape = (
        token_data["image_shape"][0]
    )

    random_T = (
        token_data["random_T"][0]
        if "random_T" in token_data
        else None
    )

    print()
    print(
        "Stereo out:",
        tuple(stereo_out.shape),
        stereo_out.dtype,
    )

    print(
        "coordinates_3d:",
        tuple(coordinates_3d.shape),
        coordinates_3d.dtype,
    )

    print(
        "C/D reference:",
        tuple(voxel_ref.shape),
        voxel_ref.dtype,
    )

    print(
        "AMP:",
        backbone.use_amp,
    )

    # ========================================================
    # Reconstruct FULL C exactly.
    # ========================================================

    full_roi = (
        0,
        H,
        0,
        W,
    )

    with torch.no_grad():
        c_full_reconstructed = (
            run_c_roi(
                backbone,
                stereo_out,
                coordinates_3d,
                full_roi,
                left.shape[2:],
                image_shape,
                P2,
                random_T=random_T,
            )
        )

    torch.cuda.synchronize()

    full_diff = (
        c_full_reconstructed.float()
        - voxel_ref.float()
    ).abs()

    print()
    print(
        "Full C reconstruction max diff:",
        float(
            full_diff.max().item()
        ),
    )

    print(
        "Full C reconstruction mean diff:",
        float(
            full_diff.mean().item()
        ),
    )

    # Do not trust ROI measurements if exact reconstruction fails.
    if float(
        full_diff.max().item()
    ) > 1e-4:
        raise RuntimeError(
            "Full C reconstruction does not "
            "match original model forward."
        )

    # ========================================================
    # Full C latency:
    # mapping + grid_sample + valid mask
    # ========================================================

    full_stats = base.measure(
        lambda:
            run_c_roi(
                backbone,
                stereo_out,
                coordinates_3d,
                full_roi,
                left.shape[2:],
                image_shape,
                P2,
                random_T=random_T,
            ),
        args.warmup,
        args.repeat,
    )

    print()
    print("=" * 88)

    print(
        "FULL C:",
        f"mean={full_stats['mean_ms']:.3f}",
        f"p50={full_stats['p50_ms']:.3f}",
        f"p95={full_stats['p95_ms']:.3f}",
        f"p99={full_stats['p99_ms']:.3f}",
    )

    print("=" * 88)


    # ========================================================
    # Original-forward-style C timing.
    #
    # Unlike run_c_roi(), this deliberately recreates the
    # per-forward coordinate/P2 preparation done by the
    # original backbone.  This tells us how much of the old
    # ~2 ms C-stage belongs to setup rather than query compute.
    # ========================================================

    def run_c_full_original_style():

        with torch.amp.autocast(
            "cuda",
            enabled=backbone.use_amp,
        ):

            tensor_dtype_local = (
                torch.float16
                if backbone.use_amp
                else torch.float32
            )

            coords = (
                backbone.coordinates_3d
                .cuda()
            )

            if backbone.use_amp:
                coords = coords.half()

            Z0, H0, W0, _ = (
                coords.shape
            )

            c3d = coords.view(
                -1,
                3,
            )

            if random_T is not None:
                c3d = (
                    torch.matmul(
                        c3d,
                        random_T[:3, :3].T,
                    )
                    + random_T[:3, 3]
                )

            c3d = (
                project_pseudo_lidar_to_rectcam(
                    c3d
                )
            )

            P2_local = torch.as_tensor(
                calib.P2,
                device="cuda",
                dtype=tensor_dtype_local,
            )

            coord_img, grid = (
                backbone.compute_mapping(
                    c3d,
                    left.shape[2:],
                    P2_local,
                    [
                        backbone.CV_DEPTH_MIN,
                        backbone.CV_DEPTH_MAX,
                    ],
                    use_amp=backbone.use_amp,
                )
            )

            coord_img = (
                coord_img.view(
                    Z0,
                    H0,
                    W0,
                    3,
                )
            )

            grid = (
                grid.view(
                    1,
                    Z0,
                    H0,
                    W0,
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
                (grid[0, ..., 2] >= -1.0)
                &
                (grid[0, ..., 2] <= 1.0)
            ).float()

            voxel = F.grid_sample(
                stereo_out,
                grid,
                align_corners=True,
            )

            voxel = (
                voxel
                *
                valid[
                    None,
                    None,
                    ...,
                ]
            )

            return voxel


    with torch.no_grad():
        original_style_c = (
            run_c_full_original_style()
        )

    torch.cuda.synchronize()

    original_style_diff = (
        original_style_c.float()
        - voxel_ref.float()
    ).abs()

    original_style_stats = (
        base.measure(
            run_c_full_original_style,
            args.warmup,
            args.repeat,
        )
    )

    print()
    print("=" * 88)

    print(
        "FULL C original-style:",
        f"mean={original_style_stats['mean_ms']:.3f}",
        f"p50={original_style_stats['p50_ms']:.3f}",
        f"p95={original_style_stats['p95_ms']:.3f}",
        f"p99={original_style_stats['p99_ms']:.3f}",
    )

    print(
        "Original-style C max diff:",
        float(
            original_style_diff.max().item()
        ),
    )

    print("=" * 88)


    # ========================================================
    # Selective C sweep
    # ========================================================

    records = []

    print()
    print("=" * 124)

    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'ExecSum%':>10} "
        f"{'Mean':>9} "
        f"{'P50':>9} "
        f"{'P95':>9} "
        f"{'P99':>9} "
        f"{'Speedup':>9} "
        f"{'MaxDiff':>12}"
    )

    print("-" * 124)

    for ratio in args.ratios:

        for frag in args.fragments:

            semantic_rois = (
                base.make_fragmented_rois(
                    H,
                    W,
                    ratio,
                    frag,
                )
            )

            exec_rois = [
                base.expand_align(
                    r,
                    args.halo,
                    H,
                    W,
                    args.align,
                )
                for r in semantic_rois
            ]

            exec_sum_area = sum(
                base.roi_area(r)
                for r in exec_rois
            )

            exec_ratio = (
                exec_sum_area
                / float(H * W)
            )

            def run_plan():

                last = None

                for e in exec_rois:
                    last = run_c_roi(
                        backbone,
                        stereo_out,
                        coordinates_3d,
                        e,
                        left.shape[2:],
                        image_shape,
                        P2,
                        random_T=random_T,
                    )

                return last

            s = base.measure(
                run_plan,
                args.warmup,
                args.repeat,
            )

            # -----------------------------------------------
            # Exactness check against original D input.
            # -----------------------------------------------

            max_diff = 0.0

            with torch.no_grad():

                for e in exec_rois:

                    y0, y1, x0, x1 = e

                    pred = run_c_roi(
                        backbone,
                        stereo_out,
                        coordinates_3d,
                        e,
                        left.shape[2:],
                        image_shape,
                        P2,
                        random_T=random_T,
                    )

                    ref = voxel_ref[
                        ...,
                        y0:y1,
                        x0:x1
                    ]

                    diff = (
                        pred.float()
                        - ref.float()
                    ).abs()

                    max_diff = max(
                        max_diff,
                        float(
                            diff.max().item()
                        ),
                    )

            speedup = (
                full_stats["mean_ms"]
                / s["mean_ms"]
            )

            rec = {
                "ratio":
                    ratio,

                "fragments":
                    frag,

                "exec_sum_area_ratio":
                    exec_ratio,

                "latency":
                    s,

                "speedup_vs_full":
                    speedup,

                "max_abs_diff":
                    max_diff,

                "semantic_rois": [
                    list(r)
                    for r in semantic_rois
                ],

                "exec_rois": [
                    list(r)
                    for r in exec_rois
                ],
            }

            records.append(
                rec
            )

            print(
                f"{ratio*100:6.1f}% "
                f"{frag:5d} "
                f"{exec_ratio*100:9.2f}% "
                f"{s['mean_ms']:9.3f} "
                f"{s['p50_ms']:9.3f} "
                f"{s['p95_ms']:9.3f} "
                f"{s['p99_ms']:9.3f} "
                f"{speedup:9.3f} "
                f"{max_diff:12.6g}"
            )

    # ========================================================
    # Save
    # ========================================================

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
                "full_c":
                    full_stats,

                "full_reconstruction_max_diff":
                    float(
                        full_diff.max().item()
                    ),

                "stereo_out_shape":
                    list(stereo_out.shape),

                "voxel_shape":
                    list(voxel_ref.shape),

                "coordinates_shape":
                    list(coordinates_3d.shape),

                "halo":
                    args.halo,

                "align":
                    args.align,

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
