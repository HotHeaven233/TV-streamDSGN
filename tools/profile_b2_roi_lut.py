#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

import profile_b2_roi_fragmentation as base


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():
    p = argparse.ArgumentParser(
        "Profile B2 ROI latency shape LUT"
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
        "--halo",
        type=int,
        default=1,
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
        default=150,
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "b2_roi_latency_lut.json"
        ),
    )

    return p.parse_args()


def aligned_sizes(
    full,
    align,
):
    """
    Dense enough sampling in middle/high ROI ranges.

    For B2, shape matters much more than simple area,
    so we intentionally keep a reasonably dense grid.
    """

    ratios = [
        0.10,
        0.125,
        0.1875,
        0.25,
        0.3125,
        0.375,
        0.4375,
        0.50,
        0.5625,
        0.625,
        0.6875,
        0.75,
        0.8125,
        0.875,
        0.9375,
        1.0,
    ]

    values = set()

    for r in ratios:

        v = int(
            round(
                full * r
            )
        )

        v = max(
            align,
            (v // align) * align,
        )

        v = min(
            full,
            v,
        )

        values.add(v)

    values.add(full)

    return sorted(values)


def stats(values):

    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean_ms":
            float(x.mean()),

        "p50_ms":
            float(
                np.percentile(
                    x,
                    50,
                )
            ),

        "p95_ms":
            float(
                np.percentile(
                    x,
                    95,
                )
            ),

        "p99_ms":
            float(
                np.percentile(
                    x,
                    99,
                )
            ),
    }


def measure(
    fn,
    warmup,
    repeat,
):

    with torch.no_grad():

        for _ in range(
            warmup
        ):
            fn()

    torch.cuda.synchronize()

    values = []

    with torch.no_grad():

        for _ in range(
            repeat
        ):

            st = torch.cuda.Event(
                enable_timing=True
            )

            ed = torch.cuda.Event(
                enable_timing=True
            )

            st.record()

            fn()

            ed.record()

            ed.synchronize()

            values.append(
                float(
                    st.elapsed_time(
                        ed
                    )
                )
            )

    return stats(
        values
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
            dataset_cfg=
                cfg.DATA_CONFIG,

            class_names=
                cfg.CLASS_NAMES,

            batch_size=1,
            dist=False,
            workers=0,
            logger=logger,
            training=False,
        )
    )

    model = build_network(
        model_cfg=
            cfg.MODEL,

        num_class=
            len(
                cfg.CLASS_NAMES
            ),

        dataset=
            dataset,
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

    if backbone.num_hg != 0:
        raise RuntimeError(
            "This profiler assumes "
            f"num_hg == 0, got {backbone.num_hg}"
        )

    # ========================================================
    # Capture real cost-volume output
    # ========================================================

    captured = {}

    def hook(
        module,
        inputs,
    ):

        captured["x"] = (
            inputs[0]
            .detach()
            .clone()
        )

    handle = (
        backbone.dres0
        .register_forward_pre_hook(
            hook
        )
    )

    sample = dataset[0]

    batch = (
        dataset.collate_batch(
            [sample]
        )
    )

    load_data_to_gpu(
        batch
    )

    with torch.no_grad():
        model(batch)

    torch.cuda.synchronize()

    handle.remove()

    x_full = (
        captured["x"]
        .contiguous()
    )

    _, C, D, H, W = (
        x_full.shape
    )

    print()
    print(
        "B2 input:",
        tuple(
            x_full.shape
        ),
        x_full.dtype,
    )

    print(
        "AMP:",
        bool(
            getattr(
                backbone,
                "use_amp",
                False,
            )
        ),
    )

    # ========================================================
    # Full reference
    # ========================================================

    with torch.no_grad():

        full_output = (
            base.run_b2(
                backbone,
                x_full,
            )
        )

    torch.cuda.synchronize()

    full_stats = measure(
        lambda:
            base.run_b2(
                backbone,
                x_full,
            ),

        args.warmup,
        args.repeat,
    )

    print()
    print(
        "Full B2:",
        f"mean={full_stats['mean_ms']:.3f}",
        f"p95={full_stats['p95_ms']:.3f}",
        f"p99={full_stats['p99_ms']:.3f}",
    )

    # ========================================================
    # Semantic core shape candidates
    # ========================================================

    hs = aligned_sizes(
        H,
        args.align,
    )

    ws = aligned_sizes(
        W,
        args.align,
    )

    print()
    print(
        f"H candidates ({len(hs)}):",
        hs,
    )

    print(
        f"W candidates ({len(ws)}):",
        ws,
    )

    print(
        "Total shapes:",
        len(hs) * len(ws),
    )

    records = []

    print()
    print("=" * 120)

    print(
        f"{'CoreH':>6} "
        f"{'CoreW':>6} "
        f"{'ExecH':>6} "
        f"{'ExecW':>6} "
        f"{'Core%':>8} "
        f"{'Exec%':>8} "
        f"{'Aspect':>8} "
        f"{'Mean':>9} "
        f"{'P95':>9} "
        f"{'P99':>9}"
    )

    print("-" * 120)

    for core_h in hs:

        for core_w in ws:

            # -----------------------------------------------
            # Central semantic ROI
            # -----------------------------------------------

            cy0 = (
                H - core_h
            ) // 2

            cx0 = (
                W - core_w
            ) // 2

            cy1 = (
                cy0 + core_h
            )

            cx1 = (
                cx0 + core_w
            )

            core_roi = (
                cy0,
                cy1,
                cx0,
                cx1,
            )

            exec_roi = (
                base.expand_align(
                    core_roi,
                    args.halo,
                    H,
                    W,
                    args.align,
                )
            )

            (
                ey0,
                ey1,
                ex0,
                ex1,
            ) = exec_roi

            exec_h = (
                ey1 - ey0
            )

            exec_w = (
                ex1 - ex0
            )

            ry0 = (
                cy0 - ey0
            )

            rx0 = (
                cx0 - ex0
            )

            # Full-size destination so the measured write
            # matches online composition behavior.
            canvas = (
                torch.empty_like(
                    full_output
                )
            )

            def fn():

                crop = (
                    x_full[
                        ...,
                        ey0:ey1,
                        ex0:ex1
                    ]
                    .contiguous()
                )

                y = (
                    base.run_b2(
                        backbone,
                        crop,
                    )
                )

                canvas[
                    ...,
                    cy0:cy1,
                    cx0:cx1
                ].copy_(
                    y[
                        ...,
                        ry0:
                            ry0 + core_h,
                        rx0:
                            rx0 + core_w
                    ]
                )

                return canvas

            s = measure(
                fn,
                args.warmup,
                args.repeat,
            )

            core_ratio = (
                core_h
                * core_w
                / float(
                    H * W
                )
            )

            exec_ratio = (
                exec_h
                * exec_w
                / float(
                    H * W
                )
            )

            aspect = (
                core_w
                / float(
                    core_h
                )
            )

            row = {
                "core_h":
                    int(core_h),

                "core_w":
                    int(core_w),

                "exec_h":
                    int(exec_h),

                "exec_w":
                    int(exec_w),

                "core_area_ratio":
                    float(
                        core_ratio
                    ),

                "exec_area_ratio":
                    float(
                        exec_ratio
                    ),

                "aspect_ratio":
                    float(
                        aspect
                    ),

                **s,
            }

            records.append(
                row
            )

            print(
                f"{core_h:6d} "
                f"{core_w:6d} "
                f"{exec_h:6d} "
                f"{exec_w:6d} "
                f"{core_ratio*100:8.2f} "
                f"{exec_ratio*100:8.2f} "
                f"{aspect:8.3f} "
                f"{s['mean_ms']:9.3f} "
                f"{s['p95_ms']:9.3f} "
                f"{s['p99_ms']:9.3f}"
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
                "gpu":
                    torch.cuda.get_device_name(
                        0
                    ),

                "input_shape":
                    list(
                        x_full.shape
                    ),

                "halo":
                    args.halo,

                "align":
                    args.align,

                "warmup":
                    args.warmup,

                "repeat":
                    args.repeat,

                "full_b2":
                    full_stats,

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
