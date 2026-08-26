#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl_5090_eval.yaml"
)

DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


def parse_args():
    p = argparse.ArgumentParser(
        "B2 stereo dres0+dres1 ROI fragmentation profiler"
    )

    p.add_argument("--cfg_file", default=DEFAULT_CFG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)

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

    # dres1 is 3x3x3, so spatial radius starts from 1.
    p.add_argument("--halo", type=int, default=1)

    # Hardware-friendly H/W alignment.
    p.add_argument("--align", type=int, default=4)

    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeat", type=int, default=500)

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "b2_roi_fragmentation.json"
        ),
    )

    return p.parse_args()


def find_backbone(model):
    for m in model.modules():
        if m.__class__.__name__ == "StreamDSGN2Backbone":
            return m

    raise RuntimeError(
        "StreamDSGN2Backbone not found"
    )


def stats(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(x.mean()),
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
    }


def measure(fn, warmup, repeat):
    with torch.no_grad():
        for _ in range(warmup):
            fn()

    torch.cuda.synchronize()

    values = []

    with torch.no_grad():
        for _ in range(repeat):

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
                    st.elapsed_time(ed)
                )
            )

    return stats(values)


def fragmentation_grid(n):
    known = {
        1: (1, 1),
        2: (1, 2),
        4: (2, 2),
        8: (2, 4),
        16: (4, 4),
    }

    if n in known:
        return known[n]

    gh = max(
        1,
        int(math.sqrt(n)),
    )

    while n % gh != 0:
        gh -= 1

    return gh, n // gh


def make_fragmented_rois(
    H,
    W,
    ratio,
    n,
):
    """
    Construct n separated semantic ROIs with approximately
    the requested total useful area ratio.
    """

    gh, gw = fragmentation_grid(n)

    scale = math.sqrt(ratio)

    rois = []

    for iy in range(gh):

        sy0 = round(
            iy * H / gh
        )

        sy1 = round(
            (iy + 1) * H / gh
        )

        sh = sy1 - sy0

        for ix in range(gw):

            sx0 = round(
                ix * W / gw
            )

            sx1 = round(
                (ix + 1) * W / gw
            )

            sw = sx1 - sx0

            rh = max(
                1,
                round(sh * scale),
            )

            rw = max(
                1,
                round(sw * scale),
            )

            rh = min(rh, sh)
            rw = min(rw, sw)

            y0 = (
                sy0
                + (sh - rh) // 2
            )

            x0 = (
                sx0
                + (sw - rw) // 2
            )

            rois.append(
                (
                    y0,
                    y0 + rh,
                    x0,
                    x0 + rw,
                )
            )

    return rois


def expand_align(
    roi,
    halo,
    H,
    W,
    align,
):
    y0, y1, x0, x1 = roi

    y0 = max(
        0,
        y0 - halo,
    )

    x0 = max(
        0,
        x0 - halo,
    )

    y1 = min(
        H,
        y1 + halo,
    )

    x1 = min(
        W,
        x1 + halo,
    )

    y0 = (
        y0 // align
    ) * align

    x0 = (
        x0 // align
    ) * align

    y1 = min(
        H,
        int(
            math.ceil(
                y1 / align
            )
            * align
        ),
    )

    x1 = min(
        W,
        int(
            math.ceil(
                x1 / align
            )
            * align
        ),
    )

    return (
        y0,
        y1,
        x0,
        x1,
    )


def roi_area(r):
    y0, y1, x0, x1 = r

    return (
        max(0, y1 - y0)
        * max(0, x1 - x0)
    )


def unique_area(
    rois,
    H,
    W,
):
    """
    Offline statistic only.
    This is NOT included in measured GPU latency.
    """

    mask = np.zeros(
        (H, W),
        dtype=np.bool_,
    )

    for (
        y0,
        y1,
        x0,
        x1,
    ) in rois:

        mask[
            y0:y1,
            x0:x1
        ] = True

    return int(
        mask.sum()
    )


def run_dres0(
    backbone,
    x,
):
    use_amp = bool(
        getattr(
            backbone,
            "use_amp",
            True,
        )
    )

    with torch.amp.autocast(
        "cuda",
        enabled=use_amp,
    ):
        return backbone.dres0(x)


def run_dres1(
    backbone,
    x,
):
    use_amp = bool(
        getattr(
            backbone,
            "use_amp",
            True,
        )
    )

    with torch.amp.autocast(
        "cuda",
        enabled=use_amp,
    ):
        return backbone.dres1(x)


def run_b2(
    backbone,
    x,
):
    """
    Current config has num_hg == 0.

    Stereo processing after cost-volume construction:

        dres0
          ->
        dres1 + residual

    We include the residual addition.
    """

    use_amp = bool(
        getattr(
            backbone,
            "use_amp",
            True,
        )
    )

    with torch.amp.autocast(
        "cuda",
        enabled=use_amp,
    ):

        y0 = backbone.dres0(x)

        y1 = backbone.dres1(y0)

        y = y1 + y0

    return y


def make_roi_exec_fn(
    backbone,
    full_input,
    exec_rois,
):
    """
    Physical ROI path.

    Crop + contiguous are deliberately inside the timed path
    so this includes actual selective-execution overhead.
    """

    def run():

        last = None

        for (
            y0,
            y1,
            x0,
            x1,
        ) in exec_rois:

            crop = (
                full_input[
                    ...,
                    y0:y1,
                    x0:x1
                ]
                .contiguous()
            )

            last = run_b2(
                backbone,
                crop,
            )

        return last

    return run


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

    backbone = find_backbone(
        model
    )

    print(
        "backbone.num_hg =",
        backbone.num_hg,
    )

    if backbone.num_hg != 0:
        raise RuntimeError(
            "This profiler assumes num_hg == 0. "
            f"Current num_hg={backbone.num_hg}"
        )

    # ------------------------------------------------------
    # Capture true inputs from an actual model forward
    # ------------------------------------------------------

    captured = {}

    def hook_dres0(
        module,
        inputs,
    ):
        captured[
            "dres0_in"
        ] = (
            inputs[0]
            .detach()
            .clone()
        )

    def hook_dres1(
        module,
        inputs,
    ):
        captured[
            "dres1_in"
        ] = (
            inputs[0]
            .detach()
            .clone()
        )

    h0 = (
        backbone.dres0
        .register_forward_pre_hook(
            hook_dres0
        )
    )

    h1 = (
        backbone.dres1
        .register_forward_pre_hook(
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

    if "dres0_in" not in captured:
        raise RuntimeError(
            "Failed to capture dres0 input."
        )

    if "dres1_in" not in captured:
        raise RuntimeError(
            "Failed to capture dres1 input."
        )

    x0 = (
        captured["dres0_in"]
        .contiguous()
    )

    x1 = (
        captured["dres1_in"]
        .contiguous()
    )

    print()
    print(
        "Captured dres0 input:",
        tuple(x0.shape),
        x0.dtype,
    )

    print(
        "Captured dres1 input:",
        tuple(x1.shape),
        x1.dtype,
    )

    _, _, D, H, W = x0.shape

    print(
        f"B2 volume: D={D}, H={H}, W={W}"
    )

    # ------------------------------------------------------
    # Full module baselines
    # ------------------------------------------------------

    full_dres0 = measure(
        lambda:
        run_dres0(
            backbone,
            x0,
        ),
        args.warmup,
        args.repeat,
    )

    full_dres1 = measure(
        lambda:
        run_dres1(
            backbone,
            x1,
        ),
        args.warmup,
        args.repeat,
    )

    full_b2 = measure(
        lambda:
        run_b2(
            backbone,
            x0,
        ),
        args.warmup,
        args.repeat,
    )

    print()
    print("=" * 78)
    print("Full B2 component latency")
    print("=" * 78)

    print(
        f"dres0 : "
        f"mean={full_dres0['mean_ms']:.3f} "
        f"p95={full_dres0['p95_ms']:.3f} "
        f"p99={full_dres0['p99_ms']:.3f}"
    )

    print(
        f"dres1 : "
        f"mean={full_dres1['mean_ms']:.3f} "
        f"p95={full_dres1['p95_ms']:.3f} "
        f"p99={full_dres1['p99_ms']:.3f}"
    )

    print(
        f"B2    : "
        f"mean={full_b2['mean_ms']:.3f} "
        f"p95={full_b2['p95_ms']:.3f} "
        f"p99={full_b2['p99_ms']:.3f}"
    )

    # ------------------------------------------------------
    # ROI fragmentation sweep
    # ------------------------------------------------------

    records = []

    print()
    print("=" * 112)

    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'Useful%':>9} "
        f"{'ExecSum%':>9} "
        f"{'ExecUniq%':>10} "
        f"{'Mean':>9} "
        f"{'P95':>9} "
        f"{'P99':>9} "
        f"{'Speedup':>9}"
    )

    print("-" * 112)

    full_area = H * W

    for ratio in args.ratios:

        for frag in args.fragments:

            semantic_rois = (
                make_fragmented_rois(
                    H,
                    W,
                    ratio,
                    frag,
                )
            )

            exec_rois = [
                expand_align(
                    r,
                    args.halo,
                    H,
                    W,
                    args.align,
                )
                for r in semantic_rois
            ]

            useful_area = sum(
                roi_area(r)
                for r in semantic_rois
            )

            exec_sum_area = sum(
                roi_area(r)
                for r in exec_rois
            )

            exec_unique_area = (
                unique_area(
                    exec_rois,
                    H,
                    W,
                )
            )

            fn = make_roi_exec_fn(
                backbone,
                x0,
                exec_rois,
            )

            result = measure(
                fn,
                args.warmup,
                args.repeat,
            )

            speedup = (
                full_b2["mean_ms"]
                / result["mean_ms"]
            )

            rec = {
                "ratio": ratio,
                "fragments": frag,
                "num_rois": len(exec_rois),

                "useful_area_ratio":
                    useful_area
                    / full_area,

                "exec_sum_area_ratio":
                    exec_sum_area
                    / full_area,

                "exec_unique_area_ratio":
                    exec_unique_area
                    / full_area,

                "latency": result,

                "speedup_vs_full":
                    speedup,

                "semantic_rois": [
                    list(r)
                    for r in semantic_rois
                ],

                "exec_rois": [
                    list(r)
                    for r in exec_rois
                ],
            }

            records.append(rec)

            print(
                f"{ratio*100:6.1f}% "
                f"{frag:5d} "
                f"{useful_area/full_area*100:8.2f}% "
                f"{exec_sum_area/full_area*100:8.2f}% "
                f"{exec_unique_area/full_area*100:9.2f}% "
                f"{result['mean_ms']:9.3f} "
                f"{result['p95_ms']:9.3f} "
                f"{result['p99_ms']:9.3f} "
                f"{speedup:9.3f}"
            )

    # ------------------------------------------------------
    # Simple crossover summary
    # ------------------------------------------------------

    print()
    print("=" * 88)
    print("Naive B2 selective/full crossover")
    print("=" * 88)

    guard_ms = 0.10

    for frag in args.fragments:

        rows = [
            r
            for r in records
            if r["fragments"] == frag
        ]

        good = [
            r
            for r in rows
            if (
                full_b2["mean_ms"]
                - r["latency"]["mean_ms"]
            ) > guard_ms
        ]

        if good:

            max_ratio = max(
                r["ratio"]
                for r in good
            )

            print(
                f"frag={frag:2d}: "
                f"selective useful through "
                f"~{max_ratio*100:.1f}% "
                f"with guard={guard_ms:.2f} ms"
            )

        else:

            print(
                f"frag={frag:2d}: "
                "no ROI ratio clears guard"
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
                "input_shape":
                    list(x0.shape),

                "halo":
                    args.halo,

                "align":
                    args.align,

                "full_dres0":
                    full_dres0,

                "full_dres1":
                    full_dres1,

                "full_b2":
                    full_b2,

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
