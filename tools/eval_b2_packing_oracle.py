#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

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
        "B2 packing oracle with scatter + correctness check"
    )

    p.add_argument("--cfg_file", default=DEFAULT_CFG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)

    p.add_argument(
        "--ratios",
        nargs="+",
        type=float,
        default=[
            0.30, 0.50, 0.60,
            0.70, 0.80, 0.90, 1.00,
        ],
    )

    p.add_argument(
        "--fragments",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
    )

    p.add_argument("--halo", type=int, default=1)
    p.add_argument("--align", type=int, default=4)

    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--repeat", type=int, default=300)

    p.add_argument(
        "--guard-ms",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--output",
        default=(
            "outputs/backbone_profile/"
            "b2_packing_oracle.json"
        ),
    )

    return p.parse_args()


# ============================================================
# Packing utilities
# ============================================================

def merge_bbox(a, b):
    return (
        min(a[0], b[0]),
        max(a[1], b[1]),
        min(a[2], b[2]),
        max(a[3], b[3]),
    )


def union_bbox(rois):
    out = rois[0]

    for r in rois[1:]:
        out = merge_bbox(
            out,
            r,
        )

    return out


def expanded_area(
    roi,
    halo,
    H,
    W,
    align,
):
    e = base.expand_align(
        roi,
        halo,
        H,
        W,
        align,
    )

    return base.roi_area(e)


def area_greedy_to_cap(
    input_rois,
    cap,
    halo,
    H,
    W,
    align,
):
    """
    Merge the pair with the smallest increase
    in physical expanded execution area.
    """

    rois = list(input_rois)

    while len(rois) > cap:

        best = None

        for i in range(len(rois)):
            for j in range(i + 1, len(rois)):

                merged = merge_bbox(
                    rois[i],
                    rois[j],
                )

                old_cost = (
                    expanded_area(
                        rois[i],
                        halo,
                        H,
                        W,
                        align,
                    )
                    +
                    expanded_area(
                        rois[j],
                        halo,
                        H,
                        W,
                        align,
                    )
                )

                new_cost = expanded_area(
                    merged,
                    halo,
                    H,
                    W,
                    align,
                )

                delta = (
                    new_cost
                    - old_cost
                )

                if (
                    best is None
                    or delta < best[0]
                ):
                    best = (
                        delta,
                        i,
                        j,
                        merged,
                    )

        _, i, j, merged = best

        rois = [
            r
            for k, r in enumerate(rois)
            if k not in (i, j)
        ]

        rois.append(
            merged
        )

    return rois


# ============================================================
# Physical execution
# ============================================================

def make_exec_pairs(
    physical_rois,
    halo,
    H,
    W,
    align,
):
    pairs = []

    for core in physical_rois:

        exec_roi = base.expand_align(
            core,
            halo,
            H,
            W,
            align,
        )

        pairs.append(
            (
                core,
                exec_roi,
            )
        )

    return pairs


def make_exec_fn(
    backbone,
    full_input,
    physical_rois,
    canvas,
    halo,
    align,
):
    """
    IMPORTANT:
      crop + contiguous + B2 + scatter
    are ALL included inside measured latency.
    """

    H = full_input.shape[-2]
    W = full_input.shape[-1]

    pairs = make_exec_pairs(
        physical_rois,
        halo,
        H,
        W,
        align,
    )

    def run():

        for core, e in pairs:

            cy0, cy1, cx0, cx1 = core
            ey0, ey1, ex0, ex1 = e

            crop = (
                full_input[
                    ...,
                    ey0:ey1,
                    ex0:ex1
                ]
                .contiguous()
            )

            y = base.run_b2(
                backbone,
                crop,
            )

            ry0 = cy0 - ey0
            rx0 = cx0 - ex0

            rh = cy1 - cy0
            rw = cx1 - cx0

            # Actual scatter/composition overhead
            canvas[
                ...,
                cy0:cy1,
                cx0:cx1
            ].copy_(
                y[
                    ...,
                    ry0:ry0 + rh,
                    rx0:rx0 + rw
                ]
            )

        return canvas

    return run


# ============================================================
# Exactness validation
# ============================================================

@torch.no_grad()
def validate_plan(
    backbone,
    full_input,
    full_output,
    physical_rois,
    halo,
    align,
):
    """
    Compare every physical ROI core against full B2 output.

    halo=1 should theoretically be enough here because:
      dres0 = 1x1x1
      dres1 = 3x3x3
    and depth dimension is never cropped.
    """

    H = full_input.shape[-2]
    W = full_input.shape[-1]

    pairs = make_exec_pairs(
        physical_rois,
        halo,
        H,
        W,
        align,
    )

    max_abs = 0.0
    sum_abs = 0.0
    count = 0

    for core, e in pairs:

        cy0, cy1, cx0, cx1 = core
        ey0, ey1, ex0, ex1 = e

        crop = (
            full_input[
                ...,
                ey0:ey1,
                ex0:ex1
            ]
            .contiguous()
        )

        y = base.run_b2(
            backbone,
            crop,
        )

        ry0 = cy0 - ey0
        rx0 = cx0 - ex0

        rh = cy1 - cy0
        rw = cx1 - cx0

        pred = y[
            ...,
            ry0:ry0 + rh,
            rx0:rx0 + rw
        ].float()

        ref = full_output[
            ...,
            cy0:cy1,
            cx0:cx1
        ].float()

        diff = (
            pred
            - ref
        ).abs()

        max_abs = max(
            max_abs,
            float(
                diff.max().item()
            ),
        )

        sum_abs += float(
            diff.sum().item()
        )

        count += diff.numel()

    mean_abs = (
        sum_abs / max(count, 1)
    )

    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
    }


# ============================================================
# Main
# ============================================================

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

    if backbone.num_hg != 0:
        raise RuntimeError(
            f"Expected num_hg=0, got {backbone.num_hg}"
        )

    # --------------------------------------------------------
    # Capture real cost-volume output = dres0 input
    # --------------------------------------------------------

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

    batch = dataset.collate_batch(
        [sample]
    )

    load_data_to_gpu(
        batch
    )

    with torch.no_grad():
        model(batch)

    torch.cuda.synchronize()

    handle.remove()

    x = (
        captured["x"]
        .contiguous()
    )

    _, _, _, H, W = x.shape

    print()
    print(
        "B2 input:",
        tuple(x.shape),
        x.dtype,
    )

    # --------------------------------------------------------
    # Full reference
    # --------------------------------------------------------

    with torch.no_grad():
        full_output = (
            base.run_b2(
                backbone,
                x,
            )
        )

    torch.cuda.synchronize()

    canvas = torch.empty_like(
        full_output
    )

    full_stats = base.measure(
        lambda:
        base.run_b2(
            backbone,
            x,
        ),
        args.warmup,
        args.repeat,
    )

    full_ms = full_stats[
        "mean_ms"
    ]

    print(
        "Full B2:",
        f"mean={full_stats['mean_ms']:.3f}",
        f"p95={full_stats['p95_ms']:.3f}",
        f"p99={full_stats['p99_ms']:.3f}",
    )

    strategies = [
        "naive",
        "cap4",
        "cap2",
        "bbox1",
    ]

    records = []

    print()
    print("=" * 132)

    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'Strategy':>10} "
        f"{'ROI':>4} "
        f"{'Mean':>9} "
        f"{'P95':>9} "
        f"{'P99':>9} "
        f"{'Save':>9} "
        f"{'Speedup':>9} "
        f"{'MaxDiff':>12}"
    )

    print("-" * 132)

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

            group_records = []

            for strategy in strategies:

                if strategy == "naive":

                    physical_rois = list(
                        semantic_rois
                    )

                elif strategy == "cap4":

                    physical_rois = (
                        area_greedy_to_cap(
                            semantic_rois,
                            min(
                                4,
                                len(semantic_rois),
                            ),
                            args.halo,
                            H,
                            W,
                            args.align,
                        )
                    )

                elif strategy == "cap2":

                    physical_rois = (
                        area_greedy_to_cap(
                            semantic_rois,
                            min(
                                2,
                                len(semantic_rois),
                            ),
                            args.halo,
                            H,
                            W,
                            args.align,
                        )
                    )

                elif strategy == "bbox1":

                    physical_rois = [
                        union_bbox(
                            semantic_rois
                        )
                    ]

                else:
                    raise ValueError(
                        strategy
                    )

                fn = make_exec_fn(
                    backbone,
                    x,
                    physical_rois,
                    canvas,
                    args.halo,
                    args.align,
                )

                latency = base.measure(
                    fn,
                    args.warmup,
                    args.repeat,
                )

                correctness = (
                    validate_plan(
                        backbone,
                        x,
                        full_output,
                        physical_rois,
                        args.halo,
                        args.align,
                    )
                )

                saving = (
                    full_ms
                    - latency["mean_ms"]
                )

                speedup = (
                    full_ms
                    / latency["mean_ms"]
                )

                rec = {
                    "ratio": ratio,
                    "fragments": frag,
                    "strategy": strategy,
                    "num_rois":
                        len(physical_rois),
                    "latency": latency,
                    "saving_ms": saving,
                    "speedup_vs_full":
                        speedup,
                    "correctness":
                        correctness,
                    "physical_rois": [
                        list(r)
                        for r in physical_rois
                    ],
                }

                records.append(rec)
                group_records.append(rec)

                print(
                    f"{ratio*100:6.1f}% "
                    f"{frag:5d} "
                    f"{strategy:>10} "
                    f"{len(physical_rois):4d} "
                    f"{latency['mean_ms']:9.3f} "
                    f"{latency['p95_ms']:9.3f} "
                    f"{latency['p99_ms']:9.3f} "
                    f"{saving:9.3f} "
                    f"{speedup:9.3f} "
                    f"{correctness['max_abs_diff']:12.6g}"
                )

            # ------------------------------------------------
            # Hardware oracle + no-regret decision
            # ------------------------------------------------

            best = min(
                group_records,
                key=lambda r:
                r["latency"]["mean_ms"],
            )

            best_saving = (
                full_ms
                - best["latency"]["mean_ms"]
            )

            if (
                best_saving
                > args.guard_ms
            ):
                decision = (
                    best["strategy"]
                )
                final_ms = (
                    best["latency"]["mean_ms"]
                )
            else:
                decision = "FULL"
                final_ms = full_ms

            print(
                f"    -> oracle={best['strategy']} "
                f"{best['latency']['mean_ms']:.3f} ms, "
                f"no-regret={decision}, "
                f"final={final_ms:.3f} ms"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
                "full_b2":
                    full_stats,
                "halo":
                    args.halo,
                "align":
                    args.align,
                "guard_ms":
                    args.guard_ms,
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
