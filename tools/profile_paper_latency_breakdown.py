#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    ElasticBEVBranch,
)
from pcdet.models.backbones_3d_stream.elastic_v4_bn_hybrid import (
    FULL_SCHEDULE,
    enable_fused_bn_static_cache,
)

# Reuse the exact stage boundaries and native/full-path implementation
# already used by the released TV-streamDSGN profiling pipeline.
import profile_fixed_forward_components as baseprof


torch.backends.cudnn.benchmark = True


PAPER_COMPONENTS = OrderedDict([
    ("Stem + Res1", "fixed_prefix_ms"),
    ("Res2--Res4", "res2_res4_ms"),
    ("FPN", "fpn_ms"),
    ("Stereo processing", "stereo_ms"),
    ("Geometry + 3D voxel refinement", "geometry_voxel_ms"),
    ("Temporal fusion", "temporal_fusion_ms"),
    ("VAN backbone", "van_ms"),
    ("Detection head", "det_head_ms"),
])

ELASTIC_COMPONENTS = {
    "Stem + Res1": False,
    "Res2--Res4": True,
    "FPN": True,
    "Stereo processing": True,
    "Geometry + 3D voxel refinement": True,
    "Temporal fusion": False,
    "VAN backbone": False,
    "Detection head": False,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Paper-oriented forward-only latency breakdown for TV-streamDSGN. "
            "The profiler reuses the existing component profiler and further "
            "splits the fixed tail into temporal fusion, VAN backbone, and "
            "StreamDetHead."
        )
    )

    p.add_argument("--full_cfg", required=True)
    p.add_argument("--full_ckpt", required=True)
    p.add_argument("--elastic_ckpt", required=True)

    p.add_argument(
        "--schedule",
        default="1,1,1,1,1,1",
        help=(
            "Six-stage schedule: Res2,Res3,Res4,FPN,Stereo,RPN. "
            "For the paper latency-scope table, use full width."
        ),
    )

    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1024)

    p.add_argument(
        "--output_dir",
        required=True,
    )

    return p.parse_args()


def percentile(x, q):
    return float(
        np.percentile(
            np.asarray(x, dtype=np.float64),
            q,
        )
    )


def stats(x):
    x = np.asarray(x, dtype=np.float64)

    return {
        "n": int(x.size),
        "mean_ms": float(np.mean(x)),
        "std_ms": float(np.std(x)),
        "p50_ms": percentile(x, 50),
        "p90_ms": percentile(x, 90),
        "p95_ms": percentile(x, 95),
        "p99_ms": percentile(x, 99),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


class TailEventRecorder:
    """
    Add CUDA events inside the fixed downstream path without changing
    the model forward.

    The original profiler already records:

        after_rpn -------------------------- forward_end
                        fixed_tail

    Here we insert:

        after_rpn
            |
            | history snapshot + temporal fusion
            v
        after_temporal_fusion
            |
            | VANBackbone
            v
        after_van
            |
            | StreamDetHead
            v
        after_det_head
            |
            | completion-side bookkeeping
            v
        forward_end

    No synchronize() is inserted by these hooks.
    """

    def __init__(self):
        self.current = None

    def begin(self):
        self.current = {}

    def hook(self, key):
        def _hook(module, inputs, output):
            if self.current is not None:
                self.current[key] = baseprof.record()
        return _hook

    def finish(self):
        if self.current is None:
            raise RuntimeError("TailEventRecorder.finish() without begin()")
        out = self.current
        self.current = None
        return out


def install_tail_hooks(model, recorder):
    if not hasattr(model, "fusion_module") or len(model.fusion_module) == 0:
        raise RuntimeError(
            "No fusion_module found; cannot locate temporal fusion boundary."
        )

    if (
        not hasattr(model, "after_fusion_blocks")
        or len(model.after_fusion_blocks) == 0
    ):
        raise RuntimeError(
            "No after_fusion_blocks found; cannot locate VAN/head boundaries."
        )

    fusion_names = [
        type(m).__name__
        for m in model.fusion_module
    ]

    tail_names = [
        type(m).__name__
        for m in model.after_fusion_blocks
    ]

    print("[MODEL] fusion_module       =", fusion_names)
    print("[MODEL] after_fusion_blocks =", tail_names)

    # The K3 final model is expected to use one temporal-fusion stage
    # followed by VANBackbone and StreamDetHead.
    #
    # We explicitly check the fixed tail instead of silently assigning
    # timings to the wrong component if the config changes.
    expected_tail = [
        "VANBackbone",
        "StreamDetHead",
    ]

    if tail_names != expected_tail:
        raise RuntimeError(
            "Unexpected fixed downstream modules.\n"
            f"Expected: {expected_tail}\n"
            f"Actual:   {tail_names}\n"
            "Update the profiler before using the resulting table."
        )

    handles = []

    # If fusion_module contains more than one module, hook the final one.
    # The interval from after_rpn to this event therefore covers the
    # complete temporal-fusion path.
    handles.append(
        model.fusion_module[-1].register_forward_hook(
            recorder.hook("after_temporal_fusion")
        )
    )

    handles.append(
        model.after_fusion_blocks[0].register_forward_hook(
            recorder.hook("after_van")
        )
    )

    handles.append(
        model.after_fusion_blocks[1].register_forward_hook(
            recorder.hook("after_det_head")
        )
    )

    return handles


def make_row(
    frame_index,
    dataset_index,
    schedule,
    events,
):
    required = [
        "start",
        "after_prefix",
        "after_res2",
        "after_res3",
        "after_res4",
        "after_fpn",
        "after_stereo",
        "after_rpn",
        "after_temporal_fusion",
        "after_van",
        "after_det_head",
        "forward_end",
    ]

    missing = [
        x for x in required
        if x not in events
    ]

    if missing:
        raise RuntimeError(
            f"Missing CUDA events: {missing}"
        )

    ms = baseprof.event_ms

    row = {
        "frame_index": int(frame_index),
        "dataset_index": int(dataset_index),
        "schedule": ",".join(
            str(float(x))
            for x in schedule
        ),

        # Detailed diagnostic boundaries.
        "fixed_prefix_ms":
            ms(events["start"], events["after_prefix"]),

        "res2_ms":
            ms(events["after_prefix"], events["after_res2"]),

        "res3_ms":
            ms(events["after_res2"], events["after_res3"]),

        "res4_ms":
            ms(events["after_res3"], events["after_res4"]),

        # Paper row: Res2--Res4 as one component.
        "res2_res4_ms":
            ms(events["after_prefix"], events["after_res4"]),

        "fpn_ms":
            ms(events["after_res4"], events["after_fpn"]),

        "stereo_ms":
            ms(events["after_fpn"], events["after_stereo"]),

        # IMPORTANT:
        # The existing "RPN" execution interval contains geometry
        # projection/grid sampling plus 3D voxel refinement.
        "geometry_voxel_ms":
            ms(events["after_stereo"], events["after_rpn"]),

        # This interval also includes creation of the current BEV snapshot
        # used by the temporal history, matching the forward-only timing
        # semantics of the released runtime.
        "temporal_fusion_ms":
            ms(
                events["after_rpn"],
                events["after_temporal_fusion"],
            ),

        "van_ms":
            ms(
                events["after_temporal_fusion"],
                events["after_van"],
            ),

        "det_head_ms":
            ms(
                events["after_van"],
                events["after_det_head"],
            ),

        # Small residual after the detector-head CUDA work, e.g.
        # completion-side Python/history bookkeeping before forward_end.
        "post_head_residual_ms":
            ms(
                events["after_det_head"],
                events["forward_end"],
            ),

        "profiled_model_ms":
            ms(
                events["start"],
                events["after_det_head"],
            ),

        "forward_total_ms":
            ms(
                events["start"],
                events["forward_end"],
            ),
    }

    row["coverage_pct"] = (
        100.0
        * row["profiled_model_ms"]
        / row["forward_total_ms"]
    )

    return row


def write_raw_csv(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows, args, schedule, scene_name):
    all_metric_keys = [
        "fixed_prefix_ms",
        "res2_ms",
        "res3_ms",
        "res4_ms",
        "res2_res4_ms",
        "fpn_ms",
        "stereo_ms",
        "geometry_voxel_ms",
        "temporal_fusion_ms",
        "van_ms",
        "det_head_ms",
        "post_head_residual_ms",
        "profiled_model_ms",
        "forward_total_ms",
        "coverage_pct",
    ]

    metrics = {
        key: stats(
            [row[key] for row in rows]
        )
        for key in all_metric_keys
    }

    forward_mean = metrics[
        "forward_total_ms"
    ]["mean_ms"]

    component_rows = []

    for display_name, key in PAPER_COMPONENTS.items():
        component_stats = metrics[key]

        component_rows.append({
            "component": display_name,
            "metric_key": key,
            "elastic": bool(
                ELASTIC_COMPONENTS[display_name]
            ),
            "mean_ms": component_stats["mean_ms"],
            "p50_ms": component_stats["p50_ms"],
            "p95_ms": component_stats["p95_ms"],
            "p99_ms": component_stats["p99_ms"],
            "share_pct": (
                100.0
                * component_stats["mean_ms"]
                / forward_mean
            ),
        })

    elastic_keys = [
        "res2_res4_ms",
        "fpn_ms",
        "stereo_ms",
        "geometry_voxel_ms",
    ]

    fixed_downstream_keys = [
        "temporal_fusion_ms",
        "van_ms",
        "det_head_ms",
    ]

    fixed_prefix_mean = metrics[
        "fixed_prefix_ms"
    ]["mean_ms"]

    elastic_mean = sum(
        metrics[x]["mean_ms"]
        for x in elastic_keys
    )

    fixed_downstream_mean = sum(
        metrics[x]["mean_ms"]
        for x in fixed_downstream_keys
    )

    residual_mean = metrics[
        "post_head_residual_ms"
    ]["mean_ms"]

    summary = {
        "gpu": {
            "name": torch.cuda.get_device_name(
                torch.cuda.current_device()
            ),
            "cuda_runtime": torch.version.cuda,
            "pytorch": torch.__version__,
        },

        "timing_scope": (
            "model forward only; data loading and H2D occur before the "
            "start CUDA event; post_processing/NMS are not executed"
        ),

        "measurement": (
            "CUDA events on the detector stream; no synchronization at "
            "component boundaries; one synchronization after forward_end"
        ),

        "schedule": [
            float(x)
            for x in schedule
        ],

        "scene": scene_name,
        "warmup": int(args.warmup),
        "frames": int(args.frames),

        "metrics": metrics,
        "paper_components": component_rows,

        "scope_summary": {
            "forward_mean_ms": forward_mean,

            "fixed_prefix_mean_ms":
                fixed_prefix_mean,

            "fixed_prefix_share_pct":
                100.0 * fixed_prefix_mean / forward_mean,

            "elastic_region_mean_ms":
                elastic_mean,

            "elastic_region_share_pct":
                100.0 * elastic_mean / forward_mean,

            "fixed_downstream_mean_ms":
                fixed_downstream_mean,

            "fixed_downstream_share_pct":
                100.0 * fixed_downstream_mean / forward_mean,

            "post_head_residual_mean_ms":
                residual_mean,

            "post_head_residual_share_pct":
                100.0 * residual_mean / forward_mean,

            "profiled_model_coverage_pct":
                100.0
                * metrics["profiled_model_ms"]["mean_ms"]
                / forward_mean,
        },
    }

    return summary


def write_paper_csv(path, summary):
    rows = summary["paper_components"]

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Component",
                "Mean latency (ms)",
                "p50 (ms)",
                "p95 (ms)",
                "p99 (ms)",
                "Share (%)",
                "Elastic",
            ],
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                "Component":
                    row["component"],

                "Mean latency (ms)":
                    f"{row['mean_ms']:.4f}",

                "p50 (ms)":
                    f"{row['p50_ms']:.4f}",

                "p95 (ms)":
                    f"{row['p95_ms']:.4f}",

                "p99 (ms)":
                    f"{row['p99_ms']:.4f}",

                "Share (%)":
                    f"{row['share_pct']:.2f}",

                "Elastic":
                    "Yes" if row["elastic"] else "No",
            })


def tex_escape_component(text):
    return (
        text
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def write_latex_table(path, summary):
    rows = summary["paper_components"]
    scope = summary["scope_summary"]

    lines = []

    lines.append(r"\begin{table}[!t]")
    lines.append(r"    \centering")
    lines.append(
        r"    \caption{Forward-latency breakdown and elasticity scope "
        r"of the full-width TV-streamDSGN. "
        r"Latency reports the mean CUDA execution time; "
        r"Share is normalized by the measured forward-only latency.}"
    )
    lines.append(r"    \label{tab:latency_breakdown}")
    lines.append(r"    \begin{tabular}{lccc}")
    lines.append(r"        \toprule")
    lines.append(
        r"        \textbf{Component} & "
        r"\textbf{Latency (ms)} & "
        r"\textbf{Share (\%)} & "
        r"\textbf{Elastic} \\"
    )
    lines.append(r"        \midrule")

    for row in rows:
        name = tex_escape_component(
            row["component"]
        )

        elastic = (
            "Yes"
            if row["elastic"]
            else "No"
        )

        lines.append(
            "        "
            f"{name} & "
            f"{row['mean_ms']:.2f} & "
            f"{row['share_pct']:.1f} & "
            f"{elastic} \\\\"
        )

    residual_share = scope[
        "post_head_residual_share_pct"
    ]

    # Keep the main architecture table honest if the residual is not
    # completely negligible.
    if residual_share >= 0.1:
        lines.append(
            "        "
            f"Other forward overhead & "
            f"{scope['post_head_residual_mean_ms']:.2f} & "
            f"{residual_share:.1f} & No \\\\"
        )

    lines.append(r"        \midrule")
    lines.append(
        "        "
        f"Total & "
        f"{scope['forward_mean_ms']:.2f} & "
        f"100.0 & -- \\\\"
    )
    lines.append(r"        \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"\end{table}")

    path.write_text(
        "\n".join(lines) + "\n"
    )


def print_console_table(summary):
    rows = summary["paper_components"]
    scope = summary["scope_summary"]

    print()
    print("=" * 86)
    print(
        f"{'Component':38s} "
        f"{'Mean(ms)':>10s} "
        f"{'p99(ms)':>10s} "
        f"{'Share':>9s} "
        f"{'Elastic':>8s}"
    )
    print("-" * 86)

    for row in rows:
        print(
            f"{row['component']:38s} "
            f"{row['mean_ms']:10.3f} "
            f"{row['p99_ms']:10.3f} "
            f"{row['share_pct']:8.2f}% "
            f"{('Yes' if row['elastic'] else 'No'):>8s}"
        )

    print("-" * 86)

    print(
        f"{'Forward total':38s} "
        f"{scope['forward_mean_ms']:10.3f} "
        f"{'--':>10s} "
        f"{100.0:8.2f}% "
        f"{'--':>8s}"
    )

    print("=" * 86)

    print(
        "[SCOPE] fixed prefix      : "
        f"{scope['fixed_prefix_mean_ms']:.3f} ms "
        f"({scope['fixed_prefix_share_pct']:.2f}%)"
    )

    print(
        "[SCOPE] elastic region    : "
        f"{scope['elastic_region_mean_ms']:.3f} ms "
        f"({scope['elastic_region_share_pct']:.2f}%)"
    )

    print(
        "[SCOPE] fixed downstream  : "
        f"{scope['fixed_downstream_mean_ms']:.3f} ms "
        f"({scope['fixed_downstream_share_pct']:.2f}%)"
    )

    print(
        "[SCOPE] post-head residual: "
        f"{scope['post_head_residual_mean_ms']:.3f} ms "
        f"({scope['post_head_residual_share_pct']:.2f}%)"
    )

    print(
        "[CHECK] profiled coverage : "
        f"{scope['profiled_model_coverage_pct']:.2f}%"
    )

    print("=" * 86)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    schedule = baseprof.parse_schedule(
        args.schedule
    )

    if tuple(schedule) != FULL_SCHEDULE:
        print(
            "[WARNING] Paper scope table is normally measured "
            "with the full schedule (1,1,1,1,1,1)."
        )

    cfg_obj = baseprof.make_cfg(
        args.full_cfg
    )

    logger = common_utils.create_logger()

    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg_obj.DATA_CONFIG,
        class_names=cfg_obj.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    base_model = build_network(
        model_cfg=cfg_obj.MODEL,
        num_class=len(cfg_obj.CLASS_NAMES),
        dataset=dataset,
    )

    base_model.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )

    base_model.cuda().eval()

    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(
            cfg_obj.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
        ),
    ).cuda().eval()

    elastic_ckpt = torch.load(
        args.elastic_ckpt,
        map_location="cpu",
    )

    if "branch" not in elastic_ckpt:
        raise RuntimeError(
            "elastic checkpoint has no 'branch' state_dict"
        )

    branch.load_state_dict(
        elastic_ckpt["branch"],
        strict=True,
    )

    if tuple(schedule) != FULL_SCHEDULE:
        fused_counts = enable_fused_bn_static_cache(
            branch
        )
        print(
            "[ELASTIC] fused static cache =",
            fused_counts,
        )

    recorder = TailEventRecorder()

    handles = install_tail_hooks(
        base_model,
        recorder,
    )

    needed = (
        int(args.warmup)
        + int(args.frames)
    )

    scene_name, indices = (
        baseprof.choose_scene_indices(
            dataset,
            needed,
        )
    )

    print("[DATA] timing scene =", scene_name)
    print("[DATA] warmup       =", args.warmup)
    print("[DATA] frames       =", args.frames)
    print("[DATA] schedule     =", tuple(schedule))
    print(
        "[GPU] ",
        torch.cuda.get_device_name(
            torch.cuda.current_device()
        ),
    )

    if (
        getattr(
            base_model,
            "history_feature_queue",
            None,
        )
        is not None
    ):
        base_model.history_feature_queue.clear()

    model_stream = (
        baseprof.make_high_priority_detector_stream(
            torch.cuda.current_device()
        )
    )

    print(
        "[CUDA] stream priority range =",
        baseprof.stream_priority_range(),
    )

    print(
        "[CUDA] detector priority     =",
        baseprof.stream_priority_value(
            model_stream
        ),
    )

    def run_one(batch):
        # H2D has already completed/enqueued on the caller/default stream.
        # The detector stream waits for it before recording the model timer.
        ready = torch.cuda.Event(
            enable_timing=False
        )
        ready.record(
            torch.cuda.current_stream()
        )

        recorder.begin()

        with torch.cuda.stream(model_stream):
            model_stream.wait_event(ready)

            events, bev = (
                baseprof.stagewise_forward(
                    base_model,
                    branch,
                    batch,
                    schedule,
                )
            )

        tail_events = recorder.finish()
        events.update(tail_events)

        return events, bev

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    last_bev = None

    for i in range(int(args.warmup)):
        batch = baseprof.load_one(
            dataset,
            indices[i],
        )

        _, last_bev = run_one(batch)

        if (
            (i + 1) % 20 == 0
            or i + 1 == int(args.warmup)
        ):
            print(
                f"[warmup] "
                f"{i + 1}/{args.warmup}"
            )

    # Match the released component profiler: start measured samples with a
    # deterministic full three-slot history.
    if last_bev is not None:
        baseprof.fill_history(
            base_model.history_feature_queue,
            last_bev,
        )

    # --------------------------------------------------------
    # Timed frames
    # --------------------------------------------------------

    rows = []
    offset = int(args.warmup)

    for i in range(int(args.frames)):
        dataset_index = indices[
            offset + i
        ]

        # dataset/collate/H2D are outside the timer.
        batch = baseprof.load_one(
            dataset,
            dataset_index,
        )

        events, _ = run_one(batch)

        row = make_row(
            i,
            dataset_index,
            schedule,
            events,
        )

        rows.append(row)

        if (
            i < 5
            or (i + 1) % 50 == 0
            or i + 1 == int(args.frames)
        ):
            print(
                f"[{i + 1:04d}/{args.frames:04d}] "
                f"forward={row['forward_total_ms']:.3f} ms | "
                f"elastic="
                f"{row['res2_res4_ms'] + row['fpn_ms'] + row['stereo_ms'] + row['geometry_voxel_ms']:.3f} ms | "
                f"fixed-tail="
                f"{row['temporal_fusion_ms'] + row['van_ms'] + row['det_head_ms']:.3f} ms"
            )

    for h in handles:
        h.remove()

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_csv = (
        out_dir
        / "latency_breakdown_raw.csv"
    )

    summary_json = (
        out_dir
        / "latency_breakdown_summary.json"
    )

    paper_csv = (
        out_dir
        / "paper_latency_breakdown.csv"
    )

    paper_tex = (
        out_dir
        / "paper_latency_breakdown.tex"
    )

    write_raw_csv(
        raw_csv,
        rows,
    )

    summary = build_summary(
        rows,
        args,
        schedule,
        scene_name,
    )

    summary_json.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )

    write_paper_csv(
        paper_csv,
        summary,
    )

    write_latex_table(
        paper_tex,
        summary,
    )

    print_console_table(
        summary
    )

    print()
    print("[RESULT]", raw_csv)
    print("[RESULT]", summary_json)
    print("[RESULT]", paper_csv)
    print("[RESULT]", paper_tex)

    coverage = summary[
        "scope_summary"
    ]["profiled_model_coverage_pct"]

    if coverage < 99.0:
        print()
        print(
            "[WARNING] Profiled component coverage is below 99%. "
            "Inspect post_head_residual_ms before using the table "
            "in the paper."
        )


if __name__ == "__main__":
    main()
