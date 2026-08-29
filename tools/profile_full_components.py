#!/usr/bin/env python3

import argparse
import csv
import json
import random
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch

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
            "Component-level CUDA latency profiler for StreamDSGN / "
            "Multi-History StreamDSGN. "
            "Data loading, preprocessing and H2D are outside the timer."
        )
    )

    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Chronological model forwards used only for GPU warmup.",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=500,
        help="Steady-state frames to profile. <=0 means all available.",
    )

    parser.add_argument(
        "--skip_scene_prefix",
        type=int,
        default=3,
        help=(
            "Run but do not profile the first N frames of every scene. "
            "For K=3 memory, N=3 means measured frames have a full "
            "three-slot history queue."
        ),
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
        default="outputs/profile_full_components",
    )

    parser.add_argument(
        "--variant_name",
        type=str,
        default="Full",
        help="Display name only: Full / Light / etc.",
    )

    return parser.parse_args()


# ============================================================
# Basic utilities
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clear_history(model):
    queue = getattr(
        model,
        "history_feature_queue",
        None,
    )

    if queue is not None:
        queue.clear()


def frame_token(dataset, dataset_index):
    info = dataset.kitti_infos[dataset_index]

    return str(
        info["sample_idx"]["frame_tag"]["token"]
    )


def frame_sort_key(dataset, dataset_index):
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
    Group dataset indices by scene and sort every scene chronologically.
    """

    scenes = OrderedDict()

    for dataset_index, info in enumerate(
        dataset.kitti_infos
    ):
        scene = str(
            info["sample_idx"]["scene"]
        )

        scenes.setdefault(
            scene,
            [],
        ).append(dataset_index)

    for scene in scenes:
        scenes[scene].sort(
            key=lambda idx: frame_sort_key(
                dataset,
                idx,
            )
        )

    return scenes


def load_one(dataset, dataset_index):
    """
    Loading + preprocessing + H2D.

    IMPORTANT:
    These operations are outside the timer.

    This matches the model-service timing boundary used by
    test_stream_buffer_timestamp.py.
    """

    data_dict = dataset[dataset_index]

    batch_dict = dataset.collate_batch(
        [data_dict]
    )

    load_data_to_gpu(
        batch_dict
    )

    return batch_dict


# ============================================================
# CUDA event profiler
# ============================================================

class CudaEventProfiler:

    def __init__(self):
        self.handles = []

        self.active = False

        self.current_pairs = defaultdict(
            list
        )

        self._stacks = defaultdict(
            list
        )

        self.registered = []


    @staticmethod
    def _event():
        return torch.cuda.Event(
            enable_timing=True
        )


    def add_pair(
        self,
        label,
        start_event,
        end_event,
    ):
        if self.active:
            self.current_pairs[label].append(
                (
                    start_event,
                    end_event,
                )
            )


    def register_module(
        self,
        label,
        module,
    ):
        if module is None:
            return

        module_id = id(module)


        def pre_hook(
            _module,
            _inputs,
        ):
            if not self.active:
                return

            start = self._event()
            start.record()

            self._stacks[module_id].append(
                start
            )


        def post_hook(
            _module,
            _inputs,
            _output,
        ):
            if not self.active:
                return

            if (
                module_id not in self._stacks
                or
                len(self._stacks[module_id]) == 0
            ):
                raise RuntimeError(
                    f"Profiler stack mismatch for {label}"
                )

            start = self._stacks[
                module_id
            ].pop()

            end = self._event()
            end.record()

            self.add_pair(
                label,
                start,
                end,
            )


        self.handles.append(
            module.register_forward_pre_hook(
                pre_hook
            )
        )

        self.handles.append(
            module.register_forward_hook(
                post_hook
            )
        )

        self.registered.append(
            label
        )


    def wrap_method(
        self,
        obj,
        method_name,
        label,
    ):
        """
        Time tensor-heavy methods that are not nn.Module objects.

        getattr(obj, method_name) gives us the already-bound original
        callable, so the replacement receives only explicit call args.
        """

        if not hasattr(
            obj,
            method_name,
        ):
            return

        original = getattr(
            obj,
            method_name,
        )

        if not callable(original):
            return


        def wrapped(
            *args,
            **kwargs,
        ):
            if not self.active:
                return original(
                    *args,
                    **kwargs,
                )

            start = self._event()
            start.record()

            out = original(
                *args,
                **kwargs,
            )

            end = self._event()
            end.record()

            self.add_pair(
                label,
                start,
                end,
            )

            return out


        setattr(
            obj,
            method_name,
            wrapped,
        )

        self.registered.append(
            label
        )


    def start_frame(self):
        if self.active:
            raise RuntimeError(
                "Profiler frame already active"
            )

        self.current_pairs = defaultdict(
            list
        )

        self._stacks = defaultdict(
            list
        )

        self.active = True


    def finish_frame(self):
        if not self.active:
            raise RuntimeError(
                "Profiler frame is not active"
            )

        self.active = False

        times_ms = {}
        calls = {}

        for label, pairs in (
            self.current_pairs.items()
        ):

            values = [
                float(
                    start.elapsed_time(end)
                )
                for start, end
                in pairs
            ]

            times_ms[label] = float(
                sum(values)
            )

            calls[label] = int(
                len(values)
            )

        for module_id, stack in (
            self._stacks.items()
        ):
            if len(stack) != 0:
                raise RuntimeError(
                    "Unclosed profiler event stack: "
                    f"module_id={module_id}, "
                    f"depth={len(stack)}"
                )

        return times_ms, calls


# ============================================================
# Component registration
# ============================================================

def register_components(
    model,
    profiler,
):
    """
    Top-level modules correspond directly to STREAM.forward_test().

    Backbone probes are selected so that most of them are disjoint.
    The part not captured by these probes is reported as:

        backbone.other_functional

    which contains things such as:
        calibration / coordinate work
        grid_sample PSV -> 3D volume
        tensor manipulation
        other functional operations
    """

    # ========================================================
    # Top-level StreamDSGN stages
    # ========================================================

    profiler.register_module(
        "backbone_3d",
        getattr(
            model,
            "backbone_3d",
            None,
        ),
    )

    profiler.register_module(
        "height_compression",
        getattr(
            model,
            "map_to_bev_module",
            None,
        ),
    )

    profiler.register_module(
        "temporal_fusion",
        getattr(
            model,
            "spatial_feature_fusion_module",
            None,
        ),
    )

    profiler.register_module(
        "van_backbone",
        getattr(
            model,
            "backbone_2d",
            None,
        ),
    )

    profiler.register_module(
        "spatial_feature_2d_fusion",
        getattr(
            model,
            "spatial_feature_2d_fusion_module",
            None,
        ),
    )

    profiler.register_module(
        "dense_head_2d",
        getattr(
            model,
            "dense_head_2d",
            None,
        ),
    )

    profiler.register_module(
        "dense_head",
        getattr(
            model,
            "dense_head",
            None,
        ),
    )


    # ========================================================
    # StreamDSGN2Backbone internals
    # ========================================================

    backbone = getattr(
        model,
        "backbone_3d",
        None,
    )

    if backbone is None:
        return


    # --------------------------------------------------------
    # Left + right image feature extraction
    #
    # Same module is called twice, so the profiler automatically
    # sums both calls for one frame.
    # --------------------------------------------------------

    profiler.register_module(
        "backbone.feature_backbone",
        getattr(
            backbone,
            "feature_backbone",
            None,
        ),
    )

    profiler.register_module(
        "backbone.feature_neck",
        getattr(
            backbone,
            "feature_neck",
            None,
        ),
    )


    # ========================================================
    # FINE_GRAIN_RESNET_NECK_PROBES
    # ========================================================

    feature_backbone = getattr(
        backbone,
        "feature_backbone",
        None,
    )

    if feature_backbone is not None:

        profiler.register_module(
            "backbone.resnet.stem.conv1",
            getattr(
                feature_backbone,
                "conv1",
                None,
            ),
        )

        # MMDetection ResNet normally exposes norm1 as a property.
        norm1 = getattr(
            feature_backbone,
            "norm1",
            None,
        )

        if norm1 is None:
            norm1 = getattr(
                feature_backbone,
                "bn1",
                None,
            )

        profiler.register_module(
            "backbone.resnet.stem.norm1",
            norm1,
        )

        profiler.register_module(
            "backbone.resnet.stem.relu",
            getattr(
                feature_backbone,
                "relu",
                None,
            ),
        )

        profiler.register_module(
            "backbone.resnet.stem.maxpool",
            getattr(
                feature_backbone,
                "maxpool",
                None,
            ),
        )

        res_layers = list(
            getattr(
                feature_backbone,
                "res_layers",
                [],
            )
        )

        if len(res_layers) == 0:
            res_layers = [
                name
                for name in (
                    "layer1",
                    "layer2",
                    "layer3",
                    "layer4",
                )
                if hasattr(
                    feature_backbone,
                    name,
                )
            ]

        for layer_name in res_layers:

            profiler.register_module(
                f"backbone.resnet.{layer_name}",
                getattr(
                    feature_backbone,
                    layer_name,
                    None,
                ),
            )


    feature_neck = getattr(
        backbone,
        "feature_neck",
        None,
    )

    if feature_neck is not None:

        spp_branches = getattr(
            feature_neck,
            "spp_branches",
            None,
        )

        if spp_branches is not None:

            for branch in spp_branches:

                profiler.register_module(
                    "backbone.neck.spp_branches",
                    branch,
                )

        profiler.register_module(
            "backbone.neck.upconv_module",
            getattr(
                feature_neck,
                "upconv_module",
                None,
            ),
        )

        profiler.register_module(
            "backbone.neck.lastconv",
            getattr(
                feature_neck,
                "lastconv",
                None,
            ),
        )

        profiler.register_module(
            "backbone.neck.upconv_module_voxel",
            getattr(
                feature_neck,
                "upconv_module_voxel",
                None,
            ),
        )

        profiler.register_module(
            "backbone.neck.rpnconv",
            getattr(
                feature_neck,
                "rpnconv",
                None,
            ),
        )


    profiler.register_module(
        "backbone.sem_neck",
        getattr(
            backbone,
            "sem_neck",
            None,
        ),
    )


    # --------------------------------------------------------
    # Plane sweep / stereo matching
    # --------------------------------------------------------

    profiler.register_module(
        "backbone.build_cost",
        getattr(
            backbone,
            "build_cost",
            None,
        ),
    )

    profiler.register_module(
        "backbone.dres0",
        getattr(
            backbone,
            "dres0",
            None,
        ),
    )

    profiler.register_module(
        "backbone.dres1",
        getattr(
            backbone,
            "dres1",
            None,
        ),
    )


    hg_stereo = getattr(
        backbone,
        "hg_stereo",
        None,
    )

    if hg_stereo is not None:
        for module in hg_stereo:
            profiler.register_module(
                "backbone.hg_stereo",
                module,
            )


    # --------------------------------------------------------
    # Depth prediction module
    # --------------------------------------------------------

    pred_stereo = getattr(
        backbone,
        "pred_stereo",
        None,
    )

    if pred_stereo is not None:
        for module in pred_stereo:
            profiler.register_module(
                "backbone.pred_stereo_module",
                module,
            )


    # --------------------------------------------------------
    # 3D RPN / voxel processing
    # --------------------------------------------------------

    profiler.register_module(
        "backbone.rpn3d_convs",
        getattr(
            backbone,
            "rpn3d_convs",
            None,
        ),
    )


    rpn3d_hgs = getattr(
        backbone,
        "rpn3d_hgs",
        None,
    )

    if rpn3d_hgs is not None:
        for module in rpn3d_hgs:
            profiler.register_module(
                "backbone.rpn3d_hgs",
                module,
            )


    profiler.register_module(
        "backbone.rpn3d_pool",
        getattr(
            backbone,
            "rpn3d_pool",
            None,
        ),
    )


    # ========================================================
    # Functional / method-level operations
    # ========================================================

    profiler.wrap_method(
        backbone,
        "compute_disp_channels",
        "backbone.compute_disp_channels",
    )

    profiler.wrap_method(
        backbone,
        "compute_mapping",
        "backbone.compute_mapping",
    )

    profiler.wrap_method(
        backbone,
        "get_local_depth",
        "backbone.get_local_depth",
    )


def install_post_processing_timer(
    model,
    profiler,
):
    """
    post_processing is a Python method rather than nn.Module.
    """

    original = model.post_processing


    def wrapped(
        batch_dict,
    ):
        if not profiler.active:
            return original(
                batch_dict
            )

        start = profiler._event()
        start.record()

        out = original(
            batch_dict
        )

        end = profiler._event()
        end.record()

        profiler.add_pair(
            "post_processing",
            start,
            end,
        )

        return out


    model.post_processing = wrapped


# ============================================================
# Derived timing
# ============================================================

def get_value(
    times,
    key,
):
    return float(
        times.get(
            key,
            0.0,
        )
    )


def add_derived_metrics(
    times,
):
    """
    Add groups useful for designing the Light branch.
    """

    # ========================================================
    # Backbone accounted time
    # ========================================================

    backbone_accounted_keys = [
        "backbone.feature_backbone",
        "backbone.feature_neck",
        "backbone.sem_neck",

        "backbone.compute_disp_channels",
        "backbone.build_cost",
        "backbone.dres0",
        "backbone.dres1",
        "backbone.hg_stereo",
        "backbone.pred_stereo_module",
        "backbone.get_local_depth",

        "backbone.compute_mapping",

        "backbone.rpn3d_convs",
        "backbone.rpn3d_hgs",
        "backbone.rpn3d_pool",
    ]

    backbone_accounted = sum(
        get_value(
            times,
            key,
        )
        for key in backbone_accounted_keys
    )

    times[
        "backbone.accounted"
    ] = float(
        backbone_accounted
    )

    times[
        "backbone.other_functional"
    ] = float(
        get_value(
            times,
            "backbone_3d",
        )
        -
        backbone_accounted
    )


    # ========================================================
    # Fine-grained ResNet / neck accounting
    # ========================================================

    resnet_stem_keys = [
        "backbone.resnet.stem.conv1",
        "backbone.resnet.stem.norm1",
        "backbone.resnet.stem.relu",
        "backbone.resnet.stem.maxpool",
    ]

    times[
        "backbone.resnet.stem"
    ] = float(
        sum(
            get_value(
                times,
                key,
            )
            for key in resnet_stem_keys
        )
    )

    resnet_stage_keys = [
        "backbone.resnet.layer1",
        "backbone.resnet.layer2",
        "backbone.resnet.layer3",
        "backbone.resnet.layer4",
    ]

    times[
        "backbone.resnet.stages"
    ] = float(
        sum(
            get_value(
                times,
                key,
            )
            for key in resnet_stage_keys
        )
    )

    times[
        "backbone.resnet.other_functional"
    ] = float(
        get_value(
            times,
            "backbone.feature_backbone",
        )
        -
        get_value(
            times,
            "backbone.resnet.stem",
        )
        -
        get_value(
            times,
            "backbone.resnet.stages",
        )
    )

    neck_detail_keys = [
        "backbone.neck.spp_branches",
        "backbone.neck.upconv_module",
        "backbone.neck.lastconv",
        "backbone.neck.upconv_module_voxel",
        "backbone.neck.rpnconv",
    ]

    times[
        "backbone.neck.accounted"
    ] = float(
        sum(
            get_value(
                times,
                key,
            )
            for key in neck_detail_keys
        )
    )

    times[
        "backbone.neck.other_functional"
    ] = float(
        get_value(
            times,
            "backbone.feature_neck",
        )
        -
        get_value(
            times,
            "backbone.neck.accounted",
        )
    )


    # ========================================================
    # Coarse backbone groups
    # ========================================================

    times[
        "backbone.2d_feature_extraction"
    ] = float(
        get_value(
            times,
            "backbone.feature_backbone",
        )
        +
        get_value(
            times,
            "backbone.feature_neck",
        )
        +
        get_value(
            times,
            "backbone.sem_neck",
        )
    )


    times[
        "backbone.stereo_matching_modules"
    ] = float(
        get_value(
            times,
            "backbone.compute_disp_channels",
        )
        +
        get_value(
            times,
            "backbone.build_cost",
        )
        +
        get_value(
            times,
            "backbone.dres0",
        )
        +
        get_value(
            times,
            "backbone.dres1",
        )
        +
        get_value(
            times,
            "backbone.hg_stereo",
        )
        +
        get_value(
            times,
            "backbone.pred_stereo_module",
        )
        +
        get_value(
            times,
            "backbone.get_local_depth",
        )
    )


    times[
        "backbone.rpn3d"
    ] = float(
        get_value(
            times,
            "backbone.rpn3d_convs",
        )
        +
        get_value(
            times,
            "backbone.rpn3d_hgs",
        )
        +
        get_value(
            times,
            "backbone.rpn3d_pool",
        )
    )


    # ========================================================
    # State generation
    #
    # Stored memory H_t is spatial_features after HeightCompression
    # and before temporal fusion.
    # ========================================================

    times[
        "state_generation"
    ] = float(
        get_value(
            times,
            "backbone_3d",
        )
        +
        get_value(
            times,
            "height_compression",
        )
    )


    # ========================================================
    # State consumption
    # ========================================================

    consumption_keys = [
        "temporal_fusion",
        "van_backbone",
        "spatial_feature_2d_fusion",
        "dense_head_2d",
        "dense_head",
        "post_processing",
    ]

    times[
        "state_consumption"
    ] = float(
        sum(
            get_value(
                times,
                key,
            )
            for key in consumption_keys
        )
    )


    # ========================================================
    # Whole service accounting
    # ========================================================

    top_level_keys = [
        "backbone_3d",
        "height_compression",
        "temporal_fusion",
        "van_backbone",
        "spatial_feature_2d_fusion",
        "dense_head_2d",
        "dense_head",
        "post_processing",
    ]

    accounted = sum(
        get_value(
            times,
            key,
        )
        for key in top_level_keys
    )

    times[
        "service.accounted"
    ] = float(
        accounted
    )

    times[
        "service.other"
    ] = float(
        get_value(
            times,
            "service_total",
        )
        -
        accounted
    )


# ============================================================
# Statistics
# ============================================================

def stats(values):
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    if x.size == 0:
        return {
            "count": 0,
            "mean_ms": None,
            "std_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p99_ms": None,
            "min_ms": None,
            "max_ms": None,
        }

    return {
        "count": int(
            x.size
        ),

        "mean_ms": float(
            np.mean(x)
        ),

        "std_ms": float(
            np.std(x)
        ),

        "p50_ms": float(
            np.percentile(
                x,
                50,
            )
        ),

        "p90_ms": float(
            np.percentile(
                x,
                90,
            )
        ),

        "p99_ms": float(
            np.percentile(
                x,
                99,
            )
        ),

        "min_ms": float(
            np.min(x)
        ),

        "max_ms": float(
            np.max(x)
        ),
    }


def aggregate_records(
    records,
):
    all_keys = set()

    for record in records:
        all_keys.update(
            record[
                "times_ms"
            ].keys()
        )

    summary = {}

    for key in sorted(
        all_keys
    ):

        values = [
            record[
                "times_ms"
            ].get(
                key,
                0.0,
            )
            for record in records
        ]

        summary[key] = stats(
            values
        )


        call_values = [
            record[
                "calls"
            ].get(
                key,
                0,
            )
            for record in records
        ]

        summary[key][
            "mean_calls_per_frame"
        ] = float(
            np.mean(
                call_values
            )
        )


    total_mean = summary.get(
        "service_total",
        {},
    ).get(
        "mean_ms",
        None,
    )

    backbone_mean = summary.get(
        "backbone_3d",
        {},
    ).get(
        "mean_ms",
        None,
    )


    for key, item in summary.items():

        mean_ms = item.get(
            "mean_ms",
            None,
        )


        if (
            mean_ms is not None
            and
            total_mean is not None
            and
            total_mean > 0
        ):
            item[
                "pct_of_service_mean"
            ] = float(
                100.0
                *
                mean_ms
                /
                total_mean
            )
        else:
            item[
                "pct_of_service_mean"
            ] = None


        if (
            key == "backbone_3d"
            and
            backbone_mean is not None
            and
            backbone_mean > 0
        ):
            item[
                "pct_of_backbone_mean"
            ] = 100.0

        elif (
            key.startswith(
                "backbone."
            )
            and
            mean_ms is not None
            and
            backbone_mean is not None
            and
            backbone_mean > 0
        ):
            item[
                "pct_of_backbone_mean"
            ] = float(
                100.0
                *
                mean_ms
                /
                backbone_mean
            )

        else:
            item[
                "pct_of_backbone_mean"
            ] = None


    return summary


# ============================================================
# Print
# ============================================================

def print_table(
    title,
    keys,
    summary,
    pct_key,
):
    print()
    print("=" * 118)
    print(title)
    print("=" * 118)

    print(
        f"{'Component':42s}"
        f"{'Mean(ms)':>11s}"
        f"{'P50':>11s}"
        f"{'P90':>11s}"
        f"{'P99':>11s}"
        f"{'%':>9s}"
        f"{'Calls/f':>11s}"
    )

    print("-" * 118)

    for key in keys:

        if key not in summary:
            continue

        item = summary[key]

        mean_ms = item[
            "mean_ms"
        ]

        if mean_ms is None:
            continue

        pct = item.get(
            pct_key,
            None,
        )

        if pct is None:
            pct_text = "-"
        else:
            pct_text = (
                f"{pct:.2f}"
            )

        print(
            f"{key:42s}"
            f"{mean_ms:11.3f}"
            f"{item['p50_ms']:11.3f}"
            f"{item['p90_ms']:11.3f}"
            f"{item['p99_ms']:11.3f}"
            f"{pct_text:>9s}"
            f"{item['mean_calls_per_frame']:11.2f}"
        )


def save_summary_csv(
    path,
    summary,
):
    fields = [
        "component",
        "count",
        "mean_ms",
        "std_ms",
        "p50_ms",
        "p90_ms",
        "p99_ms",
        "min_ms",
        "max_ms",
        "mean_calls_per_frame",
        "pct_of_service_mean",
        "pct_of_backbone_mean",
    ]

    with path.open(
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for key in sorted(
            summary.keys()
        ):

            row = {
                "component": key,
            }

            row.update(
                summary[key]
            )

            writer.writerow(
                {
                    field: row.get(
                        field,
                        None,
                    )
                    for field in fields
                }
            )


# ============================================================
# Warmup
# ============================================================

def warmup_model(
    model,
    dataset,
    scene_to_indices,
    warmup,
):
    if warmup <= 0:
        return

    print()
    print(
        f"[warmup] chronological forwards: {warmup}"
    )

    done = 0

    with torch.no_grad():

        for scene, indices in (
            scene_to_indices.items()
        ):

            clear_history(
                model
            )

            for dataset_index in indices:

                batch_dict = load_one(
                    dataset,
                    dataset_index,
                )

                torch.cuda.synchronize()

                model(
                    batch_dict
                )

                torch.cuda.synchronize()

                done += 1

                if (
                    done % 10 == 0
                    or
                    done == warmup
                ):
                    print(
                        f"[warmup] "
                        f"{done}/{warmup}"
                    )

                if done >= warmup:

                    clear_history(
                        model
                    )

                    return

    clear_history(
        model
    )


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required."
        )

    set_seed(
        args.seed
    )


    # --------------------------------------------------------
    # Config
    # --------------------------------------------------------

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    cfg.TAG = Path(
        args.cfg_file
    ).stem


    # Modern torchvision compatibility:
    # full checkpoint is loaded immediately afterwards.
    if hasattr(
        cfg.MODEL,
        "BACKBONE_3D",
    ):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None


    # Direct timing only.
    cfg.DATA_CONFIG.INFER_TIME_PATH = None


    # Force normal forward_test().
    cfg.MODEL.SAVE_TIME = False


    logger = common_utils.create_logger()

    logger.info(
        f"cfg_file: {args.cfg_file}"
    )

    logger.info(
        f"ckpt: {args.ckpt}"
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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
            "Empty test dataset."
        )


    scene_to_indices = build_scene_index(
        dataset
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Profiler
    # --------------------------------------------------------

    profiler = CudaEventProfiler()

    register_components(
        model,
        profiler,
    )

    install_post_processing_timer(
        model,
        profiler,
    )


    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print("=" * 88)
    print(
        f"StreamDSGN {args.variant_name} Component Profiler"
    )
    print("=" * 88)

    print(
        "GPU                 : "
        f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
    )

    print(
        f"Config              : "
        f"{args.cfg_file}"
    )

    print(
        f"Checkpoint          : "
        f"{args.ckpt}"
    )

    print(
        f"Dataset frames      : "
        f"{len(dataset)}"
    )

    print(
        f"Scenes              : "
        f"{len(scene_to_indices)}"
    )

    print(
        f"Warmup forwards     : "
        f"{args.warmup}"
    )

    print(
        "Profile samples     : "
        f"{args.num_samples if args.num_samples > 0 else 'ALL'}"
    )

    print(
        f"Skip scene prefix   : "
        f"{args.skip_scene_prefix}"
    )

    print(
        "Timing boundary     : "
        "model forward + post-processing; "
        "loading/preprocess/H2D excluded"
    )

    print(
        "History semantics   : "
        "chronological inference; "
        "first scene frames populate the K-slot queue"
    )

    print()
    print(
        "Registered probes:"
    )

    for label in profiler.registered:
        print(
            f"  - {label}"
        )

    print("=" * 88)


    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    warmup_model(
        model=model,
        dataset=dataset,
        scene_to_indices=scene_to_indices,
        warmup=args.warmup,
    )


    # --------------------------------------------------------
    # Measurement
    # --------------------------------------------------------

    records = []

    target = (
        args.num_samples
        if args.num_samples > 0
        else None
    )

    stop = False


    with torch.no_grad():

        for scene, indices in (
            scene_to_indices.items()
        ):

            clear_history(
                model
            )


            for scene_pos, dataset_index in (
                enumerate(indices)
            ):

                batch_dict = load_one(
                    dataset,
                    dataset_index,
                )


                collect = (
                    scene_pos
                    >=
                    args.skip_scene_prefix
                )


                if collect:

                    history_depth_before = len(
                        getattr(
                            model,
                            "history_feature_queue",
                            [],
                        )
                    )


                    # Exclude H2D and all previous GPU work.
                    torch.cuda.synchronize()


                    profiler.start_frame()


                    total_start = torch.cuda.Event(
                        enable_timing=True
                    )

                    total_end = torch.cuda.Event(
                        enable_timing=True
                    )


                    total_start.record()


                    model(
                        batch_dict
                    )


                    total_end.record()


                    # Every event is complete before elapsed_time().
                    torch.cuda.synchronize()


                    times_ms, calls = (
                        profiler.finish_frame()
                    )


                    times_ms[
                        "service_total"
                    ] = float(
                        total_start.elapsed_time(
                            total_end
                        )
                    )


                    add_derived_metrics(
                        times_ms
                    )


                    record = {

                        "scene": str(
                            scene
                        ),

                        "scene_pos": int(
                            scene_pos
                        ),

                        "dataset_index": int(
                            dataset_index
                        ),

                        "frame_id": frame_token(
                            dataset,
                            dataset_index,
                        ),

                        "history_depth_before": int(
                            history_depth_before
                        ),

                        "times_ms": times_ms,

                        "calls": calls,
                    }


                    records.append(
                        record
                    )


                    n = len(
                        records
                    )


                    if (
                        n == 1
                        or
                        n % 50 == 0
                    ):
                        print(
                            f"[profile] "
                            f"{n} frames | "
                            f"service="
                            f"{times_ms['service_total']:.3f} ms | "
                            f"backbone="
                            f"{times_ms.get('backbone_3d', 0.0):.3f} ms | "
                            f"state_gen="
                            f"{times_ms.get('state_generation', 0.0):.3f} ms"
                        )


                    if (
                        target is not None
                        and
                        len(records) >= target
                    ):
                        stop = True
                        break


                else:

                    # Scene-prefix frames are still processed so that
                    # t-3/t-2/t-1 are actually present for measured frames.
                    model(
                        batch_dict
                    )


            if stop:
                break


    clear_history(
        model
    )


    if len(records) == 0:
        raise RuntimeError(
            "No frames were profiled. "
            "Reduce --skip_scene_prefix."
        )


    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------

    summary = aggregate_records(
        records
    )


    top_level_keys = [

        "service_total",

        "state_generation",

        "backbone_3d",

        "height_compression",

        "state_consumption",

        "temporal_fusion",

        "van_backbone",

        "spatial_feature_2d_fusion",

        "dense_head_2d",

        "dense_head",

        "post_processing",

        "service.other",
    ]


    print_table(
        title=(
            "TOP LEVEL / MODEL SERVICE "
            "(% = percent of mean service time)"
        ),
        keys=top_level_keys,
        summary=summary,
        pct_key="pct_of_service_mean",
    )


    backbone_keys = [

        "backbone_3d",

        "backbone.2d_feature_extraction",

        "backbone.feature_backbone",

        "backbone.resnet.stem",
        "backbone.resnet.stem.conv1",
        "backbone.resnet.stem.norm1",
        "backbone.resnet.stem.relu",
        "backbone.resnet.stem.maxpool",

        "backbone.resnet.stages",
        "backbone.resnet.layer1",
        "backbone.resnet.layer2",
        "backbone.resnet.layer3",
        "backbone.resnet.layer4",
        "backbone.resnet.other_functional",

        "backbone.feature_neck",

        "backbone.neck.spp_branches",
        "backbone.neck.upconv_module",
        "backbone.neck.lastconv",
        "backbone.neck.upconv_module_voxel",
        "backbone.neck.rpnconv",
        "backbone.neck.accounted",
        "backbone.neck.other_functional",

        "backbone.sem_neck",

        "backbone.stereo_matching_modules",

        "backbone.compute_disp_channels",

        "backbone.build_cost",

        "backbone.dres0",

        "backbone.dres1",

        "backbone.hg_stereo",

        "backbone.pred_stereo_module",

        "backbone.get_local_depth",

        "backbone.compute_mapping",

        "backbone.rpn3d",

        "backbone.rpn3d_convs",

        "backbone.rpn3d_hgs",

        "backbone.rpn3d_pool",

        "backbone.accounted",

        "backbone.other_functional",
    ]


    print_table(
        title=(
            "StreamDSGN2Backbone BREAKDOWN "
            "(% = percent of mean backbone_3d time)"
        ),
        keys=backbone_keys,
        summary=summary,
        pct_key="pct_of_backbone_mean",
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    summary_path = (
        output_dir
        /
        "profile_summary.json"
    )

    frames_path = (
        output_dir
        /
        "profile_frames.json"
    )

    csv_path = (
        output_dir
        /
        "profile_summary.csv"
    )


    payload = {

        "config": args.cfg_file,

        "checkpoint": args.ckpt,

        "gpu": torch.cuda.get_device_name(
            torch.cuda.current_device()
        ),

        "warmup": int(
            args.warmup
        ),

        "num_profiled_frames": int(
            len(records)
        ),

        "skip_scene_prefix": int(
            args.skip_scene_prefix
        ),

        "timing_scope": (
            "model forward + post-processing; "
            "loading/preprocess/H2D excluded"
        ),

        "summary": summary,
    }


    with summary_path.open(
        "w"
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
        )


    with frames_path.open(
        "w"
    ) as f:
        json.dump(
            records,
            f,
            indent=2,
        )


    save_summary_csv(
        csv_path,
        summary,
    )


    print()
    print("=" * 88)

    print(
        f"Profiled frames : "
        f"{len(records)}"
    )

    print(
        f"Summary JSON    : "
        f"{summary_path}"
    )

    print(
        f"Per-frame JSON  : "
        f"{frames_path}"
    )

    print(
        f"Summary CSV     : "
        f"{csv_path}"
    )


    other_mean = summary.get(
        "backbone.other_functional",
        {},
    ).get(
        "mean_ms",
        None,
    )


    if (
        other_mean is not None
        and
        other_mean < -0.5
    ):
        print()
        print(
            "WARNING: backbone.other_functional is substantially "
            "negative. This indicates overlapping probes or "
            "excessive timing noise."
        )


    print("=" * 88)


if __name__ == "__main__":
    main()
