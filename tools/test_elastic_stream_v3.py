#!/usr/bin/env python3

import argparse
import copy
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v3 import (
    ElasticBEVBranch,
    MONOTONIC_SCHEDULES,
    WIDTH_CHOICES,
    cached_full_2d_prefix,
    extract_fixed_layer1_prefix,
    extract_full_2d_from_layer1,
    validate_schedule,
)

from test_stream_buffer_timestamp import (
    attach_timestamp,
    build_scene_index,
    frame_token,
    load_one,
    strip_eval_metadata,
    timestamp_align_scene,
)


torch.backends.cudnn.benchmark = True

STAGE_NAMES = ("res2", "res3", "res4", "fpn", "stereo", "rpn")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Timestamp-aligned capacity-1 streaming evaluation for Elastic-v3: "
            "frozen ResNet stem+layer1 timing probe, elastic ResNet layer2-4, "
            "elastic FPN/stereo/RPN, frozen K3 fusion/head."
        )
    )
    parser.add_argument("--full_cfg", required=True)
    parser.add_argument("--full_ckpt", required=True)
    parser.add_argument("--elastic_ckpt", required=True)
    parser.add_argument("--input_hz", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--profile_runs", type=int, default=1)
    parser.add_argument("--safety", type=float, default=1.12)
    parser.add_argument("--deadline_periods", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=("dynamic", "baseline_full", "elastic_fixed"),
        default="dynamic",
    )
    parser.add_argument(
        "--fixed_schedule",
        type=str,
        default="0.5,0.5,0.5,0.5,0.5,0.5",
        help=(
            "Six ratios: Res2,Res3,Res4,FPN,Stereo,RPN. "
            "Example 1,0.75,0.75,0.5,0.5,0.25"
        ),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_cfg(path):
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = Path(path).stem
    cfg.DATA_CONFIG.INFER_TIME_PATH = None
    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None
    return cfg


def parse_schedule(text):
    return validate_schedule(tuple(float(x.strip()) for x in text.split(",")))


def sync_time_call(fn):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter_ns() - start) / 1e6


def run_elastic_downstream(base_model, batch_dict, bev, valids):
    cur_data = batch_dict["token"]
    cur_data["spatial_features"] = bev
    cur_data["spatial_features_stride"] = 1
    cur_data["valids"] = valids
    cur_data["history_features"] = base_model.history_feature_queue

    history_feature = {"spatial_features": bev.detach().clone()}

    for module in base_model.fusion_module:
        cur_data = module(cur_data)
    for module in base_model.after_fusion_blocks:
        cur_data = module(cur_data)

    pred_dicts, ret_dicts = base_model.post_processing(cur_data)
    if base_model.history_feature_queue is not None:
        base_model.history_feature_queue.append(
            (cur_data["this_sample_idx"], history_feature)
        )
    return pred_dicts, ret_dicts


def fill_three_history(queue, bev):
    if queue is None:
        return
    queue.clear()
    for i in range(3):
        queue.append((f"profile_{i}", {"spatial_features": bev.detach().clone()}))


def full_bev_after_prefix(base_model, frame_dict, prefix_cache):
    backbone = base_model.backbone_3d
    full_2d = extract_full_2d_from_layer1(backbone, frame_dict, prefix_cache)
    data = copy.copy(frame_dict)
    with cached_full_2d_prefix(backbone, full_2d):
        data = backbone(data)
    data = base_model.map_to_bev_module(data)
    return data["spatial_features"], full_2d


def run_full_after_prefix(base_model, batch_dict, prefix_cache):
    backbone = base_model.backbone_3d
    full_2d = extract_full_2d_from_layer1(
        backbone, batch_dict["token"], prefix_cache
    )
    with cached_full_2d_prefix(backbone, full_2d):
        return base_model(batch_dict)


def _key_res2(schedule):
    return str(schedule[0])


def _key_res3(schedule):
    return f"{schedule[0]}>{schedule[1]}"


def _key_res4(schedule):
    return f"{schedule[1]}>{schedule[2]}"


def _key_fpn(schedule):
    return ">".join(str(x) for x in schedule[:4])


def _key_stereo(schedule):
    return f"{schedule[3]}>{schedule[4]}"


def _key_rpn(schedule):
    return f"{schedule[4]}>{schedule[5]}"


def profile_runtime(base_model, branch, dataset, dataset_index, warmup, runs):
    backbone = base_model.backbone_3d
    amp_enabled = bool(base_model.use_amp_dict["TEST"])
    profile_schedule = (0.5,) * 6

    # Warmup both original Full and Elastic.
    for _ in range(max(1, warmup)):
        if base_model.history_feature_queue is not None:
            base_model.history_feature_queue.clear()
        batch = load_one(dataset, dataset_index)
        with torch.no_grad():
            base_model(batch)

        if base_model.history_feature_queue is not None:
            base_model.history_feature_queue.clear()
        batch = load_one(dataset, dataset_index)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            prefix = extract_fixed_layer1_prefix(backbone, batch["token"])
            bev, valids = branch(
                batch["token"], prefix, backbone, profile_schedule
            )
            fill_three_history(base_model.history_feature_queue, bev)
            run_elastic_downstream(base_model, batch, bev, valids)
    torch.cuda.synchronize()

    prefix_samples = []
    full_remaining_samples = []
    post_samples = []

    # Full remaining time after the fixed stem+layer1 timing prefix.
    for _ in range(max(2, runs * 2)):
        if base_model.history_feature_queue is not None:
            base_model.history_feature_queue.clear()
        batch = load_one(dataset, dataset_index)

        def prefix_call():
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                return extract_fixed_layer1_prefix(backbone, batch["token"])

        prefix, prefix_ms = sync_time_call(prefix_call)
        prefix_samples.append(prefix_ms)

        # Populate three history entries so downstream timing is representative.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            bev, _ = full_bev_after_prefix(base_model, batch["token"], prefix)
        fill_three_history(base_model.history_feature_queue, bev)

        def full_remaining_call():
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                return run_full_after_prefix(base_model, batch, prefix)

        _, remain_ms = sync_time_call(full_remaining_call)
        full_remaining_samples.append(remain_ms)

    samples = {
        "res2": defaultdict(list),
        "res3": defaultdict(list),
        "res4": defaultdict(list),
        "fpn": defaultdict(list),
        "stereo": defaultdict(list),
        "rpn": defaultdict(list),
    }

    # One pass over all 84 monotonic schedules already gives repeated samples
    # for the cheaper stage keys. profile_runs>1 repeats the whole table.
    for _ in range(max(1, runs)):
        for schedule in MONOTONIC_SCHEDULES:
            if base_model.history_feature_queue is not None:
                base_model.history_feature_queue.clear()
            batch = load_one(dataset, dataset_index)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                prefix = extract_fixed_layer1_prefix(backbone, batch["token"])

                state, t = sync_time_call(
                    lambda: branch.stage_res2(prefix, schedule[0])
                )
                samples["res2"][_key_res2(schedule)].append(t)

                state, t = sync_time_call(
                    lambda: branch.stage_res3(state, schedule[1])
                )
                samples["res3"][_key_res3(schedule)].append(t)

                state, t = sync_time_call(
                    lambda: branch.stage_res4(state, schedule[2])
                )
                samples["res4"][_key_res4(schedule)].append(t)

                state, t = sync_time_call(
                    lambda: branch.stage_fpn(batch["token"], state, schedule[3])
                )
                samples["fpn"][_key_fpn(schedule)].append(t)

                stereo, t = sync_time_call(
                    lambda: branch.stage_stereo(
                        batch["token"], state, backbone, schedule[4]
                    )
                )
                samples["stereo"][_key_stereo(schedule)].append(t)

                (bev, valids), t = sync_time_call(
                    lambda: branch.stage_rpn(
                        batch["token"], stereo, backbone, schedule[5]
                    )
                )
                samples["rpn"][_key_rpn(schedule)].append(t)

                fill_three_history(base_model.history_feature_queue, bev)
                _, post_ms = sync_time_call(
                    lambda: run_elastic_downstream(
                        base_model, batch, bev, valids
                    )
                )
                post_samples.append(post_ms)

    profile = {
        "fixed_prefix_ms": float(np.median(prefix_samples)),
        "full_remaining_ms": float(np.median(full_remaining_samples)),
        "post_ms": float(np.median(post_samples)),
    }
    for stage in STAGE_NAMES:
        profile[f"{stage}_ms"] = {
            key: float(np.median(values))
            for key, values in samples[stage].items()
        }

    if base_model.history_feature_queue is not None:
        base_model.history_feature_queue.clear()
    return profile


def stage_reference(profile, stage_index, schedule):
    if stage_index == 0:
        return profile["res2_ms"][_key_res2(schedule)]
    if stage_index == 1:
        return profile["res3_ms"][_key_res3(schedule)]
    if stage_index == 2:
        return profile["res4_ms"][_key_res4(schedule)]
    if stage_index == 3:
        return profile["fpn_ms"][_key_fpn(schedule)]
    if stage_index == 4:
        return profile["stereo_ms"][_key_stereo(schedule)]
    if stage_index == 5:
        return profile["rpn_ms"][_key_rpn(schedule)]
    raise IndexError(stage_index)


def remaining_reference(profile, schedule, next_stage_index):
    total = 0.0
    for stage_index in range(next_stage_index, 6):
        total += stage_reference(profile, stage_index, schedule)
    total += profile["post_ms"]
    return total


def schedule_score(schedule):
    # Earlier stages influence a larger fraction of the representation, so
    # ties prefer keeping them wide. The lexicographic suffix makes selection
    # deterministic.
    weighted = sum((6 - i) * r for i, r in enumerate(schedule))
    return (weighted, *schedule)


class ElasticRuntimeModel(nn.Module):
    def __init__(
        self,
        base_model,
        branch,
        profile,
        deadline_ms,
        safety,
        mode,
        fixed_schedule,
    ):
        super().__init__()
        self.base_model = base_model
        self.branch = branch
        self.profile = profile
        self.deadline_ms = float(deadline_ms)
        self.safety = float(safety)
        self.mode = mode
        self.fixed_schedule = validate_schedule(fixed_schedule)
        self.initial_age_ms = 0.0
        self.last_runtime_meta = {}

    @property
    def history_feature_queue(self):
        return self.base_model.history_feature_queue

    def effective_deadline(self):
        return max(self.deadline_ms - float(self.initial_age_ms), 0.0)

    def _choose_schedule(self, fixed_prefix, elapsed_ms, rho, next_stage):
        feasible = []
        for schedule in MONOTONIC_SCHEDULES:
            if any(
                schedule[i] != fixed_prefix[i]
                for i in range(len(fixed_prefix))
            ):
                continue
            estimate = elapsed_ms + self.safety * max(rho, 1.0) * remaining_reference(
                self.profile, schedule, next_stage
            )
            if estimate <= self.effective_deadline():
                feasible.append(schedule)
        if feasible:
            return max(feasible, key=schedule_score)

        # Deadline cannot be met even by the profiled minimum suffix. Preserve
        # already executed ratios and immediately minimize all remaining stages.
        schedule = list(fixed_prefix)
        previous = schedule[-1] if schedule else 1.0
        while len(schedule) < 6:
            next_ratio = min(previous, 0.25)
            schedule.append(next_ratio)
            previous = next_ratio
        return tuple(schedule)

    @staticmethod
    def _update_rho(old_rho, actual_ms, reference_ms):
        observed = actual_ms / max(reference_ms, 1e-6)
        # Smooth noisy per-stage measurements but never assume the GPU is
        # faster than its nominal profile when making a deadline decision.
        return max(1.0, 0.5 * float(old_rho) + 0.5 * float(observed))

    def _run_elastic(self, batch_dict, prefix_cache, prefix_ms):
        backbone = self.base_model.backbone_3d
        amp_enabled = bool(self.base_model.use_amp_dict["TEST"])

        # Fixed-width evaluation must measure the actual end-to-end model path,
        # not the dynamic controller.  In particular, do NOT synchronize after
        # every elastic stage and do NOT access the dynamic timing profile.
        # Otherwise fixed-width sAP/service time would include six artificial
        # CPU/GPU synchronization barriers and the empty fixed-mode profile
        # would also trigger KeyError.
        if self.mode == "elastic_fixed":
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                bev, valids = self.branch(
                    batch_dict["token"],
                    prefix_cache,
                    backbone,
                    self.fixed_schedule,
                )
                pred_dicts, ret_dicts = run_elastic_downstream(
                    self.base_model, batch_dict, bev, valids
                )

            self.last_runtime_meta = {
                "branch": "elastic_fixed",
                "schedule": [float(x) for x in self.fixed_schedule],
                "initial_age_ms": float(self.initial_age_ms),
                "effective_deadline_ms": float(self.effective_deadline()),
            }
            return pred_dicts, ret_dicts

        # From here onward the only valid elastic mode is dynamic.  Dynamic
        # execution intentionally synchronizes at each checkpoint because the
        # CPU controller needs the just-finished stage time before selecting
        # the remaining suffix.
        elapsed = float(prefix_ms)
        rho = max(
            1.0,
            prefix_ms / max(self.profile["fixed_prefix_ms"], 1e-6),
        )
        schedule = self._choose_schedule((), elapsed, rho, 0)

        stage_times = []
        chosen = list(schedule)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            state, t = sync_time_call(
                lambda: self.branch.stage_res2(prefix_cache, chosen[0])
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["res2_ms"][_key_res2(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:1]), elapsed, rho, 1))

            state, t = sync_time_call(
                lambda: self.branch.stage_res3(state, chosen[1])
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["res3_ms"][_key_res3(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:2]), elapsed, rho, 2))

            state, t = sync_time_call(
                lambda: self.branch.stage_res4(state, chosen[2])
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["res4_ms"][_key_res4(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:3]), elapsed, rho, 3))

            state, t = sync_time_call(
                lambda: self.branch.stage_fpn(
                    batch_dict["token"], state, chosen[3]
                )
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["fpn_ms"][_key_fpn(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:4]), elapsed, rho, 4))

            stereo, t = sync_time_call(
                lambda: self.branch.stage_stereo(
                    batch_dict["token"], state, backbone, chosen[4]
                )
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["stereo_ms"][_key_stereo(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:5]), elapsed, rho, 5))

            (bev, valids), t = sync_time_call(
                lambda: self.branch.stage_rpn(
                    batch_dict["token"], stereo, backbone, chosen[5]
                )
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["rpn_ms"][_key_rpn(tuple(chosen))]
            )

            pred_dicts, ret_dicts = run_elastic_downstream(
                self.base_model, batch_dict, bev, valids
            )

        self.last_runtime_meta = {
            "branch": "elastic",
            "schedule": [float(x) for x in chosen],
            "fixed_prefix_ms": float(prefix_ms),
            "stage_ms": {
                name: float(value) for name, value in zip(STAGE_NAMES, stage_times)
            },
            "slowdown": float(rho),
            "initial_age_ms": float(self.initial_age_ms),
            "effective_deadline_ms": float(self.effective_deadline()),
        }
        return pred_dicts, ret_dicts

    def forward(self, batch_dict):
        cur_data = batch_dict["token"]
        if (
            self.base_model.history_feature_queue is not None
            and ("prev_sample_idx" not in cur_data or cur_data["prev_sample_idx"] == "")
        ):
            self.base_model.history_feature_queue.clear()

        if self.mode == "baseline_full":
            with torch.no_grad():
                pred_dicts, ret_dicts = self.base_model(batch_dict)
            self.last_runtime_meta = {
                "branch": "baseline_full",
                "schedule": None,
                "initial_age_ms": float(self.initial_age_ms),
            }
            return pred_dicts, ret_dicts

        backbone = self.base_model.backbone_3d
        amp_enabled = bool(self.base_model.use_amp_dict["TEST"])

        # Fixed-width testing has no online timing decision.  Keep the frozen
        # stem+layer1 in the normal CUDA stream so its latency is counted once
        # by the outer end-to-end service timer, without an artificial sync.
        if self.mode == "elastic_fixed":
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                prefix_cache = extract_fixed_layer1_prefix(backbone, cur_data)
            return self._run_elastic(batch_dict, prefix_cache, 0.0)

        # Dynamic mode needs a real in-band observation after layer1.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            prefix_cache, prefix_ms = sync_time_call(
                lambda: extract_fixed_layer1_prefix(backbone, cur_data)
            )

        if self.mode == "dynamic":
            rho = max(
                1.0,
                prefix_ms / max(self.profile["fixed_prefix_ms"], 1e-6),
            )
            estimated_full = (
                prefix_ms
                + self.safety * rho * self.profile["full_remaining_ms"]
            )
            if estimated_full <= self.effective_deadline():
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred_dicts, ret_dicts = run_full_after_prefix(
                        self.base_model, batch_dict, prefix_cache
                    )
                self.last_runtime_meta = {
                    "branch": "full",
                    "schedule": None,
                    "fixed_prefix_ms": float(prefix_ms),
                    "estimated_full_ms": float(estimated_full),
                    "slowdown": float(rho),
                    "initial_age_ms": float(self.initial_age_ms),
                    "effective_deadline_ms": float(self.effective_deadline()),
                }
                return pred_dicts, ret_dicts

        return self._run_elastic(batch_dict, prefix_cache, prefix_ms)


def run_one_prediction(runtime_model, dataset, dataset_index, initial_age_ms=0.0):
    runtime_model.initial_age_ms = float(initial_age_ms)
    batch_dict = load_one(dataset, dataset_index)
    torch.cuda.synchronize()
    start_ns = time.perf_counter_ns()
    with torch.no_grad():
        pred_dicts, _ = runtime_model(batch_dict)
    torch.cuda.synchronize()
    finish_ns = time.perf_counter_ns()

    service_ms = (finish_ns - start_ns) / 1e6
    wall_finish_ns = time.time_ns()
    annos = dataset.generate_prediction_dicts(
        batch_dict, pred_dicts, dataset.class_names, output_path=None
    )
    if len(annos) != 1:
        raise RuntimeError("Streaming evaluator requires batch_size=1")
    return (
        annos[0],
        service_ms,
        wall_finish_ns,
        copy.deepcopy(runtime_model.last_runtime_meta),
    )


def run_scene(runtime_model, dataset, scene, indices, period_ms, trace, processed_predictions):
    if runtime_model.history_feature_queue is not None:
        runtime_model.history_feature_queue.clear()
    n = len(indices)
    if n == 0:
        return [], 0, 0

    pos = 0
    virtual_now_ms = 0.0
    processed_count = 0
    dropped_count = 0
    scene_outputs = []

    while pos < n:
        dataset_index = indices[pos]
        input_frame_id = frame_token(dataset, dataset_index)
        arrival_ms = pos * period_ms
        start_ms = max(virtual_now_ms, arrival_ms)

        anno, service_ms, wall_finish_ns, runtime_meta = run_one_prediction(
            runtime_model,
            dataset,
            dataset_index,
            initial_age_ms=(start_ms - arrival_ms),
        )
        finish_ms = start_ms + service_ms

        stamped = attach_timestamp(
            anno=anno,
            scene=scene,
            input_frame_id=input_frame_id,
            arrival_ms=arrival_ms,
            start_ms=start_ms,
            finish_ms=finish_ms,
            service_ms=service_ms,
            wall_finish_ns=wall_finish_ns,
        )
        stamped["_stream_runtime_meta"] = runtime_meta
        processed_predictions.append(stamped)
        scene_outputs.append(stamped)
        processed_count += 1

        latest_arrived_pos = pos
        probe = pos + 1
        while probe < n and probe * period_ms <= finish_ms:
            latest_arrived_pos = probe
            probe += 1

        if latest_arrived_pos > pos:
            next_pos = latest_arrived_pos
            dropped_now = max(0, next_pos - pos - 1)
            next_reason = "buffer_latest"
        else:
            next_pos = pos + 1
            dropped_now = 0
            next_reason = "wait_next_arrival"

        dropped_count += dropped_now
        trace.append(
            {
                "scene": scene,
                "dataset_index": int(dataset_index),
                "frame_id": input_frame_id,
                "scene_pos": int(pos),
                "arrival_ms": float(arrival_ms),
                "start_ms": float(start_ms),
                "finish_ms": float(finish_ms),
                "service_ms": float(service_ms),
                "buffer_wait_ms": float(start_ms - arrival_ms),
                "dropped_waiting_frames_after_this_output": int(dropped_now),
                "next_scene_pos": int(next_pos) if next_pos < n else None,
                "next_reason": next_reason if next_pos < n else "scene_end",
                "wall_output_time_ns": int(wall_finish_ns),
                "runtime": runtime_meta,
            }
        )

        print(
            f"[{scene}] frame={input_frame_id} pos={pos:03d}/{n-1:03d} "
            f"service={service_ms:7.3f}ms drop+={dropped_now} "
            f"branch={runtime_meta.get('branch')} "
            f"width={runtime_meta.get('schedule')}"
        )

        virtual_now_ms = finish_ms
        pos = next_pos

    return scene_outputs, processed_count, dropped_count


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)

    full_cfg = make_cfg(args.full_cfg)
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(
        dataset_cfg=full_cfg.DATA_CONFIG,
        class_names=full_cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("Empty test dataset")

    base_model = build_network(
        model_cfg=full_cfg.MODEL,
        num_class=len(full_cfg.CLASS_NAMES),
        dataset=dataset,
    )
    base_model.load_params_from_file(
        filename=args.full_ckpt, logger=logger, to_cpu=True
    )
    base_model.cuda().eval()

    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(full_cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES),
    ).cuda().eval()
    checkpoint = torch.load(args.elastic_ckpt, map_location="cpu")
    branch.load_state_dict(checkpoint["branch"], strict=True)

    fixed_schedule = parse_schedule(args.fixed_schedule)
    period_ms = 1000.0 / float(args.input_hz)
    deadline_ms = float(args.deadline_periods) * period_ms
    scene_to_indices = build_scene_index(dataset)
    first_scene = next(iter(scene_to_indices))
    first_scene_indices = scene_to_indices[first_scene]
    first_index = first_scene_indices[min(5, len(first_scene_indices) - 1)]

    if args.mode == "dynamic":
        print("Profiling six-stage elastic timing table...")
        profile = profile_runtime(
            base_model,
            branch,
            dataset,
            first_index,
            args.warmup,
            args.profile_runs,
        )
    else:
        profile = {
            "fixed_prefix_ms": 1.0,
            "full_remaining_ms": 1.0,
            "post_ms": 1.0,
            **{f"{stage}_ms": {} for stage in STAGE_NAMES},
        }

    runtime_model = ElasticRuntimeModel(
        base_model=base_model,
        branch=branch,
        profile=profile,
        deadline_ms=deadline_ms,
        safety=args.safety,
        mode=args.mode,
        fixed_schedule=fixed_schedule,
    ).cuda().eval()

    print("=" * 80)
    print("Elastic-v3 K3 streaming evaluation")
    print("fixed prefix      : original ResNet stem + layer1")
    print("elastic stages    : Res2, Res3, Res4, FPN, Stereo3D, RPN3D")
    print(f"mode              : {args.mode}")
    print(f"input_hz          : {args.input_hz}")
    print(f"period_ms         : {period_ms:.3f}")
    print(f"deadline_ms       : {deadline_ms:.3f}")
    print(f"safety            : {args.safety:.3f}")
    print(f"fixed_schedule    : {fixed_schedule}")
    print("=" * 80)

    output_dir = Path(args.output_dir) / full_cfg.TAG / f"{args.input_hz:g}Hz"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "dynamic":
        with open(output_dir / "timing_profile.json", "w") as f:
            json.dump(profile, f, indent=2)

    trace = []
    processed_predictions = []
    outputs_by_scene = {}
    total_processed = 0
    total_dropped = 0

    for scene, indices in scene_to_indices.items():
        outputs, processed, dropped = run_scene(
            runtime_model,
            dataset,
            scene,
            indices,
            period_ms,
            trace,
            processed_predictions,
        )
        outputs_by_scene[scene] = outputs
        total_processed += processed
        total_dropped += dropped

    all_gt_annos = []
    all_stream_det_annos = []
    for scene, indices in scene_to_indices.items():
        gt, det = timestamp_align_scene(
            dataset=dataset,
            scene=scene,
            indices=indices,
            scene_outputs=outputs_by_scene[scene],
            period_ms=period_ms,
        )
        all_gt_annos.extend(gt)
        all_stream_det_annos.extend(det)

    eval_det_annos = [
        strip_eval_metadata(copy.deepcopy(x)) for x in all_stream_det_annos
    ]
    full_result_str, _ = dataset.evaluation_offline(
        all_gt_annos, eval_det_annos, dataset.class_names, "3d"
    )
    paper_result_str = eval_utils.format_paper_metrics(full_result_str)

    with open(output_dir / "timeline.json", "w") as f:
        json.dump(trace, f, indent=2)
    with open(output_dir / "processed_predictions_timestamped.pkl", "wb") as f:
        pickle.dump(processed_predictions, f)
    with open(output_dir / "sap_timestamp_aligned_predictions.pkl", "wb") as f:
        pickle.dump(all_stream_det_annos, f)
    with open(output_dir / "paper_sap.txt", "w") as f:
        f.write(paper_result_str + "\n")

    service = np.asarray([x["service_ms"] for x in trace], dtype=np.float64)
    total_sensor_frames = sum(len(x) for x in scene_to_indices.values())
    branch_counts = defaultdict(int)
    schedule_counts = defaultdict(int)
    deadline_miss = 0
    for x in trace:
        runtime = x["runtime"]
        branch_counts[str(runtime.get("branch"))] += 1
        if runtime.get("schedule") is not None:
            schedule_counts[str(runtime["schedule"])] += 1
        if x["buffer_wait_ms"] + x["service_ms"] > deadline_ms:
            deadline_miss += 1

    summary = {
        "mode": args.mode,
        "full_cfg": args.full_cfg,
        "full_ckpt": args.full_ckpt,
        "elastic_ckpt": args.elastic_ckpt,
        "input_hz": args.input_hz,
        "period_ms": period_ms,
        "deadline_ms": deadline_ms,
        "sensor_frames": total_sensor_frames,
        "processed_frames": total_processed,
        "dropped_frames": total_dropped,
        "process_rate": total_processed / max(total_sensor_frames, 1),
        "drop_rate": total_dropped / max(total_sensor_frames, 1),
        "deadline_miss_count": deadline_miss,
        "deadline_miss_rate": deadline_miss / max(total_processed, 1),
        "mean_service_ms": float(service.mean()) if service.size else None,
        "p50_service_ms": float(np.percentile(service, 50)) if service.size else None,
        "p90_service_ms": float(np.percentile(service, 90)) if service.size else None,
        "p99_service_ms": float(np.percentile(service, 99)) if service.size else None,
        "branch_counts": dict(branch_counts),
        "schedule_counts": dict(schedule_counts),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print(paper_result_str)
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
