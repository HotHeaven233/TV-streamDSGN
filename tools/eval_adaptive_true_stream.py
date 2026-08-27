#!/usr/bin/env python3
"""
True-stream accuracy evaluation of the trained adaptive StreamDSGN method.

This evaluator executes the CURRENT physical adaptive path instead of using the
training surrogate:

  A_s  : full stem + layer1 (left/right), always recomputed
  Q    : trained StageImportancePredictor, before expensive selective stages
  sched: deadline-first action selection from a 5x5x5 latency LUT
  A_d  : physical selective layer2/3/4 + local feature-neck + dense A cache
  B    : physical DPS cost-volume ROI + dres0/dres1 + dense B cache
  CD   : physical C mapping/grid_sample + D 3D conv/hourglass/pool ROI
  BEV  : trained temporal predictor + Recompute/Predict/Reuse composition
  tail : FeatureAlignment -> VANBackbone -> StreamDetHead -> post_processing

Streaming semantics are identical to tools/eval_true_stream_baseline.py:

  - data preparation / load_data_to_gpu / H2D are NOT timed;
  - only the accepted frame's actual adaptive forward wall time is used as the
    simulated accelerator service time;
  - if a sensor frame arrives while the accelerator is busy, that frame is
    DROPPED and is never forwarded through the model;
  - a dropped frame therefore creates no A/B/BEV cache entry and no history BEV;
  - age maps advance across dropped sensor frames;
  - sAP uses only the newest prediction that has actually completed by each
    sensor timestamp.

Important scheduler behavior
----------------------------
If a measured Global-Full P99 fits the current deadline, Global Full is selected
without running the importance network. Full has maximum possible utility because
each normalized Q map integrates to 1 over the whole map.

If Full does not fit, the 125 selective actions are filtered by the supplied LUT
and the feasible action with maximum summed learned importance is selected.
The *actual* measured adaptive forward latency, not the LUT estimate, determines
whether later frames are dropped.

The supplied component LUT is currently provisional. This evaluator remains
honest about streaming because actual physical forward time controls busy/drop;
the provisional LUT affects only which action the scheduler chooses.
"""

import argparse

# STREAMDSGN_LATENCY_OPT_V1
import copy
import csv
import json
import math
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.models.adaptive_single_roi import (
    StageImportancePredictor,
    BEVTemporalPredictor,
    roi_hw_from_ratio,
)

# Existing repository physical primitives. When this script is run as
# `python tools/eval_adaptive_true_stream.py`, the tools directory is on sys.path.
import profile_c_roi_fragmentation as cprof
import profile_roi_fragmentation as dprof

try:
    from eval_utils.eval_utils import format_paper_metrics
except Exception:
    format_paper_metrics = None


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-"
    "lka_7-mcl_5090_eval.yaml"
)
DEFAULT_BASE_CKPT = "extra_data/checkpoint_epoch_20.pth"
DEFAULT_ADAPTIVE_CKPT = "outputs/adaptive_joint/adaptive_joint_full_10ep.pth"
DEFAULT_LUT = (
    "outputs/adaptive_profile/latency_lut_full/"
    "adaptive_latency_lut_provisional.json"
)


# =============================================================================
# Arguments
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(
        description="True-stream evaluation of trained physical adaptive StreamDSGN"
    )

    p.add_argument("--cfg_file", default=DEFAULT_CFG)
    p.add_argument("--ckpt", default=DEFAULT_BASE_CKPT)
    p.add_argument("--adaptive_ckpt", default=DEFAULT_ADAPTIVE_CKPT)
    p.add_argument("--latency_lut", default=DEFAULT_LUT)

    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--rates", default="2,3,4",
                   help="Input stream-rate multipliers: 1x=10Hz, 2x=20Hz, ...")
    p.add_argument("--fps", type=float, default=None,
                   help="Base sensor rate; defaults to DATA_CONFIG.ANNOS_FREQUENCY")

    p.add_argument("--timestamp_json", default=None)
    p.add_argument("--timestamp_unit", choices=["ms", "s"], default="ms")
    p.add_argument("--require_timestamp_json", action="store_true")

    p.add_argument("--levels", default="0.15,0.25,0.40,0.60,0.80")
    p.add_argument("--roi_align", type=int, default=4)
    p.add_argument("--b_halo", type=int, default=2)
    p.add_argument("--cd_halo", type=int, default=2)
    p.add_argument("--pred_margin", type=int, default=12)
    p.add_argument("--age_cap", type=float, default=8.0)

    p.add_argument("--lambda_a", type=float, default=1.0)
    p.add_argument("--lambda_b", type=float, default=1.0)
    p.add_argument("--lambda_cd", type=float, default=1.0)

    p.add_argument("--importance_hidden", type=int, default=32)
    p.add_argument("--predictor_hidden", type=int, default=64)

    p.add_argument(
        "--scheduler_extra_ms",
        type=float,
        default=2.0,
        help=(
            "Extra conservative scheduler guard added to every LUT action and "
            "Global-Full P99. The completed component profile had ~1.7 ms "
            "closure gap, so 2.0 ms is a conservative starting value."
        ),
    )

    p.add_argument("--warmup_frames", type=int, default=10)
    p.add_argument("--self_check_tol", type=float, default=0.02)
    p.add_argument("--skip_self_check", action="store_true")
    p.add_argument("--max_frames", type=int, default=-1,
                   help="Debug only. -1 evaluates the complete validation set.")

    p.add_argument("--out_dir", default="outputs/adaptive_true_stream")
    p.add_argument("--paper_metrics_only", action="store_true")

    return p.parse_args()


# =============================================================================
# Generic helpers
# =============================================================================


def parse_float_list(text):
    out = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not out:
        raise ValueError("empty float list")
    return out


def first_scalar(x):
    if isinstance(x, (list, tuple)):
        return first_scalar(x[0])
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return first_scalar(x.reshape(-1)[0])
    if torch.is_tensor(x):
        if x.ndim == 0:
            return x.item()
        return first_scalar(x.reshape(-1)[0])
    return x


def get_scene_frame(batch_dict):
    token = batch_dict["token"]
    scene = str(first_scalar(token["scene"]))
    frame = str(first_scalar(token["this_sample_idx"]))
    return scene, frame


def frame_key(scene, frame):
    return f"{scene}/{frame}"


def amp_enabled(model):
    d = getattr(model, "use_amp_dict", {})
    if isinstance(d, dict):
        return bool(d.get("TEST", False))
    return False


def autocast_ctx(model):
    return torch.amp.autocast("cuda", enabled=amp_enabled(model))


def find_module(model, class_name):
    for m in model.modules():
        if type(m).__name__ == class_name:
            return m
    raise RuntimeError(f"{class_name} not found")


def to_jsonable(x):
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if torch.is_tensor(x):
        if x.numel() == 1:
            return x.detach().cpu().item()
        return x.detach().cpu().tolist()
    return x


def flatten_numeric(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, p))
    elif isinstance(obj, (int, float, np.integer, np.floating)):
        out[prefix] = float(obj)
    return out


def stats(xs):
    a = np.asarray(xs, dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None,
                "p99": None, "min": None, "max": None}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def empty_prediction(scene, frame, next_frame=""):
    return {
        "name": np.zeros((0,), dtype="<U1"),
        "truncated": np.zeros((0,), dtype=np.float64),
        "occluded": np.zeros((0,), dtype=np.float64),
        "alpha": np.zeros((0,), dtype=np.float64),
        "bbox": np.zeros((0, 4), dtype=np.float64),
        "dimensions": np.zeros((0, 3), dtype=np.float64),
        "location": np.zeros((0, 3), dtype=np.float64),
        "rotation_y": np.zeros((0,), dtype=np.float64),
        "score": np.zeros((0,), dtype=np.float64),
        "boxes_lidar": np.zeros((0, 7), dtype=np.float64),
        "scene": scene,
        "frame_id": frame,
        "next_frame_id": next_frame,
    }


def retag_prediction(pred, scene, frame, next_frame=""):
    if pred is None:
        return empty_prediction(scene, frame, next_frame)
    out = copy.deepcopy(pred)
    out["scene"] = scene
    out["frame_id"] = frame
    out["next_frame_id"] = next_frame
    return out


# =============================================================================
# Timestamp / stream clock
# =============================================================================


def parse_timestamp_value(v, unit):
    if isinstance(v, (int, float)):
        x = float(v)
        return x * 1000.0 if unit == "s" else x

    s = str(v).strip()
    try:
        x = float(s)
        return x * 1000.0 if unit == "s" else x
    except ValueError:
        pass

    s = s.replace("Z", "+00:00")
    return datetime.fromisoformat(s).timestamp() * 1000.0


def load_timestamps(args):
    if args.timestamp_json is None:
        if args.require_timestamp_json:
            raise RuntimeError(
                "--require_timestamp_json was set but --timestamp_json was omitted"
            )
        return None

    raw = json.loads(Path(args.timestamp_json).read_text())
    out = {}

    if all(not isinstance(v, dict) for v in raw.values()):
        for k, v in raw.items():
            out[str(k)] = parse_timestamp_value(v, args.timestamp_unit)
        return out

    for scene, frames in raw.items():
        for frame, v in frames.items():
            out[frame_key(str(scene), str(frame))] = parse_timestamp_value(
                v, args.timestamp_unit
            )
    return out


class StreamClock:
    def __init__(self, timestamp_map, base_fps, rate_multiplier):
        self.timestamp_map = timestamp_map
        self.base_fps = float(base_fps)
        self.rate = float(rate_multiplier)
        self.scene_first_frame = {}
        self.scene_first_real_ts = {}

    @property
    def nominal_interval_ms(self):
        return 1000.0 / (self.base_fps * self.rate)

    def _raw_ts(self, scene, frame):
        key = frame_key(scene, frame)
        if self.timestamp_map is None:
            return None
        if key not in self.timestamp_map:
            raise KeyError(f"Missing physical timestamp for {key}")
        return float(self.timestamp_map[key])

    def arrival_ms(self, scene, frame):
        if self.timestamp_map is not None:
            t = self._raw_ts(scene, frame)
            if scene not in self.scene_first_real_ts:
                self.scene_first_real_ts[scene] = t
            # Replay the physical timestamp sequence rate-times faster.
            return (t - self.scene_first_real_ts[scene]) / self.rate

        f = int(frame)
        if scene not in self.scene_first_frame:
            self.scene_first_frame[scene] = f
        return (
            (f - self.scene_first_frame[scene])
            * self.nominal_interval_ms
        )

    def deadline_ms(self, scene, frame, next_frame=""):
        if self.timestamp_map is not None and next_frame:
            k0 = frame_key(scene, frame)
            k1 = frame_key(scene, next_frame)
            if k0 in self.timestamp_map and k1 in self.timestamp_map:
                dt = float(self.timestamp_map[k1]) - float(self.timestamp_map[k0])
                if dt > 0:
                    return dt / self.rate
        return self.nominal_interval_ms


# =============================================================================
# ROI helpers / learned-utility scheduler
# =============================================================================


def expand_align(rect, halo, H, W, align):
    y0, y1, x0, x1 = [int(v) for v in rect]
    y0 = max(0, y0 - int(halo))
    y1 = min(H, y1 + int(halo))
    x0 = max(0, x0 - int(halo))
    x1 = min(W, x1 + int(halo))

    y0 = (y0 // align) * align
    x0 = (x0 // align) * align
    y1 = min(H, int(math.ceil(y1 / align) * align))
    x1 = min(W, int(math.ceil(x1 / align) * align))

    if y1 <= y0 or x1 <= x0:
        raise RuntimeError(f"invalid expanded ROI: {(y0, y1, x0, x1)}")
    return (y0, y1, x0, x1)


def stage_plan_gpu(q, levels, align):
    # STREAMDSGN_LATENCY_OPT_V1: one integral image per Q map.
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 1:
        raise RuntimeError(f"Expected Q [1,1,H,W], got {tuple(q.shape)}")
    H, W = q.shape[-2:]
    qf = q.float()
    integ = F.pad(qf, (1, 0, 1, 0), mode="constant", value=0.0)
    integ = integ.cumsum(dim=-2).cumsum(dim=-1)
    plans = []
    for level, ratio in enumerate(levels):
        h, w = roi_hw_from_ratio(H, W, ratio, align=align)
        score = (
            integ[..., h:, w:] - integ[..., :-h, w:]
            - integ[..., h:, :-w] + integ[..., :-h, :-w]
        )
        flat = score.flatten(1)
        value, idx = flat.max(dim=1)
        plans.append({
            "level": level, "ratio": float(ratio), "h": int(h), "w": int(w),
            "Wv": int(score.shape[-1]), "value_gpu": value[0], "idx_gpu": idx[0],
        })
    return plans


def finalize_stage_plans(groups):
    """Transfer all 15 scalar utilities/indices with one synchronization point."""
    flat = []
    for name in ("A", "B", "CD"):
        for rec in groups[name]:
            flat.append((name, rec))

    values = torch.stack([rec["value_gpu"] for _, rec in flat])
    indices = torch.stack([rec["idx_gpu"] for _, rec in flat])

    values_cpu = values.float().detach().cpu().numpy()
    indices_cpu = indices.long().detach().cpu().numpy()

    out = {"A": [], "B": [], "CD": []}
    for n, ((name, rec), val, idx) in enumerate(zip(flat, values_cpu, indices_cpu)):
        idx = int(idx)
        y0 = idx // rec["Wv"]
        x0 = idx % rec["Wv"]
        r = dict(rec)
        r.pop("value_gpu")
        r.pop("idx_gpu")
        r["utility"] = float(val)
        r["rect"] = (
            int(y0),
            int(y0 + rec["h"]),
            int(x0),
            int(x0 + rec["w"]),
        )
        out[name].append(r)
    return out


class DeadlineScheduler:
    def __init__(
        self,
        lut_path,
        levels,
        lambda_a,
        lambda_b,
        lambda_cd,
        extra_ms,
    ):
        self.path = Path(lut_path)
        self.lut = json.loads(self.path.read_text())
        self.levels = [float(x) for x in levels]
        self.lambda_a = float(lambda_a)
        self.lambda_b = float(lambda_b)
        self.lambda_cd = float(lambda_cd)
        self.extra_ms = float(extra_ms)

        lut_levels = self.lut.get("levels", {})
        for key in ("A", "B", "CD"):
            xs = [float(x) for x in lut_levels.get(key, [])]
            if len(xs) != len(self.levels) or any(
                abs(a - b) > 1e-8 for a, b in zip(xs, self.levels)
            ):
                raise RuntimeError(
                    f"LUT {key} levels {xs} do not match requested levels {self.levels}"
                )

        self.actions = self.lut["actions"]
        expected = len(self.levels) ** 3
        if len(self.actions) != expected:
            raise RuntimeError(
                f"Expected {expected} LUT actions, found {len(self.actions)}"
            )

        # Global-Full P99 comes from the component profile adjacent to the LUT.
        comp_path = self.path.parent / "adaptive_latency_components.json"
        if not comp_path.exists():
            source = self.lut.get("source_components", "")
            candidate = Path(source) if source else None
            if candidate is not None and candidate.exists():
                comp_path = candidate
            else:
                raise FileNotFoundError(
                    "adaptive_latency_components.json not found beside LUT"
                )

        self.components_path = comp_path
        comp = json.loads(comp_path.read_text())
        self.full_p99_ms = float(
            comp["reference"]["original_full_forward"]["p99_ms"]
        )

    @property
    def full_safe_ms(self):
        return self.full_p99_ms + self.extra_ms

    def full_fits(self, deadline_ms):
        return self.full_safe_ms <= float(deadline_ms)

    def choose_gpu(self, gpu_plans, deadline_ms):
        # STREAMDSGN_LATENCY_OPT_V1: GPU 5x5x5 scheduler.
        device = gpu_plans["A"][0]["value_gpu"].device
        K = len(self.levels)
        if not hasattr(self, "_latency_lut_cpu"):
            vals = []
            for a in range(K):
                for b in range(K):
                    for cd in range(K):
                        vals.append(float(self.actions[f"{a},{b},{cd}"]["safe_ms"]) + self.extra_ms)
            self._latency_lut_cpu = torch.tensor(vals, dtype=torch.float32).view(K, K, K)
            self._latency_lut_gpu = None
            self._latency_lut_device = None
        if self._latency_lut_gpu is None or self._latency_lut_device != device:
            self._latency_lut_gpu = self._latency_lut_cpu.to(device=device)
            self._latency_lut_device = device

        ua = torch.stack([x["value_gpu"].float() for x in gpu_plans["A"]])
        ub = torch.stack([x["value_gpu"].float() for x in gpu_plans["B"]])
        uc = torch.stack([x["value_gpu"].float() for x in gpu_plans["CD"]])
        util = self.lambda_a * ua[:, None, None] + self.lambda_b * ub[None, :, None] + self.lambda_cd * uc[None, None, :]
        lat = self._latency_lut_gpu
        feasible = lat <= float(deadline_ms)
        feasible_count = feasible.sum()
        no_feasible = feasible_count == 0
        masked = util.masked_fill(~feasible, -float("inf"))
        max_utility = masked.max()
        # Preserve the old scheduler's latency tie-break among actions with
        # the same maximum utility.
        best_utility_mask = feasible & (util >= max_utility - 1e-12)
        tie_lat = lat.masked_fill(~best_utility_mask, float("inf"))
        best_flat = tie_lat.reshape(-1).argmin()
        fastest_flat = lat.reshape(-1).argmin()
        best_flat = torch.where(no_feasible, fastest_flat, best_flat)

        a_gpu = best_flat // (K * K)
        rem = best_flat % (K * K)
        b_gpu = rem // K
        cd_gpu = rem % K
        idx_a = torch.stack([x["idx_gpu"] for x in gpu_plans["A"]])[a_gpu]
        idx_b = torch.stack([x["idx_gpu"] for x in gpu_plans["B"]])[b_gpu]
        idx_c = torch.stack([x["idx_gpu"] for x in gpu_plans["CD"]])[cd_gpu]

        selected_lat = lat[a_gpu, b_gpu, cd_gpu]
        selected_util = util[a_gpu, b_gpu, cd_gpu]
        packed = torch.stack([
            a_gpu.float(), b_gpu.float(), cd_gpu.float(),
            idx_a.float(), idx_b.float(), idx_c.float(),
            feasible_count.float(), no_feasible.float(),
            selected_lat.float(), selected_util.float(),
        ]).detach().cpu().tolist()
        a, b, cd, ia, ib, ic, nfeas, nofeas = [int(x) for x in packed[:8]]
        selected_lat_cpu = float(packed[8])
        selected_util_cpu = float(packed[9])

        def _plan(stage, level, idx):
            rec = gpu_plans[stage][level]
            y0 = idx // rec["Wv"]
            x0 = idx % rec["Wv"]
            return {
                "level": level, "ratio": rec["ratio"], "h": rec["h"], "w": rec["w"],
                "rect": (y0, y0 + rec["h"], x0, x0 + rec["w"]),
            }

        return {
            "key": f"{a},{b},{cd}", "a": a, "b": b, "cd": cd,
            "estimated_safe_ms": selected_lat_cpu,
            "utility": selected_util_cpu,
            "feasible_count": nfeas, "no_feasible_action": bool(nofeas),
        }, {"A": _plan("A", a, ia), "B": _plan("B", b, ib), "CD": _plan("CD", cd, ic)}

    def choose(self, plans, deadline_ms):
        best = None
        feasible_count = 0
        fastest = None

        for a in range(len(self.levels)):
            for b in range(len(self.levels)):
                for cd in range(len(self.levels)):
                    key = f"{a},{b},{cd}"
                    lut_rec = self.actions[key]
                    est_ms = float(lut_rec["safe_ms"]) + self.extra_ms

                    utility = (
                        self.lambda_a * plans["A"][a]["utility"]
                        + self.lambda_b * plans["B"][b]["utility"]
                        + self.lambda_cd * plans["CD"][cd]["utility"]
                    )

                    rec = {
                        "key": key,
                        "a": a,
                        "b": b,
                        "cd": cd,
                        "estimated_safe_ms": est_ms,
                        "utility": float(utility),
                    }

                    if fastest is None or est_ms < fastest["estimated_safe_ms"]:
                        fastest = rec

                    if est_ms <= float(deadline_ms):
                        feasible_count += 1
                        if best is None:
                            best = rec
                        else:
                            # Primary objective: maximum learned refresh utility.
                            # Tie-break: lower conservative latency estimate.
                            if (
                                rec["utility"] > best["utility"] + 1e-12
                                or (
                                    abs(rec["utility"] - best["utility"]) <= 1e-12
                                    and rec["estimated_safe_ms"]
                                    < best["estimated_safe_ms"]
                                )
                            ):
                                best = rec

        if best is None:
            best = dict(fastest)
            best["no_feasible_action"] = True
        else:
            best["no_feasible_action"] = False

        best["feasible_count"] = int(feasible_count)
        return best


# =============================================================================
# Model / state construction
# =============================================================================


def build_env(args):
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.LOCAL_RANK = 0
    cfg.MODEL.SAVE_TIME = False
    cfg.DATA_CONFIG.INFER_TIME_PATH = None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(out_dir / "eval.log", rank=0)

    dataset, loader, _ = build_dataloader(
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
        num_class=len(cfg.CLASS_NAMES),
        dataset=dataset,
    )
    model.load_params_from_file(
        filename=args.ckpt,
        logger=logger,
        to_cpu=True,
    )
    model.cuda().eval()
    for p in model.parameters():
        p.requires_grad_(False)

    backbone = find_module(model, "StreamDSGN2Backbone")
    backbone.use_amp = amp_enabled(model)

    if backbone.cat_img_feature or backbone.cat_right_img_feature:
        raise RuntimeError(
            "Current physical adaptive evaluator requires "
            "cat_img_feature=False and cat_right_img_feature=False"
        )
    if len(backbone.hg_stereo) != 0:
        raise RuntimeError(
            "Current physical B path assumes num_hg=0; found hg_stereo modules"
        )
    if backbone.use_stereo_out_type != "feature":
        raise RuntimeError(
            "Current physical adaptive evaluator requires use_stereo_out_type='feature'"
        )

    # The trained modules are not part of the base StreamDSGN checkpoint.
    try:
        adaptive_ckpt = torch.load(
            args.adaptive_ckpt, map_location="cpu", weights_only=False
        )
    except TypeError:
        # Compatibility with older PyTorch releases without weights_only.
        adaptive_ckpt = torch.load(args.adaptive_ckpt, map_location="cpu")
    if "importance_net" not in adaptive_ckpt or "pred_net" not in adaptive_ckpt:
        raise KeyError(
            "Adaptive checkpoint must contain 'importance_net' and 'pred_net'"
        )

    importance_net = StageImportancePredictor(
        shallow_channels=64,
        bev_channels=96,
        hidden=args.importance_hidden,
        age_cap=args.age_cap,
    ).cuda().eval()
    pred_net = BEVTemporalPredictor(
        channels=96,
        hidden=args.predictor_hidden,
        age_cap=args.age_cap,
    ).cuda().eval()

    importance_net.load_state_dict(adaptive_ckpt["importance_net"], strict=True)
    pred_net.load_state_dict(adaptive_ckpt["pred_net"], strict=True)
    for m in (importance_net, pred_net):
        for p in m.parameters():
            p.requires_grad_(False)

    return dataset, loader, model, backbone, importance_net, pred_net, logger, out_dir


# =============================================================================
# Adaptive runtime engine
# =============================================================================


class AdaptiveEngine:
    def __init__(
        self,
        model,
        backbone,
        importance_net,
        pred_net,
        scheduler,
        levels,
        roi_align,
        b_halo,
        cd_halo,
        pred_margin,
        age_cap,
    ):
        self.model = model
        self.backbone = backbone
        self.importance_net = importance_net
        self.pred_net = pred_net
        self.scheduler = scheduler
        self.levels = list(levels)
        self.roi_align = int(roi_align)
        self.b_halo = int(b_halo)
        self.cd_halo = int(cd_halo)
        self.pred_margin = int(pred_margin)
        self.age_cap = float(age_cap)
        self.state = None

        self.backbone._ensure_adaptive_a_runtime()

    def reset(self):
        self.state = None
        self.backbone.reset_adaptive_a_cache()
        self.backbone.clear_adaptive_a_profile()
        if hasattr(self.backbone, "clear_adaptive_training_state"):
            self.backbone.clear_adaptive_training_state()
        q = getattr(self.model, "history_feature_queue", None)
        if q is not None:
            q.clear()

    @staticmethod
    def _sample_idx(token):
        return token.get("this_sample_idx", "")

    def increment_age_for_drop(self):
        """One sensor frame elapsed without any model execution/refresh."""
        if self.state is None:
            return
        with torch.no_grad():
            for key in ("age_a", "age_b", "age_f"):
                self.state[key].add_(1.0).clamp_(max=self.age_cap)

    def _history(self):
        if self.state is None or getattr(self.model, "history_tag", None) is None:
            return deque()
        q = deque(maxlen=max(1, len(self.model.history_tag)))
        q.append(
            (
                self.state["last_sample_idx"],
                {"spatial_features": self.state["emit1"]},
            )
        )
        return q

    def _run_downstream(self, cur, history):
        cur["history_features"] = history
        with autocast_ctx(self.model):
            for module in self.model.fusion_module:
                cur = module(cur)
            for module in self.model.after_fusion_blocks:
                cur = module(cur)
            pred_dicts, ret_dict = self.model.post_processing(cur)
        return pred_dicts, ret_dict

    def _full_feature_extractor(self, token):
        self.backbone.clear_adaptive_a_profile()
        if hasattr(self.backbone, "clear_adaptive_training_state"):
            self.backbone.clear_adaptive_training_state()

        x = dict(token)
        with autocast_ctx(self.model):
            for module in self.model.feature_extractor:
                x = module(x)

        required = [
            "left_shallow_feature",
            "right_shallow_feature",
            "adaptive_stage_b",
            "spatial_features",
        ]
        missing = [k for k in required if k not in x]
        if missing:
            raise KeyError(
                f"Missing adaptive training/runtime taps {missing}. "
                "The repository patch used for adaptive training must still be installed."
            )
        return x

    def _update_state_from_full(self, full_x, token, old_emit1, pre_fusion_bev):
        # IMPORTANT: FeatureAlignment replaces full_x['spatial_features'] with a
        # fused tensor. The temporal cache must store the PRE-FUSION BEV, exactly
        # like STREAM.forward_test() clones it before the fusion module.
        bev = pre_fusion_bev.detach()
        b = full_x["adaptive_stage_b"].detach()
        shallow_l = full_x["left_shallow_feature"].detach()
        shallow_r = full_x["right_shallow_feature"].detach()

        B = bev.shape[0]
        b_hw = b.shape[-2:]
        f_hw = bev.shape[-2:]

        # A native grid equals B H/W in the current configuration.
        a_hw = b_hw

        # true_bev must NOT alias emitted history because future R updates are
        # in-place while predictions must never recursively overwrite true cache.
        true_bev = bev.clone()

        if old_emit1 is None:
            emit2 = bev
        else:
            emit2 = old_emit1

        self.state = {
            "b": b,
            "true_bev": true_bev,
            "emit1": bev,
            "emit2": emit2,
            "prev_shallow_l": shallow_l,
            "prev_shallow_r": shallow_r,
            "age_a": torch.zeros(
                (B, 1, *a_hw), device=bev.device, dtype=bev.dtype
            ),
            "age_b": torch.zeros(
                (B, 1, *b_hw), device=bev.device, dtype=bev.dtype
            ),
            "age_f": torch.zeros(
                (B, 1, *f_hw), device=bev.device, dtype=bev.dtype
            ),
            "last_sample_idx": self._sample_idx(token),
        }

    def run_global_full(self, token, label="GLOBAL_FULL"):
        history = self._history()
        old_emit1 = None if self.state is None else self.state["emit1"]

        full_x = self._full_feature_extractor(token)
        # Hold a reference to the exact pre-fusion BEV before FeatureAlignment
        # reassigns full_x['spatial_features']. This tensor is the history state.
        pre_fusion_bev = full_x["spatial_features"]
        pred_dicts, ret_dict = self._run_downstream(full_x, history)
        self._update_state_from_full(
            full_x, token, old_emit1, pre_fusion_bev
        )

        return pred_dicts, ret_dict, {
            "action": label,
            "action_key": "FULL",
            "a_level": -1,
            "b_level": -1,
            "cd_level": -1,
            "a_ratio": 1.0,
            "b_ratio": 1.0,
            "cd_ratio": 1.0,
            "estimated_safe_ms": self.scheduler.full_safe_ms,
            "utility": (
                self.scheduler.lambda_a
                + self.scheduler.lambda_b
                + self.scheduler.lambda_cd
            ),
            "feasible_count": None,
            "no_feasible_action": False,
            "a_rect": None,
            "b_rect": None,
            "cd_rect": None,
        }

    def _run_a_explicit(self, img, shallow, side, rect):
        """
        Exact current physical Stage-A implementation, but reuses the A_s tensor
        already computed for the importance predictor instead of recomputing A_s.
        """
        bb = self.backbone
        bb._ensure_adaptive_a_runtime()

        if side not in ("left", "right"):
            raise ValueError(side)

        cache = bb._adaptive_a_cache[side]
        if any(x is None for x in cache):
            raise RuntimeError(
                "Adaptive A cache is uninitialized. Each scene must start with Full."
            )

        H2, W2 = cache[0].shape[-2:]
        if rect is None or len(rect) != 4:
            raise RuntimeError(f"invalid A rect: {rect}")
        y0, y1, x0, x1 = [int(v) for v in rect]
        if not (0 <= y0 < y1 <= H2 and 0 <= x0 < x1 <= W2):
            raise RuntimeError(
                f"A rect {rect} outside native grid {(H2, W2)}"
            )

        with autocast_ctx(self.model):
            bb2 = bb.feature_backbone

            patch2 = bb._adaptive_a_run_layer2_roi(
                shallow, rect, bb._adaptive_a_halos[1]
            )
            bb._adaptive_a_update_cache(cache[0], patch2, rect)

            layer3 = getattr(bb2, bb2.res_layers[2])
            patch3 = bb._adaptive_a_run_stride1_roi(
                layer3, cache[0], rect, bb._adaptive_a_halos[2]
            )
            bb._adaptive_a_update_cache(cache[1], patch3, rect)

            layer4 = getattr(bb2, bb2.res_layers[3])
            patch4 = bb._adaptive_a_run_stride1_roi(
                layer4, cache[1], rect, bb._adaptive_a_halos[3]
            )
            bb._adaptive_a_update_cache(cache[2], patch4, rect)

            feats = [img, shallow, cache[0], cache[1], cache[2]]
            stereo_cache = bb._adaptive_a_stereo_cache[side]
            if stereo_cache is None:
                raise RuntimeError("Adaptive A stereo cache is uninitialized")

            stereo_patch, output_rect = bb.feature_neck.forward_stereo_roi(
                feats,
                base_rect=rect,
                base_halo=bb._adaptive_a_neck_halo,
                spp_cache=bb._adaptive_spp_cache[side],
            )
            bb._adaptive_a_update_dense_cache(
                stereo_cache, stereo_patch, output_rect
            )

        return stereo_cache

    def _prepare_b_shared(self, token, left_stereo):
        """
        Reproduce the ORIGINAL backbone B-preparation path exactly.

        The original STREAM forward wraps the whole backbone in autocast.
        compute_disp_channels() contains interpolation/tensor math whose AMP
        execution context can change the discretized DPS channel indices.
        Therefore this preparation MUST run inside the same autocast context.
        """
        bb = self.backbone
        left = token["left_img"]
        calib = token["calib"]
        tensor_dtype = torch.float16 if bb.use_amp else torch.float32

        with autocast_ctx(self.model):
            fu_mul_baseline = torch.as_tensor(
                [x.fu_mul_baseline for x in calib],
                dtype=tensor_dtype,
                device=left.device,
            )

            # Match stream_dsgn2_backbone.py literally:
            #   self.downsampled_depth.cuda().half() if use_amp else .cuda()
            downsampled_depth = (
                bb._adaptive_downsampled_depth_half if bb.use_amp else bb.downsampled_depth
            )

            shift = (
                fu_mul_baseline[:, None]
                / downsampled_depth[None, :]
                / (
                    bb.downsample_disp
                    if not bb.fullres_stereo_feature
                    else 1
                )
            )

            if left_stereo.shape[1] <= bb.cv_dim:
                raise RuntimeError(
                    "Current physical B ROI implementation expects the DPS path"
                )

            psv = bb.compute_disp_channels(
                shift[0],
                left_stereo.shape[1],
                inv_ratio=bb.inv_smooth_psv,
            ).to(torch.int32)

        return shift, psv

    def _run_b_roi(self, left_stereo, right_stereo, shift, psv, rect):
        bb = self.backbone
        H, W = self.state["b"].shape[-2:]
        e = expand_align(rect, self.b_halo, H, W, self.roi_align)
        cy0, cy1, cx0, cx1 = rect
        ey0, ey1, ex0, ex1 = e

        with autocast_ctx(self.model):
            cost_roi = bb.build_cost.forward_roi(
                left_stereo,
                right_stereo,
                None,
                None,
                shift,
                ph0=ey0,
                ph1=ey1,
                pw0=ex0,
                pw1=ex1,
                psv_disps_channels=psv,
            )
            x0v = bb.dres0(cost_roi)
            out_roi = bb.dres1(x0v) + x0v

        ly0 = cy0 - ey0
        lx0 = cx0 - ex0
        rh = cy1 - cy0
        rw = cx1 - cx0
        patch = out_roi[..., ly0:ly0 + rh, lx0:lx0 + rw]

        if tuple(patch.shape[-2:]) != (rh, rw):
            raise RuntimeError(
                f"B core crop mismatch: got {tuple(patch.shape[-2:])}, "
                f"expected {(rh, rw)}"
            )

        self.state["b"][..., cy0:cy1, cx0:cx1].copy_(patch)
        return e

    def _run_cd_roi_to_bev_patch(self, token, rect):
        bb = self.backbone
        true_bev = self.state["true_bev"]
        H, W = true_bev.shape[-2:]
        e = expand_align(rect, self.cd_halo, H, W, self.roi_align)
        cy0, cy1, cx0, cx1 = rect
        ey0, ey1, ex0, ex1 = e

        coordinates_3d = (
            bb._adaptive_coordinates_3d_half if bb.use_amp else bb.coordinates_3d
        )

        calib = token["calib"][0]
        tensor_dtype = torch.float16 if bb.use_amp else torch.float32
        P2 = torch.as_tensor(calib.P2, device="cuda", dtype=tensor_dtype)
        image_shape = token["image_shape"][0]
        random_T = token["random_T"][0] if "random_T" in token else None

        c_roi = cprof.run_c_roi(
            bb,
            self.state["b"],
            coordinates_3d,
            e,
            token["left_img"].shape[2:],
            image_shape,
            P2,
            random_T=random_T,
        )
        d_roi = dprof.run_d_stage(bb, c_roi)

        ly0 = cy0 - ey0
        lx0 = cx0 - ex0
        rh = cy1 - cy0
        rw = cx1 - cx0
        core_5d = d_roi[..., ly0:ly0 + rh, lx0:lx0 + rw].contiguous()

        if tuple(core_5d.shape[-2:]) != (rh, rw):
            raise RuntimeError(
                f"CD core crop mismatch: got {tuple(core_5d.shape[-2:])}, "
                f"expected {(rh, rw)}"
            )

        N, C, D, h, w = core_5d.shape
        bev_patch = core_5d.view(N, C * D, h, w)
        if bev_patch.shape[1] != true_bev.shape[1]:
            raise RuntimeError(
                f"BEV channel mismatch: patch={bev_patch.shape[1]}, "
                f"cache={true_bev.shape[1]}"
            )
        return bev_patch, e

    def _age_update_rect(self, age, rect):
        with torch.no_grad():
            age.add_(1.0).clamp_(max=self.age_cap)
            y0, y1, x0, x1 = rect
            age[..., y0:y1, x0:x1].zero_()

    def run_adaptive(self, token, deadline_ms):
        if self.state is None:
            raise RuntimeError("Adaptive frame requested before Full initialization")

        history = self._history()
        history_idx = self.state["last_sample_idx"]
        history_bev = self.state["emit1"]

        left = token["left_img"]
        right = token["right_img"]

        # ------------------------------------------------------------
        # Cheap pre-action context: A_s left/right.
        # ------------------------------------------------------------
        with autocast_ctx(self.model):
            shallow_l = self.backbone.forward_2d_shallow(left)
            shallow_r = self.backbone.forward_2d_shallow(right)

            a_hw = self.state["age_a"].shape[-2:]
            b_hw = self.state["age_b"].shape[-2:]
            cd_hw = self.state["age_f"].shape[-2:]

            imp = self.importance_net(
                cur_left=shallow_l,
                cur_right=shallow_r,
                prev_left=self.state["prev_shallow_l"],
                prev_right=self.state["prev_shallow_r"],
                prev_bev=self.state["emit1"],
                age_a=self.state["age_a"],
                age_b=self.state["age_b"],
                age_f=self.state["age_f"],
                a_hw=a_hw,
                b_hw=b_hw,
                cd_hw=cd_hw,
            )

            gpu_plans = {
                "A": stage_plan_gpu(imp["q_a"], self.levels, self.roi_align),
                "B": stage_plan_gpu(imp["q_b"], self.levels, self.roi_align),
                "CD": stage_plan_gpu(imp["q_cd"], self.levels, self.roi_align),
            }

        choice, selected_plans = self.scheduler.choose_gpu(gpu_plans, deadline_ms)

        a = choice["a"]
        b = choice["b"]
        cd = choice["cd"]
        a_rect = selected_plans["A"]["rect"]
        b_rect = selected_plans["B"]["rect"]
        cd_rect = selected_plans["CD"]["rect"]

        # ------------------------------------------------------------
        # Physical A_d from the already-computed A_s.
        # ------------------------------------------------------------
        left_stereo = self._run_a_explicit(left, shallow_l, "left", a_rect)
        right_stereo = self._run_a_explicit(right, shallow_r, "right", a_rect)

        # ------------------------------------------------------------
        # Physical B selective update on a persistent dense B cache.
        # ------------------------------------------------------------
        shift, psv = self._prepare_b_shared(token, left_stereo)
        b_exec = self._run_b_roi(
            left_stereo, right_stereo, shift, psv, b_rect
        )

        # ------------------------------------------------------------
        # Physical C+D selective recompute.
        # C intentionally consumes current/cache dense B, matching training
        # semantics. We do NOT force CD to depend on a Full-current B support.
        # ------------------------------------------------------------
        bev_patch, cd_exec = self._run_cd_roi_to_bev_patch(token, cd_rect)

        # Predictor uses PRE-FRAME emitted history and PRE-FRAME age, matching
        # the training rollout.
        with autocast_ctx(self.model):
            pred_bev = self.pred_net(
                self.state["emit1"],
                self.state["emit2"],
                self.state["age_f"],
            )

        # True recompute cache: only R writes this cache.
        cy0, cy1, cx0, cx1 = cd_rect
        self.state["true_bev"][..., cy0:cy1, cx0:cx1].copy_(bev_patch)

        # P = dilate(R, margin) \\ R; U = complement(R union P).
        R = torch.zeros(
            (self.state["true_bev"].shape[0], 1, *cd_hw),
            device=self.state["true_bev"].device,
            dtype=self.state["true_bev"].dtype,
        )
        R[..., cy0:cy1, cx0:cx1] = 1
        if self.pred_margin > 0:
            k = self.pred_margin * 2 + 1
            outer = F.max_pool2d(
                R.float(), kernel_size=k, stride=1, padding=self.pred_margin
            )
            P = ((outer > 0.5) & (R.float() < 0.5))
        else:
            P = torch.zeros_like(R, dtype=torch.bool)

        with autocast_ctx(self.model):
            bev_mix = torch.where(
                P,
                pred_bev,
                self.state["true_bev"],
            )

        cur = dict(token)
        cur["spatial_features"] = bev_mix
        cur["spatial_features_stride"] = 1

        pred_dicts, ret_dict = self._run_downstream(cur, history)

        # ------------------------------------------------------------
        # State transition AFTER the frame's pre-fusion output is formed.
        # ------------------------------------------------------------
        with torch.no_grad():
            old_emit1 = self.state["emit1"]
            self.state["emit2"] = old_emit1
            self.state["emit1"] = bev_mix.detach()
            self.state["prev_shallow_l"] = shallow_l.detach()
            self.state["prev_shallow_r"] = shallow_r.detach()

            self._age_update_rect(self.state["age_a"], a_rect)
            self._age_update_rect(self.state["age_b"], b_rect)
            self._age_update_rect(self.state["age_f"], cd_rect)
            self.state["last_sample_idx"] = self._sample_idx(token)

        return pred_dicts, ret_dict, {
            "action": "ADAPTIVE",
            "action_key": choice["key"],
            "a_level": a,
            "b_level": b,
            "cd_level": cd,
            "a_ratio": self.levels[a],
            "b_ratio": self.levels[b],
            "cd_ratio": self.levels[cd],
            "estimated_safe_ms": choice["estimated_safe_ms"],
            "utility": choice["utility"],
            "feasible_count": choice["feasible_count"],
            "no_feasible_action": choice["no_feasible_action"],
            "a_rect": list(a_rect),
            "b_rect": list(b_rect),
            "cd_rect": list(cd_rect),
            "b_exec": list(b_exec),
            "cd_exec": list(cd_exec),
            "history_idx": str(first_scalar(history_idx)),
        }

    def forward(self, token, deadline_ms, force_full=False):
        if force_full or self.state is None:
            return self.run_global_full(token, label="INIT_FULL" if self.state is None else "GLOBAL_FULL")

        # Global Full has maximum possible stage utility (= 1+1+1). If it fits,
        # do not waste A_s/ImportanceNet time selecting an inferior selective action.
        if self.scheduler.full_fits(deadline_ms):
            return self.run_global_full(token, label="GLOBAL_FULL")

        return self.run_adaptive(token, deadline_ms)


# =============================================================================
# Startup correctness check
# =============================================================================


@torch.no_grad()
def self_check_physical_bcd(
    engine,
    token,
    tol,
    logger,
):
    """
    Strong startup correctness check.

    During an exact original Full feature-extractor pass we capture the ACTUAL
    BuildCostVolume inputs used by StreamDSGN. We then verify:

      1. adaptive_stage_a_left/right taps equal the exact tensors entering B;
      2. our runtime reconstruction of shift and DPS channel indices is exact;
      3. our reconstructed Full B equals adaptive_stage_b;
      4. our physical Full C+D->BEV equals the original pre-fusion BEV.

    This prevents silently evaluating thousands of frames with a mismatched
    B-preparation path.
    """
    engine.reset()
    bb = engine.backbone

    captured = {}

    def cost_pre_hook(module, hook_args, hook_kwargs):
        captured["args"] = tuple(
            x.detach() if torch.is_tensor(x) else x
            for x in hook_args
        )
        captured["kwargs"] = dict(hook_kwargs)

    h = bb.build_cost.register_forward_pre_hook(
        cost_pre_hook,
        with_kwargs=True,
    )

    try:
        full_x = engine._full_feature_extractor(token)
    finally:
        h.remove()

    if "args" not in captured:
        raise RuntimeError("SELF-CHECK failed to capture BuildCostVolume inputs")

    ca = captured["args"]
    ck = captured["kwargs"]

    if len(ca) < 5:
        raise RuntimeError(
            f"Unexpected BuildCostVolume argument count: {len(ca)}"
        )

    exact_left = ca[0]
    exact_right = ca[1]
    exact_shift = ca[4]
    exact_psv = ca[5] if len(ca) >= 6 else ck.get(
        "psv_disps_channels", None
    )

    left_stereo = full_x.get("adaptive_stage_a_left")
    right_stereo = full_x.get("adaptive_stage_a_right")
    b_ref = full_x["adaptive_stage_b"]
    bev_ref = full_x["spatial_features"]

    if left_stereo is None or right_stereo is None:
        raise KeyError(
            "adaptive_stage_a_left/right taps are required for startup self-check"
        )
    if exact_psv is None:
        raise RuntimeError("Exact Full B call did not contain DPS channel indices")

    left_tap_diff = float(
        (left_stereo.float() - exact_left.float()).abs().max().item()
    )
    right_tap_diff = float(
        (right_stereo.float() - exact_right.float()).abs().max().item()
    )

    # This is the path used by every adaptive frame.
    shift, psv = engine._prepare_b_shared(token, left_stereo)

    shift_diff = float(
        (shift.float() - exact_shift.float()).abs().max().item()
    )
    psv_equal = bool(torch.equal(psv, exact_psv.to(torch.int32)))
    psv_mismatch = int(
        (psv != exact_psv.to(torch.int32)).sum().item()
    )

    with autocast_ctx(engine.model):
        cost = bb.build_cost(
            left_stereo,
            right_stereo,
            None,
            None,
            shift,
            psv,
        )
        x0 = bb.dres0(cost)
        b_test = bb.dres1(x0) + x0

    b_diff = float(
        (b_test.float() - b_ref.float()).abs().max().item()
    )

    H, W = bev_ref.shape[-2:]
    full_rect = (0, H, 0, W)

    coordinates_3d = (
        bb._adaptive_coordinates_3d_half if bb.use_amp else bb.coordinates_3d
    )

    calib = token["calib"][0]
    dtype = torch.float16 if bb.use_amp else torch.float32
    P2 = torch.as_tensor(
        calib.P2,
        device="cuda",
        dtype=dtype,
    )
    random_T = (
        token["random_T"][0]
        if "random_T" in token
        else None
    )

    c = cprof.run_c_roi(
        bb,
        b_test,
        coordinates_3d,
        full_rect,
        token["left_img"].shape[2:],
        token["image_shape"][0],
        P2,
        random_T=random_T,
    )
    d = dprof.run_d_stage(bb, c)

    N, C, D, h2, w2 = d.shape
    bev_test = d.view(N, C * D, h2, w2)
    bev_diff = float(
        (bev_test.float() - bev_ref.float()).abs().max().item()
    )

    logger.info(
        "[SELF-CHECK] A taps: left=%.8f right=%.8f | "
        "B prep: shift_diff=%.8f psv_equal=%s psv_mismatch=%d | "
        "B max_diff=%.8f BEV max_diff=%.8f tol=%.8f",
        left_tap_diff,
        right_tap_diff,
        shift_diff,
        psv_equal,
        psv_mismatch,
        b_diff,
        bev_diff,
        tol,
    )

    engine.reset()

    # Tap and discrete-DPS preparation must be exact. B/BEV allow small AMP
    # numerical differences controlled by tol.
    if left_tap_diff != 0.0 or right_tap_diff != 0.0:
        raise RuntimeError(
            "Physical runtime self-check failed: A-stage taps are not the "
            "actual tensors entering BuildCostVolume"
        )

    if shift_diff != 0.0 or not psv_equal:
        raise RuntimeError(
            "Physical runtime self-check failed: adaptive B preparation "
            f"does not reproduce original Full B inputs: "
            f"shift_diff={shift_diff}, psv_mismatch={psv_mismatch}"
        )

    if b_diff > tol or bev_diff > tol:
        raise RuntimeError(
            "Physical runtime self-check failed after exact B preparation: "
            f"B diff={b_diff}, BEV diff={bev_diff}, tol={tol}"
        )


# =============================================================================
# Warmup
# =============================================================================


@torch.no_grad()
def warmup_engine(loader, engine, clock, num_frames):
    if num_frames <= 0:
        return

    engine.reset()
    current_scene = None
    used = 0

    for batch in loader:
        scene, frame = get_scene_frame(batch)
        token = batch["token"]
        if scene != current_scene:
            engine.reset()
            current_scene = scene

        load_data_to_gpu(batch)
        next_frame = ""
        if "next_sample_idx" in token:
            next_frame = str(first_scalar(token["next_sample_idx"]))
        deadline = clock.deadline_ms(scene, frame, next_frame)

        engine.forward(token, deadline, force_full=(engine.state is None))
        torch.cuda.synchronize()

        used += 1
        if used >= num_frames:
            break

    engine.reset()


# =============================================================================
# True-stream evaluation
# =============================================================================


@torch.no_grad()
def evaluate_rate(
    args,
    dataset,
    loader,
    engine,
    logger,
    out_dir,
    rate,
    clock,
):
    engine.reset()

    stream_annos = []
    pending = deque()
    latest_completed = None

    current_scene = None
    busy_until_ms = -math.inf

    processed = 0
    dropped = 0
    no_output_frames = 0
    stale_output_frames = 0
    prediction_ages = []
    forward_times = []
    estimated_times = []
    deadline_misses_on_accepted = 0
    action_counts = Counter()
    level_counts = Counter()

    for i, batch in enumerate(loader):
        if args.max_frames > 0 and i >= args.max_frames:
            break

        scene, frame = get_scene_frame(batch)
        token = batch["token"]
        next_frame = ""
        if "next_sample_idx" in token:
            next_frame = str(first_scalar(token["next_sample_idx"]))

        if scene != current_scene:
            current_scene = scene
            engine.reset()
            pending.clear()
            latest_completed = None
            busy_until_ms = -math.inf

        arrival_ms = clock.arrival_ms(scene, frame)
        deadline_ms = clock.deadline_ms(scene, frame, next_frame)

        # Publish only results that have truly completed by this sensor timestamp.
        while pending and pending[0]["completion_ms"] <= arrival_ms + 1e-9:
            latest_completed = pending.popleft()

        if latest_completed is None:
            eval_pred = empty_prediction(scene, frame, next_frame)
            source_frame = ""
            source_arrival_ms = math.nan
            source_completion_ms = math.nan
            prediction_age_ms = math.nan
            no_output_frames += 1
        else:
            eval_pred = retag_prediction(
                latest_completed["anno"], scene, frame, next_frame
            )
            source_frame = latest_completed["source_frame"]
            source_arrival_ms = latest_completed["source_arrival_ms"]
            source_completion_ms = latest_completed["completion_ms"]
            prediction_age_ms = arrival_ms - source_arrival_ms
            prediction_ages.append(prediction_age_ms)
            if source_frame != frame:
                stale_output_frames += 1

        stream_annos.append(eval_pred)

        accepted = arrival_ms >= busy_until_ms - 1e-9

        service_ms = math.nan
        completion_ms = math.nan
        meta = {
            "action": "DROP",
            "action_key": "DROP",
            "a_level": -2,
            "b_level": -2,
            "cd_level": -2,
            "a_ratio": math.nan,
            "b_ratio": math.nan,
            "cd_ratio": math.nan,
            "estimated_safe_ms": math.nan,
            "utility": math.nan,
            "feasible_count": None,
            "no_feasible_action": False,
            "a_rect": None,
            "b_rect": None,
            "cd_rect": None,
        }

        if accepted:
            # H2D is deliberately outside the measured service time.
            load_data_to_gpu(batch)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            pred_dicts, ret_dict, meta = engine.forward(
                token,
                deadline_ms,
                force_full=(engine.state is None),
            )

            torch.cuda.synchronize()
            service_ms = (time.perf_counter() - t0) * 1000.0

            annos = dataset.generate_prediction_dicts(
                batch,
                pred_dicts,
                dataset.class_names,
            )
            if len(annos) != 1:
                raise RuntimeError(f"Expected one prediction dict, got {len(annos)}")

            completion_ms = arrival_ms + service_ms
            busy_until_ms = completion_ms

            pending.append({
                "completion_ms": completion_ms,
                "source_arrival_ms": arrival_ms,
                "source_frame": frame,
                "anno": annos[0],
            })

            processed += 1
            forward_times.append(service_ms)
            if math.isfinite(float(meta.get("estimated_safe_ms", math.nan))):
                estimated_times.append(float(meta["estimated_safe_ms"]))

            if service_ms > deadline_ms + 1e-9:
                deadline_misses_on_accepted += 1

            action_counts[meta["action"]] += 1
            if meta["action"] == "ADAPTIVE":
                level_counts[meta["action_key"]] += 1

        else:
            # No model call, no cache write, no shallow history. Only age advances
            # because one more sensor frame has elapsed since the last refresh.
            engine.increment_age_for_drop()
            dropped += 1
            action_counts["DROP"] += 1



        if (i + 1) % 200 == 0:
            n = processed + dropped
            logger.info(
                "[RATE %.2fx] %d/%d processed=%d dropped=%d drop=%.2f%% "
                "last_action=%s actual=%.3fms deadline=%.3fms",
                rate,
                i + 1,
                len(loader),
                processed,
                dropped,
                100.0 * dropped / max(n, 1),
                meta["action_key"],
                service_ms,
                deadline_ms,
            )

    n = processed + dropped
    if len(stream_annos) != n:
        raise RuntimeError(
            f"Streaming annotation mismatch: annos={len(stream_annos)}, frames={n}"
        )

    # Streaming alignment is already complete. offline_3d is used only as the
    # KITTI AP calculator, exactly as in the true-stream baseline evaluator.
    result_str_dict, result_dict = dataset.evaluation(
        stream_annos,
        dataset.class_names,
        eval_metric=["offline_3d"],
    )
    result_text = result_str_dict["offline_3d"]
    printable = (
        format_paper_metrics(result_text)
        if args.paper_metrics_only and format_paper_metrics is not None
        else result_text
    )

    frequency_hz = float(clock.base_fps * clock.rate)
    hz_tag = (
        str(int(round(frequency_hz)))
        if abs(frequency_hz - round(frequency_hz)) < 1e-9
        else str(frequency_hz).replace(".", "p")
    )

    # User-facing artifacts: accuracy only.
    precision_path = out_dir / f"precision_{hz_tag}hz.txt"
    precision_path.write_text(printable)

    summary = {
        "frequency_hz": frequency_hz,
        "rate_multiplier": float(rate),
        "precision": to_jsonable(result_dict),

        # Runtime diagnostics remain in memory / final JSON so the accuracy is
        # auditable, but no comparison files or per-frame trace files are made.
        "runtime": {
            "num_frames": n,
            "processed_frames": processed,
            "dropped_frames": dropped,
            "processed_ratio": processed / max(n, 1),
            "drop_ratio": dropped / max(n, 1),
            "frames_without_any_completed_output": no_output_frames,
            "stale_output_frames": stale_output_frames,
            "accepted_forward_deadline_misses": deadline_misses_on_accepted,
            "actual_forward_ms": stats(forward_times),
            "prediction_age_ms": stats(prediction_ages),
            "action_counts": dict(action_counts),
            "adaptive_action_counts": dict(level_counts),
        },
        "precision_text_file": str(precision_path),
    }

    logger.info("=" * 100)
    logger.info(
        "ADAPTIVE TRUE STREAM %.3f Hz (rate %.2fx)",
        frequency_hz,
        rate,
    )
    logger.info(
        "processed=%d/%d (%.2f%%), dropped=%d (%.2f%%)",
        processed,
        n,
        100.0 * summary["runtime"]["processed_ratio"],
        dropped,
        100.0 * summary["runtime"]["drop_ratio"],
    )
    logger.info(
        "actual forward ms: %s",
        summary["runtime"]["actual_forward_ms"],
    )
    logger.info("\n%s", printable)
    logger.info("precision: %s", precision_path)
    logger.info("=" * 100)

    return summary


# =============================================================================
# Main
# =============================================================================


def main():
    args = parse_args()
    levels = parse_float_list(args.levels)
    if len(levels) != 5:
        raise ValueError(f"Expected exactly 5 levels, got {levels}")
    if any(r <= 0 or r > 1 for r in levels):
        raise ValueError(f"Invalid levels: {levels}")

    rates = parse_float_list(args.rates)
    if any(r <= 0 for r in rates):
        raise ValueError(f"Invalid rates: {rates}")

    (
        dataset,
        loader,
        model,
        backbone,
        importance_net,
        pred_net,
        logger,
        out_dir,
    ) = build_env(args)

    base_fps = (
        float(args.fps)
        if args.fps is not None
        else float(cfg.DATA_CONFIG.get("ANNOS_FREQUENCY", 10))
    )
    timestamp_map = load_timestamps(args)

    scheduler = DeadlineScheduler(
        args.latency_lut,
        levels,
        args.lambda_a,
        args.lambda_b,
        args.lambda_cd,
        args.scheduler_extra_ms,
    )

    engine = AdaptiveEngine(
        model=model,
        backbone=backbone,
        importance_net=importance_net,
        pred_net=pred_net,
        scheduler=scheduler,
        levels=levels,
        roi_align=args.roi_align,
        b_halo=args.b_halo,
        cd_halo=args.cd_halo,
        pred_margin=args.pred_margin,
        age_cap=args.age_cap,
    )

    logger.info("=" * 100)
    logger.info("Physical adaptive true-stream evaluation")
    logger.info("dataset frames       : %d", len(dataset))
    logger.info("base fps             : %.3f", base_fps)
    logger.info("rates                : %s", rates)
    logger.info("levels               : %s", levels)
    logger.info("adaptive checkpoint  : %s", args.adaptive_ckpt)
    logger.info("latency LUT          : %s", args.latency_lut)
    logger.info("LUT authoritative    : %s", scheduler.lut.get("authoritative", False))
    logger.info("Full P99 from profile: %.3f ms", scheduler.full_p99_ms)
    logger.info("Full scheduler safe  : %.3f ms", scheduler.full_safe_ms)
    logger.info("scheduler extra guard: %.3f ms", args.scheduler_extra_ms)
    logger.info("H2D/data preparation : EXCLUDED from service time")
    logger.info("=" * 100)

    # -------------------------------------------------------------------------
    # One physical reconstruction check before the full dataset run.
    # -------------------------------------------------------------------------
    if not args.skip_self_check:
        first_batch = next(iter(loader))
        load_data_to_gpu(first_batch)
        self_check_physical_bcd(
            engine,
            first_batch["token"],
            args.self_check_tol,
            logger,
        )

    timestamp_source = (
        f"timestamp_json:{args.timestamp_json}"
        if timestamp_map is not None
        else f"fixed_rate:{base_fps}Hz"
    )

    all_summaries = []

    for rate in rates:
        clock = StreamClock(timestamp_map, base_fps, rate)

        logger.info("")
        logger.info(
            "[RATE] %.2fx -> nominal %.3f Hz, interval %.3f ms, clock=%s",
            rate,
            base_fps * rate,
            clock.nominal_interval_ms,
            timestamp_source,
        )

        # Warm only the action regime actually used by this deadline, then reset.
        warmup_engine(loader, engine, clock, args.warmup_frames)

        # Use a fresh clock because warmup consumed its scene-normalization state.
        clock = StreamClock(timestamp_map, base_fps, rate)

        summary = evaluate_rate(
            args,
            dataset,
            loader,
            engine,
            logger,
            out_dir,
            rate,
            clock,
        )
        summary["clock_source"] = timestamp_source
        all_summaries.append(summary)

    # One aggregate file containing only per-frequency accuracy plus the
    # minimal runtime diagnostics needed to audit true-stream behavior.
    aggregate_path = out_dir / "precision_all_rates.json"
    aggregate_path.write_text(
        json.dumps(to_jsonable(all_summaries), indent=2)
    )

    # Human-readable concatenation of the KITTI precision output.
    text_parts = []
    for rec in all_summaries:
        p = Path(rec["precision_text_file"])
        text_parts.append(
            "=" * 100
            + "\n"
            + f"{rec['frequency_hz']:.3f} Hz"
            + "\n"
            + "=" * 100
            + "\n"
            + p.read_text().rstrip()
            + "\n"
        )

    all_txt_path = out_dir / "precision_all_rates.txt"
    all_txt_path.write_text("\n".join(text_parts))

    logger.info("")
    logger.info("All rates complete.")
    logger.info("Precision JSON: %s", aggregate_path)
    logger.info("Precision TXT : %s", all_txt_path)


if __name__ == "__main__":
    main()