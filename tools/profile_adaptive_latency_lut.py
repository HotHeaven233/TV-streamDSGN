#!/usr/bin/env python3
"""
Profile the current StreamDSGN selective A / B / CD physical primitives and
build a *provisional* 5x5x5 adaptive latency LUT.

Why "provisional"?
------------------
The current repository has:
  - physical Stage-A selective execution integrated in StreamDSGN2Backbone;
  - physical B cost-volume ROI primitive (BuildCostVolume.forward_roi);
  - physical C/D ROI primitives used by the existing profiling/oracle tools;

but B/CD + learned importance/predictor + R/P/U + scheduler are not yet exposed
as one unified adaptive detector forward. Therefore this script does NOT pretend
that an additive component LUT is the final authoritative 125-action latency LUT.

It profiles real physical component kernels with the same latency semantics used
by the true-stream baseline:

    load_data_to_gpu(...) is NOT timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    forward/component work
    torch.cuda.synchronize()
    elapsed = t1 - t0

Outputs
-------
1) adaptive_latency_components.json
   Raw samples and statistics for:
     - original full forward reference
     - A_s diagnostic
     - original full A
     - A selective levels
     - original full B
     - B selective levels
     - original full CD
     - CD selective levels
     - fixed downstream path
     - learned adaptive fixed overhead (when local modules are available)

2) adaptive_latency_lut_provisional.json
   125 additive estimates:

       T_hat(a,b,cd)
         = T_A(a)
         + T_B(b)
         + T_CD(cd)
         + T_fixed_downstream
         + T_adaptive_fixed

   The JSON explicitly contains:
       "authoritative": false

Final paper/scheduler numbers must be replaced by a true joint LUT after the
unified adaptive runtime is implemented.
"""

import argparse
import copy
import csv
import json
import math
import random
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

# Reuse the repository's already validated physical primitives.
import profile_b2_roi_fragmentation as bbase
import profile_c_roi_fragmentation as cprof
import profile_roi_fragmentation as dprof


DEFAULT_CFG = (
    "configs/stream/kitti_models/"
    "stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-"
    "lka_7-mcl_5090_eval.yaml"
)
DEFAULT_CKPT = "extra_data/checkpoint_epoch_20.pth"


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Profile adaptive A/B/CD component latency and build provisional 5x5x5 LUT"
    )

    p.add_argument("--cfg_file", default=DEFAULT_CFG)
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument(
        "--adaptive_ckpt",
        default=None,
        help=(
            "Checkpoint produced by train_adaptive_joint.py. "
            "If supplied, loads importance_net and pred_net weights."
        ),
    )

    p.add_argument(
        "--levels",
        default="0.15,0.25,0.40,0.60,0.80",
        help="Five area ratios for A/B/CD, comma separated.",
    )
    p.add_argument(
        "--positions",
        default="center,random,boundary",
        help="ROI positions used to test location sensitivity.",
    )

    p.add_argument(
        "--sample_indices",
        default="10,50,100",
        help=(
            "Dataset indices used for component profiling. "
            "Use one index for a quick smoke test."
        ),
    )

    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=30)

    p.add_argument("--b_halo", type=int, default=1)
    p.add_argument("--cd_halo", type=int, default=12)
    p.add_argument("--align", type=int, default=4)
    p.add_argument(
        "--numeric_tol",
        type=float,
        default=0.005,
        help=(
            "Core-region max-abs-diff warning threshold for physical ROI "
            "equivalence checks. Existing joint B/CD oracle uses 0.005."
        ),
    )
    p.add_argument(
        "--strict_numeric",
        action="store_true",
        help="Abort when a physical ROI core exceeds --numeric_tol.",
    )

    p.add_argument("--importance_hidden", type=int, default=32)
    p.add_argument("--predictor_hidden", type=int, default=64)
    p.add_argument("--age_cap", type=float, default=8.0)

    p.add_argument(
        "--margin_ratio",
        type=float,
        default=0.02,
        help="Multiplicative safety margin applied to provisional P99 sum.",
    )
    p.add_argument(
        "--margin_ms",
        type=float,
        default=0.20,
        help="Fixed safety margin applied after multiplicative margin.",
    )

    p.add_argument(
        "--output_dir",
        default="outputs/adaptive_profile/latency_lut",
    )
    p.add_argument(
        "--no_raw_samples",
        action="store_true",
        help="Do not store raw timing samples in component JSON.",
    )
    p.add_argument(
        "--skip_adaptive_fixed",
        action="store_true",
        help="Skip ImportanceNet / BEV predictor / RPU overhead profiling.",
    )

    return p.parse_args()


# ============================================================================
# General helpers
# ============================================================================

def parse_float_list(s):
    xs = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if not xs:
        raise ValueError("empty float list")
    return xs


def parse_int_list(s):
    xs = [int(x.strip()) for x in str(s).split(",") if x.strip()]
    if not xs:
        raise ValueError("empty integer list")
    return xs


def parse_str_list(s):
    xs = [x.strip() for x in str(s).split(",") if x.strip()]
    if not xs:
        raise ValueError("empty string list")
    return xs


def percentile(xs, q):
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))


def summarize(xs):
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "min_ms": float("nan"),
            "p50_ms": float("nan"),
            "p90_ms": float("nan"),
            "p95_ms": float("nan"),
            "p99_ms": float("nan"),
            "max_ms": float("nan"),
        }

    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "p50_ms": percentile(arr, 50),
        "p90_ms": percentile(arr, 90),
        "p95_ms": percentile(arr, 95),
        "p99_ms": percentile(arr, 99),
        "max_ms": float(arr.max()),
    }


def timed_once(fn):
    """
    Forward-only wall-time measurement.

    This intentionally matches the user's true-stream baseline timing boundary:
    data preparation/H2D is outside; CUDA work is synchronized around the call.
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    return float(ms), out


def measure(fn, warmup, repeat):
    with torch.no_grad():
        for _ in range(warmup):
            fn()

    torch.cuda.synchronize()

    samples = []
    with torch.no_grad():
        for _ in range(repeat):
            ms, _ = timed_once(fn)
            samples.append(ms)

    return samples


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


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(data),
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )
    )


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


def fresh_batch(batch):
    """
    Fresh Python dictionaries without copying CUDA tensor storage.
    """
    out = {}
    for k, v in batch.items():
        out[k] = dict(v) if isinstance(v, dict) else v
    return out


def fresh_dict(d):
    return dict(d)


def find_module_by_name(model, class_name):
    for m in model.modules():
        if type(m).__name__ == class_name:
            return m
    raise RuntimeError(f"{class_name} not found")


def reset_stream_history(model):
    q = getattr(model, "history_feature_queue", None)
    if q is not None:
        q.clear()


def amp_enabled(model):
    d = getattr(model, "use_amp_dict", {})
    if isinstance(d, dict):
        return bool(d.get("TEST", False))
    return bool(getattr(model, "use_amp", False))


def autocast_ctx(model):
    return torch.amp.autocast("cuda", enabled=amp_enabled(model))


# ============================================================================
# ROI geometry
# ============================================================================

def rect_from_ratio(H, W, ratio, position, align):
    """
    One rectangular semantic ROI preserving the full-map aspect ratio
    approximately:

        roi_h / H ~= roi_w / W ~= sqrt(ratio)
    """
    ratio = float(ratio)

    if ratio <= 0:
        return None
    if ratio >= 1:
        return (0, H, 0, W)

    scale = math.sqrt(ratio)

    rh = int(round(H * scale / align)) * align
    rw = int(round(W * scale / align)) * align

    rh = min(H, max(align, rh))
    rw = min(W, max(align, rw))

    if position == "center":
        y0 = (H - rh) // 2
        x0 = (W - rw) // 2

    elif position == "boundary":
        y0 = H - rh
        x0 = W - rw

    elif position == "random":
        # Deterministic pseudo-random position: stable across repeated profiling.
        sy = H - rh
        sx = W - rw
        y0 = 0 if sy == 0 else ((37 * sy + 11 * sx + 17) % (sy + 1))
        x0 = 0 if sx == 0 else ((53 * sx + 7 * sy + 29) % (sx + 1))

    else:
        raise ValueError(f"unknown position: {position}")

    return (int(y0), int(y0 + rh), int(x0), int(x0 + rw))


def expand_align(rect, halo, H, W, align):
    y0, y1, x0, x1 = rect

    y0 = max(0, y0 - halo)
    y1 = min(H, y1 + halo)
    x0 = max(0, x0 - halo)
    x1 = min(W, x1 + halo)

    y0 = (y0 // align) * align
    x0 = (x0 // align) * align

    y1 = min(H, int(math.ceil(y1 / align) * align))
    x1 = min(W, int(math.ceil(x1 / align) * align))

    if y1 <= y0 or x1 <= x0:
        raise RuntimeError(f"invalid aligned ROI: {(y0, y1, x0, x1)}")

    return (y0, y1, x0, x1)


def area(rect):
    if rect is None:
        return 0
    y0, y1, x0, x1 = rect
    return max(0, y1 - y0) * max(0, x1 - x0)


def core_max_abs_diff(pred, ref, rect):
    y0, y1, x0, x1 = rect
    a = pred[..., y0:y1, x0:x1].float()
    b = ref[..., y0:y1, x0:x1].float()
    if tuple(a.shape) != tuple(b.shape):
        raise RuntimeError(
            f"core shape mismatch pred={tuple(a.shape)} ref={tuple(b.shape)} rect={rect}"
        )
    return float((a - b).abs().max().item())


def check_numeric_or_warn(logger, stage, level, position, diff, tol, strict):
    if diff <= tol:
        return
    msg = (
        f"[{stage} NUMERIC] level={level} position={position} "
        f"max_abs_diff={diff:.6g} > tol={tol:.6g}"
    )
    if strict:
        raise RuntimeError(msg)
    logger.warning(msg)


# ============================================================================
# Build environment
# ============================================================================

def build_env(args):
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.LOCAL_RANK = 0
    cfg.MODEL.SAVE_TIME = False

    random.seed(1024)
    np.random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)

    logger = common_utils.create_logger(rank=0)

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

    backbone = find_module_by_name(model, "StreamDSGN2Backbone")
    height_compression = find_module_by_name(model, "HeightCompression")

    return dataset, loader, model, backbone, height_compression, logger


# ============================================================================
# Capture a real frame's physical stage inputs
# ============================================================================

@torch.no_grad()
def capture_context(dataset, model, backbone, sample_index):
    """
    Run the full feature extractor once and capture exact real tensors used by
    B, C and D. H2D is performed before profiling and is never timed.
    """
    sample = dataset[sample_index]
    batch = dataset.collate_batch([sample])
    load_data_to_gpu(batch)

    token = batch["token"]

    # Never accidentally capture an A-selective forward here.
    if hasattr(backbone, "clear_adaptive_a_profile"):
        backbone.clear_adaptive_a_profile()

    captured = {}

    def cost_pre_hook(module, inputs):
        captured["b_left"] = inputs[0].detach().clone()
        captured["b_right"] = inputs[1].detach().clone()
        captured["b_shift"] = inputs[4].detach().clone()
        captured["b_psv"] = (
            inputs[5].detach().clone()
            if len(inputs) >= 6 and inputs[5] is not None
            else None
        )

    def dres0_hook(module, inputs, output):
        # BuildCostVolume.forward_roi() is defined on the COST-VOLUME output
        # H/W grid (80x312 in the current config), not on b_left/b_right
        # (320x1248). dres0 input is the exact full cost-volume tensor, so its
        # H/W is the authoritative B ROI coordinate system. Store only shape to
        # avoid an unnecessary large tensor clone.
        captured["b_hw"] = tuple(int(v) for v in inputs[0].shape[-2:])
        captured["dres0"] = output.detach().clone()

    def dres1_hook(module, inputs, output):
        captured["dres1"] = output.detach().clone()

    def d_input_hook(module, inputs):
        captured["voxel_ref"] = inputs[0].detach().clone()

    h_cost = backbone.build_cost.register_forward_pre_hook(cost_pre_hook)
    h_d0 = backbone.dres0.register_forward_hook(dres0_hook)
    h_d1 = backbone.dres1.register_forward_hook(dres1_hook)
    h_din = backbone.rpn3d_convs.register_forward_pre_hook(d_input_hook)

    reset_stream_history(model)

    try:
        cur = dict(token)
        pre_hc = None

        with autocast_ctx(model):
            for m in model.feature_extractor:
                if type(m).__name__ == "HeightCompression":
                    pre_hc = dict(cur)
                cur = m(cur)

        torch.cuda.synchronize()

    finally:
        h_cost.remove()
        h_d0.remove()
        h_d1.remove()
        h_din.remove()

    if pre_hc is None:
        raise RuntimeError("HeightCompression boundary not found in feature_extractor")

    required = [
        "b_left", "b_right", "b_shift", "b_hw",
        "dres0", "dres1", "voxel_ref",
    ]
    missing = [k for k in required if k not in captured]
    if missing:
        raise RuntimeError(f"capture failed, missing: {missing}")

    if captured["b_psv"] is None:
        raise RuntimeError(
            "B ROI primitive requires DPS psv_disps_channels, but capture returned None"
        )

    hg = getattr(backbone, "hg_stereo", [])
    if len(hg) != 0:
        raise RuntimeError(
            "Current component B profiler only covers cost-volume + dres0 + dres1. "
            f"hg_stereo has {len(hg)} modules, so do not use this profiler for this config."
        )

    stereo_out = (captured["dres0"] + captured["dres1"]).contiguous()

    b_hw = tuple(captured["b_hw"])
    if tuple(stereo_out.shape[-2:]) != b_hw:
        raise RuntimeError(
            "Current B ROI profiler assumes dres0/dres1 preserve cost-volume "
            f"H/W, but cost grid={b_hw} and B output={tuple(stereo_out.shape[-2:])}."
        )

    volume_features = pre_hc["volume_features"].detach().clone().contiguous()
    spatial_features = cur["spatial_features"].detach().clone().contiguous()

    left_img = token["left_img"]
    right_img = token["right_img"]

    # C-stage shared geometry tensors.
    coordinates_3d = backbone.coordinates_3d.cuda()
    if bool(getattr(backbone, "use_amp", False)):
        coordinates_3d = coordinates_3d.half()
    coordinates_3d = coordinates_3d.contiguous()

    cd_hw = tuple(int(v) for v in coordinates_3d.shape[1:3])
    if tuple(volume_features.shape[-2:]) != cd_hw:
        raise RuntimeError(
            "Current CD ROI profiler assumes D output preserves BEV H/W, "
            f"but coordinates grid={cd_hw} and volume_features="
            f"{tuple(volume_features.shape[-2:])}."
        )

    calib = token["calib"][0]
    p2_dtype = torch.float16 if bool(getattr(backbone, "use_amp", False)) else torch.float32
    P2 = torch.as_tensor(calib.P2, device="cuda", dtype=p2_dtype)

    image_shape = token["image_shape"][0]
    random_T = token["random_T"][0] if "random_T" in token else None

    # Precompute A_s outputs for learned ImportanceNet timing. A_s timing itself
    # is profiled separately; these tensors are not recomputed inside learned
    # adaptive-fixed timing.
    with autocast_ctx(model):
        shallow_left = backbone.forward_2d_shallow(left_img).detach().clone()
        shallow_right = backbone.forward_2d_shallow(right_img).detach().clone()

    torch.cuda.synchronize()

    scene = str(first_scalar(token.get("scene", "")))
    frame = str(first_scalar(token.get("this_sample_idx", sample_index)))

    return {
        "batch": batch,
        "token": token,
        "scene": scene,
        "frame": frame,

        "left_img": left_img,
        "right_img": right_img,
        "shallow_left": shallow_left,
        "shallow_right": shallow_right,

        "b_left": captured["b_left"].contiguous(),
        "b_right": captured["b_right"].contiguous(),
        "b_shift": captured["b_shift"].contiguous(),
        "b_psv": captured["b_psv"].contiguous(),
        "b_hw": b_hw,

        "stereo_out": stereo_out,
        "voxel_ref": captured["voxel_ref"].contiguous(),

        "coordinates_3d": coordinates_3d,
        "P2": P2,
        "image_shape": image_shape,
        "random_T": random_T,

        "pre_hc": pre_hc,
        "post_hc": cur,
        "volume_features": volume_features,
        "spatial_features": spatial_features,
    }


# ============================================================================
# Component execution functions
# ============================================================================

@torch.no_grad()
def run_a_shallow_pair(model, backbone, ctx):
    with autocast_ctx(model):
        backbone.forward_2d_shallow(ctx["left_img"])
        backbone.forward_2d_shallow(ctx["right_img"])


@torch.no_grad()
def run_a_full_pair(model, backbone, ctx):
    with autocast_ctx(model):
        backbone.forward_2d(ctx["left_img"])
        backbone.forward_2d(ctx["right_img"])


@torch.no_grad()
def run_a_adaptive_pair(model, backbone, ctx):
    with autocast_ctx(model):
        backbone.forward_2d_adaptive(ctx["left_img"], side="left")
        backbone.forward_2d_adaptive(ctx["right_img"], side="right")


def init_a_cache(model, backbone, ctx):
    backbone.reset_adaptive_a_cache()
    backbone.set_adaptive_a_profile(
        ratio=1.0,
        position_mode="center",
    )
    run_a_adaptive_pair(model, backbone, ctx)
    torch.cuda.synchronize()


@torch.no_grad()
def run_b_full(backbone, ctx):
    cost = backbone.build_cost(
        ctx["b_left"],
        ctx["b_right"],
        None,
        None,
        ctx["b_shift"],
        ctx["b_psv"],
    )
    return bbase.run_b2(backbone, cost)


def make_b_selective_fn(backbone, ctx, ratio, position, halo, align):
    # IMPORTANT: forward_roi() coordinates are on the output cost-volume grid.
    H, W = ctx["b_hw"]
    core = rect_from_ratio(H, W, ratio, position, align)
    if core is None:
        raise RuntimeError("zero-area B level is not supported by this profiler")

    exec_rect = expand_align(core, halo, H, W, align)
    canvas = torch.empty_like(ctx["stereo_out"])

    cy0, cy1, cx0, cx1 = core
    ey0, ey1, ex0, ex1 = exec_rect

    def fn():
        cost_roi = backbone.build_cost.forward_roi(
            ctx["b_left"],
            ctx["b_right"],
            None,
            None,
            ctx["b_shift"],
            ey0,
            ey1,
            ex0,
            ex1,
            ctx["b_psv"],
        )

        out_roi = bbase.run_b2(backbone, cost_roi)

        ry0 = cy0 - ey0
        rx0 = cx0 - ex0
        rh = cy1 - cy0
        rw = cx1 - cx0

        canvas[..., cy0:cy1, cx0:cx1].copy_(
            out_roi[
                ...,
                ry0:ry0 + rh,
                rx0:rx0 + rw,
            ]
        )
        return canvas

    meta = {
        "semantic_roi": list(core),
        "exec_roi": list(exec_rect),
        "semantic_area_ratio": area(core) / float(H * W),
        "exec_area_ratio": area(exec_rect) / float(H * W),
    }
    return fn, meta


@torch.no_grad()
def run_cd_full(backbone, ctx):
    _, H, W, _ = ctx["coordinates_3d"].shape
    full = (0, H, 0, W)

    c = cprof.run_c_roi(
        backbone,
        ctx["stereo_out"],
        ctx["coordinates_3d"],
        full,
        ctx["left_img"].shape[2:],
        ctx["image_shape"],
        ctx["P2"],
        random_T=ctx["random_T"],
    )
    return dprof.run_d_stage(backbone, c)


def make_cd_selective_fn(backbone, ctx, ratio, position, halo, align):
    _, H, W, _ = ctx["coordinates_3d"].shape

    core = rect_from_ratio(H, W, ratio, position, align)
    if core is None:
        raise RuntimeError("zero-area CD level is not supported by this profiler")

    exec_rect = expand_align(core, halo, H, W, align)

    canvas = torch.empty_like(ctx["volume_features"])

    cy0, cy1, cx0, cx1 = core
    ey0, ey1, ex0, ex1 = exec_rect

    def fn():
        c_roi = cprof.run_c_roi(
            backbone,
            ctx["stereo_out"],
            ctx["coordinates_3d"],
            exec_rect,
            ctx["left_img"].shape[2:],
            ctx["image_shape"],
            ctx["P2"],
            random_T=ctx["random_T"],
        )

        d_roi = dprof.run_d_stage(backbone, c_roi)

        ry0 = cy0 - ey0
        rx0 = cx0 - ex0
        rh = cy1 - cy0
        rw = cx1 - cx0

        if d_roi.shape[-2] < ry0 + rh or d_roi.shape[-1] < rx0 + rw:
            raise RuntimeError(
                "D ROI output spatial shape is incompatible with core scatter: "
                f"d_roi={tuple(d_roi.shape)}, core={core}, exec={exec_rect}"
            )

        canvas[..., cy0:cy1, cx0:cx1].copy_(
            d_roi[
                ...,
                ry0:ry0 + rh,
                rx0:rx0 + rw,
            ]
        )

        return canvas

    meta = {
        "semantic_roi": list(core),
        "exec_roi": list(exec_rect),
        "semantic_area_ratio": area(core) / float(H * W),
        "exec_area_ratio": area(exec_rect) / float(H * W),
    }
    return fn, meta


def make_history(model, spatial_features):
    tags = getattr(model, "history_tag", None)
    if tags is None:
        return deque()

    q = deque(maxlen=len(tags))
    for i in range(len(tags)):
        q.append(
            (
                f"profile_hist_{i}",
                {"spatial_features": spatial_features},
            )
        )
    return q


def make_downstream_fn(model, height_compression, ctx):
    """
    Fixed path beginning at full-size D output:
      HeightCompression -> FeatureAlignment -> VAN -> StreamDetHead -> post_processing

    The history queue is pre-existing state and not constructed by an expensive
    network operation; only a lightweight Python reference assignment occurs.
    """
    base_pre_hc = dict(ctx["pre_hc"])
    volume = ctx["volume_features"]
    history = make_history(model, ctx["spatial_features"])

    def fn():
        cur = dict(base_pre_hc)
        cur["volume_features"] = volume

        with autocast_ctx(model):
            cur = height_compression(cur)
            cur["history_features"] = history

            for m in model.fusion_module:
                cur = m(cur)

            for m in model.after_fusion_blocks:
                cur = m(cur)

            out = model.post_processing(cur)

        return out

    return fn


def make_height_compression_fn(model, height_compression, ctx):
    base = dict(ctx["pre_hc"])
    volume = ctx["volume_features"]

    def fn():
        cur = dict(base)
        cur["volume_features"] = volume
        with autocast_ctx(model):
            return height_compression(cur)

    return fn


def make_after_hc_downstream_fn(model, ctx):
    base = dict(ctx["post_hc"])
    history = make_history(model, ctx["spatial_features"])

    def fn():
        cur = dict(base)
        cur["spatial_features"] = ctx["spatial_features"]
        cur["history_features"] = history

        with autocast_ctx(model):
            for m in model.fusion_module:
                cur = m(cur)
            for m in model.after_fusion_blocks:
                cur = m(cur)
            return model.post_processing(cur)

    return fn


def make_full_forward_fn(model, ctx):
    batch = ctx["batch"]

    def fn():
        return model(fresh_batch(batch))

    return fn


# ============================================================================
# Learned adaptive fixed overhead
# ============================================================================

def build_adaptive_modules(args, ctx, logger):
    if args.skip_adaptive_fixed:
        return None

    try:
        from pcdet.models.adaptive_single_roi.stage_importance_predictor import (
            StageImportancePredictor,
        )
        from pcdet.models.adaptive_single_roi.bev_predictor import (
            BEVTemporalPredictor,
        )
    except Exception as e:
        logger.warning(
            "Adaptive fixed overhead will be omitted because local adaptive "
            "modules could not be imported: %r",
            e,
        )
        return {
            "available": False,
            "reason": repr(e),
        }

    shallow_channels = int(ctx["shallow_left"].shape[1])
    bev_channels = int(ctx["spatial_features"].shape[1])

    importance = StageImportancePredictor(
        shallow_channels=shallow_channels,
        bev_channels=bev_channels,
        hidden=args.importance_hidden,
        age_cap=args.age_cap,
    ).cuda().eval()

    predictor = BEVTemporalPredictor(
        channels=bev_channels,
        hidden=args.predictor_hidden,
        age_cap=args.age_cap,
    ).cuda().eval()

    weights_loaded = False

    if args.adaptive_ckpt is not None:
        ckpt = torch.load(args.adaptive_ckpt, map_location="cpu")

        if "importance_net" not in ckpt or "pred_net" not in ckpt:
            raise KeyError(
                "adaptive checkpoint must contain 'importance_net' and 'pred_net'"
            )

        importance.load_state_dict(ckpt["importance_net"], strict=True)
        predictor.load_state_dict(ckpt["pred_net"], strict=True)
        weights_loaded = True

    for m in (importance, predictor):
        for p in m.parameters():
            p.requires_grad_(False)

    return {
        "available": True,
        "importance": importance,
        "predictor": predictor,
        "weights_loaded": weights_loaded,
    }


def make_adaptive_fixed_fn(model, adaptive_modules, ctx):
    if adaptive_modules is None:
        return None, {"available": False, "reason": "--skip_adaptive_fixed"}

    if not adaptive_modules.get("available", False):
        return None, adaptive_modules

    importance = adaptive_modules["importance"]
    predictor = adaptive_modules["predictor"]

    cur_l = ctx["shallow_left"]
    cur_r = ctx["shallow_right"]
    prev_l = cur_l
    prev_r = cur_r

    bev = ctx["spatial_features"]
    prev1 = bev
    prev2 = bev

    # Training defines both Q_A and Q_B on the native 80x312 grid.
    # A's dense stereo output itself is 4x larger (320x1248), so b_left.shape
    # must NOT be used here.
    aH, aW = ctx["b_hw"]
    bH, bW = aH, aW
    cdH, cdW = bev.shape[-2:]

    age_a = torch.zeros(
        (bev.shape[0], 1, aH, aW),
        device=bev.device,
        dtype=torch.float32,
    )
    age_b = torch.zeros(
        (bev.shape[0], 1, bH, bW),
        device=bev.device,
        dtype=torch.float32,
    )
    age_f = torch.zeros(
        (bev.shape[0], 1, cdH, cdW),
        device=bev.device,
        dtype=torch.float32,
    )

    # Representative R/P/U masks. Their values do not materially affect latency.
    mask_r = torch.zeros(
        (bev.shape[0], 1, cdH, cdW),
        device=bev.device,
        dtype=bev.dtype,
    )
    y0, y1 = cdH // 4, 3 * cdH // 4
    x0, x1 = cdW // 4, 3 * cdW // 4
    mask_r[..., y0:y1, x0:x1] = 1

    mask_p = torch.zeros_like(mask_r)
    py0, py1 = max(0, y0 - 12), min(cdH, y1 + 12)
    px0, px1 = max(0, x0 - 12), min(cdW, x1 + 12)
    mask_p[..., py0:py1, px0:px1] = 1
    mask_p = (mask_p - mask_r).clamp(min=0)

    mask_u = (1 - mask_r - mask_p).clamp(min=0)

    def fn():
        with autocast_ctx(model):
            q = importance(
                cur_l,
                cur_r,
                prev_l,
                prev_r,
                prev1,
                age_a,
                age_b,
                age_f,
                (aH, aW),
                (bH, bW),
                (cdH, cdW),
            )

            pred = predictor(
                prev1,
                prev2,
                age_f,
            )

            # R/P/U dense composition only.
            mixed = (
                mask_r * bev
                + mask_p * pred
                + mask_u * prev1
            )

        return q, pred, mixed

    meta = {
        "available": True,
        "weights_loaded": bool(adaptive_modules["weights_loaded"]),
        "includes": [
            "StageImportancePredictor",
            "BEVTemporalPredictor",
            "dense R/P/U composition",
        ],
        "omits": [
            "hard ROI window scoring / argmax",
            "125-action scheduler search",
            "joint B<->CD dependency-completion control logic",
        ],
    }

    return fn, meta


# ============================================================================
# Profiling accumulation
# ============================================================================

def nested_samples(levels, positions):
    return {
        i: {pos: [] for pos in positions}
        for i in range(len(levels))
    }


def add_samples(dst, xs):
    dst.extend(float(x) for x in xs)


def stats_with_optional_raw(xs, save_raw):
    out = summarize(xs)
    if save_raw:
        out["samples_ms"] = [float(x) for x in xs]
    return out


def aggregate_level_position(samples_dict, levels, positions, save_raw):
    records = []

    for i, ratio in enumerate(levels):
        pos_stats = {}
        for pos in positions:
            pos_stats[pos] = stats_with_optional_raw(
                samples_dict[i][pos],
                save_raw,
            )

        worst_pos = max(
            positions,
            key=lambda p: pos_stats[p]["p99_ms"],
        )

        records.append(
            {
                "level": i,
                "requested_ratio": float(ratio),
                "positions": pos_stats,
                "worst_position": worst_pos,
                "worst_p99_ms": float(pos_stats[worst_pos]["p99_ms"]),
                "worst_mean_ms": float(
                    max(pos_stats[p]["mean_ms"] for p in positions)
                ),
            }
        )

    return records


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    levels = parse_float_list(args.levels)
    positions = parse_str_list(args.positions)
    sample_indices = parse_int_list(args.sample_indices)

    if len(levels) != 5:
        raise ValueError(
            f"Current method requires exactly 5 levels; got {len(levels)}: {levels}"
        )

    for r in levels:
        if not (0.0 < r <= 1.0):
            raise ValueError(f"ratio must be in (0,1], got {r}")

    for p in positions:
        if p not in ("center", "random", "boundary"):
            raise ValueError(f"unsupported position: {p}")

    if args.repeat <= 0 or args.warmup < 0:
        raise ValueError("invalid warmup/repeat")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        dataset,
        loader,
        model,
        backbone,
        height_compression,
        logger,
    ) = build_env(args)

    # ------------------------------------------------------------------------
    # State containers.
    # ------------------------------------------------------------------------
    save_raw = not args.no_raw_samples

    full_forward_samples = []
    a_s_samples = []
    a_full_samples = []
    b_full_samples = []
    cd_full_samples = []
    fixed_from_volume_samples = []
    hc_samples = []
    after_hc_downstream_samples = []
    adaptive_fixed_samples = []

    a_samples = nested_samples(levels, positions)
    b_samples = nested_samples(levels, positions)
    cd_samples = nested_samples(levels, positions)

    a_meta = defaultdict(dict)
    b_meta = defaultdict(dict)
    cd_meta = defaultdict(dict)

    frame_records = []

    adaptive_modules = None
    adaptive_meta = {
        "available": False,
        "reason": "not initialized",
    }

    logger.info("=" * 100)
    logger.info("Adaptive latency component profiler")
    logger.info("latency semantics : forward-only wall time")
    logger.info("H2D/data prep      : EXCLUDED")
    logger.info("levels             : %s", levels)
    logger.info("positions          : %s", positions)
    logger.info("sample indices     : %s", sample_indices)
    logger.info("warmup/repeat      : %d / %d", args.warmup, args.repeat)
    logger.info("B halo / CD halo   : %d / %d", args.b_halo, args.cd_halo)
    logger.info("align              : %d", args.align)
    logger.info("=" * 100)

    # ------------------------------------------------------------------------
    # Profile each real frame independently, then aggregate.
    # ------------------------------------------------------------------------
    for frame_no, sample_index in enumerate(sample_indices):
        if sample_index < 0 or sample_index >= len(dataset):
            raise IndexError(
                f"sample index {sample_index} outside dataset length {len(dataset)}"
            )

        logger.info("")
        logger.info(
            "[FRAME %d/%d] dataset index=%d",
            frame_no + 1,
            len(sample_indices),
            sample_index,
        )

        ctx = capture_context(
            dataset,
            model,
            backbone,
            sample_index,
        )

        logger.info(
            "  scene/frame=%s/%s B-grid=%s A-stereo-HW=%s CD-HW=%s BEV-HW=%s",
            ctx["scene"],
            ctx["frame"],
            tuple(ctx["b_hw"]),
            tuple(ctx["b_left"].shape[-2:]),
            tuple(ctx["coordinates_3d"].shape[1:3]),
            tuple(ctx["spatial_features"].shape[-2:]),
        )

        # Instantiate learned adaptive modules once dimensions are known.
        if adaptive_modules is None and not args.skip_adaptive_fixed:
            adaptive_modules = build_adaptive_modules(
                args,
                ctx,
                logger,
            )

        # --------------------------------------------------------------------
        # Original full forward reference.
        # --------------------------------------------------------------------
        if hasattr(backbone, "clear_adaptive_a_profile"):
            backbone.clear_adaptive_a_profile()
        reset_stream_history(model)

        full_fn = make_full_forward_fn(model, ctx)
        xs = measure(full_fn, args.warmup, args.repeat)
        add_samples(full_forward_samples, xs)
        logger.info("  Full forward mean=%.3f ms", np.mean(xs))

        # --------------------------------------------------------------------
        # A diagnostics: A_s and original full A.
        # --------------------------------------------------------------------
        xs = measure(
            lambda: run_a_shallow_pair(model, backbone, ctx),
            args.warmup,
            args.repeat,
        )
        add_samples(a_s_samples, xs)

        xs = measure(
            lambda: run_a_full_pair(model, backbone, ctx),
            args.warmup,
            args.repeat,
        )
        add_samples(a_full_samples, xs)

        # --------------------------------------------------------------------
        # A selective.
        # --------------------------------------------------------------------
        for i, ratio in enumerate(levels):
            for pos in positions:
                init_a_cache(model, backbone, ctx)

                backbone.set_adaptive_a_profile(
                    ratio=ratio,
                    position_mode=pos,
                )

                xs = measure(
                    lambda: run_a_adaptive_pair(model, backbone, ctx),
                    args.warmup,
                    args.repeat,
                )
                add_samples(a_samples[i][pos], xs)

                # Same-frame Full cache initialization makes this an exact/local
                # equivalence check of the physical A ROI implementation.
                a_left_now = backbone._adaptive_a_stereo_cache["left"]
                a_right_now = backbone._adaptive_a_stereo_cache["right"]
                a_diff = max(
                    float((a_left_now.float() - ctx["b_left"].float()).abs().max().item()),
                    float((a_right_now.float() - ctx["b_right"].float()).abs().max().item()),
                )
                check_numeric_or_warn(
                    logger, "A", i, pos, a_diff, args.numeric_tol, args.strict_numeric
                )

                roi = getattr(backbone, "_adaptive_a_last_roi", None)
                if roi is not None:
                    H = backbone._adaptive_a_cache["left"][0].shape[-2]
                    W = backbone._adaptive_a_cache["left"][0].shape[-1]
                    a_meta[i][pos] = {
                        "roi": list(roi),
                        "actual_ratio": area(roi) / float(H * W),
                        "max_abs_diff_same_frame": a_diff,
                    }

        backbone.clear_adaptive_a_profile()

        # --------------------------------------------------------------------
        # B full and B selective.
        # --------------------------------------------------------------------
        xs = measure(
            lambda: run_b_full(backbone, ctx),
            args.warmup,
            args.repeat,
        )
        add_samples(b_full_samples, xs)

        for i, ratio in enumerate(levels):
            for pos in positions:
                fn, meta = make_b_selective_fn(
                    backbone,
                    ctx,
                    ratio,
                    pos,
                    args.b_halo,
                    args.align,
                )

                # One untimed physical equivalence check on the semantic core.
                with torch.no_grad():
                    b_canvas = fn()
                b_core = tuple(meta["semantic_roi"])
                b_diff = core_max_abs_diff(
                    b_canvas, ctx["stereo_out"], b_core
                )
                meta["max_abs_diff_core"] = b_diff
                check_numeric_or_warn(
                    logger, "B", i, pos, b_diff, args.numeric_tol, args.strict_numeric
                )

                xs = measure(
                    fn,
                    args.warmup,
                    args.repeat,
                )
                add_samples(b_samples[i][pos], xs)
                b_meta[i][pos] = meta

        # --------------------------------------------------------------------
        # CD full and CD selective.
        # --------------------------------------------------------------------
        xs = measure(
            lambda: run_cd_full(backbone, ctx),
            args.warmup,
            args.repeat,
        )
        add_samples(cd_full_samples, xs)

        for i, ratio in enumerate(levels):
            for pos in positions:
                fn, meta = make_cd_selective_fn(
                    backbone,
                    ctx,
                    ratio,
                    pos,
                    args.cd_halo,
                    args.align,
                )

                # One untimed physical equivalence check on the semantic core.
                with torch.no_grad():
                    cd_canvas = fn()
                cd_core = tuple(meta["semantic_roi"])
                cd_diff = core_max_abs_diff(
                    cd_canvas, ctx["volume_features"], cd_core
                )
                meta["max_abs_diff_core"] = cd_diff
                check_numeric_or_warn(
                    logger, "CD", i, pos, cd_diff, args.numeric_tol, args.strict_numeric
                )

                xs = measure(
                    fn,
                    args.warmup,
                    args.repeat,
                )
                add_samples(cd_samples[i][pos], xs)
                cd_meta[i][pos] = meta

        # --------------------------------------------------------------------
        # Fixed downstream.
        # --------------------------------------------------------------------
        fixed_fn = make_downstream_fn(
            model,
            height_compression,
            ctx,
        )
        xs = measure(
            fixed_fn,
            args.warmup,
            args.repeat,
        )
        add_samples(fixed_from_volume_samples, xs)

        hc_fn = make_height_compression_fn(
            model,
            height_compression,
            ctx,
        )
        xs = measure(
            hc_fn,
            args.warmup,
            args.repeat,
        )
        add_samples(hc_samples, xs)

        after_hc_fn = make_after_hc_downstream_fn(
            model,
            ctx,
        )
        xs = measure(
            after_hc_fn,
            args.warmup,
            args.repeat,
        )
        add_samples(after_hc_downstream_samples, xs)

        # --------------------------------------------------------------------
        # Learned adaptive fixed overhead.
        # --------------------------------------------------------------------
        if not args.skip_adaptive_fixed:
            adaptive_fn, this_meta = make_adaptive_fixed_fn(
                model,
                adaptive_modules,
                ctx,
            )
            adaptive_meta = this_meta

            if adaptive_fn is not None:
                xs = measure(
                    adaptive_fn,
                    args.warmup,
                    args.repeat,
                )
                add_samples(adaptive_fixed_samples, xs)

        frame_records.append(
            {
                "sample_index": sample_index,
                "scene": ctx["scene"],
                "frame": ctx["frame"],
                "b_hw": list(ctx["b_hw"]),
                "a_stereo_hw": list(ctx["b_left"].shape[-2:]),
                "cd_hw": list(ctx["coordinates_3d"].shape[1:3]),
                "bev_hw": list(ctx["spatial_features"].shape[-2:]),
            }
        )

        # Avoid retaining multiple large physical stage canvases between frames.
        del ctx
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------------
    # Aggregate component statistics.
    # ------------------------------------------------------------------------
    a_records = aggregate_level_position(
        a_samples,
        levels,
        positions,
        save_raw,
    )
    b_records = aggregate_level_position(
        b_samples,
        levels,
        positions,
        save_raw,
    )
    cd_records = aggregate_level_position(
        cd_samples,
        levels,
        positions,
        save_raw,
    )

    for rec in a_records:
        rec["roi_meta"] = a_meta.get(rec["level"], {})
    for rec in b_records:
        rec["roi_meta"] = b_meta.get(rec["level"], {})
    for rec in cd_records:
        rec["roi_meta"] = cd_meta.get(rec["level"], {})

    full_forward_stat = stats_with_optional_raw(
        full_forward_samples,
        save_raw,
    )
    a_s_stat = stats_with_optional_raw(a_s_samples, save_raw)
    a_full_stat = stats_with_optional_raw(a_full_samples, save_raw)
    b_full_stat = stats_with_optional_raw(b_full_samples, save_raw)
    cd_full_stat = stats_with_optional_raw(cd_full_samples, save_raw)
    fixed_stat = stats_with_optional_raw(
        fixed_from_volume_samples,
        save_raw,
    )
    hc_stat = stats_with_optional_raw(hc_samples, save_raw)
    after_hc_stat = stats_with_optional_raw(
        after_hc_downstream_samples,
        save_raw,
    )

    if adaptive_fixed_samples:
        adaptive_fixed_stat = stats_with_optional_raw(
            adaptive_fixed_samples,
            save_raw,
        )
        adaptive_fixed_included = True
    else:
        adaptive_fixed_stat = None
        adaptive_fixed_included = False

    # ------------------------------------------------------------------------
    # Diagnostic component reconstruction of full baseline.
    # This is intentionally not used as a correctness proof.
    # ------------------------------------------------------------------------
    component_full_mean = (
        a_full_stat["mean_ms"]
        + b_full_stat["mean_ms"]
        + cd_full_stat["mean_ms"]
        + fixed_stat["mean_ms"]
    )

    component_full_p99_sum = (
        a_full_stat["p99_ms"]
        + b_full_stat["p99_ms"]
        + cd_full_stat["p99_ms"]
        + fixed_stat["p99_ms"]
    )

    closure_gap_mean_ms = (
        full_forward_stat["mean_ms"]
        - component_full_mean
    )
    closure_ratio = (
        component_full_mean / full_forward_stat["mean_ms"]
        if full_forward_stat["mean_ms"] > 0
        else float("nan")
    )

    component_payload = {
        "schema_version": 1,
        "profile_type": "physical_components",
        "latency_semantics": {
            "timing": "forward-only wall time with CUDA synchronize before/after",
            "excluded": [
                "dataset IO",
                "CPU preprocessing",
                "load_data_to_gpu",
                "CPU->GPU transfer",
            ],
            "included": "Python launch overhead and CUDA execution inside each profiled component call",
        },
        "config": {
            "cfg_file": args.cfg_file,
            "base_ckpt": args.ckpt,
            "adaptive_ckpt": args.adaptive_ckpt,
            "levels": levels,
            "positions": positions,
            "sample_indices": sample_indices,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "b_halo": args.b_halo,
            "cd_halo": args.cd_halo,
            "align": args.align,
            "numeric_tol": args.numeric_tol,
            "strict_numeric": args.strict_numeric,
        },
        "frames": frame_records,

        "reference": {
            "original_full_forward": full_forward_stat,
            "component_full_mean_sum_ms": component_full_mean,
            "component_full_p99_sum_ms": component_full_p99_sum,
            "component_vs_full_mean_ratio": closure_ratio,
            "full_minus_component_mean_ms": closure_gap_mean_ms,
            "note": (
                "Component sums are diagnostic only. Separately profiled CUDA "
                "components are not guaranteed to add exactly to model.forward latency. "
                "The gap also contains model/backbone glue that is not isolated here."
            ),
        },

        "A": {
            "definition": (
                "A level timing = left+right A_s (stem+layer1) + selective A_d "
                "(layer2-4) + current physical feature-neck/cache path."
            ),
            "a_s_diagnostic": a_s_stat,
            "original_full_A": a_full_stat,
            "levels": a_records,
        },

        "B": {
            "definition": (
                "DPS BuildCostVolume.forward_roi + dres0+dres1 residual + "
                "semantic-core scatter to dense B canvas."
            ),
            "halo": args.b_halo,
            "original_full_B": b_full_stat,
            "levels": b_records,
        },

        "CD": {
            "definition": (
                "C mapping+grid_sample+valid mask on expanded ROI, then physical "
                "D rpn3d_convs/hgs/pool and semantic-core scatter."
            ),
            "halo": args.cd_halo,
            "original_full_CD": cd_full_stat,
            "levels": cd_records,
        },

        "fixed_downstream": {
            "definition": (
                "Full-size D output -> HeightCompression -> FeatureAlignment -> "
                "AFTER_FUSION blocks (VAN/StreamDetHead) -> post_processing."
            ),
            "aggregate": fixed_stat,
            "breakdown_diagnostic": {
                "HeightCompression": hc_stat,
                "after_HeightCompression_through_post_processing": after_hc_stat,
            },
        },

        "adaptive_fixed": {
            "included_in_provisional_lut": adaptive_fixed_included,
            "meta": adaptive_meta,
            "aggregate": adaptive_fixed_stat,
        },

        "limitations": [
            "B and CD are not yet one unified adaptive detector forward.",
            "Independent A/B/CD component timings do not capture joint CUDA interaction.",
            "Independent B timing does not include extra B dependency completion demanded by an arbitrary downstream CD ROI.",
            "Hard ROI scoring / argmax and the 125-action scheduler are not included unless they become part of the unified runtime.",
            "Age-map updates, true-BEV/emit state writes, and StreamDSGN history-cache writes are not fully represented in the independent component sum.",
            "Backbone/model glue such as calibration tensor setup and disparity-channel preparation is not isolated as a component.",
            "Each component is synchronized independently, so summing component P99 values is deliberately conservative and is not the P99 of a real joint forward.",
            "Therefore the 125-action additive LUT generated from this file is provisional and must not be used as the final paper latency claim.",
        ],
    }

    components_path = out_dir / "adaptive_latency_components.json"
    save_json(components_path, component_payload)

    # ------------------------------------------------------------------------
    # Build provisional 125-action LUT.
    # ------------------------------------------------------------------------
    adaptive_p99 = (
        adaptive_fixed_stat["p99_ms"]
        if adaptive_fixed_stat is not None
        else 0.0
    )
    adaptive_mean = (
        adaptive_fixed_stat["mean_ms"]
        if adaptive_fixed_stat is not None
        else 0.0
    )

    actions = {}
    csv_rows = []

    for a in range(5):
        for b in range(5):
            for cd in range(5):
                A = a_records[a]
                B = b_records[b]
                C = cd_records[cd]

                mean_est = (
                    A["worst_mean_ms"]
                    + B["worst_mean_ms"]
                    + C["worst_mean_ms"]
                    + fixed_stat["mean_ms"]
                    + adaptive_mean
                )

                p99_sum = (
                    A["worst_p99_ms"]
                    + B["worst_p99_ms"]
                    + C["worst_p99_ms"]
                    + fixed_stat["p99_ms"]
                    + adaptive_p99
                )

                safe_ms = (
                    p99_sum * (1.0 + args.margin_ratio)
                    + args.margin_ms
                )

                key = f"{a},{b},{cd}"

                rec = {
                    "a_level": a,
                    "b_level": b,
                    "cd_level": cd,
                    "a_ratio": levels[a],
                    "b_ratio": levels[b],
                    "cd_ratio": levels[cd],

                    "estimated_mean_ms": mean_est,
                    "sum_component_p99_ms": p99_sum,
                    "safe_ms": safe_ms,

                    "component_p99_ms": {
                        "A": A["worst_p99_ms"],
                        "B": B["worst_p99_ms"],
                        "CD": C["worst_p99_ms"],
                        "fixed_downstream": fixed_stat["p99_ms"],
                        "adaptive_fixed": adaptive_p99,
                    },

                    "worst_positions": {
                        "A": A["worst_position"],
                        "B": B["worst_position"],
                        "CD": C["worst_position"],
                    },
                }

                actions[key] = rec

                csv_rows.append(
                    {
                        "action": key,
                        "a_level": a,
                        "b_level": b,
                        "cd_level": cd,
                        "a_ratio": levels[a],
                        "b_ratio": levels[b],
                        "cd_ratio": levels[cd],
                        "estimated_mean_ms": mean_est,
                        "sum_component_p99_ms": p99_sum,
                        "safe_ms": safe_ms,
                    }
                )

    lut_payload = {
        "schema_version": 1,
        "authoritative": False,
        "status": "PROVISIONAL_ADDITIVE_COMPONENT_LUT",
        "reason": (
            "Current repo does not yet execute A/B/CD + learned control path as "
            "one unified physical adaptive detector forward. Final scheduler LUT "
            "must be re-profiled from actual 125 joint forwards."
        ),
        "levels": {
            "A": levels,
            "B": levels,
            "CD": levels,
        },
        "latency_semantics": "forward_only",
        "provisional_formula": (
            "T_hat = T_A(levelA) + T_B(levelB) + T_CD(levelCD) + "
            "T_fixed_downstream + T_adaptive_fixed"
        ),
        "safe_formula": (
            f"safe_ms = sum_component_p99_ms * (1 + {args.margin_ratio}) "
            f"+ {args.margin_ms}"
        ),
        "adaptive_fixed_included": adaptive_fixed_included,
        "source_components": str(components_path),
        "actions": actions,
        "do_not_use_as_final_paper_lut": True,
    }

    lut_path = out_dir / "adaptive_latency_lut_provisional.json"
    save_json(lut_path, lut_payload)

    csv_path = out_dir / "adaptive_latency_lut_provisional.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(csv_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    # ------------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------------
    print()
    print("=" * 116)
    print("PROFILE SUMMARY -- FORWARD ONLY")
    print("=" * 116)
    print(
        f"Original full forward : mean={full_forward_stat['mean_ms']:.3f} "
        f"p50={full_forward_stat['p50_ms']:.3f} "
        f"p99={full_forward_stat['p99_ms']:.3f} ms"
    )
    print(
        f"A_s diagnostic        : mean={a_s_stat['mean_ms']:.3f} "
        f"p99={a_s_stat['p99_ms']:.3f} ms"
    )
    print(
        f"Full A/B/CD           : "
        f"{a_full_stat['mean_ms']:.3f} / "
        f"{b_full_stat['mean_ms']:.3f} / "
        f"{cd_full_stat['mean_ms']:.3f} ms"
    )
    print(
        f"Fixed downstream      : mean={fixed_stat['mean_ms']:.3f} "
        f"p99={fixed_stat['p99_ms']:.3f} ms"
    )
    print(
        f"Component closure     : sum_mean={component_full_mean:.3f} ms, "
        f"full_mean={full_forward_stat['mean_ms']:.3f} ms, "
        f"ratio={closure_ratio:.3f}, gap={closure_gap_mean_ms:+.3f} ms"
    )

    if adaptive_fixed_stat is not None:
        print(
            f"Adaptive fixed        : mean={adaptive_fixed_stat['mean_ms']:.3f} "
            f"p99={adaptive_fixed_stat['p99_ms']:.3f} ms "
            f"(weights_loaded={adaptive_meta.get('weights_loaded', False)})"
        )
    else:
        print("Adaptive fixed        : OMITTED")

    print()
    print("Worst-position P99 by level (ms)")
    print(f"{'L':>2} {'ratio':>7} {'A':>10} {'B':>10} {'CD':>10}")
    print("-" * 46)
    for i, ratio in enumerate(levels):
        print(
            f"{i:2d} {ratio:7.2f} "
            f"{a_records[i]['worst_p99_ms']:10.3f} "
            f"{b_records[i]['worst_p99_ms']:10.3f} "
            f"{cd_records[i]['worst_p99_ms']:10.3f}"
        )

    print()
    print("Saved:")
    print("  components :", components_path)
    print("  provisional:", lut_path)
    print("  csv        :", csv_path)
    print()
    print("IMPORTANT:")
    print("  adaptive_latency_lut_provisional.json is NOT the final scheduler/paper LUT.")
    print("  Re-profile all 125 actions after B/CD/control are unified in one real forward.")
    print("=" * 116)


if __name__ == "__main__":
    main()