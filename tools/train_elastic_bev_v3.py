#!/usr/bin/env python3

import argparse
import copy
import itertools
import json
import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v3 import (
    ElasticBEVBranch,
    MONOTONIC_SCHEDULES,
    WIDTH_CHOICES,
    cached_full_2d_prefix,
    extract_fixed_layer1_prefix,
    extract_full_2d_from_layer1,
)


torch.backends.cudnn.benchmark = True

HISTORY_OFFSETS = (1, 2, 3, 4, 5)
CANONICAL_HISTORY = (3, 2, 1)  # oldest -> newest
ALL_HISTORY_TRIPLETS = tuple(
    tuple(sorted(x, reverse=True))
    for x in itertools.combinations(HISTORY_OFFSETS, 3)
)
NON_CANONICAL_TRIPLETS = tuple(
    x for x in ALL_HISTORY_TRIPLETS if x != CANONICAL_HISTORY
)
ANCHOR_SCHEDULES = tuple((r,) * 6 for r in WIDTH_CHOICES)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Elastic-v3 profile-space robust training for StreamDSGN K3. "
            "The Full K3 base stays frozen; only the elastic generator and "
            "its tiny per-width normalization/projection parameters are trained."
        )
    )
    parser.add_argument("--full_cfg", required=True)
    parser.add_argument("--full_ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--exp_name", type=str, default="elastic_bev_v3")
    parser.add_argument("--output_root", type=str, default="outputs/elastic_bev")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--init_elastic_ckpt",
        type=str,
        default=None,
        help=(
            "Optional v2 checkpoint. Only state entries whose names and shapes "
            "still match v3 are copied; old BN and shared BEV projection states "
            "are intentionally skipped."
        ),
    )

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--det_weight", type=float, default=1.0)
    parser.add_argument("--bev_weight", type=float, default=1.5)
    parser.add_argument("--cos_weight", type=float, default=0.15)
    parser.add_argument("--stage_weight", type=float, default=0.50)
    parser.add_argument("--stage_cos_ratio", type=float, default=0.10)
    parser.add_argument("--history_bev_weight", type=float, default=0.75)
    parser.add_argument("--history_cos_weight", type=float, default=0.10)
    parser.add_argument("--smooth_l1_beta", type=float, default=0.1)

    # Profile-space robust current-frame sampling.
    parser.add_argument("--uniform_prob", type=float, default=0.40)
    parser.add_argument("--transition_prob", type=float, default=0.30)
    parser.add_argument("--hard_prob", type=float, default=0.20)
    parser.add_argument("--anchor_prob", type=float, default=0.10)
    parser.add_argument("--hard_ema_beta", type=float, default=0.95)
    parser.add_argument("--hard_gamma", type=float, default=1.0)
    parser.add_argument("--hard_explore", type=float, default=0.25)

    # Temporal sampling.
    parser.add_argument(
        "--canonical_history_prob",
        type=float,
        default=0.60,
        help=(
            "Use [t-3,t-2,t-1] with this probability. Otherwise choose three "
            "distinct frames from t-5...t-1. The newest selected history is not "
            "forced to be t-1."
        ),
    )
    parser.add_argument("--history_same_profile_prob", type=float, default=0.40)
    parser.add_argument("--history_independent_profile_prob", type=float, default=0.40)
    parser.add_argument("--history_jump_profile_prob", type=float, default=0.20)
    parser.add_argument(
        "--history_grad_slots",
        type=int,
        default=1,
        help=(
            "Number of selected history frames that keep an elastic graph. "
            "1 is recommended on a single 4090: all history BEVs are elastic, "
            "but only one random history slot plus the current frame receives "
            "temporal/detection gradients per iteration."
        ),
    )

    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=1)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def frame_sort_key(token):
    token = str(token)
    try:
        return (0, int(token))
    except ValueError:
        return (1, token)


def inject_history_offsets(train_set, logger, max_offset=5):
    by_scene = {}
    for index, info in enumerate(train_set.kitti_infos):
        scene = str(info["sample_idx"]["scene"])
        token = str(info["sample_idx"]["frame_tag"]["token"])
        by_scene.setdefault(scene, []).append((index, token))

    counts = {offset: 0 for offset in range(3, max_offset + 1)}
    for _, items in by_scene.items():
        items.sort(key=lambda x: frame_sort_key(x[1]))
        for pos, (index, _) in enumerate(items):
            info = train_set.kitti_infos[index]
            frame_tag = info["sample_idx"]["frame_tag"]
            for offset in range(3, max_offset + 1):
                key = f"prev{offset}"
                if pos < offset:
                    frame_tag[key] = ""
                    info["infos"].pop(key, None)
                    continue
                prev_index, prev_token = items[pos - offset]
                prev_info = train_set.kitti_infos[prev_index]
                frame_tag[key] = str(prev_token)
                info["infos"][key] = copy.deepcopy(prev_info["infos"]["token"])
                counts[offset] += 1

    if getattr(train_set, "data_augmentor", None) is not None:
        for augmentor in train_set.data_augmentor.data_augmentor_queue:
            if hasattr(augmentor, "all_sample_tag"):
                for offset in range(3, max_offset + 1):
                    augmentor.all_sample_tag[f"prev{offset}"] = -offset

    logger.info(
        "Injected extra history: "
        + ", ".join(f"t-{k}:{v}" for k, v in counts.items())
    )


def choose_history_triplet(available_offsets, canonical_probability):
    available_offsets = tuple(sorted(set(available_offsets)))
    if len(available_offsets) < 3:
        return tuple(sorted(available_offsets, reverse=True))

    available_triples = [
        tuple(sorted(x, reverse=True))
        for x in itertools.combinations(available_offsets, 3)
    ]
    if (
        CANONICAL_HISTORY in available_triples
        and random.random() < float(canonical_probability)
    ):
        return CANONICAL_HISTORY

    alternatives = [x for x in available_triples if x != CANONICAL_HISTORY]
    return random.choice(alternatives) if alternatives else CANONICAL_HISTORY


def remap_random_history(batch_dict, canonical_probability):
    candidates = {
        1: batch_dict.get("prev"),
        2: batch_dict.get("prev2"),
        3: batch_dict.get("prev3"),
        4: batch_dict.get("prev4"),
        5: batch_dict.get("prev5"),
    }
    available = [k for k, v in candidates.items() if v is not None]
    selected = choose_history_triplet(available, canonical_probability)

    for key in ("prev3", "prev2", "prev"):
        batch_dict[key] = None

    # K3 queue order remains oldest -> newest, but the newest selected history
    # is allowed to be t-2/t-3/etc.
    target_keys = ("prev3", "prev2", "prev")[-len(selected):]
    for target_key, offset in zip(target_keys, selected):
        batch_dict[target_key] = candidates[offset]
    return tuple(selected)


def make_cfg(path):
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = Path(path).stem
    cfg.DATA_CONFIG.INFER_TIME_PATH = None
    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None
    return cfg


def snapshot_base_model(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def verify_base_unchanged(model, snapshot):
    changed = []
    current = model.state_dict()
    for name, reference in snapshot.items():
        now = current[name].detach().cpu()
        if reference.dtype.is_floating_point:
            diff = float((now.float() - reference.float()).abs().max().item())
            if diff != 0.0:
                changed.append((name, diff))
        elif not torch.equal(now, reference):
            changed.append((name, 1.0))
    if changed:
        detail = "\n".join(f"{n}: {d:.3e}" for n, d in changed[:20])
        raise RuntimeError(
            "Frozen K3 base changed during Elastic-v3 training:\n" + detail
        )


class ProfileRobustSampler:
    """
    Current-frame profile sampler:
      40% uniform over all 84 legal profiles,
      30% transition-balanced,
      20% hard-profile mining,
      10% diagonal anchors.

    The percentages are configurable. Transition balancing operates on all
    five adjacent stage boundaries and all legal predecessor->successor width
    pairs. Hard mining uses an EMA loss plus an inverse-count exploration term.
    """

    def __init__(
        self,
        uniform_prob,
        transition_prob,
        hard_prob,
        anchor_prob,
        ema_beta,
        hard_gamma,
        hard_explore,
    ):
        self.schedules = tuple(MONOTONIC_SCHEDULES)
        self.index = {s: i for i, s in enumerate(self.schedules)}
        self.counts = [0 for _ in self.schedules]
        self.ema = [1.0 for _ in self.schedules]
        self.ema_initialized = [False for _ in self.schedules]
        self.ema_beta = float(ema_beta)
        self.hard_gamma = float(hard_gamma)
        self.hard_explore = float(hard_explore)

        probs = [
            float(uniform_prob),
            float(transition_prob),
            float(hard_prob),
            float(anchor_prob),
        ]
        if any(p < 0.0 for p in probs) or sum(probs) <= 0.0:
            raise ValueError(f"Invalid sampler probabilities: {probs}")
        total = sum(probs)
        self.mode_probs = tuple(p / total for p in probs)
        self.mode_names = ("uniform", "transition", "hard", "anchor")

        self.transition_candidates = {}
        self.transition_counts = {}
        for boundary in range(5):
            for a in WIDTH_CHOICES:
                for b in WIDTH_CHOICES:
                    if a < b:
                        continue
                    key = (boundary, float(a), float(b))
                    candidates = [
                        s for s in self.schedules
                        if s[boundary] == float(a)
                        and s[boundary + 1] == float(b)
                    ]
                    if candidates:
                        self.transition_candidates[key] = candidates
                        self.transition_counts[key] = 0

    def _weighted_choice(self, items, weights):
        total = float(sum(weights))
        if total <= 0.0:
            return random.choice(items)
        threshold = random.random() * total
        acc = 0.0
        for item, weight in zip(items, weights):
            acc += float(weight)
            if acc >= threshold:
                return item
        return items[-1]

    def _sample_transition(self):
        min_count = min(self.transition_counts.values())
        # Choose among the least-observed transition states, not always the
        # same deterministic one.
        underused = [
            key for key, count in self.transition_counts.items()
            if count <= min_count + 1
        ]
        key = random.choice(underused)
        candidates = self.transition_candidates[key]
        # Within that transition, prefer globally under-trained profiles.
        weights = [
            1.0 / math.sqrt(self.counts[self.index[s]] + 1.0)
            for s in candidates
        ]
        return self._weighted_choice(candidates, weights)

    def _hard_score(self, schedule):
        i = self.index[schedule]
        ema = max(float(self.ema[i]), 1e-6)
        return (
            ema ** self.hard_gamma
            + self.hard_explore / math.sqrt(self.counts[i] + 1.0)
        )

    def _sample_hard(self):
        scores = [self._hard_score(s) for s in self.schedules]
        # Sample from the hardest quarter to avoid a single noisy profile
        # monopolizing training.
        ranked = sorted(
            zip(self.schedules, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        top_k = max(8, len(ranked) // 4)
        pool = ranked[:top_k]
        return self._weighted_choice(
            [x[0] for x in pool],
            [x[1] for x in pool],
        )

    def sample(self):
        mode = random.choices(
            self.mode_names,
            weights=self.mode_probs,
            k=1,
        )[0]
        if mode == "uniform":
            schedule = random.choice(self.schedules)
        elif mode == "transition":
            schedule = self._sample_transition()
        elif mode == "hard":
            schedule = self._sample_hard()
        elif mode == "anchor":
            schedule = random.choice(ANCHOR_SCHEDULES)
        else:
            raise RuntimeError(mode)

        i = self.index[schedule]
        self.counts[i] += 1
        for boundary in range(5):
            key = (
                boundary,
                float(schedule[boundary]),
                float(schedule[boundary + 1]),
            )
            self.transition_counts[key] += 1
        return schedule, mode

    def update(self, schedule, loss_signal):
        i = self.index[tuple(schedule)]
        value = float(loss_signal)
        if not self.ema_initialized[i]:
            self.ema[i] = value
            self.ema_initialized[i] = True
        else:
            beta = self.ema_beta
            self.ema[i] = beta * self.ema[i] + (1.0 - beta) * value

    def hardest(self, n=5):
        ranked = sorted(
            self.schedules,
            key=lambda s: self._hard_score(s),
            reverse=True,
        )
        return [
            (s, self.ema[self.index[s]], self.counts[self.index[s]])
            for s in ranked[:n]
        ]

    def state_dict(self):
        transition_keys = sorted(self.transition_counts.keys())
        return {
            "counts": list(self.counts),
            "ema": list(self.ema),
            "ema_initialized": list(self.ema_initialized),
            "transition_keys": transition_keys,
            "transition_counts": [
                self.transition_counts[k] for k in transition_keys
            ],
        }

    def load_state_dict(self, state):
        if not state:
            return
        if len(state.get("counts", [])) == len(self.counts):
            self.counts = list(state["counts"])
        if len(state.get("ema", [])) == len(self.ema):
            self.ema = list(state["ema"])
        if len(state.get("ema_initialized", [])) == len(self.ema_initialized):
            self.ema_initialized = list(state["ema_initialized"])
        keys = state.get("transition_keys", [])
        values = state.get("transition_counts", [])
        for key, value in zip(keys, values):
            key = tuple(key)
            if key in self.transition_counts:
                self.transition_counts[key] = int(value)


def profile_distance(a, b):
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b))


def sample_history_profiles(
    current_schedule,
    count,
    same_prob,
    independent_prob,
    jump_prob,
):
    if count <= 0:
        return [], "none"

    probs = [float(same_prob), float(independent_prob), float(jump_prob)]
    total = sum(probs)
    if total <= 0.0:
        raise ValueError("history profile probabilities sum to zero")
    probs = [p / total for p in probs]
    mode = random.choices(
        ("same", "independent", "jump"),
        weights=probs,
        k=1,
    )[0]

    if mode == "same":
        schedules = [tuple(current_schedule) for _ in range(count)]
    elif mode == "independent":
        schedules = [
            random.choice(MONOTONIC_SCHEDULES)
            for _ in range(count)
        ]
    else:
        ranked = sorted(
            MONOTONIC_SCHEDULES,
            key=lambda s: profile_distance(s, current_schedule),
            reverse=True,
        )
        pool = ranked[: max(12, len(ranked) // 3)]
        schedules = [random.choice(pool) for _ in range(count)]
    return schedules, mode


def full_teacher_from_prefix(base_model, frame_dict, prefix_cache):
    """
    Full teacher BEV plus six-stage teacher features. Hooks recover the Full
    stereo feature before geometry mapping without modifying StreamDSGN.
    """
    backbone = base_model.backbone_3d
    full_2d = extract_full_2d_from_layer1(
        backbone, frame_dict, prefix_cache
    )

    hook_cache = {}

    def save_dres0(_module, _inputs, output):
        hook_cache["dres0"] = output.detach()

    def save_dres1(_module, _inputs, output):
        hook_cache["dres1"] = output.detach()

    h0 = backbone.dres0.register_forward_hook(save_dres0)
    h1 = backbone.dres1.register_forward_hook(save_dres1)
    try:
        teacher_data = copy.copy(frame_dict)
        with cached_full_2d_prefix(backbone, full_2d):
            teacher_data = backbone(teacher_data)
        teacher_data = base_model.map_to_bev_module(teacher_data)
    finally:
        h0.remove()
        h1.remove()

    if "dres0" not in hook_cache or "dres1" not in hook_cache:
        raise RuntimeError("Failed to capture Full stereo teacher features")

    left_backbone = full_2d["left_backbone"]
    right_backbone = full_2d["right_backbone"]
    targets = {
        "bev": teacher_data["spatial_features"].detach(),
        "res2_left": left_backbone[1].detach(),
        "res2_right": right_backbone[1].detach(),
        "res3_left": left_backbone[2].detach(),
        "res3_right": right_backbone[2].detach(),
        "res4_left": left_backbone[3].detach(),
        "res4_right": right_backbone[3].detach(),
        "fpn_left": full_2d["left_stereo"].detach(),
        "fpn_right": full_2d["right_stereo"].detach(),
        "stereo": (hook_cache["dres0"] + hook_cache["dres1"]).detach(),
        "rpn_pool": teacher_data["volume_features"].detach(),
    }
    return targets


def full_bev_from_prefix(base_model, frame_dict, prefix_cache):
    backbone = base_model.backbone_3d
    full_2d = extract_full_2d_from_layer1(
        backbone, frame_dict, prefix_cache
    )
    teacher_data = copy.copy(frame_dict)
    with cached_full_2d_prefix(backbone, full_2d):
        teacher_data = backbone(teacher_data)
    teacher_data = base_model.map_to_bev_module(teacher_data)
    return teacher_data["spatial_features"].detach()


def generic_distill(student, teacher, beta, cosine_weight=1.0):
    if student.ndim != teacher.ndim:
        raise RuntimeError(
            f"Distill rank mismatch: {student.shape} vs {teacher.shape}"
        )
    if tuple(student.shape[2:]) != tuple(teacher.shape[2:]):
        raise RuntimeError(
            f"Distill spatial mismatch: {student.shape} vs {teacher.shape}"
        )
    c = int(student.shape[1])
    if c > int(teacher.shape[1]):
        raise RuntimeError(
            f"Student channels {c} exceed teacher {teacher.shape[1]}"
        )
    teacher_prefix = teacher[:, :c]

    smooth = F.smooth_l1_loss(
        student.float(),
        teacher_prefix.float(),
        beta=float(beta),
        reduction="mean",
    )
    cosine = 1.0 - F.cosine_similarity(
        student.float(),
        teacher_prefix.float(),
        dim=1,
        eps=1e-6,
    ).mean()
    return smooth + float(cosine_weight) * cosine, smooth, cosine


def bev_distill_loss(student_bev, teacher_bev, beta):
    smooth = F.smooth_l1_loss(
        student_bev.float(),
        teacher_bev.float(),
        beta=float(beta),
        reduction="mean",
    )
    cosine = 1.0 - F.cosine_similarity(
        student_bev.float(),
        teacher_bev.float(),
        dim=1,
        eps=1e-6,
    ).mean()
    l1 = (student_bev.float() - teacher_bev.float()).abs().mean()
    return smooth, cosine, l1


def stage_distill_loss(
    student_features,
    teacher_features,
    beta,
    stage_cos_ratio,
):
    stage_names = (
        ("res2", ("res2_left", "res2_right")),
        ("res3", ("res3_left", "res3_right")),
        ("res4", ("res4_left", "res4_right")),
        ("fpn", ("fpn_left", "fpn_right")),
        ("stereo", ("stereo",)),
        ("rpn", ("rpn_pool",)),
    )
    losses = {}
    total = None
    for stage_name, keys in stage_names:
        stage_terms = []
        for key in keys:
            term, _, _ = generic_distill(
                student_features[key],
                teacher_features[key],
                beta,
                cosine_weight=stage_cos_ratio,
            )
            stage_terms.append(term)
        stage_loss = sum(stage_terms) / len(stage_terms)
        losses[stage_name] = stage_loss
        total = stage_loss if total is None else total + stage_loss
    return total / len(stage_names), losses


def run_frozen_k3_downstream(
    base_model,
    batch_dict,
    elastic_bev,
    valids,
    history_queue,
):
    cur_data = batch_dict["token"]
    cur_data["spatial_features"] = elastic_bev
    cur_data["spatial_features_stride"] = 1
    cur_data["valids"] = valids
    cur_data["history_features"] = history_queue

    for module in base_model.fusion_module:
        cur_data = module(cur_data)
    for module in base_model.after_fusion_blocks:
        cur_data = module(cur_data)

    batch_dict["token"] = cur_data
    return cur_data


def detection_loss_without_trend(base_model, batch_dict):
    # Random temporal gaps invalidate the equal-step velocity/trend loss.
    head = base_model.dense_head
    supervision = batch_dict[head.box3d_supervision]
    targets = head.assign_targets(
        gt_boxes=supervision["gt_boxes"],
        data_dict=supervision,
    )
    head.forward_ret_dict.update(targets)
    cls_loss, cls_tb = head.get_cls_layer_loss()
    box_loss, box_tb = head.get_box_reg_layer_loss()
    tb = {}
    tb.update(cls_tb)
    tb.update(box_tb)
    return cls_loss + box_loss, tb


def lr_for_step(step, total_steps, base_lr, min_lr, warmup_ratio):
    warmup_steps = max(100, int(total_steps * warmup_ratio))
    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(
        max(total_steps - warmup_steps, 1)
    )
    cosine = 0.5 * (
        1.0
        + math.cos(
            math.pi * min(max(progress, 0.0), 1.0)
        )
    )
    return min_lr + (base_lr - min_lr) * cosine


def partial_init_from_elastic(branch, checkpoint_path, logger):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint["branch"]
    target = branch.state_dict()
    matched = []
    skipped = []
    for name, value in source.items():
        if name in target and tuple(target[name].shape) == tuple(value.shape):
            target[name] = value
            matched.append(name)
        else:
            skipped.append(name)
    branch.load_state_dict(target, strict=True)
    logger.info(
        f"Partial Elastic init from {checkpoint_path}: "
        f"matched={len(matched)}, skipped={len(skipped)}"
    )
    return matched, skipped


def save_checkpoint(
    path,
    branch,
    optimizer,
    scaler,
    sampler,
    epoch,
    global_step,
    args,
):
    torch.save(
        {
            "version": "elastic_v3_profile_robust",
            "epoch": int(epoch),
            "global_step": int(global_step),
            "branch": branch.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "sampler": sampler.state_dict(),
            "args": vars(args),
            "width_choices": list(WIDTH_CHOICES),
            "num_elastic_stages": 6,
            "num_monotonic_schedules": len(MONOTONIC_SCHEDULES),
            "canonical_history": list(CANONICAL_HISTORY),
            "history_candidates": list(HISTORY_OFFSETS),
        },
        path,
    )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 <= args.canonical_history_prob <= 1.0:
        raise ValueError("canonical_history_prob must be in [0,1]")
    if args.history_grad_slots < 0 or args.history_grad_slots > 3:
        raise ValueError("history_grad_slots must be in [0,3]")

    set_seed(args.seed)
    full_cfg = make_cfg(args.full_cfg)

    output_dir = Path(args.output_root) / args.exp_name
    ckpt_dir = output_dir / "ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(output_dir / "log_train.txt")
    tb_log = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    logger.info("=" * 80)
    logger.info("Elastic-v3 profile-space robust training")
    logger.info("Frozen timing prefix: original ResNet stem + layer1")
    logger.info("Elastic stages: Res2, Res3, Res4, FPN, Stereo3D, RPN3D")
    logger.info("Normalization: per-width GroupNorm (no running statistics)")
    logger.info("BEV interface: independent tiny projection per output width")
    logger.info(
        "Current profile sampler: "
        f"uniform={args.uniform_prob}, transition={args.transition_prob}, "
        f"hard={args.hard_prob}, anchor={args.anchor_prob}"
    )
    logger.info(
        "History profile sampler: "
        f"same={args.history_same_profile_prob}, "
        f"independent={args.history_independent_profile_prob}, "
        f"jump={args.history_jump_profile_prob}"
    )
    logger.info(
        f"History time: {CANONICAL_HISTORY} prob={args.canonical_history_prob}; "
        "otherwise choose 3 distinct frames from t-5...t-1"
    )
    logger.info(f"legal schedules: {len(MONOTONIC_SCHEDULES)}")
    logger.info(f"epochs={args.epochs}, lr={args.lr}->{args.min_lr}")
    logger.info("=" * 80)

    train_set, train_loader, _ = build_dataloader(
        dataset_cfg=full_cfg.DATA_CONFIG,
        class_names=full_cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=False,
        total_epochs=args.epochs,
    )
    inject_history_offsets(train_set, logger, max_offset=5)

    base_model = build_network(
        model_cfg=full_cfg.MODEL,
        num_class=len(full_cfg.CLASS_NAMES),
        dataset=train_set,
    )
    base_model.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )
    base_model.cuda().eval()
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(
            full_cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES
        ),
    ).cuda()

    if args.init_elastic_ckpt is not None:
        partial_init_from_elastic(
            branch,
            args.init_elastic_ckpt,
            logger,
        )

    trainable = sum(
        p.numel() for p in branch.parameters() if p.requires_grad
    )
    logger.info(f"Elastic-v3 trainable parameters: {trainable:,}")
    logger.info("Base K3 trainable parameters: 0")

    optimizer = torch.optim.AdamW(
        branch.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(
        full_cfg.MODEL.USE_AMP.get("TRAIN", True)
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    sampler = ProfileRobustSampler(
        args.uniform_prob,
        args.transition_prob,
        args.hard_prob,
        args.anchor_prob,
        args.hard_ema_beta,
        args.hard_gamma,
        args.hard_explore,
    )

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        branch.load_state_dict(checkpoint["branch"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        sampler.load_state_dict(checkpoint.get("sampler"))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        logger.info(
            f"Resumed Elastic-v3 from {args.resume} at epoch={start_epoch}"
        )

    base_snapshot = snapshot_base_model(base_model)
    total_steps = args.epochs * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)
        branch.train()

        running = {
            "loss": 0.0,
            "det": 0.0,
            "bev": 0.0,
            "cos": 0.0,
            "stage": 0.0,
            "hist_bev": 0.0,
            "hist_cos": 0.0,
            "count": 0,
        }

        for iteration, batch_dict in enumerate(train_loader):
            load_data_to_gpu(batch_dict)
            selected_history = remap_random_history(
                batch_dict,
                args.canonical_history_prob,
            )

            optimizer.zero_grad(set_to_none=True)
            lr = lr_for_step(
                global_step,
                total_steps,
                args.lr,
                args.min_lr,
                args.warmup_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            current_schedule, sample_mode = sampler.sample()
            history_frames = [
                batch_dict[key]
                for key in ("prev3", "prev2", "prev")
                if batch_dict.get(key) is not None
            ]
            history_schedules, history_profile_mode = (
                sample_history_profiles(
                    current_schedule,
                    len(history_frames),
                    args.history_same_profile_prob,
                    args.history_independent_profile_prob,
                    args.history_jump_profile_prob,
                )
            )

            # Select history slots that retain graphs. With one slot, the
            # temporal gradient rotates across oldest/middle/newest histories.
            grad_count = min(
                int(args.history_grad_slots),
                len(history_frames),
            )
            grad_slots = set(
                random.sample(
                    range(len(history_frames)),
                    grad_count,
                )
            )

            # Frozen current timing prefix + Full teacher.
            with torch.no_grad(), torch.cuda.amp.autocast(
                enabled=amp_enabled
            ):
                current_prefix = extract_fixed_layer1_prefix(
                    base_model.backbone_3d,
                    batch_dict["token"],
                )
                current_teacher = full_teacher_from_prefix(
                    base_model,
                    batch_dict["token"],
                    current_prefix,
                )

            # Build an all-elastic history queue. No history frame is replaced
            # by Full at runtime. Only selected grad slots retain a graph.
            history_queue = deque(maxlen=3)
            hist_bev_terms = []
            hist_cos_terms = []

            for hist_index, (
                frame,
                hist_schedule,
            ) in enumerate(
                zip(history_frames, history_schedules)
            ):
                with torch.no_grad(), torch.cuda.amp.autocast(
                    enabled=amp_enabled
                ):
                    hist_prefix = extract_fixed_layer1_prefix(
                        base_model.backbone_3d,
                        frame,
                    )

                if hist_index in grad_slots:
                    with torch.no_grad(), torch.cuda.amp.autocast(
                        enabled=amp_enabled
                    ):
                        hist_teacher_bev = full_bev_from_prefix(
                            base_model,
                            frame,
                            hist_prefix,
                        )
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        hist_bev, _ = branch(
                            frame,
                            hist_prefix,
                            base_model.backbone_3d,
                            hist_schedule,
                        )
                        h_bev, h_cos, _ = bev_distill_loss(
                            hist_bev,
                            hist_teacher_bev,
                            args.smooth_l1_beta,
                        )
                    hist_bev_terms.append(h_bev)
                    hist_cos_terms.append(h_cos)
                    queue_bev = hist_bev
                else:
                    with torch.no_grad(), torch.cuda.amp.autocast(
                        enabled=amp_enabled
                    ):
                        queue_bev, _ = branch(
                            frame,
                            hist_prefix,
                            base_model.backbone_3d,
                            hist_schedule,
                        )
                    queue_bev = queue_bev.detach()

                history_queue.append(
                    (
                        frame["this_sample_idx"],
                        {"spatial_features": queue_bev},
                    )
                )

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                elastic_bev, valids, student_features = (
                    branch.forward_with_features(
                        batch_dict["token"],
                        current_prefix,
                        base_model.backbone_3d,
                        current_schedule,
                    )
                )

                run_frozen_k3_downstream(
                    base_model,
                    batch_dict,
                    elastic_bev,
                    valids,
                    history_queue,
                )
                det_loss, _ = detection_loss_without_trend(
                    base_model,
                    batch_dict,
                )

                bev_loss, cos_loss, bev_l1 = bev_distill_loss(
                    elastic_bev,
                    current_teacher["bev"],
                    args.smooth_l1_beta,
                )
                stage_loss, stage_parts = stage_distill_loss(
                    student_features,
                    current_teacher,
                    args.smooth_l1_beta,
                    args.stage_cos_ratio,
                )

                if hist_bev_terms:
                    history_bev_loss = (
                        sum(hist_bev_terms) / len(hist_bev_terms)
                    )
                    history_cos_loss = (
                        sum(hist_cos_terms) / len(hist_cos_terms)
                    )
                else:
                    history_bev_loss = elastic_bev.new_zeros(())
                    history_cos_loss = elastic_bev.new_zeros(())

                ramp_steps = max(len(train_loader), 1)
                det_ramp = 0.35 + 0.65 * min(
                    1.0,
                    float(global_step + 1) / float(ramp_steps),
                )

                total_loss = (
                    args.det_weight * det_ramp * det_loss
                    + args.bev_weight * bev_loss
                    + args.cos_weight * cos_loss
                    + args.stage_weight * stage_loss
                    + args.history_bev_weight * history_bev_loss
                    + args.history_cos_weight * history_cos_loss
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                branch.parameters(),
                args.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()

            # Hard-profile signal is current-profile centered. History-profile
            # noise is intentionally not included.
            hard_signal = (
                float(det_loss.detach().item())
                + 1.5 * float(bev_loss.detach().item())
                + 0.5 * float(stage_loss.detach().item())
            )
            sampler.update(current_schedule, hard_signal)

            running["loss"] += float(total_loss.detach().item())
            running["det"] += float(det_loss.detach().item())
            running["bev"] += float(bev_loss.detach().item())
            running["cos"] += float(cos_loss.detach().item())
            running["stage"] += float(stage_loss.detach().item())
            running["hist_bev"] += float(
                history_bev_loss.detach().item()
            )
            running["hist_cos"] += float(
                history_cos_loss.detach().item()
            )
            running["count"] += 1

            tb_log.add_scalar(
                "train/loss",
                total_loss.detach().item(),
                global_step,
            )
            tb_log.add_scalar(
                "train/det",
                det_loss.detach().item(),
                global_step,
            )
            tb_log.add_scalar(
                "train/bev_smooth_l1",
                bev_loss.detach().item(),
                global_step,
            )
            tb_log.add_scalar(
                "train/bev_cos",
                cos_loss.detach().item(),
                global_step,
            )
            tb_log.add_scalar(
                "train/bev_l1",
                bev_l1.detach().item(),
                global_step,
            )
            tb_log.add_scalar(
                "train/stage",
                stage_loss.detach().item(),
                global_step,
            )
            tb_log.add_scalar(
                "train/history_bev",
                history_bev_loss.detach().item(),
                global_step,
            )
            tb_log.add_scalar(
                "train/history_cos",
                history_cos_loss.detach().item(),
                global_step,
            )
            tb_log.add_scalar("train/lr", lr, global_step)

            for stage_name, value in stage_parts.items():
                tb_log.add_scalar(
                    f"train/stage_{stage_name}",
                    value.detach().item(),
                    global_step,
                )
            for idx, ratio in enumerate(
                current_schedule,
                start=1,
            ):
                tb_log.add_scalar(
                    f"train/width_stage_{idx}",
                    ratio,
                    global_step,
                )
            tb_log.add_scalar(
                "train/history_is_canonical",
                float(selected_history == CANONICAL_HISTORY),
                global_step,
            )

            if (iteration + 1) % args.log_interval == 0:
                c = max(running["count"], 1)
                hardest = sampler.hardest(1)[0]
                logger.info(
                    f"epoch={epoch + 1:02d}/{args.epochs:02d} "
                    f"iter={iteration + 1:05d}/{len(train_loader):05d} "
                    f"lr={lr:.3e} "
                    f"sample={sample_mode} "
                    f"hist_time={selected_history} "
                    f"hist_profile={history_profile_mode} "
                    f"width={current_schedule} "
                    f"loss={running['loss']/c:.4f} "
                    f"det={running['det']/c:.4f} "
                    f"bev={running['bev']/c:.4f} "
                    f"stage={running['stage']/c:.4f} "
                    f"hbev={running['hist_bev']/c:.4f} "
                    f"hardest={hardest[0]} "
                    f"ema={hardest[1]:.3f} n={hardest[2]}"
                )
                running = {
                    "loss": 0.0,
                    "det": 0.0,
                    "bev": 0.0,
                    "cos": 0.0,
                    "stage": 0.0,
                    "hist_bev": 0.0,
                    "hist_cos": 0.0,
                    "count": 0,
                }

            global_step += 1

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = (
                ckpt_dir
                / f"checkpoint_epoch_{epoch + 1}.pth"
            )
            save_checkpoint(
                ckpt_path,
                branch,
                optimizer,
                scaler,
                sampler,
                epoch,
                global_step,
                args,
            )
            logger.info(f"Saved {ckpt_path}")

        verify_base_unchanged(base_model, base_snapshot)
        logger.info(
            "PASS: frozen K3 base parameters and buffers are bitwise unchanged"
        )
        logger.info(
            "Hard-profile snapshot: "
            + "; ".join(
                f"{s} ema={ema:.3f} n={count}"
                for s, ema, count in sampler.hardest(5)
            )
        )

    with open(
        output_dir / "training_design.json",
        "w",
    ) as f:
        json.dump(
            {
                "version": "elastic_v3_profile_robust",
                "fixed_prefix": "original ResNet stem + layer1",
                "elastic_stages": [
                    "ResNet-layer2",
                    "ResNet-layer3",
                    "ResNet-layer4",
                    "FPN",
                    "Stereo3D",
                    "RPN3D-to-BEV",
                ],
                "normalization": (
                    "per-width GroupNorm without running statistics"
                ),
                "bev_projection": (
                    "independent 1x1 projection per output width"
                ),
                "width_choices": list(WIDTH_CHOICES),
                "num_monotonic_schedules": len(
                    MONOTONIC_SCHEDULES
                ),
                "profile_sampler": {
                    "uniform": args.uniform_prob,
                    "transition_balanced": args.transition_prob,
                    "hard_profile": args.hard_prob,
                    "anchor": args.anchor_prob,
                },
                "history_offsets": list(HISTORY_OFFSETS),
                "canonical_history": list(CANONICAL_HISTORY),
                "canonical_probability": (
                    args.canonical_history_prob
                ),
                "history_profile_sampler": {
                    "same": args.history_same_profile_prob,
                    "independent": (
                        args.history_independent_profile_prob
                    ),
                    "jump": args.history_jump_profile_prob,
                    "history_grad_slots": (
                        args.history_grad_slots
                    ),
                },
                "loss": {
                    "det_weight": args.det_weight,
                    "bev_weight": args.bev_weight,
                    "cos_weight": args.cos_weight,
                    "stage_weight": args.stage_weight,
                    "history_bev_weight": (
                        args.history_bev_weight
                    ),
                    "history_cos_weight": (
                        args.history_cos_weight
                    ),
                },
            },
            f,
            indent=2,
        )

    tb_log.close()
    logger.info("Elastic-v3 training completed")


if __name__ == "__main__":
    main()
