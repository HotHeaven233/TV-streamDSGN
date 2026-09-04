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

from pcdet.models.backbones_3d_stream.elastic_bev_branch import (
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the six-stage parameter-sharing elastic branch on top of "
            "a frozen K3 residual StreamDSGN base. ResNet stem+layer1, K3 "
            "fusion, VAN and detection head remain unchanged."
        )
    )
    parser.add_argument("--full_cfg", required=True)
    parser.add_argument("--full_ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--exp_name", type=str, default="elastic_bev_v2")
    parser.add_argument("--output_root", type=str, default="outputs/elastic_bev")
    parser.add_argument("--resume", type=str, default=None)

    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=2e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--det_weight", type=float, default=1.0)
    parser.add_argument("--bev_weight", type=float, default=2.0)
    parser.add_argument("--cos_weight", type=float, default=0.20)
    parser.add_argument("--aux_width_weight", type=float, default=0.35)
    parser.add_argument("--smooth_l1_beta", type=float, default=0.1)
    parser.add_argument(
        "--canonical_history_prob",
        type=float,
        default=0.60,
        help=(
            "Probability of using the original [t-3,t-2,t-1] history. "
            "The remaining probability is uniform over the other 9 triples "
            "selected from t-5...t-1."
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
    """
    Extend the already prepared K3 training infos to prev3/prev4/prev5 by
    copying the corresponding token entry from the same scene. No KITTI
    preprocessing regeneration is required.
    """
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
                key = f"prev{offset}" if offset > 1 else "prev"
                if pos < offset:
                    frame_tag[key] = ""
                    info["infos"].pop(key, None)
                    continue
                prev_index, prev_token = items[pos - offset]
                prev_info = train_set.kitti_infos[prev_index]
                frame_tag[key] = str(prev_token)
                info["infos"][key] = copy.deepcopy(prev_info["infos"]["token"])
                counts[offset] += 1

    # StreamingSampler expects every temporal key present in the data to also
    # exist in ALL_SAMPLE_TAG. Add only the missing offsets; existing K3 tags
    # are left untouched.
    if getattr(train_set, "data_augmentor", None) is not None:
        for augmentor in train_set.data_augmentor.data_augmentor_queue:
            if hasattr(augmentor, "all_sample_tag"):
                for offset in range(3, max_offset + 1):
                    key = f"prev{offset}"
                    augmentor.all_sample_tag[key] = -offset

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
    canonical_available = CANONICAL_HISTORY in available_triples

    if canonical_available and random.random() < float(canonical_probability):
        return CANONICAL_HISTORY

    alternatives = [x for x in available_triples if x != CANONICAL_HISTORY]
    if alternatives:
        return random.choice(alternatives)
    return CANONICAL_HISTORY


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

    # Canonical K3 queue order is oldest -> newest. The newest selected frame
    # is intentionally NOT forced to t-1; e.g. [t-5,t-4,t-2] is legal.
    for key in ("prev3", "prev2", "prev"):
        batch_dict[key] = None
    target_keys = ("prev3", "prev2", "prev")[-len(selected):]
    for target_key, offset in zip(target_keys, selected):
        batch_dict[target_key] = candidates[offset]
    return tuple(selected)


def sample_primary_schedule():
    return random.choice(MONOTONIC_SCHEDULES)


def auxiliary_schedule(global_step):
    # Sandwich-style endpoint coverage: alternate the narrowest and widest
    # subnet. The random primary schedule covers all intermediate paths.
    if global_step % 2 == 0:
        return (0.25,) * 6
    return (1.0,) * 6


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
            "Frozen K3 base changed during elastic training:\n" + detail
        )


def full_bev_from_prefix(base_model, frame_dict, prefix_cache):
    backbone = base_model.backbone_3d
    amp_enabled = bool(base_model.use_amp_dict["TEST"])
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
        full_2d = extract_full_2d_from_layer1(backbone, frame_dict, prefix_cache)
        teacher_data = copy.copy(frame_dict)
        with cached_full_2d_prefix(backbone, full_2d):
            teacher_data = backbone(teacher_data)
        teacher_data = base_model.map_to_bev_module(teacher_data)
    return teacher_data["spatial_features"].detach()


def full_bev(base_model, frame_dict):
    backbone = base_model.backbone_3d
    amp_enabled = bool(base_model.use_amp_dict["TEST"])
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
        prefix = extract_fixed_layer1_prefix(backbone, frame_dict)
    return full_bev_from_prefix(base_model, frame_dict, prefix)


def build_history_queue(base_model, batch_dict):
    history_queue = deque(maxlen=3)
    for key in ("prev3", "prev2", "prev"):
        frame = batch_dict.get(key)
        if frame is None:
            continue
        bev = full_bev(base_model, frame)
        history_queue.append(
            (frame["this_sample_idx"], {"spatial_features": bev})
        )
    return history_queue


def run_frozen_k3_downstream(base_model, batch_dict, elastic_bev, valids, history_queue):
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
    """
    Randomly spaced histories invalidate the equal-step velocity/trend loss.
    Keep current task classification/regression supervision while allowing its
    gradient to flow through the frozen K3 downstream into the elastic BEV.
    """
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


def lr_for_step(step, total_steps, base_lr, min_lr, warmup_ratio):
    warmup_steps = max(100, int(total_steps * warmup_ratio))
    warmup_steps = min(warmup_steps, max(total_steps - 1, 1))
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
    return min_lr + (base_lr - min_lr) * cosine


def save_checkpoint(path, branch, optimizer, scaler, epoch, global_step, args):
    torch.save(
        {
            "version": "elastic_v2_resnet_fpn",
            "epoch": int(epoch),
            "global_step": int(global_step),
            "branch": branch.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "width_choices": list(WIDTH_CHOICES),
            "num_elastic_stages": 6,
            "canonical_history": list(CANONICAL_HISTORY),
            "canonical_history_probability": float(args.canonical_history_prob),
            "history_candidates": list(HISTORY_OFFSETS),
        },
        path,
    )


def main():
    args = parse_args()
    if not 0.0 <= args.canonical_history_prob <= 1.0:
        raise ValueError("canonical_history_prob must be in [0,1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_seed(args.seed)
    full_cfg = make_cfg(args.full_cfg)

    output_dir = Path(args.output_root) / args.exp_name
    ckpt_dir = output_dir / "ckpt"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(output_dir / "log_train.txt")
    tb_log = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    logger.info("=" * 80)
    logger.info("Elastic-v2: frozen K3 base + elastic ResNet2-4/FPN/stereo/RPN")
    logger.info("Frozen timing prefix: original ResNet stem + layer1")
    logger.info(f"full_cfg       : {args.full_cfg}")
    logger.info(f"full_ckpt      : {args.full_ckpt}")
    logger.info(f"epochs         : {args.epochs}")
    logger.info(f"lr             : {args.lr} -> {args.min_lr}")
    logger.info(f"history main   : {CANONICAL_HISTORY} prob={args.canonical_history_prob}")
    logger.info("history random : choose 3 distinct frames from t-5...t-1")
    logger.info(f"legal schedules: {len(MONOTONIC_SCHEDULES)}")
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
    base_model.cuda()
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    if base_model.history_feature_queue is not None and base_model.history_feature_queue.maxlen != 3:
        logger.warning(
            f"Base history queue maxlen={base_model.history_feature_queue.maxlen}; "
            "elastic training explicitly supplies the selected three frames."
        )

    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(full_cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES),
    ).cuda()

    trainable = sum(p.numel() for p in branch.parameters() if p.requires_grad)
    logger.info(f"Elastic trainable parameters: {trainable:,}")
    logger.info("Base K3 trainable parameters: 0")

    optimizer = torch.optim.AdamW(
        branch.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(full_cfg.MODEL.USE_AMP.get("TRAIN", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu")
        branch.load_state_dict(checkpoint["branch"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint.get("global_step", 0))
        logger.info(f"Resumed from {args.resume} at epoch={start_epoch}")

    base_snapshot = snapshot_base_model(base_model)
    total_steps = args.epochs * len(train_loader)

    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_set, "set_epoch"):
            train_set.set_epoch(epoch)
        branch.train()

        running = {"loss": 0.0, "det": 0.0, "bev": 0.0, "cos": 0.0, "aux": 0.0, "count": 0}

        for iteration, batch_dict in enumerate(train_loader):
            load_data_to_gpu(batch_dict)
            selected_history = remap_random_history(
                batch_dict, args.canonical_history_prob
            )

            optimizer.zero_grad(set_to_none=True)
            lr = lr_for_step(
                global_step, total_steps, args.lr, args.min_lr, args.warmup_ratio
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Current frame: execute the frozen timing prefix only once. It is
            # shared by the Full BEV teacher and the elastic student.
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                prefix_cache = extract_fixed_layer1_prefix(
                    base_model.backbone_3d, batch_dict["token"]
                )
                full_bev_target = full_bev_from_prefix(
                    base_model, batch_dict["token"], prefix_cache
                )
                history_queue = build_history_queue(base_model, batch_dict)

            primary_schedule = sample_primary_schedule()
            endpoint_schedule = auxiliary_schedule(global_step)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                elastic_bev, valids = branch(
                    batch_dict["token"],
                    prefix_cache,
                    base_model.backbone_3d,
                    primary_schedule,
                )
                run_frozen_k3_downstream(
                    base_model,
                    batch_dict,
                    elastic_bev,
                    valids,
                    history_queue,
                )
                det_loss, _ = detection_loss_without_trend(base_model, batch_dict)
                bev_loss, cos_loss, bev_l1 = bev_distill_loss(
                    elastic_bev, full_bev_target, args.smooth_l1_beta
                )

                # The inherited 100% weights start close to Full. During the
                # first epoch, let BEV alignment dominate before exposing the
                # shared kernels to the full detection gradient.
                ramp_steps = max(len(train_loader), 1)
                det_ramp = 0.25 + 0.75 * min(
                    1.0, float(global_step + 1) / float(ramp_steps)
                )
                primary_loss = (
                    args.det_weight * det_ramp * det_loss
                    + args.bev_weight * bev_loss
                    + args.cos_weight * cos_loss
                )

            scaler.scale(primary_loss).backward()

            # Endpoint auxiliary path. Only BEV distillation is used here to
            # cover the all-25% and all-100% subnet without a second expensive
            # frozen K3 downstream pass.
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                aux_bev, _ = branch(
                    batch_dict["token"],
                    prefix_cache,
                    base_model.backbone_3d,
                    endpoint_schedule,
                )
                aux_bev_loss, aux_cos_loss, _ = bev_distill_loss(
                    aux_bev, full_bev_target, args.smooth_l1_beta
                )
                aux_loss = aux_bev_loss + args.cos_weight * aux_cos_loss
                aux_weighted = args.aux_width_weight * aux_loss

            scaler.scale(aux_weighted).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(branch.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            total_loss_value = float(primary_loss.detach().item() + aux_weighted.detach().item())
            running["loss"] += total_loss_value
            running["det"] += float(det_loss.detach().item())
            running["bev"] += float(bev_loss.detach().item())
            running["cos"] += float(cos_loss.detach().item())
            running["aux"] += float(aux_loss.detach().item())
            running["count"] += 1

            tb_log.add_scalar("train/loss", total_loss_value, global_step)
            tb_log.add_scalar("train/det", det_loss.detach().item(), global_step)
            tb_log.add_scalar("train/bev_smooth_l1", bev_loss.detach().item(), global_step)
            tb_log.add_scalar("train/bev_cos", cos_loss.detach().item(), global_step)
            tb_log.add_scalar("train/bev_l1", bev_l1.detach().item(), global_step)
            tb_log.add_scalar("train/aux_width", aux_loss.detach().item(), global_step)
            tb_log.add_scalar("train/lr", lr, global_step)
            for idx, ratio in enumerate(primary_schedule, start=1):
                tb_log.add_scalar(f"train/width_stage_{idx}", ratio, global_step)
            tb_log.add_scalar(
                "train/history_is_canonical",
                float(selected_history == CANONICAL_HISTORY),
                global_step,
            )

            if (iteration + 1) % args.log_interval == 0:
                c = max(running["count"], 1)
                logger.info(
                    f"epoch={epoch + 1:02d}/{args.epochs:02d} "
                    f"iter={iteration + 1:05d}/{len(train_loader):05d} "
                    f"lr={lr:.3e} hist={selected_history} "
                    f"width={primary_schedule} "
                    f"loss={running['loss']/c:.4f} "
                    f"det={running['det']/c:.4f} "
                    f"bev={running['bev']/c:.4f} "
                    f"cos={running['cos']/c:.4f}"
                )
                running = {"loss": 0.0, "det": 0.0, "bev": 0.0, "cos": 0.0, "aux": 0.0, "count": 0}

            global_step += 1

        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = ckpt_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            save_checkpoint(
                ckpt_path, branch, optimizer, scaler, epoch, global_step, args
            )
            logger.info(f"Saved {ckpt_path}")

        verify_base_unchanged(base_model, base_snapshot)
        logger.info("PASS: frozen K3 base parameters and buffers are bitwise unchanged")

    with open(output_dir / "training_design.json", "w") as f:
        json.dump(
            {
                "version": "elastic_v2_resnet_fpn",
                "fixed_prefix": "original ResNet stem + layer1",
                "elastic_stages": [
                    "ResNet-layer2",
                    "ResNet-layer3",
                    "ResNet-layer4",
                    "FPN",
                    "Stereo3D",
                    "RPN3D-to-BEV",
                ],
                "width_choices": list(WIDTH_CHOICES),
                "num_monotonic_schedules": len(MONOTONIC_SCHEDULES),
                "history_offsets": list(HISTORY_OFFSETS),
                "canonical_history": list(CANONICAL_HISTORY),
                "canonical_probability": args.canonical_history_prob,
                "noncanonical_probability_each": (
                    (1.0 - args.canonical_history_prob) / len(NON_CANONICAL_TRIPLETS)
                ),
                "loss": {
                    "det_weight": args.det_weight,
                    "bev_weight": args.bev_weight,
                    "cos_weight": args.cos_weight,
                    "aux_width_weight": args.aux_width_weight,
                },
            },
            f,
            indent=2,
        )

    tb_log.close()
    logger.info("Elastic-v2 training completed")


if __name__ == "__main__":
    main()
