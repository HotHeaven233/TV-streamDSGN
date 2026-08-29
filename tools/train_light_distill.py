#!/usr/bin/env python3

import argparse
import copy
import glob
import os
from collections import namedtuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
from tensorboardX import SummaryWriter

from pcdet.config import (
    cfg,
    cfg_from_yaml_file,
    cfg_from_list,
    log_config_to_file,
)
from pcdet.datasets import build_dataloader
from pcdet.models import (
    build_network,
    load_data_to_gpu,
)
from pcdet.utils import common_utils

from train_utils.optimization import (
    build_optimizer,
    build_scheduler,
)
from train_utils.train_utils import train_model


torch.backends.cudnn.benchmark = True


# ============================================================
# Arguments
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train StreamDSGN Light-v1 under a frozen Full teacher "
            "with pre-fusion BEV feature distillation."
        )
    )

    parser.add_argument(
        "--cfg_file",
        type=str,
        required=True,
        help="Light-v1 config.",
    )

    parser.add_argument(
        "--full_cfg",
        type=str,
        required=True,
        help="Frozen Full model config.",
    )

    parser.add_argument(
        "--full_ckpt",
        type=str,
        required=True,
        help="Frozen Full epoch-5 checkpoint.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--bev_weight",
        type=float,
        default=1.0,
        help="lambda_bev for SmoothL1(B_light, B_full).",
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        default="light_distill_v1",
    )

    parser.add_argument(
        "--fix_random_seed",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--ckpt_save_interval",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--max_ckpt_save_num",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Resume Light distillation checkpoint.",
    )

    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser.parse_args()


# ============================================================
# Config
# ============================================================

def load_light_cfg(args):
    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    cfg.TAG = Path(
        args.cfg_file
    ).stem

    cfg.EXP_GROUP_PATH = "_".join(
        args.cfg_file.split("/")[1:-1]
    )

    cfg.OPTIMIZATION.USE_AMP = (
        cfg.MODEL.get(
            "USE_AMP",
            {
                "TRAIN": False,
                "TEST": False,
            },
        )
    )

    if args.set_cfgs is not None:
        cfg_from_list(
            args.set_cfgs,
            cfg,
        )

    # We immediately initialize from the Full checkpoint.
    # Do not trigger legacy torchvision://resnet18 loading.
    if hasattr(
        cfg.MODEL,
        "BACKBONE_3D",
    ):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None

    return cfg


def load_full_cfg(args):
    full_cfg = EasyDict()

    full_cfg.ROOT_DIR = cfg.ROOT_DIR
    full_cfg.LOCAL_RANK = 0

    cfg_from_yaml_file(
        args.full_cfg,
        full_cfg,
    )

    # Full checkpoint contains the trained ResNet already.
    if hasattr(
        full_cfg.MODEL,
        "BACKBONE_3D",
    ):
        full_cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None

    return full_cfg


# ============================================================
# Freeze policy
# ============================================================

def configure_light_trainable(
    model,
    logger,
):
    """
    Train only the Light state generator.

    FROZEN
    ------
    backbone_3d.feature_backbone
        shared Full prefix:
            stem
            layer1
            layer2

    temporal fusion
    MH residual
    VAN
    detection head
    all other downstream parameters

    TRAINABLE
    ---------
    backbone_3d except feature_backbone:
        Light feature_neck
        Light stereo modules
        Light depth modules
        Light RPN3D

    map_to_bev_module.bev_projection:
        48 -> 96
    """

    trainable_names = []
    frozen_names = []

    for name, param in (
        model.named_parameters()
    ):

        train_this = False

        # --------------------------------------------
        # Light state generator after shared ResNet
        # --------------------------------------------

        if name.startswith(
            "backbone_3d."
        ):

            # Shared Full prefix must remain bitwise frozen.
            if not name.startswith(
                "backbone_3d.feature_backbone."
            ):
                train_this = True

        # --------------------------------------------
        # Light 48 -> 96 memory interface
        # --------------------------------------------

        if name.startswith(
            "map_to_bev_module.bev_projection."
        ):
            train_this = True

        param.requires_grad = (
            train_this
        )

        if train_this:
            trainable_names.append(
                name
            )
        else:
            frozen_names.append(
                name
            )

    if len(trainable_names) == 0:
        raise RuntimeError(
            "No Light parameters were marked trainable."
        )

    model._train_light_distill = True

    trainable_num = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_num = sum(
        p.numel()
        for p in model.parameters()
    )

    logger.info(
        "=" * 80
    )
    logger.info(
        "Light-v1 distillation freeze policy"
    )
    logger.info(
        f"Trainable parameters: "
        f"{trainable_num:,} / {total_num:,} "
        f"({100.0 * trainable_num / total_num:.3f}%)"
    )

    logger.info(
        "TRAINABLE PARAMETER GROUPS:"
    )

    for name in trainable_names:
        logger.info(
            f"  TRAINABLE: {name}"
        )

    logger.info(
        "=" * 80
    )

    return (
        trainable_names,
        frozen_names,
    )


# ============================================================
# Frozen-state safety
# ============================================================

def set_frozen_bn_eval(
    model,
):
    """
    train_model() calls model.train() at every iteration.

    requires_grad=False does NOT stop BatchNorm running_mean/running_var
    from changing.

    Therefore, after model.train(), restore every frozen BN to eval mode.

    Trainable Light BN remains in train mode.
    """

    for module in model.modules():

        if not isinstance(
            module,
            nn.modules.batchnorm._BatchNorm,
        ):
            continue

        direct_params = list(
            module.parameters(
                recurse=False
            )
        )

        has_trainable_parameter = any(
            p.requires_grad
            for p in direct_params
        )

        if not has_trainable_parameter:
            module.eval()


def snapshot_frozen_state(
    model,
):
    """
    Snapshot frozen parameters + frozen BN buffers.

    Used as a hard safety check after training.
    """

    snapshot = {}

    param_dict = dict(
        model.named_parameters()
    )

    for name, tensor in (
        model.state_dict().items()
    ):

        # Frozen parameter
        if name in param_dict:

            if not param_dict[
                name
            ].requires_grad:

                snapshot[name] = (
                    tensor.detach()
                    .cpu()
                    .clone()
                )

            continue

        # Frozen BN mutable buffers
        if (
            name.endswith(
                "running_mean"
            )
            or
            name.endswith(
                "running_var"
            )
            or
            name.endswith(
                "num_batches_tracked"
            )
        ):

            snapshot[name] = (
                tensor.detach()
                .cpu()
                .clone()
            )

    return snapshot


def verify_frozen_state(
    model,
    snapshot,
    logger,
):
    current = (
        model.state_dict()
    )

    worst_name = None
    worst_diff = 0.0

    changed = []

    for name, reference in (
        snapshot.items()
    ):

        now = (
            current[name]
            .detach()
            .cpu()
        )

        if reference.dtype.is_floating_point:

            diff = float(
                (
                    now.float()
                    -
                    reference.float()
                )
                .abs()
                .max()
                .item()
            )

        else:

            diff = (
                0.0
                if torch.equal(
                    now,
                    reference,
                )
                else 1.0
            )

        if diff > worst_diff:
            worst_diff = diff
            worst_name = name

        if diff != 0.0:
            changed.append(
                (
                    name,
                    diff,
                )
            )

    logger.info(
        "=" * 80
    )
    logger.info(
        "Frozen-state verification"
    )
    logger.info(
        f"Checked frozen tensors: "
        f"{len(snapshot)}"
    )
    logger.info(
        f"Maximum change: "
        f"{worst_diff:.10e}"
    )
    logger.info(
        f"Worst tensor: "
        f"{worst_name}"
    )

    if len(changed) != 0:

        logger.error(
            "Frozen Full/shared state changed!"
        )

        for name, diff in (
            changed[:20]
        ):
            logger.error(
                f"  CHANGED: {name} "
                f"max_diff={diff:.10e}"
            )

        raise RuntimeError(
            "Full-preservation violated: "
            "at least one frozen parameter/buffer changed."
        )

    logger.info(
        "PASS: all frozen parameters and BN "
        "buffers are bitwise unchanged."
    )
    logger.info(
        "=" * 80
    )


# ============================================================
# Shared-prefix equality
# ============================================================

def verify_shared_prefix(
    student,
    teacher,
    logger,
):
    """
    Student Light stem/layer1/layer2 should come directly from
    the Full checkpoint and must initially equal the Full teacher.
    """

    teacher_params = dict(
        teacher.named_parameters()
    )

    checked = 0
    worst_diff = 0.0
    worst_name = None

    for name, student_param in (
        student.named_parameters()
    ):

        if not name.startswith(
            "backbone_3d.feature_backbone."
        ):
            continue

        if name not in teacher_params:
            raise RuntimeError(
                "Shared-prefix parameter is missing "
                f"from Full teacher: {name}"
            )

        teacher_param = (
            teacher_params[name]
        )

        if (
            student_param.shape
            !=
            teacher_param.shape
        ):
            raise RuntimeError(
                "Shared-prefix shape mismatch: "
                f"{name}: "
                f"student={student_param.shape}, "
                f"teacher={teacher_param.shape}"
            )

        diff = float(
            (
                student_param.detach().float()
                -
                teacher_param.detach().float()
            )
            .abs()
            .max()
            .item()
        )

        checked += 1

        if diff > worst_diff:
            worst_diff = diff
            worst_name = name

    if checked == 0:
        raise RuntimeError(
            "No shared ResNet prefix parameters "
            "were checked."
        )

    logger.info(
        "=" * 80
    )
    logger.info(
        "Shared-prefix initialization verification"
    )
    logger.info(
        f"Checked parameters: {checked}"
    )
    logger.info(
        f"Max |Light-Full|: "
        f"{worst_diff:.10e}"
    )
    logger.info(
        f"Worst parameter: "
        f"{worst_name}"
    )

    if worst_diff != 0.0:
        raise RuntimeError(
            "Light shared prefix is not identical "
            "to Full teacher after checkpoint loading."
        )

    logger.info(
        "PASS: Light shared ResNet prefix "
        "is bitwise identical to Full."
    )
    logger.info(
        "=" * 80
    )


# ============================================================
# Student BEV capture
# ============================================================

def install_student_bev_capture(
    student,
):
    """
    Student forward_train order:

        1. current/token feature extractor
        2. history feature extraction
        3. temporal fusion

    Therefore the FIRST map_to_bev output in each forward is B_t^L,
    the current frame's pre-fusion BEV.

    Later calls correspond to history frames and are ignored.
    """

    student._light_bev_capture = []
    student._capture_light_bev = False

    module = getattr(
        student,
        "map_to_bev_module",
        None,
    )

    if module is None:
        raise RuntimeError(
            "Student has no map_to_bev_module."
        )

    def hook(
        _module,
        _inputs,
        output,
    ):
        if not getattr(
            student,
            "_capture_light_bev",
            False,
        ):
            return

        if not isinstance(
            output,
            dict,
        ):
            raise RuntimeError(
                "map_to_bev output is not dict."
            )

        if "spatial_features" not in output:
            raise RuntimeError(
                "map_to_bev output has no "
                "`spatial_features`."
            )

        student._light_bev_capture.append(
            output[
                "spatial_features"
            ]
        )

    return module.register_forward_hook(
        hook
    )


# ============================================================
# Full teacher target
# ============================================================

def extract_full_bev(
    teacher,
    token_dict,
):
    """
    Only run the Full state generator:

        Full StreamDSGN2Backbone
        -> HeightCompression
        -> B_t^F

    No temporal fusion / VAN / detection head is executed.
    """

    teacher_data = copy.copy(
        token_dict
    )

    amp_enabled = bool(
        teacher.use_amp_dict[
            "TEST"
        ]
    )

    with torch.no_grad():

        with torch.cuda.amp.autocast(
            enabled=amp_enabled
        ):

            for module in (
                teacher.feature_extractor
            ):

                teacher_data = module(
                    teacher_data
                )

    if "spatial_features" not in (
        teacher_data
    ):
        raise RuntimeError(
            "Full teacher did not produce "
            "`spatial_features`."
        )

    return (
        teacher_data[
            "spatial_features"
        ]
        .detach()
    )


# ============================================================
# Distillation model function
# ============================================================

def light_distill_model_fn(
    teacher,
    bev_weight,
):
    ModelReturn = namedtuple(
        "ModelReturn",
        [
            "loss",
            "tb_dict",
            "disp_dict",
        ],
    )

    def model_func(
        student,
        batch_dict,
    ):
        # train_model() has already called student.train().
        # Restore frozen BN state before every iteration.
        set_frozen_bn_eval(
            student
        )

        teacher.eval()

        # --------------------------------------------
        # H2D
        # --------------------------------------------

        load_data_to_gpu(
            batch_dict
        )

        if (
            "token" not in batch_dict
            or
            batch_dict["token"] is None
        ):
            raise RuntimeError(
                "Training batch has no token frame."
            )

        # --------------------------------------------
        # Frozen Full target
        # --------------------------------------------

        full_bev = extract_full_bev(
            teacher,
            batch_dict["token"],
        )

        # --------------------------------------------
        # Student normal training forward
        # --------------------------------------------

        student._light_bev_capture.clear()
        student._capture_light_bev = True

        try:

            ret_dict, tb_dict, disp_dict = (
                student(
                    batch_dict
                )
            )

        finally:

            student._capture_light_bev = False

        if len(
            student._light_bev_capture
        ) == 0:

            raise RuntimeError(
                "Student BEV hook captured nothing."
            )

        # FIRST call = current/token frame.
        light_bev = (
            student._light_bev_capture[0]
        )

        # Release references to history BEVs immediately.
        student._light_bev_capture.clear()

        if (
            light_bev.shape
            !=
            full_bev.shape
        ):
            raise RuntimeError(
                "BEV distillation shape mismatch: "
                f"Light={tuple(light_bev.shape)}, "
                f"Full={tuple(full_bev.shape)}"
            )

        # --------------------------------------------
        # Loss
        # --------------------------------------------

        det_loss = (
            ret_dict[
                "loss"
            ].mean()
        )

        # FP32 distillation for numerical stability.
        bev_loss = F.smooth_l1_loss(
            light_bev.float(),
            full_bev.float(),
            reduction="mean",
        )

        total_loss = (
            det_loss
            +
            float(bev_weight)
            *
            bev_loss
        )

        # Additional easy-to-interpret metric.
        with torch.no_grad():

            bev_l1 = (
                light_bev.float()
                .sub(
                    full_bev.float()
                )
                .abs()
                .mean()
            )

        tb_dict = dict(
            tb_dict
        )

        tb_dict.update(
            {
                "loss_light_det": float(
                    det_loss.detach().item()
                ),
                "loss_bev_distill": float(
                    bev_loss.detach().item()
                ),
                "loss_bev_weighted": float(
                    (
                        float(bev_weight)
                        *
                        bev_loss.detach()
                    ).item()
                ),
                "bev_l1": float(
                    bev_l1.item()
                ),
            }
        )

        if hasattr(
            student,
            "update_global_step",
        ):
            student.update_global_step()

        return ModelReturn(
            total_loss,
            tb_dict,
            disp_dict,
        )

    return model_func


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    light_cfg = load_light_cfg(
        args
    )

    full_cfg = load_full_cfg(
        args
    )

    if args.fix_random_seed:
        common_utils.set_random_seed(
            666
        )

    if args.batch_size is None:
        args.batch_size = int(
            light_cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
        )

    # ========================================================
    # Output
    # ========================================================

    output_dir = (
        light_cfg.ROOT_DIR
        /
        "outputs"
        /
        light_cfg.EXP_GROUP_PATH
        /
        (
            light_cfg.TAG
            +
            "."
            +
            args.exp_name
        )
    )

    ckpt_dir = (
        output_dir
        /
        "ckpt"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = common_utils.create_logger(
        output_dir
        /
        "log_train.txt"
    )

    logger.info(
        "=" * 80
    )
    logger.info(
        "Frozen-Full / Light-v1 BEV distillation"
    )
    logger.info(
        "=" * 80
    )

    logger.info(
        f"Light config : {args.cfg_file}"
    )
    logger.info(
        f"Full config  : {args.full_cfg}"
    )
    logger.info(
        f"Full ckpt    : {args.full_ckpt}"
    )
    logger.info(
        f"epochs       : {args.epochs}"
    )
    logger.info(
        f"batch_size   : {args.batch_size}"
    )
    logger.info(
        f"bev_weight   : {args.bev_weight}"
    )
    logger.info(
        f"output_dir   : {output_dir}"
    )

    log_config_to_file(
        light_cfg,
        logger=logger,
    )

    tb_log = SummaryWriter(
        log_dir=str(
            output_dir
            /
            "tensorboard"
        )
    )

    # ========================================================
    # Dataset
    # ========================================================

    train_set, train_loader, train_sampler = (
        build_dataloader(
            dataset_cfg=light_cfg.DATA_CONFIG,
            class_names=light_cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=False,
            workers=args.workers,
            logger=logger,
            training=True,
            merge_all_iters_to_one_epoch=False,
            total_epochs=args.epochs,
        )
    )

    # ========================================================
    # Full teacher
    # ========================================================

    logger.info(
        "Building frozen Full teacher..."
    )

    teacher = build_network(
        model_cfg=full_cfg.MODEL,
        num_class=len(
            full_cfg.CLASS_NAMES
        ),
        dataset=train_set,
    )

    teacher.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )

    teacher.cuda()
    teacher.eval()

    for param in (
        teacher.parameters()
    ):
        param.requires_grad = False

    # ========================================================
    # Light student
    # ========================================================

    logger.info(
        "Building Light-v1 student..."
    )

    student = build_network(
        model_cfg=light_cfg.MODEL,
        num_class=len(
            light_cfg.CLASS_NAMES
        ),
        dataset=train_set,
    )

    # Partial load:
    # matching Full weights initialize shared prefix/downstream.
    # New/narrow Light layers remain their own initialization.
    student.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )

    student.cuda()

    # ========================================================
    # Freeze
    # ========================================================

    configure_light_trainable(
        student,
        logger,
    )

    # Hard check:
    # shared stem/layer1/layer2 must equal Full exactly.
    verify_shared_prefix(
        student,
        teacher,
        logger,
    )

    # Snapshot frozen state BEFORE optimization.
    frozen_snapshot = (
        snapshot_frozen_state(
            student
        )
    )

    # Student current-BEV capture.
    bev_hook = (
        install_student_bev_capture(
            student
        )
    )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = build_optimizer(
        student,
        light_cfg.OPTIMIZATION,
    )

    start_epoch = 0
    start_iter = 0
    last_epoch = -1

    if args.ckpt is not None:

        start_iter, start_epoch = (
            student.load_params_with_optimizer(
                args.ckpt,
                to_cpu=True,
                optimizer=optimizer,
                logger=logger,
            )
        )

        last_epoch = (
            start_epoch + 1
        )

        logger.info(
            f"Resume Light training from "
            f"epoch={start_epoch}, "
            f"iter={start_iter}"
        )

    lr_scheduler, lr_warmup_scheduler = (
        build_scheduler(
            optimizer,
            total_iters_each_epoch=len(
                train_loader
            ),
            total_epochs=args.epochs,
            last_epoch=last_epoch,
            optim_cfg=light_cfg.OPTIMIZATION,
        )
    )

    # ========================================================
    # Train
    # ========================================================

    student.train()

    logger.info(
        "=" * 80
    )
    logger.info(
        "Start Light-v1 training"
    )
    logger.info(
        "Full teacher: frozen/eval/no_grad"
    )
    logger.info(
        "Shared ResNet prefix: frozen"
    )
    logger.info(
        "Downstream temporal/head: frozen"
    )
    logger.info(
        "Only Light state generator is trainable"
    )
    logger.info(
        "=" * 80
    )

    train_model(
        student,
        optimizer,
        train_loader,
        model_func=light_distill_model_fn(
            teacher=teacher,
            bev_weight=args.bev_weight,
        ),
        lr_scheduler=lr_scheduler,
        optim_cfg=light_cfg.OPTIMIZATION,
        start_epoch=start_epoch,
        total_epochs=args.epochs,
        start_iter=start_iter,
        rank=0,
        tb_log=tb_log,
        ckpt_save_dir=ckpt_dir,
        train_sampler=train_sampler,
        lr_warmup_scheduler=lr_warmup_scheduler,
        ckpt_save_interval=args.ckpt_save_interval,
        max_ckpt_save_num=args.max_ckpt_save_num,
        merge_all_iters_to_one_epoch=False,
        dist_train=False,
        logger=logger,
    )

    # ========================================================
    # Full-preservation safety check
    # ========================================================

    verify_frozen_state(
        student,
        frozen_snapshot,
        logger,
    )

    bev_hook.remove()

    tb_log.close()

    logger.info(
        "=" * 80
    )
    logger.info(
        "Light-v1 distillation completed."
    )
    logger.info(
        f"Checkpoints: {ckpt_dir}"
    )
    logger.info(
        "=" * 80
    )


if __name__ == "__main__":
    main()
