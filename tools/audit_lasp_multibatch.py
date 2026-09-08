#!/usr/bin/env python3

import argparse
import math

import torch

from pcdet.config import (
    cfg,
    cfg_from_yaml_file,
)
from pcdet.datasets import (
    build_dataloader,
)
from pcdet.models import (
    build_network,
    load_data_to_gpu,
)
from pcdet.utils import (
    common_utils,
)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--cfg_file",
        required=True,
    )

    ap.add_argument(
        "--pretrained_model",
        required=True,
    )

    ap.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--num_batches",
        type=int,
        default=20,
    )

    args = ap.parse_args()

    cfg_from_yaml_file(
        args.cfg_file,
        cfg,
    )

    logger = common_utils.create_logger(
        rank=0
    )

    (
        dataset,
        loader,
        _,
    ) = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=True,
    )

    model = build_network(
        model_cfg=cfg.MODEL,
        num_class=len(
            cfg.CLASS_NAMES
        ),
        dataset=dataset,
    )

    model.cuda()

    model.load_params_from_file(
        filename=args.pretrained_model,
        logger=logger,
        to_cpu=True,
    )

    model.train()

    trainable = [
        n
        for n, p
        in model.named_parameters()
        if p.requires_grad
    ]

    assert not any(
        n.startswith(
            "backbone_3d."
        )
        for n in trainable
    )

    assert any(
        n.startswith(
            "backbone_2d."
        )
        for n in trainable
    )

    assert any(
        n.startswith(
            "dense_head."
        )
        for n in trainable
    )

    assert any(
        n.startswith(
            "lasp."
        )
        for n in trainable
    )

    print(
        "[PASS] trainable parameter scope"
    )

    count = 0
    total_matches = 0
    zero_match_batches = 0
    positive_traj_batches = 0
    positive_intention_batches = 0

    losses = []

    for batch_idx, batch in enumerate(
        loader
    ):

        if batch_idx >= args.num_batches:
            break

        model.zero_grad(
            set_to_none=True
        )

        load_data_to_gpu(
            batch
        )

        ret, tb, _ = model(
            batch
        )

        loss = ret[
            "loss"
        ]

        if not torch.isfinite(
            loss
        ):
            raise RuntimeError(
                f"batch {batch_idx}: "
                f"non-finite loss={loss}"
            )

        if not loss.requires_grad:
            raise RuntimeError(
                f"batch {batch_idx}: "
                "loss has no grad_fn"
            )

        loss.backward()

        grad_sum = 0.0
        grad_count = 0

        for name, param in (
            model.named_parameters()
        ):

            if (
                name.startswith(
                    "lasp."
                )
                and
                param.grad is not None
            ):
                g = float(
                    param.grad
                    .detach()
                    .float()
                    .norm()
                    .item()
                )

                if not math.isfinite(g):
                    raise RuntimeError(
                        f"batch {batch_idx}: "
                        f"non-finite gradient "
                        f"in {name}"
                    )

                grad_sum += g
                grad_count += 1

        matches = int(
            tb.get(
                "lasp_matches",
                0,
            )
        )

        traj = float(
            tb.get(
                "lasp_traj_loss",
                0.0,
            )
        )

        intention = float(
            tb.get(
                "lasp_intention_loss",
                0.0,
            )
        )

        total_matches += matches

        if matches == 0:
            zero_match_batches += 1

        if traj > 0:
            positive_traj_batches += 1

        if intention > 0:
            positive_intention_batches += 1

        if (
            matches > 0
            and (
                grad_count == 0
                or grad_sum <= 0
            )
        ):
            raise RuntimeError(
                f"batch {batch_idx}: "
                "matched LASP objects but "
                "LASP gradients are zero"
            )

        value = float(
            loss.detach().item()
        )

        losses.append(
            value
        )

        print(
            f"[{batch_idx + 1:02d}/"
            f"{args.num_batches}] "
            f"loss={value:.6f} "
            f"matches={matches} "
            f"traj={traj:.6f} "
            f"intent={intention:.6f} "
            f"lasp_grad={grad_sum:.6f}"
        )

        count += 1

        del batch
        del ret
        del loss

    if count != args.num_batches:
        raise RuntimeError(
            f"requested {args.num_batches} "
            f"batches but processed {count}"
        )

    if total_matches == 0:
        raise RuntimeError(
            "zero LASP matches across "
            "all audit batches"
        )

    if positive_traj_batches == 0:
        raise RuntimeError(
            "trajectory loss was never positive"
        )

    if positive_intention_batches == 0:
        raise RuntimeError(
            "H8 intention loss was never positive"
        )

    print()
    print(
        "=========================================="
    )
    print(
        "LASP MULTI-BATCH AUDIT PASS"
    )
    print(
        "=========================================="
    )
    print(
        "batches                 =",
        count,
    )
    print(
        "mean loss               =",
        sum(losses) / len(losses),
    )
    print(
        "total matches            =",
        total_matches,
    )
    print(
        "zero-match batches       =",
        zero_match_batches,
    )
    print(
        "positive trajectory batch=",
        positive_traj_batches,
    )
    print(
        "positive intention batch =",
        positive_intention_batches,
    )


if __name__ == "__main__":
    main()
