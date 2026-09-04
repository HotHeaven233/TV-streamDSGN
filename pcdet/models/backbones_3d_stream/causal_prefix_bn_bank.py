#!/usr/bin/env python3
"""
Utilities for causal-prefix BatchNorm banks in Elastic-v4-BN.

The elastic convolution weights and BN affine parameters (gamma/beta) stay
shared exactly as trained.  Only BN running_mean/running_var are indexed by the
causal width prefix that has already been chosen at a given stage.

For a six-stage monotonic profile there are:
  Res2=3, Res3=9, Res4=19, FPN=34, Stereo=55, RPN=83
non-Full causal prefixes, 203 states in total.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    MONOTONIC_SCHEDULES,
    SwitchableBatchNorm2d,
    SwitchableBatchNorm3d,
    _ratio_key,
    validate_schedule,
)

STAGE_NAMES = ("res2", "res3", "res4", "fpn", "stereo", "rpn")


def ratio_token(r):
    r = float(r)
    return {
        1.0: "100",
        0.75: "075",
        0.5: "050",
        0.25: "025",
    }[r]


def prefix_key(prefix):
    return ">".join(ratio_token(x) for x in prefix)


def causal_prefixes_by_stage():
    out = OrderedDict()
    for stage_idx, stage_name in enumerate(STAGE_NAMES):
        length = stage_idx + 1
        values = {
            tuple(float(x) for x in schedule[:length])
            for schedule in MONOTONIC_SCHEDULES
            # Final width 1.0 means this stage is still native Full.
            if float(schedule[stage_idx]) < 1.0
        }
        out[stage_name] = tuple(sorted(values, reverse=True))
    counts = [len(out[name]) for name in STAGE_NAMES]
    if counts != [3, 9, 19, 34, 55, 83]:
        raise RuntimeError(f"unexpected causal-prefix counts: {counts}")
    return out


def stage_index_from_bn_name(name):
    # ElasticResNetTail stores layer2/layer3/layer4 in a ModuleList named
    # `stages`, so named_modules() yields resnet.stages.0/1/2..., not
    # resnet.stage2/3/4.
    if name.startswith("resnet.stages.0."):
        return 0
    if name.startswith("resnet.stages.1."):
        return 1
    if name.startswith("resnet.stages.2."):
        return 2
    if name.startswith("fpn."):
        return 3
    if name.startswith("dres0.") or name.startswith("dres1."):
        return 4
    if name.startswith("rpn_in.") or name.startswith("rpn_hg."):
        return 5
    return None


def stage_switchable_bns(branch, stage_idx):
    result = []
    for name, module in branch.named_modules():
        if isinstance(module, (SwitchableBatchNorm2d, SwitchableBatchNorm3d)):
            if stage_index_from_bn_name(name) == int(stage_idx):
                result.append((name, module))
    if not result:
        raise RuntimeError(
            f"No switchable BN modules found for stage {STAGE_NAMES[stage_idx]}"
        )
    return result


def discovered_stage_bn_counts(branch):
    counts = {name: 0 for name in STAGE_NAMES}
    unknown = []
    for name, module in branch.named_modules():
        if isinstance(module, (SwitchableBatchNorm2d, SwitchableBatchNorm3d)):
            idx = stage_index_from_bn_name(name)
            if idx is None:
                unknown.append(name)
            else:
                counts[STAGE_NAMES[idx]] += 1
    return counts, unknown


@torch.no_grad()
def reset_stage_active_bn(branch, stage_idx, ratio, cumulative=True):
    active = []
    key = _ratio_key(ratio)
    for name, switch_bn in stage_switchable_bns(branch, stage_idx):
        bn = switch_bn.bns[key]
        bn.reset_running_stats()
        bn.momentum = None if cumulative else 0.1
        bn.train()
        active.append((name, bn))
    return active


@torch.no_grad()
def capture_stage_stats(branch, stage_idx, prefix):
    ratio = float(prefix[-1])
    key = _ratio_key(ratio)
    stats = {}
    for name, switch_bn in stage_switchable_bns(branch, stage_idx):
        bn = switch_bn.bns[key]
        stats[name] = {
            "ratio_key": key,
            "running_mean": bn.running_mean.detach().cpu().clone(),
            "running_var": bn.running_var.detach().cpu().clone(),
            "num_batches_tracked": bn.num_batches_tracked.detach().cpu().clone(),
            "eps": float(bn.eps),
        }
    return stats


@torch.no_grad()
def apply_stage_stats(branch, bank, stage_idx, prefix, strict=True):
    stage_name = STAGE_NAMES[stage_idx]
    pkey = prefix_key(prefix)
    stage_bank = bank["stages"].get(stage_name, {})
    if pkey not in stage_bank:
        if strict:
            raise KeyError(f"Missing prefix BN state: {stage_name}:{pkey}")
        return False

    expected_ratio_key = _ratio_key(prefix[-1])
    named = dict(branch.named_modules())
    for module_name, state in stage_bank[pkey]["bn"].items():
        switch_bn = named.get(module_name)
        if switch_bn is None:
            raise KeyError(f"BN module no longer exists: {module_name}")
        ratio_key = state["ratio_key"]
        if ratio_key != expected_ratio_key:
            raise RuntimeError(
                f"ratio mismatch for {stage_name}:{pkey}: "
                f"{ratio_key} vs {expected_ratio_key}"
            )
        bn = switch_bn.bns[ratio_key]
        bn.running_mean.copy_(
            state["running_mean"].to(
                device=bn.running_mean.device,
                dtype=bn.running_mean.dtype,
            )
        )
        bn.running_var.copy_(
            state["running_var"].to(
                device=bn.running_var.device,
                dtype=bn.running_var.dtype,
            )
        )
        bn.num_batches_tracked.copy_(
            state["num_batches_tracked"].to(
                device=bn.num_batches_tracked.device
            )
        )
        bn.eval()
    return True


@torch.no_grad()
def apply_earlier_prefix_stats(branch, bank, prefix, current_stage_idx):
    for stage_idx in range(int(current_stage_idx)):
        if float(prefix[stage_idx]) >= 1.0:
            continue
        apply_stage_stats(
            branch,
            bank,
            stage_idx,
            tuple(prefix[: stage_idx + 1]),
            strict=True,
        )


@torch.no_grad()
def apply_schedule_bank(branch, bank, schedule, strict=True):
    schedule = validate_schedule(schedule)
    applied = []
    branch.eval()
    for stage_idx, ratio in enumerate(schedule):
        if float(ratio) >= 1.0:
            continue
        prefix = tuple(schedule[: stage_idx + 1])
        ok = apply_stage_stats(
            branch, bank, stage_idx, prefix, strict=strict
        )
        if ok:
            applied.append(
                (STAGE_NAMES[stage_idx], prefix_key(prefix))
            )
    return applied


def required_prefix_keys(schedule):
    schedule = validate_schedule(schedule)
    return [
        (STAGE_NAMES[i], prefix_key(schedule[: i + 1]))
        for i, r in enumerate(schedule)
        if float(r) < 1.0
    ]

