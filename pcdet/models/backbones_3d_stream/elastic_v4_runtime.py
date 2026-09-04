#!/usr/bin/env python3
"""
Elastic-v4 latency probe runtime.

Purpose
-------
This module is intentionally inference-only. It lets an already-trained v3
checkpoint answer two latency questions *before* spending another training run:

1) How much time is lost to per-frame slimmable weight slicing/index_select?
2) How fast would the same elastic convolutional workload be if runtime
   normalization were fuseable (BN -> Conv fusion), i.e. with no GroupNorm
   reductions in the deployed graph?

It also implements the deployment rule proposed for v4:
all leading 1.0 stages execute the original frozen StreamDSGN modules. Only
after the first ratio < 1.0 does execution enter the elastic branch.

This file does NOT change v3 weights and does NOT claim fused_proxy accuracy.
`fused_proxy` is a latency-only proxy because v3 was trained with GroupNorm.
"""

from __future__ import annotations

import types
from typing import Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v3 import (
    SlimConv2d,
    SlimConv3d,
    SlimConvTranspose3d,
    extract_fixed_layer1_prefix,
    ratio_to_channels,
    validate_schedule,
)


def _indices_key(input_indices):
    if input_indices is None:
        return None
    if isinstance(input_indices, torch.Tensor):
        return tuple(int(x) for x in input_indices.detach().cpu().tolist())
    return tuple(int(x) for x in input_indices)


def _cached_conv2d_forward(self, x, out_ratio, input_indices=None):
    out_c = ratio_to_channels(self.max_out_channels, out_ratio)
    in_c = int(x.shape[1])
    indices_key = _indices_key(input_indices)
    key = (float(out_ratio), in_c, indices_key)

    cache = self.__dict__.setdefault("_v4_weight_cache", {})
    cached = cache.get(key)
    if cached is None:
        with torch.no_grad():
            if input_indices is None:
                if in_c > self.max_in_channels:
                    raise RuntimeError((in_c, self.max_in_channels))
                weight = self.weight[:out_c, :in_c].detach().contiguous()
            else:
                indices = torch.tensor(
                    indices_key, device=self.weight.device, dtype=torch.long
                )
                if len(indices_key) != in_c:
                    raise RuntimeError(
                        f"input index count {len(indices_key)} != tensor channels {in_c}"
                    )
                weight = (
                    self.weight[:out_c]
                    .index_select(1, indices)
                    .detach()
                    .contiguous()
                )
            bias = (
                self.bias[:out_c].detach().contiguous()
                if self.bias is not None
                else None
            )
            cached = (weight, bias)
            cache[key] = cached

    weight, bias = cached
    y = F.conv2d(
        x,
        weight,
        bias=bias,
        stride=self.stride,
        padding=self.padding,
        dilation=self.dilation,
    )
    if self.norm is not None and not getattr(self, "_v4_bypass_norm", False):
        y = self.norm(y, out_ratio)
    if self.activation:
        y = F.relu(y, inplace=False)
    return y


def _cached_conv3d_forward(self, x, out_ratio, input_indices=None):
    out_c = ratio_to_channels(self.max_out_channels, out_ratio)
    in_c = int(x.shape[1])
    indices_key = _indices_key(input_indices)
    key = (float(out_ratio), in_c, indices_key)

    cache = self.__dict__.setdefault("_v4_weight_cache", {})
    cached = cache.get(key)
    if cached is None:
        with torch.no_grad():
            if input_indices is None:
                if in_c > self.max_in_channels:
                    raise RuntimeError((in_c, self.max_in_channels))
                weight = self.weight[:out_c, :in_c].detach().contiguous()
            else:
                indices = torch.tensor(
                    indices_key, device=self.weight.device, dtype=torch.long
                )
                if len(indices_key) != in_c:
                    raise RuntimeError(
                        f"input index count {len(indices_key)} != tensor channels {in_c}"
                    )
                weight = (
                    self.weight[:out_c]
                    .index_select(1, indices)
                    .detach()
                    .contiguous()
                )
            bias = (
                self.bias[:out_c].detach().contiguous()
                if self.bias is not None
                else None
            )
            cached = (weight, bias)
            cache[key] = cached

    weight, bias = cached
    y = F.conv3d(
        x,
        weight,
        bias=bias,
        stride=self.stride,
        padding=self.padding,
    )
    if self.norm is not None and not getattr(self, "_v4_bypass_norm", False):
        y = self.norm(y, out_ratio)
    if self.activation:
        y = F.relu(y, inplace=False)
    return y


def _cached_deconv3d_forward(self, x, out_ratio):
    in_c = int(x.shape[1])
    out_c = ratio_to_channels(self.max_out_channels, out_ratio)
    key = (float(out_ratio), in_c)

    cache = self.__dict__.setdefault("_v4_weight_cache", {})
    cached = cache.get(key)
    if cached is None:
        with torch.no_grad():
            weight = self.weight[:in_c, :out_c].detach().contiguous()
            bias = (
                self.bias[:out_c].detach().contiguous()
                if self.bias is not None
                else None
            )
            cached = (weight, bias)
            cache[key] = cached

    weight, bias = cached
    y = F.conv_transpose3d(
        x,
        weight,
        bias=bias,
        stride=self.stride,
        padding=self.padding,
        output_padding=self.output_padding,
    )
    if self.norm is not None and not getattr(self, "_v4_bypass_norm", False):
        y = self.norm(y, out_ratio)
    return y


def enable_static_weight_cache(branch, bypass_groupnorm=False):
    """
    Replace v3 slimmable forwards with eval-only cached contiguous kernels.

    The first occurrence of a width/transition creates the contiguous kernel.
    All later frames reuse it, so per-frame slicing, torch.as_tensor and
    index_select disappear from the steady-state timing path.
    """
    if branch.training:
        raise RuntimeError("Static runtime cache is inference-only; call branch.eval().")

    n2d = n3d = ndeconv = 0
    for module in branch.modules():
        if isinstance(module, SlimConv2d):
            module._v4_weight_cache = {}
            module._v4_bypass_norm = bool(bypass_groupnorm)
            module.forward = types.MethodType(_cached_conv2d_forward, module)
            n2d += 1
        elif isinstance(module, SlimConv3d):
            module._v4_weight_cache = {}
            module._v4_bypass_norm = bool(bypass_groupnorm)
            module.forward = types.MethodType(_cached_conv3d_forward, module)
            n3d += 1
        elif isinstance(module, SlimConvTranspose3d):
            module._v4_weight_cache = {}
            module._v4_bypass_norm = bool(bypass_groupnorm)
            module.forward = types.MethodType(_cached_deconv3d_forward, module)
            ndeconv += 1

    return {"conv2d": n2d, "conv3d": n3d, "deconv3d": ndeconv}


def _native_res_stage(resnet, stage_name, left, right):
    stage = getattr(resnet, stage_name)
    left = stage(left)
    right = None if right is None else stage(right)
    return left, right


def _native_fpn(full_backbone, frame, state):
    left_stereo, left_sem = full_backbone.feature_neck(
        [
            frame["left_img"],
            state["left_l1"],
            state["left_l2"],
            state["left_l3"],
            state["left_l4"],
        ]
    )
    if full_backbone.mono:
        right_stereo = right_sem = None
    else:
        right_stereo, right_sem = full_backbone.feature_neck(
            [
                frame["right_img"],
                state["right_l1"],
                state["right_l2"],
                state["right_l3"],
                state["right_l4"],
            ]
        )

    state = dict(state)
    state["left_stereo"] = left_stereo
    state["right_stereo"] = right_stereo
    state["left_sem"] = left_sem
    state["right_sem"] = right_sem
    return state


def _native_stereo(full_backbone, frame, state):
    """
    Execute exactly the original fixed-width stereo cost/dres0/dres1 path.
    This is used only when every stage up through Stereo remains at 1.0 and
    the first shrink is RPN.
    """
    left = state["left_stereo"]
    right = state["right_stereo"]
    dtype = left.dtype
    calib = frame["calib"]

    fu_mul_baseline = torch.as_tensor(
        [x.fu_mul_baseline for x in calib],
        dtype=dtype,
        device=left.device,
    )
    depth = (
        full_backbone._downsampled_depth_fp16
        if dtype == torch.float16
        else full_backbone.downsampled_depth
    ).to(left.device)
    downsampled_disp = (
        fu_mul_baseline[:, None]
        / depth[None, :]
        / (
            full_backbone.downsample_disp
            if not full_backbone.fullres_stereo_feature
            else 1
        )
    )

    if int(left.shape[1]) > int(full_backbone.cv_dim):
        psv = full_backbone.compute_disp_channels(
            downsampled_disp[0],
            int(left.shape[1]),
            inv_ratio=float(full_backbone.inv_smooth_psv),
        )
        cost = full_backbone.build_cost(
            left,
            right,
            None,
            None,
            downsampled_disp,
            psv.to(torch.int32),
        )
    else:
        cost = full_backbone.build_cost(
            left, right, None, None, downsampled_disp
        )

    x = full_backbone.dres0(cost)
    x = full_backbone.dres1(x) + x
    return x


class HybridFixedBranch:
    """
    Deployment-shaped fixed-profile executor.

    Leading 1.0 stages are run by the original StreamDSGN modules. After the
    first ratio < 1.0, all remaining stages run through the elastic branch.

    Because legal profiles are monotonic non-increasing, there is at most one
    Full -> Elastic transition.
    """

    def __init__(self, elastic_branch):
        self.elastic = elastic_branch

    def forward_from_prefix(self, frame, prefix_cache, full_backbone, schedule):
        schedule = validate_schedule(schedule)
        if all(float(r) == 1.0 for r in schedule):
            raise RuntimeError(
                "All-1.0 should bypass HybridFixedBranch and use native Full directly."
            )

        state = {
            "left_l1": prefix_cache["left_l1"],
            "right_l1": prefix_cache["right_l1"],
        }
        elastic_started = False

        # Res2
        r = float(schedule[0])
        if not elastic_started and r == 1.0:
            state["left_l2"], state["right_l2"] = _native_res_stage(
                full_backbone.feature_backbone,
                "layer2",
                state["left_l1"],
                state["right_l1"],
            )
        else:
            elastic_started = True
            e = self.elastic.stage_res2(prefix_cache, r)
            state.update(e)

        # Res3
        r = float(schedule[1])
        if not elastic_started and r == 1.0:
            state["left_l3"], state["right_l3"] = _native_res_stage(
                full_backbone.feature_backbone,
                "layer3",
                state["left_l2"],
                state["right_l2"],
            )
        else:
            elastic_started = True
            state = self.elastic.stage_res3(state, r)

        # Res4
        r = float(schedule[2])
        if not elastic_started and r == 1.0:
            state["left_l4"], state["right_l4"] = _native_res_stage(
                full_backbone.feature_backbone,
                "layer4",
                state["left_l3"],
                state["right_l3"],
            )
        else:
            elastic_started = True
            state = self.elastic.stage_res4(state, r)

        # FPN
        r = float(schedule[3])
        if not elastic_started and r == 1.0:
            state = _native_fpn(full_backbone, frame, state)
        else:
            elastic_started = True
            state = self.elastic.stage_fpn(frame, state, r)

        # Stereo
        r = float(schedule[4])
        if not elastic_started and r == 1.0:
            stereo = _native_stereo(full_backbone, frame, state)
        else:
            elastic_started = True
            stereo = self.elastic.stage_stereo(
                frame, state, full_backbone, r
            )

        # RPN. For a non-all-1 profile, either a shrink has already happened or
        # the first shrink happens here.
        r = float(schedule[5])
        bev, valids = self.elastic.stage_rpn(
            frame, stereo, full_backbone, r
        )
        return bev, valids

