#!/usr/bin/env python3
"""
Hybrid execution and BN-fused deployment helpers for Elastic-v4-BN.

Runtime semantics:
  * all leading 1.0 stages use the original frozen StreamDSGN modules;
  * after the first ratio < 1.0, execution enters the elastic branch;
  * because schedules are monotonic, there is only one Full -> Elastic switch;
  * elastic BatchNorm is fused into materialized Conv/ConvTranspose weights.

The same hybrid forward is also used during v4 training/calibration so the
feature distribution seen by BatchNorm matches deployment.
"""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    SlimConv2d,
    SlimConv3d,
    SlimConvTranspose3d,
    _ratio_key,
    ratio_to_channels,
    validate_schedule,
)


FULL_SCHEDULE = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


def first_elastic_stage(schedule):
    schedule = validate_schedule(schedule)
    for i, ratio in enumerate(schedule):
        if float(ratio) < 1.0:
            return i
    return None


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


def hybrid_forward_with_features(
    branch,
    frame,
    prefix_cache,
    full_backbone,
    schedule,
):
    """
    Run a known legal profile with native leading-1.0 stages.

    Returned feature dict contains only elastic stages. Distillation should skip
    missing native-prefix entries because they are already the exact Full path.
    """
    schedule = validate_schedule(schedule)
    first = first_elastic_stage(schedule)
    if first is None:
        raise RuntimeError(
            "All-1.0 is native Full and has no trainable elastic path."
        )

    state = {
        "left_l1": prefix_cache["left_l1"],
        "right_l1": prefix_cache["right_l1"],
    }
    features = {}

    # Res2
    if first > 0:
        state["left_l2"], state["right_l2"] = _native_res_stage(
            full_backbone.feature_backbone,
            "layer2",
            state["left_l1"],
            state["right_l1"],
        )
    else:
        state = branch.stage_res2(prefix_cache, schedule[0])
        features["res2_left"] = state["left_l2"]
        features["res2_right"] = state["right_l2"]

    # Res3
    if first > 1:
        state["left_l3"], state["right_l3"] = _native_res_stage(
            full_backbone.feature_backbone,
            "layer3",
            state["left_l2"],
            state["right_l2"],
        )
    else:
        state = branch.stage_res3(state, schedule[1])
        features["res3_left"] = state["left_l3"]
        features["res3_right"] = state["right_l3"]

    # Res4
    if first > 2:
        state["left_l4"], state["right_l4"] = _native_res_stage(
            full_backbone.feature_backbone,
            "layer4",
            state["left_l3"],
            state["right_l3"],
        )
    else:
        state = branch.stage_res4(state, schedule[2])
        features["res4_left"] = state["left_l4"]
        features["res4_right"] = state["right_l4"]

    # FPN
    if first > 3:
        state = _native_fpn(full_backbone, frame, state)
    else:
        state = branch.stage_fpn(frame, state, schedule[3])
        features["fpn_left"] = state["left_stereo"]
        features["fpn_right"] = state["right_stereo"]

    # Stereo
    if first > 4:
        stereo = _native_stereo(full_backbone, frame, state)
    else:
        stereo = branch.stage_stereo(
            frame, state, full_backbone, schedule[4]
        )
        features["stereo"] = stereo

    # Any non-Full monotonic profile has an elastic RPN width < 1.0 here.
    bev, valids, rpn_pool = branch.stage_rpn(
        frame,
        stereo,
        full_backbone,
        schedule[5],
        return_feature=True,
    )
    features["rpn_pool"] = rpn_pool
    return bev, valids, features


def hybrid_forward(
    branch,
    frame,
    prefix_cache,
    full_backbone,
    schedule,
):
    bev, valids, _ = hybrid_forward_with_features(
        branch,
        frame,
        prefix_cache,
        full_backbone,
        schedule,
    )
    return bev, valids


def _indices_key(input_indices):
    if input_indices is None:
        return None
    if isinstance(input_indices, torch.Tensor):
        return tuple(int(x) for x in input_indices.detach().cpu().tolist())
    return tuple(int(x) for x in input_indices)


@torch.no_grad()
def _fuse_bn_conv2d(module, out_ratio, in_c, indices_key):
    out_c = ratio_to_channels(module.max_out_channels, out_ratio)
    if indices_key is None:
        weight = module.weight[:out_c, :in_c]
    else:
        indices = torch.tensor(
            indices_key, device=module.weight.device, dtype=torch.long
        )
        weight = module.weight[:out_c].index_select(1, indices)

    if module.bias is None:
        base_bias = torch.zeros(
            out_c, device=weight.device, dtype=weight.dtype
        )
    else:
        base_bias = module.bias[:out_c]

    if module.norm is None:
        return weight.detach().contiguous(), base_bias.detach().contiguous()

    bn = module.norm.bns[_ratio_key(out_ratio)]
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = bn.bias + (base_bias.float() - bn.running_mean) * scale
    fused_weight = weight.float() * scale[:, None, None, None]
    return (
        fused_weight.to(weight.dtype).detach().contiguous(),
        bias.to(weight.dtype).detach().contiguous(),
    )


@torch.no_grad()
def _fuse_bn_conv3d(module, out_ratio, in_c, indices_key):
    out_c = ratio_to_channels(module.max_out_channels, out_ratio)
    if indices_key is None:
        weight = module.weight[:out_c, :in_c]
    else:
        indices = torch.tensor(
            indices_key, device=module.weight.device, dtype=torch.long
        )
        weight = module.weight[:out_c].index_select(1, indices)

    if module.bias is None:
        base_bias = torch.zeros(
            out_c, device=weight.device, dtype=weight.dtype
        )
    else:
        base_bias = module.bias[:out_c]

    if module.norm is None:
        return weight.detach().contiguous(), base_bias.detach().contiguous()

    bn = module.norm.bns[_ratio_key(out_ratio)]
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = bn.bias + (base_bias.float() - bn.running_mean) * scale
    fused_weight = weight.float() * scale[:, None, None, None, None]
    return (
        fused_weight.to(weight.dtype).detach().contiguous(),
        bias.to(weight.dtype).detach().contiguous(),
    )


@torch.no_grad()
def _fuse_bn_deconv3d(module, out_ratio, in_c):
    out_c = ratio_to_channels(module.max_out_channels, out_ratio)
    weight = module.weight[:in_c, :out_c]
    if module.bias is None:
        base_bias = torch.zeros(
            out_c, device=weight.device, dtype=weight.dtype
        )
    else:
        base_bias = module.bias[:out_c]

    if module.norm is None:
        return weight.detach().contiguous(), base_bias.detach().contiguous()

    bn = module.norm.bns[_ratio_key(out_ratio)]
    scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    bias = bn.bias + (base_bias.float() - bn.running_mean) * scale
    # ConvTranspose3d weight is [in_channels, out_channels/groups, kD,kH,kW].
    fused_weight = weight.float() * scale[None, :, None, None, None]
    return (
        fused_weight.to(weight.dtype).detach().contiguous(),
        bias.to(weight.dtype).detach().contiguous(),
    )


def _fused_conv2d_forward(self, x, out_ratio, input_indices=None):
    in_c = int(x.shape[1])
    indices_key = _indices_key(input_indices)
    key = (float(out_ratio), in_c, indices_key)
    cache = self.__dict__.setdefault("_v4_fused_cache", {})
    cached = cache.get(key)
    if cached is None:
        cached = _fuse_bn_conv2d(self, out_ratio, in_c, indices_key)
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
    if self.activation:
        y = F.relu(y, inplace=False)
    return y


def _fused_conv3d_forward(self, x, out_ratio, input_indices=None):
    in_c = int(x.shape[1])
    indices_key = _indices_key(input_indices)
    key = (float(out_ratio), in_c, indices_key)
    cache = self.__dict__.setdefault("_v4_fused_cache", {})
    cached = cache.get(key)
    if cached is None:
        cached = _fuse_bn_conv3d(self, out_ratio, in_c, indices_key)
        cache[key] = cached
    weight, bias = cached
    y = F.conv3d(
        x,
        weight,
        bias=bias,
        stride=self.stride,
        padding=self.padding,
    )
    if self.activation:
        y = F.relu(y, inplace=False)
    return y


def _fused_deconv3d_forward(self, x, out_ratio):
    in_c = int(x.shape[1])
    key = (float(out_ratio), in_c)
    cache = self.__dict__.setdefault("_v4_fused_cache", {})
    cached = cache.get(key)
    if cached is None:
        cached = _fuse_bn_deconv3d(self, out_ratio, in_c)
        cache[key] = cached
    weight, bias = cached
    return F.conv_transpose3d(
        x,
        weight,
        bias=bias,
        stride=self.stride,
        padding=self.padding,
        output_padding=self.output_padding,
    )


def enable_fused_bn_static_cache(branch):
    """
    Materialize/fuse lazily on first use, then reuse contiguous kernels.

    Call only after loading a BN-calibrated checkpoint and branch.eval().
    The fused output is mathematically equivalent to eval-mode BatchNorm up to
    normal floating-point roundoff, but no BN kernel runs per frame.
    """
    if branch.training:
        raise RuntimeError("Call branch.eval() before BN fusion.")

    counts = {"conv2d": 0, "conv3d": 0, "deconv3d": 0}
    for module in branch.modules():
        if isinstance(module, SlimConv2d):
            module._v4_fused_cache = {}
            module.forward = types.MethodType(_fused_conv2d_forward, module)
            counts["conv2d"] += 1
        elif isinstance(module, SlimConv3d):
            module._v4_fused_cache = {}
            module.forward = types.MethodType(_fused_conv3d_forward, module)
            counts["conv3d"] += 1
        elif isinstance(module, SlimConvTranspose3d):
            module._v4_fused_cache = {}
            module.forward = types.MethodType(_fused_deconv3d_forward, module)
            counts["deconv3d"] += 1
    return counts

