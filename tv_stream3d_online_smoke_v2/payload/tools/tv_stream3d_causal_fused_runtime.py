#!/usr/bin/env python3
from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    SlimConv2d,
    SlimConv3d,
    SlimConvTranspose3d,
)
from pcdet.models.backbones_3d_stream.elastic_v4_bn_hybrid import (
    _fuse_bn_conv2d,
    _fuse_bn_conv3d,
    _fuse_bn_deconv3d,
    _indices_key,
)
from pcdet.models.backbones_3d_stream.causal_prefix_bn_bank import (
    apply_stage_stats,
    prefix_key,
)


def _context(self):
    ctx = getattr(self, "_tv_prefix_context", None)
    if ctx is None:
        raise RuntimeError(
            "Elastic fused module executed without a causal-prefix context"
        )
    return ctx


def _cache_get_or_build(self, key, build_fn):
    cache = self.__dict__.setdefault("_tv_fused_cache", {})
    cached = cache.get(key)
    if cached is not None:
        return cached

    if bool(getattr(self, "_tv_forbid_cache_miss", False)):
        raise RuntimeError(
            "Unexpected causal-prefix fused-cache miss during measured forward: "
            f"context={getattr(self, '_tv_prefix_context', None)}, key={key}. "
            "Increase warmup or inspect controller path variability."
        )

    cached = build_fn()
    cache[key] = cached
    self._tv_fused_miss_count = int(
        getattr(self, "_tv_fused_miss_count", 0)
    ) + 1
    return cached


def _tv_fused_conv2d_forward(self, x, out_ratio, input_indices=None):
    in_c = int(x.shape[1])
    indices_key = _indices_key(input_indices)
    ctx = _context(self)
    key = (ctx, float(out_ratio), in_c, indices_key)

    weight, bias = _cache_get_or_build(
        self,
        key,
        lambda: _fuse_bn_conv2d(
            self,
            out_ratio,
            in_c,
            indices_key,
        ),
    )

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


def _tv_fused_conv3d_forward(self, x, out_ratio, input_indices=None):
    in_c = int(x.shape[1])
    indices_key = _indices_key(input_indices)
    ctx = _context(self)
    key = (ctx, float(out_ratio), in_c, indices_key)

    weight, bias = _cache_get_or_build(
        self,
        key,
        lambda: _fuse_bn_conv3d(
            self,
            out_ratio,
            in_c,
            indices_key,
        ),
    )

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


def _tv_fused_deconv3d_forward(self, x, out_ratio):
    in_c = int(x.shape[1])
    ctx = _context(self)
    key = (ctx, float(out_ratio), in_c)

    weight, bias = _cache_get_or_build(
        self,
        key,
        lambda: _fuse_bn_deconv3d(
            self,
            out_ratio,
            in_c,
        ),
    )

    return F.conv_transpose3d(
        x,
        weight,
        bias=bias,
        stride=self.stride,
        padding=self.padding,
        output_padding=self.output_padding,
    )


def enable_causal_prefix_fused_cache(branch):
    """
    Install a prefix-keyed lazy Conv-BN fusion cache.

    Unlike enable_fused_bn_static_cache(), the cache key contains the causal
    execution-prefix identity. The same width can therefore carry different
    BN running statistics after different earlier prefixes.

    Cache entries are created only during warmup. Measured forwards can forbid
    new cache entries, which prevents hidden first-use fusion overhead.
    """
    if branch.training:
        raise RuntimeError("Call branch.eval() before enabling fused cache")

    counts = {"conv2d": 0, "conv3d": 0, "deconv3d": 0}

    for module in branch.modules():
        if isinstance(module, SlimConv2d):
            module._tv_fused_cache = {}
            module._tv_fused_miss_count = 0
            module._tv_forbid_cache_miss = False
            module._tv_prefix_context = None
            module.forward = types.MethodType(
                _tv_fused_conv2d_forward,
                module,
            )
            counts["conv2d"] += 1

        elif isinstance(module, SlimConv3d):
            module._tv_fused_cache = {}
            module._tv_fused_miss_count = 0
            module._tv_forbid_cache_miss = False
            module._tv_prefix_context = None
            module.forward = types.MethodType(
                _tv_fused_conv3d_forward,
                module,
            )
            counts["conv3d"] += 1

        elif isinstance(module, SlimConvTranspose3d):
            module._tv_fused_cache = {}
            module._tv_fused_miss_count = 0
            module._tv_forbid_cache_miss = False
            module._tv_prefix_context = None
            module.forward = types.MethodType(
                _tv_fused_deconv3d_forward,
                module,
            )
            counts["deconv3d"] += 1

    branch._tv_ready_prefixes = set()
    return counts


def fused_modules(branch):
    for module in branch.modules():
        if isinstance(
            module,
            (SlimConv2d, SlimConv3d, SlimConvTranspose3d),
        ):
            yield module


@torch.no_grad()
def begin_stage_prefix(
    branch,
    bank,
    stage_idx,
    prefix,
    allow_prepare,
):
    prefix = tuple(float(x) for x in prefix)
    if float(prefix[-1]) >= 1.0:
        raise ValueError(
            "Causal-prefix BN is only used after entering the elastic path"
        )

    ctx = (int(stage_idx), prefix_key(prefix))
    ready = branch._tv_ready_prefixes

    if ctx not in ready:
        if not allow_prepare:
            raise RuntimeError(
                "Measured forward reached an unprepared causal prefix: "
                f"{ctx}. Increase warmup before paper timing."
            )

        # Stats must be active while the context-specific fused kernels are
        # first materialized.
        apply_stage_stats(
            branch,
            bank,
            int(stage_idx),
            prefix,
            strict=True,
        )

    for module in fused_modules(branch):
        module._tv_prefix_context = ctx

    return ctx


def finish_stage_prefix(branch, ctx, allow_prepare):
    if allow_prepare:
        branch._tv_ready_prefixes.add(ctx)


def set_forbid_cache_miss(branch, value=True):
    value = bool(value)
    for module in fused_modules(branch):
        module._tv_forbid_cache_miss = value


def fused_cache_stats(branch):
    entries = 0
    misses = 0

    for module in fused_modules(branch):
        entries += len(
            getattr(module, "_tv_fused_cache", {})
        )
        misses += int(
            getattr(module, "_tv_fused_miss_count", 0)
        )

    return {
        "entries": int(entries),
        "misses": int(misses),
        "ready_prefixes": int(
            len(
                getattr(
                    branch,
                    "_tv_ready_prefixes",
                    set(),
                )
            )
        ),
    }
