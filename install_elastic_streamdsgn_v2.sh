#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-/data/jhb/workspace/streamDSGN}"
cd "${ROOT}"

echo "[Elastic-v2] repo: ${ROOT}"
mkdir -p pcdet/models/backbones_3d_stream tools scripts/stream_exp

# Backup only files that this installer overwrites.
STAMP="$(date +%Y%m%d_%H%M%S)"
for f in     pcdet/models/backbones_3d_stream/elastic_bev_branch.py     tools/train_elastic_bev.py     tools/test_elastic_stream.py     scripts/stream_exp/14_train_elastic.sh     scripts/stream_exp/15_test_elastic.sh     scripts/stream_exp/16_sweep_elastic.sh; do
    if [[ -f "${f}" ]]; then
        cp -a "${f}" "${f}.bak_${STAMP}"
    fi
done

cat > pcdet/models/backbones_3d_stream/elastic_bev_branch.py <<'PY_ELASTIC_V2'
import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from pcdet.ops.build_cost_volume import build_cost_volume
from pcdet.ops.build_dps_cost_volume import build_dps_cost_volume
from pcdet.utils.torch_utils import project_pseudo_lidar_to_rectcam


WIDTH_CHOICES = (1.0, 0.75, 0.50, 0.25)
NUM_ELASTIC_STAGES = 6


def _ratio_key(ratio):
    ratio = float(ratio)
    mapping = {1.0: "100", 0.75: "075", 0.50: "050", 0.25: "025"}
    if ratio not in mapping:
        raise ValueError(f"Unsupported width ratio: {ratio}")
    return mapping[ratio]


def ratio_to_channels(max_channels, ratio):
    ratio = float(ratio)
    if ratio not in WIDTH_CHOICES:
        raise ValueError(f"Unsupported width ratio: {ratio}")
    channels = int(round(int(max_channels) * ratio))
    if channels <= 0:
        raise ValueError((max_channels, ratio, channels))
    return channels


def validate_schedule(schedule):
    if len(schedule) != NUM_ELASTIC_STAGES:
        raise ValueError(
            f"schedule must have {NUM_ELASTIC_STAGES} ratios, got {schedule}"
        )
    schedule = tuple(float(x) for x in schedule)
    if any(r not in WIDTH_CHOICES for r in schedule):
        raise ValueError(f"invalid schedule {schedule}")
    if any(schedule[i] < schedule[i + 1] for i in range(len(schedule) - 1)):
        raise ValueError(
            "Elastic schedule must be non-increasing from early to late stages: "
            f"{schedule}"
        )
    return schedule


def all_monotonic_schedules():
    schedules = []
    for r2 in WIDTH_CHOICES:
        for r3 in WIDTH_CHOICES:
            for r4 in WIDTH_CHOICES:
                for rf in WIDTH_CHOICES:
                    for rs in WIDTH_CHOICES:
                        for rr in WIDTH_CHOICES:
                            schedule = (r2, r3, r4, rf, rs, rr)
                            if all(
                                schedule[i] >= schedule[i + 1]
                                for i in range(len(schedule) - 1)
                            ):
                                schedules.append(schedule)
    return schedules


MONOTONIC_SCHEDULES = tuple(all_monotonic_schedules())


# ============================================================================
# Switchable normalization and slimmable convolutions
# ============================================================================


class SwitchableBatchNorm2d(nn.Module):
    """
    The convolutional kernel is shared by all widths, while each width keeps
    its own tiny BN affine/running statistics. This avoids mixing the feature
    distributions of 25/50/75/100% sub-networks.
    """

    def __init__(self, max_channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.max_channels = int(max_channels)
        self.bns = nn.ModuleDict()
        for ratio in WIDTH_CHOICES:
            c = ratio_to_channels(self.max_channels, ratio)
            self.bns[_ratio_key(ratio)] = nn.BatchNorm2d(
                c, eps=eps, momentum=momentum
            )

    def forward(self, x, ratio):
        return self.bns[_ratio_key(ratio)](x)

    @torch.no_grad()
    def initialize_from_full(self, source_bn):
        for ratio in WIDTH_CHOICES:
            bn = self.bns[_ratio_key(ratio)]
            c = bn.num_features
            if source_bn.affine:
                bn.weight.copy_(source_bn.weight[:c])
                bn.bias.copy_(source_bn.bias[:c])
            if source_bn.track_running_stats:
                bn.running_mean.copy_(source_bn.running_mean[:c])
                bn.running_var.copy_(source_bn.running_var[:c])
                bn.num_batches_tracked.copy_(source_bn.num_batches_tracked)


class SwitchableBatchNorm3d(nn.Module):
    def __init__(self, max_channels, eps=1e-5, momentum=0.1):
        super().__init__()
        self.max_channels = int(max_channels)
        self.bns = nn.ModuleDict()
        for ratio in WIDTH_CHOICES:
            c = ratio_to_channels(self.max_channels, ratio)
            self.bns[_ratio_key(ratio)] = nn.BatchNorm3d(
                c, eps=eps, momentum=momentum
            )

    def forward(self, x, ratio):
        return self.bns[_ratio_key(ratio)](x)

    @torch.no_grad()
    def initialize_from_full(self, source_bn):
        for ratio in WIDTH_CHOICES:
            bn = self.bns[_ratio_key(ratio)]
            c = bn.num_features
            if source_bn.affine:
                bn.weight.copy_(source_bn.weight[:c])
                bn.bias.copy_(source_bn.bias[:c])
            if source_bn.track_running_stats:
                bn.running_mean.copy_(source_bn.running_mean[:c])
                bn.running_var.copy_(source_bn.running_var[:c])
                bn.num_batches_tracked.copy_(source_bn.num_batches_tracked)


class SlimConv2d(nn.Module):
    def __init__(
        self,
        max_in_channels,
        max_out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        norm=True,
        activation=True,
    ):
        super().__init__()
        self.max_in_channels = int(max_in_channels)
        self.max_out_channels = int(max_out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(
            torch.empty(
                self.max_out_channels,
                self.max_in_channels,
                self.kernel_size,
                self.kernel_size,
            )
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.max_out_channels)) if bias else None
        )
        self.norm = (
            SwitchableBatchNorm2d(self.max_out_channels) if norm else None
        )
        self.activation = bool(activation)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, out_ratio, input_indices=None):
        out_c = ratio_to_channels(self.max_out_channels, out_ratio)
        in_c = int(x.shape[1])
        if input_indices is None:
            if in_c > self.max_in_channels:
                raise RuntimeError((in_c, self.max_in_channels))
            weight = self.weight[:out_c, :in_c]
        else:
            indices = torch.as_tensor(
                input_indices, device=self.weight.device, dtype=torch.long
            )
            if len(indices) != in_c:
                raise RuntimeError(
                    f"input index count {len(indices)} != tensor channels {in_c}"
                )
            weight = self.weight[:out_c].index_select(1, indices)
        bias = self.bias[:out_c] if self.bias is not None else None
        y = F.conv2d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )
        if self.norm is not None:
            y = self.norm(y, out_ratio)
        if self.activation:
            y = F.relu(y, inplace=False)
        return y

    @torch.no_grad()
    def initialize_from_full(self, source_conv, source_bn=None):
        if tuple(source_conv.weight.shape) != tuple(self.weight.shape):
            raise RuntimeError(
                f"Conv2d shape mismatch: source={tuple(source_conv.weight.shape)} "
                f"target={tuple(self.weight.shape)}"
            )
        self.weight.copy_(source_conv.weight)
        if self.bias is not None and source_conv.bias is not None:
            self.bias.copy_(source_conv.bias)
        if self.norm is not None and source_bn is not None:
            self.norm.initialize_from_full(source_bn)


class SlimConv3d(nn.Module):
    def __init__(
        self,
        max_in_channels,
        max_out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=False,
        norm=True,
        activation=True,
    ):
        super().__init__()
        self.max_in_channels = int(max_in_channels)
        self.max_out_channels = int(max_out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(
            torch.empty(
                self.max_out_channels,
                self.max_in_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
            )
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.max_out_channels)) if bias else None
        )
        self.norm = (
            SwitchableBatchNorm3d(self.max_out_channels) if norm else None
        )
        self.activation = bool(activation)
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, out_ratio, input_indices=None):
        out_c = ratio_to_channels(self.max_out_channels, out_ratio)
        in_c = int(x.shape[1])
        if input_indices is None:
            if in_c > self.max_in_channels:
                raise RuntimeError((in_c, self.max_in_channels))
            weight = self.weight[:out_c, :in_c]
        else:
            indices = torch.as_tensor(
                input_indices, device=self.weight.device, dtype=torch.long
            )
            if len(indices) != in_c:
                raise RuntimeError(
                    f"input index count {len(indices)} != tensor channels {in_c}"
                )
            weight = self.weight[:out_c].index_select(1, indices)
        bias = self.bias[:out_c] if self.bias is not None else None
        y = F.conv3d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
        )
        if self.norm is not None:
            y = self.norm(y, out_ratio)
        if self.activation:
            y = F.relu(y, inplace=False)
        return y

    @torch.no_grad()
    def initialize_from_full(self, source_conv, source_bn=None):
        if tuple(source_conv.weight.shape) != tuple(self.weight.shape):
            raise RuntimeError(
                f"Conv3d shape mismatch: source={tuple(source_conv.weight.shape)} "
                f"target={tuple(self.weight.shape)}"
            )
        self.weight.copy_(source_conv.weight)
        if self.bias is not None and source_conv.bias is not None:
            self.bias.copy_(source_conv.bias)
        if self.norm is not None and source_bn is not None:
            self.norm.initialize_from_full(source_bn)


class SlimConvTranspose3d(nn.Module):
    def __init__(
        self,
        max_in_channels,
        max_out_channels,
        kernel_size,
        stride,
        padding,
        output_padding,
        bias=False,
        norm=True,
    ):
        super().__init__()
        self.max_in_channels = int(max_in_channels)
        self.max_out_channels = int(max_out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.weight = nn.Parameter(
            torch.empty(
                self.max_in_channels,
                self.max_out_channels,
                self.kernel_size,
                self.kernel_size,
                self.kernel_size,
            )
        )
        self.bias = (
            nn.Parameter(torch.zeros(self.max_out_channels)) if bias else None
        )
        self.norm = (
            SwitchableBatchNorm3d(self.max_out_channels) if norm else None
        )
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, out_ratio):
        in_c = int(x.shape[1])
        out_c = ratio_to_channels(self.max_out_channels, out_ratio)
        weight = self.weight[:in_c, :out_c]
        bias = self.bias[:out_c] if self.bias is not None else None
        y = F.conv_transpose3d(
            x,
            weight,
            bias=bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
        )
        if self.norm is not None:
            y = self.norm(y, out_ratio)
        return y

    @torch.no_grad()
    def initialize_from_full(self, source_conv, source_bn=None):
        if tuple(source_conv.weight.shape) != tuple(self.weight.shape):
            raise RuntimeError(
                "ConvTranspose3d shape mismatch: "
                f"source={tuple(source_conv.weight.shape)} "
                f"target={tuple(self.weight.shape)}"
            )
        self.weight.copy_(source_conv.weight)
        if self.bias is not None and source_conv.bias is not None:
            self.bias.copy_(source_conv.bias)
        if self.norm is not None and source_bn is not None:
            self.norm.initialize_from_full(source_bn)


# ============================================================================
# Fixed timing prefix: original stem + original layer1, fully frozen
# ============================================================================


def run_frozen_resnet_layer1(resnet, image):
    """Run exactly the original ResNet stem and layer1."""
    if resnet.deep_stem:
        x = resnet.stem(image)
    else:
        x = resnet.conv1(image)
        x = resnet.norm1(x)
        x = resnet.relu(x)
    if resnet.with_max_pool:
        x = resnet.maxpool(x)
    x = resnet.layer1(x)
    return x


def extract_fixed_layer1_prefix(full_backbone, batch_dict):
    """
    This is the only non-elastic image backbone prefix. It is executed for
    both left/right images before the runtime decision and serves as the
    in-band timing probe.
    """
    left_l1 = run_frozen_resnet_layer1(
        full_backbone.feature_backbone, batch_dict["left_img"]
    )
    if full_backbone.mono:
        right_l1 = None
    else:
        right_l1 = run_frozen_resnet_layer1(
            full_backbone.feature_backbone, batch_dict["right_img"]
        )
    return {"left_l1": left_l1, "right_l1": right_l1}


def run_full_resnet_tail(resnet, layer1_feature):
    x = layer1_feature
    outputs = [x]
    for name in ("layer2", "layer3", "layer4"):
        x = getattr(resnet, name)(x)
        outputs.append(x)
    return tuple(outputs)


def extract_full_2d_from_layer1(full_backbone, batch_dict, prefix_cache):
    left_backbone = run_full_resnet_tail(
        full_backbone.feature_backbone, prefix_cache["left_l1"]
    )
    left_stereo, left_sem = full_backbone.feature_neck(
        [batch_dict["left_img"]] + list(left_backbone)
    )

    cache = {
        "left_backbone": left_backbone,
        "left_stereo": left_stereo,
        "left_sem": left_sem,
    }

    if full_backbone.mono:
        cache.update(
            {
                "right_backbone": None,
                "right_stereo": None,
                "right_sem": None,
            }
        )
    else:
        right_backbone = run_full_resnet_tail(
            full_backbone.feature_backbone, prefix_cache["right_l1"]
        )
        right_stereo, right_sem = full_backbone.feature_neck(
            [batch_dict["right_img"]] + list(right_backbone)
        )
        cache.update(
            {
                "right_backbone": right_backbone,
                "right_stereo": right_stereo,
                "right_sem": right_sem,
            }
        )
    return cache


@contextlib.contextmanager
def cached_full_2d_prefix(full_backbone, cache):
    """Replay already-computed original ResNet/FPN outputs into Full 3D path."""
    original_backbone_forward = full_backbone.feature_backbone.forward
    original_neck_forward = full_backbone.feature_neck.forward

    backbone_queue = [cache["left_backbone"]]
    neck_queue = [(cache["left_stereo"], cache["left_sem"])]
    if not full_backbone.mono:
        backbone_queue.append(cache["right_backbone"])
        neck_queue.append((cache["right_stereo"], cache["right_sem"]))

    def cached_backbone_forward(_image):
        if not backbone_queue:
            raise RuntimeError("Full ResNet cache underflow")
        return backbone_queue.pop(0)

    def cached_neck_forward(_features, *args, **kwargs):
        del args, kwargs
        if not neck_queue:
            raise RuntimeError("Full FPN cache underflow")
        return neck_queue.pop(0)

    full_backbone.feature_backbone.forward = cached_backbone_forward
    full_backbone.feature_neck.forward = cached_neck_forward
    try:
        yield
    finally:
        full_backbone.feature_backbone.forward = original_backbone_forward
        full_backbone.feature_neck.forward = original_neck_forward


# ============================================================================
# Elastic ResNet layer2/3/4
# ============================================================================


class ElasticBasicBlock(nn.Module):
    def __init__(self, max_in, max_out, stride=1, dilation=1):
        super().__init__()
        self.max_in = int(max_in)
        self.max_out = int(max_out)
        self.stride = int(stride)
        self.conv1 = SlimConv2d(
            self.max_in,
            self.max_out,
            3,
            stride=self.stride,
            padding=int(dilation),
            dilation=int(dilation),
            norm=True,
            activation=True,
        )
        # Original modified ResNet BasicBlock uses padding=1 on conv2.
        self.conv2 = SlimConv2d(
            self.max_out,
            self.max_out,
            3,
            stride=1,
            padding=1,
            dilation=1,
            norm=True,
            activation=False,
        )
        self.shortcut = SlimConv2d(
            self.max_in,
            self.max_out,
            1,
            stride=self.stride,
            padding=0,
            norm=True,
            activation=False,
        )
        self._init_shortcut_identity()

    @torch.no_grad()
    def _init_shortcut_identity(self):
        self.shortcut.weight.zero_()
        diagonal = min(self.max_in, self.max_out)
        for i in range(diagonal):
            self.shortcut.weight[i, i, 0, 0] = 1.0
        for ratio in WIDTH_CHOICES:
            bn = self.shortcut.norm.bns[_ratio_key(ratio)]
            bn.weight.fill_(1.0)
            bn.bias.zero_()
            bn.running_mean.zero_()
            bn.running_var.fill_(1.0)

    def forward(self, x, ratio):
        identity = x
        out = self.conv1(x, ratio)
        out = self.conv2(out, ratio)
        out_c = ratio_to_channels(self.max_out, ratio)
        if self.stride != 1 or int(identity.shape[1]) != out_c:
            identity = self.shortcut(identity, ratio)
        return out + identity

    @torch.no_grad()
    def initialize_from_full(self, source_block):
        self.conv1.initialize_from_full(source_block.conv1, source_block.norm1)
        self.conv2.initialize_from_full(source_block.conv2, source_block.norm2)
        if source_block.downsample is not None:
            src_conv = None
            src_bn = None
            for module in source_block.downsample.modules():
                if module is source_block.downsample:
                    continue
                if src_conv is None and isinstance(module, nn.Conv2d):
                    src_conv = module
                elif src_bn is None and isinstance(module, nn.BatchNorm2d):
                    src_bn = module
            if src_conv is not None:
                self.shortcut.initialize_from_full(src_conv, src_bn)


class ElasticResStage(nn.Module):
    def __init__(self, max_in, max_out, stride, dilation, num_blocks=2):
        super().__init__()
        blocks = [ElasticBasicBlock(max_in, max_out, stride, dilation)]
        for _ in range(1, int(num_blocks)):
            blocks.append(ElasticBasicBlock(max_out, max_out, 1, dilation))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, ratio):
        for block in self.blocks:
            x = block(x, ratio)
        return x

    @torch.no_grad()
    def initialize_from_full(self, source_stage):
        if len(source_stage) != len(self.blocks):
            raise RuntimeError("ResNet block-count mismatch")
        for dst, src in zip(self.blocks, source_stage):
            dst.initialize_from_full(src)


class ElasticResNetTail(nn.Module):
    def __init__(self, source_resnet):
        super().__init__()
        source_stages = [source_resnet.layer2, source_resnet.layer3, source_resnet.layer4]
        self.max_channels = []
        self.stages = nn.ModuleList()
        max_in = int(source_resnet.layer1[-1].conv2.out_channels)
        for stage in source_stages:
            first = stage[0]
            max_out = int(first.conv2.out_channels)
            self.max_channels.append(max_out)
            self.stages.append(
                ElasticResStage(
                    max_in=max_in,
                    max_out=max_out,
                    stride=int(first.stride),
                    dilation=int(first.dilation),
                    num_blocks=len(stage),
                )
            )
            max_in = max_out
        self.initialize_from_full(source_resnet)

    @torch.no_grad()
    def initialize_from_full(self, source_resnet):
        for dst, src in zip(
            self.stages, [source_resnet.layer2, source_resnet.layer3, source_resnet.layer4]
        ):
            dst.initialize_from_full(src)

    def stage2(self, x, ratio):
        return self.stages[0](x, ratio)

    def stage3(self, x, ratio):
        return self.stages[1](x, ratio)

    def stage4(self, x, ratio):
        return self.stages[2](x, ratio)


# ============================================================================
# Elastic FPN / feature_extraction_neck
# ============================================================================


class ElasticFPN(nn.Module):
    def __init__(self, source_neck, res_max_channels):
        super().__init__()
        if source_neck.upconv_type != "fpn":
            raise RuntimeError("Elastic v2 currently requires upconv_type='fpn'")
        if not source_neck.with_upconv:
            raise RuntimeError("Elastic v2 requires with_upconv=True")
        if source_neck.start_level != 2:
            raise RuntimeError("Elastic v2 assumes feature_neck.start_level=2")

        self.res_max_channels = tuple(int(x) for x in res_max_channels)
        self.spp_dim = int(source_neck.spp_dim)
        self.spp_pooling_kernel = list(source_neck.spp_pooling_kernel)
        self.up1_max = int(source_neck.up_dims[0])
        self.up2_max = int(source_neck.up_dims[1])
        self.hidden_max = int(source_neck.stereo_dim[0])
        self.output_max = int(source_neck.stereo_dim[1])

        self.spp_convs = nn.ModuleList(
            [
                SlimConv2d(
                    self.res_max_channels[2],
                    self.spp_dim,
                    1,
                    norm=True,
                    activation=True,
                )
                for _ in self.spp_pooling_kernel
            ]
        )

        concat_max = sum(self.res_max_channels) + self.spp_dim * len(self.spp_convs)
        self.concat_max = int(concat_max)
        self.up_stage0 = SlimConv2d(
            self.concat_max,
            self.up1_max,
            1 if source_neck.kernel1 else 3,
            padding=0 if source_neck.kernel1 else 1,
            norm=True,
            activation=False,
        )
        self.redir0 = SlimConv2d(
            int(source_neck.in_dims[1]),
            self.up1_max,
            3,
            padding=1,
            norm=True,
            activation=False,
        )
        self.up_stage1 = SlimConv2d(
            self.up1_max,
            self.up2_max,
            3,
            padding=1,
            norm=True,
            activation=False,
        )
        self.redir1 = SlimConv2d(
            int(source_neck.in_dims[0]),
            self.up2_max,
            3,
            padding=1,
            norm=True,
            activation=False,
        )
        self.last_hidden = SlimConv2d(
            self.up2_max,
            self.hidden_max,
            3,
            padding=1,
            norm=True,
            activation=True,
        )
        self.last_output = SlimConv2d(
            self.hidden_max,
            self.output_max,
            1,
            padding=0,
            norm=False,
            activation=False,
        )

        self.initialize_from_full(source_neck)

    @torch.no_grad()
    def initialize_from_full(self, source_neck):
        for dst, src in zip(self.spp_convs, source_neck.spp_branches):
            dst.initialize_from_full(src[1][0], src[1][1])

        self.up_stage0.initialize_from_full(
            source_neck.upconv_module.conv[0][0],
            source_neck.upconv_module.conv[0][1],
        )
        self.redir0.initialize_from_full(
            source_neck.upconv_module.redir[0][0],
            source_neck.upconv_module.redir[0][1],
        )
        self.up_stage1.initialize_from_full(
            source_neck.upconv_module.conv[1][0],
            source_neck.upconv_module.conv[1][1],
        )
        self.redir1.initialize_from_full(
            source_neck.upconv_module.redir[1][0],
            source_neck.upconv_module.redir[1][1],
        )
        self.last_hidden.initialize_from_full(
            source_neck.lastconv[0][0], source_neck.lastconv[0][1]
        )
        self.last_output.initialize_from_full(source_neck.lastconv[2], None)

    def _concat_input_indices(self, c2, c3, c4, cspp):
        offsets = [0]
        offsets.append(offsets[-1] + self.res_max_channels[0])
        offsets.append(offsets[-1] + self.res_max_channels[1])
        offsets.append(offsets[-1] + self.res_max_channels[2])
        for _ in self.spp_convs:
            offsets.append(offsets[-1] + self.spp_dim)

        indices = []
        indices.extend(range(offsets[0], offsets[0] + c2))
        indices.extend(range(offsets[1], offsets[1] + c3))
        indices.extend(range(offsets[2], offsets[2] + c4))
        for branch_idx in range(len(self.spp_convs)):
            base = offsets[3 + branch_idx]
            indices.extend(range(base, base + cspp))
        return indices

    def forward(self, image, layer1, layer2, layer3, layer4, ratio):
        feat_shape = tuple(layer2.shape[2:])
        cspp = ratio_to_channels(self.spp_dim, ratio)
        spp = []
        for kernel, conv in zip(self.spp_pooling_kernel, self.spp_convs):
            pooled = F.avg_pool2d(layer4, kernel_size=kernel, stride=kernel)
            y = conv(pooled, ratio)
            y = F.interpolate(y, feat_shape, mode="bilinear", align_corners=True)
            spp.append(y)

        packed = torch.cat([layer2, layer3, layer4] + spp, dim=1)
        indices = self._concat_input_indices(
            int(layer2.shape[1]),
            int(layer3.shape[1]),
            int(layer4.shape[1]),
            cspp,
        )
        x = self.up_stage0(packed, ratio, input_indices=indices)
        redir = self.redir0(layer1, ratio)
        x = F.relu(
            F.interpolate(x, scale_factor=2, mode="bilinear") + redir,
            inplace=False,
        )

        x = self.up_stage1(x, ratio)
        redir = self.redir1(image, ratio)
        x = F.relu(
            F.interpolate(x, scale_factor=2, mode="bilinear") + redir,
            inplace=False,
        )
        x = self.last_hidden(x, ratio)
        x = self.last_output(x, ratio)
        return x


# ============================================================================
# Elastic 3D hourglass and BEV projection
# ============================================================================


class ElasticHourglass3D(nn.Module):
    def __init__(self, max_channels):
        super().__init__()
        c = int(max_channels)
        c2 = c * 2
        self.max_channels = c
        self.conv1 = SlimConv3d(c, c2, 3, stride=2, padding=1, norm=True, activation=True)
        self.conv2 = SlimConv3d(c2, c2, 3, stride=1, padding=1, norm=True, activation=False)
        self.conv3 = SlimConv3d(c2, c2, 3, stride=2, padding=1, norm=True, activation=True)
        self.conv4 = SlimConv3d(c2, c2, 3, stride=1, padding=1, norm=True, activation=True)
        self.conv5 = SlimConvTranspose3d(
            c2, c2, 3, stride=2, padding=1, output_padding=1, norm=True
        )
        self.conv6 = SlimConvTranspose3d(
            c2, c, 3, stride=2, padding=1, output_padding=1, norm=True
        )

    def forward(self, x, ratio):
        # Match the current StreamDSGN RPN3D-hourglass call site exactly.
        # In stream_dsgn2_backbone.py the single hourglass is invoked with
        # presqu=True and postsqu=True, so the original implementation adds
        # scalar 1 at the two skip locations and returns conv6(post) directly.
        out = self.conv1(x, ratio)
        pre = F.relu(self.conv2(out, ratio) + 1.0, inplace=False)
        out = self.conv3(pre, ratio)
        out = self.conv4(out, ratio)
        post = F.relu(self.conv5(out, ratio) + 1.0, inplace=False)
        return self.conv6(post, ratio)

    @torch.no_grad()
    def initialize_from_full(self, source_hg):
        self.conv1.initialize_from_full(source_hg.conv1[0][0], source_hg.conv1[0][1])
        self.conv2.initialize_from_full(source_hg.conv2[0], source_hg.conv2[1])
        self.conv3.initialize_from_full(source_hg.conv3[0][0], source_hg.conv3[0][1])
        self.conv4.initialize_from_full(source_hg.conv4[0][0], source_hg.conv4[0][1])
        self.conv5.initialize_from_full(source_hg.conv5[0], source_hg.conv5[1])
        self.conv6.initialize_from_full(source_hg.conv6[0], source_hg.conv6[1])


class ElasticBEVProjection(nn.Module):
    def __init__(self, max_input_channels, output_channels):
        super().__init__()
        self.max_input_channels = int(max_input_channels)
        self.output_channels = int(output_channels)
        self.weight = nn.Parameter(
            torch.zeros(self.output_channels, self.max_input_channels, 1, 1)
        )
        self.bias = nn.Parameter(torch.zeros(self.output_channels))
        with torch.no_grad():
            diagonal = min(self.output_channels, self.max_input_channels)
            for i in range(diagonal):
                self.weight[i, i, 0, 0] = 1.0

    def forward(self, x):
        in_c = int(x.shape[1])
        return F.conv2d(x, self.weight[:, :in_c], self.bias)


# ============================================================================
# Main elastic BEV branch
# ============================================================================


class ElasticBEVBranch(nn.Module):
    """
    Six width-controlled stages after the frozen original ResNet layer1:

      E1: ResNet layer2
      E2: ResNet layer3
      E3: ResNet layer4
      E4: FPN / feature_extraction_neck
      E5: cost volume + stereo 3D refinement
      E6: 3D geometry/RPN + BEV projection

    The original K3 temporal fusion, VAN and detection head are not included
    here and remain frozen in the base model.
    """

    def __init__(self, full_backbone, output_bev_channels=96):
        super().__init__()
        if full_backbone.mono:
            raise RuntimeError("Elastic v2 requires stereo input")
        if full_backbone.drop_psv:
            raise RuntimeError("Elastic v2 requires PSV/cost-volume path")
        if full_backbone.cat_img_feature or full_backbone.cat_right_img_feature:
            raise RuntimeError(
                "Elastic v2 currently assumes cat_img_feature=False and "
                "cat_right_img_feature=False"
            )
        if len(full_backbone.hg_stereo) != 0:
            raise RuntimeError(
                "Elastic v2 currently targets the current K3 config with num_hg=0"
            )
        if len(full_backbone.rpn3d_hgs) != 1:
            raise RuntimeError(
                "Elastic v2 expects exactly one RPN3D hourglass in the K3 base"
            )

        self.cv_max = int(full_backbone.cv_dim)
        self.rpn_max = int(full_backbone.rpn3d_dim)
        self.output_bev_channels = int(output_bev_channels)
        self.cost_downsample = int(
            full_backbone.build_cost._get_cfg_value(
                full_backbone.build_cost.volume_cfgs[0], "downsample", 4
            )
        )
        self.cost_interval = int(
            full_backbone.build_cost._get_cfg_value(
                full_backbone.build_cost.volume_cfgs[0], "shift", 1
            )
        )

        self.resnet = ElasticResNetTail(full_backbone.feature_backbone)
        self.fpn = ElasticFPN(full_backbone.feature_neck, self.resnet.max_channels)

        self.dres0 = SlimConv3d(
            2 * self.cv_max,
            self.cv_max,
            1,
            padding=0,
            norm=True,
            activation=True,
        )
        self.dres1 = SlimConv3d(
            self.cv_max,
            self.cv_max,
            3,
            padding=1,
            norm=True,
            activation=False,
        )

        self.rpn_in = SlimConv3d(
            self.cv_max,
            self.rpn_max,
            1,
            padding=0,
            norm=True,
            activation=True,
        )
        self.rpn_hg = ElasticHourglass3D(self.rpn_max)
        self.rpn_pool = nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))

        bev_depth = int(full_backbone.coordinates_3d.shape[0]) // 4
        self.bev_depth = bev_depth
        self.bev_projection = ElasticBEVProjection(
            self.rpn_max * self.bev_depth,
            self.output_bev_channels,
        )

        self.initialize_from_full(full_backbone)

    @torch.no_grad()
    def initialize_from_full(self, full_backbone):
        self.dres0.initialize_from_full(
            full_backbone.dres0[0][0], full_backbone.dres0[0][1]
        )
        self.dres1.initialize_from_full(
            full_backbone.dres1[0][0], full_backbone.dres1[0][1]
        )
        self.rpn_in.initialize_from_full(
            full_backbone.rpn3d_convs[0][0][0],
            full_backbone.rpn3d_convs[0][0][1],
        )
        self.rpn_hg.initialize_from_full(full_backbone.rpn3d_hgs[0])

    @staticmethod
    def validate_schedule(schedule):
        return validate_schedule(schedule)

    def stage_res2(self, prefix_cache, ratio):
        return {
            "left_l1": prefix_cache["left_l1"],
            "right_l1": prefix_cache["right_l1"],
            "left_l2": self.resnet.stage2(prefix_cache["left_l1"], ratio),
            "right_l2": self.resnet.stage2(prefix_cache["right_l1"], ratio),
        }

    def stage_res3(self, state, ratio):
        state = dict(state)
        state["left_l3"] = self.resnet.stage3(state["left_l2"], ratio)
        state["right_l3"] = self.resnet.stage3(state["right_l2"], ratio)
        return state

    def stage_res4(self, state, ratio):
        state = dict(state)
        state["left_l4"] = self.resnet.stage4(state["left_l3"], ratio)
        state["right_l4"] = self.resnet.stage4(state["right_l3"], ratio)
        return state

    def stage_fpn(self, batch_dict, state, ratio):
        state = dict(state)
        state["left_stereo"] = self.fpn(
            batch_dict["left_img"],
            state["left_l1"],
            state["left_l2"],
            state["left_l3"],
            state["left_l4"],
            ratio,
        )
        state["right_stereo"] = self.fpn(
            batch_dict["right_img"],
            state["right_l1"],
            state["right_l2"],
            state["right_l3"],
            state["right_l4"],
            ratio,
        )
        return state

    @staticmethod
    def _dynamic_disp_channels(voxel_disps, img_channels, sep, inv_ratio):
        shift_channels = int(img_channels) - int(sep) + 1
        if shift_channels <= 0:
            raise RuntimeError((img_channels, sep, shift_channels))
        x = F.interpolate(
            voxel_disps[None, None], (shift_channels,), mode="linear"
        )[0, 0]
        if inv_ratio > 0.0:
            x = x ** inv_ratio
        denom = (x.max() - x.min()) / shift_channels
        x = x / torch.clamp(denom, min=1e-6)
        x -= x.min()
        x = shift_channels - 1 - x.to(torch.int64).clamp(0, shift_channels - 1)
        return x

    def _build_dynamic_cost(self, left, right, shift, full_backbone, ratio):
        sep = ratio_to_channels(self.cv_max, ratio)
        if int(left.shape[1]) < sep:
            raise RuntimeError(
                f"FPN active channels={left.shape[1]} smaller than cost sep={sep}"
            )
        original_dtype = left.dtype
        left_f = left.float() if left.dtype == torch.float16 else left
        right_f = right.float() if right.dtype == torch.float16 else right
        shift_f = shift.float() if shift.dtype == torch.float16 else shift

        if int(left.shape[1]) == sep:
            cost = build_cost_volume(
                left_f, right_f, shift_f, self.cost_downsample
            )
        else:
            psv_channels = self._dynamic_disp_channels(
                shift_f[0],
                int(left.shape[1]),
                sep,
                float(full_backbone.inv_smooth_psv),
            ).to(torch.int32)
            cost = build_dps_cost_volume(
                left_f,
                right_f,
                shift_f,
                psv_channels,
                self.cost_downsample,
                sep,
                self.cost_interval,
            )
        return cost.half() if original_dtype == torch.float16 else cost

    def stage_stereo(self, batch_dict, state, full_backbone, ratio):
        dtype = state["left_stereo"].dtype
        calib = batch_dict["calib"]
        fu_mul_baseline = torch.as_tensor(
            [x.fu_mul_baseline for x in calib],
            dtype=dtype,
            device=state["left_stereo"].device,
        )
        depth = (
            full_backbone._downsampled_depth_fp16
            if dtype == torch.float16
            else full_backbone.downsampled_depth
        ).to(state["left_stereo"].device)
        downsampled_disp = (
            fu_mul_baseline[:, None]
            / depth[None, :]
            / (
                full_backbone.downsample_disp
                if not full_backbone.fullres_stereo_feature
                else 1
            )
        )
        cost = self._build_dynamic_cost(
            state["left_stereo"],
            state["right_stereo"],
            downsampled_disp,
            full_backbone,
            ratio,
        )

        sep = ratio_to_channels(self.cv_max, ratio)
        # Packed reduced cost is [left(0:sep), right(0:sep)]. Map it to the
        # corresponding columns of the Full dres0 kernel [left32,right32].
        input_indices = list(range(0, sep)) + list(
            range(self.cv_max, self.cv_max + sep)
        )
        x = self.dres0(cost, ratio, input_indices=input_indices)
        residual = self.dres1(x, ratio)
        return F.relu(x + residual, inplace=False)

    def _geometry_grid(self, batch_dict, full_backbone, dtype):
        coordinates_3d = (
            full_backbone._coordinates_3d_fp16
            if dtype == torch.float16
            else full_backbone.coordinates_3d
        ).to(batch_dict["left_img"].device)
        batch_dict["coord"] = coordinates_3d

        norm_coord_imgs = []
        valids2d = []
        left = batch_dict["left_img"]
        n = int(batch_dict["batch_size"])
        calib = batch_dict["calib"]

        for i in range(n):
            c3d = coordinates_3d.view(-1, 3)
            if "random_T" in batch_dict:
                random_t = batch_dict["random_T"][i]
                c3d = torch.matmul(c3d, random_t[:3, :3].T) + random_t[:3, 3]
            c3d = project_pseudo_lidar_to_rectcam(c3d)
            coord_img, norm_coord_img = full_backbone.compute_mapping(
                c3d,
                left.shape[2:],
                torch.as_tensor(calib[i].P2, device=left.device, dtype=dtype),
                [full_backbone.CV_DEPTH_MIN, full_backbone.CV_DEPTH_MAX],
                use_amp=(dtype == torch.float16),
            )
            coord_img = coord_img.view(*full_backbone.coordinates_3d.shape[:3], 3)
            norm_coord_img = norm_coord_img.view(
                *full_backbone.coordinates_3d.shape[:3], 3
            )
            norm_coord_imgs.append(norm_coord_img)
            image_shape = batch_dict["image_shape"][i]
            valids2d.append(
                (coord_img[..., 0] >= 0)
                & (coord_img[..., 0] <= image_shape[1])
                & (coord_img[..., 1] >= 0)
                & (coord_img[..., 1] <= image_shape[0])
            )

        norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
        valids2d = torch.stack(valids2d, dim=0)
        valids = (
            valids2d
            & (norm_coord_imgs[..., 2] >= -1.0)
            & (norm_coord_imgs[..., 2] <= 1.0)
        )
        batch_dict["norm_coord_imgs"] = norm_coord_imgs
        batch_dict["valids"] = valids
        return norm_coord_imgs, valids

    def stage_rpn(self, batch_dict, stereo_feature, full_backbone, ratio):
        norm_coord_imgs, valids = self._geometry_grid(
            batch_dict, full_backbone, stereo_feature.dtype
        )
        voxel = F.grid_sample(stereo_feature, norm_coord_imgs, align_corners=True)
        voxel = voxel * valids.float()[:, None].to(voxel.dtype)

        x = self.rpn_in(voxel, ratio)
        x = self.rpn_hg(x, ratio)
        x = self.rpn_pool(x)
        n, c, d, h, w = x.shape
        if d != self.bev_depth:
            raise RuntimeError(
                f"Expected pooled depth={self.bev_depth}, got {d}, volume={tuple(x.shape)}"
            )
        bev = x.reshape(n, c * d, h, w)
        bev = self.bev_projection(bev)
        return bev, valids

    def forward_from_prefix(
        self,
        batch_dict,
        prefix_cache,
        full_backbone,
        schedule=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ):
        schedule = validate_schedule(schedule)
        state = self.stage_res2(prefix_cache, schedule[0])
        state = self.stage_res3(state, schedule[1])
        state = self.stage_res4(state, schedule[2])
        state = self.stage_fpn(batch_dict, state, schedule[3])
        stereo = self.stage_stereo(batch_dict, state, full_backbone, schedule[4])
        return self.stage_rpn(batch_dict, stereo, full_backbone, schedule[5])

    def forward(
        self,
        batch_dict,
        prefix_cache,
        full_backbone,
        schedule=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ):
        return self.forward_from_prefix(
            batch_dict, prefix_cache, full_backbone, schedule
        )
PY_ELASTIC_V2

cat > tools/train_elastic_bev.py <<'PY_TRAIN_V2'
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
PY_TRAIN_V2

cat > tools/test_elastic_stream.py <<'PY_TEST_V2'
#!/usr/bin/env python3

import argparse
import copy
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils

from pcdet.models.backbones_3d_stream.elastic_bev_branch import (
    ElasticBEVBranch,
    MONOTONIC_SCHEDULES,
    WIDTH_CHOICES,
    cached_full_2d_prefix,
    extract_fixed_layer1_prefix,
    extract_full_2d_from_layer1,
    validate_schedule,
)

from test_stream_buffer_timestamp import (
    attach_timestamp,
    build_scene_index,
    frame_token,
    load_one,
    strip_eval_metadata,
    timestamp_align_scene,
)


torch.backends.cudnn.benchmark = True

STAGE_NAMES = ("res2", "res3", "res4", "fpn", "stereo", "rpn")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Timestamp-aligned capacity-1 streaming evaluation for Elastic-v2: "
            "frozen ResNet stem+layer1 timing probe, elastic ResNet layer2-4, "
            "elastic FPN/stereo/RPN, frozen K3 fusion/head."
        )
    )
    parser.add_argument("--full_cfg", required=True)
    parser.add_argument("--full_ckpt", required=True)
    parser.add_argument("--elastic_ckpt", required=True)
    parser.add_argument("--input_hz", type=float, default=10.0)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--profile_runs", type=int, default=1)
    parser.add_argument("--safety", type=float, default=1.12)
    parser.add_argument("--deadline_periods", type=float, default=1.0)
    parser.add_argument(
        "--mode",
        choices=("dynamic", "baseline_full", "elastic_fixed"),
        default="dynamic",
    )
    parser.add_argument(
        "--fixed_schedule",
        type=str,
        default="0.5,0.5,0.5,0.5,0.5,0.5",
        help=(
            "Six ratios: Res2,Res3,Res4,FPN,Stereo,RPN. "
            "Example 1,0.75,0.75,0.5,0.5,0.25"
        ),
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_cfg(path):
    cfg_from_yaml_file(path, cfg)
    cfg.TAG = Path(path).stem
    cfg.DATA_CONFIG.INFER_TIME_PATH = None
    if hasattr(cfg.MODEL, "BACKBONE_3D"):
        cfg.MODEL.BACKBONE_3D.feature_backbone_pretrained = None
    return cfg


def parse_schedule(text):
    return validate_schedule(tuple(float(x.strip()) for x in text.split(",")))


def sync_time_call(fn):
    torch.cuda.synchronize()
    start = time.perf_counter_ns()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter_ns() - start) / 1e6


def run_elastic_downstream(base_model, batch_dict, bev, valids):
    cur_data = batch_dict["token"]
    cur_data["spatial_features"] = bev
    cur_data["spatial_features_stride"] = 1
    cur_data["valids"] = valids
    cur_data["history_features"] = base_model.history_feature_queue

    history_feature = {"spatial_features": bev.detach().clone()}

    for module in base_model.fusion_module:
        cur_data = module(cur_data)
    for module in base_model.after_fusion_blocks:
        cur_data = module(cur_data)

    pred_dicts, ret_dicts = base_model.post_processing(cur_data)
    if base_model.history_feature_queue is not None:
        base_model.history_feature_queue.append(
            (cur_data["this_sample_idx"], history_feature)
        )
    return pred_dicts, ret_dicts


def fill_three_history(queue, bev):
    if queue is None:
        return
    queue.clear()
    for i in range(3):
        queue.append((f"profile_{i}", {"spatial_features": bev.detach().clone()}))


def full_bev_after_prefix(base_model, frame_dict, prefix_cache):
    backbone = base_model.backbone_3d
    full_2d = extract_full_2d_from_layer1(backbone, frame_dict, prefix_cache)
    data = copy.copy(frame_dict)
    with cached_full_2d_prefix(backbone, full_2d):
        data = backbone(data)
    data = base_model.map_to_bev_module(data)
    return data["spatial_features"], full_2d


def run_full_after_prefix(base_model, batch_dict, prefix_cache):
    backbone = base_model.backbone_3d
    full_2d = extract_full_2d_from_layer1(
        backbone, batch_dict["token"], prefix_cache
    )
    with cached_full_2d_prefix(backbone, full_2d):
        return base_model(batch_dict)


def _key_res2(schedule):
    return str(schedule[0])


def _key_res3(schedule):
    return f"{schedule[0]}>{schedule[1]}"


def _key_res4(schedule):
    return f"{schedule[1]}>{schedule[2]}"


def _key_fpn(schedule):
    return ">".join(str(x) for x in schedule[:4])


def _key_stereo(schedule):
    return f"{schedule[3]}>{schedule[4]}"


def _key_rpn(schedule):
    return f"{schedule[4]}>{schedule[5]}"


def profile_runtime(base_model, branch, dataset, dataset_index, warmup, runs):
    backbone = base_model.backbone_3d
    amp_enabled = bool(base_model.use_amp_dict["TEST"])
    profile_schedule = (0.5,) * 6

    # Warmup both original Full and Elastic.
    for _ in range(max(1, warmup)):
        if base_model.history_feature_queue is not None:
            base_model.history_feature_queue.clear()
        batch = load_one(dataset, dataset_index)
        with torch.no_grad():
            base_model(batch)

        if base_model.history_feature_queue is not None:
            base_model.history_feature_queue.clear()
        batch = load_one(dataset, dataset_index)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            prefix = extract_fixed_layer1_prefix(backbone, batch["token"])
            bev, valids = branch(
                batch["token"], prefix, backbone, profile_schedule
            )
            fill_three_history(base_model.history_feature_queue, bev)
            run_elastic_downstream(base_model, batch, bev, valids)
    torch.cuda.synchronize()

    prefix_samples = []
    full_remaining_samples = []
    post_samples = []

    # Full remaining time after the fixed stem+layer1 timing prefix.
    for _ in range(max(2, runs * 2)):
        if base_model.history_feature_queue is not None:
            base_model.history_feature_queue.clear()
        batch = load_one(dataset, dataset_index)

        def prefix_call():
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                return extract_fixed_layer1_prefix(backbone, batch["token"])

        prefix, prefix_ms = sync_time_call(prefix_call)
        prefix_samples.append(prefix_ms)

        # Populate three history entries so downstream timing is representative.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            bev, _ = full_bev_after_prefix(base_model, batch["token"], prefix)
        fill_three_history(base_model.history_feature_queue, bev)

        def full_remaining_call():
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                return run_full_after_prefix(base_model, batch, prefix)

        _, remain_ms = sync_time_call(full_remaining_call)
        full_remaining_samples.append(remain_ms)

    samples = {
        "res2": defaultdict(list),
        "res3": defaultdict(list),
        "res4": defaultdict(list),
        "fpn": defaultdict(list),
        "stereo": defaultdict(list),
        "rpn": defaultdict(list),
    }

    # One pass over all 84 monotonic schedules already gives repeated samples
    # for the cheaper stage keys. profile_runs>1 repeats the whole table.
    for _ in range(max(1, runs)):
        for schedule in MONOTONIC_SCHEDULES:
            if base_model.history_feature_queue is not None:
                base_model.history_feature_queue.clear()
            batch = load_one(dataset, dataset_index)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                prefix = extract_fixed_layer1_prefix(backbone, batch["token"])

                state, t = sync_time_call(
                    lambda: branch.stage_res2(prefix, schedule[0])
                )
                samples["res2"][_key_res2(schedule)].append(t)

                state, t = sync_time_call(
                    lambda: branch.stage_res3(state, schedule[1])
                )
                samples["res3"][_key_res3(schedule)].append(t)

                state, t = sync_time_call(
                    lambda: branch.stage_res4(state, schedule[2])
                )
                samples["res4"][_key_res4(schedule)].append(t)

                state, t = sync_time_call(
                    lambda: branch.stage_fpn(batch["token"], state, schedule[3])
                )
                samples["fpn"][_key_fpn(schedule)].append(t)

                stereo, t = sync_time_call(
                    lambda: branch.stage_stereo(
                        batch["token"], state, backbone, schedule[4]
                    )
                )
                samples["stereo"][_key_stereo(schedule)].append(t)

                (bev, valids), t = sync_time_call(
                    lambda: branch.stage_rpn(
                        batch["token"], stereo, backbone, schedule[5]
                    )
                )
                samples["rpn"][_key_rpn(schedule)].append(t)

                fill_three_history(base_model.history_feature_queue, bev)
                _, post_ms = sync_time_call(
                    lambda: run_elastic_downstream(
                        base_model, batch, bev, valids
                    )
                )
                post_samples.append(post_ms)

    profile = {
        "fixed_prefix_ms": float(np.median(prefix_samples)),
        "full_remaining_ms": float(np.median(full_remaining_samples)),
        "post_ms": float(np.median(post_samples)),
    }
    for stage in STAGE_NAMES:
        profile[f"{stage}_ms"] = {
            key: float(np.median(values))
            for key, values in samples[stage].items()
        }

    if base_model.history_feature_queue is not None:
        base_model.history_feature_queue.clear()
    return profile


def stage_reference(profile, stage_index, schedule):
    if stage_index == 0:
        return profile["res2_ms"][_key_res2(schedule)]
    if stage_index == 1:
        return profile["res3_ms"][_key_res3(schedule)]
    if stage_index == 2:
        return profile["res4_ms"][_key_res4(schedule)]
    if stage_index == 3:
        return profile["fpn_ms"][_key_fpn(schedule)]
    if stage_index == 4:
        return profile["stereo_ms"][_key_stereo(schedule)]
    if stage_index == 5:
        return profile["rpn_ms"][_key_rpn(schedule)]
    raise IndexError(stage_index)


def remaining_reference(profile, schedule, next_stage_index):
    total = 0.0
    for stage_index in range(next_stage_index, 6):
        total += stage_reference(profile, stage_index, schedule)
    total += profile["post_ms"]
    return total


def schedule_score(schedule):
    # Earlier stages influence a larger fraction of the representation, so
    # ties prefer keeping them wide. The lexicographic suffix makes selection
    # deterministic.
    weighted = sum((6 - i) * r for i, r in enumerate(schedule))
    return (weighted, *schedule)


class ElasticRuntimeModel(nn.Module):
    def __init__(
        self,
        base_model,
        branch,
        profile,
        deadline_ms,
        safety,
        mode,
        fixed_schedule,
    ):
        super().__init__()
        self.base_model = base_model
        self.branch = branch
        self.profile = profile
        self.deadline_ms = float(deadline_ms)
        self.safety = float(safety)
        self.mode = mode
        self.fixed_schedule = validate_schedule(fixed_schedule)
        self.initial_age_ms = 0.0
        self.last_runtime_meta = {}

    @property
    def history_feature_queue(self):
        return self.base_model.history_feature_queue

    def effective_deadline(self):
        return max(self.deadline_ms - float(self.initial_age_ms), 0.0)

    def _choose_schedule(self, fixed_prefix, elapsed_ms, rho, next_stage):
        feasible = []
        for schedule in MONOTONIC_SCHEDULES:
            if any(
                schedule[i] != fixed_prefix[i]
                for i in range(len(fixed_prefix))
            ):
                continue
            estimate = elapsed_ms + self.safety * max(rho, 1.0) * remaining_reference(
                self.profile, schedule, next_stage
            )
            if estimate <= self.effective_deadline():
                feasible.append(schedule)
        if feasible:
            return max(feasible, key=schedule_score)

        # Deadline cannot be met even by the profiled minimum suffix. Preserve
        # already executed ratios and immediately minimize all remaining stages.
        schedule = list(fixed_prefix)
        previous = schedule[-1] if schedule else 1.0
        while len(schedule) < 6:
            next_ratio = min(previous, 0.25)
            schedule.append(next_ratio)
            previous = next_ratio
        return tuple(schedule)

    @staticmethod
    def _update_rho(old_rho, actual_ms, reference_ms):
        observed = actual_ms / max(reference_ms, 1e-6)
        # Smooth noisy per-stage measurements but never assume the GPU is
        # faster than its nominal profile when making a deadline decision.
        return max(1.0, 0.5 * float(old_rho) + 0.5 * float(observed))

    def _run_elastic(self, batch_dict, prefix_cache, prefix_ms):
        backbone = self.base_model.backbone_3d
        amp_enabled = bool(self.base_model.use_amp_dict["TEST"])
        elapsed = float(prefix_ms)
        rho = max(
            1.0,
            prefix_ms / max(self.profile["fixed_prefix_ms"], 1e-6),
        )

        if self.mode == "elastic_fixed":
            schedule = self.fixed_schedule
        else:
            schedule = self._choose_schedule((), elapsed, rho, 0)

        stage_times = []
        chosen = list(schedule)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            state, t = sync_time_call(
                lambda: self.branch.stage_res2(prefix_cache, chosen[0])
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["res2_ms"][_key_res2(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:1]), elapsed, rho, 1))

            state, t = sync_time_call(
                lambda: self.branch.stage_res3(state, chosen[1])
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["res3_ms"][_key_res3(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:2]), elapsed, rho, 2))

            state, t = sync_time_call(
                lambda: self.branch.stage_res4(state, chosen[2])
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["res4_ms"][_key_res4(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:3]), elapsed, rho, 3))

            state, t = sync_time_call(
                lambda: self.branch.stage_fpn(
                    batch_dict["token"], state, chosen[3]
                )
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["fpn_ms"][_key_fpn(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:4]), elapsed, rho, 4))

            stereo, t = sync_time_call(
                lambda: self.branch.stage_stereo(
                    batch_dict["token"], state, backbone, chosen[4]
                )
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["stereo_ms"][_key_stereo(tuple(chosen))]
            )
            if self.mode == "dynamic":
                chosen = list(self._choose_schedule(tuple(chosen[:5]), elapsed, rho, 5))

            (bev, valids), t = sync_time_call(
                lambda: self.branch.stage_rpn(
                    batch_dict["token"], stereo, backbone, chosen[5]
                )
            )
            stage_times.append(t)
            elapsed += t
            rho = self._update_rho(
                rho, t, self.profile["rpn_ms"][_key_rpn(tuple(chosen))]
            )

            pred_dicts, ret_dicts = run_elastic_downstream(
                self.base_model, batch_dict, bev, valids
            )

        self.last_runtime_meta = {
            "branch": "elastic",
            "schedule": [float(x) for x in chosen],
            "fixed_prefix_ms": float(prefix_ms),
            "stage_ms": {
                name: float(value) for name, value in zip(STAGE_NAMES, stage_times)
            },
            "slowdown": float(rho),
            "initial_age_ms": float(self.initial_age_ms),
            "effective_deadline_ms": float(self.effective_deadline()),
        }
        return pred_dicts, ret_dicts

    def forward(self, batch_dict):
        cur_data = batch_dict["token"]
        if (
            self.base_model.history_feature_queue is not None
            and ("prev_sample_idx" not in cur_data or cur_data["prev_sample_idx"] == "")
        ):
            self.base_model.history_feature_queue.clear()

        if self.mode == "baseline_full":
            with torch.no_grad():
                pred_dicts, ret_dicts = self.base_model(batch_dict)
            self.last_runtime_meta = {
                "branch": "baseline_full",
                "schedule": None,
                "initial_age_ms": float(self.initial_age_ms),
            }
            return pred_dicts, ret_dicts

        backbone = self.base_model.backbone_3d
        amp_enabled = bool(self.base_model.use_amp_dict["TEST"])
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
            prefix_cache, prefix_ms = sync_time_call(
                lambda: extract_fixed_layer1_prefix(backbone, cur_data)
            )

        if self.mode == "dynamic":
            rho = max(
                1.0,
                prefix_ms / max(self.profile["fixed_prefix_ms"], 1e-6),
            )
            estimated_full = (
                prefix_ms
                + self.safety * rho * self.profile["full_remaining_ms"]
            )
            if estimated_full <= self.effective_deadline():
                with torch.no_grad(), torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred_dicts, ret_dicts = run_full_after_prefix(
                        self.base_model, batch_dict, prefix_cache
                    )
                self.last_runtime_meta = {
                    "branch": "full",
                    "schedule": None,
                    "fixed_prefix_ms": float(prefix_ms),
                    "estimated_full_ms": float(estimated_full),
                    "slowdown": float(rho),
                    "initial_age_ms": float(self.initial_age_ms),
                    "effective_deadline_ms": float(self.effective_deadline()),
                }
                return pred_dicts, ret_dicts

        return self._run_elastic(batch_dict, prefix_cache, prefix_ms)


def run_one_prediction(runtime_model, dataset, dataset_index, initial_age_ms=0.0):
    runtime_model.initial_age_ms = float(initial_age_ms)
    batch_dict = load_one(dataset, dataset_index)
    torch.cuda.synchronize()
    start_ns = time.perf_counter_ns()
    with torch.no_grad():
        pred_dicts, _ = runtime_model(batch_dict)
    torch.cuda.synchronize()
    finish_ns = time.perf_counter_ns()

    service_ms = (finish_ns - start_ns) / 1e6
    wall_finish_ns = time.time_ns()
    annos = dataset.generate_prediction_dicts(
        batch_dict, pred_dicts, dataset.class_names, output_path=None
    )
    if len(annos) != 1:
        raise RuntimeError("Streaming evaluator requires batch_size=1")
    return (
        annos[0],
        service_ms,
        wall_finish_ns,
        copy.deepcopy(runtime_model.last_runtime_meta),
    )


def run_scene(runtime_model, dataset, scene, indices, period_ms, trace, processed_predictions):
    if runtime_model.history_feature_queue is not None:
        runtime_model.history_feature_queue.clear()
    n = len(indices)
    if n == 0:
        return [], 0, 0

    pos = 0
    virtual_now_ms = 0.0
    processed_count = 0
    dropped_count = 0
    scene_outputs = []

    while pos < n:
        dataset_index = indices[pos]
        input_frame_id = frame_token(dataset, dataset_index)
        arrival_ms = pos * period_ms
        start_ms = max(virtual_now_ms, arrival_ms)

        anno, service_ms, wall_finish_ns, runtime_meta = run_one_prediction(
            runtime_model,
            dataset,
            dataset_index,
            initial_age_ms=(start_ms - arrival_ms),
        )
        finish_ms = start_ms + service_ms

        stamped = attach_timestamp(
            anno=anno,
            scene=scene,
            input_frame_id=input_frame_id,
            arrival_ms=arrival_ms,
            start_ms=start_ms,
            finish_ms=finish_ms,
            service_ms=service_ms,
            wall_finish_ns=wall_finish_ns,
        )
        stamped["_stream_runtime_meta"] = runtime_meta
        processed_predictions.append(stamped)
        scene_outputs.append(stamped)
        processed_count += 1

        latest_arrived_pos = pos
        probe = pos + 1
        while probe < n and probe * period_ms <= finish_ms:
            latest_arrived_pos = probe
            probe += 1

        if latest_arrived_pos > pos:
            next_pos = latest_arrived_pos
            dropped_now = max(0, next_pos - pos - 1)
            next_reason = "buffer_latest"
        else:
            next_pos = pos + 1
            dropped_now = 0
            next_reason = "wait_next_arrival"

        dropped_count += dropped_now
        trace.append(
            {
                "scene": scene,
                "dataset_index": int(dataset_index),
                "frame_id": input_frame_id,
                "scene_pos": int(pos),
                "arrival_ms": float(arrival_ms),
                "start_ms": float(start_ms),
                "finish_ms": float(finish_ms),
                "service_ms": float(service_ms),
                "buffer_wait_ms": float(start_ms - arrival_ms),
                "dropped_waiting_frames_after_this_output": int(dropped_now),
                "next_scene_pos": int(next_pos) if next_pos < n else None,
                "next_reason": next_reason if next_pos < n else "scene_end",
                "wall_output_time_ns": int(wall_finish_ns),
                "runtime": runtime_meta,
            }
        )

        print(
            f"[{scene}] frame={input_frame_id} pos={pos:03d}/{n-1:03d} "
            f"service={service_ms:7.3f}ms drop+={dropped_now} "
            f"branch={runtime_meta.get('branch')} "
            f"width={runtime_meta.get('schedule')}"
        )

        virtual_now_ms = finish_ms
        pos = next_pos

    return scene_outputs, processed_count, dropped_count


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)

    full_cfg = make_cfg(args.full_cfg)
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(
        dataset_cfg=full_cfg.DATA_CONFIG,
        class_names=full_cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )
    if len(dataset) == 0:
        raise RuntimeError("Empty test dataset")

    base_model = build_network(
        model_cfg=full_cfg.MODEL,
        num_class=len(full_cfg.CLASS_NAMES),
        dataset=dataset,
    )
    base_model.load_params_from_file(
        filename=args.full_ckpt, logger=logger, to_cpu=True
    )
    base_model.cuda().eval()

    branch = ElasticBEVBranch(
        base_model.backbone_3d,
        output_bev_channels=int(full_cfg.MODEL.MAP_TO_BEV.NUM_BEV_FEATURES),
    ).cuda().eval()
    checkpoint = torch.load(args.elastic_ckpt, map_location="cpu")
    branch.load_state_dict(checkpoint["branch"], strict=True)

    fixed_schedule = parse_schedule(args.fixed_schedule)
    period_ms = 1000.0 / float(args.input_hz)
    deadline_ms = float(args.deadline_periods) * period_ms
    scene_to_indices = build_scene_index(dataset)
    first_scene = next(iter(scene_to_indices))
    first_scene_indices = scene_to_indices[first_scene]
    first_index = first_scene_indices[min(5, len(first_scene_indices) - 1)]

    if args.mode == "dynamic":
        print("Profiling six-stage elastic timing table...")
        profile = profile_runtime(
            base_model,
            branch,
            dataset,
            first_index,
            args.warmup,
            args.profile_runs,
        )
    else:
        profile = {
            "fixed_prefix_ms": 1.0,
            "full_remaining_ms": 1.0,
            "post_ms": 1.0,
            **{f"{stage}_ms": {} for stage in STAGE_NAMES},
        }

    runtime_model = ElasticRuntimeModel(
        base_model=base_model,
        branch=branch,
        profile=profile,
        deadline_ms=deadline_ms,
        safety=args.safety,
        mode=args.mode,
        fixed_schedule=fixed_schedule,
    ).cuda().eval()

    print("=" * 80)
    print("Elastic-v2 K3 streaming evaluation")
    print("fixed prefix      : original ResNet stem + layer1")
    print("elastic stages    : Res2, Res3, Res4, FPN, Stereo3D, RPN3D")
    print(f"mode              : {args.mode}")
    print(f"input_hz          : {args.input_hz}")
    print(f"period_ms         : {period_ms:.3f}")
    print(f"deadline_ms       : {deadline_ms:.3f}")
    print(f"safety            : {args.safety:.3f}")
    print(f"fixed_schedule    : {fixed_schedule}")
    print("=" * 80)

    output_dir = Path(args.output_dir) / full_cfg.TAG / f"{args.input_hz:g}Hz"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "dynamic":
        with open(output_dir / "timing_profile.json", "w") as f:
            json.dump(profile, f, indent=2)

    trace = []
    processed_predictions = []
    outputs_by_scene = {}
    total_processed = 0
    total_dropped = 0

    for scene, indices in scene_to_indices.items():
        outputs, processed, dropped = run_scene(
            runtime_model,
            dataset,
            scene,
            indices,
            period_ms,
            trace,
            processed_predictions,
        )
        outputs_by_scene[scene] = outputs
        total_processed += processed
        total_dropped += dropped

    all_gt_annos = []
    all_stream_det_annos = []
    for scene, indices in scene_to_indices.items():
        gt, det = timestamp_align_scene(
            dataset=dataset,
            scene=scene,
            indices=indices,
            scene_outputs=outputs_by_scene[scene],
            period_ms=period_ms,
        )
        all_gt_annos.extend(gt)
        all_stream_det_annos.extend(det)

    eval_det_annos = [
        strip_eval_metadata(copy.deepcopy(x)) for x in all_stream_det_annos
    ]
    full_result_str, _ = dataset.evaluation_offline(
        all_gt_annos, eval_det_annos, dataset.class_names, "3d"
    )
    paper_result_str = eval_utils.format_paper_metrics(full_result_str)

    with open(output_dir / "timeline.json", "w") as f:
        json.dump(trace, f, indent=2)
    with open(output_dir / "processed_predictions_timestamped.pkl", "wb") as f:
        pickle.dump(processed_predictions, f)
    with open(output_dir / "sap_timestamp_aligned_predictions.pkl", "wb") as f:
        pickle.dump(all_stream_det_annos, f)
    with open(output_dir / "paper_sap.txt", "w") as f:
        f.write(paper_result_str + "\n")

    service = np.asarray([x["service_ms"] for x in trace], dtype=np.float64)
    total_sensor_frames = sum(len(x) for x in scene_to_indices.values())
    branch_counts = defaultdict(int)
    schedule_counts = defaultdict(int)
    deadline_miss = 0
    for x in trace:
        runtime = x["runtime"]
        branch_counts[str(runtime.get("branch"))] += 1
        if runtime.get("schedule") is not None:
            schedule_counts[str(runtime["schedule"])] += 1
        if x["buffer_wait_ms"] + x["service_ms"] > deadline_ms:
            deadline_miss += 1

    summary = {
        "mode": args.mode,
        "full_cfg": args.full_cfg,
        "full_ckpt": args.full_ckpt,
        "elastic_ckpt": args.elastic_ckpt,
        "input_hz": args.input_hz,
        "period_ms": period_ms,
        "deadline_ms": deadline_ms,
        "sensor_frames": total_sensor_frames,
        "processed_frames": total_processed,
        "dropped_frames": total_dropped,
        "process_rate": total_processed / max(total_sensor_frames, 1),
        "drop_rate": total_dropped / max(total_sensor_frames, 1),
        "deadline_miss_count": deadline_miss,
        "deadline_miss_rate": deadline_miss / max(total_processed, 1),
        "mean_service_ms": float(service.mean()) if service.size else None,
        "p50_service_ms": float(np.percentile(service, 50)) if service.size else None,
        "p90_service_ms": float(np.percentile(service, 90)) if service.size else None,
        "p99_service_ms": float(np.percentile(service, 99)) if service.size else None,
        "branch_counts": dict(branch_counts),
        "schedule_counts": dict(schedule_counts),
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 80)
    print(paper_result_str)
    print(json.dumps(summary, indent=2))
    print(f"Saved to: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
PY_TEST_V2

cat > scripts/stream_exp/14_train_elastic.sh <<'SH_TRAIN_V2'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

ELASTIC_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v2}"
ELASTIC_EPOCHS="${ELASTIC_EPOCHS:-20}"
ELASTIC_WORKERS="${ELASTIC_WORKERS:-4}"
ELASTIC_LR="${ELASTIC_LR:-2e-4}"
ELASTIC_MIN_LR="${ELASTIC_MIN_LR:-2e-6}"
ELASTIC_WD="${ELASTIC_WD:-1e-4}"
ELASTIC_BEV_W="${ELASTIC_BEV_W:-2.0}"
ELASTIC_COS_W="${ELASTIC_COS_W:-0.20}"
ELASTIC_AUX_W="${ELASTIC_AUX_W:-0.35}"
ELASTIC_CANONICAL_HISTORY_PROB="${ELASTIC_CANONICAL_HISTORY_PROB:-0.60}"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "tools/train_elastic_bev.py"
require_file "pcdet/models/backbones_3d_stream/elastic_bev_branch.py"

OUT_DIR="outputs/elastic_bev/${ELASTIC_EXP_NAME}"
if compgen -G "${OUT_DIR}/ckpt/checkpoint_epoch_*.pth" > /dev/null; then
    echo "[ERROR] Existing Elastic-v2 checkpoints found: ${OUT_DIR}/ckpt"
    echo "Use another name, e.g. ELASTIC_EXP_NAME=elastic_bev_v2_b $0"
    exit 1
fi

banner "Train Elastic-v2 | Frozen K3 Full | Elastic Res2-4 + FPN + Stereo/RPN"
echo "[BASE CFG]      ${FULL_CFG}"
echo "[BASE CKPT]     ${FULL_CKPT}"
echo "[EXP]           ${ELASTIC_EXP_NAME}"
echo "[EPOCHS]        ${ELASTIC_EPOCHS}"
echo "[LR]            ${ELASTIC_LR} -> ${ELASTIC_MIN_LR}"
echo "[WEIGHT DECAY]  ${ELASTIC_WD}"
echo "[LOSS]          det=1.0 bev=${ELASTIC_BEV_W} cos=${ELASTIC_COS_W} aux=${ELASTIC_AUX_W}"
echo "[FIXED PREFIX]  original ResNet stem + layer1"
echo "[ELASTIC]       ResNet layer2/layer3/layer4 + FPN + Stereo3D + RPN3D"
echo "[WIDTHS]        25%, 50%, 75%, 100%; non-increasing across stages"
echo "[HISTORY]       choose 3 from t-5..t-1; [t-3,t-2,t-1] prob=${ELASTIC_CANONICAL_HISTORY_PROB}"
echo "[BASE]          all existing K3 parameters/buffers frozen and bitwise checked"

python tools/train_elastic_bev.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --epochs "${ELASTIC_EPOCHS}" \
    --workers "${ELASTIC_WORKERS}" \
    --exp_name "${ELASTIC_EXP_NAME}" \
    --lr "${ELASTIC_LR}" \
    --min_lr "${ELASTIC_MIN_LR}" \
    --weight_decay "${ELASTIC_WD}" \
    --det_weight 1.0 \
    --bev_weight "${ELASTIC_BEV_W}" \
    --cos_weight "${ELASTIC_COS_W}" \
    --aux_width_weight "${ELASTIC_AUX_W}" \
    --smooth_l1_beta 0.1 \
    --canonical_history_prob "${ELASTIC_CANONICAL_HISTORY_PROB}" \
    --warmup_ratio 0.05 \
    --grad_clip 5.0 \
    --log_interval 20 \
    --save_interval 1
SH_TRAIN_V2

cat > scripts/stream_exp/15_test_elastic.sh <<'SH_TEST_V2'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_env.sh"

ELASTIC_EXP_NAME="${ELASTIC_EXP_NAME:-elastic_bev_v2}"
EPOCH="${1:-20}"
MODE="${2:-dynamic}"
HZ="${3:-10}"
FIXED_SCHEDULE="${4:-0.5,0.5,0.5,0.5,0.5,0.5}"
PROFILE_RUNS="${PROFILE_RUNS:-1}"
SAFETY="${ELASTIC_SAFETY:-1.12}"
DEADLINE_PERIODS="${ELASTIC_DEADLINE_PERIODS:-1.0}"

ELASTIC_CKPT="outputs/elastic_bev/${ELASTIC_EXP_NAME}/ckpt/checkpoint_epoch_${EPOCH}.pth"
OUT_DIR="outputs/elastic_bev_stream/${ELASTIC_EXP_NAME}_e${EPOCH}/${MODE}"

require_file "${FULL_CFG}"
require_file "${FULL_CKPT}"
require_file "${ELASTIC_CKPT}"
require_file "tools/test_elastic_stream.py"
require_file "pcdet/models/backbones_3d_stream/elastic_bev_branch.py"

banner "Elastic-v2 K3 Streaming Test | ${MODE} | ${HZ} Hz | epoch ${EPOCH}"
echo "[BASE]          ${FULL_CKPT}"
echo "[ELASTIC]       ${ELASTIC_CKPT}"
echo "[MODE]          ${MODE}"
echo "[HZ]            ${HZ}"
echo "[FIXED]         ${FIXED_SCHEDULE}"
echo "[STAGES]        Res2,Res3,Res4,FPN,Stereo,RPN"
echo "[DEADLINE]      ${DEADLINE_PERIODS} sensor period(s)"
echo "[SAFETY]        ${SAFETY}"
echo "[PROFILE RUNS]  ${PROFILE_RUNS}"

python tools/test_elastic_stream.py \
    --full_cfg "${FULL_CFG}" \
    --full_ckpt "${FULL_CKPT}" \
    --elastic_ckpt "${ELASTIC_CKPT}" \
    --input_hz "${HZ}" \
    --mode "${MODE}" \
    --fixed_schedule "${FIXED_SCHEDULE}" \
    --deadline_periods "${DEADLINE_PERIODS}" \
    --safety "${SAFETY}" \
    --warmup 8 \
    --profile_runs "${PROFILE_RUNS}" \
    --workers 0 \
    --output_dir "${OUT_DIR}"
SH_TEST_V2

cat > scripts/stream_exp/16_sweep_elastic.sh <<'SH_SWEEP_V2'
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EPOCH="${1:-20}"
HZ="${2:-10}"

# Exact existing K3 Full baseline.
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" baseline_full "${HZ}"

# Uniform width ablations across all six elastic stages.
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 1,1,1,1,1,1
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 0.75,0.75,0.75,0.75,0.75,0.75
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 0.5,0.5,0.5,0.5,0.5,0.5
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 0.25,0.25,0.25,0.25,0.25,0.25

# Two progressive examples: keep early visual stages wider and compress later.
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 1,0.75,0.75,0.5,0.5,0.25
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" elastic_fixed "${HZ}" 0.75,0.75,0.5,0.5,0.25,0.25

# Runtime timing-aware controller.
"${SCRIPT_DIR}/15_test_elastic.sh" "${EPOCH}" dynamic "${HZ}"
SH_SWEEP_V2

chmod +x     scripts/stream_exp/14_train_elastic.sh     scripts/stream_exp/15_test_elastic.sh     scripts/stream_exp/16_sweep_elastic.sh

python -m py_compile     pcdet/models/backbones_3d_stream/elastic_bev_branch.py     tools/train_elastic_bev.py     tools/test_elastic_stream.py

bash -n scripts/stream_exp/14_train_elastic.sh
bash -n scripts/stream_exp/15_test_elastic.sh
bash -n scripts/stream_exp/16_sweep_elastic.sh

echo
printf '%s
'   '[Elastic-v2] install complete.'   'Train: ./scripts/stream_exp/14_train_elastic.sh'   'Test : ./scripts/stream_exp/15_test_elastic.sh 20 dynamic 10'   'Sweep: ./scripts/stream_exp/16_sweep_elastic.sh 20 10'