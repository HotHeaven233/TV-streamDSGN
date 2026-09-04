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

        # IMPORTANT: the RPN3D hourglass changes the voxel-depth length before
        # rpn3d_pool.  In the current K3 model the geometry grid has depth 10,
        # but the hourglass maps 10 -> 12 and AvgPool3d(4, stride=4) maps
        # 12 -> 3.  Therefore coordinates_3d.shape[0] // 4 (=2) is NOT the
        # BEV depth used by the original StreamDSGN path.  The stable interface
        # after HeightCompression is 96 = 32 * 3, so derive the pooled depth
        # from the frozen base model's BEV interface instead.  This also makes
        # the all-100% elastic path use a 96->96 identity-initialized projection.
        if self.output_bev_channels % self.rpn_max != 0:
            raise RuntimeError(
                "output_bev_channels must be divisible by rpn3d_dim: "
                f"{self.output_bev_channels} vs {self.rpn_max}"
            )
        self.bev_depth = self.output_bev_channels // self.rpn_max
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

