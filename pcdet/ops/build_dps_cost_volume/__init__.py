import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from pcdet.ops.build_dps_cost_volume import build_dps_cost_volume_cuda


class _BuildDpsCostVolume(Function):
    @staticmethod
    def forward(ctx, left, right, shift, psv_channels, downsample, sep=32, interval=1):
        ctx.save_for_backward(shift, psv_channels)
        ctx.downsample = downsample
        ctx.channels = left.shape[1]
        ctx.sep = sep
        ctx.interval = interval
        assert torch.all(shift >= 0.)
        output = build_dps_cost_volume_cuda.build_dps_cost_volume_forward(
            left, right, shift, psv_channels, downsample, sep, interval)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        shift, psv_channels = ctx.saved_tensors
        grad_left, grad_right = build_dps_cost_volume_cuda.build_dps_cost_volume_backward(
            grad_output, shift, psv_channels, ctx.downsample, ctx.channels, ctx.sep, ctx.interval)
        return grad_left, grad_right, None, None, None, None, None


build_dps_cost_volume = _BuildDpsCostVolume.apply



def build_dps_cost_volume_roi(
    left,
    right,
    shift,
    psv_channels,
    downsample,
    sep=32,
    interval=1,
    ph0=0,
    ph1=None,
    pw0=0,
    pw1=None,
):
    """
    Output-space ROI forward for DPS cost volume.

    Inference/profiling-only for now.

    left/right remain the FULL feature maps, which preserves
    global stereo disparity correspondence.
    """

    if ph1 is None:
        ph1 = (
            left.shape[2]
            // downsample
        )

    if pw1 is None:
        pw1 = (
            left.shape[3]
            // downsample
        )

    if (
        torch.is_grad_enabled()
        and
        (
            left.requires_grad
            or right.requires_grad
        )
    ):
        raise RuntimeError(
            "build_dps_cost_volume_roi currently "
            "implements forward only. "
            "Use it under torch.no_grad() for profiling/inference."
        )

    # Deliberately avoid:
    #
    #   assert torch.all(shift >= 0)
    #
    # in this hot ROI path, because it introduces a
    # GPU->CPU synchronization. The original model construction
    # already guarantees non-negative disparity shifts.

    return (
        build_dps_cost_volume_cuda
        .build_dps_cost_volume_forward_roi(
            left,
            right,
            shift,
            psv_channels,
            int(downsample),
            int(sep),
            int(interval),
            int(ph0),
            int(ph1),
            int(pw0),
            int(pw1),
        )
    )
