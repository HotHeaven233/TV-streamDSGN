# Stereo cost volume builder.

import torch
from torch import nn

from pcdet.ops.build_cost_volume import build_cost_volume
from pcdet.ops.build_dps_cost_volume import build_dps_cost_volume


class BuildCostVolume(nn.Module):
    """
    Cost-volume builder with configurable separated stereo channels.

    Original StreamDSGN:
        default_channels = 32
        output channels  = 2 * 32 = 64

    Light-v1:
        default_channels = 16
        output channels  = 2 * 16 = 32

    The CUDA DPS operator already accepts `sep` as a runtime argument,
    so changing 32 -> 16 does not require changing the CUDA kernel.
    """

    def __init__(
        self,
        volume_cfgs,
        default_channels=32,
    ):
        super().__init__()

        self.volume_cfgs = volume_cfgs
        self.default_channels = int(
            default_channels
        )

    def _get_channels(
        self,
        volume_cfg,
    ):
        if isinstance(
            volume_cfg,
            dict,
        ):
            return int(
                volume_cfg.get(
                    "channels",
                    self.default_channels,
                )
            )

        return int(
            getattr(
                volume_cfg,
                "channels",
                self.default_channels,
            )
        )

    @staticmethod
    def _get_cfg_value(
        volume_cfg,
        key,
        default,
    ):
        if isinstance(
            volume_cfg,
            dict,
        ):
            return volume_cfg.get(
                key,
                default,
            )

        return getattr(
            volume_cfg,
            key,
            default,
        )

    def get_dim(
        self,
        feature_channel,
    ):
        del feature_channel

        dim = 0

        for volume_cfg in self.volume_cfgs:

            volume_type = (
                volume_cfg["type"]
                if isinstance(volume_cfg, dict)
                else volume_cfg.type
            )

            if volume_type == "concat":

                sep = self._get_channels(
                    volume_cfg
                )

                dim += (
                    sep * 2
                )

            else:
                raise NotImplementedError(
                    volume_type
                )

        return dim

    def forward(
        self,
        left,
        right,
        left_raw,
        right_raw,
        shift,
        psv_disps_channels=None,
    ):
        # Preserve the numerical behavior of the original implementation.
        if left.dtype == torch.float16:
            left = left.float()

        if right.dtype == torch.float16:
            right = right.float()

        if shift.dtype == torch.float16:
            shift = shift.float()

        volumes = []

        for volume_cfg in self.volume_cfgs:

            volume_type = (
                volume_cfg["type"]
                if isinstance(volume_cfg, dict)
                else volume_cfg.type
            )

            if volume_type != "concat":
                raise NotImplementedError(
                    volume_type
                )

            downsample = int(
                self._get_cfg_value(
                    volume_cfg,
                    "downsample",
                    1,
                )
            )

            sep = self._get_channels(
                volume_cfg
            )

            interval = int(
                self._get_cfg_value(
                    volume_cfg,
                    "shift",
                    1,
                )
            )

            # If input feature itself already has exactly sep channels,
            # use the standard cost-volume implementation.
            if left.shape[1] == sep:

                volume = build_cost_volume(
                    left,
                    right,
                    shift,
                    downsample,
                )

            else:

                if psv_disps_channels is None:
                    raise RuntimeError(
                        "DPS cost volume requires "
                        "psv_disps_channels when "
                        f"feature channels={left.shape[1]} "
                        f"and sep={sep}."
                    )

                volume = build_dps_cost_volume(
                    left,
                    right,
                    shift,
                    psv_disps_channels,
                    downsample,
                    sep,
                    interval,
                )

            volumes.append(
                volume
            )

        if len(volumes) > 1:
            ret_volume = torch.cat(
                volumes,
                dim=1,
            )
        else:
            ret_volume = volumes[0]

        # Keep the same behavior as the original implementation.
        if left.dtype == torch.float16:
            ret_volume = (
                ret_volume.half()
            )

        return ret_volume

    def __repr__(self):
        return self.__class__.__name__
