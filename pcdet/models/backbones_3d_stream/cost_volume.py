# Stereo cost volume builder.


import torch
from torch import nn

from pcdet.ops.build_cost_volume import build_cost_volume
from pcdet.ops.build_dps_cost_volume import build_dps_cost_volume, build_dps_cost_volume_roi

class BuildCostVolume(nn.Module):
    def __init__(self, volume_cfgs):
        self.volume_cfgs = volume_cfgs
        super(BuildCostVolume, self).__init__()

    def get_dim(self, feature_channel):
        d = 0
        for cfg in self.volume_cfgs:
            volume_type = cfg["type"]
            if volume_type == "concat":
                d += 32 * 2
        return d

    def forward(self, left, right, left_raw, right_raw, shift, psv_disps_channels=None):
        if left.dtype == torch.float16:
            left = left.float()
        if right.dtype == torch.float16:
            right = right.float()
        if shift.dtype == torch.float16:
            shift = shift.float()
            
        volumes = []
        for cfg in self.volume_cfgs:
            volume_type = cfg["type"]

            if volume_type == "concat":
                downsample = getattr(cfg, "downsample", 1)
                if left.shape[1] == 32:
                    volumes.append(build_cost_volume(left, right, shift, downsample))
                else:
                    volumes.append(build_dps_cost_volume(left, right, shift, psv_disps_channels, downsample, 32, getattr(cfg, "shift", 1)))
            else:
                raise NotImplementedError
        if len(volumes) > 1:
            ret_volume =  torch.cat(volumes, dim=1)
        else:
            ret_volume = volumes[0]
        
        if left.dtype == torch.float16:
            ret_volume = ret_volume.half()
        return ret_volume


    def forward_roi(
        self,
        left,
        right,
        left_raw,
        right_raw,
        shift,
        ph0,
        ph1,
        pw0,
        pw1,
        psv_disps_channels=None,
    ):
        """
        Output-space ROI version of forward().

        Current project configuration uses the DPS path
        because stereo feature channels > cv_dim.

        ROI coordinates use the output cost-volume grid,
        i.e. H=80, W=312 in the current configuration.
        """

        if left.dtype == torch.float16:
            left = left.float()

        if right.dtype == torch.float16:
            right = right.float()

        if shift.dtype == torch.float16:
            shift = shift.float()

        volumes = []

        for cfg in self.volume_cfgs:

            volume_type = cfg["type"]

            if volume_type != "concat":
                raise NotImplementedError

            downsample = getattr(
                cfg,
                "downsample",
                1,
            )

            if left.shape[1] == 32:

                raise NotImplementedError(
                    "Current ROI implementation targets "
                    "the DPS cost-volume path used by this "
                    "StreamDSGN configuration. "
                    "Standard 32-channel cost-volume ROI "
                    "can be added separately if needed."
                )

            if psv_disps_channels is None:
                raise RuntimeError(
                    "psv_disps_channels is required "
                    "for DPS ROI cost-volume construction"
                )

            volumes.append(
                build_dps_cost_volume_roi(
                    left,
                    right,
                    shift,
                    psv_disps_channels,
                    downsample,
                    32,
                    getattr(
                        cfg,
                        "shift",
                        1,
                    ),
                    ph0,
                    ph1,
                    pw0,
                    pw1,
                )
            )

        if len(volumes) > 1:
            return torch.cat(
                volumes,
                dim=1,
            )

        return volumes[0]


    def __repr__(self):
        tmpstr = self.__class__.__name__
        return tmpstr
