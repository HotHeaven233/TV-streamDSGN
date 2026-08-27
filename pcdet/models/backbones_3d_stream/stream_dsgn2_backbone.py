# DSGN++ backbone (fv+tv)
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import numpy as np

from mmdet.models.builder import build_backbone, build_neck
from . import submodule
from .submodule import convbn_3d, convbn, feature_extraction_neck
from .cost_volume import BuildCostVolume
from pcdet.ops.build_geometry_volume import build_geometry_volume
from pcdet.ops.build_dps_geometry_volume import build_dps_geometry_volume
from pcdet.utils.torch_utils import *


class StreamDSGN2Backbone(nn.Module):
    def __init__(self, model_cfg, class_names, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.use_amp_dict = kwargs.get('use_amp_dict', {'TRAIN':False, 'TEST':False})
        self.use_amp = self.use_amp_dict['TRAIN'] if self.training else self.use_amp_dict['TEST']
        # general config
        self.class_names = class_names
        self.GN = model_cfg.GN
        self.fullres_stereo_feature = model_cfg.feature_neck.with_upconv

        # stereo config
        self.mono = getattr(model_cfg, 'mono', False)
        self.maxdisp = model_cfg.maxdisp
        self.downsample_disp = model_cfg.downsample_disp
        self.voxel_occupancy_downsample_disp = getattr(model_cfg, 'voxel_occupancy_downsample_disp', self.downsample_disp)
        self.downsampled_depth_offset = model_cfg.downsampled_depth_offset
        self.num_hg = getattr(model_cfg, 'num_hg', 1)
        self.use_stereo_out_type = getattr(model_cfg, 'use_stereo_out_type', 'feature')
        self.sup_geometry = getattr(model_cfg, 'sup_geometry', 'volume')
        assert self.sup_geometry in ['volume', 'pooledvolume']
        assert self.use_stereo_out_type in ["feature", "cost", "prob"]

        # volume construction config
        self.cat_img_feature = model_cfg.cat_img_feature
        self.cat_right_img_feature = getattr(model_cfg, 'cat_right_img_feature', False)
        self.rpn3d_dim = model_cfg.rpn3d_dim
        self.voxel_occupancy = getattr(model_cfg, 'voxel_occupancy', False)
        self.voxel_pred_convs = getattr(model_cfg, 'voxel_pred_convs', 1)
        self.voxel_pred_hgs = getattr(model_cfg, 'voxel_pred_hgs', 0)
        self.voxel_occupancy_upsample = getattr(model_cfg, 'voxel_occupancy_upsample', (2,2,2))
        self.drop_psv = getattr(model_cfg, 'drop_psv', False)
        self.geometry_volume_shift = getattr(model_cfg, 'geometry_volume_shift', 1)

        self.inv_smooth_psv = getattr(model_cfg, 'inv_smooth_psv', -1)
        self.inv_smooth_geo = getattr(model_cfg, 'inv_smooth_geo', -1)
        self.drop_psv_loss = getattr(model_cfg, 'drop_psv_loss', False)
        self.squeeze_geo = getattr(model_cfg, 'squeeze_geo', False)

        # volume config
        self.num_3dconvs = model_cfg.num_3dconvs
        self.num_3dconvs_hg = getattr(model_cfg, 'num_3dconvs_hg', 0)
        self.cv_dim = model_cfg.cv_dim

        # feature extraction
        self.feature_backbone = build_backbone(model_cfg.feature_backbone)
    
        feature_backbone_pretrained = getattr(model_cfg, 'feature_backbone_pretrained', None)
        if feature_backbone_pretrained:
            self.feature_backbone.init_weights(pretrained=feature_backbone_pretrained)

        self.feature_neck = feature_extraction_neck(model_cfg.feature_neck)
        if getattr(model_cfg, 'sem_neck', None):
            self.sem_neck = build_neck(model_cfg.sem_neck)
        else:
            self.sem_neck = None

        if not self.drop_psv:
            # cost volume
            self.build_cost = BuildCostVolume(model_cfg.cost_volume)

            # stereo network
            CV_INPUT_DIM = self.build_cost.get_dim(self.feature_neck.stereo_dim[-1]) if not self.mono else self.cv_dim
            self.dres0 = nn.Sequential(
                convbn_3d(CV_INPUT_DIM, self.cv_dim, 1, 1, 0, gn=self.GN),
                nn.ReLU(inplace=True))
            self.dres1 = nn.Sequential(
                convbn_3d(self.cv_dim, self.cv_dim, 3, 1, 1, gn=self.GN))
            self.hg_stereo = nn.ModuleList()
            for _ in range(self.num_hg):
                self.hg_stereo.append(submodule.hourglass(self.cv_dim, gn=self.GN))

        self.front_surface_depth = self.model_cfg.get('front_surface_depth', False)
        if (not self.drop_psv and not self.drop_psv_loss) or self.front_surface_depth:
            # stereo predictions
            self.pred_stereo = nn.ModuleList()
            for _ in range(max(self.num_hg, 1)):
                self.pred_stereo.append(self.build_depth_pred_module(self.cv_dim if not self.front_surface_depth else self.rpn3d_dim))
            self.dispregression = submodule.disparityregression()

        if self.voxel_occupancy:
            self.pred_occupancy = self.build_voxel_pred_module(upsample_ratio=self.voxel_occupancy_upsample, voxel_pred_convs=self.voxel_pred_convs, voxel_pred_hgs=self.voxel_pred_hgs)

        # rpn3d convs
        if self.drop_psv:
            RPN3D_INPUT_DIM = 0
        else:
            RPN3D_INPUT_DIM = self.cv_dim if not (self.use_stereo_out_type != "feature") else 1
        
        if self.squeeze_geo:
            assert self.cat_img_feature
            RPN3D_INPUT_DIM += self.rpn3d_dim
            self.squeeze_geo_conv = nn.Sequential(
                convbn_3d(self.cv_dim * (2 if self.cat_right_img_feature else 1), self.cv_dim, 1, 1, 0, gn=self.GN and self.cv_dim >= 32),
                nn.ReLU(inplace=True),
                convbn_3d(self.cv_dim, self.rpn3d_dim, 3, 1, 1, gn=self.GN),
                nn.ReLU(inplace=True),
            )
        else:
            if self.cat_img_feature:
                RPN3D_INPUT_DIM += self.rpn3d_dim #self.feature_neck.sem_dim[-1]
            if self.cat_right_img_feature:
                RPN3D_INPUT_DIM += self.rpn3d_dim #self.feature_neck.sem_dim[-1]
            
        rpn3d_convs = []
        for i in range(self.num_3dconvs):
            rpn3d_convs.append(
                nn.Sequential(
                    convbn_3d(RPN3D_INPUT_DIM if i == 0 else self.rpn3d_dim,
                              self.rpn3d_dim, 1 if i == 0 else 3, 1, 0 if i == 0 else 1, gn=self.GN and self.rpn3d_dim >= 32),
                    nn.ReLU(inplace=True)))
        self.rpn3d_convs = nn.Sequential(*rpn3d_convs)
        
        if self.num_3dconvs_hg > 0:
            self.rpn3d_hgs = nn.ModuleList()
            for i in range(self.num_3dconvs_hg):
                self.rpn3d_hgs.append(submodule.hourglass(self.rpn3d_dim, gn=self.GN and self.rpn3d_dim >= 32, planes_mul=[2,2]))
        self.rpn3d_pool = torch.nn.AvgPool3d((4, 1, 1), stride=(4, 1, 1))

        # prepare tensors
        self.point_cloud_range = kwargs.get('stereo_point_cloud_range', point_cloud_range)
        # (0.2, 0.2, 0.2); (288, 304, 20)
        self.voxel_size, self.grid_size = voxel_size, grid_size
        self.prepare_depth(self.point_cloud_range, in_camera_view=False)
        self.prepare_coordinates_3d(self.point_cloud_range, voxel_size, grid_size)
        self.max_crop_shape = kwargs.get('max_crop_shape', (320, 1248))
        if self.front_surface_depth:
            crop_x1, crop_x2, crop_y1, crop_y2 = 0, self.max_crop_shape[1], 0, self.max_crop_shape[0]
            self.coordinates_psv = self.prepare_coordinates_psv(crop_x1, crop_x2, crop_y1, crop_y2, 
                img_height=self.max_crop_shape[0], img_width=self.max_crop_shape[1], 
                downsample_disp=(self.downsample_disp, self.downsample_disp, self.voxel_occupancy_downsample_disp))
        self.init_params()
    
    def build_depth_pred_module(self, cv_dim):
        return nn.Sequential(
            convbn_3d(cv_dim, cv_dim, 3, 1, 1, gn=self.GN and cv_dim>=32),
            nn.ReLU(inplace=True),
            nn.Conv3d(cv_dim, 1, 3, 1, 1, bias=True),
            nn.Upsample(scale_factor=self.downsample_disp, mode='trilinear', align_corners=True))

    def build_voxel_pred_module(self, upsample_ratio=(2,2,2), voxel_pred_convs=1, voxel_pred_hgs=0):
        voxel_module = []
        for i in range(voxel_pred_convs):
            voxel_module.extend([
                convbn_3d(self.rpn3d_dim, self.rpn3d_dim, 3, 1, 1, gn=self.GN),
                nn.ReLU(inplace=True),
            ])
        for i in range(voxel_pred_hgs):
            voxel_module.extend([
                submodule.hourglass(self.rpn3d_dim, gn=self.GN),
            ])
        if upsample_ratio and upsample_ratio[0] > 1:
            voxel_module.append(nn.Upsample(scale_factor=tuple(upsample_ratio), mode='trilinear'))
        else:
            voxel_module.append(nn.Identity())
        return nn.Sequential(*voxel_module)

    def prepare_depth(self, point_cloud_range, in_camera_view=True):
        if in_camera_view:
            self.CV_DEPTH_MIN = point_cloud_range[2]
            self.CV_DEPTH_MAX = point_cloud_range[5]
        else:
            self.CV_DEPTH_MIN = point_cloud_range[0]
            self.CV_DEPTH_MAX = point_cloud_range[3]
        assert self.CV_DEPTH_MIN >= 0 and self.CV_DEPTH_MAX > self.CV_DEPTH_MIN
        depth_interval = (self.CV_DEPTH_MAX - self.CV_DEPTH_MIN) / self.maxdisp
        print('stereo volume depth range: {} -> {}, interval {}'.format(self.CV_DEPTH_MIN,
                                                                        self.CV_DEPTH_MAX, depth_interval))
        # prepare downsampled depth
        self.downsampled_depth = torch.zeros(
            (self.maxdisp // self.downsample_disp), dtype=torch.float32)
        for i in range(self.maxdisp // self.downsample_disp):
            self.downsampled_depth[i] = (
                i + self.downsampled_depth_offset) * self.downsample_disp * depth_interval + self.CV_DEPTH_MIN
        self.downsampledx2_depth = torch.zeros(
            (self.maxdisp // 2), dtype=torch.float32)
        for i in range(self.maxdisp // 2):
            self.downsampledx2_depth[i] = (
                i + self.downsampled_depth_offset) * 2 * depth_interval + self.CV_DEPTH_MIN
        # prepare depth
        self.depth = torch.zeros((self.maxdisp), dtype=torch.float32)
        for i in range(self.maxdisp):
            self.depth[i] = (
                i + 0.5) * depth_interval + self.CV_DEPTH_MIN

    def prepare_coordinates_3d(self, point_cloud_range, voxel_size, grid_size, sample_rate=(1, 1, 1)):
        self.X_MIN, self.Y_MIN, self.Z_MIN = point_cloud_range[:3]
        self.X_MAX, self.Y_MAX, self.Z_MAX = point_cloud_range[3:]
        self.VOXEL_X_SIZE, self.VOXEL_Y_SIZE, self.VOXEL_Z_SIZE = voxel_size
        self.GRID_X_SIZE, self.GRID_Y_SIZE, self.GRID_Z_SIZE = grid_size.tolist()

        self.VOXEL_X_SIZE /= sample_rate[0]
        self.VOXEL_Y_SIZE /= sample_rate[1]
        self.VOXEL_Z_SIZE /= sample_rate[2]

        self.GRID_X_SIZE *= sample_rate[0]
        self.GRID_Y_SIZE *= sample_rate[1]
        self.GRID_Z_SIZE *= sample_rate[2]

        zs = torch.linspace(self.Z_MIN + self.VOXEL_Z_SIZE / 2., self.Z_MAX - self.VOXEL_Z_SIZE / 2.,
                            self.GRID_Z_SIZE, dtype=torch.float32)
        ys = torch.linspace(self.Y_MIN + self.VOXEL_Y_SIZE / 2., self.Y_MAX - self.VOXEL_Y_SIZE / 2.,
                            self.GRID_Y_SIZE, dtype=torch.float32)
        xs = torch.linspace(self.X_MIN + self.VOXEL_X_SIZE / 2., self.X_MAX - self.VOXEL_X_SIZE / 2.,
                            self.GRID_X_SIZE, dtype=torch.float32)
        zs, ys, xs = torch.meshgrid(zs, ys, xs)
        coordinates_3d = torch.stack([xs, ys, zs], dim=-1)
        self.coordinates_3d = coordinates_3d.float()

    def prepare_coordinates_psv(self, crop_x1, crop_x2, crop_y1, crop_y2, img_height, img_width, downsample_disp):
        us = torch.linspace(crop_y1 + 0.5 * downsample_disp[0], crop_y2 - 0.5 * downsample_disp[0], img_height // downsample_disp[0], dtype=torch.float32, device='cuda') # height
        vs = torch.linspace(crop_x1 + 0.5 * downsample_disp[1], crop_x2 - 0.5 * downsample_disp[1], img_width // downsample_disp[1], dtype=torch.float32, device='cuda') # width 
        if downsample_disp[2] == 4:
            ds = self.downsampled_depth.cuda()
        elif downsample_disp[2] == 2:
            ds = self.downsampledx2_depth.cuda()
        elif downsample_disp[2] == 1:
            ds = self.depth.cuda()
        ds, us, vs = torch.meshgrid(ds, us, vs)
        coordinates_psv = torch.stack([vs, us, ds], dim=-1)
        return coordinates_psv

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[
                    2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        if self.voxel_occupancy:
            torch.nn.init.normal_(self.pred_occupancy[0][0].weight, mean=0, std=0.01)
            # torch.nn.init.constant_(self.pred_occupancy[0][0].bias, -2.19)

    def pred_depth(self, depth_conv_module, cost1, img_shape):
        cost1 = depth_conv_module(cost1)
        if cost1.shape[2] != self.maxdisp:
            cost1 = F.interpolate(
                cost1, [self.maxdisp, *img_shape],
                mode='trilinear',
                align_corners=True)
        cost1 = torch.squeeze(cost1, 1)
        cost1_softmax = F.softmax(cost1, dim=1)
        pred1 = self.dispregression(cost1_softmax,
                                    depth=self.depth.cuda())
        return cost1, cost1_softmax, pred1

    def pred_voxel(self, voxel_conv_module, voxel):
        voxel_occupancy = voxel_conv_module(voxel)
        return voxel_occupancy

    def get_local_depth(self, d_prob):
        with torch.no_grad():
            mean_d = []
            for i in range(len(d_prob)):
                d = self.depth.cuda()[None, :, None, None]
                d_mul_p = d * d_prob[i:i+1]
                local_window = 5
                p_local_sum = 0
                for off in range(0, local_window):
                    cur_p = d_prob[i:i+1, off:off + d_prob.shape[1] - local_window + 1]
                    p_local_sum += cur_p
                max_indices = p_local_sum.max(1, keepdim=True).indices
                pd_local_sum_for_max = 0
                for off in range(0, local_window):
                    cur_pd = torch.gather(d_mul_p, 1, max_indices + off).squeeze(1)  # d_prob[:, off:off + d_prob.shape[1] - local_window + 1]
                    pd_local_sum_for_max += cur_pd
                mean_d.append(pd_local_sum_for_max / torch.gather(p_local_sum, 1, max_indices).squeeze(1))
            mean_d = torch.cat(mean_d, dim=0)
        return mean_d

    def forward_2d(self, img):
        features = self.feature_backbone(img)
        features = [img] + list(features)
        return self.feature_neck(features)


    # ================================================================
    # Adaptive stage-A support
    #
    # A_s = stem + layer1, always full.
    # A_d = layer2 + layer3 + layer4, optionally selective.
    #
    # IMPORTANT:
    #   feature_neck is intentionally kept FULL in this first version.
    #   We first profile whether localizing layer2--4 is physically useful
    #   before making the SPP/FPN neck selective.
    # ================================================================

    def _ensure_adaptive_a_runtime(self):
        if hasattr(self, '_adaptive_a_enabled'):
            return

        self._adaptive_a_enabled = False
        self._adaptive_a_ratio = 1.0
        self._adaptive_a_position_mode = 'center'

        # Cache layer2/layer3/layer4 output for left/right independently.
        self._adaptive_a_cache = {
            'left': [None, None, None],
            'right': [None, None, None],
        }

        # Final dense stereo-feature cache seen by stage B.
        #
        # Current shape:
        #     [N, 96, 320, 1248]
        #
        # layer2/3/4 caches above are internal A_d execution caches.
        # This stereo cache is the actual A-stage interface cache.
        self._adaptive_a_stereo_cache = {
            'left': None,
            'right': None,
        }

        # Halo on the layer2/3/4 common 80x312 grid used by
        # local FPN/upconv/lastconv execution.
        self._adaptive_a_neck_halo = 4

        # ROI is defined on layer2/3/4 common H/W grid.
        # For current R18 config:
        # layer1: 160x624
        # layer2:  80x312
        # layer3:  80x312
        # layer4:  80x312
        #
        # Halo is measured on the corresponding output grid.
        # These are deliberately conservative starting values.
        self._adaptive_a_halos = {
            1: 4,   # layer2
            2: 8,   # layer3, dilation=2
            3: 16,  # layer4, dilation=4
        }

        self._adaptive_a_last_roi = None
        self._adaptive_a_last_output_roi = None

    def reset_adaptive_a_cache(self):
        self._ensure_adaptive_a_runtime()

        self._adaptive_a_cache = {
            'left': [None, None, None],
            'right': [None, None, None],
        }

        self._adaptive_a_stereo_cache = {
            'left': None,
            'right': None,
        }

        self._adaptive_a_last_roi = None
        self._adaptive_a_last_output_roi = None

    def set_adaptive_a_profile(self, ratio, position_mode='center'):
        """
        Offline profiling override.

        ratio:
            0.0 -> no layer2--4 refresh; use stage cache
            1.0 -> full layer2--4 refresh
            (0,1) -> one rectangular ROI

        position_mode:
            center / boundary / random

        This function is for profiling/debugging. The final learned scheduler
        will provide its own ROI metadata on GPU.
        """
        self._ensure_adaptive_a_runtime()

        ratio = float(ratio)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError(f'ratio must be in [0,1], got {ratio}')

        if position_mode not in ('center', 'boundary', 'random'):
            raise ValueError(position_mode)

        self._adaptive_a_enabled = True
        self._adaptive_a_ratio = ratio
        self._adaptive_a_position_mode = position_mode

    def clear_adaptive_a_profile(self):
        self._ensure_adaptive_a_runtime()
        self._adaptive_a_enabled = False
        self._adaptive_a_ratio = 1.0
        self._adaptive_a_position_mode = 'center'


    # ================================================================
    # Differentiable adaptive-training surrogate
    #
    # This is deliberately separate from the physical ROI inference path.
    # Training computes the full current stage, then composes current/cache
    # with a straight-through rectangular mask. This keeps ROI location
    # differentiable while preserving hard rectangular execution semantics in
    # the forward pass. Physical ROI kernels are profiled/used after training.
    # ================================================================

    def set_adaptive_training_state(
        self,
        a_mask,
        b_mask,
        a_cache_left,
        a_cache_right,
        b_cache,
    ):
        self._adaptive_training_state = {
            'a_mask': a_mask,
            'b_mask': b_mask,
            'a_cache_left': a_cache_left,
            'a_cache_right': a_cache_right,
            'b_cache': b_cache,
        }

    def clear_adaptive_training_state(self):
        self._adaptive_training_state = None

    @staticmethod
    def _adaptive_training_mix_2d(current, cache, mask):
        if cache is None:
            return current
        if tuple(cache.shape) != tuple(current.shape):
            raise RuntimeError(
                f'adaptive A cache shape mismatch: cache={tuple(cache.shape)}, '
                f'current={tuple(current.shape)}'
            )
        m = F.interpolate(mask.float(), size=current.shape[-2:], mode='nearest')
        m = m.to(device=current.device, dtype=current.dtype)
        return m * current + (1.0 - m) * cache.to(current.dtype)

    @staticmethod
    def _adaptive_training_mix_3d(current, cache, mask):
        if cache is None:
            return current
        if tuple(cache.shape) != tuple(current.shape):
            raise RuntimeError(
                f'adaptive B cache shape mismatch: cache={tuple(cache.shape)}, '
                f'current={tuple(current.shape)}'
            )
        m = F.interpolate(mask.float(), size=current.shape[-2:], mode='nearest')
        m = m[:, :, None].to(device=current.device, dtype=current.dtype)
        return m * current + (1.0 - m) * cache.to(current.dtype)

    def adaptive_a_is_enabled(self):
        self._ensure_adaptive_a_runtime()
        return bool(self._adaptive_a_enabled)

    def _capture_adaptive_a_full_cache(self, side, backbone_outputs):
        """
        backbone_outputs:
            layer1, layer2, layer3, layer4

        Only layer2--4 are cached because layer1 belongs to A_s and is
        recomputed fully every frame.
        """
        self._ensure_adaptive_a_runtime()

        if self.training:
            return

        assert side in ('left', 'right')
        assert len(backbone_outputs) == 4

        # detach() does not copy GPU data; it only keeps a persistent reference.
        self._adaptive_a_cache[side] = [
            backbone_outputs[1].detach(),
            backbone_outputs[2].detach(),
            backbone_outputs[3].detach(),
        ]

    def _capture_adaptive_a_stereo_cache(
        self,
        side,
        stereo_feature,
    ):
        """
        Capture complete stage-A output after a Full A execution.

        detach() keeps the tensor on GPU and does not perform a CPU copy.
        """
        self._ensure_adaptive_a_runtime()

        if self.training:
            return

        if side not in ('left', 'right'):
            raise ValueError(side)

        self._adaptive_a_stereo_cache[side] = \
            stereo_feature.detach()

        self._adaptive_a_last_output_roi = (
            0,
            stereo_feature.shape[-2],
            0,
            stereo_feature.shape[-1],
        )

    @staticmethod
    def _adaptive_a_update_dense_cache(
        cache,
        patch,
        rect,
    ):
        """
        Update only the recomputed current-frame patch in a dense cache.
        """
        y0, y1, x0, x1 = [int(v) for v in rect]

        expected_hw = (
            y1 - y0,
            x1 - x0,
        )

        if tuple(patch.shape[-2:]) != expected_hw:
            raise RuntimeError(
                'A stereo cache patch mismatch: '
                f'patch={tuple(patch.shape[-2:])}, '
                f'expected={expected_hw}, '
                f'rect={rect}'
            )

        cache[
            ...,
            y0:y1,
            x0:x1
        ].copy_(patch)

    def forward_2d_shallow(self, img):
        """
        Exact ResNet stem + layer1 forward.

        Current configuration:
            deep_stem=False
            with_max_pool=False

        The code still handles with_max_pool=True.
        """
        bb = self.feature_backbone

        if getattr(bb, 'deep_stem', False):
            x = bb.stem(img)
        else:
            x = bb.conv1(img)
            x = bb.norm1(x)
            x = bb.relu(x)

        if getattr(bb, 'with_max_pool', False):
            x = bb.maxpool(x)

        if not hasattr(bb, 'res_layers') or len(bb.res_layers) != 4:
            raise RuntimeError(
                'Adaptive A currently expects a 4-stage MMDetection ResNet'
            )

        layer1 = getattr(bb, bb.res_layers[0])
        x = layer1(x)

        return x

    def forward_2d_deep_full_from_shallow(
        self,
        img,
        shallow,
        return_stage_features=False,
    ):
        """
        Run layer2--4 + the original feature_neck from a full layer1 tensor.

        Numerically this should match:
            feature_backbone(img) -> feature_neck(...)
        """
        bb = self.feature_backbone

        out_indices = tuple(getattr(bb, 'out_indices', (0, 1, 2, 3)))
        if out_indices != (0, 1, 2, 3):
            raise RuntimeError(
                f'Adaptive A expects out_indices=(0,1,2,3), got {out_indices}'
            )

        feats = [shallow]
        x = shallow

        for layer_name in bb.res_layers[1:]:
            x = getattr(bb, layer_name)(x)
            feats.append(x)

        stereo_feature, sem_feature = self.feature_neck([img] + feats)

        if return_stage_features:
            return stereo_feature, sem_feature, feats

        return stereo_feature, sem_feature

    @staticmethod
    def _adaptive_a_expand_rect(rect, H, W, halo):
        y0, y1, x0, x1 = rect

        return (
            max(0, y0 - halo),
            min(H, y1 + halo),
            max(0, x0 - halo),
            min(W, x1 + halo),
        )

    @staticmethod
    def _adaptive_a_rect_from_ratio(
        H,
        W,
        ratio,
        position_mode='center',
        align=4,
    ):
        """
        Convert an area ratio into one rectangular ROI.

        The rectangle approximately preserves the full feature-map aspect ratio:
            roi_h/H ~= roi_w/W ~= sqrt(ratio)
        """
        if ratio <= 0:
            return None

        if ratio >= 1:
            return (0, H, 0, W)

        scale = ratio ** 0.5

        rh = int(round(H * scale / align)) * align
        rw = int(round(W * scale / align)) * align

        rh = min(H, max(align, rh))
        rw = min(W, max(align, rw))

        if position_mode == 'center':
            y0 = (H - rh) // 2
            x0 = (W - rw) // 2

        elif position_mode == 'boundary':
            # bottom-right legal boundary
            y0 = H - rh
            x0 = W - rw

        elif position_mode == 'random':
            # deterministic pseudo-random location.
            # Offline profiling only; no GPU->CPU synchronization.
            sy = H - rh
            sx = W - rw

            y0 = 0 if sy == 0 else ((37 * sy + 11 * sx + 17) % (sy + 1))
            x0 = 0 if sx == 0 else ((53 * sx + 7 * sy + 29) % (sx + 1))

        else:
            raise ValueError(position_mode)

        return (
            int(y0),
            int(y0 + rh),
            int(x0),
            int(x0 + rw),
        )

    def _adaptive_a_run_layer2_roi(self, shallow, rect, halo):
        """
        Run ResNet layer2 on a cropped layer1 feature.

        rect is defined on layer2 output coordinates.

        Current config has:
            layer1 -> layer2 spatial stride = 2
        """
        bb = self.feature_backbone
        layer2 = getattr(bb, bb.res_layers[1])

        H1, W1 = shallow.shape[-2:]

        # Expected full layer2 shape for stride=2.
        H2 = (H1 + 1) // 2
        W2 = (W1 + 1) // 2

        y0, y1, x0, x1 = rect

        ey0, ey1, ex0, ex1 = self._adaptive_a_expand_rect(
            rect, H2, W2, halo
        )

        # Preserve global stride phase.
        iy0 = ey0 * 2
        iy1 = min(H1, ey1 * 2)
        ix0 = ex0 * 2
        ix1 = min(W1, ex1 * 2)

        x = shallow[..., iy0:iy1, ix0:ix1].contiguous()
        y = layer2(x)

        expected_hw = (ey1 - ey0, ex1 - ex0)
        if tuple(y.shape[-2:]) != expected_hw:
            raise RuntimeError(
                'layer2 local shape mismatch: '
                f'got={tuple(y.shape[-2:])}, expected={expected_hw}'
            )

        cy0 = y0 - ey0
        cy1 = cy0 + (y1 - y0)
        cx0 = x0 - ex0
        cx1 = cx0 + (x1 - x0)

        return y[..., cy0:cy1, cx0:cx1]

    def _adaptive_a_run_stride1_roi(
        self,
        layer,
        dense_input,
        rect,
        halo,
    ):
        """
        Local execution for layer3/layer4.

        Current config uses stride=1 for both, so input/output H/W are equal.
        """
        H, W = dense_input.shape[-2:]

        y0, y1, x0, x1 = rect

        ey0, ey1, ex0, ex1 = self._adaptive_a_expand_rect(
            rect, H, W, halo
        )

        x = dense_input[..., ey0:ey1, ex0:ex1].contiguous()
        y = layer(x)

        expected_hw = (ey1 - ey0, ex1 - ex0)
        if tuple(y.shape[-2:]) != expected_hw:
            raise RuntimeError(
                'stride-1 local shape mismatch: '
                f'got={tuple(y.shape[-2:])}, expected={expected_hw}'
            )

        cy0 = y0 - ey0
        cy1 = cy0 + (y1 - y0)
        cx0 = x0 - ex0
        cx1 = cx0 + (x1 - x0)

        return y[..., cy0:cy1, cx0:cx1]

    @staticmethod
    def _adaptive_a_update_cache(cache, patch, rect):
        """
        Patch-update without cloning the full tensor.
        """
        y0, y1, x0, x1 = rect

        if tuple(patch.shape[-2:]) != (y1 - y0, x1 - x0):
            raise RuntimeError(
                f'patch={tuple(patch.shape[-2:])}, '
                f'roi={(y1-y0, x1-x0)}'
            )

        cache[..., y0:y1, x0:x1].copy_(patch)

    def forward_2d_adaptive(self, img, side):
        """
        A_s:
            full stem + layer1

        A_d:
            selective layer2/3/4 with persistent dense cache

        feature_neck:
            full in this first implementation.

        This path is inference-only.
        """
        self._ensure_adaptive_a_runtime()

        if self.training:
            raise RuntimeError(
                'forward_2d_adaptive is currently inference/profiling only'
            )

        if side not in ('left', 'right'):
            raise ValueError(side)

        shallow = self.forward_2d_shallow(img)

        ratio = float(self._adaptive_a_ratio)
        position_mode = self._adaptive_a_position_mode
        cache = self._adaptive_a_cache[side]

        # Cache not initialized yet:
        # perform one full layer2--4 pass and initialize it.
        if any(x is None for x in cache):
            stereo_feature, sem_feature, feats = \
                self.forward_2d_deep_full_from_shallow(
                    img,
                    shallow,
                    return_stage_features=True,
                )

            self._adaptive_a_cache[side] = [
                feats[1].detach(),
                feats[2].detach(),
                feats[3].detach(),
            ]

            self._capture_adaptive_a_stereo_cache(
                side,
                stereo_feature,
            )

            self._adaptive_a_last_roi = (
                0,
                feats[1].shape[-2],
                0,
                feats[1].shape[-1],
            )

            return shallow, stereo_feature, sem_feature

        # Stage-Full A_d.
        if ratio >= 1.0 - 1e-8:
            stereo_feature, sem_feature, feats = \
                self.forward_2d_deep_full_from_shallow(
                    img,
                    shallow,
                    return_stage_features=True,
                )

            self._adaptive_a_cache[side] = [
                feats[1].detach(),
                feats[2].detach(),
                feats[3].detach(),
            ]

            self._capture_adaptive_a_stereo_cache(
                side,
                stereo_feature,
            )

            self._adaptive_a_last_roi = (
                0,
                feats[1].shape[-2],
                0,
                feats[1].shape[-1],
            )

            return shallow, stereo_feature, sem_feature

        # All layer2--4 caches share the same H/W in the current R18 config.
        H2, W2 = cache[0].shape[-2:]

        for c in cache[1:]:
            if tuple(c.shape[-2:]) != (H2, W2):
                raise RuntimeError(
                    'Current adaptive A implementation expects '
                    'layer2/layer3/layer4 to share H/W'
                )

        rect = self._adaptive_a_rect_from_ratio(
            H2,
            W2,
            ratio,
            position_mode=position_mode,
            align=4,
        )

        # ratio == 0:
        # skip all layer2--4 recomputation and directly use the stage caches.
        if rect is not None:
            bb = self.feature_backbone

            # --------------------------------------------------------
            # layer2
            # --------------------------------------------------------
            patch2 = self._adaptive_a_run_layer2_roi(
                shallow,
                rect,
                self._adaptive_a_halos[1],
            )

            self._adaptive_a_update_cache(
                cache[0],
                patch2,
                rect,
            )

            # --------------------------------------------------------
            # layer3
            # Input is the CURRENT/CACHED dense layer2 map.
            # --------------------------------------------------------
            layer3 = getattr(bb, bb.res_layers[2])

            patch3 = self._adaptive_a_run_stride1_roi(
                layer3,
                cache[0],
                rect,
                self._adaptive_a_halos[2],
            )

            self._adaptive_a_update_cache(
                cache[1],
                patch3,
                rect,
            )

            # --------------------------------------------------------
            # layer4
            # Input is the CURRENT/CACHED dense layer3 map.
            # --------------------------------------------------------
            layer4 = getattr(bb, bb.res_layers[3])

            patch4 = self._adaptive_a_run_stride1_roi(
                layer4,
                cache[1],
                rect,
                self._adaptive_a_halos[3],
            )

            self._adaptive_a_update_cache(
                cache[2],
                patch4,
                rect,
            )

        # Current full shallow layer1 + dense current/cache layer2--4.
        feats = [
            img,
            shallow,
            cache[0],
            cache[1],
            cache[2],
        ]

        stereo_cache = self._adaptive_a_stereo_cache[side]

        if stereo_cache is None:
            raise RuntimeError(
                'Adaptive A stereo cache is not initialized. '
                'Each scene must start from a Full frame.'
            )

        # ------------------------------------------------------------
        # Reuse-only A_d:
        #
        # A_s still runs fully, but layer2--4 and feature_neck are
        # completely skipped.
        # ------------------------------------------------------------
        if rect is None:
            self._adaptive_a_last_roi = None
            self._adaptive_a_last_output_roi = None

            stereo_feature = stereo_cache

            # Current config:
            #   cat_img_feature=False
            #   with_sem_neck=False
            sem_feature = None

            return shallow, stereo_feature, sem_feature

        # ------------------------------------------------------------
        # Local neck execution.
        #
        # SPP keeps global context, while FPN/upconv/lastconv execute
        # only around the selected image-space ROI.
        # ------------------------------------------------------------
        stereo_patch, output_rect = \
            self.feature_neck.forward_stereo_roi(
                feats,
                base_rect=rect,
                base_halo=self._adaptive_a_neck_halo,
            )

        # Current ROI overwrites the corresponding part of the
        # persistent dense A-stage output cache.
        self._adaptive_a_update_dense_cache(
            stereo_cache,
            stereo_patch,
            output_rect,
        )

        self._adaptive_a_last_roi = rect
        self._adaptive_a_last_output_roi = output_rect

        # Dense stage-A output consumed by B:
        #
        #   current ROI + historical A cache outside.
        stereo_feature = stereo_cache

        sem_feature = None

        return shallow, stereo_feature, sem_feature

    @staticmethod
    def compute_mapping(c3d, image_shape, calib_proj, depth_range, pose_transform=None, use_amp=False):
        coord_img = project_rect_to_image(
            c3d,
            calib_proj,
            pose_transform)
        coord_img = torch.cat(
            [coord_img, c3d[..., 2:]], dim=-1)
        # TODO: crop augmentation
        crop_x1, crop_x2 = 0, image_shape[1]
        crop_y1, crop_y2 = 0, image_shape[0]
        # assert (crop_x1, crop_x2, crop_y1, crop_y2) == (0, 1248, 0, 320)
        norm_coord_img = (coord_img - torch.as_tensor([crop_x1, crop_y1, depth_range[0]], device=coord_img.device)) / torch.as_tensor(
            [crop_x2 - 1 - crop_x1, crop_y2 - 1 - crop_y1, depth_range[1] - depth_range[0]], device=coord_img.device)
        # resize to [-1, 1]
        norm_coord_img = norm_coord_img * 2. - 1.
        if use_amp:
            coord_img = coord_img.half()
            norm_coord_img = norm_coord_img.half()
        return coord_img, norm_coord_img

    def compute_disp_channels(self, voxel_disps, img_channels, inv_ratio=0.1):
        shift_channels = (img_channels - self.cv_dim + 1)
        voxel_disps_channels = F.interpolate(voxel_disps[None,None], (shift_channels,), mode='linear')[0,0]
        if inv_ratio > 0.:
            voxel_disps_channels = voxel_disps_channels ** inv_ratio
        voxel_disps_channels = voxel_disps_channels / ((voxel_disps_channels.max() - voxel_disps_channels.min()) / shift_channels)
        voxel_disps_channels -= voxel_disps_channels.min()
        voxel_disps_channels = shift_channels-1 - voxel_disps_channels.to(int).clamp(0, shift_channels-1)
        return voxel_disps_channels
    
    def build_3d_geometry_volume(self, RPN_feature, norm_coord_imgs, voxel_disps):
        if RPN_feature.shape[1] <= self.cv_dim:
            norm_coord_imgs_2d = norm_coord_imgs.clone().detach()
            norm_coord_imgs_2d[..., 2] = 0
            Voxel_2D = F.grid_sample(RPN_feature.unsqueeze(2), norm_coord_imgs_2d, align_corners=True)
            # Voxel_2D = build_geometry_volume(RPN_feature, norm_coord_imgs_2d[..., :2])
        else:
            voxel_disps_channels = self.compute_disp_channels(voxel_disps, 
                img_channels=RPN_feature.shape[1], inv_ratio=self.inv_smooth_geo)
            Voxel_2D = build_dps_geometry_volume(RPN_feature, norm_coord_imgs[..., :2], \
                voxel_disps_channels.to(torch.int32), self.cv_dim, self.geometry_volume_shift)
        return Voxel_2D

    def build_plane_sweep_volume(self, RPN_feature, norm_coord_imgs, voxel_disps):
        if RPN_feature.shape[1] <= self.cv_dim:
            norm_coord_imgs_2d = norm_coord_imgs.clone().detach()
            norm_coord_imgs_2d[..., 2] = 0
            Voxel_2D = F.grid_sample(RPN_feature.unsqueeze(2), norm_coord_imgs_2d, align_corners=True)
            # Voxel_2D = build_geometry_volume(RPN_feature, norm_coord_imgs_2d[..., :2])
        else:
            voxel_disps_channels = self.compute_disp_channels(voxel_disps, 
                img_channels=RPN_feature.shape[1])
            Voxel_2D = build_dps_geometry_volume(RPN_feature, norm_coord_imgs[..., :2], \
                voxel_disps_channels.to(torch.int32), 32, self.geometry_volume_shift)
        return Voxel_2D

    def forward(self, batch_dict):
        # update self.use_amp
        self.use_amp = self.use_amp_dict['TRAIN'] if self.training else self.use_amp_dict['TEST']
        tensor_dtype = torch.float16 if self.use_amp else torch.float32

        left = batch_dict['left_img']
        calib = batch_dict['calib']
        fu_mul_baseline = torch.as_tensor(
            [x.fu_mul_baseline for x in calib], dtype=tensor_dtype, device=left.device)
        calibs_Proj = torch.as_tensor(
            [x.P2 for x in calib], dtype=tensor_dtype, device=left.device)
        calibs_Proj_R = torch.as_tensor(
            [x.P3 for x in calib], dtype=tensor_dtype, device=left.device)

        N = batch_dict['batch_size']

        # feature extraction
        #
        # Normal / Global-Full path remains the original backbone+neck path.
        # Adaptive A path uses:
        #     full stem+layer1
        #     selective layer2--4 + cache
        #     full feature_neck
        #
        if self.adaptive_a_is_enabled():
            left_shallow, left_stereo_feat, left_sem_feat = \
                self.forward_2d_adaptive(left, side='left')

            if not self.mono:
                right = batch_dict['right_img']
                right_shallow, right_stereo_feat, right_sem_feat = \
                    self.forward_2d_adaptive(right, side='right')
            else:
                right_shallow = None
                right_stereo_feat, right_sem_feat = None, None

        else:
            # Exact original full path.
            left_backbone_features = self.feature_backbone(left)
            left_shallow = left_backbone_features[0]

            # Keep references to layer2--4 so the next adaptive frame can
            # immediately use them as a valid cache. No GPU copy is performed.
            self._capture_adaptive_a_full_cache(
                'left',
                left_backbone_features,
            )

            left_features = [left] + list(left_backbone_features)
            left_stereo_feat, left_sem_feat = self.feature_neck(left_features)

            # Full frame initializes / refreshes the complete A-stage cache.
            self._capture_adaptive_a_stereo_cache(
                'left',
                left_stereo_feat,
            )

            if not self.mono:
                right = batch_dict['right_img']
                right_backbone_features = self.feature_backbone(right)
                right_shallow = right_backbone_features[0]

                self._capture_adaptive_a_full_cache(
                    'right',
                    right_backbone_features,
                )

                right_features = [right] + list(right_backbone_features)
                right_stereo_feat, right_sem_feat = \
                    self.feature_neck(right_features)

                self._capture_adaptive_a_stereo_cache(
                    'right',
                    right_stereo_feat,
                )
            else:
                right_shallow = None
                right_stereo_feat, right_sem_feat = None, None

        # ------------------------------------------------------------
        # Training-time A-stage interface composition.
        # Q_A lives on the 80x312 A_d native grid; the mask is resized to the
        # dense stereo-output grid (320x1248 in the current configuration).
        # ------------------------------------------------------------
        train_state = getattr(self, '_adaptive_training_state', None)
        batch_dict['adaptive_stage_a_left_current'] = left_stereo_feat
        if not self.mono:
            batch_dict['adaptive_stage_a_right_current'] = right_stereo_feat

        if train_state is not None:
            if self.cat_img_feature or self.cat_right_img_feature:
                raise NotImplementedError(
                    'The current adaptive-training surrogate assumes '
                    'cat_img_feature=False and cat_right_img_feature=False, '
                    'which matches the supplied StreamDSGN config.'
                )
            left_stereo_feat = self._adaptive_training_mix_2d(
                left_stereo_feat,
                train_state['a_cache_left'],
                train_state['a_mask'],
            )
            if not self.mono:
                right_stereo_feat = self._adaptive_training_mix_2d(
                    right_stereo_feat,
                    train_state['a_cache_right'],
                    train_state['a_mask'],
                )

        batch_dict['adaptive_stage_a_left'] = left_stereo_feat
        if not self.mono:
            batch_dict['adaptive_stage_a_right'] = right_stereo_feat

        # Expose A_s output for the importance predictor.
        batch_dict['left_shallow_feature'] = left_shallow
        if right_shallow is not None:
            batch_dict['right_shallow_feature'] = right_shallow

        if self.sem_neck is not None:
            batch_dict['sem_features'] = self.sem_neck([left_sem_feat])
        else:
            batch_dict['sem_features'] = [left_sem_feat]
        batch_dict['left_rpn_feature'] = left_sem_feat
        if not self.mono:
            batch_dict['right_rpn_feature'] = right_sem_feat
        
        # torch.cuda.synchronize()
        # t1 = time.time()
        if not self.drop_psv:
            # stereo matching: build stereo volume
            downsampled_depth = self.downsampled_depth.cuda().half() if self.use_amp else self.downsampled_depth.cuda()
            downsampled_disp = fu_mul_baseline[:, None] / \
                downsampled_depth[None, :] / (self.downsample_disp if not self.fullres_stereo_feature else 1)
            if left_stereo_feat.shape[1] > self.cv_dim:
                psv_disps_channels = self.compute_disp_channels(downsampled_disp[0], left_stereo_feat.shape[1], inv_ratio=self.inv_smooth_psv)
                cost_raw = self.build_cost(left_stereo_feat, right_stereo_feat,
                                        None, None, downsampled_disp, psv_disps_channels.to(torch.int32))
            else:
                cost_raw = self.build_cost(left_stereo_feat, right_stereo_feat,
                                        None, None, downsampled_disp)
                
            # stereo matching network
            cost0 = self.dres0(cost_raw)
            cost0 = self.dres1(cost0) + cost0
            if len(self.hg_stereo) > 0:
                all_costs = []
                cur_cost = cost0
                assert len(self.hg_stereo) == 1
                for hg_stereo_module in self.hg_stereo:
                    cost_residual = hg_stereo_module(cur_cost, None, None)
                    cur_cost = cur_cost + cost_residual
                    all_costs.append(cur_cost)
            else:
                all_costs = [cost0]
            assert len(all_costs) > 0, 'at least one hourglass'
            if not self.drop_psv_loss:
                # stereo matching: outputs
                batch_dict['depth_preds'] = []
                if not self.training:
                    batch_dict['depth_preds_local'] = []
                batch_dict['depth_volumes'] = []
                batch_dict['depth_samples'] = self.depth.clone().detach().cuda()
                for idx in range(len(all_costs)):
                    upcost_i, cost_softmax_i, pred_i = self.pred_depth(self.pred_stereo[idx], all_costs[idx], left.shape[2:4])
                    batch_dict['depth_volumes'].append(upcost_i)
                    batch_dict['depth_preds'].append(pred_i)
                    if not self.training:
                        batch_dict['depth_preds_local'].append(self.get_local_depth(cost_softmax_i))
                
            # beginning of 3d detection part
            if self.use_stereo_out_type == "feature":
                out = all_costs[-1]
            elif self.use_stereo_out_type == "prob":
                out = cost_softmax_i.unsqueeze(1)
            elif self.use_stereo_out_type == "cost":
                out = upcost_i.unsqueeze(1)
            else:
                raise ValueError('wrong self.use_stereo_out_type option')


            # --------------------------------------------------------
            # Training-time B-stage interface composition.
            # B native ROI grid is H=80,W=312 in the current configuration.
            # The mask is broadcast over disparity/depth D.
            # --------------------------------------------------------
            batch_dict['adaptive_stage_b_current'] = out
            train_state = getattr(self, '_adaptive_training_state', None)
            if train_state is not None:
                out = self._adaptive_training_mix_3d(
                    out,
                    train_state['b_cache'],
                    train_state['b_mask'],
                )
            batch_dict['adaptive_stage_b'] = out

        # torch.cuda.synchronize()
        # t2 = time.time()
        # print('PSV:', t2-t1)

        # convert plane-sweep into 3d volume
        # torch.cuda.synchronize()
        # t1 = time.time()
        coordinates_3d = self.coordinates_3d.cuda().half() if self.use_amp else self.coordinates_3d.cuda()
        batch_dict['coord'] = coordinates_3d
        norm_coord_imgs = []
        if self.cat_right_img_feature:
            norm_coord_imgs_R = []
        valids2d = []
        for i in range(N):
            # map to rect camera coordinates
            c3d = coordinates_3d.view(-1, 3)
            if 'random_T' in batch_dict:
                random_T = batch_dict['random_T'][i]
                c3d = torch.matmul(c3d, random_T[:3, :3].T) + random_T[:3, 3]

            # in pseudo lidar coord
            c3d = project_pseudo_lidar_to_rectcam(c3d)
            #------------ left images ----------------------
            coord_img, norm_coord_img = self.compute_mapping(c3d,
                left.shape[2:],
                torch.as_tensor(calib[i].P2, device='cuda', dtype=tensor_dtype),
                [self.CV_DEPTH_MIN, self.CV_DEPTH_MAX],
                use_amp=self.use_amp)
            coord_img = coord_img.view(*self.coordinates_3d.shape[:3], 3)
            norm_coord_img = norm_coord_img.view(*self.coordinates_3d.shape[:3], 3)
            norm_coord_imgs.append(norm_coord_img)

            if self.cat_right_img_feature:
                #------------ right images ----------------------
                coord_img_R, norm_coord_img_R = self.compute_mapping(c3d,
                    right.shape[2:],
                    torch.as_tensor(calib[i].P3, device='cuda', dtype=tensor_dtype),
                    [self.CV_DEPTH_MIN, self.CV_DEPTH_MAX],
                    use_amp=self.use_amp)
                coord_img_R = coord_img_R.view(*self.coordinates_3d.shape[:3], 3)
                norm_coord_img_R = norm_coord_img_R.view(*self.coordinates_3d.shape[:3], 3)
                norm_coord_imgs_R.append(norm_coord_img_R)

            # valid: within images
            img_shape = batch_dict['image_shape'][i]
            valid_mask_2d = (coord_img[..., 0] >= 0) & (coord_img[..., 0] <= img_shape[1]) & \
                (coord_img[..., 1] >= 0) & (coord_img[..., 1] <= img_shape[0])
            valids2d.append(valid_mask_2d)

        norm_coord_imgs = torch.stack(norm_coord_imgs, dim=0)
        if self.cat_right_img_feature:
            norm_coord_imgs_R = torch.stack(norm_coord_imgs_R, dim=0)
        valids2d = torch.stack(valids2d, dim=0)
        batch_dict['norm_coord_imgs'] = norm_coord_imgs

        valids = valids2d & (norm_coord_imgs[..., 2] >= -1.) & (norm_coord_imgs[..., 2] <= 1.)
        batch_dict['valids'] = valids
        valids = valids.float()
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print('PSV to 3DV: ', t2-t1)
        if not self.drop_psv:
            # Retrieve Voxel Feature from Cost Volume Feature
            Voxel = F.grid_sample(out, norm_coord_imgs, align_corners=True)
            Voxel = Voxel * valids[:, None, :, :, :]
            Voxels = [Voxel]
        else:
            Voxels = []

        voxel_depths = c3d.view(coordinates_3d.shape)[0,0,:,2]
        voxel_disps = calib[0].fu_mul_baseline / voxel_depths

        # Retrieve Voxel Feature from 2D Img Feature
        if self.cat_img_feature:
            Voxel_2D = self.build_3d_geometry_volume(left_sem_feat, norm_coord_imgs, voxel_disps)
            Voxel_2D *= valids2d.float()[:, None, :, :, :]
            Voxels.append(Voxel_2D)

        if self.cat_right_img_feature:
            Voxel_2D_R = self.build_3d_geometry_volume(right_sem_feat, norm_coord_imgs_R, voxel_disps)
            Voxel_2D_R *= valids2d.float()[:, None, :, :, :]
            Voxels.append(Voxel_2D_R)

        if self.squeeze_geo:
            Voxel = self.squeeze_geo_conv(torch.cat([Voxels[-2], Voxels[-1]], dim=1) if self.cat_right_img_feature else Voxels[-1] )
            if not self.drop_psv:
                Voxel = torch.cat([Voxels[0], Voxel], dim=1)
        else:
            Voxel = Voxels[0] if len(Voxels) == 1 else torch.cat(Voxels, dim=1)

        # torch.cuda.synchronize()
        # t1 = time.time()
        Voxel = self.rpn3d_convs(Voxel)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print('rpn3d: ', t2-t1)
        # torch.cuda.synchronize()
        # t1 = time.time()
        if self.num_3dconvs_hg > 0:
            if self.num_3dconvs_hg == 1:
                pre, post = True, True
                for hg_stereo_module in self.rpn3d_hgs:
                    Voxel, pre, post = hg_stereo_module(Voxel, pre, post)
            else:
                pre, post = None, None
                for hg_stereo_module in self.rpn3d_hgs:
                    Voxel = hg_stereo_module(Voxel, pre, post)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print('hg3d: ', t2-t1)

        batch_dict['volume_features_nopool'] = Voxel
        if self.sup_geometry == 'volume':
            Voxel_for_geo = Voxel
        Voxel = self.rpn3d_pool(Voxel) 
        if self.sup_geometry == 'pooledvolume':
            Voxel_for_geo = Voxel
        batch_dict['volume_features'] = Voxel
        if self.training and self.voxel_occupancy:
            batch_dict = self.forward_voxel_occupancy(batch_dict, Voxel_for_geo)

        if self.training and self.front_surface_depth:
            batch_dict = self.forward_front_surface_depth_head(batch_dict, Voxel_for_geo, calibs_Proj)
        return batch_dict

    def forward_voxel_occupancy(self, batch_dict, Voxel):
        VoxelOccupancy = self.pred_voxel(self.pred_occupancy, Voxel)
        batch_dict['voxel_occupancy'] = VoxelOccupancy

        return batch_dict

    def forward_front_surface_depth_head(self, batch_dict, Voxel, calibs_Proj):
        coordinates_psv = self.coordinates_psv.cuda().half() if self.use_amp else self.coordinates_3d.cuda()
        dim1, dim2, dim3, _ = coordinates_psv.shape
        coordinates_psv = coordinates_psv.view(-1, 3)
        N = len(calibs_Proj)

        coordinates_psv_to_pseudo_lidars = []
        for i in range(N):
            coordinates_psv_to_pseudo_lidar = unproject_image_to_pseudo_lidar(coordinates_psv, calibs_Proj[i].float().cuda())
            if 'random_T' in batch_dict:
                inv_random_T = batch_dict['inv_random_T'][i]
                coordinates_psv_to_pseudo_lidar = torch.matmul(coordinates_psv_to_pseudo_lidar, inv_random_T[:3, :3].T) + inv_random_T[:3, 3]
            coordinates_psv_to_pseudo_lidars.append(coordinates_psv_to_pseudo_lidar)
        coordinates_psv_to_pseudo_lidars = torch.stack(coordinates_psv_to_pseudo_lidars, axis=0)
        norm_coordinates_psv_to_3d = (coordinates_psv_to_pseudo_lidars - torch.as_tensor([self.X_MIN, self.Y_MIN, self.Z_MIN], device=coordinates_psv_to_pseudo_lidars.device)) / torch.as_tensor(
            [self.X_MAX - self.X_MIN, self.Y_MAX - self.Y_MIN, self.Z_MAX - self.Z_MIN], device=coordinates_psv_to_pseudo_lidars.device)
        norm_coordinates_psv_to_3d = norm_coordinates_psv_to_3d.half() if self.use_amp else norm_coordinates_psv_to_3d
        norm_coordinates_psv_to_3d = norm_coordinates_psv_to_3d * 2 - 1.
        norm_coordinates_psv_to_3d = norm_coordinates_psv_to_3d.view(N, dim1, dim2, dim3, 3)

        PSV_from_3dgv = F.grid_sample(Voxel, norm_coordinates_psv_to_3d)
        batch_dict['depth_preds'] = []
        if not self.training:
            batch_dict['depth_preds_local'] = []
        batch_dict['depth_volumes'] = []
        batch_dict['depth_samples'] = self.depth.clone().detach().cuda()
        upcost_i, cost_softmax_i, pred_i = self.pred_depth(self.pred_stereo[0], PSV_from_3dgv, batch_dict['left_img'].shape[2:])
        batch_dict['depth_volumes'].append(upcost_i)
        batch_dict['depth_preds'].append(pred_i)
        if not self.training:
            batch_dict['depth_preds_local'].append(self.get_local_depth(cost_softmax_i))

        return batch_dict
