import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict

from pcdet.models.fusion_module.submodule import BaseConv
from pcdet.ops.feature_flow import feature_flow_utils


def deconv_module(in_channels, out_channels, ksize, stride, padding, output_padding, gn=False, use_norm=True, use_act=True):
    if use_norm:
        norm = nn.BatchNorm2d(
            out_channels) if not gn else nn.GroupNorm(32, out_channels)
    else:
        norm = nn.Identity()
    act = nn.ReLU() if use_act else nn.Identity()
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, ksize,
                           stride, padding, output_padding, bias=False),
        norm,
        act
    )


def warp_feature(x, flow):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow

    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow

    """

    if x.dtype == torch.float16:
        use_amp = True
        x = x.float()
        flow = flow.float()
    else:
        use_amp = False

    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.cuda()

   # warp cur to last
    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0
    vgrid = vgrid.permute(0, 2, 3, 1)  # B H,W,C
    x_warp = F.grid_sample(x, vgrid, padding_mode='zeros')
    if x.is_cuda:
        mask = torch.ones(x.size(), requires_grad=False).cuda()
    else:
        mask = torch.ones(x.size(), requires_grad=False)  # .cuda()
    mask = F.grid_sample(mask, vgrid)
    mask = (mask >= 1.0).float()

    if use_amp:
        x_warp = x_warp.half()
        mask = mask.half()

    return x_warp * mask

class SptialAttention(nn.Module):
    def __init__(self, model_cfg):
        super(SptialAttention, self).__init__()
        self.model_cfg = model_cfg
        padding = 3 if self.model_cfg.kernel_size == 7 else 1
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=self.model_cfg.kernel_size,
                                      stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        max_x = torch.max(x, dim=1, keepdim=True)[0]
        avg_x = torch.mean(x, dim=1, keepdim=True)
        x = torch.cat([max_x, avg_x], dim=1)
        x = self.spatial_conv(x)
        x = self.sigmoid(x)
        return x


class FeatureAlignment(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.fusion_type = self.model_cfg.get('FUSION_TYPE', 'avg')
        self.GN = self.model_cfg.get('GN', False)
        self.use_deform = self.model_cfg.get('USE_DEFORM', False)
        self.history_tag = self.model_cfg.get(
            'HISTORY_TAG', None)
        self.fusion_features_name = self.model_cfg.get(
            'HISTORY_FEATURES_NAME', None)
        assert self.history_tag is not None, 'You need to specify HISTORY_TAG!'
        assert self.fusion_features_name is not None, 'You need to specify HISTORY_FEATURES_NAME!'

        self.use_kd = self.model_cfg.get('USE_KD', False)
        self.do_warping = self.model_cfg.get('DO_WARPING', True)

        matching_range = self.model_cfg.get('MATCHING_RANGE', [-3, 3, -3, 3])
        self.shift_range_last = self.get_shift_range(
            matching_range, reverse=False)

        self.with_grad_in_aligned_feature = self.model_cfg.get(
            'WITH_GRAD_IN_ALIGNED_FEATURE', True)

        if self.fusion_type == 'avg':
            self.fusion_layer = BaseConv(
                input_channels,
                input_channels // (2 + len(self.history_tag)),
                ksize=1,
                stride=1,
            )
        elif self.fusion_type == 'dil':
            self.fusion_layer_cur = BaseConv(
                input_channels,
                input_channels // 2,
                ksize=1,
                stride=1,
            )
            self.fusion_layer = BaseConv(
                input_channels,
                # +1 for the next frame
                input_channels // (2 * (len(self.history_tag)+1)),
                ksize=1,
                stride=1,
            )

        self.pool_cfg = self.model_cfg.get('POOL_CFG', None)
        assert self.pool_cfg is not None, 'You need to specify POOL_CFG!'
        self.pool = nn.MaxPool2d(self.pool_cfg.factor, self.pool_cfg.factor) if self.pool_cfg.type == 'max' else nn.AvgPool2d(
            self.pool_cfg.factor, self.pool_cfg.factor)
        self.factor = self.pool_cfg.factor
        self.feature_padding = self.model_cfg.get('FEATURE_PADDING', True)

        self.interp_type = self.model_cfg.get('INTERP_TYPE', 'nearest')
        self.interp_coord = nn.Upsample(
            scale_factor=self.factor, mode=self.interp_type)

        matching_threshold_factor = self.model_cfg.get(
            'MATCHING_THRESHOLD_FACTOR', 0.8)
        self.matching_threshold = 1 / \
            (len(self.shift_range_last) * matching_threshold_factor)

        self.pixel_level_warping = self.model_cfg.get(
            'PIXEL_LEVEL_WARPING', False)
        if self.pixel_level_warping:
            self.warp_feature = feature_flow_utils.feature_adjusting
        else:
            self.warp_feature = warp_feature

        self.attn_cfg = self.model_cfg.get('ATTN_CFG', None)
        if self.attn_cfg is not None:
            if isinstance(self.attn_cfg, list):
                self.spatial_attn = nn.ModuleList()
                for attn_cfg in self.attn_cfg:
                    self.spatial_attn.append(SptialAttention(attn_cfg))
            elif isinstance(self.attn_cfg, dict):
                self.spatial_attn = nn.ModuleList()
                for _ in range(len(self.history_tag) + 2): # history, current, pseudo future
                    self.spatial_attn.append(SptialAttention(self.attn_cfg))
        
        self.use_conv_trans = self.model_cfg.get('USE_CONV_TRANS', False)
        if self.use_conv_trans:
            self.conv_trans = BaseConv(input_channels, input_channels, ksize=3, stride=1, padding=1, use_norm=False)

        self.num_bev_features = input_channels

    def cuda(self, device=None):
        model = super().cuda(device)
        try:
            device = next(self.parameters()).device
            if hasattr(self, "shift_range_last"):
                self.shift_range_last = self.shift_range_last.to(device)
            if hasattr(self, "shift_range_next"):
                self.shift_range_next = self.shift_range_next.to(device)
            return model
        except StopIteration:
            return model

    def get_shift_range(self, matching_range, reverse=False):
        '''
            reverse: bool indicating if calculating the deviation of the next frame from the current frame (True);
                     or the deviation of the previous frame from the current frame (False).
            if reverse is True, the shift range will be from right to left, bottom to top;

        '''
        left, right, top, bottom = matching_range
        if reverse:
            shift_range = torch.tensor([(i, j) for i in range(right, left-1, -1)
                                        for j in range(bottom, top-1, -1)])
        else:
            shift_range = torch.tensor([(i, j) for i in range(left, right+1, 1)
                                        for j in range(top, bottom+1, 1)])
        return shift_range

    def get_pseudo_future_feature(self, last_feature, cur_feature, matching_range):
        last_feature_down = self.pool(last_feature)
        cur_feature_down = self.pool(cur_feature)
        matching_dist = feature_flow_utils.offset_matching_gpu(
            last_feature_down, cur_feature_down, matching_range)

        shift_coord = feature_flow_utils.get_shift_coord(
            matching_dist, matching_range, self.matching_threshold).float()
        shift_coord = self.interp_coord(shift_coord) * self.factor
        if self.pixel_level_warping:
            shift_coord[:, [0, 1], :, :] = shift_coord[:, [1, 0], :, :]
            shift_coord = shift_coord.long()
            adjusted_feature = feature_flow_utils.feature_adjusting(
                cur_feature, shift_coord, self.feature_padding, self.with_grad_in_aligned_feature)
        else:
            adjusted_feature = self.warp_feature(cur_feature, -shift_coord)

        # adjusted_feature = feature_flow_utils.feature_adjusting(
        #     cur_feature, shift_coord, self.feature_padding, self.with_grad_in_aligned_feature)
        return adjusted_feature, shift_coord

    def forward(self, batch_dict):
        assert len(
            self.fusion_features_name) == 1, 'Only support one fusion feature!'
        history_features = batch_dict['history_features']
        if len(history_features) == 0:
            return batch_dict
        name = self.fusion_features_name[0]
        cur_feature = batch_dict[name]

        if self.fusion_type == 'only_align':
            if not self.do_warping:
                adjusted_feature = cur_feature
            else:
                last_feature = history_features[-1][1][name]
                if self.shift_range_last.device != last_feature.device:
                    self.shift_range_last = self.shift_range_last.to(
                        last_feature.device)
                adjusted_feature, feature_flow = self.get_pseudo_future_feature(
                    last_feature, cur_feature, self.shift_range_last)
            if self.use_conv_trans:
                adjusted_feature = self.conv_trans(adjusted_feature)

            batch_dict[name] = adjusted_feature
            if self.use_kd is not None:
                batch_dict['student_feature'] = adjusted_feature
            return batch_dict

        reduction_features = []
        for i in range(len(history_features)):
            reduction_features.append(
                self.fusion_layer(history_features[i][1][name]))

        if self.fusion_type == 'avg':
            reduction_features.append(self.fusion_layer(cur_feature))
        elif self.fusion_type == 'dil':
            reduction_features.append(self.fusion_layer_cur(cur_feature))

        if not self.do_warping:
            adjusted_feature = cur_feature
        else:
            last_feature = history_features[-1][1][name]
            adjusted_feature, feature_flow = self.get_pseudo_future_feature(
                last_feature, cur_feature, self.shift_range_last)
        if self.use_conv_trans:
            adjusted_feature = self.conv_trans(adjusted_feature)

        if self.use_kd is not None:
            batch_dict['student_feature'] = adjusted_feature

        adjusted_feature_reduced = self.fusion_layer(adjusted_feature)
        reduction_features.append(adjusted_feature_reduced)

        if len(self.history_tag) == 2 and len(history_features) == 1:
            reduction_features.append(adjusted_feature_reduced)

        if self.attn_cfg is not None:
            spatial_attn = self.spatial_attn[0]
            for i in range(len(reduction_features)):
                reduction_features[i] = reduction_features[i] * \
                    spatial_attn(reduction_features[i])

        fused_features = torch.cat(reduction_features, dim=1)
        fused_features += cur_feature
        batch_dict[name] = fused_features

        # for save
        # scene = batch_dict['scene'][0]
        # sample_idx = batch_dict['this_sample_idx'][0]
        # save_name = scene + '_' + sample_idx
        # save_name = './debug/features/' + save_name
        # save_warped_name = save_name + '_warped'
        # save_flow_name = save_name + '_flow'

        # save_feature = cur_feature.detach().cpu().numpy()
        # save_warped_feature = adjusted_feature.detach().cpu().numpy()
        # save_flow_map = feature_flow.detach().cpu().numpy()
        # import numpy as np
        # np.save(save_name, save_feature)
        # np.save(save_warped_name, save_warped_feature)
        # np.save(save_flow_name, save_flow_map)

        return batch_dict


if __name__ == '__main__':
    import easydict as edict
    import numpy as np

    model_cfg = edict.EasyDict({
        'FUSION_TYPE': 'only_align',
        'GN': False,
        'HISTORY_TAG': ['prev'],
        'HISTORY_FEATURES_NAME': ['spatial_features'],
        'MATCHING_RANGE': [-3, 3, -3, 3],
        'POOL_CFG': {
            'type': 'max',
            'factor': 4,
        },
        'INTERP_TYPE': 'nearest',
    })

    batch_dict = {}
    # batch_dict['history_features'] = [
    #     (idx, {'spatial_features': torch.randn(size=(1, 96, 288, 256)).cuda().half()}) for idx in range(len(model_cfg.HISTORY_TAG))]
    # batch_dict['spatial_features'] = torch.randn(
    #     size=(1, 96, 288, 256)).cuda().half()
    history_features = torch.from_numpy(
        np.load('debug/features/0001_7_000306.npy')).cuda().half()
    spatial_features = torch.from_numpy(
        np.load('debug/features/0001_7_000307.npy')).cuda().half()

    batch_dict['history_features'] = [
        (0, {'spatial_features': history_features})]
    batch_dict['spatial_features'] = spatial_features

    from thop import profile, clever_format
    model = FeatureAlignment(
        model_cfg=model_cfg, input_channels=96).cuda().half()

    flops, params = profile(model, inputs=(batch_dict,))
    flops, params = clever_format([flops, params], "%.3f")
    print(f"计算量：{flops}")
    print(f"参数量：{params}")

    import time
    spend_time = []
    for i in range(500):
        torch.cuda.synchronize()
        t1 = time.time()
        batch_dict = model(batch_dict)
        torch.cuda.synchronize()
        t2 = time.time()
        print(t2-t1)
        spend_time.append(t2 - t1)
    print(f"平均耗时：{sum(spend_time[1:]) / len(spend_time[1:])}")
