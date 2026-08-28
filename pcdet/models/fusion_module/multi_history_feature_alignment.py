import torch
import torch.nn as nn

from .feature_alignment import FeatureAlignment
from .submodule import BaseConv


class MultiHistoryFeatureAlignment(FeatureAlignment):
    """
    Multi-history extension of StreamDSGN FFF.

    History:
        t-3, t-2, t-1

    Current:
        t

    Pseudo future:
        t+1

    Important:
        Only t-1 and t are used by the original FFF feature-flow module
        to estimate pseudo-(t+1).

        t-2 and t-3 are ONLY used in the later temporal feature fusion.

    Feature dimensions for the default StreamDSGN:
        each BEV: 96 channels
        each temporal branch: 96 -> 32
        five branches: 5 * 32 = 160
        final temporal fusion: 160 -> 96
        residual: + current BEV
    """

    def __init__(self, model_cfg, input_channels):
        # Build all original FFF components:
        # feature-flow matching, warping, pooling, optional attention, etc.
        super().__init__(
            model_cfg=model_cfg,
            input_channels=input_channels
        )

        if self.fusion_type != 'avg':
            raise ValueError(
                "MultiHistoryFeatureAlignment currently only supports "
                "FUSION_TYPE='avg'."
            )

        self.num_history = len(self.history_tag)

        if self.num_history != 3:
            raise ValueError(
                "MultiHistoryFeatureAlignment is designed for exactly "
                "three history BEVs: ['prev3', 'prev2', 'prev']. "
                f"Current HISTORY_TAG={self.history_tag}"
            )

        # ------------------------------------------------------------------
        # Preserve the ORIGINAL StreamDSGN FFF branch capacity.
        #
        # Original:
        #   t-1       : 96 -> 32
        #   t         : 96 -> 32
        #   pseudo t+1: 96 -> 32
        #
        # We DO NOT reduce these three branches to 19 or 24 channels just
        # because more history frames are added.
        # ------------------------------------------------------------------
        self.branch_channels = self.model_cfg.get(
            'BRANCH_CHANNELS',
            input_channels // 3
        )

        if self.branch_channels * 3 != input_channels:
            raise ValueError(
                "To preserve original FFF capacity, "
                "3 * BRANCH_CHANNELS must equal input_channels. "
                f"Got branch_channels={self.branch_channels}, "
                f"input_channels={input_channels}"
            )

        # Replace FeatureAlignment's original history-count-dependent:
        #
        #   input_channels // (2 + len(history_tag))
        #
        # with the ORIGINAL fixed StreamDSGN projection:
        #
        #   96 -> 32
        #
        # Keep the attribute name 'fusion_layer' unchanged so the released
        # StreamDSGN checkpoint can directly load these weights.
        self.fusion_layer = BaseConv(
            input_channels,
            self.branch_channels,
            ksize=1,
            stride=1,
        )

        # ------------------------------------------------------------------
        # Temporal branches:
        #
        #   t-3        32
        #   t-2        32
        #   t-1        32
        #   t          32
        #   pseudo t+1 32
        #
        #             = 160
        #
        # Then compress 160 -> 96.
        # ------------------------------------------------------------------
        multi_in_channels = (
            self.num_history + 2
        ) * self.branch_channels

        self.multi_history_fusion = nn.Conv2d(
            multi_in_channels,
            input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Initialize the new layer so that the model initially behaves
        # exactly like original StreamDSGN:
        #
        # ignore t-3 and t-2,
        # directly preserve [t-1, t, pseudo-t+1].
        self._init_as_original_fff()

    @torch.no_grad()
    def _init_as_original_fff(self):
        """
        Concatenation order:

            [t-3, t-2, t-1, t, pseudo-(t+1)]

        Each branch has branch_channels=32.

        Initial 160 -> 96 convolution:

            t-3 : ignored
            t-2 : ignored

            [t-1, t, pseudo-(t+1)] -> identity

        Therefore, immediately after loading the original checkpoint,
        the network starts from exactly the original FFF representation.
        """

        weight = self.multi_history_fusion.weight
        weight.zero_()

        # For three histories:
        #
        # input:
        # 0:32    -> t-3
        # 32:64   -> t-2
        # 64:96   -> t-1
        # 96:128  -> t
        # 128:160 -> pseudo t+1
        #
        # Original FFF core starts at channel 64.
        core_start = (
            self.num_history - 1
        ) * self.branch_channels

        if self.num_bev_features != (
            3 * self.branch_channels
        ):
            raise RuntimeError(
                "Original FFF core channel count is inconsistent."
            )

        idx = torch.arange(
            self.num_bev_features,
            device=weight.device
        )

        weight[
            idx,
            core_start + idx,
            0,
            0
        ] = 1.0

    def _reduce_history_with_padding(
        self,
        history_features,
        name,
        ref_feature
    ):
        """
        Always return exactly three history branches:

            [t-3, t-2, t-1]

        At the beginning of a scene there may be fewer histories.

        Examples:

            only t-1:
                [0, 0, t-1]

            t-2, t-1:
                [0, t-2, t-1]

            t-3, t-2, t-1:
                [t-3, t-2, t-1]
        """

        history_features = list(
            history_features
        )[-self.num_history:]

        reduced_history = [
            self.fusion_layer(
                item[1][name]
            )
            for item in history_features
        ]

        num_missing = (
            self.num_history
            - len(reduced_history)
        )

        if num_missing > 0:
            zero_history = [
                torch.zeros_like(ref_feature)
                for _ in range(num_missing)
            ]

            reduced_history = (
                zero_history
                + reduced_history
            )

        return reduced_history

    def forward(self, batch_dict):
        assert len(
            self.fusion_features_name
        ) == 1, (
            'Only support one fusion feature!'
        )

        history_features = batch_dict[
            'history_features'
        ]

        # First frame of a scene:
        # preserve original behavior and do not perform FFF.
        if len(history_features) == 0:
            return batch_dict

        name = self.fusion_features_name[0]

        # Current BEV:
        #
        # [B, 96, H, W]
        cur_feature = batch_dict[name]

        # ================================================================
        # Part 1:
        # Original StreamDSGN FFF feature-flow.
        #
        # IMPORTANT:
        # ONLY t-1 and t participate here.
        # ================================================================

        if not self.do_warping:
            adjusted_feature = cur_feature

        else:
            # Always use the MOST RECENT history:
            #
            # t-1
            last_feature = history_features[
                -1
            ][1][name]

            if (
                self.shift_range_last.device
                != last_feature.device
            ):
                self.shift_range_last = (
                    self.shift_range_last.to(
                        last_feature.device
                    )
                )

            adjusted_feature, _ = (
                self.get_pseudo_future_feature(
                    last_feature,
                    cur_feature,
                    self.shift_range_last
                )
            )

        # adjusted_feature is pseudo-(t+1)
        if self.use_conv_trans:
            adjusted_feature = (
                self.conv_trans(
                    adjusted_feature
                )
            )

        if self.use_kd is not None:
            batch_dict[
                'student_feature'
            ] = adjusted_feature

        # ================================================================
        # Part 2:
        # Reduce every temporal branch from 96 -> 32.
        # ================================================================

        # current t
        cur_reduced = self.fusion_layer(
            cur_feature
        )

        # pseudo t+1
        pseudo_reduced = self.fusion_layer(
            adjusted_feature
        )

        # t-3, t-2, t-1
        history_reduced = (
            self._reduce_history_with_padding(
                history_features,
                name,
                ref_feature=cur_reduced
            )
        )

        # Final temporal order:
        #
        # t-3
        # t-2
        # t-1
        # t
        # pseudo-(t+1)
        reduction_features = (
            history_reduced
            + [
                cur_reduced,
                pseudo_reduced
            ]
        )

        # Keep compatibility with the optional
        # spatial-attention implementation.
        if self.attn_cfg is not None:

            if len(self.spatial_attn) == 1:
                attn_modules = (
                    [self.spatial_attn[0]]
                    * len(reduction_features)
                )

            else:
                attn_modules = [
                    self.spatial_attn[
                        min(
                            i,
                            len(self.spatial_attn) - 1
                        )
                    ]
                    for i in range(
                        len(reduction_features)
                    )
                ]

            reduction_features = [
                feat * attn(feat)
                for feat, attn
                in zip(
                    reduction_features,
                    attn_modules
                )
            ]

        # ================================================================
        # Part 3:
        #
        # 5 * 32 = 160
        #        ->
        # 96
        # ================================================================

        fused_features = torch.cat(
            reduction_features,
            dim=1
        )

        expected_channels = (
            self.num_history + 2
        ) * self.branch_channels

        if (
            fused_features.shape[1]
            != expected_channels
        ):
            raise RuntimeError(
                "Unexpected temporal fusion "
                f"channels: "
                f"{fused_features.shape[1]} "
                f"!= {expected_channels}"
            )

        fused_features = (
            self.multi_history_fusion(
                fused_features
            )
        )

        # ================================================================
        # Part 4:
        # Preserve original StreamDSGN current-BEV residual.
        # ================================================================

        fused_features = (
            fused_features
            + cur_feature
        )

        batch_dict[name] = fused_features

        return batch_dict
