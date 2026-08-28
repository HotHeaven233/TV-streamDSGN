import torch
import torch.nn as nn

from pcdet.models.fusion_module.feature_alignment import FeatureAlignment
from pcdet.models.fusion_module.submodule import BaseConv


class MultiHistoryResidualFeatureAlignment(FeatureAlignment):
    """
    Baseline-preserving multi-history fusion.

    Main branch:
        exactly preserve original StreamDSGN FFF:

            H_{t-1} ----\
            B_t ----------> original FFF -> F_base
            P_{t+1} ----/

    Residual history branch:
        H_{t-3} -> 96 -> 32 --\
        H_{t-2} -> 96 -> 32 ---- concat -> 96 -> 96 -> Delta
        H_{t-1} -> 96 -> 32 --/

        F_out = F_base + Delta

    The final 96->96 adapter convolution is ZERO initialized.

    Therefore at initialization:

        Delta = 0
        F_out = F_base

    i.e. the network exactly falls back to original StreamDSGN.
    """

    def __init__(self, model_cfg, input_channels):
        super().__init__(
            model_cfg=model_cfg,
            input_channels=input_channels,
        )

        if self.fusion_type != 'avg':
            raise NotImplementedError(
                "MultiHistoryResidualFeatureAlignment "
                "currently supports only FUSION_TYPE='avg'."
            )

        if len(self.history_tag) != 3:
            raise ValueError(
                "This module requires exactly three history tags: "
                "['prev3', 'prev2', 'prev'], "
                f"but got {self.history_tag}"
            )

        if self.attn_cfg is not None:
            raise NotImplementedError(
                "First residual baseline does not use ATTN_CFG."
            )

        self.num_history = 3

        self.branch_channels = int(
            self.model_cfg.get(
                'BRANCH_CHANNELS',
                32,
            )
        )

        # Original StreamDSGN FFF has:
        #
        #   t-1       : 32
        #   current   : 32
        #   pseudo t+1: 32
        #
        # total = 96, followed directly by current residual.
        if (
            self.branch_channels * 3
            !=
            input_channels
        ):
            raise ValueError(
                "For baseline-preserving FFF, "
                "3 * BRANCH_CHANNELS must equal input_channels. "
                f"Got 3 * {self.branch_channels} "
                f"!= {input_channels}."
            )

        # ------------------------------------------------------------
        # IMPORTANT:
        #
        # FeatureAlignment.__init__ sees HISTORY_TAG length = 3,
        # therefore it would originally construct 96 -> 19.
        #
        # Replace it with the ORIGINAL StreamDSGN projection:
        #
        #       96 -> 32
        #
        # Keep the same attribute name `fusion_layer`, so the released
        # StreamDSGN checkpoint can load its original FFF weights into it.
        # ------------------------------------------------------------

        self.fusion_layer = BaseConv(
            input_channels,
            self.branch_channels,
            ksize=1,
            stride=1,
        )

        # ------------------------------------------------------------
        # Three-history residual adapter.
        #
        # Separate from fusion_layer deliberately:
        #
        # - fusion_layer = protected original FFF path
        # - history_adapter_proj = new t-3/t-2/t-1 path
        #
        # No BN here because batch size is 1 and this is a lightweight
        # adapter. This also avoids additional BN running statistics.
        # ------------------------------------------------------------

        self.history_adapter_proj = BaseConv(
            input_channels,
            self.branch_channels,
            ksize=1,
            stride=1,
            use_norm=False,
            use_act=True,
        )

        # 3 * 32 = 96 -> 96
        self.history_adapter_fusion = nn.Conv2d(
            self.num_history * self.branch_channels,
            input_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # ------------------------------------------------------------
        # Exact baseline-preserving initialization.
        #
        # F_out = F_base + 0
        #
        # Do NOT zero history_adapter_proj as well.
        # If both layers were zero, gradients would initially vanish.
        # ------------------------------------------------------------

        nn.init.zeros_(
            self.history_adapter_fusion.weight
        )

        self.num_bev_features = input_channels

    def _build_history_slots(
        self,
        history_features,
    ):
        """
        Return exactly three history slots:

            [t-3, t-2, t-1]

        During the first frames of a scene:

            1 history:
                [None, None, t-1]

            2 histories:
                [None, t-2, t-1]

            3 histories:
                [t-3, t-2, t-1]

        The queue always contains actually processed frames at inference.
        """

        history_features = list(
            history_features
        )

        recent = history_features[
            -self.num_history:
        ]

        slots = [
            None
            for _ in range(
                self.num_history
            )
        ]

        if len(recent) > 0:
            slots[
                -len(recent):
            ] = recent

        return slots

    def forward(
        self,
        batch_dict,
    ):
        assert (
            len(self.fusion_features_name)
            ==
            1
        ), (
            "Only one fusion feature is supported."
        )

        history_features = batch_dict[
            'history_features'
        ]

        # Same numerical semantics as original StreamDSGN:
        # no temporal history -> current BEV must remain unchanged.
        #
        # During adapter-only training all original StreamDSGN parameters
        # are frozen. If a scene-start sample has zero history and we
        # directly return here, the loss has no connection to any
        # trainable parameter, causing:
        #
        #   RuntimeError:
        #   element 0 of tensors does not require grad
        #
        # Therefore attach a ZERO-VALUED graph term involving the two
        # trainable adapter weights.
        #
        # Numerically:
        #
        #   spatial_features_out == spatial_features_in
        #
        # exactly, while backward() remains valid. Gradients to the
        # adapter on this sample are exactly zero.
        if len(history_features) == 0:

            name = self.fusion_features_name[0]

            zero_adapter_graph = (
                self.history_adapter_fusion.weight.sum()
                * 0.0
            )

            zero_adapter_graph = (
                zero_adapter_graph
                +
                self.history_adapter_proj.conv.weight.sum()
                * 0.0
            )

            batch_dict[name] = (
                batch_dict[name]
                +
                zero_adapter_graph
            )

            return batch_dict

        name = self.fusion_features_name[0]

        cur_feature = batch_dict[
            name
        ]

        # ============================================================
        # 1. ORIGINAL STREAMDSGN FFF MAIN BRANCH
        # ============================================================

        # Only the newest history H_{t-1} participates in feature flow.
        last_feature = (
            history_features[-1][1][name]
        )

        if (
            self.shift_range_last.device
            !=
            last_feature.device
        ):
            self.shift_range_last = (
                self.shift_range_last.to(
                    last_feature.device
                )
            )

        if not self.do_warping:
            adjusted_feature = cur_feature
        else:
            (
                adjusted_feature,
                feature_flow,
            ) = self.get_pseudo_future_feature(
                last_feature,
                cur_feature,
                self.shift_range_last,
            )

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

        # Original FFF:
        #
        #   R(H_{t-1})      : 32
        #   R(B_t)          : 32
        #   R(P_{t+1})      : 32
        #
        # concat = 96
        base_parts = [
            self.fusion_layer(
                last_feature
            ),
            self.fusion_layer(
                cur_feature
            ),
            self.fusion_layer(
                adjusted_feature
            ),
        ]

        base_core = torch.cat(
            base_parts,
            dim=1,
        )

        if (
            base_core.shape[1]
            !=
            cur_feature.shape[1]
        ):
            raise RuntimeError(
                "Original FFF channel mismatch: "
                f"base_core={base_core.shape}, "
                f"cur_feature={cur_feature.shape}"
            )

        # Exactly original StreamDSGN residual.
        base_output = (
            cur_feature
            +
            base_core
        )

        # ============================================================
        # 2. THREE-HISTORY RESIDUAL ADAPTER
        # ============================================================

        history_slots = (
            self._build_history_slots(
                history_features
            )
        )

        B, _, H, W = (
            cur_feature.shape
        )

        adapter_parts = []

        for history_slot in history_slots:

            if history_slot is None:

                # Missing history at scene start.
                reduced = (
                    cur_feature.new_zeros(
                        (
                            B,
                            self.branch_channels,
                            H,
                            W,
                        )
                    )
                )

            else:

                history_bev = (
                    history_slot[1][name]
                )

                reduced = (
                    self.history_adapter_proj(
                        history_bev
                    )
                )

            adapter_parts.append(
                reduced
            )

        # Explicit order:
        #
        #   [t-3, t-2, t-1]
        adapter_input = torch.cat(
            adapter_parts,
            dim=1,
        )

        expected_channels = (
            self.num_history
            *
            self.branch_channels
        )

        if (
            adapter_input.shape[1]
            !=
            expected_channels
        ):
            raise RuntimeError(
                "History adapter channel mismatch: "
                f"got {adapter_input.shape[1]}, "
                f"expected {expected_channels}"
            )

        # Initially exactly zero.
        history_delta = (
            self.history_adapter_fusion(
                adapter_input
            )
        )

        # ============================================================
        # 3. FINAL OUTPUT
        # ============================================================

        batch_dict[name] = (
            base_output
            +
            history_delta
        )

        return batch_dict
