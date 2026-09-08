import random

import torch

from .stream import STREAM


class TRANSTREAMING_STREAM(STREAM):
    """
    StreamDSGN stereo trunk
        -> Transtreaming BEV TAT
        -> shared VANBackbone
        -> shared StreamDetHead

    Training:
        one shared model predicts multiple future horizons.
        No H1/H2/H3 sibling heads.
    """

    def __init__(
        self,
        model_cfg,
        num_class,
        dataset,
    ):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            dataset=dataset,
        )

        self.ts_cfg = model_cfg.get(
            'TRANSTREAMING',
            None,
        )

        if self.ts_cfg is None:
            raise ValueError(
                'MODEL.TRANSTREAMING '
                'configuration is required'
            )

        self.freeze_feature_extractor = bool(
            self.ts_cfg.get(
                'FREEZE_FEATURE_EXTRACTOR',
                True,
            )
        )

        if self.freeze_feature_extractor:
            for module in self.feature_extractor:
                for param in module.parameters():
                    param.requires_grad_(False)

                if (
                    module
                    not in
                    self.model_info_dict[
                        'fixed_module_list'
                    ]
                ):
                    self.model_info_dict[
                        'fixed_module_list'
                    ].append(module)

    def train(
        self,
        mode=True,
    ):
        super().train(mode)

        if (
            mode
            and
            self.freeze_feature_extractor
        ):
            for module in self.feature_extractor:
                module.eval()

        return self

    @staticmethod
    def _offset_to_tag(offset):
        offset = int(offset)

        if offset == 0:
            return 'token'

        if offset == 1:
            return 'next'

        if offset > 1:
            return f'next{offset}'

        step = -offset

        if step == 1:
            return 'prev'

        return f'prev{step}'

    def _choose_past_pattern(self):
        patterns = self.ts_cfg.get(
            'TRAIN_PAST_PATTERNS',
            [
                [-3, -2, -1],
            ],
        )

        weights = self.ts_cfg.get(
            'PAST_PATTERN_WEIGHTS',
            None,
        )

        patterns = [
            list(map(int, x))
            for x in patterns
        ]

        if weights is None:
            return random.choice(
                patterns
            )

        weights = [
            float(x)
            for x in weights
        ]

        return random.choices(
            patterns,
            weights=weights,
            k=1,
        )[0]

    def _training_future_steps(self):
        steps = self.ts_cfg.get(
            'TRAIN_FUTURE_STEPS',
            [1, 2, 4, 8],
        )

        steps = sorted(
            set(
                int(x)
                for x in steps
            )
        )

        if (
            len(steps) == 0
            or
            steps[0] <= 0
        ):
            raise ValueError(
                'TRAIN_FUTURE_STEPS '
                'must be positive'
            )

        return steps

    def forward_train(
        self,
        batch_dict,
    ):
        cur_data = batch_dict[
            'token'
        ]

        # ============================================================
        # 1. Shared expensive stereo feature extractor
        # ============================================================

        for module in self.feature_extractor:
            cur_data = module(
                cur_data
            )

        # ============================================================
        # 2. Mixed-speed historical feature sampling
        # ============================================================

        pattern = (
            self._choose_past_pattern()
        )

        available_offsets = []

        # Make sure this training iteration
        # starts with a clean queue.
        if self.history_feature_queue is not None:
            self.history_feature_queue.clear()

        for offset in pattern:
            tag = self._offset_to_tag(
                offset
            )

            history_frame = batch_dict.get(
                tag,
                None,
            )

            if history_frame is None:
                continue

            self.obtain_history_feature(
                history_frame
            )

            available_offsets.append(
                int(offset)
            )

        cur_data[
            'history_features'
        ] = self.history_feature_queue

        cur_data[
            'transtreaming_past_offsets'
        ] = available_offsets

        # ============================================================
        # 3. Dynamic future proposal set during training
        # ============================================================

        future_steps = (
            self._training_future_steps()
        )

        # ------------------------------------------------------------
        # CRITICAL:
        #
        # Snapshot every future supervision BEFORE we start aliasing
        # dense_head.box3d_supervision.
        #
        # dense_head.box3d_supervision is configured as 'next8'.
        # The old code did:
        #
        #   H4: batch_dict['next8'] = batch_dict['next4']
        #   H8: target = batch_dict['next8']
        #
        # Therefore H8 accidentally reused H4 GT.
        # ------------------------------------------------------------
        future_targets = {}

        for step in future_steps:
            target_tag = self._offset_to_tag(
                step
            )

            future_targets[
                step
            ] = batch_dict.get(
                target_tag,
                None,
            )

        cur_data[
            'transtreaming_future_offsets'
        ] = future_steps

        # ============================================================
        # 4. TAT: one module -> multiple future BEV features
        # ============================================================

        for module in self.fusion_module:
            cur_data = module(
                cur_data
            )

        future_bev = cur_data[
            'transtreaming_future_spatial_features'
        ]

        if future_bev.ndim != 5:
            raise RuntimeError(
                'expected future BEV '
                '[B, TF, C, H, W], '
                f'got {future_bev.shape}'
            )

        if (
            future_bev.shape[1]
            !=
            len(future_steps)
        ):
            raise RuntimeError(
                'future BEV count mismatch'
            )

        # ============================================================
        # 5. ONE shared VANBackbone + ONE shared StreamDetHead
        #    evaluated for every future feature.
        # ============================================================

        supervision_key = (
            self.dense_head.box3d_supervision
        )

        sentinel = object()

        old_supervision = batch_dict.get(
            supervision_key,
            sentinel,
        )

        old_token = batch_dict[
            'token'
        ]

        total_loss = None
        valid_horizons = 0
        tb_dict = {}
        disp_dict = {}

        base_data = dict(
            cur_data
        )

        try:
            for future_idx, step in enumerate(
                future_steps
            ):
                target_tag = (
                    self._offset_to_tag(
                        step
                    )
                )

                # IMPORTANT:
                # use the immutable supervision snapshot.
                # Do NOT read batch_dict[target_tag] here because
                # batch_dict['next8'] is temporarily used as the
                # StreamDetHead supervision alias.
                target = future_targets.get(
                    step,
                    None,
                )

                if target is None:
                    continue

                branch_data = dict(
                    base_data
                )

                branch_data[
                    'spatial_features'
                ] = future_bev[
                    :,
                    future_idx,
                ]

                # Avoid stale values if another branch
                # has already run.
                for key in [
                    'spatial_features_2d',
                    'batch_cls_preds',
                    'batch_box_preds',
                    'cls_preds_normalized',
                    'reg_features',
                ]:
                    branch_data.pop(
                        key,
                        None,
                    )

                for module in (
                    self.after_fusion_blocks
                ):
                    branch_data = module(
                        branch_data
                    )

                batch_dict[
                    'token'
                ] = branch_data

                # Reuse one fixed supervision key
                # inside StreamDetHead, but alias it
                # to the current future target.
                batch_dict[
                    supervision_key
                ] = target

                loss_i, tb_i = (
                    self.dense_head.get_loss(
                        batch_dict,
                        {},
                    )
                )

                if total_loss is None:
                    total_loss = loss_i
                else:
                    total_loss = (
                        total_loss
                        +
                        loss_i
                    )

                valid_horizons += 1

                tb_dict[
                    f'ts_h{step}_loss'
                ] = float(
                    loss_i.detach().item()
                )

                for key, value in (
                    tb_i.items()
                ):
                    if isinstance(
                        value,
                        (int, float),
                    ):
                        tb_dict[
                            f'ts_h{step}_{key}'
                        ] = value

            if valid_horizons == 0:
                raise RuntimeError(
                    'no valid Transtreaming '
                    'future supervision'
                )

            total_loss = (
                total_loss
                /
                float(valid_horizons)
            )

            tb_dict[
                'ts_valid_horizons'
            ] = valid_horizons

            tb_dict[
                'ts_total_loss'
            ] = float(
                total_loss.detach().item()
            )

            tb_dict[
                'ts_history_count'
            ] = len(
                available_offsets
            )

        finally:
            batch_dict[
                'token'
            ] = old_token

            if (
                old_supervision
                is
                sentinel
            ):
                batch_dict.pop(
                    supervision_key,
                    None,
                )
            else:
                batch_dict[
                    supervision_key
                ] = old_supervision

            if (
                self.history_feature_queue
                is not None
            ):
                self.history_feature_queue.clear()

        return (
            {
                'loss': total_loss,
            },
            tb_dict,
            disp_dict,
        )
