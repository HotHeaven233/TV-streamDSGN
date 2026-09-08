import random
from collections import deque

import torch

from .stream import STREAM


class TRANSTREAMING_STREAM_V2(STREAM):
    """
    Transtreaming-style StreamDSGN, correctness-fixed V2.

    Key invariants:
      1. history queue capacity == TAT PAST_LENGTH
      2. actual P^P length == actual history feature count
      3. one TAT forward produces all requested future BEV features
      4. all horizons share the same VANBackbone + StreamDetHead
      5. every horizon gets an independent loss_batch
      6. standard offline evaluation is explicitly H=+1
      7. current feature enters history only after current inference completes

    The dedicated streaming evaluator will later override P^P / P^F using
    actual processed sensor positions.  forward_test() here is the ordinary
    contiguous offline-validation path.
    """

    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            dataset=dataset,
        )

        ts_cfg = model_cfg.get('TRANSTREAMING', {})

        fusion_cfg = model_cfg.get(
            'FUSION_IN_SPATIAL_FEATURES',
            {},
        )

        self.ts_past_length = int(
            fusion_cfg.get('PAST_LENGTH', 3)
        )

        if self.ts_past_length <= 0:
            raise ValueError(
                f'PAST_LENGTH must be positive, '
                f'got {self.ts_past_length}'
            )

        # IMPORTANT:
        # HISTORY_TAG contains only runtime capacity semantics.
        # Candidate training tags such as prev8/prev6 are dataset inputs,
        # not queue capacity.
        self.history_feature_queue = deque(
            maxlen=self.ts_past_length
        )

        self.ts_freeze_feature_extractor = bool(
            ts_cfg.get(
                'FREEZE_FEATURE_EXTRACTOR',
                True,
            )
        )

        # Parameter-efficient Transtreaming adaptation:
        #
        #   frozen original StreamDSGN:
        #       stereo feature extractor
        #       VANBackbone
        #       StreamDetHead
        #
        #   trainable:
        #       TranstreamingBEVTAT only
        #
        self.ts_freeze_original_shared = bool(
            ts_cfg.get(
                'FREEZE_ORIGINAL_SHARED',
                False,
            )
        )

        self.ts_train_future_steps = [
            int(x)
            for x in ts_cfg.get(
                'TRAIN_FUTURE_STEPS',
                [1, 2, 4, 8],
            )
        ]

        self.ts_eval_future_steps = [
            int(x)
            for x in ts_cfg.get(
                'EVAL_FUTURE_STEPS',
                [1],
            )
        ]

        raw_patterns = ts_cfg.get(
            'TRAIN_PAST_PATTERNS',
            [
                [-4, -2, -1],
                [-8, -4, -2],
                [-6, -4, -2],
                [-3, -2, -1],
            ],
        )

        self.ts_train_past_patterns = [
            [int(x) for x in pattern]
            for pattern in raw_patterns
        ]

        self.ts_past_pattern_weights = [
            float(x)
            for x in ts_cfg.get(
                'PAST_PATTERN_WEIGHTS',
                [10, 5, 5, 5],
            )
        ]

        if (
            len(self.ts_train_past_patterns)
            != len(self.ts_past_pattern_weights)
        ):
            raise ValueError(
                'TRAIN_PAST_PATTERNS and '
                'PAST_PATTERN_WEIGHTS length mismatch'
            )

        for pattern in self.ts_train_past_patterns:
            if len(pattern) > self.ts_past_length:
                raise ValueError(
                    f'past pattern {pattern} contains '
                    f'{len(pattern)} history frames, '
                    f'but PAST_LENGTH={self.ts_past_length}'
                )

            if any(x >= 0 for x in pattern):
                raise ValueError(
                    f'past offsets must be negative: {pattern}'
                )

        if any(x <= 0 for x in self.ts_train_future_steps):
            raise ValueError(
                'TRAIN_FUTURE_STEPS must all be positive'
            )

        if any(x <= 0 for x in self.ts_eval_future_steps):
            raise ValueError(
                'EVAL_FUTURE_STEPS must all be positive'
            )

        # Ordinary StreamDSGN offline AP must be evaluated at +1 only.
        if self.ts_eval_future_steps != [1]:
            raise ValueError(
                'Standard offline validation must use '
                'EVAL_FUTURE_STEPS=[1]. '
                'Dynamic P^F belongs in the dedicated '
                'streaming evaluator.'
            )

        if self.ts_freeze_feature_extractor:
            for module in self.feature_extractor:
                module.eval()

                for param in module.parameters():
                    param.requires_grad_(False)

        if self.ts_freeze_original_shared:
            # Freeze all original post-fusion StreamDSGN modules.
            #
            # IMPORTANT:
            # only requires_grad=False + eval().
            # Do NOT wrap their forward pass in torch.no_grad(),
            # because gradients with respect to their INPUT must
            # still propagate backward into TranstreamingBEVTAT.
            for module in self.after_fusion_blocks:
                module.eval()

                for param in module.parameters():
                    param.requires_grad_(False)

            # Sanity: TAT itself must remain trainable.
            for module in self.fusion_module:
                for param in module.parameters():
                    param.requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)

        # StreamDetector3DTemplate.train() recursively puts children
        # into train mode.  Restore the frozen stereo trunk to eval.
        if mode and self.ts_freeze_feature_extractor:
            for module in self.feature_extractor:
                module.eval()

        if mode and self.ts_freeze_original_shared:
            # Keep frozen original StreamDSGN modules in eval mode.
            # This is especially important for BN statistics.
            for module in self.after_fusion_blocks:
                module.eval()

            # TAT is the only trainable block.
            for module in self.fusion_module:
                module.train(True)

        return self

    @staticmethod
    def _past_offset_to_tag(offset):
        offset = int(offset)

        if offset >= 0:
            raise ValueError(
                f'past offset must be negative, got {offset}'
            )

        step = -offset

        return 'prev' if step == 1 else f'prev{step}'

    @staticmethod
    def _future_step_to_tag(step):
        step = int(step)

        if step <= 0:
            raise ValueError(
                f'future step must be positive, got {step}'
            )

        return 'next' if step == 1 else f'next{step}'

    def _choose_past_pattern(self):
        return list(
            random.choices(
                self.ts_train_past_patterns,
                weights=self.ts_past_pattern_weights,
                k=1,
            )[0]
        )

    def _run_feature_extractor(self, data):
        if data is None:
            return None

        if self.ts_freeze_feature_extractor:
            with torch.no_grad():
                for module in self.feature_extractor:
                    data = module(data)
        else:
            for module in self.feature_extractor:
                data = module(data)

        return data

    def _snapshot_history_feature(
        self,
        data,
        clone=False,
    ):
        history_feature = {}

        for key in self.history_features_name:
            if key not in data:
                raise KeyError(
                    f'history feature "{key}" missing '
                    f'from feature extractor output'
                )

            value = data[key]

            if torch.is_tensor(value):
                value = value.detach()

                if clone:
                    value = value.clone()

            history_feature[key] = value

        return history_feature

    def _set_temporal_proposals(
        self,
        cur_data,
        past_offsets,
        future_offsets,
    ):
        history_features = cur_data.get(
            'history_features',
            None,
        )

        if history_features is None:
            raise RuntimeError(
                'history_features must be set before TAT'
            )

        if len(history_features) != len(past_offsets):
            raise RuntimeError(
                'Transtreaming P^P/history mismatch: '
                f'len(P^P)={len(past_offsets)} '
                f'but history_count={len(history_features)}; '
                f'P^P={past_offsets}'
            )

        feature_name = self.history_features_name[0]

        if feature_name not in cur_data:
            raise KeyError(
                f'current feature "{feature_name}" missing'
            )

        feature = cur_data[feature_name]

        if not torch.is_tensor(feature):
            raise TypeError(
                f'{feature_name} must be a tensor'
            )

        batch_size = int(feature.shape[0])
        device = feature.device

        past_tensor = torch.tensor(
            past_offsets,
            dtype=torch.long,
            device=device,
        ).reshape(1, -1)

        future_tensor = torch.tensor(
            future_offsets,
            dtype=torch.long,
            device=device,
        ).reshape(1, -1)

        if batch_size > 1:
            past_tensor = past_tensor.expand(
                batch_size,
                -1,
            )

            future_tensor = future_tensor.expand(
                batch_size,
                -1,
            )

        cur_data[
            'transtreaming_past_offsets'
        ] = past_tensor

        cur_data[
            'transtreaming_future_offsets'
        ] = future_tensor

    def _run_tat(self, cur_data):
        for module in self.fusion_module:
            cur_data = module(cur_data)

        future_features = cur_data.get(
            'transtreaming_future_spatial_features',
            None,
        )

        if future_features is None:
            raise RuntimeError(
                'TranstreamingBEVTAT did not produce '
                '"transtreaming_future_spatial_features"'
            )

        if future_features.ndim != 5:
            raise RuntimeError(
                'expected future BEV shape [B,T,C,H,W], '
                f'got {tuple(future_features.shape)}'
            )

        return cur_data, future_features

    def _run_shared_detection_path(
        self,
        fused_data,
        future_feature,
    ):
        # Shallow dict copy is intentional:
        # tensor storage may be shared, but keys written by VAN/head
        # must not leak between horizons.
        horizon_data = dict(fused_data)

        horizon_data['spatial_features'] = future_feature

        # Remove products left by another path defensively.
        horizon_data.pop(
            'spatial_features_2d',
            None,
        )

        for module in self.after_fusion_blocks:
            horizon_data = module(horizon_data)

        return horizon_data

    def forward_train(self, batch_dict):
        # Never allow a stale queue from a previous failed iteration.
        self.history_feature_queue.clear()

        try:
            # ----------------------------------------------------------
            # 1. Current backbone feature
            # ----------------------------------------------------------
            cur_data = self._run_feature_extractor(
                batch_dict['token']
            )

            # ----------------------------------------------------------
            # 2. Sample one dynamic historical proposal P^P.
            #    Only actually available frames enter the queue.
            # ----------------------------------------------------------
            sampled_pattern = self._choose_past_pattern()

            actual_past_offsets = []

            for offset in sampled_pattern:
                tag = self._past_offset_to_tag(
                    offset
                )

                history_data = batch_dict.get(
                    tag,
                    None,
                )

                if history_data is None:
                    continue

                history_data = (
                    self._run_feature_extractor(
                        history_data
                    )
                )

                history_feature = (
                    self._snapshot_history_feature(
                        history_data,
                        clone=False,
                    )
                )

                sample_idx = history_data.get(
                    'this_sample_idx',
                    '',
                )

                self.history_feature_queue.append(
                    (
                        sample_idx,
                        history_feature,
                    )
                )

                actual_past_offsets.append(
                    int(offset)
                )

            # The queue must contain exactly the selected available P^P.
            if (
                len(self.history_feature_queue)
                != len(actual_past_offsets)
            ):
                raise RuntimeError(
                    'internal history queue mismatch: '
                    f'queue={len(self.history_feature_queue)} '
                    f'offsets={actual_past_offsets}'
                )

            # ----------------------------------------------------------
            # Startup / zero-real-history handling.
            #
            # When the sampled frame has no valid historical input,
            # feeding an empty history into TAT may cause the temporal
            # block to degenerate to an identity/current-feature path.
            #
            # In the frozen-shared training setting that makes the whole
            # loss graph non-differentiable:
            #
            #   frozen current BEV
            #       -> TAT bypass
            #       -> frozen VAN/head
            #
            # Use one causal pseudo-history built from the current
            # pre-TAT feature at temporal offset 0.  This is consistent
            # with the V2 inference startup policy and guarantees that
            # the trainable TAT is actually executed.
            # ----------------------------------------------------------
            real_history_count = len(
                actual_past_offsets
            )

            if real_history_count == 0:
                startup_queue = deque(
                    maxlen=self.ts_past_length
                )

                startup_feature = (
                    self._snapshot_history_feature(
                        cur_data,
                        clone=False,
                    )
                )

                startup_queue.append(
                    (
                        cur_data.get(
                            'this_sample_idx',
                            '',
                        ),
                        startup_feature,
                    )
                )

                cur_data[
                    'history_features'
                ] = startup_queue

                tat_past_offsets = [0]

            else:
                cur_data[
                    'history_features'
                ] = self.history_feature_queue

                tat_past_offsets = list(
                    actual_past_offsets
                )

            # ----------------------------------------------------------
            # 3. Snapshot ALL future target references before any
            #    per-horizon loss plumbing.
            # ----------------------------------------------------------
            future_targets = {}

            valid_future_steps = []

            for step in self.ts_train_future_steps:
                tag = self._future_step_to_tag(
                    step
                )

                target = batch_dict.get(
                    tag,
                    None,
                )

                future_targets[step] = target

                if target is not None:
                    valid_future_steps.append(
                        step
                    )

            if len(valid_future_steps) == 0:
                raise RuntimeError(
                    'no valid Transtreaming future '
                    'supervision in this batch'
                )

            # ----------------------------------------------------------
            # 4. One TAT pass -> all future BEV maps
            # ----------------------------------------------------------
            self._set_temporal_proposals(
                cur_data=cur_data,
                past_offsets=tat_past_offsets,
                future_offsets=valid_future_steps,
            )

            cur_data, future_features = (
                self._run_tat(cur_data)
            )

            if (
                future_features.shape[1]
                != len(valid_future_steps)
            ):
                raise RuntimeError(
                    'TAT future count mismatch: '
                    f'feature T={future_features.shape[1]} '
                    f'proposal count={len(valid_future_steps)}'
                )

            # ----------------------------------------------------------
            # 5. SAME VAN + SAME StreamDetHead for every horizon.
            #
            #    IMPORTANT FIX:
            #    Each horizon receives an INDEPENDENT shallow copy of
            #    batch_dict.  We never overwrite shared batch_dict[next8].
            # ----------------------------------------------------------
            if getattr(
                self,
                'dense_head',
                None,
            ) is None:
                raise RuntimeError(
                    'Transtreaming requires dense_head'
                )

            supervision_key = getattr(
                self.dense_head,
                'box3d_supervision',
                None,
            )

            if supervision_key is None:
                supervision_key = (
                    self.model_cfg
                    .DENSE_HEAD
                    .BOX3D_SUPERVISION
                )

            losses = []
            tb_dict = {}

            for future_index, step in enumerate(
                valid_future_steps
            ):
                future_feature = (
                    future_features[
                        :,
                        future_index,
                    ]
                )

                horizon_data = (
                    self._run_shared_detection_path(
                        fused_data=cur_data,
                        future_feature=future_feature,
                    )
                )

                target = future_targets[step]

                # CRITICAL:
                # Do not mutate batch_dict itself.
                loss_batch = dict(batch_dict)

                loss_batch['token'] = (
                    horizon_data
                )

                loss_batch[
                    supervision_key
                ] = target

                # StreamDetHead.forward_ret_dict is shared and gets
                # overwritten by each head call, therefore get_loss()
                # must happen immediately after that horizon's forward.
                loss_h, _ = (
                    self.dense_head.get_loss(
                        loss_batch,
                        {},
                    )
                )

                losses.append(loss_h)

                tb_dict[
                    f'ts_h{step}_loss'
                ] = float(
                    loss_h.detach().item()
                )

            loss = torch.stack(
                losses
            ).mean()

            # Hard fail here with useful temporal diagnostics instead
            # of letting train_utils fail later inside backward().
            if not loss.requires_grad:
                raise RuntimeError(
                    'Transtreaming training loss has no grad_fn: '
                    f'real_history_count={real_history_count}, '
                    f'TAT_P^P={tat_past_offsets}, '
                    f'future_steps={valid_future_steps}. '
                    'This means the trainable TAT was bypassed.'
                )

            tb_dict[
                'ts_total_loss'
            ] = float(
                loss.detach().item()
            )

            tb_dict[
                'ts_valid_horizons'
            ] = float(
                len(valid_future_steps)
            )

            tb_dict[
                'ts_history_count'
            ] = float(
                len(actual_past_offsets)
            )

            ret_dict = {
                'loss': loss,
            }

            disp_dict = {}

            return (
                ret_dict,
                tb_dict,
                disp_dict,
            )

        finally:
            # Training batches are independent temporal samples.
            self.history_feature_queue.clear()

    def _copy_aux_outputs(
        self,
        cur_data,
        pred_dicts,
        ret_dicts,
    ):
        for key in cur_data.keys():
            if not key.startswith(
                'depth_error_'
            ):
                continue

            value = cur_data[key]

            if isinstance(value, list):
                ret_dicts[key] = value
            elif (
                torch.is_tensor(value)
                and value.ndim == 0
            ):
                ret_dicts[key] = (
                    value.item()
                )

        if (
            getattr(
                self,
                'dense_head_2d',
                None,
            )
            and
            'boxes_2d_pred'
            in cur_data
        ):
            assert (
                len(pred_dicts)
                ==
                len(
                    cur_data[
                        'boxes_2d_pred'
                    ]
                )
            )

            for pred_dict, pred_2d_dict in zip(
                pred_dicts,
                cur_data['boxes_2d_pred'],
            ):
                pred_dict[
                    'pred_boxes_2d'
                ] = pred_2d_dict[
                    'pred_boxes_2d'
                ]

                pred_dict[
                    'pred_scores_2d'
                ] = pred_2d_dict[
                    'pred_scores_2d'
                ]

                pred_dict[
                    'pred_labels_2d'
                ] = pred_2d_dict[
                    'pred_labels_2d'
                ]

        return pred_dicts, ret_dicts

    def forward_test(self, batch_dict):
        """
        Ordinary contiguous offline validation.

        This is intentionally fixed to P^F=[+1] so that offline KITTI AP
        remains comparable to the original StreamDSGN checkpoint.

        Formal streaming evaluation will use a dedicated evaluator with
        actual sensor indices, dropped frames, dynamic P^P/P^F, planner,
        and output buffer.
        """
        cur_data = batch_dict['token']

        # Scene boundary.
        if (
            'prev_sample_idx'
            not in cur_data
            or
            cur_data['prev_sample_idx'] == ''
        ):
            self.history_feature_queue.clear()

        # Current backbone only once.
        cur_data = self._run_feature_extractor(
            cur_data
        )

        # Cache current PRE-TAT feature.  It only enters persistent
        # history after the current inference finishes.
        current_history_feature = (
            self._snapshot_history_feature(
                cur_data,
                clone=True,
            )
        )

        real_history_count = len(
            self.history_feature_queue
        )

        if real_history_count > 0:
            # Standard offline loader processes every frame contiguously.
            # Formal streaming gaps are handled by the dedicated runtime.
            actual_past_offsets = list(
                range(
                    -real_history_count,
                    0,
                )
            )

            cur_data['history_features'] = (
                self.history_feature_queue
            )
        else:
            # Startup fallback.
            #
            # Official Transtreaming pads unavailable history using
            # available backbone features.  Here, a zero-delta current
            # feature serves as one causal pseudo-history at t=0.
            startup_queue = deque(
                maxlen=self.ts_past_length
            )

            startup_queue.append(
                (
                    cur_data.get(
                        'this_sample_idx',
                        '',
                    ),
                    current_history_feature,
                )
            )

            cur_data['history_features'] = (
                startup_queue
            )

            actual_past_offsets = [0]

        # Offline evaluation is explicitly H=+1 only.
        self._set_temporal_proposals(
            cur_data=cur_data,
            past_offsets=actual_past_offsets,
            future_offsets=[1],
        )

        cur_data, future_features = (
            self._run_tat(cur_data)
        )

        if future_features.shape[1] != 1:
            raise RuntimeError(
                'offline validation expected exactly '
                f'one future feature, got '
                f'{future_features.shape[1]}'
            )

        cur_data = (
            self._run_shared_detection_path(
                fused_data=cur_data,
                future_feature=(
                    future_features[:, 0]
                ),
            )
        )

        pred_dicts, ret_dicts = (
            self.post_processing(
                cur_data
            )
        )

        # IMPORTANT:
        # Completed current frame becomes history only now.
        self.history_feature_queue.append(
            (
                cur_data.get(
                    'this_sample_idx',
                    '',
                ),
                current_history_feature,
            )
        )

        pred_dicts, ret_dicts = (
            self._copy_aux_outputs(
                cur_data,
                pred_dicts,
                ret_dicts,
            )
        )

        return pred_dicts, ret_dicts

    def forward_test_save_time(
        self,
        batch_dict,
    ):
        # Do not silently fall back to STREAM.forward_test_save_time(),
        # because that path does not know P^P/P^F and would produce an
        # invalid Transtreaming timing result.
        raise RuntimeError(
            'TRANSTREAMING_STREAM_V2 intentionally disables '
            'STREAM.forward_test_save_time(). '
            'Use the dedicated causal Transtreaming runtime evaluator.'
        )
