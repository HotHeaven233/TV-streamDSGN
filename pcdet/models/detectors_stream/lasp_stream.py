from collections import deque
from contextlib import contextmanager
import re

import numpy as np
import torch

from .stream import STREAM
from pcdet.models.model_utils.lasp_query import LASPQueryModule


class LASP_STREAM(STREAM):
    """
    LASP-style 3D adaptation on top of
    StreamDSGN dense proposals.
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

        if (
            model_cfg.get(
                'LASP',
                None,
            )
            is None
        ):
            raise ValueError(
                'MODEL.LASP config '
                'is required for stream_lasp'
            )

        self.lasp_cfg = (
            model_cfg.LASP
        )

        self.lasp = LASPQueryModule(
            self.lasp_cfg,
            num_class=num_class,
            point_cloud_range=(
                dataset.point_cloud_range
            ),
        )

        self.lasp_memory = deque(
            maxlen=int(
                self.lasp_cfg.get(
                    'MEMORY_FRAMES',
                    4,
                )
            )
        )

        self.train_history_tags = list(
            self.lasp_cfg.get(
                'TRAIN_HISTORY_TAGS',
                [
                    'prev3',
                    'prev2',
                    'prev',
                ],
            )
        )

        self.history_keep_prob = float(
            self.lasp_cfg.get(
                'HISTORY_KEEP_PROB',
                0.75,
            )
        )

        self.freeze_feature_extractor = bool(
            self.lasp_cfg.get(
                'FREEZE_FEATURE_EXTRACTOR',
                True,
            )
        )

        self._last_lasp_output = None
        self._last_post_data = None

        #
        # LASP uses query history.
        # Original BEV FeatureAlignment must not remain active.
        #
        if len(
            self.fusion_module
        ) != 0:
            raise RuntimeError(
                'LASP baseline must bypass '
                'original BEV fusion. '
                'Set MODEL.FUSION_STAGE=[] and '
                'MODEL.FUSION_IN_SPATIAL_FEATURES=null.'
            )

        if self.freeze_feature_extractor:

            for module in (
                self.feature_extractor
            ):

                for p in (
                    module.parameters()
                ):
                    p.requires_grad_(
                        False
                    )

                module.eval()

    def train(
        self,
        mode=True,
    ):

        super().train(
            mode
        )

        if self.freeze_feature_extractor:

            for module in (
                self.feature_extractor
            ):
                module.eval()

        return self

    @contextmanager
    def _base_eval_for_history(
        self,
    ):

        modules = (
            list(
                self.feature_extractor
            )
            +
            list(
                self.after_fusion_blocks
            )
        )

        states = [
            m.training
            for m
            in modules
        ]

        try:

            for m in modules:
                m.eval()

            yield

        finally:

            for (
                m,
                state,
            ) in zip(
                modules,
                states,
            ):
                m.train(
                    state
                )

            if (
                self.freeze_feature_extractor
            ):
                for m in (
                    self.feature_extractor
                ):
                    m.eval()

    def _run_base(
        self,
        frame,
    ):

        x = frame

        for module in (
            self.feature_extractor
        ):
            x = module(
                x
            )

        #
        # Intentionally NO original FeatureAlignment.
        #

        for module in (
            self.after_fusion_blocks
        ):
            x = module(
                x
            )

        return x

    @staticmethod
    def _tag_position(
        tag,
    ):

        if tag == 'prev':
            return -1.0

        m = re.fullmatch(
            r'prev(\d+)',
            str(
                tag
            ),
        )

        if m:

            return -float(
                int(
                    m.group(
                        1
                    )
                )
            )

        raise ValueError(
            'unsupported LASP '
            f'history tag: {tag}'
        )

    @staticmethod
    def _scalar_string(
        x,
    ):

        if isinstance(
            x,
            np.ndarray,
        ):

            if x.size == 0:
                return ''

            x = x.reshape(
                -1
            )[0]

        elif isinstance(
            x,
            (
                list,
                tuple,
            ),
        ):

            if len(
                x
            ) == 0:
                return ''

            x = x[
                0
            ]

        return str(
            x
        )

    @classmethod
    def _frame_position(
        cls,
        frame,
    ):

        raw = cls._scalar_string(
            frame.get(
                'this_sample_idx',
                '',
            )
        )

        m = re.search(
            r'(\d+)$',
            raw,
        )

        if not m:

            raise RuntimeError(
                'cannot parse numeric '
                'sensor position from '
                f'this_sample_idx={raw!r}'
            )

        return float(
            int(
                m.group(
                    1
                )
            )
        )

    @classmethod
    def _scene_start(
        cls,
        frame,
    ):

        if (
            'prev_sample_idx'
            not in frame
        ):
            return True

        return (
            cls._scalar_string(
                frame[
                    'prev_sample_idx'
                ]
            )
            == ''
        )

    def _history_forward(
        self,
        frame,
        temp_memory,
        pos,
    ):

        with (
            self._base_eval_for_history(),
            torch.no_grad(),
        ):

            hist = self._run_base(
                frame
            )

            out = self.lasp(
                hist,
                list(
                    temp_memory
                ),
                pos,
            )

        return self.lasp.memory_entry(
            out,
            pos,
        )

    def forward_train(
        self,
        batch_dict,
    ):

        temp_memory = deque(
            maxlen=int(
                self.lasp_cfg.get(
                    'MEMORY_FRAMES',
                    4,
                )
            )
        )

        ordered = []

        for tag in (
            self.train_history_tags
        ):

            if (
                tag in batch_dict
                and batch_dict[
                    tag
                ]
                is not None
            ):

                ordered.append(
                    (
                        self._tag_position(
                            tag
                        ),
                        tag,
                    )
                )

        ordered.sort(
            key=lambda x: x[
                0
            ]
        )

        #
        # Randomly remove historical observations.
        # LASP therefore sees irregular temporal gaps.
        #
        for (
            pos,
            tag,
        ) in ordered:

            random_value = torch.rand(
                (),
                device=(
                    batch_dict[
                        'token'
                    ][
                        'left_img'
                    ].device
                ),
            ).item()

            if (
                self.training
                and random_value
                > self.history_keep_prob
            ):
                continue

            temp_memory.append(
                self._history_forward(
                    batch_dict[
                        tag
                    ],
                    temp_memory,
                    pos,
                )
            )

        #
        # Current dense StreamDSGN proposal.
        #
        cur = self._run_base(
            batch_dict[
                'token'
            ]
        )

        #
        # LASP continuous history integration
        # + sparse decoder
        # + intention trajectory.
        #
        out = self.lasp(
            cur,
            list(
                temp_memory
            ),
            0.0,
        )

        #
        # Final prediction path uses sparse LASP queries.
        #
        cur[
            'batch_box_preds'
        ] = out.boxes

        cur[
            'batch_cls_preds'
        ] = out.logits

        cur[
            'cls_preds_normalized'
        ] = False

        batch_dict[
            'token'
        ] = cur

        #
        # Original detector current-token losses.
        #
        (
            base_loss,
            tb_dict,
            disp_dict,
        ) = self.get_training_loss(
            batch_dict
        )

        #
        # LASP velocity / trajectory /
        # intention / sparse refinement losses.
        #
        (
            lasp_loss,
            lasp_tb,
        ) = self.lasp.get_loss(
            out,
            batch_dict,
        )

        loss = (
            base_loss
            + lasp_loss
        )

        tb_dict.update(
            lasp_tb
        )

        tb_dict[
            'base_loss'
        ] = float(
            base_loss
            .detach()
            .item()
        )

        tb_dict[
            'total_loss'
        ] = float(
            loss
            .detach()
            .item()
        )

        return (
            {
                'loss': loss
            },
            tb_dict,
            disp_dict,
        )

    def _make_post_data(
        self,
        cur,
        out,
    ):

        keep = {
            'batch_size':
                cur[
                    'batch_size'
                ],

            'batch_box_preds':
                out.boxes,

            'batch_cls_preds':
                out.logits,

            'cls_preds_normalized':
                False,
        }

        for key in [
            'gt_boxes',
            'has_class_labels',
            'roi_labels',
            'batch_pred_labels',
            'multihead_label_mapping',
        ]:

            if key in cur:

                keep[
                    key
                ] = cur[
                    key
                ]

        return keep

    @torch.no_grad()
    def forward_stream_core(
        self,
        batch_dict,
    ):
        """
        Operators that belong inside
        the frozen CUDA-event timing window.

        Includes:
          stereo feature extraction
          VAN
          StreamDetHead
          LASP query extraction
          ODE history integration
          sparse temporal decoder
          trajectory prediction

        Excludes:
          NMS/post-processing
          posterior trajectory lookup
        """

        cur = batch_dict[
            'token'
        ]

        if self._scene_start(
            cur
        ):
            self.lasp_memory.clear()

        pos = self._frame_position(
            cur
        )

        cur = self._run_base(
            cur
        )

        out = self.lasp(
            cur,
            list(
                self.lasp_memory
            ),
            pos,
        )

        #
        # Only a completed processed frame
        # enters LASP history.
        #
        self.lasp_memory.append(
            self.lasp.memory_entry(
                out,
                pos,
            )
        )

        self._last_lasp_output = (
            out
        )

        self._last_post_data = (
            self._make_post_data(
                cur,
                out,
            )
        )

        return {
            'lasp_sensor_position':
                pos,

            'lasp_memory_frames':
                len(
                    self.lasp_memory
                ),
        }

    @torch.no_grad()
    def postprocess_last(
        self,
        delta_frames: float = 0.0,
    ):

        if (
            self._last_lasp_output
            is None
            or self._last_post_data
            is None
        ):
            raise RuntimeError(
                'no completed LASP '
                'forward is available '
                'for post-processing'
            )

        post_data = dict(
            self._last_post_data
        )

        post_data[
            'batch_box_preds'
        ] = (
            self.lasp.compensate_boxes(
                self._last_lasp_output,
                float(
                    delta_frames
                ),
            )
        )

        return self.post_processing(
            post_data
        )

    def forward_test(
        self,
        batch_dict,
    ):

        meta = self.forward_stream_core(
            batch_dict
        )

        (
            pred_dicts,
            ret_dicts,
        ) = self.postprocess_last(
            0.0
        )

        ret_dicts.update(
            meta
        )

        return (
            pred_dicts,
            ret_dicts,
        )

    def forward_test_save_time(
        self,
        batch_dict,
    ):

        raise RuntimeError(
            'Use the dedicated streaming '
            'evaluator with CUDA events. '
            'STREAM.forward_test_save_time '
            'is not the frozen timing protocol.'
        )

    @torch.no_grad()
    def get_compensated_prediction(
        self,
        delta_frames: float,
    ):
        """
        Re-query already-predicted future trajectory.

        delta_frames =
            posterior_elapsed_ms
            / input_period_ms

        No backbone / detector /
        trajectory NN is rerun here.
        """

        if (
            self._last_lasp_output
            is None
            or self._last_post_data
            is None
        ):
            raise RuntimeError(
                'no completed LASP forward '
                'is available for compensation'
            )

        return self.postprocess_last(
            float(
                delta_frames
            )
        )
