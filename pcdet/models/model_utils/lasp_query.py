import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LASPOutput:
    query: torch.Tensor
    boxes: torch.Tensor
    logits: torch.Tensor
    scores: torch.Tensor
    labels: torch.Tensor
    velocity: torch.Tensor
    trajectories: torch.Tensor
    mode_logits: torch.Tensor


class SparseTemporalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_mult: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ffn_mult, dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: Optional[torch.Tensor],
    ) -> torch.Tensor:

        y, _ = self.self_attn(
            x,
            x,
            x,
            need_weights=False,
        )
        x = self.norm1(x + y)

        if (
            memory is not None
            and memory.numel() > 0
        ):
            y, _ = self.cross_attn(
                x,
                memory,
                memory,
                need_weights=False,
            )
            x = self.norm2(x + y)
        else:
            x = self.norm2(x)

        x = self.norm3(
            x + self.ffn(x)
        )

        return x


def _sinusoidal_xy(
    xy: torch.Tensor,
    dim: int,
) -> torch.Tensor:

    if dim % 4 != 0:
        raise ValueError(
            f'positional dim must be divisible by 4, got {dim}'
        )

    quarter = dim // 4

    freq = torch.arange(
        quarter,
        device=xy.device,
        dtype=xy.dtype,
    )

    freq = torch.exp(
        -math.log(10000.0)
        * freq
        / max(quarter - 1, 1)
    )

    x = xy[..., 0:1] * freq
    y = xy[..., 1:2] * freq

    return torch.cat(
        [
            x.sin(),
            x.cos(),
            y.sin(),
            y.cos(),
        ],
        dim=-1,
    )


class LASPQueryModule(nn.Module):
    """
    LASP-style sparse-query adaptation for StreamDSGN.

    Time is represented in sensor-frame units.

    Example:
        dt=2 means that two sensor positions separate
        the historical observation and the current input.
    """

    def __init__(
        self,
        cfg,
        num_class: int,
        point_cloud_range: Sequence[float],
    ):
        super().__init__()

        self.cfg = cfg
        self.num_class = int(num_class)

        self.pc_range = tuple(
            float(x)
            for x in point_cloud_range
        )

        self.topk = int(
            cfg.get('TOPK', 128)
        )

        self.dim = int(
            cfg.get('EMBED_DIM', 128)
        )

        self.num_basis = int(
            cfg.get('NUM_BASIS', 10)
        )

        self.num_intentions = int(
            cfg.get('NUM_INTENTIONS', 6)
        )

        self.future_steps = [
            int(x)
            for x in cfg.get(
                'FUTURE_STEPS',
                list(range(1, 9)),
            )
        ]

        self.feature_channels = int(
            cfg.get(
                'QUERY_FEATURE_CHANNELS',
                64,
            )
        )

        self.eig_clip = float(
            cfg.get(
                'EIG_CLIP',
                1.0,
            )
        )

        input_dim = (
            self.feature_channels
            + 7
            + self.num_class
        )

        #
        # dense StreamDetHead proposal
        # -> sparse query context
        #
        self.query_encoder = nn.Sequential(
            nn.Linear(
                input_dim,
                self.dim,
            ),
            nn.LayerNorm(
                self.dim
            ),
            nn.GELU(),
            nn.Linear(
                self.dim,
                self.dim,
            ),
        )

        self.box_encoder = nn.Sequential(
            nn.Linear(
                7,
                self.dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.dim,
                self.dim,
            ),
        )

        #
        # LASP hidden-space transform
        #
        self.phi = nn.Sequential(
            nn.Linear(
                self.dim,
                self.dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.dim,
                self.dim,
            ),
        )

        self.phi_inv = nn.Sequential(
            nn.Linear(
                self.dim,
                self.dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.dim,
                self.dim,
            ),
        )

        #
        # motion attributes:
        #
        # [relative vx, relative vy, delta_t]
        #
        # Original LASP additionally has ego-pose transformation.
        # StreamDSGN/KITTI adaptation uses apparent relative motion
        # in pseudo-LiDAR coordinates so no extra sensing input is added.
        #
        motion_hidden = int(
            cfg.get(
                'MOTION_HIDDEN',
                64,
            )
        )

        self.alpha_net = nn.Sequential(
            nn.Linear(
                3,
                motion_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                motion_hidden,
                self.num_basis,
            ),
        )

        self.eig_net = nn.Sequential(
            nn.Linear(
                3,
                motion_hidden,
            ),
            nn.GELU(),
            nn.Linear(
                motion_hidden,
                self.num_basis
                * self.dim,
            ),
        )

        #
        # Identity continuous transition at initialization.
        #
        nn.init.zeros_(
            self.eig_net[-1].weight
        )
        nn.init.zeros_(
            self.eig_net[-1].bias
        )

        #
        # Shared eigenvectors E in:
        #
        # A(k) = E D(k) E^T
        #
        self.eigenvectors_raw = nn.Parameter(
            torch.eye(
                self.dim
            )
        )

        #
        # sparse spatial-temporal interaction
        #
        self.temporal_blocks = nn.ModuleList(
            [
                SparseTemporalBlock(
                    self.dim,
                    int(
                        cfg.get(
                            'NUM_HEADS',
                            4,
                        )
                    ),
                    int(
                        cfg.get(
                            'FFN_MULT',
                            4,
                        )
                    ),
                    float(
                        cfg.get(
                            'DROPOUT',
                            0.0,
                        )
                    ),
                )
                for _ in range(
                    int(
                        cfg.get(
                            'NUM_DECODER_LAYERS',
                            2,
                        )
                    )
                )
            ]
        )

        #
        # current object state
        #
        self.velocity_head = nn.Sequential(
            nn.Linear(
                self.dim,
                self.dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.dim,
                2,
            ),
        )

        #
        # residual refinement keeps pretrained StreamDSGN
        # proposal unchanged at initialization.
        #
        self.state_delta = nn.Linear(
            self.dim,
            7,
        )

        self.cls_delta = nn.Linear(
            self.dim,
            self.num_class,
        )

        nn.init.zeros_(
            self.state_delta.weight
        )
        nn.init.zeros_(
            self.state_delta.bias
        )
        nn.init.zeros_(
            self.cls_delta.weight
        )
        nn.init.zeros_(
            self.cls_delta.bias
        )

        #
        # class-conditioned intention centers
        #
        centers_path = str(
            cfg.get(
                'INTENTION_CENTERS',
                '',
            )
        )

        if not centers_path:
            raise ValueError(
                'MODEL.LASP.INTENTION_CENTERS must be set'
            )

        centers = np.load(
            centers_path
        ).astype(
            np.float32
        )

        expected = (
            self.num_class,
            self.num_intentions,
            2,
        )

        if centers.shape != expected:
            raise ValueError(
                f'intention centers shape '
                f'{centers.shape}, '
                f'expected {expected}'
            )

        self.register_buffer(
            'intention_centers',
            torch.from_numpy(
                centers
            ),
            persistent=True,
        )

        pe_dim = int(
            cfg.get(
                'INTENTION_PE_DIM',
                64,
            )
        )

        if pe_dim % 4 != 0:
            raise ValueError(
                'INTENTION_PE_DIM '
                'must be divisible by 4'
            )

        self.intention_encoder = nn.Sequential(
            nn.Linear(
                pe_dim,
                self.dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.dim,
                self.dim,
            ),
        )

        self.endpoint_encoder = nn.Sequential(
            nn.Linear(
                pe_dim,
                self.dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.dim,
                self.dim,
            ),
        )

        traj_layers = int(
            cfg.get(
                'TRAJ_LAYERS',
                3,
            )
        )

        self.traj_decoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        self.dim,
                        self.dim,
                    ),
                    nn.GELU(),
                    nn.Linear(
                        self.dim,
                        len(
                            self.future_steps
                        ) * 2,
                    ),
                )
                for _ in range(
                    traj_layers
                )
            ]
        )

        self.mode_head = nn.Sequential(
            nn.Linear(
                self.dim,
                self.dim,
            ),
            nn.GELU(),
            nn.Linear(
                self.dim,
                1,
            ),
        )

    def _sample_reg_features(
        self,
        reg_features: torch.Tensor,
        boxes: torch.Tensor,
    ) -> torch.Tensor:

        if (
            reg_features.ndim != 4
            or boxes.ndim != 3
        ):
            raise ValueError(
                'expected reg_features '
                '[B,C,H,W] and boxes [B,N,7]'
            )

        if (
            reg_features.shape[1]
            != self.feature_channels
        ):
            raise RuntimeError(
                'LASP QUERY_FEATURE_CHANNELS='
                f'{self.feature_channels}, '
                'but StreamDetHead reg_features '
                f'has C={reg_features.shape[1]}'
            )

        (
            x0,
            y0,
            _,
            x1,
            y1,
            _,
        ) = self.pc_range

        gx = (
            2.0
            * (
                boxes[..., 0]
                - x0
            )
            / max(
                x1 - x0,
                1e-6,
            )
            - 1.0
        )

        gy = (
            2.0
            * (
                boxes[..., 1]
                - y0
            )
            / max(
                y1 - y0,
                1e-6,
            )
            - 1.0
        )

        grid = torch.stack(
            [
                gx,
                gy,
            ],
            dim=-1,
        ).unsqueeze(
            2
        )

        sampled = F.grid_sample(
            reg_features,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

        return (
            sampled
            .squeeze(-1)
            .transpose(
                1,
                2,
            )
            .contiguous()
        )

    def extract_queries(
        self,
        data_dict: Dict,
    ) -> Dict[str, torch.Tensor]:

        boxes = (
            data_dict[
                'batch_box_preds'
            ][..., :7]
        )

        logits = data_dict[
            'batch_cls_preds'
        ]

        if isinstance(
            logits,
            list,
        ):
            raise NotImplementedError(
                'LASP adapter currently '
                'expects a single '
                'StreamDetHead cls tensor'
            )

        if boxes.shape[0] != 1:
            raise RuntimeError(
                'StreamDSGN/LASP baseline '
                'currently supports '
                'batch size 1 only'
            )

        prob = torch.sigmoid(
            logits
        )

        scores, labels0 = prob.max(
            dim=-1
        )

        k = min(
            self.topk,
            scores.shape[1],
        )

        top_scores, idx = scores.topk(
            k,
            dim=1,
        )

        gather_box = (
            idx
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                boxes.shape[-1],
            )
        )

        gather_cls = (
            idx
            .unsqueeze(-1)
            .expand(
                -1,
                -1,
                logits.shape[-1],
            )
        )

        boxes = boxes.gather(
            1,
            gather_box,
        )

        logits = logits.gather(
            1,
            gather_cls,
        )

        labels0 = labels0.gather(
            1,
            idx,
        )

        feat = self._sample_reg_features(
            data_dict[
                'reg_features'
            ],
            boxes,
        )

        q = self.query_encoder(
            torch.cat(
                [
                    feat,
                    boxes,
                    logits,
                ],
                dim=-1,
            )
        )

        return {
            'query': q,
            'boxes': boxes,
            'logits': logits,
            'scores': top_scores,
            'labels': labels0 + 1,
        }

    def _orthogonal_e(
        self,
    ) -> torch.Tensor:

        return torch.linalg.qr(
            self.eigenvectors_raw.float()
        ).Q

    def _continuous_propagate(
        self,
        q: torch.Tensor,
        velocity: torch.Tensor,
        dt: float,
        e: torch.Tensor,
    ) -> torch.Tensor:

        if abs(
            float(dt)
        ) < 1e-12:
            return q

        e = e.to(
            dtype=q.dtype,
            device=q.device,
        )

        n = q.shape[1]

        dt_tensor = q.new_full(
            (
                n,
                1,
            ),
            float(dt),
        )

        motion = torch.cat(
            [
                velocity[0],
                dt_tensor,
            ],
            dim=-1,
        )

        alpha = torch.softmax(
            self.alpha_net(
                motion
            ),
            dim=-1,
        )

        eig = (
            self.eig_clip
            * torch.tanh(
                self.eig_net(
                    motion
                )
            )
        ).view(
            n,
            self.num_basis,
            self.dim,
        )

        rate = (
            alpha.unsqueeze(-1)
            * eig
        ).sum(
            dim=1
        )

        z = self.phi(
            q[0]
        )

        #
        # Eq. 11 style:
        #
        # z_t =
        # E exp(
        #   dt * sum(alpha_k D_k)
        # ) E^T z_tau
        #
        z_e = z @ e

        z_e = z_e * torch.exp(
            torch.clamp(
                dt_tensor
                * rate,
                min=-8.0,
                max=8.0,
            )
        )

        z_t = (
            z_e
            @ e.transpose(
                0,
                1,
            )
        )

        return self.phi_inv(
            z_t
        ).unsqueeze(
            0
        )

    def _propagate_memory(
        self,
        memory: List[Dict],
        current_pos: float,
    ) -> Optional[torch.Tensor]:

        if not memory:
            return None

        e = self._orthogonal_e()

        propagated = []

        for item in memory:

            dt = (
                float(
                    current_pos
                )
                - float(
                    item[
                        'position'
                    ]
                )
            )

            boxes = item[
                'boxes'
            ].clone()

            boxes[
                ...,
                0:2
            ] = (
                boxes[
                    ...,
                    0:2
                ]
                + item[
                    'velocity'
                ]
                * dt
            )

            q = self._continuous_propagate(
                item[
                    'query'
                ],
                item[
                    'velocity'
                ],
                dt,
                e,
            )

            q = (
                q
                + self.box_encoder(
                    boxes
                )
            )

            propagated.append(
                q
            )

        return torch.cat(
            propagated,
            dim=1,
        )

    def _predict_trajectories(
        self,
        q: torch.Tensor,
        logits: torch.Tensor,
    ):

        class_prob = torch.softmax(
            logits,
            dim=-1,
        )

        centers = (
            self.intention_centers
            .to(
                dtype=q.dtype
            )
        )

        pe_dim = int(
            self.cfg.get(
                'INTENTION_PE_DIM',
                64,
            )
        )

        pe = _sinusoidal_xy(
            centers,
            pe_dim,
        )

        intention_bank = (
            self.intention_encoder(
                pe
            )
        )

        #
        # class prior selects/blends
        # class-specific intention bank.
        #
        intention = torch.einsum(
            'bnc,ckd->bnkd',
            class_prob,
            intention_bank,
        )

        all_traj = []

        for decoder in (
            self.traj_decoders
        ):

            h = (
                q.unsqueeze(2)
                + intention
            )

            traj = decoder(
                h
            ).view(
                q.shape[0],
                q.shape[1],
                self.num_intentions,
                len(
                    self.future_steps
                ),
                2,
            )

            all_traj.append(
                traj
            )

            #
            # LASP iterative intention
            # refinement from previous endpoint.
            #
            endpoint = traj[
                ...,
                -1,
                :
            ]

            endpoint_pe = _sinusoidal_xy(
                endpoint,
                pe_dim,
            )

            intention = (
                intention
                + self.endpoint_encoder(
                    endpoint_pe
                )
            )

        mode_logits = (
            self.mode_head(
                q.unsqueeze(2)
                + intention
            )
            .squeeze(-1)
        )

        return (
            all_traj,
            mode_logits,
        )

    def forward(
        self,
        data_dict: Dict,
        memory: List[Dict],
        current_pos: float,
    ) -> LASPOutput:

        base = self.extract_queries(
            data_dict
        )

        q = base[
            'query'
        ]

        temporal_memory = (
            self._propagate_memory(
                memory,
                current_pos,
            )
        )

        for block in (
            self.temporal_blocks
        ):
            q = block(
                q,
                temporal_memory,
            )

        velocity = self.velocity_head(
            q
        )

        delta = self.state_delta(
            q
        )

        boxes = base[
            'boxes'
        ].clone()

        boxes[
            ...,
            :3
        ] = (
            boxes[
                ...,
                :3
            ]
            + delta[
                ...,
                :3
            ]
        )

        boxes[
            ...,
            3:6
        ] = torch.clamp(
            boxes[
                ...,
                3:6
            ]
            + delta[
                ...,
                3:6
            ],
            min=0.05,
        )

        boxes[
            ...,
            6
        ] = (
            boxes[
                ...,
                6
            ]
            + delta[
                ...,
                6
            ]
        )

        logits = (
            base[
                'logits'
            ]
            + self.cls_delta(
                q
            )
        )

        scores, labels0 = (
            torch.sigmoid(
                logits
            )
            .max(
                dim=-1
            )
        )

        (
            all_traj,
            mode_logits,
        ) = self._predict_trajectories(
            q,
            logits,
        )

        return LASPOutput(
            query=q,
            boxes=boxes,
            logits=logits,
            scores=scores,
            labels=labels0 + 1,
            velocity=velocity,
            trajectories=all_traj[-1],
            mode_logits=mode_logits,
        )

    def memory_entry(
        self,
        out: LASPOutput,
        position: float,
    ) -> Dict:

        return {
            'query': (
                out.query
                .detach()
            ),
            'boxes': (
                out.boxes
                .detach()
            ),
            'logits': (
                out.logits
                .detach()
            ),
            'velocity': (
                out.velocity
                .detach()
            ),
            'position': float(
                position
            ),
        }

    def compensate_boxes(
        self,
        out: LASPOutput,
        delta_frames: float,
    ) -> torch.Tensor:
        """
        Query already-computed trajectory.

        No backbone / detector / trajectory NN is rerun.
        """

        delta = float(
            delta_frames
        )

        if delta <= 0.0:
            return out.boxes

        steps = out.boxes.new_tensor(
            self.future_steps,
            dtype=out.boxes.dtype,
        )

        d = out.boxes.new_tensor(
            delta,
            dtype=out.boxes.dtype,
        ).clamp(
            min=0.0,
            max=float(
                self.future_steps[-1]
            ),
        )

        hi = int(
            torch.searchsorted(
                steps,
                d,
                right=False,
            ).item()
        )

        mode = out.mode_logits.argmax(
            dim=-1
        )

        gather_mode = (
            mode[
                ...,
                None,
                None,
                None
            ]
            .expand(
                -1,
                -1,
                1,
                len(
                    self.future_steps
                ),
                2,
            )
        )

        traj = (
            out.trajectories
            .gather(
                2,
                gather_mode,
            )
            .squeeze(2)
        )

        if hi <= 0:

            ratio = (
                d
                / steps[
                    0
                ].clamp(
                    min=1e-6
                )
            )

            offset = (
                traj[
                    ...,
                    0,
                    :
                ]
                * ratio
            )

        elif hi >= len(
            self.future_steps
        ):

            offset = traj[
                ...,
                -1,
                :
            ]

        else:

            lo = hi - 1

            denom = (
                steps[
                    hi
                ]
                - steps[
                    lo
                ]
            ).clamp(
                min=1e-6
            )

            ratio = (
                d
                - steps[
                    lo
                ]
            ) / denom

            offset = (
                traj[
                    ...,
                    lo,
                    :
                ]
                + ratio
                * (
                    traj[
                        ...,
                        hi,
                        :
                    ]
                    - traj[
                        ...,
                        lo,
                        :
                    ]
                )
            )

        boxes = out.boxes.clone()

        boxes[
            ...,
            0:2
        ] = (
            boxes[
                ...,
                0:2
            ]
            + offset
        )

        return boxes

    @staticmethod
    def _valid_gt(
        frame: Optional[Dict],
    ):

        if (
            frame is None
            or 'object_id'
            not in frame
            or 'gt_boxes'
            not in frame
        ):
            return (
                [],
                None,
            )

        ids = frame[
            'object_id'
        ][0]

        n = len(
            ids
        )

        return (
            [
                str(x)
                for x in ids
            ],
            frame[
                'gt_boxes'
            ][
                0,
                :n,
                :
            ],
        )

    def _build_track_targets(
        self,
        batch_dict: Dict,
        device,
    ):

        (
            token_ids,
            token_gt,
        ) = self._valid_gt(
            batch_dict.get(
                'token'
            )
        )

        if (
            token_gt is None
            or len(
                token_ids
            ) == 0
        ):
            return None

        token_gt = token_gt.to(
            device
        )

        n = len(
            token_ids
        )

        traj = token_gt.new_zeros(
            (
                n,
                len(
                    self.future_steps
                ),
                2,
            )
        )

        mask = torch.zeros(
            (
                n,
                len(
                    self.future_steps
                ),
            ),
            dtype=torch.bool,
            device=device,
        )

        id_to_cur = {
            oid: i
            for i, oid
            in enumerate(
                token_ids
            )
        }

        for (
            si,
            step,
        ) in enumerate(
            self.future_steps
        ):

            tag = (
                'next'
                if step == 1
                else f'next{step}'
            )

            (
                ids,
                boxes,
            ) = self._valid_gt(
                batch_dict.get(
                    tag
                )
            )

            if boxes is None:
                continue

            boxes = boxes.to(
                device
            )

            id_to_future = {
                oid: i
                for i, oid
                in enumerate(
                    ids
                )
            }

            for (
                oid,
                ci,
            ) in id_to_cur.items():

                fi = id_to_future.get(
                    oid
                )

                if fi is None:
                    continue

                traj[
                    ci,
                    si
                ] = (
                    boxes[
                        fi,
                        0:2
                    ]
                    - token_gt[
                        ci,
                        0:2
                    ]
                )

                mask[
                    ci,
                    si
                ] = True

        return (
            token_ids,
            token_gt,
            traj,
            mask,
        )

    @staticmethod
    def _greedy_match(
        query_boxes: torch.Tensor,
        query_logits: torch.Tensor,
        gt_boxes: torch.Tensor,
    ):

        if (
            gt_boxes.numel() == 0
            or query_boxes.numel() == 0
        ):
            return []

        dist = torch.cdist(
            query_boxes[
                :,
                :2
            ],
            gt_boxes[
                :,
                :2
            ],
            p=2,
        )

        gt_cls = (
            gt_boxes[
                :,
                7
            ]
            .long()
            .clamp(
                min=1
            )
            - 1
        )

        cls_prob = torch.sigmoid(
            query_logits
        )

        class_bonus = cls_prob[
            :,
            gt_cls
        ]

        cost = (
            dist
            - 2.0
            * class_bonus
        )

        matches = []
        used_q = set()

        gt_order = torch.argsort(
            cost.min(
                dim=0
            ).values
        ).tolist()

        for gi in gt_order:

            q_order = torch.argsort(
                cost[
                    :,
                    gi
                ]
            ).tolist()

            qi = next(
                (
                    q
                    for q
                    in q_order
                    if q
                    not in used_q
                ),
                None,
            )

            if qi is None:
                break

            used_q.add(
                qi
            )

            matches.append(
                (
                    qi,
                    gi,
                )
            )

        return matches

    def get_loss(
        self,
        out: LASPOutput,
        batch_dict: Dict,
    ):

        built = self._build_track_targets(
            batch_dict,
            out.boxes.device,
        )

        if built is None:

            zero = (
                out.query.sum()
                * 0.0
            )

            return (
                zero,
                {
                    'lasp_loss': 0.0,
                    'lasp_matches': 0,
                },
            )

        (
            _,
            gt_boxes,
            gt_traj,
            gt_mask,
        ) = built

        q_boxes = out.boxes[
            0
        ]

        q_logits = out.logits[
            0
        ]

        matches = self._greedy_match(
            q_boxes.detach(),
            q_logits.detach(),
            gt_boxes,
        )

        if not matches:

            zero = (
                out.query.sum()
                * 0.0
            )

            return (
                zero,
                {
                    'lasp_loss': 0.0,
                    'lasp_matches': 0,
                },
            )

        qidx = torch.tensor(
            [
                x[0]
                for x
                in matches
            ],
            device=q_boxes.device,
            dtype=torch.long,
        )

        gidx = torch.tensor(
            [
                x[1]
                for x
                in matches
            ],
            device=q_boxes.device,
            dtype=torch.long,
        )

        pred_box = q_boxes[
            qidx
        ]

        tgt_box = gt_boxes[
            gidx,
            :7
        ]

        loc_loss = F.smooth_l1_loss(
            pred_box[
                :,
                :6
            ],
            tgt_box[
                :,
                :6
            ],
            reduction='mean',
        )

        yaw_loss = (
            1.0
            - torch.cos(
                pred_box[
                    :,
                    6
                ]
                - tgt_box[
                    :,
                    6
                ]
            )
        ).mean()

        cls_target = torch.zeros_like(
            q_logits
        )

        gt_cls = (
            gt_boxes[
                gidx,
                7
            ]
            .long()
            .clamp(
                1,
                self.num_class,
            )
            - 1
        )

        cls_target[
            qidx,
            gt_cls
        ] = 1.0

        cls_loss = (
            F.binary_cross_entropy_with_logits(
                q_logits,
                cls_target,
                reduction='mean',
            )
        )

        matched_traj = gt_traj[
            gidx
        ]

        matched_mask = gt_mask[
            gidx
        ]

        vel_valid = matched_mask[
            :,
            0
        ]

        if vel_valid.any():

            step1 = float(
                self.future_steps[
                    0
                ]
            )

            vel_target = (
                matched_traj[
                    vel_valid,
                    0
                ]
                / max(
                    step1,
                    1e-6,
                )
            )

            vel_loss = (
                F.smooth_l1_loss(
                    out.velocity[
                        0,
                        qidx[
                            vel_valid
                        ]
                    ],
                    vel_target,
                    reduction='mean',
                )
            )

        else:

            vel_loss = (
                out.velocity.sum()
                * 0.0
            )

        pred_traj = out.trajectories[
            0,
            qidx
        ]

        mode_logits = out.mode_logits[
            0,
            qidx
        ]

        # ------------------------------------------------------------
        # Trajectory regression and intention classification are not
        # exactly the same supervision problem.
        #
        # INTENTION_CENTERS are clustered from the final H8 endpoint.
        # Therefore intention CE is valid only when H8 GT exists.
        #
        # Shorter valid trajectories can still supervise trajectory
        # regression. For those objects we choose the best trajectory
        # mode under their available GT steps, but DO NOT assign an
        # incorrect H8 intention class from a shorter endpoint.
        # ------------------------------------------------------------

        num_matched = len(matches)

        any_future = matched_mask.any(
            dim=1
        )

        final_idx = (
            len(
                self.future_steps
            )
            - 1
        )

        full_horizon = matched_mask[
            :,
            final_idx
        ]

        # ------------------------------------------------------------
        # 1. Best regression mode for every object that has at least
        #    one future GT.
        # ------------------------------------------------------------

        target_all = matched_traj[
            :,
            None,
            :,
            :
        ]

        mask_all = matched_mask[
            :,
            None,
            :,
            None
        ]

        traj_error_all = (
            F.smooth_l1_loss(
                pred_traj,
                target_all.expand_as(
                    pred_traj
                ),
                reduction='none',
            )
            * mask_all
        )

        per_mode_error = (
            traj_error_all.sum(
                dim=(-1, -2)
            )
            /
            (
                mask_all.sum(
                    dim=(-1, -2)
                )
                .clamp(
                    min=1
                )
            )
        )

        best_reg_mode = (
            per_mode_error.argmin(
                dim=1
            )
        )

        # ------------------------------------------------------------
        # 2. H8 intention target.
        # ------------------------------------------------------------

        full_indices = torch.where(
            full_horizon
        )[0]

        intention_targets = []

        for mi_tensor in full_indices:

            mi = int(
                mi_tensor.item()
            )

            cls_i = int(
                gt_cls[
                    mi
                ].item()
            )

            endpoint = matched_traj[
                mi,
                final_idx
            ]

            centers = (
                self.intention_centers[
                    cls_i
                ]
                .to(
                    endpoint
                )
            )

            intention_targets.append(
                int(
                    torch.argmin(
                        (
                            (
                                centers
                                - endpoint
                            )
                            ** 2
                        ).sum(
                            dim=-1
                        )
                    ).item()
                )
            )

        if len(
            intention_targets
        ) > 0:

            intention_targets = torch.tensor(
                intention_targets,
                device=q_boxes.device,
                dtype=torch.long,
            )

            mode_loss = F.cross_entropy(
                mode_logits[
                    full_indices
                ],
                intention_targets,
            )

            # For full-H8 trajectories the cluster assignment defines
            # which intention branch should learn that trajectory.
            best_reg_mode = (
                best_reg_mode.clone()
            )

            best_reg_mode[
                full_indices
            ] = intention_targets

        else:

            mode_loss = (
                out.mode_logits.sum()
                * 0.0
            )

        # ------------------------------------------------------------
        # 3. Trajectory regression for ALL objects with at least one
        #    valid future timestamp.
        # ------------------------------------------------------------

        traj_indices = torch.where(
            any_future
        )[0]

        if len(
            traj_indices
        ) > 0:

            chosen_modes = best_reg_mode[
                traj_indices
            ]

            chosen = pred_traj[
                traj_indices,
                chosen_modes
            ]

            target = matched_traj[
                traj_indices
            ]

            mask = matched_mask[
                traj_indices
            ].unsqueeze(
                -1
            )

            traj_abs = F.smooth_l1_loss(
                chosen,
                target,
                reduction='none',
            )

            # Normalize by valid XY coordinates, not only valid frames.
            denom = (
                mask.sum()
                * chosen.shape[-1]
            ).clamp(
                min=1
            )

            traj_loss = (
                (
                    traj_abs
                    * mask
                ).sum()
                / denom
            )

        else:

            traj_loss = (
                out.trajectories.sum()
                * 0.0
            )

        w = self.cfg.get(
            'LOSS_WEIGHTS',
            {},
        )

        total = (
            float(
                w.get(
                    'box',
                    1.0,
                )
            )
            * (
                loc_loss
                + yaw_loss
            )
            +
            float(
                w.get(
                    'cls',
                    1.0,
                )
            )
            * cls_loss
            +
            float(
                w.get(
                    'velocity',
                    0.5,
                )
            )
            * vel_loss
            +
            float(
                w.get(
                    'trajectory',
                    1.0,
                )
            )
            * traj_loss
            +
            float(
                w.get(
                    'intention',
                    0.2,
                )
            )
            * mode_loss
        )

        tb = {
            'lasp_loss':
                float(
                    total.detach().item()
                ),

            'lasp_box_loss':
                float(
                    (
                        loc_loss
                        + yaw_loss
                    )
                    .detach()
                    .item()
                ),

            'lasp_cls_loss':
                float(
                    cls_loss
                    .detach()
                    .item()
                ),

            'lasp_velocity_loss':
                float(
                    vel_loss
                    .detach()
                    .item()
                ),

            'lasp_traj_loss':
                float(
                    traj_loss
                    .detach()
                    .item()
                ),

            'lasp_intention_loss':
                float(
                    mode_loss
                    .detach()
                    .item()
                ),

            'lasp_matches':
                len(
                    matches
                ),
        }

        return (
            total,
            tb,
        )
