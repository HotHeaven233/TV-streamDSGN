import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2d_long(values, batch_size, device):
    if isinstance(values, torch.Tensor):
        x = values.to(device=device, dtype=torch.long)
    else:
        x = torch.as_tensor(
            values,
            dtype=torch.long,
            device=device,
        )

    if x.ndim == 0:
        x = x.view(1, 1)
    elif x.ndim == 1:
        x = x.view(1, -1)

    if x.shape[0] == 1 and batch_size > 1:
        x = x.expand(batch_size, -1)

    if x.shape[0] != batch_size:
        raise RuntimeError(
            f'temporal offset batch mismatch: '
            f'{x.shape[0]} != {batch_size}'
        )

    return x


def _window_partition(x, window_size):
    """
    x:
        [B, T, H, W, C]

    return:
        windows: [B, N, T*Wh*Ww, C]
        meta
    """
    B, T, H, W, C = x.shape
    Wh, Ww = window_size

    pad_h = (Wh - H % Wh) % Wh
    pad_w = (Ww - W % Ww) % Ww

    if pad_h > 0 or pad_w > 0:
        x = x.permute(0, 1, 4, 2, 3)
        x = F.pad(
            x,
            (0, pad_w, 0, pad_h),
        )
        x = x.permute(0, 1, 3, 4, 2)

    Hp = H + pad_h
    Wp = W + pad_w

    nh = Hp // Wh
    nw = Wp // Ww

    x = x.view(
        B,
        T,
        nh,
        Wh,
        nw,
        Ww,
        C,
    )

    x = x.permute(
        0, 2, 4, 1, 3, 5, 6
    ).contiguous()

    windows = x.view(
        B,
        nh * nw,
        T * Wh * Ww,
        C,
    )

    meta = dict(
        B=B,
        T=T,
        H=H,
        W=W,
        C=C,
        Hp=Hp,
        Wp=Wp,
        nh=nh,
        nw=nw,
    )

    return windows, meta


def _window_reverse(windows, meta, window_size):
    """
    windows:
        [B, N, T*Wh*Ww, C]

    return:
        [B, T, H, W, C]
    """
    B = meta['B']
    T = meta['T']
    H = meta['H']
    W = meta['W']
    C = meta['C']
    Hp = meta['Hp']
    Wp = meta['Wp']
    nh = meta['nh']
    nw = meta['nw']

    Wh, Ww = window_size

    x = windows.view(
        B,
        nh,
        nw,
        T,
        Wh,
        Ww,
        C,
    )

    x = x.permute(
        0, 3, 1, 4, 2, 5, 6
    ).contiguous()

    x = x.view(
        B,
        T,
        Hp,
        Wp,
        C,
    )

    return x[:, :, :H, :W]


class RelativeTemporalPositionBias(nn.Module):
    """
    Transtreaming-style relative temporal-spatial positional encoding.

    Query position:
        (future_time, q_y, q_x)

    Key position:
        (past_time, k_y, k_x)

    Continuous relative coordinates:
        [future_time - past_time, q_y-k_y, q_x-k_x]
    """

    def __init__(
        self,
        window_size: Tuple[int, int],
        num_heads: int,
        max_time: int = 32,
        hidden_dim: int = 256,
        rpe_coef: float = 16.0,
    ):
        super().__init__()

        self.window_size = tuple(window_size)
        self.num_heads = int(num_heads)
        self.max_time = int(max_time)
        self.rpe_coef = float(rpe_coef)

        self.cpb_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(
                hidden_dim,
                self.num_heads,
                bias=False,
            ),
        )

        Wh, Ww = self.window_size

        y, x = torch.meshgrid(
            torch.arange(Wh),
            torch.arange(Ww),
            indexing='ij',
        )

        spatial = torch.stack(
            [y, x],
            dim=-1,
        ).reshape(-1, 2).float()

        # [HW, HW, 2]
        rel = (
            spatial[:, None, :]
            -
            spatial[None, :, :]
        )

        self.register_buffer(
            'spatial_relative',
            rel,
            persistent=False,
        )

    def forward(
        self,
        past_offsets,
        future_offsets,
    ):
        """
        past_offsets:
            [B, TP], negative

        future_offsets:
            [B, TF], positive

        return:
            [B, heads, TF*HW, TP*HW]
        """
        B, TP = past_offsets.shape
        _, TF = future_offsets.shape

        Wh, Ww = self.window_size
        HW = Wh * Ww

        # [B, TF, TP]
        dt = (
            future_offsets[:, :, None]
            -
            past_offsets[:, None, :]
        ).float()

        # [B, TF, TP, HW, HW]
        dt = dt[:, :, :, None, None].expand(
            B,
            TF,
            TP,
            HW,
            HW,
        )

        spatial = self.spatial_relative.to(
            dt.device
        )

        dy = spatial[..., 0]
        dx = spatial[..., 1]

        dy = dy.view(
            1, 1, 1, HW, HW
        ).expand(
            B, TF, TP, HW, HW
        )

        dx = dx.view(
            1, 1, 1, HW, HW
        ).expand(
            B, TF, TP, HW, HW
        )

        coords = torch.stack(
            [dt, dy, dx],
            dim=-1,
        )

        coords[..., 0] /= max(
            float(self.max_time),
            1.0,
        )

        coords[..., 1] /= max(
            float(Wh - 1),
            1.0,
        )

        coords[..., 2] /= max(
            float(Ww - 1),
            1.0,
        )

        # Same continuous log-coordinate idea as Transtreaming RTPE.
        rct_coef = 8.0

        coords = (
            torch.sign(coords)
            *
            torch.log2(
                torch.abs(coords) * rct_coef
                + 1.0
            )
            /
            math.log2(rct_coef)
        )

        # Need:
        # [B, TF, qHW, TP, kHW, 3]
        coords = coords.permute(
            0, 1, 3, 2, 4, 5
        ).contiguous()

        coords = coords.view(
            B,
            TF * HW,
            TP * HW,
            3,
        )

        bias = self.cpb_mlp(coords)

        # [B, heads, Lq, Lv]
        bias = bias.permute(
            0, 3, 1, 2
        ).contiguous()

        bias = (
            self.rpe_coef
            *
            torch.sigmoid(bias)
        )

        return bias


class TATLayer(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        window_size,
        num_heads,
        dropout=0.0,
        max_time=32,
    ):
        super().__init__()

        self.channels = int(channels)
        self.hidden_channels = int(
            hidden_channels
        )
        self.window_size = tuple(
            window_size
        )
        self.num_heads = int(num_heads)

        if (
            self.hidden_channels
            %
            self.num_heads
            !=
            0
        ):
            raise ValueError(
                'hidden_channels must be '
                'divisible by num_heads'
            )

        self.fc_in = nn.Sequential(
            nn.Linear(
                self.channels,
                self.hidden_channels,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_channels,
                self.hidden_channels,
            ),
            nn.Dropout(dropout),
        )

        self.fc_out = nn.Sequential(
            nn.Linear(
                self.hidden_channels,
                self.hidden_channels,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.hidden_channels,
                self.channels,
            ),
            nn.Dropout(dropout),
        )

        self.norm1 = nn.LayerNorm(
            self.channels
        )

        self.norm2 = nn.LayerNorm(
            self.channels
        )

        self.mlp = nn.Sequential(
            nn.Linear(
                self.channels,
                self.channels,
            ),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(
                self.channels,
                self.channels,
            ),
            nn.Dropout(dropout),
        )

        self.logit_scale = nn.Parameter(
            torch.log(
                10.0
                *
                torch.ones(
                    self.num_heads,
                    1,
                    1,
                )
            )
        )

        self.register_buffer(
            'logit_max',
            torch.log(
                torch.tensor(
                    100.0
                )
            ),
            persistent=False,
        )

        self.rpe = RelativeTemporalPositionBias(
            window_size=self.window_size,
            num_heads=self.num_heads,
            max_time=max_time,
        )

    def forward(
        self,
        past_features,
        future_features,
        past_offsets,
        future_offsets,
    ):
        """
        past_features:
            [B, TP+1, C, H, W]
            last slot = current

        future_features:
            [B, TF, C, H, W]

        past_offsets:
            [B, TP]

        future_offsets:
            [B, TF]
        """
        B, TP1, C, H, W = (
            past_features.shape
        )

        TP = TP1 - 1

        if TP <= 0:
            return future_features

        past = past_features.permute(
            0, 1, 3, 4, 2
        ).contiguous()

        future = future_features.permute(
            0, 1, 3, 4, 2
        ).contiguous()

        past_in = self.fc_in(past)
        future_in = self.fc_in(future)

        # Transtreaming:
        # Q = future/current proposal
        # K = historical feature
        # V = historical delta w.r.t. current feature.
        query, qmeta = _window_partition(
            future_in,
            self.window_size,
        )

        historical = past_in[:, :-1]
        current = past_in[:, -1:]

        key, _ = _window_partition(
            historical,
            self.window_size,
        )

        value, _ = _window_partition(
            historical - current,
            self.window_size,
        )

        B, N, Lq, HC = query.shape
        _, _, Lv, _ = key.shape

        D = (
            HC
            //
            self.num_heads
        )

        q = query.view(
            B,
            N,
            Lq,
            self.num_heads,
            D,
        ).permute(
            0, 1, 3, 2, 4
        )

        k = key.view(
            B,
            N,
            Lv,
            self.num_heads,
            D,
        ).permute(
            0, 1, 3, 2, 4
        )

        v = value.view(
            B,
            N,
            Lv,
            self.num_heads,
            D,
        ).permute(
            0, 1, 3, 2, 4
        )

        attn = (
            q
            @
            k.transpose(-2, -1)
        )

        scale = torch.clamp(
            self.logit_scale,
            max=self.logit_max,
        ).exp()

        attn = attn * scale

        position = self.rpe(
            past_offsets,
            future_offsets,
        )

        attn = (
            attn
            +
            position[:, None]
        )

        attn = torch.softmax(
            attn,
            dim=-1,
        )

        out = attn @ v

        out = out.permute(
            0, 1, 3, 2, 4
        ).contiguous()

        out = out.view(
            B,
            N,
            Lq,
            HC,
        )

        out = _window_reverse(
            out,
            qmeta,
            self.window_size,
        )

        out = self.fc_out(out)

        x = (
            future
            +
            self.norm1(out)
        )

        x = (
            x
            +
            self.norm2(
                self.mlp(x)
            )
        )

        x = x.permute(
            0, 1, 4, 2, 3
        ).contiguous()

        return x


class TranstreamingBEVTAT(nn.Module):
    """
    Transtreaming-style temporal adaptive transformer
    on StreamDSGN BEV spatial_features.

    Input:
        current BEV:
            batch_dict['spatial_features']

        processed history:
            batch_dict['history_features']

        optional exact temporal coordinates:
            batch_dict['transtreaming_past_offsets']
            batch_dict['transtreaming_future_offsets']

    Output:
        batch_dict[
            'transtreaming_future_spatial_features'
        ]:
            [B, TF, C, H, W]

    For compatibility:
        batch_dict['spatial_features']
        is set to the first future feature.
    """

    def __init__(
        self,
        model_cfg,
        input_channels,
    ):
        super().__init__()

        self.model_cfg = model_cfg

        self.feature_name = model_cfg.get(
            'FEATURE_NAME',
            'spatial_features',
        )

        self.window_size = tuple(
            model_cfg.get(
                'WINDOW_SIZE',
                [8, 8],
            )
        )

        self.num_heads = int(
            model_cfg.get(
                'NUM_HEADS',
                4,
            )
        )

        hidden_channels = int(
            model_cfg.get(
                'HIDDEN_CHANNELS',
                input_channels,
            )
        )

        depth = int(
            model_cfg.get(
                'DEPTH',
                1,
            )
        )

        dropout = float(
            model_cfg.get(
                'DROPOUT',
                0.0,
            )
        )

        max_time = int(
            model_cfg.get(
                'MAX_TIME',
                32,
            )
        )

        self.past_length = int(
            model_cfg.get(
                'PAST_LENGTH',
                3,
            )
        )

        self.layers = nn.ModuleList([
            TATLayer(
                channels=input_channels,
                hidden_channels=hidden_channels,
                window_size=self.window_size,
                num_heads=self.num_heads,
                dropout=dropout,
                max_time=max_time,
            )
            for _ in range(depth)
        ])

        self.num_bev_features = int(
            input_channels
        )

    def forward(self, batch_dict):
        name = self.feature_name

        current = batch_dict[name]

        B, C, H, W = (
            current.shape
        )

        history = list(
            batch_dict.get(
                'history_features',
                [],
            )
        )

        explicit_offsets = batch_dict.get(
            'transtreaming_past_offsets',
            None,
        )

        if explicit_offsets is not None:
            if isinstance(
                explicit_offsets,
                torch.Tensor,
            ):
                n_explicit = (
                    explicit_offsets.shape[-1]
                    if explicit_offsets.ndim > 0
                    else 1
                )
            else:
                n_explicit = len(
                    explicit_offsets
                )

            history = history[
                -n_explicit:
            ]
        else:
            history = history[
                -self.past_length:
            ]

        historical_features = []

        for item in history:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
            ):
                raise RuntimeError(
                    'history_features item '
                    'must be '
                    '(sample_idx, feature_dict)'
                )

            feature_dict = item[1]

            if name not in feature_dict:
                raise KeyError(
                    f'{name} missing from '
                    'history feature'
                )

            historical_features.append(
                feature_dict[name]
            )

        num_history = len(
            historical_features
        )

        if explicit_offsets is None:
            # Offline/no-drop fallback only.
            # Formal streaming evaluator will always
            # supply the real offsets.
            past_offsets = list(
                range(
                    -num_history,
                    0,
                )
            )
        else:
            if isinstance(
                explicit_offsets,
                torch.Tensor,
            ):
                if explicit_offsets.ndim == 2:
                    past_offsets = (
                        explicit_offsets
                    )
                else:
                    past_offsets = (
                        explicit_offsets
                        .view(-1)
                        .tolist()
                    )
            else:
                past_offsets = list(
                    explicit_offsets
                )

        if num_history == 0:
            past = current[:, None]
        else:
            past = torch.stack(
                [
                    *historical_features,
                    current,
                ],
                dim=1,
            )

        future_offsets = batch_dict.get(
            'transtreaming_future_offsets',
            [1],
        )

        future_offsets = _to_2d_long(
            future_offsets,
            B,
            current.device,
        )

        TF = future_offsets.shape[1]

        # Initial future query = current feature.
        future = (
            current[:, None]
            .expand(
                -1,
                TF,
                -1,
                -1,
                -1,
            )
            .contiguous()
        )

        if num_history > 0:
            past_offsets = _to_2d_long(
                past_offsets,
                B,
                current.device,
            )

            if (
                past_offsets.shape[1]
                !=
                num_history
            ):
                raise RuntimeError(
                    'past temporal coordinate '
                    'count does not match '
                    f'history count: '
                    f'{past_offsets.shape[1]} '
                    f'!= {num_history}'
                )

            for layer in self.layers:
                future = layer(
                    past_features=past,
                    future_features=future,
                    past_offsets=past_offsets,
                    future_offsets=future_offsets,
                )

        batch_dict[
            'transtreaming_future_spatial_features'
        ] = future

        batch_dict[
            'transtreaming_future_offsets_used'
        ] = future_offsets

        batch_dict[name] = future[:, 0]

        return batch_dict
