# Convert sparse or dense 3D tensors into BEV 2D tensors by dimension rearrangement

import torch.nn as nn


class HeightCompression(nn.Module):
    """
    Original StreamDSGN HeightCompression.
    """

    def __init__(
        self,
        model_cfg,
        **kwargs,
    ):
        super().__init__()

        self.model_cfg = model_cfg

        self.num_bev_features = (
            self.model_cfg.NUM_BEV_FEATURES
        )

        self.sparse_input = getattr(
            self.model_cfg,
            'SPARSE_INPUT',
            True,
        )

    def forward(
        self,
        batch_dict,
    ):
        if 'volume_features' not in batch_dict:

            encoded_spconv_tensor = (
                batch_dict[
                    'encoded_spconv_tensor'
                ]
            )

            spatial_features = (
                encoded_spconv_tensor.dense()
            )

            batch_dict[
                'volume_features'
            ] = spatial_features

        else:

            spatial_features = (
                batch_dict[
                    'volume_features'
                ]
            )

        N, C, D, H, W = (
            spatial_features.shape
        )

        spatial_features = (
            spatial_features.view(
                N,
                C * D,
                H,
                W,
            )
        )

        batch_dict[
            'spatial_features'
        ] = spatial_features

        if self.sparse_input:

            batch_dict[
                'spatial_features_stride'
            ] = batch_dict[
                'encoded_spconv_tensor_stride'
            ]

        else:

            batch_dict[
                'spatial_features_stride'
            ] = 1

        return batch_dict


class ProjectedHeightCompression(nn.Module):
    """
    HeightCompression for the Light state generator.

    Example Light-v1:

        RPN3D output:
            [N, 16, 3, H, W]

        flatten height:
            [N, 48, H, W]

        projection:
            48 -> 96

        final temporal-memory interface:
            [N, 96, H, W]

    The projection is linear (1x1 Conv only):
        no BN
        no activation

    so it acts purely as a channel adapter.
    """

    def __init__(
        self,
        model_cfg,
        **kwargs,
    ):
        super().__init__()

        self.model_cfg = model_cfg

        self.num_bev_features = int(
            self.model_cfg.NUM_BEV_FEATURES
        )

        self.input_bev_features = int(
            self.model_cfg.INPUT_BEV_FEATURES
        )

        self.sparse_input = getattr(
            self.model_cfg,
            'SPARSE_INPUT',
            True,
        )

        self.bev_projection = nn.Conv2d(
            self.input_bev_features,
            self.num_bev_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

    def forward(
        self,
        batch_dict,
    ):
        if 'volume_features' not in batch_dict:

            encoded_spconv_tensor = (
                batch_dict[
                    'encoded_spconv_tensor'
                ]
            )

            spatial_features = (
                encoded_spconv_tensor.dense()
            )

            batch_dict[
                'volume_features'
            ] = spatial_features

        else:

            spatial_features = (
                batch_dict[
                    'volume_features'
                ]
            )

        N, C, D, H, W = (
            spatial_features.shape
        )

        spatial_features = (
            spatial_features.view(
                N,
                C * D,
                H,
                W,
            )
        )

        if (
            spatial_features.shape[1]
            !=
            self.input_bev_features
        ):
            raise RuntimeError(
                "ProjectedHeightCompression expected "
                f"{self.input_bev_features} input BEV channels, "
                f"but got {spatial_features.shape[1]}. "
                f"Raw volume shape was "
                f"{(N, C, D, H, W)}."
            )

        spatial_features = (
            self.bev_projection(
                spatial_features
            )
        )

        batch_dict[
            'spatial_features'
        ] = spatial_features

        if self.sparse_input:

            batch_dict[
                'spatial_features_stride'
            ] = batch_dict[
                'encoded_spconv_tensor_stride'
            ]

        else:

            batch_dict[
                'spatial_features_stride'
            ] = 1

        return batch_dict
