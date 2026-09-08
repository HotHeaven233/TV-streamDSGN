#!/usr/bin/env python3

import argparse
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--base_cfg',
        required=True,
    )

    parser.add_argument(
        '--output',
        required=True,
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
    )

    args = parser.parse_args()

    src = Path(args.base_cfg)
    dst = Path(args.output)

    with src.open() as f:
        cfg = yaml.safe_load(f)

    data = cfg[
        'DATA_CONFIG'
    ]

    model = cfg[
        'MODEL'
    ]

    optim = cfg[
        'OPTIMIZATION'
    ]

    # ============================================================
    # Dataset temporal metadata
    # ============================================================

    # Existing dataset helper creates arbitrary nextK.
    data[
        'MTD_FUTURE_STEPS'
    ] = [
        2,
        4,
        8,
    ]

    data[
        'TRANSTREAMING_PAST_STEPS'
    ] = [
        3,
        4,
        6,
        8,
    ]

    # Requiring next8 valid guarantees
    # next1/2/4 are valid too.
    data[
        'BOX3D_SUPERVISION'
    ] = 'next8'

    # Database sampling was prepared for the
    # original temporal tag set. Do not invent
    # DB entries for prev8/next8.
    #
    # This is temporal-module fine-tuning from
    # the pretrained K3 model, so disabling GT
    # database sampling is the safest choice.
    augmentors = data.get(
        'TRAIN_DATA_AUGMENTOR',
        [],
    )

    data[
        'TRAIN_DATA_AUGMENTOR'
    ] = [
        x
        for x in augmentors
        if x.get(
            'NAME',
            None,
        ) != 'gt_sampling'
    ]

    # ============================================================
    # Detector
    # ============================================================

    model[
        'NAME'
    ] = 'transtreaming_stream'

    # Superset of candidate historical frames.
    # During each training iteration only one
    # 3-frame temporal pattern is actually run.
    model[
        'HISTORY_TAG'
    ] = [
        'prev8',
        'prev6',
        'prev4',
        'prev3',
        'prev2',
        'prev',
    ]

    model[
        'HISTORY_FEATURES_NAME'
    ] = [
        'spatial_features',
    ]

    model[
        'FUSION_STAGE'
    ] = [
        'TranstreamingBEVTAT',
    ]

    # Replace original FeatureAlignment /
    # MultiHistoryResidualFeatureAlignment.
    model[
        'FUSION_IN_SPATIAL_FEATURES'
    ] = {
        'NAME':
            'TranstreamingBEVTAT',

        'FEATURE_NAME':
            'spatial_features',

        'WINDOW_SIZE':
            [8, 8],

        'NUM_HEADS':
            4,

        'HIDDEN_CHANNELS':
            96,

        'DEPTH':
            1,

        'DROPOUT':
            0.0,

        'MAX_TIME':
            32,

        # Historical feature buffers excluding
        # current feature.
        'PAST_LENGTH':
            3,
    }

    model[
        'TRANSTREAMING'
    ] = {
        'ENABLED':
            True,

        'FREEZE_FEATURE_EXTRACTOR':
            True,

        # Same shared head receives all horizons.
        'TRAIN_FUTURE_STEPS':
            [1, 2, 4, 8],

        # Three history features + current.
        'TRAIN_PAST_PATTERNS': [
            [-4, -2, -1],
            [-8, -4, -2],
            [-6, -4, -2],
            [-3, -2, -1],
        ],

        'PAST_PATTERN_WEIGHTS':
            [10, 5, 5, 5],
    }

    # ============================================================
    # Shared detection head
    # ============================================================

    dense = model[
        'DENSE_HEAD'
    ]

    # Fixed internal key; the detector aliases this
    # key to next/next2/next4/next8 during each
    # shared-head training pass.
    dense[
        'BOX3D_SUPERVISION'
    ] = 'next8'

    # Original StreamDSGN trend loss is defined
    # around token/prev/prev2 and is not a valid
    # arbitrary-horizon Transtreaming loss.
    dense[
        'HISTORY_TAG'
    ] = []

    loss_cfg = dense.get(
        'LOSS_CONFIG',
        {},
    )

    loss_weights = loss_cfg.get(
        'LOSS_WEIGHTS',
        {},
    )

    if (
        'trend_weight'
        in
        loss_weights
    ):
        loss_weights[
            'trend_weight'
        ] = 0.0

    # No old future-feature KD branch.
    model[
        'USE_KD'
    ] = None

    # ============================================================
    # Fine tuning
    # ============================================================

    optim[
        'NUM_EPOCHS'
    ] = int(
        args.epochs
    )

    optim[
        'LR'
    ] = float(
        args.lr
    )

    dst.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with dst.open('w') as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
        )

    print(
        'wrote:',
        dst,
    )

    print(
        'future steps:',
        model[
            'TRANSTREAMING'
        ][
            'TRAIN_FUTURE_STEPS'
        ],
    )

    print(
        'past patterns:',
        model[
            'TRANSTREAMING'
        ][
            'TRAIN_PAST_PATTERNS'
        ],
    )


if __name__ == '__main__':
    main()
