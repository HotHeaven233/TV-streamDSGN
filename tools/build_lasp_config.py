#!/usr/bin/env python3

import argparse
import copy
from pathlib import Path

import yaml


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--base_cfg',
        required=True,
    )

    ap.add_argument(
        '--output',
        required=True,
    )

    ap.add_argument(
        '--intention_centers',
        default=(
            'outputs/lasp/'
            'intention_centers_h8_k6.npy'
        ),
    )

    ap.add_argument(
        '--epochs',
        type=int,
        default=15,
    )

    ap.add_argument(
        '--lr',
        type=float,
        default=2e-4,
    )

    args = ap.parse_args()

    base = Path(
        args.base_cfg
    )

    if not base.is_file():
        raise FileNotFoundError(
            base
        )

    cfg = yaml.safe_load(
        base.read_text()
    )

    if (
        'MODEL'
        not in cfg
        or
        'DATA_CONFIG'
        not in cfg
        or
        'OPTIMIZATION'
        not in cfg
    ):
        raise RuntimeError(
            f'{base} is not a '
            'complete StreamDSGN '
            'training config'
        )

    out = copy.deepcopy(
        cfg
    )

    model = out[
        'MODEL'
    ]

    data = out[
        'DATA_CONFIG'
    ]

    optim = out[
        'OPTIMIZATION'
    ]

    #
    # New detector.
    #
    model[
        'NAME'
    ] = 'stream_lasp'

    #
    # IMPORTANT:
    # The LASP model is initialized from the complete
    # Original StreamDSGN checkpoint via --pretrained_model.
    #
    # Do not trigger the legacy MMCV torchvision://resnet18
    # loader during model construction.
    #
    model[
        'BACKBONE_3D'
    ][
        'feature_backbone_pretrained'
    ] = None

    #
    # LASP freezes the stereo feature extractor.
    #
    # Original StreamDSGN's front-surface depth branch is an
    # auxiliary TRAINING-only supervision branch. Keeping its
    # loss enabled while the extractor is fixed is both useless
    # and inconsistent: the fixed extractor is intentionally kept
    # in eval mode and therefore does not emit depth_preds.
    #
    model[
        'BACKBONE_3D'
    ][
        'front_surface_depth'
    ] = False

    model[
        'DEPTH_LOSS_HEAD'
    ] = None

    #
    # Disable original BEV history.
    #
    model[
        'HISTORY_TAG'
    ] = None

    model[
        'FUSION_STAGE'
    ] = []

    model[
        'FUSION_IN_SPATIAL_FEATURES'
    ] = None

    model[
        'SAVE_TIME'
    ] = False

    #
    # Validation/checkpoint selection is OFFLINE only.
    #
    # Do not inherit Original StreamDSGN's stream_copy/stream_kf
    # metrics here: those require an empirical inference-time model
    # and are not used for selecting the LASP checkpoint.
    #
    model[
        'POST_PROCESSING'
    ][
        'EVAL_METRIC'
    ] = [
        'offline_3d'
    ]

    dense = model[
        'DENSE_HEAD'
    ]

    #
    # Important:
    # current detector state is supervised
    # at token time.
    #
    # Future motion is learned by LASP
    # trajectory head.
    #
    dense[
        'BOX3D_SUPERVISION'
    ] = 'token'

    dense[
        'HISTORY_TAG'
    ] = None

    dense[
        'predict_boxes_when_training'
    ] = True

    if (
        'LOSS_CONFIG'
        in dense
    ):

        dense[
            'LOSS_CONFIG'
        ][
            'TREND_LOSS_TYPE'
        ] = None

        if (
            'LOSS_WEIGHTS'
            in dense[
                'LOSS_CONFIG'
            ]
        ):

            lw = dense[
                'LOSS_CONFIG'
            ][
                'LOSS_WEIGHTS'
            ]

            lw[
                'trend_weight'
            ] = 0.0

            lw[
                'bal_hyparam'
            ] = 0.0

            lw[
                'adap_bal_hyparam'
            ] = 0.0

    data[
        'BOX3D_SUPERVISION'
    ] = 'token'

    #
    # Existing dataset helper creates
    # next2 ... next8 along same-scene
    # next chain.
    #
    data[
        'MTD_FUTURE_STEPS'
    ] = list(
        range(
            2,
            9,
        )
    )

    #
    # DB sampled objects do not carry
    # valid real future trajectories.
    #
    aug = data.get(
        'TRAIN_DATA_AUGMENTOR',
        [],
    )

    data[
        'TRAIN_DATA_AUGMENTOR'
    ] = [
        x
        for x
        in aug
        if x.get(
            'NAME'
        )
        != 'gt_sampling'
    ]

    model[
        'LASP'
    ] = {

        #
        # 128 proposal queries ×
        # 4 memory frames = 512
        # historical query slots max.
        #
        'TOPK':
            128,

        'MEMORY_FRAMES':
            4,

        #
        # Missing tags are simply ignored,
        # so official prev2/prev config works;
        # your extended datasets can also
        # supply prev3.
        #
        'TRAIN_HISTORY_TAGS':
            [
                'prev3',
                'prev2',
                'prev',
            ],

        #
        # Random history dropping creates
        # irregular training intervals.
        #
        'HISTORY_KEEP_PROB':
            0.75,

        #
        # Preserve expensive stereo extractor.
        #
        'FREEZE_FEATURE_EXTRACTOR':
            True,

        #
        # StreamDetHead.NUM_FILTERS
        #
        'QUERY_FEATURE_CHANNELS':
            int(
                dense.get(
                    'NUM_FILTERS',
                    64,
                )
            ),

        'EMBED_DIM':
            128,

        'NUM_HEADS':
            4,

        'NUM_DECODER_LAYERS':
            2,

        'FFN_MULT':
            4,

        'DROPOUT':
            0.0,

        #
        # LASP paper setting.
        #
        'NUM_BASIS':
            10,

        'EIG_CLIP':
            1.0,

        #
        # LASP paper setting.
        #
        'NUM_INTENTIONS':
            6,

        'INTENTION_CENTERS':
            args.intention_centers,

        'INTENTION_PE_DIM':
            64,

        'TRAJ_LAYERS':
            3,

        #
        # Continuous interpolation at runtime,
        # trained on future positions 1..8.
        #
        'FUTURE_STEPS':
            list(
                range(
                    1,
                    9,
                )
            ),

        'LOSS_WEIGHTS':
            {
                'box':
                    1.0,

                'cls':
                    1.0,

                'velocity':
                    0.5,

                'trajectory':
                    1.0,

                'intention':
                    0.2,
            },
    }

    optim[
        'NUM_EPOCHS'
    ] = args.epochs

    optim[
        'LR'
    ] = args.lr

    optim[
        'LR_WARMUP'
    ] = True

    optim[
        'WARMUP_EPOCH'
    ] = 1

    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        yaml.safe_dump(
            out,
            sort_keys=False,
            width=120,
        )
    )

    print(
        output
    )


if __name__ == '__main__':
    main()
