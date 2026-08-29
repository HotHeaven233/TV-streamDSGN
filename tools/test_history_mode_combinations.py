#!/usr/bin/env python3

import argparse
import copy
import json
import pickle
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch

# Reuse the already verified fixed-frame counterfactual utilities.
from test_counterfactual_state_value import (
    CLASSES,
    build_scene_index,
    clone_feature_dict,
    evaluate_predictions,
    extract_state,
    get_gt,
    make_cfg,
    mean_3d_moderate,
    run_light_prediction,
    set_seed,
)

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


# ============================================================
# History pattern definition
#
# IMPORTANT:
#
# pattern order = oldest -> newest
#
#   index 0 = H_{t-3}
#   index 1 = H_{t-2}
#   index 2 = H_{t-1}
#
# Examples:
#
#   LLL = [L(t-3), L(t-2), L(t-1)]
#   LLF = [L(t-3), L(t-2), F(t-1)]
#   FLL = [F(t-3), L(t-2), L(t-1)]
#
# Current t is ALWAYS Light.
# ============================================================

PATTERNS = (
    "LLL",
    "FLL",
    "LFL",
    "LLF",
    "FFL",
    "FLF",
    "LFF",
    "FFF",
)


# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fixed-frame cross-mode history compatibility experiment. "
            "Current frame always runs Light. "
            "Only the three stored history states vary between Full/Light."
        )
    )

    parser.add_argument(
        "--full_cfg",
        required=True,
    )

    parser.add_argument(
        "--full_ckpt",
        required=True,
    )

    parser.add_argument(
        "--light_cfg",
        required=True,
    )

    parser.add_argument(
        "--light_ckpt",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--anchor_stride",
        type=int,
        default=1,
        help=(
            "Evaluate every N-th valid current frame. "
            "Use >1 for quick sanity checks."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--save_predictions",
        action="store_true",
    )

    return parser.parse_args()


# ============================================================
# Helpers
# ============================================================

def clone_state_entry(entry):
    sample_idx, feature_dict = entry

    return (
        copy.deepcopy(sample_idx),
        clone_feature_dict(
            feature_dict
        ),
    )


def build_pattern_queue(
    pattern,
    history_positions,
    full_cache,
    light_cache,
):
    """
    Build:

        [
            H_{t-3}^{pattern[0]},
            H_{t-2}^{pattern[1]},
            H_{t-1}^{pattern[2]}
        ]

    oldest -> newest
    """

    if len(pattern) != 3:
        raise ValueError(
            f"Pattern must have length 3: {pattern}"
        )

    if len(history_positions) != 3:
        raise ValueError(
            "Expected exactly three history positions"
        )

    queue = deque(
        maxlen=3
    )

    for mode, pos in zip(
        pattern,
        history_positions,
    ):
        if mode == "F":
            entry = full_cache[pos]
        elif mode == "L":
            entry = light_cache[pos]
        else:
            raise ValueError(
                f"Unknown mode {mode}"
            )

        queue.append(
            clone_state_entry(
                entry
            )
        )

    return queue


def count_full_states(pattern):
    return pattern.count("F")


def pattern_description(pattern):
    labels = []

    slots = (
        "oldest(t-3)",
        "middle(t-2)",
        "newest(t-1)",
    )

    for slot, mode in zip(
        slots,
        pattern,
    ):
        labels.append(
            f"{slot}={mode}"
        )

    return ", ".join(labels)


def metric_for_class(
    metrics,
    cls,
    iou="0.7",
):
    return metrics[
        cls
    ][
        iou
    ][
        "3D"
    ][
        "moderate"
    ]


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.anchor_stride <= 0:
        raise ValueError(
            "--anchor_stride must be > 0"
        )

    set_seed(
        args.seed
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger = common_utils.create_logger(
        log_file=str(
            output_dir / "run.log"
        )
    )

    # ========================================================
    # Config
    # ========================================================

    full_cfg = make_cfg(
        args.full_cfg
    )

    light_cfg = make_cfg(
        args.light_cfg
    )

    if (
        list(full_cfg.CLASS_NAMES)
        !=
        list(light_cfg.CLASS_NAMES)
    ):
        raise RuntimeError(
            "Full and Light CLASS_NAMES differ"
        )

    # ========================================================
    # Dataset
    # ========================================================

    dataset, _, _ = build_dataloader(
        dataset_cfg=
            light_cfg.DATA_CONFIG,
        class_names=
            light_cfg.CLASS_NAMES,
        batch_size=1,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "Empty dataset"
        )

    # ========================================================
    # Models
    # ========================================================

    full_model = build_network(
        model_cfg=
            full_cfg.MODEL,
        num_class=
            len(full_cfg.CLASS_NAMES),
        dataset=dataset,
    )

    light_model = build_network(
        model_cfg=
            light_cfg.MODEL,
        num_class=
            len(light_cfg.CLASS_NAMES),
        dataset=dataset,
    )

    full_model.load_params_from_file(
        filename=args.full_ckpt,
        logger=logger,
        to_cpu=True,
    )

    light_model.load_params_from_file(
        filename=args.light_ckpt,
        logger=logger,
        to_cpu=True,
    )

    full_model.cuda()
    light_model.cuda()

    full_model.eval()
    light_model.eval()

    # ========================================================
    # Interface checks
    # ========================================================

    if (
        full_model.history_feature_queue
        is None
        or
        light_model.history_feature_queue
        is None
    ):
        raise RuntimeError(
            "Both models must have history queues"
        )

    if (
        light_model.history_feature_queue.maxlen
        !=
        3
    ):
        raise RuntimeError(
            "This experiment assumes K=3, but "
            f"Light queue maxlen="
            f"{light_model.history_feature_queue.maxlen}"
        )

    if (
        full_model.history_features_name
        !=
        light_model.history_features_name
    ):
        raise RuntimeError(
            "Full / Light state feature names differ"
        )

    # ========================================================
    # Scene chronology
    # ========================================================

    scene_to_indices = build_scene_index(
        dataset
    )

    valid_anchors = []

    for scene, indices in (
        scene_to_indices.items()
    ):
        # Current frame t requires:
        #
        # t-3, t-2, t-1
        #
        # Therefore current position starts at 3.
        for pos in range(
            3,
            len(indices),
            args.anchor_stride,
        ):
            valid_anchors.append(
                (
                    scene,
                    pos,
                )
            )

    if not valid_anchors:
        raise RuntimeError(
            "No valid current-frame anchors"
        )

    # ========================================================
    # Print protocol
    # ========================================================

    print()
    print("=" * 100)
    print(
        "Cross-Mode History Compatibility Experiment"
    )
    print("=" * 100)

    print(
        f"Full cfg      : {args.full_cfg}"
    )
    print(
        f"Full ckpt     : {args.full_ckpt}"
    )

    print(
        f"Light cfg     : {args.light_cfg}"
    )
    print(
        f"Light ckpt    : {args.light_ckpt}"
    )

    print(
        f"anchor stride : {args.anchor_stride}"
    )

    print(
        f"valid anchors : {len(valid_anchors)}"
    )

    print()
    print(
        "Current compute:"
    )
    print(
        "    ALWAYS Light"
    )

    print()
    print(
        "History order:"
    )
    print(
        "    [oldest, middle, newest]"
    )
    print(
        "    [H_{t-3}, H_{t-2}, H_{t-1}]"
    )

    print()
    print(
        "Patterns:"
    )

    for pattern in PATTERNS:
        print(
            f"    {pattern}: "
            f"{pattern_description(pattern)}"
        )

    print()
    print(
        "No latency / no buffer / no dropping."
    )
    print(
        "All patterns evaluate exactly the same current frames."
    )
    print("=" * 100)
    print()

    # ========================================================
    # Prediction storage
    # ========================================================

    gt_annos = []

    predictions = {
        pattern: []
        for pattern in PATTERNS
    }

    # State difference diagnostics.
    state_mean_abs_diffs = []

    # ========================================================
    # Run scene by scene
    #
    # We use a rolling state cache.
    #
    # For adjacent anchors:
    #
    #   t     needs t-3,t-2,t-1
    #   t+1   needs t-2,t-1,t
    #
    # therefore a state only needs to be extracted once
    # while it remains in the 3-frame window.
    # ========================================================

    processed_anchor_count = 0

    for scene_no, (
        scene,
        indices,
    ) in enumerate(
        scene_to_indices.items(),
        start=1,
    ):

        if len(indices) < 4:
            continue

        print()
        print(
            f"[scene {scene_no}/{len(scene_to_indices)}] "
            f"{scene} | frames={len(indices)}"
        )

        # key: scene position
        #
        # value:
        #   (sample_idx, feature_dict)
        full_cache = {}
        light_cache = {}

        current_positions = list(
            range(
                3,
                len(indices),
                args.anchor_stride,
            )
        )

        for pos in current_positions:

            history_positions = (
                pos - 3,
                pos - 2,
                pos - 1,
            )

            # --------------------------------------------
            # Ensure both F/L states exist for every
            # history slot.
            # --------------------------------------------

            for hist_pos in (
                history_positions
            ):
                if hist_pos not in (
                    light_cache
                ):
                    dataset_index = (
                        indices[
                            hist_pos
                        ]
                    )

                    light_cache[
                        hist_pos
                    ] = extract_state(
                        light_model,
                        dataset,
                        dataset_index,
                    )

                if hist_pos not in (
                    full_cache
                ):
                    dataset_index = (
                        indices[
                            hist_pos
                        ]
                    )

                    full_cache[
                        hist_pos
                    ] = extract_state(
                        full_model,
                        dataset,
                        dataset_index,
                    )

                    # Unified-interface diagnostic:
                    #
                    # compare F/L state for same frame.
                    (
                        _,
                        f_state,
                    ) = full_cache[
                        hist_pos
                    ]

                    (
                        _,
                        l_state,
                    ) = light_cache[
                        hist_pos
                    ]

                    for feature_name in (
                        f_state
                    ):
                        f = f_state[
                            feature_name
                        ]

                        l = l_state[
                            feature_name
                        ]

                        if (
                            torch.is_tensor(f)
                            and
                            torch.is_tensor(l)
                        ):
                            if f.shape != l.shape:
                                raise RuntimeError(
                                    "Full / Light state "
                                    "shape mismatch: "
                                    f"{f.shape} vs {l.shape}"
                                )

                            state_mean_abs_diffs.append(
                                (
                                    f.float()
                                    -
                                    l.float()
                                )
                                .abs()
                                .mean()
                                .item()
                            )

            # --------------------------------------------
            # Current target frame is identical across
            # every history pattern.
            # --------------------------------------------

            target_index = (
                indices[pos]
            )

            gt_annos.append(
                get_gt(
                    dataset,
                    target_index,
                )
            )

            # --------------------------------------------
            # Enumerate all 8 history compositions.
            #
            # Current t is ALWAYS Light.
            # --------------------------------------------

            for pattern in PATTERNS:

                history_queue = (
                    build_pattern_queue(
                        pattern=
                            pattern,
                        history_positions=
                            history_positions,
                        full_cache=
                            full_cache,
                        light_cache=
                            light_cache,
                    )
                )

                (
                    anno,
                    _,
                ) = run_light_prediction(
                    light_model,
                    dataset,
                    target_index,
                    history_queue,
                )

                predictions[
                    pattern
                ].append(
                    anno
                )

            processed_anchor_count += 1

            if (
                processed_anchor_count % 25
                ==
                0
            ):
                print(
                    f"[progress] "
                    f"{processed_anchor_count}/"
                    f"{len(valid_anchors)}"
                )

            # --------------------------------------------
            # Rolling-cache cleanup.
            #
            # For the immediately following anchor t+1,
            # the oldest required position is t-2.
            #
            # With anchor_stride > 1 this cleanup is still
            # safe because missing positions are regenerated.
            # --------------------------------------------

            oldest_to_keep = (
                pos
                -
                2
            )

            for cache in (
                full_cache,
                light_cache,
            ):
                stale_keys = [
                    key
                    for key in cache
                    if key < oldest_to_keep
                ]

                for key in stale_keys:
                    del cache[key]

    # ========================================================
    # Evaluate
    # ========================================================

    if len(gt_annos) != (
        len(valid_anchors)
    ):
        raise RuntimeError(
            "GT count mismatch: "
            f"{len(gt_annos)} vs "
            f"{len(valid_anchors)}"
        )

    results = {}

    print()
    print("=" * 120)
    print(
        "MAIN RESULT | current = Light "
        "| fixed-frame AP_R40 3D@0.7 Moderate"
    )
    print("=" * 120)

    # First evaluate all patterns.
    for pattern in PATTERNS:

        (
            full_text,
            paper_text,
            metrics,
        ) = evaluate_predictions(
            dataset,
            gt_annos,
            predictions[
                pattern
            ],
        )

        mean_07 = (
            mean_3d_moderate(
                metrics,
                "0.7",
            )
        )

        mean_05 = (
            mean_3d_moderate(
                metrics,
                "0.5",
            )
        )

        results[
            pattern
        ] = {
            "metrics":
                metrics,

            "mean_3d_moderate_0.7":
                mean_07,

            "mean_3d_moderate_0.5":
                mean_05,

            "num_full_states":
                count_full_states(
                    pattern
                ),

            "description":
                pattern_description(
                    pattern
                ),
        }

        (
            output_dir
            /
            f"{pattern}_full.txt"
        ).write_text(
            full_text
        )

        (
            output_dir
            /
            f"{pattern}_paper.txt"
        ).write_text(
            paper_text
        )

    # ========================================================
    # Delta relative to LLL
    # ========================================================

    baseline_07 = (
        results[
            "LLL"
        ][
            "mean_3d_moderate_0.7"
        ]
    )

    baseline_05 = (
        results[
            "LLL"
        ][
            "mean_3d_moderate_0.5"
        ]
    )

    for pattern in PATTERNS:
        results[
            pattern
        ][
            "delta_vs_LLL_0.7"
        ] = (
            results[
                pattern
            ][
                "mean_3d_moderate_0.7"
            ]
            -
            baseline_07
        )

        results[
            pattern
        ][
            "delta_vs_LLL_0.5"
        ] = (
            results[
                pattern
            ][
                "mean_3d_moderate_0.5"
            ]
            -
            baseline_05
        )

    # ========================================================
    # Main table
    # ========================================================

    print(
        f"{'Pattern':>8} "
        f"{'#F':>4} "
        f"{'Car':>10} "
        f"{'Ped':>10} "
        f"{'Cyc':>10} "
        f"{'Mean':>10} "
        f"{'Δ vs LLL':>12}"
    )

    print("-" * 120)

    for pattern in PATTERNS:

        metrics = (
            results[
                pattern
            ][
                "metrics"
            ]
        )

        car = (
            metrics[
                "Car"
            ][
                "0.7"
            ][
                "3D"
            ][
                "moderate"
            ]
        )

        ped = (
            metrics[
                "Pedestrian"
            ][
                "0.7"
            ][
                "3D"
            ][
                "moderate"
            ]
        )

        cyc = (
            metrics[
                "Cyclist"
            ][
                "0.7"
            ][
                "3D"
            ][
                "moderate"
            ]
        )

        mean = (
            results[
                pattern
            ][
                "mean_3d_moderate_0.7"
            ]
        )

        delta = (
            results[
                pattern
            ][
                "delta_vs_LLL_0.7"
            ]
        )

        print(
            f"{pattern:>8} "
            f"{count_full_states(pattern):>4d} "
            f"{car:>10.4f} "
            f"{ped:>10.4f} "
            f"{cyc:>10.4f} "
            f"{mean:>10.4f} "
            f"{delta:>+12.4f}"
        )

    print("-" * 120)

    # ========================================================
    # Position-value analysis
    # ========================================================

    position_effect = {
        "oldest_F_FLL":
            results[
                "FLL"
            ][
                "delta_vs_LLL_0.7"
            ],

        "middle_F_LFL":
            results[
                "LFL"
            ][
                "delta_vs_LLL_0.7"
            ],

        "newest_F_LLF":
            results[
                "LLF"
            ][
                "delta_vs_LLL_0.7"
            ],
    }

    print()
    print("=" * 80)
    print(
        "SINGLE-F POSITION EFFECT"
    )
    print("=" * 80)

    print(
        "FLL | Full at oldest  t-3 : "
        f"{position_effect['oldest_F_FLL']:+.4f} AP"
    )

    print(
        "LFL | Full at middle  t-2 : "
        f"{position_effect['middle_F_LFL']:+.4f} AP"
    )

    print(
        "LLF | Full at newest  t-1 : "
        f"{position_effect['newest_F_LLF']:+.4f} AP"
    )

    # ========================================================
    # Group by number of Full states
    #
    # NOTE:
    # These are simple averages of pattern-level mean AP,
    # not a new KITTI evaluation.
    # ========================================================

    grouped = defaultdict(
        list
    )

    for pattern in PATTERNS:
        grouped[
            count_full_states(
                pattern
            )
        ].append(
            results[
                pattern
            ][
                "mean_3d_moderate_0.7"
            ]
        )

    full_count_summary = {}

    print()
    print("=" * 80)
    print(
        "FULL-STATE COUNT SUMMARY"
    )
    print(
        "(mean of pattern-level 3D@0.7 Moderate mean AP)"
    )
    print("=" * 80)

    for n_full in sorted(
        grouped.keys()
    ):
        values = grouped[
            n_full
        ]

        value = float(
            np.mean(
                values
            )
        )

        full_count_summary[
            str(n_full)
        ] = value

        print(
            f"#Full={n_full}: "
            f"{value:.4f}"
        )

    # ========================================================
    # Pairwise marginal additions
    #
    # Useful for looking at diminishing returns.
    # ========================================================

    marginal = {
        # Starting from LLL.
        "LLL_to_LLF_add_newest":
            (
                results["LLF"]
                ["mean_3d_moderate_0.7"]
                -
                results["LLL"]
                ["mean_3d_moderate_0.7"]
            ),

        "LLL_to_LFL_add_middle":
            (
                results["LFL"]
                ["mean_3d_moderate_0.7"]
                -
                results["LLL"]
                ["mean_3d_moderate_0.7"]
            ),

        "LLL_to_FLL_add_oldest":
            (
                results["FLL"]
                ["mean_3d_moderate_0.7"]
                -
                results["LLL"]
                ["mean_3d_moderate_0.7"]
            ),

        # Add another Full when newest is already Full.
        "LLF_to_LFF_add_middle":
            (
                results["LFF"]
                ["mean_3d_moderate_0.7"]
                -
                results["LLF"]
                ["mean_3d_moderate_0.7"]
            ),

        "LLF_to_FLF_add_oldest":
            (
                results["FLF"]
                ["mean_3d_moderate_0.7"]
                -
                results["LLF"]
                ["mean_3d_moderate_0.7"]
            ),

        # Final move to FFF.
        "LFF_to_FFF_add_oldest":
            (
                results["FFF"]
                ["mean_3d_moderate_0.7"]
                -
                results["LFF"]
                ["mean_3d_moderate_0.7"]
            ),

        "FLF_to_FFF_add_middle":
            (
                results["FFF"]
                ["mean_3d_moderate_0.7"]
                -
                results["FLF"]
                ["mean_3d_moderate_0.7"]
            ),

        "FFL_to_FFF_add_newest":
            (
                results["FFF"]
                ["mean_3d_moderate_0.7"]
                -
                results["FFL"]
                ["mean_3d_moderate_0.7"]
            ),
    }

    print()
    print("=" * 80)
    print(
        "SELECTED MARGINAL FULL-STATE GAINS"
    )
    print("=" * 80)

    for key, value in (
        marginal.items()
    ):
        print(
            f"{key:35s}: "
            f"{value:+.4f} AP"
        )

    # ========================================================
    # Save optional predictions
    # ========================================================

    if args.save_predictions:
        with open(
            output_dir
            /
            "predictions.pkl",
            "wb",
        ) as f:
            pickle.dump(
                {
                    "gt":
                        gt_annos,

                    "predictions":
                        predictions,
                },
                f,
                protocol=
                    pickle.HIGHEST_PROTOCOL,
            )

    # ========================================================
    # Save summary
    # ========================================================

    summary = {
        "experiment":
            "cross_mode_history_compatibility",

        "history_order":
            [
                "t-3_oldest",
                "t-2_middle",
                "t-1_newest",
            ],

        "current_mode":
            "Light",

        "full_cfg":
            args.full_cfg,

        "full_ckpt":
            args.full_ckpt,

        "light_cfg":
            args.light_cfg,

        "light_ckpt":
            args.light_ckpt,

        "anchor_stride":
            args.anchor_stride,

        "num_samples":
            len(gt_annos),

        "mean_abs_full_light_state_difference":
            (
                float(
                    np.mean(
                        state_mean_abs_diffs
                    )
                )
                if state_mean_abs_diffs
                else None
            ),

        "patterns":
            results,

        "single_full_position_effect_0.7":
            position_effect,

        "mean_by_number_of_full_states_0.7":
            full_count_summary,

        "selected_marginal_gains_0.7":
            marginal,
    }

    with open(
        output_dir
        /
        "summary.json",
        "w",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Final print
    # ========================================================

    print()
    print("=" * 100)

    print(
        "Mean |H^F - H^L| = "
        f"{summary['mean_abs_full_light_state_difference']:.6f}"
    )

    print()

    print(
        f"Saved results: {output_dir}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
