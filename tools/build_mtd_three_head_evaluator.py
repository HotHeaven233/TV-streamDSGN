#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(
    text,
    old,
    new,
    name,
):
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{name}: expected one marker, "
            f"got {count}"
        )

    return text.replace(
        old,
        new,
        1,
    )


def build(
    base_path,
    out_path,
):
    base_path = Path(base_path)
    out_path = Path(out_path)

    src = base_path.read_text()

    # ------------------------------------------------------------
    # Runtime import
    # ------------------------------------------------------------

    src = replace_once(
        src,
        "\n\ndef parse_args():",
        """
from mtd_three_head_runtime import (
    MTDDelayAnalyzer,
    MTDThreeHeadBank,
    mtd_forward_no_post,
)


def parse_args():""",
        "runtime import",
    )

    # ------------------------------------------------------------
    # H2 / H3 checkpoints
    # ------------------------------------------------------------

    src = replace_once(
        src,
        '    p.add_argument("--ckpt", required=True)\n',
        '''    p.add_argument("--ckpt", required=True)
    p.add_argument("--h2_ckpt", required=True)
    p.add_argument("--h3_ckpt", required=True)
''',
        "checkpoint arguments",
    )

    # ------------------------------------------------------------
    # Build real neural head bank + causal DAM
    # ------------------------------------------------------------

    src = replace_once(
        src,
        "    model.cuda().eval()\n",
        '''    model.cuda().eval()

    # ============================================================
    # TRUE MTD:
    # shared trunk + H1/H2/H3 neural prediction heads
    # ============================================================

    mtd_heads = MTDThreeHeadBank(
        model=model,
        h2_ckpt=args.h2_ckpt,
        h3_ckpt=args.h3_ckpt,
        logger=logger,
    )

    dam = MTDDelayAnalyzer(
        period_ms=period_ms,
        max_horizon=3,
    )

    mtd_decisions = []
    mtd_branch_hist = Counter()
''',
        "MTD runtime construction",
    )

    # ------------------------------------------------------------
    # Warm all THREE branch kernels.
    #
    # Warmup is NOT allowed to seed formal DAM history.
    # ------------------------------------------------------------

    old_warmup = '''        original_forward_no_post(
            model=model,
            batch=batch,
            model_stream=model_stream,
            contender=(
                contenders[
                    warm_level
                ]
            ),
        )

        last_batch = batch'''

    new_warmup = '''        # Warm H1/H2/H3 CUDA kernels evenly.
        # These runtimes are deliberately NOT passed to DAM.
        warm_branch = (
            wi % 3
        ) + 1

        mtd_forward_no_post(
            model=model,
            batch=batch,
            model_stream=model_stream,
            contender=(
                contenders[
                    warm_level
                ]
            ),
            head_bank=mtd_heads,
            branch_step=warm_branch,
        )

        last_batch = batch'''

    src = replace_once(
        src,
        old_warmup,
        new_warmup,
        "three-head warmup",
    )

    # ------------------------------------------------------------
    # Formal evaluation must start with a clean causal DAM.
    # First formal frame therefore selects H1.
    # ------------------------------------------------------------

    src = replace_once(
        src,
        "    total_dataset = len(\n",
        '''    # Engineering warmup is outside the formal stream.
    # Do not leak its timing into DAM.
    dam.reset()

    total_dataset = len(
''',
        "formal DAM reset",
    )

    # ------------------------------------------------------------
    # Replace ONLY the actual detector forward.
    #
    # DAM decision happens before current forward:
    #
    #   queue wait known now
    #   + previous runtimes
    #   -> choose H1/H2/H3
    #
    # Current true forward time is observed only AFTER inference.
    # ------------------------------------------------------------

    old_forward = '''                _, forward_ms = (
                    original_forward_no_post(
                        model=model,
                        batch=batch,
                        model_stream=(
                            model_stream
                        ),
                        contender=(
                            contenders[
                                true_level
                            ]
                        ),
                    )
                )

                finish_ms = ('''

    new_forward = '''                # --------------------------------------------
                # MTD Delay Analysis Module
                #
                # CAUSAL: decision is made before current forward.
                # --------------------------------------------

                dam_queue_wait_ms = (
                    start_ms
                    -
                    arrival_ms
                )

                mtd_decision = (
                    dam.select(
                        queue_wait_ms=(
                            dam_queue_wait_ms
                        )
                    )
                )

                branch_step = int(
                    mtd_decision.branch_step
                )

                mtd_branch_hist[
                    branch_step
                ] += 1

                _, forward_ms = (
                    mtd_forward_no_post(
                        model=model,
                        batch=batch,
                        model_stream=(
                            model_stream
                        ),
                        contender=(
                            contenders[
                                true_level
                            ]
                        ),
                        head_bank=mtd_heads,
                        branch_step=(
                            branch_step
                        ),
                    )
                )

                # Current runtime becomes available ONLY now.
                dam.observe(
                    forward_ms
                )

                actual_response_for_branch_ms = (
                    dam_queue_wait_ms
                    +
                    forward_ms
                )

                # Oracle branch is diagnostic only.
                # It is NEVER used by the detector.
                oracle_branch_step = (
                    dam.branch_for_response_ms(
                        actual_response_for_branch_ms
                    )
                )

                mtd_decisions.append({
                    "global_index":
                        idx,

                    "scene":
                        scene,

                    "local_pos":
                        pos,

                    "frame_id":
                        frame_id,

                    "true_level":
                        true_level,

                    "queue_wait_ms":
                        dam_queue_wait_ms,

                    "estimated_forward_ms":
                        mtd_decision.estimated_forward_ms,

                    "estimated_response_ms":
                        mtd_decision.estimated_response_ms,

                    "estimated_delay_slots":
                        mtd_decision.delay_slots,

                    "runtime_history_count":
                        mtd_decision.runtime_history_count,

                    "branch_step":
                        branch_step,

                    "actual_forward_ms":
                        forward_ms,

                    "actual_response_ms":
                        actual_response_for_branch_ms,

                    "oracle_branch_step":
                        oracle_branch_step,

                    "branch_match_oracle":
                        int(
                            branch_step
                            ==
                            oracle_branch_step
                        ),
                })

                finish_ms = ('''

    src = replace_once(
        src,
        old_forward,
        new_forward,
        "formal MTD forward",
    )

    # ------------------------------------------------------------
    # Save routing audit
    # ------------------------------------------------------------

    old_outputs = '''        # ========================================================
        # Outputs
        # ========================================================

        timeline_path = ('''

    new_outputs = '''        # ========================================================
        # Outputs
        # ========================================================

        # --------------------------------------------------------
        # MTD routing audit
        # --------------------------------------------------------

        mtd_decision_path = (
            out_dir
            /
            "mtd_decisions.csv"
        )

        if len(mtd_decisions) > 0:
            with mtd_decision_path.open(
                "w",
                newline="",
            ) as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=list(
                        mtd_decisions[0].keys()
                    ),
                )

                w.writeheader()
                w.writerows(
                    mtd_decisions
                )

        timeline_path = ('''

    src = replace_once(
        src,
        old_outputs,
        new_outputs,
        "MTD audit output",
    )

    # ------------------------------------------------------------
    # Method identity
    # ------------------------------------------------------------

    src = src.replace(
        '"Original StreamDSGN"',
        '"MTD Three-Head StreamDSGN"',
    )

    src = src.replace(
        '"ORIGINAL StreamDSGN "',
        '"MTD Three-Head StreamDSGN "',
    )

    src = src.replace(
        '"original_streamdsgn_',
        '"mtd_three_head_streamdsgn_',
    )

    src = src.replace(
        '"vanilla_streamdsgn"',
        '"k3_streamdsgn_shared_trunk"',
    )

    # ------------------------------------------------------------
    # Summary metadata
    # ------------------------------------------------------------

    old_summary = '''            "model_ckpt":
                str(args.ckpt),

            "input_hz":'''

    new_summary = '''            "model_ckpt":
                str(args.ckpt),

            "mtd_heads": {
                "H1": {
                    "target": "next",
                    "checkpoint": str(
                        args.ckpt
                    ),
                },

                "H2": {
                    "target": "next2",
                    "checkpoint": str(
                        args.h2_ckpt
                    ),
                },

                "H3": {
                    "target": "next3",
                    "checkpoint": str(
                        args.h3_ckpt
                    ),
                },
            },

            "mtd_routing": {
                "decision_time":
                    "before_current_forward",

                "runtime_estimator":
                    "min_last_two_forward_ms",

                "estimated_response":
                    "queue_wait_plus_estimated_forward",

                "branch_rule":
                    "clip(floor(response/period)+1,1,3)",

                "branch_histogram":
                    dict(
                        mtd_branch_hist
                    ),

                "decision_csv":
                    str(
                        mtd_decision_path
                    ),

                "oracle_match_rate":
                    (
                        sum(
                            x[
                                "branch_match_oracle"
                            ]
                            for x
                            in mtd_decisions
                        )
                        /
                        len(mtd_decisions)
                        if len(
                            mtd_decisions
                        ) > 0
                        else 0.0
                    ),
            },

            "input_hz":'''

    src = replace_once(
        src,
        old_summary,
        new_summary,
        "MTD summary",
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_path.write_text(src)

    print(
        f"[WRITE] {out_path}"
    )


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--base",
        required=True,
    )

    p.add_argument(
        "--out",
        required=True,
    )

    a = p.parse_args()

    build(
        a.base,
        a.out,
    )


if __name__ == "__main__":
    main()
