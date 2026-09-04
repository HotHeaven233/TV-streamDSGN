#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pcdet.models.backbones_3d_stream.causal_prefix_bn_bank import (
    STAGE_NAMES,
    causal_prefixes_by_stage,
    prefix_key,
)

EXPECTED_COUNTS = {
    "res2": 3,
    "res3": 9,
    "res4": 19,
    "fpn": 34,
    "stereo": 55,
    "rpn": 83,
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bank", required=True)
    p.add_argument("--expect_batches", type=int, default=None)
    p.add_argument("--json_out", default=None)
    return p.parse_args()


def finite(x):
    return bool(torch.isfinite(x).all().item())


def close_to_default_stats(mean, var, atol=1e-6):
    return bool(
        torch.allclose(mean, torch.zeros_like(mean), atol=atol, rtol=0.0)
        and torch.allclose(var, torch.ones_like(var), atol=atol, rtol=0.0)
    )


def main():
    args = parse_args()
    bank = torch.load(args.bank, map_location="cpu")

    errors = []
    notes = []

    if bank.get("version") != "elastic_v4_bn_causal_prefix_bank_v1":
        errors.append(f"unexpected version: {bank.get('version')}")

    expected = causal_prefixes_by_stage()
    actual_counts = {}
    total = 0
    bn_entries = 0

    nbt_min = None
    nbt_max = 0
    zero_call_bn = 0
    positive_call_bn = 0

    for stage_name in STAGE_NAMES:
        stage_bank = bank.get("stages", {}).get(stage_name, {})
        actual_counts[stage_name] = len(stage_bank)
        total += len(stage_bank)

        if len(stage_bank) != EXPECTED_COUNTS[stage_name]:
            errors.append(
                f"{stage_name}: expected {EXPECTED_COUNTS[stage_name]} prefixes, "
                f"found {len(stage_bank)}"
            )

        expected_keys = {prefix_key(p) for p in expected[stage_name]}
        actual_keys = set(stage_bank.keys())

        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        if missing:
            errors.append(f"{stage_name}: missing prefixes: {missing}")
        if extra:
            errors.append(f"{stage_name}: unexpected prefixes: {extra}")

        for pkey, entry in stage_bank.items():
            bn_dict = entry.get("bn", {})
            if not bn_dict:
                errors.append(f"{stage_name}:{pkey}: empty BN dictionary")
                continue

            # This is the number of calibration DATASET FRAMES.  It is the
            # quantity that should equal --expect_batches.
            if args.expect_batches is not None:
                frames = int(entry.get("frames", -1))
                if frames != args.expect_batches:
                    errors.append(
                        f"{stage_name}:{pkey}: frames={frames}, "
                        f"expected={args.expect_batches}"
                    )

            for module_name, state in bn_dict.items():
                bn_entries += 1
                mean = state.get("running_mean")
                var = state.get("running_var")
                nbt = state.get("num_batches_tracked")

                if mean is None or var is None or nbt is None:
                    errors.append(
                        f"{stage_name}:{pkey}:{module_name}: incomplete BN state"
                    )
                    continue

                if not finite(mean):
                    errors.append(
                        f"{stage_name}:{pkey}:{module_name}: running_mean has NaN/Inf"
                    )
                if not finite(var):
                    errors.append(
                        f"{stage_name}:{pkey}:{module_name}: running_var has NaN/Inf"
                    )
                if bool((var < 0).any().item()):
                    errors.append(
                        f"{stage_name}:{pkey}:{module_name}: negative variance"
                    )

                n = int(nbt.item())
                nbt_min = n if nbt_min is None else min(nbt_min, n)
                nbt_max = max(nbt_max, n)

                # IMPORTANT:
                # num_batches_tracked is the number of BN FORWARD CALLS, not
                # the number of dataset frames.
                #
                # In the stereo StreamDSGN path, many 2D modules are applied to
                # both left and right images, so N input frames naturally give
                # 2N BN calls (e.g. 100 -> 200).
                #
                # Some residual shortcut BN modules are conditionally bypassed
                # for a given causal prefix.  For those exact prefixes the
                # shortcut is also bypassed at inference, so n==0 is valid.
                if n == 0:
                    zero_call_bn += 1
                    if not close_to_default_stats(mean, var):
                        errors.append(
                            f"{stage_name}:{pkey}:{module_name}: "
                            "num_batches_tracked=0 but running stats are not default"
                        )
                else:
                    positive_call_bn += 1

    if total != 203:
        errors.append(f"total prefix states={total}, expected=203")

    if bank.get("completed") is not True:
        errors.append("bank metadata completed != True")

    if int(bank.get("completed_prefix_states", -1)) != 203:
        errors.append(
            f"completed_prefix_states={bank.get('completed_prefix_states')}, "
            "expected=203"
        )

    if args.expect_batches is not None:
        notes.append(
            "frames/prefix is checked against --expect_batches. "
            "num_batches_tracked is intentionally NOT required to equal it, "
            "because a BN may run twice per frame (left/right) or be bypassed."
        )

    summary = {
        "bank": str(args.bank),
        "version": bank.get("version"),
        "dataset_split": bank.get("dataset_split"),
        "batches_per_prefix": bank.get("batches_per_prefix"),
        "actual_counts": actual_counts,
        "total_prefix_states": total,
        "bn_state_entries": bn_entries,
        "bn_forward_calls_min": nbt_min,
        "bn_forward_calls_max": nbt_max,
        "zero_call_conditional_bn_entries": zero_call_bn,
        "positive_call_bn_entries": positive_call_bn,
        "errors": errors,
        "notes": notes,
        "status": "PASS" if not errors else "FAIL",
    }

    print("=" * 80)
    print("203-prefix BN bank structural verification")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 80)

    if args.json_out:
        p = Path(args.json_out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

