#!/usr/bin/env python3
"""
Materialize one fixed profile from a causal-prefix BN bank.

No convolution weights are changed.  The selected causal-prefix running
mean/variance entries are copied into the width-specific BN slots used by that
profile.  The resulting checkpoint can be evaluated by the existing
test_elastic_stream_v4_bn.py, which then fuses Conv+BN into static kernels.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pcdet.models.backbones_3d_stream.elastic_bev_branch_v4_bn import (
    validate_schedule,
)
from pcdet.models.backbones_3d_stream.causal_prefix_bn_bank import (
    STAGE_NAMES,
    prefix_key,
    required_prefix_keys,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--elastic_ckpt", required=True)
    p.add_argument("--prefix_bn_bank", required=True)
    p.add_argument("--schedule", required=True)
    p.add_argument("--output_ckpt", required=True)
    return p.parse_args()


def parse_schedule(text):
    return validate_schedule(tuple(float(x.strip()) for x in text.split(",")))


def main():
    args = parse_args()
    schedule = parse_schedule(args.schedule)

    source = torch.load(args.elastic_ckpt, map_location="cpu")
    bank = torch.load(args.prefix_bn_bank, map_location="cpu")
    if bank.get("version") != "elastic_v4_bn_causal_prefix_bank_v1":
        raise RuntimeError("Unsupported prefix BN bank")

    state = source["branch"]
    applied = []

    for stage_idx, ratio in enumerate(schedule):
        if float(ratio) >= 1.0:
            continue
        stage_name = STAGE_NAMES[stage_idx]
        prefix = tuple(schedule[: stage_idx + 1])
        pkey = prefix_key(prefix)
        entry = bank["stages"][stage_name].get(pkey)
        if entry is None:
            raise KeyError(f"Missing bank entry {stage_name}:{pkey}")

        for module_name, bn_state in entry["bn"].items():
            rk = bn_state["ratio_key"]
            base = f"{module_name}.bns.{rk}"
            keys = {
                "running_mean": f"{base}.running_mean",
                "running_var": f"{base}.running_var",
                "num_batches_tracked": f"{base}.num_batches_tracked",
            }
            for field, state_key in keys.items():
                if state_key not in state:
                    raise KeyError(f"Missing checkpoint key: {state_key}")
                state[state_key] = bn_state[field].clone()
        applied.append((stage_name, pkey))

    out = dict(source)
    out["branch"] = state
    out["bn_calibration"] = {
        "type": "causal_prefix_bank_materialized",
        "schedule": [float(x) for x in schedule],
        "prefix_bn_bank": str(args.prefix_bn_bank),
        "applied_prefixes": applied,
        "dataset_split": bank.get("dataset_split"),
        "affine_policy": bank.get("affine_policy"),
    }

    output = Path(args.output_ckpt)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, output)

    print("=" * 76)
    print("Materialized causal-prefix BN profile")
    print(f"schedule : {tuple(schedule)}")
    print(f"bank     : {args.prefix_bn_bank}")
    print(f"source   : {args.elastic_ckpt}")
    print(f"output   : {output}")
    print("prefixes :")
    for stage_name, pkey in applied:
        print(f"  {stage_name:7s} {pkey}")
    print("=" * 76)


if __name__ == "__main__":
    main()

