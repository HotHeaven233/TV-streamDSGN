#!/usr/bin/env python3

import argparse
import json
import math
import re
from pathlib import Path


FLOAT_RE = (
    r"[+-]?"
    r"(?:"
    r"\d+(?:\.\d*)?"
    r"|"
    r"\.\d+"
    r")"
    r"(?:[eE][+-]?\d+)?"
)


STRICT_HEADERS = {
    "Car": re.compile(
        r"^\s*Car\s+AP_R40@\s*"
        r"0\.70\s*,\s*0\.70\s*,\s*0\.70\s*:"
    ),

    "Pedestrian": re.compile(
        r"^\s*Pedestrian\s+AP_R40@\s*"
        r"0\.50\s*,\s*0\.50\s*,\s*0\.50\s*:"
    ),

    "Cyclist": re.compile(
        r"^\s*Cyclist\s+AP_R40@\s*"
        r"0\.50\s*,\s*0\.50\s*,\s*0\.50\s*:"
    ),
}


#
# IMPORTANT:
# Stop one AP block as soon as ANY next class AP header starts.
#
ANY_AP_HEADER = re.compile(
    r"^\s*"
    r"(?:Car|Pedestrian|Cyclist)"
    r"\s+AP(?:_R40)?@"
)


#
# Actual KITTI output is e.g.
#
#     3d   AP:63.3784, 46.8941, 41.2036
#
# so whitespace between "3d" and "AP" must be flexible.
#
AP3D_LINE = re.compile(
    rf"^\s*3d\s+AP:\s*"
    rf"({FLOAT_RE})\s*,\s*"
    rf"({FLOAT_RE})\s*,\s*"
    rf"({FLOAT_RE})\s*$",
    re.IGNORECASE,
)


def parse_log(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(
        errors="replace"
    )

    lines = text.splitlines()

    #
    # Reject incomplete/stale logs.
    #
    if "Evaluation done." not in text:
        raise RuntimeError(
            f"incomplete evaluation log: {path}"
        )

    if "eval metric: offline_3d" not in text:
        raise RuntimeError(
            f"offline_3d result not found: {path}"
        )

    values = {}

    detailed = {}

    for cls, header_re in STRICT_HEADERS.items():

        starts = [
            i
            for i, line in enumerate(lines)
            if header_re.search(line)
        ]

        if len(starts) != 1:
            raise RuntimeError(
                f"{cls}: expected exactly one strict "
                f"AP_R40 header, found {len(starts)} "
                f"in {path}"
            )

        start = starts[0]

        found = []

        for j in range(
            start + 1,
            len(lines),
        ):
            line = lines[j]

            #
            # We have entered the next AP block.
            #
            if ANY_AP_HEADER.search(line):
                break

            m = AP3D_LINE.search(line)

            if m is not None:
                found.append(
                    (
                        float(m.group(1)),
                        float(m.group(2)),
                        float(m.group(3)),
                    )
                )

        if len(found) != 1:
            raise RuntimeError(
                f"{cls}: expected exactly one 3d AP "
                f"line inside strict AP_R40 block, "
                f"found {len(found)} in {path}"
            )

        easy, moderate, hard = found[0]

        for name, x in (
            ("easy", easy),
            ("moderate", moderate),
            ("hard", hard),
        ):
            if not math.isfinite(x):
                raise RuntimeError(
                    f"{cls} {name} AP is non-finite: {x}"
                )

        detailed[cls] = {
            "easy": easy,
            "moderate": moderate,
            "hard": hard,
        }

        values[cls] = moderate

    macro = (
        values["Car"]
        + values["Pedestrian"]
        + values["Cyclist"]
    ) / 3.0

    if not math.isfinite(macro):
        raise RuntimeError(
            f"non-finite Macro AP: {macro}"
        )

    return {
        "Car": values["Car"],
        "Pedestrian": values["Pedestrian"],
        "Cyclist": values["Cyclist"],
        "Macro": macro,
        "detail": detailed,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--log",
        required=True,
    )

    ap.add_argument(
        "--epoch",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--format",
        choices=[
            "tsv",
            "json",
            "pretty",
        ],
        default="pretty",
    )

    args = ap.parse_args()

    result = parse_log(
        Path(args.log)
    )

    if args.format == "tsv":
        print(
            f"{args.epoch}\t"
            f"{result['Car']:.6f}\t"
            f"{result['Pedestrian']:.6f}\t"
            f"{result['Cyclist']:.6f}\t"
            f"{result['Macro']:.6f}"
        )

    elif args.format == "json":
        payload = {
            "epoch": args.epoch,
            **result,
        }

        print(
            json.dumps(
                payload,
                indent=2,
            )
        )

    else:
        print(
            f"epoch={args.epoch} "
            f"Car={result['Car']:.4f} "
            f"Ped={result['Pedestrian']:.4f} "
            f"Cyc={result['Cyclist']:.4f} "
            f"Macro={result['Macro']:.4f}"
        )


if __name__ == "__main__":
    main()
