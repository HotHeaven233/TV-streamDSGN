#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

WIDTHS = (1.0, 0.75, 0.5, 0.25)


def is_monotonic(schedule):
    return all(schedule[i + 1] <= schedule[i] for i in range(len(schedule) - 1))


def tag(schedule):
    m = {1.0: "100", 0.75: "075", 0.5: "050", 0.25: "025"}
    return "_".join(m[float(x)] for x in schedule)


def schedule_text(schedule):
    return ",".join(
        "1.0" if x == 1.0 else
        "0.75" if x == 0.75 else
        "0.5" if x == 0.5 else
        "0.25"
        for x in schedule
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    args = p.parse_args()

    schedules = [
        s for s in itertools.product(WIDTHS, repeat=6)
        if is_monotonic(s)
    ]
    schedules.sort(reverse=True)

    if len(schedules) != 84:
        raise RuntimeError(f"Expected 84 schedules, got {len(schedules)}")
    if len(set(schedules)) != 84:
        raise RuntimeError("Duplicate schedules found")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "id", "schedule", "tag",
            "res2", "res3", "res4", "fpn", "stereo", "rpn"
        ])
        for idx, s in enumerate(schedules, 1):
            w.writerow([
                idx, schedule_text(s), tag(s),
                *[f"{x:g}" for x in s]
            ])

    print(f"[OK] wrote {len(schedules)} legal monotonic profiles -> {output}")
    print(f"[FIRST] {schedule_text(schedules[0])}")
    print(f"[LAST ] {schedule_text(schedules[-1])}")


if __name__ == "__main__":
    main()

