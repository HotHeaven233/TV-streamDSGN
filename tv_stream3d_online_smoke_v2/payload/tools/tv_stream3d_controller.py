#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CHECKPOINTS = (
    "after_prefix",
    "after_res2",
    "after_res3",
    "after_res4",
    "after_fpn",
    "after_stereo",
    "after_rpn",
)

NEXT_STAGE_INDEX = {
    "after_prefix": 0,   # choose Res2
    "after_res2": 1,     # choose Res3
    "after_res3": 2,     # choose Res4
    "after_res4": 3,     # choose FPN
    "after_fpn": 4,      # choose Stereo
    "after_stereo": 5,   # choose RPN
    "after_rpn": None,   # fixed tail only
}

# Number of future CPU/GPU control boundaries that remain after each
# checkpoint, including the decision made at the current checkpoint.
# There is no decision after RPN; the fixed tail is launched immediately.
REMAINING_CONTROL_BOUNDARIES = {
    "after_prefix": 6,
    "after_res2": 5,
    "after_res3": 4,
    "after_res4": 3,
    "after_fpn": 2,
    "after_stereo": 1,
    "after_rpn": 0,
}


def parse_schedule(text: str) -> Tuple[float, ...]:
    xs = tuple(float(x.strip()) for x in str(text).split(","))
    if len(xs) != 6:
        raise ValueError(f"schedule must contain 6 widths: {text}")
    return xs


def is_monotonic_schedule(s: Sequence[float]) -> bool:
    return all(float(s[i + 1]) <= float(s[i]) for i in range(len(s) - 1))


@dataclass(frozen=True)
class Decision:
    checkpoint: str
    observed_level: str
    elapsed_ms: float
    deadline_ms: float
    remaining_budget_ms: float
    feasible: bool
    profile_id: int
    schedule: Tuple[float, ...]
    quality: float
    remaining_bound_ms: float
    control_guard_ms: float
    required_remaining_ms: float
    next_width: Optional[float]


class TVStream3DController:
    """
    Table-driven forward-only controller.

    Inputs
    ------
    controller_csv:
        5 contention levels x 84 legal schedules x 7 checkpoints.

    contention_levels_json:
        Calibration file containing L0..L4 probe statistics.

    Bound semantics
    ---------------
    The controller uses empirical direct suffix p99 by default:
        B(L,j,s) = Q_0.99(C_remaining^(j,s))

    At each checkpoint:
        R_j = deadline_ms - elapsed_ms

    Feasible schedules satisfy:
        schedule prefix == already executed widths
        B(L,j,s) <= R_j

    Among feasible schedules, maximize empirical complete-schedule quality.
    Only the NEXT stage width is returned. The caller replans at the next
    checkpoint using the newly observed elapsed forward time.
    """

    def __init__(
        self,
        controller_csv: str | Path,
        contention_levels_json: str | Path,
        bound_column: str = "remaining_p99_ms",
        classifier: str = "conservative_gap",
        control_guard_per_boundary_ms: float = 0.0,
    ):
        self.controller_csv = Path(controller_csv)
        self.levels_json = Path(contention_levels_json)
        self.bound_column = str(bound_column)
        self.classifier = str(classifier)
        self.control_guard_per_boundary_ms = float(
            control_guard_per_boundary_ms
        )
        if self.control_guard_per_boundary_ms < 0:
            raise ValueError(
                "control_guard_per_boundary_ms must be >= 0"
            )

        with self.levels_json.open() as f:
            self.levels_data = json.load(f)

        self.level_rows = {
            x["level"]: x
            for x in self.levels_data["levels"]
        }
        self.level_names = [
            x["level"]
            for x in self.levels_data["levels"]
        ]

        self.rows: Dict[Tuple[str, str, int], dict] = {}
        self.profile_schedule: Dict[int, Tuple[float, ...]] = {}
        self.profile_quality: Dict[int, float] = {}

        with self.controller_csv.open(newline="") as f:
            for r in csv.DictReader(f):
                level = r["level"]
                checkpoint = r["checkpoint"]
                pid = int(r["profile_id"])
                schedule = parse_schedule(r["schedule"])
                quality = float(r["reference_quality"])
                bound = float(r[self.bound_column])

                rr = dict(r)
                rr["_schedule"] = schedule
                rr["_quality"] = quality
                rr["_bound"] = bound

                self.rows[(level, checkpoint, pid)] = rr
                self.profile_schedule[pid] = schedule
                self.profile_quality[pid] = quality

        self.profile_ids = sorted(self.profile_schedule)

        if len(self.profile_ids) != 84:
            raise RuntimeError(
                f"expected 84 schedules, got {len(self.profile_ids)}"
            )

        for pid, s in self.profile_schedule.items():
            if not is_monotonic_schedule(s):
                raise RuntimeError(
                    f"profile {pid} is not monotonic: {s}"
                )

        self.centroid_thresholds = list(
            self.levels_data.get(
                "nearest_centroid_thresholds_ms",
                [],
            )
        )
        self.conservative_thresholds = self._build_conservative_thresholds()

    def _build_conservative_thresholds(self) -> List[float]:
        """
        Prefer the midpoint between lower-level p90 and upper-level p10.

        This biases the ambiguous L2/L3 boundary toward the higher contention
        level compared with nearest-centroid classification, which is desirable
        for deadline safety. If required quantiles are unavailable or overlap,
        fall back to the existing centroid midpoint threshold.
        """
        thresholds = []

        for i in range(len(self.level_names) - 1):
            lo = self.level_rows[self.level_names[i]]
            hi = self.level_rows[self.level_names[i + 1]]

            lo_p90 = lo.get("probe_p90_ms")
            hi_p10 = hi.get("probe_p10_ms")

            if (
                lo_p90 is not None
                and hi_p10 is not None
                and float(lo_p90) < float(hi_p10)
            ):
                thresholds.append(
                    0.5 * (float(lo_p90) + float(hi_p10))
                )
            elif i < len(self.centroid_thresholds):
                thresholds.append(
                    float(self.centroid_thresholds[i])
                )
            else:
                c0 = float(lo["probe_p50_ms"])
                c1 = float(hi["probe_p50_ms"])
                thresholds.append(0.5 * (c0 + c1))

        return thresholds

    def classify_probe(self, probe_ms: float) -> str:
        x = float(probe_ms)

        if self.classifier == "nearest_centroid":
            thresholds = self.centroid_thresholds
        elif self.classifier == "conservative_gap":
            thresholds = self.conservative_thresholds
        else:
            raise ValueError(
                "classifier must be nearest_centroid or conservative_gap"
            )

        idx = 0
        while idx < len(thresholds) and x >= float(thresholds[idx]):
            idx += 1
        return self.level_names[idx]

    def compatible_profile_ids(
        self,
        executed_prefix: Sequence[float],
    ) -> List[int]:
        prefix = tuple(float(x) for x in executed_prefix)
        k = len(prefix)

        out = []
        for pid in self.profile_ids:
            s = self.profile_schedule[pid]
            if s[:k] == prefix:
                out.append(pid)
        return out

    def decide(
        self,
        checkpoint: str,
        elapsed_ms: float,
        deadline_ms: float,
        observed_level: str,
        executed_prefix: Sequence[float],
    ) -> Decision:
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint: {checkpoint}")
        if observed_level not in self.level_names:
            raise ValueError(f"unknown contention level: {observed_level}")

        elapsed_ms = float(elapsed_ms)
        deadline_ms = float(deadline_ms)
        remaining_budget = deadline_ms - elapsed_ms

        compatible = self.compatible_profile_ids(executed_prefix)
        if not compatible:
            raise RuntimeError(
                f"no schedule compatible with prefix {tuple(executed_prefix)}"
            )

        guard_ms = (
            self.control_guard_per_boundary_ms
            * REMAINING_CONTROL_BOUNDARIES[checkpoint]
        )

        candidates = []
        for pid in compatible:
            row = self.rows[(observed_level, checkpoint, pid)]
            bound_ms = float(row["_bound"])
            required_ms = bound_ms + guard_ms
            candidates.append(
                (
                    pid,
                    self.profile_quality[pid],
                    bound_ms,
                    required_ms,
                    self.profile_schedule[pid],
                )
            )

        feasible = [
            x for x in candidates
            if x[3] <= remaining_budget
        ]

        if feasible:
            # Main objective: maximize empirical complete-schedule quality.
            # Tie-break by smaller remaining bound, then smaller profile id.
            chosen = sorted(
                feasible,
                key=lambda x: (-x[1], x[3], x[0]),
            )[0]
            is_feasible = True
        else:
            # Best-effort fallback: minimize the empirical suffix bound.
            # This does NOT claim deadline feasibility.
            chosen = sorted(
                candidates,
                key=lambda x: (x[3], -x[1], x[0]),
            )[0]
            is_feasible = False

        pid, quality, bound, required, schedule = chosen
        next_idx = NEXT_STAGE_INDEX[checkpoint]
        next_width = (
            None
            if next_idx is None
            else float(schedule[next_idx])
        )

        return Decision(
            checkpoint=checkpoint,
            observed_level=observed_level,
            elapsed_ms=elapsed_ms,
            deadline_ms=deadline_ms,
            remaining_budget_ms=remaining_budget,
            feasible=is_feasible,
            profile_id=int(pid),
            schedule=tuple(schedule),
            quality=float(quality),
            remaining_bound_ms=float(bound),
            control_guard_ms=float(guard_ms),
            required_remaining_ms=float(required),
            next_width=next_width,
        )

    def decide_from_probe(
        self,
        probe_ms: float,
        checkpoint: str,
        elapsed_ms: float,
        deadline_ms: float,
        executed_prefix: Sequence[float],
    ) -> Decision:
        level = self.classify_probe(probe_ms)
        return self.decide(
            checkpoint=checkpoint,
            elapsed_ms=elapsed_ms,
            deadline_ms=deadline_ms,
            observed_level=level,
            executed_prefix=executed_prefix,
        )
