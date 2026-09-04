from __future__ import annotations

import copy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment


class MTDDelayAnalyzer:
    """
    MTD-style Delay Analysis Module.

    Before job i:
        C_hat_i = min(C_{i-1}, C_{i-2})
        d_i = floor(C_hat_i / T)

    d_i is clipped to available timestep branches.
    """

    def __init__(
        self,
        period_ms,
        initial_runtime_ms,
        num_branches=3,
    ):
        if period_ms <= 0:
            raise ValueError(
                "period_ms must be > 0"
            )

        if initial_runtime_ms <= 0:
            raise ValueError(
                "initial_runtime_ms must be > 0"
            )

        if num_branches < 1:
            raise ValueError(
                "num_branches must be >= 1"
            )

        self.period_ms = float(period_ms)

        self.initial_runtime_ms = float(
            initial_runtime_ms
        )

        self.num_branches = int(
            num_branches
        )

        self.reset()

    def reset(self):
        self.last_runtime_ms = (
            self.initial_runtime_ms
        )

        self.prev_runtime_ms = (
            self.initial_runtime_ms
        )

        self.observed_count = 0

    @property
    def estimated_runtime_ms(self):
        return float(
            min(
                self.last_runtime_ms,
                self.prev_runtime_ms,
            )
        )

    def select_branch(self):
        estimate = (
            self.estimated_runtime_ms
        )

        raw_delay = int(
            math.floor(
                estimate
                /
                self.period_ms
                +
                1e-12
            )
        )

        branch = min(
            max(
                raw_delay,
                0,
            ),
            self.num_branches - 1,
        )

        return (
            branch,
            raw_delay,
            estimate,
        )

    def update(
        self,
        observed_runtime_ms,
    ):
        observed_runtime_ms = float(
            observed_runtime_ms
        )

        if observed_runtime_ms <= 0:
            raise ValueError(
                "observed_runtime_ms must be > 0"
            )

        self.prev_runtime_ms = (
            self.last_runtime_ms
        )

        self.last_runtime_ms = (
            observed_runtime_ms
        )

        self.observed_count += 1


class MTDTimeStepBranchBank:
    """
    StreamDSGN adaptation of MTD timestep branches.

    Original StreamDSGN:
        branch 0 -> h=1
        native next-frame prediction

    Extra branches:
        branch 1 -> h=2
        branch 2 -> h=3

    h=2/h=3 use only causal motion estimated from completed
    base predictions. No future observation or GT is used.
    """

    def __init__(
        self,
        match_threshold_m=10.0,
    ):
        if match_threshold_m <= 0:
            raise ValueError(
                "match_threshold_m must be > 0"
            )

        self.match_threshold_m = float(
            match_threshold_m
        )

        self.reset()

    def reset(self):
        self.prev_base_anno = None
        self.prev_local_pos = None

        self.total_current_boxes = 0
        self.total_matched_boxes = 0

        self.routed_counts = {
            0: 0,
            1: 0,
            2: 0,
        }

    @staticmethod
    def _locations(
        anno,
    ):
        if anno is None:
            return np.zeros(
                (0, 3),
                dtype=np.float64,
            )

        return np.asarray(
            anno.get(
                "location",
                np.zeros((0, 3)),
            ),
            dtype=np.float64,
        ).reshape(
            -1,
            3,
        )

    @staticmethod
    def _names(
        anno,
    ):
        if anno is None:
            return np.empty(
                (0,),
                dtype=object,
            )

        return np.asarray(
            anno.get(
                "name",
                np.empty(
                    (0,),
                    dtype=object,
                ),
            ),
            dtype=object,
        ).reshape(-1)

    def _estimate_step_motion(
        self,
        current_anno,
        current_local_pos,
    ):
        cur_loc = self._locations(
            current_anno
        )

        cur_names = self._names(
            current_anno
        )

        motion = np.zeros(
            (
                len(cur_loc),
                2,
            ),
            dtype=np.float64,
        )

        self.total_current_boxes += (
            len(cur_loc)
        )

        if (
            self.prev_base_anno is None
            or
            self.prev_local_pos is None
            or
            len(cur_loc) == 0
        ):
            return motion

        prev_loc = self._locations(
            self.prev_base_anno
        )

        prev_names = self._names(
            self.prev_base_anno
        )

        step_gap = (
            int(current_local_pos)
            -
            int(self.prev_local_pos)
        )

        if (
            step_gap <= 0
            or
            len(prev_loc) == 0
        ):
            return motion

        cost = np.linalg.norm(
            cur_loc[:, None, :]
            -
            prev_loc[None, :, :],
            axis=2,
        )

        same_class = (
            cur_names[:, None]
            ==
            prev_names[None, :]
        )

        cost[
            ~same_class
        ] = 1e9

        cur_idx, prev_idx = (
            linear_sum_assignment(
                cost
            )
        )

        for i, j in zip(
            cur_idx,
            prev_idx,
        ):
            if (
                cost[i, j]
                >=
                self.match_threshold_m
            ):
                continue

            # KITTI camera ground plane = x-z.
            motion[i, 0] = (
                cur_loc[i, 0]
                -
                prev_loc[j, 0]
            ) / step_gap

            motion[i, 1] = (
                cur_loc[i, 2]
                -
                prev_loc[j, 2]
            ) / step_gap

            self.total_matched_boxes += 1

        return motion

    @staticmethod
    def _refresh_alpha(
        anno,
    ):
        if (
            "rotation_y" not in anno
            or
            "location" not in anno
        ):
            return

        loc = np.asarray(
            anno["location"],
            dtype=np.float64,
        ).reshape(
            -1,
            3,
        )

        ry = np.asarray(
            anno["rotation_y"],
            dtype=np.float64,
        ).reshape(-1)

        if len(loc) != len(ry):
            return

        alpha = (
            ry
            -
            np.arctan2(
                loc[:, 0],
                loc[:, 2],
            )
        )

        alpha = (
            alpha + np.pi
        ) % (
            2.0 * np.pi
        ) - np.pi

        anno["alpha"] = (
            alpha.astype(
                np.float32
            )
        )

    def route(
        self,
        base_anno,
        current_local_pos,
        branch_index,
    ):
        branch_index = int(
            branch_index
        )

        if branch_index not in (
            0,
            1,
            2,
        ):
            raise ValueError(
                f"invalid branch_index="
                f"{branch_index}"
            )

        out = copy.deepcopy(
            base_anno
        )

        motion = (
            self._estimate_step_motion(
                base_anno,
                current_local_pos,
            )
        )

        if (
            out is not None
            and
            branch_index > 0
            and
            len(motion) > 0
        ):
            loc = np.asarray(
                out["location"],
                dtype=np.float64,
            ).reshape(
                -1,
                3,
            ).copy()

            # branch 0 -> native h=1
            # branch 1 -> h=2
            # branch 2 -> h=3
            loc[:, 0] += (
                motion[:, 0]
                *
                branch_index
            )

            loc[:, 2] += (
                motion[:, 1]
                *
                branch_index
            )

            out["location"] = (
                loc.astype(
                    np.float32
                )
            )

            self._refresh_alpha(
                out
            )

        self.routed_counts[
            branch_index
        ] += 1

        # Always keep the unmodified base prediction as history.
        self.prev_base_anno = (
            copy.deepcopy(
                base_anno
            )
        )

        self.prev_local_pos = int(
            current_local_pos
        )

        return out

    @property
    def match_ratio(
        self,
    ):
        if (
            self.total_current_boxes
            ==
            0
        ):
            return 0.0

        return (
            self.total_matched_boxes
            /
            self.total_current_boxes
        )


# MTD_STYLE_RUNTIME_EOF
