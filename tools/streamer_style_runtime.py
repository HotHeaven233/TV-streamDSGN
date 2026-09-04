from __future__ import annotations

import copy
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

from pcdet.datasets.kitti_streaming.kalman_filter import KalmanFilter


def tail(x: float) -> float:
    return x - math.floor(x)


def shrinking_tail_should_wait(
    finish_ms: float,
    runtime_estimate_ms: float,
    period_ms: float,
) -> bool:
    """ECCV'20 Streamer shrinking-tail rule in physical milliseconds."""
    if period_ms <= 0.0:
        raise ValueError("period_ms must be > 0")
    if runtime_estimate_ms <= 0.0:
        raise ValueError("runtime_estimate_ms must be > 0")

    r = runtime_estimate_ms / period_ms
    if r <= 1.0 + 1e-12:
        return False

    s = finish_ms / period_ms
    return tail(s + r) + 1e-12 < tail(s)


class RuntimeEstimator:
    def __init__(self, initial_ms: float, mode: str, alpha: float):
        if initial_ms <= 0.0:
            raise ValueError("initial_ms must be > 0")
        if mode not in {"static_median", "ewma"}:
            raise ValueError(f"unsupported runtime estimator: {mode}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must satisfy 0 < alpha <= 1")

        self.value_ms = float(initial_ms)
        self.mode = mode
        self.alpha = float(alpha)

    def update(self, observed_ms: float) -> None:
        observed_ms = float(observed_ms)
        if observed_ms <= 0.0:
            raise ValueError("observed_ms must be > 0")

        if self.mode == "ewma":
            self.value_ms = (
                self.alpha * observed_ms
                + (1.0 - self.alpha) * self.value_ms
            )


class GroundPlaneKalman(KalmanFilter):
    """
    Reuse StreamDSGN's existing 5D KF, but use KITTI camera x-z as the
    ground plane. State semantics: [x, y, z, vx, vz].
    """

    def make_F(self, delta_time):
        self.F = np.eye(5)
        self.F[0, 3] = delta_time
        self.F[2, 4] = delta_time


class Async3DKalman:
    """
    Streamer-style asynchronous 3D KF.

    Each completed detector package is corrected at its SOURCE sensor time.
    At a later sAP query time, the latest available state is extrapolated to
    that query. No future sensor observation is used.
    """

    def __init__(
        self,
        match_threshold_m: float = 4.0,
        r_fac: float = 40.0,
    ):
        if match_threshold_m <= 0.0:
            raise ValueError("match_threshold_m must be > 0")
        if r_fac <= 0.0:
            raise ValueError("r_fac must be > 0")

        self.match_threshold_m = float(match_threshold_m)
        self.r_fac = float(r_fac)
        self.reset()

    def reset(self):
        self.kf = GroundPlaneKalman(R_fac=self.r_fac)
        self.last_raw_loc = np.zeros((0, 3), dtype=np.float64)
        self.last_names = np.empty((0,), dtype=object)
        self.source_time_ms = None
        self.state = np.zeros((0, 5), dtype=np.float64)
        self.latest_anno = None
        self.class_ids = {}
        self.total_measurements = 0
        self.total_matches = 0
        self.forecast_queries = 0

    def _class_ids(self, names):
        ids = []
        for name in names:
            key = str(name)
            if key not in self.class_ids:
                self.class_ids[key] = len(self.class_ids) + 1
            ids.append(self.class_ids[key])
        return np.asarray(ids, dtype=np.float32)

    def _velocity_measurements(self, loc, names, dt_s):
        vel = np.zeros((len(loc), 2), dtype=np.float64)
        if len(loc) == 0 or len(self.last_raw_loc) == 0 or dt_s <= 1e-9:
            return vel

        dist = np.linalg.norm(
            loc[:, None, :] - self.last_raw_loc[None, :, :],
            axis=2,
        )
        same_cls = names[:, None] == self.last_names[None, :]
        cost = dist.copy()
        cost[~same_cls] = 1e9

        cur_idx, prev_idx = linear_sum_assignment(cost)
        for i, j in zip(cur_idx, prev_idx):
            if cost[i, j] >= self.match_threshold_m:
                continue
            vel[i, 0] = (loc[i, 0] - self.last_raw_loc[j, 0]) / dt_s
            vel[i, 1] = (loc[i, 2] - self.last_raw_loc[j, 2]) / dt_s
            self.total_matches += 1

        return vel

    def update(self, anno, source_time_ms: float):
        if anno is None:
            return

        source_time_ms = float(source_time_ms)
        if (
            self.source_time_ms is not None
            and source_time_ms + 1e-9 < self.source_time_ms
        ):
            raise RuntimeError(
                f"non-monotonic source time: {source_time_ms} < {self.source_time_ms}"
            )

        new_anno = copy.deepcopy(anno)
        loc = np.asarray(
            new_anno.get("location", np.zeros((0, 3))),
            dtype=np.float64,
        ).reshape(-1, 3)
        names = np.asarray(
            new_anno.get("name", np.empty((0,), dtype=object)),
            dtype=object,
        ).reshape(-1)

        if len(loc) != len(names):
            raise RuntimeError(
                f"annotation mismatch: location={len(loc)} names={len(names)}"
            )

        dt_s = (
            0.0
            if self.source_time_ms is None
            else max(0.0, (source_time_ms - self.source_time_ms) / 1000.0)
        )
        vel = self._velocity_measurements(loc, names, dt_s)

        z = np.zeros((len(loc), 5), dtype=np.float64)
        z[:, :3] = loc
        z[:, 3:] = vel

        self.state = np.asarray(
            self.kf(z, self._class_ids(names), dt_s),
            dtype=np.float64,
        ).copy()

        self.total_measurements += len(loc)
        self.last_raw_loc = loc.copy()
        self.last_names = names.copy()
        self.source_time_ms = source_time_ms
        self.latest_anno = new_anno

    def forecast(self, query_time_ms: float):
        if self.latest_anno is None:
            return None

        out = copy.deepcopy(self.latest_anno)
        self.forecast_queries += 1
        if len(self.state) == 0:
            return out

        dt_s = max(
            0.0,
            (float(query_time_ms) - self.source_time_ms) / 1000.0,
        )
        state = self.state.copy()
        state[:, 0] += state[:, 3] * dt_s
        state[:, 2] += state[:, 4] * dt_s
        out["location"] = state[:, :3].astype(np.float32)

        if "rotation_y" in out:
            ry = np.asarray(out["rotation_y"], dtype=np.float64).reshape(-1)
            if len(ry) == len(state):
                alpha = ry - np.arctan2(state[:, 0], state[:, 2])
                alpha = (alpha + np.pi) % (2.0 * np.pi) - np.pi
                out["alpha"] = alpha.astype(np.float32)

        return out
