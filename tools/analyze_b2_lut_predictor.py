#!/usr/bin/env python3

import argparse
import json
import math
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        "Validate B2 shape-LUT predictor against measured packing oracle"
    )

    p.add_argument(
        "--lut",
        required=True,
    )

    p.add_argument(
        "--oracle",
        required=True,
    )

    p.add_argument(
        "--metric",
        choices=[
            "mean_ms",
            "p95_ms",
            "p99_ms",
        ],
        default="mean_ms",
    )

    p.add_argument(
        "--guard-ms",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--output",
        required=True,
    )

    return p.parse_args()


def expand_align(
    roi,
    halo,
    H,
    W,
    align,
):
    y0, y1, x0, x1 = roi

    y0 = max(
        0,
        y0 - halo,
    )

    x0 = max(
        0,
        x0 - halo,
    )

    y1 = min(
        H,
        y1 + halo,
    )

    x1 = min(
        W,
        x1 + halo,
    )

    y0 = (
        y0 // align
    ) * align

    x0 = (
        x0 // align
    ) * align

    y1 = min(
        H,
        int(
            math.ceil(
                y1 / align
            ) * align
        ),
    )

    x1 = min(
        W,
        int(
            math.ceil(
                x1 / align
            ) * align
        ),
    )

    return (
        y0,
        y1,
        x0,
        x1,
    )


class B2LUTPredictor:

    def __init__(
        self,
        data,
        metric,
    ):
        self.metric = metric
        self.records = data[
            "records"
        ]

        self.H = int(
            data[
                "input_shape"
            ][-2]
        )

        self.W = int(
            data[
                "input_shape"
            ][-1]
        )

        self.halo = int(
            data["halo"]
        )

        self.align = int(
            data["align"]
        )

        self.full_ms = float(
            data[
                "full_b2"
            ][metric]
        )

    def _distance(
        self,
        r,
        core_h,
        core_w,
        exec_h,
        exec_w,
    ):
        """
        Exec shape dominates the cost.
        Core shape is kept as a weaker secondary term
        because scatter size is also included in LUT timing.
        """

        deh = abs(
            r["exec_h"]
            - exec_h
        ) / self.H

        dew = abs(
            r["exec_w"]
            - exec_w
        ) / self.W

        dch = abs(
            r["core_h"]
            - core_h
        ) / self.H

        dcw = abs(
            r["core_w"]
            - core_w
        ) / self.W

        return (
            4.0 * deh
            + 4.0 * dew
            + dch
            + dcw
        )

    def predict_roi(
        self,
        roi,
    ):
        cy0, cy1, cx0, cx1 = roi

        core_h = (
            cy1 - cy0
        )

        core_w = (
            cx1 - cx0
        )

        e = expand_align(
            roi,
            self.halo,
            self.H,
            self.W,
            self.align,
        )

        ey0, ey1, ex0, ex1 = e

        exec_h = (
            ey1 - ey0
        )

        exec_w = (
            ex1 - ex0
        )

        # --------------------------------------------------
        # 1. Prefer exact physical execution shape.
        # --------------------------------------------------

        exact = [
            r
            for r in self.records
            if (
                int(r["exec_h"])
                == exec_h
                and
                int(r["exec_w"])
                == exec_w
            )
        ]

        if exact:

            best = min(
                exact,
                key=lambda r:
                    self._distance(
                        r,
                        core_h,
                        core_w,
                        exec_h,
                        exec_w,
                    ),
            )

            return (
                float(
                    best[
                        self.metric
                    ]
                ),
                best,
            )

        # --------------------------------------------------
        # 2. Prefer a LUT shape that covers requested
        #    physical H/W.
        # --------------------------------------------------

        covering = [
            r
            for r in self.records
            if (
                int(r["exec_h"])
                >= exec_h
                and
                int(r["exec_w"])
                >= exec_w
            )
        ]

        if covering:

            best = min(
                covering,
                key=lambda r:
                    self._distance(
                        r,
                        core_h,
                        core_w,
                        exec_h,
                        exec_w,
                    ),
            )

            return (
                float(
                    best[
                        self.metric
                    ]
                ),
                best,
            )

        # --------------------------------------------------
        # 3. Fallback to globally nearest LUT point.
        # --------------------------------------------------

        best = min(
            self.records,
            key=lambda r:
                self._distance(
                    r,
                    core_h,
                    core_w,
                    exec_h,
                    exec_w,
                ),
        )

        return (
            float(
                best[
                    self.metric
                ]
            ),
            best,
        )

    def predict_plan(
        self,
        rois,
    ):
        total = 0.0

        details = []

        for roi in rois:

            p, lut_row = (
                self.predict_roi(
                    roi
                )
            )

            total += p

            details.append(
                {
                    "roi":
                        list(roi),

                    "pred_ms":
                        p,

                    "lut_core_h":
                        lut_row["core_h"],

                    "lut_core_w":
                        lut_row["core_w"],

                    "lut_exec_h":
                        lut_row["exec_h"],

                    "lut_exec_w":
                        lut_row["exec_w"],
                }
            )

        return (
            total,
            details,
        )


def percentile(
    values,
    q,
):
    if not values:
        return 0.0

    return float(
        np.percentile(
            np.asarray(
                values,
                dtype=np.float64,
            ),
            q,
        )
    )


def main():
    args = parse_args()

    with open(
        args.lut,
        "r",
    ) as f:
        lut_data = json.load(f)

    with open(
        args.oracle,
        "r",
    ) as f:
        oracle_data = json.load(f)

    predictor = B2LUTPredictor(
        lut_data,
        args.metric,
    )

    full_pred = (
        predictor.full_ms
    )

    full_real = float(
        oracle_data[
            "full_b2"
        ][args.metric]
    )

    print()
    print("=" * 116)

    print(
        f"B2 LUT validation"
        f" | metric={args.metric}"
        f" | predicted FULL={full_pred:.3f} ms"
        f" | measured FULL={full_real:.3f} ms"
        f" | guard={args.guard_ms:.3f} ms"
    )

    print("=" * 116)

    rows = []

    errors = []
    abs_errors = []
    rel_errors = []
    under_errors = []

    # ------------------------------------------------------
    # Predict every previously measured strategy.
    # ------------------------------------------------------

    for r in oracle_data[
        "records"
    ]:

        physical_rois = [
            tuple(x)
            for x in r[
                "physical_rois"
            ]
        ]

        pred_ms, details = (
            predictor.predict_plan(
                physical_rois
            )
        )

        real_ms = float(
            r[
                "latency"
            ][args.metric]
        )

        err = (
            pred_ms
            - real_ms
        )

        errors.append(
            err
        )

        abs_errors.append(
            abs(err)
        )

        rel_errors.append(
            abs(err)
            / max(
                real_ms,
                1e-9,
            )
        )

        # Positive means real was slower than prediction.
        under_errors.append(
            real_ms
            - pred_ms
        )

        rows.append(
            {
                "ratio":
                    float(
                        r["ratio"]
                    ),

                "fragments":
                    int(
                        r["fragments"]
                    ),

                "strategy":
                    str(
                        r["strategy"]
                    ),

                "num_rois":
                    int(
                        r["num_rois"]
                    ),

                "pred_ms":
                    pred_ms,

                "real_ms":
                    real_ms,

                "error_ms":
                    err,

                "details":
                    details,
            }
        )

    # ------------------------------------------------------
    # Group-wise decision validation.
    # ------------------------------------------------------

    groups = {}

    for row in rows:

        key = (
            round(
                row["ratio"],
                8,
            ),
            row[
                "fragments"
            ],
        )

        groups.setdefault(
            key,
            []
        ).append(
            row
        )

    decisions = []

    mode_correct = 0
    strategy_correct = 0

    regrets = []

    print()
    print(
        f"{'Ratio':>7} "
        f"{'Frag':>5} "
        f"{'PredBest':>10} "
        f"{'Pred(ms)':>9} "
        f"{'RealBest':>10} "
        f"{'Real(ms)':>9} "
        f"{'Decision':>10} "
        f"{'Oracle':>10} "
        f"{'Regret':>9}"
    )

    print("-" * 116)

    for (
        ratio,
        frag
    ) in sorted(
        groups.keys()
    ):

        candidates = groups[
            (ratio, frag)
        ]

        pred_best = min(
            candidates,
            key=lambda x:
                x["pred_ms"],
        )

        real_best = min(
            candidates,
            key=lambda x:
                x["real_ms"],
        )

        # --------------------------------------------------
        # Predicted no-regret decision
        # --------------------------------------------------

        if (
            pred_best["pred_ms"]
            + args.guard_ms
            < full_pred
        ):
            pred_mode = (
                "SELECTIVE"
            )

            pred_decision = (
                pred_best[
                    "strategy"
                ]
            )

            chosen_real = (
                pred_best[
                    "real_ms"
                ]
            )

        else:
            pred_mode = "FULL"
            pred_decision = "FULL"
            chosen_real = full_real

        # --------------------------------------------------
        # Measured no-regret oracle
        # --------------------------------------------------

        if (
            real_best["real_ms"]
            + args.guard_ms
            < full_real
        ):
            oracle_mode = (
                "SELECTIVE"
            )

            oracle_decision = (
                real_best[
                    "strategy"
                ]
            )

            oracle_real = (
                real_best[
                    "real_ms"
                ]
            )

        else:
            oracle_mode = "FULL"
            oracle_decision = "FULL"
            oracle_real = full_real

        if (
            pred_mode
            == oracle_mode
        ):
            mode_correct += 1

        if (
            pred_decision
            == oracle_decision
        ):
            strategy_correct += 1

        regret = max(
            0.0,
            chosen_real
            - oracle_real,
        )

        regrets.append(
            regret
        )

        decisions.append(
            {
                "ratio":
                    ratio,

                "fragments":
                    frag,

                "pred_best":
                    pred_best[
                        "strategy"
                    ],

                "pred_best_ms":
                    pred_best[
                        "pred_ms"
                    ],

                "real_best":
                    real_best[
                        "strategy"
                    ],

                "real_best_ms":
                    real_best[
                        "real_ms"
                    ],

                "pred_decision":
                    pred_decision,

                "oracle_decision":
                    oracle_decision,

                "chosen_real_ms":
                    chosen_real,

                "oracle_real_ms":
                    oracle_real,

                "regret_ms":
                    regret,
            }
        )

        print(
            f"{ratio*100:6.1f}% "
            f"{frag:5d} "
            f"{pred_best['strategy']:>10} "
            f"{pred_best['pred_ms']:9.3f} "
            f"{real_best['strategy']:>10} "
            f"{real_best['real_ms']:9.3f} "
            f"{pred_decision:>10} "
            f"{oracle_decision:>10} "
            f"{regret:9.3f}"
        )

    n_groups = len(
        decisions
    )

    mae = float(
        np.mean(
            abs_errors
        )
    )

    mape = (
        float(
            np.mean(
                rel_errors
            )
        )
        * 100.0
    )

    bias = float(
        np.mean(
            errors
        )
    )

    under_p95 = percentile(
        under_errors,
        95,
    )

    under_p99 = percentile(
        under_errors,
        99,
    )

    regret_mean = float(
        np.mean(
            regrets
        )
    )

    regret_p95 = percentile(
        regrets,
        95,
    )

    regret_max = float(
        np.max(
            regrets
        )
    )

    print()
    print("=" * 88)
    print("Prediction summary")
    print("=" * 88)

    print(
        f"MAE                 : "
        f"{mae:.4f} ms"
    )

    print(
        f"MAPE                : "
        f"{mape:.2f}%"
    )

    print(
        f"Bias(pred-real)     : "
        f"{bias:+.4f} ms"
    )

    print(
        f"Underprediction P95 : "
        f"{under_p95:+.4f} ms"
    )

    print(
        f"Underprediction P99 : "
        f"{under_p99:+.4f} ms"
    )

    print(
        f"Mode accuracy       : "
        f"{mode_correct}/{n_groups} "
        f"= "
        f"{100.0*mode_correct/n_groups:.2f}%"
    )

    print(
        f"Exact strategy acc  : "
        f"{strategy_correct}/{n_groups} "
        f"= "
        f"{100.0*strategy_correct/n_groups:.2f}%"
    )

    print(
        f"Mean regret         : "
        f"{regret_mean:.4f} ms"
    )

    print(
        f"P95 regret          : "
        f"{regret_p95:.4f} ms"
    )

    print(
        f"Max regret          : "
        f"{regret_max:.4f} ms"
    )

    print()
    print(
        "Suggested extra predictor margin "
        "(based on this validation set): "
        f"max(0, P95 underprediction) = "
        f"{max(0.0, under_p95):.4f} ms"
    )

    out = Path(
        args.output
    )

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out.write_text(
        json.dumps(
            {
                "metric":
                    args.metric,

                "guard_ms":
                    args.guard_ms,

                "full_pred_ms":
                    full_pred,

                "full_real_ms":
                    full_real,

                "summary": {
                    "mae_ms":
                        mae,

                    "mape_percent":
                        mape,

                    "bias_ms":
                        bias,

                    "underprediction_p95_ms":
                        under_p95,

                    "underprediction_p99_ms":
                        under_p99,

                    "mode_accuracy":
                        (
                            mode_correct
                            / n_groups
                        ),

                    "strategy_accuracy":
                        (
                            strategy_correct
                            / n_groups
                        ),

                    "mean_regret_ms":
                        regret_mean,

                    "p95_regret_ms":
                        regret_p95,

                    "max_regret_ms":
                        regret_max,
                },

                "decisions":
                    decisions,

                "records":
                    rows,
            },
            indent=2,
        )
    )

    print()
    print(
        "Saved:",
        out,
    )


if __name__ == "__main__":
    main()
