#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, csv, json, math, pickle
from collections import Counter, OrderedDict
from pathlib import Path
import numpy as np
import torch

from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils
from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
from test_stream_buffer_timestamp import load_one
from test_tv_stream3d_online_forward import make_cfg, choose_scene_indices, configure_contender
from smooth_cuda_contention import make_high_priority_detector_stream
from eval_original_streamdsgn_30hz_random50 import reset_history, original_forward_no_post
from eval_tv_stream3d_30hz_random50 import (
    build_balanced_trace, frame_meta, scene_groups, copy_det,
    disable_recall, make_prediction, stats, builtin,
)
from streamer_style_runtime import Async3DKalman, RuntimeEstimator, shrinking_tail_should_wait


def parse_args():
    p = argparse.ArgumentParser(description="Streamer-style StreamDSGN")
    p.add_argument("--cfg", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--levels_json", required=True)
    p.add_argument("--pressure_level", required=True, choices=["L0","L1","L2","L3","L4"])
    p.add_argument("--pressure_fraction", type=float, default=0.5)
    p.add_argument("--trace_seed", type=int, default=20260903)
    p.add_argument("--input_hz", type=float, default=35.0)
    p.add_argument("--runtime_warmup_frames", type=int, default=80)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1024)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--runtime_estimator", choices=["static_median","ewma"], default="ewma")
    p.add_argument("--ewma_alpha", type=float, default=0.5)
    p.add_argument("--forecast_mode", choices=["copy","kf"], default="kf")
    return p.parse_args()


def validate_args(a):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if a.input_hz <= 0 or a.runtime_warmup_frames < 1 or a.max_frames < 0:
        raise ValueError("invalid input_hz/runtime_warmup_frames/max_frames")
    if not (0.0 < a.ewma_alpha <= 1.0):
        raise ValueError("ewma_alpha must satisfy 0 < alpha <= 1")
    if a.pressure_level == "L0":
        if abs(a.pressure_fraction) > 1e-12:
            raise ValueError("L0 requires pressure_fraction=0")
    elif not (0.0 < a.pressure_fraction < 1.0):
        raise ValueError("random contention requires 0 < pressure_fraction < 1")


def make_drop_row(dataset, indices, pos, T, trace, reason):
    idx = indices[pos]
    scene, fid, _ = frame_meta(dataset, idx)
    arrival = pos * T
    return idx, {
        "global_index":idx, "scene":scene, "local_pos":pos, "frame_id":fid,
        "arrival_ms":arrival, "absolute_deadline_ms":arrival+T,
        "status":"dropped", "drop_reason":reason,
        "forward_start_ms":"", "forward_finish_ms":"", "prediction_timestamp_ms":"",
        "queue_wait_ms":"", "forward_ms":"", "arrival_to_finish_ms":"",
        "deadline_slack_ms":"", "deadline_miss":"",
        "true_level":trace[idx], "runtime_estimate_ms":"",
    }


def make_trace(groups, pressure, fraction, seed):
    if pressure == "L0":
        trace = {}
        summary = {}
        for scene, ids in groups.items():
            for idx in ids:
                trace[idx] = "L0"
            summary[scene] = {"sensor_frames":len(ids), "L0":len(ids), "pressure":0}
        return trace, summary
    return build_balanced_trace(groups=groups, pressure_level=pressure, fraction=fraction, seed=seed)


def main():
    a = parse_args()
    validate_args(a)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    T = 1000.0 / a.input_hz
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(a.cfg)
    logger = common_utils.create_logger()
    dataset, _, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES,
        batch_size=1, dist=False, workers=a.workers, logger=logger, training=False,
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=a.ckpt, logger=logger, to_cpu=True)
    model.cuda().eval()
    for name in ("feature_extractor","fusion_module","after_fusion_blocks"):
        if not hasattr(model, name):
            raise RuntimeError(f"model lacks {name}; not original STREAM detector")

    levels = json.loads(Path(a.levels_json).read_text())
    contenders = {lv:configure_contender(levels, lv) for lv in {"L0",a.pressure_level}}
    model_stream = make_high_priority_detector_stream(torch.cuda.current_device())
    original_recall = disable_recall(model)

    # Warmup is outside the logical timeline and always uses L0.
    _, warm_idx = choose_scene_indices(dataset, max(1, a.runtime_warmup_frames + 1))
    reset_history(model)
    warm_times, last_batch = [], None
    for wi in range(a.runtime_warmup_frames):
        batch = load_one(dataset, warm_idx[wi % len(warm_idx)])
        _, ms = original_forward_no_post(model, batch, model_stream, contenders["L0"])
        warm_times.append(float(ms))
        last_batch = batch
    if last_batch is not None:
        make_prediction(model, dataset, last_batch)  # initialize post/NMS outside logical time
    reset_history(model)
    torch.cuda.synchronize()
    runtime_init_ms = float(np.median(warm_times))

    total = len(dataset)
    n_eval = total if a.max_frames == 0 else min(total, a.max_frames)
    eval_indices = list(range(n_eval))
    groups = scene_groups(dataset, eval_indices)
    trace, trace_scene_summary = make_trace(groups, a.pressure_level, a.pressure_fraction, a.trace_seed)
    sensor_levels = Counter(trace[i] for i in eval_indices)

    trace_csv = out / "contention_trace.csv"
    with trace_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["global_index","scene","local_pos","frame_id","true_level"])
        w.writeheader()
        for scene, ids in groups.items():
            for pos, idx in enumerate(ids):
                _, fid, _ = frame_meta(dataset, idx)
                w.writerow({"global_index":idx,"scene":scene,"local_pos":pos,"frame_id":fid,"true_level":trace[idx]})

    print("=" * 100)
    print("STREAMER-STYLE StreamDSGN")
    print(f"cfg={a.cfg}")
    print(f"ckpt={a.ckpt}")
    print(f"Hz={a.input_hz} T={T:.6f} ms | timing=forward-only")
    print(f"pressure={a.pressure_level} fraction={a.pressure_fraction} seed={a.trace_seed}")
    print(f"scheduler=shrinking-tail estimator={a.runtime_estimator} L0_init={runtime_init_ms:.4f} ms alpha={a.ewma_alpha}")
    print(f"forecast={a.forecast_mode}")
    print("=" * 100)

    rows, events = {}, []
    forward_times, queue_times, response_times, rhat_times = [], [], [], []
    processed_levels = Counter()
    processed = overwritten = sched_skipped = misses = wait_count = 0
    rhat = RuntimeEstimator(runtime_init_ms, a.runtime_estimator, a.ewma_alpha)

    try:
        for scene_i, (scene, ids) in enumerate(groups.items(), 1):
            reset_history(model)
            pos, gpu_free, n = 0, 0.0, len(ids)
            print(f"[scene {scene_i:02d}/{len(groups):02d}] {scene}: {n}")

            while pos < n:
                decision_t = max(gpu_free, pos*T)
                est_ms = rhat.value_ms
                should_wait = (
                    pos + 1 < n
                    and shrinking_tail_should_wait(decision_t, est_ms, T)
                )
                if should_wait:
                    next_pos = int(math.floor((decision_t + 1e-9) / T)) + 1
                    next_pos = min(n-1, max(pos+1, next_pos))
                    if next_pos > pos:
                        for sp in range(pos, next_pos):
                            didx, drow = make_drop_row(
                                dataset, ids, sp, T, trace, "streamer_shrinking_tail_wait"
                            )
                            if didx in rows:
                                raise RuntimeError(f"duplicate status {didx}")
                            rows[didx] = drow
                            sched_skipped += 1
                        wait_count += 1
                        pos = next_pos
                        gpu_free = max(gpu_free, pos*T)
                        continue

                idx = ids[pos]
                meta_scene, fid, _ = frame_meta(dataset, idx)
                if meta_scene != scene:
                    raise RuntimeError("scene mismatch")
                arrival = pos*T
                deadline = arrival+T
                start = max(gpu_free, arrival)
                true_lv = trace[idx]
                batch = load_one(dataset, idx)
                _, fwd = original_forward_no_post(
                    model, batch, model_stream, contenders[true_lv]
                )
                finish = start+fwd
                queue = start-arrival
                response = finish-arrival
                slack = deadline-finish
                miss = finish > deadline + 1e-9
                pred = make_prediction(model, dataset, batch)  # outside logical time

                if idx in rows:
                    raise RuntimeError(f"duplicate processed status {idx}")
                rows[idx] = {
                    "global_index":idx, "scene":scene, "local_pos":pos, "frame_id":fid,
                    "arrival_ms":arrival, "absolute_deadline_ms":deadline,
                    "status":"processed", "drop_reason":"",
                    "forward_start_ms":start, "forward_finish_ms":finish,
                    "prediction_timestamp_ms":finish, "queue_wait_ms":queue,
                    "forward_ms":fwd, "arrival_to_finish_ms":response,
                    "deadline_slack_ms":slack, "deadline_miss":int(miss),
                    "true_level":true_lv, "runtime_estimate_ms":est_ms,
                }
                events.append({
                    "scene":scene, "local_pos":pos, "source_index":idx,
                    "source_frame_id":fid, "source_arrival_ms":arrival,
                    "finish_ms":finish, "anno":pred,
                })

                processed += 1
                misses += int(miss)
                forward_times.append(float(fwd))
                queue_times.append(float(queue))
                response_times.append(float(response))
                rhat_times.append(float(est_ms))
                processed_levels[true_lv] += 1
                rhat.update(fwd)

                latest_arrived = min(n-1, int(math.floor((finish+1e-9)/T)))
                if latest_arrived >= pos+1:
                    next_pos = latest_arrived
                    for dp in range(pos+1, next_pos):
                        didx, drow = make_drop_row(
                            dataset, ids, dp, T, trace, "stale_replaced_by_latest"
                        )
                        if didx in rows:
                            raise RuntimeError(f"duplicate overwritten status {didx}")
                        rows[didx] = drow
                        overwritten += 1
                    pos = next_pos
                else:
                    pos += 1
                gpu_free = finish

                if processed <= 3 or processed % 100 == 0:
                    print(
                        f"[{true_lv}] processed={processed:04d} {scene}/{fid} "
                        f"start={start:.3f} finish={finish:.3f} fwd={fwd:.3f} "
                        f"rhat={est_ms:.3f} miss={int(miss)}"
                    )

        missing = [i for i in eval_indices if i not in rows]
        if missing:
            raise RuntimeError(f"missing statuses: {missing[:10]}")
        dropped = overwritten + sched_skipped
        if processed + dropped != n_eval:
            raise RuntimeError(f"frame accounting mismatch: {processed}+{dropped}!={n_eval}")

        # Causal sAP: only packages completed by q may be used.
        by_scene = OrderedDict((s, []) for s in groups)
        for evnt in events:
            by_scene[evnt["scene"]].append(evnt)
        for s in by_scene:
            by_scene[s].sort(key=lambda x:x["finish_ms"])

        aligned = {}
        kf_measurements = kf_matches = kf_queries = 0
        for scene, ids in groups.items():
            scene_events, ptr, latest_copy = by_scene[scene], 0, None
            kf = Async3DKalman()
            for pos, idx in enumerate(ids):
                q = pos*T
                while ptr < len(scene_events) and scene_events[ptr]["finish_ms"] <= q + 1e-9:
                    evnt = scene_events[ptr]
                    latest_copy = evnt["anno"]
                    if a.forecast_mode == "kf":
                        kf.update(evnt["anno"], evnt["source_arrival_ms"])
                    ptr += 1
                latest = kf.forecast(q) if a.forecast_mode == "kf" else latest_copy
                _, fid, next_fid = frame_meta(dataset, idx)
                aligned[idx] = copy_det(latest, scene, fid, next_fid)
            kf_measurements += kf.total_measurements
            kf_matches += kf.total_matches
            kf_queries += kf.forecast_queries

        gt_annos = [
            copy.deepcopy(dataset.kitti_infos[i]["infos"]["token"]["annos"])
            for i in eval_indices
        ]
        det_annos = [aligned[i] for i in eval_indices]
        result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos, det_annos, dataset.class_names
        )
        car = float(ap_dict.get("Car_3d/moderate_R40", np.nan))
        ped = float(ap_dict.get("Pedestrian_3d/moderate_R40", np.nan))
        cyc = float(ap_dict.get("Cyclist_3d/moderate_R40", np.nan))
        macro = float(np.nanmean([car,ped,cyc]))

        fields = [
            "global_index","scene","local_pos","frame_id","arrival_ms",
            "absolute_deadline_ms","status","drop_reason","forward_start_ms",
            "forward_finish_ms","prediction_timestamp_ms","queue_wait_ms",
            "forward_ms","arrival_to_finish_ms","deadline_slack_ms",
            "deadline_miss","true_level","runtime_estimate_ms",
        ]
        with (out/"frame_timeline.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for i in eval_indices:
                w.writerow(rows[i])
        with (out/"prediction_events.pkl").open("wb") as f:
            pickle.dump(events, f, protocol=pickle.HIGHEST_PROTOCOL)
        with (out/"stream_det_annos.pkl").open("wb") as f:
            pickle.dump(det_annos, f, protocol=pickle.HIGHEST_PROTOCOL)
        (out/"stream_sap_result.txt").write_text(result_str)
        (out/"stream_sap_dict.json").write_text(json.dumps(builtin(ap_dict), indent=2)+"\n")

        fs, qs = stats(forward_times), stats(queue_times)
        rs, es = stats(response_times), stats(rhat_times)
        match_ratio = kf_matches/kf_measurements if kf_measurements else 0.0
        summary = {
            "version":"streamer_style_streamdsgn_forward_only_v1",
            "method":"Streamer-style StreamDSGN",
            "base_detector":"vanilla_streamdsgn",
            "model_cfg":str(a.cfg), "model_ckpt":str(a.ckpt),
            "input_hz":a.input_hz, "period_ms":T, "timing_scope":"forward_only",
            "no_load":bool(a.pressure_level=="L0" and a.pressure_fraction==0.0),
            "scheduler":{
                "name":"shrinking_tail", "runtime_estimator":a.runtime_estimator,
                "runtime_warmup_source":"L0_only",
                "runtime_warmup_median_ms":runtime_init_ms,
                "ewma_alpha":a.ewma_alpha, "wait_count":wait_count,
            },
            "forecast":{
                "mode":a.forecast_mode,
                "type":"asynchronous_constant_velocity_3d_kalman" if a.forecast_mode=="kf" else "copy",
                "coordinate_state":"[camera_x,camera_y,camera_z,vx,vz]",
                "match_threshold_m":4.0, "r_fac":40.0,
                "measurements":kf_measurements, "matches":kf_matches,
                "match_ratio":match_ratio, "forecast_queries":kf_queries,
            },
            "contention_trace":{
                "type":"no_load_L0" if a.pressure_level=="L0" else "balanced_random_per_sensor_frame",
                "base_level":"L0", "pressure_level":a.pressure_level,
                "pressure_fraction_target":a.pressure_fraction,
                "trace_seed":a.trace_seed, "trace_csv":str(trace_csv),
                "scene_summary":trace_scene_summary,
            },
            "true_sensor_levels":dict(sensor_levels),
            "true_processed_levels":dict(processed_levels),
            "buffer_policy":"latest_frame_only_plus_streamer_wait",
            "history_policy":"processed_frames_only",
            "sensor_frames":n_eval, "processed_frames":processed,
            "overwritten_frames":overwritten,
            "scheduler_skipped_frames":sched_skipped,
            "dropped_frames":dropped, "drop_rate":dropped/n_eval,
            "deadline_miss_count":misses,
            "deadline_miss_rate":misses/processed if processed else 0.0,
            "forward_latency":fs, "queue_wait":qs,
            "arrival_to_finish":rs, "runtime_estimate":es,
            "stream_sap_3d_moderate_R40":{
                "Car":car, "Pedestrian":ped, "Cyclist":cyc, "Macro":macro,
            },
        }
        summary_path = out/"summary.json"
        summary_path.write_text(json.dumps(builtin(summary), indent=2)+"\n")

        print("=" * 100)
        print(f"Streamer-style @ {a.input_hz:g} Hz / {a.pressure_level}")
        print(f"processed/dropped={processed}/{dropped} overwritten={overwritten} scheduler_skip={sched_skipped}")
        print(f"shrinking-tail waits={wait_count} miss={misses}/{processed} ({100*misses/processed if processed else 0:.3f}%)")
        print(f"forward p50/p90/p99={fs['p50_ms']:.4f}/{fs['p90_ms']:.4f}/{fs['p99_ms']:.4f} ms")
        print(f"sAP Car/Ped/Cyc/Macro={car:.4f}/{ped:.4f}/{cyc:.4f}/{macro:.4f}")
        if a.forecast_mode == "kf":
            print(f"KF match ratio={100*match_ratio:.2f}%")
        print(f"summary={summary_path}")
        print("=" * 100)
    finally:
        model.generate_recall_record = original_recall


if __name__ == "__main__":
    main()

# STREAMER_STYLE_STREAMDSGN_EOF
