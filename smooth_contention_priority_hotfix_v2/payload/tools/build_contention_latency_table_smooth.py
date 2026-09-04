#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path

METRICS=(
    "fixed_prefix_ms","res2_ms","res3_ms","res4_ms","fpn_ms","stereo_ms",
    "rpn_ms","fixed_tail_ms","forward_total_ms","remain_after_prefix_ms",
    "remain_after_res2_ms","remain_after_res3_ms","remain_after_res4_ms",
    "remain_after_fpn_ms","remain_after_stereo_ms","remain_after_rpn_ms",
)
CHECKPOINTS=(
    ("after_prefix","remain_after_prefix_ms"),
    ("after_res2","remain_after_res2_ms"),
    ("after_res3","remain_after_res3_ms"),
    ("after_res4","remain_after_res4_ms"),
    ("after_fpn","remain_after_fpn_ms"),
    ("after_stereo","remain_after_stereo_ms"),
    ("after_rpn","remain_after_rpn_ms"),
)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--levels-json",required=True)
    p.add_argument("--schedule-tsv",required=True)
    p.add_argument("--quality-csv",required=True)
    p.add_argument("--baseline-root",required=True)
    p.add_argument("--contention-root",required=True)
    p.add_argument("--output-wide-csv",required=True)
    p.add_argument("--output-controller-csv",required=True)
    p.add_argument("--output-json",required=True)
    p.add_argument("--allow-partial",action="store_true")
    a=p.parse_args()

    levels_data=json.loads(Path(a.levels_json).read_text())
    levels={x["level"]:x for x in levels_data["levels"]}
    with open(a.schedule_tsv,newline="") as f:
        schedules=list(csv.DictReader(f,delimiter="\t"))
    with open(a.quality_csv,newline="") as f:
        qrows=list(csv.DictReader(f))
    qmap={int(r["id"]):r for r in qrows}

    wide=[]; controller=[]; missing=[]
    for level_name in ("L0","L1","L2","L3","L4"):
        lev=levels[level_name]
        root=Path(a.baseline_root) if level_name=="L0" else Path(a.contention_root)/level_name
        for s in schedules:
            pid=int(s["id"]); tag=s["tag"]
            summary=root/f"{pid:03d}_{tag}"/"forward_profile_summary.json"
            if not summary.exists():
                missing.append({"level":level_name,"id":pid}); continue
            t=json.loads(summary.read_text()); q=qmap.get(pid,{})
            row={
                "level":level_name,
                "strength":lev.get("strength",0.0),
                "calibration_probe_p50_ms":lev["probe_p50_ms"],
                "calibration_probe_p90_ms":lev["probe_p90_ms"],
                "calibration_slowdown":lev["slowdown"],
                "id":pid,"schedule":s["schedule"],"tag":tag,
                "res2":s["res2"],"res3":s["res3"],"res4":s["res4"],
                "fpn":s["fpn"],"stereo":s["stereo"],"rpn":s["rpn"],
                "reference_mean3d_moderate":q.get("reference_mean3d_moderate",""),
                "car_3d_iou07_moderate":q.get("car_3d_iou07_moderate",""),
                "pedestrian_3d_iou05_moderate":q.get("pedestrian_3d_iou05_moderate",""),
                "cyclist_3d_iou05_moderate":q.get("cyclist_3d_iou05_moderate",""),
            }
            for metric in METRICS:
                st=t["metrics"][metric]
                for stat in ("mean_ms","p50_ms","p90_ms","p99_ms"):
                    row[f"{metric}_{stat.replace('_ms','')}"]=st[stat]
            row["summary_path"]=str(summary); wide.append(row)
            for checkpoint,metric in CHECKPOINTS:
                st=t["metrics"][metric]
                controller.append({
                    "level":level_name,
                    "strength":lev.get("strength",0.0),
                    "calibration_probe_p50_ms":lev["probe_p50_ms"],
                    "calibration_slowdown":lev["slowdown"],
                    "profile_id":pid,"schedule":s["schedule"],"checkpoint":checkpoint,
                    "remaining_mean_ms":st["mean_ms"],
                    "remaining_p50_ms":st["p50_ms"],
                    "remaining_p90_ms":st["p90_ms"],
                    "remaining_p99_ms":st["p99_ms"],
                    "reference_quality":q.get("reference_mean3d_moderate",""),
                })

    wp=Path(a.output_wide_csv); wp.parent.mkdir(parents=True,exist_ok=True)
    if wide:
        with wp.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(wide[0].keys())); w.writeheader(); w.writerows(wide)
    cp=Path(a.output_controller_csv); cp.parent.mkdir(parents=True,exist_ok=True)
    if controller:
        with cp.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(controller[0].keys())); w.writeheader(); w.writerows(controller)
    result={
        "expected_wide_rows":420,"completed_wide_rows":len(wide),
        "expected_controller_rows":2940,"completed_controller_rows":len(controller),
        "missing":missing,
        "contention_workload":levels_data.get("workload",{}),
        "classifier_validation":levels_data.get("classifier_validation",{}),
        "classifier_thresholds_ms":levels_data.get("nearest_centroid_thresholds_ms",[]),
        "wide_csv":str(wp),"controller_csv":str(cp),
    }
    op=Path(a.output_json); op.parent.mkdir(parents=True,exist_ok=True)
    op.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps(result,indent=2,ensure_ascii=False))
    if missing and not a.allow_partial:
        raise SystemExit(2)
if __name__=="__main__":
    main()
