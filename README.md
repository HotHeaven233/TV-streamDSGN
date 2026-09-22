# TV-streamDSGN

Official implementation and reproduction code for:

**Deadline-Aware Elastic Execution for Streaming Stereo 3D Perception**

TV-streamDSGN extends StreamDSGN with:

- multi-history temporal enhancement;
- stage-wise elastic BEV execution;
- contention-aware rolling runtime control;
- deadline-aware suffix selection.

This README focuses on reproducing the two main streaming experiments reported in the paper:

- **Fig. 6:** no-load frequency scalability at 35/40/45/50 Hz;
- **Fig. 7:** random GPU contention at 35 Hz.

The experiments use the KITTI Tracking benchmark and report streaming 3D AP_R40 at Moderate difficulty for Car, Pedestrian, and Cyclist. Macro AP is the arithmetic mean of the three class AP values.

---

## 1. Repository layout

```text
TV-streamDSGN/
├── configs/
├── data/
├── extra_data/
│   ├── checkpoint_epoch_20.pth
│   └── planes.zip
├── mmdetection-v2.22.0/
├── pcdet/
├── scripts/
│   ├── reproduce_final/
│   └── stream_exp/
├── tools/
├── requirements.txt
└── setup.py
```

The provided `extra_data/checkpoint_epoch_20.pth` is the original StreamDSGN checkpoint used to initialize the proposed models and the corresponding baselines.

---

## 2. Experimental environment

The paper results were obtained on:

```text
GPU:      NVIDIA GeForce RTX 4090, 24 GB
PyTorch:  2.7.1
CUDA:     12.8
```

The bundled MMDetection v2.22.0 requires:

```text
MMCV >= 1.3.17 and <= 1.5.0
```

The repository also requires `spconv`, CUDA compilation support, and the Python dependencies listed in `requirements.txt`.

The exact Python and spconv versions are not pinned in the current repository. Install versions compatible with your PyTorch/CUDA setup.

### 2.1 Create an environment

```bash
conda create -n tv-streamdsgn python=3.10 -y
conda activate tv-streamdsgn
```

Install PyTorch 2.7.1 with a CUDA 12.8-compatible build using the official PyTorch installation method for your system.

Then install a compatible MMCV build in the range required by the bundled MMDetection:

```text
1.3.17 <= MMCV <= 1.5.0
```

and install a PyTorch/CUDA-compatible version of `spconv`.

Install the remaining packages:

```bash
pip install -r requirements.txt
pip install -e mmdetection-v2.22.0
pip install -e .
```

A successful `pip install -e .` compiles the CUDA extensions required by the detector.

### 2.2 Verify the environment

```bash
python - <<'PY'
import torch
import mmcv
import pcdet

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.version.cuda)
print("MMCV    :", mmcv.__version__)
print("CUDA OK :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
PY
```

---

## 3. Configure local paths

Experiment scripts source `scripts/stream_exp/00_env.sh`, which in turn optionally reads `.streamdsgn_local_env.sh`.

Create/edit `.streamdsgn_local_env.sh` for your machine:

```bash
cat > .streamdsgn_local_env.sh <<EOF_LOCAL
export STREAMDSGN_REPO_ROOT="$(pwd)"
export STREAMDSGN_ORIGINAL_CKPT="$(pwd)/extra_data/checkpoint_epoch_20.pth"
export CUDA_VISIBLE_DEVICES="0"
EOF_LOCAL
```

If the Conda environment is already activated manually, no `STREAMDSGN_CONDA_ENV` variable is required.

Check the environment:

```bash
source scripts/stream_exp/00_env.sh
```

---

## 4. KITTI Tracking data preparation

Download the KITTI Tracking dataset, including stereo images, Velodyne point clouds, calibration files, and tracking labels.

Place the dataset under:

```text
data/kitti_tracking/
```

The current data loader expects:

```text
data/kitti_tracking/
└── training/
    ├── calib/
    ├── image_02/
    ├── image_03/
    ├── label_02/
    ├── velodyne/
    └── planes/
```

The repository provides pre-computed road planes in `extra_data/planes.zip`.

```bash
unzip extra_data/planes.zip -d data/kitti_tracking/training/
```

Make sure the final plane files are stored as:

```text
data/kitti_tracking/training/planes/<scene>/<frame>.txt
```

---

## 5. Generate streaming splits

The original StreamDSGN configuration uses the `prev2/prev/token/next` history layout:

```bash
python tools/gen_split_kitti_tracking.py \
    --root_dir ./data/kitti_tracking \
    --sample_stride 1 \
    --len_frames 40 \
    --sample_mode 3 \
    --save_split
```

TV-streamDSGN additionally uses three historical frames `prev3/prev2/prev`:

```bash
python tools/gen_split_kitti_tracking_prev3.py \
    --root_dir ./data/kitti_tracking \
    --sample_stride 1 \
    --len_frames 40 \
    --save_split
```

This produces preparation directories such as:

```text
data/kitti_tracking/
├── frame_stride_1-len_frames_40-token_prev2_prev_next/
└── frame_stride_1-len_frames_40-token_prev3_prev2_prev_next/
```

---

## 6. Generate KITTI information files and GT database

For the original StreamDSGN data layout:

```bash
python -m pcdet.datasets.kitti_streaming.lidar_kitti_streaming \
    create_kitti_infos \
    --cfg configs/lidar/dataset_configs/kitti_tracking_streaming-token_prev2_prev_next.yaml \
    --workers 8

python -m pcdet.datasets.kitti_streaming.lidar_kitti_streaming \
    create_gt_database_only \
    --cfg configs/lidar/dataset_configs/kitti_tracking_streaming-token_prev2_prev_next.yaml \
    --image_crops \
    --workers 8
```

For the three-history TV-streamDSGN layout:

```bash
python -m pcdet.datasets.kitti_streaming.lidar_kitti_streaming \
    create_kitti_infos \
    --cfg configs/lidar/dataset_configs/kitti_tracking_streaming-token_prev3_prev2_prev_next.yaml \
    --workers 8

python -m pcdet.datasets.kitti_streaming.lidar_kitti_streaming \
    create_gt_database_only \
    --cfg configs/lidar/dataset_configs/kitti_tracking_streaming-token_prev3_prev2_prev_next.yaml \
    --image_crops \
    --workers 8
```

---

## 7. Required final experiment configurations

The final experiment pipeline expects:

```text
configs/stream/kitti_models/
├── stream_dsgn_r18-token_prev_next-feature_align_avg_fusion-lka_7-mcl.yaml
├── stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual.yaml
└── stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual-transtreaming_tat_v2_15ep.yaml
```

The three-history KITTI preparation configuration is:

```text
configs/lidar/dataset_configs/
└── kitti_tracking_streaming-token_prev3_prev2_prev_next.yaml
```

Before a full reproduction, run:

```bash
bash scripts/reproduce_final/00_preflight.sh
```

---

# Reproducing TV-streamDSGN

## 8. Check the original StreamDSGN model

The original checkpoint is:

```text
extra_data/checkpoint_epoch_20.pth
```

Run:

```bash
bash scripts/reproduce_final/01_sanity_original.sh
```

Output:

```text
outputs/stream_buffer_timestamp/original_10hz/
```

---

## 9. Train the Multi-History Residual Adapter

The original detector is kept fixed and the adapter is trained for 5 epochs.

```bash
bash scripts/reproduce_final/02_train_k3_full.sh
```

Expected final checkpoint:

```text
outputs/stream_kitti_models/
stream_dsgn_r18-token_prev3_prev2_prev_next-mh_residual.mh3_residual_retrain/
ckpt/checkpoint_epoch_5.pth
```

---

## 10. Train the elastic TV-streamDSGN model

```bash
bash scripts/reproduce_final/03_train_tv_stream3d.sh
```

Default experiment name:

```text
elastic_bev_v4_bn_from_k3
```

Final checkpoint:

```text
outputs/elastic_bev/elastic_bev_v4_bn_from_k3/
ckpt/checkpoint_epoch_20.pth
```

The training script uses:

```text
initial learning rate : 2e-4
minimum learning rate : 2e-6
weight decay          : 1e-4
warm-up               : 5%
gradient clipping     : 5.0
epochs                : 20
```

---

## 11. Build runtime profiles and controller tables

Before online evaluation, TV-streamDSGN requires causal-prefix BN statistics, 84-schedule quality measurements, L0-L4 forward/suffix latency profiles, contention calibration, and controller lookup tables.

Run:

```bash
bash scripts/reproduce_final/04_prepare_tv_runtime.sh
```

Main controller table:

```text
outputs/elastic_bev/elastic_bev_v4_bn_from_k3/
all84_contention_profile_v6/e20_n100/
controller_remaining_latency_table.csv
```

The complete table contains:

```text
5 contention levels × 84 schedules × 7 checkpoints = 2940 rows
```

---

# Baselines

## 12. Original StreamDSGN

No additional training is required because the original checkpoint is provided in `extra_data/`.

## 13. Streamer-style baseline

No additional network training is required.

```text
tools/streamer_style_runtime.py
tools/eval_streamer_style_streamdsgn.py
scripts/stream_exp/49_run_streamer_style.sh
```

## 14. MTD baseline

Prepare the H2/H3 training configurations and evaluators:

```bash
bash scripts/stream_exp/49_prepare_mtd_three_head.sh
```

Train H2 and H3:

```bash
bash scripts/stream_exp/50_train_mtd_three_head.sh
```

The final evaluator uses H1/H2/H3 prediction branches.

## 15. Transtreaming-style baseline

Train:

```bash
bash scripts/reproduce_final/06_train_transtreaming_baseline.sh
```

The model is trained for 15 epochs and the paper evaluation uses the validation-selected epoch-13 checkpoint.

Some Transtreaming helper scripts in the current tree contain the original local path `/data/jhb/workspace/streamDSGN`. If the repository is located elsewhere, update the initial `cd` line before running them.

---

# Main paper experiments

## 16. Fig. 6: no-load frequency scalability

Fig. 6 evaluates all methods without external GPU contention at 35, 40, 45, and 50 Hz. Runtime warm-up is 80 frames.

### 16.1 TV-streamDSGN

```bash
bash scripts/reproduce_final/07_run_tv_stream3d_formal.sh
```

No-load outputs:

```text
outputs/elastic_bev/elastic_bev_v4_bn_from_k3/formal_streaming/
├── 35Hz_forward_only/L0/
├── 40Hz_forward_only/L0/
├── 45Hz_forward_only/L0/
└── 50Hz_forward_only/L0/
```

### 16.2 Original StreamDSGN

```bash
bash scripts/reproduce_final/08_run_original_formal.sh
```

Outputs:

```text
outputs/original_streamdsgn/formal_streaming_no_load/
```

### 16.3 Streamer-style

```bash
for HZ in 35 40 45 50
do
    bash scripts/stream_exp/49_run_streamer_style.sh \
        "${HZ}" 80 "L0" 0 20260903 0.0 0.5 kf
done
```

### 16.4 MTD

```bash
bash scripts/stream_exp/51_run_mtd_three_head_no_load.sh \
    "35,40,45,50" 80 0 20260903
```

### 16.5 Transtreaming-style

```bash
bash scripts/stream_exp/55_run_transtreaming_v2_best_no_load.sh \
    "35,40,45,50" 80 0 20260903
```

---

## 17. Expected Fig. 6 results

The paper reports **Macro AP / deadline-miss rate (%)**:

| Method | 35 Hz | 40 Hz | 45 Hz | 50 Hz |
|---|---:|---:|---:|---:|
| StreamDSGN | 49.49 / 0.11 | 19.78 / 97.60 | 17.54 / 97.33 | 16.11 / 100.00 |
| Streamer-style | 25.47 / 0.00 | 20.01 / 97.60 | 19.11 / 97.26 | 17.31 / 100.00 |
| MTD | 54.04 / 0.14 | 33.97 / 97.59 | 29.37 / 97.26 | 25.96 / 100.00 |
| Transtreaming-style | 31.65 / 48.79 | 18.62 / 100.00 | 16.98 / 100.00 | 15.34 / 100.00 |
| TV-streamDSGN | **54.10 / 0.00** | **42.51 / 0.00** | **42.46 / 0.00** | **36.21 / 0.00** |

Small numerical differences may occur across GPU/software installations because the runtime controller uses measured latency.

---

## 18. Fig. 7: random GPU contention at 35 Hz

The random-contention experiment uses:

```text
input rate        : 35 Hz
warm-up           : 80 frames
contention traces : L0/L1, L0/L2, L0/L3, L0/L4
pressure fraction : 0.5
trace seed        : 20260903
TV guard          : 0.25 ms per remaining control decision
```

`Random50` is retained in some internal script/output names for historical reasons. In the paper this experiment is called **random GPU contention**.

### 18.1 TV-streamDSGN

Already included in:

```bash
bash scripts/reproduce_final/07_run_tv_stream3d_formal.sh
```

or run directly:

```bash
bash scripts/stream_exp/45_run_tv_stream3d_30hz_random50.sh \
    20 100 35 80 "L1,L2,L3,L4" 0 0.25 20260903 0.5
```

### 18.2 Original StreamDSGN

Included in:

```bash
bash scripts/reproduce_final/08_run_original_formal.sh
```

or:

```bash
bash scripts/stream_exp/46_run_original_streamdsgn_30hz_random50.sh \
    35 80 "L1,L2,L3,L4" 0 20260903 0.5
```

### 18.3 Streamer-style

```bash
bash scripts/stream_exp/49_run_streamer_style.sh \
    35 80 "L1,L2,L3,L4" 0 20260903 0.5 0.5 kf
```

### 18.4 MTD

```bash
bash scripts/stream_exp/52_run_mtd_three_head_random50.sh \
    35 80 "L1,L2,L3,L4" 0 20260903 0.5
```

### 18.5 Transtreaming-style

```bash
bash scripts/stream_exp/56_run_transtreaming_v2_best_random50.sh \
    35 80 "L1,L2,L3,L4" 0 20260903 0.5
```

---

## 19. Expected Fig. 7 results

The paper reports **Macro AP / deadline-miss rate (%)**:

| Method | L1 | L2 | L3 | L4 |
|---|---:|---:|---:|---:|
| StreamDSGN | 33.14 / 49.02 | 22.17 / 87.50 | 18.75 / 89.68 | 18.39 / 89.06 |
| Streamer-style | 22.89 / 48.63 | 20.78 / 86.84 | 18.82 / 80.35 | 18.71 / 71.47 |
| MTD | 34.84 / 51.74 | 33.30 / 88.20 | 28.66 / 89.07 | 24.65 / 89.30 |
| Transtreaming-style | 20.21 / 97.78 | 18.23 / 98.15 | 17.11 / 97.80 | 17.17 / 97.92 |
| TV-streamDSGN | **48.19 / 0.00** | **44.91 / 0.00** | **38.82 / 1.42** | **36.60 / 4.91** |

The deadline-miss rate is computed over processed frames. Under stronger contention, more waiting frames may be replaced before inference, so the processed-frame miss rate is not necessarily monotonic.

---

## 20. Collect Fig. 6 and Fig. 7 numbers

After all formal evaluations finish:

```bash
python - <<'PY'
import json
from pathlib import Path

EXP = "elastic_bev_v4_bn_from_k3"
SEED = "20260903"

methods = {
    "StreamDSGN": {
        "nl": lambda hz: Path(f"outputs/original_streamdsgn/formal_streaming_no_load/{hz}Hz_forward_only/L0/summary.json"),
        "rnd": lambda lv: Path(f"outputs/original_streamdsgn/formal_streaming_random50/35Hz_forward_only_seed{SEED}/L0_{lv}_p0.5/summary.json"),
    },
    "Streamer-style": {
        "nl": lambda hz: Path(f"outputs/streamer_style_streamdsgn/formal_streaming_no_load/{hz}Hz_forward_only/L0/summary.json"),
        "rnd": lambda lv: Path(f"outputs/streamer_style_streamdsgn/formal_streaming_random50/35Hz_forward_only_seed{SEED}/L0_{lv}_p0.5/summary.json"),
    },
    "MTD": {
        "nl": lambda hz: Path(f"outputs/mtd_three_head/formal_streaming_no_load/{hz}Hz_forward_only/L0/summary.json"),
        "rnd": lambda lv: Path(f"outputs/mtd_three_head/formal_streaming_random50/35Hz_forward_only_seed{SEED}/L0_{lv}_p0.5/summary.json"),
    },
    "Transtreaming-style": {
        "nl": lambda hz: Path(f"outputs/transtreaming_v2_best/formal_streaming_no_load/{hz}Hz_forward_only/L0/summary.json"),
        "rnd": lambda lv: Path(f"outputs/transtreaming_v2_best/formal_streaming_random50/35Hz_forward_only_seed{SEED}/L0_{lv}_p0.5/summary.json"),
    },
    "TV-streamDSGN": {
        "nl": lambda hz: Path(f"outputs/elastic_bev/{EXP}/formal_streaming/{hz}Hz_forward_only/L0/summary.json"),
        "rnd": lambda lv: Path(f"outputs/elastic_bev/{EXP}/formal_streaming_random50/35Hz_forward_only_seed{SEED}/L0_{lv}_p0.5/summary.json"),
    },
}


def read_result(path):
    if not path.is_file():
        return None
    s = json.loads(path.read_text())
    q = s["stream_sap_3d_moderate_R40"]
    return float(q["Macro"]), 100.0 * float(s["deadline_miss_rate"])


print("\nFIG. 6 | NO-LOAD FREQUENCY SWEEP")
for name, cfg in methods.items():
    vals = []
    for hz in (35, 40, 45, 50):
        r = read_result(cfg["nl"](hz))
        vals.append("MISSING" if r is None else f"{r[0]:.2f}/{r[1]:.2f}")
    print(f"{name:<22}" + "  ".join(f"{v:>14}" for v in vals))

print("\nFIG. 7 | RANDOM GPU CONTENTION @ 35 Hz")
for name, cfg in methods.items():
    vals = []
    for lv in ("L1", "L2", "L3", "L4"):
        r = read_result(cfg["rnd"](lv))
        vals.append("MISSING" if r is None else f"{r[0]:.2f}/{r[1]:.2f}")
    print(f"{name:<22}" + "  ".join(f"{v:>14}" for v in vals))
PY
```

---

## 21. Final consistency audit

```bash
bash scripts/reproduce_final/11_final_result_audit.sh
```

This checks the formal TV-streamDSGN, Original StreamDSGN, MTD, and Transtreaming-style output trees and verifies that the formal random-contention traces are byte-identical across those methods.

The current audit script does not include Streamer-style; use the result-collection command above to include all five methods.

---

## 22. Recommended reproduction order

```bash
bash scripts/reproduce_final/00_preflight.sh
bash scripts/reproduce_final/01_sanity_original.sh
bash scripts/reproduce_final/02_train_k3_full.sh
bash scripts/reproduce_final/03_train_tv_stream3d.sh
bash scripts/reproduce_final/04_prepare_tv_runtime.sh

bash scripts/stream_exp/49_prepare_mtd_three_head.sh
bash scripts/stream_exp/50_train_mtd_three_head.sh

bash scripts/reproduce_final/06_train_transtreaming_baseline.sh

bash scripts/reproduce_final/07_run_tv_stream3d_formal.sh
bash scripts/reproduce_final/08_run_original_formal.sh

for HZ in 35 40 45 50
do
    bash scripts/stream_exp/49_run_streamer_style.sh \
        "${HZ}" 80 "L0" 0 20260903 0.0 0.5 kf
done

bash scripts/stream_exp/49_run_streamer_style.sh \
    35 80 "L1,L2,L3,L4" 0 20260903 0.5 0.5 kf

bash scripts/reproduce_final/09_run_mtd_formal.sh
bash scripts/reproduce_final/10_run_transtreaming_formal.sh
bash scripts/reproduce_final/11_final_result_audit.sh
```

A complete reproduction including training, 84-schedule profiling, contention calibration, and all formal baseline runs can take substantial time.

---

## 23. Metrics and runtime semantics

The streaming evaluator is asynchronous.

When the detector is busy, only the most recent unprocessed sensor frame is retained. A newer arrival replaces the previously waiting frame.

At each sensor query time, the evaluator returns the most recent prediction that has already completed.

For a frame arriving at time `a_i` with relative deadline `D`, the absolute deadline is:

```text
d_i = a_i + D
```

A processed frame is counted as a deadline miss if its completion time is later than `d_i`.

For the 35-Hz experiments:

```text
D = 1000 / 35 = 28.57 ms
```

Only frames that actually complete inference contribute BEV features to temporal history.

---

## 24. Notes on paper figures

The current repository contains the evaluation pipeline required to regenerate the numerical data underlying the main frequency and contention experiments.

The paper-facing Fig. 6 and Fig. 7 plotting scripts are not part of the current reproduction tree. The formal `summary.json` files contain the Car, Pedestrian, Cyclist, Macro AP, deadline-miss rate, drop rate, and latency values required to recreate the figures.

---

## 25. Citation

If this repository is useful for your research, please cite the TV-streamDSGN paper.

The implementation is based on StreamDSGN and also incorporates components derived from OpenPCDet, DSGN++, MMDetection, and related open-source projects. Please also cite the corresponding upstream projects when appropriate.

---

## License

See [LICENSE](LICENSE).
