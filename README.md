# PROSIT 2: Autonomous Driving Perception (Ghanaian Roads)

A PyTorch-based driver-assistance vision module tackling **Visual Domain Shift** on West African roads — potholes, laterite dust haze, informal speed bumps, faded lane markings, and chaotic mixed traffic (okadas, trotros, livestock).

Built from scratch for the Department of Urban Roads (DUR), Ghana.

**Group Members**: Joseph Ajegetina Ajongba · Nana Yaw Adjei Koranteng · Kevin Takyi Yeboah · Abubakari Sadik Osman

---

## Results

### Sprint 1 — Road Segmentation Ablation

| Model | Training | Zero-shot mIoU (Ghana) |
|---|---|---|
| Baseline | CamVid only | 5.07% |
| Augmented | CamVid + Ghana-condition augs | **8.96%** |
| **Δ** | | **+3.89 pp (+77%)** |

### Sprint 2 — Custom Object Detector (IDD val, 977 samples)

| Class | GT | Detections | AP@0.5 |
|---|---|---|---|
| pedestrian | 2,532 | 1,676 | 4.55% |
| motorcycle | 1,258 | 494 | 9.09% |
| car | 2,967 | 2,743 | 12.52% |
| traffic_cone | 369 | 14 | 0.00% |
| animal | 163 | 45 | 0.00% |
| **mAP@0.5** | | | **5.23%** |

### Sprint 3 — VLM Failure Mining + Retraining

| | Sprint 2 | Sprint 3 | Δ |
|---|---|---|---|
| mAP@0.5 (IDD val) | 5.23% | **6.11%** | **+16.8%** |
| Pedestrian AP | 4.55% | 9.09% | +100% |
| Ghana detections | 916 frames | **1,418 frames** | +54.8% |
| Ghana coverage | 43.4% | **48.8%** | +5.4pp |

VLM diagnosed: `dust_haze` (76%) · `low_contrast` (24%) · avg difficulty 7.6/10

---

## Architecture

```
Sprint 1: CamVid → U-Net (BCE+Dice) → ablation vs Ghana pseudo-masks
Sprint 2: U-Net encoder → 13×13 grid head → mAP@0.5 + event_log.jsonl
Sprint 3: detector → Ghana failure mining → Gemini VLM → retrain
```

**Key constraints met** (PROSIT hard requirements):
- ✅ No library shortcuts — NMS, all 3 loss components implemented in pure PyTorch
- ✅ 8 domain-relevant classes (pothole, speed_bump, pedestrian, motorcycle, car, traffic_cone, animal, open_drain)
- ✅ Structured event log: `[frame_id, class, confidence, distance_x_m, offset_y_m]`
- ✅ Ground-plane distance projection via pinhole camera model
- ✅ VLM integration (OpenRouter) with structured JSON failure diagnosis prompts
- ✅ Ablation studies in both Sprint 1 and Sprint 3

---

## Repo Structure

```
src/
├── models/          segmentation_model.py, detection_model.py
├── data/            dataset.py, detection_dataset.py, augmentations.py, convert_idd.py
├── metrics/         detection_loss.py, nms.py
├── utils/           projection.py
├── vlm/             vlm_client.py, failure_miner.py, prompts.py
├── train.py / train_augmented.py          Sprint 1
├── train_detector.py / train_detector_v2.py  Sprint 2/3
├── ablation.py, evaluate_detector.py
└── infer_ghana.py   Inference on unlabelled Ghana footage
docs/
├── report.md        Full technical report with real results
└── prosit2-reference.md
PROSIT2_Colab.ipynb  Orchestrator notebook — run all sprints on Colab
```

---

## Quick Start (Colab)

Open `PROSIT2_Colab.ipynb` in Google Colab. Set:
1. `REPO_URL` — your GitHub repo URL (Cell 0.1)
2. `DRIVE_ROOT` — your Google Drive path (Cell 0.2)
3. `OPENROUTER_KEY` — your OpenRouter API key (Cell 3.1)

Then **Runtime → Run All**.

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO/prosits/object-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

> Datasets and checkpoints are excluded from git (see `.gitignore`). Training was done on Google Colab with GPU.

## Key Commands

```bash
# Sprint 1 — train segmentation
python src/train.py --data_dir data/camvid --checkpoint_dir checkpoints/

# Sprint 2 — train detector
python src/train_detector.py --data_dir data/idd_yolo/train --epochs 30

# Sprint 2 — evaluate mAP@0.5
python src/evaluate_detector.py --data_dir data/idd_yolo/val \
    --model_path checkpoints/detector_best.pth --output_dir results/sprint2/

# Sprint 3 — VLM failure mining (OpenRouter)
python src/vlm/failure_miner.py --ghana_dir data/raw/ghana \
    --backend openrouter --api_key $OPENROUTER_API_KEY \
    --model_name "meta-llama/llama-3.2-11b-vision-instruct:free"

# Sprint 3 — retrain
python src/train_detector_v2.py --data_dir data/idd_yolo/train \
    --pretrained checkpoints/detector_best.pth \
    --failure_analysis results/failure_mining/failure_analysis.json

# Inference on Ghana footage (no labels needed)
python src/infer_ghana.py --ghana_dir data/raw/ghana \
    --model_path checkpoints/detector_v2_best.pth \
    --output_dir results/ghana_detections/
```

## Documentation

Full technical report: [`docs/report.md`](docs/report.md)
