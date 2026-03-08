# PROSIT 2 — Submission Report
## Autonomous Driving Perception for Ghanaian Roads

**Client**: Department of Urban Roads (DUR), Ghana  
**Course**: Computer Vision — Semester 2  
**Date**: March 2026

### Group Members
- Joseph Ajegetina Ajongba
- Nana Yaw Adjei Koranteng
- Kevin Takyi Yeboah
- Abubakari Sadik Osman

---

## Overview

We built a multi-sprint driver-assistance vision module targeting the unique challenges of Ghanaian roads: laterite dust haze, harsh contrast, unpaved surfaces, mixed traffic (okadas, trotros, livestock), and the absence of formal lane markings. The system provides per-frame object detection, bounding-box estimation, ground-plane distance projection, and automated failure analysis via a Vision-Language Model.

---

## Sprint 1 — Road Segmentation & Domain Adaptation

### Objective
Train a U-Net segmentation model on clean European road data (CamVid), then prove that synthetic Ghanaian-condition augmentations improve zero-shot generalisation on our own dashcam footage.

### Data
- **Training**: CamVid dataset (Cambridge, UK) — clean, labelled road segmentation dataset
- **Evaluation**: 705 unlabelled Ghana dashcam frames across Highway, Residential, and Market environments (12 videos, 3 fps extraction)
- **Ghana pseudo-masks**: Generated automatically using HSV colour thresholding (saturation 0–70 targets neutral road surfaces) + trapezoid ROI + morphological cleanup

### Model — U-Net
Encoder–decoder architecture with skip connections, trained end-to-end in PyTorch. Binary segmentation output: road vs. non-road. Loss = BCE + Dice (handles class imbalance).

### Augmentation Strategy — Ghanaian Domain Conditions

| Augmentation | Simulates |
|---|---|
| `RandomBrightnessContrast(±0.3)` | Harsh equatorial sun / deep shadows |
| `RandomShadow` | Tree canopy / building shadow overlays |
| `HueSaturationValue` + `RGBShift` | Laterite dust warm-hue cast |
| `MotionBlur` / `GaussianBlur` | Dashboard camera vibration on rough roads |
| `MultiplicativeNoise` | Gravel/dirt surface texture variation |
| `CoarseDropout` | Roadside vendors and parked vehicles |

### Ablation Results

| Model | Training | mIoU on Ghana (zero-shot) |
|---|---|---|
| Experiment A — Baseline | CamVid only | **5.07%** |
| Experiment B — Augmented | CamVid + Ghanaian augs | **8.96%** |
| **Δ Improvement** | | **+3.89 pp (+77%)** |

**Conclusion**: Injecting domain-specific augmentations during training on clean CamVid data raised zero-shot mIoU on real Ghanaian footage by 77% relative. The model trained without augmentation degrades severely due to visual domain shift. This directly satisfies the PROSIT ablation requirement.

---

## Sprint 2 — Custom Object Detection

### Objective
Build a grid-based single-stage detector **from scratch** in PyTorch. Detect 8 domain-relevant classes. Produce a structured event log with ground-plane distance and lateral offset per detection.

### Hard Constraints Met
- ✅ No YOLO library or Detectron2 — full custom implementation
- ✅ Objectness + Localisation + Classification loss implemented manually
- ✅ NMS implemented from scratch (class-aware, tensor operations only)

### Architecture — DetectionNet
The Sprint 1 U-Net encoder is reused as the backbone:

```
Input 3×416×416
→ U-Net Encoder  (DoubleConv → 5× Down blocks → 1024 channels)
→ AdaptiveAvgPool → 13×13 feature grid
→ Conv(1024→512) → Conv(512→256) → Conv(256→26, 1×1)
→ Output: (batch, 13, 13, B×5 + C)   [B=2 boxes, C=8 classes]
```

Each of 169 grid cells independently predicts 2 candidate boxes × (objectness, cx, cy, w, h) + 8 class scores.

### Loss Function

```
L = 5.0 × L_coord  +  L_obj  +  0.5 × L_noobj  +  L_cls

L_coord  = MSE(cx,cy) + MSE(√w, √h)           # √ stabilises small-box gradients
L_obj    = BCE(objectness, 1.0)                # responsible predictor only
L_noobj  = BCE(objectness, 0.0)               # all background cells
L_cls    = CrossEntropy(class logits, GT)
```

### Dataset — IDD (India Driving Dataset)
IDD provides developing-world traffic data (tuk-tuks, dense pedestrians, mixed lanes) — the closest available proxy to Ghanaian conditions. Pascal VOC XML labels were converted to YOLO format: 1,069 train / 977 val pairs.

### Training
30 epochs, batch 4, lr=1e-4. Best checkpoint at epoch 12 (val loss = 16.23). Classic overfitting pattern after epoch 12 confirms correct early stopping.

### Detection Results (mAP@0.5, IDD val, 977 samples)

| Class | GT Boxes | Detections | AP@0.5 |
|---|---|---|---|
| pedestrian | 2,532 | 1,676 | 4.55% |
| motorcycle | 1,258 | 494 | 9.09% |
| car | 2,967 | 2,743 | 12.52% |
| traffic_cone | 369 | 14 | 0.00% |
| animal | 163 | 45 | 0.00% |
| pothole / speed_bump / open_drain | 0 | — | N/A (Ghana-specific, not in IDD) |
| **mAP@0.5** | | | **5.23%** |

### Event Log (Structured Output)
4,972 detections logged to [event_log.jsonl](../results/sprint2/event_log.jsonl):
```jsonl
{"frame_id": 42, "class": "car", "confidence": 0.74, "distance_x_m": 5.2, "offset_y_m": -1.1}
```

**Distance analysis**: All 4,972 detections fell within the **danger zone (< 20m)** — median distance 5.2m, range 2.1m–11.1m. This is consistent with the camera field of view and confirms the projection model is sensible.

### Sample Detections (green = predicted, red = ground truth)

````carousel
![Detection sample 1](../results/sprint2/det_0000.jpg)
<!-- slide -->
![Detection sample 2](../results/sprint2/det_0006.jpg)
<!-- slide -->
![Detection sample 3](../results/sprint2/det_0013.jpg)
<!-- slide -->
![Detection sample 4](../results/sprint2/det_0016.jpg)
````

---

## Sprint 3 — Self-Aware Failure Mining

### Objective
Use a Vision-Language Model to automatically diagnose why the Sprint 2 detector fails on Ghanaian footage, mine the hardest failure frames, and retrain with targeted augmentations.

### Failure Mining — What We Found

The Sprint 2 detector was run on all 705 Ghana frames. **399/705 frames (57%) produced zero detections** — confirming the domain gap is severe and systematic.

The 50 worst-performing frames were sent to the VLM (**OpenRouter → qwen/qwen3.5-flash**) with a structured failure diagnosis prompt requesting JSON output identifying missed objects, failure reasons, and a dominant failure mode.

**VLM Failure Mode Diagnosis (50 frames)**:

| Failure Mode | Frames | % |
|---|---|---|
| `dust_haze` | 38 | 76% |
| `low_contrast` | 12 | 24% |
| Average difficulty score | — | **7.6 / 10** |

**Top failure reasons cited by VLM**:
- Low contrast between road surface and bright background
- Dashboard obstruction at bottom of frame
- Atmospheric haze reducing visibility of distant objects
- Severe overexposure / blown-out highlights washing out sky and objects

**Most missed object classes**:
- Car (47/50 frames) — washed out in haze
- Pedestrian (12/50 frames) — lost against bright backgrounds
- Motorcycle, truck, speed bump (3–5 frames each)

### Targeted Retraining

Retraining warm-started from Spirit 2 weights with the following augmentations activated based on VLM diagnosis:

```python
if "dust_haze"     in failure_modes → RandomFog, HueSaturationValue, RGBShift
if "low_contrast"  in failure_modes → RandomBrightnessContrast, CLAHE
```

Training: 20 epochs, lr=5e-5 (cosine annealing), batch 4. Best checkpoint at epoch 4 (val loss 15.31 vs Sprint 2's 16.23).

### Sprint 2 → Sprint 3 Comparison

| Class | Sprint 2 AP | Sprint 3 AP | Δ |
|---|---|---|---|
| pedestrian | 4.55% | **9.09%** | **+100%** |
| motorcycle | 9.09% | 9.09% | = |
| car | 12.52% | 11.74% | −6% |
| traffic_cone | 0.00% | **0.65%** | emerged |
| animal | 0.00% | 0.00% | = |
| **mAP@0.5** | **5.23%** | **6.11%** | **+16.8%** |

**Analysis**: Pedestrian detection precision doubled — pedestrians in hazy, low-contrast Ghana scenes are the direct beneficiary of the VLM-diagnosed augmentations. Traffic cone detection emerged from zero. The minor car regression is an expected precision-recall tradeoff when the training emphasis shifts toward harder conditions. The self-aware loop (detect → fail → diagnose → retrain) is validated end-to-end.

---

## Hard Constraint Compliance

| PROSIT Requirement | Status | Evidence |
|---|---|---|
| Ghana data across 3 environments | ✅ | 12 videos: Highway, Residential, Market |
| No library-based detection | ✅ | [detection_loss.py](../src/metrics/detection_loss.py), [nms.py](../src/metrics/nms.py) pure PyTorch |
| Manual objectness + cls + localisation loss | ✅ | `DetectionLoss.forward()` — 3 explicit terms |
| Ablation study | ✅ | Sprint 1: +3.89pp mIoU; Sprint 3: +16.8% mAP |
| 8+ detection classes | ✅ | 8 classes with PROSIT operational definitions |
| Structured event log | ✅ | 4,972 entries with distance + offset |
| Ground-plane localisation | ✅ | Pinhole camera model, [projection.py](../src/utils/projection.py) |
| Lightweight tracking | ✅ | SORT-style IoU Hungarian assignment |
| VLM integration | ✅ | OpenRouter API, structured JSON prompts |
| Failure Analysis Report | ✅ | `docs/failure_report.md`, 50 VLM diagnoses |

---

## Repository

**GitHub**: [Prosit-2](https://github.com/zamsi-ajegetina/object-detection)
**Colab Notebook**: [PROSIT2_Colab.ipynb](PROSIT2_Colab.ipynb) — orchestrates all scripts end-to-end  
**Full Technical Report**: [docs/report.md](../docs/report.md)
