# PROSIT 2 — Reference Document

> **Source**: `PROSIT 2.pdf`
> **Context**: Autonomous Driving Perception, Road Safety, Embedded Vision, African Visual Domain Shift
> **Client**: Department of Urban Roads (DUR), Ghana
> **Duration**: 3 Sprints (~360 hours total)
> **Prerequisites**: Linear Algebra, Calculus, Python, Deep Learning, Machine Learning

---

## Scenario

DUR wants a low-cost, high-reliability **driver-assistance vision module** deployable on mid-range Android smartphones. Existing CV models (trained on European/American data) fail in Ghana due to **visual domain shift** — informal speed bumps, faded lane markings, varied vehicle types (taxis, trotros, okadas), high dust, and extreme camera vibration.

### Core Objective (NOT a black box demo)

| Pillar | Description |
|---|---|
| **Scene Content** | Identification of hazards and traffic participants |
| **Spatial Reasoning** | Precise object distance and lateral offset (metres) |
| **Failure Awareness** | Understanding of _why_ the system fails (bias & error analysis) |

---

## Per-Frame Requirements

- **Object Detection**: Bounding boxes for **8+ specific classes**
- **Road Segmentation**: Semantic mask of the drivable area / road boundaries
- **Ground-Plane Localisation**: Projection of detections onto 3D road plane (geometric constraints from PROSIT 1)
- **Structured Event Log**: `.jsonl` / `.csv` export → `[Frame_ID, Object_ID, Class, Confidence, Distance_X(m), Offset_Y(m)]`

## Video-Level Requirements

- **Lightweight Tracking**: Consistent ID assignment across frames (e.g. SORT)
- **Incident Summary**: Summary report per clip (e.g. "Detected 2 speed bumps and 1 pedestrian; high confidence in road boundary despite dust haze")

---

## Equipment

1. **Smartphone** — student's own
2. **Vehicle** — any car / uber / trotro for data collection
3. **Tape Measure** — for ground truth validation
4. **GPU** — for deep learning models

---

## Label Space & Operational Definitions

> Must provide **3 borderline examples** per class (e.g. "Is this a pothole or just a shadow?")

| Class | Type | Operational Definition |
|---|---|---|
| **Pothole** | Hazard / Road Defect | Any cavity >10 cm in diameter likely to cause vehicle damage or swerving |
| **Speed Bump / Rumble Strip** | Hazard | Any intentional lateral road elevation (asphalt, concrete, or dirt) |
| **Pedestrian** | Vulnerable | Any human on or within 2 m of the road, including roadside vendors |
| **Motorcycle / Okada** | Vehicle | Two-wheeled motorized vehicles + rider + any attached cargo |
| **Car / Taxi / Trotro** | Vehicle | Four-wheeled passenger vehicles (merged class) |
| **Traffic Cone / Temporary** | Obstacle | Formal cones or informal markers (stones/branches) used for signaling |
| **Custom (2+)** | Select from list | Stray animals, open drains, road signs, or construction barriers |

---

## Hard Constraints

1. **Ghana-Specific Data**: Capture **8 videos** (10–30 s) across **3 environments** — Highway, Residential, Market/Urban
2. **No One-Line Detectors**: Must implement **detection loss** (Classification + Localisation + Objectness) and **assignment logic** manually in PyTorch
3. **Ablation Requirement**: Must prove the impact of choices (e.g. training with vs. without custom Ghanaian road augmentations)

---

## Sprint Structure Overview

### Sprint 1 — Surviving Visual Domain Shift (0–120 hrs)

**Goal**: CNN-based semantic segmentation → pixel-level binary mask (road vs. non-road)

**Deliverables**:
- Demonstrate strong segmentation on clean developed-road imagery
- Quantify performance degradation on Ghanaian footage
- Design domain-inspired augmentation strategies
- Retrain and show measurable robustness improvement

**Evaluation**: Segmentation outputs, IoU comparisons across domains, failure analysis

---

### Sprint 2 — From Surfaces to Objects (120–240 hrs)

**Goal**: Custom object detection module integrated with road-segmentation backbone

**Deliverables**:
- Novel CNN-based detection head (grid-based or anchor-free) from scratch in PyTorch
- Manual classification, localisation, and objectness losses
- NMS without library shortcuts
- Detect 8+ domain-relevant classes
- Optionally constrain detection to drivable region via segmentation

**Evaluation**: mAP@0.5 across object classes, mIoU for segmentation, domain comparative analysis

---

### Sprint 3 — Self-Aware Failure Mining (240–360 hrs)

**Goal**: VLM-assisted failure mining module interfacing with detection + segmentation

**Deliverables**:
- Integrate a Vision–Language Model (VLM) for image–text reasoning
- Design language prompts for failure pattern queries
- Extract and catalogue hard negative examples from unlabelled footage
- Retrain detection model with targeted failure cases

**Evaluation**: mAP increase, reduction in specific failure modes, Failure Analysis Report

---

## Available Data

12 video files in `videos/`:

| File | Format | Size |
|---|---|---|
| IMG_3744.MP4 | MP4 | ~21 MB |
| IMG_3745.MP4 | MP4 | ~24 MB |
| IMG_3746.MP4 | MP4 | ~24 MB |
| IMG_3757.MOV | MOV | ~68 MB |
| IMG_3759.MOV | MOV | ~63 MB |
| IMG_3761.MOV | MOV | ~60 MB |
| IMG_3763.MOV | MOV | ~54 MB |
| IMG_3765.MOV | MOV | ~72 MB |
| IMG_3767.MOV | MOV | ~46 MB |
| IMG_3769.MOV | MOV | ~56 MB |
| IMG_3771.MOV | MOV | ~41 MB |
| IMG_3773.MOV | MOV | ~42 MB |
