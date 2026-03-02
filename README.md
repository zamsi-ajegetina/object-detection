# PROSIT 2: Autonomous Driving Perception (Ghanaian Roads)

A comprehensive PyTorch-based driver-assistance vision module designed to tackle extreme **Visual Domain Shift** on unstructured West African roads (potholes, faded lane markings, extreme dust, informal speed bumps, and dense traffic like *okadas*).

Built from scratch as part of the PROSIT 2 assignment for the Department of Urban Roads (DUR), Ghana.

## 🚀 Key Features

- **No Black Box Libraries**: The Object Detection architecture, Non-Maximum Suppression (NMS), and composite Multi-Part Loss functions are implemented entirely **from scratch** in pure PyTorch (no `ultralytics` YOLO or `torchvision.models.detection` shortcuts).
- **Physical Spatial Reasoning**: Uses Inverse Perspective Mapping (IPM) to estimate the physical distance (in meters) and lateral offset of detected objects directly from a monocular dashboard camera.
- **Domain-Specific Augmentation Strategy**: Uses `albumentations` to heavily simulate Ghanaian environmental hazards like laterite dust haze, extreme shadow casting, and motion blur during training.
- **Self-Aware Failure Mining**: Integrates Google's Vision-Language Model (Gemini VLM) to automatically diagnose detector failures on unlabelled frames, generate a failure report, and mine hard negatives for targeted retraining.
- **Lightweight Tracking**: Assigns consistent object IDs across consecutive video frames using Simple Online and Realtime Tracking (SORT).

## 📂 Project Structure (3 Sprints)

The project was executed over three massive sprints (~360 hours theory/practice):

### Sprint 1: Surviving Visual Domain Shift (Road Segmentation)
- Extracted and auto-annotated unlabelled dashboard footage using classical CV techniques.
- Built a native `U-Net` semantic segmentation model to classify drivable road surface vs background.
- Conducted a zero-shot ablation study proving that injecting custom domain augmentations (shadows, hue shifts) improved generalisation on Ghanaian roads by **+19.5% mIoU** compared to a clean-data baseline.

### Sprint 2: From Surfaces to Objects (Object Detection)
- Recycled the trained U-Net Encoder to serve as the feature extraction backbone.
- Built a custom `13x13` grid-prediction head predicting objectness, bounding boxes, and 8 African-context classes (e.g. `pedestrian`, `motorcycle`, `car`, `speed_bump`, `pothole`, `open_drain`).
- Converted and trained on the India Driving Dataset (IDD) as a proxy for chaotic developing-world traffic.
- Wrote custom Evaluation pipelines outputting 11-point interpolated `mAP@0.5` and JSONL structured event logs with physical object distances.

### Sprint 3: Self-Aware Failure Mining (VLM Integration)
- Built an end-to-end pipeline (`failure_miner.py`) that runs the Sprint 2 detector on unlabelled Ghanaian frames to identify low-confidence or zero-detection failures.
- Pinged the Gemini API locally with failing frames, asking the VLM *why* the detector failed (e.g., "heavy occlusion by dust haze").
- Automatically hardened the data augmentation pipeline based on identified failure modes and retrained the model on mined hard-negatives.

## 🛠️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME/prosits/object-detection

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

> **Note**: Datasets (`data/raw`, `data/idd_yolo`) and model weights (`checkpoints/`) are excluded from version control due to GitHub size limits. Model training was performed on Google Colab.

## 📊 Running the Code

### 1. Object Detection Training (Sprint 2/3)
```bash
python src/train_detector.py \
    --data_dir data/idd_yolo/train \
    --batch_size 8 \
    --epochs 30
```

### 2. Evaluation & Projection
Computes mAP@0.5 and exports bounding box visualisations plus `event_log.jsonl`.
```bash
python src/evaluate_detector.py \
    --data_dir data/idd_yolo/val \
    --model_path checkpoints/detector_best.pth \
    --output_dir results/sprint2_detection/
```

### 3. VLM Failure Mining Pipeline (Sprint 3)
Requires a valid Google Gemini API Key.
```bash
export GEMINI_API_KEY="your_api_key_here"
python src/vlm/failure_miner.py \
    --ghana_dir data/raw/ghana \
    --model_path checkpoints/detector_best.pth
```

## 📝 Documentation
Detailed walkthroughs of the engineering challenges and ablation studies are available in the `docs/` folder (or via the system-generated AI artifacts).
