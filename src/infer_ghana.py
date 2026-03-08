"""
Inference script — run the detector on unlabelled Ghana dashcam frames
and save annotated visualizations + event log.

Usage:
    python src/infer_ghana.py \
        --ghana_dir   data/raw/ghana \
        --model_path  checkpoints/detector_v2_best.pth \
        --output_dir  results/ghana_detections/ \
        --img_size    416
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.detection_model import DetectionNet
from src.data.detection_dataset import CLASS_NAMES, NUM_CLASSES, get_detection_transforms
from src.metrics.nms import non_max_suppression
from src.utils.projection import project_detection

# Colour palette — one per class (BGR)
PALETTE = [
    (0,   255, 0),    # pothole      — green
    (0,   200, 255),  # speed_bump   — yellow-ish
    (255, 0,   0),    # pedestrian   — blue
    (0,   0,   255),  # motorcycle   — red
    (255, 128, 0),    # car          — orange
    (128, 0,   255),  # traffic_cone — purple
    (0,   255, 255),  # animal       — cyan
    (255, 0,   255),  # open_drain   — magenta
]


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    S = args.img_size // 32
    model = DetectionNet(
        num_classes=NUM_CLASSES, S=S, B=args.B, img_size=args.img_size
    ).to(device)

    ckpt = Path(args.model_path)
    if ckpt.exists():
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        print(f"Loaded: {ckpt}")
    else:
        print(f"WARNING: checkpoint not found at {ckpt} — using random weights")
    model.eval()

    # ── Collect frames ────────────────────────────────────────────────────────
    ghana_dir = Path(args.ghana_dir)
    frames = sorted(ghana_dir.rglob("*.jpg")) + sorted(ghana_dir.rglob("*.png"))
    print(f"Found {len(frames)} Ghana frames in {ghana_dir}")

    if not frames:
        print("No images found. Check --ghana_dir path.")
        return

    # ── Output dirs ───────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    vis_dir = out_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    transforms = get_detection_transforms(img_size=args.img_size, train=False)
    event_log  = []
    total_det  = 0
    zero_det   = 0

    with torch.no_grad():
        for frame_id, img_path in enumerate(tqdm(frames, desc="Detecting")):

            # Load + preprocess
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            aug = transforms(image=img_rgb)
            tensor = aug["image"]
            if isinstance(tensor, np.ndarray):
                tensor = torch.from_numpy(tensor.transpose(2, 0, 1)).float()
            tensor = tensor.unsqueeze(0).to(device)

            # Inference
            preds  = model(tensor)
            decoded = model.decode_predictions(preds, conf_thresh=args.conf_thresh)
            boxes, scores, class_ids = decoded[0]

            # NMS
            if boxes.numel() > 0:
                kept = non_max_suppression(
                    boxes, scores, class_ids, iou_threshold=args.nms_thresh
                )
                boxes     = boxes[kept]
                scores    = scores[kept]
                class_ids = class_ids[kept]

            n_det = boxes.shape[0]
            total_det += n_det
            if n_det == 0:
                zero_det += 1

            # ── Visualisation ─────────────────────────────────────────────────
            vis = img_bgr.copy()
            h_orig, w_orig = vis.shape[:2]

            for i in range(n_det):
                cx, cy, w, h = boxes[i].tolist()
                cls_id  = class_ids[i].item()
                score   = scores[i].item()
                colour  = PALETTE[cls_id % len(PALETTE)]

                # Scale from model-normalised coords → original image pixels
                x1 = int((cx - w / 2) * w_orig)
                y1 = int((cy - h / 2) * h_orig)
                x2 = int((cx + w / 2) * w_orig)
                y2 = int((cy + h / 2) * h_orig)

                cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)
                label = f"{CLASS_NAMES[cls_id]} {score:.2f}"
                cv2.putText(
                    vis, label, (x1, max(y1 - 6, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2
                )

                # Distance projection
                proj = project_detection(
                    [cx * args.img_size, cy * args.img_size,
                     w  * args.img_size, h  * args.img_size],
                    args.img_size, args.img_size
                )
                dist_label = f"{proj['distance_m']:.1f}m"
                cv2.putText(
                    vis, dist_label, (x1, min(y2 + 16, h_orig - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1
                )

                event_log.append({
                    "frame_id":    frame_id,
                    "frame_file":  img_path.name,
                    "object_id":   i,
                    "class":       CLASS_NAMES[cls_id],
                    "confidence":  round(score, 3),
                    "distance_x_m": proj["distance_m"],
                    "offset_y_m":  proj["offset_m"],
                })

            # Save annotated image
            out_name = vis_dir / f"{img_path.stem}_det.jpg"
            cv2.imwrite(str(out_name), vis)

    # ── Save event log ────────────────────────────────────────────────────────
    log_path = out_dir / "ghana_event_log.jsonl"
    with open(log_path, "w") as f:
        for entry in event_log:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'='*50}")
    print(f"  Ghana Inference Complete")
    print(f"{'='*50}")
    print(f"  Frames processed : {len(frames)}")
    print(f"  Frames with dets : {len(frames) - zero_det}  ({100*(len(frames)-zero_det)/max(len(frames),1):.1f}%)")
    print(f"  Frames zero dets : {zero_det}  ({100*zero_det/max(len(frames),1):.1f}%)")
    print(f"  Total detections : {total_det}")
    print(f"  Avg per frame    : {total_det/max(len(frames),1):.1f}")
    print(f"  Event log        : {log_path}")
    print(f"  Visualizations   : {vis_dir}  ({len(frames)} files)")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference on unlabelled Ghana dashcam frames"
    )
    parser.add_argument(
        "--ghana_dir", type=str, default="data/raw/ghana",
        help="Directory with Ghana dashcam frames"
    )
    parser.add_argument(
        "--model_path", type=str, default="checkpoints/detector_v2_best.pth",
        help="Path to detector checkpoint (Sprint 3 retrained = detector_v2_best.pth)"
    )
    parser.add_argument("--output_dir",  type=str, default="results/ghana_detections/")
    parser.add_argument("--img_size",    type=int, default=416)
    parser.add_argument("--B",           type=int, default=2)
    parser.add_argument("--conf_thresh", type=float, default=0.25,
                        help="Lower than eval — Ghana is harder, so be more permissive")
    parser.add_argument("--nms_thresh",  type=float, default=0.5)
    args = parser.parse_args()
    run_inference(args)
