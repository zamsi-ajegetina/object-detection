"""
Evaluation script for Sprint 2 Object Detection.
Computes mAP@0.5 and generates visual outputs with bounding boxes.
"""
import torch
import numpy as np
import cv2
import json
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.detection_model import DetectionNet
from src.data.detection_dataset import (
    DetectionDataset, get_detection_transforms, collate_fn, CLASS_NAMES, NUM_CLASSES
)
from src.metrics.nms import non_max_suppression
from src.utils.projection import project_detection


def compute_ap(recalls, precisions):
    """Compute Average Precision using the 11-point interpolation method."""
    ap = 0.0
    for t in np.arange(0., 1.1, 0.1):
        precision_at_t = precisions[recalls >= t]
        if len(precision_at_t) > 0:
            ap += precision_at_t.max()
    return ap / 11.0


def evaluate_detector(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    S = args.img_size // 32
    model = DetectionNet(num_classes=NUM_CLASSES, S=S, B=args.B, img_size=args.img_size).to(device)

    if Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded weights from {args.model_path}")
    else:
        print(f"Warning: {args.model_path} not found!")

    model.eval()

    # Dataset
    transforms = get_detection_transforms(img_size=args.img_size, train=False)
    dataset = DetectionDataset(args.data_dir, transforms=transforms, img_size=args.img_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all predictions and ground truths for mAP calculation
    all_detections = {c: [] for c in range(NUM_CLASSES)}
    all_ground_truths = {c: [] for c in range(NUM_CLASSES)}
    event_log = []

    print(f"Evaluating on {len(dataset)} samples...")

    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(loader)):
            images = images.to(device)
            predictions = model(images)

            # Decode predictions
            decoded = model.decode_predictions(predictions, conf_thresh=args.conf_thresh)
            boxes, scores, class_ids = decoded[0]

            # Apply NMS
            if boxes.numel() > 0:
                kept = non_max_suppression(boxes, scores, class_ids, iou_threshold=args.nms_thresh)
                boxes = boxes[kept]
                scores = scores[kept]
                class_ids = class_ids[kept]

            # Ground truth
            gt_boxes = targets[0]['boxes']
            gt_labels = targets[0]['labels']

            # Store for mAP computation
            for i in range(boxes.shape[0]):
                cls = class_ids[i].item()
                all_detections[cls].append({
                    'image_id': idx,
                    'score': scores[i].item(),
                    'box': boxes[i].tolist()
                })

            for i in range(gt_boxes.shape[0]):
                cls = gt_labels[i].item()
                all_ground_truths[cls].append({
                    'image_id': idx,
                    'box': gt_boxes[i].tolist(),
                    'matched': False
                })

            # Event log entry
            for i in range(boxes.shape[0]):
                cx, cy, w, h = boxes[i].tolist()
                proj = project_detection(
                    [cx * args.img_size, cy * args.img_size, w * args.img_size, h * args.img_size],
                    args.img_size, args.img_size
                )
                event_log.append({
                    'frame_id': idx,
                    'object_id': i,
                    'class': CLASS_NAMES[class_ids[i].item()],
                    'confidence': round(scores[i].item(), 3),
                    'distance_x_m': proj['distance_m'],
                    'offset_y_m': proj['offset_m'],
                })

            # Save visualizations for first N samples
            if idx < args.num_save:
                # Denormalize image
                img_np = images[0].cpu().permute(1, 2, 0).numpy()
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = (img_np * std + mean) * 255
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                # Draw predicted boxes (green)
                for i in range(boxes.shape[0]):
                    cx, cy, w, h = boxes[i].tolist()
                    x1 = int((cx - w / 2) * args.img_size)
                    y1 = int((cy - h / 2) * args.img_size)
                    x2 = int((cx + w / 2) * args.img_size)
                    y2 = int((cy + h / 2) * args.img_size)
                    cls_name = CLASS_NAMES[class_ids[i].item()]
                    score = scores[i].item()

                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{cls_name} {score:.2f}"
                    cv2.putText(img_bgr, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Draw GT boxes (red)
                for i in range(gt_boxes.shape[0]):
                    cx, cy, w, h = gt_boxes[i].tolist()
                    x1 = int((cx - w / 2) * args.img_size)
                    y1 = int((cy - h / 2) * args.img_size)
                    x2 = int((cx + w / 2) * args.img_size)
                    y2 = int((cy + h / 2) * args.img_size)
                    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

                cv2.imwrite(str(out_dir / f"det_{idx:04d}.jpg"), img_bgr)

    # --- Compute mAP@0.5 ---
    aps = []
    print("\n--- Per-Class AP@0.5 ---")
    for c in range(NUM_CLASSES):
        dets = all_detections[c]
        gts = all_ground_truths[c]
        n_gt = len(gts)

        if n_gt == 0:
            print(f"  {CLASS_NAMES[c]}: No GT samples")
            continue

        # Sort detections by score (desc)
        dets = sorted(dets, key=lambda x: x['score'], reverse=True)

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))

        for d_idx, det in enumerate(dets):
            best_iou = 0
            best_gt = -1

            for g_idx, gt in enumerate(gts):
                if gt['image_id'] == det['image_id'] and not gt['matched']:
                    from src.metrics.detection_loss import compute_iou
                    iou = compute_iou(
                        torch.tensor([det['box']]),
                        torch.tensor([gt['box']])
                    ).item()
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = g_idx

            if best_iou >= 0.5 and best_gt >= 0:
                tp[d_idx] = 1
                gts[best_gt]['matched'] = True
            else:
                fp[d_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / n_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recalls, precisions)
        aps.append(ap)
        print(f"  {CLASS_NAMES[c]}: AP={ap:.4f} ({n_gt} GT, {len(dets)} detections)")

    mAP = np.mean(aps) * 100 if aps else 0.0
    print(f"\nmAP@0.5: {mAP:.2f}%")

    # Save metrics
    with open(out_dir / 'metrics.txt', 'w') as f:
        f.write(f"mAP@0.5: {mAP:.2f}%\n")
        for c, ap in zip(range(NUM_CLASSES), aps):
            f.write(f"{CLASS_NAMES[c]}: {ap:.4f}\n")

    # Save structured event log
    with open(out_dir / 'event_log.jsonl', 'w') as f:
        for entry in event_log:
            f.write(json.dumps(entry) + '\n')

    print(f"Saved {len(event_log)} event log entries to {out_dir / 'event_log.jsonl'}")
    print(f"Saved {min(args.num_save, len(dataset))} visualizations to {out_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Sprint 2 Object Detector")
    parser.add_argument('--model_path', type=str, default='checkpoints/detector_best.pth')
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset")
    parser.add_argument('--output_dir', type=str, default='results/detection/')
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--B', type=int, default=2)
    parser.add_argument('--conf_thresh', type=float, default=0.3)
    parser.add_argument('--nms_thresh', type=float, default=0.5)
    parser.add_argument('--num_save', type=int, default=20)
    args = parser.parse_args()
    evaluate_detector(args)
