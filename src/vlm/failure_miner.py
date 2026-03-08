"""
Hard-Negative Failure Mining Pipeline for Sprint 3.

End-to-end workflow:
  1. Run Sprint 2 detector on all unlabelled Ghanaian frames.
  2. Identify failure frames (low confidence, 0 detections, etc.).
  3. Query the VLM to diagnose WHY each frame is difficult.
  4. Catalogue failures by mode and generate a structured report.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm

import torch
import numpy as np
from PIL import Image

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.detection_model import DetectionNet
from src.metrics.nms import non_max_suppression
from src.data.detection_dataset import CLASS_NAMES, NUM_CLASSES, get_detection_transforms
from src.vlm.vlm_client import VLMClient
from src.vlm.prompts import failure_diagnosis_prompt, scene_description_prompt


def run_detector_on_frame(model, image_path, transforms, device, conf_thresh=0.3):
    """Run the Sprint 2 detector on a single frame and return detections."""
    img = np.array(Image.open(image_path).convert('RGB'))

    transformed = transforms(image=img, bboxes=[], class_labels=[])
    img_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)
        decoded = model.decode_predictions(predictions, conf_thresh=conf_thresh)

    boxes, scores, class_ids = decoded[0]

    if boxes.numel() > 0:
        kept = non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5)
        boxes = boxes[kept]
        scores = scores[kept]
        class_ids = class_ids[kept]

    return {
        'num_detections': boxes.shape[0],
        'max_conf': scores.max().item() if scores.numel() > 0 else 0.0,
        'mean_conf': scores.mean().item() if scores.numel() > 0 else 0.0,
        'detections': [
            {
                'class': CLASS_NAMES[class_ids[i].item()],
                'confidence': scores[i].item(),
                'box': boxes[i].tolist()
            }
            for i in range(boxes.shape[0])
        ]
    }


def identify_failure_frames(frame_results, max_failures=50):
    """
    Identify frames where the detector likely failed.
    Criteria:
      - 0 detections (model sees nothing)
      - Very low max confidence (model is uncertain)
      - Very few detections on a busy scene
    """
    # Sort by confidence ascending (worst first)
    sorted_frames = sorted(frame_results, key=lambda x: x['max_conf'])

    failures = []
    for frame in sorted_frames:
        if len(failures) >= max_failures:
            break
        # Include if: no detections or low confidence
        if frame['num_detections'] == 0 or frame['max_conf'] < 0.5:
            failures.append(frame)

    return failures


def mine_failures(args):
    """Main failure mining pipeline."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load model ---
    S = args.img_size // 32
    model = DetectionNet(num_classes=NUM_CLASSES, S=S, B=2, img_size=args.img_size).to(device)

    if Path(args.model_path).exists():
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded detector from {args.model_path}")
    else:
        print(f"Warning: {args.model_path} not found, using random weights!")

    model.eval()
    transforms = get_detection_transforms(img_size=args.img_size, train=False)

    # --- Find all Ghana frames ---
    ghana_dir = Path(args.ghana_dir)
    image_exts = {'.jpg', '.jpeg', '.png'}
    all_frames = sorted([
        f for f in ghana_dir.rglob('*')
        if f.suffix.lower() in image_exts
    ])
    print(f"Found {len(all_frames)} Ghanaian frames in {ghana_dir}")

    # --- Step 1: Run detector on every frame ---
    print("\n=== Step 1: Running detector on all frames ===")
    frame_results = []
    for img_path in tqdm(all_frames, desc="Detecting"):
        try:
            result = run_detector_on_frame(model, img_path, transforms, device, args.conf_thresh)
            result['image_path'] = str(img_path)
            frame_results.append(result)
        except Exception as e:
            print(f"  Error on {img_path.name}: {e}")

    # Summary stats
    total_detections = sum(r['num_detections'] for r in frame_results)
    zero_det_frames = sum(1 for r in frame_results if r['num_detections'] == 0)
    print(f"\nDetection summary:")
    print(f"  Total detections: {total_detections}")
    print(f"  Frames with 0 detections: {zero_det_frames}/{len(frame_results)}")
    print(f"  Average detections/frame: {total_detections/max(len(frame_results),1):.1f}")

    # --- Step 2: Identify failure frames ---
    print(f"\n=== Step 2: Identifying top {args.max_failures} failure frames ===")
    failures = identify_failure_frames(frame_results, max_failures=args.max_failures)
    print(f"Found {len(failures)} failure frames")

    # --- Step 3: VLM Diagnosis ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vlm_results = []

    if args.api_key or os.environ.get('GEMINI_API_KEY'):
        print(f"\n=== Step 3: Querying VLM for failure diagnosis ===")
        client = VLMClient(api_key=args.api_key)

        for i, failure in enumerate(tqdm(failures, desc="VLM Diagnosis")):
            prompt = failure_diagnosis_prompt(
                CLASS_NAMES,
                failure['num_detections'],
                {'max_conf': failure['max_conf'], 'mean_conf': failure['mean_conf']}
            )
            vlm_response = client.query_structured(failure['image_path'], prompt)

            vlm_results.append({
                'image_path': failure['image_path'],
                'num_detections': failure['num_detections'],
                'max_conf': failure['max_conf'],
                'vlm_analysis': vlm_response,
            })

            # Rate limit
            if i < len(failures) - 1:
                import time
                time.sleep(1.5)
    else:
        print("\n=== Step 3: Skipping VLM (no API key) — using detection stats only ===")
        for failure in failures:
            vlm_results.append({
                'image_path': failure['image_path'],
                'num_detections': failure['num_detections'],
                'max_conf': failure['max_conf'],
                'vlm_analysis': {'note': 'VLM skipped — no API key provided'},
            })

    # --- Step 4: Catalogue & Report ---
    print(f"\n=== Step 4: Generating failure report ===")

    # Count failure modes
    failure_modes = Counter()
    for result in vlm_results:
        analysis = result.get('vlm_analysis', {})
        if isinstance(analysis, dict):
            mode = analysis.get('dominant_failure_mode', 'unknown')
            failure_modes[mode] += 1

    # Save raw results
    with open(out_dir / 'failure_analysis.json', 'w') as f:
        json.dump(vlm_results, f, indent=2)

    # Save failure frame list (for retraining)
    failure_paths = [r['image_path'] for r in vlm_results]
    with open(out_dir / 'hard_negative_frames.txt', 'w') as f:
        f.write('\n'.join(failure_paths))

    # Generate markdown report
    report_lines = [
        "# Sprint 3 — Failure Analysis Report\n",
        f"Analyzed **{len(frame_results)}** Ghanaian frames with the Sprint 2 detector.\n",
        "## Detection Summary\n",
        f"| Metric | Value |",
        f"|---|---|",
        f"| Total frames | {len(frame_results)} |",
        f"| Total detections | {total_detections} |",
        f"| Frames with 0 detections | {zero_det_frames} |",
        f"| Avg detections/frame | {total_detections/max(len(frame_results),1):.1f} |",
        f"| Failure frames analyzed by VLM | {len(vlm_results)} |\n",
    ]

    if failure_modes:
        report_lines.append("## Failure Mode Distribution\n")
        report_lines.append("| Failure Mode | Count | Percentage |")
        report_lines.append("|---|---|---|")
        total_modes = sum(failure_modes.values())
        for mode, count in failure_modes.most_common():
            pct = count / total_modes * 100
            report_lines.append(f"| {mode} | {count} | {pct:.1f}% |")
        report_lines.append("")

    # Sample failures
    report_lines.append("## Sample Failure Cases\n")
    for i, result in enumerate(vlm_results[:5]):
        report_lines.append(f"### Case {i+1}: `{Path(result['image_path']).name}`")
        report_lines.append(f"- Detections: {result['num_detections']}")
        report_lines.append(f"- Max confidence: {result['max_conf']:.3f}")
        analysis = result.get('vlm_analysis', {})
        if isinstance(analysis, dict):
            reasons = analysis.get('failure_reasons', [])
            if reasons:
                report_lines.append(f"- Failure reasons: {', '.join(reasons)}")
            missed = analysis.get('objects_missed', [])
            if missed:
                report_lines.append(f"- Objects missed: {', '.join(missed)}")
        report_lines.append("")

    report_lines.append("## Recommendations\n")
    report_lines.append("Based on the analysis, the following targeted improvements are recommended:\n")
    report_lines.append("1. Include the identified hard-negative frames in the training set")
    report_lines.append("2. Apply targeted augmentations matching the top failure modes")
    report_lines.append("3. Retrain with `train_detector_v2.py` and compare mAP\n")

    report_path = Path('docs/failure_report.md')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"\n✅ Failure mining complete!")
    print(f"  - Analysis: {out_dir / 'failure_analysis.json'}")
    print(f"  - Hard negatives: {out_dir / 'hard_negative_frames.txt'} ({len(failure_paths)} frames)")
    print(f"  - Report: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sprint 3 Failure Mining")
    parser.add_argument('--ghana_dir', type=str, default='data/raw/ghana',
                        help="Directory containing unlabelled Ghanaian frames")
    parser.add_argument('--model_path', type=str, default='checkpoints/detector_best.pth')
    parser.add_argument('--api_key', type=str, default=None,
                        help="API key (OpenRouter key or Gemini key depending on --backend)")
    parser.add_argument('--backend', type=str, default='openrouter',
                        choices=['openrouter', 'gemini'],
                        help="VLM backend to use (default: openrouter)")
    parser.add_argument('--model_name', type=str, default=None,
                        help="Model name override. "
                             "OpenRouter default: google/gemini-2.0-flash-exp:free  "
                             "Gemini default: gemini-2.0-flash")
    parser.add_argument('--output_dir', type=str, default='results/failure_mining/')
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--conf_thresh', type=float, default=0.3)
    parser.add_argument('--max_failures', type=int, default=50,
                        help="Max failure frames to analyze with VLM")
    args = parser.parse_args()
    mine_failures(args)
