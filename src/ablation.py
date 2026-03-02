import os
import argparse
import torch
import sys
from pathlib import Path

# Add project root to python path to allow direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

# In a real run, this script would programmatically run evaluation twice
# and parse the metrics into a table. Since evaluation logic already exists
# in evaluate.py, we'll wrap it here.
from src.evaluate import evaluate as run_eval

class DummyArgs:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def run_ablation(args):
    """
    Runs the baseline and augmented models against the Ghanaian evaluation dataset
    and prints a clear comparison table.
    """
    print("=== Sprint 1 Ablation Study: Visual Domain Shift ===")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Evaluate Baseline Model
    baseline_args = DummyArgs(
        model_path=args.baseline_model,
        data_dir=args.ghana_data_dir,
        mask_dir=args.ghana_mask_dir,
        split='',
        dataset_type='binary',
        output_dir=str(out_dir / 'baseline'),
        num_save=args.num_save
    )
    print("\n[Experiment A] Evaluating Baseline Model...")
    try:
        run_eval(baseline_args)
        with open(out_dir / 'baseline' / 'metrics.txt', 'r') as f:
            base_iou_str = f.read().strip().split(': ')[1]
            base_iou = float(base_iou_str.replace('%', ''))
    except Exception as e:
        print(f"Error evaluating baseline: {e}")
        base_iou = 0.0

    # 2. Evaluate Augmented Model
    augmented_args = DummyArgs(
        model_path=args.augmented_model,
        data_dir=args.ghana_data_dir,
        mask_dir=args.ghana_mask_dir,
        split='',
        dataset_type='binary',
        output_dir=str(out_dir / 'augmented'),
        num_save=args.num_save
    )
    print("\n[Experiment B] Evaluating Domain-Augmented Model...")
    try:
        run_eval(augmented_args)
        with open(out_dir / 'augmented' / 'metrics.txt', 'r') as f:
            aug_iou_str = f.read().strip().split(': ')[1]
            aug_iou = float(aug_iou_str.replace('%', ''))
    except Exception as e:
        print(f"Error evaluating augmented model: {e}")
        aug_iou = 0.0
        
    # Print final comparison table
    print("\n" + "="*50)
    print("                  FINAL ABLATION RESULTS                 ")
    print("="*50)
    print(f"| {'Model':<20} | {'Ghana mIoU':<10} |")
    print("-" * 50)
    print(f"| {'Baseline (A)':<20} | {base_iou:>6.2f}%    |")
    print(f"| {'Augmented (B)':<20} | {aug_iou:>6.2f}%    |")
    print("="*50)
    
    diff = aug_iou - base_iou
    if diff > 0:
        print(f"Conclusion: Domain augmentations improved performance by +{diff:.2f}% mIoU.")
    else:
        print(f"Conclusion: Domain augmentations did not improve performance (Diff: {diff:.2f}% mIoU).")
        
    with open(out_dir / "summary.txt", "w") as f:
        f.write("Ablation Results\n")
        f.write("-" * 20 + "\n")
        f.write(f"Baseline (A): {base_iou:.2f}%\n")
        f.write(f"Augmented (B): {aug_iou:.2f}%\n")
        f.write(f"Difference: {diff:+.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete Sprint 1 Ablation Study")
    parser.add_argument('--baseline_model', type=str, default='checkpoints/baseline_best.pth')
    parser.add_argument('--augmented_model', type=str, default='checkpoints/augmented_best.pth')
    parser.add_argument('--ghana_data_dir', type=str, required=True, help="Path to Ghana frames")
    parser.add_argument('--ghana_mask_dir', type=str, default='data/annotations/classical', help="Path to Ghana pseudo-masks")
    parser.add_argument('--output_dir', type=str, default='results/ablation/')
    parser.add_argument('--num_save', type=int, default=10, help="Vis saves per model")
    
    args = parser.parse_args()
    run_ablation(args)
