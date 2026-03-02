import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import torchvision.transforms as T

# Add project root to python path to allow direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.segmentation_model import UNet
from src.data.dataset import RoadSegDataset

def calculate_iou(preds, labels, num_classes=2):
    """Calculates Intersection over Union per class, ignoring background if needed."""
    ious = []
    # Class 1 is road
    pred_inds = preds == 1
    target_inds = labels == 1
    intersection = (pred_inds[target_inds]).sum().item()
    union = pred_inds.sum().item() + target_inds.sum().item() - intersection
    
    if union == 0:
        return float('nan')
    else:
        return float(intersection) / float(max(union, 1))

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    model = UNet(n_channels=3, n_classes=1).to(device)
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Loaded weights from {args.model_path}")
    else:
        print(f"Warning: Model weights {args.model_path} not found! Evaluating with random weights.")

    model.eval()

    # Load dataset
    road_color = [128, 64, 128] if args.dataset_type == 'camvid' else None
    dataset = RoadSegDataset(args.data_dir, split=args.split, mask_dir=args.mask_dir, road_color=road_color)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ious = []

    print(f"Evaluating on {len(dataset)} samples from {args.data_dir}")
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(loader)):
            images = images.to(device)
            # targets are B, 1, H, W
            
            logits = model(images)
            preds = (torch.sigmoid(logits) > 0.5).long()
            
            # Move to CPU for metrics and saving
            preds_cpu = preds.cpu().squeeze().numpy()
            masks_cpu = masks.cpu().squeeze().numpy()
            img_cpu = images.cpu().squeeze().permute(1, 2, 0).numpy()
            
            # calculate IoU
            iou = calculate_iou(preds_cpu, masks_cpu)
            if not np.isnan(iou):
                ious.append(iou)

            # Save qualitative visualisations for the first N samples
            if i < args.num_save:
                # Denormalize image if necessary (standard ToTensor is just 0-1)
                img_bgr = cv2.cvtColor((img_cpu * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                
                # Create an overlay (Green for predicted road, Red for ground truth)
                overlay = img_bgr.copy()
                
                # Add green where predicted road
                green_mask = np.zeros_like(overlay)
                green_mask[preds_cpu == 1] = [0, 255, 0]
                
                # Add red where ground truth road
                red_mask = np.zeros_like(overlay)
                red_mask[masks_cpu == 1] = [0, 0, 255]

                # Blend
                alpha = 0.5
                overlay = cv2.addWeighted(overlay, 1.0, green_mask, alpha, 0)
                
                out_path = out_dir / f"pred_{i:04d}_iou_{iou:.3f}.jpg"
                cv2.imwrite(str(out_path), overlay)

    mIoU = np.nanmean(ious) * 100 if ious else 0.0
    print(f"\n--- Evaluation Results ---")
    print(f"Mean IoU (Road Class): {mIoU:.2f}%")
    print(f"Saved {min(args.num_save, len(dataset))} visualisations to {out_dir}")
    
    with open(out_dir / "metrics.txt", "w") as f:
        f.write(f"Mean IoU: {mIoU:.2f}%\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path to best baseline.pth")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to evaluate (e.g. data/raw/ghana/highway)")
    parser.add_argument('--mask_dir', type=str, default=None, help="Explicit path to masks if different from data_dir")
    parser.add_argument('--split', type=str, default='', help="Split to evaluate (leave empty if evaluating flat ghana directory)")
    parser.add_argument('--dataset_type', type=str, default='binary', choices=['camvid', 'binary'])
    parser.add_argument('--output_dir', type=str, default='results/domain_shift/')
    parser.add_argument('--num_save', type=int, default=20, help="Number of qualitative visualisations to save")
    
    args = parser.parse_args()
    evaluate(args)
