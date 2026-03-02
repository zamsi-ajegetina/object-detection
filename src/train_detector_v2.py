"""
Retraining script for Sprint 3.
Combines the original IDD dataset with VLM-identified hard negative frames
and applies targeted augmentations based on identified failure modes.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.detection_model import DetectionNet
from src.data.detection_dataset import (
    DetectionDataset, get_detection_transforms, collate_fn, NUM_CLASSES
)
from src.metrics.detection_loss import DetectionLoss

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_hardened_transforms(img_size=416, failure_modes=None):
    """
    Augmentations targeting the specific failure modes identified by the VLM.
    Applies MORE AGGRESSIVE versions of augmentations that match failure modes.
    """
    transforms_list = [
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
    ]

    # Base augmentations
    transforms_list.append(
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6)
    )

    # Targeted augmentations based on failure modes
    if failure_modes:
        if 'dust_haze' in failure_modes or 'low_contrast' in failure_modes:
            transforms_list.extend([
                A.RandomFog(fog_coef_range=(0.1, 0.3), p=0.4),
                A.CLAHE(clip_limit=4.0, p=0.3),
            ])

        if 'shadow' in failure_modes:
            transforms_list.extend([
                A.RandomShadow(num_shadows_limit=(1, 3), shadow_dimension=5, p=0.5),
                A.RandomGamma(gamma_limit=(60, 140), p=0.3),
            ])

        if 'motion_blur' in failure_modes:
            transforms_list.append(
                A.MotionBlur(blur_limit=7, p=0.4)
            )

        if 'crowded_scene' in failure_modes:
            transforms_list.append(
                A.RandomScale(scale_limit=0.2, p=0.3)
            )
    else:
        # Default aggressive augmentations if no specific modes identified
        transforms_list.extend([
            A.RandomFog(fog_coef_range=(0.05, 0.2), p=0.3),
            A.RandomShadow(num_shadows_limit=(1, 2), shadow_dimension=5, p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
        ])

    transforms_list.extend([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3)
    )


def load_failure_modes(failure_analysis_path):
    """Load failure modes from the VLM failure analysis JSON."""
    path = Path(failure_analysis_path)
    if not path.exists():
        return []

    with open(path) as f:
        results = json.load(f)

    modes = []
    for r in results:
        analysis = r.get('vlm_analysis', {})
        if isinstance(analysis, dict):
            mode = analysis.get('dominant_failure_mode', '')
            if mode and mode != 'unknown':
                modes.append(mode)

    return list(set(modes))


def train_detector_v2(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Load failure modes from VLM analysis ---
    failure_modes = load_failure_modes(args.failure_analysis)
    if failure_modes:
        print(f"Targeting failure modes: {failure_modes}")
    else:
        print("No specific failure modes found, using default aggressive augmentations")

    # --- Datasets ---
    hardened_transforms = get_hardened_transforms(img_size=args.img_size, failure_modes=failure_modes)
    val_transforms = get_detection_transforms(img_size=args.img_size, train=False)

    # Original IDD training data
    idd_dataset = DetectionDataset(args.data_dir, transforms=hardened_transforms, img_size=args.img_size)

    # Hard negative frames (if available with labels)
    combined_dataset = idd_dataset
    if args.hard_neg_dir and Path(args.hard_neg_dir).exists():
        hard_neg_dataset = DetectionDataset(args.hard_neg_dir, transforms=hardened_transforms, img_size=args.img_size)
        if len(hard_neg_dataset) > 0:
            print(f"Adding {len(hard_neg_dataset)} hard negative samples")
            combined_dataset = ConcatDataset([idd_dataset, hard_neg_dataset])

    # Split
    n_train = int(0.85 * len(combined_dataset))
    n_val = len(combined_dataset) - n_train
    train_dataset, val_dataset = random_split(combined_dataset, [n_train, n_val])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    # --- Model (optionally warm-start from Sprint 2 weights) ---
    S = args.img_size // 32
    model = DetectionNet(num_classes=NUM_CLASSES, S=S, B=args.B, img_size=args.img_size).to(device)

    if args.pretrained and Path(args.pretrained).exists():
        model.load_state_dict(torch.load(args.pretrained, map_location=device))
        print(f"Warm-starting from Sprint 2 weights: {args.pretrained}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DetectionLoss(S=S, B=args.B, C=NUM_CLASSES)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    print(f"\nStarting retraining for {args.epochs} epochs on {n_train} train, {n_val} val samples.\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")

        for images, targets in pbar:
            images = images.to(device)
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'coord': f"{loss_dict['coord']:.3f}",
                'cls': f"{loss_dict['cls']:.3f}"
            })

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                images = images.to(device)
                predictions = model(images)
                loss, _ = criterion(predictions, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)
        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_dir / 'detector_v2_best.pth')
            print(f"  → Saved best v2 model (val_loss={best_val_loss:.4f})")

        scheduler.step()

    torch.save(model.state_dict(), ckpt_dir / 'detector_v2_final.pth')
    print(f"\nRetraining complete. Best Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sprint 3 Detector Retraining")
    parser.add_argument('--data_dir', type=str, required=True, help="IDD YOLO dataset path")
    parser.add_argument('--hard_neg_dir', type=str, default=None,
                        help="Directory with hard negative images+labels")
    parser.add_argument('--failure_analysis', type=str, default='results/failure_mining/failure_analysis.json',
                        help="Path to VLM failure analysis JSON")
    parser.add_argument('--pretrained', type=str, default='checkpoints/detector_best.pth',
                        help="Sprint 2 weights to warm-start from")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--B', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    args = parser.parse_args()
    train_detector_v2(args)
