"""
Training script for Sprint 2 Object Detection.
Trains the custom DetectionNet on bounding box data.
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.detection_model import DetectionNet
from src.data.detection_dataset import (
    DetectionDataset, get_detection_transforms, collate_fn, NUM_CLASSES
)
from src.metrics.detection_loss import DetectionLoss


def train_detector(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- Dataset ---
    train_transforms = get_detection_transforms(img_size=args.img_size, train=True)
    val_transforms = get_detection_transforms(img_size=args.img_size, train=False)

    full_dataset = DetectionDataset(args.data_dir, transforms=train_transforms, img_size=args.img_size)

    # Split 80/20
    n_train = int(0.8 * len(full_dataset))
    n_val = len(full_dataset) - n_train
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])

    # Override val transforms
    val_dataset.dataset = DetectionDataset(args.data_dir, transforms=val_transforms, img_size=args.img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn
    )

    # --- Model ---
    S = args.img_size // 32  # Grid size (416/32 = 13)
    model = DetectionNet(num_classes=NUM_CLASSES, S=S, B=args.B, img_size=args.img_size).to(device)
    print(f"Model grid: {S}x{S}, B={args.B}, Classes={NUM_CLASSES}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = DetectionLoss(S=S, B=args.B, C=NUM_CLASSES)

    # --- Checkpoint directory ---
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')

    print(f"\nStarting training for {args.epochs} epochs on {n_train} train, {n_val} val samples.\n")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
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

        # --- Validate ---
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

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), ckpt_dir / 'detector_best.pth')
            print(f"  → Saved best model (val_loss={best_val_loss:.4f})")

        scheduler.step()

    # Save final
    torch.save(model.state_dict(), ckpt_dir / 'detector_final.pth')
    print(f"\nTraining complete. Best Val Loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Sprint 2 Object Detector")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to dataset with images/ and labels/")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--img_size', type=int, default=416)
    parser.add_argument('--B', type=int, default=2, help="Boxes per grid cell")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    args = parser.parse_args()
    train_detector(args)
