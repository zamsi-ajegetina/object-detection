import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import sys
from pathlib import Path

# Add project root to python path to allow direct execution
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.models.segmentation_model import UNet
from src.data.dataset import RoadSegDataset
from src.data.augmentations import get_ghana_augmentations, get_baseline_transforms
from src.train import DiceLoss

def train_augmented(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up datasets
    road_color = [128, 64, 128] if args.dataset_type == 'camvid' else None
    
    # Use Domain-Inspired Augmentations for training
    train_transforms = get_ghana_augmentations()
    # Use baseline minimal transforms for validation
    val_transforms = get_baseline_transforms()

    train_dataset = RoadSegDataset(args.data_dir, split='train', road_color=road_color, transforms=train_transforms)
    val_dataset = RoadSegDataset(args.data_dir, split='val', road_color=road_color, transforms=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = UNet(n_channels=3, n_classes=1).to(device)

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_iou = 0.0

    print(f"Starting AUGMENTED training for {args.epochs} epochs on {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            # albumentations might add a channel dim or keep it 2D. Ensure correct shape.
            if len(masks.shape) == 3:
                masks = masks.unsqueeze(1)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            
            loss_bce = criterion_bce(logits, masks)
            loss_dice = criterion_dice(logits, masks)
            loss = loss_bce + loss_dice
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # Validation
        model.eval()
        val_loss = 0.0
        intersection_sum = 0
        union_sum = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val  ]")
        with torch.no_grad():
            for images, masks in pbar_val:
                images = images.to(device)
                if len(masks.shape) == 3:
                    masks = masks.unsqueeze(1)
                masks = masks.to(device)

                logits = model(images)
                
                loss_bce = criterion_bce(logits, masks)
                loss_dice = criterion_dice(logits, masks)
                loss = loss_bce + loss_dice
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = (preds * masks).sum().item()
                union = preds.sum().item() + masks.sum().item() - intersection
                
                intersection_sum += intersection
                union_sum += union

        val_iou = (intersection_sum + 1e-6) / (union_sum + 1e-6)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            # Save separately from baseline
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'augmented_best.pth'))
            print("  -> Saved new best augmented model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to clean dataset (e.g., CamVid)")
    parser.add_argument('--dataset_type', type=str, default='camvid', choices=['camvid', 'binary'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    
    args = parser.parse_args()
    train_augmented(args)
