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

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        return 1 - dice

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up datasets
    # For CamVid, road color is [128, 64, 128]
    road_color = [128, 64, 128] if args.dataset_type == 'camvid' else None
    
    # Ideally, we'd add albumentations here for data augmentation (Phase 4).
    # Since this is Baseline (Phase 2), we use minimal standard transforms.
    train_dataset = RoadSegDataset(args.data_dir, split='train', road_color=road_color)
    val_dataset = RoadSegDataset(args.data_dir, split='val', road_color=road_color)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Initialize model
    model = UNet(n_channels=3, n_classes=1).to(device)

    # Losses & Optimizer
    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    best_iou = 0.0

    print(f"Starting training for {args.epochs} epochs on {len(train_dataset)} train samples, {len(val_dataset)} val samples.")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            
            # Loss is combo of BCE and Dice
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
                masks = masks.to(device)

                logits = model(images)
                
                loss_bce = criterion_bce(logits, masks)
                loss_dice = criterion_dice(logits, masks)
                loss = loss_bce + loss_dice
                val_loss += loss.item()

                # Calculate IoU
                preds = (torch.sigmoid(logits) > 0.5).float()
                intersection = (preds * masks).sum().item()
                union = preds.sum().item() + masks.sum().item() - intersection
                
                intersection_sum += intersection
                union_sum += union

        val_iou = (intersection_sum + 1e-6) / (union_sum + 1e-6)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val IoU: {val_iou:.4f}")

        # Sav checkpoint
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, 'baseline_best.pth'))
            print("  -> Saved new best model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Path to clean dataset (e.g., CamVid root containing train/ and val/ dirs)")
    parser.add_argument('--dataset_type', type=str, default='camvid', choices=['camvid', 'binary'], help="Type of masks")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/')
    
    args = parser.parse_args()
    train(args)
