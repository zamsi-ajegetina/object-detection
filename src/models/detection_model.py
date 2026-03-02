"""
Custom Grid-Based Object Detection Model (Sprint 2).
Inspired by YOLOv1 theory but implemented entirely from scratch.

Architecture:
  - Backbone: Re-uses the U-Net encoder (DoubleConv + Down blocks)
  - Detection Head: Conv layers mapping to S x S grid
  - Each cell predicts: B bounding boxes, each with (obj, x, y, w, h) + C class probs
  - Output shape: (batch, S, S, B * 5 + C)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.models.segmentation_model import DoubleConv, Down


class DetectionNet(nn.Module):
    """
    Single-stage grid-based detector.

    Args:
        num_classes: Number of object classes (8 for PROSIT 2)
        S: Grid size (SxS cells)
        B: Number of bounding boxes per cell
        img_size: Input image size (must be divisible by 32)
    """
    def __init__(self, num_classes=8, S=13, B=2, img_size=416):
        super().__init__()
        self.num_classes = num_classes
        self.S = S
        self.B = B
        self.img_size = img_size

        # --- Backbone (U-Net Encoder) ---
        # Produces feature maps at 1/32 resolution
        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)       # /2
        self.down2 = Down(128, 256)      # /4
        self.down3 = Down(256, 512)      # /8
        self.down4 = Down(512, 1024)     # /16
        self.down5 = Down(1024, 1024)    # /32 → 416/32 = 13x13

        # --- Detection Head ---
        # Takes the 1024-channel feature map and predicts per-cell outputs
        self.det_conv1 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.det_conv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Final 1x1 conv: output channels = B * 5 + C
        # Each box: (objectness, x, y, w, h) = 5 values
        # Plus C class probabilities shared per cell
        out_channels = B * 5 + num_classes
        self.det_out = nn.Conv2d(256, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Input images (B, 3, H, W)
        Returns:
            predictions: (B, S, S, B*5 + C)
        """
        # Backbone forward
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)

        # Adaptive pool to guarantee S x S spatial dims
        x = F.adaptive_avg_pool2d(x, (self.S, self.S))

        # Detection head
        x = self.det_conv1(x)
        x = self.det_conv2(x)
        x = self.det_out(x)  # (batch, B*5+C, S, S)

        # Reshape to (batch, S, S, B*5+C)
        x = x.permute(0, 2, 3, 1).contiguous()

        return x

    def decode_predictions(self, predictions, conf_thresh=0.5):
        """
        Decode raw grid predictions into bounding boxes.

        Args:
            predictions: (batch, S, S, B*5 + C)
            conf_thresh: Minimum objectness score

        Returns:
            List of (boxes, scores, class_ids) per image
            boxes are in [cx, cy, w, h] format, normalized 0-1
        """
        batch_size = predictions.shape[0]
        S = self.S
        B = self.B
        C = self.num_classes

        results = []

        for b in range(batch_size):
            pred = predictions[b]  # (S, S, B*5+C)
            all_boxes = []
            all_scores = []
            all_classes = []

            for i in range(S):
                for j in range(S):
                    cell = pred[i, j]

                    # Class probabilities (shared per cell)
                    class_probs = torch.softmax(cell[B * 5:], dim=0)

                    for k in range(B):
                        offset = k * 5
                        obj = torch.sigmoid(cell[offset])
                        x = (torch.sigmoid(cell[offset + 1]) + j) / S
                        y = (torch.sigmoid(cell[offset + 2]) + i) / S
                        w = torch.exp(cell[offset + 3]).clamp(max=1.0)
                        h = torch.exp(cell[offset + 4]).clamp(max=1.0)

                        # Combined confidence = objectness * class_prob
                        max_class_prob, class_id = class_probs.max(0)
                        score = obj * max_class_prob

                        if score > conf_thresh:
                            all_boxes.append([x.item(), y.item(), w.item(), h.item()])
                            all_scores.append(score.item())
                            all_classes.append(class_id.item())

            if len(all_boxes) > 0:
                results.append((
                    torch.tensor(all_boxes),
                    torch.tensor(all_scores),
                    torch.tensor(all_classes)
                ))
            else:
                results.append((
                    torch.zeros((0, 4)),
                    torch.zeros((0,)),
                    torch.zeros((0,), dtype=torch.long)
                ))

        return results
