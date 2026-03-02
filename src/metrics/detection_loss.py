"""
Custom Detection Loss for Sprint 2.
Manually implements the composite YOLO-style loss:
  1. Objectness Loss (BCE)
  2. Localisation Loss (MSE on box coordinates)
  3. Classification Loss (Cross-Entropy)

No library shortcuts — everything is explicit PyTorch tensor ops.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_iou(box1, box2):
    """
    Compute IoU between two sets of boxes in [cx, cy, w, h] format.
    Both inputs: (N, 4) tensors.
    Returns: (N,) tensor of IoU values.
    """
    # Convert to [x1, y1, x2, y2]
    b1_x1 = box1[:, 0] - box1[:, 2] / 2
    b1_y1 = box1[:, 1] - box1[:, 3] / 2
    b1_x2 = box1[:, 0] + box1[:, 2] / 2
    b1_y2 = box1[:, 1] + box1[:, 3] / 2

    b2_x1 = box2[:, 0] - box2[:, 2] / 2
    b2_y1 = box2[:, 1] - box2[:, 3] / 2
    b2_x2 = box2[:, 0] + box2[:, 2] / 2
    b2_y2 = box2[:, 1] + box2[:, 3] / 2

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area + b2_area - inter_area + 1e-6

    return inter_area / union


class DetectionLoss(nn.Module):
    """
    YOLOv1-inspired composite loss.

    For each image, we:
      1. Assign each ground truth box to the grid cell it falls into.
      2. Among the B predicted boxes in that cell, pick the one with highest IoU.
      3. Compute losses only for responsible predictors.

    Loss = λ_coord * localization_loss
         + objectness_loss (obj + noobj)
         + classification_loss
    """
    def __init__(self, S=13, B=2, C=8, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (batch, S, S, B*5 + C)
            targets: list of dicts with 'boxes' (N,4) [cx,cy,w,h] and 'labels' (N,)

        Returns:
            total_loss: scalar
        """
        batch_size = predictions.shape[0]
        S = self.S
        B = self.B
        C = self.C
        device = predictions.device

        coord_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        noobj_loss = torch.tensor(0.0, device=device)
        class_loss = torch.tensor(0.0, device=device)

        for b in range(batch_size):
            pred = predictions[b]  # (S, S, B*5+C)
            target = targets[b]
            gt_boxes = target['boxes'].to(device)     # (N, 4)
            gt_labels = target['labels'].to(device)    # (N,)

            # Track which cells are responsible (have a GT object)
            obj_mask = torch.zeros(S, S, device=device)

            # --- Assign GT boxes to grid cells ---
            for g in range(gt_boxes.shape[0]):
                cx, cy, w, h = gt_boxes[g]
                # Which cell?
                gi = int(torch.clamp(cy * S, 0, S - 1).item())
                gj = int(torch.clamp(cx * S, 0, S - 1).item())

                obj_mask[gi, gj] = 1.0

                # Find the best predictor box (highest IoU)
                best_iou = -1
                best_k = 0
                for k in range(B):
                    offset = k * 5
                    pred_x = (torch.sigmoid(pred[gi, gj, offset + 1]) + gj) / S
                    pred_y = (torch.sigmoid(pred[gi, gj, offset + 2]) + gi) / S
                    pred_w = torch.exp(pred[gi, gj, offset + 3]).clamp(max=1.0)
                    pred_h = torch.exp(pred[gi, gj, offset + 4]).clamp(max=1.0)

                    pred_box = torch.stack([pred_x, pred_y, pred_w, pred_h]).unsqueeze(0)
                    gt_box = gt_boxes[g].unsqueeze(0)
                    iou = compute_iou(pred_box, gt_box).item()

                    if iou > best_iou:
                        best_iou = iou
                        best_k = k

                # --- Localization Loss (on responsible predictor) ---
                offset = best_k * 5
                pred_x = (torch.sigmoid(pred[gi, gj, offset + 1]) + gj) / S
                pred_y = (torch.sigmoid(pred[gi, gj, offset + 2]) + gi) / S
                pred_w = torch.exp(pred[gi, gj, offset + 3]).clamp(max=1.0)
                pred_h = torch.exp(pred[gi, gj, offset + 4]).clamp(max=1.0)

                coord_loss += F.mse_loss(pred_x, cx) + F.mse_loss(pred_y, cy)
                coord_loss += F.mse_loss(torch.sqrt(pred_w + 1e-6), torch.sqrt(w + 1e-6))
                coord_loss += F.mse_loss(torch.sqrt(pred_h + 1e-6), torch.sqrt(h + 1e-6))

                # --- Objectness Loss (responsible predictor) ---
                pred_obj = pred[gi, gj, offset]
                obj_loss += F.binary_cross_entropy_with_logits(pred_obj, torch.tensor(1.0, device=device))

                # --- Classification Loss ---
                pred_class = pred[gi, gj, B * 5:]  # (C,)
                class_loss += F.cross_entropy(pred_class.unsqueeze(0), gt_labels[g].unsqueeze(0))

            # --- No-Object Loss (all non-responsible predictors) ---
            for i in range(S):
                for j in range(S):
                    if obj_mask[i, j] == 0:
                        for k in range(B):
                            pred_obj = pred[i, j, k * 5]
                            noobj_loss += F.binary_cross_entropy_with_logits(
                                pred_obj, torch.tensor(0.0, device=device)
                            )

        total = (self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * noobj_loss + class_loss) / batch_size

        return total, {
            'coord': coord_loss.item() / batch_size,
            'obj': obj_loss.item() / batch_size,
            'noobj': noobj_loss.item() / batch_size,
            'cls': class_loss.item() / batch_size,
        }
