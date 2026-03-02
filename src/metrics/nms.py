"""
Manual Non-Maximum Suppression (NMS) for Sprint 2.
No torchvision.ops.nms — implemented from scratch.
"""
import torch


def compute_iou_matrix(boxes1, boxes2):
    """
    Compute pairwise IoU between two sets of boxes.
    Boxes format: [cx, cy, w, h] (normalized).

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)
    Returns:
        iou_matrix: (N, M)
    """
    # Convert to corners [x1, y1, x2, y2]
    b1_x1 = boxes1[:, 0:1] - boxes1[:, 2:3] / 2
    b1_y1 = boxes1[:, 1:2] - boxes1[:, 3:4] / 2
    b1_x2 = boxes1[:, 0:1] + boxes1[:, 2:3] / 2
    b1_y2 = boxes1[:, 1:2] + boxes1[:, 3:4] / 2

    b2_x1 = boxes2[:, 0:1] - boxes2[:, 2:3] / 2
    b2_y1 = boxes2[:, 1:2] - boxes2[:, 3:4] / 2
    b2_x2 = boxes2[:, 0:1] + boxes2[:, 2:3] / 2
    b2_y2 = boxes2[:, 1:2] + boxes2[:, 3:4] / 2

    # Intersection
    inter_x1 = torch.max(b1_x1, b2_x1.T)
    inter_y1 = torch.max(b1_y1, b2_y1.T)
    inter_x2 = torch.min(b1_x2, b2_x2.T)
    inter_y2 = torch.min(b1_y2, b2_y2.T)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Union
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union = b1_area + b2_area.T - inter_area + 1e-6

    return inter_area / union


def non_max_suppression(boxes, scores, class_ids, iou_threshold=0.5):
    """
    Perform class-aware NMS.

    Args:
        boxes: (N, 4) tensor [cx, cy, w, h]
        scores: (N,) tensor of confidence scores
        class_ids: (N,) tensor of class indices
        iou_threshold: IoU threshold for suppression

    Returns:
        kept_indices: list of indices to keep
    """
    if boxes.numel() == 0:
        return []

    # Sort by score descending
    order = torch.argsort(scores, descending=True)
    kept = []

    while order.numel() > 0:
        # Pick the highest score
        i = order[0].item()
        kept.append(i)

        if order.numel() == 1:
            break

        # Compute IoU of current box against remaining
        remaining = order[1:]
        ious = compute_iou_matrix(
            boxes[i].unsqueeze(0),
            boxes[remaining]
        ).squeeze(0)

        # Only suppress boxes of the SAME class
        same_class = class_ids[remaining] == class_ids[i]
        suppress = (ious > iou_threshold) & same_class

        # Keep boxes that are NOT suppressed
        mask = ~suppress
        order = remaining[mask]

    return kept
