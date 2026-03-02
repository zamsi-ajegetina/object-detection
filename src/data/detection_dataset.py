"""
Detection Dataset for Sprint 2.
Supports YOLO-format TXT annotations (one .txt per image).
Each line in a label file: class_id cx cy w h (normalized 0-1).

Also handles IDD → PROSIT class mapping.
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# PROSIT 2 class definitions (8 classes)
CLASS_NAMES = [
    'pothole',          # 0
    'speed_bump',       # 1
    'pedestrian',       # 2
    'motorcycle',       # 3
    'car',              # 4
    'traffic_cone',     # 5
    'animal',           # 6
    'open_drain',       # 7
]
NUM_CLASSES = len(CLASS_NAMES)

# IDD class name → PROSIT class ID mapping
# IDD has many classes; we map the relevant ones to our 8
IDD_TO_PROSIT = {
    'person': 2,
    'rider': 2,
    'motorcycle': 3,
    'bicycle': 3,
    'autorickshaw': 4,
    'car': 4,
    'truck': 4,
    'bus': 4,
    'vehicle fallback': 4,
    'animal': 6,
}


def get_detection_transforms(img_size=416, train=True):
    """Albumentations transforms that also handle bounding boxes."""
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3))


class DetectionDataset(Dataset):
    """
    PyTorch Dataset for object detection with YOLO-format labels.

    Directory structure:
        data_dir/
            images/
                img001.jpg
                ...
            labels/
                img001.txt
                ...
    Each label .txt has lines: class_id center_x center_y width height (normalized).
    """
    def __init__(self, data_dir, transforms=None, img_size=416):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / 'images'
        self.label_dir = self.data_dir / 'labels'
        self.img_size = img_size
        self.transforms = transforms

        # Only keep images that have a corresponding label file
        all_images = sorted(self.img_dir.glob('*.*'))
        self.samples = []
        for img_path in all_images:
            label_path = self.label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                self.samples.append((img_path, label_path))

        print(f"DetectionDataset: Found {len(self.samples)} image-label pairs in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        image = np.array(Image.open(img_path).convert('RGB'))

        # Parse YOLO-format labels
        boxes = []
        class_labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = [float(x) for x in parts[1:]]
                    # Clamp to strict validity bounds for Albumentations
                    # It requires x_min >= 0, y_min >= 0, x_max <= 1, y_max <= 1
                    # In YOLO format: x_min = cx - w/2
                    xmin = max(0.0, cx - w / 2.0)
                    ymin = max(0.0, cy - h / 2.0)
                    xmax = min(1.0, cx + w / 2.0)
                    ymax = min(1.0, cy + h / 2.0)
                    
                    # Prevent zero-area boxes after clamping
                    if xmax > xmin and ymax > ymin:
                        # Recompute YOLO format safely clamped
                        cx = (xmin + xmax) / 2.0
                        cy = (ymin + ymax) / 2.0
                        w = xmax - xmin
                        h = ymax - ymin
                        
                        # Add tiny epsilon to strictly avoid 0.0 or 1.0 boundary errors in Albumentations
                        cx = max(1e-5, min(1.0 - 1e-5, cx))
                        cy = max(1e-5, min(1.0 - 1e-5, cy))
                        w = max(1e-5, min(1.0 - 1e-5, w))
                        h = max(1e-5, min(1.0 - 1e-5, h))
                    if cls_id < NUM_CLASSES:
                        boxes.append([cx, cy, w, h])
                        class_labels.append(cls_id)

        # Apply transforms
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                bboxes=boxes,
                class_labels=class_labels
            )
            image = transformed['image']
            boxes = transformed['bboxes']
            class_labels = transformed['class_labels']
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            class_labels = torch.tensor(class_labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_labels = torch.zeros((0,), dtype=torch.long)

        target = {
            'boxes': boxes,
            'labels': class_labels,
        }

        return image, target


def collate_fn(batch):
    """Custom collate function for variable-length bounding boxes."""
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets
