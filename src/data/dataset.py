import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class RoadSegDataset(Dataset):
    """
    Generic PyTorch Dataset for Road Segmentation (Binary: Road vs Non-Road).
    Expects a directory structure with images/ and masks/ subdirectories.
    Masks can be either:
      - Binary grayscale (0 for bg, 1 or 255 for road).
      - RGB masks where road is a specific color (e.g. CamVid road is [128, 64, 128]).
    """
    def __init__(self, data_dir, split='train', mask_dir=None, transforms=None, road_color=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transforms = transforms
        self.road_color = list(road_color) if road_color else None
        
        # If mask_dir is explicitly provided (e.g. for Ghana eval where masks are in annotations/)
        if mask_dir is not None:
            self.image_dir = self.data_dir if not split else self.data_dir / split
            self.mask_dir = Path(mask_dir) if not split else Path(mask_dir) / split
            # Use rglob to search inside subdirectories (highway, market, etc.)
            self.images = sorted([p for p in self.image_dir.rglob('*.*') if p.is_file()])
            self.masks = sorted([p for p in self.mask_dir.rglob('*.*') if p.is_file()])
        else:
            # Existing CamVid logic
            nested_img_dir = self.data_dir / split / 'images'
            flat_img_dir = self.data_dir / split
            
            if nested_img_dir.exists():
                self.image_dir = nested_img_dir
                self.mask_dir = self.data_dir / split / 'masks'
            else:
                self.image_dir = flat_img_dir
                self.mask_dir = self.data_dir / f"{split}_labels"
                
            self.images = sorted([p for p in self.image_dir.glob('*.*') if p.is_file()])
            self.masks = sorted([p for p in self.mask_dir.glob('*.*') if p.is_file()])
        
        # Basic validation
        assert len(self.images) == len(self.masks), f"Mismatch: {len(self.images)} images vs {len(self.masks)} masks."

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image (RGB)
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load mask
        mask_path = self.masks[idx]
        if self.road_color:
            # RGB mask mapping. e.g., CamVid road color RGB = (128, 64, 128)
            mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
            mask = np.all(mask_rgb == self.road_color, axis=-1).astype(np.float32)
            mask = Image.fromarray(mask)
        else:
            # L mode grayscale mask. Assume > 0 is road.
            mask_gray = np.array(Image.open(mask_path).convert("L"))
            mask = (mask_gray > 0).astype(np.float32)
            mask = Image.fromarray(mask)

        # Apply transforms if provided
        if self.transforms is not None:
            # Note: For segmentation, typically albumentations is used because it handles 
            # images and masks simultaneously. If we use torchvision, we must apply the same
            # random transformations (e.g., cropping, flipping) to both.
            # Here we assume `transforms` is an albumentations Compose object.
            augmented = self.transforms(image=np.array(image), mask=np.array(mask))
            img_arr = augmented['image']
            mask_arr = augmented['mask']
            
            # Albumentations returns HWC numpy arrays. PyTorch expects CHW tensors.
            if isinstance(img_arr, np.ndarray):
                image = torch.from_numpy(img_arr.transpose(2, 0, 1)).float()
            else:
                image = img_arr
                
            if isinstance(mask_arr, np.ndarray):
                mask = torch.from_numpy(mask_arr).unsqueeze(0).float()
            else:
                mask = mask_arr
        else:
            # Fallback basic transforms
            image = T.ToTensor()(image)
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float()

        return image, mask
