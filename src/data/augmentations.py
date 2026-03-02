import albumentations as A
import numpy as np
import cv2

def get_baseline_transforms():
    """Minimal transforms for baseline Phase 2 training on clean dataset."""
    return A.Compose([
        A.Resize(720, 1280),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def get_ghana_augmentations():
    """
    Domain-inspired augmentations simulating Ghanaian road conditions.
    Rationale:
    - Shadow overlay -> Shadows misclassified as edges
    - Dust simulation (ColorJitter/Defocus) -> Haze over laterite/gravel
    - MotionBlur -> Camera vibration
    - MultiplicativeNoise -> Texture noise injection for dirt surfaces
    """
    return A.Compose([
        A.Resize(720, 1280),
        A.HorizontalFlip(p=0.5),
        
        # 1. Lighting & Contrast Variation
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        
        # 2. Shadow overlay
        A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.4),
        
        # 3. Dust / Haze simulation (Warm hue shift + blur)
        A.OneOf([
            A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.3, val_shift_limit=0.2, p=1.0),
            A.RGBShift(r_shift_limit=(0.0, 0.1), g_shift_limit=(0.0, 0.05), b_shift_limit=(-0.05, 0.0), p=1.0) # slightly warmer
        ], p=0.5),
        
        # 4. Camera vibration blur
        A.OneOf([
            A.MotionBlur(blur_limit=(3, 11), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0)
        ], p=0.5),
        
        # 5. Texture noise (dirt/gravel mimicking)
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), elementwise=True, p=0.4),
        
        # 6. Random Occlusions (like vendors, parked cars)
        A.CoarseDropout(max_holes=4, max_height=100, max_width=100, fill_value=0, p=0.3),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
