import os
import cv2
import numpy as np
import argparse
from pathlib import Path

def generate_classical_masks(input_dir, output_dir):
    """
    Generates pseudo-masks for road segmentation using classical CV techniques.
    Assumes dashboard camera viewpoint.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Process all images recursively
    image_paths = list(input_dir.rglob("*.jpg"))
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Auto-annotating...")
    
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        
        # 1. Define ROI (bottom 60% of image, excluding extreme bottom for ego-hood)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        top_cutoff = int(h * 0.4)
        bottom_cutoff = int(h * 0.95)
        # Trapezoid ROI roughly corresponding to the road ahead
        pts = np.array([
            [int(w * 0.3), top_cutoff],
            [int(w * 0.7), top_cutoff],
            [w, bottom_cutoff],
            [0, bottom_cutoff]
        ], np.int32)
        cv2.fillPoly(roi_mask, [pts], 255)

        # 2. Convert to HSV for robust color thresholding
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Road is typically neutral (low saturation)
        # We also want to avoid extremely dark (shadows) and extremely bright (sky reflections)
        # These bounds can be tuned for Ghanaian roads (laterite might have higher saturation, 
        # but asphalt/dirt usually falls in these ranges)
        lower_bound = np.array([0, 0, 40])
        upper_bound = np.array([179, 70, 200]) # Max saturation 70, ignoring very bright/dark
        
        road_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Apply ROI
        road_mask = cv2.bitwise_and(road_mask, road_mask, mask=roi_mask)
        
        # 3. Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        # Remove small noise (e.g. lane markers, small defects)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)
        # Fill holes (e.g. potholes, shadows within the road)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
        
        # Optional: Keep only the largest connected component within ROI
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(road_mask, connectivity=8)
        # stats: [x, y, w, h, area]
        
        final_mask = np.zeros_like(road_mask)
        if num_labels > 1:
            # exclude background (index 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            final_mask[labels == largest_label] = 255
        
        # Save structural output replicating the input directory structure
        rel_path = img_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(out_path), final_mask)
    
    print(f"Finished generating {len(image_paths)} masks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-annotate road frames using Classical CV")
    parser.add_argument("--input_dir", type=str, default="data/raw/ghana", help="Input extracted frames")
    parser.add_argument("--output_dir", type=str, default="data/annotations/classical", help="Output directory for masks")
    args = parser.parse_args()

    generate_classical_masks(args.input_dir, args.output_dir)
