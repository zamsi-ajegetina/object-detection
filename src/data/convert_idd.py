"""
Convert IDD Pascal VOC XML annotations to YOLO TXT format
and copy images into the expected directory structure.

IDD classes → PROSIT 2 class mapping:
  0: pothole          (no IDD equivalent — will be empty)
  1: speed_bump       (no IDD equivalent — will be empty)
  2: pedestrian       ← person, rider
  3: motorcycle       ← motorcycle, bicycle
  4: car              ← car, truck, bus, autorickshaw, vehicle fallback, caravan
  5: traffic_cone     ← traffic sign, traffic light
  6: animal           ← animal
  7: open_drain       (no IDD equivalent — will be empty)
"""
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
from tqdm import tqdm

# IDD class name → PROSIT class ID
IDD_CLASS_MAP = {
    'person': 2,
    'rider': 2,
    'motorcycle': 3,
    'bicycle': 3,
    'car': 4,
    'truck': 4,
    'bus': 4,
    'autorickshaw': 4,
    'vehicle fallback': 4,
    'caravan': 4,
    'traffic sign': 5,
    'traffic light': 5,
    'animal': 6,
}


def convert_voc_to_yolo(xml_path, img_width, img_height):
    """Parse a Pascal VOC XML file and return YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_lines = []

    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        if name not in IDD_CLASS_MAP:
            continue

        class_id = IDD_CLASS_MAP[name]
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        # Ensure correct order
        xmin, xmax = min(xmin, xmax), max(xmin, xmax)
        ymin, ymax = min(ymin, ymax), max(ymin, ymax)

        # Convert to YOLO format (cx, cy, w, h) normalized
        cx = ((xmin + xmax) / 2.0) / img_width
        cy = ((ymin + ymax) / 2.0) / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height

        # Clamp
        cx = max(0.001, min(0.999, cx))
        cy = max(0.001, min(0.999, cy))
        w = max(0.001, min(0.999, w))
        h = max(0.001, min(0.999, h))

        yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return yolo_lines


def convert_split(idd_dir, output_dir, split):
    """Convert one split (train/val/test) of IDD to YOLO format."""
    src_img_dir = Path(idd_dir) / split / 'images'
    src_ann_dir = Path(idd_dir) / split / 'annotations'

    out_img_dir = Path(output_dir) / split / 'images'
    out_lbl_dir = Path(output_dir) / split / 'labels'
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    if not src_ann_dir.exists():
        print(f"  Skipping {split}: no annotations directory found.")
        return 0

    xml_files = sorted(src_ann_dir.glob('*.xml'))
    converted = 0
    skipped = 0

    for xml_path in tqdm(xml_files, desc=f"  Converting {split}"):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get image dimensions from XML
        size = root.find('size')
        if size is None:
            skipped += 1
            continue
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)

        # Convert annotations
        yolo_lines = convert_voc_to_yolo(xml_path, img_width, img_height)

        if len(yolo_lines) == 0:
            skipped += 1
            continue

        # Get corresponding image
        img_stem = xml_path.stem
        img_src = None
        for ext in ['.jpg', '.jpeg', '.png']:
            candidate = src_img_dir / (img_stem + ext)
            if candidate.exists():
                img_src = candidate
                break

        if img_src is None:
            skipped += 1
            continue

        # Copy image
        shutil.copy2(img_src, out_img_dir / img_src.name)

        # Write YOLO label
        with open(out_lbl_dir / (img_stem + '.txt'), 'w') as f:
            f.write('\n'.join(yolo_lines) + '\n')

        converted += 1

    print(f"  {split}: Converted {converted} images, skipped {skipped}")
    return converted


def main():
    parser = argparse.ArgumentParser(description="Convert IDD VOC to YOLO format")
    parser.add_argument('--idd_dir', type=str, default=os.path.expanduser('~/Downloads/IDD_Detection_Organized'),
                        help="Path to IDD_Detection_Organized/")
    parser.add_argument('--output_dir', type=str, default='data/idd_yolo',
                        help="Output directory for YOLO-format dataset")
    args = parser.parse_args()

    print(f"Converting IDD dataset from: {args.idd_dir}")
    print(f"Output directory: {args.output_dir}\n")

    total = 0
    for split in ['train', 'val', 'test']:
        total += convert_split(args.idd_dir, args.output_dir, split)

    print(f"\n✅ Done! Converted {total} total images to YOLO format.")
    print(f"\nTo train:")
    print(f"  python src/train_detector.py --data_dir {args.output_dir}/train --batch_size 4")


if __name__ == '__main__':
    main()
