"""
Convert COCO format annotations to YOLO segmentation format.
"""

import json
import os
from pathlib import Path
from tqdm import tqdm


def convert_coco_to_yolo_seg(coco_json_path, images_dir, output_labels_dir):
    """
    Convert COCO segmentation annotations to YOLO format.
    
    YOLO segmentation format:
    - One txt file per image
    - Each line: class_id x1 y1 x2 y2 x3 y3 ... (normalized coordinates 0-1)
    - Coordinates are normalized by image width/height
    """
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    output_labels_dir = Path(output_labels_dir)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create image_id to image info mapping
    images_dict = {img['id']: img for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    print(f"\nConverting COCO to YOLO format...")
    print(f"Source: {coco_json_path}")
    print(f"Output: {output_labels_dir}")
    print(f"Images: {len(coco_data['images'])}")
    print(f"Annotations: {len(coco_data['annotations'])}")
    
    # Convert each image's annotations
    converted = 0
    skipped = 0
    
    for img_id, img_info in tqdm(images_dict.items(), desc="Converting"):
        # Get image dimensions
        img_width = img_info['width']
        img_height = img_info['height']
        img_filename = img_info['file_name']
        
        # Create label file path (same name as image but .txt)
        label_filename = Path(img_filename).stem + '.txt'
        label_path = output_labels_dir / label_filename
        
        # Get annotations for this image
        annotations = annotations_by_image.get(img_id, [])
        
        if not annotations:
            # Create empty file for images without annotations
            label_path.write_text('')
            skipped += 1
            continue
        
        # Convert annotations to YOLO format
        yolo_lines = []
        for ann in annotations:
            if 'segmentation' not in ann or not ann['segmentation']:
                continue
            
            category_id = ann['category_id']
            
            # Convert segmentation polygons
            for seg in ann['segmentation']:
                if len(seg) < 6:  # Need at least 3 points (6 coordinates)
                    continue
                
                # Normalize coordinates
                normalized_coords = []
                for i in range(0, len(seg), 2):
                    x = seg[i] / img_width
                    y = seg[i + 1] / img_height
                    
                    # Clip to [0, 1] range
                    x = max(0.0, min(1.0, x))
                    y = max(0.0, min(1.0, y))
                    
                    normalized_coords.append(f"{x:.6f}")
                    normalized_coords.append(f"{y:.6f}")
                
                # Create YOLO format line: class_id x1 y1 x2 y2 ...
                yolo_line = f"{category_id} " + " ".join(normalized_coords)
                yolo_lines.append(yolo_line)
        
        # Write to file
        if yolo_lines:
            label_path.write_text('\n'.join(yolo_lines))
            converted += 1
        else:
            label_path.write_text('')
            skipped += 1
    
    print(f"\n✓ Conversion complete!")
    print(f"  Converted: {converted} images with annotations")
    print(f"  Skipped: {skipped} images without valid annotations")
    print(f"  Output directory: {output_labels_dir}")
    
    return converted, skipped


def convert_dataset(dataset_path, splits=['train', 'valid']):
    """
    Convert all splits in a COCO dataset to YOLO format.
    """
    dataset_path = Path(dataset_path)
    
    print("\n" + "="*70)
    print("COCO TO YOLO SEGMENTATION CONVERTER")
    print("="*70)
    print(f"Dataset: {dataset_path}")
    print(f"Splits to convert: {', '.join(splits)}")
    print("="*70)
    
    results = {}
    
    for split in splits:
        split_path = dataset_path / split
        
        if not split_path.exists():
            print(f"\n⚠ Warning: Split '{split}' not found, skipping...")
            continue
        
        # Find COCO annotation file
        coco_files = list(split_path.glob('*.json')) + list(split_path.glob('*annotations*.json'))
        if not coco_files:
            print(f"\n⚠ Warning: No COCO JSON file found in {split_path}, skipping...")
            continue
        
        coco_json = coco_files[0]
        
        # Create labels directory
        labels_dir = split_path / 'labels'
        
        print(f"\n{'='*70}")
        print(f"Converting split: {split}")
        print(f"{'='*70}")
        
        # Convert
        converted, skipped = convert_coco_to_yolo_seg(
            coco_json_path=str(coco_json),
            images_dir=str(split_path),
            output_labels_dir=str(labels_dir)
        )
        
        results[split] = {
            'converted': converted,
            'skipped': skipped,
            'labels_dir': str(labels_dir)
        }
    
    print("\n" + "="*70)
    print("CONVERSION COMPLETE!")
    print("="*70)
    for split, result in results.items():
        print(f"  {split}: {result['converted']} annotations converted")
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\datasets\\WormBodyDetection.v10i.coco-segmentation"
    
    # Convert dataset
    results = convert_dataset(DATASET_PATH, splits=['train', 'valid'])
    
    print("✓ Dataset is now ready for YOLO training!")
    print("\nNext steps:")
    print("  1. The YAML file should automatically detect the labels folders")
    print("  2. Run train_yolo_segmentation.py to start training")
