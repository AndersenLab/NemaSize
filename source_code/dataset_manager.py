"""
Dataset Manager for managing local annotated data for RT-DETR model training.
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import random
import numpy as np
import cv2


class DatasetManager:
    """
    Manager class for organizing and managing local datasets
    for RT-DETR model training.
    """
    
    def __init__(self, dataset_root: str = "./datasets"):
        """
        Initialize the DatasetManager.
        
        Args:
            dataset_root: Root directory containing datasets
        """
        self.dataset_root = Path(dataset_root)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        
    def load_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Load and analyze a local dataset.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            Dictionary containing dataset information and paths
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        print(f"Loading dataset from: {dataset_path}")
        
        # Analyze dataset structure
        dataset_info = {
            "name": dataset_path.name,
            "location": str(dataset_path.absolute()),
            "splits": {}
        }
        
        # Check for common splits
        for split in ["train", "valid", "val", "test"]:
            split_path = dataset_path / split
            if split_path.exists():
                images = self._count_images(split_path)
                annotations = self._find_annotations(split_path)
                
                dataset_info["splits"][split] = {
                    "path": str(split_path.absolute()),
                    "image_count": images,
                    "annotations": annotations
                }
        
        # Calculate totals
        total_images = sum(split["image_count"] for split in dataset_info["splits"].values())
        dataset_info["total_images"] = total_images
        
        self._print_dataset_summary(dataset_info)
        
        # Save dataset info
        info_path = dataset_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        return dataset_info
    
    def _count_images(self, path: Path) -> int:
        """Count image files in a directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        if not path.exists():
            return 0
        
        # Use a set to avoid counting duplicates (important on case-insensitive file systems)
        image_files = set()
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.add(file_path)
        
        return len(image_files)
    
    def _find_annotations(self, path: Path) -> Dict[str, str]:
        """Find annotation files in a directory."""
        annotations = {}
        
        if path.exists():
            # COCO format
            coco_files = list(path.rglob("*annotations*.json")) + list(path.rglob("*.json"))
            if coco_files:
                annotations["coco"] = str(coco_files[0].absolute())
            
            # YOLO format
            if (path / "labels").exists() or list(path.rglob("*.txt")):
                annotations["yolo"] = "txt labels found"
            
            # Pascal VOC format
            if list(path.rglob("*.xml")):
                annotations["voc"] = "xml annotations found"
        
        return annotations
    
    def _print_dataset_summary(self, info: Dict[str, Any]):
        """Print a formatted summary of the dataset."""
        print("\n" + "="*60)
        print("Dataset Summary")
        print("="*60)
        print(f"Name: {info['name']}")
        print(f"Location: {info['location']}")
        print(f"\nSplits:")
        
        for split_name, split_info in info['splits'].items():
            print(f"  {split_name.capitalize()}:")
            print(f"    Images: {split_info['image_count']}")
            if split_info['annotations']:
                print(f"    Annotations: {', '.join(split_info['annotations'].keys())}")
        
        print(f"\nTotal Images: {info['total_images']}")
        print("="*60 + "\n")
    
    def verify_coco_format(self, annotation_path: str) -> Dict[str, Any]:
        """
        Verify and analyze a COCO format annotation file.
        
        Args:
            annotation_path: Path to COCO JSON annotation file
            
        Returns:
            Dictionary with COCO dataset statistics
        """
        annotation_path = Path(annotation_path)
        
        if not annotation_path.exists():
            raise ValueError(f"Annotation file not found: {annotation_path}")
        
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        stats = {
            "images": len(coco_data.get("images", [])),
            "annotations": len(coco_data.get("annotations", [])),
            "categories": len(coco_data.get("categories", [])),
            "category_names": [cat["name"] for cat in coco_data.get("categories", [])]
        }
        
        print(f"\nCOCO Annotation Statistics ({annotation_path.name}):")
        print(f"  Images: {stats['images']}")
        print(f"  Annotations: {stats['annotations']}")
        print(f"  Categories: {stats['categories']}")
        print(f"  Classes: {', '.join(stats['category_names'])}")
        
        return stats
    
    def get_annotation_statistics(self, annotation_path: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Get comprehensive statistics about annotations in a COCO format file.
        
        Args:
            annotation_path: Path to COCO JSON annotation file
            verbose: Whether to print detailed statistics (default: True)
            
        Returns:
            Dictionary containing comprehensive annotation statistics including:
            - Total counts (images, annotations, categories)
            - Per-category statistics (count, percentage, area stats)
            - Image statistics
            - Area statistics
        """
        annotation_path = Path(annotation_path)
        
        if not annotation_path.exists():
            raise ValueError(f"Annotation file not found: {annotation_path}")
        
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        images = coco_data.get("images", [])
        annotations = coco_data.get("annotations", [])
        categories = coco_data.get("categories", [])
        
        # Basic counts
        num_images = len(images)
        num_annotations = len(annotations)
        num_categories = len(categories)
        
        # Create category lookup
        cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
        
        # Initialize per-category statistics
        category_stats = {cat['id']: {
            'name': cat['name'],
            'count': 0,
            'areas': [],
            'bbox_widths': [],
            'bbox_heights': []
        } for cat in categories}
        
        # Count annotations per image
        annotations_per_image = {}
        
        # Process each annotation
        for ann in annotations:
            cat_id = ann['category_id']
            img_id = ann['image_id']
            
            # Count by category
            if cat_id in category_stats:
                category_stats[cat_id]['count'] += 1
                
                # Collect area statistics
                if 'area' in ann:
                    category_stats[cat_id]['areas'].append(ann['area'])
                
                # Collect bbox statistics
                if 'bbox' in ann:
                    x, y, w, h = ann['bbox']
                    category_stats[cat_id]['bbox_widths'].append(w)
                    category_stats[cat_id]['bbox_heights'].append(h)
            
            # Count by image
            annotations_per_image[img_id] = annotations_per_image.get(img_id, 0) + 1
        
        # Compute statistics for each category
        category_summary = []
        for cat_id, stats in category_stats.items():
            summary = {
                'category_id': cat_id,
                'name': stats['name'],
                'count': stats['count'],
                'percentage': (stats['count'] / num_annotations * 100) if num_annotations > 0 else 0
            }
            
            # Area statistics
            if stats['areas']:
                summary['area_mean'] = np.mean(stats['areas'])
                summary['area_std'] = np.std(stats['areas'])
                summary['area_min'] = np.min(stats['areas'])
                summary['area_max'] = np.max(stats['areas'])
                summary['area_median'] = np.median(stats['areas'])
            
            # Bbox statistics
            if stats['bbox_widths']:
                summary['bbox_width_mean'] = np.mean(stats['bbox_widths'])
                summary['bbox_height_mean'] = np.mean(stats['bbox_heights'])
                summary['bbox_width_min'] = np.min(stats['bbox_widths'])
                summary['bbox_width_max'] = np.max(stats['bbox_widths'])
                summary['bbox_height_min'] = np.min(stats['bbox_heights'])
                summary['bbox_height_max'] = np.max(stats['bbox_heights'])
            
            category_summary.append(summary)
        
        # Sort by count descending
        category_summary = sorted(category_summary, key=lambda x: x['count'], reverse=True)
        
        # Image statistics
        if annotations_per_image:
            ann_per_img_values = list(annotations_per_image.values())
            image_stats = {
                'images_with_annotations': len(annotations_per_image),
                'images_without_annotations': num_images - len(annotations_per_image),
                'annotations_per_image_mean': np.mean(ann_per_img_values),
                'annotations_per_image_std': np.std(ann_per_img_values),
                'annotations_per_image_min': np.min(ann_per_img_values),
                'annotations_per_image_max': np.max(ann_per_img_values),
                'annotations_per_image_median': np.median(ann_per_img_values)
            }
        else:
            image_stats = {
                'images_with_annotations': 0,
                'images_without_annotations': num_images,
                'annotations_per_image_mean': 0,
                'annotations_per_image_std': 0,
                'annotations_per_image_min': 0,
                'annotations_per_image_max': 0,
                'annotations_per_image_median': 0
            }
        
        # Compile results
        results = {
            'file_name': annotation_path.name,
            'file_path': str(annotation_path.absolute()),
            'total_images': num_images,
            'total_annotations': num_annotations,
            'total_categories': num_categories,
            'category_names': [cat['name'] for cat in categories],
            'categories': category_summary,
            'image_statistics': image_stats
        }
        
        # Print detailed statistics if verbose
        if verbose:
            self._print_annotation_statistics(results)
        
        return results
    
    def _print_annotation_statistics(self, stats: Dict[str, Any]):
        """
        Print formatted annotation statistics.
        
        Args:
            stats: Statistics dictionary from get_annotation_statistics()
        """
        print("\n" + "="*70)
        print(f"ANNOTATION STATISTICS - {stats['file_name']}")
        print("="*70)
        
        # Overall statistics
        print("\n📊 Overall Statistics:")
        print(f"  Total Images:      {stats['total_images']}")
        print(f"  Total Annotations: {stats['total_annotations']}")
        print(f"  Total Categories:  {stats['total_categories']}")
        
        # Image statistics
        img_stats = stats['image_statistics']
        print(f"\n🖼️  Image Statistics:")
        print(f"  Images with annotations:    {img_stats['images_with_annotations']}")
        print(f"  Images without annotations: {img_stats['images_without_annotations']}")
        print(f"  Annotations per image:")
        print(f"    Mean:   {img_stats['annotations_per_image_mean']:.2f}")
        print(f"    Std:    {img_stats['annotations_per_image_std']:.2f}")
        print(f"    Min:    {img_stats['annotations_per_image_min']:.0f}")
        print(f"    Max:    {img_stats['annotations_per_image_max']:.0f}")
        print(f"    Median: {img_stats['annotations_per_image_median']:.2f}")
        
        # Per-category statistics
        print(f"\n📁 Per-Category Statistics:")
        print("\n" + "-"*70)
        for cat_stat in stats['categories']:
            print(f"\nCategory: {cat_stat['name']} (ID: {cat_stat['category_id']})")
            print(f"  Count:      {cat_stat['count']} ({cat_stat['percentage']:.2f}%)")
            
            if 'area_mean' in cat_stat:
                print(f"  Area:")
                print(f"    Mean:   {cat_stat['area_mean']:.2f}")
                print(f"    Std:    {cat_stat['area_std']:.2f}")
                print(f"    Min:    {cat_stat['area_min']:.2f}")
                print(f"    Max:    {cat_stat['area_max']:.2f}")
                print(f"    Median: {cat_stat['area_median']:.2f}")
            
            if 'bbox_width_mean' in cat_stat:
                print(f"  Bounding Box:")
                print(f"    Width:  {cat_stat['bbox_width_mean']:.2f} (min: {cat_stat['bbox_width_min']:.2f}, max: {cat_stat['bbox_width_max']:.2f})")
                print(f"    Height: {cat_stat['bbox_height_mean']:.2f} (min: {cat_stat['bbox_height_min']:.2f}, max: {cat_stat['bbox_height_max']:.2f})")
        
        print("\n" + "="*70 + "\n")
    
    def prepare_for_rtdetr(self, dataset_path: str, output_path: Optional[str] = None) -> str:
        """
        Prepare a dataset for RT-DETR training.
        Ensures the dataset is in the correct format and structure.
        
        Args:
            dataset_path: Path to the source dataset
            output_path: Optional output path (if None, uses dataset_path)
            
        Returns:
            Path to the prepared dataset
        """
        dataset_path = Path(dataset_path)
        
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = dataset_path
        
        print(f"Preparing dataset for RT-DETR training...")
        print(f"Source: {dataset_path}")
        print(f"Output: {output_path}")
        
        # Verify required structure
        required_splits = ["train", "valid"]
        missing_splits = []
        
        for split in required_splits:
            split_variants = [split, "val" if split == "valid" else None]
            found = False
            for variant in split_variants:
                if variant and (dataset_path / variant).exists():
                    found = True
                    break
            if not found:
                missing_splits.append(split)
        
        if missing_splits:
            print(f"⚠ Warning: Missing required splits: {', '.join(missing_splits)}")
        else:
            print("✓ All required splits found (train, valid)")
        
        print(f"✓ Dataset prepared at: {output_path}")
        return str(output_path.absolute())
    
    def list_datasets(self) -> List[str]:
        """
        List all datasets in the dataset root directory.
        
        Returns:
            List of dataset names
        """
        datasets = []
        if self.dataset_root.exists():
            for item in self.dataset_root.iterdir():
                if item.is_dir():
                    datasets.append(item.name)
        
        if datasets:
            print(f"Available datasets in {self.dataset_root}:")
            for ds in datasets:
                print(f"  - {ds}")
        else:
            print(f"No datasets found in {self.dataset_root}")
        
        return datasets
    
    def visualize_annotations(
        self,
        annotation_path: str,
        image_dir: str,
        num_images: int = 5,
        image_ids: Optional[List[int]] = None,
        show_boxes: bool = True,
        show_masks: bool = True,
        show_labels: bool = True,
        save_dir: Optional[str] = None,
        display: bool = True
    ):
        """
        Visualize images with their COCO segmentation annotations.
        
        Args:
            annotation_path: Path to COCO JSON annotation file
            image_dir: Directory containing the images
            num_images: Number of random images to visualize (if image_ids not specified)
            image_ids: Specific image IDs to visualize (optional)
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show segmentation masks
            show_labels: Whether to show category labels
            save_dir: Optional directory to save visualizations
            display: Whether to display images (press any key to continue)
        """
        # Load COCO annotations
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        # Create lookup dictionaries
        images_dict = {img['id']: img for img in coco_data['images']}
        categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Select images to visualize
        if image_ids is None:
            available_ids = list(annotations_by_image.keys())
            image_ids = random.sample(available_ids, min(num_images, len(available_ids)))
        
        # Create save directory if specified
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Visualize each image
        for img_id in image_ids:
            self._visualize_single_image(
                img_id,
                images_dict,
                annotations_by_image,
                categories_dict,
                image_dir,
                show_boxes,
                show_masks,
                show_labels,
                save_dir,
                display
            )
        
        if display:
            cv2.destroyAllWindows()
        
        print(f"\n✓ Visualized {len(image_ids)} images")
    
    def _visualize_single_image(
        self,
        img_id: int,
        images_dict: Dict,
        annotations_by_image: Dict,
        categories_dict: Dict,
        image_dir: str,
        show_boxes: bool,
        show_masks: bool,
        show_labels: bool,
        save_dir: Optional[str],
        display: bool
    ):
        """
        Visualize a single image with its annotations.
        """
        if img_id not in images_dict:
            print(f"Warning: Image ID {img_id} not found")
            return
        
        img_info = images_dict[img_id]
        img_filename = img_info['file_name']
        
        # Find the image file
        image_path = self._find_image_file(image_dir, img_filename)
        if not image_path:
            print(f"Warning: Image file not found: {img_filename}")
            return
        
        # Load image with OpenCV
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Failed to load image: {img_filename}")
            return
        
        # Convert BGR to RGB for proper color display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a copy for drawing
        img_display = img_rgb.copy()
        
        # Get annotations for this image
        annotations = annotations_by_image.get(img_id, [])
        
        # Generate colors for each annotation (BGR format for OpenCV)
        colors = self._generate_colors(len(annotations))
        
        # Draw each annotation
        for ann, color in zip(annotations, colors):
            category_name = categories_dict.get(ann['category_id'], 'Unknown')
            
            # Draw segmentation mask
            if show_masks and 'segmentation' in ann:
                img_display = self._draw_segmentation(img_display, ann['segmentation'], color, img_info)
            
            # Draw bounding box
            if show_boxes and 'bbox' in ann:
                img_display = self._draw_bbox(img_display, ann['bbox'], color, category_name if show_labels else None)
        
        # Add title/info text at the top
        title = f"ID: {img_id} | File: {img_filename} | Annotations: {len(annotations)}"
        cv2.putText(img_display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Convert back to BGR for OpenCV display/save
        img_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        # Save or display
        if save_dir:
            save_path = Path(save_dir) / f"visualization_{img_id}_{Path(img_filename).stem}.png"
            cv2.imwrite(str(save_path), img_bgr)
            print(f"Saved: {save_path}")
        
        if display:
            # Resize if image is too large
            h, w = img_bgr.shape[:2]
            max_dim = 1200
            if max(h, w) > max_dim:
                scale = max_dim / max(h, w)
                img_bgr = cv2.resize(img_bgr, None, fx=scale, fy=scale)
            
            cv2.imshow(f'Image ID: {img_id}', img_bgr)
            print(f"Displaying image {img_id}. Press any key to continue...")
            cv2.waitKey(0)
    
    def _find_image_file(self, image_dir: str, filename: str) -> Optional[Path]:
        """
        Find an image file in the directory (handles nested structures).
        """
        image_dir = Path(image_dir)
        
        # Try direct path
        direct_path = image_dir / filename
        if direct_path.exists():
            return direct_path
        
        # Search recursively
        for img_path in image_dir.rglob(filename):
            return img_path
        
        return None
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n distinct colors in RGB format.
        """
        colors = []
        for i in range(n):
            hue = int(180 * i / max(n, 1))
            # Create color in HSV then convert to RGB (OpenCV uses BGR)
            hsv = np.uint8([[[hue, 255, 255]]])
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0][0]
            colors.append(tuple(map(int, rgb)))
        return colors
    
    def _draw_segmentation(self, img, segmentation, color, img_info):
        """
        Draw segmentation mask on the image.
        Returns the modified image.
        """
        overlay = img.copy()
        
        if isinstance(segmentation, list):
            # Polygon format
            for seg in segmentation:
                if len(seg) >= 6:  # At least 3 points (x, y pairs)
                    poly = np.array(seg).reshape(-1, 2).astype(np.int32)
                    # Fill polygon with transparency
                    cv2.fillPoly(overlay, [poly], color)
                    # Draw polygon outline
                    cv2.polylines(img, [poly], True, color, 2)
            
            # Blend overlay with original for transparency effect
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            
        elif isinstance(segmentation, dict):
            # RLE format
            if 'counts' in segmentation:
                mask = self._decode_rle(segmentation, img_info['height'], img_info['width'])
                # Apply colored mask
                colored_mask = np.zeros_like(img)
                colored_mask[mask > 0] = color
                cv2.addWeighted(img, 1.0, colored_mask, 0.4, 0, img)
        
        return img
    
    def _decode_rle(self, rle, height, width):
        """
        Decode RLE (Run-Length Encoding) to binary mask.
        """
        if isinstance(rle['counts'], list):
            # Uncompressed RLE
            counts = rle['counts']
            mask = np.zeros(height * width, dtype=np.uint8)
            pos = 0
            val = 0
            for count in counts:
                mask[pos:pos+count] = val
                pos += count
                val = 1 - val
            return mask.reshape((height, width))
        else:
            # Compressed RLE - would need pycocotools
            print("Warning: Compressed RLE format requires pycocotools")
            return np.zeros((height, width), dtype=np.uint8)
    
    def _draw_bbox(self, img, bbox, color, label=None):
        """
        Draw bounding box on the image.
        COCO bbox format: [x, y, width, height]
        Returns the modified image.
        """
        x, y, w, h = map(int, bbox)
        
        # Draw rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Draw label if provided
        if label:
            # Get text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw background rectangle for text
            cv2.rectangle(img, (x, y - text_h - baseline - 5), 
                         (x + text_w, y), color, -1)
            
            # Draw text
            cv2.putText(img, label, (x, y - baseline - 2), font, 
                       font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        return img
    
    def augment_dataset(
        self,
        dataset_path: str,
        split: str = 'train',
        augmentation_factor: int = 2,
        rotation_angles: Optional[List[float]] = None,
        intensity_range: Tuple[float, float] = (0.8, 1.2),
        noise_sigma: float = 0.02,
        combinations: bool = True,
        output_suffix: str = '_aug',
        seed: int = 42
    ) -> Dict[str, Any]:
        """
        Augment a dataset with various transformations.
        Creates augmented images and updates COCO annotations accordingly.
        
        Args:
            dataset_path: Path to the dataset directory
            split: Split name to augment (default: 'train')
            augmentation_factor: Number of augmented versions per image (default: 2)
            rotation_angles: List of rotation angles in degrees. If None, uses random rotations
            intensity_range: Range for intensity adjustment (min, max) as multipliers
            noise_sigma: Standard deviation for Gaussian noise (0-1 range)
            combinations: Whether to combine multiple augmentations
            output_suffix: Suffix to add to augmented image filenames
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing augmentation statistics
        """
        dataset_path = Path(dataset_path)
        split_path = dataset_path / split
        
        if not split_path.exists():
            raise ValueError(f"Split '{split}' not found: {split_path}")
        
        # Find annotation file
        annotation_files = list(split_path.glob('*annotations*.json')) + list(split_path.glob('*.json'))
        if not annotation_files:
            raise ValueError(f"No COCO annotation file found in {split_path}")
        
        annotation_path = annotation_files[0]
        print(f"Using annotation file: {annotation_path.name}")
        
        # Load COCO annotations
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        print(f"\n{'='*60}")
        print(f"Data Augmentation for '{split}' split")
        print(f"{'='*60}")
        print(f"Original images: {len(coco_data['images'])}")
        print(f"Augmentation factor: {augmentation_factor}")
        print(f"Augmentation settings:")
        print(f"  - Rotations: {rotation_angles if rotation_angles else 'Random'}")
        print(f"  - Intensity range: {intensity_range}")
        print(f"  - Noise sigma: {noise_sigma}")
        print(f"  - Combinations: {combinations}")
        
        # Track new images and annotations
        new_images = []
        new_annotations = []
        next_image_id = max(img['id'] for img in coco_data['images']) + 1
        next_ann_id = max(ann['id'] for ann in coco_data['annotations']) + 1
        
        # Group annotations by image_id
        annotations_by_image = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_by_image:
                annotations_by_image[img_id] = []
            annotations_by_image[img_id].append(ann)
        
        # Augment each image
        total_created = 0
        for img_info in coco_data['images']:
            img_path = self._find_image_file(str(split_path), img_info['file_name'])
            if not img_path:
                print(f"Warning: Image not found: {img_info['file_name']}")
                continue
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Failed to load image: {img_info['file_name']}")
                continue
            
            # Get annotations for this image
            img_annotations = annotations_by_image.get(img_info['id'], [])
            
            # Create augmented versions
            for aug_idx in range(augmentation_factor):
                # Determine augmentation parameters
                aug_params = self._get_augmentation_params(
                    rotation_angles, intensity_range, noise_sigma, combinations, aug_idx
                )
                
                # Apply augmentations
                aug_img, aug_annotations = self._apply_augmentations(
                    img.copy(), img_annotations, img_info, aug_params
                )
                
                # Save augmented image
                aug_filename = self._generate_aug_filename(
                    img_info['file_name'], aug_idx, output_suffix
                )
                aug_img_path = split_path / aug_filename
                cv2.imwrite(str(aug_img_path), aug_img)
                
                # Create new image info
                new_img_info = img_info.copy()
                new_img_info['id'] = next_image_id
                new_img_info['file_name'] = aug_filename
                new_images.append(new_img_info)
                
                # Update annotation IDs and image_id
                for ann in aug_annotations:
                    ann['id'] = next_ann_id
                    ann['image_id'] = next_image_id
                    new_annotations.append(ann)
                    next_ann_id += 1
                
                next_image_id += 1
                total_created += 1
        
        print(f"\nCreated {total_created} augmented images")
        print(f"Created {len(new_annotations)} augmented annotations")
        
        # Update COCO data
        coco_data['images'].extend(new_images)
        coco_data['annotations'].extend(new_annotations)
        
        # Save updated annotation file
        with open(annotation_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\nUpdated annotation file: {annotation_path}")
        print(f"Total images now: {len(coco_data['images'])}")
        print(f"Total annotations now: {len(coco_data['annotations'])}")
        print(f"{'='*60}\n")
        
        # Show updated statistics for the augmented split
        print(f"Getting updated annotation statistics for '{split}' split...")
        updated_stats = self.get_annotation_statistics(str(annotation_path), verbose=True)
        
        # Also show statistics for other splits (e.g., validation)
        dataset_path_obj = Path(dataset_path)
        other_splits_stats = {}
        for other_split in ['valid', 'val', 'test']:
            if other_split != split:
                other_split_path = dataset_path_obj / other_split
                if other_split_path.exists():
                    other_ann_files = list(other_split_path.glob('*annotations*.json')) + list(other_split_path.glob('*.json'))
                    if other_ann_files:
                        print(f"\nGetting annotation statistics for '{other_split}' split...")
                        other_stats = self.get_annotation_statistics(str(other_ann_files[0]), verbose=True)
                        other_splits_stats[other_split] = other_stats
        
        return {
            'original_images': len(coco_data['images']) - len(new_images),
            'augmented_images': len(new_images),
            'total_images': len(coco_data['images']),
            'augmented_annotations': len(new_annotations),
            'total_annotations': len(coco_data['annotations']),
            'updated_statistics': updated_stats,
            'other_splits_statistics': other_splits_stats
        }
    
    def _get_augmentation_params(
        self,
        rotation_angles: Optional[List[float]],
        intensity_range: Tuple[float, float],
        noise_sigma: float,
        combinations: bool,
        aug_idx: int
    ) -> Dict[str, Any]:
        """Generate augmentation parameters for a single augmentation."""
        params = {}
        
        if combinations:
            # Apply multiple augmentations
            if rotation_angles:
                params['rotation'] = rotation_angles[aug_idx % len(rotation_angles)]
            else:
                params['rotation'] = random.uniform(1, 359)
            
            params['intensity'] = random.uniform(intensity_range[0], intensity_range[1])
            params['noise'] = random.uniform(0, noise_sigma)
        else:
            # Apply single augmentation type in rotation
            aug_type = aug_idx % 3
            if aug_type == 0:  # Rotation
                if rotation_angles:
                    params['rotation'] = rotation_angles[aug_idx % len(rotation_angles)]
                else:
                    params['rotation'] = random.uniform(1, 359)
            elif aug_type == 1:  # Intensity
                params['intensity'] = random.uniform(intensity_range[0], intensity_range[1])
            else:  # Noise
                params['noise'] = random.uniform(0, noise_sigma)
        
        return params
    
    def _apply_augmentations(
        self,
        img: np.ndarray,
        annotations: List[Dict],
        img_info: Dict,
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Apply augmentations to image and annotations."""
        height, width = img.shape[:2]
        aug_annotations = []
        
        # Apply rotation
        if 'rotation' in params:
            img, rotation_matrix = self._rotate_image(img, params['rotation'])
            # Transform annotations
            for ann in annotations:
                aug_ann = ann.copy()
                aug_ann = self._transform_annotation(aug_ann, rotation_matrix, width, height)
                aug_annotations.append(aug_ann)
        else:
            aug_annotations = [ann.copy() for ann in annotations]
        
        # Apply intensity adjustment
        if 'intensity' in params:
            img = self._adjust_intensity(img, params['intensity'])
        
        # Apply noise
        if 'noise' in params:
            img = self._add_noise(img, params['noise'])
        
        return img, aug_annotations
    
    def _rotate_image(self, img: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate image around center while keeping the same size.
        Returns rotated image and rotation matrix.
        """
        height, width = img.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated = cv2.warpAffine(
            img, rotation_matrix, (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        
        return rotated, rotation_matrix
    
    def _adjust_intensity(self, img: np.ndarray, factor: float) -> np.ndarray:
        """
        Adjust overall intensity of the image.
        Factor > 1 increases brightness, < 1 decreases brightness.
        """
        # Convert to float for precision
        img_float = img.astype(np.float32)
        
        # Adjust intensity
        adjusted = img_float * factor
        
        # Clip values to valid range
        adjusted = np.clip(adjusted, 0, 255)
        
        return adjusted.astype(np.uint8)
    
    def _add_noise(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """
        Add Gaussian noise to the image.
        Sigma is in the range [0, 1] relative to image intensity range.
        """
        # Generate Gaussian noise
        noise = np.random.normal(0, sigma * 255, img.shape)
        
        # Add noise to image
        noisy = img.astype(np.float32) + noise
        
        # Clip values to valid range
        noisy = np.clip(noisy, 0, 255)
        
        return noisy.astype(np.uint8)
    
    def _transform_annotation(
        self,
        annotation: Dict,
        rotation_matrix: np.ndarray,
        orig_width: int,
        orig_height: int
    ) -> Dict:
        """Transform annotation (segmentation and bbox) using rotation matrix."""
        # Transform segmentation polygons
        if 'segmentation' in annotation and isinstance(annotation['segmentation'], list):
            transformed_segs = []
            for seg in annotation['segmentation']:
                if len(seg) >= 6:  # At least 3 points
                    # Reshape to points
                    points = np.array(seg).reshape(-1, 2)
                    # Transform points
                    transformed_points = self._transform_points(points, rotation_matrix)
                    # Flatten back to list
                    transformed_segs.append(transformed_points.flatten().tolist())
            annotation['segmentation'] = transformed_segs
        
        # Recalculate bounding box from transformed segmentation
        if annotation['segmentation']:
            annotation['bbox'] = self._calculate_bbox_from_segmentation(
                annotation['segmentation']
            )
            annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        
        return annotation
    
    def _transform_points(
        self,
        points: np.ndarray,
        rotation_matrix: np.ndarray
    ) -> np.ndarray:
        """Transform 2D points using rotation matrix."""
        # Add homogeneous coordinate
        ones = np.ones((points.shape[0], 1))
        points_homogeneous = np.hstack([points, ones])
        
        # Apply transformation
        transformed = rotation_matrix @ points_homogeneous.T
        
        return transformed.T
    
    def _calculate_bbox_from_segmentation(self, segmentation: List) -> List[float]:
        """Calculate bounding box [x, y, width, height] from segmentation polygons."""
        all_x = []
        all_y = []
        
        for seg in segmentation:
            points = np.array(seg).reshape(-1, 2)
            all_x.extend(points[:, 0])
            all_y.extend(points[:, 1])
        
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]
    
    def _generate_aug_filename(
        self,
        original_filename: str,
        aug_idx: int,
        suffix: str
    ) -> str:
        """Generate filename for augmented image."""
        path = Path(original_filename)
        stem = path.stem
        ext = path.suffix
        
        return f"{stem}{suffix}_{aug_idx}{ext}"
    
    def remove_augmented_images(
        self,
        dataset_path: str,
        split: str = 'train',
        suffix_pattern: str = '_aug'
    ) -> Dict[str, Any]:
        """
        Remove all augmented images from a dataset split.
        Useful for re-augmenting with different parameters.
        
        Args:
            dataset_path: Path to the dataset directory
            split: Split name to clean (default: 'train')
            suffix_pattern: The suffix pattern used for augmented images (default: '_aug')
            
        Returns:
            Dictionary containing removal statistics
        """
        dataset_path = Path(dataset_path)
        split_path = dataset_path / split
        
        if not split_path.exists():
            raise ValueError(f"Split '{split}' not found: {split_path}")
        
        # Find annotation file
        annotation_files = list(split_path.glob('*annotations*.json')) + list(split_path.glob('*.json'))
        if not annotation_files:
            raise ValueError(f"No COCO annotation file found in {split_path}")
        
        annotation_path = annotation_files[0]
        print(f"Using annotation file: {annotation_path.name}")
        
        # Load COCO annotations
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        print(f"\n{'='*60}")
        print(f"Removing Augmented Images from '{split}' split")
        print(f"{'='*60}")
        print(f"Looking for images with pattern: '*{suffix_pattern}_*'")
        print(f"Total images before: {len(coco_data['images'])}")
        print(f"Total annotations before: {len(coco_data['annotations'])}")
        
        # Identify augmented images (those with the suffix pattern)
        original_images = []
        augmented_images = []
        augmented_image_ids = set()
        
        for img_info in coco_data['images']:
            filename = img_info['file_name']
            stem = Path(filename).stem
            
            # Check if filename contains the augmentation suffix pattern
            if suffix_pattern in stem and any(
                stem.endswith(f"{suffix_pattern}_{i}") for i in range(1000)
            ):
                augmented_images.append(img_info)
                augmented_image_ids.add(img_info['id'])
            else:
                original_images.append(img_info)
        
        print(f"\nFound {len(augmented_images)} augmented images")
        print(f"Found {len(original_images)} original images")
        
        if len(augmented_images) == 0:
            print("\nNo augmented images found. Nothing to remove.")
            print(f"{'='*60}\n")
            return {
                'removed_images': 0,
                'removed_annotations': 0,
                'remaining_images': len(original_images),
                'remaining_annotations': len(coco_data['annotations'])
            }
        
        # Remove augmented image files from disk
        print(f"\nRemoving augmented image files from disk...")
        removed_files = 0
        for img_info in augmented_images:
            img_path = self._find_image_file(str(split_path), img_info['file_name'])
            if img_path and img_path.exists():
                img_path.unlink()
                removed_files += 1
        
        print(f"Removed {removed_files} image files")
        
        # Filter out annotations for augmented images
        original_annotations = [
            ann for ann in coco_data['annotations']
            if ann['image_id'] not in augmented_image_ids
        ]
        
        removed_annotations = len(coco_data['annotations']) - len(original_annotations)
        
        # Update COCO data
        coco_data['images'] = original_images
        coco_data['annotations'] = original_annotations
        
        # Save updated annotation file
        with open(annotation_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"\nUpdated annotation file: {annotation_path}")
        print(f"Total images now: {len(coco_data['images'])}")
        print(f"Total annotations now: {len(coco_data['annotations'])}")
        
        print(f"\n{'='*60}")
        print("Augmented Images Removed Successfully!")
        print(f"{'='*60}")
        print(f"Removed images: {len(augmented_images)}")
        print(f"Removed annotations: {removed_annotations}")
        print(f"Remaining images: {len(original_images)}")
        print(f"Remaining annotations: {len(original_annotations)}")
        print(f"{'='*60}\n")
        
        return {
            'removed_images': len(augmented_images),
            'removed_annotations': removed_annotations,
            'remaining_images': len(original_images),
            'remaining_annotations': len(original_annotations)
        }
    
    def split_train_validation(
        self,
        dataset_path: str,
        val_ratio: float = 0.2,
        seed: int = 42,
        train_split_name: str = 'train',
        val_split_name: str = 'valid'
    ) -> Dict[str, Any]:
        """
        Split the train dataset into train and validation sets.
        Creates a new validation folder with images and annotations.
        
        Args:
            dataset_path: Path to the dataset directory
            val_ratio: Ratio of data to use for validation (0.0 to 1.0)
            seed: Random seed for reproducibility
            train_split_name: Name of the training split folder (default: 'train')
            val_split_name: Name of the validation split folder (default: 'valid')
            
        Returns:
            Dictionary containing split statistics
        """
        import shutil
        
        dataset_path = Path(dataset_path)
        train_path = dataset_path / train_split_name
        val_path = dataset_path / val_split_name
        
        if not train_path.exists():
            raise ValueError(f"Training split not found: {train_path}")
        
        if val_path.exists():
            print(f"Warning: Validation split already exists at {val_path}")
            response = input("Do you want to overwrite it? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Operation cancelled.")
                return {}
            print("Removing existing validation split...")
            shutil.rmtree(val_path)
        
        print(f"\nSplitting dataset: {val_ratio*100:.1f}% for validation, {(1-val_ratio)*100:.1f}% for training")
        print(f"Random seed: {seed}")
        
        # Find the COCO annotation file
        annotation_files = list(train_path.glob('*annotations*.json')) + list(train_path.glob('*.json'))
        if not annotation_files:
            raise ValueError(f"No COCO annotation file found in {train_path}")
        
        annotation_path = annotation_files[0]
        print(f"Using annotation file: {annotation_path.name}")
        
        # Load COCO annotations
        with open(annotation_path, 'r') as f:
            coco_data = json.load(f)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Get all image IDs and shuffle them
        all_images = coco_data['images'].copy()
        random.shuffle(all_images)
        
        # Split images
        num_val = int(len(all_images) * val_ratio)
        val_images = all_images[:num_val]
        train_images = all_images[num_val:]
        
        val_image_ids = {img['id'] for img in val_images}
        train_image_ids = {img['id'] for img in train_images}
        
        print(f"\nTotal images: {len(all_images)}")
        print(f"Training images: {len(train_images)}")
        print(f"Validation images: {len(val_images)}")
        
        # Split annotations
        train_annotations = [ann for ann in coco_data['annotations'] 
                            if ann['image_id'] in train_image_ids]
        val_annotations = [ann for ann in coco_data['annotations'] 
                          if ann['image_id'] in val_image_ids]
        
        print(f"\nTotal annotations: {len(coco_data['annotations'])}")
        print(f"Training annotations: {len(train_annotations)}")
        print(f"Validation annotations: {len(val_annotations)}")
        
        # Create validation directory
        val_path.mkdir(parents=True, exist_ok=True)
        
        # Copy validation images
        print(f"\nCopying validation images to {val_path}...")
        copied_count = 0
        for img_info in val_images:
            src_img_path = self._find_image_file(str(train_path), img_info['file_name'])
            if src_img_path:
                dst_img_path = val_path / img_info['file_name']
                dst_img_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_img_path, dst_img_path)
                copied_count += 1
            else:
                print(f"Warning: Image not found: {img_info['file_name']}")
        
        print(f"Copied {copied_count} images to validation set")
        
        # Remove validation images from train directory
        print(f"\nRemoving validation images from training set...")
        removed_count = 0
        for img_info in val_images:
            img_path = self._find_image_file(str(train_path), img_info['file_name'])
            if img_path:
                img_path.unlink()
                removed_count += 1
        
        print(f"Removed {removed_count} images from training set")
        
        # Create new COCO annotation files
        train_coco = {
            'images': train_images,
            'annotations': train_annotations,
            'categories': coco_data['categories'],
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', [])
        }
        
        val_coco = {
            'images': val_images,
            'annotations': val_annotations,
            'categories': coco_data['categories'],
            'info': coco_data.get('info', {}),
            'licenses': coco_data.get('licenses', [])
        }
        
        # Save annotation files
        train_ann_path = train_path / annotation_path.name
        val_ann_path = val_path / annotation_path.name
        
        print(f"\nSaving annotation files...")
        with open(train_ann_path, 'w') as f:
            json.dump(train_coco, f, indent=2)
        print(f"Training annotations: {train_ann_path}")
        
        with open(val_ann_path, 'w') as f:
            json.dump(val_coco, f, indent=2)
        print(f"Validation annotations: {val_ann_path}")
        
        # Create summary
        summary = {
            'total_images': len(all_images),
            'train': {
                'images': len(train_images),
                'annotations': len(train_annotations),
                'ratio': 1 - val_ratio
            },
            'validation': {
                'images': len(val_images),
                'annotations': len(val_annotations),
                'ratio': val_ratio
            }
        }
        
        print("\n" + "="*60)
        print("Split Complete!")
        print("="*60)
        print(f"Training set: {summary['train']['images']} images, "
              f"{summary['train']['annotations']} annotations ({summary['train']['ratio']*100:.1f}%)")
        print(f"Validation set: {summary['validation']['images']} images, "
              f"{summary['validation']['annotations']} annotations ({summary['validation']['ratio']*100:.1f}%)")
        print("="*60 + "\n")
        
        return summary
    
    def visualize_dataset_split(
        self,
        dataset_path: str,
        split: str = 'train',
        num_images: int = 5,
        **kwargs
    ):
        """
        Convenience method to visualize a specific dataset split.
        
        Args:
            dataset_path: Path to the dataset directory
            split: Split name ('train', 'valid', 'val', 'test')
            num_images: Number of images to visualize
            **kwargs: Additional arguments passed to visualize_annotations
        """
        dataset_path = Path(dataset_path)
        split_path = dataset_path / split
        
        if not split_path.exists():
            # Try 'val' if 'valid' doesn't exist
            if split == 'valid':
                split_path = dataset_path / 'val'
        
        if not split_path.exists():
            print(f"Error: Split '{split}' not found in {dataset_path}")
            return
        
        # Find annotation file
        annotation_files = list(split_path.glob('*annotations*.json')) + list(split_path.glob('*.json'))
        if not annotation_files:
            print(f"Error: No annotation file found in {split_path}")
            return
        
        annotation_path = annotation_files[0]
        print(f"Using annotation file: {annotation_path.name}")
        print(f"Visualizing {num_images} images from '{split}' split...\n")
        
        self.visualize_annotations(
            annotation_path=str(annotation_path),
            image_dir=str(split_path),
            num_images=num_images,
            **kwargs
        )
    
    def flatten_categories(
        self,
        dataset_path: str,
        splits: Optional[List[str]] = None,
        merged_category_name: str = 'worms',
        merged_category_id: int = 0
    ) -> Dict[str, Any]:
        """
        Merge all categories into a single category across specified splits.
        Creates a backup of the original annotation files before modification.
        
        Args:
            dataset_path: Path to the dataset directory
            splits: List of split names to process (e.g., ['train', 'valid']). 
                   If None, processes all available splits.
            merged_category_name: Name for the merged category (default: 'worms')
            merged_category_id: ID for the merged category (default: 0)
            
        Returns:
            Dictionary containing merge statistics for each split
            
        Example:
            >>> manager = DatasetManager()
            >>> results = manager.flatten_categories(
            ...     dataset_path="./datasets/my_dataset",
            ...     splits=['train', 'valid'],
            ...     merged_category_name='worms'
            ... )
        """
        import shutil
        from datetime import datetime
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # If splits not specified, find all available splits
        if splits is None:
            splits = []
            for split in ["train", "valid", "val", "test"]:
                if (dataset_path / split).exists():
                    splits.append(split)
        
        if not splits:
            raise ValueError(f"No valid splits found in {dataset_path}")
        
        print("\n" + "="*70)
        print("FLATTEN CATEGORIES TO SINGLE CLASS")
        print("="*70)
        print(f"Dataset: {dataset_path}")
        print(f"Splits to process: {', '.join(splits)}")
        print(f"Merged category: '{merged_category_name}' (ID: {merged_category_id})")
        print("="*70 + "\n")
        
        results = {
            'merged_category_name': merged_category_name,
            'merged_category_id': merged_category_id,
            'splits': {}
        }
        
        for split in splits:
            split_path = dataset_path / split
            
            if not split_path.exists():
                print(f"⚠ Warning: Split '{split}' not found, skipping...")
                continue
            
            # Find annotation file
            annotation_files = list(split_path.glob('*annotations*.json')) + list(split_path.glob('*.json'))
            if not annotation_files:
                print(f"⚠ Warning: No annotation file found in {split_path}, skipping...")
                continue
            
            annotation_path = annotation_files[0]
            
            print(f"\nProcessing split: {split}")
            print(f"Annotation file: {annotation_path.name}")
            
            # Create backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_dir = split_path / "annotation_backups"
            backup_dir.mkdir(exist_ok=True)
            backup_path = backup_dir / f"{annotation_path.stem}_backup_{timestamp}{annotation_path.suffix}"
            
            shutil.copy2(annotation_path, backup_path)
            print(f"✓ Backup created: {backup_path.name}")
            
            # Load COCO annotations
            with open(annotation_path, 'r') as f:
                coco_data = json.load(f)
            
            original_categories = coco_data.get('categories', [])
            original_annotations = coco_data.get('annotations', [])
            
            print(f"  Original categories: {len(original_categories)}")
            print(f"  Original annotations: {len(original_annotations)}")
            
            # Store original category mapping
            category_mapping = {cat['id']: cat['name'] for cat in original_categories}
            
            # Create single merged category
            merged_category = {
                'id': merged_category_id,
                'name': merged_category_name,
                'supercategory': 'object'
            }
            
            # Update all annotations to use the merged category
            annotation_changes = 0
            for ann in coco_data['annotations']:
                old_cat_id = ann['category_id']
                if old_cat_id != merged_category_id:
                    ann['category_id'] = merged_category_id
                    annotation_changes += 1
            
            # Replace categories with single merged category
            coco_data['categories'] = [merged_category]
            
            # Save updated annotation file
            with open(annotation_path, 'w') as f:
                json.dump(coco_data, f, indent=2)
            
            print(f"✓ Updated {annotation_changes} annotations")
            print(f"  New categories: 1")
            print(f"  Category name: '{merged_category_name}'")
            
            # Store results for this split
            results['splits'][split] = {
                'annotation_file': str(annotation_path),
                'backup_file': str(backup_path),
                'original_categories': len(original_categories),
                'original_category_names': [cat['name'] for cat in original_categories],
                'category_mapping': category_mapping,
                'annotations_updated': annotation_changes,
                'total_annotations': len(original_annotations)
            }
        
        print("\n" + "="*70)
        print("CATEGORY FLATTENING COMPLETE!")
        print("="*70)
        print(f"Processed {len(results['splits'])} split(s)")
        print(f"All categories merged into: '{merged_category_name}'")
        print(f"Original annotation files backed up in 'annotation_backups' folders")
        print("="*70 + "\n")
        
        # Save merge report
        report_path = dataset_path / "category_merge_report.json"
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📄 Merge report saved to: {report_path}\n")
        
        return results
    
    def restore_categories_from_backup(
        self,
        dataset_path: str,
        splits: Optional[List[str]] = None,
        backup_timestamp: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Restore annotation files from backup.
        
        Args:
            dataset_path: Path to the dataset directory
            splits: List of split names to restore (e.g., ['train', 'valid']). 
                   If None, restores all available splits.
            backup_timestamp: Specific backup timestamp to restore (e.g., '20260209_143052').
                             If None, restores the most recent backup.
            
        Returns:
            Dictionary containing restoration statistics
        """
        import shutil
        
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # If splits not specified, find all available splits
        if splits is None:
            splits = []
            for split in ["train", "valid", "val", "test"]:
                if (dataset_path / split).exists():
                    splits.append(split)
        
        print("\n" + "="*70)
        print("RESTORE CATEGORIES FROM BACKUP")
        print("="*70)
        
        results = {'restored_splits': {}}
        
        for split in splits:
            split_path = dataset_path / split
            backup_dir = split_path / "annotation_backups"
            
            if not backup_dir.exists():
                print(f"⚠ No backup directory found for split '{split}'")
                continue
            
            # Find backup files
            backup_files = sorted(list(backup_dir.glob('*_backup_*.json')), reverse=True)
            
            if not backup_files:
                print(f"⚠ No backup files found for split '{split}'")
                continue
            
            # Select backup file
            if backup_timestamp:
                # Find backup with specific timestamp
                backup_file = None
                for bf in backup_files:
                    if backup_timestamp in bf.name:
                        backup_file = bf
                        break
                if not backup_file:
                    print(f"⚠ No backup found with timestamp '{backup_timestamp}' for split '{split}'")
                    continue
            else:
                # Use most recent backup
                backup_file = backup_files[0]
            
            # Find current annotation file
            annotation_files = list(split_path.glob('*annotations*.json')) + list(split_path.glob('*.json'))
            annotation_files = [f for f in annotation_files if 'backup' not in f.name]
            
            if not annotation_files:
                print(f"⚠ No annotation file found in split '{split}'")
                continue
            
            annotation_path = annotation_files[0]
            
            print(f"\nRestoring split: {split}")
            print(f"  From backup: {backup_file.name}")
            print(f"  To file: {annotation_path.name}")
            
            # Restore the backup
            shutil.copy2(backup_file, annotation_path)
            print(f"  ✓ Restored successfully")
            
            results['restored_splits'][split] = {
                'backup_file': str(backup_file),
                'restored_to': str(annotation_path)
            }
        
        print("\n" + "="*70)
        print(f"Restored {len(results['restored_splits'])} split(s) from backup")
        print("="*70 + "\n")
        
        return results
    
    def process_dataset(
        self,
        dataset_path: str,
        # Split options
        do_split: bool = False,
        val_ratio: float = 0.2,
        split_seed: int = 42,
        # Flatten categories option
        flatten_categories: bool = False,
        merged_category_name: str = 'worms',
        flatten_splits: Optional[List[str]] = None,
        # Remove augmented options
        remove_augmented: bool = False,
        suffix_pattern: str = '_aug',
        # Augmentation options
        do_augment: bool = False,
        augmentation_factor: int = 2,
        rotation_angles: Optional[List[float]] = None,
        intensity_range: Tuple[float, float] = (0.8, 1.2),
        noise_sigma: float = 0.02,
        combinations: bool = True,
        output_suffix: str = '_aug',
        aug_seed: int = 42,
        # Visualization options
        visualize: bool = False,
        num_images: int = 3,
        display_images: bool = True
    ) -> Dict[str, Any]:
        """
        Process dataset with options to split, flatten categories, remove augmented images, augment, and visualize.
        This is a comprehensive workflow function that handles all common dataset operations.
        
        Args:
            dataset_path: Path to the dataset directory
            
            # Split options
            do_split: Whether to split training data into train/validation (default: False)
            val_ratio: Validation ratio for splitting (default: 0.2 = 20%)
            split_seed: Random seed for splitting (default: 42)
            
            # Flatten categories option
            flatten_categories: Whether to merge all categories into a single category (default: False)
            merged_category_name: Name for the merged category (default: 'worms')
            flatten_splits: List of splits to flatten (default: None = all splits)
            
            # Remove augmented options
            remove_augmented: Whether to remove previously augmented images (default: False)
            suffix_pattern: Suffix pattern to identify augmented images (default: '_aug')
            
            # Augmentation options
            do_augment: Whether to augment the training dataset (default: False)
            augmentation_factor: Number of augmented versions per image (default: 2)
            rotation_angles: List of rotation angles in degrees, None for random (default: None)
            intensity_range: Range for intensity adjustment (min, max) (default: (0.8, 1.2))
            noise_sigma: Standard deviation for Gaussian noise (default: 0.02)
            combinations: Whether to combine multiple augmentations (default: True)
            output_suffix: Suffix for augmented image filenames (default: '_aug')
            aug_seed: Random seed for augmentation (default: 42)
            
            # Visualization options
            visualize: Whether to visualize sample images (default: False)
            num_images: Number of images to visualize (default: 3)
            display_images: Whether to display images interactively (default: True)
            
        Returns:
            Dictionary containing results from all operations performed
            
        Example:
            >>> manager = DatasetManager()
            >>> # Split and augment
            >>> results = manager.process_dataset(
            ...     dataset_path="./datasets/my_dataset",
            ...     do_split=True,
            ...     val_ratio=0.2,
            ...     do_augment=True,
            ...     augmentation_factor=3,
            ...     visualize=True
            ... )
            >>> 
            >>> # Flatten categories to single class
            >>> results = manager.process_dataset(
            ...     dataset_path="./datasets/my_dataset",
            ...     flatten_categories=True,
            ...     merged_category_name='worms'
            ... )
            >>> 
            >>> # Remove old augmentations and re-augment
            >>> results = manager.process_dataset(
            ...     dataset_path="./datasets/my_dataset",
            ...     remove_augmented=True,
            ...     do_augment=True,
            ...     augmentation_factor=5,
            ...     combinations=False
            ... )
        """
        dataset_path = Path(dataset_path)
        results = {
            'dataset_path': str(dataset_path.absolute()),
            'operations': []
        }
        
        print("\n" + "="*70)
        print("DATASET PROCESSING PIPELINE")
        print("="*70)
        print(f"Dataset: {dataset_path}")
        print(f"\nOperations to perform:")
        if do_split:
            print(f"  ✓ Split train/validation ({val_ratio*100:.0f}% validation)")
        if flatten_categories:
            print(f"  ✓ Flatten categories to '{merged_category_name}'")
        if remove_augmented:
            print(f"  ✓ Remove augmented images (pattern: '{suffix_pattern}')")
        if do_augment:
            print(f"  ✓ Augment dataset (factor: {augmentation_factor})")
        if visualize:
            print(f"  ✓ Visualize images ({num_images} samples)")
        if not any([do_split, flatten_categories, remove_augmented, do_augment, visualize]):
            print("  (None - will only load and verify dataset)")
        print("="*70 + "\n")
        
        try:
            # Load and analyze dataset
            print("Loading and analyzing dataset...")
            dataset_info = self.load_dataset(str(dataset_path))
            results['dataset_info'] = dataset_info
            results['operations'].append('load')
            
            # Verify COCO annotations and get detailed statistics
            annotation_stats = {}
            for split_name, split_info in dataset_info["splits"].items():
                if "coco" in split_info.get("annotations", {}):
                    coco_path = split_info["annotations"]["coco"]
                    self.verify_coco_format(coco_path)
                    # Get detailed annotation statistics
                    stats = self.get_annotation_statistics(coco_path, verbose=True)
                    annotation_stats[split_name] = stats
            
            results['annotation_statistics'] = annotation_stats
            
            # Save annotation statistics to file
            if annotation_stats:
                stats_file = dataset_path / "annotation_statistics.json"
                with open(stats_file, 'w') as f:
                    json.dump(annotation_stats, f, indent=2, default=str)
                print(f"\n💾 Annotation statistics saved to: {stats_file}")
                results['statistics_file'] = str(stats_file)
            
            # Split train/validation
            if do_split:
                print("\n" + "="*70)
                print("SPLIT TRAIN/VALIDATION")
                print("="*70)
                split_summary = self.split_train_validation(
                    dataset_path=str(dataset_path),
                    val_ratio=val_ratio,
                    seed=split_seed,
                    train_split_name='train',
                    val_split_name='valid'
                )
                results['split_summary'] = split_summary
                results['operations'].append('split')
            
            # Flatten categories to single category
            if flatten_categories:
                flatten_summary = self.flatten_categories(
                    dataset_path=str(dataset_path),
                    splits=flatten_splits,
                    merged_category_name=merged_category_name,
                    merged_category_id=0
                )
                results['flatten_summary'] = flatten_summary
                results['operations'].append('flatten_categories')
                
                # Re-load dataset info after flattening
                dataset_info = self.load_dataset(str(dataset_path))
                results['dataset_info'] = dataset_info
                
                # Get updated annotation statistics
                annotation_stats = {}
                for split_name, split_info in dataset_info["splits"].items():
                    if "coco" in split_info.get("annotations", {}):
                        coco_path = split_info["annotations"]["coco"]
                        stats = self.get_annotation_statistics(coco_path, verbose=True)
                        annotation_stats[split_name] = stats
                results['annotation_statistics'] = annotation_stats
            
            # Remove previously augmented images
            if remove_augmented:
                print("\n" + "="*70)
                print("REMOVING PREVIOUS AUGMENTATIONS")
                print("="*70)
                remove_summary = self.remove_augmented_images(
                    dataset_path=str(dataset_path),
                    split='train',
                    suffix_pattern=suffix_pattern
                )
                results['remove_summary'] = remove_summary
                results['operations'].append('remove_augmented')
            
            # Augment training data
            if do_augment:
                print("\n" + "="*70)
                print("DATA AUGMENTATION")
                print("="*70)
                aug_summary = self.augment_dataset(
                    dataset_path=str(dataset_path),
                    split='train',
                    augmentation_factor=augmentation_factor,
                    rotation_angles=rotation_angles,
                    intensity_range=intensity_range,
                    noise_sigma=noise_sigma,
                    combinations=combinations,
                    output_suffix=output_suffix,
                    seed=aug_seed
                )
                results['augmentation_summary'] = aug_summary
                results['operations'].append('augment')
            
            # Prepare for RT-DETR training
            prepared_path = self.prepare_for_rtdetr(str(dataset_path))
            results['prepared_path'] = prepared_path
            
            # Visualize sample images
            if visualize:
                print("\n" + "="*70)
                print("VISUALIZING ANNOTATED IMAGES")
                print("="*70)
                self.visualize_dataset_split(
                    dataset_path=str(dataset_path),
                    split='train',
                    num_images=num_images,
                    show_boxes=True,
                    show_masks=True,
                    show_labels=True,
                    display=display_images
                )
                results['operations'].append('visualize')
            
            # Final summary
            print("\n" + "="*70)
            print("PROCESSING COMPLETE!")
            print("="*70)
            print(f"Operations performed: {', '.join(results['operations'])}")
            print(f"Dataset ready at: {prepared_path}")
            print("="*70 + "\n")
            
            results['success'] = True
            return results
            
        except Exception as e:
            print(f"\n❌ Error during processing: {e}")
            import traceback
            traceback.print_exc()
            results['success'] = False
            results['error'] = str(e)
            return results


def main():
    """
    Main function demonstrating the use of process_dataset().
    Edit the parameters below to customize the dataset processing pipeline.
    """
    # ==================== CONFIGURATION ====================
    # Update this path to point to your dataset
    DATASET_PATH = "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\datasets\\WormBodyDetection.v10i.coco-segmentation"
    
    # Initialize the manager
    manager = DatasetManager()
    
    # ==================== PROCESSING OPTIONS ====================
    # Set these options to control what operations are performed
    
    # Option 1: Just load and analyze (default - all operations disabled)
    # results = manager.process_dataset(
    #     dataset_path=DATASET_PATH
    # )
    
    # Option 2: Split dataset into train/validation
    # results = manager.process_dataset(
    #     dataset_path=DATASET_PATH,
    #     do_split=True,
    #     val_ratio=0.2,  # 20% for validation
    #     split_seed=42
    # )
    
    # Option 3: Remove previous augmentations and re-augment
    # results = manager.process_dataset(
    #     dataset_path=DATASET_PATH,
    #     remove_augmented=True,
    #     do_augment=True,
    #     augmentation_factor=3,
    #     combinations=True
    # )
    
    # Option 4: Flatten all categories to single 'worms' category
    # results = manager.process_dataset(
    #     dataset_path=DATASET_PATH,
    #     flatten_categories=True,
    #     merged_category_name='worms',
    #     flatten_splits=['train', 'valid']  # Or None for all splits
    # )
    
    # Option 5: Full pipeline - split, augment, and visualize
    results = manager.process_dataset(
        dataset_path=DATASET_PATH,
        # Split options
        do_split=False,  # Set to True to split train/validation
        val_ratio=0.2,
        split_seed=42,
        # Flatten categories options
        flatten_categories=False,  # Set to True to merge all categories into one
        merged_category_name='worms',
        flatten_splits=None,  # None = all splits, or specify like ['train', 'valid']
        # Remove augmented options
        remove_augmented=True,  # Set to True to remove previous augmentations
        suffix_pattern='_aug',
        # Augmentation options
        do_augment=True,  # Set to True to augment training data
        augmentation_factor=1,  # Number of augmented versions per image
        rotation_angles=None,  # Use None for random, or specify like [-15, 15, -30, 30]
        intensity_range=(0.7, 1.3),  # Brightness adjustment range
        noise_sigma=0.05,  # Gaussian noise amount
        combinations=True,  # Combine all augmentations
        output_suffix='_aug',
        aug_seed=42,
        # Visualization options
        visualize=True,  # Set to True to visualize samples
        num_images=3,
        display_images=True
    )
    
    # Check results
    if results['success']:
        print(f"\n✓ All operations completed successfully!")
        print(f"Operations performed: {', '.join(results['operations'])}")
        return 0
    else:
        print(f"\n❌ Processing failed: {results.get('error', 'Unknown error')}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())