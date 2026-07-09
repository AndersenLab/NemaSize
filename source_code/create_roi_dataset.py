"""
Create ROI Dataset from YOLO Segmentation Annotations

This script processes an existing YOLO segmentation dataset and creates a new dataset
where each annotation mask is cropped into a square ROI image centered on the mask centroid.
The ROI dimension is 20% larger than the larger of the mask's width or height.
"""

import os
import cv2
import json
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import random
from shapely.geometry import Polygon, box
from shapely.validation import make_valid
from shapely.ops import unary_union


def parse_yolo_segmentation(label_path, img_width, img_height):
    """
    Parse YOLO segmentation format labels.
    
    Args:
        label_path: Path to the YOLO label file
        img_width: Original image width
        img_height: Original image height
    
    Returns:
        List of annotations, each containing class_id and polygon points
    """
    annotations = []
    
    if not os.path.exists(label_path):
        return annotations
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # Need at least 7 values: class_id (1) + 3 points (6 coords)
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            
            # Convert normalized coordinates to pixel coordinates
            points = []
            for i in range(0, len(coords), 2):
                x = coords[i] * img_width
                y = coords[i + 1] * img_height
                points.append([x, y])
            
            annotations.append({
                'class_id': class_id,
                'points': np.array(points, dtype=np.float32)
            })
    
    return annotations


def calculate_centroid(points):
    """Calculate the centroid of a polygon as the center of its bounding box."""
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    # Center of bounding box
    cx = (np.min(x_coords) + np.max(x_coords)) / 2
    cy = (np.min(y_coords) + np.max(y_coords)) / 2
    
    return np.array([cx, cy])


def calculate_bbox_dimensions(points):
    """Calculate width and height of the bounding box."""
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    width = np.max(x_coords) - np.min(x_coords)
    height = np.max(y_coords) - np.min(y_coords)
    
    return width, height


def crop_roi(image, centroid, roi_size):
    """
    Crop a square ROI centered on the centroid.
    
    Args:
        image: Input image
        centroid: (x, y) coordinates of the center
        roi_size: Size of the square ROI
    
    Returns:
        Cropped ROI image, top-left corner coordinates
    """
    img_height, img_width = image.shape[:2]
    cx, cy = centroid
    half_size = roi_size / 2
    
    # Calculate ROI boundaries
    x1 = int(cx - half_size)
    y1 = int(cy - half_size)
    x2 = int(cx + half_size)
    y2 = int(cy + half_size)
    
    # Handle boundary cases with padding
    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_width)
    pad_bottom = max(0, y2 - img_height)
    
    # Clip to image boundaries
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(img_width, x2)
    y2_clip = min(img_height, y2)
    
    # Crop the region
    roi = image[y1_clip:y2_clip, x1_clip:x2_clip]
    
    # Add padding if necessary to maintain square shape
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        roi = cv2.copyMakeBorder(
            roi, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
    
    return roi, (x1, y1)


def transform_annotation_to_roi(points, roi_top_left, roi_size, min_area_ratio=0.001, is_target=False):
    """
    Transform annotation coordinates from original image to ROI coordinates.
    Properly clips polygons that extend beyond ROI boundaries.
    
    Args:
        points: Original polygon points
        roi_top_left: (x, y) of ROI top-left corner in original image
        roi_size: Size of the ROI
        min_area_ratio: Minimum area ratio (relative to ROI) to keep a polygon piece (default: 0.001)
        is_target: If True, keeps all pieces regardless of area (for center target object)
    
    Returns:
        List of transformed and normalized points for YOLO format (clipped to ROI bounds).
        Returns a list because clipping may result in multiple polygon pieces.
    """
    x_offset, y_offset = roi_top_left
    
    # Calculate minimum area threshold
    roi_area = roi_size * roi_size
    min_area = roi_area * min_area_ratio

    # For the target annotation (the centered worm), bypass Shapely entirely.
    # The ROI is sized to contain the target, so no clipping is needed.
    # Using Shapely on self-intersecting polygons (e.g. worms whose annotation
    # trace crosses itself) would trigger make_valid() which splits the polygon
    # into a MultiPolygon — incorrectly fragmenting one worm into multiple pieces.
    if is_target:
        transformed_points = points.copy()
        transformed_points[:, 0] -= x_offset
        transformed_points[:, 1] -= y_offset
        transformed_points[:, 0] /= roi_size
        transformed_points[:, 1] /= roi_size
        transformed_points = np.clip(transformed_points, 0, 1)
        return [transformed_points]

    # For neighbor annotations: use Shapely to clip against the ROI boundary.
    # Repair strategy for invalid (e.g. self-intersecting) polygons:
    #   1. Try make_valid() — if it returns a single Polygon, use it directly.
    #   2. If make_valid() fragments into multiple pieces (e.g. self-intersecting
    #      omega worm), union them back together.  This recovers the true worm
    #      footprint — much tighter than the convex hull.
    #   3. If the union is still a MultiPolygon (pieces are genuinely disconnected),
    #      try a small buffer to close any hairline gaps, then union again.
    #   4. Last resort: convex hull (avoids fragmenting the neighbor entirely).
    try:
        original_polygon = Polygon(points)
        if not original_polygon.is_valid:
            repaired = make_valid(original_polygon)
            if repaired.geom_type == 'Polygon':
                original_polygon = repaired
            else:
                # Union the pieces back into one polygon — preserves the actual
                # worm outline rather than inflating to the convex hull.
                merged = unary_union(repaired)
                if merged.geom_type == 'Polygon':
                    original_polygon = merged
                else:
                    # Pieces are still disjoint after union (very rare).
                    # Apply a tiny outward then inward buffer to close hairline
                    # gaps at crossing points, then try unioning once more.
                    buf_size = max(
                        (original_polygon.bounds[2] - original_polygon.bounds[0]),
                        (original_polygon.bounds[3] - original_polygon.bounds[1])
                    ) * 0.005  # 0.5% of bounding-box size
                    merged_buf = unary_union(repaired).buffer(buf_size).buffer(-buf_size)
                    if merged_buf.geom_type == 'Polygon':
                        original_polygon = merged_buf
                    else:
                        # Absolute last resort: convex hull.
                        original_polygon = original_polygon.convex_hull
    except Exception as e:
        # Fallback to simple transform if Shapely polygon creation fails entirely.
        # WARNING: cut lines will NOT be flat — out-of-bounds vertices are snapped
        # to the ROI edge rather than properly clipped.
        print(f"Warning: Shapely polygon creation/validation failed (using snap fallback, cut lines may be incorrect): {e}")
        transformed_points = points.copy()
        transformed_points[:, 0] -= x_offset
        transformed_points[:, 1] -= y_offset
        transformed_points[:, 0] /= roi_size
        transformed_points[:, 1] /= roi_size
        transformed_points = np.clip(transformed_points, 0, 1)
        return [transformed_points]  # Return as list for consistency
    
    # Create ROI bounding box in original image coordinates
    roi_box = box(x_offset, y_offset, x_offset + roi_size, y_offset + roi_size)
    
    # Clip polygon to ROI boundaries
    try:
        clipped_polygon = original_polygon.intersection(roi_box)
        
        # Handle different geometry types
        if clipped_polygon.is_empty:
            return []
        
        # Collect all valid polygon pieces
        polygon_pieces = []
        
        if clipped_polygon.geom_type == 'Polygon':
            # Apply the same area threshold as MultiPolygon pieces — a neighbor that
            # barely grazes the ROI corner produces a single tiny Polygon, not a
            # MultiPolygon, so we still want to discard it if it's too small.
            polygon_pieces = [clipped_polygon] if clipped_polygon.area >= min_area else []
        elif clipped_polygon.geom_type == 'MultiPolygon':
            # Keep pieces that meet the minimum area threshold.
            # is_target is never True here (target returns early above).
            polygon_pieces = [poly for poly in clipped_polygon.geoms if poly.area >= min_area]
        elif clipped_polygon.geom_type in ['LineString', 'MultiLineString', 'Point', 'MultiPoint']:
            # Degenerate case - polygon was clipped to line/point, skip it
            return []
        else:
            # Fallback for other geometry types
            return []
        
        # Process each polygon piece
        result_polygons = []
        for poly in polygon_pieces:
            # Ensure polygon is valid and properly oriented
            if not poly.is_valid:
                poly = make_valid(poly)
            
            # Skip if became invalid after cleaning
            if poly.is_empty or poly.geom_type != 'Polygon':
                continue
            
            # Extract exterior coordinates (shapely polygons are always closed)
            clipped_coords = np.array(poly.exterior.coords[:-1])  # Remove duplicate last point
            
            # Transform to ROI coordinate system
            transformed_points = clipped_coords.copy()
            transformed_points[:, 0] -= x_offset
            transformed_points[:, 1] -= y_offset
            
            # Normalize to [0, 1]
            transformed_points[:, 0] /= roi_size
            transformed_points[:, 1] /= roi_size
            
            # Final safety clip (should be very close to [0,1] already)
            transformed_points = np.clip(transformed_points, 0, 1)
            
            # Remove consecutive duplicate points that might have appeared after normalization
            unique_consecutive = [transformed_points[0]]
            for i in range(1, len(transformed_points)):
                if not np.allclose(transformed_points[i], unique_consecutive[-1], atol=1e-6):
                    unique_consecutive.append(transformed_points[i])
            
            # Ensure polygon is closed by checking if first and last points are different
            # (YOLO format assumes implicit closure, so we don't add duplicate point)
            transformed_points = np.array(unique_consecutive)
            
            # Validate minimum points for a valid polygon
            if len(transformed_points) >= 3:
                result_polygons.append(transformed_points)
        
        return result_polygons
        
    except Exception as e:
        # Fallback to simple transformation if clipping fails.
        # WARNING: cut lines will NOT be flat — out-of-bounds vertices are snapped
        # to the ROI edge rather than properly clipped.
        print(f"Warning: Shapely polygon clipping failed (using snap fallback, cut lines may be incorrect): {e}")
        transformed_points = points.copy()
        transformed_points[:, 0] -= x_offset
        transformed_points[:, 1] -= y_offset
        transformed_points[:, 0] /= roi_size
        transformed_points[:, 1] /= roi_size
        transformed_points = np.clip(transformed_points, 0, 1)
        return [transformed_points]  # Return as list for consistency


def polygon_intersects_roi(points, roi_bbox):
    """
    Check if a polygon intersects with the ROI bounding box.
    
    Args:
        points: Polygon points (N x 2 array)
        roi_bbox: (x1, y1, x2, y2) of ROI
    
    Returns:
        True if polygon intersects with ROI
    """
    x1, y1, x2, y2 = roi_bbox
    
    # Check if any point is inside the ROI
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    
    points_inside = np.any(
        (x_coords >= x1) & (x_coords <= x2) & 
        (y_coords >= y1) & (y_coords <= y2)
    )
    
    if points_inside:
        return True
    
    # Check if ROI corners are inside the polygon (using cv2.pointPolygonTest)
    roi_corners = np.array([
        [x1, y1], [x2, y1], [x2, y2], [x1, y2]
    ])
    
    for corner in roi_corners:
        if cv2.pointPolygonTest(points.astype(np.float32), tuple(corner), False) >= 0:
            return True
    
    return False


def save_yolo_annotation(label_path, annotations_list):
    """Save multiple annotations in YOLO segmentation format.
    
    Args:
        label_path: Path to save the label file
        annotations_list: List of dicts containing 'class_id' and 'normalized_points'
    """
    with open(label_path, 'w') as f:
        for annotation in annotations_list:
            class_id = annotation['class_id']
            normalized_points = annotation['normalized_points']
            coords_str = ' '.join([f'{coord:.6f}' for point in normalized_points for coord in point])
            f.write(f'{class_id} {coords_str}\n')


def sanitize_folder_name(name):
    """Return a Windows-safe folder name derived from a class label."""
    invalid_chars = '<>:"/\\|?*'
    cleaned = ''.join('_' if ch in invalid_chars else ch for ch in str(name)).strip()
    cleaned = cleaned.rstrip('.')
    return cleaned if cleaned else 'unnamed_class'


def load_class_names_from_source_dataset(source_dataset_path):
    """Load class names from the source dataset data.yaml."""
    candidate_paths = [
        os.path.join(source_dataset_path, 'data.yaml'),
        os.path.join(source_dataset_path, 'train', 'data.yaml'),
        os.path.join(source_dataset_path, 'valid', 'data.yaml'),
        os.path.join(source_dataset_path, 'val', 'data.yaml'),
        os.path.join(source_dataset_path, 'test', 'data.yaml'),
    ]

    data_yaml_path = None
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            data_yaml_path = candidate
            break

    if data_yaml_path is None:
        checked = '\n'.join(candidate_paths)
        raise FileNotFoundError(f"Could not find source data.yaml. Checked:\n{checked}")

    print(f"Using source class map from: {data_yaml_path}")

    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}

    names = data.get('names')
    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        parsed = {}
        for key, value in names.items():
            idx = int(key)
            parsed[idx] = str(value)
        max_idx = max(parsed.keys())
        return [parsed.get(i, f'class_{i}') for i in range(max_idx + 1)]

    nc = data.get('nc')
    if isinstance(nc, int) and nc > 0:
        return [f'class_{i}' for i in range(nc)]

    raise ValueError(f"Could not parse class names from: {data_yaml_path}")


def find_present_class_ids(source_dataset_path, split_names):
    """Find class IDs that actually appear in source label files."""
    present_ids = set()

    for split in split_names:
        labels_dir = os.path.join(source_dataset_path, split, 'labels')
        if not os.path.exists(labels_dir):
            continue

        label_files = [
            f for f in os.listdir(labels_dir)
            if f.lower().endswith('.txt')
        ]

        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 1:
                            continue
                        try:
                            class_id = int(parts[0])
                            present_ids.add(class_id)
                        except ValueError:
                            continue
            except Exception as e:
                print(f"Warning: Failed to read label file {label_path}: {e}")

    return sorted(present_ids)


def process_dataset(source_path, output_path, split_name, roi_size_scale_factor=1.2,
                   output_image_size=640, min_area_ratio=0.001,
                   class_output_paths=None, class_names=None,
                   separate_by_target_class=False):
    """
    Process a single split (train/test/valid) of the dataset.
    
    Args:
        source_path: Path to the source dataset split
        output_path: Path to save the ROI dataset
        split_name: Name of the split (train/test/valid)
        roi_size_scale_factor: Scale factor for ROI size (default: 1.2 for 20% padding)
        output_image_size: Final size for all output ROI images (default: 640x640)
        min_area_ratio: Minimum area ratio to keep polygon pieces after clipping (default: 0.001)
    """
    images_dir = os.path.join(source_path, split_name, 'images')
    labels_dir = os.path.join(source_path, split_name, 'labels')
    
    if separate_by_target_class:
        if not class_output_paths:
            raise ValueError("class_output_paths is required when separate_by_target_class=True")
        output_images_dir = None
        output_labels_dir = None
        split_catalogs = {class_id: {} for class_id in class_output_paths.keys()}
    else:
        output_images_dir = os.path.join(output_path, split_name, 'images')
        output_labels_dir = os.path.join(output_path, split_name, 'labels')

        # Create output directories
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
    
    # Get list of images
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"\nProcessing {split_name} split: {len(image_files)} images")
    
    total_rois = 0
    catalog = {}  # ROI transformation catalog for this split
    
    for img_filename in tqdm(image_files, desc=f"Processing {split_name}"):
        # Read image
        img_path = os.path.join(images_dir, img_filename)
        image = cv2.imread(img_path)
        
        if image is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Get corresponding label file
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        # Parse annotations
        annotations = parse_yolo_segmentation(label_path, img_width, img_height)
        
        # Process each annotation as a target (center ROI on each annotation)
        for idx, target_annotation in enumerate(annotations):
            target_points = target_annotation['points']
            target_class_id = target_annotation['class_id']

            if separate_by_target_class and target_class_id not in class_output_paths:
                print(f"Warning: target class {target_class_id} not found in class_output_paths, skipping")
                continue

            target_class_name = None
            if class_names is not None and 0 <= target_class_id < len(class_names):
                target_class_name = class_names[target_class_id]
            
            # Calculate centroid of target annotation
            centroid = calculate_centroid(target_points)
            
            # Calculate bounding box dimensions of target annotation
            width, height = calculate_bbox_dimensions(target_points)
            
            # Calculate ROI size (scale factor applied to max dimension)
            max_dimension = max(width, height)
            roi_size = int(max_dimension * roi_size_scale_factor)
            
            # Ensure minimum ROI size
            roi_size = max(roi_size, 32)
            
            # Calculate ROI bounding box
            cx, cy = centroid
            half_size = roi_size / 2
            roi_bbox = (
                cx - half_size,  # x1
                cy - half_size,  # y1
                cx + half_size,  # x2
                cy + half_size   # y2
            )
            
            # Crop ROI
            roi_image, roi_top_left = crop_roi(image, centroid, roi_size)
            
            # Resize ROI to standard output size (e.g., 640x640)
            # Use INTER_AREA when downscaling (best quality), INTER_LANCZOS4 when upscaling
            interp = cv2.INTER_AREA if roi_size > output_image_size else cv2.INTER_LANCZOS4
            roi_image_resized = cv2.resize(roi_image, (output_image_size, output_image_size),
                                          interpolation=interp)
            
            # Find ALL annotations that intersect with this ROI
            roi_annotations = []
            for annotation in annotations:
                points = annotation['points']
                class_id = annotation['class_id']

                # In per-class mode, keep only same-class objects in that class dataset.
                if separate_by_target_class and class_id != target_class_id:
                    continue
                
                # Check if this annotation intersects with the ROI
                if polygon_intersects_roi(points, roi_bbox):
                    # Check if this is the target annotation (centered object)
                    is_target_annotation = (points is target_points)
                    
                    # Transform annotation to ROI coordinates (may return multiple pieces)
                    # Target annotation pieces are always kept regardless of size
                    transformed_polygons = transform_annotation_to_roi(points, roi_top_left, roi_size, min_area_ratio, is_target_annotation)
                    
                    # Add each valid polygon piece as a separate annotation
                    for transformed_points in transformed_polygons:
                        # Validate the transformed polygon has at least 3 unique points
                        if len(transformed_points) >= 3:
                            # Check for unique points to avoid degenerate polygons
                            unique_points = np.unique(transformed_points, axis=0)
                            if len(unique_points) >= 3:
                                output_class_id = 0 if separate_by_target_class else class_id
                                roi_annotations.append({
                                    'class_id': output_class_id,
                                    'normalized_points': transformed_points
                                })
            
            # Skip this ROI if no valid annotations (shouldn't happen for target)
            if len(roi_annotations) == 0:
                continue
            
            # Generate output filename
            base_name = os.path.splitext(img_filename)[0]
            roi_key = f"{base_name}_roi_{idx}"
            roi_img_filename = f"{roi_key}.png"
            roi_label_filename = f"{roi_key}.txt"

            if separate_by_target_class:
                class_output_base = class_output_paths[target_class_id]
                current_output_images_dir = os.path.join(class_output_base, split_name, 'images')
                current_output_labels_dir = os.path.join(class_output_base, split_name, 'labels')
                os.makedirs(current_output_images_dir, exist_ok=True)
                os.makedirs(current_output_labels_dir, exist_ok=True)
            else:
                current_output_images_dir = output_images_dir
                current_output_labels_dir = output_labels_dir
            
            # Save resized ROI image (lossless PNG, compression=0 for fastest I/O)
            roi_img_path = os.path.join(current_output_images_dir, roi_img_filename)
            cv2.imwrite(roi_img_path, roi_image_resized, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            
            # Save ROI annotations (all annotations in this ROI)
            # Note: annotations are already normalized [0,1], so they remain valid after resize
            roi_label_path = os.path.join(current_output_labels_dir, roi_label_filename)
            save_yolo_annotation(roi_label_path, roi_annotations)
            
            # Record catalog entry for this ROI
            catalog_entry = {
                "source_image":      img_path,
                "original_width":    img_width,
                "original_height":   img_height,
                "split":             split_name,
                "annotation_index":  idx,
                "class_id":          target_class_id,
                "class_name":        target_class_name,
                # Target annotation bounding box in original image pixels
                "bbox_original": [
                    round(float(np.min(target_points[:, 0])), 2),
                    round(float(np.min(target_points[:, 1])), 2),
                    round(float(np.max(target_points[:, 0])), 2),
                    round(float(np.max(target_points[:, 1])), 2),
                ],
                "centroid_original": [round(float(cx), 2), round(float(cy), 2)],
                "roi_top_left":      [roi_top_left[0], roi_top_left[1]],
                "roi_size_original": roi_size,
                "output_image_size": output_image_size,
                "scale_factor":      round(output_image_size / roi_size, 6),
                "output_image_path": roi_img_path,
            }

            if separate_by_target_class:
                split_catalogs[target_class_id][roi_key] = catalog_entry
            else:
                catalog[roi_key] = catalog_entry
            
            total_rois += 1

    if separate_by_target_class:
        for class_id, class_catalog in split_catalogs.items():
            class_output_base = class_output_paths[class_id]
            catalog_path = os.path.join(class_output_base, split_name, 'roi_catalog.json')
            with open(catalog_path, 'w') as f:
                json.dump(class_catalog, f, indent=2)
            class_name = class_names[class_id] if class_names and class_id < len(class_names) else f'class_{class_id}'
            print(f"Catalog written to: {catalog_path}  ({len(class_catalog)} entries, class={class_name})")
    else:
        # Write per-split catalog JSON
        catalog_path = os.path.join(output_path, split_name, 'roi_catalog.json')
        with open(catalog_path, 'w') as f:
            json.dump(catalog, f, indent=2)
        print(f"Catalog written to: {catalog_path}  ({len(catalog)} entries)")

    print(f"{split_name} split: Created {total_rois} ROI images")
    return total_rois


def create_data_yaml(output_path, class_names=None, flat_structure=False, split_names=None):
    """Create data.yaml file for the ROI dataset."""
    if class_names is None:
        class_names = ['worm']  # Default class name

    if flat_structure:
        train_path = 'images'
        val_path = 'images'
        test_path = 'images'
    else:
        split_set = set(split_names or ['train', 'valid', 'test'])

        train_path = 'train/images' if 'train' in split_set else None
        if 'valid' in split_set:
            val_path = 'valid/images'
        elif 'val' in split_set:
            val_path = 'val/images'
        else:
            val_path = None
        test_path = 'test/images' if 'test' in split_set else None
    
    yaml_lines = [
        "# ROI Dataset Configuration",
        f"path: {output_path}"
    ]
    if train_path is not None:
        yaml_lines.append(f"train: {train_path}")
    if val_path is not None:
        yaml_lines.append(f"val: {val_path}")
    if test_path is not None:
        yaml_lines.append(f"test: {test_path}")
    yaml_lines.extend([
        "",
        "# Classes",
        "names:"
    ])

    yaml_content = "\n".join(yaml_lines) + "\n"
    
    for idx, name in enumerate(class_names):
        yaml_content += f"  {idx}: {name}\n"
    
    yaml_path = os.path.join(output_path, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nCreated data.yaml at {yaml_path}")


def visualize_roi_samples(output_path, num_samples=20, splits=['train', 'valid', 'test'], image_paths=None, random_seed=None):
    """
    Visualize samples from the generated ROI dataset with annotations overlaid.
    
    Args:
        output_path: Path to the ROI dataset
        num_samples: Number of random samples to visualize (ignored when image_paths is provided)
        splits: List of splits to sample from (ignored when image_paths is provided)
        image_paths: Optional list of specific image file paths to visualize. When provided,
                     these images are shown instead of random samples. The corresponding label
                     file is expected alongside the image (same dir, 'labels' instead of
                     'images', .txt extension).
        random_seed: Optional integer seed for reproducible random sampling (default: None).
    """
    # --- Mode: specific image paths provided ---
    if image_paths:
        print(f"\n{'=' * 60}")
        print(f"Visualizing {len(image_paths)} specified image(s)")
        print(f"{'=' * 60}")

        sampled_images = []
        for img_path in image_paths:
            img_path = os.path.normpath(img_path)
            if not os.path.exists(img_path):
                print(f"Warning: Image not found, skipping: {img_path}")
                continue

            # Derive label path: swap 'images' directory component for 'labels', change ext
            img_p = Path(img_path)
            parts = img_p.parts
            # Replace the last occurrence of 'images' folder with 'labels'
            parts_list = list(parts)
            for i in range(len(parts_list) - 1, -1, -1):
                if parts_list[i].lower() == 'images':
                    parts_list[i] = 'labels'
                    break
            label_path = str(Path(*parts_list).with_suffix('.txt'))

            if not os.path.exists(label_path):
                print(f"Warning: Label file not found for {img_path} (expected {label_path})")
                continue

            # Infer split name from path for display
            split_name = 'unknown'
            for part in parts:
                if part.lower() in ('train', 'valid', 'test', 'val'):
                    split_name = part
                    break

            sampled_images.append({
                'image': img_path,
                'label': label_path,
                'split': split_name,
                'filename': img_p.name
            })

        if len(sampled_images) == 0:
            print("No valid images found to visualize!")
            return

        num_samples = len(sampled_images)

    # --- Mode: random sampling ---
    else:
        print(f"\n{'=' * 60}")
        print(f"Visualizing {num_samples} random ROI samples")
        print(f"{'=' * 60}")

        # Collect all image paths from all splits
        all_image_paths = []
        for split in splits:
            images_dir = os.path.join(output_path, split, 'images')
            labels_dir = os.path.join(output_path, split, 'labels')

            if not os.path.exists(images_dir):
                continue

            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

            for img_file in image_files:
                img_path = os.path.join(images_dir, img_file)
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)

                if os.path.exists(label_path):
                    all_image_paths.append({
                        'image': img_path,
                        'label': label_path,
                        'split': split,
                        'filename': img_file
                    })

        if len(all_image_paths) == 0:
            print("No images found to visualize!")
            return

        num_samples = min(num_samples, len(all_image_paths))
        if random_seed is not None:
            random.seed(random_seed)
        sampled_images = random.sample(all_image_paths, num_samples)
    
    # Define colors for different annotations (BGR format)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 128, 0), (0, 128, 255), (128, 255, 0)
    ]
    
    # Display each image one by one
    for idx, sample in enumerate(sampled_images, 1):
        # Read image
        image = cv2.imread(sample['image'])
        if image is None:
            continue
        
        img_height, img_width = image.shape[:2]
        
        # Read annotations
        annotations = []
        with open(sample['label'], 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  # Need at least 7 values: class_id (1) + 3 points (6 coords)
                    print(f"  Warning: Skipping annotation with only {len(parts)} values in {sample['filename']}")
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Convert normalized coordinates to pixel coordinates
                points = []
                for i in range(0, len(coords), 2):
                    x = int(coords[i] * img_width)
                    y = int(coords[i + 1] * img_height)
                    points.append([x, y])
                
                # Validate we have enough unique points
                points_array = np.array(points, dtype=np.int32)
                unique_points = np.unique(points_array, axis=0)
                
                if len(unique_points) < 3:
                    print(f"  Warning: Annotation has only {len(unique_points)} unique points after discretization in {sample['filename']}")
                    continue
                
                annotations.append({
                    'class_id': class_id,
                    'points': points_array
                })
        
        # Draw annotations
        for ann_idx, annotation in enumerate(annotations):
            points = annotation['points']
            
            # Additional validation before drawing
            if len(points) < 2:
                print(f"  Warning: Cannot draw polygon with {len(points)} points")
                continue
            
            color = colors[ann_idx % len(colors)]
            
            # Draw point-connecting lines (polylines) without fill, useful for self-overlapping objects
            try:
                # Draw closed polygon boundary as lines connecting each vertex
                cv2.polylines(image, [points], isClosed=True, color=color, thickness=4, lineType=cv2.LINE_AA)
                
                # Draw a small circle at each vertex
                for pt in points:
                    cv2.circle(image, (int(pt[0]), int(pt[1])), 3, color, -1)
                
                # Draw centroid using bounding box center (consistent with dataset creation)
                x_coords = points[:, 0]
                y_coords = points[:, 1]
                cx = int((np.min(x_coords) + np.max(x_coords)) / 2)
                cy = int((np.min(y_coords) + np.max(y_coords)) / 2)
                cv2.circle(image, (cx, cy), 5, color, -1)
            except Exception as e:
                print(f"  Warning: Failed to draw annotation: {e}")
        
        # Create a canvas with black margin around the image and white text area below
        margin = 20       # black border on top, left, right
        text_height = 100 # white text area at the bottom
        canvas_h = margin + img_height + margin + text_height
        canvas_w = margin + img_width + margin
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)  # all black by default
        # White text strip at the bottom
        canvas[margin + img_height + margin:, :] = 255
        
        # Place the image inside the black margin
        canvas[margin:margin + img_height, margin:margin + img_width] = image
        
        # Add title text in the white text area
        title_text = f"[{idx}/{num_samples}] {sample['split']}: {sample['filename']}"
        count_text = f"{len(annotations)} object(s)"
        text_y0 = margin + img_height + margin
        cv2.putText(canvas, title_text, (5, text_y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(canvas, count_text, (5, text_y0 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Display image
        cv2.imshow('ROI Samples - Press any key for next, ESC to skip remaining', canvas)
        print(f"Displaying image {idx}/{num_samples}: {sample['filename']}")
        
        key = cv2.waitKey(0)
        if key == 27:  # ESC key
            print("Visualization cancelled by user")
            break
    
    cv2.destroyAllWindows()
    print(f"\nVisualization complete")



def main():
    # ========== CONFIGURATION ==========
    # Source dataset path
    # source_dataset = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyDetection.v10i.coco-segmentation"
    # source_dataset = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\small_test"
    source_dataset = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\WormBodyDetection.v10i.yolo26"


    # Output dataset configuration
    #output_base = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
    #output_dataset_name = "WormBodyROI_fix_overlap_catalog"  # Name of the output ROI dataset folder

    output_base = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test"
    output_dataset_name = "GT_rois"  # Name of the output ROI dataset folder
    
    # ROI size scale factor: ROI dimension = max(width, height) * roi_size_scale_factor
    # For example, 1.2 means 20% larger, 1.5 means 50% larger
    roi_size_scale_factor = 1.2
    
    # Final output image size (all ROIs will be resized to this dimension)
    output_image_size = 640  # pixels (square: 640x640)
    
    # Minimum area ratio for keeping polygon pieces after clipping
    # Pieces smaller than this ratio (relative to ROI area) will be discarded
    # For example, 0.001 means keep pieces larger than 0.1% of ROI area
    min_area_ratio = 0.001
    
    # Dataset construction
    build_dataset = True  # Set to False to skip dataset construction

    # If True, create one output dataset folder per source class.
    # Each class folder contains images/, labels/, data.yaml and stores
    # only ROIs centered on that class.
    separate_by_target_class = True
    
    # Visualization settings
    visualize_samples = True  # Set to False to skip visualizations
    num_visualization_samples = 3  # Number of random samples to visualize
    visualization_random_seed = 42  # Set to None for non-reproducible random sampling
    # Specific images to visualize (overrides random sampling when non-empty).
    # Provide a list of absolute image file paths, e.g.:
    #   specific_visualization_images = [
    #       r"C:\path\to\dataset\train\images\img001.jpg",
    #       r"C:\path\to\dataset\valid\images\img042.jpg",
    #   ]
    specific_visualization_images = [
        #r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap\train\images\154_png.rf.14ad9631fdb98926098b64b2cc23b499_roi_15.jpg"
    ]  # Empty list = use random sampling
    # ===================================
    
    output_path = os.path.join(output_base, output_dataset_name)
    
    print("=" * 60)
    print("ROI Dataset Creation")
    print("=" * 60)
    print(f"Source dataset: {source_dataset}")
    print(f"Output dataset: {output_path}")
    print(f"ROI size scale factor: {roi_size_scale_factor}")
    print(f"Output image size: {output_image_size}x{output_image_size}")
    print(f"Min area ratio: {min_area_ratio}")
    print(f"Build dataset: {build_dataset}")
    print(f"Separate by target class: {separate_by_target_class}")
    print(f"Visualize samples: {visualize_samples}")
    print("=" * 60)
    
    total_all_rois = 0
    available_splits = []
    class_output_paths = {}
    
    # Build dataset if enabled
    if build_dataset:
        # Mirror only splits that actually exist in the source dataset.
        split_candidates = ['train', 'valid', 'val', 'test']
        available_splits = [
            split for split in split_candidates
            if os.path.exists(os.path.join(source_dataset, split, 'images'))
            and os.path.exists(os.path.join(source_dataset, split, 'labels'))
        ]
        if len(available_splits) == 0:
            raise FileNotFoundError(
                f"No valid source splits found under {source_dataset}. "
                "Expected split/images and split/labels directories."
            )

        print(f"Detected source splits: {available_splits}")

        class_names = load_class_names_from_source_dataset(source_dataset)
        present_class_ids = find_present_class_ids(source_dataset, available_splits)
        print(f"Detected class IDs present in source labels: {present_class_ids}")

        if separate_by_target_class and len(present_class_ids) == 0:
            print("Warning: No class IDs found in source labels. No class folders will be created.")

        if separate_by_target_class:
            print(f"Detected classes from source data.yaml: {class_names}")
            for class_id in present_class_ids:
                if not (0 <= class_id < len(class_names)):
                    print(f"Warning: Class id {class_id} not found in source data.yaml names, skipping")
                    continue

                class_name = class_names[class_id]
                class_folder = sanitize_folder_name(class_name)
                class_output_base = os.path.join(output_path, class_folder)
                for split in available_splits:
                    os.makedirs(os.path.join(class_output_base, split, 'images'), exist_ok=True)
                    os.makedirs(os.path.join(class_output_base, split, 'labels'), exist_ok=True)
                create_data_yaml(
                    class_output_base,
                    class_names=['worm'],
                    flat_structure=False,
                    split_names=available_splits
                )
                class_output_paths[class_id] = class_output_base

        # Process only detected splits.
        for split in available_splits:
            num_rois = process_dataset(source_dataset, output_path, split,
                                      roi_size_scale_factor, output_image_size, min_area_ratio,
                                      class_output_paths=class_output_paths,
                                      class_names=class_names,
                                      separate_by_target_class=separate_by_target_class)
            total_all_rois += num_rois
        
        # Create data.yaml for non-separated mode
        if not separate_by_target_class:
            create_data_yaml(output_path, class_names=class_names, split_names=available_splits)
        
        print("\n" + "=" * 60)
        print(f"ROI Dataset Creation Complete!")
        print(f"Total ROI images created: {total_all_rois}")
        print(f"Output location: {output_path}")
        print("=" * 60)
    else:
        print("\nSkipping dataset construction (build_dataset = False)")
        # Check if output dataset exists for visualization
        if not os.path.exists(output_path):
            print(f"Warning: Output dataset does not exist at {output_path}")
            print("Set build_dataset = True to create it first.")
            visualize_samples = False
    
    # Visualize samples
    if visualize_samples:
        if specific_visualization_images:
            visualize_roi_samples(
                output_path,
                num_samples=num_visualization_samples,
                image_paths=specific_visualization_images,
                random_seed=visualization_random_seed
            )
        elif separate_by_target_class:
            # Sample across all class sub-datasets so visualization works in class-separated mode.
            all_class_images = []
            class_roots = list(class_output_paths.values())

            # If not built in this run, discover class folders from disk.
            if len(class_roots) == 0 and os.path.exists(output_path):
                for child in os.listdir(output_path):
                    child_path = os.path.join(output_path, child)
                    if not os.path.isdir(child_path):
                        continue
                    if os.path.exists(os.path.join(child_path, 'data.yaml')):
                        class_roots.append(child_path)

            # If splits are unknown (e.g., build_dataset=False), infer from folder structure.
            splits_for_visualization = available_splits if len(available_splits) > 0 else ['train', 'valid', 'val', 'test']

            for class_root in class_roots:
                for split in splits_for_visualization:
                    images_dir = os.path.join(class_root, split, 'images')
                    if not os.path.exists(images_dir):
                        continue
                    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    for img_file in image_files:
                        all_class_images.append(os.path.join(images_dir, img_file))

            if len(all_class_images) == 0:
                print("No images found to visualize!")
            else:
                if visualization_random_seed is not None:
                    random.seed(visualization_random_seed)
                sample_count = min(num_visualization_samples, len(all_class_images))
                sampled_paths = random.sample(all_class_images, sample_count)
                visualize_roi_samples(
                    output_path,
                    num_samples=sample_count,
                    image_paths=sampled_paths,
                    random_seed=visualization_random_seed
                )
        else:
            visualize_roi_samples(
                output_path,
                num_samples=num_visualization_samples,
                splits=available_splits if len(available_splits) > 0 else ['train', 'valid', 'test'],
                image_paths=None,
                random_seed=visualization_random_seed
            )


if __name__ == "__main__":
    main()
