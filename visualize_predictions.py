"""
YOLO Segmentation Prediction Visualization Script
Loads trained model and generates mask annotations on validation images
"""

import os
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm


def get_image_files(image_dir: str) -> list:
    """Return a sorted, deduplicated list of image Paths in *image_dir*."""
    extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    files = []
    for ext in extensions:
        files.extend(Path(image_dir).glob(f'*{ext}'))
        files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
    return sorted(set(files))


def visualize_segmentation_predictions(
    model_path: str,
    image_dir: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    line_thickness: int = 2,
    alpha: float = 0.35,
    draw_boxes_and_labels: bool = True,
):
    """
    Run YOLO segmentation inference and save annotated images
    
    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        image_dir: Directory containing validation images
        output_dir: Directory to save annotated images
        conf_threshold: Confidence threshold for predictions
        line_thickness: Thickness of mask contours
        alpha: Transparency of mask overlay (0=transparent, 1=opaque)
        draw_boxes_and_labels: If False, skip bounding boxes, confidence
            labels, and all text overlays (No Detection / Detections count).
    """

    print("\n" + "="*70)
    print("YOLO SEGMENTATION VISUALIZATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Images: {image_dir}")
    print(f"Output: {output_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    print("="*70 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load trained model
    print("Loading model...")
    model = YOLO(model_path)
    print(f"✓ Model loaded: {model_path}\n")
    
    # Get all images
    image_files = get_image_files(image_dir)
    print(f"Found {len(image_files)} images to process\n")
    
    if len(image_files) == 0:
        print("❌ No images found! Check the image directory path.")
        return
    
    # Generate random colors for each class
    np.random.seed(42)
    colors = {}
    for name in model.names.values():
        colors[name] = tuple(map(int, np.random.randint(0, 255, 3)))
    
    # Process each image
    print("Processing images...\n")
    stats = {
        'total_images': len(image_files),
        'images_with_detections': 0,
        'total_detections': 0
    }
    
    for img_path in tqdm(image_files, desc="Annotating images"):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ Warning: Could not read {img_path.name}")
            continue
        
        # Run inference
        results = model.predict(
            source=img,
            conf=conf_threshold,
            verbose=False
        )[0]
        
        # Check if any detections
        if results.masks is None or len(results.masks) == 0:
            # No detections - save original image with "No Detection" text
            annotated = img.copy()
            if draw_boxes_and_labels:
                cv2.putText(
                    annotated,
                    "No Detection",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
        else:
            # Create annotated image
            annotated = img.copy()
            overlay = img.copy()
            
            # Get masks and boxes
            masks = results.masks.data.cpu().numpy()  # (N, H, W)
            boxes = results.boxes.xyxy.cpu().numpy()  # (N, 4)
            confs = results.boxes.conf.cpu().numpy()  # (N,)
            class_ids = results.boxes.cls.cpu().numpy().astype(int)  # (N,)
            
            stats['images_with_detections'] += 1
            stats['total_detections'] += len(masks)
            
            # Draw each mask
            for i, (mask, box, conf, cls_id) in enumerate(zip(masks, boxes, confs, class_ids)):
                # Get class name and color
                class_name = model.names[cls_id]
                color = colors[class_name]
                
                # Resize mask to image size
                mask_resized = cv2.resize(
                    mask,
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                mask_binary = (mask_resized > 0.5).astype(np.uint8)
                
                # Fill mask on overlay — use the same yellow as skeletonize_worms.py
                overlay[mask_binary == 1] = (0, 200, 255)
                
                # Draw contour
                contours, _ = cv2.findContours(
                    mask_binary,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, (0, 150, 200), line_thickness)
                
                if draw_boxes_and_labels:
                    # Draw bounding box — darker amber (0,150,200) vs mask yellow (0,200,255)
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 150, 200), 2)

                    # Add label
                    label = f"{class_name} {conf:.2f}"
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    label_y = max(y1 - 10, label_size[1] + 10)

                    # Draw label background
                    cv2.rectangle(
                        annotated,
                        (x1, label_y - label_size[1] - 4),
                        (x1 + label_size[0] + 4, label_y + 4),
                        (0, 150, 200),
                        -1
                    )

                    # Draw label text
                    cv2.putText(
                        annotated,
                        label,
                        (x1 + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
            
            # Blend overlay with original image
            annotated = cv2.addWeighted(annotated, 1 - alpha, overlay, alpha, 0)
            
            if draw_boxes_and_labels:
                # Add detection count
                det_count_text = f"Detections: {len(masks)}"
                cv2.putText(
                    annotated,
                    det_count_text,
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
        
        # Save annotated image
        output_path = Path(output_dir) / img_path.name
        cv2.imwrite(str(output_path), annotated)
    
    # Print statistics
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"Total images processed: {stats['total_images']}")
    print(f"Images with detections: {stats['images_with_detections']}")
    print(f"Images without detections: {stats['total_images'] - stats['images_with_detections']}")
    print(f"Total detections: {stats['total_detections']}")
    if stats['images_with_detections'] > 0:
        avg_det = stats['total_detections'] / stats['images_with_detections']
        print(f"Average detections per image: {avg_det:.2f}")
    print(f"\n✓ Annotated images saved to: {output_dir}")
    print("="*70 + "\n")


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\segment\worm_seg_train\weights\best.pt"
    IMAGE_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyDetection.v10i.coco-segmentation\valid\images"
    OUTPUT_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\validation_predictions_yellow"

    # ROI YOLO validation
    #MODEL_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt"
    #IMAGE_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI\valid\images"
    #OUTPUT_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\yolo_roi_validation_predictions_fix_overlap_text_disabled"
    
    # Visualization parameters
    CONF_THRESHOLD = 0.25        # Confidence threshold (0.0-1.0)
    LINE_THICKNESS = 2           # Contour line thickness
    MASK_ALPHA = 0.35            # Mask transparency (0.0-1.0) — matches skeletonize_worms.py
    DRAW_BOXES_AND_LABELS = True # Set False to hide bounding boxes and text
    
    print("\n" + "="*70)
    print("YOLO SEGMENTATION PREDICTION VISUALIZATION SCRIPT")
    print("="*70)
    print(f"Model weights: {MODEL_PATH}")
    print(f"Validation images: {IMAGE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"Mask transparency: {MASK_ALPHA}")
    print("="*70 + "\n")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("Please update MODEL_PATH to point to your trained model weights.")
        exit(1)
    
    # Check if image directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"❌ Error: Image directory not found at {IMAGE_DIR}")
        print("Please update IMAGE_DIR to point to your validation images.")
        exit(1)
    
    # Run visualization
    visualize_segmentation_predictions(
        model_path=MODEL_PATH,
        image_dir=IMAGE_DIR,
        output_dir=OUTPUT_DIR,
        conf_threshold=CONF_THRESHOLD,
        line_thickness=LINE_THICKNESS,
        alpha=MASK_ALPHA,
        draw_boxes_and_labels=DRAW_BOXES_AND_LABELS,
    )
    
    print("You can now review the annotated images in the output directory.")
    print("To adjust visualization settings, modify the parameters at the top of this script.")
