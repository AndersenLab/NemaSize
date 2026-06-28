"""
RF-DETR Instance Segmentation Inference Script
Performs inference with trained RF-DETR-Seg models on a folder of images.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import json
import numpy as np
import cv2
import torch
from tqdm import tqdm


class RFDETRSegmentationInference:
    """
    Inference class for RF-DETR instance segmentation models.
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        model_size: str = 'l',
        device: str = 'cuda',
        conf_threshold: float = 0.3
    ):
        """
        Initialize the inference pipeline.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            model_size: Model size ('n', 's', 'm', 'l', 'xl', '2xl')
            device: Device to run inference on ('cuda', 'cpu', 'mps')
            conf_threshold: Confidence threshold for detections
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model_size = model_size.lower()
        self.device = device
        self.conf_threshold = conf_threshold
        
        # Check if checkpoint exists
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model
        self.model = self._load_model()
        
        print(f"✓ Loaded RF-DETR-Seg-{model_size.upper()} model from: {checkpoint_path}")
        print(f"✓ Device: {device}")
        print(f"✓ Confidence threshold: {conf_threshold}")
    
    def _load_model(self):
        """Load the RF-DETR model from checkpoint."""
        try:
            from rfdetr import (
                RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium,
                RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge
            )
        except ImportError:
            raise ImportError("rfdetr package not found. Install with: pip install rfdetr")
        
        # Model size mapping
        model_classes = {
            'n': RFDETRSegNano,
            's': RFDETRSegSmall,
            'm': RFDETRSegMedium,
            'l': RFDETRSegLarge,
            'xl': RFDETRSegXLarge,
            '2xl': RFDETRSeg2XLarge
        }
        
        if self.model_size not in model_classes:
            raise ValueError(f"Invalid model_size. Choose from: {list(model_classes.keys())}")
        
        # Initialize model with pretrained weights
        model_class = model_classes[self.model_size]
        print(f"Loading checkpoint: {self.checkpoint_path}")
        model = model_class(pretrain_weights=str(self.checkpoint_path))
        
        return model
    
    def predict_image(
        self,
        image_path: str,
        return_masks: bool = True,
        return_boxes: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference on a single image.
        
        Args:
            image_path: Path to the image file
            return_masks: Return segmentation masks
            return_boxes: Return bounding boxes
            
        Returns:
            Dictionary containing predictions
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Run inference using RF-DETR's predict method
        # Note: RF-DETR API uses 'threshold' parameter, not 'conf'
        results = self.model.predict(
            str(image_path),
            threshold=self.conf_threshold
        )
        
        # Convert results to list if single result
        if not isinstance(results, list):
            results = [results]
        
        return {
            'image_path': str(image_path),
            'image_shape': image.shape,
            'results': results
        }
    
    def visualize_predictions(
        self,
        image_path: str,
        predictions: Dict[str, Any],
        output_path: Optional[str] = None,
        show_boxes: bool = True,
        show_masks: bool = True,
        show_labels: bool = True,
        mask_alpha: float = 0.5,
        box_thickness: int = 2
    ) -> np.ndarray:
        """
        Visualize predictions on the image.
        
        Args:
            image_path: Path to the original image
            predictions: Predictions from predict_image()
            output_path: Path to save the annotated image (optional)
            show_boxes: Draw bounding boxes
            show_masks: Draw segmentation masks
            show_labels: Show class labels and confidence scores
            mask_alpha: Transparency of mask overlay (0-1)
            box_thickness: Thickness of bounding box lines
            
        Returns:
            Annotated image as numpy array
        """
        # Read original image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Create a copy for annotations
        annotated = image.copy()
        
        # Get results (list of supervision.Detections objects)
        results = predictions['results']
        
        if len(results) == 0:
            # No detections
            if output_path:
                cv2.imwrite(str(output_path), annotated)
            return annotated
        
        # Extract predictions from results
        # RF-DETR returns supervision.Detections object
        detections = results[0]  # Get first (and only) result
        
        # Check if there are any detections
        if len(detections) == 0:
            if output_path:
                cv2.imwrite(str(output_path), annotated)
            return annotated
        
        # Generate colors for each instance
        num_instances = len(detections)
        colors = self._generate_colors(num_instances)
        
        # Draw masks first (so boxes are on top)
        if show_masks and detections.mask is not None:
            masks = detections.mask  # [N, H, W] numpy array
            
            # Create overlay for all masks
            overlay = annotated.copy()
            
            for i, mask in enumerate(masks):
                # Resize mask to image size if needed
                if mask.shape != (image.shape[0], image.shape[1]):
                    mask = cv2.resize(mask.astype(np.uint8), (image.shape[1], image.shape[0]))
                
                # Convert to binary mask
                mask_binary = (mask > 0.5).astype(np.uint8)
                
                # Apply color to mask region
                color = colors[i]
                overlay[mask_binary > 0] = overlay[mask_binary > 0] * (1 - mask_alpha) + \
                                            np.array(color) * mask_alpha
            
            # Blend overlay with original
            annotated = overlay.astype(np.uint8)
        
        # Draw bounding boxes and labels
        if show_boxes and detections.xyxy is not None:
            boxes = detections.xyxy  # [N, 4] in xyxy format
            scores = detections.confidence if hasattr(detections, 'confidence') else None  # [N]
            classes = detections.class_id if hasattr(detections, 'class_id') else None  # [N]
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.astype(int)
                color = colors[i]
                
                # Draw bounding box
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, box_thickness)
                
                # Draw label
                if show_labels:
                    # Get class name and score
                    if classes is not None and scores is not None:
                        class_id = int(classes[i])
                        score = float(scores[i])
                        # Try to get class name from detections data if available
                        class_name = f"Class {class_id}"
                        label = f"{class_name}: {score:.2f}"
                    elif scores is not None:
                        score = float(scores[i])
                        label = f"{score:.2f}"
                    else:
                        label = f"Detection {i}"
                    
                    # Get label size
                    (label_w, label_h), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated,
                        (x1, y1 - label_h - baseline - 5),
                        (x1 + label_w, y1),
                        color,
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - baseline - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
        
        return annotated
    
    def _generate_colors(self, n: int) -> List[Tuple[int, int, int]]:
        """
        Generate n distinct colors for visualization.
        
        Args:
            n: Number of colors to generate
            
        Returns:
            List of RGB color tuples
        """
        colors = []
        for i in range(n):
            hue = int(180 * i / max(n, 1))
            color = cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]),
                cv2.COLOR_HSV2BGR
            )[0][0]
            colors.append((int(color[0]), int(color[1]), int(color[2])))
        return colors
    
    def process_folder(
        self,
        input_folder: str,
        output_folder: str,
        image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        save_json: bool = True,
        **viz_kwargs
    ) -> Dict[str, Any]:
        """
        Process all images in a folder.
        
        Args:
            input_folder: Folder containing input images
            output_folder: Folder to save annotated images
            image_extensions: List of image file extensions to process
            save_json: Save predictions as JSON file
            **viz_kwargs: Additional arguments for visualize_predictions()
            
        Returns:
            Dictionary with processing statistics
        """
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        
        if not input_folder.exists():
            raise ValueError(f"Input folder does not exist: {input_folder}")
        
        # Create output folder
        output_folder.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_folder.glob(f"*{ext}"))
            image_files.extend(input_folder.glob(f"*{ext.upper()}"))
        
        image_files = sorted(list(set(image_files)))  # Remove duplicates and sort
        
        if len(image_files) == 0:
            print(f"⚠️  No images found in {input_folder}")
            return {'num_images': 0, 'num_detections': 0}
        
        print(f"\n{'='*70}")
        print(f"Processing {len(image_files)} images from: {input_folder}")
        print(f"Saving results to: {output_folder}")
        print(f"{'='*70}\n")
        
        # Process each image
        all_predictions = []
        total_detections = 0
        
        for image_path in tqdm(image_files, desc="Processing images"):
            try:
                # Run inference
                predictions = self.predict_image(str(image_path))
                
                # Count detections
                results = predictions['results']
                if len(results) > 0:
                    detections = results[0]
                    num_detections = len(detections) if detections is not None else 0
                    total_detections += num_detections
                else:
                    num_detections = 0
                
                # Visualize and save
                output_path = output_folder / image_path.name
                self.visualize_predictions(
                    str(image_path),
                    predictions,
                    str(output_path),
                    **viz_kwargs
                )
                
                # Store predictions for JSON export
                if save_json:
                    all_predictions.append({
                        'image': image_path.name,
                        'num_detections': num_detections,
                        'predictions': self._format_predictions_for_json(predictions)
                    })
                
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                continue
        
        # Save predictions to JSON
        if save_json:
            json_path = output_folder / 'predictions.json'
            with open(json_path, 'w') as f:
                json.dump(all_predictions, f, indent=2)
            print(f"\n✓ Predictions saved to: {json_path}")
        
        # Print summary
        print(f"\n{'='*70}")
        print("INFERENCE COMPLETE")
        print(f"{'='*70}")
        print(f"Processed images: {len(image_files)}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections / len(image_files):.2f}")
        print(f"Results saved to: {output_folder.absolute()}")
        print(f"{'='*70}\n")
        
        return {
            'num_images': len(image_files),
            'num_detections': total_detections,
            'output_folder': str(output_folder.absolute())
        }
    
    def _format_predictions_for_json(self, predictions: Dict[str, Any]) -> List[Dict]:
        """Format predictions for JSON export."""
        results = predictions['results']
        
        if len(results) == 0:
            return []
        
        detections = results[0]
        
        if detections is None or len(detections) == 0:
            return []
        
        formatted = []
        
        # Access supervision.Detections attributes
        boxes = detections.xyxy if detections.xyxy is not None else None
        scores = detections.confidence if hasattr(detections, 'confidence') else None
        classes = detections.class_id if hasattr(detections, 'class_id') else None
        masks = detections.mask if hasattr(detections, 'mask') and detections.mask is not None else None
        
        if boxes is not None:
            for i in range(len(boxes)):
                pred = {
                    'box': boxes[i].tolist(),
                }
                
                if scores is not None:
                    pred['confidence'] = float(scores[i])
                
                if classes is not None:
                    pred['class_id'] = int(classes[i])
                    pred['class_name'] = f"Class {int(classes[i])}"
                
                # Add mask area if available
                if masks is not None:
                    mask = masks[i]
                    pred['mask_area'] = int(mask.sum())
                
                formatted.append(pred)
        
        return formatted


def main():
    """Main function to run inference with configuration specified below."""
    
    # ========================================================================
    # CONFIGURATION - Edit these paths and settings as needed
    # ========================================================================
    
    # Model checkpoint path
    CHECKPOINT_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\rfdetr_segment\worm_seg_train\checkpoint_best_total.pth"
    
    # Input/Output folders
    INPUT_FOLDER = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyDetection.v10i.coco-segmentation\valid\images"
    OUTPUT_FOLDER = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\rfdetr_validation_predictions"
    
    # Model configuration
    MODEL_SIZE = 'l'  # Options: 'n', 's', 'm', 'l', 'xl', '2xl'
    DEVICE = 'cuda'   # Options: 'cuda', 'cpu', 'mps'
    
    # Detection thresholds
    CONF_THRESHOLD = 0.5  # Confidence threshold (0.0 - 1.0)
    
    # Visualization settings
    SHOW_BOXES = True      # Draw bounding boxes
    SHOW_MASKS = True      # Draw segmentation masks
    SHOW_LABELS = True     # Show class labels and confidence scores
    MASK_ALPHA = 0.5       # Mask transparency (0.0 - 1.0)
    BOX_THICKNESS = 2      # Bounding box line thickness
    
    # Output settings
    SAVE_JSON = True       # Save predictions to JSON file
    
    # ========================================================================
    # END CONFIGURATION
    # ========================================================================
    
    print("\n" + "="*70)
    print("RF-DETR SEGMENTATION INFERENCE")
    print("="*70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"Model size: {MODEL_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print("="*70 + "\n")
    
    # Initialize inference pipeline
    inferencer = RFDETRSegmentationInference(
        checkpoint_path=CHECKPOINT_PATH,
        model_size=MODEL_SIZE,
        device=DEVICE,
        conf_threshold=CONF_THRESHOLD
    )
    
    # Process all images in the folder
    results = inferencer.process_folder(
        input_folder=INPUT_FOLDER,
        output_folder=OUTPUT_FOLDER,
        save_json=SAVE_JSON,
        show_boxes=SHOW_BOXES,
        show_masks=SHOW_MASKS,
        show_labels=SHOW_LABELS,
        mask_alpha=MASK_ALPHA,
        box_thickness=BOX_THICKNESS
    )
    
    print("\n✓ Inference completed successfully!")
    return results


if __name__ == '__main__':
    main()
