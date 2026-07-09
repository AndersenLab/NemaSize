"""
YOLO Instance Segmentation Training Script
Trains YOLOv8/YOLOv11 models for instance segmentation tasks.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import yaml
from datetime import datetime
import shutil


class YOLOSegmentationTrainer:
    """
    Trainer for YOLO instance segmentation models.
    Supports YOLO26-seg, YOLOv8-seg and YOLOv11-seg architectures.
    """
    
    def __init__(self):
        """Initialize the trainer."""
        self.ultralytics_available = self._check_ultralytics()
    
    def _check_ultralytics(self) -> bool:
        """Check if ultralytics is installed."""
        try:
            from ultralytics import YOLO
            return True
        except ImportError:
            return False
    
    def prepare_yolo_dataset_config(
        self,
        dataset_path: str,
        output_yaml_path: Optional[str] = None,
        train_split: str = 'train',
        val_split: str = 'valid'
    ) -> str:
        """
        Create a YAML configuration file for YOLO training.
        Converts COCO dataset structure to YOLO format configuration.
        
        Args:
            dataset_path: Path to the dataset directory (should contain train/valid folders)
            output_yaml_path: Path where to save the YAML config (optional)
            train_split: Name of training split folder (default: 'train')
            val_split: Name of validation split folder (default: 'valid')
            
        Returns:
            Path to the created YAML configuration file
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Find train and validation splits
        train_path = dataset_path / train_split
        val_path = dataset_path / val_split
        
        # Try alternate validation names if 'valid' doesn't exist
        if not val_path.exists():
            val_path = dataset_path / 'val'
        
        if not train_path.exists():
            raise ValueError(f"Training split not found: {train_path}")
        
        if not val_path.exists():
            raise ValueError(f"Validation split not found. Tried: {dataset_path / val_split}, {dataset_path / 'val'}")
        
        # Find annotation files to extract category information
        train_ann_files = list(train_path.glob('*.json'))
        if not train_ann_files:
            raise ValueError(f"No COCO annotation file found in {train_path}")
        
        # Load categories from the annotation file
        with open(train_ann_files[0], 'r') as f:
            coco_data = json.load(f)
        
        # Extract category names (YOLO uses index-based class list)
        categories = sorted(coco_data.get('categories', []), key=lambda x: x['id'])
        class_names = [cat['name'] for cat in categories]
        
        print(f"\n{'='*70}")
        print("Creating YOLO Dataset Configuration")
        print(f"{'='*70}")
        print(f"Dataset path: {dataset_path.absolute()}")
        print(f"Training split: {train_path.name}")
        print(f"Validation split: {val_path.name}")
        print(f"Number of classes: {len(class_names)}")
        print(f"Classes: {', '.join(class_names)}")
        print(f"{'='*70}\n")
        
        # Create YAML configuration
        # Note: For COCO format, YOLO expects specific structure
        config = {
            'path': str(dataset_path.absolute()),  # Dataset root directory
            'train': train_split,  # Relative to 'path'
            'val': val_split if val_path.exists() else 'val',  # Relative to 'path'
            'names': {i: name for i, name in enumerate(class_names)}  # Class names dict
        }
        
        # Determine output path
        if output_yaml_path is None:
            output_yaml_path = dataset_path / 'dataset.yaml'
        else:
            output_yaml_path = Path(output_yaml_path)
        
        # Save YAML file
        with open(output_yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ YAML configuration saved to: {output_yaml_path}")
        print(f"\nYAML contents:")
        print("-" * 70)
        with open(output_yaml_path, 'r') as f:
            print(f.read())
        print("-" * 70 + "\n")
        
        return str(output_yaml_path)
    
    def train(
        self,
        dataset_yaml: str,
        model_size: str = 'x',  # n, s, m, l, x
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        workers: int = 8,
        device: str = '0',  # '0' for GPU 0, 'cpu' for CPU, '0,1' for multiple GPUs
        project: str = 'runs/segment',
        name: str = 'train',
        pretrained: bool = True,
        optimizer: str = 'SGD',  # SGD, Adam, AdamW
        lr0: float = 0.01,
        momentum: float = 0.937,
        weight_decay: float = 0.0005,
        patience: int = 50,
        save_period: int = 10,
        resume: bool = False,
        resume_from: Optional[str] = None,
        transfer_weights: Optional[str] = None,
        augment: bool = False,  # Test-Time Augmentation during validation (not supported by all models)
        cache: bool = False,
        rect: bool = False,
        cos_lr: bool = False,
        close_mosaic: int = 10,
        amp: bool = True,
        val: bool = True,
        plots: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train a YOLO segmentation model.
        
        Args:
            dataset_yaml: Path to the dataset YAML configuration file
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra-large)
            epochs: Number of training epochs
            batch_size: Batch size for training
            image_size: Input image size
            workers: Number of dataloader workers
            device: Device to train on ('0', 'cpu', '0,1', etc.)
            project: Project directory where results will be saved
            name: Experiment name
            pretrained: Use pretrained weights
            optimizer: Optimizer type (SGD, Adam, AdamW)
            lr0: Initial learning rate
            momentum: SGD momentum / Adam beta1
            weight_decay: Optimizer weight decay
            patience: Epochs to wait for no improvement before early stopping
            save_period: Save checkpoint every x epochs
            resume: Resume training from last checkpoint
            resume_from: Specific checkpoint path to resume from
            transfer_weights: Path to a custom .pt file to use as starting weights for transfer learning
            augment: Use data augmentation
            cache: Cache images for faster training (True/False/'ram'/'disk')
            rect: Rectangular training
            cos_lr: Use cosine learning rate scheduler
            close_mosaic: Disable mosaic augmentation for final N epochs
            amp: Automatic Mixed Precision training
            val: Validate during training
            plots: Generate training plots
            verbose: Verbose output
            
        Returns:
            Dictionary containing training results and metrics
        """
        if not self.ultralytics_available:
            print("\n❌ ERROR: ultralytics package not found!")
            print("Please install it with: pip install ultralytics")
            return {'success': False, 'error': 'ultralytics not installed'}
        
        from ultralytics import YOLO
        
        # Validate model size
        valid_sizes = ['n', 's', 'm', 'l', 'x']
        if model_size not in valid_sizes:
            raise ValueError(f"Invalid model_size. Choose from: {valid_sizes}")
        
        # Prepare model
        if transfer_weights:
            print(f"\n🔀 Transfer learning from: {transfer_weights}")
            model = YOLO(transfer_weights)
        elif resume and resume_from:
            print(f"\n📥 Resuming training from: {resume_from}")
            model = YOLO(resume_from)
        elif resume:
            print(f"\n📥 Resuming training from last checkpoint in {project}/{name}")
            # Let YOLO find the last checkpoint automatically
            model_name = f'yolo26{model_size}-seg.pt'
            model = YOLO(model_name)
        else:
            # Load pretrained or initialize new model
            if pretrained:
                # Define model path in Dropbox to avoid downloading to repo
                models_dir = Path("C:/Users/jl200/Dropbox/JHU_2026_spring/NemaSeg/models")
                models_dir.mkdir(parents=True, exist_ok=True)
                model_path = models_dir / f'yolo26{model_size}-seg.pt'
                
                if model_path.exists():
                    print(f"\n📥 Loading pretrained model from: {model_path}")
                    model = YOLO(str(model_path))
                else:
                    print(f"\n📥 Downloading pretrained model: yolo26{model_size}-seg.pt")
                    print(f"   Will be saved to: {models_dir}")
                    # Download to current dir first, then move to models dir
                    import os
                    original_dir = os.getcwd()
                    os.chdir(str(models_dir))
                    model = YOLO(f'yolo26{model_size}-seg.pt')
                    os.chdir(original_dir)
            else:
                model_name = f'yolo26{model_size}-seg.yaml'
                print(f"\n🔨 Initializing model from scratch: {model_name}")
                model = YOLO(model_name)
        
        print("\n" + "="*70)
        print("YOLO SEGMENTATION TRAINING")
        print("="*70)
        if transfer_weights:
            print(f"Model: Transfer learning from {transfer_weights}")
        else:
            print(f"Model: YOLO26{model_size}-seg")
        print(f"Dataset: {dataset_yaml}")
        print(f"Epochs: {epochs}")
        print(f"Batch Size: {batch_size}")
        print(f"Image Size: {image_size}")
        print(f"Device: {device}")
        print(f"Optimizer: {optimizer}")
        print(f"Learning Rate: {lr0}")
        print(f"Output: {project}/{name}")
        print("="*70 + "\n")
        
        try:
            # Train the model
            results = model.train(
                data=dataset_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=image_size,
                workers=workers,
                device=device,
                project=project,
                name=name,
                pretrained=pretrained,
                optimizer=optimizer,
                lr0=lr0,
                momentum=momentum,
                weight_decay=weight_decay,
                patience=patience,
                save_period=save_period,
                resume=resume if resume_from else False,
                augment=augment,
                cache=cache,
                rect=rect,
                cos_lr=cos_lr,
                close_mosaic=close_mosaic,
                amp=amp,
                val=val,
                plots=plots,
                verbose=verbose
            )
            
            print("\n" + "="*70)
            print("TRAINING COMPLETE!")
            print("="*70)
            
            # Get the output directory
            output_dir = Path(project) / name
            print(f"Results saved to: {output_dir.absolute()}")
            print(f"Best weights: {output_dir / 'weights' / 'best.pt'}")
            print(f"Last weights: {output_dir / 'weights' / 'last.pt'}")
            print("="*70 + "\n")
            
            return {
                'success': True,
                'results': results,
                'output_dir': str(output_dir.absolute()),
                'best_weights': str((output_dir / 'weights' / 'best.pt').absolute()),
                'last_weights': str((output_dir / 'weights' / 'last.pt').absolute())
            }
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def validate(
        self,
        weights_path: str,
        dataset_yaml: str,
        batch_size: int = 16,
        image_size: int = 640,
        device: str = '0',
        split: str = 'val',
        save_json: bool = True,
        save_txt: bool = False,
        plots: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Validate a trained YOLO segmentation model.
        
        Args:
            weights_path: Path to model weights (.pt file)
            dataset_yaml: Path to dataset YAML configuration
            batch_size: Batch size for validation
            image_size: Input image size
            device: Device to use for validation
            split: Dataset split to validate on ('val', 'test', 'train')
            save_json: Save results to JSON file
            save_txt: Save results to txt files
            plots: Generate validation plots
            verbose: Verbose output
            
        Returns:
            Dictionary containing validation metrics
        """
        if not self.ultralytics_available:
            print("\n❌ ERROR: ultralytics package not found!")
            return {'success': False, 'error': 'ultralytics not installed'}
        
        from ultralytics import YOLO
        
        print("\n" + "="*70)
        print("YOLO SEGMENTATION VALIDATION")
        print("="*70)
        print(f"Weights: {weights_path}")
        print(f"Dataset: {dataset_yaml}")
        print(f"Split: {split}")
        print("="*70 + "\n")
        
        try:
            # Load model
            model = YOLO(weights_path)
            
            # Run validation
            metrics = model.val(
                data=dataset_yaml,
                batch=batch_size,
                imgsz=image_size,
                device=device,
                split=split,
                save_json=save_json,
                save_txt=save_txt,
                plots=plots,
                verbose=verbose
            )
            
            print("\n" + "="*70)
            print("VALIDATION RESULTS")
            print("="*70)
            print(f"Box mAP50: {metrics.box.map50:.4f}")
            print(f"Box mAP50-95: {metrics.box.map:.4f}")
            print(f"Mask mAP50: {metrics.seg.map50:.4f}")
            print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
            print("="*70 + "\n")
            
            return {
                'success': True,
                'metrics': metrics,
                'box_map50': float(metrics.box.map50),
                'box_map': float(metrics.box.map),
                'seg_map50': float(metrics.seg.map50),
                'seg_map': float(metrics.seg.map)
            }
            
        except Exception as e:
            print(f"\n❌ Validation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def predict(
        self,
        weights_path: str,
        source: str,
        save: bool = True,
        save_txt: bool = False,
        save_conf: bool = False,
        show: bool = False,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        device: str = '0',
        project: str = 'runs/segment',
        name: str = 'predict',
        line_width: int = 2,
        show_labels: bool = True,
        show_conf: bool = True,
        retina_masks: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run inference with a trained YOLO segmentation model.
        
        Args:
            weights_path: Path to model weights
            source: Input source (image file, folder, video, webcam, etc.)
            save: Save results
            save_txt: Save results as txt files
            save_conf: Save confidence scores in txt files
            show: Show results
            conf: Confidence threshold
            iou: NMS IoU threshold
            max_det: Maximum number of detections per image
            device: Device to use
            project: Project directory
            name: Experiment name
            line_width: Bounding box line width
            show_labels: Show labels on predictions
            show_conf: Show confidence scores on predictions
            retina_masks: Use high-resolution masks
            verbose: Verbose output
            
        Returns:
            Dictionary containing prediction results
        """
        if not self.ultralytics_available:
            print("\n❌ ERROR: ultralytics package not found!")
            return {'success': False, 'error': 'ultralytics not installed'}
        
        from ultralytics import YOLO
        
        print("\n" + "="*70)
        print("YOLO SEGMENTATION INFERENCE")
        print("="*70)
        print(f"Weights: {weights_path}")
        print(f"Source: {source}")
        print(f"Confidence: {conf}")
        print(f"IoU: {iou}")
        print("="*70 + "\n")
        
        try:
            # Load model
            model = YOLO(weights_path)
            
            # Run prediction
            results = model.predict(
                source=source,
                save=save,
                save_txt=save_txt,
                save_conf=save_conf,
                show=show,
                conf=conf,
                iou=iou,
                max_det=max_det,
                device=device,
                project=project,
                name=name,
                line_width=line_width,
                show_labels=show_labels,
                show_conf=show_conf,
                retina_masks=retina_masks,
                verbose=verbose
            )
            
            output_dir = Path(project) / name
            print(f"\n✓ Predictions saved to: {output_dir.absolute()}")
            
            return {
                'success': True,
                'results': results,
                'output_dir': str(output_dir.absolute())
            }
            
        except Exception as e:
            print(f"\n❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}
    
    def export(
        self,
        weights_path: str,
        format: str = 'onnx',
        imgsz: int = 640,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 12,
        half: bool = False
    ) -> Dict[str, Any]:
        """
        Export trained model to various formats.
        
        Args:
            weights_path: Path to model weights
            format: Export format (onnx, torchscript, coreml, tensorrt, etc.)
            imgsz: Image size for export
            dynamic: Dynamic input shapes
            simplify: Simplify ONNX model
            opset: ONNX opset version
            half: FP16 quantization
            
        Returns:
            Dictionary containing export results
        """
        if not self.ultralytics_available:
            print("\n❌ ERROR: ultralytics package not found!")
            return {'success': False, 'error': 'ultralytics not installed'}
        
        from ultralytics import YOLO
        
        print(f"\n📦 Exporting model to {format.upper()} format...")
        
        try:
            model = YOLO(weights_path)
            export_path = model.export(
                format=format,
                imgsz=imgsz,
                dynamic=dynamic,
                simplify=simplify,
                opset=opset,
                half=half
            )
            
            print(f"✓ Model exported to: {export_path}")
            
            return {
                'success': True,
                'export_path': str(export_path)
            }
            
        except Exception as e:
            print(f"\n❌ Export failed: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """
    Main training function.
    Configure training parameters here.
    """
    
    # ==================== CONFIGURATION ====================
    
    # Dataset configuration
    # DATASET_PATH = "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\datasets\\WormBodyDetection.v10i.coco-segmentation"
    DATASET_PATH = "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\datasets\\WormBodyROI_fix_overlap"
    
    # Transfer learning: path to a previously trained .pt file to start from.
    # Set to None to use the official yolo26x-seg.pt weights pretrained on COCO instead.
    TRANSFER_WEIGHTS = "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\runs\\segment\\worm_seg_train\\weights\\best.pt"

    # Training configuration
    CONFIG = {
        # Model settings
        'model_size': 'x',  # Must match the architecture of TRANSFER_WEIGHTS (yolo26x-seg)
        'pretrained': True,  # Ignored when transfer_weights is set
        'transfer_weights': TRANSFER_WEIGHTS,
        
        # Training hyperparameters
        'epochs': 200,  # epoch number, can reduce for quick tests
        'batch_size': 16,  # Optimized for RTX 3090 (24GB) at 640x640
        'image_size': 640,
        'workers': 8,
        'device': '0',  # '0' for GPU 0, 'cpu' for CPU, '0,1' for multi-GPU
        
        # Optimizer settings
        'optimizer': 'SGD',  # Options: 'SGD', 'Adam', 'AdamW'
        'lr0': 0.01,  # Initial learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        
        # Training behavior
        'patience': 300,  # Early stopping patience
        'save_period': 10,  # Save checkpoint every N epochs
        # Note: augment=True would enable Test-Time Augmentation (TTA) during validation,
        # which YOLO26x-seg does not support. Standard training augmentations (mosaic,
        # flips, color jitter, etc.) are applied automatically and are unaffected.
        'cos_lr': False,  # Cosine learning rate scheduler
        'close_mosaic': 60,  # Disable mosaic augmentation for the final N epochs
        
        # Performance
        'amp': True,  # Automatic Mixed Precision
        'cache': False,  # Cache images in RAM (True) or disk ('disk') for faster training
        'rect': False,  # Rectangular training
        
        # Output (stored outside repo in Dropbox)
        'project': 'C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\runs\\roi_yolo_segment_fix_overlap_no_erlystp',
        'name': 'roi_seg_train_fix_overlap_no_erlystp',
        
        # Visualization
        'plots': True,
        'verbose': True
    }
    
    # Resume training settings (optional)
    RESUME = False  # Set to True to resume training
    RESUME_FROM = None  # Specific checkpoint path, or None to use last checkpoint
    
    # ==================== EXECUTION ====================
    
    print("\n" + "="*70)
    print("YOLO SEGMENTATION TRAINING SCRIPT")
    print("="*70)
    print(f"Dataset: {DATASET_PATH}")
    if CONFIG.get('transfer_weights'):
        print(f"Transfer weights: {CONFIG['transfer_weights']}")
    else:
        print(f"Model: YOLO26{CONFIG['model_size']}-seg")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Batch Size: {CONFIG['batch_size']}")
    print(f"Device: {CONFIG['device']}")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = YOLOSegmentationTrainer()
    
    # Check if ultralytics is installed
    if not trainer.ultralytics_available:
        print("\n❌ ERROR: 'ultralytics' package is not installed!")
        print("\nTo install, run:")
        print("  pip install ultralytics")
        print("\nOr with conda:")
        print("  conda install -c conda-forge ultralytics")
        return 1
    
    # Step 1: Prepare YAML configuration
    print("Step 1: Preparing dataset configuration...")
    # Check if the dataset already has a YOLO-format data.yaml (images/labels subfolders)
    existing_yaml = Path(DATASET_PATH) / 'data.yaml'
    train_images_dir = Path(DATASET_PATH) / 'train' / 'images'
    if existing_yaml.exists() and train_images_dir.exists():
        yaml_path = str(existing_yaml)
        print(f"✓ Dataset is already in YOLO format. Using existing YAML: {yaml_path}")
        print("\nYAML contents:")
        print("-" * 70)
        with open(yaml_path, 'r') as f:
            print(f.read())
        print("-" * 70 + "\n")
    else:
        # Dataset is in COCO format — generate YAML from annotation JSON files
        try:
            yaml_path = trainer.prepare_yolo_dataset_config(
                dataset_path=DATASET_PATH,
                train_split='train',
                val_split='valid'
            )
        except Exception as e:
            print(f"\n❌ Failed to prepare dataset configuration: {e}")
            return 1
    
    # Step 2: Train the model
    print("\nStep 2: Starting training...")
    results = trainer.train(
        dataset_yaml=yaml_path,
        resume=RESUME,
        resume_from=RESUME_FROM,
        **CONFIG
    )
    
    if not results['success']:
        print("\n❌ Training failed!")
        return 1
    
    # Step 3: Validate the best model
    print("\nStep 3: Validating best model...")
    best_weights = results['best_weights']
    val_results = trainer.validate(
        weights_path=best_weights,
        dataset_yaml=yaml_path,
        batch_size=CONFIG['batch_size'],
        image_size=CONFIG['image_size'],
        device=CONFIG['device']
    )
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"✓ Best model: {best_weights}")
    print(f"✓ Training results: {results['output_dir']}")
    if val_results['success']:
        print(f"✓ Box mAP50-95: {val_results['box_map']:.4f}")
        print(f"✓ Mask mAP50-95: {val_results['seg_map']:.4f}")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("  1. Review training plots in the output directory")
    print("  2. Test inference with trainer.predict()")
    print("  3. Export model with trainer.export() for deployment")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
