"""
RF-DETR Instance Segmentation Training Script
Trains RF-DETR-Seg models for instance segmentation tasks.
Based on documentation: https://rfdetr.roboflow.com/
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime


class RFDETRSegmentationTrainer:
    """
    Trainer for RF-DETR instance segmentation models.
    Supports RF-DETR-Seg models from Nano to 2XLarge.
    """
    
    def __init__(self):
        """Initialize the trainer."""
        self.rfdetr_available = self._check_rfdetr()
    
    def _check_rfdetr(self) -> bool:
        """Check if rfdetr is installed."""
        try:
            import rfdetr
            return True
        except ImportError:
            return False
    
    def _get_pretrained_model_path(self, model_size: str) -> Optional[str]:
        """
        Get or download pretrained model to Dropbox location.
        Keeps repo clean by storing models outside the repository.
        
        Args:
            model_size: Model size ('n', 's', 'm', 'l', 'xl', '2xl')
            
        Returns:
            Path to pretrained model if it should be used, None otherwise
        """
        # Define Dropbox models directory (outside repo)
        models_dir = Path("C:/Users/jl200/Dropbox/JHU_2026_spring/NemaSeg/rfdetr_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model filename mapping
        model_files = {
            'n': 'rfdetr_seg_nano.pth',
            's': 'rfdetr_seg_small.pth',
            'm': 'rfdetr_seg_medium.pth',
            'l': 'rfdetr_seg_large.pth',
            'xl': 'rfdetr_seg_xlarge.pth',
            '2xl': 'rfdetr_seg_2xlarge.pth'
        }
        
        model_filename = model_files.get(model_size.lower())
        if not model_filename:
            return None
        
        model_path = models_dir / model_filename
        
        # Check if model already exists in Dropbox
        if model_path.exists():
            print(f"✓ Found cached pretrained model: {model_path}")
            return str(model_path)
        
        # Model doesn't exist yet - will be downloaded by RF-DETR to its cache
        # We'll let RF-DETR handle the download, then copy it
        print(f"Pretrained model will be downloaded on first use.")
        print(f"Model will be cached to: {model_path}")
        return None
    
    def _cache_pretrained_model(self, model, model_size: str):
        """
        Cache the downloaded pretrained model to Dropbox location.
        This attempts to save the model weights after RF-DETR downloads them.
        
        Args:
            model: The initialized RF-DETR model
            model_size: Model size string
        """
        import torch
        
        models_dir = Path("C:/Users/jl200/Dropbox/JHU_2026_spring/NemaSeg/rfdetr_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_files = {
            'n': 'rfdetr_seg_nano.pth',
            's': 'rfdetr_seg_small.pth',
            'm': 'rfdetr_seg_medium.pth',
            'l': 'rfdetr_seg_large.pth',
            'xl': 'rfdetr_seg_xlarge.pth',
            '2xl': 'rfdetr_seg_2xlarge.pth'
        }
        
        model_filename = model_files.get(model_size.lower())
        if not model_filename:
            return
        
        model_path = models_dir / model_filename
        
        # Save model state to our Dropbox cache
        print(f"\nCaching pretrained model to: {model_path}")
        try:
            # Save the model's state dict
            torch.save(model.model.state_dict(), model_path)
            print(f"✓ Model cached successfully. Future runs will use this cached version.")
        except Exception as e:
            print(f"Could not cache model: {e}")
    
    def validate_dataset_structure(self, dataset_path: str) -> Dict[str, Any]:
        """
        Validate that the dataset has the correct COCO format structure.
        RF-DETR expects: dataset/train/, dataset/valid/, dataset/test/
        Each with _annotations.coco.json and image files.
        
        Args:
            dataset_path: Path to the dataset directory
            
        Returns:
            Dictionary with validation results and metadata
        """
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {dataset_path}")
        
        # Check for required splits
        splits = {}
        required_splits = ['train', 'valid']
        optional_splits = ['test']
        
        for split_name in required_splits + optional_splits:
            split_path = dataset_path / split_name
            if split_path.exists():
                ann_file = split_path / '_annotations.coco.json'
                if ann_file.exists():
                    splits[split_name] = {
                        'path': str(split_path),
                        'annotations': str(ann_file)
                    }
                    
                    # Load and validate annotations
                    with open(ann_file, 'r') as f:
                        coco_data = json.load(f)
                    
                    splits[split_name]['num_images'] = len(coco_data.get('images', []))
                    splits[split_name]['num_annotations'] = len(coco_data.get('annotations', []))
                    splits[split_name]['categories'] = coco_data.get('categories', [])
                else:
                    if split_name in required_splits:
                        raise ValueError(f"Missing _annotations.coco.json in {split_path}")
        
        # Ensure we have at least train and valid
        for split_name in required_splits:
            if split_name not in splits:
                raise ValueError(f"Required split '{split_name}' not found in {dataset_path}")
        
        # Print dataset summary
        print(f"\n{'='*70}")
        print("RF-DETR Dataset Validation")
        print(f"{'='*70}")
        print(f"Dataset path: {dataset_path.absolute()}")
        
        for split_name, split_info in splits.items():
            print(f"\n{split_name.upper()} split:")
            print(f"  Path: {split_info['path']}")
            print(f"  Images: {split_info['num_images']}")
            print(f"  Annotations: {split_info['num_annotations']}")
        
        if 'train' in splits:
            categories = splits['train']['categories']
            print(f"\nNumber of classes: {len(categories)}")
            print(f"Classes: {', '.join([cat['name'] for cat in categories])}")
        
        print(f"{'='*70}\n")
        
        return {
            'valid': True,
            'splits': splits,
            'num_classes': len(categories) if 'train' in splits else 0,
            'categories': categories if 'train' in splits else []
        }
    
    def train(
        self,
        dataset_dir: str,
        model_size: str = 'm',  # Options: 'n', 's', 'm', 'l', 'xl', '2xl'
        epochs: int = 100,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        lr: float = 1e-4,
        lr_encoder: Optional[float] = None,
        resolution: Optional[int] = None,
        weight_decay: float = 1e-4,
        device: str = 'cuda',
        output_dir: str = 'output',
        use_ema: bool = True,
        gradient_checkpointing: bool = False,
        checkpoint_interval: int = 10,
        resume: Optional[str] = None,
        tensorboard: bool = True,
        wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_run: Optional[str] = None,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        early_stopping_min_delta: float = 0.001,
        early_stopping_use_ema: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train an RF-DETR segmentation model.
        
        Args:
            dataset_dir: Path to the dataset directory (COCO format with train/valid/test splits)
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 
                       'xl' (xlarge), '2xl' (2xlarge)
            epochs: Number of training epochs
            batch_size: Batch size per GPU (effective batch = batch_size * grad_accum_steps * num_gpus)
            grad_accum_steps: Gradient accumulation steps for larger effective batch size
            lr: Learning rate for most model components
            lr_encoder: Learning rate for the backbone encoder (if None, uses lr * 1.5)
            resolution: Input image resolution (must be divisible by 56). If None, uses model default
            weight_decay: L2 regularization coefficient
            device: Device to train on ('cuda', 'cpu', 'mps')
            output_dir: Directory where checkpoints and logs are saved
            use_ema: Enable Exponential Moving Average of weights
            gradient_checkpointing: Trade compute for memory (reduces memory usage)
            checkpoint_interval: Save checkpoint every N epochs
            resume: Path to checkpoint to resume training
            tensorboard: Enable TensorBoard logging
            wandb: Enable Weights & Biases logging
            wandb_project: W&B project name
            wandb_run: W&B run name
            early_stopping: Enable early stopping
            early_stopping_patience: Epochs without improvement before stopping
            early_stopping_min_delta: Minimum mAP change to qualify as improvement
            early_stopping_use_ema: Use EMA model for early stopping metrics
            verbose: Verbose output
            
        Returns:
            Dictionary containing training results and paths
        """
        if not self.rfdetr_available:
            print("\n❌ ERROR: rfdetr package not found!")
            print("Please install it with: pip install rfdetr")
            return {'success': False, 'error': 'rfdetr not installed'}
        
        # Import RF-DETR segmentation models
        from rfdetr import (
            RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium,
            RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge
        )
        
        # Model size mapping
        model_classes = {
            'n': RFDETRSegNano,
            's': RFDETRSegSmall,
            'm': RFDETRSegMedium,
            'l': RFDETRSegLarge,
            'xl': RFDETRSegXLarge,
            '2xl': RFDETRSeg2XLarge
        }
        
        # Validate model size
        if model_size.lower() not in model_classes:
            raise ValueError(f"Invalid model_size. Choose from: {list(model_classes.keys())}")
        
        model_size = model_size.lower()
        
        # Validate dataset structure
        print("\nValidating dataset structure...")
        dataset_info = self.validate_dataset_structure(dataset_dir)
        
        # Get or prepare pretrained model path
        pretrained_path = self._get_pretrained_model_path(model_size)
        
        # Initialize model
        print(f"\n📦 Initializing RF-DETR-Seg-{model_size.upper()} model...")
        model_class = model_classes[model_size]
        
        if pretrained_path:
            # Load from our cached Dropbox location
            print(f"Loading pretrained weights from: {pretrained_path}")
            model = model_class(pretrain_weights=pretrained_path)
        else:
            # First time - RF-DETR will download to its cache
            print("Downloading pretrained COCO weights...")
            model = model_class()
            
            # After download, try to cache the model to Dropbox for future use
            try:
                self._cache_pretrained_model(model, model_size)
            except Exception as e:
                print(f"Note: Could not cache model to Dropbox: {e}")
                print("Model will be re-downloaded on next run.")
        
        # Set default encoder learning rate if not specified
        if lr_encoder is None:
            lr_encoder = lr * 1.5
        
        # Calculate effective batch size
        effective_batch_size = batch_size * grad_accum_steps
        
        print("\n" + "="*70)
        print("RF-DETR SEGMENTATION TRAINING")
        print("="*70)
        print(f"Model: RF-DETR-Seg-{model_size.upper()}")
        print(f"Dataset: {dataset_dir}")
        print(f"Number of classes: {dataset_info['num_classes']}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Effective batch size: {effective_batch_size}")
        print(f"Learning rate: {lr}")
        print(f"Encoder learning rate: {lr_encoder}")
        if resolution:
            print(f"Resolution: {resolution}")
        print(f"Device: {device}")
        print(f"Output directory: {output_dir}")
        print(f"Use EMA: {use_ema}")
        print(f"TensorBoard: {tensorboard}")
        print(f"W&B: {wandb}")
        if early_stopping:
            print(f"Early stopping: patience={early_stopping_patience}, min_delta={early_stopping_min_delta}")
        print("="*70 + "\n")
        
        # Prepare training arguments
        train_kwargs = {
            'dataset_dir': dataset_dir,
            'epochs': epochs,
            'batch_size': batch_size,
            'grad_accum_steps': grad_accum_steps,
            'lr': lr,
            'lr_encoder': lr_encoder,
            'weight_decay': weight_decay,
            'device': device,
            'output_dir': output_dir,
            'use_ema': use_ema,
            'gradient_checkpointing': gradient_checkpointing,
            'checkpoint_interval': checkpoint_interval,
            'tensorboard': tensorboard,
            'wandb': wandb,
            'early_stopping': early_stopping,
            'early_stopping_patience': early_stopping_patience,
            'early_stopping_min_delta': early_stopping_min_delta,
            'early_stopping_use_ema': early_stopping_use_ema
        }
        
        # Add optional parameters
        if resolution is not None:
            train_kwargs['resolution'] = resolution
        if resume is not None:
            train_kwargs['resume'] = resume
        if wandb_project is not None:
            train_kwargs['project'] = wandb_project
        if wandb_run is not None:
            train_kwargs['run'] = wandb_run
        
        try:
            print("🚀 Starting training...\n")
            
            # Train the model
            model.train(**train_kwargs)
            
            print("\n" + "="*70)
            print("TRAINING COMPLETE!")
            print("="*70)
            
            # Output directory
            output_path = Path(output_dir)
            print(f"Results saved to: {output_path.absolute()}")
            
            # List available checkpoints
            checkpoints = {
                'latest': output_path / 'checkpoint.pth',
                'best_ema': output_path / 'checkpoint_best_ema.pth',
                'best_regular': output_path / 'checkpoint_best_regular.pth',
                'best_total': output_path / 'checkpoint_best_total.pth'
            }
            
            print("\nAvailable checkpoints:")
            for name, path in checkpoints.items():
                if path.exists():
                    print(f"  ✓ {name}: {path}")
            
            print("="*70 + "\n")
            
            return {
                'success': True,
                'output_dir': str(output_path.absolute()),
                'checkpoints': {k: str(v) for k, v in checkpoints.items() if v.exists()},
                'model_size': model_size,
                'num_classes': dataset_info['num_classes']
            }
            
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def predict(
        self,
        checkpoint_path: str,
        image_path: str,
        model_size: str = 'm',
        threshold: float = 0.5,
        device: str = 'cuda',
        save_output: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run inference with a trained RF-DETR segmentation model.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            image_path: Path to the input image
            model_size: Model size used during training
            threshold: Detection confidence threshold
            device: Device to run on ('cuda', 'cpu', 'mps')
            save_output: Optional path to save annotated image
            
        Returns:
            Dictionary containing predictions and metadata
        """
        if not self.rfdetr_available:
            print("\n❌ ERROR: rfdetr package not found!")
            return {'success': False, 'error': 'rfdetr not installed'}
        
        from PIL import Image
        import supervision as sv
        from rfdetr import (
            RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium,
            RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge
        )
        
        # Model size mapping
        model_classes = {
            'n': RFDETRSegNano,
            's': RFDETRSegSmall,
            'm': RFDETRSegMedium,
            'l': RFDETRSegLarge,
            'xl': RFDETRSegXLarge,
            '2xl': RFDETRSeg2XLarge
        }
        
        if model_size.lower() not in model_classes:
            raise ValueError(f"Invalid model_size. Choose from: {list(model_classes.keys())}")
        
        print(f"\n📦 Loading RF-DETR-Seg-{model_size.upper()} model from {checkpoint_path}...")
        
        # Load model with trained weights
        model_class = model_classes[model_size.lower()]
        model = model_class(pretrain_weights=checkpoint_path)
        
        # Load image
        print(f"📸 Loading image: {image_path}")
        image = Image.open(image_path)
        
        # Run prediction
        print(f"🔍 Running inference (threshold={threshold})...")
        detections = model.predict(image, threshold=threshold)
        
        print(f"\n✓ Detected {len(detections)} instances")
        
        # Optionally save annotated image
        if save_output:
            print(f"💾 Saving annotated image to: {save_output}")
            
            # Annotate with masks and labels
            annotated_image = sv.MaskAnnotator().annotate(image.copy(), detections)
            
            # Save
            annotated_image.save(save_output)
        
        return {
            'success': True,
            'num_detections': len(detections),
            'detections': detections,
            'image_path': image_path,
            'output_path': save_output
        }
    
    def export(
        self,
        checkpoint_path: str,
        model_size: str = 'm',
        output_dir: str = 'output'
    ) -> Dict[str, Any]:
        """
        Export RF-DETR model to ONNX format.
        Requires: pip install rfdetr[onnxexport]
        
        Args:
            checkpoint_path: Path to the trained model checkpoint
            model_size: Model size used during training
            output_dir: Directory to save ONNX model
            
        Returns:
            Dictionary containing export results
        """
        if not self.rfdetr_available:
            print("\n❌ ERROR: rfdetr package not found!")
            return {'success': False, 'error': 'rfdetr not installed'}
        
        from rfdetr import (
            RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium,
            RFDETRSegLarge, RFDETRSegXLarge, RFDETRSeg2XLarge
        )
        
        model_classes = {
            'n': RFDETRSegNano,
            's': RFDETRSegSmall,
            'm': RFDETRSegMedium,
            'l': RFDETRSegLarge,
            'xl': RFDETRSegXLarge,
            '2xl': RFDETRSeg2XLarge
        }
        
        if model_size.lower() not in model_classes:
            raise ValueError(f"Invalid model_size. Choose from: {list(model_classes.keys())}")
        
        print(f"\n📦 Loading RF-DETR-Seg-{model_size.upper()} model from {checkpoint_path}...")
        
        try:
            # Load model
            model_class = model_classes[model_size.lower()]
            model = model_class(pretrain_weights=checkpoint_path)
            
            # Export to ONNX
            print(f"📤 Exporting to ONNX format...")
            model.export()
            
            print(f"\n✓ Model exported successfully to: {output_dir}")
            
            return {
                'success': True,
                'output_dir': output_dir
            }
            
        except ImportError as e:
            print("\n❌ Export failed: onnxexport extension not installed")
            print("Install with: pip install rfdetr[onnxexport]")
            return {'success': False, 'error': 'onnxexport not installed'}
        except Exception as e:
            print(f"\n❌ Export failed: {e}")
            return {'success': False, 'error': str(e)}


def main():
    """Main function for training RF-DETR segmentation models."""
    
    # Set PyTorch memory allocator to reduce fragmentation
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # ==================== CONFIGURATION ====================
    
    # Dataset configuration
    DATASET_PATH = "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\datasets\\WormBodyDetection.v10i.coco-segmentation"
    
    # Training configuration
    CONFIG = {
        # Model settings
        'model_size': 'l',  # Options: 'n', 's', 'm', 'l', 'xl', '2xl' 
                            # (n=fastest/smallest, 2xl=best accuracy/largest)
        
        # Training hyperparameters
        'epochs': 100, # Number of training epochs
        'batch_size': 4,  # Per-GPU batch size (reduced from 8 to 4 for 24GB GPU)
        'grad_accum_steps': 4,  # Gradient accumulation (effective batch = 4*4 = 16)
        'lr': 1e-4,  # Learning rate
        'lr_encoder': None,  # If None, will be set to lr * 1.5
        'weight_decay': 1e-4,
        
        # Resolution (optional - if None, uses model default)
        # Must be divisible by both 56 and 24 (backbone requirement)
        # Valid values: 168, 336, 504, 672, 840, 1008...
        'resolution': 840,  # Higher resolution for better accuracy (closest to 784)
        
        # Hardware settings
        'device': 'cuda',  # Options: 'cuda', 'cpu', 'mps' (for Apple Silicon)
        'gradient_checkpointing': False,  # Reduce memory usage at cost of speed
        
        # Output directory (stored outside repo in Dropbox)
        'output_dir': 'C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\runs\\rfdetr_segment\\worm_seg_train',
        
        # Model behavior
        'use_ema': True,  # Exponential Moving Average of weights
        'checkpoint_interval': 10,  # Save checkpoint every N epochs
        
        # Logging
        'tensorboard': True,  # Enable TensorBoard logging
        'wandb': False,  # Enable Weights & Biases logging
        'wandb_project': None,  # W&B project name
        'wandb_run': None,  # W&B run name
        
        # Early stopping (optional)
        'early_stopping': False,
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'early_stopping_use_ema': False,
        
        # Other
        'verbose': True
    }
    
    # Resume training settings (optional)
    RESUME_CHECKPOINT = None  # Path to checkpoint, or None for fresh training
    
    # ==================== EXECUTION ====================
    
    print("\n" + "="*70)
    print("RF-DETR SEGMENTATION TRAINING SCRIPT")
    print("="*70)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: RF-DETR-Seg-{CONFIG['model_size'].upper()}")
    print(f"Epochs: {CONFIG['epochs']}")
    print(f"Effective Batch Size: {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
    print(f"Device: {CONFIG['device']}")
    print("="*70 + "\n")
    
    # Initialize trainer
    trainer = RFDETRSegmentationTrainer()
    
    # Check if rfdetr is installed
    if not trainer.rfdetr_available:
        print("\n❌ ERROR: 'rfdetr' package is not installed!")
        print("\nTo install, run:")
        print("  pip install rfdetr")
        print("\nFor TensorBoard/W&B support:")
        print("  pip install rfdetr[metrics]")
        return 1
    
    # Check if supervision is installed (needed for visualization)
    try:
        import supervision
    except ImportError:
        print("\n⚠️  WARNING: 'supervision' package not installed.")
        print("This is needed for inference visualization.")
        print("To install: pip install supervision")
    
    # Train the model
    print("Starting training...")
    results = trainer.train(
        dataset_dir=DATASET_PATH,
        resume=RESUME_CHECKPOINT,
        **CONFIG
    )
    
    if not results['success']:
        print("\n❌ Training failed!")
        return 1
    
    # Print final summary
    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*70)
    print(f"✓ Output directory: {results['output_dir']}")
    if 'best_total' in results.get('checkpoints', {}):
        print(f"✓ Best model: {results['checkpoints']['best_total']}")
    print("="*70 + "\n")
    
    print("Next steps:")
    print("  1. Review training logs in the output directory")
    if CONFIG['tensorboard']:
        print("  2. View TensorBoard: tensorboard --logdir=" + results['output_dir'])
    print("  3. Test inference with trainer.predict()")
    print("  4. Export model with trainer.export() for deployment")
    print("\nInference example:")
    print("  results = trainer.predict(")
    print("      checkpoint_path='path/to/checkpoint_best_total.pth',")
    print("      image_path='path/to/image.jpg',")
    print(f"      model_size='{CONFIG['model_size']}',")
    print("      threshold=0.5")
    print("  )")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
