# RF-DETR Segmentation Training Guide

## Overview

This guide provides information on training RF-DETR (Real-Time DETR) instance segmentation models and compares them with YOLO segmentation models.

## What is RF-DETR?

RF-DETR is a real-time transformer architecture for object detection and instance segmentation developed by Roboflow. Built on a DINOv2 vision transformer backbone, RF-DETR delivers state-of-the-art accuracy and latency trade-offs.

**Key Features:**
- Transformer-based architecture (vs YOLO's CNN-based)
- State-of-the-art accuracy for real-time models
- Supports both detection and segmentation
- Multiple model sizes from Nano to 2XLarge
- Apache 2.0 license (core models)

## Installation

```bash
# Basic installation
pip install rfdetr

# With logging support (TensorBoard, Weights & Biases)
pip install rfdetr[metrics]

# With ONNX export support
pip install rfdetr[onnxexport]

# For visualization during inference
pip install supervision
```

## Dataset Format

### COCO Format (RF-DETR Native)

RF-DETR expects datasets in COCO JSON format:

```
dataset/
├── train/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── valid/
│   ├── _annotations.coco.json
│   ├── image1.jpg
│   └── ...
└── test/  (optional)
    ├── _annotations.coco.json
    └── ...
```

**Your existing dataset (`WormBodyDetection.v10i.coco-segmentation`) is already in the correct format!**

## RF-DETR vs YOLO: Key Differences

| Aspect | RF-DETR | YOLO |
|--------|---------|------|
| **Architecture** | Vision Transformer (DINOv2) | CNN-based |
| **Approach** | DETR (DEtection TRansformer) | Anchor-based / Anchor-free |
| **Dataset Format** | COCO JSON (native) | YOLO format (native), COCO (supported) |
| **Training Style** | Simpler hyperparameters | More hyperparameters to tune |
| **Batch Size** | Explicit gradient accumulation | Standard batching |
| **Best For** | State-of-the-art accuracy | Speed/efficiency balance |

## Model Sizes

### RF-DETR Segmentation Models

| Model | Size | mAP (COCO) | Latency (T4) | Resolution | Memory |
|-------|------|------------|--------------|------------|--------|
| RF-DETR-Seg-N | Nano | 40.3 | 3.4ms | 312×312 | ~4 GB |
| RF-DETR-Seg-S | Small | 43.1 | 4.4ms | 384×384 | ~6 GB |
| RF-DETR-Seg-M | Medium | 45.3 | 5.9ms | 432×432 | ~8 GB |
| RF-DETR-Seg-L | Large | 47.1 | 8.8ms | 504×504 | ~12 GB |
| RF-DETR-Seg-XL | XLarge | 48.8 | 13.5ms | 624×624 | ~16 GB |
| RF-DETR-Seg-2XL | 2XLarge | 49.9 | 21.8ms | 768×768 | ~24 GB |

**Recommendation for RTX 3090 (24GB):** Start with Medium or Large

## Training Configuration

### Basic Training

```python
from train_rfdetr_segmentation import RFDETRSegmentationTrainer

trainer = RFDETRSegmentationTrainer()

results = trainer.train(
    dataset_dir="path/to/dataset",
    model_size='m',  # Options: 'n', 's', 'm', 'l', 'xl', '2xl'
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,  # Effective batch size = 4*4 = 16
    lr=1e-4,
    output_dir='output/rfdetr_seg'
)
```

### Understanding Batch Size

RF-DETR uses gradient accumulation for effective large batch sizes:

**Effective Batch Size = batch_size × grad_accum_steps × num_gpus**

Recommended configurations for RTX 3090 (24GB):
- Small model: `batch_size=8, grad_accum_steps=2` (effective=16)
- Medium model: `batch_size=4, grad_accum_steps=4` (effective=16)
- Large model: `batch_size=2, grad_accum_steps=8` (effective=16)

### Key Training Parameters

```python
CONFIG = {
    # Model
    'model_size': 'm',  # n, s, m, l, xl, 2xl
    
    # Core hyperparameters
    'epochs': 100,
    'batch_size': 4,  # Per-GPU batch size
    'grad_accum_steps': 4,  # Gradient accumulation steps
    'lr': 1e-4,  # Learning rate
    'lr_encoder': None,  # If None, uses lr * 1.5
    'weight_decay': 1e-4,
    
    # Resolution (optional, uses model default if None)
    'resolution': None,  # Can be 560, 672, 784, 896, etc.
    
    # Hardware
    'device': 'cuda',  # 'cuda', 'cpu', or 'mps'
    'gradient_checkpointing': False,  # Reduce memory, slower training
    
    # Model behavior
    'use_ema': True,  # Exponential Moving Average
    'checkpoint_interval': 10,  # Save every N epochs
    
    # Logging
    'tensorboard': True,
    'wandb': False,
    'wandb_project': None,
    'wandb_run': None,
    
    # Early stopping
    'early_stopping': False,
    'early_stopping_patience': 10,
    'early_stopping_min_delta': 0.001,
}
```

## Quick Start

### 1. Install RF-DETR

```bash
pip install rfdetr supervision
# Optional: pip install rfdetr[metrics]
```

### 2. Train a Model

```bash
python train_rfdetr_segmentation.py
```

The script is pre-configured for your worm dataset!

### 3. Monitor Training

If TensorBoard is enabled:
```bash
tensorboard --logdir=path/to/output_dir
```

### 4. Run Inference

```python
from train_rfdetr_segmentation import RFDETRSegmentationTrainer

trainer = RFDETRSegmentationTrainer()

results = trainer.predict(
    checkpoint_path='output/checkpoint_best_total.pth',
    image_path='test_image.jpg',
    model_size='m',
    threshold=0.5,
    save_output='annotated_output.jpg'
)
```

## Checkpoints Explained

RF-DETR saves multiple checkpoints during training:

| Checkpoint | Description | Use Case |
|------------|-------------|----------|
| `checkpoint.pth` | Most recent checkpoint | Resume training |
| `checkpoint_<N>.pth` | Periodic checkpoints | Backup, analysis |
| `checkpoint_best_ema.pth` | Best with EMA weights | Usually best for deployment |
| `checkpoint_best_regular.pth` | Best with raw weights | Alternative choice |
| `checkpoint_best_total.pth` | **Final best model** | **Use this for inference** |

**Recommendation:** Use `checkpoint_best_total.pth` for inference and deployment.

## Advanced Features

### Resume Training

```python
results = trainer.train(
    dataset_dir="path/to/dataset",
    resume='path/to/checkpoint.pth',  # Resume from checkpoint
    # ... other parameters
)
```

### Early Stopping

```python
results = trainer.train(
    dataset_dir="path/to/dataset",
    early_stopping=True,
    early_stopping_patience=15,
    early_stopping_min_delta=0.005,
    # ... other parameters
)
```

### Custom Resolution

```python
results = trainer.train(
    dataset_dir="path/to/dataset",
    resolution=784,  # Must be divisible by 56
    # ... other parameters
)
```

### Weights & Biases Integration

```python
results = trainer.train(
    dataset_dir="path/to/dataset",
    wandb=True,
    wandb_project='worm-segmentation',
    wandb_run='rfdetr-medium-001',
    # ... other parameters
)
```

## ONNX Export

For deployment optimization:

```python
trainer = RFDETRSegmentationTrainer()

# First install the export extension
# pip install rfdetr[onnxexport]

trainer.export(
    checkpoint_path='output/checkpoint_best_total.pth',
    model_size='m',
    output_dir='exported_models'
)
```

## Comparison: Training Commands

### YOLO Training
```python
# YOLO requires YAML config, converts formats
trainer = YOLOSegmentationTrainer()
yaml_path = trainer.prepare_yolo_dataset_config(dataset_path)
results = trainer.train(
    dataset_yaml=yaml_path,
    model_size='x',
    epochs=100,
    batch_size=8,
    lr0=0.01,
    optimizer='SGD',
    # Many more hyperparameters...
)
```

### RF-DETR Training
```python
# RF-DETR uses COCO format directly, fewer hyperparameters
trainer = RFDETRSegmentationTrainer()
results = trainer.train(
    dataset_dir=dataset_path,  # No YAML needed
    model_size='m',
    epochs=100,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    # Simpler configuration
)
```

## Tips for Your Worm Dataset

1. **Start with Medium model** (`model_size='m'`) - good balance of accuracy and speed
2. **Use default resolution** - models are pre-optimized
3. **Enable EMA** (`use_ema=True`) - usually improves final performance
4. **Monitor with TensorBoard** - easy visualization of training progress
5. **Try lower batch size** if you hit memory limits with RTX 3090

## Expected Training Time

On RTX 3090 (24GB), for your worm dataset (~1000 images):
- **Nano:** ~15-20 minutes for 100 epochs
- **Small:** ~20-30 minutes for 100 epochs
- **Medium:** ~30-45 minutes for 100 epochs
- **Large:** ~45-60 minutes for 100 epochs

## Performance Comparison Strategy

To compare YOLO vs RF-DETR on your worm dataset:

1. **Train both models** with similar configurations (100 epochs)
2. **Compare metrics:**
   - mAP (mean Average Precision)
   - Training time
   - Inference speed
   - Model size
3. **Test on real data** - visual quality matters!
4. **Consider deployment** - which model better fits your inference environment?

## Troubleshooting

### Out of Memory
- Reduce `batch_size` (e.g., from 4 to 2)
- Increase `grad_accum_steps` to compensate
- Enable `gradient_checkpointing=True`
- Use a smaller model size

### Slow Training
- Disable `gradient_checkpointing`
- Reduce logging frequency
- Use fewer workers if I/O bound

### Model Not Converging
- Try reducing learning rate (e.g., `lr=5e-5`)
- Increase `grad_accum_steps` for larger effective batch
- Enable early stopping to catch convergence issues

## Resources

- **Documentation:** https://rfdetr.roboflow.com/
- **PyPI:** https://pypi.org/project/rfdetr/
- **GitHub:** https://github.com/roboflow/rf-detr
- **Paper:** https://arxiv.org/abs/2511.09554
- **Tutorial Video:** How to Train RF-DETR on Custom Dataset

## Next Steps

1. Run `python train_rfdetr_segmentation.py` to start training
2. Monitor training with TensorBoard
3. Compare results with your YOLO26 model
4. Test inference on validation images
5. Choose the best model for your application!

Good luck with your training! 🚀
