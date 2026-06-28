# Data Augmentation Guide

This guide explains how to use the data augmentation features in the DatasetManager.

## Features

The augmentation system supports:

1. **Rotation** - Rotate images around the center while maintaining image size
2. **Intensity Adjustment** - Adjust overall brightness of images
3. **Gaussian Noise** - Add random noise to images
4. **Combinations** - Apply multiple augmentations together

**Important:** All augmentations automatically update COCO annotations (segmentation polygons and bounding boxes) to match the transformed images!

## Quick Start

### Method 1: Using the Standalone Script

```bash
python augment_data.py --dataset "path/to/your/dataset" --factor 2
```

### Method 2: From Python Code

```python
from dataset_manager import DatasetManager

manager = DatasetManager()

# Augment with default settings
manager.augment_dataset(
    dataset_path="path/to/your/dataset",
    split='train',
    augmentation_factor=2
)
```

## Augmentation Parameters

### Basic Parameters

- `dataset_path` (str): Path to your dataset directory
- `split` (str): Which split to augment (default: 'train')
- `augmentation_factor` (int): Number of augmented versions per image (default: 2)
- `seed` (int): Random seed for reproducibility (default: 42)

### Rotation

- `rotation_angles` (list): Specific angles in degrees, e.g., `[-15, 15, -30, 30]`
  - If `None`, uses random angles between -30° and 30°

### Intensity Adjustment

- `intensity_range` (tuple): Min and max brightness multipliers
  - Default: `(0.8, 1.2)` means 80% to 120% of original brightness
  - Values > 1.0 increase brightness
  - Values < 1.0 decrease brightness

### Noise

- `noise_sigma` (float): Standard deviation for Gaussian noise
  - Range: 0.0 to 1.0 (relative to pixel intensity range)
  - Default: 0.02 (2% noise)
  - Higher values = more noise

### Combinations

- `combinations` (bool): Whether to combine multiple augmentations
  - `True`: Each augmented image has rotation + intensity + noise
  - `False`: Each augmented image has only one type of augmentation

## Examples

### Example 1: Basic Augmentation (doubles your dataset)

```python
manager.augment_dataset(
    dataset_path="C:/datasets/my_dataset",
    augmentation_factor=2  # Creates 2 augmented versions per image
)
```

### Example 2: Specific Rotations

```python
manager.augment_dataset(
    dataset_path="C:/datasets/my_dataset",
    augmentation_factor=4,
    rotation_angles=[-90, -45, 45, 90],  # 4 specific angles
    combinations=False  # Only rotation, no intensity or noise
)
```

### Example 3: Intensity and Noise Only (no rotation)

```python
manager.augment_dataset(
    dataset_path="C:/datasets/my_dataset",
    augmentation_factor=3,
    rotation_angles=[0],  # No rotation (0 degrees)
    intensity_range=(0.7, 1.3),  # More aggressive brightness change
    noise_sigma=0.05  # More noise
)
```

### Example 4: Aggressive Augmentation (5x dataset size)

```python
manager.augment_dataset(
    dataset_path="C:/datasets/my_dataset",
    augmentation_factor=5,
    rotation_angles=None,  # Random rotations
    intensity_range=(0.6, 1.4),  # Wide intensity range
    noise_sigma=0.03,
    combinations=True  # Apply all augmentations
)
```

### Example 5: Using Command Line

```bash
# Basic augmentation
python augment_data.py --dataset "C:/datasets/my_dataset" --factor 2

# With specific rotations
python augment_data.py --dataset "C:/datasets/my_dataset" \
    --factor 4 \
    --rotations -90 -45 45 90

# Aggressive augmentation with combinations
python augment_data.py --dataset "C:/datasets/my_dataset" \
    --factor 5 \
    --intensity-min 0.6 \
    --intensity-max 1.4 \
    --noise 0.05

# Single augmentation type per image
python augment_data.py --dataset "C:/datasets/my_dataset" \
    --factor 3 \
    --no-combinations
```

### Example 6: Remove and Re-augment

```python
# Remove all previously augmented images
manager.remove_augmented_images(
    dataset_path="C:/datasets/my_dataset",
    split='train',
    suffix_pattern='_aug'  # Must match the suffix used during augmentation
)

# This will:
# - Delete all image files with pattern "*_aug_*" from disk
# - Remove their entries from the COCO annotation file
# - Keep all original images untouched

# Now you can augment again with completely different settings
manager.augment_dataset(
    dataset_path="C:/datasets/my_dataset",
    augmentation_factor=5,
    rotation_angles=[-45, -22, 22, 45],  # Different angles
    intensity_range=(0.7, 1.3),  # Different range
    combinations=False  # Different strategy
)
```

**Why use this?** You might want to experiment with different augmentation strategies:
- Try different rotation angles
- Test single vs. combined augmentations
- Adjust intensity or noise levels
- Change augmentation factor

Without this function, you'd need to manually delete augmented images or restore from backup.

## What Happens During Augmentation?

1. **Reads** the COCO annotation file from the specified split
2. **For each image**:
   - Creates N augmented versions (based on `augmentation_factor`)
   - Applies transformations to the image
   - Transforms segmentation polygons and bounding boxes accordingly
   - Saves augmented images with suffix (e.g., `image_aug_0.jpg`, `image_aug_1.jpg`)
3. **Updates** the COCO annotation file with:
   - New image entries for augmented images
   - New annotation entries with transformed coordinates
   - Preserved category information

## Tips

1. **Start small**: Begin with `augmentation_factor=2` and see the results
2. **Check visually**: Use the visualization function to verify augmentations look correct
3. **Train first**: Always split train/validation before augmenting (only augment training data)
4. **Combinations**: Use `combinations=True` for more diverse augmentations
5. **Reproducibility**: Keep the same `seed` value for consistent results

## Typical Workflow

```python
from dataset_manager import DatasetManager

manager = DatasetManager()

# 1. Split data (create validation set)
manager.split_train_validation(
    dataset_path="C:/datasets/my_dataset",
    val_ratio=0.2
)

# 2. Augment ONLY the training set
manager.augment_dataset(
    dataset_path="C:/datasets/my_dataset",
    split='train',  # Only augment training data
    augmentation_factor=3
)

# 3. Visualize to check
manager.visualize_dataset_split(
    dataset_path="C:/datasets/my_dataset",
    split='train',
    num_images=5
)

# 4. Train your model!
```

## Workflow with Experimentation

If you want to try different augmentation strategies:

```python
from dataset_manager import DatasetManager

manager = DatasetManager()

# 1. Try first augmentation strategy
print("Trying augmentation strategy 1...")
manager.augment_dataset(
    dataset_path="C:/datasets/my_dataset",
    split='train',
    augmentation_factor=2,
    combinations=True
)

# 2. Visualize and check results
manager.visualize_dataset_split(
    dataset_path="C:/datasets/my_dataset",
    split='train',
    num_images=5
)

# 3. Not satisfied? Remove and try different parameters
print("\nRemoving previous augmentations...")
manager.remove_augmented_images(
    dataset_path="C:/datasets/my_dataset",
    split='train'
)

# 4. Try different strategy
print("Trying augmentation strategy 2...")
manager.augment_dataset(
    dataset_path="C:/datasets/my_dataset",
    split='train',
    augmentation_factor=5,
    rotation_angles=[-45, -22, 0, 22, 45],
    combinations=False  # Different approach
)

# 5. Visualize again
manager.visualize_dataset_split(
    dataset_path="C:/datasets/my_dataset",
    split='train',
    num_images=5
)

# 6. When satisfied, train your model!
```

## Troubleshooting

**Q: My augmented images look wrong**
- Check that OpenCV is properly installed: `pip install opencv-python`
- Verify annotation file is in COCO format

**Q: Annotations don't match the images**
- This shouldn't happen! The code automatically transforms both images and annotations
- If you see mismatches, please report it as a bug

**Q: How much should I augment?**
- For small datasets (<100 images): augmentation_factor of 5-10
- For medium datasets (100-1000 images): augmentation_factor of 2-4
- For large datasets (>1000 images): augmentation_factor of 1-2

**Q: Should I augment validation data?**
- **NO!** Only augment training data. Keep validation data untouched to get accurate performance metrics.

**Q: The remove function didn't delete all augmented images**
- Make sure `suffix_pattern` matches what you used in `output_suffix` during augmentation
- Default is `'_aug'`, so augmented images are named like `image_aug_0.jpg`, `image_aug_1.jpg`
- If you used a different suffix (e.g., `'_augmented'`), you must specify it when removing

**Q: Can I safely remove augmented images if I'm not sure?**
- Yes! The function only removes images with the specific suffix pattern
- Original images are never touched
- You can always re-augment if needed

