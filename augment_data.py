"""
Standalone script for data augmentation.
Apply various augmentations to your training dataset.
"""

from dataset_manager import DatasetManager
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Augment dataset with rotations, intensity adjustments, and noise"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to the dataset directory'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Split to augment (default: train)'
    )
    parser.add_argument(
        '--factor',
        type=int,
        default=2,
        help='Number of augmented versions per image (default: 2)'
    )
    parser.add_argument(
        '--rotations',
        type=float,
        nargs='+',
        default=None,
        help='Specific rotation angles in degrees (e.g., -15 15 -30 30). If not specified, uses random rotations.'
    )
    parser.add_argument(
        '--intensity-min',
        type=float,
        default=0.8,
        help='Minimum intensity multiplier (default: 0.8)'
    )
    parser.add_argument(
        '--intensity-max',
        type=float,
        default=1.2,
        help='Maximum intensity multiplier (default: 1.2)'
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.02,
        help='Gaussian noise sigma (0-1 range, default: 0.02)'
    )
    parser.add_argument(
        '--no-combinations',
        action='store_true',
        help='Apply only single augmentation per image instead of combinations'
    )
    parser.add_argument(
        '--suffix',
        type=str,
        default='_aug',
        help='Suffix for augmented image filenames (default: _aug)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DATA AUGMENTATION TOOL")
    print("="*70)
    print(f"\nDataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Augmentation factor: {args.factor}")
    print(f"\nAugmentation settings:")
    print(f"  Rotations: {args.rotations if args.rotations else 'Random (-30 to 30 degrees)'}")
    print(f"  Intensity range: ({args.intensity_min}, {args.intensity_max})")
    print(f"  Noise sigma: {args.noise}")
    print(f"  Combinations: {not args.no_combinations}")
    print(f"  Random seed: {args.seed}")
    print()
    
    # Create manager and augment
    manager = DatasetManager()
    
    try:
        summary = manager.augment_dataset(
            dataset_path=args.dataset,
            split=args.split,
            augmentation_factor=args.factor,
            rotation_angles=args.rotations,
            intensity_range=(args.intensity_min, args.intensity_max),
            noise_sigma=args.noise,
            combinations=not args.no_combinations,
            output_suffix=args.suffix,
            seed=args.seed
        )
        
        print("\n" + "="*70)
        print("AUGMENTATION COMPLETE!")
        print("="*70)
        print(f"Original images: {summary['original_images']}")
        print(f"Augmented images: {summary['augmented_images']}")
        print(f"Total images: {summary['total_images']}")
        print(f"Total annotations: {summary['total_annotations']}")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
