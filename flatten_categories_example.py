"""
Example script demonstrating how to flatten/merge all categories into a single category.
This is useful when you want to detect objects (e.g., worms) without distinguishing between types.
"""

from dataset_manager import DatasetManager

def main():
    # Path to your dataset
    DATASET_PATH = "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\datasets\\WormBodyDetection.v10i.coco-segmentation"
    
    # Initialize manager
    manager = DatasetManager()
    
    print("="*70)
    print("FLATTEN CATEGORIES EXAMPLE")
    print("="*70)
    print("This will merge all worm categories into a single 'worms' category.")
    print("Original annotation files will be backed up automatically.")
    print("="*70)
    
    # Example 1: Flatten categories for all splits
    print("\n\nExample 1: Flatten all splits")
    print("-" * 70)
    results = manager.flatten_categories(
        dataset_path=DATASET_PATH,
        splits=None,  # None = process all available splits
        merged_category_name='worms',
        merged_category_id=0
    )
    
    # Show results
    print("\nFlattening Results:")
    for split, info in results['splits'].items():
        print(f"\n{split.upper()}:")
        print(f"  Original categories: {info['original_categories']}")
        print(f"  Original category names: {', '.join(info['original_category_names'][:5])}...")
        print(f"  Annotations updated: {info['annotations_updated']}/{info['total_annotations']}")
        print(f"  Backup saved to: {info['backup_file']}")
    
    # Example 2: Verify the flattened dataset
    print("\n\n" + "="*70)
    print("VERIFY FLATTENED DATASET")
    print("="*70)
    
    # Get statistics for the flattened dataset
    dataset_info = manager.load_dataset(DATASET_PATH)
    
    for split_name, split_info in dataset_info["splits"].items():
        if "coco" in split_info.get("annotations", {}):
            coco_path = split_info["annotations"]["coco"]
            print(f"\nChecking {split_name} split...")
            stats = manager.get_annotation_statistics(coco_path, verbose=True)
    
    # Example 3: How to restore from backup if needed
    print("\n\n" + "="*70)
    print("TO RESTORE CATEGORIES FROM BACKUP")
    print("="*70)
    print("If you want to restore the original categories, use:")
    print(">>> results = manager.restore_categories_from_backup(")
    print(f"...     dataset_path='{DATASET_PATH}',")
    print("...     splits=['train', 'valid'],  # or None for all")
    print("...     backup_timestamp=None  # None = most recent backup")
    print("... )")
    print("="*70)


if __name__ == "__main__":
    main()
