"""
Quick script to restore categories from backup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset_manager import DatasetManager

def main():
    DATASET_PATH = "C:\\Users\\jl200\\Dropbox\\JHU_2026_spring\\NemaSeg\\datasets\\WormBodyDetection.v10i.coco-segmentation"
    
    manager = DatasetManager()
    
    print("\n" + "="*70)
    print("RESTORING ORIGINAL CATEGORIES FROM BACKUP")
    print("="*70)
    
    # Restore the most recent backup for all splits
    results = manager.restore_categories_from_backup(
        dataset_path=DATASET_PATH,
        splits=None,  # None = restore all available splits
        backup_timestamp=None  # None = use most recent backup
    )
    
    print("\n✅ Categories restored successfully!")
    print("\nVerifying restoration...")
    
    # Verify by checking statistics
    dataset_info = manager.load_dataset(DATASET_PATH)
    
    for split_name, split_info in dataset_info["splits"].items():
        if "coco" in split_info.get("annotations", {}):
            coco_path = split_info["annotations"]["coco"]
            print(f"\n{split_name.upper()} split:")
            stats = manager.get_annotation_statistics(coco_path, verbose=False)
            print(f"  Categories: {stats['total_categories']}")
            print(f"  Category names: {', '.join(stats['category_names'][:5])}...")
            print(f"  Total annotations: {stats['total_annotations']}")


if __name__ == "__main__":
    main()
