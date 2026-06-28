#!/usr/bin/env python3
"""
Lookup converted image names and output original .tif names.

Expected converted format example:
    10_png.rf.2a5422c48a272b06a5ed1e4811e31377.jpg

Expected mapping file line example (tab-separated preferred):
    10.png    20210205-assayA-p007-m2X_H11.png

Output original should always use .tif extension:
    20210205-assayA-p007-m2X_H11.tif
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


# Edit these three paths directly in the script.
CONFIG_IMAGES_DIR = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\WormBodyDetection.v10i.yolo26\train\images"
)
CONFIG_MAPPING_FILE = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\RF_name_mapping\name_mapping.txt"
)
CONFIG_OUTPUT_FILE = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\Orig_img_names\original_name_all.csv"
)


# Matches names like "10_png.rf.<hash>.jpg" and captures "10" as the stem.
CONVERTED_NAME_PATTERN = re.compile(
    r"^(?P<stem>.+?)_png\.rf\.[^.]+\.(?:jpg|jpeg|png)$", re.IGNORECASE
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map converted image filenames to original .tif names."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=CONFIG_IMAGES_DIR,
        help="Directory containing converted images.",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=CONFIG_MAPPING_FILE,
        help="Lookup table file path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG_OUTPUT_FILE,
        help="Output CSV file path.",
    )
    return parser.parse_args()


def load_mapping(mapping_file: Path) -> dict[str, str]:
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    mapping: dict[str, str] = {}
    with mapping_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for raw_line in reader:
            if not raw_line:
                continue

            if len(raw_line) >= 2:
                converted_key = raw_line[0].strip()
                mapped_name = raw_line[1].strip()
            else:
                # Fallback for whitespace-delimited lines.
                parts = raw_line[0].split()
                if len(parts) < 2:
                    continue
                converted_key = parts[0].strip()
                mapped_name = parts[1].strip()

            if converted_key and mapped_name:
                mapping[converted_key] = mapped_name

    return mapping


def converted_filename_to_key(filename: str) -> str | None:
    match = CONVERTED_NAME_PATTERN.match(filename)
    if not match:
        return None
    stem = match.group("stem")
    return f"{stem}.png"


def to_tif_name(filename: str) -> str:
    stem = Path(filename).stem
    if stem.lower().endswith("_copy"):
        stem = stem[:-5]
    return f"{stem}.tif"


def resolve_names(images_dir: Path, mapping: dict[str, str]) -> list[tuple[str, str]]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not images_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {images_dir}")

    results: list[tuple[str, str]] = []
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file():
            continue

        key = converted_filename_to_key(image_path.name)
        if key is None:
            continue

        mapped_png = mapping.get(key)
        if mapped_png is None:
            results.append((image_path.name, "NOT_FOUND"))
            continue

        results.append((image_path.name, to_tif_name(mapped_png)))

    return results


def write_output(results: list[tuple[str, str]], output_file: Path | None) -> None:
    if output_file is None:
        print("converted_name,original_tif_name")
        for converted_name, original_tif_name in results:
            print(f"{converted_name},{original_tif_name}")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["converted_name", "original_tif_name"])
        writer.writerows(results)


def main() -> int:
    args = parse_args()
    try:
        mapping = load_mapping(args.mapping_file)
        results = resolve_names(args.images_dir, mapping)
        write_output(results, args.output)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
