#!/usr/bin/env python3
"""
Create a copy of a CSV and add an `original_tif_name` column based on converted names.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


# Edit these paths/column names in-script.
CONFIG_INPUT_CSV = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Pipeline\Skeleton_infoParsed\worm_lengths.csv"
)
CONFIG_OUTPUT_CSV = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\RF_name_mapping\worm_lengths_validset.csv"
)
CONFIG_MAPPING_FILE = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\RF_name_mapping\name_mapping.txt"
)
CONFIG_CONVERTED_NAME_COLUMN = "image"
CONFIG_OUTPUT_COLUMN = "original_tif_name"
CONFIG_NOT_FOUND_VALUE = "NOT_FOUND"


# Matches names like "10_png.rf.<hash>.jpg" and captures "10" as the stem.
CONVERTED_NAME_PATTERN = re.compile(
    r"^(?P<stem>.+?)_png\.rf\.[^.]+\.(?:jpg|jpeg|png)$", re.IGNORECASE
)


def load_mapping(mapping_file: Path) -> tuple[dict[str, str], dict[str, str]]:
    if not mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_file}")

    mapping_exact: dict[str, str] = {}
    mapping_lower: dict[str, str] = {}
    with mapping_file.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for raw_line in reader:
            if not raw_line:
                continue

            if len(raw_line) >= 2:
                converted_key = raw_line[0].strip()
                mapped_name = raw_line[1].strip()
            else:
                parts = raw_line[0].split()
                if len(parts) < 2:
                    continue
                converted_key = parts[0].strip()
                mapped_name = parts[1].strip()

            if converted_key and mapped_name:
                mapping_exact[converted_key] = mapped_name
                mapping_lower[converted_key.lower()] = mapped_name

    return mapping_exact, mapping_lower


def normalized_lookup_key(raw_name: str) -> str | None:
    name = Path(raw_name.strip()).name
    if not name:
        return None

    match = CONVERTED_NAME_PATTERN.match(name)
    if match:
        return f"{match.group('stem')}.png"

    if name.lower().endswith(".png"):
        return name

    return None


def to_tif_name(mapped_name: str) -> str:
    stem = Path(mapped_name).stem
    if stem.lower().endswith("_copy"):
        stem = stem[:-5]
    return f"{stem}.tif"


def lookup_original_tif(
    converted_name: str,
    mapping_exact: dict[str, str],
    mapping_lower: dict[str, str],
) -> str:
    key = normalized_lookup_key(converted_name)
    if key is None:
        return CONFIG_NOT_FOUND_VALUE

    mapped_name = mapping_exact.get(key)
    if mapped_name is None:
        mapped_name = mapping_lower.get(key.lower())
    if mapped_name is None:
        return CONFIG_NOT_FOUND_VALUE

    return to_tif_name(mapped_name)


def add_column_and_write_copy() -> None:
    if not CONFIG_INPUT_CSV.exists():
        raise FileNotFoundError(f"Input CSV not found: {CONFIG_INPUT_CSV}")

    mapping_exact, mapping_lower = load_mapping(CONFIG_MAPPING_FILE)

    with CONFIG_INPUT_CSV.open("r", encoding="utf-8", newline="") as in_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        if CONFIG_CONVERTED_NAME_COLUMN not in reader.fieldnames:
            raise ValueError(
                f"Column '{CONFIG_CONVERTED_NAME_COLUMN}' not found in input CSV."
            )

        output_fields = list(reader.fieldnames)
        if CONFIG_OUTPUT_COLUMN not in output_fields:
            output_fields.append(CONFIG_OUTPUT_COLUMN)

        CONFIG_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with CONFIG_OUTPUT_CSV.open("w", encoding="utf-8", newline="") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                converted_name = row.get(CONFIG_CONVERTED_NAME_COLUMN, "")
                row[CONFIG_OUTPUT_COLUMN] = lookup_original_tif(
                    converted_name, mapping_exact, mapping_lower
                )
                writer.writerow(row)


def main() -> int:
    try:
        add_column_and_write_copy()
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
