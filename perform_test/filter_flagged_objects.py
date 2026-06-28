"""Filter out flagged entries from a CellProfiler results CSV.

Removes rows where any of `edge_ObjectFlag`, `cluster_ObjectFlag`, or
`outlier_ObjectFlag` indicate the object is flagged. In this CSV, flagged
rows contain a non-empty string (e.g. "edge", "cluster", "outlier") while
unflagged rows are NA / empty.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FLAG_COLUMNS = ("edge_ObjectFlag", "cluster_ObjectFlag", "outlier_ObjectFlag")
# Strings (case-insensitive) that should be treated as "not flagged"
# in addition to actual NaN / empty values.
_UNFLAGGED_STRINGS = {"", "na", "nan", "none", "false", "0"}


def _is_flagged(series: pd.Series) -> pd.Series:
    stripped = series.astype("string").str.strip()
    lowered = stripped.str.lower()
    return stripped.notna() & ~lowered.isin(_UNFLAGGED_STRINGS)


def filter_flagged(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv, low_memory=False)

    missing = [c for c in FLAG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected flag column(s): {missing}")

    flagged_mask = pd.Series(False, index=df.index)
    per_column_counts = {}
    for col in FLAG_COLUMNS:
        col_mask = _is_flagged(df[col])
        per_column_counts[col] = int(col_mask.sum())
        flagged_mask |= col_mask

    filtered = df.loc[~flagged_mask].copy()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    total = len(df)
    removed = int(flagged_mask.sum())
    kept = len(filtered)
    print(f"Input rows:    {total}")
    for col, count in per_column_counts.items():
        print(f"  {col} flagged: {count}")
    print(f"Rows removed (any flag): {removed}")
    print(f"Rows kept:     {kept}")
    print(f"Wrote: {output_csv}")


def main() -> None:
    default_input = Path(
        r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test"
        r"\cellprofiler_results\Missing_data\20260519_cp_missing_objects.csv"
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", "-i", type=Path, default=default_input,
        help="Path to the input CellProfiler CSV.",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Path for the filtered output CSV. Defaults to "
             "<input_stem>_filtered.csv next to the input.",
    )
    args = parser.parse_args()

    input_csv: Path = args.input
    output_csv: Path = args.output or input_csv.with_name(
        f"{input_csv.stem}_filtered.csv"
    )
    filter_flagged(input_csv, output_csv)


if __name__ == "__main__":
    main()
