"""
Convert FileName_RawBF values in CellProfiler CSV output to FileName_RoboFlow
using a tab-separated lookup table.

Lookup table format (tab-separated):
    <RF_name>.png  <Original_name>.png

The CSV column FileName_RawBF typically uses .TIF extensions. Matching is done
by stripping file extensions and any trailing "_copy" suffix, case-insensitively.

Output: a new CSV alongside each input CSV with "_rfnames" appended to the stem.
"""

import os

import pandas as pd


LOOKUP_PATH = (
    r"C:\Users\lizih\Dropbox\JHU_2026_spring\NemaSeg"
    r"\datasets\RF_name_mapping\name_mapping.txt"
)
CSV_PATH = (
    r"C:\Users\lizih\Dropbox\JHU_2026_spring\NemaSeg"
    r"\datasets\Perform_test\cellprofiler_results\20260402_combined_ms_ef_cf_o_nemasegcomparison.csv"
)
# Output CSV path. If None, defaults to CSV_PATH with "_rfnames" appended to the stem.
OUTPUT_PATH: str | None = (
    r"C:\Users\lizih\Dropbox\JHU_2026_spring\NemaSeg"
    r"\datasets\Perform_test\cellprofiler_results\20260402_combined_ms_ef_cf_o_nemasegcomparison_rfnames_debug.csv"
)


def normalize(name: str) -> str:
    """Strip file extension and trailing '_copy' for comparison."""
    base = os.path.splitext(name)[0]
    if base.endswith("_copy"):
        base = base[:-5]
    return base.lower()


def load_lookup(path: str) -> dict[str, str]:
    """Build a reverse mapping: normalized_original_name -> RF_name."""
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rf_name = parts[0].strip()
            orig_name = parts[1].strip()
            key = normalize(orig_name)
            mapping[key] = rf_name
    return mapping


def process_csv(csv_path: str, lookup: dict[str, str], output_path: str | None = None) -> None:
    df = pd.read_csv(csv_path)

    if "FileName_RawBF" not in df.columns:
        print(f"SKIP (no FileName_RawBF): {csv_path}")
        return

    df["FileName_RoboFlow"] = df["FileName_RawBF"].apply(
        lambda x: lookup.get(normalize(str(x)))
    )

    unmatched = df.loc[df["FileName_RoboFlow"].isna(), "FileName_RawBF"].unique()
    if len(unmatched):
        print(f"  WARNING – {len(unmatched)} unmatched name(s) will be dropped: {list(unmatched)}")
    df = df.dropna(subset=["FileName_RoboFlow"])

    keep_cols = (
        ["FileName_RawBF", "FileName_RoboFlow"]
        + [c for c in df.columns if c.startswith("po_")]
        + [c for c in ["worm_length_um"] if c in df.columns]
    )
    df = df[keep_cols]

    if output_path:
        out_path = output_path
    else:
        stem, ext = os.path.splitext(csv_path)
        out_path = stem + "_rfnames" + ext

    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


def main() -> None:
    lookup = load_lookup(LOOKUP_PATH)
    print(f"Loaded {len(lookup)} entries from lookup table.")

    if not os.path.isfile(CSV_PATH):
        print(f"CSV file not found: {CSV_PATH}")
        return

    print(f"Processing: {CSV_PATH}")
    process_csv(CSV_PATH, lookup, OUTPUT_PATH)


if __name__ == "__main__":
    main()
