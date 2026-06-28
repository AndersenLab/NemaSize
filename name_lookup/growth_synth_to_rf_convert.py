"""
Convert FileName_RawBF values produced by the toxin-pipeline workaround back
to RoboFlow image names.

Background:
    perform_test/cluster_workaround/rename_growth_for_toxin_pipeline.sh
    creates symlinks of the form
        <YYYYMMDD>-growth<HID>-<pNN>-m2X_<WELL>.TIF
    e.g. 20260518-growthH01-p01-m2X_A02.TIF
    so that the andersenlab/cellprofiler-nf toxin pipeline accepts the
    "growth" images, whose real names look like
        p01-growth-H01-2X_A02.TIF

    The CellProfiler CSV therefore records the synthesized name in the
    FileName_RawBF column. To compare those rows against RoboFlow-named
    predictions we need to:
        1. Reverse the rename to recover the original growth filename:
               20260518-growthH01-p01-m2X_A02.TIF
            -> p01-growth-H01-2X_A02.TIF
        2. Look up the RoboFlow name (e.g. 179.png) in image_key.tsv,
           whose lines look like
               p01-growth-H01-2X_A02.TIF<TAB>179.png

Output: a CSV next to the input (or at OUTPUT_PATH) keeping
FileName_RawBF, FileName_RoboFlow, all po_* columns, and worm_length_um
when present. Rows whose FileName_RawBF cannot be mapped are dropped with
a warning.
"""

import os
import re

import pandas as pd


LOOKUP_PATH = (
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg"
    r"\datasets\RF_name_mapping_new\image_key.tsv"
)
CSV_PATH = (
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg"
    r"\datasets\Perform_test\cellprofiler_results\Missing_data"
    r"\20260519_cp_missing_objects_filtered.csv"
)
# Output CSV path. If None, defaults to CSV_PATH with "_rfnames" appended.
OUTPUT_PATH: str | None = None


# Synthesized name written by rename_growth_for_toxin_pipeline.sh:
#   <DATE>-growth<HID>-<pNN>-m2X_<WELL>.TIF
SYNTH_RE = re.compile(
    r"^(?P<date>\d{8})-growth(?P<hid>H\d+)-(?P<plate>p\d+)-m2X_(?P<well>[A-Z]\d{2})\.TIF$",
    re.IGNORECASE,
)


def synth_to_original(name: str) -> str | None:
    """Reverse the rename script: synthesized name -> original growth name.

    Returns None if `name` does not match the synthesized pattern.
    """
    m = SYNTH_RE.match(str(name).strip())
    if not m:
        return None
    return f"{m['plate']}-growth-{m['hid']}-2X_{m['well']}.TIF"


def normalize(name: str) -> str:
    """Lowercase + strip extension for case-insensitive matching."""
    return os.path.splitext(str(name))[0].lower()


def load_lookup(path: str) -> dict[str, str]:
    """Build mapping: normalized_original_name -> RF_name.

    image_key.tsv format: <original>.TIF\t<rf>.png
    """
    mapping: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            orig_name = parts[0].strip()
            rf_name = parts[1].strip()
            if not orig_name or not rf_name:
                continue
            mapping[normalize(orig_name)] = rf_name
    return mapping


def process_csv(csv_path: str, lookup: dict[str, str], output_path: str | None = None) -> None:
    df = pd.read_csv(csv_path)

    if "FileName_RawBF" not in df.columns:
        print(f"SKIP (no FileName_RawBF): {csv_path}")
        return

    original_names = df["FileName_RawBF"].apply(lambda x: synth_to_original(str(x)))

    bad_synth = df.loc[original_names.isna(), "FileName_RawBF"].unique()
    if len(bad_synth):
        print(
            f"  WARNING – {len(bad_synth)} value(s) did not match the synthesized "
            f"growth pattern and will be dropped: {list(bad_synth)}"
        )

    df = df.assign(_orig_growth_name=original_names)

    df["FileName_RoboFlow"] = df["_orig_growth_name"].apply(
        lambda x: lookup.get(normalize(x)) if isinstance(x, str) else None
    )

    unmatched_lookup = df.loc[
        df["_orig_growth_name"].notna() & df["FileName_RoboFlow"].isna(),
        "_orig_growth_name",
    ].unique()
    if len(unmatched_lookup):
        print(
            f"  WARNING – {len(unmatched_lookup)} reversed name(s) not found in "
            f"lookup and will be dropped: {list(unmatched_lookup)}"
        )

    df = df.dropna(subset=["FileName_RoboFlow"]).drop(columns=["_orig_growth_name"])

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
    print(f"Saved: {out_path}  ({len(df)} rows)")


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
