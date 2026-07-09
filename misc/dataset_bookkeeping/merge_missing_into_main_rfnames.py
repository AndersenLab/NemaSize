"""
Merge the "missing data" RF-named CSV into the main combined RF-named CSV
and save the result to the Merged_data directory.

Inputs:
    MAIN_CSV    – the main combined CSV (already converted to RF names).
    EXTRA_CSV   – the missing-data CSV (already converted to RF names by
                  growth_synth_to_rf_convert.py).

Output:
    OUTPUT_DIR / OUTPUT_NAME – vertical concat of the two, with column
    union preserved (MAIN_CSV column order first, then any columns that
    exist only in EXTRA_CSV appended at the end).
"""

import os

import pandas as pd


MAIN_CSV = (
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test"
    r"\cellprofiler_results\20260402_combined_ms_ef_cf_o_nemasegcomparison_rfnames.csv"
)
EXTRA_CSV = (
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test"
    r"\cellprofiler_results\Missing_data"
    r"\20260519_cp_missing_objects_filtered_rfnames.csv"
)
OUTPUT_DIR = (
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test"
    r"\cellprofiler_results\Merged_data"
)
OUTPUT_NAME = "20260402_combined_ms_ef_cf_o_nemasegcomparison_rfnames_merged.csv"


def main() -> None:
    if not os.path.isfile(MAIN_CSV):
        raise FileNotFoundError(MAIN_CSV)
    if not os.path.isfile(EXTRA_CSV):
        raise FileNotFoundError(EXTRA_CSV)

    df_main = pd.read_csv(MAIN_CSV)
    df_extra = pd.read_csv(EXTRA_CSV)

    print(f"Main : {df_main.shape}  ({MAIN_CSV})")
    print(f"Extra: {df_extra.shape}  ({EXTRA_CSV})")

    only_in_main = [c for c in df_main.columns if c not in df_extra.columns]
    only_in_extra = [c for c in df_extra.columns if c not in df_main.columns]
    if only_in_main:
        print(f"  Columns only in main  ({len(only_in_main)}): {only_in_main}")
    if only_in_extra:
        print(f"  Columns only in extra ({len(only_in_extra)}): {only_in_extra}")

    merged = pd.concat([df_main, df_extra], ignore_index=True, sort=False)
    # Keep main column order first, then any columns unique to extra.
    ordered_cols = list(df_main.columns) + only_in_extra
    merged = merged[ordered_cols]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
    merged.to_csv(out_path, index=False)

    print(f"Merged: {merged.shape}")
    print(f"Saved : {out_path}")


if __name__ == "__main__":
    main()
