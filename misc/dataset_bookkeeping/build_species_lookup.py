"""Build a lookup table: renamed image (png) -> raw image -> species / strain.

Inputs
  image_key_filtered.tsv   raw image name <TAB> renamed png name (no header)
  training_data - Sheet1.csv   sheet with species/strain per raw image

The sheet spells raw names inconsistently (" copy.TIF", "_copy.TIF",
"_overlay", no extension), so both sides are normalised before joining.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_DIR = Path(r"C:\Users\lizih\Dropbox\JHU_2026_spring\NemaSeg\datasets\Species_test")
SHEET_COLUMNS = ["species", "strain", "condition", "assay", "phenotype"]

# Images absent from the sheet; values confirmed manually.
MANUAL_OVERRIDES: dict[str, dict[str, str]] = {
	"20210205-assayA-p012-m2X_B12.TIF": {"species": "C_elegans", "strain": "CB4856"},
	"p01-growth-H01-2X_B10.TIF": {"species": "C_elegans", "strain": "N2"},
}


def normalize(name: str) -> str:
	s = str(name).strip()
	s = re.sub(r"\.tiff?$", "", s, flags=re.I)
	s = re.sub(r"[ _]copy$", "", s, flags=re.I)
	s = re.sub(r"_overlay$", "", s, flags=re.I)
	return s.lower()


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--key-tsv", type=Path, default=DEFAULT_DIR / "image_key_filtered.tsv")
	parser.add_argument("--sheet-csv", type=Path, default=DEFAULT_DIR / "training_data - Sheet1.csv")
	parser.add_argument("--output", type=Path, default=DEFAULT_DIR / "image_species_lookup.csv")
	args = parser.parse_args()

	key = pd.read_csv(args.key_tsv, sep="\t", header=None, names=["raw_image", "png_name"])
	sheet = pd.read_csv(args.sheet_csv, usecols=["file_name", *SHEET_COLUMNS])

	key["_join"] = key["raw_image"].map(normalize)
	sheet["_join"] = sheet["file_name"].map(normalize)
	if sheet["_join"].duplicated().any():
		dups = sheet.loc[sheet["_join"].duplicated(keep=False), "file_name"].tolist()
		raise ValueError(f"Ambiguous sheet entries after normalisation: {dups}")

	merged = key.merge(sheet.drop(columns="file_name"), on="_join", how="left", indicator=True)
	merged["match_status"] = merged["_merge"].map({"both": "matched", "left_only": "not_in_sheet"})
	merged.loc[(merged["match_status"] == "matched") & merged["strain"].isna(), "match_status"] = "no_strain_in_sheet"
	for raw, values in MANUAL_OVERRIDES.items():
		mask = merged["raw_image"] == raw
		if not mask.any():
			raise ValueError(f"Override target not found in key TSV: {raw}")
		for col, val in values.items():
			merged.loc[mask, col] = val
		merged.loc[mask, "match_status"] = "manual_override"
	out = merged[["png_name", "raw_image", "species", "strain", "condition", "assay", "phenotype", "match_status"]]

	args.output.parent.mkdir(parents=True, exist_ok=True)
	out.to_csv(args.output, index=False)
	print(f"Wrote {len(out)} rows to {args.output}")
	print(out["match_status"].value_counts().to_string())
	print(out["species"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
	main()
