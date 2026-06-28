"""
Match worm masks between human annotations and CellProfiler detections.

The script computes bounding-box IoU for every human/CellProfiler pair,
finds a one-to-one assignment that maximizes total IoU, and then filters
matches by an IoU threshold.

CellProfiler provides bounding boxes and length measurements in a CSV
spreadsheet (no masks), so matching uses bounding-box IoU rather than
pixel-level mask IoU.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

PathOrPaths = Path | list[Path]


@dataclass(frozen=True)
class MatchConfig:
	human_dir: PathOrPaths
	cellprofiler_csv: Path
	human_measurements_csv: PathOrPaths
	output_dir: PathOrPaths
	iou_threshold: float
	save_iou_matrix: bool
	show_progress: bool


@dataclass(frozen=True)
class MatchRunConfig:
	human_dir: Path
	cellprofiler_csv: Path
	human_measurements_csv: Path
	output_dir: Path
	iou_threshold: float
	save_iou_matrix: bool
	show_progress: bool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Each path field accepts either:
# - a single Path (one run), or
# - a list[Path] (batch runs, paired by index; single-item lists are broadcast).
#
# To add/remove GT categories, edit HUMAN_GROUPS only.
HUMAN_GROUPS: list[str] = []

GT_ROIS_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\GT_rois"
)
CELLPROFILER_CSV = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\cellprofiler_results\Merged_data"
	r"\20260402_combined_ms_ef_cf_o_nemasegcomparison_rfnames_merged.csv"
)
OUTPUT_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\GT_vs_CellProf_missingvalid_added"
)


def resolve_human_groups(gt_rois_root: Path, configured_groups: list[str]) -> list[str]:
	if configured_groups:
		return configured_groups

	if not gt_rois_root.exists() or not gt_rois_root.is_dir():
		raise FileNotFoundError(f"GT_ROIS_ROOT not found or not a directory: {gt_rois_root}")

	auto_groups = sorted(
		path.name
		for path in gt_rois_root.iterdir()
		if path.is_dir() and (path / "train" / "skeleton" / "target_masks_original").is_dir()
	)

	if not auto_groups:
		raise RuntimeError(
			"HUMAN_GROUPS is empty and no valid categories were found under GT_ROIS_ROOT. "
			"Expected folders like <category>/train/skeleton/target_masks_original."
		)

	return auto_groups


RESOLVED_HUMAN_GROUPS = resolve_human_groups(GT_ROIS_ROOT, HUMAN_GROUPS)

CONFIG = MatchConfig(
	human_dir=[
		GT_ROIS_ROOT / group / "train" / "skeleton" / "target_masks_original"
		for group in RESOLVED_HUMAN_GROUPS
	],
	cellprofiler_csv=CELLPROFILER_CSV,

	human_measurements_csv=[
		GT_ROIS_ROOT / group / "train" / "skeleton" / "worm_lengths.csv"
		for group in RESOLVED_HUMAN_GROUPS
	],

	output_dir=[OUTPUT_ROOT / group for group in RESOLVED_HUMAN_GROUPS],

	iou_threshold=0.30,
	save_iou_matrix=False,
	show_progress=True,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaskItem:
	path: Path
	area: int
	image_number: str
	bbox: tuple[int, int, int, int] | None  # (x_min, y_min, x_max, y_max)


@dataclass(frozen=True)
class CPItem:
	source_filename: str          # e.g. "143.png"
	image_number: str             # e.g. "143"
	bbox: tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
	length_um: float
	row_index: int                # row in the original CSV


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------
def extract_image_number_from_mask(path: Path) -> str:
	"""Extract image number from a GT mask filename.

	Example:
		143_png.rf.e21cdb67d45225a97432fb9511bacd58_roi_19.png -> "143"
	"""
	name = path.stem
	if "_png" in name:
		return name.split("_png", 1)[0]
	if "_jpg" in name:
		return name.split("_jpg", 1)[0]
	if "_jpeg" in name:
		return name.split("_jpeg", 1)[0]
	if "_tif" in name:
		return name.split("_tif", 1)[0]
	return name


def extract_image_number_from_cp(filename: str) -> str:
	"""Extract image number from a CellProfiler filename.

	Example:
		143.png -> "143"
	"""
	return Path(filename).stem


# ---------------------------------------------------------------------------
# File listing / loading
# ---------------------------------------------------------------------------
def list_mask_files(folder: Path) -> list[Path]:
	if not folder.exists() or not folder.is_dir():
		raise FileNotFoundError(f"Mask folder not found: {folder}")

	files = [
		path
		for path in folder.iterdir()
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	]
	files.sort()
	return files


def load_binary_mask(path: Path) -> np.ndarray:
	img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
	if img is None:
		raise ValueError(f"Failed to read mask image: {path}")

	if img.ndim == 3:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	return img > 0


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
	ys, xs = np.where(mask)
	if len(xs) == 0:
		return None
	return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def load_masks(paths: Iterable[Path], show_progress: bool, desc: str) -> list[MaskItem]:
	items: list[MaskItem] = []
	iterator = tqdm(list(paths), desc=desc, disable=not show_progress)
	for path in iterator:
		mask = load_binary_mask(path)
		area = int(mask.sum())
		items.append(
			MaskItem(
				path=path,
				area=area,
				image_number=extract_image_number_from_mask(path),
				bbox=mask_bbox(mask),
			)
		)
	return items


def load_cellprofiler_csv(csv_path: Path) -> list[CPItem]:
	if not csv_path.exists():
		raise FileNotFoundError(f"CellProfiler CSV not found: {csv_path}")

	df = pd.read_csv(csv_path)
	df.columns = [
		str(c).strip().replace("\r", "").replace("\n", "").replace(" ", "")
		for c in df.columns
	]

	required_cols = [
		"FileName_RoboFlow",
		"po_AreaShape_BoundingBoxMinimum_X",
		"po_AreaShape_BoundingBoxMinimum_Y",
		"po_AreaShape_BoundingBoxMaximum_X",
		"po_AreaShape_BoundingBoxMaximum_Y",
		"worm_length_um",
	]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise KeyError(f"CellProfiler CSV missing columns: {missing}. Available: {list(df.columns)}")

	items: list[CPItem] = []
	for idx, row in df.iterrows():
		filename = str(row["FileName_RoboFlow"])
		bbox = (
			int(row["po_AreaShape_BoundingBoxMinimum_X"]),
			int(row["po_AreaShape_BoundingBoxMinimum_Y"]),
			int(row["po_AreaShape_BoundingBoxMaximum_X"]),
			int(row["po_AreaShape_BoundingBoxMaximum_Y"]),
		)
		items.append(
			CPItem(
				source_filename=filename,
				image_number=extract_image_number_from_cp(filename),
				bbox=bbox,
				length_um=float(row["worm_length_um"]),
				row_index=int(idx),
			)
		)
	return items


# ---------------------------------------------------------------------------
# Bounding-box IoU
# ---------------------------------------------------------------------------
def bbox_area(box: tuple[int, int, int, int]) -> int:
	x0, y0, x1, y1 = box
	return max(0, x1 - x0) * max(0, y1 - y0)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
	ax0, ay0, ax1, ay1 = a
	bx0, by0, bx1, by1 = b

	ix0 = max(ax0, bx0)
	iy0 = max(ay0, by0)
	ix1 = min(ax1, bx1)
	iy1 = min(ay1, by1)

	inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
	if inter == 0:
		return 0.0

	area_a = bbox_area(a)
	area_b = bbox_area(b)
	union = area_a + area_b - inter
	if union == 0:
		return 0.0

	return inter / union


def bboxes_overlap(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> bool:
	if a is None or b is None:
		return False
	ax0, ay0, ax1, ay1 = a
	bx0, by0, bx1, by1 = b
	if ax1 < bx0 or bx1 < ax0:
		return False
	if ay1 < by0 or by1 < ay0:
		return False
	return True


# ---------------------------------------------------------------------------
# IoU matrix and matching
# ---------------------------------------------------------------------------
def build_iou_matrix(
	human_items: list[MaskItem],
	cp_items: list[CPItem],
	show_progress: bool,
) -> np.ndarray:
	n_h = len(human_items)
	n_c = len(cp_items)
	matrix = np.zeros((n_h, n_c), dtype=np.float32)

	cp_by_image: dict[str, list[int]] = defaultdict(list)
	for idx, item in enumerate(cp_items):
		cp_by_image[item.image_number].append(idx)

	iterator = tqdm(
		range(n_h),
		desc="Computing bbox IoU matrix",
		disable=not show_progress,
	)

	for i in iterator:
		h_item = human_items[i]
		if h_item.bbox is None or h_item.area == 0:
			continue
		candidate_indices = cp_by_image.get(h_item.image_number, [])
		if not candidate_indices:
			continue

		for j in candidate_indices:
			c_item = cp_items[j]
			if not bboxes_overlap(h_item.bbox, c_item.bbox):
				continue
			matrix[i, j] = bbox_iou(h_item.bbox, c_item.bbox)

	return matrix


def match_detections(
	iou_matrix: np.ndarray, iou_threshold: float,
) -> tuple[list[dict], set[int], set[int]]:
	n_h, n_c = iou_matrix.shape
	rows_all = set(range(n_h))
	cols_all = set(range(n_c))

	if n_h == 0 or n_c == 0:
		return [], rows_all, cols_all

	row_ind, col_ind = linear_sum_assignment(-iou_matrix)

	matches: list[dict] = []
	matched_rows: set[int] = set()
	matched_cols: set[int] = set()

	for r, c in zip(row_ind.tolist(), col_ind.tolist()):
		iou = float(iou_matrix[r, c])
		if iou >= iou_threshold:
			matches.append({"human_idx": r, "computer_idx": c, "iou": iou})
			matched_rows.add(r)
			matched_cols.add(c)

	unmatched_rows = rows_all - matched_rows
	unmatched_cols = cols_all - matched_cols
	return matches, unmatched_rows, unmatched_cols


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df.columns = [
		str(c).strip().replace("\r", "").replace("\n", "").replace(" ", "")
		for c in df.columns
	]
	return df


def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
	for col in candidates:
		if col in df.columns:
			return col
	raise KeyError(
		f"Could not find {label} column. Tried {candidates}. Available: {list(df.columns)}"
	)


def load_human_measurement_table(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"Measurement CSV not found for human: {csv_path}")

	df = pd.read_csv(csv_path)
	df = normalize_columns(df)

	filename_col = find_column(df, ["filename", "image", "mask", "mask_name"], "human filename")
	length_col = find_column(df, ["length_um", "length", "worm_length_um"], "human length")
	width_col = find_column(df, ["width_um", "width", "worm_width_um"], "human width")

	df = df[[filename_col, length_col, width_col]].copy()
	df.rename(
		columns={
			filename_col: "filename",
			length_col: "length_um_human",
			width_col: "width_um_human",
		},
		inplace=True,
	)

	dup_count = int(df["filename"].duplicated(keep=False).sum())
	if dup_count > 0:
		print(f"Warning: {dup_count} duplicate filename rows in human measurements. Keeping first occurrence.")
		df = df.drop_duplicates(subset=["filename"], keep="first")

	return df


def build_matched_measurements(
	match_df: pd.DataFrame,
	human_measurements_csv: Path,
	cp_items: list[CPItem],
) -> pd.DataFrame:
	human_meas = load_human_measurement_table(human_measurements_csv)

	# Build a lookup from CPItem row_index to length_um.
	cp_length_map = {item.row_index: item.length_um for item in cp_items}

	merged = match_df.merge(
		human_meas,
		left_on="human_file",
		right_on="filename",
		how="left",
	)

	if "filename" in merged.columns:
		merged.drop(columns=["filename"], inplace=True)

	merged["length_um_cellprofiler"] = merged["cp_row_index"].map(cp_length_map)
	merged["length_diff_um"] = merged["length_um_cellprofiler"] - merged["length_um_human"]
	merged["abs_length_diff_um"] = merged["length_diff_um"].abs()

	cols = [
		"human_file",
		"cp_source_filename",
		"cp_row_index",
		"iou",
		"length_um_human",
		"width_um_human",
		"length_um_cellprofiler",
		"length_diff_um",
		"abs_length_diff_um",
	]
	return merged[cols]


# ---------------------------------------------------------------------------
# Path helpers (mirrored from worm_match.py)
# ---------------------------------------------------------------------------
def to_path_list(value: PathOrPaths, label: str) -> list[Path]:
	if isinstance(value, Path):
		return [value]
	if not isinstance(value, list) or not value:
		raise ValueError(f"{label} must be a Path or a non-empty list of Path values.")
	if not all(isinstance(v, Path) for v in value):
		raise TypeError(f"All entries in {label} must be Path objects.")
	return value


def expand_to_length(values: list[Path], target_len: int, label: str) -> list[Path]:
	if len(values) == target_len:
		return values
	if len(values) == 1:
		return values * target_len
	raise ValueError(
		f"{label} has length {len(values)} but expected 1 or {target_len} "
		"to match other path lists."
	)


def resolve_match_runs(config: MatchConfig) -> list[MatchRunConfig]:
	human_dirs = to_path_list(config.human_dir, "human_dir")
	human_csvs = to_path_list(config.human_measurements_csv, "human_measurements_csv")
	output_dirs = to_path_list(config.output_dir, "output_dir")

	num_runs = max(len(human_dirs), len(human_csvs), len(output_dirs))

	human_dirs = expand_to_length(human_dirs, num_runs, "human_dir")
	human_csvs = expand_to_length(human_csvs, num_runs, "human_measurements_csv")
	output_dirs = expand_to_length(output_dirs, num_runs, "output_dir")

	runs: list[MatchRunConfig] = []
	for i in range(num_runs):
		runs.append(
			MatchRunConfig(
				human_dir=human_dirs[i],
				cellprofiler_csv=config.cellprofiler_csv,
				human_measurements_csv=human_csvs[i],
				output_dir=output_dirs[i],
				iou_threshold=config.iou_threshold,
				save_iou_matrix=config.save_iou_matrix,
				show_progress=config.show_progress,
			)
		)
	return runs


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single_match(
	config: MatchRunConfig,
	run_idx: int,
	num_runs: int,
	cp_items: list[CPItem] | None = None,
) -> dict[str, float | int]:
	config.output_dir.mkdir(parents=True, exist_ok=True)

	human_files = list_mask_files(config.human_dir)
	if cp_items is None:
		cp_items = load_cellprofiler_csv(config.cellprofiler_csv)

	if not human_files:
		raise RuntimeError(f"No mask images found in human folder: {config.human_dir}")
	if not cp_items:
		raise RuntimeError(f"No CellProfiler detections loaded from: {config.cellprofiler_csv}")

	# Filter CellProfiler items to only those whose image_number appears in
	# the human mask set for this run (avoids polluting the matrix with
	# irrelevant images from other categories).
	human_image_numbers: set[str] = set()
	for path in human_files:
		human_image_numbers.add(extract_image_number_from_mask(path))

	filtered_cp = [item for item in cp_items if item.image_number in human_image_numbers]

	print(f"\n=== Run {run_idx}/{num_runs} ===")
	print(f"Human masks:          {len(human_files)} from {config.human_dir}")
	print(f"CellProfiler dets:    {len(filtered_cp)} (of {len(cp_items)} total) from {config.cellprofiler_csv}")
	print(f"Shared images:        {len(human_image_numbers & {c.image_number for c in filtered_cp})}")
	print(f"IoU threshold:        {config.iou_threshold:.3f}")

	human_items = load_masks(human_files, show_progress=config.show_progress, desc="Loading human masks")

	iou_matrix = build_iou_matrix(
		human_items,
		filtered_cp,
		show_progress=config.show_progress,
	)

	matches, unmatched_h, unmatched_c = match_detections(iou_matrix, iou_threshold=config.iou_threshold)

	match_rows = []
	for m in matches:
		hi = m["human_idx"]
		ci = m["computer_idx"]
		match_rows.append(
			{
				"human_file": human_items[hi].path.name,
				"cp_source_filename": filtered_cp[ci].source_filename,
				"cp_row_index": filtered_cp[ci].row_index,
				"iou": m["iou"],
			}
		)

	unmatched_h_rows = [
		{"human_file": human_items[idx].path.name}
		for idx in sorted(unmatched_h)
	]

	matched_measurements_csv = config.output_dir / "worm_matched_measurements.csv"
	unmatched_h_csv = config.output_dir / "unmatched_human_masks.csv"
	summary_json = config.output_dir / "matching_summary.json"

	match_df = pd.DataFrame(match_rows)
	if not match_df.empty:
		match_df = match_df.sort_values(by="iou", ascending=False)

	if not match_df.empty:
		matched_meas_df = build_matched_measurements(
			match_df=match_df,
			human_measurements_csv=config.human_measurements_csv,
			cp_items=filtered_cp,
		)
		matched_meas_df.to_csv(matched_measurements_csv, index=False)
	else:
		if matched_measurements_csv.exists():
			matched_measurements_csv.unlink()
		print("No matched pairs above IoU threshold, so no matched measurement CSV was saved.")

	if unmatched_h_rows:
		pd.DataFrame(unmatched_h_rows).to_csv(unmatched_h_csv, index=False)
	elif unmatched_h_csv.exists():
		unmatched_h_csv.unlink()

	summary = {
		"human_dir": str(config.human_dir),
		"cellprofiler_csv": str(config.cellprofiler_csv),
		"iou_threshold": config.iou_threshold,
		"num_human_masks": len(human_items),
		"num_cellprofiler_detections": len(filtered_cp),
		"num_matched": len(matches),
		"num_unmatched_human": len(unmatched_h),
		"num_unmatched_cellprofiler": len(unmatched_c),
		"mean_iou_of_matches": float(np.mean([m["iou"] for m in matches])) if matches else 0.0,
		"max_iou": float(np.max(iou_matrix)) if iou_matrix.size else 0.0,
	}

	summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	if config.save_iou_matrix:
		matrix_csv = config.output_dir / "iou_matrix.csv"
		matrix_df = pd.DataFrame(
			iou_matrix,
			index=[p.path.name for p in human_items],
			columns=[f"{c.source_filename}_row{c.row_index}" for c in filtered_cp],
		)
		matrix_df.to_csv(matrix_csv)

	print("\nMatching complete.")
	print(f"Matched pairs:                {summary['num_matched']}")
	print(f"Unmatched human masks:        {summary['num_unmatched_human']}")
	print(f"Unmatched CellProfiler dets:  {summary['num_unmatched_cellprofiler']}")
	print(f"Mean IoU (matched):           {summary['mean_iou_of_matches']:.4f}")
	if matches:
		print(f"Saved: {matched_measurements_csv}")
	if unmatched_h_rows:
		print(f"Saved: {unmatched_h_csv}")
	print(f"Saved: {summary_json}")

	return {
		"num_matched": int(summary["num_matched"]),
		"num_unmatched_human": int(summary["num_unmatched_human"]),
		"num_unmatched_cellprofiler": int(summary["num_unmatched_cellprofiler"]),
	}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
	runs = resolve_match_runs(CONFIG)
	print(f"Planned runs: {len(runs)}")

	# Load CellProfiler CSV once and reuse across runs.
	cp_items = load_cellprofiler_csv(runs[0].cellprofiler_csv)
	print(f"Loaded {len(cp_items)} CellProfiler detections from {runs[0].cellprofiler_csv}")

	totals = {
		"num_matched": 0,
		"num_unmatched_human": 0,
		"num_unmatched_cellprofiler": 0,
	}

	for idx, run_config in enumerate(runs, start=1):
		run_summary = run_single_match(
			run_config,
			run_idx=idx,
			num_runs=len(runs),
			cp_items=cp_items,
		)
		totals["num_matched"] += int(run_summary["num_matched"])
		totals["num_unmatched_human"] += int(run_summary["num_unmatched_human"])
		totals["num_unmatched_cellprofiler"] += int(run_summary["num_unmatched_cellprofiler"])

	if len(runs) > 1:
		print("\n=== Batch totals ===")
		print(f"Total matched pairs:              {totals['num_matched']}")
		print(f"Total unmatched human masks:      {totals['num_unmatched_human']}")
		print(f"Total unmatched CellProfiler:     {totals['num_unmatched_cellprofiler']}")


if __name__ == "__main__":
	main()
