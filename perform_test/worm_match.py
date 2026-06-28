"""
Match worm masks between human annotations and computer detections.

The script computes IoU for every human/computer mask pair, finds a
one-to-one assignment that maximizes total IoU, and then filters matches by
an IoU threshold.
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
	computer_dir: PathOrPaths
	human_measurements_csv: PathOrPaths
	computer_measurements_csv: PathOrPaths
	output_dir: PathOrPaths
	iou_threshold: float
	resize_mismatched: bool
	save_iou_matrix: bool
	show_progress: bool


@dataclass(frozen=True)
class MatchRunConfig:
	human_dir: Path
	computer_dir: Path
	human_measurements_csv: Path
	computer_measurements_csv: Path
	output_dir: Path
	iou_threshold: float
	resize_mismatched: bool
	save_iou_matrix: bool
	show_progress: bool


# Edit these values directly to configure matching.
# Each path field accepts either:
# - a single Path (one run), or
# - a list[Path] (batch runs, paired by index; single-item lists are broadcast).
#
# To add/remove GT categories, edit HUMAN_GROUPS only.
HUMAN_GROUPS = []

GT_ROIS_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\GT_rois"
)
# Masks live under the original "skeleton" folder; only the worm_lengths.csv
# files (with the Topology_Warnings column) live under "skeleton_warned".
# Set the toggles below to False to read worm_lengths.csv from the original
# "skeleton" folder instead of "skeleton_warned" for either side (this
# disables the warning column for that side, so matched pairs won't be
# filtered by warnings from that side).
USE_HUMAN_WARNED_CSV = True
USE_COMPUTER_WARNED_CSV = True
# Even when a warned CSV is loaded, set these to True to ignore its
# Topology_Warnings column (matched pairs will not be filtered by that side).
IGNORE_HUMAN_WARNINGS = True
IGNORE_COMPUTER_WARNINGS = True
HUMAN_SKELETON_SUBDIR = "skeleton_warned" if USE_HUMAN_WARNED_CSV else "skeleton"

COMPUTER_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\NeSg_perform\skeleton"
)
COMPUTER_WARNED_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\NeSg_perform\skeleton_warned"
)
COMPUTER_CSV_ROOT = COMPUTER_WARNED_ROOT if USE_COMPUTER_WARNED_CSV else COMPUTER_ROOT
OUTPUT_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\GT_vs_NeSg_unwarned"
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
	computer_dir=COMPUTER_ROOT / "target_masks_original",

	human_measurements_csv=[
		GT_ROIS_ROOT / group / "train" / HUMAN_SKELETON_SUBDIR / "worm_lengths.csv"
		for group in RESOLVED_HUMAN_GROUPS
	],

	computer_measurements_csv=COMPUTER_CSV_ROOT / "worm_lengths.csv",

	output_dir=[OUTPUT_ROOT / group for group in RESOLVED_HUMAN_GROUPS],
		
	iou_threshold=0.70,
	resize_mismatched=False,
	save_iou_matrix=False,
	show_progress=True,
)


@dataclass(frozen=True)
class MaskItem:
	path: Path
	area: int
	prefix: str
	bbox: tuple[int, int, int, int] | None  # (x_min, y_min, x_max, y_max)


def extract_mask_prefix(path: Path) -> str:
	"""Return filename prefix used to group candidate masks.

	Example:
		100_png.rf.xxx_roi_17.png -> 100_png.rf.xxx
	"""
	name = path.stem
	if "_roi_" in name:
		return name.split("_roi_", 1)[0]
	return name


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
	computer_dirs = to_path_list(config.computer_dir, "computer_dir")
	human_csvs = to_path_list(config.human_measurements_csv, "human_measurements_csv")
	computer_csvs = to_path_list(config.computer_measurements_csv, "computer_measurements_csv")
	output_dirs = to_path_list(config.output_dir, "output_dir")

	num_runs = max(
		len(human_dirs),
		len(computer_dirs),
		len(human_csvs),
		len(computer_csvs),
		len(output_dirs),
	)

	human_dirs = expand_to_length(human_dirs, num_runs, "human_dir")
	computer_dirs = expand_to_length(computer_dirs, num_runs, "computer_dir")
	human_csvs = expand_to_length(human_csvs, num_runs, "human_measurements_csv")
	computer_csvs = expand_to_length(computer_csvs, num_runs, "computer_measurements_csv")
	output_dirs = expand_to_length(output_dirs, num_runs, "output_dir")

	runs: list[MatchRunConfig] = []
	for i in range(num_runs):
		runs.append(
			MatchRunConfig(
				human_dir=human_dirs[i],
				computer_dir=computer_dirs[i],
				human_measurements_csv=human_csvs[i],
				computer_measurements_csv=computer_csvs[i],
				output_dir=output_dirs[i],
				iou_threshold=config.iou_threshold,
				resize_mismatched=config.resize_mismatched,
				save_iou_matrix=config.save_iou_matrix,
				show_progress=config.show_progress,
			)
		)
	return runs


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
				prefix=extract_mask_prefix(path),
				bbox=mask_bbox(mask),
			)
		)
	return items


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


def compute_iou(h_mask: np.ndarray, human_area: int, computer: MaskItem, resize_mismatched: bool) -> float:
	c_mask = load_binary_mask(computer.path)

	if h_mask.shape != c_mask.shape:
		if not resize_mismatched:
			return 0.0
		c_mask = cv2.resize(
			c_mask.astype(np.uint8),
			(h_mask.shape[1], h_mask.shape[0]),
			interpolation=cv2.INTER_NEAREST,
		).astype(bool)

	if human_area == 0 or int(c_mask.sum()) == 0:
		return 0.0

	inter = int(np.logical_and(h_mask, c_mask).sum())
	if inter == 0:
		return 0.0

	union = int(np.logical_or(h_mask, c_mask).sum())
	if union == 0:
		return 0.0

	return inter / union


def build_iou_matrix(
	human_items: list[MaskItem],
	computer_items: list[MaskItem],
	resize_mismatched: bool,
	show_progress: bool,
) -> np.ndarray:
	n_h = len(human_items)
	n_c = len(computer_items)
	matrix = np.zeros((n_h, n_c), dtype=np.float32)

	computer_by_prefix: dict[str, list[int]] = defaultdict(list)
	for idx, item in enumerate(computer_items):
		computer_by_prefix[item.prefix].append(idx)

	iterator = tqdm(
		range(n_h),
		desc="Computing IoU matrix",
		disable=not show_progress,
	)

	for i in iterator:
		h_item = human_items[i]
		if h_item.area == 0:
			continue
		candidate_indices = computer_by_prefix.get(h_item.prefix, [])
		if not candidate_indices:
			continue

		h_mask = load_binary_mask(h_item.path)

		for j in candidate_indices:
			c_item = computer_items[j]
			if c_item.area == 0:
				continue
			if not bboxes_overlap(h_item.bbox, c_item.bbox):
				continue
			matrix[i, j] = compute_iou(h_mask, h_item.area, c_item, resize_mismatched)

	return matrix


def match_masks(iou_matrix: np.ndarray, iou_threshold: float) -> tuple[list[dict], set[int], set[int]]:
	n_h, n_c = iou_matrix.shape
	rows_all = set(range(n_h))
	cols_all = set(range(n_c))

	if n_h == 0 or n_c == 0:
		return [], rows_all, cols_all

	# Minimize negative IoU to maximize IoU globally under one-to-one matching.
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


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize column names to improve robustness to formatting differences."""
	df = df.copy()
	df.columns = [
		str(c).strip().replace("\r", "").replace("\n", "").replace(" ", "").lower()
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


def load_measurement_table(
	csv_path: Path,
	side: str,
	ignore_warnings: bool = False,
) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"Measurement CSV not found for {side}: {csv_path}")

	df = pd.read_csv(csv_path)
	df = normalize_columns(df)

	filename_col = find_column(df, ["filename", "image", "mask", "mask_name"], f"{side} filename")
	length_col = find_column(df, ["length_um", "length", "worm_length_um"], f"{side} length")
	width_col = find_column(df, ["width_um", "width", "worm_width_um"], f"{side} width")

	# Optional column: present only in CSVs produced by the updated
	# skeletonize_worms.py.  When missing, treat every row as warning-free.
	# When ignore_warnings=True, deliberately skip detection so this side never
	# contributes to filtering, even if the column exists in the loaded CSV.
	warning_col: str | None = None
	if not ignore_warnings:
		for candidate in ("topology_warnings", "topologywarnings", "skeleton_warning", "skeletonwarnings"):
			if candidate in df.columns:
				warning_col = candidate
				break

	keep_cols = [filename_col, length_col, width_col]
	if warning_col is not None:
		keep_cols.append(warning_col)

	df = df[keep_cols].copy()
	rename_map = {
		filename_col: "filename",
		length_col: f"length_um_{side}",
		width_col: f"width_um_{side}",
	}
	if warning_col is not None:
		rename_map[warning_col] = f"topology_warnings_{side}"
	else:
		df[f"topology_warnings_{side}"] = ""
	df.rename(columns=rename_map, inplace=True)

	# Normalize warning column: NaN → "", strip whitespace.
	warn_out = f"topology_warnings_{side}"
	df[warn_out] = df[warn_out].fillna("").astype(str).str.strip()

	dup_count = int(df["filename"].duplicated(keep=False).sum())
	if dup_count > 0:
		print(f"Warning: {dup_count} duplicate filename rows in {side} measurements. Keeping first occurrence.")
		df = df.drop_duplicates(subset=["filename"], keep="first")

	return df


def build_matched_measurements(
	match_df: pd.DataFrame,
	human_measurements_csv: Path,
	computer_measurements_csv: Path,
	ignore_human_warnings: bool = False,
	ignore_computer_warnings: bool = False,
) -> pd.DataFrame:
	human_meas = load_measurement_table(
		human_measurements_csv, side="human", ignore_warnings=ignore_human_warnings
	)
	computer_meas = load_measurement_table(
		computer_measurements_csv, side="computer", ignore_warnings=ignore_computer_warnings
	)

	merged = match_df.merge(
		human_meas,
		left_on="human_file",
		right_on="filename",
		how="left",
	)
	merged = merged.merge(
		computer_meas,
		left_on="computer_file",
		right_on="filename",
		how="left",
		suffixes=("", "_computer_meas"),
	)

	# Remove helper join columns from both merges.
	for col in ["filename", "filename_computer_meas"]:
		if col in merged.columns:
			merged.drop(columns=[col], inplace=True)

	merged["length_diff_um"] = merged["length_um_computer"] - merged["length_um_human"]
	merged["width_diff_um"] = merged["width_um_computer"] - merged["width_um_human"]
	merged["abs_length_diff_um"] = merged["length_diff_um"].abs()
	merged["abs_width_diff_um"] = merged["width_diff_um"].abs()

	# Filter out matched pairs flagged by topology warnings on either side.
	for col in ("topology_warnings_human", "topology_warnings_computer"):
		if col not in merged.columns:
			merged[col] = ""
		merged[col] = merged[col].fillna("").astype(str).str.strip()

	warn_mask = (
		(merged["topology_warnings_human"] != "")
		| (merged["topology_warnings_computer"] != "")
	)
	n_dropped = int(warn_mask.sum())
	if n_dropped > 0:
		print(
			f"Filtered out {n_dropped} matched pair(s) with topology warnings "
			"on the human and/or computer side."
		)
	merged = merged.loc[~warn_mask].copy()

	cols = [
		"human_file",
		"computer_file",
		"iou",
		"length_um_human",
		"width_um_human",
		"length_um_computer",
		"width_um_computer",
		"length_diff_um",
		"width_diff_um",
		"abs_length_diff_um",
		"abs_width_diff_um",
	]
	return merged[cols]


def run_single_match(
	config: MatchRunConfig,
	run_idx: int,
	num_runs: int,
	computer_files: list[Path] | None = None,
	computer_items: list[MaskItem] | None = None,
) -> dict[str, float | int]:
	config.output_dir.mkdir(parents=True, exist_ok=True)

	human_files = list_mask_files(config.human_dir)
	if computer_files is None:
		computer_files = list_mask_files(config.computer_dir)
	if computer_items is None:
		computer_items = load_masks(
			computer_files,
			show_progress=config.show_progress,
			desc="Loading computer masks",
		)

	if not human_files:
		raise RuntimeError(f"No mask images found in human folder: {config.human_dir}")
	if not computer_files:
		raise RuntimeError(f"No mask images found in computer folder: {config.computer_dir}")
	if not computer_items:
		raise RuntimeError(f"No computer mask metadata loaded for folder: {config.computer_dir}")

	print(f"\n=== Run {run_idx}/{num_runs} ===")
	print(f"Human masks:    {len(human_files)} from {config.human_dir}")
	print(f"Computer masks: {len(computer_files)} from {config.computer_dir}")
	print(f"IoU threshold:  {config.iou_threshold:.3f}")

	human_items = load_masks(human_files, show_progress=config.show_progress, desc="Loading human masks")

	iou_matrix = build_iou_matrix(
		human_items,
		computer_items,
		resize_mismatched=config.resize_mismatched,
		show_progress=config.show_progress,
	)

	matches, unmatched_h, unmatched_c = match_masks(iou_matrix, iou_threshold=config.iou_threshold)

	match_rows = []
	for m in matches:
		hi = m["human_idx"]
		ci = m["computer_idx"]
		match_rows.append(
			{
				"human_file": human_items[hi].path.name,
				"computer_file": computer_items[ci].path.name,
				"iou": m["iou"],
			}
		)

	unmatched_h_rows = [
		{
			"human_file": human_items[idx].path.name,
		}
		for idx in sorted(unmatched_h)
	]

	matches_csv = config.output_dir / "worm_matches.csv"
	matched_measurements_csv = config.output_dir / "worm_matched_measurements.csv"
	unmatched_h_csv = config.output_dir / "unmatched_human_masks.csv"
	summary_json = config.output_dir / "matching_summary.json"

	match_df = pd.DataFrame(match_rows).sort_values(by="iou", ascending=False)
	if matches_csv.exists():
		matches_csv.unlink()

	if not match_df.empty:
		matched_meas_df = build_matched_measurements(
			match_df=match_df,
			human_measurements_csv=config.human_measurements_csv,
			computer_measurements_csv=config.computer_measurements_csv,
			ignore_human_warnings=IGNORE_HUMAN_WARNINGS,
			ignore_computer_warnings=IGNORE_COMPUTER_WARNINGS,
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
		"computer_dir": str(config.computer_dir),
		"iou_threshold": config.iou_threshold,
		"resize_mismatched": bool(config.resize_mismatched),
		"num_human_masks": len(human_items),
		"num_computer_masks": len(computer_items),
		"num_matched": len(matches),
		"num_unmatched_human": len(unmatched_h),
		"num_unmatched_computer": len(unmatched_c),
		"mean_iou_of_matches": float(np.mean([m["iou"] for m in matches])) if matches else 0.0,
		"max_iou": float(np.max(iou_matrix)) if iou_matrix.size else 0.0,
	}

	summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

	if config.save_iou_matrix:
		matrix_csv = config.output_dir / "iou_matrix.csv"
		matrix_df = pd.DataFrame(
			iou_matrix,
			index=[p.path.name for p in human_items],
			columns=[p.path.name for p in computer_items],
		)
		matrix_df.to_csv(matrix_csv)

	print("\nMatching complete.")
	print(f"Matched pairs:          {summary['num_matched']}")
	print(f"Unmatched human masks:  {summary['num_unmatched_human']}")
	print(f"Unmatched computer:     {summary['num_unmatched_computer']}")
	print(f"Mean IoU (matched):     {summary['mean_iou_of_matches']:.4f}")
	print("\nSaved matched info in measurement table only (worm_matches.csv not written).")
	if matches:
		print(f"Saved: {matched_measurements_csv}")
	if unmatched_h_rows:
		print(f"Saved: {unmatched_h_csv}")
	print(f"Saved: {summary_json}")

	return {
		"num_matched": int(summary["num_matched"]),
		"num_unmatched_human": int(summary["num_unmatched_human"]),
		"num_unmatched_computer": int(summary["num_unmatched_computer"]),
	}


def main() -> None:
	runs = resolve_match_runs(CONFIG)
	print(f"Planned runs: {len(runs)}")
	computer_cache: dict[Path, tuple[list[Path], list[MaskItem]]] = {}

	totals = {
		"num_matched": 0,
		"num_unmatched_human": 0,
		"num_unmatched_computer": 0,
	}

	for idx, run_config in enumerate(runs, start=1):
		cache_key = run_config.computer_dir.resolve()
		if cache_key not in computer_cache:
			print(f"Loading shared computer masks once: {run_config.computer_dir}")
			cached_files = list_mask_files(run_config.computer_dir)
			if not cached_files:
				raise RuntimeError(f"No mask images found in computer folder: {run_config.computer_dir}")
			cached_items = load_masks(
				cached_files,
				show_progress=run_config.show_progress,
				desc="Loading computer masks (shared cache)",
			)
			computer_cache[cache_key] = (cached_files, cached_items)
		else:
			print(f"Reusing cached computer masks: {run_config.computer_dir}")

		cached_files, cached_items = computer_cache[cache_key]
		run_summary = run_single_match(
			run_config,
			run_idx=idx,
			num_runs=len(runs),
			computer_files=cached_files,
			computer_items=cached_items,
		)
		totals["num_matched"] += int(run_summary["num_matched"])
		totals["num_unmatched_human"] += int(run_summary["num_unmatched_human"])
		totals["num_unmatched_computer"] += int(run_summary["num_unmatched_computer"])

	if len(runs) > 1:
		print("\n=== Batch totals ===")
		print(f"Total matched pairs:         {totals['num_matched']}")
		print(f"Total unmatched human masks: {totals['num_unmatched_human']}")
		print(f"Total unmatched computer:    {totals['num_unmatched_computer']}")


if __name__ == "__main__":
	main()
