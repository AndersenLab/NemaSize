"""Generate a combined summary of human vs computer measurement differences.

This script scans category folders under GT_VS_COMP_ROOT, reads each
`worm_matched_measurements.csv`, and creates summary bar charts using
difference columns (for example `length_diff_um`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


# Doubled-font rc params used for box plot figures (publication figures are
# left with default font sizes).
_BASE_FONT_SIZE = float(plt.rcParams["font.size"])
_FONT_SCALE = 2.0
LARGE_FONT_RC = {
	"font.size": _BASE_FONT_SIZE * _FONT_SCALE,
	"axes.titlesize": _BASE_FONT_SIZE * _FONT_SCALE * 1.2,
	"axes.labelsize": _BASE_FONT_SIZE * _FONT_SCALE,
	"xtick.labelsize": _BASE_FONT_SIZE * _FONT_SCALE,
	"ytick.labelsize": _BASE_FONT_SIZE * _FONT_SCALE,
	"legend.fontsize": _BASE_FONT_SIZE * _FONT_SCALE,
	"figure.titlesize": _BASE_FONT_SIZE * _FONT_SCALE * 1.2,
}


# Set this to your Windows username to point all paths at your Dropbox folder.
DROPBOX_USER: str = "lizih"

_DROPBOX_BASE = Path(f"C:/Users/{DROPBOX_USER}/Dropbox/JHU_2026_spring/NemaSeg/datasets")

GT_VS_NESG_ROOT: Path | None = _DROPBOX_BASE / "Perform_test/GT_vs_NeSg"
GT_VS_CP_ROOT: Path | None = _DROPBOX_BASE / "Perform_test/GT_vs_CellProf_missingvalid_added"
COMPARISON_OUTPUT_ROOT = _DROPBOX_BASE / "Perform_test"
GT_ROIS_ROOT = _DROPBOX_BASE / "Perform_test/GT_rois"
NESG_PROCESSED_SKELETON_ROOT = _DROPBOX_BASE / "Perform_test/NeSg_perform/skeleton"

# --- Validation-only mode ----------------------------------------------------
# When True, every CSV is filtered down to rows whose source image stem
# appears in VALIDATION_IMAGES_DIR. All output figures/tables are written to a
# sibling "validation_only" folder so they don't overwrite the full-dataset
# results. Discoverability is intentionally skipped in this mode.
VALIDATION_ONLY: bool = True
VALIDATION_IMAGES_DIR: Path = (
	_DROPBOX_BASE / "WormBodyDetection.v10i.coco-segmentation/valid/images"
)
VALIDATION_SUBDIR_NAME: str = "validation_only"

CSV_NAME = "worm_matched_measurements.csv"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Merge low-sample classes into stronger destination classes.
CLASS_MERGE_MAP: dict[str, str] = {
	"overlapping_curly_tiny_worm": "curly_tiny_worm",
	"overlapping_tiny_worm": "tiny_worm",
	"self-overlapping_tiny_worm": "tiny_worm",
}

# --- Publication-figure axes ---------------------------------------------------
# Multi-axis (orthogonal) re-grouping of the original source classes for paper
# figures. Each axis maps every (relevant) source class to exactly ONE parent
# group on that axis, so a worm never appears twice in the same axis figure
# but can appear in several axis figures (e.g. tiny_bent_worm -> "bent" on
# curvature axis AND "tiny" on size axis).
PUBLICATION_AXES: dict[str, dict[str, list[str]]] = {
	"curvature": {
		# self-overlapping_* are intentionally excluded from this axis
		# because their underlying centerline curvature is not labeled.
		"straight": ["straight_worm"],
		"bent": ["bent_worm", "tiny_bent_worm"],
		"curly": [
			"curly_worm",
			"curly_tiny_worm",
			"incomplete_curly_worm",
			"overlapping_curly_worm",
			"overlapping_curly_tiny_worm",
		],
	},
	"size": {
		"normal": [
			"straight_worm",
			"bent_worm",
			"curly_worm",
			"incomplete_worm",
			"incomplete_curly_worm",
			"overlapping_worm",
			"overlapping_curly_worm",
			"overlapping_incomplete_worm",
			"self-overlapping_worm",
		],
		"tiny": [
			"tiny_worm",
			"tiny_bent_worm",
			"curly_tiny_worm",
			"incomplete_tiny_worm",
			"overlapping_tiny_worm",
			"overlapping_curly_tiny_worm",
			"self-overlapping_tiny_worm",
		],
	},
	"completeness": {
		"complete": [
			"straight_worm",
			"bent_worm",
			"tiny_bent_worm",
			"curly_worm",
			"curly_tiny_worm",
			"tiny_worm",
			"overlapping_worm",
			"overlapping_curly_worm",
			"overlapping_tiny_worm",
			"overlapping_curly_tiny_worm",
			"self-overlapping_worm",
			"self-overlapping_tiny_worm",
		],
		"incomplete": [
			"incomplete_worm",
			"incomplete_curly_worm",
			"incomplete_tiny_worm",
			"overlapping_incomplete_worm",
		],
	},
	"occlusion": {
		"isolated": [
			"straight_worm",
			"bent_worm",
			"tiny_bent_worm",
			"curly_worm",
			"curly_tiny_worm",
			"tiny_worm",
			"incomplete_worm",
			"incomplete_curly_worm",
			"incomplete_tiny_worm",
		],
		"self-overlapping": [
			"self-overlapping_worm",
			"self-overlapping_tiny_worm",
		],
		"overlapping": [
			"overlapping_worm",
			"overlapping_curly_worm",
			"overlapping_tiny_worm",
			"overlapping_incomplete_worm",
			"overlapping_curly_tiny_worm",
		],
	},
}

# Display order for parent groups within each axis figure.
PUBLICATION_AXIS_ORDER: dict[str, list[str]] = {
	"curvature": ["straight", "bent", "curly"],
	"size": ["normal", "tiny"],
	"completeness": ["complete", "incomplete"],
	"occlusion": ["isolated", "self-overlapping", "overlapping"],
}

# Baseline ("easy") group per axis to exclude from the publication figure so
# the high-n baseline does not visually dominate the harder cases.
PUBLICATION_AXIS_BASELINE: dict[str, str] = {
	"curvature": "straight",
	"size": "normal",
	"completeness": "complete",
	"occlusion": "isolated",
}

# Shape-encoded buckets for the pooled "all_categories" publication figure.
# Each source class is assigned to exactly one bucket using priority:
#   occlusion-hard > curvature-hard > baseline
# (so e.g. overlapping_curly_worm -> triangle, not square).
PUBLICATION_POOLED_BUCKETS: list[tuple[str, str, list[str]]] = [
	# (bucket_label, marker, source classes)
	(
		"Overlapping",
		"^",
		[
			"overlapping_worm",
			"overlapping_curly_worm",
			"overlapping_tiny_worm",
			"overlapping_incomplete_worm",
			"overlapping_curly_tiny_worm",
			"self-overlapping_worm",
			"self-overlapping_tiny_worm",
			"incomplete_worm",
			"incomplete_curly_worm",
			"incomplete_tiny_worm",
		],
	),
	(
		"Bent or curly",
		"s",
		[
			"bent_worm",
			"tiny_bent_worm",
			"curly_worm",
			"curly_tiny_worm",
		],
	),
	(
		"Straight and isolated",
		"o",
		[
			"straight_worm",
			"tiny_worm",
		],
	),
]


def _build_pooled_bucket_lookup() -> dict[str, tuple[str, str]]:
	"""Invert PUBLICATION_POOLED_BUCKETS into {source_class: (label, marker)}.

	The list order encodes priority (first match wins), so a source listed in
	multiple buckets resolves to the earlier (higher-priority) bucket.
	"""
	lookup: dict[str, tuple[str, str]] = {}
	for label, marker, sources in PUBLICATION_POOLED_BUCKETS:
		for src in sources:
			lookup.setdefault(src, (label, marker))
	return lookup


PUBLICATION_OUTPUT_DIRNAME = "publication_figures"

# Number of highest-|percentage-error| images to print/copy from the pooled
# "all_categories" group when generating difference_boxplot.png. Affects both
# the console printout, the stats file block, and the number of skeleton
# images copied to top10_abs_pct_error_images/.
TOP_N_HIGHEST_PCT_ERROR_IMAGES: int = 20

# Font sizes for the three final publication figures.
PUB_AXIS_FONTSIZE:   float = 12   # axis labels and tick labels
PUB_LEGEND_FONTSIZE: float = 8   # legend text           

# Figure sizes for the three final publication figures (inches).
# difference_boxplot and width_difference_boxplot heights are dynamic
# (scale with category count); only the width is fixed here.
SCATTER_FIG_WIDTH:  float = 3.7          # all_categories_scatter.png width
SCATTER_FIG_HEIGHT: float = 3.7          # all_categories_scatter.png height
BOXPLOT_FIG_WIDTH:  float = 3.7          # difference_boxplot / width_difference_boxplot width
BOXPLOT_FIG_HEIGHT: float = 3.7  		 # None = auto-scale with category count

# Marker sizes for the publication figures (matplotlib `s` area units).
SCATTER_MARKER_SIZE_CP: float = 22.0*0.7    # CellProfiler hollow markers in all_categories_scatter
SCATTER_MARKER_SIZE_NS: float = 29.0    # NemaSize solid markers in all_categories_scatter
BOXPLOT_MARKER_SIZE:    float = 12.0*0.7    # jitter-strip dots in difference_boxplot / width_difference_boxplot

# Legend position for all_categories_scatter (matplotlib loc + bbox_to_anchor).
SCATTER_LEGEND_LOC:   str             = "upper left"   # e.g. "upper left", "upper right"
SCATTER_LEGEND_BBOX:  tuple[float, float] = (-0.01, 1.05)  # (x, y) in axes-fraction coordinates


def get_category_folders(root: Path) -> list[Path]:
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Root folder not found: {root}")

	return sorted(path for path in root.iterdir() if path.is_dir())


def sem_for_plot(values: pd.Series) -> float:
	if values.shape[0] <= 1:
		return 0.0
	return float(values.std(ddof=1) / np.sqrt(values.shape[0]))


def percentage_diff(delta: pd.Series, ground_truth: pd.Series) -> pd.Series:
	valid = ground_truth != 0
	if not valid.any():
		return pd.Series(dtype=float)
	pct = (delta[valid] / ground_truth[valid]) * 100.0
	pct = pd.to_numeric(pct, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
	return pct


def annotate_sample_sizes(ax: plt.Axes, x_values: np.ndarray, sample_sizes: list[int]) -> None:
	for xi, n in zip(x_values, sample_sizes):
		ax.text(
			xi,
			0.98,
			f"n={n}",
			transform=ax.get_xaxis_transform(),
			ha="center",
			va="top",
			fontsize=8,
			rotation=90,
			color="#333333",
		)


def find_censor_file_for_category(category_name: str, category_dir: Path) -> Path | None:
	"""Find optional censor list for a category.

	Supported locations:
	- <GT_rois>/<category>/train/skeleton/censor.txt
	- <GT_VS_COMP_ROOT>/<category>/censor.txt
	"""
	candidates = [
		GT_ROIS_ROOT / category_name / "train" / "skeleton" / "censor.txt",
		category_dir / "censor.txt",
	]
	for path in candidates:
		if path.is_file():
			return path
	return None


def strip_image_extension(name: str) -> str:
	name = name.strip()
	lower_name = name.lower()
	for ext in IMAGE_EXTENSIONS:
		if lower_name.endswith(ext):
			return name[: -len(ext)]
	return name


def load_censor_entries(censor_file: Path) -> set[str]:
	"""Load censored human mask prefixes from censor.txt.

	Assumes each line in censor.txt is a mask prefix (typically filename without
	image extension).
	"""
	prefixes: set[str] = set()
	for raw in censor_file.read_text(encoding="utf-8").splitlines():
		line = raw.strip()
		if not line or line.startswith("#"):
			continue
		name = Path(line).name
		if not name:
			continue
		prefixes.add(strip_image_extension(name))
	return prefixes


def get_merged_category_name(category_name: str) -> str:
	return CLASS_MERGE_MAP.get(category_name, category_name)


_ROI_SUFFIX_RE = re.compile(r"_roi_\d+$")
_VALIDATION_STEMS_CACHE: set[str] | None = None


def extract_image_stem_from_roi(human_file: str) -> str:
	"""Recover the source image stem from a `<imagestem>_roi_<N>.<ext>` filename."""
	stem = strip_image_extension(Path(str(human_file)).name)
	return _ROI_SUFFIX_RE.sub("", stem)


def get_validation_image_stems() -> set[str]:
	"""Return cached set of image stems in VALIDATION_IMAGES_DIR."""
	global _VALIDATION_STEMS_CACHE
	if _VALIDATION_STEMS_CACHE is not None:
		return _VALIDATION_STEMS_CACHE

	if not VALIDATION_IMAGES_DIR.is_dir():
		raise FileNotFoundError(
			f"Validation images directory not found: {VALIDATION_IMAGES_DIR}"
		)
	stems = {
		p.stem
		for p in VALIDATION_IMAGES_DIR.iterdir()
		if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
	}
	if not stems:
		print(f"[warn] validation: no image files found in {VALIDATION_IMAGES_DIR}")
	else:
		print(f"[validation] Loaded {len(stems)} validation image stems from {VALIDATION_IMAGES_DIR}")
	_VALIDATION_STEMS_CACHE = stems
	return stems


def apply_validation_filter(df: pd.DataFrame, category_name: str) -> pd.DataFrame:
	"""Keep only rows whose human_file's source image stem is in the validation set."""
	if "human_file" not in df.columns or df.empty:
		return df
	val_stems = get_validation_image_stems()
	if not val_stems:
		return df.iloc[0:0].copy()

	human_names = df["human_file"].astype(str)
	image_stems = human_names.map(extract_image_stem_from_roi)
	keep_mask = image_stems.isin(val_stems)
	kept = int(keep_mask.sum())
	total = int(len(df))
	print(f"[validation] {category_name}: kept {kept}/{total} rows")
	return df[keep_mask].copy()


def get_merged_category_order(category_dirs: list[Path], merged_data: dict[str, dict[str, list[float]]]) -> list[str]:
	"""Return merged category names ordered by original destination class position."""
	original_order = {category_dir.name: idx for idx, category_dir in enumerate(category_dirs)}

	def sort_key(category_name: str) -> tuple[int, int, str]:
		# Keep destination class position unchanged when sources are merged into it.
		if category_name in original_order:
			return (0, original_order[category_name], category_name)
		return (1, 10**9, category_name)

	return sorted(merged_data.keys(), key=sort_key)


def apply_censor_filter(df: pd.DataFrame, category_name: str, category_dir: Path) -> pd.DataFrame:
	"""Drop rows whose human_file is listed in censor.txt (if available).

	When VALIDATION_ONLY is True, also drop rows whose source image stem is
	not present in the validation images directory.
	"""
	if "human_file" not in df.columns:
		print(f"[info] {category_name}: no human_file column, censor filtering skipped")
		return df

	censor_file = find_censor_file_for_category(category_name, category_dir)
	if censor_file is None:
		print(f"[info] {category_name}: no censor.txt found")
	else:
		censor_prefixes = load_censor_entries(censor_file)
		if censor_prefixes:
			human_names = df["human_file"].astype(str)
			human_prefixes = human_names.map(strip_image_extension)
			keep_mask = ~human_prefixes.isin(censor_prefixes)

			removed = int((~keep_mask).sum())
			print(f"[info] {category_name}: using censor file {censor_file}")
			if removed > 0:
				print(f"[info] {category_name}: censored {removed} rows using {censor_file}")
				removed_names = sorted(human_names[~keep_mask].dropna().unique().tolist())
				for name in removed_names:
					print(f"[info] {category_name}: filtered human_file -> {name}")
			else:
				print(f"[info] {category_name}: censor.txt found but no rows matched entries")
			df = df[keep_mask].copy()

	if VALIDATION_ONLY:
		df = apply_validation_filter(df, category_name=category_name)

	return df


def collect_merged_data(
	category_dirs: list[Path],
) -> dict[str, dict[str, list[float]]]:
	merged_data: dict[str, dict[str, list[float]]] = {}

	for category_dir in category_dirs:
		csv_path = category_dir / CSV_NAME
		if not csv_path.exists():
			continue

		source_category = category_dir.name
		merged_category = get_merged_category_name(source_category)
		if source_category != merged_category:
			print(f"[info] Merging class '{source_category}' -> '{merged_category}'")

		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=source_category, category_dir=category_dir)
		if "length_diff_um" not in df.columns:
			print(f"[skip] {source_category}: missing column 'length_diff_um'")
			continue

		bucket = merged_data.setdefault(
			merged_category,
			{"length_diff": [], "width_diff": [], "length_pct": [], "width_pct": []},
		)

		length_diff_vals = pd.to_numeric(df["length_diff_um"], errors="coerce").dropna().to_list()
		if not length_diff_vals:
			print(f"[skip] {source_category}: no valid numeric values in 'length_diff_um'")
			continue
		bucket["length_diff"].extend(length_diff_vals)

		if "width_diff_um" in df.columns:
			width_diff_vals = pd.to_numeric(df["width_diff_um"], errors="coerce").dropna().to_list()
			bucket["width_diff"].extend(width_diff_vals)

		if "length_um_human" in df.columns:
			length_human = pd.to_numeric(df["length_um_human"], errors="coerce")
			length_pct = percentage_diff(pd.to_numeric(df["length_diff_um"], errors="coerce"), length_human)
			if not length_pct.empty:
				bucket["length_pct"].extend(length_pct.to_list())

		if {"width_diff_um", "width_um_human"}.issubset(df.columns):
			width_human = pd.to_numeric(df["width_um_human"], errors="coerce")
			width_pct = percentage_diff(pd.to_numeric(df["width_diff_um"], errors="coerce"), width_human)
			if not width_pct.empty:
				bucket["width_pct"].extend(width_pct.to_list())

	return merged_data


def create_combined_summary_plot(category_dirs: list[Path], output_root: Path) -> Path | None:
	merged_data: dict[str, dict[str, list[float]]] = {}

	for category_dir in category_dirs:
		csv_path = category_dir / CSV_NAME
		if not csv_path.exists():
			continue

		source_category = category_dir.name
		merged_category = get_merged_category_name(source_category)
		if source_category != merged_category:
			print(f"[info] Merging class '{source_category}' -> '{merged_category}'")

		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=source_category, category_dir=category_dir)
		if "length_diff_um" not in df.columns:
			print(f"[skip] {source_category}: missing column 'length_diff_um'")
			continue

		bucket = merged_data.setdefault(
			merged_category,
			{
				"length_diff": [],
				"width_diff": [],
			},
		)

		length_diff_vals = pd.to_numeric(df["length_diff_um"], errors="coerce").dropna().to_list()
		if not length_diff_vals:
			print(f"[skip] {source_category}: no valid numeric values in 'length_diff_um'")
			continue
		bucket["length_diff"].extend(length_diff_vals)

		if "width_diff_um" in df.columns:
			width_diff_vals = pd.to_numeric(df["width_diff_um"], errors="coerce").dropna().to_list()
			bucket["width_diff"].extend(width_diff_vals)

	categories: list[str] = []
	sample_sizes: list[int] = []
	mean_length_diff: list[float] = []
	sem_length_diff: list[float] = []
	mean_width_diff: list[float] = []
	sem_width_diff: list[float] = []
	all_length_diff_values: list[float] = []
	all_width_diff_values: list[float] = []
	has_width_data = False

	for category_name in get_merged_category_order(category_dirs, merged_data):
		bucket = merged_data[category_name]
		length_diff = pd.Series(bucket["length_diff"], dtype=float)
		if length_diff.empty:
			continue

		categories.append(category_name)
		sample_sizes.append(int(length_diff.shape[0]))
		mean_length_diff.append(float(length_diff.mean()))
		sem_length_diff.append(sem_for_plot(length_diff))
		all_length_diff_values.extend(length_diff.to_list())

		width_diff = pd.Series(bucket["width_diff"], dtype=float)
		if not width_diff.empty:
			has_width_data = True
			mean_width_diff.append(float(width_diff.mean()))
			sem_width_diff.append(sem_for_plot(width_diff))
			all_width_diff_values.extend(width_diff.to_list())
		else:
			mean_width_diff.append(np.nan)
			sem_width_diff.append(np.nan)

	if not categories:
		print("[skip] Combined summary: no valid category data found")
		return None

	# Add one pooled bar that aggregates all valid rows from every category.
	categories.append("all_categories")
	all_length = pd.Series(all_length_diff_values, dtype=float)
	sample_sizes.append(int(all_length.shape[0]))
	mean_length_diff.append(float(all_length.mean()))
	sem_length_diff.append(sem_for_plot(all_length))

	if has_width_data and all_width_diff_values:
		all_width = pd.Series(all_width_diff_values, dtype=float)
		mean_width_diff.append(float(all_width.mean()))
		sem_width_diff.append(sem_for_plot(all_width))
	else:
		mean_width_diff.append(np.nan)
		sem_width_diff.append(np.nan)

	if has_width_data:
		fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
	else:
		fig, axes = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
		axes = [axes]

	x = np.arange(len(categories))

	axes[0].bar(x, mean_length_diff, yerr=sem_length_diff, capsize=4, color="#4C78A8")
	axes[0].axhline(0.0, color="black", linewidth=1)
	axes[0].set_title("Mean Length Difference (Computer - Human, +/- 1 SEM)")
	axes[0].set_ylabel("Difference (um)")
	axes[0].set_xticks(x)
	axes[0].set_xticklabels(categories, rotation=25, ha="right")
	axes[0].grid(True, axis="y", alpha=0.3)
	annotate_sample_sizes(axes[0], x, sample_sizes)

	if has_width_data:
		axes[1].bar(x, mean_width_diff, yerr=sem_width_diff, capsize=4, color="#54A24B")
		axes[1].axhline(0.0, color="black", linewidth=1)
		axes[1].set_title("Mean Width Difference (Computer - Human, +/- 1 SEM)")
		axes[1].set_ylabel("Difference (um)")
		axes[1].set_xticks(x)
		axes[1].set_xticklabels(categories, rotation=25, ha="right")
		axes[1].grid(True, axis="y", alpha=0.3)
		annotate_sample_sizes(axes[1], x, sample_sizes)

	fig.suptitle("Combined Summary of Human vs Computer Measurement Differences")

	output_path = output_root / "combined_difference_summary.png"
	fig.savefig(output_path, dpi=220)
	plt.close(fig)
	print(f"[ok] Saved: {output_path}")
	return output_path


def create_percentage_summary_plot(category_dirs: list[Path], output_root: Path) -> Path | None:
	merged_data: dict[str, dict[str, list[float]]] = {}

	for category_dir in category_dirs:
		csv_path = category_dir / CSV_NAME
		if not csv_path.exists():
			continue

		source_category = category_dir.name
		merged_category = get_merged_category_name(source_category)
		if source_category != merged_category:
			print(f"[info] Merging class '{source_category}' -> '{merged_category}'")

		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=source_category, category_dir=category_dir)

		required_len_cols = {"length_diff_um", "length_um_human"}
		if not required_len_cols.issubset(df.columns):
			print(
				f"[skip] {source_category}: missing columns for length % diff: "
				f"{sorted(required_len_cols - set(df.columns))}"
			)
			continue

		bucket = merged_data.setdefault(
			merged_category,
			{
				"length_pct": [],
				"width_pct": [],
			},
		)

		length_diff = pd.to_numeric(df["length_diff_um"], errors="coerce")
		length_human = pd.to_numeric(df["length_um_human"], errors="coerce")
		length_pct = percentage_diff(length_diff, length_human)
		if length_pct.empty:
			print(f"[skip] {source_category}: no valid rows for length % diff")
			continue
		bucket["length_pct"].extend(length_pct.to_list())

		required_w_cols = {"width_diff_um", "width_um_human"}
		if required_w_cols.issubset(df.columns):
			width_diff = pd.to_numeric(df["width_diff_um"], errors="coerce")
			width_human = pd.to_numeric(df["width_um_human"], errors="coerce")
			width_pct = percentage_diff(width_diff, width_human)
			if not width_pct.empty:
				bucket["width_pct"].extend(width_pct.to_list())

	categories: list[str] = []
	sample_sizes: list[int] = []
	mean_length_pct: list[float] = []
	sem_length_pct: list[float] = []
	mean_width_pct: list[float] = []
	sem_width_pct: list[float] = []
	all_length_pct_values: list[float] = []
	all_width_pct_values: list[float] = []
	has_width_data = False

	for category_name in get_merged_category_order(category_dirs, merged_data):
		bucket = merged_data[category_name]
		length_pct = pd.Series(bucket["length_pct"], dtype=float)
		if length_pct.empty:
			continue

		categories.append(category_name)
		sample_sizes.append(int(length_pct.shape[0]))
		mean_length_pct.append(float(length_pct.mean()))
		sem_length_pct.append(sem_for_plot(length_pct))
		all_length_pct_values.extend(length_pct.to_list())

		width_pct = pd.Series(bucket["width_pct"], dtype=float)
		if not width_pct.empty:
			has_width_data = True
			mean_width_pct.append(float(width_pct.mean()))
			sem_width_pct.append(sem_for_plot(width_pct))
			all_width_pct_values.extend(width_pct.to_list())
		else:
			mean_width_pct.append(np.nan)
			sem_width_pct.append(np.nan)

	if not categories:
		print("[skip] Percentage summary: no valid category data found")
		return None

	categories.append("all_categories")
	all_length_pct = pd.Series(all_length_pct_values, dtype=float)
	sample_sizes.append(int(all_length_pct.shape[0]))
	mean_length_pct.append(float(all_length_pct.mean()))
	sem_length_pct.append(sem_for_plot(all_length_pct))

	if has_width_data and all_width_pct_values:
		all_width_pct = pd.Series(all_width_pct_values, dtype=float)
		mean_width_pct.append(float(all_width_pct.mean()))
		sem_width_pct.append(sem_for_plot(all_width_pct))
	else:
		mean_width_pct.append(np.nan)
		sem_width_pct.append(np.nan)

	if has_width_data:
		fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
	else:
		fig, axes = plt.subplots(1, 1, figsize=(7.2, 4.8), constrained_layout=True)
		axes = [axes]

	x = np.arange(len(categories))

	axes[0].bar(x, mean_length_pct, yerr=sem_length_pct, capsize=4, color="#2C7FB8")
	axes[0].axhline(0.0, color="black", linewidth=1)
	axes[0].set_title("Mean Length Percentage Difference (+/- 1 SEM)")
	axes[0].set_ylabel("Difference (%)")
	axes[0].set_xticks(x)
	axes[0].set_xticklabels(categories, rotation=25, ha="right")
	axes[0].grid(True, axis="y", alpha=0.3)
	annotate_sample_sizes(axes[0], x, sample_sizes)

	if has_width_data:
		axes[1].bar(x, mean_width_pct, yerr=sem_width_pct, capsize=4, color="#41AB5D")
		axes[1].axhline(0.0, color="black", linewidth=1)
		axes[1].set_title("Mean Width Percentage Difference (+/- 1 SEM)")
		axes[1].set_ylabel("Difference (%)")
		axes[1].set_xticks(x)
		axes[1].set_xticklabels(categories, rotation=25, ha="right")
		axes[1].grid(True, axis="y", alpha=0.3)
		annotate_sample_sizes(axes[1], x, sample_sizes)

	fig.suptitle("Combined Summary of Human vs Computer Percentage Differences")

	output_path = output_root / "combined_percentage_difference_summary.png"
	fig.savefig(output_path, dpi=220)
	plt.close(fig)
	print(f"[ok] Saved: {output_path}")
	return output_path


def create_unified_summary_plot(category_dirs: list[Path], output_root: Path) -> Path | None:
	merged_data = collect_merged_data(category_dirs)

	categories: list[str] = []

	# Difference stats (units: um)
	sample_sizes_diff: list[int] = []
	mean_length_diff: list[float] = []
	sem_length_diff: list[float] = []
	mean_width_diff: list[float] = []
	sem_width_diff: list[float] = []
	all_length_diff_values: list[float] = []
	all_width_diff_values: list[float] = []
	has_width_diff_data = False

	# Percentage stats (units: %)
	sample_sizes_pct: list[int] = []
	mean_length_pct: list[float] = []
	sem_length_pct: list[float] = []
	mean_width_pct: list[float] = []
	sem_width_pct: list[float] = []
	all_length_pct_values: list[float] = []
	all_width_pct_values: list[float] = []
	has_width_pct_data = False

	for category_name in get_merged_category_order(category_dirs, merged_data):
		bucket = merged_data[category_name]
		length_diff = pd.Series(bucket["length_diff"], dtype=float)
		if length_diff.empty:
			continue

		categories.append(category_name)

		sample_sizes_diff.append(int(length_diff.shape[0]))
		mean_length_diff.append(float(length_diff.mean()))
		sem_length_diff.append(sem_for_plot(length_diff))
		all_length_diff_values.extend(length_diff.to_list())

		width_diff = pd.Series(bucket["width_diff"], dtype=float)
		if not width_diff.empty:
			has_width_diff_data = True
			mean_width_diff.append(float(width_diff.mean()))
			sem_width_diff.append(sem_for_plot(width_diff))
			all_width_diff_values.extend(width_diff.to_list())
		else:
			mean_width_diff.append(np.nan)
			sem_width_diff.append(np.nan)

		length_pct = pd.Series(bucket["length_pct"], dtype=float)
		if not length_pct.empty:
			sample_sizes_pct.append(int(length_pct.shape[0]))
			mean_length_pct.append(float(length_pct.mean()))
			sem_length_pct.append(sem_for_plot(length_pct))
			all_length_pct_values.extend(length_pct.to_list())
		else:
			sample_sizes_pct.append(0)
			mean_length_pct.append(np.nan)
			sem_length_pct.append(np.nan)

		width_pct = pd.Series(bucket["width_pct"], dtype=float)
		if not width_pct.empty:
			has_width_pct_data = True
			mean_width_pct.append(float(width_pct.mean()))
			sem_width_pct.append(sem_for_plot(width_pct))
			all_width_pct_values.extend(width_pct.to_list())
		else:
			mean_width_pct.append(np.nan)
			sem_width_pct.append(np.nan)

	if not categories:
		print("[skip] Unified summary: no valid category data found")
		return None

	# Add pooled bars
	categories.append("all_categories")

	all_length_diff = pd.Series(all_length_diff_values, dtype=float)
	sample_sizes_diff.append(int(all_length_diff.shape[0]))
	mean_length_diff.append(float(all_length_diff.mean()))
	sem_length_diff.append(sem_for_plot(all_length_diff))

	if has_width_diff_data and all_width_diff_values:
		all_width_diff = pd.Series(all_width_diff_values, dtype=float)
		mean_width_diff.append(float(all_width_diff.mean()))
		sem_width_diff.append(sem_for_plot(all_width_diff))
	else:
		mean_width_diff.append(np.nan)
		sem_width_diff.append(np.nan)

	all_length_pct = pd.Series(all_length_pct_values, dtype=float)
	sample_sizes_pct.append(int(all_length_pct.shape[0]))
	mean_length_pct.append(float(all_length_pct.mean()) if not all_length_pct.empty else np.nan)
	sem_length_pct.append(sem_for_plot(all_length_pct) if not all_length_pct.empty else np.nan)

	if has_width_pct_data and all_width_pct_values:
		all_width_pct = pd.Series(all_width_pct_values, dtype=float)
		mean_width_pct.append(float(all_width_pct.mean()))
		sem_width_pct.append(sem_for_plot(all_width_pct))
	else:
		mean_width_pct.append(np.nan)
		sem_width_pct.append(np.nan)

	fig, axes = plt.subplots(2, 2, figsize=(15, 9.2), constrained_layout=True)
	x = np.arange(len(categories))

	# Top row: raw differences
	axes[0, 0].bar(x, mean_length_diff, yerr=sem_length_diff, capsize=4, color="#4C78A8")
	axes[0, 0].axhline(0.0, color="black", linewidth=1)
	axes[0, 0].set_title("Mean Length Difference (Computer - Human, +/- 1 SEM)")
	axes[0, 0].set_ylabel("Difference (um)")
	axes[0, 0].set_xticks(x)
	axes[0, 0].set_xticklabels(categories, rotation=25, ha="right")
	axes[0, 0].grid(True, axis="y", alpha=0.3)
	annotate_sample_sizes(axes[0, 0], x, sample_sizes_diff)

	axes[0, 1].bar(x, mean_width_diff, yerr=sem_width_diff, capsize=4, color="#54A24B")
	axes[0, 1].axhline(0.0, color="black", linewidth=1)
	axes[0, 1].set_title("Mean Width Difference (Computer - Human, +/- 1 SEM)")
	axes[0, 1].set_ylabel("Difference (um)")
	axes[0, 1].set_xticks(x)
	axes[0, 1].set_xticklabels(categories, rotation=25, ha="right")
	axes[0, 1].grid(True, axis="y", alpha=0.3)
	annotate_sample_sizes(axes[0, 1], x, sample_sizes_diff)
	if not has_width_diff_data:
		axes[0, 1].text(0.5, 0.5, "No valid width-difference data", ha="center", va="center", transform=axes[0, 1].transAxes)

	# Bottom row: percentage differences
	axes[1, 0].bar(x, mean_length_pct, yerr=sem_length_pct, capsize=4, color="#2C7FB8")
	axes[1, 0].axhline(0.0, color="black", linewidth=1)
	axes[1, 0].set_title("Mean Length Percentage Difference (+/- 1 SEM)")
	axes[1, 0].set_ylabel("Difference (%)")
	axes[1, 0].set_xticks(x)
	axes[1, 0].set_xticklabels(categories, rotation=25, ha="right")
	axes[1, 0].grid(True, axis="y", alpha=0.3)
	annotate_sample_sizes(axes[1, 0], x, sample_sizes_pct)

	axes[1, 1].bar(x, mean_width_pct, yerr=sem_width_pct, capsize=4, color="#41AB5D")
	axes[1, 1].axhline(0.0, color="black", linewidth=1)
	axes[1, 1].set_title("Mean Width Percentage Difference (+/- 1 SEM)")
	axes[1, 1].set_ylabel("Difference (%)")
	axes[1, 1].set_xticks(x)
	axes[1, 1].set_xticklabels(categories, rotation=25, ha="right")
	axes[1, 1].grid(True, axis="y", alpha=0.3)
	annotate_sample_sizes(axes[1, 1], x, sample_sizes_pct)
	if not has_width_pct_data:
		axes[1, 1].text(0.5, 0.5, "No valid width-percentage data", ha="center", va="center", transform=axes[1, 1].transAxes)

	fig.suptitle("Combined Summary: Difference and Percentage Difference")

	output_path = output_root / "combined_difference_and_percentage_summary.png"
	fig.savefig(output_path, dpi=220)
	plt.close(fig)
	print(f"[ok] Saved: {output_path}")
	return output_path


def collect_discoverability(
	category_dirs: list[Path],
) -> dict[str, dict[str, int]]:
	"""Read matching_summary.json from each category dir and aggregate by merged category.

	Returns a dict mapping merged_category -> {num_gt, num_matched}.
	"""
	result: dict[str, dict[str, int]] = {}
	for category_dir in category_dirs:
		summary_path = category_dir / "matching_summary.json"
		if not summary_path.exists():
			print(f"[skip] {category_dir.name}: matching_summary.json not found")
			continue
		with summary_path.open(encoding="utf-8") as f:
			summary = json.load(f)
		source_category = category_dir.name
		merged_category = get_merged_category_name(source_category)
		bucket = result.setdefault(merged_category, {"num_gt": 0, "num_matched": 0})
		bucket["num_gt"] += int(summary.get("num_human_masks", 0))
		bucket["num_matched"] += int(summary.get("num_matched", 0))
	return result


def create_comparison_plot(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> Path | None:
	print("\n[comparison] Collecting NemaSize data...")
	nesg_data = collect_merged_data(nesg_dirs)
	print("[comparison] Collecting CellProfiler data...")
	cp_data = collect_merged_data(cp_dirs)

	combined_for_ordering: dict[str, dict[str, list[float]]] = {**nesg_data, **cp_data}
	category_names = get_merged_category_order(nesg_dirs, combined_for_ordering)

	if not category_names:
		print("[skip] Comparison plot: no valid categories found in either method")
		return None

	display_categories = list(category_names) + ["all_categories"]

	# Build per-category raw-value dicts for box plots.
	def _raw(data: dict[str, dict[str, list[float]]], key: str) -> dict[str, list[float]]:
		out: dict[str, list[float]] = {}
		for cat in category_names:
			out[cat] = list(pd.Series(data.get(cat, {}).get(key, []), dtype=float).dropna())
		out["all_categories"] = [v for cat in category_names for v in out[cat]]
		return out

	nesg_diff_raw = _raw(nesg_data, "length_diff")
	cp_diff_raw = _raw(cp_data, "length_diff")
	nesg_pct_raw = _raw(nesg_data, "length_pct")
	cp_pct_raw = _raw(cp_data, "length_pct")

	from matplotlib.patches import Patch

	def _jitter(n: int, spread: float = 0.06) -> np.ndarray:
		rng = np.random.default_rng(42)
		return rng.uniform(-spread, spread, size=n)

	def _draw_comparison_boxplot(
		ax: plt.Axes,
		cats: list[str],
		nesg_raw: dict[str, list[float]],
		cp_raw: dict[str, list[float]],
		ylabel: str,
		title: str,
	) -> None:
		offset = 0.2
		data_nesg = [nesg_raw.get(c, []) for c in cats]
		data_cp = [cp_raw.get(c, []) for c in cats]
		pos_n = [i - offset for i in range(len(cats))]
		pos_c = [i + offset for i in range(len(cats))]

		bp_n = ax.boxplot(
			data_nesg, positions=pos_n, widths=0.3, patch_artist=True,
			showfliers=False, medianprops=dict(color="black", linewidth=1.5),
		)
		bp_c = ax.boxplot(
			data_cp, positions=pos_c, widths=0.3, patch_artist=True,
			showfliers=False, medianprops=dict(color="black", linewidth=1.5),
		)
		for patch in bp_n["boxes"]:
			patch.set_facecolor("#4C78A8"); patch.set_alpha(0.5)
		for patch in bp_c["boxes"]:
			patch.set_facecolor("#E45756"); patch.set_alpha(0.5)

		for i, cat in enumerate(cats):
			n_vals = np.array(nesg_raw.get(cat, []))
			c_vals = np.array(cp_raw.get(cat, []))
			if n_vals.size:
				ax.scatter(i - offset + _jitter(n_vals.size), n_vals,
				           s=16, alpha=0.7, color="#4C78A8", edgecolors="white",
				           linewidths=0.3, zorder=3)
			if c_vals.size:
				ax.scatter(i + offset + _jitter(c_vals.size), c_vals,
				           s=16, alpha=0.7, color="#E45756", edgecolors="white",
				           linewidths=0.3, zorder=3)
			ax.text(i - offset, 0.98, f"n={n_vals.size}",
			        transform=ax.get_xaxis_transform(), ha="center", va="top",
			        fontsize=14, rotation=90, color="#333333")
			ax.text(i + offset, 0.98, f"n={c_vals.size}",
			        transform=ax.get_xaxis_transform(), ha="center", va="top",
			        fontsize=14, rotation=90, color="#333333")

		ax.axhline(0.0, color="black", linewidth=1)
		ax.set_ylabel(ylabel)
		ax.set_xticks(range(len(cats)))
		ax.set_xticklabels(cats, rotation=25, ha="right")
		ax.grid(True, axis="y", alpha=0.3)
		ax.legend(
			handles=[Patch(facecolor="#4C78A8", alpha=0.5, label="NemaSize"),
			         Patch(facecolor="#E45756", alpha=0.5, label="CellProfiler")],
			loc="upper left",
			bbox_to_anchor=(1.02, 1.0),
			borderaxespad=0.0,
		)

	box_dir = output_root / "box_plots"
	box_dir.mkdir(parents=True, exist_ok=True)

	with plt.rc_context(LARGE_FONT_RC):
		fig_diff, ax_diff = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
		_draw_comparison_boxplot(
			ax_diff, display_categories, nesg_diff_raw, cp_diff_raw,
			"Difference (um)", "Length Difference (Computer − Human)",
		)
		diff_output_path = box_dir / "nesg_vs_cellprofiler_comparison_diff.png"
		fig_diff.savefig(diff_output_path, dpi=220)
		plt.close(fig_diff)
		print(f"[ok] Saved: {diff_output_path}")

		fig_pct, ax_pct = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
		_draw_comparison_boxplot(
			ax_pct, display_categories, nesg_pct_raw, cp_pct_raw,
			"Percentage error (%)", "Length Percentage Difference",
		)
		output_path = box_dir / "nesg_vs_cellprofiler_comparison_pct.png"
		fig_pct.savefig(output_path, dpi=220)
		plt.close(fig_pct)
		print(f"[ok] Saved: {output_path}")

	# --- Discoverability panel (bar chart – single value per category) ---
	if VALIDATION_ONLY:
		print("[validation] Skipping discoverability panel (not computed in validation-only mode)")
		return output_path

	nesg_disc = collect_discoverability(nesg_dirs)
	cp_disc = collect_discoverability(cp_dirs)

	nesg_disc_pcts: list[float] = []
	cp_disc_pcts: list[float] = []
	gt_sample_sizes: list[int] = []

	for cat in display_categories:
		if cat == "all_categories":
			ng = sum(v["num_gt"] for v in nesg_disc.values())
			nm = sum(v["num_matched"] for v in nesg_disc.values())
			cg = sum(v["num_gt"] for v in cp_disc.values())
			cm = sum(v["num_matched"] for v in cp_disc.values())
		else:
			nesg_b = nesg_disc.get(cat, {"num_gt": 0, "num_matched": 0})
			cp_b = cp_disc.get(cat, {"num_gt": 0, "num_matched": 0})
			ng, nm = nesg_b["num_gt"], nesg_b["num_matched"]
			cg, cm = cp_b["num_gt"], cp_b["num_matched"]
		nesg_disc_pcts.append((nm / ng * 100.0) if ng > 0 else 0.0)
		cp_disc_pcts.append((cm / cg * 100.0) if cg > 0 else 0.0)
		# GT counts should match between methods (same human masks); fall back to
		# the larger one if they differ for any reason.
		gt_sample_sizes.append(max(ng, cg))

	x = np.arange(len(display_categories))
	bar_w = 0.38
	bar_offset = bar_w / 2

	with plt.rc_context(LARGE_FONT_RC):
		disc_fig, disc_ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
		disc_ax.bar(x - bar_offset, nesg_disc_pcts, bar_w, color="#4C78A8", label="NemaSize")
		disc_ax.bar(x + bar_offset, cp_disc_pcts, bar_w, color="#E45756", label="CellProfiler")
		for xi, n_gt in zip(x, gt_sample_sizes):
			disc_ax.text(
				xi, 101.5,
				f"n = {n_gt}",
				ha="left", va="bottom", fontsize=14, rotation=45,
			)
		disc_ax.set_ylim(0, 115)
		disc_ax.axhline(100.0, color="gray", linewidth=0.8, linestyle="--")
		disc_ax.set_ylabel("Percentage processed (%)")
		disc_ax.set_xticks(x)
		disc_ax.set_xticklabels(display_categories, rotation=25, ha="right")
		disc_ax.tick_params(axis="both", labelsize=plt.rcParams["xtick.labelsize"])
		disc_ax.grid(True, axis="y", alpha=0.3)
		disc_ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
		disc_output_path = box_dir / "nesg_vs_cellprofiler_discoverability.png"
		disc_fig.savefig(disc_output_path, dpi=220)
	plt.close(disc_fig)
	print(f"[ok] Saved: {disc_output_path}")

	return output_path


def create_cross_method_scatter_plots(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> tuple[list[Path], list[tuple[str, float, float, float]]]:
	"""For each category, produce a scatter plot of GT vs Computer for worms matched in both methods.

	Only worms whose human_file appears in *both* the NemaSize and CellProfiler
	matched CSVs are included.  Each category gets one figure with two panels
	(NemaSize on the left, CellProfiler on the right).
	"""
	nesg_by_cat: dict[str, tuple[pd.DataFrame, Path]] = {}
	for d in nesg_dirs:
		csv_path = d / CSV_NAME
		if csv_path.exists():
			src = d.name
			merged = get_merged_category_name(src)
			df = pd.read_csv(csv_path)
			df = apply_censor_filter(df, category_name=src, category_dir=d)
			if merged in nesg_by_cat:
				nesg_by_cat[merged] = (pd.concat([nesg_by_cat[merged][0], df], ignore_index=True), d)
			else:
				nesg_by_cat[merged] = (df, d)

	cp_by_cat: dict[str, tuple[pd.DataFrame, Path]] = {}
	for d in cp_dirs:
		csv_path = d / CSV_NAME
		if csv_path.exists():
			src = d.name
			merged = get_merged_category_name(src)
			df = pd.read_csv(csv_path)
			df = apply_censor_filter(df, category_name=src, category_dir=d)
			if merged in cp_by_cat:
				cp_by_cat[merged] = (pd.concat([cp_by_cat[merged][0], df], ignore_index=True), d)
			else:
				cp_by_cat[merged] = (df, d)

	common_categories = sorted(set(nesg_by_cat) & set(cp_by_cat))
	if not common_categories:
		print("[skip] Cross-method scatter: no categories common to both methods")
		return [], []

	scatter_dir = output_root / "scatter_plots"
	scatter_dir.mkdir(parents=True, exist_ok=True)
	saved: list[Path] = []
	delta_r_by_category: list[tuple[str, float, float, float]] = []

	for cat in common_categories:
		nesg_df = nesg_by_cat[cat][0].copy()
		cp_df = cp_by_cat[cat][0].copy()

		if "human_file" not in nesg_df.columns or "human_file" not in cp_df.columns:
			print(f"[skip] {cat}: missing human_file column")
			continue

		# Keep only worms discovered by both methods.
		common_humans = set(nesg_df["human_file"]) & set(cp_df["human_file"])
		if not common_humans:
			print(f"[skip] {cat}: no worms matched by both methods")
			continue

		nesg_df = nesg_df[nesg_df["human_file"].isin(common_humans)].copy()
		cp_df = cp_df[cp_df["human_file"].isin(common_humans)].copy()

		# Resolve computer-length column names.
		nesg_gt = pd.to_numeric(nesg_df["length_um_human"], errors="coerce")
		if "length_um_computer" in nesg_df.columns:
			nesg_comp = pd.to_numeric(nesg_df["length_um_computer"], errors="coerce")
		else:
			nesg_comp = nesg_gt + pd.to_numeric(nesg_df["length_diff_um"], errors="coerce")

		cp_gt = pd.to_numeric(cp_df["length_um_human"], errors="coerce")
		if "length_um_cellprofiler" in cp_df.columns:
			cp_comp = pd.to_numeric(cp_df["length_um_cellprofiler"], errors="coerce")
		elif "length_um_computer" in cp_df.columns:
			cp_comp = pd.to_numeric(cp_df["length_um_computer"], errors="coerce")
		else:
			cp_comp = cp_gt + pd.to_numeric(cp_df["length_diff_um"], errors="coerce")

		# Drop NaN rows.
		nesg_valid = nesg_gt.notna() & nesg_comp.notna()
		cp_valid = cp_gt.notna() & cp_comp.notna()
		nesg_gt, nesg_comp = nesg_gt[nesg_valid], nesg_comp[nesg_valid]
		cp_gt, cp_comp = cp_gt[cp_valid], cp_comp[cp_valid]

		if nesg_gt.empty and cp_gt.empty:
			print(f"[skip] {cat}: no valid numeric data after filtering")
			continue

		# Shared axis limits.
		all_vals = pd.concat([nesg_gt, nesg_comp, cp_gt, cp_comp]).dropna()
		lo = float(all_vals.min()) * 0.9
		hi = float(all_vals.max()) * 1.1

		fig, ax = plt.subplots(1, 1, figsize=(7, 6.5), constrained_layout=True)
		ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")

		nesg_r = np.nan
		cp_r = np.nan
		stat_parts: list[str] = []
		for gt, comp, label, color in [
			(cp_gt, cp_comp, "CellProfiler", "#E45756"),
			(nesg_gt, nesg_comp, "NemaSize", "#4C78A8"),
		]:
			ax.scatter(gt, comp, s=28, alpha=0.4, color=color, edgecolors="white",
			           linewidths=0.3, label=label)
			if gt.shape[0] >= 2:
				r = float(gt.corr(comp))
				if label == "NemaSize":
					nesg_r = r
				else:
					cp_r = r
				stat_parts.append(f"{label}: n={int(gt.shape[0])}, r={r:.3f}")
			else:
				stat_parts.append(f"{label}: n={int(gt.shape[0])}")

		if np.isfinite(nesg_r) and np.isfinite(cp_r):
			delta_r = nesg_r - cp_r
			delta_r_by_category.append((cat, delta_r, nesg_r, cp_r))
			delta_label = f"\u0394r (NemaSize \u2212 CellProfiler) = {delta_r:.3f}"
		else:
			delta_label = "\u0394r (NemaSize \u2212 CellProfiler) = n/a"
		stat_parts.append(delta_label)

		ax.set_xlim(lo, hi)
		ax.set_ylim(lo, hi)
		ax.set_aspect("equal", adjustable="box")
		ax.set_xlabel("Ground Truth Length (um)")
		ax.set_ylabel("Computer Length (um)")
		ax.set_title("\n".join(stat_parts), fontsize=18)
		ax.grid(True, alpha=0.3)
		ax.legend(loc="upper left")
		fig.suptitle(f"{cat}: GT vs Computer Length (common matches only)")

		out_path = scatter_dir / f"{cat}_scatter.png"
		fig.savefig(out_path, dpi=220)
		plt.close(fig)
		print(f"[ok] Saved: {out_path}")
		saved.append(out_path)

	# --- Pooled scatter across all categories ---
	all_nesg_gt_vals: list[float] = []
	all_nesg_comp_vals: list[float] = []
	all_cp_gt_vals: list[float] = []
	all_cp_comp_vals: list[float] = []

	for cat in common_categories:
		nesg_df = nesg_by_cat[cat][0].copy()
		cp_df = cp_by_cat[cat][0].copy()
		common_humans = set(nesg_df["human_file"]) & set(cp_df["human_file"])
		if not common_humans:
			continue
		nesg_df = nesg_df[nesg_df["human_file"].isin(common_humans)]
		cp_df = cp_df[cp_df["human_file"].isin(common_humans)]

		n_gt = pd.to_numeric(nesg_df["length_um_human"], errors="coerce")
		if "length_um_computer" in nesg_df.columns:
			n_comp = pd.to_numeric(nesg_df["length_um_computer"], errors="coerce")
		else:
			n_comp = n_gt + pd.to_numeric(nesg_df["length_diff_um"], errors="coerce")
		mask = n_gt.notna() & n_comp.notna()
		all_nesg_gt_vals.extend(n_gt[mask].to_list())
		all_nesg_comp_vals.extend(n_comp[mask].to_list())

		c_gt = pd.to_numeric(cp_df["length_um_human"], errors="coerce")
		if "length_um_cellprofiler" in cp_df.columns:
			c_comp = pd.to_numeric(cp_df["length_um_cellprofiler"], errors="coerce")
		elif "length_um_computer" in cp_df.columns:
			c_comp = pd.to_numeric(cp_df["length_um_computer"], errors="coerce")
		else:
			c_comp = c_gt + pd.to_numeric(cp_df["length_diff_um"], errors="coerce")
		mask = c_gt.notna() & c_comp.notna()
		all_cp_gt_vals.extend(c_gt[mask].to_list())
		all_cp_comp_vals.extend(c_comp[mask].to_list())

	if all_nesg_gt_vals or all_cp_gt_vals:
		pool_nesg_gt = pd.Series(all_nesg_gt_vals, dtype=float)
		pool_nesg_comp = pd.Series(all_nesg_comp_vals, dtype=float)
		pool_cp_gt = pd.Series(all_cp_gt_vals, dtype=float)
		pool_cp_comp = pd.Series(all_cp_comp_vals, dtype=float)

		all_vals = pd.concat([pool_nesg_gt, pool_nesg_comp, pool_cp_gt, pool_cp_comp]).dropna()
		lo = float(all_vals.min()) * 0.9
		hi = float(all_vals.max()) * 1.1

		fig, ax = plt.subplots(1, 1, figsize=(7, 6.5), constrained_layout=True)
		ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")

		pool_nesg_r = np.nan
		pool_cp_r = np.nan
		stat_parts: list[str] = []
		for gt, comp, label, color in [
			(pool_cp_gt, pool_cp_comp, "CellProfiler", "#E45756"),
			(pool_nesg_gt, pool_nesg_comp, "NemaSize", "#4C78A8"),
		]:
			ax.scatter(gt, comp, s=28, alpha=0.4, color=color, edgecolors="white",
			           linewidths=0.3, label=label)
			if gt.shape[0] >= 2:
				r = float(gt.corr(comp))
				if label == "NemaSize":
					pool_nesg_r = r
				else:
					pool_cp_r = r
				stat_parts.append(f"{label}: n={int(gt.shape[0])}, r={r:.3f}")
			else:
				stat_parts.append(f"{label}: n={int(gt.shape[0])}")

		if np.isfinite(pool_nesg_r) and np.isfinite(pool_cp_r):
			pool_delta_r = pool_nesg_r - pool_cp_r
			pool_delta_label = f"\u0394r (NemaSize \u2212 CellProfiler) = {pool_delta_r:.3f}"
		else:
			pool_delta_label = "\u0394r (NemaSize \u2212 CellProfiler) = n/a"
		stat_parts.append(pool_delta_label)

		ax.set_xlim(lo, hi)
		ax.set_ylim(lo, hi)
		ax.set_aspect("equal", adjustable="box")
		ax.set_xlabel("Ground Truth Length (um)")
		ax.set_ylabel("Computer Length (um)")
		ax.set_title("\n".join(stat_parts), fontsize=18)
		ax.grid(True, alpha=0.3)
		ax.legend(loc="upper left")
		fig.suptitle("All Categories: GT vs Computer Length (common matches only)")
		out_path = scatter_dir / "all_categories_scatter.png"
		fig.savefig(out_path, dpi=220)
		plt.close(fig)
		print(f"[ok] Saved: {out_path}")
		saved.append(out_path)

	return saved, delta_r_by_category


def create_cross_method_difference_boxplots(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> list[Path]:
	"""Box plots of measurement differences for worms common to both methods.

	For each category (and pooled), shows NemaSize vs CellProfiler box plots
	side by side, overlaid with individual jittered data points.
	"""
	nesg_by_cat: dict[str, pd.DataFrame] = {}
	for d in nesg_dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		nesg_by_cat[merged] = pd.concat([nesg_by_cat[merged], df], ignore_index=True) if merged in nesg_by_cat else df

	cp_by_cat: dict[str, pd.DataFrame] = {}
	for d in cp_dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		cp_by_cat[merged] = pd.concat([cp_by_cat[merged], df], ignore_index=True) if merged in cp_by_cat else df

	common_categories = sorted(set(nesg_by_cat) & set(cp_by_cat))
	if not common_categories:
		print("[skip] Cross-method box plots: no categories common to both methods")
		return []

	# Collect per-category diff values restricted to common worms.
	cat_nesg_length_diff: dict[str, list[float]] = {}
	cat_cp_length_diff: dict[str, list[float]] = {}
	cat_nesg_length_pct: dict[str, list[float]] = {}
	cat_cp_length_pct: dict[str, list[float]] = {}
	cat_nesg_width_diff: dict[str, list[float]] = {}
	cat_cp_width_diff: dict[str, list[float]] = {}
	cat_nesg_width_pct: dict[str, list[float]] = {}
	cat_cp_width_pct: dict[str, list[float]] = {}

	for cat in common_categories:
		nesg_df_full = nesg_by_cat[cat].copy()
		cp_df = cp_by_cat[cat].copy()
		if "human_file" not in nesg_df_full.columns or "human_file" not in cp_df.columns:
			continue
		common_humans = set(nesg_df_full["human_file"]) & set(cp_df["human_file"])
		if not common_humans:
			continue

		nesg_df = nesg_df_full[nesg_df_full["human_file"].isin(common_humans)]
		cp_df = cp_df[cp_df["human_file"].isin(common_humans)]

		# Length diff (um).
		n_ld = pd.to_numeric(nesg_df["length_diff_um"], errors="coerce").dropna()
		c_ld = pd.to_numeric(cp_df["length_diff_um"], errors="coerce").dropna()
		if not n_ld.empty and not c_ld.empty:
			cat_nesg_length_diff[cat] = n_ld.tolist()
			cat_cp_length_diff[cat] = c_ld.tolist()

		# Length pct diff.
		if "length_um_human" in nesg_df.columns:
			n_pct = percentage_diff(
				pd.to_numeric(nesg_df["length_diff_um"], errors="coerce"),
				pd.to_numeric(nesg_df["length_um_human"], errors="coerce"),
			)
			c_pct = percentage_diff(
				pd.to_numeric(cp_df["length_diff_um"], errors="coerce"),
				pd.to_numeric(cp_df["length_um_human"], errors="coerce"),
			)
			if not n_pct.empty and not c_pct.empty:
				cat_nesg_length_pct[cat] = n_pct.tolist()
				cat_cp_length_pct[cat] = c_pct.tolist()

		# Width diff (um) — NemaSize only, using ALL NemaSize worms (not
		# restricted to common matches with CellProfiler, since CellProfiler
		# does not measure width).
		if "width_diff_um" in nesg_df_full.columns:
			n_wd = pd.to_numeric(nesg_df_full["width_diff_um"], errors="coerce").dropna()
			if not n_wd.empty:
				cat_nesg_width_diff[cat] = n_wd.tolist()

		# Width pct diff — NemaSize only, using ALL NemaSize worms.
		if {"width_diff_um", "width_um_human"}.issubset(nesg_df_full.columns):
			n_wpct = percentage_diff(
				pd.to_numeric(nesg_df_full["width_diff_um"], errors="coerce"),
				pd.to_numeric(nesg_df_full["width_um_human"], errors="coerce"),
			)
			if not n_wpct.empty:
				cat_nesg_width_pct[cat] = n_wpct.tolist()

	if not cat_nesg_length_diff:
		print("[skip] Cross-method box plots: no valid common-worm data")
		return []

	# Add pooled "all_categories".
	display_cats = list(common_categories) + ["all_categories"]
	cat_nesg_length_diff["all_categories"] = [v for vals in cat_nesg_length_diff.values() for v in vals]
	cat_cp_length_diff["all_categories"] = [v for vals in cat_cp_length_diff.values() for v in vals]
	if cat_nesg_length_pct:
		cat_nesg_length_pct["all_categories"] = [v for vals in cat_nesg_length_pct.values() for v in vals]
		cat_cp_length_pct["all_categories"] = [v for vals in cat_cp_length_pct.values() for v in vals]
	if cat_nesg_width_diff:
		cat_nesg_width_diff["all_categories"] = [v for vals in cat_nesg_width_diff.values() if vals for v in vals]
	if cat_nesg_width_pct:
		cat_nesg_width_pct["all_categories"] = [v for vals in cat_nesg_width_pct.values() if vals for v in vals]

	has_pct = bool(cat_nesg_length_pct)
	has_width = bool(cat_nesg_width_diff)
	has_width_pct = bool(cat_nesg_width_pct)
	box_dir = output_root / "box_plots"
	box_dir.mkdir(parents=True, exist_ok=True)
	saved: list[Path] = []

	def _jitter(n: int, spread: float = 0.06) -> np.ndarray:
		rng = np.random.default_rng(42)
		return rng.uniform(-spread, spread, size=n)

	def _draw_boxplot_panel(
		ax: plt.Axes,
		display_cats: list[str],
		nesg_data: dict[str, list[float]],
		cp_data: dict[str, list[float]],
		ylabel: str,
		title: str,
	) -> None:
		positions_nesg = []
		positions_cp = []
		data_nesg = []
		data_cp = []
		offset = 0.2

		for i, cat in enumerate(display_cats):
			n_vals = nesg_data.get(cat, [])
			c_vals = cp_data.get(cat, [])
			data_nesg.append(n_vals)
			data_cp.append(c_vals)
			positions_nesg.append(i - offset)
			positions_cp.append(i + offset)

		bp_n = ax.boxplot(
			data_nesg,
			positions=positions_nesg,
			widths=0.3,
			patch_artist=True,
			showfliers=False,
			medianprops=dict(color="black", linewidth=1.5),
		)
		bp_c = ax.boxplot(
			data_cp,
			positions=positions_cp,
			widths=0.3,
			patch_artist=True,
			showfliers=False,
			medianprops=dict(color="black", linewidth=1.5),
		)
		for patch in bp_n["boxes"]:
			patch.set_facecolor("#4C78A8")
			patch.set_alpha(0.5)
		for patch in bp_c["boxes"]:
			patch.set_facecolor("#E45756")
			patch.set_alpha(0.5)

		# Overlay individual points with jitter.
		for i, cat in enumerate(display_cats):
			n_vals = np.array(nesg_data.get(cat, []))
			c_vals = np.array(cp_data.get(cat, []))
			if n_vals.size:
				ax.scatter(
					i - offset + _jitter(n_vals.size),
					n_vals,
					s=16, alpha=0.7, color="#4C78A8", edgecolors="white",
					linewidths=0.3, zorder=3,
				)
			if c_vals.size:
				ax.scatter(
					i + offset + _jitter(c_vals.size),
					c_vals,
					s=16, alpha=0.7, color="#E45756", edgecolors="white",
					linewidths=0.3, zorder=3,
				)
			# Annotate n once per category (centered between the two boxes).
			n_total = max(int(n_vals.size), int(c_vals.size))
			ax.text(
				i, 0.98, f"n = {n_total}",
				transform=ax.get_xaxis_transform(),
				ha="center", va="top", fontsize=20, rotation=90, color="#333333",
			)

		ax.axhline(0.0, color="black", linewidth=1)
		ax.set_ylabel(ylabel)
		ax.set_xticks(range(len(display_cats)))
		ax.set_xticklabels(display_cats, rotation=25, ha="right")
		ax.grid(True, axis="y", alpha=0.3)
		# Legend.
		from matplotlib.patches import Patch
		ax.legend(
			handles=[Patch(facecolor="#4C78A8", alpha=0.5, label="NemaSize"),
			         Patch(facecolor="#E45756", alpha=0.5, label="CellProfiler")],
			loc="upper left",
			bbox_to_anchor=(1.02, 1.0),
			borderaxespad=0.0,
		)

	def _draw_single_method_boxplot_panel(
		ax: plt.Axes,
		display_cats: list[str],
		data: dict[str, list[float]],
		ylabel: str,
		title: str,
		method_label: str,
		color: str,
	) -> None:
		positions = list(range(len(display_cats)))
		series = [data.get(cat, []) for cat in display_cats]

		bp = ax.boxplot(
			series,
			positions=positions,
			widths=0.5,
			patch_artist=True,
			showfliers=False,
			medianprops=dict(color="black", linewidth=1.5),
		)
		for patch in bp["boxes"]:
			patch.set_facecolor(color)
			patch.set_alpha(0.5)

		for i, cat in enumerate(display_cats):
			vals = np.array(data.get(cat, []))
			if vals.size:
				ax.scatter(
					i + _jitter(vals.size),
					vals,
					s=16, alpha=0.7, color=color, edgecolors="white",
					linewidths=0.3, zorder=3,
				)
			ax.text(
				i, 0.98, f"n={vals.size}",
				transform=ax.get_xaxis_transform(),
				ha="center", va="top", fontsize=20, rotation=90, color="#333333",
			)

		ax.axhline(0.0, color="black", linewidth=1)
		ax.set_ylabel(ylabel)
		ax.set_xticks(positions)
		ax.set_xticklabels(display_cats, rotation=25, ha="right")
		ax.grid(True, axis="y", alpha=0.3)
		from matplotlib.patches import Patch
		ax.legend(
			handles=[Patch(facecolor=color, alpha=0.5, label=method_label)],
			loc="upper left",
			bbox_to_anchor=(1.02, 1.0),
			borderaxespad=0.0,
		)

	# Absolute difference figure
	fig_diff, ax_diff = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
	_draw_boxplot_panel(
		ax_diff,
		display_cats,
		cat_nesg_length_diff,
		cat_cp_length_diff,
		"Difference (um)",
		"Length Difference (Computer − Human)",
	)
	out_path_diff = box_dir / "nesg_vs_cellprofiler_difference_boxplots.png"
	fig_diff.savefig(out_path_diff, dpi=220)
	plt.close(fig_diff)
	print(f"[ok] Saved: {out_path_diff}")
	saved.append(out_path_diff)

	# Percentage difference figure
	if has_pct:
		fig_pct, ax_pct = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
		_draw_boxplot_panel(
			ax_pct,
			display_cats,
			cat_nesg_length_pct,
			cat_cp_length_pct,
			"Percentage error (%)",
			"Length Percentage Difference",
		)
		out_path_pct = box_dir / "nesg_vs_cellprofiler_pct_difference_boxplots.png"
		fig_pct.savefig(out_path_pct, dpi=220)
		plt.close(fig_pct)
		print(f"[ok] Saved: {out_path_pct}")
		saved.append(out_path_pct)

	# Width absolute difference figure (NemaSize only — CellProfiler has no width).
	if has_width:
		fig_wd, ax_wd = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
		_draw_single_method_boxplot_panel(
			ax_wd,
			display_cats,
			cat_nesg_width_diff,
			"Difference (um)",
			"Width Difference (NemaSize − Human)",
			method_label="NemaSize",
			color="#4C78A8",
		)
		out_path_wd = box_dir / "nesg_vs_cellprofiler_width_difference_boxplots.png"
		fig_wd.savefig(out_path_wd, dpi=220)
		plt.close(fig_wd)
		print(f"[ok] Saved: {out_path_wd}")
		saved.append(out_path_wd)

	# Width percentage difference figure (NemaSize only).
	if has_width_pct:
		fig_wpct, ax_wpct = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
		_draw_single_method_boxplot_panel(
			ax_wpct,
			display_cats,
			cat_nesg_width_pct,
			"Percentage error (%)",
			"Width Percentage Difference (NemaSize vs Human)",
			method_label="NemaSize",
			color="#4C78A8",
		)
		out_path_wpct = box_dir / "nesg_vs_cellprofiler_width_pct_difference_boxplots.png"
		fig_wpct.savefig(out_path_wpct, dpi=220)
		plt.close(fig_wpct)
		print(f"[ok] Saved: {out_path_wpct}")
		saved.append(out_path_wpct)

	return saved


def create_anova_table(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> Path | None:
	"""One-way ANOVA (GT vs NemaSize vs CellProfiler) per category + pooled.

	For each category the three groups are the length measurements of worms
	common to both methods.  Results are saved as a CSV table and printed.
	"""
	from scipy.stats import f_oneway, tukey_hsd

	nesg_by_cat: dict[str, pd.DataFrame] = {}
	for d in nesg_dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		nesg_by_cat[merged] = pd.concat([nesg_by_cat[merged], df], ignore_index=True) if merged in nesg_by_cat else df

	cp_by_cat: dict[str, pd.DataFrame] = {}
	for d in cp_dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		cp_by_cat[merged] = pd.concat([cp_by_cat[merged], df], ignore_index=True) if merged in cp_by_cat else df

	common_categories = sorted(set(nesg_by_cat) & set(cp_by_cat))
	if not common_categories:
		print("[skip] ANOVA: no categories common to both methods")
		return None

	rows: list[dict[str, object]] = []

	# Collect pooled arrays alongside per-category.
	pool_gt: list[float] = []
	pool_nesg: list[float] = []
	pool_cp: list[float] = []

	for cat in common_categories:
		nesg_df = nesg_by_cat[cat].copy()
		cp_df = cp_by_cat[cat].copy()
		if "human_file" not in nesg_df.columns or "human_file" not in cp_df.columns:
			continue
		common_humans = set(nesg_df["human_file"]) & set(cp_df["human_file"])
		if not common_humans:
			continue
		nesg_df = nesg_df[nesg_df["human_file"].isin(common_humans)].copy()
		cp_df = cp_df[cp_df["human_file"].isin(common_humans)].copy()

		# Align by human_file so all three arrays correspond to the same worms.
		nesg_df["_human_key"] = nesg_df["human_file"].astype(str)
		cp_df["_human_key"] = cp_df["human_file"].astype(str)

		# GT length (from NemaSize side – same human masks).
		nesg_df["_gt"] = pd.to_numeric(nesg_df["length_um_human"], errors="coerce")

		# NemaSize computer length.
		if "length_um_computer" in nesg_df.columns:
			nesg_df["_nesg"] = pd.to_numeric(nesg_df["length_um_computer"], errors="coerce")
		else:
			nesg_df["_nesg"] = (pd.to_numeric(nesg_df["length_um_human"], errors="coerce")
			                    + pd.to_numeric(nesg_df["length_diff_um"], errors="coerce"))

		# CellProfiler computer length.
		if "length_um_cellprofiler" in cp_df.columns:
			cp_df["_cp"] = pd.to_numeric(cp_df["length_um_cellprofiler"], errors="coerce")
		elif "length_um_computer" in cp_df.columns:
			cp_df["_cp"] = pd.to_numeric(cp_df["length_um_computer"], errors="coerce")
		else:
			cp_df["_cp"] = (pd.to_numeric(cp_df["length_um_human"], errors="coerce")
			                + pd.to_numeric(cp_df["length_diff_um"], errors="coerce"))

		# Merge on human_file to keep only worms with valid values in all three.
		merged = nesg_df[["_human_key", "_gt", "_nesg"]].merge(
			cp_df[["_human_key", "_cp"]], on="_human_key", how="inner",
		).dropna(subset=["_gt", "_nesg", "_cp"])

		if merged.empty:
			continue

		gt_vals = merged["_gt"].values
		nesg_vals = merged["_nesg"].values
		cp_vals = merged["_cp"].values

		pool_gt.extend(gt_vals.tolist())
		pool_nesg.extend(nesg_vals.tolist())
		pool_cp.extend(cp_vals.tolist())

		row = _compute_anova_row(cat, gt_vals, nesg_vals, cp_vals)
		if row is not None:
			rows.append(row)

	# Pooled across all categories.
	if pool_gt:
		row = _compute_anova_row(
			"all_categories",
			np.array(pool_gt), np.array(pool_nesg), np.array(pool_cp),
		)
		if row is not None:
			rows.append(row)

	if not rows:
		print("[skip] ANOVA: no valid data")
		return None

	table = pd.DataFrame(rows)
	out_path = output_root / "anova_gt_nesg_cp.csv"
	table.to_csv(out_path, index=False)
	print(f"\n[ok] Saved ANOVA table: {out_path}")
	_print_anova_table(table)
	return out_path


def _compute_anova_row(
	category: str,
	gt: np.ndarray,
	nesg: np.ndarray,
	cp: np.ndarray,
) -> dict[str, object] | None:
	from scipy.stats import f_oneway, tukey_hsd

	# Need at least 2 observations per group for meaningful ANOVA.
	if gt.shape[0] < 2 or nesg.shape[0] < 2 or cp.shape[0] < 2:
		print(f"[skip] ANOVA {category}: n={gt.shape[0]}, need >= 2 per group")
		return None

	f_stat, p_anova = f_oneway(gt, nesg, cp)
	# Tukey HSD pairwise.
	tukey = tukey_hsd(gt, nesg, cp)
	# tukey.pvalue is a 3x3 matrix; indices 0=GT, 1=NemaSize, 2=CellProfiler
	p_gt_nesg = float(tukey.pvalue[0, 1])
	p_gt_cp = float(tukey.pvalue[0, 2])
	p_nesg_cp = float(tukey.pvalue[1, 2])

	return {
		"category": category,
		"n": int(gt.shape[0]),
		"GT_mean": round(float(gt.mean()), 2),
		"GT_std": round(float(gt.std(ddof=1)), 2),
		"NemaSize_mean": round(float(nesg.mean()), 2),
		"NemaSize_std": round(float(nesg.std(ddof=1)), 2),
		"CellProfiler_mean": round(float(cp.mean()), 2),
		"CellProfiler_std": round(float(cp.std(ddof=1)), 2),
		"F_statistic": round(float(f_stat), 4),
		"p_ANOVA": float(f"{p_anova:.4g}"),
		"p_GT_vs_NemaSize": float(f"{p_gt_nesg:.4g}"),
		"p_GT_vs_CellProfiler": float(f"{p_gt_cp:.4g}"),
		"p_NemaSize_vs_CellProfiler": float(f"{p_nesg_cp:.4g}"),
	}


def _print_anova_table(table: pd.DataFrame) -> None:
	print("\n[ANOVA] Ground Truth vs NemaSize vs CellProfiler (length, um):")
	header = (
		f"  {'Category':<30} {'n':>5}  "
		f"{'GT mean':>9} {'NeSg mean':>10} {'CP mean':>9}  "
		f"{'F':>9} {'p(ANOVA)':>10}  "
		f"{'p(GT-NS)':>10} {'p(GT-CP)':>10} {'p(NS-CP)':>10}"
	)
	print(header)
	print(f"  {'-'*len(header.strip())}")
	for _, r in table.iterrows():
		print(
			f"  {r['category']:<30} {r['n']:>5}  "
			f"{r['GT_mean']:>9.2f} {r['NemaSize_mean']:>10.2f} {r['CellProfiler_mean']:>9.2f}  "
			f"{r['F_statistic']:>9.4f} {r['p_ANOVA']:>10.4g}  "
			f"{r['p_GT_vs_NemaSize']:>10.4g} {r['p_GT_vs_CellProfiler']:>10.4g} {r['p_NemaSize_vs_CellProfiler']:>10.4g}"
		)


def _build_axis_lookup(axis_name: str) -> dict[str, str]:
	"""Invert PUBLICATION_AXES[axis] into {source_class: parent_group}."""
	mapping: dict[str, str] = {}
	for parent, sources in PUBLICATION_AXES[axis_name].items():
		for src in sources:
			if src in mapping:
				raise ValueError(
					f"Source class '{src}' is mapped to multiple parents on axis "
					f"'{axis_name}': '{mapping[src]}' and '{parent}'. "
					"Each source must map to exactly one parent per axis."
				)
			mapping[src] = parent
	return mapping


def _load_method_dataframes(
	dirs: list[Path],
	axis_lookup: dict[str, str] | None,
) -> dict[str, pd.DataFrame]:
	"""Read each category CSV, apply censor filter, route to a parent group.

	If `axis_lookup` is provided, source classes absent from the lookup are
	skipped. If `axis_lookup` is None, the source class itself is used as the
	parent key (one bucket per source).
	"""
	by_parent: dict[str, list[pd.DataFrame]] = {}
	for d in dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		if axis_lookup is not None:
			parent = axis_lookup.get(src)
			if parent is None:
				continue
		else:
			parent = src
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		by_parent.setdefault(parent, []).append(df)
	return {
		parent: pd.concat(frames, ignore_index=True)
		for parent, frames in by_parent.items()
	}


def _extract_gt_and_computer_lengths(
	df: pd.DataFrame, method: str
) -> tuple[pd.Series, pd.Series]:
	"""Return (gt_length, computer_length) with NaN rows dropped."""
	gt = pd.to_numeric(df["length_um_human"], errors="coerce")
	if method == "NemaSize":
		if "length_um_computer" in df.columns:
			comp = pd.to_numeric(df["length_um_computer"], errors="coerce")
		else:
			comp = gt + pd.to_numeric(df["length_diff_um"], errors="coerce")
	else:  # CellProfiler
		if "length_um_cellprofiler" in df.columns:
			comp = pd.to_numeric(df["length_um_cellprofiler"], errors="coerce")
		elif "length_um_computer" in df.columns:
			comp = pd.to_numeric(df["length_um_computer"], errors="coerce")
		else:
			comp = gt + pd.to_numeric(df["length_diff_um"], errors="coerce")
	mask = gt.notna() & comp.notna()
	return gt[mask].reset_index(drop=True), comp[mask].reset_index(drop=True)


def _common_worm_subframes(
	nesg_df: pd.DataFrame, cp_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
	"""Intersect on human_file; return per-method subframes (may be empty)."""
	if "human_file" not in nesg_df.columns or "human_file" not in cp_df.columns:
		return nesg_df.iloc[0:0], cp_df.iloc[0:0]
	common = set(nesg_df["human_file"]) & set(cp_df["human_file"])
	if not common:
		return nesg_df.iloc[0:0], cp_df.iloc[0:0]
	return (
		nesg_df[nesg_df["human_file"].isin(common)].copy(),
		cp_df[cp_df["human_file"].isin(common)].copy(),
	)


# Method colors (NemaSize vs CellProfiler).
_PUBLICATION_METHOD_COLORS = {
	"NemaSize": "#E45756",
	"CellProfiler": "#4C78A8",
}

# Per-bucket color variation within each method's family. Order matches
# PUBLICATION_POOLED_BUCKETS (occlusion -> curvature -> straight). Hues stay
# inside the red/blue theme but separate the buckets visually.
_PUBLICATION_BUCKET_COLORS: dict[str, dict[str, str]] = {
	"NemaSize": {
		"Overlapping":           "#7A0E0E",  # deep maroon
		"Bent or curly":         "#E45756",  # base red
		"Straight and isolated": "#F4A6A1",  # light salmon
	},
	"CellProfiler": {
		"Overlapping":           "#0B3D6B",  # deep navy
		"Bent or curly":         "#4C78A8",  # base blue
		"Straight and isolated": "#9EC4E5",  # light steel blue
	},
}

# Marker shape cycle for parent groups within an axis figure.
_PUBLICATION_GROUP_MARKERS = ["o", "^", "s", "D", "P", "X", "v", "*"]


def _draw_publication_axis_scatter(
	ax: plt.Axes,
	groups_data: list[
		tuple[str, pd.Series, pd.Series, pd.Series, pd.Series]
	],
	title: str,
	xy_limits: tuple[float, float] | None = None,
) -> None:
	"""Draw a single axis scatter panel.

	Color encodes method (NemaSize vs CellProfiler); marker shape encodes the
	parent group on the axis.

	Parameters
	----------
	groups_data
		List of (group_name, nesg_gt, nesg_comp, cp_gt, cp_comp) tuples in the
		order they should appear in the legend.
	xy_limits
		Optional (lo, hi) shared axis limits. If None, limits are derived from
		`groups_data`.
	"""
	# Derive limits from this panel's data if no shared limits were supplied.
	if xy_limits is None:
		all_vals: list[float] = []
		for _, ng, nc, cg, cc in groups_data:
			all_vals.extend(ng.tolist())
			all_vals.extend(nc.tolist())
			all_vals.extend(cg.tolist())
			all_vals.extend(cc.tolist())
		if not all_vals:
			ax.text(0.5, 0.5, "No common worms", ha="center", va="center",
			        transform=ax.transAxes)
			ax.set_title(title)
			return
		lo = float(min(all_vals)) * 0.9
		hi = float(max(all_vals)) * 1.1
	else:
		lo, hi = xy_limits

	yx_line, = ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")

	pooled_n_gt: list[float] = []
	pooled_n_comp: list[float] = []
	pooled_c_gt: list[float] = []
	pooled_c_comp: list[float] = []

	# Collect per-group handles so legend can list NemaSize above CellProfiler.
	legend_pairs: list[tuple] = []  # (ns_handle_or_None, cp_handle_or_None, group)

	for idx, (group, ng, nc, cg, cc) in enumerate(groups_data):
		marker = _PUBLICATION_GROUP_MARKERS[idx % len(_PUBLICATION_GROUP_MARKERS)]
		ns_h = None
		cp_h = None
		if cg.size:
			cp_h = ax.scatter(cg, cc, s=34, alpha=0.55,
			           color=_PUBLICATION_METHOD_COLORS["CellProfiler"], marker=marker,
			           edgecolors="white", linewidths=0.3,
			           label=f"{group} – CellProfiler (n={cg.size})")
			pooled_c_gt.extend(cg.tolist())
			pooled_c_comp.extend(cc.tolist())
		if ng.size:
			ns_h = ax.scatter(ng, nc, s=30, alpha=0.55,
			           color=_PUBLICATION_METHOD_COLORS["NemaSize"], marker=marker,
			           edgecolors="white", linewidths=0.3,
			           label=f"{group} – NemaSize (n={ng.size})")
			pooled_n_gt.extend(ng.tolist())
			pooled_n_comp.extend(nc.tolist())
		legend_pairs.append((ns_h, cp_h, group))

	# Pooled r per method across the figure.
	def _pool_r(a: list[float], b: list[float]) -> float:
		if len(a) >= 2:
			return float(pd.Series(a).corr(pd.Series(b)))
		return float("nan")

	r_nesg = _pool_r(pooled_n_gt, pooled_n_comp)
	r_cp = _pool_r(pooled_c_gt, pooled_c_comp)
	if np.isfinite(r_nesg) and np.isfinite(r_cp):
		delta_r = r_nesg - r_cp
		pooled_line = (
			f"Pooled: NS r={r_nesg:.3f} (n={len(pooled_n_gt)}),  "
			f"CP r={r_cp:.3f} (n={len(pooled_c_gt)}),  "
			f"\u0394r = {delta_r:+.3f}"
		)
	else:
		pooled_line = (
			f"Pooled: NS n={len(pooled_n_gt)}, CP n={len(pooled_c_gt)}"
		)

	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)
	ax.set_aspect("equal", adjustable="box")
	ax.set_xlabel("Ground Truth Length (um)")
	ax.set_ylabel("Computer Length (um)")
	ax.set_title(f"{title}\n{pooled_line}", fontsize=10)
	ax.grid(True, alpha=0.3)

	# Build legend with NemaSize listed above CellProfiler within each group.
	ordered_handles = [yx_line]
	ordered_labels = ["y = x"]
	for ns_h, cp_h, _grp in legend_pairs:
		if ns_h is not None:
			ordered_handles.append(ns_h)
			ordered_labels.append(ns_h.get_label())
		if cp_h is not None:
			ordered_handles.append(cp_h)
			ordered_labels.append(cp_h.get_label())
	ax.legend(ordered_handles, ordered_labels,
	          loc="upper left", fontsize=8, framealpha=0.9)


def _collect_axis_groups_data(
	axis_name: str,
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
) -> list[tuple[str, pd.Series, pd.Series, pd.Series, pd.Series]]:
	"""Build per-group (NemaSize, CellProfiler) common-worm length pairs."""
	axis_lookup = _build_axis_lookup(axis_name)
	nesg_by_parent = _load_method_dataframes(nesg_dirs, axis_lookup)
	cp_by_parent = _load_method_dataframes(cp_dirs, axis_lookup)

	groups_data: list[tuple[str, pd.Series, pd.Series, pd.Series, pd.Series]] = []
	baseline = PUBLICATION_AXIS_BASELINE.get(axis_name)
	for group in PUBLICATION_AXIS_ORDER[axis_name]:
		if group == baseline:
			print(f"[info] publication/{axis_name}: excluding baseline group '{group}'")
			continue
		nesg_df = nesg_by_parent.get(group, pd.DataFrame())
		cp_df = cp_by_parent.get(group, pd.DataFrame())
		empty = (pd.Series(dtype=float),) * 4
		if nesg_df.empty or cp_df.empty:
			print(f"[skip] publication/{axis_name}/{group}: no data on one or both methods")
			groups_data.append((group, *empty))
			continue
		nesg_sub, cp_sub = _common_worm_subframes(nesg_df, cp_df)
		if nesg_sub.empty or cp_sub.empty:
			print(f"[skip] publication/{axis_name}/{group}: no common human_file matches")
			groups_data.append((group, *empty))
			continue
		ng, nc = _extract_gt_and_computer_lengths(nesg_sub, "NemaSize")
		cg, cc = _extract_gt_and_computer_lengths(cp_sub, "CellProfiler")
		groups_data.append((group, ng, nc, cg, cc))
	return groups_data


def _collect_pooled_data(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
) -> tuple[list[float], list[float], list[float], list[float]]:
	"""Pooled (across all source classes) common-worm length pairs."""
	by_src = _collect_pooled_data_by_source(nesg_dirs, cp_dirs)

	all_n_gt: list[float] = []
	all_n_comp: list[float] = []
	all_c_gt: list[float] = []
	all_c_comp: list[float] = []
	for ng, nc, cg, cc in by_src.values():
		all_n_gt.extend(ng.tolist())
		all_n_comp.extend(nc.tolist())
		all_c_gt.extend(cg.tolist())
		all_c_comp.extend(cc.tolist())
	return all_n_gt, all_n_comp, all_c_gt, all_c_comp


def _collect_pooled_data_by_source(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
) -> dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
	"""Per-source-class common-worm length pairs.

	Returns {source_class: (nesg_gt, nesg_comp, cp_gt, cp_comp)}.
	"""
	nesg_by_src = _load_method_dataframes(nesg_dirs, axis_lookup=None)
	cp_by_src = _load_method_dataframes(cp_dirs, axis_lookup=None)

	result: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
	for src in sorted(set(nesg_by_src) & set(cp_by_src)):
		nesg_sub, cp_sub = _common_worm_subframes(nesg_by_src[src], cp_by_src[src])
		if nesg_sub.empty or cp_sub.empty:
			continue
		ng, nc = _extract_gt_and_computer_lengths(nesg_sub, "NemaSize")
		cg, cc = _extract_gt_and_computer_lengths(cp_sub, "CellProfiler")
		result[src] = (ng, nc, cg, cc)
	return result


def create_publication_axis_scatter(
	axis_name: str,
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
	xy_limits: tuple[float, float] | None = None,
) -> Path | None:
	"""One scatter figure for an axis, marker shape per parent group.

	Color encodes method (NemaSize vs CellProfiler). Only worms whose
	human_file appears in BOTH methods (within the same parent group) are
	plotted.
	"""
	groups_data = _collect_axis_groups_data(axis_name, nesg_dirs, cp_dirs)

	fig, ax = plt.subplots(1, 1, figsize=(8, 7.5), constrained_layout=True)
	_draw_publication_axis_scatter(
		ax, groups_data,
		title=f"{axis_name.capitalize()} axis: GT vs Computer length",
		xy_limits=xy_limits,
	)
	out_path = output_root / f"{axis_name}_scatter.png"
	fig.savefig(out_path, dpi=300)
	plt.close(fig)
	print(f"[ok] Saved: {out_path}")
	return out_path


def create_publication_pooled_scatter(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
	xy_limits: tuple[float, float] | None = None,
) -> Path | None:
	"""All-categories pooled scatter, common worms only.

	Color encodes method (NemaSize=red, CellProfiler=blue). Marker shape
	encodes a curvature/occlusion bucket derived from the source class
	(see PUBLICATION_POOLED_BUCKETS) — not the size axis, since the plot
	axes already encode length.
	"""
	by_src = _collect_pooled_data_by_source(nesg_dirs, cp_dirs)
	if not by_src:
		print("[skip] publication/all_categories: no valid common-worm data")
		return None

	bucket_lookup = _build_pooled_bucket_lookup()
	# Aggregate into buckets, preserving the priority order.
	bucket_order = [label for label, _, _ in PUBLICATION_POOLED_BUCKETS]
	bucket_marker = {label: marker for label, marker, _ in PUBLICATION_POOLED_BUCKETS}
	per_bucket: dict[str, dict[str, list[float]]] = {
		label: {"n_gt": [], "n_comp": [], "c_gt": [], "c_comp": []}
		for label in bucket_order
	}

	unbucketed: list[str] = []
	for src, (ng, nc, cg, cc) in by_src.items():
		if src not in bucket_lookup:
			unbucketed.append(src)
			continue
		label, _ = bucket_lookup[src]
		b = per_bucket[label]
		b["n_gt"].extend(ng.tolist())
		b["n_comp"].extend(nc.tolist())
		b["c_gt"].extend(cg.tolist())
		b["c_comp"].extend(cc.tolist())
	if unbucketed:
		print(f"[warn] publication/all_categories: source classes without a bucket: {sorted(unbucketed)}")

	# Report per-bucket sample sizes (common worms appearing in both methods).
	print("[publication/all_categories] per-category common-worm sample sizes:")
	for src in sorted(by_src):
		ng, _, cg, _ = by_src[src]
		bucket = bucket_lookup.get(src, ("(none)", None))[0]
		print(f"  {src:<35} [{bucket:<25}] NemaSize n={len(ng):<5d} CellProfiler n={len(cg)}")
	print("[publication/all_categories] per-bucket sample sizes:")
	total_ns = 0
	total_cp = 0
	for label in bucket_order:
		b = per_bucket[label]
		print(f"  {label:<25} NemaSize n={len(b['n_gt']):<5d} CellProfiler n={len(b['c_gt'])}")
		total_ns += len(b['n_gt'])
		total_cp += len(b['c_gt'])
	print(f"  {'all_categories':<25} NemaSize n={total_ns:<5d} CellProfiler n={total_cp}")

	# Pooled values (for r and shared-limits fallback).
	all_n_gt = [v for b in per_bucket.values() for v in b["n_gt"]]
	all_n_comp = [v for b in per_bucket.values() for v in b["n_comp"]]
	all_c_gt = [v for b in per_bucket.values() for v in b["c_gt"]]
	all_c_comp = [v for b in per_bucket.values() for v in b["c_comp"]]

	if not all_n_gt and not all_c_gt:
		print("[skip] publication/all_categories: no valid common-worm data after bucketing")
		return None

	if xy_limits is None:
		all_vals = all_n_gt + all_n_comp + all_c_gt + all_c_comp
		lo = float(min(all_vals)) * 0.9
		hi = float(max(all_vals)) * 1.1
	else:
		lo, hi = xy_limits

	fig, ax = plt.subplots(1, 1, figsize=(SCATTER_FIG_WIDTH, SCATTER_FIG_HEIGHT), constrained_layout=True)
	ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="_nolegend_")

	# Draw order: largest-n bucket first (bottom), smallest-n on top so the
	# rarer/harder cases are visible. PUBLICATION_POOLED_BUCKETS lists the
	# occlusion bucket first (smallest n) -> straight last (largest n), so
	# we reverse to draw straight first.
	draw_order = list(reversed(bucket_order))
	# Within each bucket, draw CellProfiler first so NemaSize sits on top.
	# Track handles per bucket so the legend can list NemaSize above CP.
	bucket_handles: dict[str, dict[str, object]] = {}
	# Per-(bucket, method) r² values for the legend annotations.
	bucket_r2: dict[str, dict[str, float]] = {}
	for label in draw_order:
		b = per_bucket[label]
		marker = bucket_marker[label]
		cp_color = _PUBLICATION_BUCKET_COLORS["CellProfiler"][label]
		ns_color = _PUBLICATION_BUCKET_COLORS["NemaSize"][label]
		cp_h = None
		ns_h = None
		r2_ns = float("nan")
		r2_cp = float("nan")
		if len(b["n_gt"]) >= 2:
			r_ns = float(pd.Series(b["n_gt"]).corr(pd.Series(b["n_comp"])))
			if np.isfinite(r_ns):
				r2_ns = r_ns * r_ns
		if len(b["c_gt"]) >= 2:
			r_cp = float(pd.Series(b["c_gt"]).corr(pd.Series(b["c_comp"])))
			if np.isfinite(r_cp):
				r2_cp = r_cp * r_cp
		bucket_r2[label] = {"ns": r2_ns, "cp": r2_cp}

		def _fmt_label(method: str, r2: float) -> str:
			base = f"{label} - {method}"
			if np.isfinite(r2):
				return f"{base} (r²={r2:.3f})"
			return base

		if b["c_gt"]:
			# Hollow marker (no face, colored edge) for CellProfiler.
			cp_h = ax.scatter(
				b["c_gt"], b["c_comp"], s=SCATTER_MARKER_SIZE_CP, alpha=0.95,
				facecolors="none",
				edgecolors=cp_color,
				linewidths=1.2, marker=marker,
				label=_fmt_label("CP", r2_cp),
			)
		if b["n_gt"]:
			# Solid filled marker for NemaSize with white edge for separation.
			# Sized so its outer diameter matches the CellProfiler hollow
			# marker above (CP s=42 + 1.2pt edge ≈ NS s=58).
			ns_h = ax.scatter(
				b["n_gt"], b["n_comp"], s=SCATTER_MARKER_SIZE_NS, alpha=0.95,
				color=ns_color, marker=marker,
				edgecolors="white", linewidths=0.5,
				label=_fmt_label("NS", r2_ns),
			)
		bucket_handles[label] = {"ns": ns_h, "cp": cp_h}

	r_nesg = float(pd.Series(all_n_gt).corr(pd.Series(all_n_comp))) if len(all_n_gt) >= 2 else float("nan")
	r_cp = float(pd.Series(all_c_gt).corr(pd.Series(all_c_comp))) if len(all_c_gt) >= 2 else float("nan")
	if np.isfinite(r_nesg) and np.isfinite(r_cp):
		print(
			f"[publication] all_categories: NemaSize r={r_nesg:.3f} (n={len(all_n_gt)}), "
			f"CellProfiler r={r_cp:.3f} (n={len(all_c_gt)}), "
			f"\u0394r = {r_nesg - r_cp:+.3f}"
		)
	# Print per-(bucket, method) r² values for traceability.
	print("[publication/all_categories] per-(bucket, method) r²:")
	for label in bucket_order:
		r2 = bucket_r2.get(label, {})
		print(
			f"  {label:<25} NemaSize r²={r2.get('ns', float('nan')):.3f}  "
			f"CellProfiler r²={r2.get('cp', float('nan')):.3f}"
		)

	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)
	ax.set_aspect("equal", adjustable="box")
	_pub_ticks = np.arange(200, 1201, 200)
	ax.set_xticks(_pub_ticks)
	ax.set_yticks(_pub_ticks)
	ax.set_xlabel("Ground-truth animal length (µm)", fontfamily="Arial", fontsize=PUB_AXIS_FONTSIZE)
	ax.set_ylabel("Computer measured animal length (µm)", fontfamily="Arial", fontsize=PUB_AXIS_FONTSIZE)
	# No title, no grid, and no surrounding box – only the x/y axes remain.
	ax.grid(False)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	# Inward-pointing ticks, 1.7x larger tick label font.
	ax.tick_params(axis="both", direction="in", labelsize=PUB_AXIS_FONTSIZE, length=6.5, width=1.05)
	# Refresh tick label fonts (set_xlabel/set_ylabel above don't touch ticks).
	for lbl in ax.get_xticklabels() + ax.get_yticklabels():
		lbl.set_fontfamily("Arial")

	# Legend order: list buckets from "Straight" to "Overlapping" (i.e. reverse
	# of priority order), and within each bucket put NemaSize above CellProfiler.
	ordered_handles: list[object] = []
	ordered_labels: list[str] = []
	for label in reversed(bucket_order):
		pair = bucket_handles.get(label, {})
		ns_h = pair.get("ns")
		cp_h = pair.get("cp")
		if ns_h is not None:
			ordered_handles.append(ns_h)
			ordered_labels.append(ns_h.get_label())
		if cp_h is not None:
			ordered_handles.append(cp_h)
			ordered_labels.append(cp_h.get_label())
	leg = ax.legend(ordered_handles, ordered_labels,
	                loc=SCATTER_LEGEND_LOC, bbox_to_anchor=SCATTER_LEGEND_BBOX,
	                fontsize=PUB_LEGEND_FONTSIZE, frameon=False,
	                handletextpad=0,
	                prop={"family": "Arial", "size": PUB_LEGEND_FONTSIZE})
	for txt in leg.get_texts():
		txt.set_fontfamily("Arial")

	out_path = output_root / "all_categories_scatter.png"
	fig.savefig(out_path, dpi=300)
	svg_path = out_path.with_suffix(".svg")
	fig.savefig(svg_path)
	plt.close(fig)
	print(f"[ok] Saved: {out_path}")
	print(f"[ok] Saved: {svg_path}")

	# --- Per-group statistics file ---
	stats_path = output_root / "all_categories_scatter_stats.txt"
	header = f"{'bucket':<25} {'method':<8} {'n':>6} {'r2':>10}"
	lines = [
		"Per-group statistics for all_categories_scatter.png",
		"r² = squared Pearson correlation between ground-truth length and "
		"computer-measured length, computed on common worms within each bucket.",
		"",
		header,
		"-" * len(header),
	]

	def _fmt(bucket: str, method: str, n: int, r2: float) -> str:
		r2_str = f"{r2:.3f}" if np.isfinite(r2) else "nan"
		return f"{bucket:<25} {method:<8} {n:>6d} {r2_str:>10}"

	total_ns_n = 0
	total_cp_n = 0
	total_ns_gt: list[float] = []
	total_ns_comp: list[float] = []
	total_cp_gt: list[float] = []
	total_cp_comp: list[float] = []
	for label in bucket_order:
		b = per_bucket[label]
		r2 = bucket_r2.get(label, {})
		lines.append(_fmt(label, "NS", len(b["n_gt"]), r2.get("ns", float("nan"))))
		lines.append(_fmt(label, "CP", len(b["c_gt"]), r2.get("cp", float("nan"))))
		total_ns_n += len(b["n_gt"])
		total_cp_n += len(b["c_gt"])
		total_ns_gt.extend(b["n_gt"])
		total_ns_comp.extend(b["n_comp"])
		total_cp_gt.extend(b["c_gt"])
		total_cp_comp.extend(b["c_comp"])

	def _r2_pool(x: list[float], y: list[float]) -> float:
		if len(x) < 2:
			return float("nan")
		r = float(pd.Series(x).corr(pd.Series(y)))
		return r * r if np.isfinite(r) else float("nan")

	lines.append(_fmt("all_categories", "NS", total_ns_n, _r2_pool(total_ns_gt, total_ns_comp)))
	lines.append(_fmt("all_categories", "CP", total_cp_n, _r2_pool(total_cp_gt, total_cp_comp)))

	stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"[ok] Saved: {stats_path}")

	return out_path


def create_publication_difference_boxplot(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> Path | None:
	"""Publication-styled horizontal box plot of length-diff per category.

	Saves `<output_root>/difference_boxplot.png`. Categories run along the
	y-axis (boxes are horizontal). Only worms whose `human_file` appears in
	both methods (per merged category) are included. Style matches the
	pooled scatter: Arial, no title/grid, only bottom+left spines, inward
	ticks, NemaSize above CellProfiler in the legend.
	"""
	# --- Collect data (mirrors create_cross_method_difference_boxplots) ---
	nesg_by_cat: dict[str, pd.DataFrame] = {}
	for d in nesg_dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		# Remember the on-disk source folder so we can later locate the
		# annotated skeleton image for top-error reporting.
		df = df.copy()
		df["_src_cat"] = src
		nesg_by_cat[merged] = (
			pd.concat([nesg_by_cat[merged], df], ignore_index=True)
			if merged in nesg_by_cat else df
		)

	cp_by_cat: dict[str, pd.DataFrame] = {}
	for d in cp_dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		df = df.copy()
		df["_src_cat"] = src
		cp_by_cat[merged] = (
			pd.concat([cp_by_cat[merged], df], ignore_index=True)
			if merged in cp_by_cat else df
		)

	common_categories = sorted(set(nesg_by_cat) & set(cp_by_cat))
	cat_nesg: dict[str, list[float]] = {}
	cat_cp: dict[str, list[float]] = {}
	# Parallel lists tracking the source `human_file` for each value so we can
	# report the top-N highest absolute-percentage-error images later.
	cat_nesg_files: dict[str, list[str]] = {}
	cat_cp_files: dict[str, list[str]] = {}
	# Parallel lists tracking the on-disk source category folder per value so
	# we can locate the annotated skeleton image to copy.
	cat_nesg_srcs: dict[str, list[str]] = {}
	cat_cp_srcs: dict[str, list[str]] = {}
	for cat in common_categories:
		ndf = nesg_by_cat[cat]
		cdf = cp_by_cat[cat]
		if "human_file" not in ndf.columns or "human_file" not in cdf.columns:
			continue
		common_humans = set(ndf["human_file"]) & set(cdf["human_file"])
		if not common_humans:
			continue
		ndf = ndf[ndf["human_file"].isin(common_humans)]
		cdf = cdf[cdf["human_file"].isin(common_humans)]
		if "length_um_human" not in ndf.columns or "length_um_human" not in cdf.columns:
			continue
		n_pct = percentage_diff(
			pd.to_numeric(ndf["length_diff_um"], errors="coerce"),
			pd.to_numeric(ndf["length_um_human"], errors="coerce"),
		)
		c_pct = percentage_diff(
			pd.to_numeric(cdf["length_diff_um"], errors="coerce"),
			pd.to_numeric(cdf["length_um_human"], errors="coerce"),
		)
		n_files = ndf["human_file"].astype(str).reset_index(drop=True)
		c_files = cdf["human_file"].astype(str).reset_index(drop=True)
		n_srcs = ndf["_src_cat"].astype(str).reset_index(drop=True)
		c_srcs = cdf["_src_cat"].astype(str).reset_index(drop=True)
		n_pct = n_pct.reset_index(drop=True)
		c_pct = c_pct.reset_index(drop=True)
		n_mask = n_pct.notna()
		c_mask = c_pct.notna()
		n_pct_valid = n_pct[n_mask]
		c_pct_valid = c_pct[c_mask]
		if not n_pct_valid.empty and not c_pct_valid.empty:
			cat_nesg[cat] = n_pct_valid.tolist()
			cat_cp[cat] = c_pct_valid.tolist()
			cat_nesg_files[cat] = n_files[n_mask].tolist()
			cat_cp_files[cat] = c_files[c_mask].tolist()
			cat_nesg_srcs[cat] = n_srcs[n_mask].tolist()
			cat_cp_srcs[cat] = c_srcs[c_mask].tolist()

	# Merge small-n categories (n < 5) into larger related buckets so every
	# row in the boxplot has enough samples. Defect type (incomplete /
	# overlapping) takes precedence over shape modifier (curly / tiny);
	# tiny_bent_worm rolls up to bent_worm on the curvature axis.
	publication_merge_map = {
		"overlapping_incomplete_worm": "overlapping_worm",
		"overlapping_curly_worm": "overlapping_worm",
		"incomplete_curly_worm": "incomplete_worm",
		"incomplete_tiny_worm": "incomplete_worm",
		"tiny_bent_worm": "bent_worm",
	}
	for src, dst in publication_merge_map.items():
		if src in cat_nesg:
			cat_nesg.setdefault(dst, []).extend(cat_nesg.pop(src))
			cat_cp.setdefault(dst, []).extend(cat_cp.pop(src))
			cat_nesg_files.setdefault(dst, []).extend(cat_nesg_files.pop(src, []))
			cat_cp_files.setdefault(dst, []).extend(cat_cp_files.pop(src, []))
			cat_nesg_srcs.setdefault(dst, []).extend(cat_nesg_srcs.pop(src, []))
			cat_cp_srcs.setdefault(dst, []).extend(cat_cp_srcs.pop(src, []))
			print(f"[publication/difference_boxplot] merged {src} -> {dst}")

	if not cat_nesg:
		print("[skip] publication/difference_boxplot: no valid common-worm data")
		return None

	# Report sample sizes per category (common worms appearing in both methods).
	print("[publication/difference_boxplot] per-category common-worm sample sizes:")
	low_n = []
	total_n = 0
	for cat in cat_nesg:
		n = len(cat_nesg[cat])
		print(f"  {cat:<35} n={n}")
		total_n += n
		if n < 20:
			low_n.append((cat, n))
	print(f"  {'all_categories':<35} n={total_n}")
	if low_n:
		print("[publication/difference_boxplot] categories with n < 20:")
		for cat, n in low_n:
			print(f"  {cat:<35} n={n}")

	display_cats = list(cat_nesg.keys()) + ["all_categories"]
	cat_nesg["all_categories"] = [v for c in cat_nesg for v in cat_nesg[c]]
	cat_cp["all_categories"] = [v for c in cat_cp for v in cat_cp[c]]
	cat_nesg_files["all_categories"] = [
		f for c in cat_nesg_files for f in cat_nesg_files[c]
	]
	cat_cp_files["all_categories"] = [
		f for c in cat_cp_files for f in cat_cp_files[c]
	]
	cat_nesg_srcs["all_categories"] = [
		s for c in cat_nesg_srcs for s in cat_nesg_srcs[c]
	]
	cat_cp_srcs["all_categories"] = [
		s for c in cat_cp_srcs for s in cat_cp_srcs[c]
	]

	# --- Top-N highest absolute percentage-error images in "all_categories" ---
	import shutil
	top_n_summary: dict[str, list[tuple[str, str, float, Path | None]]] = {}
	top_n = max(int(TOP_N_HIGHEST_PCT_ERROR_IMAGES), 0)
	top_n_dir = output_root / f"top{top_n}_abs_pct_error_images"
	# Cache human_file -> computer_file lookups per source category from the
	# NemaSize-side CSVs so we can locate the NemaSize-processed skeleton.
	_nesg_lookup_cache: dict[str, dict[str, str]] = {}

	def _load_nesg_lookup(src_cat: str) -> dict[str, str]:
		if src_cat in _nesg_lookup_cache:
			return _nesg_lookup_cache[src_cat]
		mapping: dict[str, str] = {}
		if GT_VS_NESG_ROOT is not None:
			csv_path = GT_VS_NESG_ROOT / src_cat / CSV_NAME
			if csv_path.exists():
				try:
					tmp = pd.read_csv(csv_path)
					if "human_file" in tmp.columns and "computer_file" in tmp.columns:
						for hf, cf in zip(
							tmp["human_file"].astype(str),
							tmp["computer_file"].astype(str),
						):
							if hf and cf and cf.lower() != "nan":
								mapping.setdefault(hf, cf)
				except Exception as exc:  # pragma: no cover - IO errors
					print(f"[warn] failed to read {csv_path}: {exc}")
		_nesg_lookup_cache[src_cat] = mapping
		return mapping

	for method_label, vals, files, srcs in (
		(
			"NemaSize",
			cat_nesg["all_categories"],
			cat_nesg_files["all_categories"],
			cat_nesg_srcs["all_categories"],
		),
		(
			"CellProfiler",
			cat_cp["all_categories"],
			cat_cp_files["all_categories"],
			cat_cp_srcs["all_categories"],
		),
	):
		triples = [
			(str(s), str(f), float(v))
			for s, f, v in zip(srcs, files, vals)
			if v is not None and np.isfinite(v)
		]
		triples.sort(key=lambda p: abs(p[2]), reverse=True)
		top = triples[:top_n]
		print(
			f"[publication/difference_boxplot] top {top_n} |%error| images "
			f"in all_categories ({method_label}):"
		)

		# CellProfiler has no annotated/processed images, so we only report
		# the ranking and skip file copying for that method.
		copy_images = method_label == "NemaSize"
		method_dir: Path | None = None
		human_dir: Path | None = None
		if copy_images:
			method_dir = top_n_dir / method_label
			method_dir.mkdir(parents=True, exist_ok=True)
			human_dir = top_n_dir / f"{method_label}_human"
			human_dir.mkdir(parents=True, exist_ok=True)

		annotated: list[tuple[str, str, float, Path | None]] = []
		for rank, (src_cat, fname, pct) in enumerate(top, start=1):
			copied_to: Path | None = None
			if copy_images and method_dir is not None:
				# Step 1: look up the matching computer_file via the NemaSize CSV.
				lookup = _load_nesg_lookup(src_cat)
				computer_file = lookup.get(fname)
				src_path: Path | None = None
				if computer_file:
					# Step 2: locate the NemaSize-processed skeleton image.
					candidate = NESG_PROCESSED_SKELETON_ROOT / computer_file
					if candidate.exists():
						src_path = candidate
					else:
						# Try alternative extensions if the recorded extension
						# doesn't match what's on disk.
						stem = Path(computer_file).stem
						for ext in IMAGE_EXTENSIONS:
							alt = NESG_PROCESSED_SKELETON_ROOT / f"{stem}{ext}"
							if alt.exists():
								src_path = alt
								break
				else:
					print(
						f"[warn] no computer_file mapping for "
						f"{src_cat}/{fname} in NeSg CSV"
					)

				if src_path is not None:
					sign = "pos" if pct >= 0 else "neg"
					dst_name = (
						f"rank{rank:02d}_abs{abs(pct):07.3f}pct_{sign}_"
						f"{src_cat}_{src_path.name}"
					)
					dst_path = method_dir / dst_name
					try:
						shutil.copy2(src_path, dst_path)
						copied_to = dst_path
					except Exception as exc:  # pragma: no cover - filesystem errors
						print(
							f"[warn] failed to copy {src_path} -> "
							f"{dst_path}: {exc}"
						)
				elif computer_file:
					print(
						f"[warn] NemaSize skeleton not found for "
						f"{src_cat}/{fname} (computer_file={computer_file})"
					)

				# Also copy the human-annotated skeleton image from
				# GT_ROIS_ROOT/<src_cat>/train/skeleton/<human_file>.
				human_src_dir = GT_ROIS_ROOT / src_cat / "train" / "skeleton"
				human_src: Path | None = None
				candidate_h = human_src_dir / fname
				if candidate_h.exists():
					human_src = candidate_h
				else:
					stem_h = Path(fname).stem
					for ext in IMAGE_EXTENSIONS:
						alt_h = human_src_dir / f"{stem_h}{ext}"
						if alt_h.exists():
							human_src = alt_h
							break
				if human_src is not None and human_dir is not None:
					sign = "pos" if pct >= 0 else "neg"
					human_dst_name = (
						f"rank{rank:02d}_abs{abs(pct):07.3f}pct_{sign}_"
						f"{src_cat}_{human_src.name}"
					)
					human_dst = human_dir / human_dst_name
					try:
						shutil.copy2(human_src, human_dst)
					except Exception as exc:  # pragma: no cover - filesystem errors
						print(
							f"[warn] failed to copy {human_src} -> "
							f"{human_dst}: {exc}"
						)
				else:
					print(
						f"[warn] human-annotated skeleton not found for "
						f"{src_cat}/{fname}"
					)

			annotated.append((src_cat, fname, pct, copied_to))
			if copy_images:
				tag = copied_to.name if copied_to is not None else "MISSING"
				print(
					f"  {rank:>2}. [{src_cat:<30}] {fname:<40} "
					f"{pct:+.3f}%  -> {tag}"
				)
			else:
				print(
					f"  {rank:>2}. [{src_cat:<30}] {fname:<40} "
					f"{pct:+.3f}%"
				)

		top_n_summary[method_label] = annotated

	# --- Plot ---
	ns_color = _PUBLICATION_METHOD_COLORS["NemaSize"]
	cp_color = _PUBLICATION_METHOD_COLORS["CellProfiler"]

	# Figure height scales with number of categories (each row gets ~0.55 in).
	n_rows = len(display_cats)
	fig_h = BOXPLOT_FIG_HEIGHT if BOXPLOT_FIG_HEIGHT is not None else max(5.0, 0.6 * n_rows + 1.5)
	fig, ax = plt.subplots(1, 1, figsize=(BOXPLOT_FIG_WIDTH, fig_h), constrained_layout=True)

	# Categories run top-to-bottom in display order: index 0 = top row.
	# y-position for category i: (n_rows - 1 - i). Within each row,
	# NemaSize is plotted above (larger y), CellProfiler below.
	offset = 0.2
	positions_ns: list[float] = []
	positions_cp: list[float] = []
	data_ns: list[list[float]] = []
	data_cp: list[list[float]] = []
	for i, cat in enumerate(display_cats):
		y = (n_rows - 1 - i)
		positions_ns.append(y + offset)
		positions_cp.append(y - offset)
		data_ns.append(cat_nesg.get(cat, []))
		data_cp.append(cat_cp.get(cat, []))

	bp_n = ax.boxplot(
		data_ns, positions=positions_ns, widths=0.3,
		vert=False, patch_artist=True, showfliers=False,
		medianprops=dict(color="black", linewidth=1.5),
	)
	bp_c = ax.boxplot(
		data_cp, positions=positions_cp, widths=0.3,
		vert=False, patch_artist=True, showfliers=False,
		medianprops=dict(color="black", linewidth=1.5),
	)
	for patch in bp_n["boxes"]:
		patch.set_facecolor(ns_color)
		patch.set_alpha(0.5)
	for patch in bp_c["boxes"]:
		patch.set_facecolor(cp_color)
		patch.set_alpha(0.5)

	# Jittered points.
	rng = np.random.default_rng(42)
	for i, cat in enumerate(display_cats):
		y = (n_rows - 1 - i)
		n_vals = np.array(cat_nesg.get(cat, []))
		c_vals = np.array(cat_cp.get(cat, []))
		if n_vals.size:
			ax.scatter(
				n_vals,
				y + offset + rng.uniform(-0.06, 0.06, size=n_vals.size),
				s=BOXPLOT_MARKER_SIZE, alpha=0.7, color=ns_color, edgecolors="white",
				linewidths=0.3, zorder=3,
			)
		if c_vals.size:
			ax.scatter(
				c_vals,
				y - offset + rng.uniform(-0.06, 0.06, size=c_vals.size),
				s=BOXPLOT_MARKER_SIZE, alpha=0.7, color=cp_color, edgecolors="white",
				linewidths=0.3, zorder=3,
			)

	ax.axvline(0.0, color="black", linewidth=1)

	# Y axis = categories (top-to-bottom in display_cats order).
	pretty_overrides = {
		"curly_tiny_worm": "Curly and tiny",
		"tiny_bent_worm": "Tiny and bent",
		"overlapping_worm": "Mutual-overlap",
		"self-overlapping_worm": "Self-overlap",
		"all_categories": "All categories",
	}

	def _pretty(name: str) -> str:
		if name in pretty_overrides:
			return pretty_overrides[name]
		words = [w for w in name.replace("_", " ").split() if w.lower() != "worm"]
		if not words:
			return name
		words[0] = words[0][:1].upper() + words[0][1:]
		return " ".join(words)

	ax.set_yticks([n_rows - 1 - i for i in range(n_rows)])
	ax.set_yticklabels([_pretty(c) for c in display_cats])
	ax.set_xticks(np.arange(-75, 76, 25))

	ax.set_xlabel("Percentage error of length (%)",
	              fontfamily="Arial", fontsize=PUB_AXIS_FONTSIZE)
	ax.set_ylabel("")
	ax.grid(False)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(axis="both", direction="in", labelsize=PUB_AXIS_FONTSIZE, length=6.5, width=1.05)
	for lbl in ax.get_xticklabels() + ax.get_yticklabels():
		lbl.set_fontfamily("Arial")

	from matplotlib.patches import Patch
	leg = ax.legend(
		handles=[
			Patch(facecolor=ns_color, alpha=0.5, label="NS"),
			Patch(facecolor=cp_color, alpha=0.5, label="CP"),
		],
		loc="upper right", bbox_to_anchor=(1.08, 1.0),
		fontsize=PUB_LEGEND_FONTSIZE, frameon=False,
		prop={"family": "Arial", "size": PUB_LEGEND_FONTSIZE},
	)
	for txt in leg.get_texts():
		txt.set_fontfamily("Arial")

	out_path = output_root / "difference_boxplot.png"
	fig.savefig(out_path, dpi=300)
	svg_path = out_path.with_suffix(".svg")
	fig.savefig(svg_path)
	plt.close(fig)
	print(f"[ok] Saved: {out_path}")
	print(f"[ok] Saved: {svg_path}")

	# --- Write per-box statistics file ---
	stats_path = output_root / "difference_boxplot_stats.txt"
	header = (
		f"{'category':<35} {'method':<12} {'n':>6} "
		f"{'mean':>10} {'median':>10} {'std':>10} "
		f"{'q25':>10} {'q75':>10}"
	)
	lines = [
		"Per-box statistics for difference_boxplot.png",
		"Values are percentage error of worm length (%): (computer - human) / human * 100",
		"",
		header,
		"-" * len(header),
	]

	def _fmt_row(cat_label: str, method: str, values: list[float]) -> str:
		arr = np.asarray(values, dtype=float)
		arr = arr[np.isfinite(arr)]
		n = arr.size
		if n == 0:
			return (
				f"{cat_label:<35} {method:<12} {n:>6} "
				f"{'nan':>10} {'nan':>10} {'nan':>10} "
				f"{'nan':>10} {'nan':>10}"
			)
		mean = float(arr.mean())
		median = float(np.median(arr))
		# Sample std (ddof=1) when possible, else 0 for a single-value group.
		std = float(arr.std(ddof=1)) if n > 1 else 0.0
		q25 = float(np.percentile(arr, 25))
		q75 = float(np.percentile(arr, 75))
		return (
			f"{cat_label:<35} {method:<12} {n:>6d} "
			f"{mean:>10.3f} {median:>10.3f} {std:>10.3f} "
			f"{q25:>10.3f} {q75:>10.3f}"
		)

	for cat in display_cats:
		lines.append(_fmt_row(cat, "NemaSize", cat_nesg.get(cat, [])))
		lines.append(_fmt_row(cat, "CellProfiler", cat_cp.get(cat, [])))

	# Append top-N highest |percentage-error| images for the pooled
	# all_categories group, one block per method. NemaSize copies the
	# processed skeleton image; CellProfiler has no processed image so the
	# block only lists ranking + filenames.
	for method_label in ("NemaSize", "CellProfiler"):
		lines.append("")
		lines.append(
			f"Top {top_n} highest |percentage error| images in all_categories "
			f"({method_label}):"
		)
		top = top_n_summary.get(method_label, [])
		if not top:
			lines.append("  (none)")
		elif method_label == "NemaSize":
			lines.append(
				f"  {'rank':>4}  {'source_category':<30} "
				f"{'human_file':<40} {'pct_error':>12}  copied_as"
			)
			for rank, (src_cat, fname, pct, copied) in enumerate(top, start=1):
				copied_name = copied.name if copied is not None else "MISSING"
				lines.append(
					f"  {rank:>4}  {src_cat:<30} {fname:<40} "
					f"{pct:>+11.3f}%  {copied_name}"
				)
		else:
			lines.append(
				f"  {'rank':>4}  {'source_category':<30} "
				f"{'human_file':<40} {'pct_error':>12}"
			)
			for rank, (src_cat, fname, pct, _copied) in enumerate(top, start=1):
				lines.append(
					f"  {rank:>4}  {src_cat:<30} {fname:<40} "
					f"{pct:>+11.3f}%"
				)

	stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"[ok] Saved: {stats_path}")

	return out_path


def create_publication_width_boxplot(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> Path | None:
	"""NS-only horizontal box plot of width-% error per category.

	Saves `<output_root>/width_difference_boxplot.png` and a sibling stats
	file `width_difference_boxplot_stats.txt`. CellProfiler does not produce
	width measurements, so only NemaSize is shown. To keep the worm
	population consistent with the length boxplot, we still restrict to NS
	rows whose `human_file` appears in the CP CSVs for the same category.
	Style mirrors `create_publication_difference_boxplot`.
	"""
	# --- Collect data ---
	nesg_by_cat: dict[str, pd.DataFrame] = {}
	for d in nesg_dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		nesg_by_cat[merged] = (
			pd.concat([nesg_by_cat[merged], df], ignore_index=True)
			if merged in nesg_by_cat else df
		)

	cp_by_cat: dict[str, pd.DataFrame] = {}
	for d in cp_dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_censor_filter(df, category_name=src, category_dir=d)
		cp_by_cat[merged] = (
			pd.concat([cp_by_cat[merged], df], ignore_index=True)
			if merged in cp_by_cat else df
		)

	common_categories = sorted(set(nesg_by_cat) & set(cp_by_cat))
	cat_nesg: dict[str, list[float]] = {}
	for cat in common_categories:
		ndf = nesg_by_cat[cat]
		cdf = cp_by_cat[cat]
		if "human_file" not in ndf.columns or "human_file" not in cdf.columns:
			continue
		common_humans = set(ndf["human_file"]) & set(cdf["human_file"])
		if not common_humans:
			continue
		ndf = ndf[ndf["human_file"].isin(common_humans)]
		if not {"width_diff_um", "width_um_human"}.issubset(ndf.columns):
			continue
		w_pct = percentage_diff(
			pd.to_numeric(ndf["width_diff_um"], errors="coerce"),
			pd.to_numeric(ndf["width_um_human"], errors="coerce"),
		).dropna()
		if not w_pct.empty:
			cat_nesg[cat] = w_pct.tolist()

	# Same publication-merge rules as the length boxplot.
	publication_merge_map = {
		"overlapping_incomplete_worm": "overlapping_worm",
		"overlapping_curly_worm": "overlapping_worm",
		"incomplete_curly_worm": "incomplete_worm",
		"incomplete_tiny_worm": "incomplete_worm",
		"tiny_bent_worm": "bent_worm",
	}
	for src, dst in publication_merge_map.items():
		if src in cat_nesg:
			cat_nesg.setdefault(dst, []).extend(cat_nesg.pop(src))
			print(f"[publication/width_boxplot] merged {src} -> {dst}")

	if not cat_nesg:
		print("[skip] publication/width_boxplot: no valid width data")
		return None

	print("[publication/width_boxplot] per-category sample sizes:")
	total_n = 0
	for cat in cat_nesg:
		n = len(cat_nesg[cat])
		print(f"  {cat:<35} n={n}")
		total_n += n
	print(f"  {'all_categories':<35} n={total_n}")

	display_cats = list(cat_nesg.keys()) + ["all_categories"]
	cat_nesg["all_categories"] = [v for c in cat_nesg for v in cat_nesg[c]]

	# --- Plot ---
	ns_color = _PUBLICATION_METHOD_COLORS["NemaSize"]
	n_rows = len(display_cats)
	fig_h = BOXPLOT_FIG_HEIGHT if BOXPLOT_FIG_HEIGHT is not None else max(5.0, 0.6 * n_rows + 1.5)
	fig, ax = plt.subplots(1, 1, figsize=(BOXPLOT_FIG_WIDTH, fig_h), constrained_layout=True)

	positions: list[float] = []
	data: list[list[float]] = []
	for i, cat in enumerate(display_cats):
		y = (n_rows - 1 - i)
		positions.append(y)
		data.append(cat_nesg.get(cat, []))

	bp = ax.boxplot(
		data, positions=positions, widths=0.5,
		vert=False, patch_artist=True, showfliers=False,
		medianprops=dict(color="black", linewidth=1.5),
	)
	for patch in bp["boxes"]:
		patch.set_facecolor(ns_color)
		patch.set_alpha(0.5)

	# Jittered points.
	rng = np.random.default_rng(42)
	for i, cat in enumerate(display_cats):
		y = (n_rows - 1 - i)
		vals = np.array(cat_nesg.get(cat, []))
		if vals.size:
			ax.scatter(
				vals,
				y + rng.uniform(-0.12, 0.12, size=vals.size),
				s=BOXPLOT_MARKER_SIZE, alpha=0.7, color=ns_color, edgecolors="white",
				linewidths=0.3, zorder=3,
			)

	ax.axvline(0.0, color="black", linewidth=1)

	pretty_overrides = {
		"curly_tiny_worm": "Curly and tiny",
		"tiny_bent_worm": "Tiny and bent",
		"overlapping_worm": "Mutual-overlap",
		"self-overlapping_worm": "Self-overlap",
		"all_categories": "All categories",
	}

	def _pretty(name: str) -> str:
		if name in pretty_overrides:
			return pretty_overrides[name]
		words = [w for w in name.replace("_", " ").split() if w.lower() != "worm"]
		if not words:
			return name
		words[0] = words[0][:1].upper() + words[0][1:]
		return " ".join(words)

	ax.set_yticks([n_rows - 1 - i for i in range(n_rows)])
	ax.set_yticklabels([_pretty(c) for c in display_cats])

	ax.set_xlabel("Percentage error of width (%)",
	              fontfamily="Arial", fontsize=PUB_AXIS_FONTSIZE)
	ax.set_ylabel("")
	ax.grid(False)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(axis="both", direction="in", labelsize=PUB_AXIS_FONTSIZE, length=6.5, width=1.05)
	for lbl in ax.get_xticklabels() + ax.get_yticklabels():
		lbl.set_fontfamily("Arial")

	out_path = output_root / "width_difference_boxplot.png"
	fig.savefig(out_path, dpi=300)
	svg_path = out_path.with_suffix(".svg")
	fig.savefig(svg_path)
	plt.close(fig)
	print(f"[ok] Saved: {out_path}")
	print(f"[ok] Saved: {svg_path}")

	# --- Stats file ---
	stats_path = output_root / "width_difference_boxplot_stats.txt"
	header = (
		f"{'category':<35} {'method':<12} {'n':>6} "
		f"{'mean':>10} {'median':>10} {'std':>10} "
		f"{'q25':>10} {'q75':>10}"
	)
	lines = [
		"Per-box statistics for width_difference_boxplot.png",
		"Values are percentage error of worm width (%): (computer - human) / human * 100",
		"",
		header,
		"-" * len(header),
	]

	def _fmt_row(cat_label: str, method: str, values: list[float]) -> str:
		arr = np.asarray(values, dtype=float)
		arr = arr[np.isfinite(arr)]
		n = arr.size
		if n == 0:
			return (
				f"{cat_label:<35} {method:<12} {n:>6} "
				f"{'nan':>10} {'nan':>10} {'nan':>10} "
				f"{'nan':>10} {'nan':>10}"
			)
		mean = float(arr.mean())
		median = float(np.median(arr))
		std = float(arr.std(ddof=1)) if n > 1 else 0.0
		q25 = float(np.percentile(arr, 25))
		q75 = float(np.percentile(arr, 75))
		return (
			f"{cat_label:<35} {method:<12} {n:>6d} "
			f"{mean:>10.3f} {median:>10.3f} {std:>10.3f} "
			f"{q25:>10.3f} {q75:>10.3f}"
		)

	for cat in display_cats:
		lines.append(_fmt_row(cat, "NemaSize", cat_nesg.get(cat, [])))

	stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"[ok] Saved: {stats_path}")

	return out_path


def run_publication_pipeline(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> list[Path]:
	"""Generate paper-ready scatter figures (4 axes + pooled).

	Output goes under `<output_root>/publication_figures/`. The original 16
	source classes are reduced via 4 orthogonal axes (curvature, size,
	completeness, occlusion); each axis figure has one panel with marker
	shape encoding the parent group and color encoding the method
	(NemaSize=red, CellProfiler=blue). Only worms whose human_file appears in
	both methods (within the same parent group) are included. All five
	figures share the same x/y axis limits for direct visual comparison.
	"""
	pub_root = output_root / PUBLICATION_OUTPUT_DIRNAME
	pub_root.mkdir(parents=True, exist_ok=True)

	# Sanity check: warn about source classes on disk that aren't covered by
	# any axis (so silently-dropped data is visible).
	known_sources = {
		src
		for axis_map in PUBLICATION_AXES.values()
		for sources in axis_map.values()
		for src in sources
	}
	on_disk = {d.name for d in nesg_dirs} | {d.name for d in cp_dirs}
	unknown = sorted(on_disk - known_sources)
	if unknown:
		print(f"[warn] publication: source classes not assigned to any axis: {unknown}")

	# Pre-compute all data so we can derive a single shared (lo, hi).
	axis_data: dict[str, list[tuple[str, pd.Series, pd.Series, pd.Series, pd.Series]]] = {}
	for axis_name in PUBLICATION_AXES:
		axis_data[axis_name] = _collect_axis_groups_data(axis_name, nesg_dirs, cp_dirs)
	pooled_data = _collect_pooled_data(nesg_dirs, cp_dirs)

	all_vals: list[float] = []
	for groups in axis_data.values():
		for _, ng, nc, cg, cc in groups:
			all_vals.extend(ng.tolist())
			all_vals.extend(nc.tolist())
			all_vals.extend(cg.tolist())
			all_vals.extend(cc.tolist())
	for vals in pooled_data:
		all_vals.extend(vals)

	if all_vals:
		shared_limits: tuple[float, float] | None = (
			float(min(all_vals)) * 0.9,
			float(max(all_vals)) * 1.1,
		)
		print(f"[publication] Shared axis limits: ({shared_limits[0]:.2f}, {shared_limits[1]:.2f})")
	else:
		shared_limits = None

	saved: list[Path] = []
	for axis_name in PUBLICATION_AXES:
		print(f"\n[publication] Building axis '{axis_name}' scatter...")
		out = create_publication_axis_scatter(
			axis_name, nesg_dirs, cp_dirs, pub_root, xy_limits=shared_limits,
		)
		if out is not None:
			saved.append(out)

	print("\n[publication] Building pooled all-categories scatter...")
	pooled = create_publication_pooled_scatter(
		nesg_dirs, cp_dirs, pub_root, xy_limits=shared_limits,
	)
	if pooled is not None:
		saved.append(pooled)

	print("\n[publication] Building horizontal difference boxplot...")
	box_out = create_publication_difference_boxplot(
		nesg_dirs, cp_dirs, pub_root,
	)
	if box_out is not None:
		saved.append(box_out)

	print("\n[publication] Building NS-only width boxplot...")
	width_box_out = create_publication_width_boxplot(nesg_dirs, cp_dirs, pub_root)
	if width_box_out is not None:
		saved.append(width_box_out)

	return saved


def _print_discoverability_table(label: str, disc: dict[str, dict[str, int]]) -> None:
	print(f"\n[discoverability] {label}:")
	print(f"  {'Category':<45} {'GT worms':>10} {'Matched':>10} {'% matched':>12}")
	print(f"  {'-'*45} {'-'*10} {'-'*10} {'-'*12}")
	for cat in sorted(disc.keys()):
		n_gt = disc[cat]["num_gt"]
		n_matched = disc[cat]["num_matched"]
		pct = (n_matched / n_gt * 100.0) if n_gt > 0 else 0.0
		print(f"  {cat:<45} {n_gt:>10} {n_matched:>10} {pct:>11.1f}%")


def main() -> None:
	active_roots: list[tuple[Path, str]] = []
	if GT_VS_NESG_ROOT is not None:
		active_roots.append((GT_VS_NESG_ROOT, "NemaSize"))
	if GT_VS_CP_ROOT is not None:
		active_roots.append((GT_VS_CP_ROOT, "CellProfiler"))

	if not active_roots:
		raise RuntimeError("Both GT_VS_NESG_ROOT and GT_VS_CP_ROOT are None. Set at least one.")

	if VALIDATION_ONLY:
		print(f"\n=== VALIDATION-ONLY MODE ===")
		print(f"[validation] Filtering CSVs to images in: {VALIDATION_IMAGES_DIR}")
		# Prime the cache early so the count is logged once up front.
		get_validation_image_stems()
		comparison_output_root = COMPARISON_OUTPUT_ROOT / VALIDATION_SUBDIR_NAME
	else:
		comparison_output_root = COMPARISON_OUTPUT_ROOT

	saved_files = []
	resolved_dirs: list[list[Path]] = []
	disc_results: list[tuple[str, dict[str, dict[str, int]]]] = []
	delta_r_ranking: list[tuple[str, float, float, float]] = []

	for root, label in active_roots:
		print(f"\n=== Processing {label} ({root}) ===")
		category_dirs = get_category_folders(root)
		resolved_dirs.append(category_dirs)
		if not category_dirs:
			print(f"[skip] No category folders found in: {root}")
			disc_results.append((label, {}))
			continue

		per_method_output_root = root / VALIDATION_SUBDIR_NAME if VALIDATION_ONLY else root
		per_method_output_root.mkdir(parents=True, exist_ok=True)
		unified_plot = create_unified_summary_plot(category_dirs, per_method_output_root)
		if unified_plot is not None:
			saved_files.append(unified_plot)

		if VALIDATION_ONLY:
			print(f"[validation] {label}: skipping discoverability collection")
		else:
			disc = collect_discoverability(category_dirs)
			disc_results.append((label, disc))
			_print_discoverability_table(label, disc)

	if len(active_roots) == 2 and all(resolved_dirs):
		comparison_output_root.mkdir(parents=True, exist_ok=True)
		comp_plot = create_comparison_plot(resolved_dirs[0], resolved_dirs[1], comparison_output_root)
		if comp_plot is not None:
			saved_files.append(comp_plot)

		scatter_plots, delta_r_ranking = create_cross_method_scatter_plots(
			resolved_dirs[0], resolved_dirs[1], comparison_output_root,
		)
		saved_files.extend(scatter_plots)

		with plt.rc_context(LARGE_FONT_RC):
			box_plots = create_cross_method_difference_boxplots(
				resolved_dirs[0], resolved_dirs[1], comparison_output_root,
			)
		saved_files.extend(box_plots)

		anova_path = create_anova_table(
			resolved_dirs[0], resolved_dirs[1], comparison_output_root,
		)
		if anova_path is not None:
			saved_files.append(anova_path)

		pub_files = run_publication_pipeline(
			resolved_dirs[0], resolved_dirs[1], comparison_output_root,
		)
		saved_files.extend(pub_files)

	print(f"\nCompleted. Generated {len(saved_files)} plot(s).")

	if delta_r_ranking:
		print("\n[delta_r] Categories ranked by delta_r = r_NemaSize - r_CellProfiler (largest to smallest):")
		for cat, delta_r, nesg_r, cp_r in sorted(delta_r_ranking, key=lambda x: x[1], reverse=True):
			print(f"  {cat:<35} delta_r={delta_r:+.4f}  (r_NemaSize={nesg_r:.4f}, r_CellProfiler={cp_r:.4f})")
	else:
		print("\n[delta_r] No categories had >=2 valid points in both methods; ranking not available")


if __name__ == "__main__":
	main()
