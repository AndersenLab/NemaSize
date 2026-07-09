"""Standalone reproduction script for three NemaSize publication figures.

This is a trimmed-down mirror of ``perform_stats.py`` that keeps only the
functions/utilities required to regenerate:

  - Fig_4A.png  (pooled GT-vs-computer length scatter)
  - Fig_4B.png  (length percentage-error box plot)
  - Fig_S1.png  (width percentage-error box plot, NemaSize only)

It is meant to run standalone from this folder using the data bundled under
``data/`` (no external Dropbox paths required):

    figure_replication/
        generate_figures.py   <- this file
        data/
            GT_vs_NeSg/<category>/worm_matched_measurements.csv
            GT_vs_CellProfiler/<category>/worm_matched_measurements.csv
            validation_stems.txt
        output/                <- figures are written here

Only worms whose source image is listed in ``validation_stems.txt`` are
included, matching how the published figures were generated (validation
split only).
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Paths (all relative to this script, so the folder is standalone) -------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data"
GT_VS_NESG_ROOT = DATA_ROOT / "GT_vs_NeSg"
GT_VS_CP_ROOT = DATA_ROOT / "GT_vs_CellProfiler"
OUTPUT_ROOT = SCRIPT_DIR / "output"

CSV_NAME = "worm_matched_measurements.csv"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# The published figures only include worms from the validation split.
VALIDATION_ONLY = True
VALIDATION_STEMS_FILE = DATA_ROOT / "validation_stems.txt"

# Merge low-sample classes into stronger destination classes (used by the
# two box plots; the scatter plot buckets directly from source classes).
CLASS_MERGE_MAP: dict[str, str] = {
	"overlapping_curly_tiny_worm": "curly_tiny_worm",
	"overlapping_tiny_worm": "tiny_worm",
	"self-overlapping_tiny_worm": "tiny_worm",
}

# Further merge low-n categories for the box plots (defect type takes
# precedence over shape modifier; tiny_bent_worm rolls up to bent_worm).
BOXPLOT_MERGE_MAP: dict[str, str] = {
	"overlapping_incomplete_worm": "overlapping_worm",
	"overlapping_curly_worm": "overlapping_worm",
	"incomplete_curly_worm": "incomplete_worm",
	"incomplete_tiny_worm": "incomplete_worm",
	"tiny_bent_worm": "bent_worm",
}

# Shape-encoded buckets for the pooled scatter plot. Priority order matters:
# a source class listed in multiple buckets resolves to the first match.
PUBLICATION_POOLED_BUCKETS: list[tuple[str, str, list[str]]] = [
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

# Method colors and per-bucket color variants (kept in the red/blue theme).
_PUBLICATION_METHOD_COLORS = {
	"NemaSize": "#E45756",
	"CellProfiler": "#4C78A8",
}
_PUBLICATION_BUCKET_COLORS: dict[str, dict[str, str]] = {
	"NemaSize": {
		"Overlapping":           "#7A0E0E",
		"Bent or curly":         "#E45756",
		"Straight and isolated": "#F4A6A1",
	},
	"CellProfiler": {
		"Overlapping":           "#0B3D6B",
		"Bent or curly":         "#4C78A8",
		"Straight and isolated": "#9EC4E5",
	},
}

# Font sizes / figure sizes / marker sizes for the publication figures.
PUB_AXIS_FONTSIZE:   float = 12
PUB_LEGEND_FONTSIZE: float = 8
SCATTER_FIG_WIDTH:  float = 3.7
SCATTER_FIG_HEIGHT: float = 3.7
BOXPLOT_FIG_WIDTH:  float = 3.7
BOXPLOT_FIG_HEIGHT: float = 3.7
SCATTER_MARKER_SIZE_CP: float = 22.0 * 0.7
SCATTER_MARKER_SIZE_NS: float = 29.0
BOXPLOT_MARKER_SIZE:    float = 12.0 * 0.7
SCATTER_LEGEND_LOC:  str = "upper left"
SCATTER_LEGEND_BBOX: tuple[float, float] = (-0.01, 1.05)


# --- Small shared helpers ----------------------------------------------------

def get_category_folders(root: Path) -> list[Path]:
	if not root.exists() or not root.is_dir():
		raise FileNotFoundError(f"Root folder not found: {root}")
	return sorted(path for path in root.iterdir() if path.is_dir())


def percentage_diff(delta: pd.Series, ground_truth: pd.Series) -> pd.Series:
	valid = ground_truth != 0
	if not valid.any():
		return pd.Series(dtype=float)
	pct = (delta[valid] / ground_truth[valid]) * 100.0
	pct = pd.to_numeric(pct, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
	return pct


def strip_image_extension(name: str) -> str:
	name = name.strip()
	lower_name = name.lower()
	for ext in IMAGE_EXTENSIONS:
		if lower_name.endswith(ext):
			return name[: -len(ext)]
	return name


def get_merged_category_name(category_name: str) -> str:
	return CLASS_MERGE_MAP.get(category_name, category_name)


# --- Validation-only filtering -----------------------------------------------

_ROI_SUFFIX_RE = re.compile(r"_roi_\d+$")
_VALIDATION_STEMS_CACHE: set[str] | None = None


def extract_image_stem_from_roi(human_file: str) -> str:
	"""Recover the source image stem from a `<imagestem>_roi_<N>.<ext>` filename."""
	stem = strip_image_extension(Path(str(human_file)).name)
	return _ROI_SUFFIX_RE.sub("", stem)


def get_validation_image_stems() -> set[str]:
	"""Return cached set of validation image stems from the bundled manifest."""
	global _VALIDATION_STEMS_CACHE
	if _VALIDATION_STEMS_CACHE is not None:
		return _VALIDATION_STEMS_CACHE

	if not VALIDATION_STEMS_FILE.is_file():
		raise FileNotFoundError(f"Validation stems manifest not found: {VALIDATION_STEMS_FILE}")
	stems = {line.strip() for line in VALIDATION_STEMS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()}
	print(f"[validation] Loaded {len(stems)} validation image stems from {VALIDATION_STEMS_FILE}")
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


def apply_row_filters(df: pd.DataFrame, category_name: str) -> pd.DataFrame:
	"""Apply the validation-only filter."""
	if VALIDATION_ONLY:
		df = apply_validation_filter(df, category_name=category_name)
	return df


# --- Data loading -------------------------------------------------------------

def _load_method_dataframes(dirs: list[Path]) -> dict[str, pd.DataFrame]:
	"""Read each category CSV, apply validation filtering, key by source class."""
	by_src: dict[str, list[pd.DataFrame]] = {}
	for d in dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		df = pd.read_csv(csv_path)
		df = apply_row_filters(df, category_name=src)
		by_src.setdefault(src, []).append(df)
	return {src: pd.concat(frames, ignore_index=True) for src, frames in by_src.items()}


def _extract_gt_and_computer_lengths(df: pd.DataFrame, method: str) -> tuple[pd.Series, pd.Series]:
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


def _common_worm_subframes(nesg_df: pd.DataFrame, cp_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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


def _build_pooled_bucket_lookup() -> dict[str, tuple[str, str]]:
	"""Invert PUBLICATION_POOLED_BUCKETS into {source_class: (label, marker)}."""
	lookup: dict[str, tuple[str, str]] = {}
	for label, marker, sources in PUBLICATION_POOLED_BUCKETS:
		for src in sources:
			lookup.setdefault(src, (label, marker))
	return lookup


def _collect_pooled_data_by_source(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
) -> dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
	"""Per-source-class common-worm length pairs: {src: (nesg_gt, nesg_comp, cp_gt, cp_comp)}."""
	nesg_by_src = _load_method_dataframes(nesg_dirs)
	cp_by_src = _load_method_dataframes(cp_dirs)

	result: dict[str, tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = {}
	for src in sorted(set(nesg_by_src) & set(cp_by_src)):
		nesg_sub, cp_sub = _common_worm_subframes(nesg_by_src[src], cp_by_src[src])
		if nesg_sub.empty or cp_sub.empty:
			continue
		ng, nc = _extract_gt_and_computer_lengths(nesg_sub, "NemaSize")
		cg, cc = _extract_gt_and_computer_lengths(cp_sub, "CellProfiler")
		result[src] = (ng, nc, cg, cc)
	return result


# --- Figure 1 (Fig_4A): pooled GT-vs-computer scatter -----------------------

def create_pooled_scatter(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> Path | None:
	"""Pooled GT-vs-computer length scatter, common worms only, bucketed by shape."""
	by_src = _collect_pooled_data_by_source(nesg_dirs, cp_dirs)
	if not by_src:
		print("[skip] Fig_4A: no valid common-worm data")
		return None

	bucket_lookup = _build_pooled_bucket_lookup()
	bucket_order = [label for label, _, _ in PUBLICATION_POOLED_BUCKETS]
	bucket_marker = {label: marker for label, marker, _ in PUBLICATION_POOLED_BUCKETS}
	per_bucket: dict[str, dict[str, list[float]]] = {
		label: {"n_gt": [], "n_comp": [], "c_gt": [], "c_comp": []} for label in bucket_order
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
		print(f"[warn] Fig_4A: source classes without a bucket: {sorted(unbucketed)}")

	all_n_gt = [v for b in per_bucket.values() for v in b["n_gt"]]
	all_n_comp = [v for b in per_bucket.values() for v in b["n_comp"]]
	all_c_gt = [v for b in per_bucket.values() for v in b["c_gt"]]
	all_c_comp = [v for b in per_bucket.values() for v in b["c_comp"]]
	if not all_n_gt and not all_c_gt:
		print("[skip] Fig_4A: no valid common-worm data after bucketing")
		return None

	all_vals = all_n_gt + all_n_comp + all_c_gt + all_c_comp
	lo = float(min(all_vals)) * 0.9
	hi = float(max(all_vals)) * 1.1

	fig, ax = plt.subplots(1, 1, figsize=(SCATTER_FIG_WIDTH, SCATTER_FIG_HEIGHT), constrained_layout=True)
	ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="_nolegend_")

	draw_order = list(reversed(bucket_order))
	bucket_handles: dict[str, dict[str, object]] = {}
	for label in draw_order:
		b = per_bucket[label]
		marker = bucket_marker[label]
		cp_color = _PUBLICATION_BUCKET_COLORS["CellProfiler"][label]
		ns_color = _PUBLICATION_BUCKET_COLORS["NemaSize"][label]
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

		def _fmt_label(method: str, r2: float) -> str:
			base = f"{label} - {method}"
			return f"{base} (r\u00b2={r2:.3f})" if np.isfinite(r2) else base

		cp_h = None
		ns_h = None
		if b["c_gt"]:
			cp_h = ax.scatter(
				b["c_gt"], b["c_comp"], s=SCATTER_MARKER_SIZE_CP, alpha=0.95,
				facecolors="none", edgecolors=cp_color, linewidths=1.2, marker=marker,
				label=_fmt_label("CP", r2_cp),
			)
		if b["n_gt"]:
			ns_h = ax.scatter(
				b["n_gt"], b["n_comp"], s=SCATTER_MARKER_SIZE_NS, alpha=0.95,
				color=ns_color, marker=marker, edgecolors="white", linewidths=0.5,
				label=_fmt_label("NS", r2_ns),
			)
		bucket_handles[label] = {"ns": ns_h, "cp": cp_h}

	ax.set_xlim(lo, hi)
	ax.set_ylim(lo, hi)
	ax.set_aspect("equal", adjustable="box")
	pub_ticks = np.arange(200, 1201, 200)
	ax.set_xticks(pub_ticks)
	ax.set_yticks(pub_ticks)
	ax.set_xlabel("Ground-truth animal length (\u00b5m)", fontfamily="Arial", fontsize=PUB_AXIS_FONTSIZE)
	ax.set_ylabel("Computer measured animal length (\u00b5m)", fontfamily="Arial", fontsize=PUB_AXIS_FONTSIZE)
	ax.grid(False)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(axis="both", direction="in", labelsize=PUB_AXIS_FONTSIZE, length=6.5, width=1.05)
	for lbl in ax.get_xticklabels() + ax.get_yticklabels():
		lbl.set_fontfamily("Arial")

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
	leg = ax.legend(
		ordered_handles, ordered_labels,
		loc=SCATTER_LEGEND_LOC, bbox_to_anchor=SCATTER_LEGEND_BBOX,
		fontsize=PUB_LEGEND_FONTSIZE, frameon=False, handletextpad=0,
		prop={"family": "Arial", "size": PUB_LEGEND_FONTSIZE},
	)
	for txt in leg.get_texts():
		txt.set_fontfamily("Arial")

	out_path = output_root / "Fig_4A.png"
	fig.savefig(out_path, dpi=300)
	fig.savefig(out_path.with_suffix(".svg"))
	plt.close(fig)
	print(f"[ok] Saved: {out_path}")
	return out_path


# --- Figure 2 (Fig_4B): length %-error box plot, NS vs CP -------------------

def _pretty_category_name(name: str) -> str:
	overrides = {
		"curly_tiny_worm": "Curly and tiny",
		"tiny_bent_worm": "Tiny and bent",
		"overlapping_worm": "Mutual-overlap",
		"self-overlapping_worm": "Self-overlap",
		"all_categories": "All categories",
	}
	if name in overrides:
		return overrides[name]
	words = [w for w in name.replace("_", " ").split() if w.lower() != "worm"]
	if not words:
		return name
	words[0] = words[0][:1].upper() + words[0][1:]
	return " ".join(words)


def _read_merged_by_category(dirs: list[Path]) -> dict[str, pd.DataFrame]:
	"""Read all category CSVs, apply filtering, and merge via CLASS_MERGE_MAP."""
	by_cat: dict[str, pd.DataFrame] = {}
	for d in dirs:
		csv_path = d / CSV_NAME
		if not csv_path.exists():
			continue
		src = d.name
		merged = get_merged_category_name(src)
		df = pd.read_csv(csv_path)
		df = apply_row_filters(df, category_name=src)
		by_cat[merged] = pd.concat([by_cat[merged], df], ignore_index=True) if merged in by_cat else df
	return by_cat


def create_difference_boxplot(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> Path | None:
	"""Horizontal box plot of length percentage-error per category (NS vs CP)."""
	nesg_by_cat = _read_merged_by_category(nesg_dirs)
	cp_by_cat = _read_merged_by_category(cp_dirs)

	common_categories = sorted(set(nesg_by_cat) & set(cp_by_cat))
	cat_nesg: dict[str, list[float]] = {}
	cat_cp: dict[str, list[float]] = {}
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
		if not n_pct.empty and not c_pct.empty:
			cat_nesg[cat] = n_pct.tolist()
			cat_cp[cat] = c_pct.tolist()

	for src, dst in BOXPLOT_MERGE_MAP.items():
		if src in cat_nesg:
			cat_nesg.setdefault(dst, []).extend(cat_nesg.pop(src))
			cat_cp.setdefault(dst, []).extend(cat_cp.pop(src))
			print(f"[Fig_4B] merged {src} -> {dst}")

	if not cat_nesg:
		print("[skip] Fig_4B: no valid common-worm data")
		return None

	print("[Fig_4B] per-category common-worm sample sizes:")
	total_n = 0
	for cat, vals in cat_nesg.items():
		print(f"  {cat:<35} n={len(vals)}")
		total_n += len(vals)
	print(f"  {'all_categories':<35} n={total_n}")

	display_cats = list(cat_nesg.keys()) + ["all_categories"]
	cat_nesg["all_categories"] = [v for c in cat_nesg for v in cat_nesg[c]]
	cat_cp["all_categories"] = [v for c in cat_cp for v in cat_cp[c]]

	ns_color = _PUBLICATION_METHOD_COLORS["NemaSize"]
	cp_color = _PUBLICATION_METHOD_COLORS["CellProfiler"]
	n_rows = len(display_cats)
	fig_h = BOXPLOT_FIG_HEIGHT if BOXPLOT_FIG_HEIGHT is not None else max(5.0, 0.6 * n_rows + 1.5)
	fig, ax = plt.subplots(1, 1, figsize=(BOXPLOT_FIG_WIDTH, fig_h), constrained_layout=True)

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
		data_ns, positions=positions_ns, widths=0.3, vert=False, patch_artist=True,
		showfliers=False, medianprops=dict(color="black", linewidth=1.5),
	)
	bp_c = ax.boxplot(
		data_cp, positions=positions_cp, widths=0.3, vert=False, patch_artist=True,
		showfliers=False, medianprops=dict(color="black", linewidth=1.5),
	)
	for patch in bp_n["boxes"]:
		patch.set_facecolor(ns_color)
		patch.set_alpha(0.5)
	for patch in bp_c["boxes"]:
		patch.set_facecolor(cp_color)
		patch.set_alpha(0.5)

	rng = np.random.default_rng(42)
	for i, cat in enumerate(display_cats):
		y = (n_rows - 1 - i)
		n_vals = np.array(cat_nesg.get(cat, []))
		c_vals = np.array(cat_cp.get(cat, []))
		if n_vals.size:
			ax.scatter(
				n_vals, y + offset + rng.uniform(-0.06, 0.06, size=n_vals.size),
				s=BOXPLOT_MARKER_SIZE, alpha=0.7, color=ns_color, edgecolors="white",
				linewidths=0.3, zorder=3,
			)
		if c_vals.size:
			ax.scatter(
				c_vals, y - offset + rng.uniform(-0.06, 0.06, size=c_vals.size),
				s=BOXPLOT_MARKER_SIZE, alpha=0.7, color=cp_color, edgecolors="white",
				linewidths=0.3, zorder=3,
			)

	ax.axvline(0.0, color="black", linewidth=1)
	ax.set_yticks([n_rows - 1 - i for i in range(n_rows)])
	ax.set_yticklabels([_pretty_category_name(c) for c in display_cats])
	ax.set_xticks(np.arange(-75, 76, 25))
	ax.set_xlabel("Percentage error of length (%)", fontfamily="Arial", fontsize=PUB_AXIS_FONTSIZE)
	ax.set_ylabel("")
	ax.grid(False)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(axis="both", direction="in", labelsize=PUB_AXIS_FONTSIZE, length=6.5, width=1.05)
	for lbl in ax.get_xticklabels() + ax.get_yticklabels():
		lbl.set_fontfamily("Arial")

	from matplotlib.patches import Patch
	leg = ax.legend(
		handles=[Patch(facecolor=ns_color, alpha=0.5, label="NS"), Patch(facecolor=cp_color, alpha=0.5, label="CP")],
		loc="upper right", bbox_to_anchor=(1.08, 1.0),
		fontsize=PUB_LEGEND_FONTSIZE, frameon=False,
		prop={"family": "Arial", "size": PUB_LEGEND_FONTSIZE},
	)
	for txt in leg.get_texts():
		txt.set_fontfamily("Arial")

	out_path = output_root / "Fig_4B.png"
	fig.savefig(out_path, dpi=300)
	fig.savefig(out_path.with_suffix(".svg"))
	plt.close(fig)
	print(f"[ok] Saved: {out_path}")

	_write_boxplot_stats(
		output_root / "Fig_4B_stats.txt",
		title="Per-box statistics for Fig_4B.png",
		value_desc="Values are percentage error of worm length (%): (computer - human) / human * 100",
		display_cats=display_cats,
		series_by_method={"NemaSize": cat_nesg, "CellProfiler": cat_cp},
	)
	return out_path


# --- Figure 3 (Fig_S1): width %-error box plot, NemaSize only ---------------

def create_width_difference_boxplot(
	nesg_dirs: list[Path],
	cp_dirs: list[Path],
	output_root: Path,
) -> Path | None:
	"""NemaSize-only horizontal box plot of width percentage-error per category.

	CellProfiler produces no width measurements. To keep the worm population
	consistent with the length box plot, rows are still restricted to
	NemaSize worms whose human_file also appears in the CellProfiler CSV for
	the same category.
	"""
	nesg_by_cat = _read_merged_by_category(nesg_dirs)
	cp_by_cat = _read_merged_by_category(cp_dirs)

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

	for src, dst in BOXPLOT_MERGE_MAP.items():
		if src in cat_nesg:
			cat_nesg.setdefault(dst, []).extend(cat_nesg.pop(src))
			print(f"[Fig_S1] merged {src} -> {dst}")

	if not cat_nesg:
		print("[skip] Fig_S1: no valid width data")
		return None

	print("[Fig_S1] per-category sample sizes:")
	total_n = 0
	for cat, vals in cat_nesg.items():
		print(f"  {cat:<35} n={len(vals)}")
		total_n += len(vals)
	print(f"  {'all_categories':<35} n={total_n}")

	display_cats = list(cat_nesg.keys()) + ["all_categories"]
	cat_nesg["all_categories"] = [v for c in cat_nesg for v in cat_nesg[c]]

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
		data, positions=positions, widths=0.5, vert=False, patch_artist=True,
		showfliers=False, medianprops=dict(color="black", linewidth=1.5),
	)
	for patch in bp["boxes"]:
		patch.set_facecolor(ns_color)
		patch.set_alpha(0.5)

	rng = np.random.default_rng(42)
	for i, cat in enumerate(display_cats):
		y = (n_rows - 1 - i)
		vals = np.array(cat_nesg.get(cat, []))
		if vals.size:
			ax.scatter(
				vals, y + rng.uniform(-0.12, 0.12, size=vals.size),
				s=BOXPLOT_MARKER_SIZE, alpha=0.7, color=ns_color, edgecolors="white",
				linewidths=0.3, zorder=3,
			)

	ax.axvline(0.0, color="black", linewidth=1)
	ax.set_yticks([n_rows - 1 - i for i in range(n_rows)])
	ax.set_yticklabels([_pretty_category_name(c) for c in display_cats])
	ax.set_xlabel("Percentage error of width (%)", fontfamily="Arial", fontsize=PUB_AXIS_FONTSIZE)
	ax.set_ylabel("")
	ax.grid(False)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(axis="both", direction="in", labelsize=PUB_AXIS_FONTSIZE, length=6.5, width=1.05)
	for lbl in ax.get_xticklabels() + ax.get_yticklabels():
		lbl.set_fontfamily("Arial")

	out_path = output_root / "Fig_S1.png"
	fig.savefig(out_path, dpi=300)
	fig.savefig(out_path.with_suffix(".svg"))
	plt.close(fig)
	print(f"[ok] Saved: {out_path}")

	_write_boxplot_stats(
		output_root / "Fig_S1_stats.txt",
		title="Per-box statistics for Fig_S1.png",
		value_desc="Values are percentage error of worm width (%): (computer - human) / human * 100",
		display_cats=display_cats,
		series_by_method={"NemaSize": cat_nesg},
	)
	return out_path


def _write_boxplot_stats(
	stats_path: Path,
	title: str,
	value_desc: str,
	display_cats: list[str],
	series_by_method: dict[str, dict[str, list[float]]],
) -> None:
	header = (
		f"{'category':<35} {'method':<12} {'n':>6} "
		f"{'mean':>10} {'median':>10} {'std':>10} {'q25':>10} {'q75':>10}"
	)
	lines = [title, value_desc, "", header, "-" * len(header)]

	def _fmt_row(cat_label: str, method: str, values: list[float]) -> str:
		arr = np.asarray(values, dtype=float)
		arr = arr[np.isfinite(arr)]
		n = arr.size
		if n == 0:
			return f"{cat_label:<35} {method:<12} {n:>6} {'nan':>10} {'nan':>10} {'nan':>10} {'nan':>10} {'nan':>10}"
		mean = float(arr.mean())
		median = float(np.median(arr))
		std = float(arr.std(ddof=1)) if n > 1 else 0.0
		q25 = float(np.percentile(arr, 25))
		q75 = float(np.percentile(arr, 75))
		return (
			f"{cat_label:<35} {method:<12} {n:>6d} "
			f"{mean:>10.3f} {median:>10.3f} {std:>10.3f} {q25:>10.3f} {q75:>10.3f}"
		)

	for cat in display_cats:
		for method, series in series_by_method.items():
			lines.append(_fmt_row(cat, method, series.get(cat, [])))

	stats_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print(f"[ok] Saved: {stats_path}")


def main() -> None:
	if VALIDATION_ONLY:
		print("=== VALIDATION-ONLY MODE ===")
		get_validation_image_stems()

	nesg_dirs = get_category_folders(GT_VS_NESG_ROOT)
	cp_dirs = get_category_folders(GT_VS_CP_ROOT)
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

	saved = []
	print("\n[1/3] Building Fig_4A.png ...")
	out = create_pooled_scatter(nesg_dirs, cp_dirs, OUTPUT_ROOT)
	if out is not None:
		saved.append(out)

	print("\n[2/3] Building Fig_4B.png ...")
	out = create_difference_boxplot(nesg_dirs, cp_dirs, OUTPUT_ROOT)
	if out is not None:
		saved.append(out)

	print("\n[3/3] Building Fig_S1.png ...")
	out = create_width_difference_boxplot(nesg_dirs, cp_dirs, OUTPUT_ROOT)
	if out is not None:
		saved.append(out)

	print(f"\nCompleted. Generated {len(saved)} figure(s) in {OUTPUT_ROOT}")


if __name__ == "__main__":
	main()
