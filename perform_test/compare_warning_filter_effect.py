"""Compare NemaSize-vs-human measurement summaries with and without the
topology-warning filter applied during matching.

Reads `worm_matched_measurements.csv` from each category folder under two
roots (unfiltered vs. warning-filtered) and produces one box-plot figure per
metric (length um/%, width um/%) with three groups per category:
unfiltered, warning-filtered, and warned-only (unfiltered minus filtered).

Self-contained: does not import from `perform_stats`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Configuration (edit paths here) -----------------------------------------
UNFILTERED_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\GT_vs_NeSg_unwarned"
)
FILTERED_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\GT_vs_NeSg_warned"
)
OUTPUT_ROOT = Path(
	r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
	r"\Perform_test\Topology_warning"
)
OUTPUT_FILENAME_LENGTH_UM = "warning_filter_effect_length_um.png"
OUTPUT_FILENAME_LENGTH_PCT = "warning_filter_effect_length_pct.png"
OUTPUT_FILENAME_WIDTH_UM = "warning_filter_effect_width_um.png"
OUTPUT_FILENAME_WIDTH_PCT = "warning_filter_effect_width_pct.png"

CSV_NAME = "worm_matched_measurements.csv"
# Columns that uniquely identify a matched worm row across the two CSVs.
ROW_KEY_COLUMNS = ("human_file", "computer_file")

UNFILTERED_LABEL = "Unfiltered"
FILTERED_LABEL = "Warning-filtered"
WARNED_LABEL = "Warned-only"
UNFILTERED_COLOR = "#4C78A8"
FILTERED_COLOR = "#F58518"
WARNED_COLOR = "#E45756"

# Jitter / overlay style for scatter points.
POINT_ALPHA = 0.45
POINT_SIZE = 9
JITTER_FRAC = 0.55  # fraction of half-bar-width used as scatter jitter


# --- Aggregation -------------------------------------------------------------
def _percentage_diff(delta: pd.Series, ground_truth: pd.Series) -> list[float]:
	valid = ground_truth != 0
	if not valid.any():
		return []
	pct = (delta[valid] / ground_truth[valid]) * 100.0
	pct = pd.to_numeric(pct, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
	return pct.to_list()


def _metrics_from_df(df: pd.DataFrame) -> dict[str, list[float]]:
	out = {"length_diff": [], "width_diff": [], "length_pct": [], "width_pct": []}
	if "length_diff_um" in df.columns:
		out["length_diff"] = pd.to_numeric(df["length_diff_um"], errors="coerce").dropna().to_list()
	if "width_diff_um" in df.columns:
		out["width_diff"] = pd.to_numeric(df["width_diff_um"], errors="coerce").dropna().to_list()
	if {"length_diff_um", "length_um_human"}.issubset(df.columns):
		out["length_pct"] = _percentage_diff(
			pd.to_numeric(df["length_diff_um"], errors="coerce"),
			pd.to_numeric(df["length_um_human"], errors="coerce"),
		)
	if {"width_diff_um", "width_um_human"}.issubset(df.columns):
		out["width_pct"] = _percentage_diff(
			pd.to_numeric(df["width_diff_um"], errors="coerce"),
			pd.to_numeric(df["width_um_human"], errors="coerce"),
		)
	return out


def aggregate_all(
	unfiltered_root: Path,
	filtered_root: Path,
) -> tuple[
	dict[str, dict[str, list[float]]],
	dict[str, dict[str, list[float]]],
	dict[str, dict[str, list[float]]],
]:
	"""Aggregate per-category metric lists for unfiltered, filtered, and warned-only.

	Warned classification is row-identity-based: an unfiltered row is "warned"
	iff its ``(human_file, computer_file)`` key is not present in the matching
	category's filtered CSV. Each metric for a row is therefore consistent
	(same `n` across length/width/% within a group).

	Returns (stats_uf, stats_fi, stats_wn), each in the shape
	``{category: {metric: [values]}, "all_categories": {...}}``.
	"""
	if not unfiltered_root.is_dir():
		raise FileNotFoundError(f"Root folder not found: {unfiltered_root}")
	if not filtered_root.is_dir():
		raise FileNotFoundError(f"Root folder not found: {filtered_root}")

	stats_uf: dict[str, dict[str, list[float]]] = {}
	stats_fi: dict[str, dict[str, list[float]]] = {}
	stats_wn: dict[str, dict[str, list[float]]] = {}

	pooled_uf = {"length_diff": [], "width_diff": [], "length_pct": [], "width_pct": []}
	pooled_fi = {"length_diff": [], "width_diff": [], "length_pct": [], "width_pct": []}
	pooled_wn = {"length_diff": [], "width_diff": [], "length_pct": [], "width_pct": []}

	fi_by_name = {p.name: p for p in filtered_root.iterdir() if p.is_dir()}

	for uf_sub in sorted(p for p in unfiltered_root.iterdir() if p.is_dir()):
		name = uf_sub.name
		uf_csv = uf_sub / CSV_NAME
		if not uf_csv.exists():
			continue
		df_uf = pd.read_csv(uf_csv)
		if not set(ROW_KEY_COLUMNS).issubset(df_uf.columns):
			print(f"[skip] {name}: unfiltered CSV missing row-key columns {ROW_KEY_COLUMNS}")
			continue

		fi_sub = fi_by_name.get(name)
		if fi_sub is None or not (fi_sub / CSV_NAME).exists():
			print(f"[warn] {name}: no matching filtered CSV; treating all rows as warned")
			df_fi = df_uf.iloc[0:0].copy()
		else:
			df_fi = pd.read_csv(fi_sub / CSV_NAME)

		# Row-identity classification.
		uf_keys = list(zip(df_uf[ROW_KEY_COLUMNS[0]].astype(str), df_uf[ROW_KEY_COLUMNS[1]].astype(str)))
		if set(ROW_KEY_COLUMNS).issubset(df_fi.columns) and not df_fi.empty:
			fi_keys = set(zip(df_fi[ROW_KEY_COLUMNS[0]].astype(str), df_fi[ROW_KEY_COLUMNS[1]].astype(str)))
		else:
			fi_keys = set()
		warned_mask = pd.Series([k not in fi_keys for k in uf_keys], index=df_uf.index)
		df_wn = df_uf[warned_mask].copy()

		uf_bucket = _metrics_from_df(df_uf)
		fi_bucket = _metrics_from_df(df_fi)
		wn_bucket = _metrics_from_df(df_wn)

		print(
			f"[info] {name}: unfiltered={len(df_uf)}, filtered={len(df_fi)}, "
			f"warned={int(warned_mask.sum())}"
		)

		if uf_bucket["length_diff"]:
			stats_uf[name] = uf_bucket
			for k in pooled_uf:
				pooled_uf[k].extend(uf_bucket[k])
		if fi_bucket["length_diff"]:
			stats_fi[name] = fi_bucket
			for k in pooled_fi:
				pooled_fi[k].extend(fi_bucket[k])
		if wn_bucket["length_diff"]:
			stats_wn[name] = wn_bucket
			for k in pooled_wn:
				pooled_wn[k].extend(wn_bucket[k])

	if pooled_uf["length_diff"]:
		stats_uf["all_categories"] = pooled_uf
	if pooled_fi["length_diff"]:
		stats_fi["all_categories"] = pooled_fi
	if pooled_wn["length_diff"]:
		stats_wn["all_categories"] = pooled_wn

	return stats_uf, stats_fi, stats_wn


def union_category_order(
	primary: dict[str, dict[str, list[float]]],
	secondary: dict[str, dict[str, list[float]]],
) -> list[str]:
	"""Order: primary keys first; extras from secondary appended; 'all_categories' last."""
	primary_keys = [k for k in primary.keys() if k != "all_categories"]
	extra = [k for k in secondary.keys() if k not in primary and k != "all_categories"]
	ordered = primary_keys + extra
	if "all_categories" in primary or "all_categories" in secondary:
		ordered.append("all_categories")
	return ordered


# --- Plotting ----------------------------------------------------------------
def _clean(values: list[float]) -> np.ndarray:
	if not values:
		return np.empty(0, dtype=float)
	arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
	return arr


def _draw_box_and_points(
	ax: plt.Axes,
	position: float,
	values: np.ndarray,
	color: str,
	bar_width: float,
	rng: np.random.Generator,
	label: str | None = None,
) -> None:
	# Box itself. positions/width control horizontal placement.
	if values.size:
		bp = ax.boxplot(
			[values],
			positions=[position],
			widths=bar_width * 0.85,
			patch_artist=True,
			showfliers=False,
			manage_ticks=False,
		)
		for patch in bp["boxes"]:
			patch.set_facecolor(color)
			patch.set_edgecolor("black")
			patch.set_alpha(0.55)
		for whisker in bp["whiskers"]:
			whisker.set_color("black")
		for cap in bp["caps"]:
			cap.set_color("black")
		for median in bp["medians"]:
			median.set_color("black")
			median.set_linewidth(1.5)

		jitter_range = bar_width * JITTER_FRAC
		jitter = rng.uniform(-jitter_range / 2.0, jitter_range / 2.0, size=values.size)
		ax.scatter(
			np.full_like(values, position) + jitter,
			values,
			s=POINT_SIZE,
			color=color,
			alpha=POINT_ALPHA,
			edgecolors="none",
			label=label,
		)
	elif label is not None:
		# Empty proxy so legend still picks the label up.
		ax.scatter([], [], s=POINT_SIZE, color=color, alpha=POINT_ALPHA, label=label)


def _annotate_paired_sample_sizes(
	ax: plt.Axes,
	x: np.ndarray,
	offsets: list[float],
	counts: list[list[int]],
) -> None:
	"""Annotate each group's sample sizes above the axis.

	`offsets` gives the horizontal offset (relative to category center) for
	each series; `counts[s]` is the list of per-category sample sizes for
	series s.
	"""
	for xi, *per_series in zip(x, *counts):
		for off, n in zip(offsets, per_series):
			ax.text(
				xi + off,
				0.98,
				f"n={n}",
				transform=ax.get_xaxis_transform(),
				ha="center",
				va="top",
				fontsize=7,
				rotation=90,
				color="#333333",
			)


def make_paired_box_panel(
	ax: plt.Axes,
	x: np.ndarray,
	series: list[tuple[str, str, list[np.ndarray]]],
	title: str,
	ylabel: str,
	bar_width: float = 0.27,
	seed: int = 0,
) -> None:
	"""Draw side-by-side boxes per category.

	`series` is a list of (label, color, values_per_category). All entries
	must have the same length as `x`.
	"""
	rng = np.random.default_rng(seed)
	n_series = len(series)
	# Center the series symmetrically around each category x position.
	offsets = [(i - (n_series - 1) / 2.0) * bar_width for i in range(n_series)]

	for s_idx, (label, color, values_per_cat) in enumerate(series):
		first = True
		for xi, vals in zip(x, values_per_cat):
			_draw_box_and_points(
				ax,
				position=xi + offsets[s_idx],
				values=vals,
				color=color,
				bar_width=bar_width,
				rng=rng,
				label=label if first else None,
			)
			first = False

	ax.axhline(0.0, color="black", linewidth=1)
	ax.set_title(title)
	ax.set_ylabel(ylabel)
	ax.grid(True, axis="y", alpha=0.3)

	counts = [[int(v.size) for v in vals_per_cat] for _, _, vals_per_cat in series]
	_annotate_paired_sample_sizes(ax, x, offsets, counts)


def _series_for(
	stats: dict[str, dict[str, list[float]]],
	categories: list[str],
	field: str,
) -> list[np.ndarray]:
	return [_clean(stats.get(cat, {}).get(field, [])) for cat in categories]


def _save_metric_figure(
	categories: list[str],
	x: np.ndarray,
	values_uf: list[np.ndarray],
	values_fi: list[np.ndarray],
	values_wn: list[np.ndarray],
	title: str,
	ylabel: str,
	suptitle: str,
	output_path: Path,
	seed: int,
) -> Path:
	fig, ax = plt.subplots(1, 1, figsize=(12, 5.6), constrained_layout=True)

	make_paired_box_panel(
		ax,
		x,
		series=[
			(UNFILTERED_LABEL, UNFILTERED_COLOR, values_uf),
			(FILTERED_LABEL, FILTERED_COLOR, values_fi),
			(WARNED_LABEL, WARNED_COLOR, values_wn),
		],
		title=title,
		ylabel=ylabel,
		seed=seed,
	)

	ax.set_xticks(x)
	ax.set_xticklabels(categories, rotation=25, ha="right")
	if x.size:
		ax.set_xlim(x[0] - 0.7, x[-1] + 0.7)

	handles, labels = ax.get_legend_handles_labels()
	seen: set[str] = set()
	unique = [(h, lab) for h, lab in zip(handles, labels) if not (lab in seen or seen.add(lab))]
	if unique:
		ax.legend(
			[h for h, _ in unique],
			[lab for _, lab in unique],
			loc="best",
			frameon=True,
		)

	fig.suptitle(suptitle)

	fig.savefig(output_path, dpi=220)
	plt.close(fig)
	print(f"[ok] Saved: {output_path}")
	return output_path


def create_warning_filter_comparison_plot(
	unfiltered_root: Path,
	filtered_root: Path,
	output_root: Path,
) -> list[Path]:
	print(f"\n=== Aggregating (row-keyed by {ROW_KEY_COLUMNS}) ===")
	stats_uf, stats_fi, stats_wn = aggregate_all(unfiltered_root, filtered_root)
	if not stats_uf:
		print(f"[skip] No category data found in: {unfiltered_root}")
		return []
	if not stats_fi:
		print(f"[skip] No category data found in: {filtered_root}")
		return []

	categories = union_category_order(stats_uf, stats_fi)
	# Make sure warned-only categories are also represented.
	for k in stats_wn:
		if k not in categories and k != "all_categories":
			categories.insert(-1 if "all_categories" in categories else len(categories), k)
	if not categories:
		print("[skip] No usable category data found.")
		return []

	x = np.arange(len(categories), dtype=float)

	ld_uf = _series_for(stats_uf, categories, "length_diff")
	ld_fi = _series_for(stats_fi, categories, "length_diff")
	ld_wn = _series_for(stats_wn, categories, "length_diff")
	wd_uf = _series_for(stats_uf, categories, "width_diff")
	wd_fi = _series_for(stats_fi, categories, "width_diff")
	wd_wn = _series_for(stats_wn, categories, "width_diff")
	lp_uf = _series_for(stats_uf, categories, "length_pct")
	lp_fi = _series_for(stats_fi, categories, "length_pct")
	lp_wn = _series_for(stats_wn, categories, "length_pct")
	wp_uf = _series_for(stats_uf, categories, "width_pct")
	wp_fi = _series_for(stats_fi, categories, "width_pct")
	wp_wn = _series_for(stats_wn, categories, "width_pct")

	output_root.mkdir(parents=True, exist_ok=True)

	suptitle_base = "Effect of Topology-Warning Filter on NemaSize vs Human"

	length_um_path = _save_metric_figure(
		categories, x, ld_uf, ld_fi, ld_wn,
		title="Length Difference (Computer - Human)",
		ylabel="Difference (um)",
		suptitle=f"{suptitle_base} Length Measurements (um)",
		output_path=output_root / OUTPUT_FILENAME_LENGTH_UM,
		seed=1,
	)
	length_pct_path = _save_metric_figure(
		categories, x, lp_uf, lp_fi, lp_wn,
		title="Length Percentage Difference",
		ylabel="Difference (%)",
		suptitle=f"{suptitle_base} Length Measurements (%)",
		output_path=output_root / OUTPUT_FILENAME_LENGTH_PCT,
		seed=2,
	)
	width_um_path = _save_metric_figure(
		categories, x, wd_uf, wd_fi, wd_wn,
		title="Width Difference (Computer - Human)",
		ylabel="Difference (um)",
		suptitle=f"{suptitle_base} Width Measurements (um)",
		output_path=output_root / OUTPUT_FILENAME_WIDTH_UM,
		seed=3,
	)
	width_pct_path = _save_metric_figure(
		categories, x, wp_uf, wp_fi, wp_wn,
		title="Width Percentage Difference",
		ylabel="Difference (%)",
		suptitle=f"{suptitle_base} Width Measurements (%)",
		output_path=output_root / OUTPUT_FILENAME_WIDTH_PCT,
		seed=4,
	)
	return [length_um_path, length_pct_path, width_um_path, width_pct_path]


def main() -> None:
	create_warning_filter_comparison_plot(
		unfiltered_root=UNFILTERED_ROOT,
		filtered_root=FILTERED_ROOT,
		output_root=OUTPUT_ROOT,
	)


if __name__ == "__main__":
	main()
