"""Reproduce Fig S2: NemaSize percentage error split by species (validation split only).

  - Fig_S2A.png  length percentage error (cf. Fig 4B)
  - Fig_S2B.png  width percentage error (cf. Fig S1)

Worm data, validation filtering and styling are reused from
``../Figs_4_and_S1/generate_figures.py``. Species comes from
``data/image_species_lookup.csv`` (see misc/dataset_bookkeeping/build_species_lookup.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent / "Figs_4_and_S1"))
import generate_figures as gf  # noqa: E402

LOOKUP_CSV = SCRIPT_DIR / "data" / "image_species_lookup.csv"
OUTPUT_ROOT = SCRIPT_DIR / "output"

SPECIES_ORDER = ["C_elegans", "C_briggsae"]
SPECIES_LABEL = {"C_elegans": "C. elegans", "C_briggsae": "C. briggsae"}
FIG_WIDTH_IN = 3.5
FONT_SIZE_PT = 12

plt.rcParams.update({
	"font.size": FONT_SIZE_PT,
	"mathtext.fontset": "custom",
	"mathtext.rm": "Arial",
	"mathtext.it": "Arial:italic",
})


def _italic(species: str) -> str:
	return r"$\it{" + SPECIES_LABEL[species].replace(" ", r"\ ") + "}$"


# --- Data ---------------------------------------------------------------------

def _attach_species(df: pd.DataFrame) -> pd.DataFrame:
	"""Map each worm to a species via the image number in its human_file name."""
	lookup = pd.read_csv(LOOKUP_CSV)
	species_by_num = dict(zip(lookup["png_name"].str.extract(r"^(\d+)")[0].astype(int), lookup["species"]))
	nums = pd.to_numeric(df["human_file"].astype(str).str.extract(r"^(\d+)_png")[0])
	out = df.copy()
	out["species"] = nums.map(species_by_num)
	if out["species"].isna().any():
		bad = out.loc[out["species"].isna(), "human_file"].unique()[:5]
		raise ValueError(f"No species for some worms, e.g. {list(bad)}")
	return out


def load_common_worms() -> pd.DataFrame:
	"""NemaSize validation worms that CellProfiler also matched (same population as Fig 4/S1), with species."""
	nesg_by_src = gf._load_method_dataframes(gf.get_category_folders(gf.GT_VS_NESG_ROOT))
	cp_by_src = gf._load_method_dataframes(gf.get_category_folders(gf.GT_VS_CP_ROOT))

	ns_parts: list[pd.DataFrame] = []
	for src in sorted(set(nesg_by_src) & set(cp_by_src)):
		ns_sub, cp_sub = gf._common_worm_subframes(nesg_by_src[src], cp_by_src[src])
		if ns_sub.empty or cp_sub.empty:
			continue
		ns_parts.append(ns_sub)
	return _attach_species(pd.concat(ns_parts, ignore_index=True))


# --- Fig S2 -------------------------------------------------------------------

def _pct_error_by_species(df: pd.DataFrame, diff_col: str, human_col: str) -> dict[str, list[float]]:
	out = {
		sp: gf.percentage_diff(
			pd.to_numeric(df.loc[df["species"] == sp, diff_col], errors="coerce"),
			pd.to_numeric(df.loc[df["species"] == sp, human_col], errors="coerce"),
		).tolist()
		for sp in SPECIES_ORDER
	}
	return out


def _draw_boxes(ax, rows: list[str], series: list[tuple[dict[str, list[float]], str, float]], width: float) -> None:
	"""Vertical box + jittered points per row; series = (data_by_row, color, x_offset)."""
	rng = np.random.default_rng(42)
	for data, color, offset in series:
		positions = [i + offset for i in range(len(rows))]
		bp = ax.boxplot(
			[data[r] for r in rows], positions=positions, widths=width, vert=True, patch_artist=True,
			showfliers=False, medianprops=dict(color="black", linewidth=1.5),
		)
		for patch in bp["boxes"]:
			patch.set_facecolor(color)
			patch.set_alpha(0.5)
		for pos, r in zip(positions, rows):
			vals = np.asarray(data[r])
			if vals.size:
				ax.scatter(
					pos + rng.uniform(-0.06, 0.06, size=vals.size), vals, s=gf.BOXPLOT_MARKER_SIZE,
					alpha=0.7, color=color, edgecolors="white", linewidths=0.3, zorder=3,
				)

	ax.axhline(0.0, color="black", linewidth=1)
	ax.grid(False)
	for side in ("top", "right"):
		ax.spines[side].set_visible(False)
	ax.tick_params(axis="both", direction="in", labelsize=FONT_SIZE_PT, length=6.5, width=1.05)


def _shared_ylim(ns: pd.DataFrame) -> tuple[float, float]:
	"""Y range covering both length and width errors so S2A and S2B are comparable."""
	vals = [
		v
		for measure in ("length", "width")
		for series in _pct_error_by_species(ns, f"{measure}_diff_um", f"{measure}_um_human").values()
		for v in series
	]
	pad = 0.05 * (max(vals) - min(vals))
	return min(vals) - pad, max(vals) + pad


def create_species_boxplot(ns: pd.DataFrame, panel: str, measure: str, ylim: tuple[float, float]) -> Path:
	"""One box plot of NemaSize percentage error of `measure` (length|width) per species."""
	rows = SPECIES_ORDER
	data = _pct_error_by_species(ns, f"{measure}_diff_um", f"{measure}_um_human")
	print(f"[Fig_S2{panel}] {measure}: " + ", ".join(f"{r} n={len(data[r])}" for r in rows))

	fig, ax = plt.subplots(1, 1, figsize=(FIG_WIDTH_IN, gf.BOXPLOT_FIG_HEIGHT), constrained_layout=True)
	_draw_boxes(ax, rows, [(data, gf._PUBLICATION_METHOD_COLORS["NemaSize"], 0.0)], width=0.5)

	ax.set_ylim(ylim)
	ax.set_xticks(range(len(rows)))
	ax.set_xticklabels([_italic(r) for r in rows])
	ax.set_ylabel(f"Percentage error of {measure} (%)", fontfamily="Arial", fontsize=FONT_SIZE_PT)
	for lbl in ax.get_xticklabels() + ax.get_yticklabels():
		lbl.set_fontfamily("Arial")

	out_path = OUTPUT_ROOT / f"Fig_S2{panel}.png"
	fig.savefig(out_path, dpi=300)
	fig.savefig(out_path.with_suffix(".svg"))
	plt.close(fig)
	print(f"[ok] Saved: {out_path}")

	gf._write_boxplot_stats(
		OUTPUT_ROOT / f"Fig_S2{panel}_stats.txt",
		title=f"Per-box statistics for Fig_S2{panel}.png",
		value_desc=f"Values are percentage error of worm {measure} (%): (computer - human) / human * 100",
		display_cats=rows,
		series_by_method={"NemaSize": data},
	)
	return out_path


def write_ranksum_tests(ns: pd.DataFrame) -> Path:
	"""Two-sided Wilcoxon rank-sum (Mann-Whitney U) test of error between the two species."""
	a, b = SPECIES_ORDER
	header = f"{'measure':<8} {'error':<9} {'n_' + a:>12} {'n_' + b:>12} {'median_' + a:>16} {'median_' + b:>16} {'U':>10} {'p':>12}"
	lines = [
		"Wilcoxon rank-sum (Mann-Whitney U, two-sided) test of NemaSize error, C. elegans vs C. briggsae",
		"signed = (computer - human) / human * 100 (as plotted); absolute = |signed|",
		"Worms are treated as independent samples.",
		"",
		header,
		"-" * len(header),
	]
	for measure in ("length", "width"):
		data = _pct_error_by_species(ns, f"{measure}_diff_um", f"{measure}_um_human")
		for kind in ("signed", "absolute"):
			x, y = (np.asarray(data[sp]) if kind == "signed" else np.abs(data[sp]) for sp in (a, b))
			res = mannwhitneyu(x, y, alternative="two-sided")
			lines.append(
				f"{measure:<8} {kind:<9} {len(x):>12d} {len(y):>12d} {np.median(x):>16.3f} {np.median(y):>16.3f} "
				f"{res.statistic:>10.1f} {res.pvalue:>12.3e}"
			)
	out_path = OUTPUT_ROOT / "Fig_S2_ranksum.txt"
	out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
	print("\n".join(lines[4:]))
	print(f"[ok] Saved: {out_path}")
	return out_path


def main() -> None:
	OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
	ns = load_common_worms()
	print(ns["species"].value_counts().to_string())
	ylim = _shared_ylim(ns)
	create_species_boxplot(ns, "A", "length", ylim)
	create_species_boxplot(ns, "B", "width", ylim)
	write_ranksum_tests(ns)


if __name__ == "__main__":
	main()
