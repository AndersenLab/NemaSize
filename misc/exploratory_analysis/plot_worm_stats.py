"""
plot_worm_stats.py
==================
Read a CSV of worm measurements and plot distributions for length and width.

Usage
-----
    python plot_worm_stats.py

Edit the CSV_PATH below to point to your output folder.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
# ── Configuration ──────────────────────────────────────────────────────────
# Optional second CSV for overlay comparison.  Set to None to disable.


CSV_PATH = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Etta_dataset\20260305_adult_images\20260305_adult_images\skeleton"
    r"\worm_lengths.csv"
)   # e.g. Path(r"C:\...\worm_lengths_2.csv")



CSV_PATH_2 = Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Etta_dataset\20260305_dauer_images\20260305_dauer_images\skeleton"
    r"\worm_lengths.csv"
)

#CSV_PATH = Path(
#    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
#    r"\WormBodyROI_fix_overlap_catalog\valid\skeleton_gt"
#    r"\worm_lengths.csv"
#)   # e.g. Path(r"C:\...\worm_lengths_2.csv")

# CSV_PATH_2 = Path(
#     r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Pipeline\Skeletonization"
#     r"\worm_lengths.csv"
# )

# Labels used in the legend when two files are provided.
LABEL_1 = "Adult"
LABEL_2 = "Dauer"

# Which columns to prefer for length / width.
# New format: length_um, width_um (already in microns)
# Backward compatibility: original/ROI pixel columns
PREFER_ORIG = True
# ── End configuration ──────────────────────────────────────────────────────


def _load_csv(csv_path: Path, prefer_orig: bool):
    """Load a measurements CSV and return (df, len_col, width_col, unit_label)."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            "Run skeletonize_worms.py first to generate the measurements."
        )
    df = pd.read_csv(csv_path)
    df = df.replace("", np.nan)

    # Preferred schema (already in microns)
    if {"length_um", "width_um"}.issubset(df.columns):
        len_col, width_col, unit_label = "length_um", "width_um", "µm"
        df[len_col] = pd.to_numeric(df[len_col], errors="coerce")
        df[width_col] = pd.to_numeric(df[width_col], errors="coerce")
        return df, len_col, width_col, unit_label

    # Backward compatibility with older pixel-based exports
    has_orig = "skeleton_length_orig_px" in df.columns and df["skeleton_length_orig_px"].notna().any()
    if prefer_orig and has_orig:
        len_col, width_col, unit_label = "skeleton_length_orig_px", "width_orig_px", "original-image px"
    else:
        len_col, width_col, unit_label = "skeleton_length_roi_px", "width_roi_px", "ROI-crop px"

    missing = [c for c in (len_col, width_col) if c not in df.columns]
    if missing:
        raise KeyError(
            f"CSV is missing required columns: {missing}. "
            "Expected either [length_um, width_um] or legacy pixel columns."
        )

    df[len_col] = pd.to_numeric(df[len_col], errors="coerce")
    df[width_col] = pd.to_numeric(df[width_col], errors="coerce")
    return df, len_col, width_col, unit_label


def _stats_annotation(values: np.ndarray, decimals: int = 1, show_range: bool = True) -> str:
    fmt = f".{decimals}f"
    base = (
        f"n = {len(values)}\n"
        f"mean   = {values.mean():{fmt}}\n"
        f"median = {np.median(values):{fmt}}\n"
        f"std    = {values.std():{fmt}}"
    )
    if show_range:
        base += f"\n[{values.min():{fmt}} \u2013 {values.max():{fmt}}]"
    return base


def _hist_overlay(ax, vals1, vals2, color1, color2, label1, label2):
    """Plot two overlapping density-normalised histograms with mean vlines."""
    all_vals = np.concatenate([vals1, vals2])
    n_bins = max(10, int(np.sqrt(len(all_vals))))
    hist_kw = dict(density=True, edgecolor="white", linewidth=0.4, alpha=0.55)
    ax.hist(vals1, bins=n_bins, color=color1, label=label1, **hist_kw)
    ax.hist(vals2, bins=n_bins, color=color2, label=label2, **hist_kw)
    ax.axvline(vals1.mean(), color=color1, lw=1.8, ls="--")
    ax.axvline(vals2.mean(), color=color2, lw=1.8, ls="--")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)


def _aspect_ratio_test_text(ratios1: np.ndarray, ratios2: np.ndarray) -> str:
    """Return a compact summary of two-group statistical tests."""
    t_res = stats.ttest_ind(ratios1, ratios2, equal_var=False, nan_policy="omit")
    mw_res = stats.mannwhitneyu(ratios1, ratios2, alternative="two-sided")
    return (
        f"Welch t-test p = {t_res.pvalue:.3e}\n"
        f"Mann-Whitney p = {mw_res.pvalue:.3e}"
    )


def plot_distributions(
    csv_path: Path,
    prefer_orig: bool = True,
    csv_path2: "Path | None" = None,
    label1: str = "Dataset 1",
    label2: str = "Dataset 2",
) -> None:
    df, len_col, width_col, unit = _load_csv(csv_path, prefer_orig)
    lengths = df[len_col].dropna().values
    widths  = df[width_col].dropna().values
    valid_pairs = df[[len_col, width_col]].dropna()
    valid_pairs = valid_pairs[valid_pairs[len_col] > 0]
    ratios = (valid_pairs[width_col] / valid_pairs[len_col]).values

    if len(lengths) == 0 or len(widths) == 0 or len(ratios) == 0:
        print("No valid measurements found in the CSV.")
        return

    # Optionally load second dataset
    overlay = csv_path2 is not None
    if overlay:
        df2, len_col2, width_col2, unit2 = _load_csv(csv_path2, prefer_orig)
        if unit2 != unit:
            raise ValueError(
                f"Unit mismatch between CSV files: {unit} vs {unit2}. "
                "Use files with the same units."
            )
        lengths2 = df2[len_col2].dropna().values
        widths2  = df2[width_col2].dropna().values
        valid_pairs2 = df2[[len_col2, width_col2]].dropna()
        valid_pairs2 = valid_pairs2[valid_pairs2[len_col2] > 0]
        ratios2 = (valid_pairs2[width_col2] / valid_pairs2[len_col2]).values
        if len(ratios2) == 0:
            print("No valid aspect-ratio measurements found in the second CSV.")
            return

    title_suffix = (
        f"{label1}: n={len(df)},  {label2}: n={len(df2)}  |  {unit}"
        if overlay else
        f"(n = {len(df)} images, {unit})"
    )

    # ── Figure 1: panels 1 & 2 (histograms) ───────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Worm Measurement Distributions  {title_suffix}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    _HIST_KWARGS = dict(edgecolor="white", linewidth=0.4, alpha=0.75)

    # ── Panel 1: Skeleton length ────────────────────────────────────────────
    ax = axes[0]
    if overlay:
        _hist_overlay(ax, lengths, lengths2, "#4C8BF5", "#F44336", label1, label2)
        stats_text = (
            f"── {label1} ──\n" + _stats_annotation(lengths, 1) +
            f"\n── {label2} ──\n" + _stats_annotation(lengths2, 1)
        )
    else:
        n_bins = max(10, int(np.sqrt(len(lengths))))
        ax.hist(lengths, bins=n_bins, color="#4C8BF5", **_HIST_KWARGS)
        ax.set_ylabel("Count")
        stats_text = _stats_annotation(lengths, 1)

    ax.set_xlabel(f"Skeleton length ({unit})")
    ax.set_title("Worm Length Distribution")
    ax.text(
        0.97, 0.97, stats_text,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.0, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # ── Panel 2: Mean body width ────────────────────────────────────────────
    ax = axes[1]
    if overlay:
        _hist_overlay(ax, widths, widths2, "#F5A623", "#9C27B0", label1, label2)
        stats_text_w = (
            f"── {label1} ──\n" + _stats_annotation(widths, 2) +
            f"\n── {label2} ──\n" + _stats_annotation(widths2, 2)
        )
    else:
        n_bins_w = max(10, int(np.sqrt(len(widths))))
        ax.hist(widths, bins=n_bins_w, color="#F5A623", **_HIST_KWARGS)
        ax.set_ylabel("Count")
        stats_text_w = _stats_annotation(widths, 2)

    ax.set_xlabel(f"Mean body width ({unit})")
    ax.set_title("Worm Width Distribution")
    ax.text(
        0.97, 0.97, stats_text_w,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8.0, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    plt.tight_layout()
    out_path1 = csv_path.parent / "worm_stats_distributions.png"
    plt.savefig(out_path1, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {out_path1}")
    plt.show()

    # ── Figure 2: panels 3 & 4 (scatter + aspect ratio) ───────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

    # ── Panel 3: Scatter length vs width ─────────────────────────────────────
    ax = axes2[0]
    both = df[[len_col, width_col]].dropna()
    ax.scatter(
        both[len_col], both[width_col],
        color="#6DBF67", alpha=0.35, edgecolors="none", s=40,
        label=label1 if overlay else None,
    )
    if overlay:
        both2 = df2[[len_col2, width_col2]].dropna()
        ax.scatter(
            both2[len_col2], both2[width_col2],
            color="#2196F3", alpha=0.35, edgecolors="none", s=40,
            label=label2,
        )
        ax.legend(fontsize=9)

    ax.set_xlabel(f"Skeleton length ({unit})")
    ax.set_ylabel(f"Mean body width ({unit})")

    # ── Panel 4: Aspect ratio box plot + datapoints ─────────────────────────
    ax = axes2[1]
    rng = np.random.default_rng(7)
    if overlay:
        data = [ratios, ratios2]
        positions = [1, 2]
        colors = ["#6DBF67", "#2196F3"]
        labels = [label1, label2]
        test_text = _aspect_ratio_test_text(ratios, ratios2)
        print("Aspect ratio two-group tests (width/length):")
        print(test_text)
    else:
        data = [ratios]
        positions = [1]
        colors = ["#6DBF67"]
        labels = [label1]

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="black", linewidth=1.4),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    for pos, vals, color in zip(positions, data, colors):
        x_jitter = pos + rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(x_jitter, vals, s=14, alpha=0.35, color=color, edgecolors="none")

    stats_text_r = (
        f"── {label1} ──\n" + _stats_annotation(ratios, 3, show_range=False) +
        (f"\n── {label2} ──\n" + _stats_annotation(ratios2, 3, show_range=False) if overlay else "")
    )
    if overlay:
        stats_text_r += f"\n\n{test_text}"
    ax.text(
        0.97, 0.97, stats_text_r,
        transform=ax.transAxes, ha="right", va="top",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Body aspect ratio (width / length)")

    plt.tight_layout()
    out_path2 = csv_path.parent / "worm_stats_scatter_aspect.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    print(f"✓ Plot saved → {out_path2}")
    plt.show()


if __name__ == "__main__":
    plot_distributions(
        CSV_PATH,
        prefer_orig=PREFER_ORIG,
        csv_path2=CSV_PATH_2,
        label1=LABEL_1,
        label2=LABEL_2,
    )
