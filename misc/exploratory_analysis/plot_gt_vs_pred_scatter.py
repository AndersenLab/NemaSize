"""
Scatter plot comparison of human (ground truth) vs computer (predicted)
worm length and width measurements in physical units (µm).

GT  : skeleton_gt/worm_lengths.csv
Pred: skeleton_pred/worm_lengths.csv
"""

import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = pathlib.Path(
    r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets"
    r"\WormBodyROI_fix_overlap_catalog\valid"
)
GT_CSV   = BASE / "skeleton_gt"   / "worm_lengths.csv"
PRED_CSV = BASE / "skeleton_pred" / "worm_lengths.csv"
OUT_DIR  = BASE / "scatter_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Pixel → micron conversion ──────────────────────────────────────────────────
# 1 pixel = (sensor_mm * 1000 / image_width_px) µm
# Adjust these values to match your imaging setup.
SENSOR_MM       = 6.58    # sensor size in mm
IMAGE_WIDTH_PX  = 1983    # image width in pixels
UM_PER_PX       = SENSOR_MM * 1000 / IMAGE_WIDTH_PX   # µm per pixel
print(f"Conversion: {UM_PER_PX:.6f} µm/px")

# ── Load & merge on image name ─────────────────────────────────────────────────
gt   = pd.read_csv(GT_CSV,   usecols=["image", "skeleton_length_orig_px", "width_orig_px"])
pred = pd.read_csv(PRED_CSV, usecols=["image", "skeleton_length_orig_px", "width_orig_px"])

df = pd.merge(gt, pred, on="image", suffixes=("_gt", "_pred"))
print(f"Matched worms: {len(df)}")

# Convert pixel columns to microns
for col in ["skeleton_length_orig_px_gt", "skeleton_length_orig_px_pred",
            "width_orig_px_gt",           "width_orig_px_pred"]:
    df[col] = df[col] * UM_PER_PX

# ── Helper: one scatter panel ──────────────────────────────────────────────────
def scatter_panel(ax, x, y, xlabel, ylabel, title):
    """Draw scatter + identity line + linear regression on ax."""
    ax.scatter(x, y, s=10, alpha=0.45, color="#2196F3", edgecolors="none")

    # Identity line (perfect agreement)
    lims = [min(x.min(), y.min()) * 0.95, max(x.max(), y.max()) * 1.05]
    ax.plot(lims, lims, color="black", lw=1.2, ls="--", label="Identity (y = x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Linear regression (no line drawn; stats shown in annotation)
    slope, intercept, r, p, _ = stats.linregress(x, y)
    r2 = r ** 2

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    # Annotate sample size + error stats
    err = y - x
    mean_err = err.mean()
    mae      = err.abs().mean()
    stats_txt = (
        f"n = {len(x)}\n"
        f"Fit: y = {slope:.3f}x + {intercept:.2f}\n"
        f"$R^2$ = {r2:.4f}\n"
        f"Mean error          = {mean_err:+.3f} µm\n"
        f"Mean Absolute Error = {mae:.3f} µm"
    )
    ax.text(0.02, 0.97, stats_txt, transform=ax.transAxes,
            va="top", ha="left", fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    return err, slope, intercept, r2

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.suptitle("Human (GT) vs Computer Prediction\n— Physical Scale (µm) —",
             fontsize=13, fontweight="bold", y=1.01)

err_len, slope_len, intercept_len, r2_len = scatter_panel(
    axes[0],
    x=df["skeleton_length_orig_px_gt"],
    y=df["skeleton_length_orig_px_pred"],
    xlabel="Human annotation (µm)",
    ylabel="Computer measurement (µm)",
    title="Worm Length",
)

err_wid, slope_wid, intercept_wid, r2_wid = scatter_panel(
    axes[1],
    x=df["width_orig_px_gt"],
    y=df["width_orig_px_pred"],
    xlabel="Human annotation (µm)",
    ylabel="Computer measurement (µm)",
    title="Worm Width",
)

# ── Print error summary to console ────────────────────────────────────────────
for label, err, slope, intercept, r2 in [
    ("Worm Length (µm)", err_len, slope_len, intercept_len, r2_len),
    ("Worm Width  (µm)", err_wid, slope_wid, intercept_wid, r2_wid),
]:
    print(f"\n{'─'*45}")
    print(f"  {label}  (computer − human)")
    print(f"{'─'*45}")
    print(f"  n          = {len(err)}")
    print(f"  Fit        = y = {slope:.4f}x + {intercept:.4f}")
    print(f"  R²         = {r2:.4f}")
    print(f"  Mean error = {err.mean():+.4f} µm")
    print(f"  MAE        = {err.abs().mean():.4f} µm")
    print(f"  Min  error = {err.min():+.4f} µm")
    print(f"  Max  error = {err.max():+.4f} µm")
    print(f"  Median err = {err.median():+.4f} µm")

plt.tight_layout()

out_path = OUT_DIR / "gt_vs_pred_length_width_scatter_um.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# ── Distribution plots ─────────────────────────────────────────────────────────
def dist_panel(ax, x_gt, x_pred, xlabel, title):
    """Overlay histograms + KDE curves for GT and Pred."""
    bins = min(50, max(15, len(x_gt) // 10))
    all_vals = np.concatenate([x_gt, x_pred])
    rng = (all_vals.min(), all_vals.max())

    ax.hist(x_gt,   bins=bins, range=rng, density=True, alpha=0.4,
            color="#4CAF50", label="Human (GT)")
    ax.hist(x_pred, bins=bins, range=rng, density=True, alpha=0.4,
            color="#F44336", label="Computer (Pred)")

    # Vertical mean lines
    ax.axvline(x_gt.mean(),   color="#2E7D32", lw=1.5, ls="--",
               label=f"GT mean = {x_gt.mean():.1f} µm")
    ax.axvline(x_pred.mean(), color="#B71C1C", lw=1.5, ls="--",
               label=f"Pred mean = {x_pred.mean():.1f} µm")

    # Stats annotation
    stats_txt = (
        f"GT   — mean={x_gt.mean():.2f}, std={x_gt.std():.2f} µm\n"
        f"Pred — mean={x_pred.mean():.2f}, std={x_pred.std():.2f} µm"
    )
    ax.text(0.98, 0.97, stats_txt, transform=ax.transAxes,
            va="top", ha="right", fontsize=8.5, color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Distribution of Worm Measurements\n— Human (GT) vs Computer Prediction (µm) —",
              fontsize=13, fontweight="bold", y=1.01)

dist_panel(axes2[0],
           df["skeleton_length_orig_px_gt"],
           df["skeleton_length_orig_px_pred"],
           xlabel="Length (µm)", title="Worm Length Distribution")

dist_panel(axes2[1],
           df["width_orig_px_gt"],
           df["width_orig_px_pred"],
           xlabel="Width (µm)", title="Worm Width Distribution")

plt.tight_layout()

out_path2 = OUT_DIR / "gt_vs_pred_length_width_distribution_um.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")

plt.show()
