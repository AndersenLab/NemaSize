"""
Box plot: median_wormlength_um per strain, NemaSize vs CellProfiler side by side.
Each data point is one well (all concentrations pooled).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

plt.rcParams["font.family"] = "Arial"

# ── Paths ─────────────────────────────────────────────────────────────────────
NEMASIZE_PATH = Path(
    r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5"
    r"\Dose_response\20260615_length_reg_delta_nemasize.csv"
)
CELLPROFILER_PATH = Path(
    r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5"
    r"\Dose_response\20260522_Cbriggsae_IVM_DRC2_regressed_delta_HTLDA_cellprofiler.csv"
)
OUT_PATH = Path(
    r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5"
    r"\Dose_response\plots\worm_length_boxplot.svg"
)

# ── Load & label ──────────────────────────────────────────────────────────────
df_ns = pd.read_csv(NEMASIZE_PATH)
df_ns = df_ns.loc[df_ns["concentration_um"] == 0, ["strain", "median_wormlength_um"]]
df_ns["tool"] = "NemaSize"

df_cp = pd.read_csv(CELLPROFILER_PATH)
df_cp = df_cp.loc[df_cp["concentration_um"] == 0, ["strain", "median_wormlength_um"]]
df_cp["tool"] = "CellProfiler"

df = pd.concat([df_ns, df_cp], ignore_index=True)
df["strain"] = df["strain"].replace("PB420", "CGC2")

# ── Layout config ─────────────────────────────────────────────────────────────
strains = sorted(df["strain"].unique())
tools = ["CellProfiler", "NemaSize"]
colors = {"NemaSize": "#E45756", "CellProfiler": "#4C78A8"}

n_tools = len(tools)
box_width = 0.3
spacing = 0.06   # gap between the two boxes within a group
group_step = 1.0  # distance between strain groups

# ── Figure size ───────────────────────────────────────────────────────────────
FIG_WIDTH  = 7   # inches
FIG_HEIGHT = 2.8   # inches

# ── Font sizes ────────────────────────────────────────────────────────────────
FONT_SIZE = 12      # base font size for tick labels and legend
LABEL_SIZE = 12    # axis label font size

# ── Legend position ───────────────────────────────────────────────────────────
LEGEND_X = 0.0    # axes fraction (0 = left edge)
LEGEND_Y = 1.08   # axes fraction (1 = top edge; >1 moves above the axes)

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

for i, strain in enumerate(strains):
    for j, tool in enumerate(tools):
        vals = (
            df.loc[(df["strain"] == strain) & (df["tool"] == tool), "median_wormlength_um"]
            .dropna()
        )
        x = i * group_step + (j - (n_tools - 1) / 2) * (box_width + spacing)
        ax.boxplot(
            vals,
            positions=[x],
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor=colors[tool], alpha=0.5, linewidth=0.8),
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
            flierprops=dict(marker="", markersize=0),  # hide built-in fliers
        )
        # Overlay individual data points with jitter
        jitter = np.random.default_rng(seed=42).uniform(-box_width * 0.25, box_width * 0.25, size=len(vals))
        ax.scatter(
            x + jitter,
            vals,
            color=colors[tool],
            s=20,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )

# ── Axes & styling ────────────────────────────────────────────────────────────
ax.set_xticks([i * group_step for i in range(len(strains))])
ax.set_xticklabels(strains, fontsize=FONT_SIZE)
ax.set_ylabel("Animal length (µm)", fontsize=LABEL_SIZE)
ax.set_ylim(top=800)
ax.tick_params(axis="both", labelsize=FONT_SIZE, direction="in")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_handles = [
    mpatches.Patch(facecolor=colors[t], alpha=0.75, label=t) for t in tools
]
ax.legend(handles=legend_handles, fontsize=FONT_SIZE, frameon=False,
          loc="upper left", bbox_to_anchor=(LEGEND_X, LEGEND_Y))

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
plt.close(fig)
