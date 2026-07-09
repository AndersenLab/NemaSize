"""
Fig_5D and Fig_5E: normalized dose-response plots of
median_wormlength_um_reg vs concentration_um. Each strain is plotted as an
independent data series (mean ± SD across well replicates).

Fig_5D = Cbriggsae dataset, Fig_5E = NemaSize dataset.
Only the normalized plot is produced (the raw/unnormalized plot is disabled).

X-axis: concentration_um (log scale)
Y-axis: median_wormlength_um_reg_delta, normalized and converted to nM (x-axis)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

plt.rcParams["font.family"] = "Arial"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
OUTPUT_DIR = SCRIPT_DIR / "output"

# ── Normalised plot font sizes ─────────────────────────────────────────────────
NORM_TICK_SIZE   = 12   # tick labels
NORM_LEGEND_SIZE = 10   # legend text
NORM_LABEL_SIZE  = 12  # axis labels

# ── Figure size (inches) ──────────────────────────────────────────────────────
NORM_FIG_WIDTH, NORM_FIG_HEIGHT = 3.5, 3.5    # normalised plot

# ── Datasets to process ───────────────────────────────────────────────────────
DATASETS = [
    {
        "data": DATA_DIR / "20260615_length_reg_delta_nemasize.csv",
        "out": OUTPUT_DIR / "Fig_5E.svg",
    },
    {
        "data": DATA_DIR / "20260522_Cbriggsae_IVM_DRC2_regressed_delta_HTLDA_cellprofiler.csv",
        "out": OUTPUT_DIR / "Fig_5D.svg",
    },
]


def aggregate(data_path: Path):
    """Load CSV and return per-strain × concentration mean/SD."""
    df = pd.read_csv(data_path)
    df["strain"] = df["strain"].replace("PB420", "CGC2")
    agg = (
        df.groupby(["strain", "concentration_um"])["median_wormlength_um_reg"]
        .agg(mean="mean", sd=lambda x: x.std(), n="count")
        .reset_index()
    )
    nonzero_concs = sorted(agg.loc[agg["concentration_um"] > 0, "concentration_um"].unique())
    log_step = np.log10(nonzero_concs[1]) - np.log10(nonzero_concs[0])
    pseudo_zero = 10 ** (np.log10(nonzero_concs[0]) - log_step)
    agg["x"] = np.where(agg["concentration_um"] == 0, pseudo_zero, agg["concentration_um"])
    return df, agg, nonzero_concs, pseudo_zero


def compute_norm(df):
    """Aggregate pre-computed delta column (nonzero doses only); convert µM → nM."""
    agg_norm = (
        df.loc[df["concentration_um"] > 0]
        .groupby(["strain", "concentration_um"])["median_wormlength_um_reg_delta"]
        .agg(mean_norm="mean", sd=lambda x: x.std())
        .reset_index()
    )
    agg_norm["x_nm"] = agg_norm["concentration_um"] * 1000
    return agg_norm


def plot_dose_response(agg_norm, out_path: Path, norm_ylim) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    strains = sorted(agg_norm["strain"].unique())
    palette = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P"]
    style = {s: {"color": palette[i], "marker": markers[i]} for i, s in enumerate(strains)}

    # ── Normalised plot ───────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(NORM_FIG_WIDTH, NORM_FIG_HEIGHT))

    for strain, grp in agg_norm.groupby("strain"):
        grp_sorted = grp.sort_values("x_nm")
        s = style[strain]
        ax2.errorbar(
            grp_sorted["x_nm"], grp_sorted["mean_norm"], yerr=grp_sorted["sd"],
            label=strain, color=s["color"], marker=s["marker"],
            markersize=5, linewidth=1.4, elinewidth=0.8,
            capsize=3, capthick=0.8, zorder=3,
        )

    ax2.set_xscale("log")
    ax2.set_ylim(norm_ylim)
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(100))
    ax2.set_xlabel("Concentration (nM)", fontsize=NORM_LABEL_SIZE)
    ax2.set_ylabel("Normalized animal length (µm)", fontsize=NORM_LABEL_SIZE)
    ax2.legend(fontsize=NORM_LEGEND_SIZE, frameon=False, loc="lower left")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.tick_params(axis="both", which="both", labelsize=NORM_TICK_SIZE, direction="in")

    plt.tight_layout()
    fig2.savefig(out_path, bbox_inches="tight")
    fig2.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_path.with_suffix('.png')}")
    plt.close(fig2)


# ── Run for all datasets ──────────────────────────────────────────────────────
# Pass 1: aggregate all data and compute shared normalised y limits
all_data = [aggregate(ds["data"]) for ds in DATASETS]
all_norms = [compute_norm(df) for df, *_ in all_data]

combined = pd.concat(all_norms)
y_lo = (combined["mean_norm"] - combined["sd"]).min()
y_hi = (combined["mean_norm"] + combined["sd"]).max()
margin = (y_hi - y_lo) * 0.05
shared_norm_ylim = (y_lo - margin, y_hi + margin)

# Pass 2: plot each dataset with the shared y range
for ds, (df, agg, _nonzero_concs, _pseudo_zero), agg_norm in zip(DATASETS, all_data, all_norms):
    print(f"\nProcessing: {ds['data'].name}")
    print("Sample sizes (n wells per strain × concentration):")
    print(agg.pivot(index="strain", columns="concentration_um", values="n").to_string())
    print()
    plot_dose_response(agg_norm, ds["out"], norm_ylim=shared_norm_ylim)
