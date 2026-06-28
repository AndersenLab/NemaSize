"""
EC10 / EC50 / EC90 + slope (b) estimation for dose-response datasets used by
plot_dose_response.py. Python port of ec_estimates.R, which itself wraps
fit_DRC_models() + safe_EC() from master_anthDRmanuscript_analysis_script.R.

Model: four-parameter log-logistic (drc::LL.4)
    f(x) = c + (d - c) / (1 + exp(b * (log(x) - log(e))))

Fit structure (mirrors safe_LL4_strain):
    pmodels = list(~strain-1, ~1, ~1, ~strain-1),
    fct     = LL.4(fixed = c(NA, -600, NA, NA))
  -> c fixed at -600
  -> d shared across strains within a dataset
  -> b and e fit per strain (joint NLS over all strains)

EC_p for LL.4 with default "relative" type (matches drc::ED defaults):
    EC_p = e * (p / (1 - p))^(1 / b)

Delta-method variance for EC_p uses the (b, e) sub-block of the joint covariance:
    Var(EC_p) = (dEC/db)^2 Var(b) + (dEC/de)^2 Var(e)
              + 2 (dEC/db)(dEC/de) Cov(b, e)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm

# Same two CSVs as plot_dose_response.py. Both datasets analyse the SAME images;
# they differ only in the processing tool (NemaSize vs CellProfiler).
DATASETS = [
    {
        "name": "NemaSize",
        "data": Path(
            r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5"
            r"\Dose_response\20260615_length_reg_delta_nemasize.csv"
        ),
        "out": Path(
            r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5"
            r"\Dose_response\ec_estimates_NemaSize.csv"
        ),
    },
    {
        "name": "CellProfiler",
        "data": Path(
            r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5"
            r"\Dose_response\20260522_Cbriggsae_IVM_DRC2_regressed_delta_HTLDA_cellprofiler.csv"
        ),
        "out": Path(
            r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5"
            r"\Dose_response\ec_estimates_CellProfiler.csv"
        ),
    },
]

# Combined side-by-side comparison output (slope + EC + SEs, both methods)
COMPARISON_OUT = Path(
    r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5"
    r"\Dose_response\ec_estimates_comparison.csv"
)

TRAIT = "median_wormlength_um_reg"
C_FIXED = -600.0        # LL.4(fixed = c(NA, -600, NA, NA))
EC_LEVELS = (0.10, 0.50, 0.90)
Z95 = norm.ppf(0.975)


# ── LL.4 model ────────────────────────────────────────────────────────────────
def ll4(x, b, c, d, e):
    """Four-parameter log-logistic; at x == 0 returns d (the x->0+ limit)."""
    x = np.asarray(x, dtype=float)
    b = np.broadcast_to(np.asarray(b, dtype=float), x.shape)
    e = np.broadcast_to(np.asarray(e, dtype=float), x.shape)
    out = np.full(x.shape, d, dtype=float)
    pos = x > 0
    if np.any(pos):
        out[pos] = c + (d - c) / (
            1.0 + np.exp(b[pos] * (np.log(x[pos]) - np.log(e[pos])))
        )
    return out


def make_joint_model(strain_idx):
    """
    Build the model callable that curve_fit will optimize.
    Parameter layout: [d, b_1, e_1, b_2, e_2, ..., b_K, e_K]
    """
    def model(x, *params):
        d = params[0]
        b_arr = np.asarray(params[1::2])
        e_arr = np.asarray(params[2::2])
        b = b_arr[strain_idx]
        e = e_arr[strain_idx]
        return ll4(x, b, C_FIXED, d, e)
    return model


# ── Fit one dataset ───────────────────────────────────────────────────────────
def fit_dataset(df: pd.DataFrame):
    """Joint LL.4 fit across all strains in `df`. Returns (strains, popt, pcov)."""
    sub = df[["strain", "concentration_um", TRAIT]].dropna().copy()
    sub = sub[np.isfinite(sub["concentration_um"]) & np.isfinite(sub[TRAIT])]

    strains = sorted(sub["strain"].unique())
    strain_to_idx = {s: i for i, s in enumerate(strains)}
    K = len(strains)

    X = sub["concentration_um"].to_numpy(dtype=float)
    Y = sub[TRAIT].to_numpy(dtype=float)
    sidx = sub["strain"].map(strain_to_idx).to_numpy(dtype=int)

    # Initial guesses
    d0 = Y[X == 0].mean() if np.any(X == 0) else float(np.nanmax(Y))
    nonzero = X[X > 0]
    e0 = float(np.median(nonzero)) if nonzero.size else 1.0

    p0 = [d0]
    for _ in range(K):
        p0.extend([1.0, e0])   # b0=1 (decreasing curve), e0 = median dose

    # Bounds: d unbounded; b unbounded; e strictly positive (log(e) needed)
    lower = [-np.inf] + [-np.inf, 1e-12] * K
    upper = [ np.inf] + [ np.inf,  np.inf] * K

    model = make_joint_model(sidx)
    popt, pcov = curve_fit(
        model, X, Y, p0=p0, bounds=(lower, upper), maxfev=50000
    )
    return strains, popt, pcov


# ── EC_p via analytic inverse + delta method ─────────────────────────────────
def ec_with_delta_se(p, b, e, var_b, var_e, cov_be):
    """EC_p = e * (p/(1-p))^(1/b) ; delta-method SE from (b, e) covariance."""
    L = np.log(p / (1.0 - p))
    ec = e * np.exp(L / b)
    dec_de = ec / e
    dec_db = -ec * L / (b ** 2)
    var = (dec_db ** 2) * var_b + (dec_de ** 2) * var_e + 2.0 * dec_db * dec_de * cov_be
    se = float(np.sqrt(max(var, 0.0)))
    return float(ec), se


def build_results(dataset_name, strains, popt, pcov) -> pd.DataFrame:
    d = float(popt[0])
    d_se = float(np.sqrt(pcov[0, 0]))

    rows = []
    for i, strain in enumerate(strains):
        bi, ei = 1 + 2 * i, 2 + 2 * i
        b = float(popt[bi])
        e = float(popt[ei])
        var_b = float(pcov[bi, bi])
        var_e = float(pcov[ei, ei])
        cov_be = float(pcov[bi, ei])

        row = {
            "dataset": dataset_name,
            "strain": strain,
            "b": b,
            "b_se": float(np.sqrt(var_b)),
            "c_fixed": C_FIXED,
            "d_shared": d,
            "d_se": d_se,
            "e_um": e,
            "e_se": float(np.sqrt(var_e)),
        }
        for p in EC_LEVELS:
            ec, se = ec_with_delta_se(p, b, e, var_b, var_e, cov_be)
            tag = f"EC{int(round(p * 100))}"
            row[f"{tag}_um"] = ec
            row[f"{tag}_se"] = se
            row[f"{tag}_lower95"] = ec - Z95 * se
            row[f"{tag}_upper95"] = ec + Z95 * se
        rows.append(row)
    return pd.DataFrame(rows)


# ── Side-by-side comparison ──────────────────────────────────────────────────
def build_comparison(per_dataset_results: dict) -> pd.DataFrame:
    """
    Transposed comparison CSV:
      - Rows: metric × method (slope, EC10, EC50, EC90; CellProfiler then NemaSize)
      - Columns: strain
      - EC values in µM, formatted to 2 significant figures
      - Slope kept at 4 decimal places (dimensionless)
    """
    import math

    def sig_fmt(val, se, n_sig=2):
        """Format 'val ± se' in scientific notation to n_sig significant figures."""
        return f"{val:.{n_sig}e} ± {se:.{n_sig}e}"

    # (val_col, se_col, row_label, convert_to_nM)
    metrics = [
        ("b",       "b_se",    "slope_b",  False),
        ("EC10_um", "EC10_se", "EC10_um",  True),
        ("EC50_um", "EC50_se", "EC50_um",  True),
        ("EC90_um", "EC90_se", "EC90_um",  True),
    ]
    method_order = ["CellProfiler", "NemaSize"]

    strains = sorted(
        set().union(*[set(df["strain"]) for df in per_dataset_results.values()])
    )
    method_dfs = {m: df.set_index("strain") for m, df in per_dataset_results.items()}

    rows = []
    for strain in strains:
        row = {"strain": strain}
        for method in method_order:
            df = method_dfs[method]
            for val_col, se_col, label, _ in metrics:
                if strain not in df.index:
                    row[f"{method}_{label}"] = None
                else:
                    r = df.loc[strain]
                    val, se = float(r[val_col]), float(r[se_col])
                    row[f"{method}_{label}"] = sig_fmt(val, se, n_sig=2)
        rows.append(row)

    ordered = ["strain"]
    for method in method_order:
        for _, _, label, _ in metrics:
            ordered.append(f"{method}_{label}")
    return pd.DataFrame(rows, columns=ordered)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    per_dataset_results: dict[str, pd.DataFrame] = {}
    for ds in DATASETS:
        print(f"\n=== {ds['name']}  ({ds['data'].name}) ===")
        df = pd.read_csv(ds["data"])
        df["strain"] = df["strain"].replace("PB420", "CGC2")   # match plot script

        strains, popt, pcov = fit_dataset(df)
        result = build_results(ds["name"], strains, popt, pcov)

        ds["out"].parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(ds["out"], index=False)
        per_dataset_results[ds["name"]] = result

        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(result.round(4).to_string(index=False))
        print(f"Saved: {ds['out']}")

    comparison = build_comparison(per_dataset_results)
    COMPARISON_OUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_OUT, index=False, encoding="utf-8-sig")
    print("\n=== NemaSize vs CellProfiler comparison ===")
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(comparison.round(4).to_string(index=False))
    print(f"Saved: {COMPARISON_OUT}")


if __name__ == "__main__":
    main()
