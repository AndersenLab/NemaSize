"""
Calculate the physical field-of-view (FOV) for a list of ROI names.

For each ROI the physical FOV is derived from the roi_catalog.json produced
by detect_and_crop_rois.py:

    FOV_um = roi_size_original  [px]  x  UM_PER_PX  [µm/px]

where UM_PER_PX = 3.2937 µm/px (calibrated from CellProfiler).

FOV formula:
    FOV_um = roi_size_original * UM_PER_PX

Also reported per ROI:
    - roi_size_original   : ROI side length in original-image pixels
    - output_image_size   : saved image side length (pixels)
    - scale_factor        : output_image_size / roi_size_original
    - um_per_px_roi       : effective µm/px in the *saved* ROI image
                            = UM_PER_PX / scale_factor
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Fig 2
ROI_CATALOG_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap_catalog\valid\roi_catalog.json"

# µm per pixel in the ORIGINAL image (CellProfiler calibration)
UM_PER_PX = 3.2937

# ROI names grouped by figure panel.  Keys can be any label, e.g. "Fig1", "Fig1a", "Fig2b".
QUERY_ROI_NAMES = {
    "Fig2c": [
        "109_png.rf.c8dd17e858d1df0c04a8d6524de5fe76_roi_6",
        "84_png.rf.58f1729113639cc3714467029dcb4780_roi_3",
        "10_png.rf.2a5422c48a272b06a5ed1e4811e31377_roi_17",
        "176_png.rf.ade8f3becc3343adf565d66c9323a444_roi_14",
    ],
    "Fig2d": [
        "155_png.rf.f50fa82504926ab35f38301b6390248d_roi_23",
        "10_png.rf.2a5422c48a272b06a5ed1e4811e31377_roi_19",
    ],

}

# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------

catalog_path = Path(ROI_CATALOG_PATH)
if not catalog_path.exists():
    raise FileNotFoundError(f"ROI catalog not found: {catalog_path}")

with open(catalog_path, "r") as f:
    catalog = json.load(f)

# ---------------------------------------------------------------------------
# Compute and print results
# ---------------------------------------------------------------------------

col_roi     = 60
col_header  = (
    f"{'ROI name':<{col_roi}}  "
    f"{'roi_size_orig (px)':>18}  "
    f"{'FOV (µm × µm)':>22}  "
    f"{'output_size (px)':>16}  "
    f"{'scale_factor':>12}  "
    f"{'µm/px (ROI)':>11}"
)
row_sep = "-" * len(col_header)

missing = []
for fig_label, roi_names in QUERY_ROI_NAMES.items():
    print(f"\n── {fig_label} {'─' * (len(col_header) - len(fig_label) - 4)}")
    print(col_header)
    print(row_sep)
    for roi_name in roi_names:
        entry = catalog.get(roi_name)
        if entry is None:
            missing.append((fig_label, roi_name))
            print(f"  {roi_name}  <NOT FOUND IN CATALOG>")
            continue

        roi_size_orig = entry["roi_size_original"]
        output_size   = entry["output_image_size"]
        scale_factor  = entry["scale_factor"]
        fov_um        = roi_size_orig * UM_PER_PX
        um_per_px_roi = UM_PER_PX / scale_factor
        fov_str       = f"{fov_um:.1f} × {fov_um:.1f}"

        print(
            f"{roi_name:<{col_roi}}  "
            f"{roi_size_orig:>18d}  "
            f"{fov_str:>22}  "
            f"{output_size:>16d}  "
            f"{scale_factor:>12.4f}  "
            f"{um_per_px_roi:>11.4f}"
        )

if missing:
    print(f"\nNot found in catalog ({len(missing)}):")
    for fig_label, name in missing:
        print(f"  [{fig_label}] {name}")

