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

ROI_CATALOG_PATH = r"C:\Users\lizih\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json"

# µm per pixel in the ORIGINAL image (CellProfiler calibration)
UM_PER_PX = 3.2937

# List of ROI names to query (stem of the saved ROI image, without extension)
QUERY_ROI_NAMES = [
    "0_png.rf.168a7f1464a843d2217034df0330d03b_roi_0",
    "0_png.rf.168a7f1464a843d2217034df0330d03b_roi_1",
    "100_png.rf.93dbb301c4dc91fe8173d8a0a7f2dd5b_roi_0",
    "101_png.rf.c4bf71f43d058b555a715c3e9695be8b_roi_3",
    "102_png.rf.17ecda620c62893042cd19d261c2d15e_roi_10",
    # Add more ROI names here as needed
]

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

header = (
    f"{'ROI name':<60}  "
    f"{'roi_size_orig (px)':>18}  "
    f"{'FOV (µm)':>10}  "
    f"{'output_size (px)':>16}  "
    f"{'scale_factor':>12}  "
    f"{'µm/px (ROI)':>11}"
)
print(header)
print("-" * len(header))

missing = []
for roi_name in QUERY_ROI_NAMES:
    entry = catalog.get(roi_name)
    if entry is None:
        missing.append(roi_name)
        continue

    roi_size_orig   = entry["roi_size_original"]     # px in original image
    output_size     = entry["output_image_size"]     # px of saved image
    scale_factor    = entry["scale_factor"]          # output / orig

    fov_um          = roi_size_orig * UM_PER_PX      # physical FOV (µm)
    um_per_px_roi   = UM_PER_PX / scale_factor       # µm/px in saved ROI image

    print(
        f"{roi_name:<60}  "
        f"{roi_size_orig:>18d}  "
        f"{fov_um:>10.2f}  "
        f"{output_size:>16d}  "
        f"{scale_factor:>12.4f}  "
        f"{um_per_px_roi:>11.4f}"
    )

if missing:
    print(f"\nNot found in catalog ({len(missing)}):")
    for name in missing:
        print(f"  {name}")

