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

# µm per pixel in the ORIGINAL image (CellProfiler calibration)
UM_PER_PX = 3.2937

# Each figure entry specifies its own catalog path and the ROI names to query.
# Keys can be any label, e.g. "Fig1", "Fig1a", "Fig2b".
FIGURES = {
    "Fig1_Straight": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\straight_worm\train\roi_catalog.json",
        "rois": [
            "121_png.rf.49e7ff8713556ad7cc37d311e25762ea_roi_26",
            "111_png.rf.8fe543cf12ca100b887933e93b334b92_roi_8",
        ],
    },
    "Fig1_Curly": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\curly_worm\train\roi_catalog.json",
        "rois": [
            "79_png.rf.5ec9f077f4d5af0f3ab029d7283bba4e_roi_1",
            "85_png.rf.75439e009b13787c8bd637122941d6c9_roi_7",
        ],
    },
    "Fig1_Tiny": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\tiny_worm\train\roi_catalog.json",
        "rois": [
            "90_png.rf.69bdac2f8992d8d3ebf3226c61ad2bd5_roi_19",  # left
            "175_png.rf.809573d6b32e2ea5b757d39421fb3e6a_roi_22",  # right
        ],
    },
    "Fig1_selfoverlap": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\self-overlapping_worm\train\roi_catalog.json",
        "rois": [
            "168_png.rf.5a44d0e8273a80cfd25e3f2fec8d8fd7_roi_20",  # left
            "175_png.rf.809573d6b32e2ea5b757d39421fb3e6a_roi_32",  # right
        ],
    },
    "Fig1_mutualoverlap_left": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\incomplete_worm\train\roi_catalog.json",
        "rois": [
            "79_png.rf.5ec9f077f4d5af0f3ab029d7283bba4e_roi_14",  # left
        ],
    },
    "Fig1_mutualoverlap_right": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\overlapping_worm\train\roi_catalog.json",
        "rois": [
            "239_png.rf.4ed938c80d208d81f455a7bd2caf1cca_roi_18",  # left
        ],
    },
    "Fig1_incomplete": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\incomplete_worm\train\roi_catalog.json",
        "rois": [
            "156_png.rf.6120be0c3a28f4801b0a45145ce1e961_roi_34",  # left
            "156_png.rf.6120be0c3a28f4801b0a45145ce1e961_roi_17",  # right
        ],
    },
    "Fig1_debris": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\straight_worm\train\roi_catalog.json",
        "rois": [
            "149_png.rf.4dfa00beb57e9649962221eb0b15f710_roi_14",
            "147_png.rf.cf2e44c074bedd11a6fc17a869ce1e4a_roi_8",

        ],
    },
    "Fig1_poorcontrast": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\tiny_worm\train\roi_catalog.json",
        "rois": [
            "183_png.rf.3b20a92fe511dac0343f295bffbeeee8_roi_31",  # left (tiny)
            "10_png.rf.2a5422c48a272b06a5ed1e4811e31377_roi_0",  # right (tiny)
        ],
    },
    "Fig1_welledge_left": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\straight_worm\train\roi_catalog.json",
        "rois": [
            "134_png.rf.004aee1bacac1673bfbc037b834a310b_roi_0",  # left (straight worms)
        ],
    },
    "Fig1_welledge_right": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\GT_rois\tiny_worm\train\roi_catalog.json",
        "rois": [
            "62_png.rf.beb309695495d75acd96567a6b6510bc_roi_19",  # right (tiny worms)
        ],
    },
    "Fig2c": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\WormBodyROI_fix_overlap_catalog\valid\roi_catalog.json",
        "rois": [
            "109_png.rf.c8dd17e858d1df0c04a8d6524de5fe76_roi_6",
            "84_png.rf.58f1729113639cc3714467029dcb4780_roi_3",
            "10_png.rf.2a5422c48a272b06a5ed1e4811e31377_roi_17",
            "176_png.rf.ade8f3becc3343adf565d66c9323a444_roi_14",
        ],
    },
    "Fig2c": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\WormBodyROI_fix_overlap_catalog\valid\roi_catalog.json",
        "rois": [
            "109_png.rf.c8dd17e858d1df0c04a8d6524de5fe76_roi_6",
            "84_png.rf.58f1729113639cc3714467029dcb4780_roi_3",
            "10_png.rf.2a5422c48a272b06a5ed1e4811e31377_roi_17",
            "176_png.rf.ade8f3becc3343adf565d66c9323a444_roi_14",
        ],
    },
    "Fig2d": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\WormBodyROI_fix_overlap_catalog\valid\roi_catalog.json",
        "rois": [
            "155_png.rf.f50fa82504926ab35f38301b6390248d_roi_23",
            "10_png.rf.2a5422c48a272b06a5ed1e4811e31377_roi_19",
        ],
    },
    "Fig3Line": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json",
        "rois": [
            "113_png.rf.e3028a1dab303081c21db80950b8398c_roi_19",
            "78_png.rf.64c5d696884a3b905e0a4e7625969623_roi_18",
            "79_png.rf.5ec9f077f4d5af0f3ab029d7283bba4e_roi_55",
        ],
    },
    "Fig3Circle": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json",
        "rois": [
            "79_png.rf.5ec9f077f4d5af0f3ab029d7283bba4e_roi_24",
            "7_png.rf.8ce77fd1bebdb67de94465bd206e1900_roi_7",
        ],
    },
    "Fig3Six": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json",
        "rois": [
            "82_png.rf.78a828637d0fdf11d14f1d4b7920f8d0_roi_28",
            "168_png.rf.5a44d0e8273a80cfd25e3f2fec8d8fd7_roi_8",
        ],
    },
    "Fig3Omega": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json",
        "rois": [
            "174_png.rf.0e87f59ae4c245e42548098544b4cc07_roi_14",
            "174_png.rf.0e87f59ae4c245e42548098544b4cc07_roi_24",
        ],
    },
    "Fig3Fish": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json",
        "rois": [
            "175_png.rf.809573d6b32e2ea5b757d39421fb3e6a_roi_17",
            "168_png.rf.5a44d0e8273a80cfd25e3f2fec8d8fd7_roi_30",
        ],
    },
    "Fig3Candy": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\Datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json",
        "rois": [
            "177_png.rf.7d111caca2be9c02b4c09af0069e3b04_roi_12",
            "176_png.rf.ade8f3becc3343adf565d66c9323a444_roi_20",
        ],
    },
    "FigS2": {
        "catalog": r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json",
        "rois": [
            "172_png.rf.f3457ec580e8349d69afeb37c6eae6f8_roi_17",
            "33_png.rf.7b2ed94dd972fe6fe24277fbb2eb0d04_roi_17",
            "85_png.rf.75439e009b13787c8bd637122941d6c9_roi_36",
            "84_png.rf.58f1729113639cc3714467029dcb4780_roi_49",
            "79_png.rf.5ec9f077f4d5af0f3ab029d7283bba4e_roi_57",
            "85_png.rf.75439e009b13787c8bd637122941d6c9_roi_31",
        ],
    },
    
}

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

# Cache loaded catalogs to avoid re-reading the same file multiple times.
_catalog_cache: dict = {}

missing = []
for fig_label, fig_cfg in FIGURES.items():
    catalog_path = Path(fig_cfg["catalog"])
    roi_names    = fig_cfg["rois"]

    if str(catalog_path) not in _catalog_cache:
        if not catalog_path.exists():
            raise FileNotFoundError(f"ROI catalog not found: {catalog_path}")
        with open(catalog_path, "r") as f:
            _catalog_cache[str(catalog_path)] = json.load(f)
    catalog = _catalog_cache[str(catalog_path)]

    print(f"\n── {fig_label} {'─' * (len(col_header) - len(fig_label) - 4)}")
    print(f"   catalog: {catalog_path}")
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
        fov_str       = f"{fov_um:.0f} × {fov_um:.0f} µm"

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

