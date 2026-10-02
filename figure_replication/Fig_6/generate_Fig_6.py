"""
Fig_6: NemaSize annotations on two public C. elegans datasets (generalizability test).

  A, B : BBBC010 (live/dead, brightfield, 696 x 520 px)
  C, D : BBBC011 (oil red O fat stain, color, 691 x 770 px)

For each panel the raw image is overlaid with the contour (yellow) and
centerline skeleton (green, blue endpoints) of every worm NemaSize found.
The coordinates come from the per-worm ``contour_skeleton_txt/*.txt`` files
written by the pipeline: normalized (x, y) in [0, 1] of the full raw image.

Layout (all paths are relative to this script, so the folder is standalone):

    Fig_6/
        generate_Fig_6.py
        data/
            raw_images/<stem>.png
            contour_skeleton_txt/<stem>_roi_<n>.txt
        output/            <- annotated panels are written here

Pipeline settings used to produce the annotations:
  - detection confidence threshold 0.1 (default 0.25)
  - detection center-distance filter disabled (default 0.94)
  - BBBC010: raw 16-bit TIFs min-max scaled to 8-bit, then gamma 0.5 before
    inference. The overlays here are drawn on the image *before* gamma
    correction; gamma does not change the image geometry.
  - BBBC011: 8-bit color PNGs used as is.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "data" / "raw_images"
TXT_DIR = SCRIPT_DIR / "data" / "contour_skeleton_txt"
OUTPUT_DIR = SCRIPT_DIR / "output"

# Panel label -> image stem (file name without extension).
PANELS = {
    "A": "BBBC010-celegans-p1-mNA_D11",
    "B": "BBBC010-celegans-p1-mNA_A13",
    "C": "BBBC011-L4440-Plate6-mNA_H10",
    "D": "BBBC011-daf2-Plate6-mNA_F12",
}

# Same colors as source_code/visualize_contour_skeleton.py (there in BGR).
CONTOUR_COLOR = "#FFFF00"   # yellow
SKELETON_COLOR = "#00FF00"  # green
ENDPOINT_COLOR = "#0064FF"  # blue
ENDPOINT_EDGE = "white"

# Sizes in image pixels, so the look is the same whatever the image size.
CONTOUR_WIDTH_PX = 1.0
SKELETON_WIDTH_PX = 1.0
ENDPOINT_DIAMETER_PX = 10.0
ENDPOINT_EDGE_PX = 1.0

DPI = 300  # PNG is saved at 3x the native pixel size (figure is built at 100 dpi)


def parse_contour_skeleton_txt(txt_path):
    """Parse one per-worm .txt file.

    Returns ``(contours, skeleton)``: a list of (N, 2) normalized (x, y)
    arrays (one per [CONTOUR] section) and an (M, 2) array or None.
    """
    contours: list[list[list[float]]] = []
    skeleton_pts: list[list[float]] = []
    section = None
    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "[CONTOUR]":
                section = "contour"
                contours.append([])
                continue
            if line == "[SKELETON]":
                section = "skeleton"
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            if section == "contour" and contours:
                contours[-1].append([x, y])
            elif section == "skeleton":
                skeleton_pts.append([x, y])
    contour_arrays = [np.asarray(c, dtype=float) for c in contours if c]
    skeleton = np.asarray(skeleton_pts, dtype=float) if skeleton_pts else None
    return contour_arrays, skeleton


def to_pixels(pts_norm, width, height):
    """Normalized (x, y) -> matplotlib image coordinates (pixel centers at integers)."""
    return np.column_stack([pts_norm[:, 0] * width - 0.5, pts_norm[:, 1] * height - 0.5])


def draw_panel(stem, out_png, out_svg):
    img = plt.imread(RAW_DIR / f"{stem}.png")
    height, width = img.shape[:2]

    txt_paths = sorted(TXT_DIR.glob(f"{stem}_roi_*.txt"))
    if not txt_paths:
        raise FileNotFoundError(f"No annotation .txt files for {stem} in {TXT_DIR}")

    # One figure unit = one image pixel at 100 dpi; saved at DPI for print.
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    else:
        ax.imshow(img[..., :3], interpolation="nearest")

    pt = 72 / 100  # points per pixel at figure dpi=100
    endpoints = []
    for txt_path in txt_paths:
        contours, skeleton = parse_contour_skeleton_txt(txt_path)
        for cnt in contours:
            xy = to_pixels(cnt, width, height)
            xy = np.vstack([xy, xy[:1]])  # close the polygon
            ax.plot(xy[:, 0], xy[:, 1], color=CONTOUR_COLOR,
                    linewidth=CONTOUR_WIDTH_PX * pt, solid_joinstyle="round")
        if skeleton is not None and len(skeleton) >= 2:
            xy = to_pixels(skeleton, width, height)
            ax.plot(xy[:, 0], xy[:, 1], color=SKELETON_COLOR,
                    linewidth=SKELETON_WIDTH_PX * pt, solid_joinstyle="round")
            endpoints.extend([xy[0], xy[-1]])

    if endpoints:
        ep = np.asarray(endpoints)
        ax.scatter(ep[:, 0], ep[:, 1], s=(ENDPOINT_DIAMETER_PX * pt) ** 2,
                   c=ENDPOINT_COLOR, edgecolors=ENDPOINT_EDGE,
                   linewidths=ENDPOINT_EDGE_PX * pt, zorder=3)

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)

    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_svg)
    plt.close(fig)
    return width, height, len(txt_paths)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for label, stem in PANELS.items():
        out_png = OUTPUT_DIR / f"Fig_6{label}.png"
        out_svg = OUTPUT_DIR / f"Fig_6{label}.svg"
        w, h, n = draw_panel(stem, out_png, out_svg)
        print(f"Panel {label}: {stem}  ({w}x{h} px, {n} worms)")
        print(f"  Saved: {out_png}")
        print(f"  Saved: {out_svg}")


if __name__ == "__main__":
    main()
