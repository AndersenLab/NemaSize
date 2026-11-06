"""
Plot pixel-intensity profile along the worm body axis.

Add-on analysis module for NemaSize.  Reads the contour/skeleton ``.txt``
file already produced by ``skeletonize_worms.py`` (``save_contour_skeleton_txt
=True``) together with the corresponding raw microscopy image, divides the
worm body contour into ``N`` segments along the skeleton (default 100), and
plots the mean raw pixel intensity in each segment as a function of body
coordinate (0 = one end of the skeleton, 1 = the other end).

This module does **not** call into the YOLO detection/segmentation pipeline
and does not modify anything in ``skeletonize_worms.py`` / ``run_pipeline.py``
— it only consumes their ``.txt`` outputs, so it can be run any time after
the main pipeline has produced ``contour_skeleton_txt/*.txt`` files.

Scope (first version)
----------------------
Intended for an intact, non-looping, non-overlapping single worm whose mask
is a single simple contour.  Segment assignment uses nearest-centerline-point
("Voronoi along the skeleton") pixel labelling, which is accurate for simple
worm shapes but can mis-assign pixels for coiled/self-touching worms — that
case is left for a future iteration.

Usage
-----
    python plot_intensity_profile.py \
        --txt-path   path/to/contour_skeleton_txt/IMG_roi_0.txt \
        --image-dir  path/to/raw_images \
        --output-dir path/to/intensity_profile

Batch mode (process every .txt file in a folder, e.g. all ROIs from one or
more source images) — use ``--txt-dir`` instead of ``--txt-path``:

    python plot_intensity_profile.py \
        --txt-dir    path/to/contour_skeleton_txt \
        --image-dir  path/to/raw_images \
        --output-dir path/to/intensity_profile

``--image-dir`` auto-resolves the raw image from each txt filename (strips the
trailing ``_roi_<id>`` suffix, as in ``visualize_contour_skeleton.py``), so a
single ``--image-dir`` works even when the txt files come from several source
images. In ``--txt-path`` single-file mode only, ``--image-path`` can be used
instead to point directly at one raw image file. In batch mode, worms that
fail (e.g. a fragmented/multi-contour mask) are skipped with a warning rather
than aborting the whole run.
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_contour_skeleton import (  # noqa: E402
    load_16bit_tif_as_8bit,
    parse_contour_skeleton_txt,
)

RAW_IMAGE_EXTENSIONS = (".tif", ".tiff", ".TIF", ".TIFF", ".png", ".jpg", ".jpeg")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def resample_polyline(points: np.ndarray, n_points: int) -> np.ndarray:
    """Resample an ordered ``(M, 2)`` (x, y) polyline to ``n_points`` positions
    evenly spaced by cumulative arc length (endpoints included)."""
    diffs = np.diff(points, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = cum[-1]
    if total <= 0:
        raise ValueError("Degenerate skeleton: zero arc length.")
    targets = np.linspace(0.0, total, n_points)
    x = np.interp(targets, cum, points[:, 0])
    y = np.interp(targets, cum, points[:, 1])
    return np.stack([x, y], axis=1)


def rasterize_mask(contours_px: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Fill one or more (x, y) pixel-space polygons into a boolean mask."""
    mask = np.zeros(shape, dtype=np.uint8)
    polys = [c.astype(np.int32).reshape(-1, 1, 2) for c in contours_px]
    cv2.fillPoly(mask, polys, 1)
    return mask.astype(bool)


def assign_segments(mask: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Label every foreground pixel with the index of its nearest skeleton
    anchor point (nearest-centerline-point / Voronoi-along-skeleton binning).

    Returns an ``int32`` array shaped like ``mask`` with values ``0..len(anchors)-1``
    for foreground pixels and ``-1`` for background.
    """
    ys, xs = np.nonzero(mask)
    pixel_xy = np.stack([xs, ys], axis=1).astype(float)
    tree = cKDTree(anchors)
    _, idx = tree.query(pixel_xy)
    label_img = np.full(mask.shape, -1, dtype=np.int32)
    label_img[ys, xs] = idx
    return label_img


def compute_intensity_profile(
    raw_img: np.ndarray, label_img: np.ndarray, n_segments: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean / std / pixel-count of raw intensity within each of ``n_segments``
    body-coordinate bins."""
    mean_intensity = np.full(n_segments, np.nan)
    std_intensity = np.full(n_segments, np.nan)
    pixel_counts = np.zeros(n_segments, dtype=int)
    for i in range(n_segments):
        sel = label_img == i
        pixel_counts[i] = int(sel.sum())
        if pixel_counts[i] > 0:
            vals = raw_img[sel].astype(np.float64)
            mean_intensity[i] = vals.mean()
            std_intensity[i] = vals.std()
    return mean_intensity, std_intensity, pixel_counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_intensity_profile(
    mean_intensity: np.ndarray,
    std_intensity: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    n = len(mean_intensity)
    body_coord = np.linspace(0.0, 1.0, n)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(body_coord, mean_intensity, color="tab:blue", lw=1.5)
    ax.fill_between(
        body_coord,
        mean_intensity - std_intensity,
        mean_intensity + std_intensity,
        alpha=0.2, color="tab:blue",
    )
    ax.set_xlabel("Body coordinate (0 = skeleton start, 1 = skeleton end)")
    ax.set_ylabel("Mean pixel intensity (raw units)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_segment_overlay(
    raw_img: np.ndarray,
    label_img: np.ndarray,
    anchors: np.ndarray,
    n_segments: int,
    out_path: Path,
) -> None:
    """Debug image with three side-by-side panels:

    1. ``Raw`` — the raw grayscale crop.
    2. ``Segmented`` — worm body colour-coded by segment index (head=segment
       0 -> tail=segment n_segments-1, "turbo" colormap: dark blue -> green
       -> yellow -> red), with a thin 1-px centerline traced through the
       skeleton anchors, a hollow ring at the head and a filled dot at the
       tail, and a colour bar below showing the segment-index -> colour
       mapping.
    3. ``Segment divisions`` — the worm outline plus every internal
       segment-to-segment boundary drawn as a thin black line on white,
       i.e. the 100 body-coordinate bins as 100 "slices".
    """
    lo, hi = float(raw_img.min()), float(raw_img.max())
    if hi > lo:
        disp = ((raw_img.astype(np.float32) - lo) / (hi - lo) * 255.0)
    else:
        disp = np.zeros_like(raw_img, dtype=np.float32)
    disp = disp.clip(0, 255).astype(np.uint8)
    disp_bgr = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    cmap = plt.get_cmap("turbo")
    colors_rgb = (np.array([cmap(i / max(n_segments - 1, 1))[:3] for i in range(n_segments)]) * 255)
    colors_bgr = colors_rgb[:, ::-1].astype(np.uint8)

    # ── Panel 2: colour-coded body ──────────────────────────────────────────
    colored = disp_bgr.copy()
    fg = label_img >= 0
    colored[fg] = colors_bgr[label_img[fg]]
    segmented = cv2.addWeighted(disp_bgr, 0.35, colored, 0.65, 0)

    # Thin 1-px centerline through all anchors (a polyline, not filled
    # circles at every point — that previously merged into one thick blob).
    anchors_int = np.round(anchors).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(segmented, [anchors_int], isClosed=False,
                  color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    # Orientation markers: hollow ring at the head (segment 0), filled dot at
    # the tail (last segment).
    hx, hy = anchors[0]
    cv2.circle(segmented, (int(round(hx)), int(round(hy))), 4, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    tx, ty = anchors[-1]
    cv2.circle(segmented, (int(round(tx)), int(round(ty))), 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    # ── Panel 3: segment-division lines on white ────────────────────────────
    # A boundary pixel is any pixel whose right or bottom neighbour belongs
    # to a different label (including -1 = background), which traces both
    # the worm outline and every internal segment-to-segment divider.
    h, w = label_img.shape
    boundary = np.zeros((h, w), dtype=bool)
    boundary[:, :-1] |= label_img[:, :-1] != label_img[:, 1:]
    boundary[:-1, :] |= label_img[:-1, :] != label_img[1:, :]
    divisions = np.full((h, w, 3), 255, dtype=np.uint8)
    divisions[boundary] = (0, 0, 0)

    # ── Compose "Raw" | "Segmented" | "Segment divisions" side by side ─────
    divider_w = 2
    caption_h = 18
    colorbar_h = 10
    colorbar_pad = 4
    tick_h = 14
    footer_h = colorbar_pad + colorbar_h + tick_h

    panels = [("Raw", disp_bgr), ("Segmented", segmented), ("Segment divisions", divisions)]
    total_w = w * len(panels) + divider_w * (len(panels) - 1)
    canvas = np.full((caption_h + h + footer_h, total_w, 3), 255, dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    x = 0
    panel_x: dict[str, int] = {}
    for i, (label, panel_img) in enumerate(panels):
        panel_x[label] = x
        canvas[caption_h:caption_h + h, x:x + w] = panel_img
        # Shrink the caption font if needed so it fits within this panel's
        # width (narrow crops can be narrower than a label like "Segment
        # divisions" at the default font scale).
        font_scale = 0.4
        (text_w, _), _ = cv2.getTextSize(label, font, font_scale, 1)
        max_text_w = w - 4
        if text_w > max_text_w > 0:
            font_scale *= max_text_w / text_w
            (text_w, _), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.putText(canvas, label, (x + max(0, (w - text_w) // 2), caption_h - 5),
                    font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        x += w
        if i < len(panels) - 1:
            canvas[caption_h:caption_h + h, x:x + divider_w] = (180, 180, 180)
            x += divider_w

    # ── Colour bar (segment index 0 -> n_segments-1) under the "Segmented"
    # panel only, with endpoint tick labels. ────────────────────────────────
    sx = panel_x["Segmented"]
    grad_frac = np.linspace(0.0, 1.0, w)
    grad_rgb = cmap(grad_frac)[:, :3] * 255
    grad_bgr = grad_rgb[:, ::-1].astype(np.uint8)
    bar = np.tile(grad_bgr[np.newaxis, :, :], (colorbar_h, 1, 1))
    by0 = caption_h + h + colorbar_pad
    canvas[by0:by0 + colorbar_h, sx:sx + w] = bar
    cv2.rectangle(canvas, (sx, by0), (sx + w - 1, by0 + colorbar_h - 1), (0, 0, 0), 1)

    ty_text = by0 + colorbar_h + 10
    cv2.putText(canvas, "0", (sx, ty_text), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    last_label = str(n_segments - 1)
    (lw, _), _ = cv2.getTextSize(last_label, font, 0.35, 1)
    cv2.putText(canvas, last_label, (sx + w - lw, ty_text), font, 0.35, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def resolve_raw_image_path(txt_path: Path, image_dir: Path) -> Path:
    """Find the raw image matching a ``..._roi_<id>.txt`` file's source image stem."""
    m = re.match(r"^(.+)_roi_\d+$", txt_path.stem)
    image_stem = m.group(1) if m else txt_path.stem
    for ext in RAW_IMAGE_EXTENSIONS:
        candidate = image_dir / (image_stem + ext)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No raw image found for stem '{image_stem}' in {image_dir} "
        f"(tried extensions {RAW_IMAGE_EXTENSIONS})"
    )


def load_raw_image(image_path: Path) -> np.ndarray:
    """Load the raw image preserving its native bit depth (e.g. uint16 TIFF)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_and_plot_intensity_profile(
    txt_path: Path,
    image_path: Path,
    output_dir: Path,
    n_segments: int = 100,
    pad_px: int = 20,
) -> dict:
    """Run the full intensity-profiling pipeline for one worm and save outputs.

    Returns a dict with ``mean_intensity``, ``std_intensity``, ``pixel_counts``
    (each ``(n_segments,)`` arrays) for programmatic use/testing.
    """
    contours_norm, skeleton_norm = parse_contour_skeleton_txt(txt_path)
    if skeleton_norm is None or len(skeleton_norm) < 2:
        raise ValueError(f"No usable skeleton found in {txt_path}")
    if not contours_norm:
        raise ValueError(f"No contour found in {txt_path}")

    raw_img = load_raw_image(image_path)
    h, w = raw_img.shape[:2]

    # Denormalise (fractions of full original image) -> full-image pixel coords.
    contours_px = [c * np.array([w, h]) for c in contours_norm]
    skeleton_px = skeleton_norm * np.array([w, h])

    if len(contours_px) > 1:
        warnings.warn(
            f"[{txt_path.name}] {len(contours_px)} contours found; expected 1 for an "
            "intact, non-fragmented worm. Using the largest by area.",
            RuntimeWarning, stacklevel=2,
        )
        contours_px = [max(
            contours_px,
            key=lambda c: cv2.contourArea(c.astype(np.float32).reshape(-1, 1, 2)),
        )]

    # Crop to the worm's bounding box (+ padding) for speed and a focused overlay.
    all_pts = np.concatenate(contours_px + [skeleton_px], axis=0)
    x0 = max(0, int(np.floor(all_pts[:, 0].min())) - pad_px)
    y0 = max(0, int(np.floor(all_pts[:, 1].min())) - pad_px)
    x1 = min(w, int(np.ceil(all_pts[:, 0].max())) + pad_px)
    y1 = min(h, int(np.ceil(all_pts[:, 1].max())) + pad_px)

    raw_crop = raw_img[y0:y1, x0:x1]
    contours_crop = [c - np.array([x0, y0]) for c in contours_px]
    skeleton_crop = skeleton_px - np.array([x0, y0])

    mask = rasterize_mask(contours_crop, raw_crop.shape[:2])
    anchors = resample_polyline(skeleton_crop, n_segments)
    label_img = assign_segments(mask, anchors)

    mean_intensity, std_intensity, pixel_counts = compute_intensity_profile(
        raw_crop, label_img, n_segments,
    )
    n_empty = int(np.sum(pixel_counts == 0))
    if n_empty > 0:
        warnings.warn(
            f"[{txt_path.name}] {n_empty}/{n_segments} body segments contained no "
            "mask pixels (NaN in the intensity profile).",
            RuntimeWarning, stacklevel=2,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = txt_path.stem
    plot_path = output_dir / f"{stem}_intensity_profile.png"
    overlay_path = output_dir / f"{stem}_segments_overlay.png"

    plot_intensity_profile(
        mean_intensity, std_intensity, plot_path,
        title=f"Body intensity profile — {stem} (n={n_segments} segments)",
    )
    render_segment_overlay(raw_crop, label_img, anchors, n_segments, overlay_path)

    return {
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "pixel_counts": pixel_counts,
        "plot_path": plot_path,
        "overlay_path": overlay_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot mean pixel intensity along the worm body skeleton, "
                    "from existing contour_skeleton_txt file(s) + raw image(s)."
    )
    txt_group = parser.add_mutually_exclusive_group(required=True)
    txt_group.add_argument("--txt-path", type=str, default=None,
                        help="Path to a single *_roi_<id>.txt contour/skeleton "
                             "file produced by skeletonize_worms.py.")
    txt_group.add_argument("--txt-dir", type=str, default=None,
                        help="Directory of *.txt contour/skeleton files to "
                             "process in batch (e.g. contour_skeleton_txt/).")
    parser.add_argument("--image-path", type=str, default=None,
                        help="Path to the raw source image (--txt-path mode only). "
                             "If omitted, --image-dir is used to auto-resolve it.")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="Directory containing the raw source image(s) "
                             "(auto-resolved per txt file from its image stem). "
                             "Required for --txt-dir.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to write the plot and overlay image(s) to. "
                             "Default: <txt file(s) parent>/intensity_profile")
    parser.add_argument("--n-segments", type=int, default=100,
                        help="Number of body-coordinate segments (default 100).")
    parser.add_argument("--pad-px", type=int, default=20,
                        help="Padding (px) around the worm's bounding box when "
                             "cropping for processing/overlay (default 20).")
    args = parser.parse_args()

    if args.txt_path:
        txt_paths = [Path(args.txt_path)]
    else:
        txt_dir = Path(args.txt_dir)
        txt_paths = sorted(txt_dir.glob("*.txt"))
        if not txt_paths:
            parser.error(f"No .txt files found in {txt_dir}")

    if args.txt_dir and not args.image_dir:
        parser.error("--image-dir is required when using --txt-dir.")

    n_ok, n_failed = 0, 0
    iterator = tqdm(txt_paths, desc="Intensity profiling") if len(txt_paths) > 1 else txt_paths
    for txt_path in iterator:
        try:
            if args.image_path:
                image_path = Path(args.image_path)
            else:
                image_path = resolve_raw_image_path(txt_path, Path(args.image_dir))

            output_dir = (
                Path(args.output_dir) if args.output_dir
                else txt_path.parent / "intensity_profile"
            )

            result = compute_and_plot_intensity_profile(
                txt_path, image_path, output_dir,
                n_segments=args.n_segments, pad_px=args.pad_px,
            )
            n_ok += 1
            if len(txt_paths) == 1:
                print(f"Intensity plot  -> {result['plot_path']}")
                print(f"Segment overlay -> {result['overlay_path']}")
        except Exception as exc:
            n_failed += 1
            warnings.warn(f"[{txt_path.name}] Skipped: {exc}", RuntimeWarning, stacklevel=2)

    if len(txt_paths) > 1:
        print(f"Done: {n_ok} succeeded, {n_failed} failed (of {len(txt_paths)} total).")


if __name__ == "__main__":
    main()
