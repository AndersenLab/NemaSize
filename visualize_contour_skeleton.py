"""
Visualize contour and skeleton annotations on original raw images.

Reads normalised-coordinate .txt files produced by skeletonize_worms.py
(``save_contour_skeleton_txt`` mode) and draws the contour outlines and
skeleton centerlines for every ROI onto the corresponding 16-bit TIF
source image (converted to 8-bit for display).

Usage
-----
    python visualize_contour_skeleton.py \
        --txt-dir   path/to/contour_skeleton_txt \
        --image-dir path/to/raw_images \
        --output-dir path/to/output
"""

import re
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_contour_skeleton_txt(
    txt_path: str | Path,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Parse a contour/skeleton .txt file.

    Returns
    -------
    contours : list of (N, 2) float arrays – normalised (x, y) per contour.
    skeleton : (M, 2) float array of normalised (x, y), or None.
    """
    contours: list[list[list[float]]] = []
    skeleton_pts: list[list[float]] = []
    current_section: str | None = None

    with open(txt_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line == "[CONTOUR]":
                current_section = "contour"
                contours.append([])
                continue
            if line == "[SKELETON]":
                current_section = "skeleton"
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            x, y = float(parts[0]), float(parts[1])
            if current_section == "contour" and contours:
                contours[-1].append([x, y])
            elif current_section == "skeleton":
                skeleton_pts.append([x, y])

    contour_arrays = [np.array(c, dtype=np.float64) for c in contours if c]
    skeleton = np.array(skeleton_pts, dtype=np.float64) if skeleton_pts else None
    return contour_arrays, skeleton


def group_txt_by_image(txt_dir: Path) -> dict[str, list[Path]]:
    """Group ROI .txt files by their source image stem.

    Filename pattern: ``{image_stem}_roi_{id}.txt``
    Returns ``{image_stem: [txt_path, ...]}``.
    """
    groups: dict[str, list[Path]] = {}
    for txt_path in sorted(txt_dir.glob("*.txt")):
        stem = txt_path.stem
        # Strip trailing _roi_<digits>
        m = re.match(r"^(.+)_roi_\d+$", stem)
        if m:
            image_stem = m.group(1)
        else:
            image_stem = stem
        groups.setdefault(image_stem, []).append(txt_path)
    return groups


def load_16bit_tif_as_8bit(image_path: Path) -> np.ndarray:
    """Load a 16-bit TIF and convert to 8-bit BGR for visualisation."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    if img.dtype == np.uint16:
        # Normalise to 0-255 using the actual data range for best contrast
        lo, hi = float(img.min()), float(img.max())
        if hi > lo:
            img_f = (img.astype(np.float32) - lo) / (hi - lo) * 255.0
        else:
            img_f = np.zeros_like(img, dtype=np.float32)
        img = img_f.clip(0, 255).astype(np.uint8)

    # Ensure 3-channel BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

# Colour constants (BGR)
CONTOUR_COLOR  = (0, 255, 255)   # yellow – contour outline
SKELETON_COLOR = (0, 255, 0)     # green  – skeleton centerline
ENDPOINT_COLOR = (255, 100, 0)   # blue   – skeleton endpoints


def draw_annotations_on_image(
    img: np.ndarray,
    contours_list: list[list[np.ndarray]],
    skeletons: list[np.ndarray | None],
    roi_labels: list[str],
    contour_thickness: int = 2,
    skeleton_thickness: int = 2,
) -> np.ndarray:
    """Draw contour outlines and skeleton centerlines for all ROIs."""
    h, w = img.shape[:2]
    annotated = img.copy()

    for roi_idx, (contours, skeleton, label) in enumerate(
        zip(contours_list, skeletons, roi_labels)
    ):
        # Draw contours
        for cnt_norm in contours:
            # Denormalise: (x, y) → pixel (col, row)
            pts_px = cnt_norm.copy()
            pts_px[:, 0] *= w
            pts_px[:, 1] *= h
            pts_int = pts_px.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(annotated, [pts_int], isClosed=True,
                          color=CONTOUR_COLOR, thickness=contour_thickness,
                          lineType=cv2.LINE_AA)

        # Draw skeleton
        if skeleton is not None and len(skeleton) >= 2:
            skel_px = skeleton.copy()
            skel_px[:, 0] *= w
            skel_px[:, 1] *= h
            skel_int = skel_px.astype(np.int32)

            # Polyline
            cv2.polylines(annotated,
                          [skel_int.reshape(-1, 1, 2)],
                          isClosed=False,
                          color=SKELETON_COLOR,
                          thickness=skeleton_thickness,
                          lineType=cv2.LINE_AA)

            # Endpoint markers – blue filled circles at both ends
            for ex, ey in [(int(skel_int[0, 0]), int(skel_int[0, 1])),
                           (int(skel_int[-1, 0]), int(skel_int[-1, 1]))]:
                cv2.circle(annotated, (ex, ey), 5, ENDPOINT_COLOR, -1, lineType=cv2.LINE_AA)
                cv2.circle(annotated, (ex, ey), 5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # ROI label near first contour centroid (disabled)
        # if contours:
        #     cx = int(contours[0][:, 0].mean() * w)
        #     cy = int(contours[0][:, 1].mean() * h)
        #     cv2.putText(annotated, label, (cx + 8, cy - 8),
        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, CONTOUR_COLOR, 1, cv2.LINE_AA)

    return annotated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Configuration ──────────────────────────────────────────────────────
    txt_dir    = Path(r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5\contour_skeleton_txt")
    image_dir  = Path(r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5\Example_images")
    output_dir = Path(r"C:\Users\lizih\Dropbox\Publication\NemaSize\Figures\Fig5\annotated_images")

    contour_thickness  = 1
    skeleton_thickness = 1
    # ──────────────────────────────────────────────────────────────────────

    output_dir.mkdir(parents=True, exist_ok=True)

    groups = group_txt_by_image(txt_dir)
    if not groups:
        print("No .txt files found in", txt_dir)
        return

    print(f"Found {sum(len(v) for v in groups.values())} ROI annotations "
          f"across {len(groups)} source images\n")

    # Supported extensions for raw images
    extensions = [".tif", ".tiff", ".TIF", ".TIFF", ".png", ".jpg", ".jpeg"]

    for image_stem, txt_paths in tqdm(groups.items(), desc="Processing images"):
        # Find the matching raw image
        img_path = None
        for ext in extensions:
            candidate = image_dir / (image_stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            print(f"  WARNING: no raw image found for '{image_stem}', skipping")
            continue

        img = load_16bit_tif_as_8bit(img_path)

        contours_list: list[list[np.ndarray]] = []
        skeletons: list[np.ndarray | None] = []
        roi_labels: list[str] = []

        for txt_path in sorted(txt_paths):
            contours, skeleton = parse_contour_skeleton_txt(txt_path)
            contours_list.append(contours)
            skeletons.append(skeleton)
            # Extract ROI id from filename
            m = re.search(r"roi_(\d+)", txt_path.stem)
            roi_labels.append(f"roi_{m.group(1)}" if m else txt_path.stem)

        annotated = draw_annotations_on_image(
            img, contours_list, skeletons, roi_labels,
            contour_thickness=contour_thickness,
            skeleton_thickness=skeleton_thickness,
        )

        out_path = output_dir / (image_stem + "_annotated.jpg")
        cv2.imwrite(str(out_path), annotated)

    print(f"\nDone – annotated images saved to {output_dir}")


if __name__ == "__main__":
    main()
