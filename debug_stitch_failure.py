"""
debug_stitch_failure.py
=======================
Diagnose WHY `stitch_centerline_fragments` fails for a single image.

Usage:
    python debug_stitch_failure.py

The script:
  1. Runs the same YOLO inference as skeletonize_worms.py.
  2. Computes mask fragments with the same parameters.
  3. For every ordered pair (A → B) and all 4 flip combinations:
       • prints the gap distance, head/tail coords, and angle cosines
       • reports which constraint(s) killed the connection
  4. Calls `stitch_centerline_fragments` and prints the final verdict.
  5. Saves a multi-panel debug image:
       Panel 1  – raw image + all fragment masks (coloured)
       Panel 2  – each fragment's skeleton + head (●) / tail (■) arrows
       Panel 3  – gap-distance matrix heat-map (all 4 flip orientations, per pair)
       Panel 4  – angle-cosine matrix heat-map (same layout)
       Panel 5  – final stitched (or fallback) centerline overlay
"""

import itertools
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpecFromSubplotSpec
from skimage.measure import label as sk_label
from ultralytics import YOLO

# ── Import helpers from skeletonize_worms ─────────────────────────────────
# All the heavy lifting is already implemented there; we just reuse it.
sys.path.insert(0, os.path.dirname(__file__))
from skeletonize_worms import (
    _gap_connection_cost,
    _gap_failure_reason,
    _skeletonize_mask_fragments,
    find_center_worm_idx,
    stitch_centerline_fragments,
    draw_centerline_on_image,
)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION  ─ edit these to match your skeletonize_worms.py settings
# ═══════════════════════════════════════════════════════════════════════════
IMAGE_PATH    = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Skeleton_test\test_img\images\256_png.rf.18d3c3ddc6e345fc5391481cf087f656_roi_45.jpg"
MODEL_PATH    = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt"
OUTPUT_DIR    = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Skeleton_test\skeleton_output\stitch_debug"

CONF_THRESHOLD       = 0.25
N_CENTERLINE_POINTS  = 200
CONTOUR_SMOOTH_SIGMA = 6.0
MAX_GAP_PX           = 100.0
MAX_JOINT_ANGLE_DEG  = 110.0
TANGENT_PATH_LEN_PX  = 50   # arc length (px) over which tangent direction is estimated
SKELETON_PX_PER_POINT = 10.0   # px; 1 skeleton point per N px of arc length
MIN_FRAGMENT_PIXELS  = 200
HOLE_FILL_RATIO      = 0.05
HOLE_CIRC_WEIGHT     = 0.75
RING_DILATION_RADIUS = 3
# ═══════════════════════════════════════════════════════════════════════════

FRAG_COLORS_BGR = [
    (0,   200, 255),   # fragment 0 – amber/yellow
    (0,   255,   0),   # fragment 1 – green
    (255,   0, 255),   # fragment 2 – magenta
    (255, 128,   0),   # fragment 3 – blue-ish
    (0,   128, 255),   # fragment 4 – orange
]
FRAG_COLORS_RGB = [(b/255, g/255, r/255) for b, g, r in FRAG_COLORS_BGR]


def _endpoint_coords(cl: np.ndarray, flip: bool):
    """Return (tail_rc, head_rc) after applying flip."""
    tail = cl[0].copy()  if flip else cl[-1].copy()
    head = cl[-1].copy() if flip else cl[0].copy()
    return tail, head


def diagnose_stitch(fragments, cos_thresh, max_gap_px, tangent_path_len_px, px_per_point):
    """
    Print a full diagnostic table for all ordered A→B pairs and all 4 flip
    combinations.  Returns a dict with per-pair results for plotting.
    """
    n_tangent_steps = max(1, round(tangent_path_len_px / px_per_point))
    k = len(fragments)
    pair_results = {}   # (i, j) → list of dicts (one per flip combo)

    sep = "─" * 78
    print(f"\n{'STITCH DIAGNOSTICS':^78}")
    print(sep)
    print(f"  max_gap_px = {max_gap_px:.0f} px   |   "
          f"max_joint_angle = {np.degrees(np.arccos(cos_thresh)):.0f}°   |   "
          f"tangent_path_len = {tangent_path_len_px:.0f} px   n_tangent_steps = {n_tangent_steps}")
    print(sep)

    for i, j in itertools.permutations(range(k), 2):
        cl_a, _ = fragments[i]
        cl_b, _ = fragments[j]
        rows = []
        for a_flip, b_flip in itertools.product([False, True], repeat=2):
            tail_a = cl_a[0] if a_flip else cl_a[-1]
            head_b = cl_b[-1] if b_flip else cl_b[0]
            gap_vec  = head_b - tail_a
            gap_dist = float(np.linalg.norm(gap_vec))

            # Outgoing direction of A
            n_a = min(n_tangent_steps, len(cl_a) - 1)
            dir_a = (cl_a[0] - cl_a[n_a]) if a_flip else (cl_a[-1] - cl_a[-1 - n_a])
            norm_a = float(np.linalg.norm(dir_a))
            cos_a = float(np.dot(dir_a / norm_a, gap_vec / (gap_dist + 1e-9))) if norm_a > 1e-6 and gap_dist > 1e-6 else 1.0

            # Incoming direction of B
            n_b = min(n_tangent_steps, len(cl_b) - 1)
            dir_b = (cl_b[-1 - n_b] - cl_b[-1]) if b_flip else (cl_b[n_b] - cl_b[0])
            norm_b = float(np.linalg.norm(dir_b))
            cos_b = float(np.dot(dir_b / norm_b, gap_vec / (gap_dist + 1e-9))) if norm_b > 1e-6 and gap_dist > 1e-6 else 1.0

            cost   = _gap_connection_cost(cl_a, a_flip, cl_b, b_flip, cos_thresh, max_gap_px, tangent_path_len_px, px_per_point)
            passed = np.isfinite(cost)

            dist_ok  = gap_dist <= max_gap_px
            angle_a_ok = cos_a >= cos_thresh
            angle_b_ok = cos_b >= cos_thresh

            fails = []
            if not dist_ok:   fails.append(f"dist={gap_dist:.1f}>{max_gap_px:.0f}")
            if not angle_a_ok: fails.append(f"cos_A={cos_a:.3f}<{cos_thresh:.3f}")
            if not angle_b_ok: fails.append(f"cos_B={cos_b:.3f}<{cos_thresh:.3f}")
            status = "✓ PASS" if passed else f"✗ FAIL ({', '.join(fails)})"

            rows.append({
                "a_flip": a_flip, "b_flip": b_flip,
                "gap_dist": gap_dist,
                "cos_a": cos_a, "cos_b": cos_b,
                "cost": cost, "passed": passed,
                "tail_a": tail_a, "head_b": head_b,
            })

        pair_results[(i, j)] = rows

        # Print table for this pair
        print(f"\n  Fragment {i} → Fragment {j}")
        print(f"  {'a_flip':>6}  {'b_flip':>6}  {'gap(px)':>8}  {'cos_A':>7}  {'cos_B':>7}  {'cost':>12}  status")
        print(f"  {'──────':>6}  {'──────':>6}  {'───────':>8}  {'─────':>7}  {'─────':>7}  {'──────────':>12}  ──────────────────────")
        for r in rows:
            cost_str = f"{r['cost']:.1f}" if np.isfinite(r['cost']) else "∞"
            print(f"  {str(r['a_flip']):>6}  {str(r['b_flip']):>6}  {r['gap_dist']:>8.1f}  "
                  f"{r['cos_a']:>7.3f}  {r['cos_b']:>7.3f}  {cost_str:>12}  {('✓' if r['passed'] else '✗ FAIL')}")

    print(f"\n{sep}")
    return pair_results


def make_gap_heatmaps(fragments, pair_results):
    """
    Build a (k, k, 4) array of gap distances and angle cosines
    for visualisation.  The 4 flip combinations are indexed as:
      0: (F,F)  1: (F,T)  2: (T,F)  3: (T,T)
    """
    k = len(fragments)
    dists   = np.full((k, k, 4), np.nan)
    cos_min = np.full((k, k, 4), np.nan)  # min of cos_a, cos_b

    flip_combos = [(False, False), (False, True), (True, False), (True, True)]
    for (i, j), rows in pair_results.items():
        for fi, (a_flip, b_flip) in enumerate(flip_combos):
            for r in rows:
                if r["a_flip"] == a_flip and r["b_flip"] == b_flip:
                    dists  [i, j, fi] = r["gap_dist"]
                    cos_min[i, j, fi] = min(r["cos_a"], r["cos_b"])
    return dists, cos_min


def _draw_skeleton_panel(ax, img_rgb, fragments, title="Fragments & Skeletons"):
    ax.imshow(img_rgb)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")
    for fi, (cl, _) in enumerate(fragments):
        color = FRAG_COLORS_RGB[fi % len(FRAG_COLORS_RGB)]
        # Skeleton path
        ax.plot(cl[:, 1], cl[:, 0], "-", color=color, linewidth=1.5, alpha=0.85)
        # Head = circle, Tail = square
        ax.plot(cl[0, 1],  cl[0, 0],  "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=0.8, label=f"Frag {fi} head")
        ax.plot(cl[-1, 1], cl[-1, 0], "s", color=color, markersize=6, markeredgecolor="white", markeredgewidth=0.8)
        # Fragment index label near the midpoint
        mid = len(cl) // 2
        ax.text(cl[mid, 1] + 3, cl[mid, 0] - 3, f"F{fi}", color=color,
                fontsize=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, ec="none"))


def _find_best_pair(pair_results):
    """Return the (i, j) key whose rows contain the globally shortest gap distance."""
    best_key  = None
    best_dist = np.inf
    for (i, j), rows in pair_results.items():
        for r in rows:
            if r["gap_dist"] < best_dist:
                best_dist = r["gap_dist"]
                best_key  = (i, j)
    return best_key


def _draw_gap_panel(sub_ax, img_rgb, cl_a, cl_b, i, j,
                    r_row, fl_label,
                    cos_thresh, max_gap_px, tangent_path_len_px, px_per_point,
                    zoom_pad=80, arrow_len=45):
    """
    Draw one zoomed gap panel into *sub_ax* for a single flip combination.

    Draws:
      • Cropped image centred on the gap region
      • Skeleton context lines for each fragment
      • dir_A arrow  — outgoing tangent of A at its connection endpoint
      • White dashed gap arrow (A-endpoint → B-endpoint) with distance label
      • dir_B arrow  — expected incoming tangent of B at its endpoint
      • Angle arc + degree label at A (dir_A vs gap vector)
      • Angle arc + degree label at B (gap vector vs dir_B)
      • Coloured spine: green = PASS, red = FAIL
    """
    n_tangent_steps = max(1, round(tangent_path_len_px / px_per_point))
    color_a  = FRAG_COLORS_RGB[i % len(FRAG_COLORS_RGB)]
    color_b  = FRAG_COLORS_RGB[j % len(FRAG_COLORS_RGB)]
    a_flip   = r_row["a_flip"]
    b_flip   = r_row["b_flip"]
    tail_a   = r_row["tail_a"]
    head_b   = r_row["head_b"]
    gap_dist = r_row["gap_dist"]
    cos_a    = r_row["cos_a"]
    cos_b    = r_row["cos_b"]
    passed   = r_row["passed"]

    # ── Skeleton context segments ─────────────────────────────────────────
    n_ctx   = min(n_tangent_steps + 8, len(cl_a))
    seg_a   = cl_a[:n_ctx]    if a_flip else cl_a[-n_ctx:]
    n_ctx_b = min(n_tangent_steps + 8, len(cl_b))
    seg_b   = cl_b[-n_ctx_b:] if b_flip else cl_b[:n_ctx_b]

    # ── Crop bounds ───────────────────────────────────────────────────────
    all_pts = np.vstack([seg_a, seg_b, [tail_a], [head_b]])
    r0 = max(int(all_pts[:, 0].min()) - zoom_pad, 0)
    r1 = min(int(all_pts[:, 0].max()) + zoom_pad, img_rgb.shape[0] - 1)
    c0 = max(int(all_pts[:, 1].min()) - zoom_pad, 0)
    c1 = min(int(all_pts[:, 1].max()) + zoom_pad, img_rgb.shape[1] - 1)

    crop = img_rgb[r0:r1 + 1, c0:c1 + 1]
    sub_ax.imshow(crop, extent=[c0, c1, r1, r0], origin="upper")
    sub_ax.set_xlim(c0, c1)
    sub_ax.set_ylim(r1, r0)   # row increases downward

    # ── Skeleton context lines ────────────────────────────────────────────
    sub_ax.plot(seg_a[:, 1], seg_a[:, 0], "-", color=color_a, lw=2.2, alpha=0.9,  zorder=3)
    sub_ax.plot(seg_b[:, 1], seg_b[:, 0], "-", color=color_b, lw=2.2, alpha=0.9,  zorder=3)

    # Endpoint dots
    sub_ax.plot(tail_a[1], tail_a[0], "o", color=color_a, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    sub_ax.plot(head_b[1], head_b[0], "o", color=color_b, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    sub_ax.text(tail_a[1] + 3, tail_a[0] - 4, f"F{i}", color=color_a,
                fontsize=8, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"))
    sub_ax.text(head_b[1] + 3, head_b[0] + 6, f"F{j}", color=color_b,
                fontsize=8, fontweight="bold", zorder=7,
                bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.55, ec="none"))

    # ── Direction vectors ─────────────────────────────────────────────────
    n_a        = min(n_tangent_steps, len(cl_a) - 1)
    dir_a_raw  = (cl_a[0] - cl_a[n_a])       if a_flip else (cl_a[-1] - cl_a[-1 - n_a])
    norm_a     = float(np.linalg.norm(dir_a_raw))
    dir_a_unit = dir_a_raw / norm_a if norm_a > 1e-6 else np.array([0.0, 1.0])

    n_b        = min(n_tangent_steps, len(cl_b) - 1)
    dir_b_raw  = (cl_b[-1 - n_b] - cl_b[-1]) if b_flip else (cl_b[n_b] - cl_b[0])
    norm_b     = float(np.linalg.norm(dir_b_raw))
    dir_b_unit = dir_b_raw / norm_b if norm_b > 1e-6 else np.array([0.0, 1.0])

    gap_vec  = head_b - tail_a
    gap_d    = float(np.linalg.norm(gap_vec))
    gap_unit = gap_vec / gap_d if gap_d > 1e-6 else np.array([0.0, 1.0])

    def _arrow(origin_rc, dir_rc, color, label, lw=2.2):
        sub_ax.annotate("",
            xy=(origin_rc[1] + dir_rc[1] * arrow_len,
                origin_rc[0] + dir_rc[0] * arrow_len),
            xytext=(origin_rc[1], origin_rc[0]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=12),
            zorder=7,
        )
        lbl_c = origin_rc[1] + dir_rc[1] * arrow_len * 1.45
        lbl_r = origin_rc[0] + dir_rc[0] * arrow_len * 1.45
        sub_ax.text(lbl_c, lbl_r, label, color=color, fontsize=8,
                    ha="center", va="center", zorder=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7, ec="none"))

    _arrow(tail_a, dir_a_unit, color_a, "dir_A")
    _arrow(head_b, dir_b_unit, color_b, "dir_B")

    # Gap vector (white dashed)
    sub_ax.annotate("",
        xy=(head_b[1], head_b[0]),
        xytext=(tail_a[1], tail_a[0]),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2.0,
                        linestyle="dashed", mutation_scale=12),
        zorder=6,
    )
    mid_r = (tail_a[0] + head_b[0]) / 2
    mid_c = (tail_a[1] + head_b[1]) / 2
    sub_ax.text(mid_c, mid_r, f"{gap_d:.1f} px",
                color="white", fontsize=8, ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.7, ec="none"))

    # ── Angle arcs ────────────────────────────────────────────────────────
    arc_r = arrow_len * 0.45

    def _angle_arc(origin_rc, vec1_rc, vec2_rc, color, angle_deg, ok):
        a1   = float(np.degrees(np.arctan2(vec1_rc[0], vec1_rc[1])))
        a2   = float(np.degrees(np.arctan2(vec2_rc[0], vec2_rc[1])))
        diff = (a2 - a1 + 360) % 360
        if diff > 180:
            a1, a2 = a2, a1
            diff   = 360 - diff
        ts     = np.linspace(np.radians(a1), np.radians(a1 + diff), 40)
        arc_cc = origin_rc[1] + arc_r * np.cos(ts)
        arc_rr = origin_rc[0] + arc_r * np.sin(ts)
        sub_ax.plot(arc_cc, arc_rr, color=color, lw=1.6, alpha=0.9, zorder=7)
        t_mid  = np.radians(a1 + diff / 2)
        lbl_c  = origin_rc[1] + arc_r * 2.0 * np.cos(t_mid)
        lbl_r_ = origin_rc[0] + arc_r * 2.0 * np.sin(t_mid)
        lbl_color = "lime" if ok else "red"
        sub_ax.text(lbl_c, lbl_r_, f"{angle_deg:.0f}°",
                    color=lbl_color, fontsize=10, ha="center", va="center",
                    fontweight="bold", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black",
                              alpha=0.8, ec="none"))

    angle_A_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    angle_B_deg = float(np.degrees(np.arccos(np.clip(cos_b, -1, 1))))
    a_ok    = cos_a >= cos_thresh
    b_ok    = cos_b >= cos_thresh
    dist_ok = gap_dist <= max_gap_px

    _angle_arc(tail_a, dir_a_unit, gap_unit,   color_a, angle_A_deg, a_ok)
    _angle_arc(head_b, gap_unit,   dir_b_unit, color_b, angle_B_deg, b_ok)

    # ── Panel title and coloured spine ────────────────────────────────────
    fails = []
    if not dist_ok: fails.append(f"dist={gap_d:.0f}>{max_gap_px:.0f}px")
    if not a_ok:    fails.append(f"ang_A={angle_A_deg:.0f}°")
    if not b_ok:    fails.append(f"ang_B={angle_B_deg:.0f}°")
    status_str = "PASS" if passed else "FAIL: " + ", ".join(fails)
    status_col = "lime" if passed else "red"

    sub_ax.set_title(
        f"{fl_label}   [{status_str}]   gap={gap_d:.1f}px",
        fontsize=9, color=status_col, pad=5,
    )
    sub_ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in sub_ax.spines.values():
        spine.set_edgecolor(status_col)
        spine.set_linewidth(2.5)


def _save_tangent_detail_figure(out_path, img_stem, img_rgb, fragments,
                                pair_results, cos_thresh, max_gap_px,
                                tangent_path_len_px, px_per_point,
                                zoom_pad=100, arrow_len=55):
    """
    Standalone figure for the flip combo with the shortest gap distance.

    Shows ONLY the information that determines dir_A and dir_B:
      • Cropped image tightly around the gap region
      • The tangent_path_len_px skeleton points of F0 used for chord averaging
        (coloured dots grading from dim to bright toward the endpoint)
      • A chord arrow from anchor-point → tail endpoint  (= dir_A direction)
      • The tail endpoint of F0 (large star marker)
      • Dashed gap connection line with distance label
      • The connection endpoint of F1 (large star marker)
      • The tangent_path_len_px skeleton points of F1 used for chord averaging
      • A chord arrow from anchor-point → connection endpoint (= dir_B direction)
      • dir_A and dir_B unit-vector arrows (scaled, labelled)
      • Angle arc + degree label at each endpoint
    """
    best_pair_key = _find_best_pair(pair_results)
    if best_pair_key is None:
        return None

    n_tangent_steps = max(1, round(tangent_path_len_px / px_per_point))
    i, j    = best_pair_key
    rows    = pair_results[best_pair_key]
    cl_a, _ = fragments[i]
    cl_b, _ = fragments[j]
    color_a  = FRAG_COLORS_RGB[i % len(FRAG_COLORS_RGB)]
    color_b  = FRAG_COLORS_RGB[j % len(FRAG_COLORS_RGB)]

    # Pick the row with the shortest gap distance
    r_row   = min(rows, key=lambda r: r["gap_dist"])
    a_flip  = r_row["a_flip"]
    b_flip  = r_row["b_flip"]
    tail_a  = r_row["tail_a"]   # connection endpoint of F0 (row, col)
    head_b  = r_row["head_b"]   # connection endpoint of F1 (row, col)
    cos_a   = r_row["cos_a"]
    cos_b   = r_row["cos_b"]
    passed  = r_row["passed"]

    flip_label = f"a_flip={a_flip}, b_flip={b_flip}"

    # ── Tangent averaging windows ────────────────────────────────────
    n_a = min(n_tangent_steps, len(cl_a) - 1)
    n_b = min(n_tangent_steps, len(cl_b) - 1)

    if a_flip:
        # tail endpoint = cl_a[0]; chord = cl_a[n_a] -> cl_a[0]
        win_a       = cl_a[:n_a + 1]        # window used for chord [0..n_a]
        chord_a_end = cl_a[0]               # tail endpoint (= tail_a)
        chord_a_src = cl_a[n_a]             # far anchor of chord
    else:
        # tail endpoint = cl_a[-1]; chord = cl_a[-1-n_a] -> cl_a[-1]
        win_a       = cl_a[-(n_a + 1):]     # window used for chord
        chord_a_end = cl_a[-1]              # tail endpoint (= tail_a)
        chord_a_src = cl_a[-1 - n_a]        # far anchor of chord

    if b_flip:
        # head endpoint = cl_b[-1]; chord = cl_b[-1-n_b] -> cl_b[-1]
        win_b       = cl_b[-(n_b + 1):]     # window used for chord
        chord_b_end = cl_b[-1]              # connection endpoint (= head_b)
        chord_b_src = cl_b[-1 - n_b]        # far anchor of chord
    else:
        # head endpoint = cl_b[0]; chord = cl_b[0] -> cl_b[n_b]
        win_b       = cl_b[:n_b + 1]
        chord_b_end = cl_b[0]
        chord_b_src = cl_b[n_b]

    # ── Direction unit vectors ───────────────────────────────────────
    dir_a_raw  = chord_a_end - chord_a_src
    norm_a     = float(np.linalg.norm(dir_a_raw))
    dir_a_unit = dir_a_raw / norm_a if norm_a > 1e-6 else np.array([0.0, 1.0])

    dir_b_raw  = chord_b_src - chord_b_end
    norm_b     = float(np.linalg.norm(dir_b_raw))
    dir_b_unit = dir_b_raw / norm_b if norm_b > 1e-6 else np.array([0.0, 1.0])

    gap_vec  = head_b - tail_a
    gap_d    = float(np.linalg.norm(gap_vec))
    gap_unit = gap_vec / gap_d if gap_d > 1e-6 else np.array([0.0, 1.0])

    angle_A_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))
    angle_B_deg = float(np.degrees(np.arccos(np.clip(cos_b, -1, 1))))
    a_ok    = cos_a >= cos_thresh
    b_ok    = cos_b >= cos_thresh
    dist_ok = gap_d  <= max_gap_px

    # ── Crop bounds ─────────────────────────────────────────────────────
    all_pts = np.vstack([win_a, win_b, [tail_a], [head_b]])
    # also include arrow tips so they aren't clipped
    arrow_pts = np.array([
        tail_a  + dir_a_unit  * arrow_len * 1.6,
        head_b  + dir_b_unit  * arrow_len * 1.6,
        chord_a_src, chord_b_src,
    ])
    all_pts = np.vstack([all_pts, arrow_pts])
    r0 = max(int(all_pts[:, 0].min()) - zoom_pad, 0)
    r1 = min(int(all_pts[:, 0].max()) + zoom_pad, img_rgb.shape[0] - 1)
    c0 = max(int(all_pts[:, 1].min()) - zoom_pad, 0)
    c1 = min(int(all_pts[:, 1].max()) + zoom_pad, img_rgb.shape[1] - 1)

    # ── Figure ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#1a1a1a")

    fails = []
    if not dist_ok: fails.append(f"dist={gap_d:.0f}>{max_gap_px:.0f}px")
    if not a_ok:    fails.append(f"ang_A={angle_A_deg:.0f}\u00b0 > {np.degrees(np.arccos(cos_thresh)):.0f}\u00b0")
    if not b_ok:    fails.append(f"ang_B={angle_B_deg:.0f}\u00b0 > {np.degrees(np.arccos(cos_thresh)):.0f}\u00b0")
    status_str = "PASS" if passed else "FAIL: " + ", ".join(fails)
    status_col = "lime" if passed else "#ff4444"

    fig.suptitle(
        f"Tangent chord detail  —  F{i} vs F{j}  [{flip_label}]   gap={gap_d:.1f}px\n"
        f"max_gap={max_gap_px:.0f}px   threshold angle={np.degrees(np.arccos(cos_thresh)):.0f}\u00b0"
        f"   tangent_path={tangent_path_len_px:.0f}px  n_steps={n_tangent_steps}   │   {img_stem}\n"
        f"[{status_str}]",
        fontsize=12, fontweight="bold", color=status_col, y=0.99,
    )

    crop = img_rgb[r0:r1 + 1, c0:c1 + 1]
    ax.imshow(crop, extent=[c0, c1, r1, r0], origin="upper", zorder=0)
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_edgecolor(status_col)
        spine.set_linewidth(2.5)

    # ── Averaging window points ────────────────────────────────────────
    # Gradient alpha: dimmer farther from the endpoint, bright at the endpoint
    def _draw_window(win, color, endpoint_rc):
        n = len(win)
        for ki, pt in enumerate(win):
            # brightness gradient: 0.15 at far anchor, 0.9 at endpoint
            alpha = 0.15 + 0.75 * ki / max(n - 1, 1)
            ax.plot(pt[1], pt[0], "o", color=color, markersize=5,
                    alpha=alpha, zorder=4, markeredgewidth=0)
        # connect them as a faint line
        ax.plot(win[:, 1], win[:, 0], "-", color=color, lw=1.2,
                alpha=0.45, zorder=3)
        # label count
        ax.text(win[0, 1], win[0, 0], f" {n} pts",
                color=color, fontsize=7, alpha=0.7, zorder=5,
                bbox=dict(boxstyle="round,pad=0.15", fc="black",
                          alpha=0.4, ec="none"))

    _draw_window(win_a, color_a, tail_a)
    _draw_window(win_b, color_b, head_b)

    # ── Chord arrows ──────────────────────────────────────────────────
    # Chord of F0: anchor → tail endpoint (same direction as dir_A)
    ax.annotate("",
        xy=(chord_a_end[1], chord_a_end[0]),
        xytext=(chord_a_src[1], chord_a_src[0]),
        arrowprops=dict(arrowstyle="-|>", color=color_a, lw=1.8,
                        linestyle="dotted", mutation_scale=12),
        zorder=6,
    )
    mid_chord_a = (chord_a_src + chord_a_end) / 2
    ax.text(mid_chord_a[1] - 8, mid_chord_a[0] - 6, "chord A",
            color=color_a, fontsize=7, alpha=0.85, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, ec="none"))

    # Chord of F1: head endpoint → anchor (same direction as dir_B)
    ax.annotate("",
        xy=(chord_b_src[1], chord_b_src[0]),
        xytext=(chord_b_end[1], chord_b_end[0]),
        arrowprops=dict(arrowstyle="-|>", color=color_b, lw=1.8,
                        linestyle="dotted", mutation_scale=12),
        zorder=6,
    )
    mid_chord_b = (chord_b_src + chord_b_end) / 2
    ax.text(mid_chord_b[1] + 4, mid_chord_b[0] + 6, "chord B",
            color=color_b, fontsize=7, alpha=0.85, zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.5, ec="none"))

    # ── Endpoint markers ──────────────────────────────────────────────
    ax.plot(tail_a[1], tail_a[0], "*", color=color_a, markersize=16,
            markeredgecolor="white", markeredgewidth=0.8, zorder=8)
    ax.plot(head_b[1], head_b[0], "*", color=color_b, markersize=16,
            markeredgecolor="white", markeredgewidth=0.8, zorder=8)
    ax.text(tail_a[1] + 5, tail_a[0] - 7,
            f"F{i} endpoint\n(tail_a)",
            color=color_a, fontsize=8, fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65, ec="none"))
    ax.text(head_b[1] + 5, head_b[0] + 8,
            f"F{j} endpoint\n(head_b)",
            color=color_b, fontsize=8, fontweight="bold", zorder=9,
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65, ec="none"))

    # Anchor markers (far end of each chord)
    ax.plot(chord_a_src[1], chord_a_src[0], "D", color=color_a, markersize=7,
            alpha=0.7, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax.plot(chord_b_src[1], chord_b_src[0], "D", color=color_b, markersize=7,
            alpha=0.7, markeredgecolor="white", markeredgewidth=0.6, zorder=5)
    ax.text(chord_a_src[1] - 5, chord_a_src[0] - 6,
            f"cl_a[ep-{n_a}]",
            color=color_a, fontsize=7, alpha=0.75, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, ec="none"))
    ax.text(chord_b_src[1] - 5, chord_b_src[0] + 8,
            f"cl_b[ep-{n_b}]",
            color=color_b, fontsize=7, alpha=0.75, zorder=6,
            bbox=dict(boxstyle="round,pad=0.15", fc="black", alpha=0.45, ec="none"))

    # ── Gap connection line ────────────────────────────────────────────
    ax.annotate("",
        xy=(head_b[1], head_b[0]),
        xytext=(tail_a[1], tail_a[0]),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2.2,
                        linestyle="dashed", mutation_scale=14),
        zorder=6,
    )
    mid_gap = (tail_a + head_b) / 2
    ax.text(mid_gap[1] + 4, mid_gap[0], f"gap = {gap_d:.1f} px",
            color="white", fontsize=9, ha="left", va="center", zorder=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="black", alpha=0.75, ec="none"))

    # ── dir_A and dir_B unit-vector arrows ───────────────────────────
    def _dir_arrow(origin, unit, color, label, ok):
        tip = origin + unit * arrow_len
        ax.annotate("",
            xy=(tip[1], tip[0]),
            xytext=(origin[1], origin[0]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2.8,
                            mutation_scale=16),
            zorder=10,
        )
        lbl_pos = origin + unit * arrow_len * 1.5
        ax.text(lbl_pos[1], lbl_pos[0], label,
                color="lime" if ok else "red",
                fontsize=11, fontweight="bold", ha="center", va="center",
                zorder=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="black",
                          alpha=0.8, ec="none"))

    _dir_arrow(tail_a, dir_a_unit, color_a, "dir_A", a_ok)
    _dir_arrow(head_b, dir_b_unit, color_b, "dir_B", b_ok)

    # ── Angle arcs + degree labels ──────────────────────────────────
    arc_r = arrow_len * 0.5

    def _arc(origin, vec1, vec2, color, angle_deg, ok):
        a1   = float(np.degrees(np.arctan2(vec1[0], vec1[1])))
        a2   = float(np.degrees(np.arctan2(vec2[0], vec2[1])))
        diff = (a2 - a1 + 360) % 360
        if diff > 180:
            a1, a2 = a2, a1
            diff   = 360 - diff
        ts     = np.linspace(np.radians(a1), np.radians(a1 + diff), 60)
        arc_cc = origin[1] + arc_r * np.cos(ts)
        arc_rr = origin[0] + arc_r * np.sin(ts)
        ax.plot(arc_cc, arc_rr, "-", color=color, lw=2.5, alpha=0.95, zorder=8)
        t_mid  = np.radians(a1 + diff / 2)
        lbl_c  = origin[1] + arc_r * 2.2 * np.cos(t_mid)
        lbl_r_ = origin[0] + arc_r * 2.2 * np.sin(t_mid)
        ax.text(lbl_c, lbl_r_, f"{angle_deg:.0f}\u00b0",
                color="lime" if ok else "red",
                fontsize=16, fontweight="bold", ha="center", va="center",
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.3", fc="black",
                          alpha=0.85, ec=("lime" if ok else "red"),
                          linewidth=1.5))

    _arc(tail_a, dir_a_unit, gap_unit,   color_a, angle_A_deg, a_ok)
    _arc(head_b, gap_unit,   dir_b_unit, color_b, angle_B_deg, b_ok)

    # ── Legend ───────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=color_a,  label=f"F{i} skeleton (averaging window)"),
        mpatches.Patch(color=color_b,  label=f"F{j} skeleton (averaging window)"),
        mpatches.Patch(color="white",  label="gap connection (dashed)"),
        plt.Line2D([0], [0], color=color_a,  linestyle="dotted", lw=2,
                   label=f"chord A  → dir_A"),
        plt.Line2D([0], [0], color=color_b,  linestyle="dotted", lw=2,
                   label=f"chord B  → dir_B"),
        mpatches.Patch(color="#444444", label="star = connection endpoint"),
        mpatches.Patch(color="#444444", label="diamond = chord far anchor"),
    ]
    ax.legend(handles=legend_items, loc="lower left",
              fontsize=8, framealpha=0.8,
              facecolor="#222222", edgecolor="#888888", labelcolor="white")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _draw_gap_zoom(fig, ax, img_rgb, fragments, pair_results,
                   max_gap_px, cos_thresh, tangent_path_len_px, px_per_point,
                   zoom_pad=70, arrow_len=35):
    """Replace *ax* with a 2×2 sub-grid — one zoomed gap panel per flip combo."""
    best_pair_key = _find_best_pair(pair_results)
    if best_pair_key is None:
        ax.text(0.5, 0.5, "No pair to show", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        return

    i, j    = best_pair_key
    rows    = pair_results[best_pair_key]
    cl_a, _ = fragments[i]
    cl_b, _ = fragments[j]
    flip_labels = ["(F,F)", "(F,T)", "(T,F)", "(T,T)"]

    ax.set_visible(False)
    ss  = ax.get_subplotspec()
    gs2 = GridSpecFromSubplotSpec(2, 2, subplot_spec=ss, hspace=0.28, wspace=0.05)

    for fi, (r_row, fl_label) in enumerate(zip(rows, flip_labels)):
        sub_ax = fig.add_subplot(gs2[fi // 2, fi % 2])
        _draw_gap_panel(sub_ax, img_rgb, cl_a, cl_b, i, j,
                        r_row, fl_label,
                        cos_thresh, max_gap_px, tangent_path_len_px, px_per_point,
                        zoom_pad=zoom_pad, arrow_len=arrow_len)


def _save_gap_zoom_standalone(out_path, img_stem, img_rgb, fragments,
                               pair_results, max_gap_px, cos_thresh,
                               tangent_path_len_px, px_per_point,
                               zoom_pad=80, arrow_len=45):
    """
    Save a full-page standalone figure (2×2, one panel per flip combo)
    for the fragment pair with the shortest gap distance.
    """
    best_pair_key = _find_best_pair(pair_results)
    if best_pair_key is None:
        return None

    i, j    = best_pair_key
    rows    = pair_results[best_pair_key]
    cl_a, _ = fragments[i]
    cl_b, _ = fragments[j]
    flip_labels = [
        "(F,F)  a=False, b=False",
        "(F,T)  a=False, b=True",
        "(T,F)  a=True,  b=False",
        "(T,T)  a=True,  b=True",
    ]

    n_tangent_steps = max(1, round(tangent_path_len_px / px_per_point))
    any_pass  = any(r["passed"] for r in rows)
    title_col = "lime" if any_pass else "red"
    best_gap  = min(r["gap_dist"] for r in rows)

    fig, ax_grid = plt.subplots(2, 2, figsize=(14, 11))
    fig.patch.set_facecolor("#111111")
    fig.suptitle(
        f"Gap zoom — F{i} vs F{j}   (shortest gap = {best_gap:.1f} px)\n"
        f"max_gap={max_gap_px:.0f}px   max_angle={np.degrees(np.arccos(cos_thresh)):.0f}°"
        f"   tangent_path={tangent_path_len_px:.0f}px  n_steps={n_tangent_steps}   |   {img_stem}",
        fontsize=11, fontweight="bold", color=title_col, y=0.99,
    )

    for fi, (r_row, fl_label) in enumerate(zip(rows, flip_labels)):
        sub_ax = ax_grid[fi // 2, fi % 2]
        sub_ax.set_facecolor("#1a1a1a")
        _draw_gap_panel(sub_ax, img_rgb, cl_a, cl_b, i, j,
                        r_row, fl_label,
                        cos_thresh, max_gap_px, tangent_path_len_px, px_per_point,
                        zoom_pad=zoom_pad, arrow_len=arrow_len)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. Load image ──────────────────────────────────────────────────────
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"❌ Cannot read image: {IMAGE_PATH}")
        sys.exit(1)
    print(f"Image loaded: {IMAGE_PATH}")
    print(f"  Shape: {img.shape[1]}×{img.shape[0]} (W×H)")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # ── 2. YOLO inference ─────────────────────────────────────────────────
    print("\nLoading model and running inference...")
    model = YOLO(MODEL_PATH)
    results = model.predict(source=img, conf=CONF_THRESHOLD, verbose=False)[0]

    if results.masks is None or len(results.masks) == 0:
        print("❌ No detections — nothing to debug.")
        sys.exit(0)

    masks_raw  = results.masks.data.cpu().numpy()
    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
    confs      = results.boxes.conf.cpu().numpy()
    n_det = len(masks_raw)
    print(f"  {n_det} detection(s) found")

    masks_full = [
        cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        for m in masks_raw
    ]

    target_idx = find_center_worm_idx(boxes_xyxy, h, w)
    print(f"  Target worm index (closest to FOV centre): {target_idx}")

    # ── 3. Fragment extraction for the target worm ────────────────────────
    mask_bin = (masks_full[target_idx] > 0.5).astype(np.uint8)
    labeled_tmp, n_frags_raw = sk_label((mask_bin > 0).astype(np.uint8), connectivity=2, return_num=True)
    print(f"\n{'═'*60}")
    print(f"  Mask connected components (before min-pixel filter): {n_frags_raw}")
    print(f"  MIN_FRAGMENT_PIXELS = {MIN_FRAGMENT_PIXELS}")
    print(f"  SKELETON_PX_PER_POINT = {SKELETON_PX_PER_POINT} px  (1 point per {SKELETON_PX_PER_POINT:.0f} px of arc length)")
    for cid in range(1, n_frags_raw + 1):
        area = int((labeled_tmp == cid).sum())
        print(f"    component {cid}: {area} px  {'→ KEEP' if area >= MIN_FRAGMENT_PIXELS else '→ DROP (too small)'}")
    print(f"{'═'*60}")

    fragments = _skeletonize_mask_fragments(
        mask_bin,
        n_points_per_fragment=50,  # fallback only; px_per_point takes precedence
        px_per_point=SKELETON_PX_PER_POINT,
        smooth_factor=None,
        contour_smooth_sigma=CONTOUR_SMOOTH_SIGMA,
        min_fragment_pixels=MIN_FRAGMENT_PIXELS,
        hole_fill_ratio_threshold=HOLE_FILL_RATIO,
        hole_keep_circularity_weight=HOLE_CIRC_WEIGHT,
        ring_dilation_radius=RING_DILATION_RADIUS,
        debug_ring_mask_prefix=None,
        img_h=h,
        img_w=w,
    )

    print(f"\n  Skeletonizable fragments (after filtering): {len(fragments)}")
    for fi, (cl, wp) in enumerate(fragments):
        print(f"    Fragment {fi}: {len(cl)} skeleton points  "
              f"head=({cl[0,0]:.0f},{cl[0,1]:.0f})  "
              f"tail=({cl[-1,0]:.0f},{cl[-1,1]:.0f})  "
              f"mean_width={wp.mean():.1f}px")

    if len(fragments) == 0:
        print("\n❌ No fragments extracted. Skeletonization failed on all components.")
        sys.exit(0)

    if len(fragments) == 1:
        print("\n✓ Only 1 fragment — stitching is not needed.")
    else:
        # ── 4. Full stitch diagnostics ────────────────────────────────────
        cos_thresh = float(np.cos(np.radians(MAX_JOINT_ANGLE_DEG)))
        pair_results = diagnose_stitch(fragments, cos_thresh, MAX_GAP_PX, TANGENT_PATH_LEN_PX, SKELETON_PX_PER_POINT)

    # ── 5. Call the real stitch function and report result ─────────────────
    cl_out, wp_out, gap_ranges, gap_exceeded, failure_reason, frag_chain = \
        stitch_centerline_fragments(
            fragments,
            n_points=N_CENTERLINE_POINTS,
            max_gap_px=MAX_GAP_PX,
            max_joint_angle_deg=MAX_JOINT_ANGLE_DEG,
            tangent_path_len_px=TANGENT_PATH_LEN_PX,
            px_per_point=SKELETON_PX_PER_POINT,
        )

    print(f"\n{'═'*60}")
    print("  STITCH RESULT")
    print(f"{'═'*60}")
    if gap_exceeded:
        print(f"  ✗ STITCH FAILED  —  reason: '{failure_reason}'")
        print("  → Fallback: fragment 0 (closest to FOV centre) will be used alone.")
    elif cl_out is None:
        print("  ✗ STITCH RETURNED None (unexpected)")
    else:
        print(f"  ✓ STITCH SUCCEEDED")
        print(f"    Fragment chain : {frag_chain}")
        print(f"    Gap ranges     : {gap_ranges}")
    print(f"{'═'*60}\n")

    # ── 6. Build output figure ─────────────────────────────────────────────
    if len(fragments) == 1:
        n_cols = 3
        fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))
        axes = list(axes)
    else:
        n_cols = 3
        n_rows = 2
        fig, ax_grid = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
        axes = list(ax_grid.flatten())

    fig.suptitle(
        f"Stitch Debug — {os.path.basename(IMAGE_PATH)}\n"
        f"max_gap={MAX_GAP_PX}px  max_angle={MAX_JOINT_ANGLE_DEG}°  "
        f"{'STITCH FAILED: ' + failure_reason if gap_exceeded else 'STITCH SUCCEEDED'}",
        fontsize=10, fontweight="bold",
        color="red" if gap_exceeded else "green",
    )

    ax_idx = 0

    # ── Panel 1: Raw image + coloured mask overlays ──────────────────────
    ax = axes[ax_idx]; ax_idx += 1
    ax.imshow(img_rgb)
    ax.set_title("Mask components (all detections)", fontsize=9, fontweight="bold")
    ax.axis("off")

    # Draw all detections faintly, highlight target
    for mi, mf32 in enumerate(masks_full):
        m_bin = (mf32 > 0.5).astype(np.uint8)
        color_bgr = FRAG_COLORS_BGR[mi % len(FRAG_COLORS_BGR)]
        alpha = 0.45 if mi == target_idx else 0.15
        overlay = np.zeros_like(img, dtype=np.float32)
        overlay[m_bin > 0] = color_bgr
        comp = cv2.addWeighted(img.astype(np.float32), 1.0, overlay, alpha, 0)
        comp_rgb = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2RGB)
        ax.imshow(comp_rgb, alpha=(0.7 if mi == target_idx else 0.3))
        # Bounding box
        x1, y1, x2, y2 = boxes_xyxy[mi]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5 if mi == target_idx else 0.7,
            edgecolor=FRAG_COLORS_RGB[mi % len(FRAG_COLORS_RGB)],
            facecolor="none",
            linestyle="-" if mi == target_idx else "--",
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 3, f"Det {mi} conf={confs[mi]:.2f}{'  ← TARGET' if mi == target_idx else ''}",
                color=FRAG_COLORS_RGB[mi % len(FRAG_COLORS_RGB)],
                fontsize=6, fontweight="bold")

    # ── Panel 2: Fragment skeletons on target mask ────────────────────────
    ax = axes[ax_idx]; ax_idx += 1
    _draw_skeleton_panel(ax, img_rgb, fragments,
                         title=f"Skeletonised fragments ({len(fragments)} total)\n"
                               f"● head  ■ tail")

    # ── Panel 3: Zoomed gap panels (one per flip combo) ──────────────────
    ax = axes[ax_idx]; ax_idx += 1
    if len(fragments) > 1:
        _draw_gap_zoom(fig, ax, img_rgb, fragments, pair_results,
                       MAX_GAP_PX, cos_thresh, TANGENT_PATH_LEN_PX, SKELETON_PX_PER_POINT)
    else:
        ax.imshow(img_rgb)
        ax.set_title("Gap zoom (N/A — single fragment)", fontsize=9)
        ax.axis("off")

    # ── Row 2 (only if >1 fragment) ────────────────────────────────────────
    if len(fragments) > 1:
        # ── Panel 4: Per-pair gap distance table ──────────────────────────
        ax = axes[ax_idx]; ax_idx += 1
        k = len(fragments)
        flip_labels = ["(F→F)", "(F→T)", "(T→F)", "(T→T)"]
        cols_labels = [f"{i}→{j}  {fl}" for i, j in itertools.permutations(range(k), 2)
                                         for fl in flip_labels]
        gap_data = []
        angle_data = []
        pass_data = []
        row_labels = []
        for (i, j), rows in sorted(pair_results.items()):
            for fl_label, r in zip(flip_labels, rows):
                row_labels.append(f"F{i}→F{j}  {fl_label}")
                gap_data.append(r["gap_dist"])
                angle_data.append(min(r["cos_a"], r["cos_b"]))
                pass_data.append(r["passed"])

        # Gap table
        table_vals = np.array(gap_data).reshape(-1, 1)
        cmap_dist = plt.cm.RdYlGn_r
        norm_dist = plt.Normalize(vmin=0, vmax=MAX_GAP_PX * 2)
        xpos = np.arange(len(row_labels))
        bars = ax.barh(xpos, gap_data, color=[cmap_dist(norm_dist(d)) for d in gap_data])
        ax.axvline(MAX_GAP_PX, color="red", linewidth=1.5, linestyle="--", label=f"max_gap={MAX_GAP_PX:.0f}px")
        ax.set_yticks(xpos)
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_xlabel("Gap distance (px)", fontsize=7)
        ax.set_title("Gap distances (all flip combos)\nred line = max_gap_px", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.invert_yaxis()
        for xi, (val, passed) in enumerate(zip(gap_data, pass_data)):
            ax.text(val + 1, xi, f"{'✓' if passed else '✗'}", va="center", fontsize=6,
                    color="green" if passed else "red")

        # ── Panel 5: Min angle-cosine table ───────────────────────────────
        ax = axes[ax_idx]; ax_idx += 1
        cmap_cos = plt.cm.RdYlGn
        norm_cos = plt.Normalize(vmin=-1, vmax=1)
        ax.barh(xpos, angle_data, color=[cmap_cos(norm_cos(c)) for c in angle_data])
        ax.axvline(cos_thresh, color="orange", linewidth=1.5, linestyle="--",
                   label=f"cos_thresh={cos_thresh:.3f} ({MAX_JOINT_ANGLE_DEG:.0f}°)")
        ax.set_yticks(xpos)
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_xlabel("min(cos_A, cos_B)", fontsize=7)
        ax.set_title("Angle cosines (all flip combos)\norange line = cos threshold", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.invert_yaxis()
        for xi, (val, passed) in enumerate(zip(angle_data, pass_data)):
            ax.text(val + 0.02, xi, f"{'✓' if passed else '✗'}", va="center", fontsize=6,
                    color="green" if passed else "red")

        # ── Panel 6: Final stitched (or fallback) centerline ──────────────
        ax = axes[ax_idx]; ax_idx += 1
        if gap_exceeded and fragments:
            cl_show, wp_show = fragments[0]
            idx_fb = np.round(np.linspace(0, len(cl_show) - 1, N_CENTERLINE_POINTS)).astype(int)
            cl_show = cl_show[idx_fb]
            gr_show = []
            title_str = f"Fallback: fragment 0 only\n(stitch failed: {failure_reason})"
            title_color = "red"
        elif cl_out is not None:
            cl_show = cl_out
            gr_show = gap_ranges
            chain_str = " → ".join(f"F{fi}{'↑' if fl else ''}" for fi, fl in frag_chain)
            title_str = f"Stitched centerline\nchain: {chain_str}"
            title_color = "green"
        else:
            cl_show = None
            gr_show = []
            title_str = "No centerline"
            title_color = "red"

        overlay_img = draw_centerline_on_image(img, mask_bin, cl_show, gap_index_ranges=gr_show) if cl_show is not None else img
        ax.imshow(cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB))
        ax.set_title(title_str, fontsize=9, fontweight="bold", color=title_color)
        ax.axis("off")

    # ── Hide unused axes ──────────────────────────────────────────────────
    for ax in axes[ax_idx:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "stitch_debug_" + os.path.splitext(os.path.basename(IMAGE_PATH))[0] + ".png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"✓ Debug figure saved → {out_path}")

    # ── Standalone gap-zoom figure ─────────────────────────────────────────
    if len(fragments) > 1:
        zoom_path = os.path.join(
            OUTPUT_DIR,
            "gap_zoom_" + os.path.splitext(os.path.basename(IMAGE_PATH))[0] + ".png",
        )
        _save_gap_zoom_standalone(
            zoom_path,
            os.path.splitext(os.path.basename(IMAGE_PATH))[0],
            img_rgb, fragments, pair_results,
            MAX_GAP_PX, cos_thresh, TANGENT_PATH_LEN_PX, SKELETON_PX_PER_POINT,
        )
        print(f"✓ Gap zoom figure saved → {zoom_path}")

        # ── Tangent chord detail figure ─────────────────────────────────
        tangent_path = os.path.join(
            OUTPUT_DIR,
            "tangent_detail_" + os.path.splitext(os.path.basename(IMAGE_PATH))[0] + ".png",
        )
        _save_tangent_detail_figure(
            tangent_path,
            os.path.splitext(os.path.basename(IMAGE_PATH))[0],
            img_rgb, fragments, pair_results,
            cos_thresh, MAX_GAP_PX, TANGENT_PATH_LEN_PX, SKELETON_PX_PER_POINT,
        )
        print(f"✓ Tangent detail figure saved → {tangent_path}")
        import subprocess
        subprocess.Popen(["explorer", tangent_path])

    # ── Concise summary ───────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print(f"  Detections total   : {n_det}")
    print(f"  Target detection   : {target_idx}")
    print(f"  Fragments produced : {len(fragments)}")
    if len(fragments) > 1:
        print(f"  Stitch outcome     : {'FAILED — ' + failure_reason if gap_exceeded else 'SUCCESS'}")
    print("═" * 60)


if __name__ == "__main__":
    main()
