"""
Step-by-step debug of the "O"-shape worm skeletonization pipeline.
Saves one annotated image per processing step so you can see exactly
where the logic breaks down.
"""

import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.color import label2rgb
from skimage.measure import label as sk_label
from skimage.morphology import medial_axis
from skan import Skeleton
from ultralytics import YOLO

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt"
IMAGE_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Skeleton_test\test_img_overlap\176_png.rf.ade8f3becc3343adf565d66c9323a444_roi_20.jpg"
OUT_DIR    = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Skeleton_test\debug_o_shape_output"
CONF       = 0.25
CONTOUR_SMOOTH_SIGMA         = 6.0
HOLE_FILL_RATIO_THRESHOLD    = 0.05   # mirrors extract_centerline default
HOLE_KEEP_CIRCULARITY_WEIGHT = 0.75    # mirrors extract_centerline default
RING_DILATION_RADIUS         = 3      # mirrors extract_centerline default
SMOOTH_FACTOR_RATIO          = 0    # multiplier on len(coords): 0=exact interp, 0.5, 1.0=default, 5.0=heavy smooth
DIR_VECTOR_LEN_PX            = 50  # arc length (pixels) used for directional averaging at each junction seam
MIN_BRIDGE_ARC_LEN_PX        = 20  # Case 6 safety: if score-preferred bridge arc is shorter than this (px),
                                   # direction scoring is unreliable → override to the longer arc as bridge
os.makedirs(OUT_DIR, exist_ok=True)


def save(name: str, img: np.ndarray) -> None:
    p = str(Path(OUT_DIR) / name)
    cv2.imwrite(p, img)
    print(f"  saved → {p}")


def bool_to_bgr(arr: np.ndarray, fg=(255, 255, 255)) -> np.ndarray:
    out = np.zeros((*arr.shape[:2], 3), np.uint8)
    out[arr > 0] = fg
    return out


def overlay_on(base_bgr: np.ndarray, mask_bool: np.ndarray,
               color=(0, 200, 255), alpha=0.45) -> np.ndarray:
    out = base_bgr.copy()
    ov  = base_bgr.copy()
    ov[mask_bool] = color
    return cv2.addWeighted(out, 1 - alpha, ov, alpha, 0)


def _arc_len_index(coords: np.ndarray, target_len: float, from_end: bool = False) -> int:
    """Walk along *coords* accumulating arc length until >= *target_len*.

    Parameters
    ----------
    coords     : (N, 2) (row, col) array.
    target_len : Desired path length in pixels.
    from_end   : True  → walk backward from coords[-1], return offset wi
                         so that coords[-1 - wi] is the incoming anchor.
                 False → walk forward  from coords[0],  return index wo
                         so that coords[wo] is the outgoing anchor.
    """
    if len(coords) < 2:
        return 0
    if from_end:
        cum = 0.0
        for i in range(len(coords) - 1, 0, -1):
            cum += float(np.linalg.norm(coords[i] - coords[i - 1]))
            if cum >= target_len:
                return len(coords) - i
        return len(coords) - 1
    else:
        cum = 0.0
        for i in range(len(coords) - 1):
            cum += float(np.linalg.norm(coords[i + 1] - coords[i]))
            if cum >= target_len:
                return i + 1
        return len(coords) - 1


# ═══════════════════════════════════════════════════════════════════════════
# Step 0 – raw image
# ═══════════════════════════════════════════════════════════════════════════
print("\n[0] Load image")
img = cv2.imread(IMAGE_PATH)
save("step0_original.jpg", img)
h, w = img.shape[:2]
print(f"    image size: {w}×{h}")

# ═══════════════════════════════════════════════════════════════════════════
# Step 1 – YOLO mask
# ═══════════════════════════════════════════════════════════════════════════
print("\n[1] YOLO inference")
model = YOLO(MODEL_PATH)
results = model.predict(source=img, conf=CONF, verbose=False)[0]

if results.masks is None or len(results.masks) == 0:
    print("    ❌ No masks detected!")
    raise SystemExit

masks_raw  = results.masks.data.cpu().numpy()   # (N, Hm, Wm)
boxes_xyxy = results.boxes.xyxy.cpu().numpy()
confs      = results.boxes.conf.cpu().numpy()
n_dets     = len(masks_raw)
print(f"    {n_dets} detection(s)")

# resize all masks to full image resolution
masks_full = [
    cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    for m in masks_raw
]

# pick the detection whose box centre is nearest image centre
img_cx, img_cy = w / 2.0, h / 2.0
box_cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
box_cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
target_idx = int(np.argmin(np.hypot(box_cx - img_cx, box_cy - img_cy)))
print(f"    target detection index: {target_idx}  conf={confs[target_idx]:.3f}")

mask_f32 = masks_full[target_idx]
mask_bin = (mask_f32 > 0.5).astype(np.uint8)

step1 = overlay_on(img, mask_bin.astype(bool), color=(0, 200, 255))
# draw all detection boxes
for i, (box, c) in enumerate(zip(boxes_xyxy, confs)):
    x1, y1, x2, y2 = box.astype(int)
    col = (0, 255, 0) if i == target_idx else (180, 180, 180)
    cv2.rectangle(step1, (x1, y1), (x2, y2), col, 2)
    cv2.putText(step1, f"{'TARGET ' if i == target_idx else ''}conf={c:.2f}",
                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
save("step1_yolo_mask.jpg", step1)

# Also save just the binary mask as a clear B&W image
save("step1b_mask_binary.jpg", mask_bin * 255)
print(f"    mask foreground pixels: {mask_bin.sum()}")

# ═══════════════════════════════════════════════════════════════════════════
# Step 2 – binary_fill_holes  (THIS IS THE CRITICAL STEP FOR "O" SHAPES)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2] binary_fill_holes")
mask_raw_bool = mask_bin.astype(bool)
mask_filled   = binary_fill_holes(mask_raw_bool)
n_orig   = mask_raw_bool.sum()
n_filled = mask_filled.sum()
n_added  = n_filled - n_orig
print(f"    pixels before fill : {n_orig}")
print(f"    pixels after  fill : {n_filled}  (+{n_added} added by fill)")
print(f"    TOPOLOGY CHANGED   : {'YES — hole detected!' if n_added > 50 else 'no'}")

# Visualise difference: original in yellow, added hole in red
step2 = img.copy()
step2[mask_raw_bool]                  = (200, 200,   0)   # original mask → yellow
step2[mask_filled & ~mask_raw_bool]   = (  0,   0, 255)   # added pixels  → red
save("step2_fill_holes.jpg", step2)

# ═══════════════════════════════════════════════════════════════════════════
# Step 2b – selective hole-filling (mirrors extract_centerline is_ring path)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[2b] Selective hole-filling (circularity + area scoring)")

added_ratio = (mask_filled.sum() - mask_raw_bool.sum()) / max(mask_raw_bool.sum(), 1)
is_ring = added_ratio > HOLE_FILL_RATIO_THRESHOLD
print(f"    added_ratio : {added_ratio:.4f}  threshold={HOLE_FILL_RATIO_THRESHOLD}  is_ring={is_ring}")

if is_ring:
    # Optional dilation to bridge narrow body-gap holes before analysis.
    if RING_DILATION_RADIUS > 0:
        ksize_dil  = 2 * RING_DILATION_RADIUS + 1
        kernel_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize_dil, ksize_dil))
        mask_work  = cv2.dilate(mask_raw_bool.astype(np.uint8), kernel_dil).astype(bool)
    else:
        mask_work = mask_raw_bool

    # Label connected components of the background (inverted mask).
    # Components touching the image border are outer background; all others
    # are enclosed holes.
    inv = ~mask_work
    labeled_bg, n_bg = sk_label(inv, connectivity=1, return_num=True)

    border_labels: set = set()
    border_labels.update(int(v) for v in labeled_bg[0, :])
    border_labels.update(int(v) for v in labeled_bg[-1, :])
    border_labels.update(int(v) for v in labeled_bg[:, 0])
    border_labels.update(int(v) for v in labeled_bg[:, -1])
    border_labels.discard(0)   # 0 = foreground, ignore

    hole_labels = [lbl for lbl in range(1, n_bg + 1) if lbl not in border_labels]
    print(f"    enclosed holes found : {len(hole_labels)}")

    if len(hole_labels) > 1:
        # Score each hole: w * circularity + (1-w) * (area / max_area)
        # Hole with the HIGHEST score = loop interior → keep unfilled.
        # All others = body-overlap gaps → fill.
        def _circularity(lbl: int) -> float:
            hole_mask_u8 = (labeled_bg == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(
                hole_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return 0.0
            area_c  = float(cv2.contourArea(cnts[0]))
            perim_c = float(cv2.arcLength(cnts[0], closed=True))
            if perim_c < 1e-6:
                return 0.0
            return (4.0 * np.pi * area_c) / (perim_c ** 2)

        hole_areas_raw = [(int((labeled_bg == lbl).sum()), lbl) for lbl in hole_labels]
        max_area_h = max(a for a, _ in hole_areas_raw) or 1
        w_circ = float(np.clip(HOLE_KEEP_CIRCULARITY_WEIGHT, 0.0, 1.0))
        hole_scores = [
            (w_circ * _circularity(lbl) + (1.0 - w_circ) * (area / max_area_h), lbl)
            for area, lbl in hole_areas_raw
        ]
        hole_scores.sort(reverse=True)
        print(f"    kept hole (loop interior) : label={hole_scores[0][1]}  score={hole_scores[0][0]:.3f}")
        for score, lbl in hole_scores[1:]:
            area_px = int((labeled_bg == lbl).sum())
            print(f"    filled hole (body gap)    : label={lbl}  score={score:.3f}  area={area_px}px")

        fill_mask = mask_work.copy()
        for _, lbl in hole_scores[1:]:
            fill_mask[labeled_bg == lbl] = True
        mask_bool = fill_mask
    else:
        print("    0 or 1 hole — no selective fill needed")
        mask_bool = mask_work

    # Visualise: yellow = original foreground, green = newly filled body-gap pixels
    step2b = img.copy()
    ov2b   = img.copy()
    ov2b[mask_bool]                  = (200, 200,   0)   # all foreground → yellow
    ov2b[mask_bool & ~mask_raw_bool] = (  0, 200,   0)   # newly filled gap pixels → green
    step2b = cv2.addWeighted(step2b, 0.55, ov2b, 0.45, 0)
    cv2.putText(step2b,
                f"Selective fill: {len(hole_labels)} holes found, 1 kept (loop interior)",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    save("step2b_selective_fill.jpg", step2b)
else:
    mask_bool = mask_filled
    print("    Not a ring — mask_bool = binary_fill_holes result (no selective fill)")

# ═══════════════════════════════════════════════════════════════════════════
# Step 3 – Gaussian blur + re-threshold (contour smoothing, ring mask)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3] Contour smoothing (sigma={})".format(CONTOUR_SMOOTH_SIGMA))

# Smooth mask_bool (the selectively-filled mask) without re-filling the hole.
# extract_centerline applies effective_sigma = contour_smooth_sigma directly
# for all worms (normal and ring alike) — no internal multiplier.
effective_sigma = CONTOUR_SMOOTH_SIGMA
if effective_sigma > 0:
    ksize = int(effective_sigma * 6) | 1
    blur_ring = cv2.GaussianBlur(
        mask_bool.astype(np.float32), (ksize, ksize), effective_sigma)
    mask_smooth_ring = (blur_ring > 0.5)
else:
    mask_smooth_ring = mask_bool.copy()

step3b = bool_to_bgr(mask_smooth_ring)
cv2.putText(step3b, f"mask_bool smoothed sigma={effective_sigma:.1f}", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
save(f"step3_smooth_sigma{effective_sigma:.0f}.jpg", step3b)

# ═══════════════════════════════════════════════════════════════════════════
# Step 3c – effect of increasing smoothing sigma on junction count
# ═══════════════════════════════════════════════════════════════════════════
print("\n[3c] Smoothing sigma sweep on RING mask (junction count vs sigma)")

sigma_sweep = [2.0, 4.0, 6.0, 8.0]
mask_smooth_best = mask_smooth_ring   # default: sigma=2.0
best_sigma       = CONTOUR_SMOOTH_SIGMA

for sig in sigma_sweep:
    ks = int(sig * 6) | 1
    blr = cv2.GaussianBlur(mask_bool.astype(np.float32), (ks, ks), sig)
    msk = blr > 0.5
    try:
        sk_tmp, dt_tmp = medial_axis(msk, return_distance=True)
        so = Skeleton(sk_tmp, source_image=dt_tmp)
        pl = so.paths_list()
        le = so.path_lengths()
        G_tmp = nx.MultiGraph()
        for idx2, (pp, ll) in enumerate(zip(pl, le)):
            s2, d2 = int(pp[0]), int(pp[-1])
            G_tmp.add_edge(s2, d2, key=idx2, length=ll, path_idx=idx2)
        n_junc = sum(1 for n in G_tmp.nodes() if G_tmp.degree(n) > 2)
        n_endp = sum(1 for n in G_tmp.nodes() if G_tmp.degree(n) == 1)
        is_cyc = not nx.is_forest(G_tmp)
        print(f"    sigma={sig:.1f} → junctions={n_junc}  endpoints={n_endp}  skel_px={sk_tmp.sum()}  has_cycle={is_cyc}")
    except Exception as e:
        print(f"    sigma={sig:.1f} → error: {e}")

# effective_sigma = CONTOUR_SMOOTH_SIGMA directly (no multiplier).
RING_SIGMA = CONTOUR_SMOOTH_SIGMA
ks_ring = int(RING_SIGMA * 6) | 1
blu_ring_eff = cv2.GaussianBlur(mask_bool.astype(np.float32), (ks_ring, ks_ring), RING_SIGMA)
mask_smooth_ring = (blu_ring_eff > 0.5)   # overwrite with pipeline-consistent version
print(f"\n    → using σ={RING_SIGMA} for subsequent skeleton steps")
save(f"step3c_smooth_sigma{RING_SIGMA:.0f}.jpg", bool_to_bgr(mask_smooth_ring))

# ═══════════════════════════════════════════════════════════════════════════
# Step 4 – Medial axis with sigma=4.0 ring mask
# ═══════════════════════════════════════════════════════════════════════════
print("\n[4] Medial axis (ring at sigma={})".format(RING_SIGMA))

skel_from_ring, dist_from_ring = medial_axis(mask_smooth_ring, return_distance=True)
print(f"    skel from RING   : {skel_from_ring.sum()} skeleton pixels")

def skel_vis(img_base, skel, dist, title):
    vis = img_base.copy()
    vis[skel] = (0, 255, 0)
    dist_norm = cv2.normalize(dist * skel.astype(float), None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    heat[~skel] = 0
    combined = cv2.addWeighted(vis, 0.6, heat, 0.4, 0)
    cv2.putText(combined, title, (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return combined

base_ring = overlay_on(img, mask_smooth_ring, color=(40, 40, 40), alpha=0.6)
save("step4_skel_from_ring.jpg", skel_vis(base_ring, skel_from_ring, dist_from_ring, f"Skel from RING sigma={RING_SIGMA}"))

# ═══════════════════════════════════════════════════════════════════════════
# Step 5 – skan graph analysis: cycle detection + node degree map
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5] Skeleton graph analysis")

for label, skel, dist in [
    ("RING", skel_from_ring, dist_from_ring),
]:
    try:
        skel_obj = Skeleton(skel, source_image=dist)
    except Exception as e:
        print(f"    [{label}] Skeleton() failed: {e}")
        continue

    paths_list = skel_obj.paths_list()
    lengths    = skel_obj.path_lengths()

    # Use MultiGraph (same as the fixed skeletonize_worms.py) so that
    # parallel ring arcs are preserved and cycles are correctly reported.
    G = nx.MultiGraph()
    for idx, (px_path, length) in enumerate(zip(paths_list, lengths)):
        src, dst = int(px_path[0]), int(px_path[-1])
        G.add_edge(src, dst, key=idx, length=length, path_idx=idx)

    n_nodes     = G.number_of_nodes()
    n_edges     = G.number_of_edges()
    is_forest   = nx.is_forest(G)
    n_endpoints = sum(1 for n in G.nodes() if G.degree(n) == 1)
    n_junctions = sum(1 for n in G.nodes() if G.degree(n) > 2)

    print(f"\n    [{label}]")
    print(f"      n_paths     : {skel_obj.n_paths}")
    print(f"      nodes       : {n_nodes}  edges: {n_edges}")
    print(f"      is_forest   : {is_forest}  (False = cycles present)")
    print(f"      endpoints   : {n_endpoints}")
    print(f"      junctions   : {n_junctions}")
    if not is_forest:
        try:
            cycle = nx.find_cycle(G)
            print(f"      cycle found : {len(cycle)} edges in cycle")
        except nx.NetworkXNoCycle:
            print(f"      (no cycle found by nx.find_cycle despite not being forest)")

    # ── Per-edge lengths and total ────────────────────────────────────────
    total_arc_length = 0.0
    total_arc_pixels = 0
    print(f"      edge list (u, v, key) → arc_length  pixel_count  min_dist_width:")
    for u, v, k, d in G.edges(data=True, keys=True):
        pi       = d.get("path_idx")
        arc_len  = d.get("length", 0.0)
        n_px     = len(skel_obj.path_coordinates(pi)) if pi is not None else 0
        if pi is not None:
            coords_e = skel_obj.path_coordinates(pi)
            re = np.clip(coords_e[:, 0].astype(int), 0, dist.shape[0] - 1)
            ce = np.clip(coords_e[:, 1].astype(int), 0, dist.shape[1] - 1)
            min_w = float(np.min(dist[re, ce]))
        else:
            min_w = float('nan')
        node_deg = lambda n: G.degree(n)
        u_type = "ep" if node_deg(u) == 1 else ("junc" if node_deg(u) > 2 else "pass")
        v_type = "ep" if node_deg(v) == 1 else ("junc" if node_deg(v) > 2 else "pass")
        print(f"        ({u}[{u_type}], {v}[{v_type}], {k}) "
              f"→ len={arc_len:7.1f}  px={n_px:4d}  min_w={min_w:.2f}")
        total_arc_length += arc_len
        total_arc_pixels += n_px
    print(f"      TOTAL arc length : {total_arc_length:.1f}")
    print(f"      TOTAL arc pixels : {total_arc_pixels}  (skel.sum={skel.sum()})")

    # Visualise the graph nodes colour-coded by degree
    vis = overlay_on(img, (skel > 0), color=(50, 50, 50), alpha=0.7)
    for node in G.nodes():
        coord = skel_obj.coordinates[node]
        r, c  = int(coord[0]), int(coord[1])
        deg   = G.degree(node)
        if deg == 1:
            colour = (0, 255, 0)    # green  = tip
        elif deg == 2:
            colour = (200, 200, 0)  # yellow = pass-through
        else:
            colour = (0, 0, 255)    # red    = junction
        cv2.circle(vis, (c, r), 5, colour, -1, lineType=cv2.LINE_AA)
    cv2.putText(vis, f"{label} — green=tip  yellow=pass  red=junction",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    save(f"step5_graph_nodes_{label.lower()}.jpg", vis)

# ═══════════════════════════════════════════════════════════════════════════
# Step 5b – cycle-breaking and endpoint/path visualisation (RING only)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[5b] Cycle-breaking and path tracing on RING skeleton")

try:
    import importlib, skeletonize_worms as sw
    importlib.reload(sw)

    skel_obj_r = Skeleton(skel_from_ring, source_image=dist_from_ring)
    paths_list_r = skel_obj_r.paths_list()
    lengths_r    = skel_obj_r.path_lengths()

    # Use MultiGraph so parallel ring arcs are preserved and cycles detected.
    G_r = nx.MultiGraph()
    for idx, (px_path, length) in enumerate(zip(paths_list_r, lengths_r)):
        src, dst = int(px_path[0]), int(px_path[-1])
        G_r.add_edge(src, dst, key=idx, length=length, path_idx=idx)

    print(f"    Before cycle-breaking : {G_r.number_of_edges()} edges, "
          f"is_forest={nx.is_forest(G_r)}, "
          f"endpoints={sum(1 for n in G_r.nodes() if G_r.degree(n)==1)}")

    # ── MultiGraph helper ────────────────────────────────────────────────────
    def _ed(G_ref, u, v):
        """First edge-data dict between u and v (post cycle-breaking = unique)."""
        return next(iter(G_ref[u][v].values()))

    def _path_coords_mg(node_path: list, G_ref: nx.MultiGraph) -> np.ndarray:
        """Row-col coords along a sequence of graph nodes (MultiGraph)."""
        segs = []
        for k in range(len(node_path) - 1):
            u2, v2 = node_path[k], node_path[k + 1]
            pi2 = _ed(G_ref, u2, v2).get("path_idx")
            if pi2 is None:
                continue
            seg2 = skel_obj_r.path_coordinates(pi2)
            pp2  = paths_list_r[pi2]
            if int(pp2[0]) != u2:
                seg2 = seg2[::-1]
            if segs:
                seg2 = seg2[1:]
            segs.append(seg2)
        return np.vstack(segs) if segs else np.empty((0, 2))

    # ── Seam annotation helpers (shared by Case 4/5 and Case 6) ─────────────
    def _seam_cos_c6(arr_prev, arr_next):
        """Returns (cos_val, angle_deg) at the seam junction using DIR_VECTOR_LEN_PX arc length."""
        if len(arr_prev) < 2 or len(arr_next) < 1:
            return 0.0, float('nan')
        wi = _arc_len_index(arr_prev, DIR_VECTOR_LEN_PX, from_end=True)
        wo = _arc_len_index(arr_next, DIR_VECTOR_LEN_PX, from_end=False)
        v_in  = arr_prev[-1] - arr_prev[-1 - wi]
        v_out = arr_next[wo] - arr_prev[-1]
        n_in, n_out = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if n_in == 0 or n_out == 0:
            return 0.0, float('nan')
        cos_val = float(np.dot(v_in, v_out) / (n_in * n_out))
        cos_val = float(np.clip(cos_val, -1.0, 1.0))
        angle_deg = float(np.degrees(np.arccos(cos_val)))
        return cos_val, angle_deg

    def _draw_seam_annotation(ax, segs, seam_idx, color, radius=18, arrow_len=28):
        """Draw direction arrows + arc + angle label at a seam junction.
        Arrow directions use DIR_VECTOR_LEN_PX arc-length chords, exactly
        matching the vectors used in _seam_cos_c6 for scoring."""
        prev, nxt = segs[seam_idx], segs[seam_idx + 1]
        if len(prev) < 2 or len(nxt) < 1:
            return
        junc   = prev[-1]
        wi     = _arc_len_index(prev, DIR_VECTOR_LEN_PX, from_end=True)
        wo     = _arc_len_index(nxt,  DIR_VECTOR_LEN_PX, from_end=False)
        in_pt  = prev[-1 - wi]
        out_pt = nxt[wo]
        jx, jy = float(junc[1]), float(junc[0])
        def _uv(a, b):
            v = np.array([b[1]-a[1], b[0]-a[0]], float)
            n = np.linalg.norm(v)
            return v / n if n > 0 else v
        uv_in  = _uv(in_pt, junc)
        uv_out = _uv(junc,  out_pt)
        ang_in  = float(np.degrees(np.arctan2(uv_in[1],  uv_in[0])))
        ang_out = float(np.degrees(np.arctan2(uv_out[1], uv_out[0])))
        cos_v, ang_deg = _seam_cos_c6(prev, nxt)
        ax.annotate("",
            xy    =(jx, jy),
            xytext=(jx - uv_in[0]*arrow_len, jy - uv_in[1]*arrow_len),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4, mutation_scale=10),
            zorder=8)
        ax.annotate("",
            xy    =(jx + uv_out[0]*arrow_len, jy + uv_out[1]*arrow_len),
            xytext=(jx, jy),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.4, mutation_scale=10),
            zorder=8)
        from matplotlib.patches import Arc
        a1, a2 = sorted([ang_in, ang_out])
        if (a2 - a1) > 180:
            a1, a2 = a2, a1 + 360
        arc = Arc((jx, jy), 2*radius, 2*radius,
                  angle=0, theta1=a1, theta2=a2,
                  color=color, lw=1.5, zorder=5)
        ax.add_patch(arc)
        mid_ang = np.radians((a1 + a2) / 2)
        tx = jx + (radius + 9) * np.cos(mid_ang)
        ty = jy + (radius + 9) * np.sin(mid_ang)
        ax.text(tx, ty, f"{ang_deg:.0f}°", color=color,
                fontsize=7, ha='center', va='center', zorder=6,
                bbox=dict(boxstyle='round,pad=0.1', fc='black', alpha=0.55, ec='none'))
        ax.plot(jx, jy, 's', color='white', ms=5, zorder=7)

    # ── Check for self-loops and describe full branch structure ───────────
    self_loop_nodes = [n for n in G_r.nodes() if G_r.has_edge(n, n)]
    print(f"    Self-loop nodes      : {self_loop_nodes}")

    if self_loop_nodes:
        # Pick the self-loop with the most pixels (dominant ring arc).
        def _best_sl(n):
            cands = [d for d in G_r[n][n].values() if d.get("path_idx") is not None]
            return max(cands, key=lambda d: len(skel_obj_r.path_coordinates(d["path_idx"])),
                       default=None)

        sl_node = max((n for n in self_loop_nodes if _best_sl(n) is not None),
                      key=lambda n: len(skel_obj_r.path_coordinates(_best_sl(n)["path_idx"])),
                      default=None)
        if sl_node is None:
            sl_node = self_loop_nodes[0]

        sl_edata    = _best_sl(sl_node)
        sl_path_idx = sl_edata["path_idx"] if sl_edata else None
        ring_coords = (skel_obj_r.path_coordinates(sl_path_idx)
                       if sl_path_idx is not None else None)
        print(f"    Ring arc pixels      : {len(ring_coords) if ring_coords is not None else 0}")

        G_no_loop = G_r.copy()
        for k in list(G_no_loop[sl_node][sl_node].keys()):
            G_no_loop.remove_edge(sl_node, sl_node, k)

        endpoint_branches = []
        for ep in [n for n in G_no_loop.nodes() if G_no_loop.degree(n) == 1]:
            try:
                npath = nx.shortest_path(G_no_loop, sl_node, ep)
                total = sum(_ed(G_no_loop, npath[k], npath[k+1])["length"]
                            for k in range(len(npath)-1))
                endpoint_branches.append((total, npath))
                branch_coords = _path_coords_mg(npath, G_no_loop)
                print(f"    Branch to ep {ep}: {len(branch_coords)} pixels  length={total:.1f}")
            except nx.NetworkXNoPath:
                pass

        endpoint_branches.sort(key=lambda t: t[0], reverse=True)

        full = None
        if ring_coords is not None and len(ring_coords) >= 4:
            if len(endpoint_branches) >= 2:
                branch_a = _path_coords_mg(endpoint_branches[0][1], G_no_loop)[::-1]
                branch_b = _path_coords_mg(endpoint_branches[1][1], G_no_loop)

                # ── Choose ring traversal direction to minimise sharp turns ──
                ring_fwd = ring_coords[1:-1]
                ring_rev = ring_coords[-2:0:-1]

                def _junction_smoothness_details(ring_interior):
                    """Returns (score, entry_cos, entry_deg, exit_cos, exit_deg)."""
                    def _cos_and_angle(va, vo):
                        na, no = np.linalg.norm(va), np.linalg.norm(vo)
                        if na == 0 or no == 0: return 0.0, float('nan')
                        c = float(np.clip(np.dot(va, vo) / (na * no), -1, 1))
                        return c, float(np.degrees(np.arccos(c)))
                    score = 0.0
                    entry_cos, entry_deg = 0.0, float('nan')
                    exit_cos,  exit_deg  = 0.0, float('nan')
                    if len(branch_a) >= 2 and len(ring_interior) >= 1:
                        wi = _arc_len_index(branch_a,      DIR_VECTOR_LEN_PX, from_end=True)
                        wo = _arc_len_index(ring_interior, DIR_VECTOR_LEN_PX, from_end=False)
                        entry_cos, entry_deg = _cos_and_angle(
                            branch_a[-1] - branch_a[-1 - wi],
                            ring_interior[wo] - branch_a[-1])
                        score += entry_cos
                    if len(ring_interior) >= 2 and len(branch_b) >= 2:
                        wi = _arc_len_index(ring_interior, DIR_VECTOR_LEN_PX, from_end=True)
                        wo = _arc_len_index(branch_b,      DIR_VECTOR_LEN_PX, from_end=False)
                        exit_cos, exit_deg = _cos_and_angle(
                            ring_interior[-1] - ring_interior[-1 - wi],
                            branch_b[wo] - ring_interior[-1])
                        score += exit_cos
                    return score, entry_cos, entry_deg, exit_cos, exit_deg

                score_fwd, fwd_ec, fwd_ed, fwd_xc, fwd_xd = _junction_smoothness_details(ring_fwd)
                score_rev, rev_ec, rev_ed, rev_xc, rev_xd = _junction_smoothness_details(ring_rev)
                ring_interior = ring_fwd if score_fwd >= score_rev else ring_rev
                direction_label = "fwd" if score_fwd >= score_rev else "rev"
                print(f"    Ring direction chosen : {direction_label}  "
                      f"(fwd score={score_fwd:.3f}  rev score={score_rev:.3f})")

                # ── Figure: side-by-side comparison of fwd vs rev ─────────
                # Uses _draw_seam_annotation (defined above) with DIR_VECTOR_LEN_PX
                # arc-length chords, exactly matching _junction_smoothness_details.

                bg_rgb = cv2.cvtColor(
                    overlay_on(img, mask_smooth_ring, color=(40, 40, 40), alpha=0.6),
                    cv2.COLOR_BGR2RGB)

                fig45, axes45 = plt.subplots(1, 2, figsize=(14, 7),
                                              facecolor='#111')
                for ax, ring_int, sc, ec_, ed_, xc_, xd_, lbl, chosen in [
                    (axes45[0], ring_fwd, score_fwd, fwd_ec, fwd_ed, fwd_xc, fwd_xd,
                     'Forward  ring_coords[1:-1]',  direction_label == 'fwd'),
                    (axes45[1], ring_rev, score_rev, rev_ec, rev_ed, rev_xc, rev_xd,
                     'Reversed  ring_coords[-2:0:-1]', direction_label == 'rev'),
                ]:
                    ax.imshow(bg_rgb, origin='upper')
                    ax.set_aspect('equal')
                    ax.axis('off')
                    # draw segments: branch_a (cyan), ring (yellow/magenta), branch_b (cyan)
                    for seg, col_seg in [
                        (branch_a, '#00e5ff'),
                        (ring_int,  '#ffeb3b'),
                        (branch_b[1:], '#00e5ff'),
                    ]:
                        if len(seg) >= 2:
                            ax.plot(seg[:, 1], seg[:, 0], '-', color=col_seg,
                                    lw=1.8, zorder=3)
                    # endpoints
                    if len(branch_a):
                        ax.plot(branch_a[0, 1], branch_a[0, 0], 'o',
                                color='#ff6d00', ms=9, zorder=7)
                    if len(branch_b):
                        ax.plot(branch_b[-1, 1], branch_b[-1, 0], 'o',
                                color='#3d5afe', ms=9, zorder=7)
                    # seam annotations: DIR_VECTOR_LEN_PX arc-length chords (matching scoring)
                    segs_c45 = [branch_a, ring_int, branch_b]
                    SEAM_COLORS_45 = ['#ff4081', '#69f0ae']
                    for si, sc_col in enumerate(SEAM_COLORS_45):
                        _draw_seam_annotation(ax, segs_c45, si, color=sc_col)
                    # legend for scored seams
                    for sname, si_leg, sc_col in [
                        ('entry @sl_node', 0, '#ff4081'),
                        ('exit  @sl_node', 1, '#69f0ae'),
                    ]:
                        _, ang_val = _seam_cos_c6(segs_c45[si_leg], segs_c45[si_leg + 1])
                        ax.plot([], [], '-', color=sc_col, lw=2,
                                label=f'{sname}: {ang_val:.0f}°')
                    ax.legend(loc='lower right', fontsize=6.5,
                              facecolor='#222', edgecolor='#555',
                              labelcolor='white', framealpha=0.8)
                    border_col = '#76ff03' if chosen else '#555'
                    for spine in ax.spines.values():
                        spine.set_edgecolor(border_col)
                        spine.set_linewidth(3 if chosen else 1)
                    title_str = (f"{'✓ CHOSEN  ' if chosen else ''}{lbl}\n"
                                 f"entry={ed_:.0f}°  exit={xd_:.0f}°  "
                                 f"score={sc:.3f}")
                    ax.set_title(title_str, color='white' if chosen else '#aaa',
                                 fontsize=9, pad=4)
                fig45.suptitle('Case 4/5 — Ring traversal direction comparison',
                               color='white', fontsize=11, y=1.01)
                fig45.tight_layout()
                fig45_path = str(Path(OUT_DIR) / 'step5c_ring_direction_comparison.jpg')
                fig45.savefig(fig45_path, dpi=120, bbox_inches='tight',
                              facecolor='#111')
                plt.close(fig45)
                print(f"  saved → {fig45_path}")

                parts = []
                if len(branch_a): parts.append(branch_a)
                parts.append(ring_interior if len(parts) else ring_coords)
                if len(branch_b): parts.append(branch_b[1:])
                full = np.vstack(parts)
                print(f"\n    FULL PATH (2 branches + ring): {len(full)} pixels total")
            elif len(endpoint_branches) == 1:
                branch_a = _path_coords_mg(endpoint_branches[0][1], G_no_loop)[::-1]
                # Always cut the ring at sl_node (index 0 == index N-1) so the
                # ring opens exactly where the branch attaches.
                # branch_a[-1] == ring_coords[0] == ring_coords[-1] == sl_node,
                # so ring_coords[1:] continues seamlessly from the junction with
                # no spatial jump and no U-turn kink at sl_node mid-traversal.
                full = np.vstack([branch_a, ring_coords[1:]])
                print(f"\n    FULL PATH (1 branch + ring cut at sl_node): {len(full)} pixels total")
            else:
                r2 = np.clip(ring_coords[:, 0].astype(int), 0, dist_from_ring.shape[0]-1)
                c2 = np.clip(ring_coords[:, 1].astype(int), 0, dist_from_ring.shape[1]-1)
                cut = int(np.argmin(dist_from_ring[r2, c2]))
                full = np.concatenate([ring_coords[cut:], ring_coords[1:cut+1]])
                print(f"\n    FULL PATH (pure ring cut): {len(full)} pixels total")
        else:
            print("    ❌ No valid ring coords — cannot assemble path")

        if full is not None and len(full) >= 4:
            vis_full = overlay_on(img, mask_smooth_ring, color=(40,40,40), alpha=0.6)
            xy_full  = full[:, ::-1].astype(np.int32)
            cv2.polylines(vis_full, [xy_full.reshape(-1,1,2)],
                          isClosed=False, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
            cv2.circle(vis_full, tuple(xy_full[0]),  9, (255,100,0), -1, lineType=cv2.LINE_AA)
            cv2.circle(vis_full, tuple(xy_full[-1]), 9, (50,50,255), -1, lineType=cv2.LINE_AA)
            cv2.putText(vis_full, f"Full path with overhangs: {len(full)} px",
                        (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
            save("step5b_full_path_with_branches.jpg", vis_full)
        else:
            print("    ❌ Could not assemble full path — skipping visualisation")

    elif not nx.is_forest(G_r):
        # Parallel-arc ring: two skan paths share the same junction endpoints.
        print("    No self-loop, but CYCLE detected (parallel arcs — omega/Ω ring)")
        try:
            cycle_edges = nx.find_cycle(G_r)
        except nx.NetworkXNoCycle:
            cycle_edges = []

        if len(cycle_edges) == 2:
            (ca_u, ca_v, ca_k), (cb_u, cb_v, cb_k) = cycle_edges
            if {ca_u, ca_v} == {cb_u, cb_v}:
                u_j, v_j = ca_u, ca_v
                arcs = [(G_r[u_j][v_j][k].get("length", np.inf), k,
                         G_r[u_j][v_j][k].get("path_idx"))
                        for k in G_r[u_j][v_j]]
                arcs.sort()
                _, short_k, short_pi = arcs[0]
                _, long_k,  long_pi  = arcs[1]

                def _arc_c(pi, from_node):
                    c = skel_obj_r.path_coordinates(pi)
                    if int(paths_list_r[pi][0]) != from_node:
                        c = c[::-1]
                    return c

                arc_a   = _arc_c(short_pi, u_j)   # u_j → v_j  (shorter arc)
                arc_b   = _arc_c(long_pi,  u_j)   # u_j → v_j  (longer  arc)
                arc_a_r = arc_a[::-1]              # v_j → u_j
                arc_b_r = arc_b[::-1]              # v_j → u_j

                G_br = G_r.copy()
                for k in list(G_br[u_j][v_j].keys()):
                    G_br.remove_edge(u_j, v_j, k)

                def _branch(start):
                    eps = [n for n in nx.node_connected_component(G_br, start)
                           if G_br.degree(n) <= 1 and n != start]
                    if not eps:
                        return np.empty((0, 2))
                    npath = nx.shortest_path(G_br, start, eps[0])
                    segs = []
                    for ki in range(len(npath) - 1):
                        u2, v2 = npath[ki], npath[ki+1]
                        pi2 = _ed(G_br, u2, v2).get("path_idx")
                        if pi2 is None: continue
                        s = skel_obj_r.path_coordinates(pi2)
                        if int(paths_list_r[pi2][0]) != u2: s = s[::-1]
                        if segs: s = s[1:]
                        segs.append(s)
                    return np.vstack(segs) if segs else np.empty((0, 2))

                ba = _branch(u_j)   # u_j outward → ep_A
                bb = _branch(v_j)   # v_j outward → ep_B

                # ── Choose which arc is the bridge (traversed twice) ──────
                # Two candidate topologies:
                #   T1: ep_A→u_j → arc_a → v_j → arc_b_r → u_j → arc_a → v_j → ep_B
                #   T2: ep_A→u_j → arc_b → v_j → arc_a_r → u_j → arc_b → v_j → ep_B
                # Pick the one with the highest sum of cos(turning angle) at all 4 seams.
                # _seam_cos_c6 is defined above (shared with Case 4/5).

                def _build_c6(bridge, single_r, bridge_name, single_name):
                    seg_entry   = ba[::-1]
                    seg_bridge  = bridge[1:]
                    seg_single  = single_r[1:]
                    seg_bridge2 = bridge[1:]
                    seg_exit    = bb[1:]
                    segs = []
                    if len(seg_entry):  segs.append(seg_entry)
                    segs.append(seg_bridge if segs else bridge)
                    segs.append(seg_single)
                    segs.append(seg_bridge2)
                    if len(seg_exit): segs.append(seg_exit)
                    # Seam labels describe what junction is being crossed
                    seam_labels = [
                        f"entry    branch_A → {bridge_name}  @u_j={u_j}",
                        f"turnaround  {bridge_name} → {single_name}  @v_j={v_j}",
                        f"turnaround  {single_name} → {bridge_name}  @u_j={u_j}",
                        f"exit     {bridge_name} → branch_B  @v_j={v_j}",
                    ]
                    seam_results = []
                    for i in range(len(segs) - 1):
                        cos_val, angle_deg = _seam_cos_c6(segs[i], segs[i+1])
                        label = seam_labels[i] if i < len(seam_labels) else f"seam {i}"
                        seam_results.append((label, cos_val, angle_deg))
                    # Only score entry (branch_A → bridge) and exit (bridge2 → branch_B).
                    # The two turnaround seams are structurally unavoidable U-turns that
                    # are roughly equal for both topologies and do not help discriminate.
                    score = sum(c for lbl, c, _ in seam_results
                                if not lbl.startswith("turnaround"))
                    return np.vstack(segs), score, seam_results

                # ── Bridge selection ──────────────────────────────────────
                # Since arcs is sorted ascending, arc_a <= arc_b always.
                # If arc_a < threshold then at least one arc is too short for
                # reliable direction scoring → skip the score entirely and use
                # the shorter arc (arc_a, T1) as bridge.
                # Only follow the score when both arcs are above the threshold.
                arc_a_len_px = float(arcs[0][0])
                arc_b_len_px = float(arcs[1][0])
                override_reason = None
                if arc_a_len_px < MIN_BRIDGE_ARC_LEN_PX:
                    override_reason = (f"OVERRIDE: arc_a {arc_a_len_px:.0f}px "
                                       f"<{MIN_BRIDGE_ARC_LEN_PX}px threshold "
                                       f"→ arc_a (shorter) as bridge (score skipped)")
                    print(f"    → Bridge override: arc_a ({arc_a_len_px:.1f}px) "
                          f"< MIN_BRIDGE_ARC_LEN_PX={MIN_BRIDGE_ARC_LEN_PX} "
                          f"— at least one arc unreliable → T1 (arc_a shorter as bridge, score skipped)")
                    arr_t1, score_t1, seams_t1 = _build_c6(arc_a, arc_b_r, "arc_a(shorter)", "arc_b_r")
                    arr_t2, score_t2, seams_t2 = None, None, None
                    use_t1 = True
                else:
                    arr_t1, score_t1, seams_t1 = _build_c6(arc_a, arc_b_r, "arc_a(shorter)", "arc_b_r")
                    arr_t2, score_t2, seams_t2 = _build_c6(arc_b, arc_a_r, "arc_b(longer)",  "arc_a_r")
                    use_t1 = score_t1 >= score_t2

                bridge_label = "arc_a (shorter)" if use_t1 else "arc_b (longer)"
                full = arr_t1 if use_t1 else arr_t2

                print(f"    u_j={u_j}  v_j={v_j}")
                print(f"    arc_a (shorter): {len(arc_a)} px  arc_b (longer): {len(arc_b)} px")
                print(f"    branch_A : {len(ba)} px  (ep→u_j)  branch_B : {len(bb)} px  (v_j→ep)")
                # ── Effective arc length diagnostic and figure (score path only) ──
                if arr_t2 is not None:
                    def _seg_arc(arr):
                        if len(arr) < 2: return 0.0
                        return float(np.sum(np.linalg.norm(np.diff(arr, axis=0), axis=1)))
                    def _eff_arc(arr, fend):
                        """Arc actually spanned by _arc_len_index on this segment."""
                        if len(arr) < 2: return 0.0
                        wi = _arc_len_index(arr, DIR_VECTOR_LEN_PX, from_end=fend)
                        sub = arr[-1-wi:] if fend else arr[:wi+1]
                        return float(np.sum(np.linalg.norm(np.diff(sub, axis=0), axis=1)))
                    print(f"    DIR_VECTOR_LEN_PX={DIR_VECTOR_LEN_PX}  \u2192  actual arc covered at scored seams:")
                    print(f"      T1/T2 entry incoming (ba):    {_eff_arc(ba[::-1], True):.1f}px  (ba total: {_seg_arc(ba):.1f}px)")
                    print(f"      T1 entry outgoing (arc_a):    {_eff_arc(arc_a[1:], False):.1f}px  (arc_a total: {_seg_arc(arc_a):.1f}px)")
                    print(f"      T2 entry outgoing (arc_b):    {_eff_arc(arc_b[1:], False):.1f}px  (arc_b total: {_seg_arc(arc_b):.1f}px)")
                    print(f"      T1 exit incoming  (arc_a):    {_eff_arc(arc_a[1:], True):.1f}px")
                    print(f"      T2 exit incoming  (arc_b):    {_eff_arc(arc_b[1:], True):.1f}px")
                    print(f"      T1/T2 exit outgoing (bb):     {_eff_arc(bb[1:], False):.1f}px  (bb total: {_seg_arc(bb):.1f}px)")
                    print(f"\n    T1 \u2014 arc_a (shorter) as bridge  [score={score_t1:.4f}]")
                    for lbl, cos_v, ang in seams_t1:
                        print(f"      {lbl:55s}  cos={cos_v:+.4f}  angle={ang:6.1f}\u00b0")
                    print(f"\n    T2 \u2014 arc_b (longer) as bridge   [score={score_t2:.4f}]")
                    for lbl, cos_v, ang in seams_t2:
                        print(f"      {lbl:55s}  cos={cos_v:+.4f}  angle={ang:6.1f}\u00b0")
                    print(f"\n    \u2192 Bridge chosen : {bridge_label}")
                    print(f"\n    FULL PATH (omega double-cross): {len(full)} pixels total")

                # ── Figure: side-by-side comparison of T1 vs T2 ──────────
                    # _draw_seam_annotation is defined above (shared with Case 4/5).

                    bg_rgb6 = cv2.cvtColor(
                        overlay_on(img, mask_smooth_ring, color=(40, 40, 40), alpha=0.6),
                        cv2.COLOR_BGR2RGB)

                    # segment colour palette per topology segment type
                    SEG_COLORS = [
                        '#00e5ff',   # branch_A  (entry)
                        '#ffeb3b',   # bridge 1st pass
                        '#ff6d00',   # single arc
                        '#ffeb3b',   # bridge 2nd pass
                        '#00e5ff',   # branch_B  (exit)
                    ]
                    SEAM_COLORS = ['#ff4081', '#69f0ae', '#ff9100', '#40c4ff']

                    fig6, axes6 = plt.subplots(1, 2, figsize=(16, 8), facecolor='#111')
                    for ax, arr, seams, sc, tlabel, chosen in [
                        (axes6[0], arr_t1, seams_t1, score_t1,
                         'T1 — arc_a (shorter) as bridge', use_t1),
                        (axes6[1], arr_t2, seams_t2, score_t2,
                         'T2 — arc_b (longer) as bridge',  not use_t1),
                    ]:
                        ax.imshow(bg_rgb6, origin='upper')
                        ax.set_aspect('equal')
                        ax.axis('off')

                        # Rebuild the segment list for this topology to draw
                        # coloured segments and annotate seams.
                        bridge_here  = arc_a if 'shorter' in tlabel else arc_b
                        single_r_here= arc_b_r if 'shorter' in tlabel else arc_a_r
                        segs_here = []
                        if len(ba): segs_here.append(ba[::-1])
                        br_start = bridge_here[1:]
                        segs_here.append(br_start if segs_here else bridge_here)
                        segs_here.append(single_r_here[1:])
                        segs_here.append(bridge_here[1:])
                        if len(bb): segs_here.append(bb[1:])

                        for si, seg in enumerate(segs_here):
                            col_seg = SEG_COLORS[si] if si < len(SEG_COLORS) else '#ccc'
                            if len(seg) >= 2:
                                ax.plot(seg[:, 1], seg[:, 0], '-',
                                        color=col_seg, lw=2, zorder=3,
                                        alpha=0.85)
                        # endpoints
                        ep_a_coord = segs_here[0][0]  if len(segs_here[0]) else segs_here[0][0]
                        ep_b_coord = segs_here[-1][-1] if len(segs_here[-1]) else segs_here[-1][-1]
                        ax.plot(ep_a_coord[1], ep_a_coord[0], 'o',
                                color='#ff6d00', ms=10, zorder=8, label='ep_A')
                        ax.plot(ep_b_coord[1], ep_b_coord[0], 'o',
                                color='#3d5afe', ms=10, zorder=8, label='ep_B')
                        # annotate only the scored seams: entry (0) and exit (last)
                        _scored_seam_indices = [0, len(segs_here) - 2]
                        for si, sc_col in zip(_scored_seam_indices, [SEAM_COLORS[0], SEAM_COLORS[3]]):
                            if si < len(segs_here) - 1:
                                _draw_seam_annotation(ax, segs_here, si, color=sc_col)
                        # legend for the two scored seams only
                        scored_seam_info = [
                            ('entry @u_j',  0,              SEAM_COLORS[0]),
                            ('exit @v_j',   len(seams) - 1, SEAM_COLORS[3]),
                        ]
                        for sname, si, sc_col in scored_seam_info:
                            ang_val = seams[si][2] if si < len(seams) else float('nan')
                            ax.plot([], [], '-', color=sc_col, lw=2,
                                    label=f'{sname}: {ang_val:.0f}°')

                        border_col = '#76ff03' if chosen else '#555'
                        for spine in ax.spines.values():
                            spine.set_edgecolor(border_col)
                            spine.set_linewidth(3 if chosen else 1)
                        ax.set_title(
                            f"{'\u2713 CHOSEN  ' if chosen else ''}{tlabel}\nscore={sc:.4f}",
                            color='white' if chosen else '#aaa',
                            fontsize=9, pad=4)
                        ax.legend(loc='lower right', fontsize=6.5,
                                  facecolor='#222', edgecolor='#555',
                                  labelcolor='white', framealpha=0.8)

                    fig6.suptitle(
                        f'Case 6 — Bridge arc choice comparison  '
                        f'(u_j={u_j}, v_j={v_j})',
                        color='white', fontsize=11, y=1.01)
                    # segment colour legend (shared)
                    seg_labels = ['branch_A / branch_B', 'bridge (×2)', 'single arc']
                    seg_cols_leg = ['#00e5ff', '#ffeb3b', '#ff6d00']
                    handles = [plt.Line2D([0], [0], color=c, lw=2, label=l)
                               for c, l in zip(seg_cols_leg, seg_labels)]
                    handles += [
                        plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='#ff6d00', ms=8, label='ep_A (orange)'),
                        plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='#3d5afe', ms=8, label='ep_B (blue)'),
                    ]
                    fig6.legend(handles=handles, loc='lower center', ncol=5,
                                fontsize=7.5, facecolor='#222',
                                edgecolor='#555', labelcolor='white',
                                framealpha=0.9, bbox_to_anchor=(0.5, -0.04))
                    fig6.tight_layout()
                    fig6_path = str(Path(OUT_DIR) / 'step5d_bridge_choice_comparison.jpg')
                    fig6.savefig(fig6_path, dpi=120, bbox_inches='tight',
                                 facecolor='#111')
                    plt.close(fig6)
                    print(f"  saved → {fig6_path}")

                vis_full = overlay_on(img, mask_smooth_ring, color=(40,40,40), alpha=0.6)
                xy_full  = full[:, ::-1].astype(np.int32)
                cv2.polylines(vis_full, [xy_full.reshape(-1,1,2)],
                              isClosed=False, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(vis_full, tuple(xy_full[0]),  9, (255,100,0), -1, lineType=cv2.LINE_AA)
                cv2.circle(vis_full, tuple(xy_full[-1]), 9, (50,50,255), -1, lineType=cv2.LINE_AA)
                cv2.putText(vis_full, f"Omega path ({bridge_label} as bridge): {len(full)} px",
                            (5,20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                if override_reason:
                    cv2.putText(vis_full, override_reason,
                                (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
                save("step5b_full_path_with_branches.jpg", vis_full)
            else:
                print("    ❌ 2-edge cycle but not parallel arcs — unexpected topology")
        else:
            print(f"    ❌ Cycle has {len(cycle_edges)} edges — not a simple parallel-arc ring")
    else:
        print("    No self-loop and no cycle — simple open arc (standard worm)")

except Exception as e:
    print(f"    ❌ Error: {e}")
    import traceback; traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════════════
# Step 6 – run the FIXED pipeline (topology-aware hole fill)
# ═══════════════════════════════════════════════════════════════════════════
print("\n[6] Run FIXED pipeline (topology-aware hole filling)")

try:
    import importlib
    import skeletonize_worms as sw
    importlib.reload(sw)   # pick up the edited source

    cl_fix, wp_fix, ok_fix = sw.extract_centerline(
        mask_bin,
        n_points=100,
        smooth_factor=None,
        smooth_factor_ratio=SMOOTH_FACTOR_RATIO,
        contour_smooth_sigma=CONTOUR_SMOOTH_SIGMA,
        hole_fill_ratio_threshold=HOLE_FILL_RATIO_THRESHOLD,
        hole_keep_circularity_weight=HOLE_KEEP_CIRCULARITY_WEIGHT,
        ring_dilation_radius=RING_DILATION_RADIUS,
        dir_vector_len_px=DIR_VECTOR_LEN_PX,
    )

    if ok_fix and cl_fix is not None:
        vis6 = img.copy()
        ov6  = img.copy()
        ov6[mask_bin == 1] = (0, 200, 255)
        vis6 = cv2.addWeighted(vis6, 0.65, ov6, 0.35, 0)
        xy = cl_fix[:, ::-1].astype(np.int32)
        cv2.polylines(vis6, [xy.reshape(-1, 1, 2)],
                      isClosed=False, color=(0, 255, 0), thickness=2,
                      lineType=cv2.LINE_AA)
        r0, c0 = int(cl_fix[0, 0]), int(cl_fix[0, 1])
        r1, c1 = int(cl_fix[-1, 0]), int(cl_fix[-1, 1])
        cv2.circle(vis6, (c0, r0), 7, (255, 100, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(vis6, (c1, r1), 7, (50, 50, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(vis6, "FIXED pipeline — O-shape preserved",
                    (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        save("step6_fixed_centerline.jpg", vis6)
        print(f"    ✓ centerline extracted: {len(cl_fix)} points")

        # ── Diagnose spline endpoint drift caused by smooth_factor_ratio ──
        # splprep with s > 0 is a least-squares approximation, NOT an
        # interpolant.  Large s values allow the spline to drift away from the
        # raw skeleton endpoints, producing a visible spatial shift of the
        # orange/blue markers relative to the skeleton tip pixels.
        #
        # Here we replicate the spline step with the same coords to quantify
        # the drift and compare several s values.
        print(f"\n    [6 diag] Spline endpoint drift analysis  (SMOOTH_FACTOR_RATIO={SMOOTH_FACTOR_RATIO})")
        from scipy.interpolate import splprep as _splprep, splev as _splev

        # Re-run the internal path-building to get raw skeleton coords,
        # replicating extract_centerline's exact pipeline (hole fill + dilation
        # + smoothing) so the raw coords match what was passed to splprep.
        from scipy.ndimage import binary_fill_holes as _bfh
        _mr  = mask_bin.astype(bool)
        _mf  = _bfh(_mr)
        _ar  = (_mf.sum() - _mr.sum()) / max(_mr.sum(), 1)
        if _ar > HOLE_FILL_RATIO_THRESHOLD and RING_DILATION_RADIUS > 0:
            _ks  = 2 * RING_DILATION_RADIUS + 1
            _ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ks, _ks))
            _mw  = cv2.dilate(_mr.astype(np.uint8), _ker).astype(bool)
        else:
            _mw = _mr
        # selective hole fill (same scoring as extract_centerline)
        _inv = ~_mw
        from skimage.measure import label as _sklbl
        _lbg, _nbg = _sklbl(_inv, connectivity=1, return_num=True)
        _bls: set = set()
        _bls.update(int(v) for v in _lbg[0, :]); _bls.update(int(v) for v in _lbg[-1, :])
        _bls.update(int(v) for v in _lbg[:, 0]); _bls.update(int(v) for v in _lbg[:, -1])
        _bls.discard(0)
        _hls = [l for l in range(1, _nbg + 1) if l not in _bls]
        _mb  = _mw.copy()
        if len(_hls) > 1:
            def _ci2(l):
                _u8 = (_lbg == l).astype(np.uint8)
                _cc, _ = cv2.findContours(_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not _cc: return 0.0
                _a = float(cv2.contourArea(_cc[0])); _p = float(cv2.arcLength(_cc[0], True))
                return (4 * np.pi * _a / _p ** 2) if _p > 1e-6 else 0.0
            _haw = [(int((_lbg == l).sum()), l) for l in _hls]
            _mxa = max(a for a, _ in _haw) or 1
            _w   = float(np.clip(HOLE_KEEP_CIRCULARITY_WEIGHT, 0, 1))
            _hs  = sorted([(_w * _ci2(l) + (1 - _w) * (a / _mxa), l) for a, l in _haw], reverse=True)
            for _, l in _hs[1:]:
                _mb[_lbg == l] = True
        _sig2 = CONTOUR_SMOOTH_SIGMA
        _ks2  = int(_sig2 * 6) | 1
        _sm2  = cv2.GaussianBlur(_mb.astype(np.float32), (_ks2, _ks2), _sig2) > 0.5
        _sk2, _dt2 = medial_axis(_sm2, return_distance=True)
        _so2 = Skeleton(_sk2, source_image=_dt2)
        _raw = sw._longest_endpoint_path_coords(_so2)

        if _raw is not None and len(_raw) >= 4:
            print(f"    raw skeleton endpoints : coords[0]={_raw[0]}  coords[-1]={_raw[-1]}")
            print(f"    spline endpoints       : cl[0]={cl_fix[0].round(1)}  cl[-1]={cl_fix[-1].round(1)}")
            drift_start = float(np.hypot(_raw[0, 0] - cl_fix[0, 0], _raw[0, 1] - cl_fix[0, 1]))
            drift_end   = float(np.hypot(_raw[-1, 0] - cl_fix[-1, 0], _raw[-1, 1] - cl_fix[-1, 1]))
            print(f"    endpoint drift (px)    : start={drift_start:.1f}  end={drift_end:.1f}")

            print(f"\n    s-value sweep (s = ratio × {len(_raw)} raw pixels):")
            for ratio in [0.0, 0.5, 1.0, 2.0, 5.0]:
                s_val = ratio * len(_raw)
                try:
                    if s_val == 0.0:
                        tck2, _ = _splprep([_raw[:, 0], _raw[:, 1]], s=0, k=3)
                    else:
                        tck2, _ = _splprep([_raw[:, 0], _raw[:, 1]], s=s_val, k=3)
                    r2, c2 = _splev([0.0, 1.0], tck2)
                    d_s = float(np.hypot(_raw[0, 0] - r2[0], _raw[0, 1] - c2[0]))
                    d_e = float(np.hypot(_raw[-1, 0] - r2[1], _raw[-1, 1] - c2[1]))
                    print(f"      ratio={ratio:.1f}  s={s_val:.0f}  → start drift={d_s:.1f}px  end drift={d_e:.1f}px")
                except Exception as ex:
                    print(f"      ratio={ratio:.1f} → error: {ex}")

            # Visualise drift: raw endpoint pixels (yellow cross) vs spline
            # endpoints (orange/blue circles) side-by-side on step 6 image.
            vis6d = vis6.copy()
            for pt, col in [(_raw[0, ::-1].astype(int), (0, 255, 255)),
                            (_raw[-1, ::-1].astype(int), (0, 255, 255))]:
                cv2.drawMarker(vis6d, tuple(pt), col, cv2.MARKER_CROSS, 15, 2, cv2.LINE_AA)
            cv2.putText(vis6d,
                        f"orange/blue=spline  cyan cross=raw skel  ratio={SMOOTH_FACTOR_RATIO}",
                        (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            save("step6b_endpoint_drift.jpg", vis6d)
        else:
            print("    (could not reconstruct raw path for drift analysis)")
    else:
        print("    ❌ extract_centerline returned failure after fix")
except Exception as e:
    print(f"    ❌ Error in step 6: {e}")
    import traceback; traceback.print_exc()

print(f"\n✓ All debug images saved to: {OUT_DIR}")
print("  Open the folder and inspect images step0 → step7 in order.")
