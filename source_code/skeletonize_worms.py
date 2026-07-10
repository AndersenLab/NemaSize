"""
C. elegans Centerline Skeletonization Script
=============================================
Fits a smooth centerline to each worm mask produced by a YOLO segmentation model.

Pipeline per mask:
    binary mask → fill holes → medial axis + distance transform
    → skan skeleton graph → longest path (by arc length)
    → parametric B-spline smoothing
    → ordered centerline (row, col) + local body width profile

Outputs:
    - Annotated images with semi-transparent mask overlay and green centerline

Reuses utilities from visualize_predictions.py (get_image_files) to avoid
code duplication.
"""

import argparse
import csv
import itertools
import os
import signal
import warnings
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import torch
from scipy.interpolate import splprep, splev
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage.measure import label as sk_label
from skimage.morphology import medial_axis
from skan import Skeleton
from tqdm import tqdm
from ultralytics import YOLO

# Allow cuDNN to benchmark and select the fastest convolution algorithm for
# the current hardware.  This improves GPU throughput at the cost of
# non-deterministic algorithm selection between runs (borderline mask pixels
# may occasionally differ).  Set deterministic=True / benchmark=False if
# exact run-to-run reproducibility is required.
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# Reuse the image-discovery helper defined in the visualization module
from visualize_predictions import get_image_files
from speed_meter import SpeedMeter, device_label


# ---------------------------------------------------------------------------
# Core skeletonization helpers
# ---------------------------------------------------------------------------

def _break_cycles_at_minimum_width(
    G: nx.MultiGraph,
    skel_obj: Skeleton,
    paths_list: list,
) -> None:
    """
    Detect and remove one edge per cycle in *G*, choosing the edge whose
    skeleton pixels have the **smallest minimum distance-transform value**.

    This handles self-overlapping worm configurations ("O" or "6" shapes)
    where the medial axis forms a closed loop.  Removing the edge at the
    narrowest point cuts the loop at the most likely self-overlap region —
    the spot where the two body sections lie closest together and local width
    is smallest.

    *G* must be a ``nx.MultiGraph`` so that parallel arcs between the same
    pair of junction nodes (both sides of a ring) are preserved as distinct
    edges.  ``nx.find_cycle`` returns ``(u, v, key)`` triples for MultiGraph,
    and ``G.remove_edge(u, v, key)`` removes only the targeted edge.

    The graph is modified **in-place**.  After all cycles are removed it is a
    forest, so the normal degree-1 endpoint search can proceed unchanged.
    Normal (non-looping) worms have no cycles and are completely unaffected.
    """
    max_iterations = G.number_of_edges()   # safety cap
    for _ in range(max_iterations):
        try:
            cycle_edges = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            break

        # Cut the shortest arc in the cycle (fewest skeleton pixels).
        # Arc *length* is a more reliable discriminator than minimum distance-
        # transform width for the parallel-arc ring case:
        #   - The crossing stub (which should be removed) is always short.
        #   - The main ring arc (which should be kept) is always long.
        # Minimum-width could wrongly cut the long arc if the worm body is
        # narrow anywhere along its length.
        min_len  = np.inf
        cut_edge = None   # (u, v, key)
        for edge in cycle_edges:   # each element is (u, v, key) for MultiGraph
            u, v = edge[0], edge[1]
            key  = edge[2] if len(edge) > 2 else 0
            if not G.has_edge(u, v, key):
                continue
            arc_len = G[u][v][key].get("length", np.inf)
            if arc_len < min_len:
                min_len  = arc_len
                cut_edge = (u, v, key)

        if cut_edge is None:
            # Fallback: no length stored — remove the first edge in the cycle
            e0 = cycle_edges[0]
            cut_edge = (e0[0], e0[1], e0[2] if len(e0) > 2 else 0)

        G.remove_edge(cut_edge[0], cut_edge[1], cut_edge[2])


def _arc_len_index(coords: np.ndarray, target_len: float, from_end: bool = False) -> int:
    """Walk along *coords* accumulating arc length until >= *target_len*.

    Parameters
    ----------
    coords     : (N, 2) array of (row, col) skeleton points.
    target_len : Desired path length in pixels.
    from_end   : If ``True``, walk backward from ``coords[-1]`` and return the
                 offset ``wi`` such that ``coords[-1 - wi]`` is the anchor for
                 an *incoming* direction vector.
                 If ``False``, walk forward from ``coords[0]`` and return the
                 direct index ``wo`` such that ``coords[wo]`` is the anchor for
                 an *outgoing* direction vector.

    Returns
    -------
    int
        Index offset (``from_end=True``) or direct index (``from_end=False``).
        Always in ``[0, len(coords) - 1]``.  Returns 0 when *coords* has fewer
        than 2 points.
    """
    if len(coords) < 2:
        return 0
    if from_end:
        cum = 0.0
        for i in range(len(coords) - 1, 0, -1):
            cum += float(np.linalg.norm(coords[i] - coords[i - 1]))
            if cum >= target_len:
                return len(coords) - i   # wi: coords[-1-wi] == coords[i-1]
        return len(coords) - 1           # whole array shorter than target_len
    else:
        cum = 0.0
        for i in range(len(coords) - 1):
            cum += float(np.linalg.norm(coords[i + 1] - coords[i]))
            if cum >= target_len:
                return i + 1             # wo: coords[wo] is the anchor
        return len(coords) - 1           # whole array shorter than target_len


def _longest_endpoint_path_coords(
    skel_obj: Skeleton,
    dir_vector_len_px: float = 20.0,
    min_bridge_arc_len_px: float = 20.0,
    img_name: str = "",
) -> np.ndarray | None:
    """
    Build a networkx graph from all skan skeleton segments and return the
    ordered (N, 2) (row, col) coordinate array for the longest
    endpoint-to-endpoint path.

    Why a graph traversal is needed
    --------------------------------
    ``skan`` splits the skeleton into *segments* — edges between junction
    nodes and degree-1 leaf nodes.  ``path_lengths()`` gives the length of
    each individual segment, not the full head-to-tail path.  When the medial
    axis has *any* junction (from mask noise, thick regions, etc.) the
    longest single segment stops at that junction, truncating the centerline
    at mid-body.  Walking the full graph finds the true longest path from one
    tip to the other.

    Self-overlapping worms ("O" / "6" shapes)
    ------------------------------------------
    When the worm curls back onto itself the medial axis becomes a closed
    loop, producing a cycle in the graph and **no** degree-1 endpoints.
    ``_break_cycles_at_minimum_width`` removes one edge per cycle at the
    narrowest skeleton point (minimum distance-transform value), which
    corresponds to the self-overlap crossing.  This converts the loop into an
    open path so the normal endpoint search can proceed.  For a "6" shape the
    free tail remains as one endpoint and the cut point in the loop becomes
    the other, yielding a full head-to-tail traversal.

    Supported graph topologies (6 cases)
    -------------------------------------
    The function handles the following skeleton graph structures.  Cases are
    evaluated in the order listed; the first match wins.

    Case 1 — Normal worm (no loops)
        Graph:   ep1 — [optional intermediate junctions] — ep2
        Detect:  no self-loop nodes; graph is a forest (acyclic).
        Build:   all-pairs longest shortest-path search over degree-1
                 endpoints; arc segments stitched in order.

    Case 2 — Pure "O" shape (full ring, no free tails)
        Graph:   J ↺   (single junction node with one self-loop edge)
        Detect:  self-loop node found; after removing the self-loop, no
                 degree-1 endpoints remain (endpoint_branches is empty).
        Build:   ring pixel array rotated so the narrowest body point
                 (minimum distance-transform value along the ring) becomes
                 both start and end, opening the loop into a linear path.

    Case 3 — "6" shape (one free tail + one self-loop junction)
        Graph:   ep1 — J ↺
        Detect:  self-loop node found; after removing the self-loop, exactly
                 one degree-1 endpoint is reachable from the self-loop node
                 (len(endpoint_branches) == 1).
        Build:   branch reversed to ep1 → J, then ring pixels appended from
                 index [1:] (ring cut at the natural junction seam, avoiding
                 spatial jumps or backtracking kinks).

    Case 4 — Two free tails flanking a self-loop junction
        Graph:   ep1 — J ↺ — ep2
        Detect:  self-loop node found; after removing the self-loop, two or
                 more degree-1 endpoints are reachable (len(endpoint_branches)
                 >= 2).  The two longest branches are kept.
        Build:   branch_A reversed (ep1 → J) + ring interior + branch_B (J → ep2).
                 The ring interior is traversed in whichever direction (forward
                 ring_coords[1:-1] or reversed ring_coords[-2:0:-1]) produces
                 the smoothest path: the direction is chosen to maximise the sum
                 of cos(turning angle) at both branch–ring junctions.

    Case 5 — Two free tails, inner self-loop junction behind an outer junction
        Graph:   ep1 — J1 — J2 ↺ — J1 — ep2
                 (J2 holds the self-loop; J1 is an intermediate junction
                  connecting J2 to both endpoint stems)
        Detect:  same code path as Case 4 — self-loop node J2 is found.
                 After removing the self-loop, nx.shortest_path from J2 to
                 each degree-1 endpoint returns 2-hop paths [J2, J1, ep_x].
                 _path_coords concatenates the two arc-segments per branch
                 transparently, so the multi-hop structure requires no special
                 handling beyond Case 4.
        Build:   identical assembly to Case 4 (with ring direction selection):
                 branch_A reversed (ep1→J1→J2) + ring interior + branch_B (J2→J1→ep2).

    Case 6 — Omega / Ω shape (two junctions connected by two parallel arcs)
        Graph:   ep1 — J1 ══(arc_A)══ J2 — ep2
                          ══(arc_B)══
                 (J1 and J2 are connected by exactly TWO distinct arcs,
                  creating a non-self-loop cycle in the MultiGraph)
        Detect:  no self-loop nodes; graph is NOT a forest; nx.find_cycle
                 returns exactly 2 edges sharing the same unordered node pair
                 {J1, J2} — i.e. they are parallel edges.
        Build:   the worm body passes through the crossing region twice
                 (once inbound, once outbound), so one arc is used as the
                 "bridge" (traversed twice) and the other arc is traversed
                 once.  Two candidate assemblies are evaluated:
                     T1:  ep1 → J1 → arc_A → J2 → arc_B_rev → J1 → arc_A → J2 → ep2
                     T2:  ep1 → J1 → arc_B → J2 → arc_A_rev → J1 → arc_B → J2 → ep2
                 If ``arc_a`` (the shorter arc) is shorter than
                 ``min_bridge_arc_len_px``, direction scoring is unreliable and
                 the shorter arc is used as bridge unconditionally (heuristic:
                 minimises total repeated path length).  Otherwise both T1 and
                 T2 are scored and the candidate that maximises the sum of
                 cos(turning angle) at the entry and exit seams is selected.
    """
    paths_list = skel_obj.paths_list()
    lengths    = skel_obj.path_lengths()

    # Use MultiGraph so that parallel arcs between the same pair of junction
    # nodes (e.g. both sides of a ring) are preserved as distinct edges.
    # nx.Graph would silently deduplicate them, keeping only the longer arc
    # and making the ring appear acyclic — causing the shorter arc to be
    # permanently abandoned from the centerline.
    G = nx.MultiGraph()
    for idx, (px_path, length) in enumerate(zip(paths_list, lengths)):
        src, dst = int(px_path[0]), int(px_path[-1])
        G.add_edge(src, dst, key=idx, length=length, path_idx=idx)

    # ── Helper: get the single edge-data dict between u and v ──────────────
    # After cycle-breaking the graph is a forest, so each adjacent pair has
    # at most one edge.  This helper retrieves that edge's data dict.
    def _edge_data(G_ref: nx.MultiGraph, u: int, v: int) -> dict:
        return next(iter(G_ref[u][v].values()))

    # ── Cases 2 / 3 / 4 / 5 — self-loop detected ──────────────────────────────
    # A worm that forms a complete ring produces a self-loop edge in the path
    # graph (the entire ring arc has src == dst = the junction node on the
    # ring).  Additional branch edges connect that junction outward through
    # any bridge segments toward the free endpoints (the "overhang" stems of
    # an omega/Ω shape).
    #
    # Strategy
    # --------
    # 1.  Identify the dominant self-loop node (most ring pixels).
    # 2.  In a graph copy with the self-loop removed, find the longest
    #     coordinate chain from the self-loop node to each degree-1 endpoint
    #     (walking through any intermediate junctions / bridge segments).
    # 3.  Assemble the full head-to-tail path:
    #         branch_to_epA_reversed  +  ring_arc[1:-1]  +  branch_to_epB
    #     If no endpoint branches exist (pure "O"), cut the ring at its
    #     narrowest point (minimum distance-transform value) and return that
    #     as an open path.
    self_loop_nodes = [n for n in G.nodes() if G.has_edge(n, n)]
    if self_loop_nodes:
        # Pick the self-loop with the most pixels (dominant ring arc).
        def _best_sl_data(n: int) -> dict | None:
            candidates = [
                d for d in G[n][n].values() if d.get("path_idx") is not None
            ]
            if not candidates:
                return None
            return max(
                candidates,
                key=lambda d: len(skel_obj.path_coordinates(d["path_idx"])),
            )

        sl_node = max(
            (n for n in self_loop_nodes if _best_sl_data(n) is not None),
            key=lambda n: len(
                skel_obj.path_coordinates(_best_sl_data(n)["path_idx"])
            ),
            default=None,
        )
        if sl_node is None:
            sl_node = self_loop_nodes[0]

        sl_edata    = _best_sl_data(sl_node)
        sl_path_idx = sl_edata["path_idx"] if sl_edata else None
        ring_coords = (
            skel_obj.path_coordinates(sl_path_idx)
            if sl_path_idx is not None else None
        )

        # ── Helper: collect pixel coords along a node path ─────────────────
        def _path_coords(
            node_path: list[int], G_ref: nx.MultiGraph
        ) -> np.ndarray:
            """Return (M, 2) row-col array for an ordered list of graph nodes."""
            segs: list[np.ndarray] = []
            for k in range(len(node_path) - 1):
                u2, v2 = node_path[k], node_path[k + 1]
                pi2    = _edge_data(G_ref, u2, v2).get("path_idx")
                if pi2 is None:
                    continue
                seg2 = skel_obj.path_coordinates(pi2)
                pp2  = paths_list[pi2]
                if int(pp2[0]) != u2:
                    seg2 = seg2[::-1]
                if segs:             # drop duplicate junction pixel
                    seg2 = seg2[1:]
                segs.append(seg2)
            return np.vstack(segs) if segs else np.empty((0, 2))

        # ── Find endpoint branches via graph without the self-loop ──────────
        G_no_loop = G.copy()
        # Remove ALL self-loop edges at sl_node.
        for k in list(G_no_loop[sl_node][sl_node].keys()):
            G_no_loop.remove_edge(sl_node, sl_node, k)

        # For each degree-1 endpoint reachable from sl_node, record the
        # node path and its total arc length (sum of edge lengths).
        endpoint_branches: list[tuple[float, list[int]]] = []
        for ep in [n for n in G_no_loop.nodes() if G_no_loop.degree(n) == 1]:
            try:
                npath = nx.shortest_path(G_no_loop, sl_node, ep)
                total = sum(
                    _edge_data(G_no_loop, npath[k], npath[k + 1])["length"]
                    for k in range(len(npath) - 1)
                )
                endpoint_branches.append((total, npath))
            except nx.NetworkXNoPath:
                pass

        # Sort longest-branch first so we always keep the two biggest stems.
        endpoint_branches.sort(key=lambda t: t[0], reverse=True)

        if ring_coords is None or len(ring_coords) < 4:
            # Self-loop was detected but the ring arc pixel array is invalid —
            # either skan returned no coordinates for the dominant self-loop arc,
            # or it has fewer than 4 pixels (too small to fit a spline).
            # Cannot handle via Cases 2/3/4/5; falling through to Case 6 /
            # Fallback A / Case 1.
            n_ring = 0 if ring_coords is None else len(ring_coords)
            _pfx = f"[{img_name}] " if img_name else ""
            warnings.warn(
                f"{_pfx}Self-loop node detected (Cases 2/3/4/5) but ring_coords is "
                f"{'None' if ring_coords is None else f'only {n_ring} pixel(s) long'} "
                f"(need \u2265 4). Cannot reconstruct centerline from the ring arc. "
                f"Falling through to Case 6 / Fallback A / Case 1.",
                RuntimeWarning,
                stacklevel=2,
            )
        if ring_coords is not None and len(ring_coords) >= 4:
            if len(endpoint_branches) >= 2:
                # ── Case 4 / 5: two free tails flanking the self-loop junction ──────
                # Case 4: ep1 — J ↺ — ep2  (direct branches from the loop node)
                # Case 5: ep1 — J1 — J2 ↺ — J1 — ep2  (branches are multi-hop;
                #          _path_coords handles the extra J1 hop transparently)
                # Two overhangs: epA → ... → sl_node → ring → sl_node → ... → epB
                # branch paths are oriented sl_node→endpoint; reverse the first.
                branch_a = _path_coords(endpoint_branches[0][1], G_no_loop)[::-1]
                branch_b = _path_coords(endpoint_branches[1][1], G_no_loop)
                # ring_coords[0] == ring_coords[-1] == sl_node pixel;
                # drop the shared junction pixels at the seams.
                #
                # ── Choose ring traversal direction to minimise sharp turns ──
                # The ring interior can be walked in two directions; pick the
                # one whose tangent at the entry junction (end of branch_a →
                # first ring pixel) AND exit junction (last ring pixel → start
                # of branch_b) together produce the smallest cumulative turning
                # angle.  Score = Σ cos(turning angle) at both junctions;
                # higher score = smoother path.
                ring_fwd = ring_coords[1:-1]     # forward around the loop
                ring_rev = ring_coords[-2:0:-1]  # reversed around the loop

                def _junction_smoothness(ring_interior: np.ndarray) -> float:
                    """Sum of cos(turning angle) at entry and exit junctions."""
                    score = 0.0
                    # Entry: branch_a end → first ring pixel
                    if len(branch_a) >= 2 and len(ring_interior) >= 1:
                        wi = _arc_len_index(branch_a,      dir_vector_len_px, from_end=True)
                        wo = _arc_len_index(ring_interior, dir_vector_len_px, from_end=False)
                        v_in  = branch_a[-1] - branch_a[-1 - wi]
                        v_out = ring_interior[wo] - branch_a[-1]
                        n_in, n_out = (np.linalg.norm(v_in),
                                       np.linalg.norm(v_out))
                        if n_in > 0 and n_out > 0:
                            score += float(np.clip(
                                np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0
                            ))
                    # Exit: last ring pixel → branch_b[1]
                    # (branch_b[0] is the duplicate sl_node pixel, skipped in
                    # the assembly below, so use branch_b[1] as the first new
                    # pixel after the junction.)
                    if len(ring_interior) >= 2 and len(branch_b) >= 2:
                        wi = _arc_len_index(ring_interior, dir_vector_len_px, from_end=True)
                        wo = _arc_len_index(branch_b,      dir_vector_len_px, from_end=False)
                        v_in  = ring_interior[-1] - ring_interior[-1 - wi]
                        v_out = branch_b[wo] - ring_interior[-1]
                        n_in, n_out = (np.linalg.norm(v_in),
                                       np.linalg.norm(v_out))
                        if n_in > 0 and n_out > 0:
                            score += float(np.clip(
                                np.dot(v_in, v_out) / (n_in * n_out), -1.0, 1.0
                            ))
                    return score

                ring_interior = (
                    ring_fwd
                    if _junction_smoothness(ring_fwd) >= _junction_smoothness(ring_rev)
                    else ring_rev
                )

                parts = []
                if len(branch_a):
                    parts.append(branch_a)
                parts.append(ring_interior if len(parts) else ring_coords)
                if len(branch_b):
                    parts.append(branch_b[1:])   # skip dup sl_node pixel
                result = np.vstack(parts)
                if len(result) >= 4:
                    return result

            elif len(endpoint_branches) == 1:
                # ── Case 3: "6" shape — one free tail + self-loop ────────────────
                # Graph: ep1 — J ↺
                # One overhang ("6" shape): epA → sl_node → ring
                # Cut the ring at sl_node (index 0 == index N-1) rather than
                # at the narrowest cross-section.  branch_a[-1] is already
                # sl_node == ring_coords[0] == ring_coords[-1], so
                # ring_coords[1:] continues seamlessly from the junction with
                # no spatial jump and no U-turn kink at sl_node mid-traversal.
                # Using argmin(dist) as the cut could place the seam anywhere
                # mid-ring, causing a chord jump at sl_node and a backtrack
                # segment that systematically biases the measured body length.
                branch_a = _path_coords(endpoint_branches[0][1], G_no_loop)[::-1]
                result = np.vstack([branch_a, ring_coords[1:]])
                if len(result) >= 4:
                    return result

            else:
                # ── Case 2: pure "O" shape — full ring, no free tails ────────────
                # Graph: J ↺  (no degree-1 endpoints after self-loop removal)
                # Pure "O" — no overhangs: cut the ring at min-width.
                r2 = np.clip(ring_coords[:, 0].astype(int), 0,
                             skel_obj.source_image.shape[0] - 1)
                c2 = np.clip(ring_coords[:, 1].astype(int), 0,
                             skel_obj.source_image.shape[1] - 1)
                cut_idx  = int(np.argmin(skel_obj.source_image[r2, c2]))
                ordered  = np.concatenate(
                    [ring_coords[cut_idx:], ring_coords[1 : cut_idx + 1]], axis=0
                )
                if len(ordered) >= 4:
                    return ordered

    # ── Case 6 — non-self-loop cycle: parallel-arc ring (omega / Ω shape) ────
    if not nx.is_forest(G):
        # ── Case 6: two junctions connected by two parallel arcs ───────────────
        # Graph: ep1 — J1 ══(short arc)══ J2 — ep2
        #                  ══(long arc) ══
        # Detected when nx.find_cycle returns exactly 2 edges sharing the same
        # unordered node pair {J1, J2} (i.e. parallel edges in the MultiGraph).
        # The ring skeleton has exactly two junction nodes (u_j, v_j) that are
        # connected by TWO parallel arcs: a short crossing-stub arc and a long
        # ring-body arc.  Endpoint branches extend outward from each junction
        # to the two free tips.
        #
        # The worm body physically crosses over itself at the junction region,
        # passing through that crossing TWICE — once on the way in and once on
        # the way out after going around the big ring.  The correct centerline
        # therefore visits the crossing arc twice:
        #
        #   ep_A → branch_A → u_j → short_arc → v_j
        #        → long_arc_reversed → u_j → short_arc → v_j
        #        → branch_B → ep_B
        #
        # This gives a continuous, gap-free path that faithfully represents the
        # worm topology.  The short arc appears twice in the coordinate array,
        # which is biologically correct: both body layers at the crossing are
        # traced.
        #
        # Detection: the cycle returned by nx.find_cycle has exactly 2 edges
        # that share the same unordered junction-node pair — i.e. they are
        # parallel edges in the MultiGraph.
        try:
            cycle_edges = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            cycle_edges = []

        parallel_arc_result = None
        if len(cycle_edges) == 2:
            (ca_u, ca_v, ca_k), (cb_u, cb_v, cb_k) = cycle_edges
            if {ca_u, ca_v} == {cb_u, cb_v}:          # same junction pair
                u_j, v_j = ca_u, ca_v

                # Identify short arc (lower arc length) and long arc.
                arcs = []
                for k in list(G[u_j][v_j].keys()):
                    d = G[u_j][v_j][k]
                    arcs.append((d.get("length", np.inf), k, d.get("path_idx")))
                arcs.sort()                            # shortest first
                if len(arcs) >= 2:
                    _, short_k, short_pi = arcs[0]
                    _, long_k,  long_pi  = arcs[1]

                    def _arc_coords(pi: int, from_node: int) -> np.ndarray:
                        """Path coords for arc pi, oriented so coords[0] is from_node."""
                        c = skel_obj.path_coordinates(pi)
                        if int(paths_list[pi][0]) != from_node:
                            c = c[::-1]
                        return c

                    # arc_a: u_j → v_j (first arc in sorted order)
                    # arc_b: u_j → v_j (second arc in sorted order)
                    # arc_a_r / arc_b_r: same arcs reversed (v_j → u_j)
                    arc_a    = _arc_coords(short_pi, u_j)   # u_j → v_j
                    arc_b    = _arc_coords(long_pi,  u_j)   # u_j → v_j
                    arc_a_r  = arc_a[::-1]                  # v_j → u_j
                    arc_b_r  = arc_b[::-1]                  # v_j → u_j

                    # Find the endpoint branch attached to each junction.
                    # Remove both parallel arcs; remaining edges are branches.
                    G_branches = G.copy()
                    for k in list(G_branches[u_j][v_j].keys()):
                        G_branches.remove_edge(u_j, v_j, k)

                    def _branch_to_ep(start_node: int) -> np.ndarray | None:
                        """Return coords from start_node to its nearest endpoint."""
                        eps = [
                            n for n in nx.node_connected_component(G_branches, start_node)
                            if G_branches.degree(n) <= 1 and n != start_node
                        ]
                        if not eps:
                            return np.empty((0, 2))
                        ep = eps[0]
                        try:
                            npath = nx.shortest_path(G_branches, start_node, ep)
                        except nx.NetworkXNoPath:
                            return np.empty((0, 2))
                        segs: list[np.ndarray] = []
                        for ki in range(len(npath) - 1):
                            u2, v2 = npath[ki], npath[ki + 1]
                            pi2 = _edge_data(G_branches, u2, v2).get("path_idx")
                            if pi2 is None:
                                continue
                            seg = skel_obj.path_coordinates(pi2)
                            if int(paths_list[pi2][0]) != u2:
                                seg = seg[::-1]
                            if segs:
                                seg = seg[1:]
                            segs.append(seg)
                        return np.vstack(segs) if segs else np.empty((0, 2))

                    branch_a = _branch_to_ep(u_j)   # u_j outward → ep_A
                    branch_b = _branch_to_ep(v_j)   # v_j outward → ep_B

                    # ── Choose which arc is the bridge (traversed twice) ──────
                    # Two candidate topologies:
                    #   T1: ep_A → u_j → arc_a → v_j → arc_b_r → u_j → arc_a → v_j → ep_B
                    #   T2: ep_A → u_j → arc_b → v_j → arc_a_r → u_j → arc_b → v_j → ep_B
                    # Score each by summing cos(turning angle) at all 4 seams.
                    def _seam_cos_c6(arr_prev: np.ndarray,
                                     arr_next: np.ndarray) -> float:
                        """Cosine of turning angle: arr_prev[-1] is the junction pixel;
                        arr_next[0] is the first pixel AFTER the junction."""
                        if len(arr_prev) < 2 or len(arr_next) < 1:
                            return 0.0
                        wi = _arc_len_index(arr_prev, dir_vector_len_px, from_end=True)
                        wo = _arc_len_index(arr_next, dir_vector_len_px, from_end=False)
                        v_in  = arr_prev[-1] - arr_prev[-1 - wi]
                        v_out = arr_next[wo] - arr_prev[-1]
                        n_in  = np.linalg.norm(v_in)
                        n_out = np.linalg.norm(v_out)
                        if n_in == 0 or n_out == 0:
                            return 0.0
                        return float(np.dot(v_in, v_out) / (n_in * n_out))

                    def _build_c6(bridge: np.ndarray,
                                  single_r: np.ndarray) -> tuple[np.ndarray, float]:
                        """
                        Assemble one Case 6 candidate and compute its smoothness
                        score.  bridge goes u_j → v_j; single_r goes v_j → u_j.
                        Layout: ep_A→u_j | bridge | single_r | bridge | v_j→ep_B
                        """
                        # Build as a list of segments (junction pixel already at
                        # end of previous segment; duplicate dropped from start
                        # of next segment — consistent with Cases 4/5).
                        seg_entry  = branch_a[::-1]          # ep_A → u_j
                        seg_bridge = bridge[1:]              # u_j → v_j  (drop dup)
                        seg_single = single_r[1:]            # v_j → u_j  (drop dup)
                        seg_bridge2= bridge[1:]              # u_j → v_j  (second pass)
                        seg_exit   = branch_b[1:]            # v_j → ep_B (drop dup)

                        segs: list[np.ndarray] = []
                        if len(seg_entry):
                            segs.append(seg_entry)
                        segs.append(seg_bridge if segs else bridge)
                        segs.append(seg_single)
                        segs.append(seg_bridge2)
                        if len(seg_exit):
                            segs.append(seg_exit)

                        # Only score the entry (branch_A → bridge) and exit
                        # (bridge2 → branch_B) seams.  The two turnaround seams
                        # at v_j / u_j are structural U-turns that are roughly
                        # equal for both topologies and do not help discriminate.
                        score = 0.0
                        if len(seg_entry):   # seam 0: entry → bridge
                            score += _seam_cos_c6(segs[0], segs[1])
                        if len(seg_exit):    # last seam: bridge2 → exit
                            score += _seam_cos_c6(segs[-2], segs[-1])
                        return np.vstack(segs), score

                    # ── Bridge selection ────────────────────────────────────
                    # Since arcs is sorted ascending, arc_a <= arc_b always.
                    # If arc_a < min_bridge_arc_len_px then at least one arc is
                    # too short for reliable direction scoring → skip scoring
                    # entirely and use the shorter arc (arc_a) as bridge.
                    # Only follow the score when both arcs exceed the threshold.
                    arc_a_len_px = float(arcs[0][0])
                    if arc_a_len_px < min_bridge_arc_len_px:
                        arr_t1, score_t1 = _build_c6(arc_a, arc_b_r)
                        result = arr_t1
                    else:
                        arr_t1, score_t1 = _build_c6(arc_a, arc_b_r)  # arc_a is bridge
                        arr_t2, score_t2 = _build_c6(arc_b, arc_a_r)  # arc_b is bridge
                        result = arr_t1 if score_t1 >= score_t2 else arr_t2
                    if len(result) >= 4:
                        parallel_arc_result = result

        if parallel_arc_result is not None:
            return parallel_arc_result

        # Fallback A — general multi-edge cycle (e.g. triangular junction blob):
        # The cycle has more than 2 edges, or its 2 edges don't share the same
        # node pair, so it doesn't match Case 6.  Break cycles at the shortest
        # arc and fall through to Case 1 below.
        n_cycles_broken = G.number_of_edges() - (G.number_of_nodes() - nx.number_connected_components(G))
        _pfx = f"[{img_name}] " if img_name else ""
        warnings.warn(
            f"{_pfx}Fallback A: skeleton graph has an unrecognised cycle topology "
            f"(cycle edges returned: {len(cycle_edges)}, "
            f"estimated cycles to break: {max(n_cycles_broken, 1)}). "
            f"Breaking cycles at minimum-width arcs and falling back to Case 1 "
            f"(normal endpoint search). The centerline may be suboptimal.",
            RuntimeWarning,
            stacklevel=3,
        )
        _break_cycles_at_minimum_width(G, skel_obj, paths_list)

    # ── Case 1 — normal worm (acyclic graph, or cycle-broken fallback) ────────
    # Graph: ep1 — [optional intermediate junctions] — ep2
    # Also handles any topology that was reduced to a forest by
    # _break_cycles_at_minimum_width above.
    endpoints = [n for n in G.nodes() if G.degree(n) == 1]
    if len(endpoints) < 2:
        # Fallback B — fewer than 2 degree-1 endpoints after all cycle handling.
        # Typical cause: degenerate / very small mask whose skeleton collapsed to
        # a single pixel or an isolated junction node with no leaf edges.  Promote
        # all nodes to candidates so the path search can still attempt a result.
        _pfx = f"[{img_name}] " if img_name else ""
        warnings.warn(
            f"{_pfx}Fallback B: only {len(endpoints)} degree-1 endpoint(s) found after "
            f"cycle handling (graph has {G.number_of_nodes()} node(s), "
            f"{G.number_of_edges()} edge(s)). "
            f"Promoting all nodes to endpoint candidates. "
            f"The skeleton may be degenerate or the mask too small.",
            RuntimeWarning,
            stacklevel=3,
        )
        endpoints = list(G.nodes())

    best_length    = -1.0
    best_node_path = None
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            try:
                # nx.shortest_path is used here because after all cycle-breaking
                # the graph is a forest (tree), so there is exactly ONE path
                # between any two nodes — "shortest" simply means "the only path"
                # at the graph-topology level.  The "longest" selection below
                # operates at the arc-length metric level: among all endpoint
                # pairs we keep whichever unique path has the greatest total
                # pixel arc length (sum of edge "length" attributes), which
                # corresponds to the true head-to-tail body length of the worm.
                node_path = nx.shortest_path(G, endpoints[i], endpoints[j])
                total = sum(
                    _edge_data(G, node_path[k], node_path[k + 1])["length"]
                    for k in range(len(node_path) - 1)
                )
                if total > best_length:
                    best_length    = total
                    best_node_path = node_path
            except nx.NetworkXNoPath:
                continue

    if best_node_path is None:
        _pfx = f"[{img_name}] " if img_name else ""
        warnings.warn(
            f"{_pfx}Skeleton graph traversal failed: no path found between any pair of "
            f"endpoints (graph has {G.number_of_nodes()} node(s), "
            f"{G.number_of_edges()} edge(s)). "
            f"The graph may be fully disconnected after cycle breaking. "
            f"Returning None centerline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    all_coords: list[np.ndarray] = []
    for k in range(len(best_node_path) - 1):
        u, v       = best_node_path[k], best_node_path[k + 1]
        edata      = _edge_data(G, u, v)
        path_idx   = edata["path_idx"]
        seg_coords = skel_obj.path_coordinates(path_idx)   # (M, 2) row-col
        px_path    = paths_list[path_idx]
        if int(px_path[0]) != u:
            seg_coords = seg_coords[::-1]
        if k > 0:
            seg_coords = seg_coords[1:]   # drop duplicate junction pixel
        all_coords.append(seg_coords)

    return np.vstack(all_coords) if all_coords else None


# ---------------------------------------------------------------------------
# Core skeletonization
# ---------------------------------------------------------------------------

def extract_centerline(
    mask: np.ndarray,
    n_points: int = 100,
    px_per_point: float | None = None,
    smooth_factor: float | None = None,
    smooth_factor_ratio: float = 1.0,
    contour_smooth_sigma: float = 2.0,
    hole_fill_ratio_threshold: float = 0.05,
    hole_keep_circularity_weight: float = 0.5,
    ring_dilation_radius: int = 0,
    dir_vector_len_px: float = 20.0,
    min_bridge_arc_len_px: float = 20.0,
    debug_ring_mask_path: str | None = None,
    img_name: str = "",
) -> tuple[np.ndarray | None, np.ndarray | None, bool, np.ndarray]:
    """
    Fit a smooth centerline to a single binary worm mask.

    Parameters
    ----------
    mask :
        2-D uint8 / bool array (H × W).  Non-zero pixels = foreground.
    n_points :
        Number of evenly-spaced points on the output spline.
    smooth_factor :
        Spline smoothing parameter passed to ``scipy.interpolate.splprep``.
        ``None`` → auto (``smooth_factor_ratio * len(skeleton_pixels)``).
    smooth_factor_ratio :
        Multiplier applied to ``len(skeleton_pixels)`` when ``smooth_factor``
        is ``None``.  ``1.0`` (default) matches SciPy's recommended midpoint.
        ``0`` → exact interpolation (wiggly);  ``>1`` → heavier smoothing.
        Typical values: 0, 0.5, 1.0, 5.0.
    contour_smooth_sigma :
        Standard deviation (pixels) of the Gaussian blur applied to the
        binary mask **before** computing the medial axis.  This rounds off
        sharp polygon corners in the YOLO mask that would otherwise produce
        spurious branches in the skeleton.  Set to 0 to disable.
    hole_fill_ratio_threshold :
        Fraction of the original mask area that ``binary_fill_holes`` must add
        before the mask is treated as a topologically significant ring ("O" /
        "6" shape) and hole-filling is skipped.  Lower values make the
        detection more sensitive to smaller loops.  Default 0.05 (5%).
    hole_keep_circularity_weight :
        Weight in ``[0, 1]`` controlling how much circularity vs normalised
        area contributes to the score used to decide which enclosed hole is
        the loop interior (and therefore kept unfilled).  The combined score
        per hole is::

            score = w * circularity + (1 - w) * (area / max_area)

        where ``w`` = this parameter.  The hole with the **highest** score is
        kept unfilled; all others (body-overlap gaps) are filled.
        ``1.0`` = pure circularity;  ``0.0`` = pure area;  default ``0.5``.
    ring_dilation_radius :
        Radius (pixels) of the elliptical dilation applied to ``mask_raw``
        **before** both the ``is_ring`` detection step and the selective hole
        analysis.  Dilation strengthens weak or narrow mask connections so
        that ``binary_fill_holes`` can see the enclosed loop area, and also
        bridges body-overlap gaps before hole labelling.  ``mask_raw`` itself
        is never modified (it is shown as the "before" panel in the debug
        image).  Set to 0 (default) to disable.
    is_ring_detection_dilation_radius :
        Deprecated — now merged into ``ring_dilation_radius``.  Ignored.
    dir_vector_len_px :
        Arc length in pixels measured from the junction seam in each direction
        to estimate the incoming and outgoing directional vectors for
        smoothness scoring.  The helper walks the skeleton coordinate array
        until the cumulative Euclidean distance reaches this value, then uses
        the chord from that anchor point to the junction.  Larger values
        average out local pixel-level noise; if a segment is shorter than the
        target the full segment is used.  Default 20.0.
    min_bridge_arc_len_px :
        Minimum arc length (pixels) required for the Case 6 (omega / Ω ring)
        direction-scoring step to be considered reliable.  Since skan arcs are
        sorted ascending, ``arc_a`` (shorter) is always <= ``arc_b`` (longer).
        If ``arc_a < min_bridge_arc_len_px`` the direction vectors are too
        short to be trustworthy, so scoring is skipped entirely and arc_a
        (the shorter arc) is used as the bridge — the heuristic that minimises
        total repeated path length.  Default 20.0.
    debug_ring_mask_path :
        Optional file path (e.g. ``".../img_f00_ring.jpg"``) at which to save
        a 2-panel debug image showing the mask before and after selective hole
        fill.  Only written when ``is_ring`` is ``True``.  ``None`` = disabled.

    Returns
    -------
    centerline :
        ``(n_points, 2)`` float array of **(row, col)** positions, or ``None``.
    width_profile :
        ``(n_points,)`` float array of local body *diameter* in pixels
        (= 2 × distance-transform value at each centerline point), or ``None``.
    success : bool
    effective_mask :
        The hole-filled boolean mask actually used for skeletonization
        (same shape as *mask*).  Useful for callers that need to know
        which pixels are considered foreground after selective hole filling.
    """
    mask_raw = mask.astype(bool)
    if mask_raw.sum() < 20:
        return None, None, False, mask_raw

    # ── Step 0: topology-aware hole filling ────────────────────────────────
    # ``binary_fill_holes`` removes small internal voids — desirable for normal
    # worms.  BUT for self-overlapping worms ("O" / "6" shapes) the worm body
    # forms a ring, so the large enclosed background region must stay unfilled
    # or the skeleton degenerates into a solid disc.
    #
    # For ring masks YOLO can produce TWO kinds of holes:
    #   • One LARGE hole  = background pixels enclosed by the loop body.
    #                       Must be kept unfilled to preserve ring topology.
    #   • One/more SMALL holes = overlapping body segments that YOLO left
    #                       transparent.  Should be filled so the skeleton
    #                       runs continuously through the crossing region.
    #
    # Strategy:
    #   1. Detect whether the mask is a ring via the hole-area heuristic.
    #   2. For normal worms: fill all holes with ``binary_fill_holes``.
    #   3. For ring worms: label all enclosed background components, keep the
    #      LARGEST one unfilled (= background loop interior), and fill every
    #      smaller component (= occluded-body gaps).
    # Optionally dilate mask_raw before hole-filling to bridge weak/narrow
    # connections (e.g. a nearly-open loop).  The dilated mask is used for
    # both the is_ring detection step and the subsequent hole analysis.
    # mask_raw itself is never modified (shown as "before" in the debug image).
    if ring_dilation_radius > 0:
        _ksize_rd = 2 * ring_dilation_radius + 1
        _kernel_rd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ksize_rd, _ksize_rd))
        mask_work = cv2.dilate(mask_raw.astype(np.uint8), _kernel_rd).astype(bool)
    else:
        mask_work = mask_raw
    mask_filled = binary_fill_holes(mask_work)
    added_ratio = (mask_filled.sum() - mask_work.sum()) / max(mask_work.sum(), 1)
    is_ring = added_ratio > hole_fill_ratio_threshold
    if is_ring:
        # Components that touch the image border are outer background; all
        # others are enclosed holes.
        inv = ~mask_work
        labeled_bg, n_bg = sk_label(inv, connectivity=1, return_num=True)

        border_labels: set[int] = set()
        border_labels.update(int(v) for v in labeled_bg[0, :])
        border_labels.update(int(v) for v in labeled_bg[-1, :])
        border_labels.update(int(v) for v in labeled_bg[:, 0])
        border_labels.update(int(v) for v in labeled_bg[:, -1])
        border_labels.discard(0)   # 0 = foreground pixels, ignore

        hole_labels = [
            lbl for lbl in range(1, n_bg + 1) if lbl not in border_labels
        ]

        if len(hole_labels) > 1:  # noqa: SIM102
            # Multiple holes: keep the hole with the highest combined score
            # of circularity (loop interior ≈ round) and normalised area
            # (loop interior is usually larger than body-overlap gaps).
            #
            # score = w * circularity + (1-w) * (area / max_area)
            # w = hole_keep_circularity_weight  (0 = pure area, 1 = pure circularity)
            def _circularity(lbl: int) -> float:
                hole_mask_u8 = (labeled_bg == lbl).astype(np.uint8)
                cnts, _ = cv2.findContours(
                    hole_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if not cnts:
                    return 0.0
                area = float(cv2.contourArea(cnts[0]))
                perim = float(cv2.arcLength(cnts[0], closed=True))
                if perim < 1e-6:
                    return 0.0
                return (4.0 * np.pi * area) / (perim ** 2)

            hole_areas_raw = [
                (int((labeled_bg == lbl).sum()), lbl) for lbl in hole_labels
            ]
            max_area = max(a for a, _ in hole_areas_raw) or 1
            w = float(np.clip(hole_keep_circularity_weight, 0.0, 1.0))
            hole_scores = [
                (
                    w * _circularity(lbl) + (1.0 - w) * (area / max_area),
                    lbl,
                )
                for area, lbl in hole_areas_raw
            ]
            hole_scores.sort(reverse=True)   # highest score first
            keep_lbl = hole_scores[0][1]      # loop interior

            fill_mask = mask_work.copy()
            for _, lbl in hole_scores[1:]:    # fill all lower-scoring holes
                fill_mask[labeled_bg == lbl] = True
            mask_bool = fill_mask
        else:
            # Zero or one enclosed hole — nothing extra to fill.
            mask_bool = mask_work
    else:
        # Normal (non-ring) worm: fill holes on the ORIGINAL (undilated) mask.
        # ring_dilation_radius was applied to mask_work solely for the is_ring
        # detection and hole-labelling step above.  Using it here would
        # artificially fatten the worm body, widening the medial axis and
        # biasing the width profile.  mask_raw is always the unmodified input.
        mask_bool = binary_fill_holes(mask_raw)

    # ── Debug: save ring mask processing stages ─────────────────────────────
    if is_ring and debug_ring_mask_path is not None:
        def _mask_panel(m: np.ndarray, title: str) -> np.ndarray:
            """Convert bool mask to an annotated BGR panel."""
            bgr = np.zeros((*m.shape, 3), dtype=np.uint8)
            bgr[m] = (200, 200, 200)
            cv2.putText(bgr, title, (4, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1, cv2.LINE_AA)
            return bgr
        dil_tag = f", dil={ring_dilation_radius}px" if ring_dilation_radius > 0 else ""
        panels = [
            _mask_panel(mask_raw,   "1: Original"),
            _mask_panel(mask_bool,  f"2: Selective fill (circ_w={hole_keep_circularity_weight:.2f}{dil_tag})"),
        ]
        sep = np.zeros((mask_raw.shape[0], 3, 3), dtype=np.uint8)
        debug_ring = np.concatenate([panels[0], sep, panels[1]], axis=1)
        cv2.imwrite(debug_ring_mask_path, debug_ring)

    # ── Step 1: smooth mask contour ─────────────────────────────────────────
    # YOLO polygon masks have a limited vertex count, leaving sharp 90° corner
    # artifacts.  The medial axis branches toward each such corner, fragmenting
    # the skeleton into many short segments at junction nodes.  A Gaussian blur
    # followed by re-thresholding at 0.5 rounds these corners before the medial
    # axis is computed.
    effective_sigma = contour_smooth_sigma
    if effective_sigma > 0:
        ksize = int(effective_sigma * 6) | 1   # next odd number ≥ 6σ
        blurred = cv2.GaussianBlur(
            mask_bool.astype(np.float32), (ksize, ksize), effective_sigma
        )
        smoothed = blurred > 0.5
        # Re-apply hole-fill only when the mask did NOT have a significant hole
        # (to avoid re-introducing the solid-disc problem on "O" shapes).
        if not is_ring:
            mask_bool = binary_fill_holes(smoothed)
        else:
            mask_bool = smoothed

    # Medial axis + Euclidean distance transform in one call.
    # `dist[r, c]` = distance from skeleton pixel (r,c) to nearest background pixel.
    skel, dist = medial_axis(mask_bool, return_distance=True)

    _pfx = f"[{img_name}] " if img_name else ""
    if skel.sum() < 4:
        warnings.warn(
            f"{_pfx}Medial axis produced fewer than 4 skeleton pixels "
            f"(skel.sum()={int(skel.sum())}, mask area={int(mask_raw.sum())} px). "
            f"The mask may be too thin or small after contour smoothing. "
            f"Returning None centerline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, False, mask_bool

    # Build an annotated skeleton graph; passing `source_image=dist` lets skan
    # weight path lengths by the distance-transform values if needed later.
    try:
        skel_obj = Skeleton(skel, source_image=dist)
    except Exception as exc:
        warnings.warn(
            f"{_pfx}skan.Skeleton() raised an exception and could not build the "
            f"skeleton graph (mask area={int(mask_raw.sum())} px, "
            f"skeleton pixels={int(skel.sum())}): {exc}. "
            f"Returning None centerline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, False, mask_bool

    if skel_obj.n_paths == 0:
        warnings.warn(
            f"{_pfx}skan.Skeleton() produced 0 paths (mask area={int(mask_raw.sum())} px, "
            f"skeleton pixels={int(skel.sum())}). "
            f"The skeleton may consist of isolated pixels with no connected arcs. "
            f"Returning None centerline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, False, mask_bool

    # ── Step 2: graph traversal for full head-to-tail path ─────────────────
    # Even after mask smoothing, a few junction nodes may remain.  Rather than
    # taking the single longest segment (which stops at the first junction),
    # we build a networkx graph and find the longest endpoint-to-endpoint path,
    # stitching segments back together in order.
    coords = _longest_endpoint_path_coords(
        skel_obj,
        dir_vector_len_px=dir_vector_len_px,
        min_bridge_arc_len_px=min_bridge_arc_len_px,
        img_name=img_name,
    )

    if coords is None or len(coords) < 4:
        n_coords = 0 if coords is None else len(coords)
        warnings.warn(
            f"{_pfx}Graph traversal returned {'None' if coords is None else f'only {n_coords} point(s)'} "
            f"(need ≥ 4 for spline fitting, mask area={int(mask_raw.sum())} px, "
            f"skeleton pixels={int(skel.sum())}). "
            f"All named topology cases (1–6) and both fallbacks failed to produce "
            f"a usable centerline. Returning None centerline.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None, False, mask_bool

    # Override n_points with arc-length-based spatial resolution if requested.
    if px_per_point is not None and px_per_point > 0:
        diffs = np.diff(coords.astype(float), axis=0)
        arc_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
        n_points = max(1, int(round(arc_len / px_per_point)))

    rows, cols = coords[:, 0], coords[:, 1]

    # Fit a parametric cubic B-spline to remove digitisation noise
    s = smooth_factor if smooth_factor is not None else max(float(smooth_factor_ratio) * len(coords), 0.0)
    try:
        tck, _ = splprep([rows, cols], s=s, k=3)
        u_new = np.linspace(0, 1, n_points)
        r_sm, c_sm = splev(u_new, tck)
        centerline = np.column_stack([r_sm, c_sm])
    except Exception:
        # Fallback: uniformly subsample the raw skeleton path
        idx = np.round(np.linspace(0, len(coords) - 1, n_points)).astype(int)
        centerline = coords[idx].astype(float)

    # Sample local body width (diameter = 2 × distance transform) along the spline
    r_int = np.clip(centerline[:, 0].astype(int), 0, dist.shape[0] - 1)
    c_int = np.clip(centerline[:, 1].astype(int), 0, dist.shape[1] - 1)
    width_profile = dist[r_int, c_int] * 2.0

    return centerline, width_profile, True, mask_bool


# ---------------------------------------------------------------------------
# Multi-fragment helpers
# ---------------------------------------------------------------------------

def _skeletonize_mask_fragments(
    mask: np.ndarray,
    n_points_per_fragment: int = 60,
    smooth_factor: float | None = None,
    smooth_factor_ratio: float = 1.0,
    contour_smooth_sigma: float = 2.0,
    min_fragment_pixels: int = 30,
    hole_fill_ratio_threshold: float = 0.05,
    hole_keep_circularity_weight: float = 0.5,
    ring_dilation_radius: int = 0,
    dir_vector_len_px: float = 20.0,
    min_bridge_arc_len_px: float = 20.0,
    px_per_point: float | None = None,
    debug_ring_mask_prefix: str | None = None,
    img_h: int | None = None,
    img_w: int | None = None,
    img_name: str = "",
    _precomputed_labeled: np.ndarray | None = None,
    _precomputed_n_comp: int | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """
    Split *mask* into connected components and run ``extract_centerline`` on
    each one that is large enough.  Returns a list of
    ``(centerline, width_profile)`` tuples — one per successfully skeletonized
    fragment.

    Ordering
    --------
    Fragment 0 (the "main body") is the component whose **bounding-box centre
    is closest to the image centre** (requires *img_h* and *img_w*).  This
    mirrors how the target worm is selected among multiple YOLO detections and
    is more robust than pure area ranking when a debris blob happens to be
    slightly larger than the actual worm fragment.  All remaining fragments
    follow in ascending distance order.  If *img_h* / *img_w* are omitted,
    the original largest-area-first order is used as a fallback (no
    image-centre reference available).
    """
    if _precomputed_labeled is not None and _precomputed_n_comp is not None:
        labeled, n_comp = _precomputed_labeled, _precomputed_n_comp
    else:
        binary = (mask > 0.5).astype(np.uint8)
        labeled, n_comp = sk_label(binary, connectivity=2, return_num=True)

    # Collect per-component stats and compute the sort key.
    use_center = img_h is not None and img_w is not None
    img_cy = img_h / 2.0 if use_center else 0.0
    img_cx = img_w / 2.0 if use_center else 0.0

    comp_info: list[tuple[float, int]] = []   # (sort_key, cid)
    for cid in range(1, n_comp + 1):
        rows, cols = np.where(labeled == cid)
        area = int(rows.size)
        if use_center:
            # Bounding-box centre distance to image centre (ascending = closer first)
            bb_cy = (int(rows.min()) + int(rows.max())) / 2.0
            bb_cx = (int(cols.min()) + int(cols.max())) / 2.0
            sort_key = float(np.hypot(bb_cx - img_cx, bb_cy - img_cy))
        else:
            # Fallback: negate area so ascending sort gives largest-area-first
            sort_key = float(-area)
        comp_info.append((sort_key, cid))

    comp_info.sort(key=lambda t: t[0])

    # effective_mask accumulates the hole-filled mask for each fragment so the
    # caller can test which resampled centerline points are truly inside the
    # (possibly hole-filled) foreground — not just the raw YOLO mask pixels.
    effective_mask = np.zeros(mask.shape[:2], dtype=bool)

    fragments: list[tuple[np.ndarray, np.ndarray]] = []
    frag_idx = 0
    for _sort_key, cid in comp_info:
        area = int((labeled == cid).sum())
        if area < min_fragment_pixels:
            continue
        comp_mask = (labeled == cid).astype(np.uint8)
        ring_debug_path = (
            f"{debug_ring_mask_prefix}_f{frag_idx:02d}_ring.jpg"
            if debug_ring_mask_prefix is not None else None
        )
        cl, wp, ok, comp_filled = extract_centerline(
            comp_mask,
            n_points=n_points_per_fragment,
            px_per_point=px_per_point,
            smooth_factor=smooth_factor,
            smooth_factor_ratio=smooth_factor_ratio,
            contour_smooth_sigma=contour_smooth_sigma,
            hole_fill_ratio_threshold=hole_fill_ratio_threshold,
            hole_keep_circularity_weight=hole_keep_circularity_weight,
            ring_dilation_radius=ring_dilation_radius,
            dir_vector_len_px=dir_vector_len_px,
            min_bridge_arc_len_px=min_bridge_arc_len_px,
            debug_ring_mask_path=ring_debug_path,
            img_name=img_name,
        )
        effective_mask |= comp_filled
        frag_idx += 1
        if ok:
            fragments.append((cl, wp))
    return fragments, effective_mask


def _gap_connection_cost(
    cl_a: np.ndarray,
    a_flip: bool,
    cl_b: np.ndarray,
    b_flip: bool,
    cos_angle_threshold: float,
    max_gap_px: float,
    tangent_path_len_px: float,
    px_per_point: float,
) -> float:
    """
    Return the cost of connecting the tail of fragment A to the head of
    fragment B.  Returns ``np.inf`` if the Euclidean gap distance exceeds
    *max_gap_px* or if the gap angle exceeds the angle threshold on either
    side — both criteria are treated symmetrically as hard ``inf`` costs so
    they participate in the same sub-chain search that drops bad fragments.

    *tangent_path_len_px* and *px_per_point* together determine how many
    skeleton steps are used to estimate the outgoing / incoming tangent
    direction: ``n_steps = round(tangent_path_len_px / px_per_point)``.
    All available points are used when the fragment is shorter than the
    requested arc length.  This gives a physically meaningful, scale-aware
    tangent estimate independent of the skeleton's point count.
    """
    # Number of skeleton steps used for the tangent chord.
    n_tangent_steps = max(1, round(tangent_path_len_px / px_per_point))
    tail_a = cl_a[0]  if a_flip else cl_a[-1]
    head_b = cl_b[-1] if b_flip else cl_b[0]
    gap_vec  = head_b - tail_a
    gap_dist = float(np.linalg.norm(gap_vec))

    # ── Distance hard cutoff ──────────────────────────────────────────────
    if gap_dist > max_gap_px:
        return np.inf

    gap_cost = float(np.sum(gap_vec ** 2))

    if gap_dist > 1e-6:
        gap_unit = gap_vec / gap_dist

        # Outgoing direction of A (tail side)
        # Outgoing tangent of A: chord from endpoint back n_tangent_steps along
        # the skeleton — far more robust than a single-pixel finite difference.
        n_a = min(n_tangent_steps, len(cl_a) - 1)
        dir_a = (cl_a[0] - cl_a[n_a]) if a_flip else (cl_a[-1] - cl_a[-1 - n_a])
        norm_a = float(np.linalg.norm(dir_a))
        if norm_a > 1e-6:
            cos_a = float(np.dot(dir_a / norm_a, gap_unit))
            if cos_a < cos_angle_threshold:
                return np.inf
            gap_cost += (1.0 - cos_a) * gap_dist ** 2

        # Incoming tangent of B (head side)
        n_b = min(n_tangent_steps, len(cl_b) - 1)
        dir_b = (cl_b[-1 - n_b] - cl_b[-1]) if b_flip else (cl_b[n_b] - cl_b[0])
        norm_b = float(np.linalg.norm(dir_b))
        if norm_b > 1e-6:
            cos_b = float(np.dot(dir_b / norm_b, gap_unit))
            if cos_b < cos_angle_threshold:
                return np.inf
            gap_cost += (1.0 - cos_b) * gap_dist ** 2

    return gap_cost


def _gap_failure_reason(
    cl_a: np.ndarray,
    cl_b: np.ndarray,
    max_gap_px: float,
    cos_angle_threshold: float,
    tangent_path_len_px: float,
    px_per_point: float,
) -> set[str]:
    """
    Check all 4 endpoint-flip combinations between two fragments and return
    the set of constraint(s) violated by the *best-case* pairing (the one
    with the fewest violations).  Possible elements: ``"distance"``, ``"angle"``.
    Used only for diagnostic labelling when stitching fails.
    """
    n_tangent_steps = max(1, round(tangent_path_len_px / px_per_point))
    best_reasons: set[str] | None = None
    for a_flip in (False, True):
        for b_flip in (False, True):
            tail_a   = cl_a[0]  if a_flip else cl_a[-1]
            head_b   = cl_b[-1] if b_flip else cl_b[0]
            gap_vec  = head_b - tail_a
            gap_dist = float(np.linalg.norm(gap_vec))

            reasons: set[str] = set()
            if gap_dist > max_gap_px:
                reasons.add("distance")

            if gap_dist > 1e-6:
                gap_unit = gap_vec / gap_dist

                n_a   = min(n_tangent_steps, len(cl_a) - 1)
                dir_a = (cl_a[0] - cl_a[n_a]) if a_flip else (cl_a[-1] - cl_a[-1 - n_a])
                norm_a = float(np.linalg.norm(dir_a))
                if norm_a > 1e-6 and float(np.dot(dir_a / norm_a, gap_unit)) < cos_angle_threshold:
                    reasons.add("angle")

                n_b   = min(n_tangent_steps, len(cl_b) - 1)
                dir_b = (cl_b[-1 - n_b] - cl_b[-1]) if b_flip else (cl_b[n_b] - cl_b[0])
                norm_b = float(np.linalg.norm(dir_b))
                if norm_b > 1e-6 and float(np.dot(dir_b / norm_b, gap_unit)) < cos_angle_threshold:
                    reasons.add("angle")

            # Keep the combination with the fewest violations
            if best_reasons is None or len(reasons) < len(best_reasons):
                best_reasons = reasons
            if not best_reasons:   # already zero violations — stop early
                break

    return best_reasons or set()


def stitch_centerline_fragments(
    fragments: list[tuple[np.ndarray, np.ndarray]],
    n_points: int = 100,
    gap_interp_points: int = 12,
    max_gap_px: float = 200.0,
    max_joint_angle_deg: float = 70.0,
    tangent_path_len_px: float = 200.0,
    px_per_point: float = 10.0,
) -> tuple[np.ndarray | None, np.ndarray | None, list[tuple[int, int]], bool, str, list[tuple[int, bool]]]:
    """
    Order and stitch a list of (centerline, width_profile) fragments into a
    single continuous centerline.

    Strategy
    --------
    1.  Enumerate all orderings **and** all per-fragment flip orientations
        (head ↔ tail) — feasible because in practice the worm is cut into at
        most 2–3 pieces.
    2.  Score each candidate chain by the sum of per-connection costs.  Each
        connection cost is ``inf`` when the gap distance exceeds *max_gap_px*
        **or** the gap angle deviates more than *max_joint_angle_deg* from the
        outgoing direction of A or the incoming direction of B; otherwise it
        combines the squared Euclidean endpoint distance with a symmetric
        direction-consistency penalty.
    3.  Both thresholds are unified inside ``_gap_connection_cost``, so the
        sub-chain fallback (step 4) naturally drops blobs that are too far
        away *or* mis-aligned — whichever is the cause.
    4.  If no full-chain arrangement is valid, progressively smaller subsets
        are tried (k−1, k−2, …) to preserve the largest valid partial chain.
        **Fragment 0 (the closest-to-image-centre fragment) is always kept** —
        only secondary fragments are dropped, so the main worm body is never
        abandoned in favour of a debris blob.
    5.  If even the size-2 sub-chains that include fragment 0 all have
        infinite cost, return ``(None, None, [], True)`` and the caller falls
        back to fragment 0 directly (closest to image centre).
    6.  Otherwise return the lowest-cost stitched chain with linear
        interpolation across each gap.

    Parameters
    ----------
    fragments :
        List of ``(centerline, width_profile)`` tuples from
        ``_skeletonize_mask_fragments``.
    n_points :
        Number of uniformly-resampled points in the final output.
    gap_interp_points :
        Number of interpolated points inserted **across each occluded gap**
        (not counting the fragment endpoints themselves).
    max_gap_px :
        Maximum allowed Euclidean distance (pixels) for a single
        fragment-to-fragment connection.  Connections that exceed this value
        are assigned ``inf`` cost so they are excluded from the chain search
        (and sub-chain fallback) exactly like angle violations.
    max_joint_angle_deg :
        Maximum allowed angle (degrees) between the gap vector and the
        outgoing direction of fragment A **or** the incoming direction of
        fragment B.  Connections that deviate beyond this threshold (e.g. a
        perpendicular stub sitting at the tip of the main body) are given an
        infinite cost and are never selected.  Default 70°.  Set to 180° to
        disable the check.
    tangent_path_len_px :
        Arc length (pixels) over which the tangent direction is estimated at
        each fragment endpoint.  The chord spans
        ``round(tangent_path_len_px / px_per_point)`` skeleton steps back from
        the endpoint (clamped to the fragment length), giving a
        physically-meaningful, scale-aware tangent.  Default 200 px.
    px_per_point :
        Spatial resolution of the fragment skeletons in pixels per point.
        Should match ``skeleton_px_per_point`` used during fragment extraction.
        Default 10.0.

    Returns
    -------
    centerline :
        ``(n_points, 2)`` float array, or ``None``.
    width_profile :
        ``(n_points,)`` float array (diameter in px), or ``None``.
    gap_index_ranges :
        List of ``(start, end)`` **index pairs** (into the *resampled*
        centerline) that mark interpolated gap regions — used by the
        visualiser to draw dashed lines over occluded sections.
    gap_exceeded :
        ``True`` when all arrangements (full chain and every sub-chain subset)
        were invalid — either due to gap distance, angle, or both.  The caller
        then falls back to the single fragment closest to the FOV centre.
        ``False`` in all normal cases.
    stitch_failure_reason :
        Human-readable string explaining *why* stitching failed, e.g.
        ``"distance"``, ``"angle"``, or ``"distance + angle"``.
        Empty string ``""`` when stitching succeeded.
    fragment_chain :
        Ordered list of ``(fragment_index, flipped)`` pairs describing which
        fragments were included and whether each was reversed before joining.
        Empty list when stitching failed or only one fragment existed.
    """
    if not fragments:
        return None, None, [], False, "", []

    if len(fragments) == 1:
        cl, wp = fragments[0]
        idx = np.round(np.linspace(0, len(cl) - 1, n_points)).astype(int)
        return cl[idx], wp[idx], [], False, "", [(0, False)]

    k = len(fragments)

    # Precompute the cosine threshold once.
    cos_angle_threshold = float(np.cos(np.deg2rad(max_joint_angle_deg)))

    def _best_chain_for_indices(indices: tuple[int, ...]) -> tuple[float, tuple | None, tuple | None]:  # noqa: E501
        """
        Over all permutations and flip combinations of the given fragment
        *indices*, return ``(best_cost, best_order, best_flips)``.
        NOTE: we do NOT fix flip[0]=False.  The "halve the search" trick
        is only valid when the cost is symmetric under full chain reversal
        (reversed order + all flips toggled).  Our direction-consistency
        penalty is NOT symmetric because gap distances differ between a
        forward and a reversed chain, so the true minimum can — and does —
        live in a state where the first fragment is flipped.  With k ≤ 5
        the full 2^k enumeration (≤ 32 combinations per ordering) is cheap.
        """
        m = len(indices)
        bc, bo, bf = np.inf, None, None
        for order in itertools.permutations(indices):
            for flips in itertools.product([False, True], repeat=m):
                cost = 0.0
                for step in range(m - 1):
                    cost += _gap_connection_cost(
                        fragments[order[step]][0],   flips[step],
                        fragments[order[step+1]][0], flips[step+1],
                        cos_angle_threshold,
                        max_gap_px,
                        tangent_path_len_px,
                        px_per_point,
                    )
                    if not np.isfinite(cost):
                        break   # short-circuit
                if cost < bc:
                    bc, bo, bf = cost, order, flips
        return bc, bo, bf

    # ── Step 1: try the full chain ─────────────────────────────────────────
    best_cost, best_order, best_flips = _best_chain_for_indices(tuple(range(k)))

    # ── Step 2: if no full-chain arrangement is valid, try sub-chains ──────
    # Drop the fewest secondary fragments possible so that the remaining set
    # can be validly stitched.  Fragment 0 (closest to image centre, i.e. the
    # main worm body) is ALWAYS kept — only indices 1..k-1 are ever dropped.
    if not np.isfinite(best_cost):
        for subset_size in range(k - 1, 1, -1):
            sub_best_cost  = np.inf
            sub_best_order = None
            sub_best_flips = None
            # Only consider subsets that contain fragment 0.
            secondary = list(range(1, k))
            for extra in itertools.combinations(secondary, subset_size - 1):
                subset = (0,) + extra
                sc, so, sf = _best_chain_for_indices(subset)
                if sc < sub_best_cost:
                    sub_best_cost, sub_best_order, sub_best_flips = sc, so, sf
            if sub_best_order is not None and np.isfinite(sub_best_cost):
                best_cost  = sub_best_cost
                best_order = sub_best_order
                best_flips = sub_best_flips
                break   # found the largest valid sub-chain; stop

    # ── All arrangements (including sub-chains) were invalid ───────────────
    if best_order is None or not np.isfinite(best_cost):
        # Diagnose why every connection involving fragment 0 failed.
        # Check fragment 0 against all other fragments and union the reasons.
        all_reasons: set[str] = set()
        for other in range(1, k):
            all_reasons |= _gap_failure_reason(
                fragments[0][0], fragments[other][0],
                max_gap_px, cos_angle_threshold, tangent_path_len_px, px_per_point,
            )
        failure_reason = " + ".join(sorted(all_reasons)) if all_reasons else "unknown"
        return None, None, [], True, failure_reason, []

    chain_len = len(best_order)

    # ── Stitch fragments in best order ─────────────────────────────────────
    raw_coords: list[np.ndarray] = []
    raw_widths: list[float]      = []
    # Track the *raw* index ranges of gap interpolation
    raw_gap_ranges: list[tuple[int, int]] = []

    for step, (frag_idx, flip) in enumerate(zip(best_order, best_flips)):
        cl, wp = fragments[frag_idx]
        if flip:
            cl = cl[::-1].copy()
            wp = wp[::-1].copy()

        if step > 0:
            # Linear interpolation across the occluded gap
            gap_start = len(raw_coords)
            prev_coord = raw_coords[-1]
            prev_width = raw_widths[-1]
            next_coord = cl[0]
            next_width = float(wp[0])
            for t in np.linspace(0.0, 1.0, gap_interp_points + 2)[1:-1]:
                raw_coords.append(prev_coord + t * (next_coord - prev_coord))
                raw_widths.append(prev_width + t * (next_width - prev_width))
            raw_gap_ranges.append((gap_start, len(raw_coords)))

        raw_coords.extend(list(cl))
        raw_widths.extend(list(wp.astype(float)))

    raw_coords_arr = np.array(raw_coords)   # (M, 2)
    raw_widths_arr = np.array(raw_widths)   # (M,)

    # Uniform re-sampling to n_points
    idx = np.round(np.linspace(0, len(raw_coords_arr) - 1, n_points)).astype(int)
    centerline    = raw_coords_arr[idx]
    width_profile = raw_widths_arr[idx]

    # Map raw gap ranges → indices in the resampled array
    # For each raw gap range (rs, re) find which resampled indices fall inside
    gap_index_ranges: list[tuple[int, int]] = []
    for rs, re in raw_gap_ranges:
        # idx is sorted, use searchsorted
        s = int(np.searchsorted(idx, rs, side="left"))
        e = int(np.searchsorted(idx, re, side="right")) - 1
        if s <= e and s < n_points:
            gap_index_ranges.append((max(s, 0), min(e, n_points - 1)))

    fragment_chain = list(zip(best_order, best_flips))
    return centerline, width_profile, gap_index_ranges, False, "", fragment_chain


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _draw_dashed_polyline(
    img: np.ndarray,
    pts: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash_px: int = 8,
    gap_px: int = 6,
) -> None:
    """
    Draw a dashed polyline on *img* (in-place).

    Parameters
    ----------
    pts :
        ``(N, 2)`` integer array of ``(x, y)`` OpenCV points (col, row).
    """
    if len(pts) < 2:
        return
    budget = 0.0   # pixels remaining in the current dash/gap segment
    drawing = True
    for i in range(len(pts) - 1):
        p0 = pts[i].astype(float)
        p1 = pts[i + 1].astype(float)
        seg_len = float(np.linalg.norm(p1 - p0))
        if seg_len < 1e-3:
            continue
        direction = (p1 - p0) / seg_len
        d = 0.0
        while d < seg_len:
            remaining_in_phase = (dash_px if drawing else gap_px) - budget
            step = min(remaining_in_phase, seg_len - d)
            if drawing:
                q0 = (p0 + direction * d).astype(np.int32)
                q1 = (p0 + direction * (d + step)).astype(np.int32)
                cv2.line(img, tuple(q0), tuple(q1), color, thickness, cv2.LINE_AA)
            d      += step
            budget += step
            if budget >= (dash_px if drawing else gap_px):
                budget  = 0.0
                drawing = not drawing


def draw_centerline_on_image(
    img: np.ndarray,
    mask: np.ndarray,
    centerline: np.ndarray,
    mask_color: tuple[int, int, int] = (0, 200, 255),
    line_color: tuple[int, int, int] = (0, 255, 0),
    gap_color: tuple[int, int, int] = (0, 165, 255),
    head_color: tuple[int, int, int] = (255, 100, 0),   # blue  – circle
    tail_color: tuple[int, int, int] = (255, 100, 0),   # blue  – circle (same as head)
    mask_alpha: float = 0.35,
    line_thickness: int = 2,
    gap_index_ranges: list[tuple[int, int]] | None = None,
    in_mask_points: np.ndarray | None = None,
) -> np.ndarray:
    """
    Overlay a semi-transparent mask fill and the fitted centerline on *img*.

    Parameters
    ----------
    img :
        BGR image ``(H × W × 3)``.
    mask :
        Binary mask ``(H × W)``, same spatial size as *img*.
    centerline :
        ``(N, 2)`` array of **(row, col)** positions.
    head_color :
        BGR colour for the **head** endpoint marker (filled circle).
    tail_color :
        BGR colour for the **tail** endpoint marker (filled circle).
    gap_index_ranges :
        Optional list of ``(start, end)`` index pairs (into *centerline*)
        that mark occluded / interpolated gap regions.  These spans are drawn
        as a dashed orange line instead of the solid green centerline.
    in_mask_points :
        Optional boolean array of length N.  When provided, a small filled
        dot is drawn at every centerline point: white (radius 2) for points
        inside the mask (included in width calculation) and red (radius 3)
        for points outside the mask (excluded from width calculation).

    Returns
    -------
    Annotated BGR image (copy of *img*).
    """
    annotated = img.copy()
    overlay   = img.copy()

    # Semi-transparent mask fill
    mask_bin = (mask > 0.5).astype(np.uint8)
    overlay[mask_bin == 1] = mask_color
    annotated = cv2.addWeighted(annotated, 1 - mask_alpha, overlay, mask_alpha, 0)

    # Build a boolean mask: True for points that belong to a gap segment
    n = len(centerline)

    # Centerline is (row, col); OpenCV wants (x=col, y=row)
    xy = centerline[:, ::-1].astype(np.int32)   # (N, 2) col-row

    # Disabled: solid green centerline and orange dashed gap polyline
    # (per-point white/red dots and head/tail circles are kept).
    # is_gap = np.zeros(n, dtype=bool)
    # if gap_index_ranges:
    #     for gs, ge in gap_index_ranges:
    #         is_gap[max(0, gs - 1) : min(n, ge + 2)] = True
    #
    # # Draw solid segments (non-gap) and dashed segments (gap) separately
    # i = 0
    # while i < n:
    #     # Find the next run of the same type
    #     is_gap_i = bool(is_gap[i])
    #     j = i + 1
    #     while j < n and bool(is_gap[j]) == is_gap_i:
    #         j += 1
    #     seg = xy[i:j]   # slice of same type
    #     if len(seg) >= 2:
    #         if is_gap_i:
    #             _draw_dashed_polyline(
    #                 annotated, seg, gap_color,
    #                 thickness=line_thickness, dash_px=8, gap_px=5,
    #             )
    #         else:
    #             pts = seg.reshape(-1, 1, 2)
    #             cv2.polylines(
    #                 annotated, [pts],
    #                 isClosed=False, color=line_color,
    #                 thickness=line_thickness, lineType=cv2.LINE_AA,
    #             )
    #     i = j

    # ── Per-point markers (in-mask = white, out-of-mask = magenta) ───────
    # Magenta (BGR 220,0,220) is colour-blind safe (distinguishable under
    # deutan/protan/tritan) and contrasts strongly with the yellow mask
    # overlay, the white in-mask dots, and the blue head/tail circles.
    if in_mask_points is not None:
        for k in range(n):
            pt = (int(xy[k, 0]), int(xy[k, 1]))   # (x, y) for OpenCV
            if bool(in_mask_points[k]):
                cv2.circle(annotated, pt, 2, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            else:
                cv2.circle(annotated, pt, 3, (255, 200, 0), -1, lineType=cv2.LINE_AA)

    # ── Head endpoint: filled circle ──────────────────────────────────────
    r0, c0 = int(centerline[0, 0]), int(centerline[0, 1])
    cv2.circle(annotated, (c0, r0), 12, head_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(annotated, (c0, r0), 12, (255, 255, 255), 1, lineType=cv2.LINE_AA)  # white ring

    # ── Tail endpoint: filled circle ──────────────────────────────────────
    r1, c1 = int(centerline[-1, 0]), int(centerline[-1, 1])
    cv2.circle(annotated, (c1, r1), 12, tail_color, -1, lineType=cv2.LINE_AA)
    cv2.circle(annotated, (c1, r1), 12, (255, 255, 255), 1, lineType=cv2.LINE_AA)  # white ring

    return annotated


def find_center_worm_idx(boxes_xyxy: np.ndarray, img_h: int, img_w: int) -> int:
    """
    Return the index of the detection whose bounding-box centre is closest
    to the image centre.

    Parameters
    ----------
    boxes_xyxy :
        ``(N, 4)`` array of ``[x1, y1, x2, y2]`` bounding boxes.
    img_h, img_w :
        Image height and width in pixels.

    Returns
    -------
    int index into the N detections.
    """
    img_cx, img_cy = img_w / 2.0, img_h / 2.0
    box_cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    box_cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    dists = np.hypot(box_cx - img_cx, box_cy - img_cy)
    return int(np.argmin(dists))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _save_fragment_debug_image(
    img: np.ndarray,
    mask_bin: np.ndarray,
    fragments: list[tuple[np.ndarray, np.ndarray]],
    out_path: str,
) -> None:
    """
    Save a debug image showing each skeletonized fragment with a coloured
    polyline and a directional arrow (filled triangle) at its tail end so
    you can see which way each fragment's skeleton is oriented before stitching.

    Arrow convention
    ----------------
    • Filled circle  = fragment ``[0]``  ("start" as returned by extract_centerline)
    • Filled triangle = fragment ``[-1]`` ("end" / tail)
    """
    debug_img = img.copy()
    overlay   = img.copy()
    mask_3ch  = (mask_bin > 0).astype(np.uint8)
    overlay[mask_3ch == 1] = (200, 200, 200)
    debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)

    palette = [
        (0,   255,   0),
        (255,   0, 255),
        (0,   200, 255),
        (255, 165,   0),
        (128,   0, 255),
    ]
    for fi, (cl, _) in enumerate(fragments):
        colour = palette[fi % len(palette)]
        xy = cl[:, ::-1].astype(np.int32)   # (row,col) → (x,y)
        cv2.polylines(
            debug_img, [xy.reshape(-1, 1, 2)],
            isClosed=False, color=colour, thickness=2, lineType=cv2.LINE_AA,
        )
        # Start marker: filled circle
        cv2.circle(debug_img, tuple(xy[0]), 7, colour, -1, lineType=cv2.LINE_AA)
        cv2.circle(debug_img, tuple(xy[0]), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        # End marker: filled triangle pointing in the exit direction
        if len(xy) >= 2:
            tip   = xy[-1].astype(float)
            shaft = xy[-2].astype(float)
            dvec  = tip - shaft
            norm  = np.linalg.norm(dvec)
            if norm > 1e-6:
                dvec /= norm
                perp  = np.array([-dvec[1], dvec[0]])
                half  = 6.0
                tri   = np.array([
                    (tip + dvec * 10).astype(np.int32),
                    (tip - dvec * 3 + perp * half).astype(np.int32),
                    (tip - dvec * 3 - perp * half).astype(np.int32),
                ])
                cv2.fillPoly(debug_img, [tri], colour, lineType=cv2.LINE_AA)
        label_pos = tuple((xy[0] + np.array([4, -8])).astype(int))
        cv2.putText(
            debug_img, f"F{fi}", label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            debug_img, f"F{fi}", label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA,
        )

    cv2.imwrite(out_path, debug_img)


def _save_stitched_debug_image(
    img: np.ndarray,
    mask_bin: np.ndarray,
    fragments: list[tuple[np.ndarray, np.ndarray]],
    centerline: np.ndarray,
    gap_index_ranges: list[tuple[int, int]],
    stitch_failure_reason: str,
    fragment_chain: list[tuple[int, bool]],
    out_path: str,
) -> None:
    """
    Save a debug image showing the per-fragment skeletons (faint, same colours
    as the pre-stitch image) **plus** the final stitched centerline on top.

    This lets you compare the before/after pair side-by-side:
     • ``<stem>_fragments.jpg``  – raw fragment orientations before stitching
     • ``<stem>_stitched.jpg``   – final result with stitched centerline overlay

    Per-fragment arrows (same circle + filled-triangle format as the
    pre-stitch image) are drawn in their **stitched orientation** (flips
    applied) using full-brightness palette colours, on top of the dimmed
    original-orientation skeletons shown for comparison.

    The stitched centerline is drawn in white (solid) with an orange dashed
    line over any interpolated gap regions.  Head = filled blue circle;
    tail = filled red square.
    If *stitch_failure_reason* is non-empty a red label is added showing why
    stitching failed (``"distance"``, ``"angle"``, or ``"distance + angle"``).
    """
    # Start from the same grey-mask background as the pre-stitch debug image
    debug_img = img.copy()
    overlay   = img.copy()
    mask_3ch  = (mask_bin > 0).astype(np.uint8)
    overlay[mask_3ch == 1] = (200, 200, 200)
    debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)

    palette = [
        (0,   255,   0),
        (255,   0, 255),
        (0,   200, 255),
        (255, 165,   0),
        (128,   0, 255),
    ]

    # Draw ALL fragments faintly in original orientation for reference
    for fi, (cl, _) in enumerate(fragments):
        base   = palette[fi % len(palette)]
        colour = tuple(int(c * 0.45) for c in base)   # dimmed
        xy = cl[:, ::-1].astype(np.int32)
        cv2.polylines(
            debug_img, [xy.reshape(-1, 1, 2)],
            isClosed=False, color=colour, thickness=1, lineType=cv2.LINE_AA,
        )

    # Draw each chain fragment in its STITCHED orientation (flip applied)
    # using the same circle/triangle markers as _save_fragment_debug_image.
    chain_fragment_indices = {fi for fi, _ in fragment_chain}
    for chain_step, (fi, flip) in enumerate(fragment_chain):
        cl = fragments[fi][0]
        if flip:
            cl = cl[::-1]
        colour = palette[fi % len(palette)]   # full brightness
        xy = cl[:, ::-1].astype(np.int32)     # (row,col) → (x,y)
        cv2.polylines(
            debug_img, [xy.reshape(-1, 1, 2)],
            isClosed=False, color=colour, thickness=2, lineType=cv2.LINE_AA,
        )
        # Start marker: filled circle
        cv2.circle(debug_img, tuple(xy[0]), 7, colour, -1, lineType=cv2.LINE_AA)
        cv2.circle(debug_img, tuple(xy[0]), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        # End marker: filled triangle pointing in the exit direction
        if len(xy) >= 2:
            tip   = xy[-1].astype(float)
            shaft = xy[-2].astype(float)
            dvec  = tip - shaft
            norm  = np.linalg.norm(dvec)
            if norm > 1e-6:
                dvec /= norm
                perp = np.array([-dvec[1], dvec[0]])
                half = 6.0
                tri  = np.array([
                    (tip + dvec * 10).astype(np.int32),
                    (tip - dvec * 3 + perp * half).astype(np.int32),
                    (tip - dvec * 3 - perp * half).astype(np.int32),
                ])
                cv2.fillPoly(debug_img, [tri], colour, lineType=cv2.LINE_AA)
        # Label: chain step index (S0, S1, …) and whether flipped
        flip_tag  = "↕" if flip else ""
        label_txt = f"F{fi}{flip_tag}"
        label_pos = tuple((xy[0] + np.array([4, -8])).astype(int))
        cv2.putText(
            debug_img, label_txt, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            debug_img, label_txt, label_pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA,
        )

    # Draw the stitched centerline
    n       = len(centerline)
    is_gap  = np.zeros(n, dtype=bool)
    for gs, ge in (gap_index_ranges or []):
        is_gap[max(0, gs - 1) : min(n, ge + 2)] = True

    xy_cl = centerline[:, ::-1].astype(np.int32)   # (row,col) → (x,y)
    i = 0
    while i < n:
        is_gap_i = bool(is_gap[i])
        j = i + 1
        while j < n and bool(is_gap[j]) == is_gap_i:
            j += 1
        seg = xy_cl[i:j]
        if len(seg) >= 2:
            if is_gap_i:
                _draw_dashed_polyline(
                    debug_img, seg, (0, 165, 255),
                    thickness=2, dash_px=8, gap_px=5,
                )
            else:
                cv2.polylines(
                    debug_img, [seg.reshape(-1, 1, 2)],
                    isClosed=False, color=(255, 255, 255),
                    thickness=2, lineType=cv2.LINE_AA,
                )
        i = j

    # Head: filled blue circle
    r0, c0 = int(centerline[0, 0]), int(centerline[0, 1])
    cv2.circle(debug_img, (c0, r0), 12, (255, 100, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(debug_img, (c0, r0), 12, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Tail: filled blue circle (same as head)
    r1, c1 = int(centerline[-1, 0]), int(centerline[-1, 1])
    cv2.circle(debug_img, (c1, r1), 12, (255, 100, 0), -1, lineType=cv2.LINE_AA)
    cv2.circle(debug_img, (c1, r1), 12, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Fallback label
    if stitch_failure_reason:
        cv2.putText(
            debug_img, f"No valid stitch [{stitch_failure_reason}] — closest-to-centre fragment used",
            (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )

    cv2.imwrite(out_path, debug_img)


def _parse_filename_metadata(stem: str) -> tuple[str, str, str, str, str, str]:
    """
    Extract structured metadata from a ROI image filename stem.

    Expected format::

        {date}-{experiment}-{plate_id}-{magnification}_{well_id}_roi_{worm_id}

    Example::

        "20260226-cryassays-p001-m2X_A01_roi_0"
        -> date="20260226", experiment="cryassays", plate_id="p001",
           magnification="m2X", well_id="A01", worm_id="0"

    Returns ``(date, experiment, plate_id, magnification, well_id, worm_id)``.
    Any field that cannot be parsed is returned as ``"NA"``.
    Fields are only populated when the filename matches the expected structure;
    a filename that does not contain ``_roi_`` or lacks at least two
    hyphen-delimited prefix tokens will return all-``"NA"`` fields rather than
    silently mis-assigning tokens to the wrong columns.
    """
    _NA = "NA"
    _empty = (_NA, _NA, _NA, _NA, _NA, _NA)
    try:
        if not isinstance(stem, str) or not stem:
            return _empty
        # Strip file extension if accidentally passed with one
        base = Path(stem).stem if "." in stem else stem
        # Require the _roi_ separator to be present; without it we cannot
        # reliably identify well_id or worm_id.
        if "_roi_" not in base:
            return _empty
        head, worm_id = base.rsplit("_roi_", 1)
        # The last "_"-delimited token of head is the well id
        parts = head.rsplit("_", 1)
        if len(parts) != 2:
            return _empty
        prefix, well_id = parts
        # prefix must have at least date and experiment (2 hyphen-delimited tokens)
        dash_parts = prefix.split("-")
        if len(dash_parts) < 2:
            return _empty
        date          = dash_parts[0]
        experiment    = dash_parts[1]
        plate_id      = dash_parts[2] if len(dash_parts) >= 3 else _NA
        magnification = dash_parts[3] if len(dash_parts) >= 4 else _NA
    except Exception:
        return _empty
    return date, experiment, plate_id, magnification, well_id, worm_id


# Shared worm-size CSV schema used by both YOLO and GT modes.
WORM_SIZES_CSV_COLUMNS = [
    "Filename",
    "Date",
    "Metadata_Experiment",
    "Metadata_Plate",
    "Magnification",
    "Metadata_Well",
    "Worm_ID",
    "Length_um",
    "Width_um",
    "Topology_Warnings",
]


# Mapping from substrings unique to each Group-A topology warning to a short
# flag code that will appear in the ``Topology_Warnings`` CSV column.  Order
# is preserved so output flags appear in a stable sequence.
_TOPOLOGY_WARNING_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("Self-loop node detected",                 "ring_fallthrough"),
    ("Fallback A:",                             "fallback_a"),
    ("Fallback B:",                             "fallback_b"),
    ("Skeleton graph traversal failed",         "traversal_failed"),
)


def _collect_topology_warnings(caught_warnings) -> str:
    """Return a ``;``-joined string of Group-A topology-warning flag codes.

    Scans the list produced by ``warnings.catch_warnings(record=True)`` for
    messages matching any of the known centerline-topology warnings emitted
    by ``_longest_endpoint_path_coords``.  Each unique flag appears at most
    once; an empty string is returned when no relevant warning fired.
    """
    seen: set[str] = set()
    flags: list[str] = []
    for w in caught_warnings:
        msg = str(w.message)
        for needle, code in _TOPOLOGY_WARNING_SIGNATURES:
            if needle in msg and code not in seen:
                seen.add(code)
                flags.append(code)
    return ";".join(flags)


def _init_worm_sizes_csv(output_dir: str) -> tuple[Path, object, csv.writer]:
    """Create worm-size CSV file and write the shared header."""
    csv_path = Path(output_dir) / "worm_sizes.csv"
    csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_fh)
    csv_writer.writerow(WORM_SIZES_CSV_COLUMNS)
    return csv_path, csv_fh, csv_writer


def _build_worm_sizes_row(
    filename: str,
    stem: str,
    length_um: float,
    width_um: float,
    topology_warnings: str = "",
) -> tuple[str, str, str, str, str, str, str, float, float, str]:
    """Build one worm-size CSV row with parsed filename metadata.

    ``topology_warnings`` is a ``;``-joined string of flag codes
    (see ``_collect_topology_warnings``) indicating that the skeleton
    extraction for this image hit one or more known centerline-topology
    fallbacks.  Empty string ⇒ clean extraction.
    """
    date, experiment, plate_id, magnification, well_id, worm_id = _parse_filename_metadata(stem)
    return (
        filename,
        date,
        experiment,
        plate_id,
        magnification,
        well_id,
        worm_id,
        length_um,
        width_um,
        topology_warnings,
    )


def _save_contour_skeleton_txt(
    mask_bin: np.ndarray,
    centerline: np.ndarray | None,
    catalog_entry: dict,
    roi_stem: str,
    out_path: Path | str,
    contour_max_points: int = 200,
) -> None:
    """Write mask contours and skeleton to a YOLO-style normalised .txt file.

    All coordinates are written as ``x y`` fractions relative to the
    **original full image** dimensions, matching the YOLO annotation
    convention (x = col / orig_width, y = row / orig_height).

    File layout::

        [CONTOUR]
        x1 y1
        x2 y2
        ...
        [CONTOUR]          ← one section per disconnected mask blob
        ...
        [SKELETON]
        x1 y1
        ...

    Requires a valid ``catalog_entry`` with ``original_width``,
    ``original_height``, ``roi_top_left`` ([x, y]) and
    ``roi_size_original``.  Emits a warning and returns early if the
    catalog entry is missing or incomplete.
    """
    # ── Validate catalog entry ─────────────────────────────────────────────
    if not isinstance(catalog_entry, dict) or not catalog_entry:
        warnings.warn(
            f"[{roi_stem}] Contour/skeleton txt export skipped: no ROI catalog entry.",
            RuntimeWarning, stacklevel=2,
        )
        return

    required = ("original_width", "original_height", "roi_top_left", "roi_size_original")
    missing = [k for k in required if k not in catalog_entry]
    if missing:
        warnings.warn(
            f"[{roi_stem}] Contour/skeleton txt export skipped: missing catalog "
            f"field(s): {', '.join(missing)}.",
            RuntimeWarning, stacklevel=2,
        )
        return

    try:
        orig_w = float(catalog_entry["original_width"])
        orig_h = float(catalog_entry["original_height"])
        roi_xy = catalog_entry["roi_top_left"]
        roi_size_orig = float(catalog_entry["roi_size_original"])
    except (TypeError, ValueError):
        warnings.warn(
            f"[{roi_stem}] Contour/skeleton txt export skipped: invalid catalog values.",
            RuntimeWarning, stacklevel=2,
        )
        return

    if orig_w <= 0 or orig_h <= 0 or roi_size_orig <= 0:
        return
    if not isinstance(roi_xy, (list, tuple)) or len(roi_xy) != 2:
        return

    try:
        roi_x = float(roi_xy[0])
        roi_y = float(roi_xy[1])
    except (TypeError, ValueError):
        return

    roi_h, roi_w = mask_bin.shape[:2]
    if roi_h <= 0 or roi_w <= 0:
        return

    # Pixel-to-normalised helpers (ROI pixel → original-image fraction).
    sx = roi_size_orig / roi_w   # col scale
    sy = roi_size_orig / roi_h   # row scale

    def _to_norm(col: float, row: float) -> tuple[float, float]:
        return ((roi_x + col * sx) / orig_w,
                (roi_y + row * sy) / orig_h)

    def _resample_contour(pts: np.ndarray, max_pts: int) -> np.ndarray:
        """Uniformly resample an (N, 2) contour down to *max_pts* points."""
        if len(pts) <= max_pts:
            return pts
        indices = np.round(np.linspace(0, len(pts) - 1, max_pts)).astype(int)
        return pts[indices]

    # ── Extract contours from mask ─────────────────────────────────────────
    contours_cv, _ = cv2.findContours(
        mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    lines: list[str] = []

    for cnt in contours_cv:
        pts = cnt.reshape(-1, 2)  # (N, 2) with cols (x=col, y=row)
        pts = _resample_contour(pts, contour_max_points)
        lines.append("[CONTOUR]")
        for col, row in pts:
            xn, yn = _to_norm(float(col), float(row))
            lines.append(f"{xn:.6f} {yn:.6f}")

    # ── Skeleton ───────────────────────────────────────────────────────────
    if centerline is not None and len(centerline) >= 2:
        lines.append("[SKELETON]")
        for row, col in centerline:   # centerline is (N, 2) as (row, col)
            xn, yn = _to_norm(float(col), float(row))
            lines.append(f"{xn:.6f} {yn:.6f}")

    if not lines:
        return

    with open(out_path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")


def _roi_mask_to_original_canvas(
    roi_mask_bin: np.ndarray,
    catalog_entry: dict,
) -> np.ndarray | None:
    """
    Project a ROI-space binary mask back onto a full-size original-image canvas.

    Required ``catalog_entry`` fields:
        - ``original_width``
        - ``original_height``
        - ``roi_top_left`` as [x, y]
        - ``roi_size_original``
    """
    if roi_mask_bin is None:
        return None

    try:
        orig_w = int(round(float(catalog_entry["original_width"])))
        orig_h = int(round(float(catalog_entry["original_height"])))
        roi_xy = catalog_entry["roi_top_left"]
        roi_size_orig = int(round(float(catalog_entry["roi_size_original"])))
    except (KeyError, TypeError, ValueError):
        return None

    if orig_w <= 0 or orig_h <= 0 or roi_size_orig <= 0:
        return None
    if not isinstance(roi_xy, (list, tuple)) or len(roi_xy) != 2:
        return None

    try:
        roi_x = int(round(float(roi_xy[0])))
        roi_y = int(round(float(roi_xy[1])))
    except (TypeError, ValueError):
        return None

    roi_u8 = (roi_mask_bin > 0).astype(np.uint8)
    roi_resized = cv2.resize(
        roi_u8,
        (roi_size_orig, roi_size_orig),
        interpolation=cv2.INTER_NEAREST,
    )

    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)

    x0 = max(0, roi_x)
    y0 = max(0, roi_y)
    x1 = min(orig_w, roi_x + roi_size_orig)
    y1 = min(orig_h, roi_y + roi_size_orig)
    if x1 <= x0 or y1 <= y0:
        return full_mask

    src_x0 = x0 - roi_x
    src_y0 = y0 - roi_y
    src_x1 = src_x0 + (x1 - x0)
    src_y1 = src_y0 + (y1 - y0)
    full_mask[y0:y1, x0:x1] = roi_resized[src_y0:src_y1, src_x0:src_x1]

    # Save as conventional binary-image intensities.
    return (full_mask * 255).astype(np.uint8)


def _save_target_mask_original_with_warning(
    roi_mask_bin: np.ndarray,
    catalog_entry: dict,
    roi_stem: str,
    out_path: Path | str,
) -> None:
    """Save original-size target mask and emit explicit warnings on failure."""
    if not isinstance(catalog_entry, dict) or not catalog_entry:
        warnings.warn(
            f"[{roi_stem}] Target-mask export skipped: no ROI catalog entry found.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    required = ("original_width", "original_height", "roi_top_left", "roi_size_original")
    missing = [k for k in required if k not in catalog_entry]
    if missing:
        warnings.warn(
            f"[{roi_stem}] Target-mask export skipped: ROI catalog entry is missing "
            f"required field(s): {', '.join(missing)}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    roi_xy = catalog_entry.get("roi_top_left")
    if not isinstance(roi_xy, (list, tuple)) or len(roi_xy) != 2:
        warnings.warn(
            f"[{roi_stem}] Target-mask export skipped: invalid roi_top_left={roi_xy!r} "
            "(expected [x, y]).",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    mask_orig = _roi_mask_to_original_canvas(roi_mask_bin, catalog_entry)
    if mask_orig is None:
        warnings.warn(
            f"[{roi_stem}] Target-mask export skipped: catalog values are invalid for "
            "projection to original image size.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    if not cv2.imwrite(str(out_path), mask_orig):
        warnings.warn(
            f"[{roi_stem}] Target-mask export failed: could not write mask to {out_path}.",
            RuntimeWarning,
            stacklevel=2,
        )


def skeletonize_worm_predictions(
    model_path: str,
    image_dir: str,
    output_dir: str,
    conf_threshold: float = 0.25,
    n_centerline_points: int = 100,
    smooth_factor: float | None = None,
    smooth_factor_ratio: float = 1.0,
    contour_smooth_sigma: float = 2.0,
    max_gap_px: float = 200.0,
    max_joint_angle_deg: float = 70.0,
    tangent_path_len_px: float = 200.0,
    min_fragment_pixels: int = 30,
    hole_fill_ratio_threshold: float = 0.05,
    hole_keep_circularity_weight: float = 0.5,
    ring_dilation_radius: int = 0,
    dir_vector_len_px: float = 20.0,
    min_bridge_arc_len_px: float = 20.0,
    skeleton_px_per_point: float | None = 10.0,
    gap_interp_points: int = 12,
    debug_fragments: bool = False,
    roi_catalog_path: str | None = None,
    save_annotated_images: bool = True,
    save_target_masks: bool = True,
    save_contour_skeleton_txt: bool = False,
    contour_max_points: int = 200,
    draw_annotation_text: bool = True,
    per_image_timeout_sec: float = 60.0,
    # ---- speed-measurement (opt-in; default off keeps original behavior) ----
    measure_speed: bool = False,
    speed_output_dir: str | None = None,
    speed_num_batches: int = 10,
    speed_warmup: int = 5,
    speed_device: str = "auto",
    device: str = "",
) -> list[tuple[str, float, float, float, float]]:
    """
    Run YOLO segmentation inference, extract a smooth centerline for the
    detected worm whose bounding-box centre is closest to the image centre,
    and optionally save annotated images.  All other detections are drawn as
    faint outlines for context but are not skeletonized.

    Parameters
    ----------
    model_path :
        Path to trained YOLO ``.pt`` weights.
    image_dir :
        Directory of input images.
    output_dir :
        Destination for annotated images.
    conf_threshold :
        YOLO detection confidence threshold.
    n_centerline_points :
        Number of points on the resampled spline centerline.
    smooth_factor :
        Passed directly to ``extract_centerline``; ``None`` = auto.
    contour_smooth_sigma :
        Gaussian blur sigma applied to binary masks before skeletonization to
        round off sharp polygon-corner artifacts from YOLO.  Set to 0 to
        disable.  Default 2.0 px works well for typical ROI crop sizes.
    max_gap_px :
        Maximum Euclidean distance (pixels) allowed between two fragment
        endpoints when stitching a multi-blob mask.  If the best gap exceeds
        this value the blobs are treated as YOLO errors and only the single
        fragment whose centroid is closest to the centre of the field-of-view
        is skeletonized.  Default 200 px.  Set to ``float('inf')`` to always
        stitch regardless of gap size.
    hole_fill_ratio_threshold :
        Passed to ``extract_centerline``.  Fraction of original mask area that
        hole-filling must add before the mask is treated as a ring shape and
        filling is skipped.  Default 0.05 (5%).
    hole_keep_circularity_weight :
        Passed to ``extract_centerline``.  Blend weight between circularity
        (1.0) and normalised area (0.0) used to select which enclosed hole is
        the loop interior.  Default 0.5.
    ring_dilation_radius :
        Passed to ``extract_centerline``.  Radius (pixels) of the elliptical
        dilation applied to ring-shaped masks before selective hole analysis.
        0 = disabled (default).
    dir_vector_len_px :
        Passed to ``extract_centerline``.  Arc length (pixels) used to
        estimate the incoming/outgoing direction vectors at junction seams for
        smoothness scoring.  Default 20.0.
    min_bridge_arc_len_px :
        Passed to ``extract_centerline``.  Minimum arc length (pixels) for
        Case 6 direction scoring to be trusted.  When the shorter of the two
        parallel arcs is below this threshold, scoring is skipped and the
        shorter arc is used as the bridge directly.  Default 20.0.
    smooth_factor_ratio :
        Passed to ``extract_centerline``.  Multiplier on ``len(skeleton_pixels)``
        used as the ``splprep`` smoothing factor when ``smooth_factor`` is
        ``None``.  ``1.0`` (default) ≈ 1 px average deviation;  ``0`` = exact
        interpolation (wiggly);  ``>1`` = heavier smoothing.
    gap_interp_points :
        Passed to ``stitch_centerline_fragments``.  Number of linearly
        interpolated points inserted across each stitched gap between
        fragments.  Default 12.
    save_annotated_images :
        If ``True`` (default), write each input image with the mask overlay
        and fitted centerline to *output_dir*.  Set ``False`` to skip saving
        annotated images (useful when only the CSV or .txt outputs are needed).
    save_target_masks :
        If ``True`` (default), export one projected binary target mask per
        input image to ``target_masks_original``.  Set ``False`` to skip mask
        image export entirely.
    """
    print("\n" + "=" * 70)
    print("C. ELEGANS CENTERLINE SKELETONIZATION")
    print("=" * 70)
    print(f"Model             : {model_path}")
    print(f"Images            : {image_dir}")
    print(f"Output            : {output_dir}")
    print(f"Conf threshold    : {conf_threshold}")
    print(f"Centerline points : {n_centerline_points}")
    print(f"Skel px/point     : {skeleton_px_per_point}")
    print(f"Smooth factor     : {smooth_factor}")
    print(f"Smooth factor ratio: {smooth_factor_ratio}")
    print(f"Contour sigma     : {contour_smooth_sigma}")
    print(f"Max gap (px)      : {max_gap_px}")
    print(f"Max joint angle   : {max_joint_angle_deg}°")
    print(f"Tangent path len  : {tangent_path_len_px} px")
    print(f"Gap interp points : {gap_interp_points}")
    print(f"Min fragment px   : {min_fragment_pixels}")
    print(f"Hole fill thresh  : {hole_fill_ratio_threshold}")
    print(f"Hole circ. weight : {hole_keep_circularity_weight}")
    print(f"Ring dilation r   : {ring_dilation_radius} px")
    print(f"Dir vector len    : {dir_vector_len_px} px")
    print(f"Min bridge arc    : {min_bridge_arc_len_px} px")
    print(f"Debug fragments   : {debug_fragments}")
    print(f"ROI catalog       : {roi_catalog_path}")
    print(f"Save annotated    : {save_annotated_images}")
    print(f"Save target masks : {save_target_masks}")
    print(f"Save contour/skel : {save_contour_skeleton_txt}")
    print(f"Contour max pts   : {contour_max_points}")
    print(f"Draw text overlay : {draw_annotation_text}")
    print("=" * 70 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    debug_dir = Path(output_dir) / "debug_fragments"
    target_mask_dir = Path(output_dir) / "target_masks_original"
    contour_skel_dir = Path(output_dir) / "contour_skeleton_txt"
    if debug_fragments:
        debug_dir.mkdir(exist_ok=True)
    if save_target_masks:
        target_mask_dir.mkdir(exist_ok=True)
    if save_contour_skeleton_txt:
        contour_skel_dir.mkdir(exist_ok=True)

    # Load ROI catalog (optional — used to convert ROI-pixel measurements back
    # to original-image pixel scale via the stored scale_factor).
    roi_catalog: dict = {}
    if roi_catalog_path is not None:
        try:
            import json
            with open(roi_catalog_path, "r") as _f:
                roi_catalog = json.load(_f)
            print(f"✓ ROI catalog loaded: {len(roi_catalog)} entries from {roi_catalog_path}\n")
        except Exception as _e:
            print(f"⚠ Could not load ROI catalog ({roi_catalog_path}): {_e}\n")

    # Load model
    print("Loading model...")
    model = YOLO(model_path)
    print(f"✓ Model loaded\n")

    # Image discovery — shared helper from visualize_predictions
    image_files = get_image_files(image_dir)
    print(f"Found {len(image_files)} images to process\n")
    if not image_files:
        print("❌ No images found! Check IMAGE_DIR.")
        return
    if save_target_masks:
        print(f"✓ Target masks (original size) → {target_mask_dir}\n")

    # ---- speed meter (no-op when measure_speed=False) ----
    speed_meter = SpeedMeter(
        name="segment", enabled=measure_speed, device=speed_device
    )
    if measure_speed:
        import sys as _sys
        _sys.stderr.write(
            f"[speed] segment: device={speed_meter.device}, "
            f"num_batches={speed_num_batches}, warmup={speed_warmup}, "
            f"images={len(image_files)}\n"
        )
        w_count = min(int(speed_warmup), len(image_files))
        speed_meter.warmup_start()
        for _wp in image_files[:w_count]:
            _wimg = cv2.imread(str(_wp))
            if _wimg is None:
                continue
            _wkw = dict(source=_wimg, conf=conf_threshold, verbose=False)
            if device:
                _wkw["device"] = device
            model.predict(**_wkw)
        speed_meter.warmup_end(w_count)

    stats = {
        "total_images"         : len(image_files),
        "images_with_detections": 0,
        "total_detections"     : 0,
        "successful_centerlines": 0,
    }
    # Ordered list of (filename, date, experiment, plate_id, magnification,
    #                    well_id, worm_id, length_px, width_px) for every processed image.
    # length_px / width_px are in original-image pixel scale when an ROI catalog
    # is provided, otherwise in ROI pixel scale.  NaN when skeletonization failed.
    worm_sizes: list[tuple[str, str, str, str, str, str, str, float, float]] = []
    csv_path, _csv_fh, _csv_writer = _init_worm_sizes_csv(output_dir)

    # ── Hang diagnostics ───────────────────────────────────────────────────
    # Write the filename of the image currently being processed to a small
    # "heartbeat" file BEFORE each iteration starts.  If the job is killed
    # (e.g. SLURM wall-time timeout) you can `cat` this file to identify the
    # exact image that was being processed when the hang occurred.
    heartbeat_path = Path(output_dir) / "_current_image.txt"
    stuck_log_path = Path(output_dir) / "_processed_images.log"
    # Also append every started image (with timestamp) to a log so that
    # post-mortem you can see the last few processed images and timings.
    import time as _time
    _stuck_fh = open(stuck_log_path, "a", buffering=1)  # line-buffered
    _stuck_fh.write(f"# === run started {_time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    # ── Per-image timeout (POSIX-only, via SIGALRM) ────────────────────────
    # When per_image_timeout_sec > 0 on a POSIX system, any image that takes
    # longer than the limit raises TimeoutError, gets logged as TIMEOUT, and
    # is skipped so the rest of the job can continue.  On Windows or when set
    # to 0, the timeout is silently disabled.
    _per_img_timeout_ok = (
        per_image_timeout_sec is not None
        and per_image_timeout_sec > 0
        and hasattr(signal, "SIGALRM")
    )
    if _per_img_timeout_ok:
        def _alarm_handler(_signum, _frame):
            raise TimeoutError(
                f"image processing exceeded {per_image_timeout_sec:.0f}s"
            )
        signal.signal(signal.SIGALRM, _alarm_handler)
        print(
            f"✓ Per-image timeout enabled: {per_image_timeout_sec:.0f}s "
            "(images exceeding this will be skipped)\n"
        )
    elif per_image_timeout_sec and per_image_timeout_sec > 0:
        print(
            "⚠ Per-image timeout requested but SIGALRM is unavailable on this "
            "platform (likely Windows) — timeout disabled.\n"
        )

    def _process_image(img_path):
        """Per-image worker.  Raises TimeoutError if the wall-clock limit fires.

        Returns nothing.  Uses closure over the enclosing function's locals
        (model, stats dict, worm_sizes list, csv writer, etc.).
        """
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ Cannot read {img_path.name}")
            return

        img_h, img_w = img.shape[:2]
        target_mask_roi = np.zeros((img_h, img_w), dtype=np.uint8)

        speed_meter.tick_start()
        _pkw = dict(source=img, conf=conf_threshold, verbose=False)
        if device:
            _pkw["device"] = device
        results = model.predict(**_pkw)[0]
        speed_meter.tick_end()

        if results.masks is None or len(results.masks) == 0:
            if save_annotated_images:
                annotated = img.copy()
                if draw_annotation_text:
                    cv2.putText(
                        annotated, "No Detection", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    )
                cv2.imwrite(str(Path(output_dir) / img_path.name), annotated)
            if save_target_masks:
                _save_target_mask_original_with_warning(
                    target_mask_roi,
                    roi_catalog.get(img_path.stem, {}),
                    img_path.stem,
                    target_mask_dir / img_path.name,
                )
            return

        masks_raw  = results.masks.data.cpu().numpy()          # (N, H_m, W_m)
        boxes_xyxy = results.boxes.xyxy.cpu().numpy()           # (N, 4)
        confs      = results.boxes.conf.cpu().numpy()           # (N,)
        class_ids  = results.boxes.cls.cpu().numpy().astype(int)

        stats["images_with_detections"] += 1
        stats["total_detections"]       += len(masks_raw)

        h, w = img_h, img_w

        # Resize all masks to image dimensions up-front
        masks_full = [
            cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            for m in masks_raw
        ]

        # Identify the single worm to skeletonize
        target_idx = find_center_worm_idx(boxes_xyxy, h, w)

        annotated = img.copy()
        target_arc_length_px: float = float("nan")
        target_width_px: float = float("nan")
        target_arc_length_orig_px: float = float("nan")
        target_width_orig_px: float = float("nan")

        for i, (mask_f32, conf, cls_id) in enumerate(zip(masks_full, confs, class_ids)):
            mask_bin = (mask_f32 > 0.5).astype(np.uint8)

            if i != target_idx:
                # Non-target: draw a faint grey contour outline only
                contours, _ = cv2.findContours(
                    mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, (160, 160, 160), 1)
                continue

            target_mask_roi = mask_bin.copy()

            # ── Target worm: multi-fragment-aware skeletonization ──────────
            # A single YOLO detection can produce a mask with multiple
            # disconnected blobs when the worm is occluded mid-body.  We
            # skeletonize each fragment independently, then stitch them into
            # one continuous head-to-tail centerline.

            # Points per fragment: use arc-length spatial resolution if set;
            # fall back to evenly distributing n_centerline_points.
            labeled_tmp, n_frags = sk_label(
                (mask_bin > 0).astype(np.uint8), connectivity=2, return_num=True
            )

            fragments, effective_mask = _skeletonize_mask_fragments(
                mask_bin,
                n_points_per_fragment=max(1, n_centerline_points // max(n_frags, 1)),
                px_per_point=skeleton_px_per_point,
                smooth_factor=smooth_factor,
                smooth_factor_ratio=smooth_factor_ratio,
                contour_smooth_sigma=contour_smooth_sigma,
                min_fragment_pixels=min_fragment_pixels,
                hole_fill_ratio_threshold=hole_fill_ratio_threshold,
                hole_keep_circularity_weight=hole_keep_circularity_weight,
                ring_dilation_radius=ring_dilation_radius,
                dir_vector_len_px=dir_vector_len_px,
                min_bridge_arc_len_px=min_bridge_arc_len_px,
                debug_ring_mask_prefix=(
                    str(debug_dir / img_path.stem) if debug_fragments else None
                ),
                img_h=h,
                img_w=w,
                img_name=img_path.name,
                _precomputed_labeled=labeled_tmp,
                _precomputed_n_comp=n_frags,
            )

            # ── Optional debug: save per-fragment orientation image ────────
            if debug_fragments and len(fragments) > 1:
                _save_fragment_debug_image(
                    img, mask_bin, fragments,
                    str(debug_dir / f"{img_path.stem}_fragments.jpg"),
                )

            centerline, width_profile, gap_ranges, gap_exceeded, stitch_failure_reason, fragment_chain = \
                stitch_centerline_fragments(
                    fragments, n_points=n_centerline_points,
                    max_gap_px=max_gap_px,
                    max_joint_angle_deg=max_joint_angle_deg,
                    tangent_path_len_px=tangent_path_len_px,
                    gap_interp_points=gap_interp_points,
                    px_per_point=skeleton_px_per_point if skeleton_px_per_point is not None else 1.0,
                )

            # ── Gap-exceeded fallback: use fragment closest to FOV centre ──
            # When the gap between blobs is larger than max_gap_px the blobs
            # are likely unrelated YOLO mask errors.  Use only the single
            # fragment whose bounding-box centre is nearest to the image centre.
            # _skeletonize_mask_fragments already returns fragments in ascending
            # bounding-box-centre-distance order, so fragment 0 is always the
            # correct fallback — no re-search needed.
            frag_label = ""
            if gap_exceeded and fragments:
                best_frag_idx = 0   # fragment 0 is closest to image centre by construction
                cl_fb, wp_fb = fragments[best_frag_idx]
                idx_fb = np.round(
                    np.linspace(0, len(cl_fb) - 1, n_centerline_points)
                ).astype(int)
                centerline    = cl_fb[idx_fb]
                width_profile = wp_fb[idx_fb]
                gap_ranges    = []
                frag_label    = f" (no valid stitch [{stitch_failure_reason}] — closest-to-centre fragment used)"
                fragment_chain = [(best_frag_idx, False)]   # single fallback fragment

            # ── Optional debug: save post-stitch overlay image ─────────────
            if debug_fragments and len(fragments) > 1 and centerline is not None:
                _save_stitched_debug_image(
                    img, mask_bin, fragments,
                    centerline, gap_ranges, stitch_failure_reason,
                    fragment_chain,
                    str(debug_dir / f"{img_path.stem}_stitched.jpg"),
                )

            ok = centerline is not None

            if ok:
                stats["successful_centerlines"] += 1
                diffs = np.diff(centerline.astype(float), axis=0)
                seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])   # (N-1,)
                target_arc_length_px = float(seg_lengths.sum())

                # ── Width = mask area / body arc length ────────────────────
                # After resampling the skeleton to N_CENTERLINE_POINTS, check
                # which resampled points actually fall inside the mask.
                # Only segments whose BOTH endpoints are inside the mask
                # contribute to body arc length.  This is more accurate than
                # excluding entire gap_index_ranges regions, because some
                # interpolated gap points may land inside the mask after
                # resampling and should therefore be counted.
                r_cl = np.clip(centerline[:, 0].astype(int), 0, h - 1)
                c_cl = np.clip(centerline[:, 1].astype(int), 0, w - 1)
                # Use the hole-filled effective mask so that points falling in
                # filled body-gap holes are correctly counted as inside the mask.
                in_mask_cl  = effective_mask[r_cl, c_cl].astype(bool)   # (N,)
                # seg_lengths[k] = distance from centerline[k] to [k+1]
                seg_in_mask = in_mask_cl[:-1] & in_mask_cl[1:]     # (N-1,)
                body_arc  = float((seg_lengths * seg_in_mask).sum())
                mask_area = float(mask_bin.sum())
                target_width_px = (mask_area / body_arc) if body_arc > 0 else float("nan")

                # ── Scale back to original-image coordinates ───────────────
                # The ROI catalog records scale_factor = output_image_size /
                # roi_size_original, so dividing by it recovers real-world px.
                _catalog_entry = roi_catalog.get(img_path.stem, {})
                _scale = _catalog_entry.get("scale_factor", None)
                if _scale is not None and _scale > 0:
                    target_arc_length_orig_px = target_arc_length_px / _scale
                    target_width_orig_px = (
                        target_width_px / _scale
                        if not np.isnan(target_width_px) else float("nan")
                    )

                if not frag_label and n_frags > 1:
                    frag_label = f" ({n_frags} fragments stitched)"
                annotated = draw_centerline_on_image(
                    annotated, mask_bin, centerline,
                    gap_index_ranges=gap_ranges,
                    in_mask_points=in_mask_cl,
                )

                # Label near head endpoint
                r0, c0 = int(centerline[0, 0]), int(centerline[0, 1])
                label = f"{model.names[cls_id]} {conf:.2f} [center]{frag_label}"
                if draw_annotation_text:
                    cv2.putText(
                        annotated, label, (c0 + 6, r0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                # Measurement label: top-right corner (ROI-pixel scale → µm)
                width_str = f"{target_width_px * UM_PER_PX:.1f}" if not np.isnan(target_width_px) else "N/A"
                meas_label = f"L={target_arc_length_px * UM_PER_PX:.1f}µm  W={width_str}µm (ROI)"
                (meas_w, meas_h), _ = cv2.getTextSize(
                    meas_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                meas_x = w - meas_w - 8
                meas_y = 22
                if draw_annotation_text:
                    cv2.putText(
                        annotated, meas_label, (meas_x, meas_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated, meas_label, (meas_x, meas_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                # Original-image-scale measurements (second line, shown only
                # when the ROI catalog provides a valid scale_factor).
                if not np.isnan(target_arc_length_orig_px):
                    orig_width_str = (
                        f"{target_width_orig_px * UM_PER_PX:.1f}" if not np.isnan(target_width_orig_px) else "N/A"
                    )
                    meas_label_orig = (
                        f"L={target_arc_length_orig_px * UM_PER_PX:.1f}µm  W={orig_width_str}µm (orig)"
                    )
                    (meas_w2, _), _ = cv2.getTextSize(
                        meas_label_orig, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                    )
                    meas_x2 = w - meas_w2 - 8
                    meas_y2 = meas_y + 20
                    if draw_annotation_text:
                        cv2.putText(
                            annotated, meas_label_orig, (meas_x2, meas_y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                        )
                        cv2.putText(
                            annotated, meas_label_orig, (meas_x2, meas_y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA,
                        )

                # ── Legend ─────────────────────────────────────────────────
                # Always show dot-marker legend; gap-line rows added only when
                # gap_ranges are present.
                if draw_annotation_text:
                    leg_x, leg_y = 8, 28          # top-left anchor
                    leg_line_len = 22              # length of colour swatch
                    leg_gap      = 18              # row spacing
                    font         = cv2.FONT_HERSHEY_SIMPLEX
                    fscale       = 0.45
                    n_legend_rows = 4 if gap_ranges else 2
                    box_w, box_h  = 280, leg_gap * n_legend_rows + 2
                    sub = annotated[leg_y - 16 : leg_y + box_h, leg_x - 4 : leg_x + box_w]
                    if sub.shape[0] > 0 and sub.shape[1] > 0:
                        black = np.zeros_like(sub)
                        cv2.addWeighted(black, 0.45, sub, 0.55, 0, sub)
                        annotated[leg_y - 16 : leg_y + box_h, leg_x - 4 : leg_x + box_w] = sub
                    # Row 0: white dot → inside-mask point (included in width)
                    cv2.circle(annotated, (leg_x + leg_line_len // 2, leg_y), 2,
                               (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated, "point inside mask (used in width)",
                                (leg_x + leg_line_len + 4, leg_y + 4),
                                font, fscale, (255, 255, 255), 1, cv2.LINE_AA)
                    # Row 1: red dot → outside-mask point (excluded from width)
                    gy1 = leg_y + leg_gap
                    cv2.circle(annotated, (leg_x + leg_line_len // 2, gy1), 3,
                               (0, 0, 220), -1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated, "point outside mask (excl. from width)",
                                (leg_x + leg_line_len + 4, gy1 + 4),
                                font, fscale, (0, 0, 220), 1, cv2.LINE_AA)
                    if gap_ranges:
                        # Row 2: green solid line → body skeleton
                        gy2 = leg_y + leg_gap * 2
                        cv2.line(annotated, (leg_x, gy2), (leg_x + leg_line_len, gy2),
                                 (0, 255, 0), 2, lineType=cv2.LINE_AA)
                        cv2.putText(annotated, "body skeleton",
                                    (leg_x + leg_line_len + 4, gy2 + 4),
                                    font, fscale, (255, 255, 255), 1, cv2.LINE_AA)
                        # Row 3: orange dashed line → stitched gap
                        gy3 = leg_y + leg_gap * 3
                        for dx in range(0, leg_line_len, 7):
                            x1 = leg_x + dx
                            x2 = min(leg_x + dx + 4, leg_x + leg_line_len)
                            cv2.line(annotated, (x1, gy3), (x2, gy3),
                                     (0, 165, 255), 2, lineType=cv2.LINE_AA)
                        cv2.putText(annotated, "stitched gap",
                                    (leg_x + leg_line_len + 4, gy3 + 4),
                                    font, fscale, (0, 165, 255), 1, cv2.LINE_AA)

            else:
                # Centerline failed — fall back to contour outline in red
                contours, _ = cv2.findContours(
                    mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

        n_cl = int(not np.isnan(target_arc_length_px))
        if draw_annotation_text:
            cv2.putText(
                annotated,
                f"Detections: {len(masks_raw)}  |  Skeletonized: {n_cl} (closest to center)",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

        if save_annotated_images:
            cv2.imwrite(str(Path(output_dir) / img_path.name), annotated)
        if save_target_masks:
            _save_target_mask_original_with_warning(
                target_mask_roi,
                roi_catalog.get(img_path.stem, {}),
                img_path.stem,
                target_mask_dir / img_path.name,
            )
        if save_contour_skeleton_txt:
            _save_contour_skeleton_txt(
                target_mask_roi,
                centerline if not np.isnan(target_arc_length_px) else None,
                roi_catalog.get(img_path.stem, {}),
                img_path.stem,
                contour_skel_dir / (img_path.stem + ".txt"),
                contour_max_points=contour_max_points,
            )
        _length = target_arc_length_orig_px if not np.isnan(target_arc_length_orig_px) else target_arc_length_px
        _width  = target_width_orig_px      if not np.isnan(target_width_orig_px)      else target_width_px
        _row = _build_worm_sizes_row(
            filename=img_path.name,
            stem=img_path.stem,
            length_um=_length * UM_PER_PX if not np.isnan(_length) else float("nan"),
            width_um=_width  * UM_PER_PX if not np.isnan(_width)  else float("nan"),
        )
        # Row is written by the caller (after collecting any topology
        # warnings emitted during skeletonization).  Returning ``None``
        # signals "no measurement row for this image".
        return _row

    pbar = tqdm(image_files, desc="Skeletonizing")
    stats["timed_out_images"] = 0
    for img_path in pbar:
        # Heartbeat: overwrite with the current filename (atomic-ish on POSIX).
        try:
            heartbeat_path.write_text(img_path.name)
        except Exception:
            pass
        pbar.set_postfix_str(img_path.name[:40], refresh=False)
        _t_start = _time.time()
        _stuck_fh.write(
            f"START\t{_time.strftime('%H:%M:%S')}\t{img_path.name}\n"
        )

        if _per_img_timeout_ok:
            signal.alarm(int(per_image_timeout_sec))
        try:
            # Capture any centerline-topology warnings fired during this
            # image so they can be surfaced in the CSV's Topology_Warnings
            # column.  The captured warnings are re-emitted after the
            # ``with`` block exits so the terminal output is unchanged.
            with warnings.catch_warnings(record=True) as _caught_warns:
                warnings.simplefilter("always", RuntimeWarning)
                _row = _process_image(img_path)
            for _w in _caught_warns:
                warnings.warn_explicit(
                    _w.message, _w.category, _w.filename, _w.lineno
                )
            if _row is not None:
                _flags = _collect_topology_warnings(_caught_warns)
                if _flags:
                    _row = _row[:-1] + (_flags,)
                worm_sizes.append(_row)
                _csv_writer.writerow(_row)
            _stuck_fh.write(
                f"END  \t{_time.strftime('%H:%M:%S')}\t{img_path.name}\t"
                f"{_time.time() - _t_start:.2f}s\n"
            )
        except TimeoutError:
            stats["timed_out_images"] += 1
            print(
                f"\n⏱  TIMEOUT after {per_image_timeout_sec:.0f}s on "
                f"{img_path.name} — skipping."
            )
            _stuck_fh.write(
                f"TIMEOUT\t{_time.strftime('%H:%M:%S')}\t{img_path.name}\t"
                f"{per_image_timeout_sec:.0f}s\n"
            )
        finally:
            if _per_img_timeout_ok:
                signal.alarm(0)

    _csv_fh.close()
    _stuck_fh.close()
    try:
        heartbeat_path.unlink()
    except FileNotFoundError:
        pass

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SKELETONIZATION COMPLETE")
    print("=" * 70)
    print(f"Total images            : {stats['total_images']}")
    print(f"Images with detections  : {stats['images_with_detections']}")
    print(f"Total YOLO detections   : {stats['total_detections']}")
    print(f"Successful centerlines  : {stats['successful_centerlines']}")
    if stats.get("timed_out_images", 0):
        print(
            f"Timed-out images        : {stats['timed_out_images']} "
            f"(skipped after {per_image_timeout_sec:.0f}s each)"
        )
    if save_annotated_images:
        print(f"\n✓ Annotated images  → {output_dir}")
    if save_target_masks:
        print(f"✓ Target masks      → {target_mask_dir}")
    if save_contour_skeleton_txt:
        print(f"✓ Contour/skeleton  → {contour_skel_dir}")
    print(f"✓ Worm sizes        → {csv_path}")
    print("=" * 70 + "\n")

    # ---- finalize speed meter (writes nothing when disabled) ----
    if measure_speed:
        _so_dir = speed_output_dir or os.path.join(output_dir, "speed")
        speed_meter.finalize(
            output_dir=_so_dir,
            num_batches=int(speed_num_batches),
            extra_metadata={
                "model_path": model_path,
                "image_dir": image_dir,
                "image_count": len(image_files),
                "conf_threshold": conf_threshold,
                "device_arg": speed_device,
                "device_label": device_label(speed_device),
            },
        )

    return worm_sizes


# ---------------------------------------------------------------------------
# Ground-truth annotation mode helpers
# ---------------------------------------------------------------------------

def _yolo_seg_label_to_masks(
    label_path: str,
    img_h: int,
    img_w: int,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """
    Parse a YOLO segmentation ``.txt`` label file and rasterize each polygon
    into a binary mask.

    YOLO segmentation format (one annotation per line)::

        class_id  x1 y1  x2 y2  ...  xN yN

    All coordinates are **normalised** to [0, 1].

    Parameters
    ----------
    label_path :
        Path to the ``.txt`` annotation file.
    img_h, img_w :
        Target image height and width (pixels).

    Returns
    -------
    masks_full :
        List of ``(img_h × img_w)`` float32 arrays in [0, 1].
        Each element corresponds to one annotation line.
    boxes_xyxy :
        ``(N, 4)`` float32 array of ``[x1, y1, x2, y2]`` bounding boxes
        (absolute pixel coordinates) derived from the polygon extents.
    class_ids :
        ``(N,)`` int32 array of class indices.
    """
    masks_full: list[np.ndarray] = []
    boxes_list: list[list[float]] = []
    class_ids_list: list[int] = []

    with open(label_path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 7:   # need at least class + 3 xy pairs
                continue
            cls_id = int(tokens[0])
            coords = [float(v) for v in tokens[1:]]
            if len(coords) % 2 != 0 or len(coords) < 6:
                continue
            xs_norm = np.array(coords[0::2], dtype=np.float32)
            ys_norm = np.array(coords[1::2], dtype=np.float32)
            xs_px = np.clip(xs_norm * img_w, 0, img_w - 1)
            ys_px = np.clip(ys_norm * img_h, 0, img_h - 1)

            # Rasterize polygon
            polygon_pts = np.stack([xs_px, ys_px], axis=1).astype(np.int32)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon_pts], 1)

            # Bounding box from polygon extents
            x1, y1 = float(xs_px.min()), float(ys_px.min())
            x2, y2 = float(xs_px.max()), float(ys_px.max())

            masks_full.append(mask.astype(np.float32))
            boxes_list.append([x1, y1, x2, y2])
            class_ids_list.append(cls_id)

    if not masks_full:
        return [], np.zeros((0, 4), dtype=np.float32), np.zeros(0, dtype=np.int32)

    boxes_xyxy = np.array(boxes_list, dtype=np.float32)
    class_ids  = np.array(class_ids_list, dtype=np.int32)
    return masks_full, boxes_xyxy, class_ids


def skeletonize_gt_annotations(
    image_dir: str,
    labels_dir: str,
    output_dir: str,
    class_names: dict[int, str] | None = None,
    n_centerline_points: int = 100,
    smooth_factor: float | None = None,
    smooth_factor_ratio: float = 1.0,
    contour_smooth_sigma: float = 2.0,
    max_gap_px: float = 200.0,
    max_joint_angle_deg: float = 70.0,
    tangent_path_len_px: float = 200.0,
    min_fragment_pixels: int = 30,
    hole_fill_ratio_threshold: float = 0.05,
    hole_keep_circularity_weight: float = 0.5,
    ring_dilation_radius: int = 0,
    dir_vector_len_px: float = 20.0,
    min_bridge_arc_len_px: float = 20.0,
    skeleton_px_per_point: float | None = 10.0,
    gap_interp_points: int = 12,
    debug_fragments: bool = False,
    roi_catalog_path: str | None = None,
    save_annotated_images: bool = True,
    save_target_masks: bool = True,
    save_contour_skeleton_txt: bool = False,
    contour_max_points: int = 200,
    draw_annotation_text: bool = True,
) -> list[tuple[str, float, float, float, float]]:
    """
    Generate ground-truth skeleton lines from human annotations (YOLO
    segmentation ``.txt`` label files) rather than from YOLO inference.

    The skeletonization pipeline is **identical** to
    ``skeletonize_worm_predictions``; the only difference is the source of the
    binary masks.  Masks are rasterized from the polygon annotations in the
    label files instead of being produced by a neural-network model.

    For each image the annotation whose bounding-box centre is closest to the
    image centre is selected as the primary worm (matching the YOLO-mode
    heuristic), and all other annotations in the same image are drawn as faint
    grey contours for context.

    Parameters
    ----------
    image_dir :
        Directory containing the input images.
    labels_dir :
        Directory containing the YOLO segmentation ``.txt`` label files.
        Each label file must have the same stem as the corresponding image.
    output_dir :
        Destination directory for annotated images.
    class_names :
        Optional mapping from integer class ID to string name.
        ``None`` → class labels are rendered as ``"class_<id>"``.
    n_centerline_points :
        Number of points on the resampled spline centerline.
    smooth_factor :
        Spline smoothing parameter (``None`` = automatic).
    smooth_factor_ratio :
        Multiplier on ``len(skeleton_pixels)`` for auto smoothing.
    contour_smooth_sigma :
        Gaussian blur sigma applied to binary masks before skeletonization.
    max_gap_px :
        Maximum Euclidean gap (pixels) allowed when stitching multi-blob masks.
    max_joint_angle_deg :
        Maximum gap-connection angle deviation (degrees).
    tangent_path_len_px :
        Arc length (pixels) used to estimate tangent direction at endpoints.
    min_fragment_pixels :
        Minimum connected-component area (pixels²) to attempt skeletonization.
    hole_fill_ratio_threshold :
        Fraction of mask area that hole-filling must add to classify as ring.
    hole_keep_circularity_weight :
        Blend weight for ring-hole selection (1.0 = circularity, 0.0 = area).
    ring_dilation_radius :
        Dilation radius (pixels) applied to ring masks before hole analysis.
    dir_vector_len_px :
        Arc length (pixels) for directional averaging at junction seams.
    min_bridge_arc_len_px :
        Minimum bridge arc length for Case-6 direction scoring.
    skeleton_px_per_point :
        Spatial resolution of fragment skeletons in pixels per point.
    gap_interp_points :
        Number of interpolated points inserted across each stitched gap.
    debug_fragments :
        If ``True``, save per-image fragment debug images.
    roi_catalog_path :
        Optional path to ``roi_catalog.json`` for scale conversion.
    save_annotated_images :
        If ``True`` (default), write each input image with the mask overlay
        and fitted centerline to *output_dir*.  Set ``False`` to skip saving
        annotated images.
    save_target_masks :
        If ``True`` (default), export one projected binary target mask per
        input image to ``target_masks_original``.  Set ``False`` to skip mask
        image export entirely.

    Returns
    -------
    list of ``(filename, date, experiment, plate_id, magnification,
               well_id, worm_id, length_um, width_um)`` tuples.
        Values are ``float('nan')`` when no annotation or skeletonization failed.
    """
    print("\n" + "=" * 70)
    print("C. ELEGANS CENTERLINE SKELETONIZATION  [GROUND-TRUTH ANNOTATION MODE]")
    print("=" * 70)
    print(f"Images            : {image_dir}")
    print(f"Labels            : {labels_dir}")
    print(f"Output            : {output_dir}")
    print(f"Centerline points : {n_centerline_points}")
    print(f"Skel px/point     : {skeleton_px_per_point}")
    print(f"Smooth factor     : {smooth_factor}")
    print(f"Smooth factor ratio: {smooth_factor_ratio}")
    print(f"Contour sigma     : {contour_smooth_sigma}")
    print(f"Max gap (px)      : {max_gap_px}")
    print(f"Max joint angle   : {max_joint_angle_deg}°")
    print(f"Tangent path len  : {tangent_path_len_px} px")
    print(f"Gap interp points : {gap_interp_points}")
    print(f"Min fragment px   : {min_fragment_pixels}")
    print(f"Hole fill thresh  : {hole_fill_ratio_threshold}")
    print(f"Hole circ. weight : {hole_keep_circularity_weight}")
    print(f"Ring dilation r   : {ring_dilation_radius} px")
    print(f"Dir vector len    : {dir_vector_len_px} px")
    print(f"Min bridge arc    : {min_bridge_arc_len_px} px")
    print(f"Debug fragments   : {debug_fragments}")
    print(f"ROI catalog       : {roi_catalog_path}")
    print(f"Save annotated    : {save_annotated_images}")
    print(f"Save target masks : {save_target_masks}")
    print(f"Save contour/skel : {save_contour_skeleton_txt}")
    print(f"Contour max pts   : {contour_max_points}")
    print(f"Draw text overlay : {draw_annotation_text}")
    print("=" * 70 + "\n")

    os.makedirs(output_dir, exist_ok=True)
    debug_dir = Path(output_dir) / "debug_fragments"
    target_mask_dir = Path(output_dir) / "target_masks_original"
    contour_skel_dir = Path(output_dir) / "contour_skeleton_txt"
    if debug_fragments:
        debug_dir.mkdir(parents=True, exist_ok=True)
    if save_target_masks:
        target_mask_dir.mkdir(parents=True, exist_ok=True)
    if save_contour_skeleton_txt:
        contour_skel_dir.mkdir(parents=True, exist_ok=True)

    # Load ROI catalog (optional)
    roi_catalog: dict = {}
    if roi_catalog_path is not None:
        try:
            import json
            with open(roi_catalog_path, "r") as fh:
                roi_catalog = json.load(fh)
        except Exception as exc:
            warnings.warn(
                f"Could not load ROI catalog from {roi_catalog_path}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    # Image discovery — shared helper from visualize_predictions
    image_files = get_image_files(image_dir)
    print(f"Found {len(image_files)} images to process\n")
    if not image_files:
        print("❌ No images found — check image_dir.")
        return []
    if save_target_masks:
        print(f"✓ Target masks (original size) → {target_mask_dir}\n")

    stats = {
        "total_images"          : len(image_files),
        "images_with_annotations": 0,
        "total_annotations"     : 0,
        "successful_centerlines": 0,
    }
    worm_sizes: list[tuple[str, str, str, str, str, str, str, float, float]] = []
    csv_path, _csv_fh, _csv_writer = _init_worm_sizes_csv(output_dir)

    # ── Per-image topology-warning capture ────────────────────────────────
    # Install a ``warnings.showwarning`` hook for the duration of the loop
    # so each iteration can collect any RuntimeWarnings (in particular the
    # Group-A centerline-topology warnings) and surface them in the CSV's
    # ``Topology_Warnings`` column without changing what is printed to the
    # terminal.
    from types import SimpleNamespace as _SimpleNamespace
    _gt_caught_warns: list = []
    _gt_prev_show = warnings.showwarning

    def _gt_show_hook(message, category, filename, lineno, file=None, line=None):
        _gt_caught_warns.append(
            _SimpleNamespace(
                message=message, category=category,
                filename=filename, lineno=lineno,
            )
        )
        return _gt_prev_show(message, category, filename, lineno, file, line)

    warnings.showwarning = _gt_show_hook

    for img_path in tqdm(image_files, desc="Skeletonizing (GT)"):
        _gt_caught_warns.clear()
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠ Cannot read {img_path.name}")
            continue

        h, w = img.shape[:2]
        target_mask_roi = np.zeros((h, w), dtype=np.uint8)

        # ── Load annotations from matching label file ──────────────────────
        label_path = Path(labels_dir) / (img_path.stem + ".txt")
        if not label_path.exists():
            if save_annotated_images:
                annotated = img.copy()
                if draw_annotation_text:
                    cv2.putText(
                        annotated, "No Annotation", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    )
                cv2.imwrite(str(Path(output_dir) / img_path.name), annotated)
            if save_target_masks:
                _save_target_mask_original_with_warning(
                    target_mask_roi,
                    roi_catalog.get(img_path.stem, {}),
                    img_path.stem,
                    target_mask_dir / img_path.name,
                )
            _nan_row = _build_worm_sizes_row(
                filename=img_path.name,
                stem=img_path.stem,
                length_um=float("nan"),
                width_um=float("nan"),
            )
            worm_sizes.append(_nan_row)
            _csv_writer.writerow(_nan_row)
            continue

        masks_full, boxes_xyxy, class_ids = _yolo_seg_label_to_masks(
            str(label_path), h, w
        )

        if len(masks_full) == 0:
            if save_annotated_images:
                annotated = img.copy()
                if draw_annotation_text:
                    cv2.putText(
                        annotated, "No Annotation", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                    )
                cv2.imwrite(str(Path(output_dir) / img_path.name), annotated)
            if save_target_masks:
                _save_target_mask_original_with_warning(
                    target_mask_roi,
                    roi_catalog.get(img_path.stem, {}),
                    img_path.stem,
                    target_mask_dir / img_path.name,
                )
            _nan_row = _build_worm_sizes_row(
                filename=img_path.name,
                stem=img_path.stem,
                length_um=float("nan"),
                width_um=float("nan"),
            )
            worm_sizes.append(_nan_row)
            _csv_writer.writerow(_nan_row)
            continue

        # Confidence is always 1.0 for ground-truth annotations
        confs = np.ones(len(masks_full), dtype=np.float32)

        stats["images_with_annotations"] += 1
        stats["total_annotations"]       += len(masks_full)

        # Identify the single worm to skeletonize (closest to image centre)
        target_idx = find_center_worm_idx(boxes_xyxy, h, w)

        annotated = img.copy()
        target_arc_length_px: float = float("nan")
        target_width_px: float = float("nan")
        target_arc_length_orig_px: float = float("nan")
        target_width_orig_px: float = float("nan")

        for i, (mask_f32, conf, cls_id) in enumerate(zip(masks_full, confs, class_ids)):
            mask_bin = (mask_f32 > 0.5).astype(np.uint8)

            if i != target_idx:
                # Non-target: draw a faint grey contour outline only
                contours, _ = cv2.findContours(
                    mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, (160, 160, 160), 1)
                continue

            target_mask_roi = mask_bin.copy()

            # ── Target worm: multi-fragment-aware skeletonization ──────────
            labeled_tmp, n_frags = sk_label(
                (mask_bin > 0).astype(np.uint8), connectivity=2, return_num=True
            )

            fragments, effective_mask = _skeletonize_mask_fragments(
                mask_bin,
                n_points_per_fragment=max(1, n_centerline_points // max(n_frags, 1)),
                px_per_point=skeleton_px_per_point,
                smooth_factor=smooth_factor,
                smooth_factor_ratio=smooth_factor_ratio,
                contour_smooth_sigma=contour_smooth_sigma,
                min_fragment_pixels=min_fragment_pixels,
                hole_fill_ratio_threshold=hole_fill_ratio_threshold,
                hole_keep_circularity_weight=hole_keep_circularity_weight,
                ring_dilation_radius=ring_dilation_radius,
                dir_vector_len_px=dir_vector_len_px,
                min_bridge_arc_len_px=min_bridge_arc_len_px,
                debug_ring_mask_prefix=(
                    str(debug_dir / img_path.stem) if debug_fragments else None
                ),
                img_h=h,
                img_w=w,
                img_name=img_path.name,
                _precomputed_labeled=labeled_tmp,
                _precomputed_n_comp=n_frags,
            )

            # ── Optional debug: save per-fragment orientation image ────────
            if debug_fragments and len(fragments) > 1:
                _save_fragment_debug_image(
                    img, mask_bin, fragments,
                    str(debug_dir / f"{img_path.stem}_fragments.jpg"),
                )

            centerline, width_profile, gap_ranges, gap_exceeded, stitch_failure_reason, fragment_chain = \
                stitch_centerline_fragments(
                    fragments, n_points=n_centerline_points,
                    max_gap_px=max_gap_px,
                    max_joint_angle_deg=max_joint_angle_deg,
                    tangent_path_len_px=tangent_path_len_px,
                    gap_interp_points=gap_interp_points,
                    px_per_point=skeleton_px_per_point if skeleton_px_per_point is not None else 1.0,
                )

            # ── Gap-exceeded fallback ──────────────────────────────────────
            frag_label = ""
            if gap_exceeded and fragments:
                best_frag_idx = 0
                cl_fb, wp_fb = fragments[best_frag_idx]
                idx_fb = np.round(
                    np.linspace(0, len(cl_fb) - 1, n_centerline_points)
                ).astype(int)
                centerline    = cl_fb[idx_fb]
                width_profile = wp_fb[idx_fb]
                gap_ranges    = []
                frag_label    = f" (no valid stitch [{stitch_failure_reason}] — closest-to-centre fragment used)"
                fragment_chain = [(best_frag_idx, False)]

            # ── Optional debug: save post-stitch overlay image ─────────────
            if debug_fragments and len(fragments) > 1 and centerline is not None:
                _save_stitched_debug_image(
                    img, mask_bin, fragments,
                    centerline, gap_ranges, stitch_failure_reason,
                    fragment_chain,
                    str(debug_dir / f"{img_path.stem}_stitched.jpg"),
                )

            ok = centerline is not None

            if ok:
                stats["successful_centerlines"] += 1
                diffs = np.diff(centerline.astype(float), axis=0)
                seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
                target_arc_length_px = float(seg_lengths.sum())

                r_cl = np.clip(centerline[:, 0].astype(int), 0, h - 1)
                c_cl = np.clip(centerline[:, 1].astype(int), 0, w - 1)
                in_mask_cl  = effective_mask[r_cl, c_cl].astype(bool)
                seg_in_mask = in_mask_cl[:-1] & in_mask_cl[1:]
                body_arc  = float((seg_lengths * seg_in_mask).sum())
                mask_area = float(mask_bin.sum())
                target_width_px = (mask_area / body_arc) if body_arc > 0 else float("nan")

                # ── Scale back to original-image coordinates ───────────────
                _catalog_entry = roi_catalog.get(img_path.stem, {})
                _scale = _catalog_entry.get("scale_factor", None)
                if _scale is not None and _scale > 0:
                    target_arc_length_orig_px = target_arc_length_px / _scale
                    target_width_orig_px = (
                        target_width_px / _scale
                        if not np.isnan(target_width_px) else float("nan")
                    )

                if not frag_label and n_frags > 1:
                    frag_label = f" ({n_frags} fragments stitched)"

                annotated = draw_centerline_on_image(
                    annotated, mask_bin, centerline,
                    gap_index_ranges=gap_ranges,
                    in_mask_points=in_mask_cl,
                )

                # Resolve class name
                cls_name = (
                    class_names[int(cls_id)]
                    if class_names and int(cls_id) in class_names
                    else f"class_{cls_id}"
                )

                # Label near head endpoint
                r0, c0 = int(centerline[0, 0]), int(centerline[0, 1])
                label = f"{cls_name} GT [center]{frag_label}"
                if draw_annotation_text:
                    cv2.putText(
                        annotated, label, (c0 + 6, r0 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                # Measurement label (ROI scale → µm)
                width_str = f"{target_width_px * UM_PER_PX:.1f}" if not np.isnan(target_width_px) else "N/A"
                meas_label = f"L={target_arc_length_px * UM_PER_PX:.1f}µm  W={width_str}µm (ROI)"
                (meas_w, meas_h_px), _ = cv2.getTextSize(
                    meas_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                )
                meas_x = w - meas_w - 8
                meas_y = 22
                if draw_annotation_text:
                    cv2.putText(
                        annotated, meas_label, (meas_x, meas_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                    )
                    cv2.putText(
                        annotated, meas_label, (meas_x, meas_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                    )
                # Original-image-scale measurements (→ µm)
                if not np.isnan(target_arc_length_orig_px):
                    orig_width_str = (
                        f"{target_width_orig_px * UM_PER_PX:.1f}"
                        if not np.isnan(target_width_orig_px) else "N/A"
                    )
                    meas_label_orig = (
                        f"L={target_arc_length_orig_px * UM_PER_PX:.1f}µm  W={orig_width_str}µm (orig)"
                    )
                    (meas_w2, _), _ = cv2.getTextSize(
                        meas_label_orig, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
                    )
                    meas_x2 = w - meas_w2 - 8
                    meas_y2 = meas_y + 20
                    if draw_annotation_text:
                        cv2.putText(
                            annotated, meas_label_orig, (meas_x2, meas_y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA,
                        )
                        cv2.putText(
                            annotated, meas_label_orig, (meas_x2, meas_y2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 100), 1, cv2.LINE_AA,
                        )

                # ── Legend ─────────────────────────────────────────────────
                if draw_annotation_text:
                    leg_x, leg_y = 8, 28
                    leg_line_len = 22
                    leg_gap      = 18
                    font         = cv2.FONT_HERSHEY_SIMPLEX
                    fscale       = 0.45
                    n_legend_rows = 4 if gap_ranges else 2
                    box_w, box_h  = 280, leg_gap * n_legend_rows + 2
                    sub = annotated[leg_y - 16 : leg_y + box_h, leg_x - 4 : leg_x + box_w]
                    if sub.shape[0] > 0 and sub.shape[1] > 0:
                        black = np.zeros_like(sub)
                        cv2.addWeighted(black, 0.45, sub, 0.55, 0, sub)
                        annotated[leg_y - 16 : leg_y + box_h, leg_x - 4 : leg_x + box_w] = sub
                    cv2.circle(annotated, (leg_x + leg_line_len // 2, leg_y), 2,
                               (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated, "point inside mask (used in width)",
                                (leg_x + leg_line_len + 4, leg_y + 4),
                                font, fscale, (255, 255, 255), 1, cv2.LINE_AA)
                    gy1 = leg_y + leg_gap
                    cv2.circle(annotated, (leg_x + leg_line_len // 2, gy1), 3,
                               (0, 0, 220), -1, lineType=cv2.LINE_AA)
                    cv2.putText(annotated, "point outside mask (excl. from width)",
                                (leg_x + leg_line_len + 4, gy1 + 4),
                                font, fscale, (0, 0, 220), 1, cv2.LINE_AA)
                    if gap_ranges:
                        gy2 = leg_y + leg_gap * 2
                        cv2.line(annotated, (leg_x, gy2), (leg_x + leg_line_len, gy2),
                                 (0, 255, 0), 2, lineType=cv2.LINE_AA)
                        cv2.putText(annotated, "body skeleton",
                                    (leg_x + leg_line_len + 4, gy2 + 4),
                                    font, fscale, (255, 255, 255), 1, cv2.LINE_AA)
                        gy3 = leg_y + leg_gap * 3
                        for dx in range(0, leg_line_len, 7):
                            x1 = leg_x + dx
                            x2 = min(leg_x + dx + 4, leg_x + leg_line_len)
                            cv2.line(annotated, (x1, gy3), (x2, gy3),
                                     (0, 165, 255), 2, lineType=cv2.LINE_AA)
                        cv2.putText(annotated, "stitched gap",
                                    (leg_x + leg_line_len + 4, gy3 + 4),
                                    font, fscale, (0, 165, 255), 1, cv2.LINE_AA)

            else:
                # Centerline failed — fall back to contour outline in red
                contours, _ = cv2.findContours(
                    mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(annotated, contours, -1, (0, 0, 255), 2)

        n_cl = int(not np.isnan(target_arc_length_px))
        if draw_annotation_text:
            cv2.putText(
                annotated,
                f"Annotations: {len(masks_full)}  |  Skeletonized: {n_cl} (closest to center)",
                (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
            )

        if save_annotated_images:
            cv2.imwrite(str(Path(output_dir) / img_path.name), annotated)
        if save_target_masks:
            _save_target_mask_original_with_warning(
                target_mask_roi,
                roi_catalog.get(img_path.stem, {}),
                img_path.stem,
                target_mask_dir / img_path.name,
            )
        if save_contour_skeleton_txt:
            _save_contour_skeleton_txt(
                target_mask_roi,
                centerline if not np.isnan(target_arc_length_px) else None,
                roi_catalog.get(img_path.stem, {}),
                img_path.stem,
                contour_skel_dir / (img_path.stem + ".txt"),
                contour_max_points=contour_max_points,
            )
        _row = _build_worm_sizes_row(
            filename=img_path.name,
            stem=img_path.stem,
            length_um=(
                target_arc_length_orig_px * UM_PER_PX
                if not np.isnan(target_arc_length_orig_px) else float("nan")
            ),
            width_um=(
                target_width_orig_px * UM_PER_PX
                if not np.isnan(target_width_orig_px) else float("nan")
            ),
            topology_warnings=_collect_topology_warnings(_gt_caught_warns),
        )
        worm_sizes.append(_row)
        _csv_writer.writerow(_row)

    # Restore the original warnings hook now that the loop is done.
    warnings.showwarning = _gt_prev_show

    _csv_fh.close()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SKELETONIZATION COMPLETE  [GROUND-TRUTH ANNOTATION MODE]")
    print("=" * 70)
    print(f"Total images              : {stats['total_images']}")
    print(f"Images with annotations   : {stats['images_with_annotations']}")
    print(f"Total annotations         : {stats['total_annotations']}")
    print(f"Successful centerlines    : {stats['successful_centerlines']}")
    if save_annotated_images:
        print(f"\n✓ Annotated images  → {output_dir}")
    if save_target_masks:
        print(f"✓ Target masks      → {target_mask_dir}")
    if save_contour_skeleton_txt:
        print(f"✓ Contour/skeleton  → {contour_skel_dir}")
    print(f"✓ Worm sizes        → {csv_path}")
    print("=" * 70 + "\n")

    return worm_sizes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── CLI entry point ────────────────────────────────────────────────────
    # When command-line arguments are provided, run YOLO skeletonization
    # directly from them.  When invoked with NO arguments, fall through to
    # the legacy hardcoded configuration below.
    _cli_parser = argparse.ArgumentParser(
        description="Skeletonize worm masks from YOLO segmentation predictions.",
        add_help=True,
    )
    _cli_parser.add_argument("--model-path", type=str, default=None,
                             help="Path to YOLO segmentation weights (.pt).")
    _cli_parser.add_argument("--image-dir", type=str, default=None,
                             help="Directory of ROI images to process.")
    _cli_parser.add_argument("--output-dir", type=str, default=None,
                             help="Output directory for skeleton images and CSV.")
    _cli_parser.add_argument("--roi-catalog-path", type=str, default=None,
                             help="Path to roi_catalog.json from detect_and_crop_rois.")
    _cli_parser.add_argument("--per-image-timeout-sec", type=float, default=60.0,
                             help="If > 0, skip any image whose processing exceeds this many "
                                  "seconds (POSIX only, via SIGALRM). Default 60s. Set to 0 to disable.")
    _cli_parser.add_argument("--measure-speed", action="store_true",
                             help="Enable inference-time benchmarking (opt-in). "
                                  "Default off — original behavior unchanged.")
    _cli_parser.add_argument("--speed-output-dir", type=str, default=None,
                             help="Where to write speed_segment_*.csv/json. "
                                  "Defaults to <output_dir>/speed.")
    _cli_parser.add_argument("--speed-num-batches", type=int, default=10,
                             help="Number of batches (trials) for mean/SD. Default 10.")
    _cli_parser.add_argument("--speed-warmup", type=int, default=5,
                             help="Warmup predict() calls excluded from samples. Default 5.")
    _cli_parser.add_argument("--speed-device", type=str, default="auto",
                             help="Device label recorded with speed output: auto|cpu|cuda.")
    _cli_parser.add_argument("--device", type=str, default="",
                             help="Inference device for YOLO: '' (auto), 'cpu', '0', etc.")

    _cli_args = _cli_parser.parse_args()

    # If the three required paths are given via CLI, override the hardcoded
    # YOLO-mode paths below so the same code path is used without duplication.
    _cli_override = (
        _cli_args.model_path and _cli_args.image_dir and _cli_args.output_dir
    )

    # ── Legacy hardcoded configuration (no CLI args) ───────────────────────
    def _as_list(value):
        """Return *value* as a list; scalars become single-item lists."""
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _discover_gt_dataset_dirs(
        gt_root_dir: str,
        split_name: str = "train",
        output_folder_name: str = "skeleton",
        include_subsets: list[str] | None = None,
    ) -> tuple[list[str], list[str], list[str], list[str | None]]:
        """
        Auto-discover GT dataset triplets under a root folder.

        Expected layout per subset:
            <gt_root>/<subset>/<split_name>/images
            <gt_root>/<subset>/<split_name>/labels
            <gt_root>/<subset>/<split_name>/roi_catalog.json  (optional)
        """
        root = Path(gt_root_dir)
        if not root.exists():
            print(f"❌ GT root directory not found: {gt_root_dir}")
            raise SystemExit(1)

        include_set = set(include_subsets or [])
        subset_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

        image_dirs: list[str] = []
        labels_dirs: list[str] = []
        output_dirs: list[str] = []
        roi_catalog_paths: list[str | None] = []

        for subset_dir in subset_dirs:
            subset_name = subset_dir.name
            if include_set and subset_name not in include_set:
                continue

            split_dir = subset_dir / split_name
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"
            if not (img_dir.exists() and lbl_dir.exists()):
                continue

            image_dirs.append(str(img_dir))
            labels_dirs.append(str(lbl_dir))
            output_dirs.append(str(split_dir / output_folder_name))

            roi_path = split_dir / "roi_catalog.json"
            roi_catalog_paths.append(str(roi_path) if roi_path.exists() else None)

        if not image_dirs:
            print(
                f"❌ No GT datasets discovered under {gt_root_dir} for split '{split_name}'. "
                "Expected <subset>/<split>/images and <subset>/<split>/labels."
            )
            raise SystemExit(1)

        return image_dirs, labels_dirs, output_dirs, roi_catalog_paths

    def _normalize_optional_list(value, n_expected: int, name: str) -> list:
        """
        Normalize optional config values (str | None | list) to a list of
        length *n_expected*. A single item is broadcast to all datasets.
        """
        items = _as_list(value)
        if len(items) == 1 and n_expected > 1:
            return items * n_expected
        if len(items) != n_expected:
            print(
                f"❌ {name} must have length 1 or match dataset count ({n_expected}); "
                f"got {len(items)}"
            )
            raise SystemExit(1)
        return items

    # ── Mode selection ─────────────────────────────────────────────────────
    # "yolo"  : run YOLO inference to obtain masks, then skeletonize
    # "gt"    : read ground-truth YOLO segmentation .txt labels to obtain
    #           masks, then skeletonize (no model needed)
    MODE = "yolo"   # ← change to "yolo" to use the inference pipeline

    # ── CLI path overrides ─────────────────────────────────────────────────
    # When paths are provided via command-line arguments, force YOLO mode
    # and override the hardcoded paths below.  All tuning parameters remain
    # the same as the legacy configuration.
    if _cli_override:
        MODE = "yolo"

    # ── Pixel → micron conversion ────────────────────────────────────────────
    # 1 pixel = (SENSOR_MM * 1000 / IMAGE_WIDTH_PX) µm
    # Adjust to match your camera sensor and objective magnification.
    #SENSOR_MM      = 6.58    # sensor size in mm
    #IMAGE_WIDTH_PX = 1983    # image width in pixels
    #UM_PER_PX      = SENSOR_MM * 1000 / IMAGE_WIDTH_PX
    UM_PER_PX = 3.2937 # data from cell profiler

    # ── Shared (skeleton) parameters ───────────────────────────────────────
    N_CENTERLINE_POINTS   = 200    # spline resolution (used only if SKELETON_PX_PER_POINT is None)
    SKELETON_PX_PER_POINT = 5.0   # px; 1 skeleton point per N px of arc length (None = fixed count)
    SMOOTH_FACTOR         = None   # None = auto; increase for a smoother curve
    SMOOTH_FACTOR_RATIO   = 1.0    # multiplier on len(skeleton_pixels): 0=exact interp, 0.5, 1.0=default, 5.0=heavy smooth
    CONTOUR_SMOOTH_SIGMA  = 6.0    # Gaussian sigma to round YOLO mask corners (0 = off)
    MAX_GAP_PX            = 110    # px; blobs farther apart than this will NOT be stitched
    MAX_JOINT_ANGLE_DEG   = 110    # deg; gap connections more skewed than this are rejected
    TANGENT_PATH_LEN_PX   = 50     # arc length (px) over which the tangent direction is estimated
    MIN_FRAGMENT_PIXELS   = 200    # px²; connected components smaller than this are ignored
    HOLE_FILL_RATIO       = 0.05   # fraction; hole fill adds more than this → treat as ring
    HOLE_CIRC_WEIGHT      = 0.75   # 0=pure area, 1=pure circularity for keeping loop hole
    RING_DILATION_RADIUS  = 5      # px; dilate mask before is_ring detection and hole analysis (0 = off)
    DIR_VECTOR_LEN_PX     = 50     # arc length (px) used for directional averaging at each junction seam
    MIN_BRIDGE_ARC_LEN_PX = 20     # Case 6 safety: if shorter parallel arc < this (px), skip scoring → use shorter arc as bridge
    GAP_INTERP_POINTS     = 12     # number of interpolated points inserted across each stitched gap

    # ── Output saving options ───────────────────────────────────────────────
    SAVE_ANNOTATED_IMAGES = False    # set False to skip saving skeleton-overlay images to output_dir
    SAVE_TARGET_MASKS     = False   # set False to skip exporting target_masks_original images
    SAVE_DEBUG_FRAGMENTS  = False   # set False to skip saving debug_fragments images
    SAVE_CONTOUR_SKELETON_TXT = True  # set True to export YOLO-style contour + skeleton .txt per ROI
    CONTOUR_MAX_POINTS    = 200    # max vertices per contour polygon in .txt export
    DRAW_ANNOTATION_TEXT  = False   # set False to suppress ALL text overlays (head label, L/W measurements, legend box, footer, "No Detection"/"No Annotation") on annotated images

    if MODE == "yolo":
        # ── YOLO inference mode configuration ──────────────────────────────

        MODEL_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt"

        # cluster test
        #MODEL_PATH = r"/scratch/eande106/ZihaoJohnLi/NemaSeg_Project/weights/worm_roi_seg/best.pt"
        #IMAGE_DIR  = r"/scratch/eande106/ZihaoJohnLi/datasets/Amanda/20260226_Cry_p001_p020_Lina/inference_rois/images"
        #OUTPUT_DIR = r"/scratch/eande106/ZihaoJohnLi/datasets/Amanda/20260226_Cry_p001_p020_Lina/skeleton_infoParsed"

        # Etta test
        #IMAGE_DIR  = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Etta_dataset\20260305_dauer_images\20260305_dauer_images\inference_rois\images"
        #OUTPUT_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Etta_dataset\20260305_dauer_images\20260305_dauer_images\skeleton"

        # Amanda test
        #MODEL_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt"
        #IMAGE_DIR  = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Amanda_dataset\inference_rois\images"
        #OUTPUT_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Amanda_dataset\skeleton_2"

        #MODEL_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt"
        #IMAGE_DIR  = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap_catalog\valid\images"
        #OUTPUT_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap_catalog\valid\skeleton_pred"

        #MODEL_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\runs\roi_yolo_segment_fix_overlap\roi_seg_train_fix_overlap\weights\best.pt"
        # Accept either:
        #   - a single string path, or
        #   - a list of paths to process multiple datasets in one run.
        IMAGE_DIR  = [
            #r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Pipeline\inference_rois\images",
            r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\NeSg_perform\inference_rois\images"
        ]
        OUTPUT_DIR = [
            #r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Pipeline\Skeleton_infoParsed",
            #r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\NeSg_perform\skeleton_warned"
            r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\NeSg_perform\skeleton_text_disabled"
        ]

        CONF_THRESHOLD = 0.25      # YOLO detection confidence threshold

        # Optional: path to roi_catalog.json produced by detect_and_crop_rois.py.

        # ROI_CATALOG_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Etta_dataset\20260305_dauer_images\20260305_dauer_images\inference_rois\roi_catalog.json"
        # ROI_CATALOG_PATH = r"/scratch/eande106/ZihaoJohnLi/datasets/Amanda/20260226_Cry_p001_p020_Lina/inference_rois/roi_catalog.json"
        # ROI_CATALOG_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap_catalog\valid\roi_catalog.json"
        # Can be None / single path / list of paths (broadcast if length=1).
        ROI_CATALOG_PATH = [
            #r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Pipeline\inference_rois\roi_catalog.json",
            r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\NeSg_perform\inference_rois\roi_catalog.json"
        ]
        # ── End YOLO mode configuration ─────────────────────────────────────

        # Apply CLI path overrides (if provided)
        if _cli_override:
            MODEL_PATH = _cli_args.model_path
            IMAGE_DIR  = _cli_args.image_dir
            OUTPUT_DIR = _cli_args.output_dir
            ROI_CATALOG_PATH = _cli_args.roi_catalog_path

        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found: {MODEL_PATH}")
            raise SystemExit(1)
        image_dirs = _as_list(IMAGE_DIR)
        output_dirs = _as_list(OUTPUT_DIR)
        if len(image_dirs) != len(output_dirs):
            print(
                "❌ IMAGE_DIR and OUTPUT_DIR must have the same length; "
                f"got {len(image_dirs)} and {len(output_dirs)}"
            )
            raise SystemExit(1)

        roi_catalog_paths = _normalize_optional_list(
            ROI_CATALOG_PATH, len(image_dirs), "ROI_CATALOG_PATH"
        )

        worm_sizes = []
        completed_output_dirs = []
        for run_idx, (img_dir, out_dir, roi_path) in enumerate(
            zip(image_dirs, output_dirs, roi_catalog_paths), start=1
        ):
            if not os.path.exists(img_dir):
                print(f"❌ Image directory not found: {img_dir}")
                raise SystemExit(1)

            roi_arg = roi_path if (roi_path and os.path.exists(roi_path)) else None
            if roi_path and roi_arg is None:
                print(f"⚠ ROI catalog not found for run {run_idx}: {roi_path}; proceeding without it.")

            print(f"\n[YOLO run {run_idx}/{len(image_dirs)}] {img_dir} → {out_dir}\n")
            run_rows = skeletonize_worm_predictions(
                model_path=MODEL_PATH,
                image_dir=img_dir,
                output_dir=out_dir,
                conf_threshold=CONF_THRESHOLD,
                n_centerline_points=N_CENTERLINE_POINTS,
                smooth_factor=SMOOTH_FACTOR,
                contour_smooth_sigma=CONTOUR_SMOOTH_SIGMA,
                max_gap_px=MAX_GAP_PX,
                max_joint_angle_deg=MAX_JOINT_ANGLE_DEG,
                tangent_path_len_px=TANGENT_PATH_LEN_PX,
                min_fragment_pixels=MIN_FRAGMENT_PIXELS,
                hole_fill_ratio_threshold=HOLE_FILL_RATIO,
                hole_keep_circularity_weight=HOLE_CIRC_WEIGHT,
                ring_dilation_radius=RING_DILATION_RADIUS,
                dir_vector_len_px=DIR_VECTOR_LEN_PX,
                min_bridge_arc_len_px=MIN_BRIDGE_ARC_LEN_PX,
                gap_interp_points=GAP_INTERP_POINTS,
                smooth_factor_ratio=SMOOTH_FACTOR_RATIO,
                skeleton_px_per_point=SKELETON_PX_PER_POINT,
                debug_fragments=SAVE_DEBUG_FRAGMENTS,
                roi_catalog_path=roi_arg,
                save_annotated_images=SAVE_ANNOTATED_IMAGES,
                save_target_masks=SAVE_TARGET_MASKS,
                save_contour_skeleton_txt=SAVE_CONTOUR_SKELETON_TXT,
                contour_max_points=CONTOUR_MAX_POINTS,
                draw_annotation_text=DRAW_ANNOTATION_TEXT,
                per_image_timeout_sec=_cli_args.per_image_timeout_sec,
                measure_speed=_cli_args.measure_speed,
                speed_output_dir=_cli_args.speed_output_dir,
                speed_num_batches=_cli_args.speed_num_batches,
                speed_warmup=_cli_args.speed_warmup,
                speed_device=_cli_args.speed_device,
                device=_cli_args.device,
            )
            worm_sizes.extend(run_rows)
            completed_output_dirs.append(out_dir)

    elif MODE == "gt":
        # ── Ground-truth annotation mode configuration ──────────────────────

        # Optional auto-discovery mode to avoid manually typing arrays.
        # Set GT_DATA_ROOT to your GT_rois directory and leave
        # GT_USE_AUTO_DISCOVERY=True to populate all arrays automatically.
        GT_USE_AUTO_DISCOVERY = True
        GT_DATA_ROOT = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\Perform_test\GT_rois"
        GT_SPLIT_NAME = "train"
        GT_OUTPUT_FOLDER_NAME = "skeleton_warned"
        # Empty list = include every subset under GT_DATA_ROOT.
        GT_SUBSETS: list[str] = []

        # Class-wise performance task
        # Accept either single strings or arrays for batch processing.
        GT_IMAGE_DIR  = []
        GT_LABELS_DIR = []
        GT_OUTPUT_DIR = []

        #GT_IMAGE_DIR  = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap_catalog\valid\images"
        #GT_LABELS_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap_catalog\valid\labels"
        #GT_OUTPUT_DIR = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap_catalog\valid\skeleton_gt"

        # Optional: path to roi_catalog.json for scale conversion.
        # GT_ROI_CATALOG_PATH = r"C:\Users\jl200\Dropbox\JHU_2026_spring\NemaSeg\datasets\WormBodyROI_fix_overlap_catalog\valid\roi_catalog.json"
        # Can be None / single path / list of paths (broadcast if length=1).
        GT_ROI_CATALOG_PATH = []

        # Optional: human-readable class names (mapping int class_id → name).
        # Set to None to use generic "class_<id>" labels.
        GT_CLASS_NAMES: dict[int, str] | None = {0: "worm"}
        # ── End GT mode configuration ────────────────────────────────────────

        if GT_USE_AUTO_DISCOVERY:
            (
                GT_IMAGE_DIR,
                GT_LABELS_DIR,
                GT_OUTPUT_DIR,
                GT_ROI_CATALOG_PATH,
            ) = _discover_gt_dataset_dirs(
                gt_root_dir=GT_DATA_ROOT,
                split_name=GT_SPLIT_NAME,
                output_folder_name=GT_OUTPUT_FOLDER_NAME,
                include_subsets=GT_SUBSETS,
            )
            print(
                f"Auto-discovered {len(GT_IMAGE_DIR)} GT dataset(s) under "
                f"{GT_DATA_ROOT} (split='{GT_SPLIT_NAME}')."
            )
            for i, (img_d, lbl_d, out_d) in enumerate(
                zip(GT_IMAGE_DIR, GT_LABELS_DIR, GT_OUTPUT_DIR), start=1
            ):
                print(f"  [{i}] {img_d} | {lbl_d} -> {out_d}")

        gt_image_dirs = _as_list(GT_IMAGE_DIR)
        gt_labels_dirs = _as_list(GT_LABELS_DIR)
        gt_output_dirs = _as_list(GT_OUTPUT_DIR)
        if not (
            len(gt_image_dirs) == len(gt_labels_dirs) == len(gt_output_dirs)
        ):
            print(
                "❌ GT_IMAGE_DIR, GT_LABELS_DIR, and GT_OUTPUT_DIR must have the "
                "same length; got "
                f"{len(gt_image_dirs)}, {len(gt_labels_dirs)}, {len(gt_output_dirs)}"
            )
            raise SystemExit(1)

        gt_roi_catalog_paths = _normalize_optional_list(
            GT_ROI_CATALOG_PATH, len(gt_image_dirs), "GT_ROI_CATALOG_PATH"
        )

        worm_sizes = []
        completed_output_dirs = []
        for run_idx, (img_dir, lbl_dir, out_dir, roi_path) in enumerate(
            zip(gt_image_dirs, gt_labels_dirs, gt_output_dirs, gt_roi_catalog_paths), start=1
        ):
            if not os.path.exists(img_dir):
                print(f"❌ Image directory not found: {img_dir}")
                raise SystemExit(1)
            if not os.path.exists(lbl_dir):
                print(f"❌ Labels directory not found: {lbl_dir}")
                raise SystemExit(1)

            roi_arg = roi_path if (roi_path and os.path.exists(roi_path)) else None
            if roi_path and roi_arg is None:
                print(f"⚠ ROI catalog not found for run {run_idx}: {roi_path}; proceeding without it.")

            print(f"\n[GT run {run_idx}/{len(gt_image_dirs)}] {img_dir} + {lbl_dir} → {out_dir}\n")
            run_rows = skeletonize_gt_annotations(
                image_dir=img_dir,
                labels_dir=lbl_dir,
                output_dir=out_dir,
                class_names=GT_CLASS_NAMES,
                n_centerline_points=N_CENTERLINE_POINTS,
                smooth_factor=SMOOTH_FACTOR,
                contour_smooth_sigma=CONTOUR_SMOOTH_SIGMA,
                max_gap_px=MAX_GAP_PX,
                max_joint_angle_deg=MAX_JOINT_ANGLE_DEG,
                tangent_path_len_px=TANGENT_PATH_LEN_PX,
                min_fragment_pixels=MIN_FRAGMENT_PIXELS,
                hole_fill_ratio_threshold=HOLE_FILL_RATIO,
                hole_keep_circularity_weight=HOLE_CIRC_WEIGHT,
                ring_dilation_radius=RING_DILATION_RADIUS,
                dir_vector_len_px=DIR_VECTOR_LEN_PX,
                min_bridge_arc_len_px=MIN_BRIDGE_ARC_LEN_PX,
                gap_interp_points=GAP_INTERP_POINTS,
                smooth_factor_ratio=SMOOTH_FACTOR_RATIO,
                skeleton_px_per_point=SKELETON_PX_PER_POINT,
                debug_fragments=SAVE_DEBUG_FRAGMENTS,
                roi_catalog_path=roi_arg,
                save_annotated_images=SAVE_ANNOTATED_IMAGES,
                save_target_masks=SAVE_TARGET_MASKS,
                save_contour_skeleton_txt=SAVE_CONTOUR_SKELETON_TXT,
                contour_max_points=CONTOUR_MAX_POINTS,
                draw_annotation_text=DRAW_ANNOTATION_TEXT,
            )
            worm_sizes.extend(run_rows)
            completed_output_dirs.append(out_dir)

    else:
        print(f"❌ Unknown MODE: {MODE!r}. Choose 'yolo' or 'gt'.")
        raise SystemExit(1)

    # CSV is already written inside each mode-specific function.
    if len(completed_output_dirs) == 1:
        csv_path = Path(completed_output_dirs[0]) / "worm_sizes.csv"
        print(f"✓ Worm sizes saved → {csv_path}")
    else:
        print("✓ Worm sizes saved:")
        for out_dir in completed_output_dirs:
            print(f"  - {Path(out_dir) / 'worm_sizes.csv'}")