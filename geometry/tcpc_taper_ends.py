#!/usr/bin/env python3
"""
tcpc_taper_ends.py

Standalone (non-CLI) script to:
  0) Optionally uncap all 4 ends (if the input is capped)
  1) Label the 4 ends with indices 0,1,2,3 (deterministic order)
  2) Let you choose a subset of ends to modify
  3) For each selected end, "prolong" it along its outward normal (≈ +/-Y)
     for a given length, while continuously scaling the rim down to a factor
     (1.0 = unchanged, 0.0 = fully closed)
     - Taper profile can be either:
         * linear  : straight frustum/cone (previous behavior)
         * smooth  : smoothly-eased radius change along the length (vessel-like)
  4) Ensure ALL 4 ends are capped in the output
  5) Export the result as STL

Dependencies: numpy, trimesh
Install: pip install numpy trimesh

Notes
-----
- This script assumes the 4 ends are planar and approximately parallel to the XZ plane,
  i.e. their outward normals are aligned with +/-Y (as described).
- End labels are assigned deterministically by sorting end-centroids by (y, x, z).
  You can change that with LABEL_SORT_* constants below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh


# =============================================================================
# USER CONFIG (edit this section)
# =============================================================================

# Input / output
INPUT_STL: str = "fantom.stl"
OUTPUT_STL: str = "fantom_tapered.stl"

# If True and the input has caps, remove caps for ALL ends and rebuild caps.
# If False, remove caps ONLY for ends you modify (WORKING_ENDS), leaving other original caps intact.
UNCAP_ALL_ENDS: bool = False

# Which ends you want to modify (subset of [0, 1, 2, 3]).
# Ends NOT listed here will simply be capped (either preserved or rebuilt as needed).
WORKING_ENDS: List[int] = [0, 1, 2]

# Taper parameters for each working end:
#   scale: 1.0 = no taper, 0.0 = completely closed (cone to a point)
#   length: extrusion distance along outward normal of that end
#
# Units are the same as your STL units (often mm).
TAPER_PARAMS: Dict[int, Tuple[float, float]] = {
    0: (0.5, 35.0),  # example: extend end 0 by 10 units without taper
    1: (1.0, 35.0),  # example: close end 1 completely over 15 units
    2: (0.4, 35.0),  # example: close end 1 completely over 15 units
}

# Taper profile control
#
# "linear":  matches the previous behavior (single frustum/cone segment)
# "smooth":  uses a smooth easing curve along the length (piecewise loft with many rings)
#
# You can set a global default and optionally override per end.
TAPER_PROFILE_DEFAULT: str = "smooth"  # "linear" or "smooth"
TAPER_PROFILE_PER_END: Dict[int, str] = {
    # 3: "smooth",
}

# Only used for the "smooth" profile: number of segments along the length.
# Larger => smoother but more triangles.
SMOOTH_SEGMENTS: int = 24

# End detection settings
EXPECTED_ENDS: int = 4
AXIS: np.ndarray = np.array([0.0, 1.0, 0.0])  # "Y axis"
NORMAL_THRESHOLD: float = 0.95  # how close cap normals must be to +/-AXIS to be considered end-caps

# End labeling: sort end centroids by these axes to get labels 0..3.
# Default: (y, x, z) ascending.
LABEL_SORT_AXES: Tuple[str, str, str] = ("y", "x", "z")
LABEL_SORT_DIRECTIONS: Dict[str, float] = {"x": 1.0, "y": 1.0, "z": 1.0}
# Example: to label the top end as 0 instead of 3:
# LABEL_SORT_DIRECTIONS = {"x": 1.0, "y": -1.0, "z": 1.0}

# Geometry tolerances
CAP_TRI_TOL: float = 1e-10  # ear-clipping tolerance
EPS: float = 1e-9


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def _normalize_profile_name(name: str) -> str:
    """Map user-friendly strings to internal taper profile names."""
    n = (name or "").strip().lower()

    # Linear taper (frustum/cone)
    if n in {"linear", "lin", "l"}:
        return "linear"

    # Smooth vessel-like taper (multi-ring loft + easing)
    if n in {"smooth", "curved", "curve", "vessel", "eased", "ease"}:
        return "smooth"

    raise ValueError(
        f"Unknown taper profile '{name}'. Supported: 'linear' or 'smooth'."
    )


def _easing_01(t: float, profile: str) -> float:
    """Return g(t) in [0,1] for t in [0,1], controlling progression of radius change."""
    t = _clamp(t, 0.0, 1.0)
    if profile == "linear":
        return t

    if profile == "smooth":
        # Quintic smootherstep: 6t^5 - 15t^4 + 10t^3
        # C2-continuous: zero first and second derivatives at endpoints.
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    # Should never happen if normalized.
    return t


# =============================================================================
# INTERNALS
# =============================================================================

@dataclass
class EndInfo:
    label: int = -1
    loop_vertices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=int))
    outward_normal: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=float))
    centroid: np.ndarray = field(default_factory=lambda: np.zeros((3,), dtype=float))
    has_cap: bool = False
    cap_faces: Optional[np.ndarray] = None       # face indices in the ORIGINAL mesh that form the cap
    cap_facet_index: Optional[int] = None
    triangles: Optional[np.ndarray] = None       # triangulation indices into loop order
    match_fraction: Optional[float] = None       # debug: only for open meshes


def _boundary_edges(mesh: trimesh.Trimesh) -> np.ndarray:
    """
    Return boundary edges as (m,2) vertex indices (unique edges referenced by only one face).
    """
    inv = mesh.edges_unique_inverse
    counts = np.bincount(inv)
    boundary_unique = np.where(counts == 1)[0]
    return mesh.edges_unique[boundary_unique]


def _order_loop_from_edges(edges: np.ndarray) -> np.ndarray:
    """
    Given undirected edges (m,2) that form ONE closed loop, return ordered vertex indices (n,).
    """
    adj: Dict[int, List[int]] = {}
    for a, b in edges:
        a = int(a); b = int(b)
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    start = min(adj.keys())
    loop = [start]
    prev = None
    curr = start

    for _ in range(len(adj) + 5):
        nbrs = adj[curr]
        if len(nbrs) < 1:
            raise RuntimeError("Boundary loop has a dead end. Mesh may be non-manifold.")
        if prev is None:
            nxt = nbrs[0]
        else:
            # pick the neighbor that's not the previous vertex
            nxt = nbrs[0] if nbrs[0] != prev else nbrs[1]
        if nxt == start:
            break
        loop.append(nxt)
        prev, curr = curr, nxt
    else:
        raise RuntimeError("Failed to order boundary loop (iteration guard triggered).")

    return np.array(loop, dtype=int)


def _polygon_normal_newell(points: np.ndarray) -> np.ndarray:
    """
    Compute polygon normal using Newell's method. Points are assumed ordered.
    """
    n = np.zeros(3, dtype=float)
    for i in range(len(points)):
        p0 = points[i]
        p1 = points[(i + 1) % len(points)]
        n[0] += (p0[1] - p1[1]) * (p0[2] + p1[2])
        n[1] += (p0[2] - p1[2]) * (p0[0] + p1[0])
        n[2] += (p0[0] - p1[0]) * (p0[1] + p1[1])
    norm = float(np.linalg.norm(n))
    if norm < EPS:
        return n
    return n / norm


def _build_boundary_oriented_edge_map(mesh: trimesh.Trimesh) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    For boundary unique edges, map undirected edge key (min,max) -> oriented edge (a,b)
    as it appears in the only adjacent face.

    Used to orient open boundary loops consistently with the outward-facing cap normal.
    """
    inv = mesh.edges_unique_inverse
    counts = np.bincount(inv)
    edges_oriented = mesh.edges  # oriented per face-edge (v0,v1), (v1,v2), (v2,v0)

    undirected_to_oriented: Dict[Tuple[int, int], Tuple[int, int]] = {}
    for fe_i, u in enumerate(inv):
        if counts[u] == 1:
            a, b = map(int, edges_oriented[fe_i])
            key = (a, b) if a < b else (b, a)
            undirected_to_oriented[key] = (a, b)

    return undirected_to_oriented


def _loop_match_fraction(loop: np.ndarray, und_to_or: Dict[Tuple[int, int], Tuple[int, int]]) -> float:
    """
    Compare the loop's directed edges against the oriented boundary edges from adjacent faces.
    If most edges match, the loop direction matches adjacent faces (and should be reversed
    to match the *cap* outward orientation).
    """
    n = len(loop)
    if n == 0:
        return 0.0
    match = 0
    for i in range(n):
        a = int(loop[i])
        b = int(loop[(i + 1) % n])
        key = (a, b) if a < b else (b, a)
        if und_to_or.get(key) == (a, b):
            match += 1
    return match / n


def _plane_basis_from_normal(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orthonormal (u,v) basis vectors spanning the plane whose normal is n.
    """
    n = n / (np.linalg.norm(n) + EPS)
    a = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 0.0, 1.0])
    u = np.cross(n, a)
    u = u / (np.linalg.norm(u) + EPS)
    v = np.cross(n, u)
    return u, v


def _earclip_triangulation(poly2d: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """
    Simple ear-clipping triangulation for a simple polygon.

    poly2d: (n,2) vertices in order (CW or CCW)

    Returns triangles as indices into poly2d (m,3), where m = n-2 for success.
    """
    n = len(poly2d)
    if n < 3:
        return np.zeros((0, 3), dtype=int)

    x = poly2d[:, 0]
    y = poly2d[:, 1]
    area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    orientation = 1.0 if area > 0.0 else -1.0  # +1 => CCW, -1 => CW

    def cross_z(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ab = b - a
        ac = c - a
        return float(ab[0] * ac[1] - ab[1] * ac[0])

    def is_convex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        return cross_z(a, b, c) * orientation > tol

    def point_in_tri(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        # barycentric
        v0 = c - a
        v1 = b - a
        v2 = p - a
        dot00 = float(np.dot(v0, v0))
        dot01 = float(np.dot(v0, v1))
        dot02 = float(np.dot(v0, v2))
        dot11 = float(np.dot(v1, v1))
        dot12 = float(np.dot(v1, v2))
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < tol:
            return False
        inv = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        return (u >= -tol) and (v >= -tol) and (u + v <= 1.0 + tol)

    indices = list(range(n))
    triangles: List[List[int]] = []

    guard = 0
    while len(indices) > 3 and guard < n * n:
        m = len(indices)
        ear_found = False

        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_curr = indices[i]
            i_next = indices[(i + 1) % m]

            a = poly2d[i_prev]
            b = poly2d[i_curr]
            c = poly2d[i_next]

            if not is_convex(a, b, c):
                continue

            ok = True
            for j in range(m):
                jj = indices[j]
                if jj in (i_prev, i_curr, i_next):
                    continue
                if point_in_tri(poly2d[jj], a, b, c):
                    ok = False
                    break

            if not ok:
                continue

            triangles.append([i_prev, i_curr, i_next])
            del indices[i]
            ear_found = True
            break

        if not ear_found:
            break

        guard += 1

    if len(indices) == 3:
        triangles.append(indices)

    return np.array(triangles, dtype=int)


def _get_triangulation(mesh: trimesh.Trimesh, end: EndInfo, tol: float) -> np.ndarray:
    """
    Triangulate an end loop into (n-2) triangles (using ear-clipping), caching results in end.triangles.
    """
    if end.triangles is not None:
        return end.triangles

    pts = mesh.vertices[end.loop_vertices]
    n = end.outward_normal / (np.linalg.norm(end.outward_normal) + EPS)
    u, v = _plane_basis_from_normal(n)

    poly2d = np.column_stack((pts.dot(u), pts.dot(v)))
    tris = _earclip_triangulation(poly2d, tol=tol)

    expected = len(end.loop_vertices) - 2
    if len(tris) != expected:
        # retry with a looser tolerance (sometimes helps with near-colinear points)
        tris2 = _earclip_triangulation(poly2d, tol=max(tol * 0.01, 1e-14))
        if len(tris2) == expected:
            tris = tris2
        else:
            raise RuntimeError(
                f"Ear clipping failed for end {end.label}: expected {expected} triangles, got {len(tris)}.\n"
                f"Try increasing CAP_TRI_TOL, or simplify/remesh the STL near the ends."
            )

    end.triangles = tris
    return tris


def _detect_ends(
    mesh: trimesh.Trimesh,
    expected_ends: int,
    axis: np.ndarray,
    normal_threshold: float,
) -> List[EndInfo]:
    """
    Detect 4 ends either from open boundaries (if mesh has boundary edges)
    or from planar end-caps (if mesh is watertight).
    """
    axis = axis / (np.linalg.norm(axis) + EPS)

    b_edges = _boundary_edges(mesh)
    if len(b_edges) > 0:
        # ------------------------------------------------------------
        # OPEN MESH: ends are boundary loops
        # ------------------------------------------------------------
        comps = trimesh.graph.connected_components(b_edges)
        comps_sorted = sorted(comps, key=lambda c: len(c), reverse=True)
        comps_take = comps_sorted[:expected_ends]

        und_to_or = _build_boundary_oriented_edge_map(mesh)

        ends: List[EndInfo] = []
        for comp in comps_take:
            edges_comp = b_edges[np.isin(b_edges[:, 0], comp) & np.isin(b_edges[:, 1], comp)]
            loop = _order_loop_from_edges(edges_comp)

            # Orient loop so it matches the outward-facing cap normal that would close the hole
            frac = _loop_match_fraction(loop, und_to_or)
            if frac > 0.5:
                loop = loop[::-1]

            pts = mesh.vertices[loop]
            n = _polygon_normal_newell(pts)
            centroid = pts.mean(axis=0)

            ends.append(
                EndInfo(
                    loop_vertices=loop,
                    outward_normal=n,
                    centroid=centroid,
                    has_cap=False,
                    cap_faces=None,
                    match_fraction=frac,
                )
            )

        return ends

    # ------------------------------------------------------------
    # CAPPED / WATERTIGHT MESH: detect planar end caps as facets
    # ------------------------------------------------------------
    facets_normal = mesh.facets_normal
    facets_area = mesh.facets_area

    # Candidate planar facets whose normal is close to +/-axis
    dot = np.abs(facets_normal.dot(axis))
    candidates = np.where(dot > normal_threshold)[0]
    if len(candidates) < expected_ends:
        raise RuntimeError(
            f"Could not find {expected_ends} end-cap facets aligned with axis. "
            f"Found only {len(candidates)} candidates at threshold {normal_threshold}."
        )

    # Pick the largest ones by area
    candidates_sorted = candidates[np.argsort(facets_area[candidates])[::-1]]
    selected = candidates_sorted[:expected_ends]

    ends: List[EndInfo] = []
    for fidx in selected:
        edges = mesh.facets_boundary[fidx]
        loop = _order_loop_from_edges(edges)

        pts = mesh.vertices[loop]
        n_loop = _polygon_normal_newell(pts)
        n_cap = facets_normal[fidx] / (np.linalg.norm(facets_normal[fidx]) + EPS)

        # Ensure loop ordering produces the same normal direction as the cap facet
        if float(np.dot(n_loop, n_cap)) < 0.0:
            loop = loop[::-1]
            pts = mesh.vertices[loop]
            n_loop = _polygon_normal_newell(pts)

        centroid = pts.mean(axis=0)
        cap_faces = np.array(mesh.facets[fidx], dtype=int)

        ends.append(
            EndInfo(
                loop_vertices=loop,
                outward_normal=n_cap,
                centroid=centroid,
                has_cap=True,
                cap_faces=cap_faces,
                cap_facet_index=int(fidx),
            )
        )

    return ends


def _assign_labels(
    ends: List[EndInfo],
    sort_axes: Tuple[str, str, str],
    directions: Dict[str, float],
) -> List[EndInfo]:
    def key(end: EndInfo) -> Tuple[float, float, float]:
        coords = {"x": float(end.centroid[0]), "y": float(end.centroid[1]), "z": float(end.centroid[2])}
        return tuple(float(directions[a]) * coords[a] for a in sort_axes)

    ends_sorted = sorted(ends, key=key)
    for i, end in enumerate(ends_sorted):
        end.label = i

    # return in label order
    return sorted(ends_sorted, key=lambda e: e.label)


def _apply_taper_and_cap(
    mesh: trimesh.Trimesh,
    ends: List[EndInfo],
    working_ends: List[int],
    taper_params: Dict[int, Tuple[float, float]],
    profile_default: str,
    profile_per_end: Dict[int, str],
    smooth_segments: int,
    uncap_all: bool,
    cap_tri_tol: float,
) -> trimesh.Trimesh:
    """
    Build a new mesh with selected end tapers and all ends capped.
    """
    # --- Remove selected caps (faces) if present ---
    faces = mesh.faces.copy()
    keep = np.ones(len(faces), dtype=bool)

    caps_removed = set()
    for end in ends:
        if end.has_cap and (uncap_all or (end.label in working_ends)):
            keep[end.cap_faces] = False
            caps_removed.add(end.label)

    faces_base = faces[keep]

    vertices_blocks = [mesh.vertices.copy()]
    faces_blocks = [faces_base.copy()]
    v_offset = len(vertices_blocks[0])

    # --- Add geometry end-by-end ---
    for end in ends:
        label = end.label
        base_idx = end.loop_vertices
        base_pts = mesh.vertices[base_idx]
        n_out = end.outward_normal / (np.linalg.norm(end.outward_normal) + EPS)

        if label in working_ends:
            if label not in taper_params:
                raise ValueError(f"Missing TAPER_PARAMS entry for working end {label}.")

            scale, length = taper_params[label]
            scale = float(scale)
            length = float(length)

            # clamp inputs to safe ranges
            scale = _clamp(scale, 0.0, 1.0)
            length = max(0.0, length)

            profile_raw = profile_per_end.get(label, profile_default)
            profile = _normalize_profile_name(profile_raw)
            n_segments = 1 if profile == "linear" else max(2, int(smooth_segments))

            if length > EPS:
                center = base_pts.mean(axis=0)
                delta = base_pts - center

                # ensure taper scaling happens in the end plane (remove normal component)
                delta_plane = delta - np.outer(delta.dot(n_out), n_out)

                # Build a loft made of multiple rings if "smooth".
                # For "linear", this degenerates to exactly one ring at t=1.
                base_ring_idx = base_idx
                prev_ring_idx = base_ring_idx

                # If we close to a point, end with an apex instead of a degenerate final ring.
                close_to_point = scale <= EPS

                # For a point closure we create rings up to t=(n_segments-1)/n_segments,
                # then connect that last ring to the apex at t=1.
                last_ring_step = n_segments - 1 if close_to_point else n_segments

                n_loop = len(base_idx)
                if n_loop < 3:
                    raise RuntimeError(f"End {label} loop has <3 vertices; cannot taper.")

                for step in range(1, last_ring_step + 1):
                    t = step / float(n_segments)
                    g = _easing_01(t, profile)
                    s_t = 1.0 + (scale - 1.0) * g

                    ring_pts = center + s_t * delta_plane + n_out * (length * t)
                    ring_idx = np.arange(v_offset, v_offset + n_loop, dtype=int)
                    vertices_blocks.append(ring_pts)
                    v_offset += n_loop

                    # Connect prev ring -> this ring
                    side = np.zeros((n_loop * 2, 3), dtype=int)
                    for i in range(n_loop):
                        j = (i + 1) % n_loop
                        side[2 * i] = [prev_ring_idx[i], prev_ring_idx[j], ring_idx[j]]
                        side[2 * i + 1] = [prev_ring_idx[i], ring_idx[j], ring_idx[i]]
                    faces_blocks.append(side)

                    prev_ring_idx = ring_idx

                if close_to_point:
                    # Apex at t=1
                    tip_point = center + n_out * length
                    tip_idx = int(v_offset)
                    vertices_blocks.append(tip_point.reshape(1, 3))
                    v_offset += 1

                    side = np.zeros((n_loop, 3), dtype=int)
                    for i in range(n_loop):
                        j = (i + 1) % n_loop
                        side[i] = [prev_ring_idx[i], prev_ring_idx[j], tip_idx]
                    faces_blocks.append(side)
                    # no cap needed at a point
                else:
                    # Cap the final ring
                    tris = _get_triangulation(mesh, end, tol=cap_tri_tol)
                    faces_blocks.append(prev_ring_idx[tris])

                # This end is now capped at the tip; do NOT cap at the base.
                continue

            # length == 0 -> treat as "no modification"; fall through to base-cap logic below.

        # --- If we didn't modify this end, ensure it is capped at the base if needed ---
        need_cap = (not end.has_cap) or (label in caps_removed)
        if need_cap:
            tris = _get_triangulation(mesh, end, tol=cap_tri_tol)
            faces_blocks.append(base_idx[tris])

    # --- Assemble final mesh ---
    vertices = np.vstack(vertices_blocks)
    faces_out = np.vstack(faces_blocks)

    out = trimesh.Trimesh(vertices=vertices, faces=faces_out, process=False)

    # Clean up (helps make STL more robust for downstream tools)
    out.update_faces(out.unique_faces())
    if hasattr(out, "remove_degenerate_faces"):
        out.remove_degenerate_faces()
    else:
        out.update_faces(out.nondegenerate_faces())
    out.remove_unreferenced_vertices()
    out.merge_vertices()

    return out


def main() -> None:
    # Basic validation of config
    for i in WORKING_ENDS:
        if i not in (0, 1, 2, 3):
            raise ValueError(f"WORKING_ENDS contains invalid end index {i}; expected only 0..3.")

    mesh = trimesh.load_mesh(INPUT_STL)
    if not isinstance(mesh, trimesh.Trimesh):
        # Sometimes trimesh returns a Scene; merge into a single mesh
        mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])

    print(f"Loaded mesh: {INPUT_STL}")
    print(f"  vertices: {len(mesh.vertices)}")
    print(f"  faces:    {len(mesh.faces)}")
    print(f"  watertight: {mesh.is_watertight}")

    ends = _detect_ends(mesh, expected_ends=EXPECTED_ENDS, axis=AXIS, normal_threshold=NORMAL_THRESHOLD)
    ends = _assign_labels(ends, sort_axes=LABEL_SORT_AXES, directions=LABEL_SORT_DIRECTIONS)

    print("\nDetected ends (label -> centroid, outward normal, loop vertices):")
    for e in ends:
        c = e.centroid
        n = e.outward_normal / (np.linalg.norm(e.outward_normal) + EPS)
        print(
            f"  End {e.label}: centroid=({c[0]: .3f}, {c[1]: .3f}, {c[2]: .3f}), "
            f"normal=({n[0]: .3f}, {n[1]: .3f}, {n[2]: .3f}), "
            f"rim_n={len(e.loop_vertices)}, has_cap={e.has_cap}"
        )

    print("\nWorking ends and parameters:")
    for idx in WORKING_ENDS:
        if idx not in TAPER_PARAMS:
            raise ValueError(f"End {idx} is in WORKING_ENDS but missing from TAPER_PARAMS.")
        s, L = TAPER_PARAMS[idx]
        prof = _normalize_profile_name(TAPER_PROFILE_PER_END.get(idx, TAPER_PROFILE_DEFAULT))
        segs = 1 if prof == "linear" else max(2, int(SMOOTH_SEGMENTS))
        print(f"  End {idx}: scale={s}, length={L}, profile={prof}, segments={segs}")

    out = _apply_taper_and_cap(
        mesh=mesh,
        ends=ends,
        working_ends=WORKING_ENDS,
        taper_params=TAPER_PARAMS,
        profile_default=TAPER_PROFILE_DEFAULT,
        profile_per_end=TAPER_PROFILE_PER_END,
        smooth_segments=SMOOTH_SEGMENTS,
        uncap_all=UNCAP_ALL_ENDS,
        cap_tri_tol=CAP_TRI_TOL,
    )

    print("\nOutput mesh:")
    print(f"  vertices: {len(out.vertices)}")
    print(f"  faces:    {len(out.faces)}")
    print(f"  watertight: {out.is_watertight}")
    if not out.is_watertight:
        print("  WARNING: output is not watertight (there are still boundary edges).")

    out.export(OUTPUT_STL)
    print(f"\nSaved: {OUTPUT_STL}")


if __name__ == "__main__":
    main()
