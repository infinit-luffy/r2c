"""Step 4 `clean`: per-object geometry -> watertight visual mesh + convex collision set.

For each object (type decides the strategy):
  rigid   : drop fragments -> close bottom onto the support plane -> repair -> decimate
  thin    : same, but collision = single convex hull (CoACD over-fragments thin plates)
  capsule : fit a capsule chain for cables/handles (not implemented in v0 -> hull)
  fuzzy   : coarse smooth shell, collision shrunk by `fuzz_margin`
Background: drop fragments, fill the holes the objects left on the table top.

Bottom closing is the part that makes a scan usable in a physics engine: the
scanned shell is open underneath (never seen by the camera); we extrude the
footprint down to the support plane and cap it, so the object has a flat base
that touches the table exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from .manifest import Manifest, hash_inputs


# ----------------------------------------------------------------------------- helpers
def largest_components(mesh: trimesh.Trimesh, keep_frac: float = 0.05) -> trimesh.Trimesh:
    """Keep components whose area >= keep_frac * largest (drops scan fragments)."""
    parts = mesh.split(only_watertight=False)
    if len(parts) <= 1:
        return mesh
    areas = np.array([p.area for p in parts])
    keep = [p for p, a in zip(parts, areas) if a >= keep_frac * areas.max()]
    return trimesh.util.concatenate(keep)


def footprint_polygon(mesh: trimesh.Trimesh, alpha: float | None = None) -> Polygon:
    """2D footprint of the mesh in XY (convex hull by default; alpha shape when
    `alpha` is given via shapely's concave_hull ratio)."""
    pts = mesh.vertices[:, :2]
    mp = MultiPoint(pts)
    if alpha is not None:
        try:
            from shapely import concave_hull

            return concave_hull(mp, ratio=alpha)
        except Exception:
            pass
    return mp.convex_hull


def close_bottom(mesh: trimesh.Trimesh, plane_z: float, sink: float = 0.0005,
                 ratio: float = 0.3) -> trimesh.Trimesh:
    """Close an open scanned shell onto the support plane.

    Strategy: build a solid 'plinth' = footprint polygon extruded from the plane up
    to the lowest scanned ring, union it with the shell, then make it watertight.
    In practice (robust and simple): take the shell, add a skirt from its lowest
    boundary ring straight down to plane_z, and cap the bottom with the footprint.
    """
    m = mesh.copy()
    # boundary edges of the open shell
    edges = m.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    z0 = plane_z - sink  # sink slightly below the plane so the cap is inside the table top
    if len(boundary) == 0:
        return m
    # keep only the boundary loops at the underside opening (lowest loops); other
    # holes (occluded patches higher up) are left for the generic hole filler
    loops_all = _edge_loops(boundary)
    if not loops_all:
        return m
    height = m.bounds[1][2] - plane_z
    loop_z = [m.vertices[l][:, 2].mean() for l in loops_all]
    zmin = min(loop_z)
    bottom_loops = [l for l, z in zip(loops_all, loop_z) if z < zmin + max(0.015, 0.15 * height)]
    low = np.array([[l[i], l[(i + 1) % len(l)]] for l in bottom_loops for i in range(len(l))])
    if len(low) == 0:
        return m
    # rings that already sit on the plane (within `snap`) are snapped flat onto it;
    # no vertex may protrude below the support plane (a single noisy vertex 2 mm
    # low makes a mug stand on a point and tip over in simulation)
    snap = 0.006
    snapped = []
    skirted = []
    for l in bottom_loops:
        if m.vertices[l][:, 2].mean() < plane_z + snap:
            m.vertices[l, 2] = z0
            snapped.append(l)
        else:
            skirted.append(l)
    m.vertices[m.vertices[:, 2] < z0, 2] = z0
    low = np.array([[l[i], l[(i + 1) % len(l)]] for l in skirted for i in range(len(l))]).reshape(-1, 2)
    # skirt: for each boundary edge (a,b) add quad a,b,b',a' with primed = projected to z0
    V = m.vertices.copy()
    F = list(m.faces)
    n0 = len(V)
    idx_map = {}
    new_pts = []
    for e in low:
        for vi in e:
            if vi not in idx_map:
                idx_map[vi] = n0 + len(new_pts)
                p = V[vi].copy(); p[2] = z0
                new_pts.append(p)
    V = np.vstack([V, np.array(new_pts)]) if new_pts else V
    # orient skirt quads consistently with the shell: the boundary edge (a,b) appears
    # in exactly one face; use that face's winding to decide the skirt winding
    edge_dir = {}
    for f in m.faces:
        for i in range(3):
            edge_dir[(f[i], f[(i + 1) % 3])] = True
    for a, b in low:
        if (a, b) not in edge_dir:  # face has edge as (b,a) -> flip so skirt faces outward
            a, b = b, a
        a2, b2 = idx_map[a], idx_map[b]
        F.append([b, a, a2]); F.append([b, a2, b2])
    # bottom cap: one fan per boundary loop, from the loop centroid (at z0).
    # Footprints of tabletop objects are star-shaped w.r.t. their centroid in practice.
    # Skirted loops are capped on their projected ring; snapped loops directly.
    for loop in skirted:
        ring_idx = [idx_map[v] for v in loop]
        ring = V[ring_idx]
        c = ring.mean(0); c[2] = z0
        ci = len(V)
        V = np.vstack([V, c[None]])
        k = len(ring_idx)
        for i in range(k):
            a2, b2 = ring_idx[i], ring_idx[(i + 1) % k]
            F.append([ci, b2, a2])  # skirt bottom edge runs a2->b2, cap runs b2->a2
    for loop in snapped:
        ring = V[loop]
        c = ring.mean(0); c[2] = z0
        ci = len(V)
        V = np.vstack([V, c[None]])
        k = len(loop)
        for i in range(k):
            a, b = loop[i], loop[(i + 1) % k]
            F.append([ci, b, a] if (a, b) in edge_dir else [ci, a, b])
    out = trimesh.Trimesh(V, np.array(F), process=True)
    trimesh.repair.fix_normals(out)
    if out.is_watertight and out.volume < 0:
        out.invert()
    return out


def _edge_loops(edges: np.ndarray) -> list[list[int]]:
    """Order undirected boundary edges into closed loops (list of vertex ids)."""
    from collections import defaultdict

    adj = defaultdict(list)
    for a, b in edges:
        adj[int(a)].append(int(b)); adj[int(b)].append(int(a))
    seen = set()
    loops = []
    for start in list(adj):
        if start in seen:
            continue
        loop = [start]
        seen.add(start)
        prev, cur = None, start
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur); seen.add(cur)
        if len(loop) >= 3:
            loops.append(loop)
    return loops


def fill_loops_fan(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Fill every remaining boundary loop with a fan from its centroid (good for
    the small occluded patches a scan leaves on undersides of handles etc.)."""
    m = mesh.copy()
    e = m.edges_sorted
    u, c = np.unique(e, axis=0, return_counts=True)
    boundary = u[c == 1]
    if len(boundary) == 0:
        return m
    edge_dir = {(f[i], f[(i + 1) % 3]) for f in m.faces for i in range(3)}
    V = m.vertices.copy()
    F = list(m.faces)
    for loop in _edge_loops(boundary):
        c0 = V[loop].mean(0)
        ci = len(V)
        V = np.vstack([V, c0[None]])
        k = len(loop)
        for i in range(k):
            a, b = loop[i], loop[(i + 1) % k]
            if (a, b) in edge_dir:
                F.append([ci, b, a])
            else:
                F.append([ci, a, b])
    out = trimesh.Trimesh(V, np.array(F), process=True)
    trimesh.repair.fix_normals(out)
    return out


def make_watertight(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    import pymeshfix

    out = mesh.copy()
    if not out.is_watertight:
        out = fill_loops_fan(out)
    if not out.is_watertight:
        v = np.ascontiguousarray(np.asarray(out.vertices, dtype=np.float64))
        f = np.ascontiguousarray(np.asarray(out.faces, dtype=np.int32))
        try:
            v2, f2 = pymeshfix.clean_from_arrays(v, f, joincomp=True, remove_smallest_components=False)
            cand = trimesh.Trimesh(v2, f2, process=True)
            # MeshFix sometimes discards most of the mesh; only accept a sane result
            if cand.is_watertight and len(cand.faces) >= 4 and \
               np.all(cand.extents > 0.8 * out.extents):
                out = cand
        except Exception:
            pass
    trimesh.repair.fix_normals(out)
    if out.is_watertight and out.volume < 0:
        out.invert()
    return out


def decimate(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(mesh.faces) <= target_faces:
        return mesh
    try:
        return mesh.simplify_quadric_decimation(face_count=target_faces)
    except Exception:
        return mesh


def convex_decompose(mesh: trimesh.Trimesh, threshold: float = 0.05, max_hulls: int = 16) -> list[trimesh.Trimesh]:
    import coacd

    coacd.set_log_level("error")
    # nearly convex already (ball, box, plate): one hull is exact enough and far cheaper
    hull = mesh.convex_hull
    if mesh.is_watertight and mesh.volume > 0 and hull.volume / mesh.volume < 1.10:
        return [hull]
    cm = coacd.Mesh(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64))
    parts = coacd.run_coacd(cm, threshold=threshold, max_convex_hull=max_hulls,
                            preprocess_mode="auto", preprocess_resolution=100)
    hulls = []
    for v, f in parts:
        h = trimesh.Trimesh(v, f, process=True)
        if h.volume > 1e-9:
            hulls.append(h.convex_hull)
    return hulls or [mesh.convex_hull]


def fill_table_holes(bg: trimesh.Trimesh, holes: list[Polygon], plane_z: float) -> trimesh.Trimesh:
    """The scan has no table surface under the objects; add flat patches there."""
    patches = []
    for poly in holes:
        if poly.is_empty:
            continue
        v, f = trimesh.creation.triangulate_polygon(poly.buffer(0.004), engine="earcut")
        v3 = np.column_stack([v, np.full(len(v), plane_z - 0.0002)])
        patches.append(trimesh.Trimesh(v3, f, process=False))
    if not patches:
        return bg
    return trimesh.util.concatenate([bg, *patches])


DENSITY = {"rigid": 600.0, "thin": 700.0, "fuzzy": 150.0, "capsule": 900.0}  # kg/m^3 defaults


# ----------------------------------------------------------------------------- driver
def clean(workdir: Path, target_faces: int = 50000, coacd_threshold: float = 0.05,
          max_hulls: int = 16, fuzz_margin: float = 0.005) -> dict:
    man = Manifest(workdir)
    seg = man.step_outputs("segment")
    params = {"target_faces": target_faces, "coacd_threshold": coacd_threshold,
              "max_hulls": max_hulls, "fuzz_margin": fuzz_margin,
              "types": {k: v.get("type", "rigid") for k, v in man.objects().items()}}
    files = [Path(o["raw_mesh"]) for o in seg["objects"].values()] + [Path(seg["background_raw"])]
    key = hash_inputs(files, params)
    if man.step_cached("clean", key):
        return man.step_outputs("clean")
    man.step_begin("clean", key, params)
    plane_z = man.scene["plane_z"]
    out = Path(workdir) / "clean"
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    holes = []
    for oid, rec in seg["objects"].items():
        otype = man.objects()[oid].get("type", "rigid")
        raw = trimesh.load(rec["raw_mesh"], force="mesh", process=False)
        m = largest_components(raw)
        holes.append(footprint_polygon(m, alpha=0.3))
        m = close_bottom(m, plane_z)
        m = make_watertight(m)
        m = decimate(m, target_faces)
        od = out / oid
        od.mkdir(exist_ok=True)
        m.export(od / "visual.ply")
        if otype == "thin":
            hulls = [m.convex_hull]
        elif otype == "fuzzy":
            shrunk = m.copy()
            shrunk.vertices -= shrunk.vertex_normals * fuzz_margin
            hulls = convex_decompose(shrunk, coacd_threshold, max_hulls)
        else:
            hulls = convex_decompose(m, coacd_threshold, max_hulls)
        for i, h in enumerate(hulls):
            h.export(od / f"collision_{i:02d}.ply")
        vol = float(m.volume)
        mass = vol * DENSITY.get(otype, 600.0)
        r = {"visual": str(od / "visual.ply"), "collision": [str(od / f"collision_{i:02d}.ply") for i in range(len(hulls))],
             "watertight": bool(m.is_watertight), "faces": int(len(m.faces)), "volume": vol,
             "mass": mass, "com": m.center_mass.tolist(), "bbox": m.bounds.tolist(), "type": otype}
        results[oid] = r
        man.set_object(oid, **r)

    bg = trimesh.load(seg["background_raw"], force="mesh", process=False)
    bg = largest_components(bg, keep_frac=0.002)
    bg = fill_table_holes(bg, holes, plane_z)
    bg = decimate(bg, max(target_faces * 4, 200000))
    bg.export(out / "background.ply")
    outputs = {"objects": results, "background": str(out / "background.ply"),
               "background_faces": int(len(bg.faces))}
    man.step_end("clean", outputs)
    (out / "clean.json").write_text(json.dumps(outputs, indent=2))
    return outputs
