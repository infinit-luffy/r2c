"""Step 1 `inspect`: what is in each input file, in which units / coordinate frame.
Step 2 `align`: bring GS, mesh and USD into one Z-up metric frame.

Alignment strategy (deterministic, no ICP needed for the common cases):
  * detect up-axis of each source from its dominant plane normal (the table / floor)
  * detect scale from bbox ratio between sources (they describe the same scene)
  * detect metres-per-unit from the plane height if a reference height is given,
    otherwise trust the USD stage metadata or the user-supplied --scale
  * final frame: Z-up, metres, dominant plane at z = plane height, scene centred in XY
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from . import io as rio
from .manifest import Manifest, hash_inputs


# ----------------------------------------------------------------------------- inspect
def inspect_inputs(mesh: Path | None, gs: Path | None, usd: Path | None) -> dict:
    report: dict = {}
    if mesh:
        groups = rio.read_mesh_groups(mesh)
        m = rio.read_mesh(mesh)
        report["mesh"] = {
            "path": str(mesh),
            "faces": int(len(m.faces)),
            "vertices": int(len(m.vertices)),
            "groups": len(groups),
            "group_names": list(groups.keys())[:20],
            "has_vertex_colors": m.visual.kind == "vertex",
            "has_uv": m.visual.kind == "texture",
            "watertight": bool(m.is_watertight),
            "bbox": m.bounds.tolist(),
            "extents": m.extents.tolist(),
            "components": int(len(m.split(only_watertight=False))),
            "up_axis_guess": _up_axis_guess(m.vertices, m.face_normals, m.area_faces),
        }
    if gs:
        if rio.is_gaussian_ply(gs):
            g = rio.read_gaussians(gs)
            report["gaussians"] = {
                "path": str(gs), "format": "3dgs", "count": int(len(g)),
                "sh_rest_fields": sum(1 for k in g.extra if k.startswith("f_rest_")),
                "bbox": [g.xyz.min(0).tolist(), g.xyz.max(0).tolist()],
                "extents": (g.xyz.max(0) - g.xyz.min(0)).tolist(),
                "median_scale": float(np.median(g.scale)),
                "up_axis_guess": _up_axis_guess(g.xyz),
            }
        else:
            pc = rio.read_pointcloud(gs)
            report["gaussians"] = {
                "path": str(gs), "format": "pointcloud", "count": int(len(pc.vertices)),
                "bbox": pc.bounds.tolist(), "extents": pc.extents.tolist(),
                "has_colors": pc.colors is not None and len(pc.colors) > 0,
                "up_axis_guess": _up_axis_guess(np.asarray(pc.vertices)),
            }
    if usd:
        s = rio.read_usd_summary(usd)
        meshes = [p for p in s["prims"] if "faces" in p]
        s["mesh_prims"] = len(meshes)
        s["rigid_bodies"] = sum(1 for p in meshes if p.get("rigid_body"))
        s["already_decomposed"] = len(meshes) > 1
        s["path"] = str(usd)
        s["prims"] = s["prims"][:50]
        report["usd"] = s

    # cross-source consistency
    ext = {k: np.array(v["extents"]) for k, v in report.items() if "extents" in v}
    if "usd" in report and "bbox" in report["usd"]:
        b = np.array(report["usd"]["bbox"])
        ext["usd"] = b[1] - b[0]
    if len(ext) >= 2:
        keys = list(ext)
        ratios = {f"{a}/{b}": float(np.linalg.norm(ext[a]) / max(np.linalg.norm(ext[b]), 1e-9))
                  for i, a in enumerate(keys) for b in keys[i + 1:]}
        report["scale_ratios"] = ratios
        report["consistent_scale"] = all(abs(r - 1) < 0.05 for r in ratios.values())
    return report


def _up_axis_guess(points: np.ndarray, normals: np.ndarray | None = None, weights: np.ndarray | None = None) -> dict:
    """Guess which axis is 'up' from the dominant planar normal (table/floor)."""
    if normals is not None and weights is not None:
        # weight by face area, find dominant axis-aligned normal direction
        n = np.abs(normals)
        score = (n * weights[:, None]).sum(0)
    else:
        # for point clouds: axis with smallest spread is usually up (tabletop scenes are flat)
        score = 1.0 / (points.std(0) + 1e-9)
    axis = int(np.argmax(score))
    sign = 1
    if normals is not None and weights is not None:
        sign = int(np.sign((normals[:, axis] * weights).sum()) or 1)
    return {"axis": "xyz"[axis], "sign": sign, "scores": score.tolist()}


# ----------------------------------------------------------------------------- align
def fit_dominant_plane(points: np.ndarray, n_iter: int = 400, thresh: float = 0.006, seed: int = 0):
    """RANSAC plane. Returns (normal, d, inlier_mask) with normal·p + d = 0."""
    rng = np.random.default_rng(seed)
    best = (None, None, np.zeros(len(points), bool))
    N = len(points)
    sub = points if N <= 200000 else points[rng.choice(N, 200000, replace=False)]
    for _ in range(n_iter):
        idx = rng.choice(len(sub), 3, replace=False)
        p0, p1, p2 = sub[idx]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n /= nn
        d = -n @ p0
        dist = np.abs(sub @ n + d)
        inl = dist < thresh
        if inl.sum() > best[2].sum():
            best = (n, d, inl)
    n, d, inl = best
    # refine with least squares on inliers
    P = sub[inl]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    d = -n @ c
    full_inl = np.abs(points @ n + d) < thresh
    return n, d, full_inl


def find_parallel_planes(points: np.ndarray, normal: np.ndarray, max_planes: int = 4,
                         thresh: float = 0.008, min_frac: float = 0.02) -> list[dict]:
    """Sequentially extract planes parallel to `normal` (1-D RANSAC on the height
    coordinate). Returns list of {height, count, point, inliers(mask over points)}."""
    h = points @ normal
    remaining = np.ones(len(points), bool)
    planes = []
    for _ in range(max_planes):
        hs = h[remaining]
        if len(hs) < min_frac * len(points):
            break
        # histogram peak as a robust mode
        bins = np.arange(hs.min(), hs.max() + thresh, thresh)
        hist, edges = np.histogram(hs, bins=bins)
        k = int(np.argmax(hist))
        centre = 0.5 * (edges[k] + edges[k + 1])
        inl = remaining & (np.abs(h - centre) < thresh)
        if inl.sum() < min_frac * len(points):
            break
        centre = float(h[inl].mean())
        planes.append({"height": centre, "count": int(inl.sum()),
                       "point": points[inl].mean(0), "inliers": inl.copy()})
        remaining &= ~inl
    return planes


def plane_to_zup_transform(normal: np.ndarray, d: float, plane_z: float = 0.0, center_xy: np.ndarray | None = None) -> np.ndarray:
    """4x4 transform mapping plane (normal,d) to z = plane_z, normal -> +z."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = n @ z
    if s < 1e-9:
        R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
    T = np.eye(4)
    T[:3, :3] = R
    # point on plane
    p0 = -d * n
    T[:3, 3] = -R @ p0
    T[2, 3] += plane_z
    if center_xy is not None:
        T[0, 3] -= center_xy[0]
        T[1, 3] -= center_xy[1]
    return T


def align(workdir: Path, mesh: Path | None, gs: Path | None, usd: Path | None,
          scale: float | None = None, table_height: float | None = None,
          prefer: str = "mesh") -> dict:
    """Write aligned copies into <workdir>/aligned/ and record transforms."""
    man = Manifest(workdir)
    params = {"scale": scale, "table_height": table_height, "prefer": prefer}
    files = [p for p in (mesh, gs, usd) if p]
    key = hash_inputs(files, params)
    if man.step_cached("align", key):
        return man.step_outputs("align")
    man.step_begin("align", key, params)
    out = Path(workdir) / "aligned"
    out.mkdir(parents=True, exist_ok=True)

    # reference geometry for plane fitting
    ref_mesh = None
    if mesh:
        ref_mesh = rio.read_mesh(mesh)
    elif usd:
        ref_mesh = rio.usd_to_trimesh(usd)
    g = rio.read_gaussians(gs) if gs and rio.is_gaussian_ply(gs) else None
    pc = rio.read_pointcloud(gs) if gs and g is None else None

    # reference points: area-weighted surface samples (mesh vertices alone are too
    # sparse on primitive-like scans), or gaussian centres
    if prefer == "mesh" and ref_mesh is not None:
        ref_pts, _ = trimesh.sample.sample_surface(ref_mesh, 200000, seed=0)
    elif g is not None:
        ref_pts = g.xyz.astype(np.float64)
    elif pc is not None:
        ref_pts = np.asarray(pc.vertices)
    else:
        ref_pts, _ = trimesh.sample.sample_surface(ref_mesh, 200000, seed=0)
    ref_pts = np.asarray(ref_pts, dtype=np.float64)

    s = 1.0 if scale is None else float(scale)
    ref_pts = ref_pts * s

    # 1) dominant plane gives the up direction (floor or table top)
    n, d, inl = fit_dominant_plane(ref_pts)
    if np.sign((ref_pts @ n + d).mean()) < 0:
        n, d = -n, -d
    # 2) find other planes parallel to it (table top vs floor) by sequential RANSAC
    planes = find_parallel_planes(ref_pts, n, max_planes=4)
    heights = np.array([p["height"] for p in planes])
    areas = np.array([p["count"] for p in planes], dtype=float)
    big = areas >= 0.10 * areas.max()
    support_i = int(np.argmax(np.where(big, heights, -np.inf)))  # highest big plane
    lowest_i = int(np.argmin(heights))
    support = planes[support_i]
    gap = float(heights[support_i] - heights[lowest_i])
    if table_height is not None:
        plane_z = float(table_height)
    elif support_i != lowest_i and gap > 0.2:
        plane_z = gap  # floor visible: table sits `gap` above it
    else:
        plane_z = 0.0
    d_support = -float(n @ support["point"])
    T = plane_to_zup_transform(n, d_support, plane_z)
    # centre the scene in XY on the support plane inliers (the table), not the whole scan
    tmp = (ref_pts[support["inliers"]] @ T[:3, :3].T + T[:3, 3])
    cxy = tmp[:, :2].mean(0)
    T[0, 3] -= cxy[0]
    T[1, 3] -= cxy[1]
    S = np.eye(4) * s
    S[3, 3] = 1.0
    T_full = T @ S

    outputs = {"transform": T_full.tolist(), "scale": s, "up_normal_src": n.tolist(),
               "planes": [{"height_rel": float(h - heights[lowest_i]), "count": int(c)} for h, c in zip(heights, areas)],
               "support_plane_index": support_i, "plane_z": plane_z}
    if ref_mesh is not None:
        m2 = ref_mesh.copy()
        m2.apply_transform(T_full)
        m2.export(out / "scene_mesh.ply")
        outputs["mesh"] = str(out / "scene_mesh.ply")
    if g is not None:
        g2 = g.transformed(T_full)
        rio.write_gaussians(out / "scene_gs.ply", g2)
        outputs["gaussians"] = str(out / "scene_gs.ply")
    if pc is not None:
        pc2 = pc.copy()
        pc2.apply_transform(T_full)
        pc2.export(out / "scene_points.ply")
        outputs["pointcloud"] = str(out / "scene_points.ply")
    man.set_scene(transform=T_full.tolist(), plane_z=plane_z, up="z", units="m")
    man.step_end("align", outputs)
    (out / "align.json").write_text(json.dumps(outputs, indent=2))
    return outputs
