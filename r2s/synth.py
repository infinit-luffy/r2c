"""Synthetic tabletop scan generator.

Produces what a real scan pipeline leaves behind, with ground truth:
  scene_gs.ply   - 3DGS-format gaussians sampled on visible surfaces (SH0 colour)
  scene_mesh.obj - ONE holistic mesh, no groups, bottoms removed (unseen from above),
                   small vertex noise, a few floater fragments
  scene.usd      - the same holistic mesh as a single Mesh prim (Z-up, metres)
  gt_labels.npy  - per-face ground-truth object id for scoring segmentation
  gt.json        - object list with true poses / sizes

Used by the test-suite and as a stand-in when no real scan is at hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from .io import write_gaussians, Gaussians

TABLE_H = 0.75


def _mug(radius=0.04, height=0.095, color=(0.85, 0.2, 0.15)):
    body = trimesh.creation.cylinder(radius=radius, height=height, sections=48)
    body.apply_translation([0, 0, height / 2])
    handle = trimesh.creation.torus(major_radius=0.025, minor_radius=0.006, major_sections=32, minor_sections=12)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    handle.apply_translation([radius + 0.018, 0, height * 0.55])
    m = trimesh.util.concatenate([body, handle])
    m.visual.vertex_colors = (np.array(color + (1.0,)) * 255).astype(np.uint8)
    return m


def _box(size, color):
    m = trimesh.creation.box(extents=size)
    m.apply_translation([0, 0, size[2] / 2])
    m.visual.vertex_colors = (np.array(color + (1.0,)) * 255).astype(np.uint8)
    return m


def _sphere(r, color):
    m = trimesh.creation.icosphere(subdivisions=3, radius=r)
    m.apply_translation([0, 0, r])
    m.visual.vertex_colors = (np.array(color + (1.0,)) * 255).astype(np.uint8)
    return m


def make_scene(out: Path, seed: int = 0, n_gauss_per_m2: int = 60000) -> dict:
    rng = np.random.default_rng(seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- table (static background) : top slab + 4 legs
    top = trimesh.creation.box(extents=[1.2, 0.8, 0.04])
    top.apply_translation([0, 0, TABLE_H - 0.02])
    top.visual.vertex_colors = [150, 120, 90, 255]
    legs = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            leg = trimesh.creation.box(extents=[0.05, 0.05, TABLE_H - 0.04])
            leg.apply_translation([sx * 0.55, sy * 0.35, (TABLE_H - 0.04) / 2])
            leg.visual.vertex_colors = [120, 95, 70, 255]
            legs.append(leg)
    floor = trimesh.creation.box(extents=[3.0, 3.0, 0.01])
    floor.apply_translation([0, 0, -0.005])
    floor.visual.vertex_colors = [80, 80, 85, 255]
    background = trimesh.util.concatenate([top, *legs, floor])

    # ---- objects (ground truth)
    objs = {
        "mug": (_mug(), [0.15, 0.10, TABLE_H], 0.35),
        "box": (_box([0.12, 0.08, 0.06], (0.2, 0.3, 0.8)), [-0.20, 0.05, TABLE_H], 0.25),
        "ball": (_sphere(0.035, (0.95, 0.85, 0.1)), [0.05, -0.20, TABLE_H], 0.06),
        "board": (_box([0.20, 0.15, 0.012], (0.9, 0.9, 0.9)), [-0.15, -0.22, TABLE_H], 0.15),
    }
    gt = {"table_height": TABLE_H, "objects": {}}
    parts = [background]
    labels = [np.zeros(len(background.faces), dtype=np.int32)]
    for k, (name, (m, pos, mass)) in enumerate(objs.items(), start=1):
        yaw = rng.uniform(0, np.pi)
        T = trimesh.transformations.rotation_matrix(yaw, [0, 0, 1])
        T[:3, 3] = pos
        m = m.copy()
        m.apply_transform(T)
        gt["objects"][name] = {"id": k, "position": pos, "yaw": float(yaw),
                               "extents": m.extents.tolist(), "mass": mass,
                               "bbox": [m.bounds[0].tolist(), m.bounds[1].tolist()]}
        parts.append(m)
        labels.append(np.full(len(m.faces), k, dtype=np.int32))
    full = trimesh.util.concatenate(parts)
    labels = np.concatenate(labels)

    # ---- make it look like a scan: drop downward-facing faces (never seen),
    #      drop floor underside, add noise, add floaters
    normals = full.face_normals
    keep = normals[:, 2] > -0.3  # bottoms invisible from above
    # also hide table underside/legs lower half partially (scanned from above)
    scan_mesh = full.submesh([np.where(keep)[0]], append=True, repair=False)
    scan_labels = labels[keep]
    noise = rng.normal(0, 0.0008, size=scan_mesh.vertices.shape)
    scan_mesh.vertices = scan_mesh.vertices + noise
    floaters = []
    for _ in range(6):
        f = trimesh.creation.icosphere(subdivisions=1, radius=rng.uniform(0.004, 0.01))
        f.apply_translation([rng.uniform(-0.5, 0.5), rng.uniform(-0.3, 0.3), TABLE_H + rng.uniform(0.05, 0.4)])
        f.visual.vertex_colors = [200, 200, 200, 255]
        floaters.append(f)
    scan_mesh = trimesh.util.concatenate([scan_mesh, *floaters])
    scan_labels = np.concatenate([scan_labels, np.full(sum(len(f.faces) for f in floaters), -1, dtype=np.int32)])

    # ---- gaussians: sample visible surface, colour from vertex colours
    area = scan_mesh.area
    n = int(area * n_gauss_per_m2)
    pts, fid = trimesh.sample.sample_surface(scan_mesh, n, seed=seed)
    vc = scan_mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
    rgb = vc[scan_mesh.faces[fid]].mean(1)
    # bake a fake shading term (light from +x,+z) so delight has something to do
    nrm = scan_mesh.face_normals[fid]
    light = np.array([0.5, 0.2, 0.8]); light /= np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(nrm @ light, 0, 1)
    rgb = np.clip(rgb * shade[:, None], 0, 1)
    scale = np.full((n, 3), 0.004, dtype=np.float32) * rng.uniform(0.7, 1.4, size=(n, 1)).astype(np.float32)
    rot = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n, 1))
    opacity = rng.uniform(0.8, 1.0, n).astype(np.float32)
    g = Gaussians(pts.astype(np.float32), rgb.astype(np.float32), opacity, scale, rot)
    write_gaussians(out / "scene_gs.ply", g)
    gauss_labels = scan_labels[fid]

    # ---- write files
    scan_mesh.export(out / "scene_mesh.obj")
    np.save(out / "gt_labels.npy", scan_labels)
    np.save(out / "gt_gauss_labels.npy", gauss_labels)
    _write_usd(out / "scene.usd", scan_mesh)
    (out / "gt.json").write_text(json.dumps(gt, indent=2))
    return gt


def _write_usd(path: Path, mesh: trimesh.Trimesh) -> None:
    from pxr import Usd, UsdGeom, Vt, Gf

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Scene")
    m = UsdGeom.Mesh.Define(stage, "/Scene/scan")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(mesh.vertices.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.faces.flatten().astype(np.int32)))
    if mesh.visual.kind == "vertex":
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", __import__("pxr").Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy((mesh.visual.vertex_colors[:, :3] / 255.0).astype(np.float32)))
    stage.SetDefaultPrim(root.GetPrim())
    stage.Save()
