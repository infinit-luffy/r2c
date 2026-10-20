"""Step 5 `texture`: colour the cleaned meshes from the gaussians.

v0 bakes per-vertex colours (SH0 of the nearest gaussians). Optional `delight`
removes the dominant directional shading baked into the scan so the asset can be
re-lit in RTX without double shading:

  lambert : fit luminance ~ c0 + c . normal over all gaussians (albedo assumed
            uncorrelated with normal); divide colours by the fitted shading.
  none    : keep the scan colours as they are.

Proper photo-projection + HDRI-based delighting (Blender) is a later step; the
interfaces here produce the same outputs (vertex colours / albedo) so pack() does
not care which one produced them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from . import io as rio
from .manifest import Manifest, hash_inputs


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def fit_directional_shading(xyz: np.ndarray, rgb: np.ndarray, mesh: trimesh.Trimesh) -> dict:
    """Estimate shading s(n) = c0 + c·n from gaussians and the mesh normals."""
    n_s = int(np.clip(mesh.area * 2e5, 50000, 2_000_000))
    sp, sf = trimesh.sample.sample_surface(mesh, n_s, seed=0)
    tree = cKDTree(sp)
    d, nn = tree.query(xyz, k=1)
    ok = d < 0.01
    normals = mesh.face_normals[sf[nn[ok]]]
    L = _luminance(rgb[ok])
    A = np.column_stack([np.ones(len(L)), normals])
    coef, *_ = np.linalg.lstsq(A, L, rcond=None)
    c0, c = coef[0], coef[1:]
    light = c / (np.linalg.norm(c) + 1e-9)
    return {"c0": float(c0), "c": c.tolist(), "light_dir": light.tolist(),
            "strength": float(np.linalg.norm(c) / max(c0, 1e-6))}


def delight_colors(rgb: np.ndarray, normals: np.ndarray, model: dict, floor: float = 0.25) -> np.ndarray:
    s = model["c0"] + normals @ np.array(model["c"])
    s_mean = model["c0"]  # normalise so a surface facing "average" keeps its brightness
    ratio = np.clip(s / max(s_mean, 1e-6), floor, 4.0)
    return np.clip(rgb / ratio[:, None], 0, 1)


def bake_vertex_colors(mesh: trimesh.Trimesh, g: rio.Gaussians, k: int = 8, max_dist: float = 0.02,
                       fallback=(0.6, 0.6, 0.6)) -> np.ndarray:
    tree = cKDTree(g.xyz)
    d, nn = tree.query(mesh.vertices, k=k, distance_upper_bound=max_dist)
    cols = np.full((len(mesh.vertices), 3), fallback, dtype=np.float32)
    for i in range(len(mesh.vertices)):
        valid = np.isfinite(d[i])
        if valid.any():
            w = 1.0 / (d[i][valid] + 1e-4)
            cols[i] = (g.rgb[nn[i][valid]] * w[:, None]).sum(0) / w.sum()
    return cols


def texture(workdir: Path, delight: str = "lambert") -> dict:
    man = Manifest(workdir)
    cl = man.step_outputs("clean")
    seg = man.step_outputs("segment")
    params = {"delight": delight}
    files = [Path(o["visual"]) for o in cl["objects"].values()] + [Path(cl["background"])]
    if "background_gs" in seg:
        files.append(Path(seg["background_gs"]))
    key = hash_inputs(files, params)
    if man.step_cached("texture", key):
        return man.step_outputs("texture")
    man.step_begin("texture", key, params)
    out = Path(workdir) / "textured"
    out.mkdir(parents=True, exist_ok=True)

    have_gs = "gaussians" in man.step_outputs("align")
    model = None
    if have_gs:
        g_all = rio.read_gaussians(Path(man.step_outputs("align")["gaussians"]))
        if delight == "lambert":
            bg = trimesh.load(cl["background"], force="mesh", process=False)
            objs = [trimesh.load(o["visual"], force="mesh", process=False) for o in cl["objects"].values()]
            full = trimesh.util.concatenate([bg, *objs])
            model = fit_directional_shading(g_all.xyz.astype(np.float64), g_all.rgb, full)
    results = {}
    for oid, rec in cl["objects"].items():
        m = trimesh.load(rec["visual"], force="mesh", process=False)
        if have_gs:
            g = rio.read_gaussians(Path(seg["objects"][oid]["raw_gs"]))
            cols = bake_vertex_colors(m, g)
            if model is not None:
                cols = delight_colors(cols, m.vertex_normals, model)
        else:
            cols = np.full((len(m.vertices), 3), 0.6, dtype=np.float32)
        m.visual.vertex_colors = np.column_stack([(cols * 255).astype(np.uint8), np.full(len(cols), 255, np.uint8)])
        od = out / oid
        od.mkdir(exist_ok=True)
        m.export(od / "visual.ply")
        results[oid] = {"visual": str(od / "visual.ply"), "mean_albedo": cols.mean(0).tolist()}
        man.set_object(oid, textured=str(od / "visual.ply"))
    bg = trimesh.load(cl["background"], force="mesh", process=False)
    if have_gs:
        gbg = rio.read_gaussians(Path(seg["background_gs"]))
        cols = bake_vertex_colors(bg, gbg)
        if model is not None:
            cols = delight_colors(cols, bg.vertex_normals, model)
    else:
        cols = np.full((len(bg.vertices), 3), 0.5, dtype=np.float32)
    bg.visual.vertex_colors = np.column_stack([(cols * 255).astype(np.uint8), np.full(len(cols), 255, np.uint8)])
    bg.export(out / "background.ply")
    outputs = {"objects": results, "background": str(out / "background.ply"), "shading_model": model}
    man.step_end("texture", outputs)
    (out / "texture.json").write_text(json.dumps(outputs, indent=2))
    return outputs
