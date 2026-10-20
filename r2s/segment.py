"""Step 3 `segment`: split the holistic scan into background + per-object pieces.

Two modes:

auto  (default, no ML):   table plane -> everything resting on it, clustered by
                          spatial proximity (DBSCAN on face centroids) with a
                          handle-merging rule and floater rejection.
masks (SAM2 etc.):        per-view 2D masks + cameras -> per-face votes by ray
                          casting -> majority label; ambiguous faces -> background.
                          Mask generation itself lives in scripts/run_sam2_views.py.

Gaussians are labelled by nearest labelled face so the GS subset of each object
is exported alongside its mesh.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

from . import io as rio
from .manifest import Manifest, hash_inputs

BACKGROUND = 0
FLOATER = -1


# ----------------------------------------------------------------------------- auto mode
def segment_auto(mesh: trimesh.Trimesh, plane_z: float, *,
                 plane_thresh: float = 0.008,
                 cluster_eps: float = 0.015,
                 min_area: float = 4e-4,
                 min_extent: float = 0.015,
                 max_height: float = 0.6,
                 merge_gap: float = 0.012) -> np.ndarray:
    """Return per-face labels: 0 background, -1 floater, 1..K objects."""
    fc = mesh.triangles_center
    fn = mesh.face_normals
    labels = np.full(len(fc), BACKGROUND, dtype=np.int32)

    # faces on / below the support plane are background (table top, legs, floor)
    on_plane = (np.abs(fc[:, 2] - plane_z) < plane_thresh) & (np.abs(fn[:, 2]) > 0.7)
    below = fc[:, 2] < plane_z - plane_thresh
    too_high = fc[:, 2] > plane_z + max_height
    cand = ~(on_plane | below | too_high)
    idx = np.where(cand)[0]
    if len(idx) == 0:
        return labels

    # cluster area-weighted surface samples (robust to low-poly / long thin faces),
    # then assign each face by the majority label of its samples
    sub = mesh.submesh([idx], append=True, repair=False)
    n_samp = int(np.clip(sub.area * 1.0e6, 20000, 500000))  # ~1 sample / mm^2
    pts, fid = trimesh.sample.sample_surface(sub, n_samp, seed=0)
    db = DBSCAN(eps=cluster_eps, min_samples=4).fit(pts)
    cl = _merge_touching(pts, db.labels_, merge_gap)
    face_cl = np.full(len(idx), -1, dtype=np.int64)
    for f in np.unique(fid):
        votes = cl[fid == f]
        votes = votes[votes >= 0]
        if len(votes):
            face_cl[f] = np.bincount(votes).argmax()
    # faces that received no sample: inherit the cluster of the nearest sampled point
    missing = np.where(face_cl < 0)[0]
    if len(missing) and (cl >= 0).any():
        tree = cKDTree(pts[cl >= 0])
        d, nn = tree.query(sub.triangles_center[missing], k=1)
        ok = d < cluster_eps
        face_cl[missing[ok]] = cl[cl >= 0][nn[ok]]
    labels[idx] = FLOATER
    next_id = 1
    for c in np.unique(face_cl):
        if c < 0:
            continue
        sel = idx[face_cl == c]
        pts_c = mesh.vertices[mesh.faces[sel]].reshape(-1, 3)
        ext = pts_c.max(0) - pts_c.min(0)
        area = mesh.area_faces[sel].sum()
        if area < min_area or ext.max() < min_extent:
            continue  # floater / scan debris
        # must rest on the support plane: lowest point close to the plane
        # (bottom faces are usually missing, so allow a fraction of the height)
        if pts_c[:, 2].min() > plane_z + max(0.02, 0.5 * ext[2]):
            continue
        labels[sel] = next_id
        next_id += 1
    return labels


def _merge_touching(pts: np.ndarray, cl: np.ndarray, gap: float) -> np.ndarray:
    ids = [c for c in np.unique(cl) if c >= 0]
    if len(ids) < 2:
        return cl
    trees = {c: cKDTree(pts[cl == c]) for c in ids}
    parent = {c: c for c in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d, _ = trees[a].query(pts[cl == b], k=1, distance_upper_bound=gap)
            if np.isfinite(d).any():
                parent[find(a)] = find(b)
    out = cl.copy()
    for c in ids:
        out[cl == c] = find(c)
    return out


def label_points_by_surface(mesh: trimesh.Trimesh, face_labels: np.ndarray, pts: np.ndarray,
                            samples_per_m2: float = 2.0e5, max_dist: float = 0.02) -> np.ndarray:
    """Label free points (gaussian centres) by the nearest *surface* point of the
    labelled mesh. Uses dense surface samples + KD-tree (fast, accurate enough);
    points further than `max_dist` from any surface become FLOATER."""
    n = int(np.clip(mesh.area * samples_per_m2, 50000, 3_000_000))
    sp, sf = trimesh.sample.sample_surface(mesh, n, seed=0)
    tree = cKDTree(sp)
    d, nn = tree.query(pts, k=1)
    out = face_labels[sf[nn]].copy()
    out[d > max_dist] = FLOATER
    return out


# ----------------------------------------------------------------------------- masks mode
def lift_masks_to_faces(mesh: trimesh.Trimesh, views: list[dict], masks_dir: Path,
                        vote_thresh: float = 0.7) -> np.ndarray:
    """views: [{"image": name, "K": 3x3, "T_world_cam": 4x4, "width", "height"}]
    masks_dir/<image_stem>_<obj_id>.png : binary masks (obj_id >= 1).
    Casts a ray per mask pixel (subsampled) and votes on the hit face."""
    from PIL import Image

    n_obj = 0
    votes: dict[int, np.ndarray] = {}
    total = np.zeros(len(mesh.faces), dtype=np.int32)
    inter = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh) if trimesh.ray.has_embree else trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    for v in views:
        K = np.array(v["K"]); T = np.array(v["T_world_cam"])
        W, H = v["width"], v["height"]
        step = max(1, int(np.sqrt(W * H / 40000)))  # ~40k rays per view
        ys, xs = np.mgrid[0:H:step, 0:W:step]
        pix = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], 1).astype(np.float64)
        dirs_cam = (np.linalg.inv(K) @ pix.T).T
        dirs = dirs_cam @ T[:3, :3].T
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        origins = np.tile(T[:3, 3], (len(dirs), 1))
        loc, ray_idx, tri = inter.intersects_location(origins, dirs, multiple_hits=False)
        hit_face = np.full(len(dirs), -1, dtype=np.int64)
        hit_face[ray_idx] = tri
        stem = Path(v["image"]).stem
        for mp in sorted(masks_dir.glob(f"{stem}_*.png")):
            oid = int(mp.stem.split("_")[-1])
            n_obj = max(n_obj, oid)
            m = np.array(Image.open(mp).convert("L")) > 127
            m_sub = m[ys.ravel(), xs.ravel()]
            f = hit_face[m_sub & (hit_face >= 0)]
            votes.setdefault(oid, np.zeros(len(mesh.faces), dtype=np.int32))
            np.add.at(votes[oid], f, 1)
        np.add.at(total, hit_face[hit_face >= 0], 1)
    labels = np.full(len(mesh.faces), BACKGROUND, dtype=np.int32)
    if not votes:
        return labels
    V = np.stack([votes.get(i, np.zeros(len(mesh.faces), dtype=np.int32)) for i in range(1, n_obj + 1)], 1)
    best = V.argmax(1) + 1
    frac = V.max(1) / np.maximum(total, 1)
    labels[(frac >= vote_thresh) & (V.max(1) > 0)] = best[(frac >= vote_thresh) & (V.max(1) > 0)]
    return labels


# ----------------------------------------------------------------------------- driver
def segment(workdir: Path, mode: str = "auto", masks_dir: Path | None = None,
            views_json: Path | None = None, **kw) -> dict:
    man = Manifest(workdir)
    al = man.step_outputs("align")
    mesh_path = Path(al["mesh"])
    gs_path = Path(al["gaussians"]) if "gaussians" in al else None
    files = [mesh_path] + ([gs_path] if gs_path else []) + ([views_json] if views_json else [])
    params = {"mode": mode, "masks_dir": str(masks_dir) if masks_dir else None, **kw}
    key = hash_inputs(files, params)
    if man.step_cached("segment", key):
        return man.step_outputs("segment")
    man.step_begin("segment", key, params)

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    plane_z = man.scene["plane_z"]
    if mode == "auto":
        labels = segment_auto(mesh, plane_z, **kw)
    elif mode == "masks":
        views = json.loads(Path(views_json).read_text())
        labels = lift_masks_to_faces(mesh, views, Path(masks_dir), kw.get("vote_thresh", 0.7))
    else:
        raise ValueError(mode)

    out = Path(workdir) / "segmented"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "face_labels.npy", labels)

    # gaussians -> nearest labelled face centroid
    g = rio.read_gaussians(gs_path) if gs_path else None
    glabels = None
    if g is not None:
        glabels = label_points_by_surface(mesh, labels, g.xyz.astype(np.float64))
        np.save(out / "gauss_labels.npy", glabels)

    objects = {}
    ids = [int(i) for i in np.unique(labels) if i > 0]
    for i in ids:
        sel = np.where(labels == i)[0]
        om = mesh.submesh([sel], append=True, repair=False)
        oid = f"obj_{i:02d}"
        od = out / oid
        od.mkdir(exist_ok=True)
        om.export(od / "raw_mesh.ply")
        rec = {"id": i, "faces": int(len(sel)), "bbox": om.bounds.tolist(),
               "extents": om.extents.tolist(), "raw_mesh": str(od / "raw_mesh.ply")}
        if g is not None:
            gs_sub = g.subset(glabels == i)
            rio.write_gaussians(od / "raw_gs.ply", gs_sub)
            rec["raw_gs"] = str(od / "raw_gs.ply"); rec["gaussians"] = int(len(gs_sub))
        objects[oid] = rec
        man.set_object(oid, **rec, type="rigid")

    bg_sel = np.where(labels == BACKGROUND)[0]
    bg = mesh.submesh([bg_sel], append=True, repair=False)
    bg.export(out / "background_raw.ply")
    outputs = {"objects": objects, "background_raw": str(out / "background_raw.ply"),
               "face_labels": str(out / "face_labels.npy"),
               "n_floater_faces": int((labels == FLOATER).sum())}
    if g is not None:
        rio.write_gaussians(out / "background_gs.ply", g.subset(glabels == BACKGROUND))
        outputs["background_gs"] = str(out / "background_gs.ply")
    man.step_end("segment", outputs)
    (out / "segmentation.json").write_text(json.dumps(outputs, indent=2))
    return outputs


def score_against_gt(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Hungarian-free scoring: for each GT object pick the best-overlapping predicted id."""
    res = {}
    for g in np.unique(gt):
        if g <= 0:
            continue
        gm = gt == g
        cands = np.unique(pred[gm])
        best, best_iou = None, 0.0
        for c in cands:
            if c <= 0:
                continue
            pm = pred == c
            iou = (gm & pm).sum() / (gm | pm).sum()
            if iou > best_iou:
                best, best_iou = int(c), float(iou)
        res[int(g)] = {"pred": best, "iou": best_iou}
    bg_gt = gt == 0
    bg_pred = pred == 0
    res["background_iou"] = float((bg_gt & bg_pred).sum() / (bg_gt | bg_pred).sum())
    return res
