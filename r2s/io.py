"""Readers for the three kinds of input a scan pipeline typically leaves behind:
3DGS .ply (gaussians), plain .ply point clouds, .obj/.glb/.ply meshes and .usd scenes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from plyfile import PlyData

SH_C0 = 0.28209479177387814


@dataclass
class Gaussians:
    """Minimal 3DGS container. Extra per-gaussian fields (SH rest, etc.) are kept
    in `extra` so they survive a round trip when we export per-object subsets."""

    xyz: np.ndarray  # (N,3)
    rgb: np.ndarray  # (N,3) in [0,1], derived from f_dc
    opacity: np.ndarray  # (N,) sigmoid-activated
    scale: np.ndarray  # (N,3) exp-activated
    rot: np.ndarray  # (N,4) wxyz
    extra: dict[str, np.ndarray] = field(default_factory=dict)
    raw_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.xyz)

    def subset(self, mask: np.ndarray) -> "Gaussians":
        return Gaussians(
            xyz=self.xyz[mask],
            rgb=self.rgb[mask],
            opacity=self.opacity[mask],
            scale=self.scale[mask],
            rot=self.rot[mask],
            extra={k: v[mask] for k, v in self.extra.items()},
            raw_names=list(self.raw_names),
        )

    def transformed(self, T: np.ndarray) -> "Gaussians":
        """Apply 4x4 transform (rotation+uniform scale+translation)."""
        R = T[:3, :3]
        s = np.cbrt(abs(np.linalg.det(R)))
        Rn = R / s
        xyz = self.xyz @ R.T + T[:3, 3]
        scale = self.scale * s
        rot = _quat_mul(_rotmat_to_quat(Rn)[None], self.rot)
        return Gaussians(xyz, self.rgb.copy(), self.opacity.copy(), scale, rot,
                         {k: v.copy() for k, v in self.extra.items()}, list(self.raw_names))


def is_gaussian_ply(path: Path) -> bool:
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        return False
    names = {p.name for p in ply["vertex"].properties}
    return {"f_dc_0", "opacity", "scale_0", "rot_0"} <= names


def read_gaussians(path: Path) -> Gaussians:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    names = [p.name for p in v.properties]
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float32)
    f_dc = np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], -1).astype(np.float32)
    rgb = np.clip(0.5 + SH_C0 * f_dc, 0, 1)
    opacity = 1.0 / (1.0 + np.exp(-v["opacity"].astype(np.float32)))
    scale = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], -1).astype(np.float32))
    rot = np.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], -1).astype(np.float32)
    rot /= np.linalg.norm(rot, axis=1, keepdims=True) + 1e-9
    core = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
    extra = {n: np.asarray(v[n]) for n in names if n not in core}
    return Gaussians(xyz, rgb, opacity, scale, rot, extra, names)


def write_gaussians(path: Path, g: Gaussians) -> None:
    """Write back in standard 3DGS ply layout (inverse-activated fields)."""
    from plyfile import PlyElement

    n = len(g)
    f_dc = (g.rgb - 0.5) / SH_C0
    op = np.clip(g.opacity, 1e-6, 1 - 1e-6)
    opacity = np.log(op / (1 - op))
    scale = np.log(np.maximum(g.scale, 1e-9))
    cols: dict[str, np.ndarray] = {
        "x": g.xyz[:, 0], "y": g.xyz[:, 1], "z": g.xyz[:, 2],
        "nx": np.zeros(n), "ny": np.zeros(n), "nz": np.zeros(n),
        "f_dc_0": f_dc[:, 0], "f_dc_1": f_dc[:, 1], "f_dc_2": f_dc[:, 2],
    }
    # keep SH rest if present
    rest = sorted((k for k in g.extra if k.startswith("f_rest_")), key=lambda s: int(s.split("_")[-1]))
    for k in rest:
        cols[k] = g.extra[k]
    cols.update({
        "opacity": opacity,
        "scale_0": scale[:, 0], "scale_1": scale[:, 1], "scale_2": scale[:, 2],
        "rot_0": g.rot[:, 0], "rot_1": g.rot[:, 1], "rot_2": g.rot[:, 2], "rot_3": g.rot[:, 3],
    })
    dtype = [(k, "f4") for k in cols]
    arr = np.empty(n, dtype=dtype)
    for k, vals in cols.items():
        arr[k] = np.asarray(vals, dtype=np.float32)
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(path))


def read_pointcloud(path: Path) -> trimesh.PointCloud:
    ply = PlyData.read(str(path))
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)
    colors = None
    names = {p.name for p in v.properties}
    if {"red", "green", "blue"} <= names:
        colors = np.stack([v["red"], v["green"], v["blue"]], -1)
    return trimesh.PointCloud(xyz, colors=colors)


def read_mesh(path: Path) -> trimesh.Trimesh:
    """Load any mesh file as ONE Trimesh (scenes are concatenated).
    Face groups from OBJ `o`/`g` are preserved in metadata['groups'] when present."""
    m = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()])
    return m


def read_mesh_groups(path: Path) -> dict[str, trimesh.Trimesh]:
    """Load keeping OBJ groups / scene nodes separate (useful for inspect)."""
    s = trimesh.load(str(path), force="scene", process=False)
    if isinstance(s, trimesh.Trimesh):
        return {"mesh": s}
    return dict(s.geometry)


def read_usd_summary(path: Path) -> dict:
    """Summarise a USD stage: prims, mesh counts, units, bbox."""
    from pxr import Usd, UsdGeom, Gf

    stage = Usd.Stage.Open(str(path))
    out = {
        "upAxis": UsdGeom.GetStageUpAxis(stage),
        "metersPerUnit": UsdGeom.GetStageMetersPerUnit(stage),
        "defaultPrim": stage.GetDefaultPrim().GetPath().pathString if stage.GetDefaultPrim() else None,
        "prims": [],
    }
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    total = Gf.Range3d()
    for prim in stage.Traverse():
        entry = {"path": prim.GetPath().pathString, "type": prim.GetTypeName()}
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            entry["faces"] = len(mesh.GetFaceVertexCountsAttr().Get() or [])
            entry["points"] = len(mesh.GetPointsAttr().Get() or [])
            entry["rigid_body"] = prim.HasAPI("PhysicsRigidBodyAPI") or prim.GetParent().HasAPI("PhysicsRigidBodyAPI")
            r = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            total.UnionWith(r)
            entry["bbox"] = [list(r.GetMin()), list(r.GetMax())]
        if prim.IsA(UsdGeom.Points) or prim.GetTypeName() in ("ParticleField3DGaussianSplat", "OmniNuRecFieldAsset"):
            entry["gaussian_like"] = True
        out["prims"].append(entry)
    if not total.IsEmpty():
        out["bbox"] = [list(total.GetMin()), list(total.GetMax())]
    return out


def usd_to_trimesh(path: Path) -> trimesh.Trimesh | None:
    """Flatten all UsdGeom.Mesh prims into one Trimesh (world space)."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    xf = UsdGeom.XformCache(Usd.TimeCode.Default())
    parts = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        cnt = np.array(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=int)
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=int)
        if len(pts) == 0 or len(cnt) == 0:
            continue
        faces = _triangulate(cnt, idx)
        M = np.array(xf.GetLocalToWorldTransform(prim)).T  # pxr matrices are row-vector
        pts = pts @ M[:3, :3].T + M[:3, 3]
        parts.append(trimesh.Trimesh(pts, faces, process=False))
    if not parts:
        return None
    return trimesh.util.concatenate(parts)


def _triangulate(counts: np.ndarray, indices: np.ndarray) -> np.ndarray:
    faces = []
    i = 0
    for c in counts:
        poly = indices[i:i + c]
        for k in range(1, c - 1):
            faces.append([poly[0], poly[k], poly[k + 1]])
        i += c
    return np.array(faces, dtype=int)


# ---- small quaternion helpers (wxyz) ----
def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    w2, x2, y2, z2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], -1)


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
