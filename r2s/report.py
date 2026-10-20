"""Visual report: top-down and side views of the segmentation + cleaned assets so a
human (or an agent) can check a scene without opening a 3D viewer."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .manifest import Manifest


def render_report(workdir: Path) -> Path | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    man = Manifest(workdir)
    seg = man.step_outputs("segment")
    mesh = trimesh.load(man.step_outputs("align")["mesh"], force="mesh", process=False)
    labels = np.load(seg["face_labels"])
    pts, fid = trimesh.sample.sample_surface(mesh, 120000, seed=0)
    lab = labels[fid]
    plane_z = man.scene["plane_z"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap = plt.get_cmap("tab10")

    def scatter(ax, x, y, l, title):
        bg = l == 0
        ax.scatter(x[bg], y[bg], s=0.3, c="#bbbbbb", label="background")
        fl = l == -1
        if fl.any():
            ax.scatter(x[fl], y[fl], s=1.5, c="k", marker="x", label="floater")
        for i in sorted(set(l[l > 0])):
            m = l == i
            ax.scatter(x[m], y[m], s=0.6, color=cmap((i - 1) % 10), label=f"obj_{i:02d}")
        ax.set_aspect("equal"); ax.set_title(title); ax.legend(markerscale=10, fontsize=8)

    near = np.abs(pts[:, 2] - plane_z) < 0.6
    scatter(axes[0], pts[near, 0], pts[near, 1], lab[near], "segmentation, top-down (x,y)")
    scatter(axes[1], pts[near, 0], pts[near, 2], lab[near], "side view (x,z)")
    axes[1].axhline(plane_z, color="r", lw=0.5)

    # cleaned objects silhouettes
    ax = axes[2]
    cl = man.data["steps"].get("clean", {}).get("outputs")
    if cl:
        for i, (oid, r) in enumerate(cl["objects"].items()):
            m = trimesh.load(r["visual"], force="mesh", process=False)
            p, _ = trimesh.sample.sample_surface(m, 4000, seed=0)
            ax.scatter(p[:, 0], p[:, 2], s=0.6, color=cmap(i % 10), label=f"{oid} ({len(r['collision'])} hulls)")
        ax.axhline(plane_z, color="r", lw=0.5, label="support plane")
        ax.set_aspect("equal"); ax.set_title("cleaned (closed) objects, side view"); ax.legend(fontsize=8)
    out = Path(workdir) / "report"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "segmentation.png"
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)
    return path
