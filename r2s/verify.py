"""Step 7 `verify`: is the packed scene physically usable?

Static checks (per object)
  watertight          visual mesh closed
  base_on_plane       lowest point within `base_tol` of the support plane
  volume_sane         volume > 0 and consistent with bbox (not a degenerate shell)
  hull_count          <= max hulls
Dynamic check (MuJoCo, headless)
  drop_test           place objects at scanned pose on the support plane, simulate
                      `duration` seconds; pass if translation < `max_drift` and tilt
                      < `max_tilt_deg`; also report inter-object penetration.

Thresholds live in VERIFY_DEFAULTS and are *not* exposed to any automated fixer:
a fixer may change assets, never the pass criteria.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from .manifest import Manifest

VERIFY_DEFAULTS = {
    "base_tol": 0.004,      # m
    "max_hulls": 16,
    "duration": 2.0,        # s
    "max_drift": 0.005,     # m
    "max_tilt_deg": 3.0,
    "max_penetration": 0.002,  # m
}


def static_checks(man: Manifest, thr: dict) -> dict:
    cl = man.step_outputs("clean")
    plane_z = man.scene["plane_z"]
    res = {}
    for oid, rec in cl["objects"].items():
        m = trimesh.load(rec["visual"], force="mesh", process=False)
        zmin = float(m.bounds[0][2])
        bbox_vol = float(np.prod(m.extents))
        r = {
            "watertight": bool(m.is_watertight),
            "base_on_plane": abs(zmin - plane_z) <= thr["base_tol"],
            "base_offset": zmin - plane_z,
            "volume_sane": bool(m.is_watertight and m.volume > 1e-7 and m.volume <= bbox_vol * 1.01),
            "volume": float(m.volume) if m.is_watertight else None,
            "hull_count": len(rec["collision"]),
            "hull_count_ok": len(rec["collision"]) <= thr["max_hulls"],
        }
        r["pass"] = all(r[k] for k in ("watertight", "base_on_plane", "volume_sane", "hull_count_ok"))
        res[oid] = r
    return res


def _quat_tilt_deg(q_wxyz: np.ndarray) -> float:
    w, x, y, z = q_wxyz
    # angle between body z-axis and world z
    zz = 1 - 2 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(zz, -1, 1))))


def drop_test(man: Manifest, thr: dict) -> dict:
    import mujoco

    cl = man.step_outputs("clean")
    pk = man.step_outputs("pack")
    plane_z = man.scene["plane_z"]
    tmp = Path(tempfile.mkdtemp(prefix="r2s_verify_"))
    meshes_xml = []
    bodies_xml = []
    for oid, rec in cl["objects"].items():
        origin = np.array(pk["objects"][oid]["origin"])
        geoms = []
        hulls = [trimesh.load(c, force="mesh", process=False) for c in rec["collision"]]
        density = rec["mass"] / max(sum(h.volume for h in hulls), 1e-9)  # mass ∝ hull volume
        for i, h in enumerate(hulls):
            h = h.copy(); h.apply_translation(-origin)
            name = f"{oid}_c{i:02d}"
            h.export(tmp / f"{name}.obj")
            meshes_xml.append(f'<mesh name="{name}" file="{name}.obj"/>')
            geoms.append(f'<geom type="mesh" mesh="{name}" density="{density:.3f}" '
                         f'friction="0.6 0.005 0.0001" condim="3"/>')
        bodies_xml.append(
            f'<body name="{oid}" pos="{origin[0]:.6f} {origin[1]:.6f} {origin[2]:.6f}">'
            f'<freejoint/>{"".join(geoms)}</body>')
    xml = f"""
<mujoco model="r2s_verify">
  <compiler meshdir="{tmp}" angle="degree"/>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>{''.join(meshes_xml)}</asset>
  <worldbody>
    <geom type="plane" pos="0 0 {plane_z:.6f}" size="5 5 0.1" friction="0.6 0.005 0.0001"/>
    {''.join(bodies_xml)}
  </worldbody>
</mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    start = {oid: data.body(oid).xpos.copy() for oid in cl["objects"]}
    steps = int(thr["duration"] / model.opt.timestep)
    max_pen = 0.0
    for _ in range(steps):
        mujoco.mj_step(model, data)
        for k in range(data.ncon):
            c = data.contact[k]
            if c.dist < -max_pen and model.geom_bodyid[c.geom1] != 0 and model.geom_bodyid[c.geom2] != 0:
                max_pen = -c.dist
    res = {}
    for oid in cl["objects"]:
        pos = data.body(oid).xpos
        q = data.body(oid).xquat
        drift = float(np.linalg.norm(pos - start[oid]))
        tilt = _quat_tilt_deg(np.array(q))
        res[oid] = {"drift": drift, "tilt_deg": tilt,
                    "pass": drift <= thr["max_drift"] and tilt <= thr["max_tilt_deg"]}
    res["max_inter_object_penetration"] = max_pen
    res["penetration_ok"] = max_pen <= thr["max_penetration"]
    return res


def verify(workdir: Path, skip_dynamic: bool = False) -> dict:
    man = Manifest(workdir)
    thr = dict(VERIFY_DEFAULTS)
    report = {"thresholds": thr, "static": static_checks(man, thr)}
    if not skip_dynamic:
        try:
            report["drop_test"] = drop_test(man, thr)
        except ImportError:
            report["drop_test"] = {"error": "mujoco not installed (pip install mujoco)"}
    per_obj = {}
    for oid, s in report["static"].items():
        d = report.get("drop_test", {}).get(oid, {"pass": True})
        per_obj[oid] = {"static": s["pass"], "dynamic": d.get("pass", True),
                        "pass": s["pass"] and d.get("pass", True)}
        man.set_object(oid, verify=per_obj[oid])
    report["objects"] = per_obj
    report["all_pass"] = all(v["pass"] for v in per_obj.values()) and report.get("drop_test", {}).get("penetration_ok", True)
    out = Path(workdir) / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(json.dumps(report, indent=2))
    man.data["steps"]["verify"] = {"status": "ok" if report["all_pass"] else "failed", "outputs": {"report": str(out / "verify.json")}}
    man.save()
    return report
