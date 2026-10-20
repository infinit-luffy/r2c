"""Step 6 `pack`: write Isaac Lab-ready USD assets.

Per object  assets/<obj>.usdc
  /<obj>                 Xform + PhysicsRigidBodyAPI + PhysicsMassAPI (mass, CoM)
    /visual              Mesh (vertex displayColor) + UsdPreviewSurface material
    /collision_NN        Mesh, PhysicsCollisionAPI, convexHull approximation, purpose=guide
    /physics_material    PhysicsMaterialAPI (friction, restitution)
Background  assets/background.usdc   static Mesh + collision (no rigid body)
Scene       assets/scene.usda        references everything at its scanned pose
            assets/scene_cfg.py      Isaac Lab InteractiveSceneCfg snippet

Object meshes are written in the object's local frame (origin at the bottom-centre
of the bbox) and placed with an xformOp:translate in scene.usda, so each asset is
reusable and Isaac Lab's RigidObjectCfg.init_state.pos is simply that translate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import trimesh

from .manifest import Manifest, hash_inputs

FRICTION = {"rigid": (0.6, 0.5), "thin": (0.6, 0.5), "fuzzy": (0.9, 0.8), "capsule": (0.5, 0.4)}


def _write_mesh_prim(stage, path: str, mesh: trimesh.Trimesh, color: bool = True):
    from pxr import UsdGeom, Vt, Sdf

    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(mesh.faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).ravel()))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateExtentAttr(Vt.Vec3fArray([tuple(mesh.bounds[0].astype(float)), tuple(mesh.bounds[1].astype(float))]))
    if color and mesh.visual.kind == "vertex":
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec3fArray.FromNumpy((mesh.visual.vertex_colors[:, :3] / 255.0).astype(np.float32)))
    nrm = UsdGeom.PrimvarsAPI(m).CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, UsdGeom.Tokens.vertex)
    nrm.Set(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertex_normals, dtype=np.float32)))
    return m


def _write_material(stage, path: str, albedo, roughness: float = 0.6, metallic: float = 0.0):
    from pxr import UsdShade, Sdf, Gf

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(x) for x in albedo]))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _add_collision(prim, approx: str = "convexHull"):
    from pxr import UsdPhysics, UsdGeom

    UsdPhysics.CollisionAPI.Apply(prim)
    mc = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mc.CreateApproximationAttr(approx)
    UsdGeom.Imageable(prim).CreatePurposeAttr(UsdGeom.Tokens.guide)


def write_object_usd(path: Path, name: str, visual: trimesh.Trimesh, hulls: list[trimesh.Trimesh],
                     mass: float, otype: str, albedo) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf

    # local frame: bottom-centre of the visual bbox
    origin = np.array([visual.centroid[0], visual.centroid[1], visual.bounds[0][2]])
    vis = visual.copy(); vis.apply_translation(-origin)
    hls = [h.copy() for h in hulls]
    for h in hls:
        h.apply_translation(-origin)

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}")
    prim = root.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    massapi = UsdPhysics.MassAPI.Apply(prim)
    massapi.CreateMassAttr(float(mass))
    com = vis.center_mass if vis.is_watertight else vis.centroid
    massapi.CreateCenterOfMassAttr(Gf.Vec3f(*[float(x) for x in com]))

    v = _write_mesh_prim(stage, f"/{name}/visual", vis)
    mat = _write_material(stage, f"/{name}/Looks/material", albedo)
    UsdShade.MaterialBindingAPI.Apply(v.GetPrim()).Bind(mat)

    pm = UsdShade.Material.Define(stage, f"/{name}/physics_material")
    pmat = UsdPhysics.MaterialAPI.Apply(pm.GetPrim())
    sf, df = FRICTION.get(otype, FRICTION["rigid"])
    pmat.CreateStaticFrictionAttr(sf); pmat.CreateDynamicFrictionAttr(df); pmat.CreateRestitutionAttr(0.0)
    for i, h in enumerate(hls):
        c = _write_mesh_prim(stage, f"/{name}/collision_{i:02d}", h, color=False)
        _add_collision(c.GetPrim(), "convexHull")
        UsdShade.MaterialBindingAPI.Apply(c.GetPrim()).Bind(pm, materialPurpose="physics")
    stage.SetDefaultPrim(prim)
    stage.Save()
    return {"origin": origin.tolist(), "extents": vis.extents.tolist()}


def write_background_usd(path: Path, mesh: trimesh.Trimesh) -> None:
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/background")
    v = _write_mesh_prim(stage, "/background/visual", mesh)
    mat = _write_material(stage, "/background/Looks/material", mesh.visual.vertex_colors[:, :3].mean(0) / 255.0 if mesh.visual.kind == "vertex" else (0.5, 0.5, 0.5), roughness=0.7)
    UsdShade.MaterialBindingAPI.Apply(v.GetPrim()).Bind(mat)
    c = _write_mesh_prim(stage, "/background/collision", mesh, color=False)
    _add_collision(c.GetPrim(), "none")  # static triangle mesh collider
    stage.SetDefaultPrim(root.GetPrim())
    stage.Save()


def write_scene_usda(path: Path, objects: dict[str, dict], background_file: str) -> None:
    from pxr import Usd, UsdGeom, Gf

    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    bg = UsdGeom.Xform.Define(stage, "/World/background")
    bg.GetPrim().GetReferences().AddReference(f"./{background_file}")
    for oid, rec in objects.items():
        x = UsdGeom.Xform.Define(stage, f"/World/{oid}")
        x.GetPrim().GetReferences().AddReference(f"./{oid}.usdc")
        x.AddTranslateOp().Set(Gf.Vec3d(*rec["origin"]))
    stage.SetDefaultPrim(world.GetPrim())
    stage.Save()


SCENE_CFG_TEMPLATE = '''"""Auto-generated by r2s pack. Drop into an Isaac Lab project."""
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

ASSET_DIR = "{asset_dir}"


@configclass
class ScannedTabletopSceneCfg(InteractiveSceneCfg):
    background = AssetBaseCfg(
        prim_path="/World/background",
        spawn=sim_utils.UsdFileCfg(usd_path=f"{{ASSET_DIR}}/background.usdc"),
    )
    dome = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1000.0),  # swap for texture_file=<site HDRI>
    )
{objects}
'''

OBJ_CFG_TEMPLATE = '''    {oid} = RigidObjectCfg(
        prim_path="{{ENV_REGEX_NS}}/{oid}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{{ASSET_DIR}}/{oid}.usdc",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass={mass:.4f}),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=({x:.4f}, {y:.4f}, {z:.4f})),
    )
'''


def pack(workdir: Path, asset_dir: Path | None = None) -> dict:
    man = Manifest(workdir)
    tx = man.step_outputs("texture")
    cl = man.step_outputs("clean")
    files = [Path(o["visual"]) for o in tx["objects"].values()] + [Path(tx["background"])]
    for o in cl["objects"].values():
        files += [Path(c) for c in o["collision"]]
    params = {"asset_dir": str(asset_dir) if asset_dir else None}
    key = hash_inputs(files, params)
    if man.step_cached("pack", key):
        return man.step_outputs("pack")
    man.step_begin("pack", key, params)
    out = Path(asset_dir) if asset_dir else Path(workdir) / "assets"
    out.mkdir(parents=True, exist_ok=True)

    objects = {}
    for oid, rec in cl["objects"].items():
        vis = trimesh.load(tx["objects"][oid]["visual"], force="mesh", process=False)
        hulls = [trimesh.load(c, force="mesh", process=False) for c in rec["collision"]]
        info = write_object_usd(out / f"{oid}.usdc", oid, vis, hulls, rec["mass"], rec["type"],
                                tx["objects"][oid]["mean_albedo"])
        info.update({"mass": rec["mass"], "type": rec["type"], "usd": str(out / f"{oid}.usdc"),
                     "n_collision": len(hulls)})
        objects[oid] = info
        man.set_object(oid, usd=info["usd"], origin=info["origin"])
    bg = trimesh.load(tx["background"], force="mesh", process=False)
    write_background_usd(out / "background.usdc", bg)
    write_scene_usda(out / "scene.usda", objects, "background.usdc")
    cfg = SCENE_CFG_TEMPLATE.format(
        asset_dir=str(out.resolve()),
        objects="".join(OBJ_CFG_TEMPLATE.format(oid=o, mass=r["mass"], x=r["origin"][0], y=r["origin"][1], z=r["origin"][2])
                        for o, r in objects.items()))
    (out / "scene_cfg.py").write_text(cfg)
    outputs = {"objects": objects, "background": str(out / "background.usdc"),
               "scene": str(out / "scene.usda"), "scene_cfg": str(out / "scene_cfg.py")}
    man.step_end("pack", outputs)
    (out / "assets.json").write_text(json.dumps(outputs, indent=2))
    return outputs
