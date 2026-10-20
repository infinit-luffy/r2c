# real2sim-assets

Turn a **holistic scene scan** (one 3DGS `.ply` + one mesh `.obj/.ply` + optionally a `.usd`)
into **per-object, simulation-ready USD assets** that load straight into Isaac Lab as
rigid bodies, with a physics check that proves they stand up.

```
scan (GS + mesh)  ──►  align  ──►  segment  ──►  clean  ──►  texture  ──►  pack  ──►  verify
                       Z-up/m      objects      watertight   albedo       USD +       MuJoCo
                       table at z  + floaters   + closed     from GS     Isaac Lab    drop test
                                   + background  bottom      (delight)   cfg
                                                 + CoACD
```

Everything is deterministic, cached per step (input hash + code hash in `manifest.json`),
and every step leaves inspectable files on disk. No ML in the core path; SAM 2 is an
optional plug-in for segmentation.

## Install

```bash
pip install -e .            # core: trimesh, coacd, pymeshfix, usd-core, shapely, scikit-learn
pip install mujoco          # for `r2s verify` drop test (optional but recommended)
```

## Quick start on the synthetic fixture

```bash
r2s synth tests/data/synth                       # table + mug/box/ball/plate, scanned-looking
r2s run -w work/synth --mesh tests/data/synth/scene_mesh.obj --gs tests/data/synth/scene_gs.ply
```

Output:

```
work/synth/assets/
  background.usdc      static table/floor: visual mesh + triangle collider
  obj_01.usdc ...      RigidBodyAPI + MassAPI, /visual (vertex colour), /collision_NN (convex), physics material
  scene.usda           references every asset at its scanned pose
  scene_cfg.py         Isaac Lab InteractiveSceneCfg with one RigidObjectCfg per object
work/synth/report/
  verify.json          static checks + drop test numbers
  segmentation.png     top-down / side views of what got split
```

```
obj_01: PASS  watertight=True base_off=-0.5mm hulls=1  drift=0.5mm tilt=0.0deg
obj_02: PASS  watertight=True base_off=-0.5mm hulls=16 drift=0.8mm tilt=0.0deg
...
ALL PASS: True
```

## On your own scan

```bash
r2s inspect --mesh scene.obj --gs scene_gs.ply --usd scene.usd      # what's in there, do scales agree
r2s align   -w work/kitchen --mesh scene.obj --gs scene_gs.ply [--scale 0.01] [--table-height 0.75]
r2s segment -w work/kitchen                                           # auto: plane + clustering
r2s types   -w work/kitchen obj_01=rigid obj_03=thin obj_04=fuzzy     # optional per-object strategy
r2s clean   -w work/kitchen
r2s texture -w work/kitchen --delight lambert
r2s pack    -w work/kitchen --asset-dir assets/kitchen
r2s verify  -w work/kitchen
```

`align` finds the floor and the table top as parallel planes; if both are visible the
table height is recovered automatically, otherwise pass `--table-height`. Pass `--scale`
when the scan is not in metres (check `inspect` extents against a known object).

### Segmentation modes

| mode  | what it needs | when |
|-------|---------------|------|
| `auto` (default) | nothing | tabletop, objects not stacked / touching |
| `masks` | per-view masks + camera poses | cluttered scenes; run `scripts/run_sam2_views.py` first, then `r2s segment --mode masks --masks-dir masks/ --views views.json` |

In `masks` mode each face gets a vote per view by ray casting; faces with < 70 % agreement
go to the background (better a slightly small object than one with a ring of table on it).

### Object types

| type | clean strategy |
|------|----------------|
| `rigid` | close bottom, repair, decimate, CoACD convex decomposition (≤16 hulls) |
| `thin` | same, collision = single convex hull (CoACD over-fragments plates) |
| `fuzzy` | collision shell shrunk by 5 mm, low density |
| `capsule` | placeholder (hull) – capsule-chain fitting for cables is on the list |

### What `verify` checks (thresholds in `r2s/verify.py`, not CLI-tunable on purpose)

- watertight visual mesh; volume consistent with bbox
- lowest point within 4 mm of the support plane
- ≤ 16 collision hulls
- MuJoCo drop test: 2 s at the scanned pose → drift < 5 mm, tilt < 3°, no inter-object penetration > 2 mm

The drop test is what catches the real failures: a single noisy vertex 2 mm below the
base makes a mug stand on a point and fall over; that is now snapped to the plane in `clean`.

## Design notes

- **Bottom closing** is the step that makes a scan usable in physics: the scanned shell is
  open underneath. Rings already on the table are snapped flat; rings higher up (a ball)
  get a vertical skirt down to the plane plus a cap, so the object stands exactly where
  it was scanned.
- **Delighting** (`--delight lambert`) fits `luminance ≈ c0 + c·n` over all gaussians and
  divides it out, removing the dominant directional shading baked into the scan so RTX
  lighting in Isaac doesn't double-shade. Proper HDRI-based delighting is the next step.
- **Gaussians are kept per object** (`segmented/obj_k/raw_gs.ply`) so the same split can
  feed a GS-rendered background or a NuRec/ParticleField variant later.
- **No thresholds are exposed to automation.** A future fixer agent may change assets and
  parameters of `clean`/`segment`; it cannot change what counts as passing.

## Roadmap

1. Photo-projection texture baking (xatlas UV + source frames) and HDRI-based delighting (Blender)
2. Capsule chain fitting for thin structures
3. Background hole filling from GS (currently a flat patch at the footprint)
4. Articulated objects (URDF joints) – out of scope for v0
5. Fixer loop: re-run `clean` with alternative parameters when `verify` fails, log what worked

## Layout

```
r2s/
  manifest.py      step cache + object registry
  io.py            3DGS ply / mesh / USD readers & writers
  inspect_align.py inspect + align (plane detection, Z-up, metres)
  segment.py       auto + mask-vote segmentation, GS labelling
  clean.py         fragments, bottom closing, watertight, decimate, CoACD
  texture.py       GS → vertex colours, lambert delight
  pack.py          USD assets, scene.usda, Isaac Lab cfg
  verify.py        static checks + MuJoCo drop test
  report.py        segmentation.png
  synth.py         synthetic scanned tabletop with ground truth
scripts/run_sam2_views.py   SAM 2 mask generation (optional)
tests/test_pipeline.py      end-to-end on the fixture
```
