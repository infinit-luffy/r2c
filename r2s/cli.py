"""`r2s` command line.

  r2s inspect  --mesh scene.obj --gs scene_gs.ply --usd scene.usd
  r2s align    -w work/scene --mesh scene.obj --gs scene_gs.ply [--table-height 0.75] [--scale 1.0]
  r2s segment  -w work/scene [--mode auto|masks --masks-dir .. --views ..]
  r2s types    -w work/scene obj_01=rigid obj_02=thin obj_03=fuzzy
  r2s clean    -w work/scene
  r2s texture  -w work/scene [--delight lambert|none]
  r2s pack     -w work/scene [--asset-dir assets/]
  r2s verify   -w work/scene [--skip-dynamic]
  r2s run      -w work/scene --mesh .. --gs ..      # align -> verify in one go
  r2s synth    tests/data/synth                     # generate the synthetic fixture
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _p(x):
    return Path(x) if x else None


def cmd_inspect(a):
    from .inspect_align import inspect_inputs

    rep = inspect_inputs(_p(a.mesh), _p(a.gs), _p(a.usd))
    if "usd" in rep:
        rep["usd"]["prims"] = rep["usd"]["prims"][:10]
    print(json.dumps(rep, indent=2))


def cmd_align(a):
    from .inspect_align import align

    out = align(Path(a.workdir), _p(a.mesh), _p(a.gs), _p(a.usd), scale=a.scale, table_height=a.table_height)
    print(json.dumps({k: v for k, v in out.items() if k != "transform"}, indent=2))


def cmd_segment(a):
    from .segment import segment

    kw = {}
    if a.cluster_eps is not None:
        kw["cluster_eps"] = a.cluster_eps
    out = segment(Path(a.workdir), mode=a.mode, masks_dir=_p(a.masks_dir), views_json=_p(a.views), **kw)
    print(f"objects: {len(out['objects'])}, floater faces: {out['n_floater_faces']}")
    for oid, r in out["objects"].items():
        e = r["extents"]
        print(f"  {oid}: faces={r['faces']} extents={e[0]:.3f}x{e[1]:.3f}x{e[2]:.3f}")


def cmd_types(a):
    from .manifest import Manifest

    man = Manifest(Path(a.workdir))
    for spec in a.assignments:
        oid, t = spec.split("=")
        assert t in ("rigid", "thin", "fuzzy", "capsule"), t
        man.set_object(oid, type=t)
    print({k: v.get("type") for k, v in man.objects().items()})


def cmd_clean(a):
    from .clean import clean

    out = clean(Path(a.workdir), target_faces=a.target_faces, coacd_threshold=a.coacd_threshold, max_hulls=a.max_hulls)
    for oid, r in out["objects"].items():
        print(f"  {oid}: watertight={r['watertight']} faces={r['faces']} hulls={len(r['collision'])} mass={r['mass']:.3f}kg")


def cmd_texture(a):
    from .texture import texture

    out = texture(Path(a.workdir), delight=a.delight)
    m = out.get("shading_model")
    if m:
        print(f"  shading: light_dir={[round(x, 2) for x in m['light_dir']]} strength={m['strength']:.2f}")
    print(f"  textured {len(out['objects'])} objects + background")


def cmd_pack(a):
    from .pack import pack

    out = pack(Path(a.workdir), _p(a.asset_dir))
    print(f"  scene: {out['scene']}\n  isaac lab cfg: {out['scene_cfg']}")


def cmd_verify(a):
    from .verify import verify
    from .report import render_report

    r = verify(Path(a.workdir), skip_dynamic=a.skip_dynamic)
    img = render_report(Path(a.workdir))
    if img:
        print(f"  report image: {img}")
    for oid, v in r["objects"].items():
        s = r["static"][oid]
        d = r.get("drop_test", {}).get(oid, {})
        extra = f" drift={d['drift'] * 1000:.1f}mm tilt={d['tilt_deg']:.1f}deg" if "drift" in d else ""
        print(f"  {oid}: {'PASS' if v['pass'] else 'FAIL'}  watertight={s['watertight']} base_off={s['base_offset'] * 1000:.1f}mm hulls={s['hull_count']}{extra}")
    print(f"ALL PASS: {r['all_pass']}  (report: {Path(a.workdir) / 'report' / 'verify.json'})")
    sys.exit(0 if r["all_pass"] else 1)


def cmd_run(a):
    cmd_align(a)
    cmd_segment(a)
    cmd_clean(a)
    cmd_texture(a)
    cmd_pack(a)
    cmd_verify(a)


def cmd_synth(a):
    from .synth import make_scene

    gt = make_scene(Path(a.out), seed=a.seed)
    print(f"wrote synthetic scene to {a.out}: {list(gt['objects'])}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="r2s", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_inputs(sp):
        sp.add_argument("--mesh"); sp.add_argument("--gs"); sp.add_argument("--usd")

    def add_work(sp):
        sp.add_argument("-w", "--workdir", required=True)

    s = sub.add_parser("inspect"); add_inputs(s); s.set_defaults(f=cmd_inspect)
    s = sub.add_parser("align"); add_work(s); add_inputs(s)
    s.add_argument("--scale", type=float); s.add_argument("--table-height", type=float); s.set_defaults(f=cmd_align)
    s = sub.add_parser("segment"); add_work(s)
    s.add_argument("--mode", default="auto", choices=["auto", "masks"])
    s.add_argument("--masks-dir"); s.add_argument("--views"); s.add_argument("--cluster-eps", type=float)
    s.set_defaults(f=cmd_segment)
    s = sub.add_parser("types"); add_work(s); s.add_argument("assignments", nargs="+"); s.set_defaults(f=cmd_types)
    s = sub.add_parser("clean"); add_work(s)
    s.add_argument("--target-faces", type=int, default=50000); s.add_argument("--coacd-threshold", type=float, default=0.05)
    s.add_argument("--max-hulls", type=int, default=16); s.set_defaults(f=cmd_clean)
    s = sub.add_parser("texture"); add_work(s); s.add_argument("--delight", default="lambert", choices=["lambert", "none"]); s.set_defaults(f=cmd_texture)
    s = sub.add_parser("pack"); add_work(s); s.add_argument("--asset-dir"); s.set_defaults(f=cmd_pack)
    s = sub.add_parser("verify"); add_work(s); s.add_argument("--skip-dynamic", action="store_true"); s.set_defaults(f=cmd_verify)
    s = sub.add_parser("run"); add_work(s); add_inputs(s)
    s.add_argument("--scale", type=float); s.add_argument("--table-height", type=float)
    s.add_argument("--mode", default="auto", choices=["auto", "masks"]); s.add_argument("--masks-dir"); s.add_argument("--views")
    s.add_argument("--cluster-eps", type=float)
    s.add_argument("--target-faces", type=int, default=50000); s.add_argument("--coacd-threshold", type=float, default=0.05)
    s.add_argument("--max-hulls", type=int, default=16); s.add_argument("--delight", default="lambert", choices=["lambert", "none"])
    s.add_argument("--asset-dir"); s.add_argument("--skip-dynamic", action="store_true"); s.set_defaults(f=cmd_run)
    s = sub.add_parser("synth"); s.add_argument("out"); s.add_argument("--seed", type=int, default=0); s.set_defaults(f=cmd_synth)

    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
