"""End-to-end test on the synthetic tabletop fixture: all four objects must be
found, closed, packed and pass the MuJoCo drop test."""
import json
from pathlib import Path

import numpy as np
import pytest

from r2s.synth import make_scene
from r2s.inspect_align import align, inspect_inputs
from r2s.segment import segment, score_against_gt
from r2s.clean import clean
from r2s.texture import texture
from r2s.pack import pack
from r2s.verify import verify


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    make_scene(d / "data", seed=0)
    return d


def test_inspect(scene):
    d = scene / "data"
    rep = inspect_inputs(d / "scene_mesh.obj", d / "scene_gs.ply", d / "scene.usd")
    assert rep["gaussians"]["format"] == "3dgs"
    assert rep["consistent_scale"]
    assert rep["usd"]["already_decomposed"] is False


def test_pipeline(scene):
    d = scene / "data"
    w = scene / "work"
    al = align(w, d / "scene_mesh.obj", d / "scene_gs.ply", None)
    assert abs(al["plane_z"] - 0.75) < 0.01  # table found above the floor

    seg = segment(w)
    gt = np.load(d / "gt_labels.npy")
    pred = np.load(seg["face_labels"])
    score = score_against_gt(pred, gt)
    for k in (1, 2, 3, 4):
        assert score[k]["iou"] > 0.9, score
    assert score["background_iou"] > 0.95

    cl = clean(w)
    for oid, r in cl["objects"].items():
        assert r["watertight"], oid
        assert len(r["collision"]) <= 16

    texture(w)
    pk = pack(w)
    assert Path(pk["scene"]).exists()
    assert Path(pk["scene_cfg"]).exists()

    rep = verify(w)
    assert rep["all_pass"], json.dumps(rep, indent=1)
