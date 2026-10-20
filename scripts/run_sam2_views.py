"""Generate per-view object masks with SAM 2 for `r2s segment --mode masks`.

Usage (needs torch + sam2, and a GPU to be fast):
  python scripts/run_sam2_views.py --frames frames/ --clicks clicks.json \
      --out masks/ --views views.json --ckpt sam2_hiera_large.pt --cfg sam2_hiera_l.yaml

clicks.json: {"frame": "00012.jpg", "objects": {"1": [[x, y], ...], "2": [[x, y]]}}
views.json : [{"image": "00012.jpg", "K": 3x3, "T_world_cam": 4x4, "width": W, "height": H}, ...]
             (from COLMAP: images.txt + cameras.txt -> OpenCV convention, camera->world)

Writes masks/<frame_stem>_<obj_id>.png for every frame in the video directory.
This script is intentionally separate from the r2s package so the core pipeline
stays free of torch; r2s only consumes the masks it writes.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="directory of ordered video frames (jpg)")
    ap.add_argument("--clicks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", required=True)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from sam2.build_sam import build_sam2_video_predictor

    clicks = json.loads(Path(a.clicks).read_text())
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    frames = sorted(Path(a.frames).glob("*.jpg"))
    names = [f.name for f in frames]
    start = names.index(clicks["frame"])

    predictor = build_sam2_video_predictor(a.cfg, a.ckpt)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(a.frames))
        for oid, pts in clicks["objects"].items():
            predictor.add_new_points_or_box(state, frame_idx=start, obj_id=int(oid),
                                            points=np.array(pts, dtype=np.float32),
                                            labels=np.ones(len(pts), dtype=np.int32))
        for direction in (False, True):  # forward then backward from the click frame
            for fidx, obj_ids, logits in predictor.propagate_in_video(state, start_frame_idx=start, reverse=direction):
                stem = frames[fidx].stem
                for k, oid in enumerate(obj_ids):
                    m = (logits[k] > 0).cpu().numpy()[0]
                    Image.fromarray((m * 255).astype(np.uint8)).save(out / f"{stem}_{oid}.png")
    print(f"wrote masks for {len(frames)} frames to {out}")


if __name__ == "__main__":
    main()
