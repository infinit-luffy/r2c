"""Manifest: single JSON file that records every step's inputs, parameters,
outputs and verification results for one scene. Steps are cached by the hash
of their inputs + parameters so re-running the CLI only redoes what changed.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"


def _file_hash(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _code_hash() -> str:
    """Hash of the r2s package sources so a code change invalidates cached steps."""
    h = hashlib.sha1()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def hash_inputs(files: list[Path], params: dict[str, Any]) -> str:
    """Hash of input file contents + JSON-serialised params + package code."""
    h = hashlib.sha1()
    h.update(_code_hash().encode())
    for p in sorted(files):
        p = Path(p)
        h.update(str(p.name).encode())
        if p.exists():
            h.update(_file_hash(p).encode())
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


class Manifest:
    """Scene manifest living at <workdir>/manifest.json."""

    def __init__(self, workdir: Path):
        self.workdir = Path(workdir)
        self.path = self.workdir / MANIFEST_NAME
        self.data: dict[str, Any] = {"version": 1, "steps": {}, "objects": {}, "scene": {}}
        if self.path.exists():
            self.data = json.loads(self.path.read_text())

    # ---- persistence ----
    def save(self) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, default=str))

    # ---- steps / caching ----
    def step_cached(self, name: str, key: str) -> bool:
        s = self.data["steps"].get(name)
        return bool(s and s.get("hash") == key and s.get("status") == "ok")

    def step_begin(self, name: str, key: str, params: dict[str, Any]) -> None:
        self.data["steps"][name] = {
            "hash": key,
            "params": params,
            "status": "running",
            "started": time.time(),
        }
        self.save()

    def step_end(self, name: str, outputs: dict[str, Any], status: str = "ok") -> None:
        s = self.data["steps"][name]
        s.update({"status": status, "outputs": outputs, "finished": time.time()})
        self.save()

    def step_outputs(self, name: str) -> dict[str, Any]:
        s = self.data["steps"].get(name)
        if not s or s.get("status") != "ok":
            raise RuntimeError(f"step '{name}' has not run successfully yet; run it first")
        return s["outputs"]

    # ---- objects ----
    def set_object(self, obj_id: str, **fields: Any) -> None:
        self.data["objects"].setdefault(obj_id, {}).update(fields)
        self.save()

    def objects(self) -> dict[str, dict[str, Any]]:
        return self.data["objects"]

    def set_scene(self, **fields: Any) -> None:
        self.data["scene"].update(fields)
        self.save()

    @property
    def scene(self) -> dict[str, Any]:
        return self.data["scene"]
