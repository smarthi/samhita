"""Load a `presets/*.toml` file (SPEC.md §4.3, §9) and build the K/V
`Pipeline`s it describes."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from .pipeline import Pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = REPO_ROOT / "presets"


@dataclass
class Preset:
    name: str
    raw: dict

    @property
    def evidence(self) -> dict | None:
        return self.raw.get("evidence")

    def build_k(self, head_dim: int, seed: int = 1) -> Pipeline:
        return Pipeline(self.raw["k"], head_dim, seed)

    def build_v(self, head_dim: int, seed: int = 2) -> Pipeline:
        return Pipeline(self.raw["v"], head_dim, seed)


def load_preset(name_or_path: str) -> Preset:
    path = Path(name_or_path)
    if not path.exists():
        path = PRESETS_DIR / f"{name_or_path}.toml"
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return Preset(name=raw["name"], raw=raw)
