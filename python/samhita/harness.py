"""The M1 evaluation harness: sweep a preset's quantizer bit budget against
a real captured-activation shard and plot error vs. measured bytes.

SPEC.md §6.2 puts K/V reconstruction MSE last in the metric priority order
("a secondary diagnostic, not a headline metric") — QKᵀ logit error and
attention-output error (items 1-2) are M2 scope ("Full metric stack").
This harness reports MSE because it's the metric the M1 stage inventory
(no attention-score wiring yet) can honestly compute; the bytes axis is
never approximated in Python — every point comes from a real run of the
Rust CLI (`rust_bridge.run_pipeline`), which is the same byte-accounting
function `crates/core/src/bytes.rs` property-tests against real allocated
buffer sizes (SPEC.md §5.3).
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import write_smhf
from .rust_bridge import preset_with_bits, run_pipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = REPO_ROOT / "presets"
DEFAULT_BITS_BY_QUANTIZER = {
    "group_rtn": [2, 3, 4, 6],
    "sign_residual": [2, 3, 4, 6],
}


@dataclass
class SweepPoint:
    bits: int
    mse: float
    effective_bits_per_element: float
    total_resident_bytes: float


def _quantizer_kind(preset_path: Path, side: str) -> str:
    import tomllib

    with open(preset_path, "rb") as f:
        raw = tomllib.load(f)
    return raw[side]["quantizer"]["kind"]


def sweep_preset(
    preset_name: str,
    side: str,
    array: np.ndarray,
    num_layers: int,
    num_kv_heads: int,
    bits_values: list[int] | None = None,
    seed: int = 1,
) -> list[SweepPoint]:
    preset_path = PRESETS_DIR / f"{preset_name}.toml"
    kind = _quantizer_kind(preset_path, side)
    bits_values = bits_values or DEFAULT_BITS_BY_QUANTIZER[kind]

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        fixture_path = tmp_dir / "shard.smhf"
        write_smhf(fixture_path, array)

        points = []
        for bits in bits_values:
            cfg_path = preset_with_bits(preset_path, side, bits, tmp_dir)
            out = run_pipeline(
                cfg_path, fixture_path, side, seed=seed,
                num_layers=num_layers, num_kv_heads=num_kv_heads, release=True,
            )
            points.append(
                SweepPoint(
                    bits=bits,
                    mse=out["mse"],
                    effective_bits_per_element=out["byte_report"]["effective_bits_per_element"],
                    total_resident_bytes=out["byte_report"]["total_resident_bytes"],
                )
            )
        return points


def load_shard(shard_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    from safetensors import safe_open

    shard_dir = Path(shard_dir)
    manifest = json.loads((shard_dir / "manifest.json").read_text())
    tensors = {}
    with safe_open(str(shard_dir / "activations.safetensors"), framework="numpy") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return manifest, tensors


def run_demo(shard_dir: Path, out_dir: Path, layer: int = 0, head: int = 0) -> Path:
    """The M1 "one command" acceptance target: sweep `kivi` and
    `turboquant_mse` on a real captured layer/head and plot error vs bytes.
    """
    manifest, tensors = load_shard(shard_dir)
    num_layers = manifest["num_layers"]
    num_kv_heads = manifest["num_key_value_heads"]

    k = tensors[f"layer{layer}.k"][head]  # (seq_len, head_dim)
    v = tensors[f"layer{layer}.v"][head]

    results: dict[str, dict[str, list[SweepPoint]]] = {}
    for preset_name in ("kivi", "turboquant_mse"):
        results[preset_name] = {
            "k": sweep_preset(preset_name, "k", k, num_layers, num_kv_heads, seed=1),
            "v": sweep_preset(preset_name, "v", v, num_layers, num_kv_heads, seed=2),
        }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    serializable = {
        preset: {side: [vars(p) for p in points] for side, points in sides.items()}
        for preset, sides in results.items()
    }
    (out_dir / "error_vs_bytes.json").write_text(
        json.dumps({"manifest": manifest, "layer": layer, "head": head, "results": serializable}, indent=2)
    )

    _plot(results, out_dir / "error_vs_bytes.png", manifest, layer, head)
    return out_dir / "error_vs_bytes.png"


def _plot(results, out_path: Path, manifest: dict, layer: int, head: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, side in zip(axes, ("k", "v")):
        for preset_name, sides in results.items():
            points = sides[side]
            xs = [p.total_resident_bytes for p in points]
            ys = [p.mse for p in points]
            ax.plot(xs, ys, marker="o", label=preset_name)
            for p in points:
                ax.annotate(f"{p.bits}b", (p.total_resident_bytes, p.mse), fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("measured bytes (total resident, this layer/head)")
        ax.set_ylabel("reconstruction MSE (diagnostic; SPEC §6.2 secondary metric)")
        ax.set_yscale("log")
        ax.set_title(f"{side.upper()} — layer {layer}, head {head}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"error vs. measured bytes — {manifest['model_id']}, seq_len={manifest['seq_len']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
