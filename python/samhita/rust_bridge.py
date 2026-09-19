"""Subprocess bridge to the Rust `samhita` CLI (`crates/cli`).

SPEC.md §5.3 makes the Rust byte-accounting function the single source of
truth; rather than re-deriving that formula in Python (and risking drift),
`harness.py` and the parity tests call the real Rust binary and read its
JSON output. This mirrors SPEC.md §7's split exactly: "Rust owns... the
bit-exact CPU reference (the oracle...); Python owns... the harness."
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


class RustCliError(RuntimeError):
    pass


def ensure_cli_built(release: bool = True) -> Path:
    """Builds `crates/cli` if needed and returns the binary path."""
    profile = "release" if release else "debug"
    bin_path = REPO_ROOT / "target" / profile / "samhita"
    cmd = ["cargo", "build", "-p", "samhita-cli"]
    if release:
        cmd.append("--release")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True)
    if not bin_path.exists():
        raise RustCliError(f"expected binary at {bin_path} after build")
    return bin_path


def run_pipeline(
    preset_toml_path: Path,
    fixture_path: Path,
    side: str,
    seed: int = 1,
    num_layers: int = 1,
    num_kv_heads: int = 1,
    batch: int = 1,
    release: bool = True,
) -> dict[str, Any]:
    """Runs one (preset, side) pipeline on `fixture_path` via the Rust CLI
    and returns the parsed JSON (`mse`, `byte_report`, ...)."""
    bin_path = ensure_cli_built(release=release)
    result = subprocess.run(
        [
            str(bin_path),
            "--preset", str(preset_toml_path),
            "--fixture", str(fixture_path),
            "--side", side,
            "--seed", str(seed),
            "--layers", str(num_layers),
            "--heads", str(num_kv_heads),
            "--batch", str(batch),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _format_inline_table(d: dict[str, Any]) -> str:
    parts = []
    for k, v in d.items():
        if isinstance(v, str):
            parts.append(f'{k} = "{v}"')
        else:
            parts.append(f"{k} = {v}")
    return "{ " + ", ".join(parts) + " }"


def _format_side_table(side_cfg: dict[str, Any]) -> str:
    lines = [f'rotation = "{side_cfg["rotation"]}"', f'scale = "{side_cfg["scale"]}"']
    lines.append(f"quantizer = {_format_inline_table(side_cfg['quantizer'])}")
    lines.append(f'residual = "{side_cfg.get("residual", "none")}"')
    lines.append(f"window = {_format_inline_table(side_cfg['window'])}")
    return "\n".join(lines)


def preset_with_bits(base_preset_path: Path, side: str, bits: int, tmp_path: Path) -> Path:
    """Writes a copy of `base_preset_path` with `[<side>].quantizer.bits`
    overridden to `bits`, for the harness's bit-budget sweep.

    Hand-rolled TOML serialization (no `tomli_w` dependency needed) scoped
    to exactly the flat `[k]`/`[v]` preset schema `pipeline.py` consumes;
    the `[evidence]` block isn't needed by the Rust CLI so it's dropped.
    """
    with open(base_preset_path, "rb") as f:
        raw = tomllib.load(f)
    raw[side]["quantizer"]["bits"] = bits

    text = f'name = "{raw["name"]}"\n\n'
    text += "[k]\n" + _format_side_table(raw["k"]) + "\n\n"
    text += "[v]\n" + _format_side_table(raw["v"]) + "\n"

    out_path = tmp_path / f"{raw['name']}_{side}_{bits}bit.toml"
    out_path.write_text(text)
    return out_path


def preset_with_bits_both_sides(base_preset_path: Path, bits: int, tmp_path: Path) -> Path:
    """Like `preset_with_bits`, but overrides `quantizer.bits` on *both*
    `[k]` and `[v]` to the same value — for the M2 harness's matched-bytes
    sweep, which treats "bits" as one shared knob per preset rather than
    sweeping K and V independently (SPEC.md §6.3: compare at matched
    measured *bytes*, here summed across K+V)."""
    with open(base_preset_path, "rb") as f:
        raw = tomllib.load(f)
    raw["k"]["quantizer"]["bits"] = bits
    raw["v"]["quantizer"]["bits"] = bits

    text = f'name = "{raw["name"]}"\n\n'
    text += "[k]\n" + _format_side_table(raw["k"]) + "\n\n"
    text += "[v]\n" + _format_side_table(raw["v"]) + "\n"

    out_path = tmp_path / f"{raw['name']}_{bits}bit.toml"
    out_path.write_text(text)
    return out_path
