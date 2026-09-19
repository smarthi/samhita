"""Cross-language parity: SPEC.md M1 acceptance criterion is "bit-exact
(or documented tolerance) agreement between the Rust reference and a torch
reference on fixtures." Both languages share the same deterministic
SplitMix64/inverse-normal-CDF seeding (see `python/samhita/prng.py` vs
`crates/core/src/{rng,normal}.rs`), so agreement here is expected to be
tight (~1e-6), not just "roughly similar" — see docs/open-questions.md for
the one documented source of divergence (`lloyd_max`'s codebook uses
`torch.special.ndtri` in Python vs an Acklam approximation in Rust).

Skipped automatically if `cargo` isn't available.
"""

from __future__ import annotations

import shutil

import pytest
import torch

from samhita.io import read_smhf
from samhita.presets import load_preset
from samhita.rust_bridge import run_pipeline

pytestmark = pytest.mark.skipif(shutil.which("cargo") is None, reason="cargo not available")


@pytest.mark.parametrize(
    ("preset_name", "side", "seed"),
    [
        ("turboquant_mse", "k", 1),
        ("turboquant_mse", "v", 2),
        ("kivi", "k", 1),
        ("kivi", "v", 2),
    ],
)
def test_mse_matches_rust_reference(preset_name, side, seed, fixtures_dir, presets_dir):
    fixture_path = fixtures_dir / "long_256x64.smhf"
    arr = read_smhf(fixture_path)
    x = torch.tensor(arr)

    preset = load_preset(preset_name)
    pipe = preset.build_k(head_dim=x.shape[1], seed=seed) if side == "k" else preset.build_v(head_dim=x.shape[1], seed=seed)
    py_mse = pipe.round_trip_mse(x)

    rust_out = run_pipeline(
        presets_dir / f"{preset_name}.toml", fixture_path, side, seed=seed, release=False
    )

    assert py_mse == pytest.approx(rust_out["mse"], abs=1e-4, rel=1e-2)
