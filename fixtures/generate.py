"""Generate small, deterministic synthetic tensors for bit-exact tests.

SPEC.md §2 / §7: synthetic tensors are permitted ONLY as unit-test fixtures,
never for codec quality claims. These fixtures exist purely so the Rust
core and the Python torch reference can be checked against each other
(tests/test_bytes_parity_with_rust.py) without needing a captured-activation
shard on every test run.

Each fixture is saved twice: as `<name>.npy` (for the Python side) and as
`<name>.smhf` (the tiny row-major-f32 format `crates/cli` reads — see the
format docstring in crates/cli/src/main.rs).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent
sys.path.insert(0, str(FIXTURES_DIR.parent / "python"))
from samhita.io import write_smhf  # noqa: E402


def make_fixture(name: str, rows: int, cols: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # A mild heavy-tailed distribution, structurally similar to real
    # attention K/V activations (a narrow bulk plus a handful of outlier
    # channels) — see dhurandhar/src/dhurandhar/turboquant.py
    # `synthesize_kv_tensor`, whose "gaussian_heavy_tail" mode this mirrors.
    # Used only as a fixture, per SPEC.md §2/§7 — never as an evaluation.
    base = rng.standard_normal((rows, cols)).astype(np.float32) * 0.3
    outlier_mask = rng.random((rows, cols)) < 0.05
    base = base + outlier_mask * rng.standard_normal((rows, cols)).astype(np.float32) * 3.0

    np.save(FIXTURES_DIR / f"{name}.npy", base)
    write_smhf(FIXTURES_DIR / f"{name}.smhf", base)
    return base


def main() -> None:
    make_fixture("small_8x16", rows=8, cols=16, seed=0)
    make_fixture("medium_64x64", rows=64, cols=64, seed=1)
    make_fixture("with_window_40x32", rows=40, cols=32, seed=2)
    # Long enough that presets with `recent=128` still leave a non-empty
    # "middle" segment that actually reaches the quantizer.
    make_fixture("long_256x64", rows=256, cols=64, seed=3)
    print("wrote fixtures to", FIXTURES_DIR)


if __name__ == "__main__":
    main()
