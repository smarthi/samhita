"""The `.smhf` fixture format shared with `crates/cli/src/main.rs` and
`fixtures/generate.py`: u32 rows (LE), u32 cols (LE), then `rows*cols`
little-endian f32 values, row-major."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def write_smhf(path: Path, arr: np.ndarray) -> None:
    assert arr.ndim == 2
    rows, cols = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", rows, cols))
        f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def read_smhf(path: Path) -> np.ndarray:
    data = Path(path).read_bytes()
    rows, cols = struct.unpack_from("<II", data, 0)
    arr = np.frombuffer(data, dtype="<f4", count=rows * cols, offset=8)
    return arr.reshape(rows, cols).copy()
