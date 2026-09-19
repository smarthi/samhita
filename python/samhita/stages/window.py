"""`sink_window`/`recent_window` layout policy: torch mirror of
`crates/core/src/stages/window.rs`."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WindowSplit:
    sink: int
    recent: int
    middle: int


@dataclass(frozen=True)
class WindowPolicy:
    sink: int
    recent: int
    high_prec_dtype_bytes: int

    def split(self, seq_len: int) -> WindowSplit:
        window = min(self.sink + self.recent, seq_len)
        sink = min(self.sink, window)
        recent = window - sink
        return WindowSplit(sink=sink, recent=recent, middle=seq_len - window)
