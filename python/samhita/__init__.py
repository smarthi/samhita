"""samhita: composable KV-cache codec framework. Stages, not codecs.

See SPEC.md at the repo root and docs/design.md for the architecture.
"""

from .pipeline import Pipeline
from .presets import Preset, load_preset

__all__ = ["Pipeline", "Preset", "load_preset"]
