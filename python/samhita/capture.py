"""Post-RoPE Q/K/V activation capture from a real model on real text
(SPEC.md §6.1, M1 acceptance: "Activation capture tool working end-to-end
on one small model").

K and V are read directly off the model's `past_key_values` cache after a
forward pass: modern `transformers` (5.x) caches use `Cache.layers[i].keys`
/ `.values`, and those are populated by `Qwen2Attention.forward` *after*
`apply_rotary_pos_emb`, i.e. they already are the real post-RoPE K/V — no
patching needed for those two.

Q is not cached, so it's captured by monkeypatching the model's own
`apply_rotary_pos_emb` module-level function for the duration of one
forward pass. This works because `Qwen2Attention.forward` (and the
Llama-family equivalents) call it by its bare name, which Python resolves
from the module's globals *at call time* — patching
`sys.modules[model.__class__.__module__].apply_rotary_pos_emb` before the
call is enough, no need to touch the model object itself.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Public-domain fallback (Bacon, "Of Studies", 1625), used only if
# WikiText-2 can't be reached (e.g. no network). SPEC.md §6.1 asks for
# WikiText-2 specifically; `load_wikitext2_prompts` below is the real path.
DEFAULT_TEXT = (
    "Studies serve for delight, for ornament, and for ability. Their chief "
    "use for delight is in privateness and retiring; for ornament, is in "
    "discourse; and for ability, is in the judgment and disposition of "
    "business. For expert men can execute, and perhaps judge of "
    "particulars, one by one; but the general counsels, and the plots and "
    "marshalling of affairs, come best from those that are learned. To "
    "spend too much time in studies is sloth; to use them too much for "
    "ornament is affectation; to make judgment wholly by their rules is "
    "the humor of a scholar. They perfect nature, and are perfected by "
    "experience: for natural abilities are like natural plants, that need "
    "proyning by study; and studies themselves do give forth directions "
    "too much at large, except they be bounded in by experience."
)


def load_wikitext2_prompts(n_prompts: int = 5, min_chars: int = 300) -> list[str]:
    """Real WikiText-2 prompts (SPEC.md §6.1), for the M2 harness's
    multi-prompt sweep. Falls back to `DEFAULT_TEXT` (repeated) if the
    dataset can't be loaded (no network, etc.) — capture still works
    end-to-end, just not on WikiText-2 in that case.

    Note the repo id: the classic `wikitext` loading-script path is broken
    under current `datasets`/`huggingface_hub` versions (a Hub API change
    around dataset loading scripts); `Salesforce/wikitext` is the same data
    re-published as a plain parquet dataset and works with this session's
    `datasets==4.8.5`.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        prompts = []
        for row in ds:
            text = row["text"].strip()
            if len(text) >= min_chars and not text.startswith("="):
                prompts.append(text)
            if len(prompts) >= n_prompts:
                break
        if prompts:
            return prompts
    except Exception:  # noqa: BLE001 - fall back to the bundled prompt below
        pass
    return [DEFAULT_TEXT] * n_prompts


@dataclass
class LayerActivations:
    q: torch.Tensor  # (num_heads, seq_len, head_dim)
    k: torch.Tensor  # (num_kv_heads, seq_len, head_dim)
    v: torch.Tensor  # (num_kv_heads, seq_len, head_dim)


@dataclass
class CaptureResult:
    layers: list[LayerActivations]
    manifest: dict = field(default_factory=dict)


def capture_activations(
    model_id: str = DEFAULT_MODEL_ID,
    text: str = DEFAULT_TEXT,
    dtype: torch.dtype = torch.float32,
    model=None,
    input_ids: torch.Tensor | None = None,
) -> CaptureResult:
    """`model`/`input_ids` are dependency-injection hooks so tests can pass
    a tiny randomly-initialized model + hand-built `input_ids` and capture
    real post-RoPE activations without downloading anything or needing a
    tokenizer (see `tests/test_capture.py`). Real usage leaves both `None`
    and loads `model_id` + tokenizes `text` normally.
    """
    if model is None:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=dtype)
    model.eval()

    if input_ids is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

    module = sys.modules[model.__class__.__module__]
    if not hasattr(module, "apply_rotary_pos_emb"):
        raise RuntimeError(
            f"{module.__name__} has no module-level apply_rotary_pos_emb; "
            "this model architecture isn't supported by the M1 capture tool."
        )
    original_rope = module.apply_rotary_pos_emb
    captured_q: list[torch.Tensor] = []

    def patched_rope(q, k, cos, sin, *args, **kwargs):
        q_out, k_out = original_rope(q, k, cos, sin, *args, **kwargs)
        captured_q.append(q_out.detach().clone())
        return q_out, k_out

    module.apply_rotary_pos_emb = patched_rope
    try:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)
    finally:
        module.apply_rotary_pos_emb = original_rope

    cache = outputs.past_key_values
    num_layers = len(cache.layers)
    if len(captured_q) != num_layers:
        raise RuntimeError(
            f"captured {len(captured_q)} post-RoPE Q tensors but the cache has "
            f"{num_layers} layers; capture and cache are out of sync."
        )

    layers = []
    for i in range(num_layers):
        layers.append(
            LayerActivations(
                q=captured_q[i][0].detach().clone(),
                k=cache.layers[i].keys[0].detach().clone(),
                v=cache.layers[i].values[0].detach().clone(),
            )
        )

    manifest = {
        "model_id": model_id,
        "dtype": str(dtype),
        "seq_len": int(input_ids.shape[1]),
        "num_layers": num_layers,
        "num_attention_heads": model.config.num_attention_heads,
        "num_key_value_heads": getattr(model.config, "num_key_value_heads", model.config.num_attention_heads),
        "head_dim": layers[0].k.shape[-1],
        "prompt_preview": text[:120],
        "capture_path": "post_rope",
    }
    return CaptureResult(layers=layers, manifest=manifest)


def save_shard(result: CaptureResult, out_dir: Path) -> Path:
    """Saves one safetensors shard + a manifest.json, per SPEC.md §6.1."""
    import json

    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {}
    for i, layer in enumerate(result.layers):
        tensors[f"layer{i}.q"] = layer.q.contiguous()
        tensors[f"layer{i}.k"] = layer.k.contiguous()
        tensors[f"layer{i}.v"] = layer.v.contiguous()

    save_file(tensors, str(out_dir / "activations.safetensors"))
    (out_dir / "manifest.json").write_text(json.dumps(result.manifest, indent=2))
    return out_dir


def capture_and_save_many(
    model_id: str, prompts: list[str], base_dir: Path, dtype: torch.dtype = torch.float32
) -> list[Path]:
    """Captures one shard per prompt into `base_dir/prompt_<i>/`, reusing
    one loaded model across all prompts. Used by the M2 harness's
    multi-prompt sweep (SPEC.md §6.3: "multiple seeds... report bootstrap
    CIs over prompts" — this is the "over prompts" half)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager", dtype=dtype)

    base_dir = Path(base_dir)
    out_dirs = []
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        result = capture_activations(model_id=model_id, model=model, input_ids=input_ids, dtype=dtype)
        result.manifest["prompt_preview"] = prompt[:120]
        out_dir = save_shard(result, base_dir / f"prompt_{i}")
        out_dirs.append(out_dir)
    return out_dirs
