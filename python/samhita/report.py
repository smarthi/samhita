"""The M2 evaluation report (SPEC.md §8 M2, §6.2, §6.3):

- Full metric stack (§6.2 items 1-3): QKᵀ logit error, attention-output
  error, V reconstruction error — not just K/V MSE (that's `harness.py`'s
  M1-era secondary diagnostic).
- Matched-bytes sweep: "bits" is one shared knob applied to both K and V
  of a preset, x-axis is real measured total K+V bytes from the Rust CLI,
  never nominal bits.
- Multiple seeds (for presets with a randomized rotation) and multiple
  real WikiText-2 prompts, aggregated with a percentile bootstrap CI.
- A versioned JSON result schema (`schema_version`) plus this module's own
  viewer (`render_markdown`), not just a one-off plot.

"First report on real activations supersedes earlier synthetic
benchmarks" (SPEC.md §8 M2) — every number in this report comes from a
real captured Qwen2.5-0.5B-Instruct shard on real WikiText-2 text, run
through the real Rust byte-accounting function.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .io import write_smhf
from .metrics import compute_attention_metrics
from .presets import load_preset
from .rust_bridge import preset_with_bits_both_sides, run_pipeline

SCHEMA_VERSION = "2.0"
REPO_ROOT = Path(__file__).resolve().parents[2]
PRESETS_DIR = REPO_ROOT / "presets"

DEFAULT_PRESETS = ("kivi", "turboquant_mse", "turboquant_prod")
DEFAULT_BITS_SWEEP = (2, 3, 4, 6)
# kivi has no rotation (group_rtn is deterministic given data): one seed is
# enough. turboquant_mse/prod's hadamard rotation is seeded, so multiple
# seeds are needed to see how much that randomness affects the result.
DEFAULT_SEEDS_BY_PRESET = {
    "kivi": (1,),
    "turboquant_mse": (1, 2, 3),
    "turboquant_prod": (1, 2, 3),
}
N_BOOTSTRAP = 2000
CI = 0.90


@dataclass
class Trial:
    preset: str
    bits: int
    seed: int
    prompt_idx: int
    qkt_logit_error: float
    attention_output_error: float
    v_reconstruction_error: float
    total_kv_bytes: float


@dataclass
class Aggregate:
    preset: str
    bits: int
    n_trials: int
    mean_bytes: float
    qkt_logit_error_mean: float
    qkt_logit_error_ci: tuple[float, float]
    attention_output_error_mean: float
    attention_output_error_ci: tuple[float, float]
    v_reconstruction_error_mean: float
    v_reconstruction_error_ci: tuple[float, float]


def _percentile_bootstrap_ci(values: list[float], n_resamples: int = N_BOOTSTRAP, ci: float = CI) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(0)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means[i] = sample.mean()
    lo = (1 - ci) / 2 * 100
    hi = (1 - (1 - ci) / 2) * 100
    return (float(np.percentile(means, lo)), float(np.percentile(means, hi)))


def run_trial(
    preset_name: str, bits: int, seed: int, prompt_idx: int,
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, head_dim: int, tmp_dir: Path,
) -> Trial:
    preset_path = PRESETS_DIR / f"{preset_name}.toml"
    cfg_path = preset_with_bits_both_sides(preset_path, bits, tmp_dir)

    k_fixture = tmp_dir / f"k_{preset_name}_{bits}_{seed}_{prompt_idx}.smhf"
    v_fixture = tmp_dir / f"v_{preset_name}_{bits}_{seed}_{prompt_idx}.smhf"
    write_smhf(k_fixture, k.numpy())
    write_smhf(v_fixture, v.numpy())
    k_report = run_pipeline(cfg_path, k_fixture, "k", seed=seed, release=True)
    v_report = run_pipeline(cfg_path, v_fixture, "v", seed=seed + 1, release=True)
    total_bytes = (
        k_report["byte_report"]["total_resident_bytes"] + v_report["byte_report"]["total_resident_bytes"]
    )

    preset = load_preset(str(cfg_path))
    k_pipe = preset.build_k(head_dim=head_dim, seed=seed)
    v_pipe = preset.build_v(head_dim=head_dim, seed=seed + 1)
    metrics = compute_attention_metrics(q, k, v, k_pipe, v_pipe)

    return Trial(
        preset=preset_name, bits=bits, seed=seed, prompt_idx=prompt_idx,
        qkt_logit_error=metrics.qkt_logit_error,
        attention_output_error=metrics.attention_output_error,
        v_reconstruction_error=metrics.v_reconstruction_error,
        total_kv_bytes=total_bytes,
    )


def aggregate_trials(trials: list[Trial]) -> list[Aggregate]:
    groups: dict[tuple[str, int], list[Trial]] = {}
    for t in trials:
        groups.setdefault((t.preset, t.bits), []).append(t)

    aggregates = []
    for (preset, bits), group in sorted(groups.items()):
        qkt = [t.qkt_logit_error for t in group]
        attn = [t.attention_output_error for t in group]
        v_err = [t.v_reconstruction_error for t in group]
        aggregates.append(
            Aggregate(
                preset=preset, bits=bits, n_trials=len(group),
                mean_bytes=float(np.mean([t.total_kv_bytes for t in group])),
                qkt_logit_error_mean=float(np.mean(qkt)), qkt_logit_error_ci=_percentile_bootstrap_ci(qkt),
                attention_output_error_mean=float(np.mean(attn)), attention_output_error_ci=_percentile_bootstrap_ci(attn),
                v_reconstruction_error_mean=float(np.mean(v_err)), v_reconstruction_error_ci=_percentile_bootstrap_ci(v_err),
            )
        )
    return aggregates


def run_m2_report(
    shard_dirs: list[Path],
    out_dir: Path,
    layer: int = 0,
    head: int = 0,
    presets: tuple[str, ...] = DEFAULT_PRESETS,
    bits_sweep: tuple[int, ...] = DEFAULT_BITS_SWEEP,
    seeds_by_preset: dict[str, tuple[int, ...]] = DEFAULT_SEEDS_BY_PRESET,
) -> Path:
    """Runs the full M2 sweep across `shard_dirs` (one real captured
    activation shard per WikiText-2 prompt) and writes a versioned JSON
    result, an error-vs-bytes plot (one panel per metric), and a markdown
    report to `out_dir`. Returns the markdown report's path.
    """
    from .harness import load_shard

    manifests = []
    qkv_by_prompt = []
    for shard_dir in shard_dirs:
        manifest, tensors = load_shard(shard_dir)
        manifests.append(manifest)
        q = torch.tensor(tensors[f"layer{layer}.q"][head])
        k = torch.tensor(tensors[f"layer{layer}.k"][head])
        v = torch.tensor(tensors[f"layer{layer}.v"][head])
        qkv_by_prompt.append((q, k, v))
    head_dim = manifests[0]["head_dim"]

    trials: list[Trial] = []
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        for preset_name in presets:
            for bits in bits_sweep:
                for seed in seeds_by_preset.get(preset_name, (1,)):
                    for prompt_idx, (q, k, v) in enumerate(qkv_by_prompt):
                        trials.append(
                            run_trial(preset_name, bits, seed, prompt_idx, q, k, v, head_dim, tmp_dir)
                        )

    aggregates = aggregate_trials(trials)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "schema_version": SCHEMA_VERSION,
        "model_id": manifests[0]["model_id"],
        "layer": layer,
        "head": head,
        "n_prompts": len(shard_dirs),
        "prompt_seq_lens": [m["seq_len"] for m in manifests],
        "trials": [asdict(t) for t in trials],
        "aggregates": [asdict(a) for a in aggregates],
    }
    json_path = out_dir / "m2_report.json"
    json_path.write_text(json.dumps(result, indent=2))

    _plot_m2(aggregates, out_dir / "m2_report.png", manifests[0], layer, head)
    md_path = out_dir / "m2_report.md"
    md_path.write_text(render_markdown(result, aggregates))
    return md_path


def _plot_m2(aggregates: list[Aggregate], out_path: Path, manifest: dict, layer: int, head: int) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("qkt_logit_error_mean", "qkt_logit_error_ci", "relative QKᵀ logit error"),
        ("attention_output_error_mean", "attention_output_error_ci", "attention-output error"),
        ("v_reconstruction_error_mean", "v_reconstruction_error_ci", "V reconstruction error"),
    ]
    presets = sorted({a.preset for a in aggregates})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (mean_key, ci_key, title) in zip(axes, metrics):
        for preset in presets:
            points = sorted([a for a in aggregates if a.preset == preset], key=lambda a: a.mean_bytes)
            xs = [p.mean_bytes for p in points]
            ys = [getattr(p, mean_key) for p in points]
            los = [getattr(p, mean_key) - getattr(p, ci_key)[0] for p in points]
            his = [getattr(p, ci_key)[1] - getattr(p, mean_key) for p in points]
            ax.errorbar(xs, ys, yerr=[los, his], marker="o", capsize=3, label=preset)
            for p in points:
                ax.annotate(f"{p.bits}b", (p.mean_bytes, getattr(p, mean_key)), fontsize=7, textcoords="offset points", xytext=(4, 4))
        ax.set_xlabel("measured bytes (total K+V, this layer/head)")
        ax.set_ylabel(title)
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"M2 report — {manifest['model_id']}, layer {layer} head {head}, {CI:.0%} bootstrap CI over prompts/seeds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_markdown(result: dict, aggregates: list[Aggregate]) -> str:
    lines = [
        f"# M2 report (schema {result['schema_version']})",
        "",
        f"Model: `{result['model_id']}`, layer {result['layer']}, head {result['head']}, "
        f"{result['n_prompts']} real WikiText-2 prompts (seq_lens: {result['prompt_seq_lens']}).",
        "",
        "Bytes are real, measured K+V totals from the Rust byte-accounting function "
        "(SPEC.md §5.3), never nominal bits. Error columns are the mean and "
        f"{CI:.0%} bootstrap CI over (seed x prompt) trials.",
        "",
        "| preset | bits | mean KV bytes | QKᵀ logit error | attn-output error | V recon error | n |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in sorted(aggregates, key=lambda a: (a.preset, a.bits)):
        lines.append(
            f"| {a.preset} | {a.bits} | {a.mean_bytes:.0f} | "
            f"{a.qkt_logit_error_mean:.4f} ({a.qkt_logit_error_ci[0]:.4f}, {a.qkt_logit_error_ci[1]:.4f}) | "
            f"{a.attention_output_error_mean:.4f} ({a.attention_output_error_ci[0]:.4f}, {a.attention_output_error_ci[1]:.4f}) | "
            f"{a.v_reconstruction_error_mean:.4f} ({a.v_reconstruction_error_ci[0]:.4f}, {a.v_reconstruction_error_ci[1]:.4f}) | "
            f"{a.n_trials} |"
        )
    lines.append("")
    lines.append("See `m2_report.png` for error-vs-bytes plots (one panel per metric).")
    return "\n".join(lines)
