"""`python -m samhita.cli` — the M1 "one command" entry point: capture real
activations from a small model (if not already cached) and produce the
error-vs-bytes report + plot for the `kivi` and `turboquant_mse` presets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SHARD_DIR = REPO_ROOT / "reports" / "activations" / "qwen2.5-0.5b-instruct"
DEFAULT_MULTI_SHARD_DIR = REPO_ROOT / "reports" / "activations" / "qwen2.5-0.5b-instruct-wikitext2"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports"


def cmd_capture(args: argparse.Namespace) -> None:
    from .capture import capture_activations, save_shard

    result = capture_activations(model_id=args.model_id)
    out = save_shard(result, args.out)
    print(f"wrote shard to {out} ({result.manifest})")


def cmd_demo(args: argparse.Namespace) -> None:
    from .capture import capture_activations, save_shard
    from .harness import run_demo

    shard_dir = Path(args.shard_dir)
    if not (shard_dir / "activations.safetensors").exists():
        print(f"no cached shard at {shard_dir}, capturing from {args.model_id} ...")
        result = capture_activations(model_id=args.model_id)
        save_shard(result, shard_dir)

    plot_path = run_demo(shard_dir, Path(args.out), layer=args.layer, head=args.head)
    print(f"wrote report to {plot_path}")


def cmd_report(args: argparse.Namespace) -> None:
    from .capture import capture_and_save_many, load_wikitext2_prompts
    from .report import run_m2_report

    base_dir = Path(args.shard_dir)
    existing = sorted(base_dir.glob("prompt_*")) if base_dir.exists() else []
    if len(existing) >= args.n_prompts:
        shard_dirs = existing[: args.n_prompts]
        print(f"reusing {len(shard_dirs)} cached prompt shards under {base_dir}")
    else:
        print(f"capturing {args.n_prompts} WikiText-2 prompts from {args.model_id} ...")
        prompts = load_wikitext2_prompts(n_prompts=args.n_prompts)
        shard_dirs = capture_and_save_many(args.model_id, prompts, base_dir)

    md_path = run_m2_report(shard_dirs, Path(args.out), layer=args.layer, head=args.head)
    print(f"wrote M2 report to {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="samhita")
    sub = parser.add_subparsers(dest="command", required=True)

    p_capture = sub.add_parser("capture", help="capture real post-RoPE Q/K/V from a small model")
    p_capture.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    p_capture.add_argument("--out", type=Path, default=DEFAULT_SHARD_DIR)
    p_capture.set_defaults(func=cmd_capture)

    p_demo = sub.add_parser("demo", help="capture (if needed) + sweep kivi/turboquant_mse + plot")
    p_demo.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    p_demo.add_argument("--shard-dir", type=Path, default=DEFAULT_SHARD_DIR)
    p_demo.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    p_demo.add_argument("--layer", type=int, default=0)
    p_demo.add_argument("--head", type=int, default=0)
    p_demo.set_defaults(func=cmd_demo)

    p_report = sub.add_parser(
        "report", help="M2: full metric stack, multi-prompt/seed sweep with bootstrap CIs, matched-bytes"
    )
    p_report.add_argument("--model-id", default="Qwen/Qwen2.5-0.5B-Instruct")
    p_report.add_argument("--shard-dir", type=Path, default=DEFAULT_MULTI_SHARD_DIR)
    p_report.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    p_report.add_argument("--n-prompts", type=int, default=5)
    p_report.add_argument("--layer", type=int, default=0)
    p_report.add_argument("--head", type=int, default=0)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
