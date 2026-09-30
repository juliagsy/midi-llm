"""Command-line interface for midi-llm."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from midi_llm.data.gigamidi import build_gigamidi_shard
from midi_llm.data.instruct import build_instruct_shard
from midi_llm.data.midicaps import build_midicaps_shard
from midi_llm.data.syntax import build_syntax_shard
from midi_llm.midi_repr.registry import REPR_NAMES, get_repr, list_reprs


def _cmd_token_stats(args: argparse.Namespace) -> int:
    from midi_llm.tokenization.stats import token_stats_for_midi

    path = Path(args.midi)
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 1

    names = list_reprs() if args.repr == "all" else [args.repr]
    rows = []
    exit_code = 0
    for name in names:
        row = token_stats_for_midi(
            path,
            name,
            model_name=args.model,
            count_bpe=not args.no_bpe,
        )
        payload = row.to_dict()
        if payload.get("error"):
            exit_code = 1
        rows.append(payload)

    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        for row in rows:
            if row.get("error"):
                print(f"{row['repr']:8}  SKIP  {row['error']}", file=sys.stderr)
                continue
            bpe = row["n_bpe_tokens"]
            bpe_tpn = row["bpe_tokens_per_note"]
            bpe_part = f"  bpe={bpe}  bpe_tpn={bpe_tpn}" if bpe is not None else ""
            print(
                f"{row['repr']:8}  repr={row['n_repr_tokens']}{bpe_part}  "
                f"repr_tpn={row['repr_tokens_per_note']}  notes={row['n_notes']}  "
                f"tracks={row['n_tracks']}  dur={row['duration_sec']}s"
            )
    if args.repr == "all" and any(not row.get("error") for row in rows):
        return 0
    return exit_code


def _cmd_roundtrip(args: argparse.Namespace) -> int:
    path = Path(args.midi)
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 1

    backend = get_repr(args.repr)
    result = backend.roundtrip(path, output_dir=args.output_dir)
    payload = {
        "repr": result.repr_name,
        "success": result.success,
        "source": str(result.source),
        "output": str(result.output) if result.output else None,
        "error": result.error,
        "fidelity_ok": result.fidelity_ok,
        "fidelity_summary": result.fidelity_summary,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    elif result.success:
        print(f"ok  {result.repr_name}  ->  {result.output}")
    else:
        print(f"fail  {result.repr_name}  {result.error}", file=sys.stderr)
    return 0 if result.success else 1


def _cmd_build_instruct_shard(args: argparse.Namespace) -> int:
    count = build_instruct_shard(
        args.manifest,
        repr_name=args.repr,
        output_path=args.output,
        split=args.split,
    )
    print(f"wrote {count} examples to {args.output}")
    return 0


def _cmd_build_syntax_shard(args: argparse.Namespace) -> int:
    count = build_syntax_shard(
        args.midi_dir,
        repr_name=args.repr,
        output_path=args.output,
        max_files=args.max_files,
    )
    print(f"wrote {count} syntax examples to {args.output}")
    return 0


def _cmd_build_gigamidi_shard(args: argparse.Namespace) -> int:
    count = build_gigamidi_shard(
        repr_name=args.repr,
        output_path=args.output,
        split=args.split,
        limit=args.limit,
        streaming=not args.no_streaming,
    )
    print(f"wrote {count} GigaMIDI syntax examples to {args.output}")
    return 0


def _cmd_build_midicaps_shard(args: argparse.Namespace) -> int:
    if not args.midi_root and not args.lakh_root:
        print("error: provide --midi-root (MidiCaps extract) or --lakh-root", file=sys.stderr)
        return 1
    count = build_midicaps_shard(
        midi_root=args.midi_root,
        lakh_root=args.lakh_root,
        repr_name=args.repr,
        output_path=args.output,
        split=args.split,
        limit=args.limit,
        bidirectional=not args.unidirectional,
    )
    print(f"wrote {count} caption examples to {args.output}")
    return 0


def _cmd_train_lora(args: argparse.Namespace) -> int:
    from midi_llm.train.lora_sft import train_lora_sft

    out = train_lora_sft(
        args.shard,
        args.output_dir,
        repr_name=args.repr,
        max_steps=args.max_steps,
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        precision=args.precision,
        seed=args.seed,
    )
    print(f"saved LoRA adapter to {out / 'lora_adapter'}")
    return 0


def _cmd_infer_edit(args: argparse.Namespace) -> int:
    from midi_llm.infer.edit import run_edit_inference

    meta = run_edit_inference(
        args.manifest,
        args.output_dir,
        repr_name=args.repr,
        model_name=args.model,
        adapter_path=args.adapter,
        split=args.split,
        max_items=args.max_items,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )
    print(json.dumps(meta, indent=2))
    return 0


def _cmd_eval_edit(args: argparse.Namespace) -> int:
    from midi_llm.eval.edit_pipeline import copy_source_as_baseline, run_edit_eval

    if args.baseline == "copy-source":
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        preds = out / "predictions.jsonl"
        copy_result = copy_source_as_baseline(args.manifest, preds, split=args.split)
        from midi_llm.eval.musicinstruct_runner import run_musicinstruct_eval

        results_path = out / "results.json"
        results = run_musicinstruct_eval(
            args.manifest,
            preds,
            output_results=results_path,
            split=args.split,
            timeout_sec=args.score_timeout,
        )
        summary = {
            "baseline": "copy-source",
            "n_predictions": copy_result.n_written,
            "n_copy_failed": copy_result.n_failed,
            "n_copy_skipped": copy_result.n_skipped,
            "overall": results.get("overall"),
        }
        print(json.dumps(summary, indent=2))
        return 0

    summary = run_edit_eval(
        args.manifest,
        args.output_dir,
        repr_name=args.repr,
        split=args.split,
        model_name=args.model,
        adapter_path=args.adapter,
        max_items=args.max_items,
        max_new_tokens=args.max_new_tokens,
        skip_infer=args.skip_infer,
        completions_path=args.completions,
        seed=args.seed,
        score_timeout_sec=args.score_timeout,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_estimate_training(args: argparse.Namespace) -> int:
    from midi_llm.train.estimate import estimate_training

    report = estimate_training(
        repr_name=args.repr,
        stage=args.stage,
        steps=args.steps,
        samples=args.samples,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        run_benchmark=args.benchmark,
        model_name=args.model,
    )
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="midi-llm", description="midi-llm representation toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    stats = sub.add_parser(
        "token-stats",
        help="Compare representation and Llama BPE token counts for one MIDI file",
    )
    stats.add_argument("midi", help="Path to a MIDI file")
    stats.add_argument("--repr", default="all", choices=[*REPR_NAMES, "all"])
    stats.add_argument(
        "--model",
        default=None,
        help="HF model id for BPE counting (default: backbone from repr config)",
    )
    stats.add_argument(
        "--no-bpe",
        action="store_true",
        help="Skip Llama BPE counting (representation counts only)",
    )
    stats.add_argument("--json", action="store_true")
    stats.set_defaults(func=_cmd_token_stats)

    rt = sub.add_parser("roundtrip", help="Encode and decode a MIDI file")
    rt.add_argument("midi", help="Path to a MIDI file")
    rt.add_argument("--repr", required=True, choices=REPR_NAMES)
    rt.add_argument("--output-dir", default=None, help="Directory for round-trip MIDI output")
    rt.add_argument("--json", action="store_true")
    rt.set_defaults(func=_cmd_roundtrip)

    shard = sub.add_parser("build-instruct-shard", help="Build Stage S3 edit SFT JSONL from MIDI-Instruct")
    shard.add_argument("manifest", help="Path to pilot.jsonl or full manifest")
    shard.add_argument("--repr", required=True, choices=REPR_NAMES)
    shard.add_argument("--output", required=True, help="Output JSONL path")
    shard.add_argument("--split", default="train", help="Split to export (default: train)")
    shard.set_defaults(func=_cmd_build_instruct_shard)

    syntax = sub.add_parser("build-syntax-shard", help="Build Stage S0 syntax JSONL from a MIDI directory")
    syntax.add_argument("midi_dir", help="Root directory containing .mid files")
    syntax.add_argument("--repr", required=True, choices=REPR_NAMES)
    syntax.add_argument("--output", required=True, help="Output JSONL path")
    syntax.add_argument("--max-files", type=int, default=None, help="Cap number of MIDI files")
    syntax.set_defaults(func=_cmd_build_syntax_shard)

    giga = sub.add_parser("build-gigamidi-shard", help="Build Stage S0 JSONL from HuggingFace GigaMIDI")
    giga.add_argument("--repr", required=True, choices=REPR_NAMES)
    giga.add_argument("--output", required=True, help="Output JSONL path")
    giga.add_argument("--split", default="train")
    giga.add_argument("--limit", type=int, default=100, help="Max examples (default: 100)")
    giga.add_argument("--no-streaming", action="store_true", help="Disable HF streaming")
    giga.set_defaults(func=_cmd_build_gigamidi_shard)

    caps = sub.add_parser("build-midicaps-shard", help="Build Stage S2 JSONL from MidiCaps")
    caps.add_argument("--midi-root", default=None, help="Local MidiCaps/Lakh MIDI directory")
    caps.add_argument("--lakh-root", default=None, help="Alias for --midi-root")
    caps.add_argument("--repr", required=True, choices=REPR_NAMES)
    caps.add_argument("--output", required=True, help="Output JSONL path")
    caps.add_argument("--split", default="train")
    caps.add_argument("--limit", type=int, default=None, help="Max examples to write")
    caps.add_argument("--unidirectional", action="store_true", help="Only caption→MIDI examples")
    caps.set_defaults(func=_cmd_build_midicaps_shard)

    train = sub.add_parser("train-lora", help="LoRA SFT on a JSONL shard (Stage S3 pilot)")
    train.add_argument("shard", help="JSONL shard path")
    train.add_argument("--output-dir", required=True, help="Checkpoint output directory")
    train.add_argument(
        "--repr",
        default=None,
        help="Representation arm (required unless shard records include repr_name)",
    )
    train.add_argument("--seed", type=int, default=42, help="Random seed for training (default: 42)")
    train.add_argument("--max-steps", type=int, default=None)
    train.add_argument("--max-seq-len", type=int, default=None)
    train.add_argument("--max-samples", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=1)
    train.add_argument("--grad-accum", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=None)
    train.add_argument("--precision", choices=["fp16", "bf16"], default=None, help="Override config precision (T4: fp16)")
    train.set_defaults(func=_cmd_train_lora)

    infer = sub.add_parser("infer-edit", help="Run LLM inference on MIDI-Instruct edit items")
    infer.add_argument("manifest", help="MIDI-Instruct manifest JSONL")
    infer.add_argument("--output-dir", required=True)
    infer.add_argument("--repr", required=True, choices=REPR_NAMES)
    infer.add_argument("--model", default=None, help="HF model id (default from config)")
    infer.add_argument("--adapter", default=None, help="LoRA adapter directory")
    infer.add_argument("--split", default="test")
    infer.add_argument("--max-items", type=int, default=None)
    infer.add_argument("--max-new-tokens", type=int, default=512)
    infer.add_argument("--temperature", type=float, default=0.2)
    infer.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")
    infer.set_defaults(func=_cmd_infer_edit)

    ev = sub.add_parser("eval-edit", help="Infer, decode, and score MIDI-Instruct edits")
    ev.add_argument("manifest", help="MIDI-Instruct manifest JSONL")
    ev.add_argument("--output-dir", required=True)
    ev.add_argument("--repr", default="remi", choices=REPR_NAMES)
    ev.add_argument("--split", default="test")
    ev.add_argument("--model", default=None)
    ev.add_argument("--adapter", default=None)
    ev.add_argument("--max-items", type=int, default=None)
    ev.add_argument("--max-new-tokens", type=int, default=512)
    ev.add_argument("--skip-infer", action="store_true", help="Decode/score existing completions only")
    ev.add_argument("--completions", default=None, help="Path to completions.jsonl when --skip-infer")
    ev.add_argument(
        "--baseline",
        choices=["copy-source"],
        default=None,
        help="Run a non-LLM baseline instead of inference",
    )
    ev.add_argument("--seed", type=int, default=42, help="Random seed for inference (default: 42)")
    ev.add_argument(
        "--score-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for musicinstruct score subprocess (default: 600)",
    )
    ev.set_defaults(func=_cmd_eval_edit)

    est = sub.add_parser("estimate-training", help="Estimate LoRA training time on this machine")
    est.add_argument("--repr", default="remi")
    est.add_argument("--stage", default="s3_edit_lora")
    est.add_argument("--steps", type=int, default=None)
    est.add_argument("--samples", type=int, default=224)
    est.add_argument("--seq-len", type=int, default=512)
    est.add_argument("--batch-size", type=int, default=1)
    est.add_argument("--grad-accum", type=int, default=8)
    est.add_argument("--model", default=None)
    est.add_argument("--benchmark", action="store_true", help="Run one training step to measure speed")
    est.set_defaults(func=_cmd_estimate_training)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
