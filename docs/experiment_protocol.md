# Experiment protocol (Paper 2)

Controlled comparison of REMI+, AMT, and Octuple under a unified Llama 3.2 1B recipe.

**ABC is disabled** in v0: music21 does not produce valid ABC from MIDI. Re-enable as a fourth arm after implementing a tested converter.

## Token protocol (v0)

All active arms serialize MIDI as **space-separated token ID strings** in the SFT prompt/completion text. Training uses the **native Llama BPE tokenizer** with **no vocabulary extension** or embedding resize.

Configs set `vocab_extension: false` and `tokenizer_mode: bpe_text`.

Octuple uses compound tokens serialized as `pitch,vel,dur,...|pitch,vel,dur,...` (pipe-separated compounds, comma-separated fields). REMI and AMT use space-separated flat integers.

### Efficiency measurement

Representation-native counts (`n_repr_tokens`) are **not** comparable across arms. For Paper 2 efficiency tables, use:

```bash
midi-llm token-stats path/to/file.mid --json
```

Fields: `n_bpe_tokens`, `bpe_tokens_per_note` (Llama BPE on the serialized payload).

## Hold constant

- Backbone: `meta-llama/Llama-3.2-1B-Instruct`
- Context: `model.max_seq_len` = 2048 tokens
- Edit generation cap: `eval.max_new_tokens` = 1024 tokens
- Optimizer / LR / schedule: see `configs/base.yaml`
- S3 data: MIDI-Instruct train split (unique-gold); pilot (`pilot_v1`, 200 train) or real v0.2 (`real_v0.2_v1`, ~645 train)
- S3 adapter: LoRA rank 16 (all arms)

## Training stages

| Stage | Objective | Data | Shard builder | Trainer |
|-------|-----------|------|---------------|---------|
| S0 | MIDI syntax CE | GigaMIDI subset | `build-syntax-shard`, `build-gigamidi-shard` | — |
| S1 | Domain CPT CE | MusicPile subset | — | planned |
| S2 | Text↔MIDI SFT | MidiCaps | `build-midicaps-shard` | — |
| S3 | Edit SFT (LoRA) | MIDI-Instruct train | `build-instruct-shard` | **`train-lora`** |

Build shards:

```bash
# S0 syntax
midi-llm build-syntax-shard MIDI_DIR --repr remi --output data/shards/syntax_remi.jsonl

# S2 caption↔MIDI (requires local MidiCaps extract)
midi-llm build-midicaps-shard --midi-root /path/to/midicaps --repr remi \
  --output data/shards/midicaps_remi.jsonl --limit 1000

# S3 editing (pilot or real v0.2 manifest — same schema)
midi-llm build-instruct-shard ../musicinstruct/data/v0.2/manifest.jsonl \
  --repr remi --output data/shards/real_v0.2_v1/edit_remi_train.jsonl --split train

# S3 LoRA (steps capped by training.s3_max_epochs; grad_accum capped by s3_grad_accum_cap)
midi-llm train-lora data/shards/edit_remi.jsonl --output-dir runs/remi_lora --repr remi --max-steps 50
# Uses sibling edit_*_validation.jsonl for eval + early stopping when present
```

## Evaluation (Paper 2 scope)

**Implemented:** instruction editing on MIDI-Instruct test split.

```bash
midi-llm eval-edit MANIFEST.jsonl --output-dir results/remi --repr remi --split test
# sanity baseline (no LLM):
midi-llm eval-edit MANIFEST.jsonl --output-dir results/copy --baseline copy-source --split test
```

Pipeline: manifest preflight (file existence + optional encode check) → LoRA inference (per-item encode failures skipped) → per-item decode → `musicinstruct score`.

**Eval knobs (defaults in `configs/base.yaml`):**

- `eval.midicaps_eval_ids: null` — when unset, score the full manifest split; set to a JSON list path to restrict scoring to a fixed id set (e.g. MidiCaps overlap subset).
- Scoring timeout: 600s per **scoring run**, not per item (`--score-timeout`).
- Inference uses greedy decoding (`temperature=0`) by default for reproducible eval scores.
- Completions record `hit_max_new_tokens` when generation reaches `eval.max_new_tokens` (1024); truncated strings may decode to partial MIDI.
- SFT `max_seq_len` must fit the full edit prompt; rows that would lose `### Instruction` after truncation are rejected at tokenize time.
- **Colab T4:** notebook 02 uses `MAX_SEQ_LEN=1024` as a hardware override; paper runs should use 2048 when VRAM allows.

**Deferred:** MidiCaps captioning, caption→MIDI generation, FAD/CLAP.

## Active representation arms

| Arm | Config | Notes |
|-----|--------|-------|
| REMI+ | `repr_remi.yaml` | MidiTok REMI+ |
| AMT | `repr_amt.yaml` | Anticipation MIDI tokenizer |
| Octuple | `repr_octuple.yaml` | MidiTok Octuple |
| ABC | `repr_abc.yaml` | **disabled** |

## Hypotheses

- H1: Compound (Octuple) → shorter BPE sequences, comparable editing quality
- H2: AMT → strong infilling/editing
- H3: REMI → longer BPE sequences, strong harmonic structure
- H4: Representation choice matters most on editing (joint score)

## Paper split

- **Paper 2**: representation comparison + basic editing (S3 LoRA on unique-gold)
- **Paper 3**: scaled editing, constraint-gold, descriptive instructions, human eval

## Implementation checklist

| Component | Status |
|-----------|--------|
| REMI / AMT / Octuple encode-decode | done |
| Round-trip MIDI fidelity checks | done |
| JSONL validation + line numbers | done |
| Instruct shard fail-on-empty | done |
| Per-item decode isolation | done |
| Infer manifest preflight | done |
| Llama BPE token stats | done |
| ABC arm | disabled |
| Vocab extension / embedding resize | not implemented (v0 uses BPE-text) |
| S0/S2 trainers | not implemented |
| S1 MusicPile CPT | planned |
| CI / LICENSE | optional (release hygiene) |
