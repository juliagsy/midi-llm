# midi-llm

Controlled comparison of **REMI+**, **AMT (MIDI-like)**, and **Octuple** representations inside a unified LLM training and evaluation pipeline (Paper 2). **ABC is disabled** until a real MIDI↔ABC converter is implemented.

**Paper 2 scope (implemented):** Stage S3 LoRA editing on MIDI-Instruct, with infer → decode → `musicinstruct score`.

**Deferred:** captioning / text→MIDI generation metrics (FAD, CLAP), S1 domain CPT, vocab extension.

## Google Colab (T4)

Training notebooks live in [`colab/`](colab/README.md):

1. `01_setup_and_data.ipynb` — install, HF login, build shards  
2. `02_train_lora_sft.ipynb` — LoRA SFT (fp16 on T4)  
3. `03_eval_musicinstruct.ipynb` — infer + MIDI-Instruct score  

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[repr,dev]"

# AMT tokenizer (Anticipatory Music Transformer — install from GitHub)
pip install "git+https://github.com/jthickstun/anticipation.git" mido
# or: pip install -e ".[amt]"

# Training + BPE token stats (Llama tokenizer)
pip install -e ".[train]"

# Evaluation against MIDI-Instruct benchmark (optional; not required for unit tests)
pip install -e ../musicinstruct
```

## Quick start

```bash
# Round-trip fidelity check on a MIDI file
midi-llm roundtrip path/to/file.mid --repr remi

# Representation + Llama BPE token counts (comparable across arms)
midi-llm token-stats path/to/file.mid
midi-llm token-stats path/to/file.mid --json

# Build SFT JSONL from a MIDI-Instruct manifest (train split; pilot or real v0.2)
midi-llm build-instruct-shard \
  ../musicinstruct/data/pilot/pilot.jsonl \
  --repr remi \
  --output data/shards/instruct_remi.jsonl \
  --split train
# Real v0.2: ../musicinstruct/data/v0.2/manifest.jsonl — see colab/README.md

# Stage S0 from local MIDIs or GigaMIDI (HF)
midi-llm build-syntax-shard path/to/midis --repr remi --output data/shards/syntax_remi.jsonl
pip install -e ".[data]"
midi-llm build-gigamidi-shard --repr remi --output data/shards/gigamidi_remi.jsonl --limit 50

# Estimate training time on your machine (add --benchmark to measure one step)
midi-llm estimate-training --repr remi --steps 50 --benchmark

# Eval loop: copy-source baseline (no LLM) or full infer+score
midi-llm eval-edit ../musicinstruct/data/pilot/pilot.jsonl \
  --output-dir results/copy_source --baseline copy-source --split test

# Stage S3 LoRA pilot (defaults: max_seq_len=2048, max_new_tokens=1024 from configs/base.yaml)
midi-llm train-lora data/shards/edit_remi.jsonl \
  --output-dir runs/remi_lora_pilot \
  --repr remi --max-steps 50 --max-samples 32
```

On 8 GB unified-memory Macs, add `--max-seq-len 512 --batch-size 1` to avoid swapping.

## Representations

| Arm | Module | Status |
|-----|--------|--------|
| `remi` | `midi_repr.remi_repr` | active — REMI+ via MidiTok |
| `amt` | `midi_repr.amt_repr` | active — AMT via `anticipation` |
| `octuple` | `midi_repr.octuple_repr` | active — Octuple via MidiTok |
| `abc` | `midi_repr.abc_repr` | **disabled** |

All active arms use **space-separated token strings + native Llama BPE** (`tokenizer_mode: bpe_text`, no vocab extension). Use `token-stats` for **BPE-comparable** efficiency numbers (`n_bpe_tokens`, `bpe_tokens_per_note`).

## Implemented CLI commands

| Command | Status |
|---------|--------|
| `token-stats` | representation + Llama BPE counts |
| `roundtrip` | encode/decode + MIDI fidelity check |
| `build-instruct-shard` | S3 edit JSONL |
| `build-syntax-shard` | S0 syntax JSONL |
| `build-gigamidi-shard` | S0 from HuggingFace GigaMIDI |
| `build-midicaps-shard` | S2 caption↔MIDI JSONL |
| `train-lora` | S3 LoRA SFT |
| `infer-edit` | manifest preflight + LLM completions |
| `eval-edit` | infer/decode/score (or copy-source baseline) |
| `estimate-training` | wall-clock estimate |

## Training stages

| Stage | Data | CLI | Training |
|-------|------|-----|----------|
| S0 | GigaMIDI / local MIDIs | `build-syntax-shard`, `build-gigamidi-shard` | not implemented |
| S1 | MusicPile + MIDI | — | planned |
| S2 | MidiCaps | `build-midicaps-shard` | not implemented |
| S3 | MIDI-Instruct train | `build-instruct-shard` | **`train-lora`** |

See `docs/experiment_protocol.md` for frozen hyperparameters and evaluation protocol.

## Project layout

```
midi_llm/
  midi_repr/     # encode/decode + round-trip per representation
  data/          # SFT templates, shard builders, JSONL validation
  tokenization/  # Llama BPE counting for efficiency tables
  train/         # LoRA SFT + dataset
  infer/         # edit prompts, preflight, generation
  eval/          # decode completions + musicinstruct runner
configs/         # shared hyperparameters per representation arm
docs/            # experiment protocol
tests/           # self-contained fixtures (no sibling repo required)
```

## Citation

TBD — Paper 2 representation study; uses MIDI-Instruct benchmark (Paper 1).
