# midi-llm

Controlled comparison of **ABC**, **REMI+**, **AMT (MIDI-like)**, and **Octuple** representations inside a unified LLM training and evaluation pipeline (Paper 2).

Evaluates **understanding**, **text→MIDI generation**, and **instruction editing** (via [MIDI-Instruct](https://github.com/juliagsy/musicinstruct)).

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

# Training (later phases)
pip install -e ".[train]"

# Evaluation against MIDI-Instruct benchmark
pip install -e ../musicinstruct
```

## Quick start

```bash
# Token stats and round-trip check on a MIDI file
midi-llm roundtrip path/to/file.mid --repr remi

# Compare all representations on one file
midi-llm token-stats path/to/file.mid

# Build SFT JSONL from a MIDI-Instruct manifest (train split)
midi-llm build-instruct-shard \
  ../musicinstruct/data/pilot/pilot.jsonl \
  --repr abc \
  --output data/shards/instruct_abc.jsonl \
  --split train

# Stage S0 from local MIDIs or GigaMIDI (HF, no Lakh)
midi-llm build-syntax-shard ../musicinstruct/data/pilot/seeds \
  --repr remi --output data/shards/syntax_remi.jsonl
pip install -e ".[data]"
midi-llm build-gigamidi-shard --repr remi --output data/shards/gigamidi_remi.jsonl --limit 50

# Estimate training time on your machine (add --benchmark to measure one step)
pip install -e ".[train]"
midi-llm estimate-training --repr remi --steps 50 --seq-len 512 --benchmark

# Eval loop: copy-source baseline (no LLM) or full infer+score
pip install -e ../musicinstruct
midi-llm eval-edit ../musicinstruct/data/pilot/pilot.jsonl \
  --output-dir results/copy_source --baseline copy-source --split test

# Stage S3 LoRA pilot (requires HF Llama license + ~8GB+ unified/GPU memory)
midi-llm train-lora data/shards/edit_remi_train.jsonl \
  --output-dir runs/remi_lora_pilot \
  --repr remi --max-steps 50 --max-samples 32 --max-seq-len 512
```

## Representations

| Arm | Module | Notes |
|-----|--------|-------|
| `abc` | `midi_repr.abc_repr` | Text notation; native LLM tokenizer at train time |
| `remi` | `midi_repr.remi_repr` | REMI+ via MidiTok |
| `amt` | `midi_repr.amt_repr` | AMT arrival-time tokens via `anticipation` |
| `octuple` | `midi_repr.octuple_repr` | Octuple compound tokens via MidiTok |

## Project layout

```
midi_llm/
  midi_repr/     # encode/decode + round-trip per representation
  data/          # SFT templates and shard builders
  eval/          # MIDI-Instruct evaluation helpers
configs/         # shared hyperparameters per representation arm
docs/            # experiment protocol
```

## Training stages (planned)

| Stage | Data | Purpose |
|-------|------|---------|
| S0 | GigaMIDI subset | MIDI syntax per representation |
| S1 | MusicPile + MIDI | Domain continued pretraining |
| S2 | MidiCaps ↔ Lakh | Text↔MIDI alignment |
| S3 | MIDI-Instruct train | Light LoRA editing adaptation |

See `docs/experiment_protocol.md` for frozen hyperparameters and evaluation protocol.

## Citation

TBD — Paper 2 representation study; uses MIDI-Instruct benchmark (Paper 1).
