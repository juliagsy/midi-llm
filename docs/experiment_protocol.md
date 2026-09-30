# Experiment protocol (Paper 2)

Frozen comparison of ABC, REMI+, AMT, and Octuple under a unified Llama 3.2 1B recipe.

## Hold constant

- Backbone: `meta-llama/Llama-3.2-1B-Instruct`
- Context: 2048 tokens
- Optimizer / LR / schedule: see `configs/base.yaml`
- S2 data: MidiCaps captions + Lakh MIDI (when available)
- S3 data: MIDI-Instruct train split (unique-gold)
- S3 adapter: LoRA rank 16 (all arms)

## Training stages

| Stage | Objective | Data |
|-------|-----------|------|
| S0 | MIDI syntax CE | GigaMIDI subset |
| S1 | Domain CPT CE | MusicPile subset + standalone MIDI |
| S2 | Text↔MIDI SFT | MidiCaps ↔ Lakh |
| S3 | Edit SFT (LoRA) | MIDI-Instruct train |

Build shards:

```bash
# S0 syntax
midi-llm build-syntax-shard MIDI_DIR --repr remi --output data/shards/syntax_remi.jsonl

# S2 caption↔MIDI (requires local Lakh + HuggingFace MidiCaps)
midi-llm build-midicaps-shard --lakh-root /path/to/lmd --repr remi \
  --output data/shards/midicaps_remi.jsonl --limit 1000

# S3 editing
midi-llm build-instruct-shard MANIFEST.jsonl --repr remi --output data/shards/edit_remi.jsonl --split train

# S3 LoRA pilot
midi-llm train-lora data/shards/edit_remi.jsonl --output-dir runs/remi_lora --repr remi --max-steps 50
```

## Evaluation axes

1. **Understanding** — MIDI→caption on MidiCaps test
2. **Generation** — caption→MIDI; FAD, CLAP, attribute accuracy
3. **Editing** — MIDI-Instruct test via `musicinstruct score`

## Hypotheses

- H1: Compound (Octuple) → shorter sequences, comparable quality
- H2: ABC → strong captioning, weak precise editing
- H3: AMT → strong infilling/editing, weaker meter reasoning
- H4: REMI → strong harmonic attributes, longer sequences
- H5: Representation choice matters most on editing (joint score)

## Paper split

- **Paper 2**: representation comparison + basic editing (S3 LoRA on unique-gold)
- **Paper 3**: scaled editing, constraint-gold, descriptive instructions, human eval
