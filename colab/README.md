# Google Colab training (T4)

Notebooks for **Paper 2** LoRA editing experiments on a Colab **T4** (~16 GB VRAM).

## Important: one runtime per notebook

Colab gives **each notebook its own VM**. You cannot attach notebooks 01/02/03 to the same session.

**Workflow:** run each notebook top-to-bottom in a **new** session. Pass artifacts through **Google Drive** (`USE_DRIVE=True`, default in all notebooks).

Set **`DATASET_MODE`** and **`RUN_ID`** consistently in notebooks **01–03**:

| Mode | `DATASET_MODE` | `RUN_ID` | Manifest |
|------|----------------|----------|----------|
| Pilot (synthetic) | `"pilot"` | `pilot_v1` | `data/pilot/pilot.jsonl` |
| Real v0.2 | `"real_v0.2"` | `real_v0.2_v1` | `data/v0.2/manifest.jsonl` |

| Drive path | Contents |
|------------|----------|
| `MyDrive/midi-llm/shards/<RUN_ID>/` | S3 edit + optional S2 MidiCaps JSONL shards (from 01 or 02 bootstrap) |
| `MyDrive/midi-llm/runs/<RUN_ID>/` | LoRA adapters + eval outputs (from 02/03) |
| `MyDrive/midi-llm/midicaps/midicaps.tar.gz` | MidiCaps tarball (~1.6 GB; extract locally — do **not** sync extracted tree) |
| `MyDrive/musicinstruct/data/pilot/` or `.../data/v0.2/` | **Shared** MIDI-Instruct manifest + `midi_in/` + `gold/` (reused, not copied to `midi-llm/`) |

With **`USE_MUSICINSTRUCT_DRIVE=True`** (default), notebooks symlink the dataset from `MyDrive/musicinstruct/data/` — run [musicinstruct colab 01](https://github.com/juliagsy/musicinstruct/tree/main/colab) first for real v0.2, or let midi-llm notebook 01 generate/restore it.

## Prerequisites

1. [Accept the Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face.
2. Create a [HF access token](https://huggingface.co/settings/tokens) (read access is enough).
3. In Colab: **Runtime → Change runtime type → T4 GPU**.

## Notebooks (run in order)

| Notebook | Purpose | Needs from prior step |
|----------|---------|------------------------|
| `01_setup_and_data.ipynb` | Clone, install, restore/generate dataset, build S3 (+ optional S2/S0) shards, **sync shards to Drive** | musicinstruct data on Drive (real v0.2) or generate pilot |
| `02_train_lora_sft.ipynb` | Self-contained bootstrap + LoRA SFT (`STAGE`: S3 edit or S2 MidiCaps) | Drive shards (or builds them) |
| `03_eval_musicinstruct.ipynb` | Self-contained bootstrap + infer/score | Drive LoRA adapters from 02 + shared manifest |

Notebooks **02** and **03** clone repos and install deps automatically. If Drive has no shards yet, **02** will build them (slower first run).

## Quick start (pilot)

1. Push `midi-llm` to GitHub; edit repo URLs in the config cell if needed.
2. Open **`01_setup_and_data.ipynb`** → `DATASET_MODE="pilot"` → run all → confirm **Sync to Drive**.
3. Disconnect → open **`02_train_lora_sft.ipynb`** → run all → confirm checkpoint sync.
4. Disconnect → open **`03_eval_musicinstruct.ipynb`** → eval (`MODE`: copy_source → zero_shot → lora).

## Real v0.2 workflow

1. Run **musicinstruct** colab 01 with `DATASET_MODE="real_v0.2"` and sync `data/v0.2/` to Drive (or run midi-llm 01 with real mode + MidiCaps tarball).
2. **midi-llm 01**: `DATASET_MODE="real_v0.2"`, `RUN_ID="real_v0.2_v1"` → build shards → sync shards only.
3. **02**: same mode + `RUN_ID`, `MAX_STEPS=300` (default for real) → train LoRA.
4. **03**: same mode + `RUN_ID` → eval on test split (~180 items).

Reuse MidiCaps tarball across repos:

```python
DRIVE_MIDICAPS_TAR = "/content/drive/MyDrive/musicinstruct/midicaps/midicaps.tar.gz"
```

## T4 timing (estimates)

| Setting | Steps | Time (approx.) |
|---------|-------|----------------|
| Pilot | 50 | 5–10 min |
| Real v0.2 | 300 | 30–45 min |
| Full S3 | 3000 | 4–6 h |
| 3 arms × 3000 | — | 12–18 h (split across sessions) |

Uses **fp16** on T4. Default **SEED=42**. Octuple shards use `a,b,c|d,e,f` wire format.

## Repos

Default clone URLs assume GitHub paths under `juliagsy/` — edit the config cell if your forks differ.
