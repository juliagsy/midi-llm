# Google Colab training (T4)

Notebooks for **Paper 2** LoRA editing experiments on a Colab **T4** (~16 GB VRAM).

## Important: one runtime per notebook

Colab gives **each notebook its own VM**. You cannot attach notebooks 01/02/03 to the same session.

**Workflow:** run each notebook top-to-bottom in a **new** session. Pass artifacts through **Google Drive** (`USE_DRIVE=True`, default in all notebooks).

| Drive path | Contents |
|------------|----------|
| `MyDrive/midi-llm/shards/<RUN_ID>/` | S3 edit JSONL shards (from 01 or 02 bootstrap) |
| `MyDrive/midi-llm/runs/<RUN_ID>/` | LoRA adapters + eval outputs (from 02/03) |

Keep **`RUN_ID`** identical across notebooks (default: `pilot_v1`).

## Prerequisites

1. [Accept the Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face.
2. Create a [HF access token](https://huggingface.co/settings/tokens) (read access is enough).
3. In Colab: **Runtime → Change runtime type → T4 GPU**.

## Notebooks (run in order)

| Notebook | Purpose | Needs from prior step |
|----------|---------|------------------------|
| `01_setup_and_data.ipynb` | Clone, install, build shards, **sync shards to Drive** | — |
| `02_train_lora_sft.ipynb` | Self-contained bootstrap + LoRA SFT | Drive shards (or builds them) |
| `03_eval_musicinstruct.ipynb` | Self-contained bootstrap + infer/score | Drive LoRA adapters from 02 |

Notebooks **02** and **03** clone repos and install deps automatically. If Drive has no shards yet, **02** will build them (slower first run).

## Quick start

1. Push `midi-llm` to GitHub; edit repo URLs in the config cell if needed.
2. Open **`01_setup_and_data.ipynb`** → run all cells → confirm **Sync to Drive**.
3. Disconnect runtime → open **`02_train_lora_sft.ipynb`** → run all cells → confirm checkpoint sync.
4. Disconnect → open **`03_eval_musicinstruct.ipynb`** → run eval (`MODE`: copy_source → zero_shot → lora).

## T4 timing (estimates)

| Setting | Steps | Time (approx.) |
|---------|-------|----------------|
| Pilot | 50 | 5–10 min |
| Medium | 300 | 30–45 min |
| Full S3 | 3000 | 4–6 h |
| 3 arms × 3000 | — | 12–18 h (split across sessions) |

Uses **fp16** on T4. Default **SEED=42**. Octuple shards use `a,b,c|d,e,f` wire format.

## Repos

Default clone URLs assume GitHub paths under `juliagsy/` — edit the config cell if your forks differ.
