# Google Colab training (T4)

Notebooks for **Paper 2** LoRA editing experiments on a Colab **T4** (~16 GB VRAM).

## Prerequisites

1. [Accept the Llama 3.2 license](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face.
2. Create a [HF access token](https://huggingface.co/settings/tokens) (read access is enough).
3. In Colab: **Runtime → Change runtime type → T4 GPU**.

## Notebooks (run in order)

| Notebook | Purpose |
|----------|---------|
| `01_setup_and_data.ipynb` | Clone repos, install deps, HF login, build edit shards |
| `02_train_lora_sft.ipynb` | LoRA SFT (Stage S3) for one or all representation arms |
| `03_eval_musicinstruct.ipynb` | Infer on test split, decode MIDI, score with MIDI-Instruct |

## Quick start

1. Upload this repo to GitHub (or open notebooks from your clone).
2. Open `01_setup_and_data.ipynb` in Colab ([Open in Colab](https://colab.research.google.com/) → upload / GitHub).
3. Set `HF_TOKEN` in the secrets cell (or paste when prompted).
4. After training in `02`, run `03` with the same `RUN_ID` and `REPR`.

## T4 timing (estimates)

| Setting | Steps | Time (approx.) |
|---------|-------|----------------|
| Pilot | 50 | 5–10 min |
| Medium | 300 | 30–45 min |
| Full S3 | 3000 | 4–6 h |
| 4 arms × 3000 | — | 16–24 h (use Colab Pro or split across sessions) |

Uses **fp16** on T4 (bf16 is for A100/H100).

## Persisting results

- Checkpoints: `/content/runs/<run_id>/`
- Download via Colab **Files** panel or the zip cell in notebook 03.
- Optional: mount Google Drive and set `RUNS_ROOT = "/content/drive/MyDrive/midi-llm/runs"`.

## Repos

Default clone URLs assume GitHub paths under `juliagsy/` — edit the first cell if your forks differ.
