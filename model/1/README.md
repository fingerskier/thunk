# Model #1 - Recursive Reasoning Diffuser

This directory implements the first recursive reasoning diffuser described in
`PLAN.md`. It is intentionally small and experimental: the supervision loop is
also the masked-denoising schedule, and the model carries only two mutable
features between steps.

## Components

- `config.py` defines `ReasoningDiffuserConfig` and the tiny preset.
- `model.py` implements the shared recursive core, answer-canvas head, and halt
  head.
- `diffusion.py` provides the absorbing-mask schedule, masked CE objective, and
  confidence remasking.
- `train.py` exposes the lower-level `deep_supervision_step` primitive and a
  CPU smoke test.
- `text_io.py` provides the script-level character tokenizer, JSONL dataset
  loader, batch encoding, and checkpoint sampling helper.
- `train_model.py` trains from model-1 JSONL data and writes checkpoints.
- `run_model.py` loads a checkpoint and answers one question from the command
  line.

## Install

Model #1 currently depends only on PyTorch plus the Python standard library.
Install a CPU or CUDA build of PyTorch appropriate for your machine:

```bash
python -m pip install torch
```

Run the commands below from the repository root unless noted otherwise.

## Generate Data

Create model-1-shaped question/answer JSONL records under `data/model/1/`:

```bash
python script/glean_datasets.py --model 1 --offline --limit 20
```

Use the non-offline path to sample configured public sources when optional
dataset dependencies or direct URLs are available:

```bash
python script/glean_datasets.py --model 1 --limit 250
```

Generate a small sample set from no-key public APIs:

```bash
python script/glean_datasets.py --model 1 --sources opentdb_trivia openlibrary_books datamuse_words --limit 20
```

`train_model.py` reads `data/model/1/combined.jsonl` by default. Each row must
contain `question` and `answer` fields.

## Train

Train with the default model-1 data path and write
`model/1/checkpoints/model1.pt`:

```bash
python model/1/train_model.py --steps 200 --batch-size 8
```

Run a smaller CPU smoke-training job:

```bash
python model/1/train_model.py --steps 5 --batch-size 2 --d-model 32 --d-ff 64 --recursion-depth 1 --supervision-steps 3 --checkpoint model/1/checkpoints/smoke.pt
```

Train from a specific JSONL file:

```bash
python model/1/train_model.py --data path/to/records.jsonl --checkpoint model/1/checkpoints/custom.pt --steps 1000
```

Continue training from an existing checkpoint:

```bash
python model/1/train_model.py --resume model/1/checkpoints/model1.pt --steps 200
```

Useful training options:

- `--max-records N` limits the loaded JSONL rows.
- `--device cpu|cuda|auto` controls placement.
- `--save-every N` writes intermediate checkpoints during longer runs.
- `--answer-len N` sets the fixed answer canvas length.
- `--max-question-len N` sets the fixed question length.
- `--supervision-steps N`, `--warmup-recursions N`, and `--recursion-depth N`
  control the recursive denoising work.

The script uses a checkpoint-local character tokenizer built from the training
rows. This is a runnable baseline for model development, not a final tokenizer
or benchmark-grade data pipeline.

## Run

Ask a trained checkpoint one question:

```bash
python model/1/run_model.py --checkpoint model/1/checkpoints/model1.pt --question "What is example 0 from wmt14_en_de?"
```

Run the smaller smoke checkpoint:

```bash
python model/1/run_model.py --checkpoint model/1/checkpoints/smoke.pt --question "What is example 0 from wmt14_en_de?"
```

Useful run options:

- `--weights ema|model` chooses EMA weights or raw online weights.
- `--steps N` overrides the checkpoint's inference denoising step count.
- `--halt-threshold X` changes the halt-head stopping threshold.
- `--show-ids` also prints generated token ids.

Outputs from an untrained or tiny smoke-trained model can be nonsensical. A
meaningful answer requires enough data, steps, and a task that fits the fixed
answer canvas.

## Smoke Test

The original low-level smoke test still verifies that the recursive training
primitive executes:

```bash
python model/1/train.py
```

An end-to-end smoke path is:

```bash
python script/glean_datasets.py --model 1 --offline --limit 2
python model/1/train_model.py --steps 2 --batch-size 2 --d-model 32 --d-ff 64 --recursion-depth 1 --supervision-steps 3 --checkpoint model/1/checkpoints/smoke.pt
python model/1/run_model.py --checkpoint model/1/checkpoints/smoke.pt --question "What is example 0 from wmt14_en_de?"
```

## Checkpoints

Checkpoints are written with:

- `config`: the `ReasoningDiffuserConfig` values.
- `model`: raw online model weights.
- `ema`: EMA model weights, used by `run_model.py` by default.
- `optimizer`: AdamW state for resume.
- `tokenizer`: character-tokenizer metadata.
- `step` and `records`: run metadata.

## Current Limits

- Answers are fixed to `--answer-len`; longer answers are truncated.
- The script tokenizer is character-level and checkpoint-local.
- `run_model.py` uses a simple greedy denoising loop over the answer canvas.
- The model is still experimental; the scripts provide a runnable path, not a
  claim of reasoning quality.
