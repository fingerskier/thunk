# thunk
A tiny, composable cognitive prosthetic
> ...a local SLM but for lightning-fast translation tasks that can integrate with an LLM workflow


## Idears
* 128-dim transformer w/ attention
* Higher reasoning by chaining multiple models together
* Multiple, pre-trained variants

## Variants
* Maths
* Coding
* Theology
* Philosophy
* Images
* Legal
* General/router (what to link next)

## Architecture
* Transformer with multi-headed attention
* Internal recursion until the latent vector stabilizes (with stop-limit)

## Repository layout
Each model iteration lives in its own numbered subdirectory under `model/`,
with its own spec, sources, and implementation:

* [`model/0/`](model/0/) — **#0**, the v0 baseline translator.
  See [`model/0/README.md`](model/0/README.md) and
  [`model/0/SPEC.md`](model/0/SPEC.md) for the design and
  [`model/0/SOURCES.md`](model/0/SOURCES.md) for training material.
* [`model/1/`](model/1/) — **#1**, the small recursive reasoning diffuser.
  See [`model/1/PLAN.md`](model/1/PLAN.md) for the phased plan.
* [`model/2/`](model/2/) — **#2**, the auto-halting recurrent LM with ternary
  weights (planned). See [`model/2/PLAN.md`](model/2/PLAN.md) for the review,
  evaluation, and phased plan.

Distributed-training strategy (Psyche/Solana evaluation, alternatives, and
re-evaluation triggers) lives in
[`DISTRIBUTED_TRAINING.md`](DISTRIBUTED_TRAINING.md).

## Dataset gleaning
Use `script/glean_datasets.py` to sample public training-data repositories into
`./data/`, which is ignored by git. The script is model-aware: `--model 0`
emits seq2seq records (`source`/`target` plus `train_text.txt` for the v0
Tokenizer workflow), while `--model 1` emits reasoning-diffuser records
(`question`/`answer`). Use `--model all` to produce both shapes.

```bash
python script/glean_datasets.py --model all --limit 250
```

The script prefers Hugging Face datasets when the optional `datasets` package is
installed, can fall back to direct public JSONL URLs where available, and also
supports `--offline` for deterministic smoke data without network access.

