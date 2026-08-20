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
  weights. See [`model/2/README.md`](model/2/README.md) for the training &
  inference runbook and [`model/2/PLAN.md`](model/2/PLAN.md) for the review,
  evaluation, and phased plan.

Distributed-training strategy (Psyche/Solana evaluation, alternatives, and
re-evaluation triggers) lives in
[`DISTRIBUTED_TRAINING.md`](DISTRIBUTED_TRAINING.md).

## Shared tokenizer
All models in the chain share one pinned SentencePiece tokenizer,
[`tokenizer/v1/tokenizer.model`](tokenizer/v1/) (vocab 8000, byte fallback,
every `<src:…>`/`<tgt:…>`/`<sep>` control tag a single reserved symbol — see
`tokenizer/v1/MANIFEST.json`). It is built once by
`script/train_tokenizer.py` and then only loaded; changing it is a breaking
change for every model, so publish a new `tokenizer/vN/` rather than
overwriting.

## Dataset gleaning
Use `script/glean_datasets.py` to sample public training-data repositories and
no-key public APIs into
`./data/`, which is ignored by git. The script is model-aware: `--model 0`
emits seq2seq records (`source`/`target` plus `train_text.txt` for the v0
Tokenizer workflow), while `--model 1` emits reasoning-diffuser records
(`question`/`answer`), and `--model 2` emits bidirectional translation pairs
(`{tagged source} <sep> {target}` lines, see
[`TRANSLATION_DATA.md`](TRANSLATION_DATA.md)). Use `--model all` for every shape.

```bash
python script/glean_datasets.py --model all --limit 250
python script/glean_datasets.py --model 1 --sources opentdb_trivia openlibrary_books datamuse_words --limit 20
```

The script prefers Hugging Face datasets when the optional `datasets` package is
installed, can fall back to direct public JSONL URLs where available, can fetch
small samples from Open Trivia DB, Open Library, and Datamuse public APIs, and
also supports `--offline` for deterministic smoke data without network access.

