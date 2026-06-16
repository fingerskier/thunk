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

* [`model/0/`](model/0/) — the current model (**#0**, v0 baseline translator).
  See [`model/0/README.md`](model/0/README.md) and
  [`model/0/SPEC.md`](model/0/SPEC.md) for the design and
  [`model/0/SOURCES.md`](model/0/SOURCES.md) for training material.
