# Model #2 — Auto-Halting Recurrent LM with Ternary Weights

The Reasoner link in the Thunk chain: a decoder-only LM that **thinks longer
on demand** by looping a small weight-tied transformer core, exits each token
as soon as its output distribution stops changing, and (in its ternary
variant) trains BitNet-b1.58-style so the exported body packs four weights
per byte. The review, evidence, and phase gates behind every choice here are
in [`PLAN.md`](PLAN.md) — this file is the runbook for actually training and
running it.

| Section | What it does |
| --- | --- |
| prelude (2 blocks) | embeds tokens into latent space, runs once |
| core (2 blocks, **weight-tied**) | looped r times per token; each iteration re-injects the prelude output through a linear adapter (Huginn recipe) |
| coda (2 blocks) | decodes the converged state into logits, runs once |

Recurrent depth is test-time compute: training randomizes the loop count
(log-normal Poisson, mean 4, max 8) with truncated backprop through the last
4 iterations, so at inference you can spend as many or as few loops as each
token needs. Halting is training-free: per token, exit when the KL between
successive loop outputs drops under `5e-4`. The looped core's KV cache is
capped at `kv_budget` slots (iteration i writes slot `i mod k`).

## Files

| File | Purpose |
| --- | --- |
| `config.py` | `RecurrentLMConfig` + presets: `smoke`, `fp` (Phase 0/1), `ternary` (Phase 2) |
| `model.py` | architecture, real stablemax CE, auto-halting `generate()`, `embed()` |
| `quant.py` | b1.58 QAT: `BitLinear` (STE, median/absmean scales), ternary packing |
| `data.py` | plain-text LM dataset, pinned-tokenizer loading, byte fallback |
| `train.py` | randomized-loop training, r-sweep eval, two-stage ternary schedule |
| `generate.py` | CLI inference with halt-depth metadata; latent-embedding hook |
| `export.py` | pack ternary 4-per-byte + int8 embeddings; round-trip verify |

Measured shapes (8k shared vocab): `fp` ≈ **8.5M** params (6.4M
non-embedding), `ternary` ≈ **23.5M** latent params — matching the PLAN.md
reference configuration. The ternary body packs to single-digit MB at export.

## Setup

```bash
cd model/2
pip install -r requirements.txt   # torch (CPU is fine for smoke), sentencepiece
```

No GPU is required for the smoke config. Real Phase 0+ runs want a single
consumer GPU (see `DISTRIBUTED_TRAINING.md` at the repo root for why that GPU
should be local/rented, not a distributed network).

## Quickstart (60 seconds, no data, no tokenizer)

```bash
python train.py --config smoke --steps 200
python generate.py --ckpt checkpoints/model2-smoke.pt --prompt "count: 1 2 3"
```

Training prints per-step loss (with the sampled loop count `r=`) and, at
eval intervals, validation loss at fixed test-time loop counts — the r-sweep
that every phase gate reads. Generation prints the continuation plus the
halt-depth metadata:

```
halt depths (20 tokens): mean 2.00 of max 8 (75% compute saved vs always-max)
depth histogram: 2:20
```

## Tokenizer: pinned, shared, never regenerated

Chained models must speak the same token space, so model #2 **loads** the
pinned tokenizer artifact shared with model/0 and refuses to train one
(`data.py` raises if the file is missing rather than silently regenerating).
The versioned artifact lives at the repo root:

```
tokenizer/v1/tokenizer.model   # SentencePiece BPE, vocab 8000, byte fallback
tokenizer/v1/MANIFEST.json     # vocab, control tags, input corpora + hashes
```

It was produced once by `script/train_tokenizer.py` (model/0's trainer
settings + the union of every `<src:…>`/`<tgt:…>`/`<sep>` control tag across
models, each a single reserved symbol). Regenerating it is a breaking change
that retrains every model sharing it — bump to `tokenizer/v2/` instead of
overwriting. Omit `--tokenizer` for the byte-level fallback (smoke only).

## Data

Any plain-text file, one document per line. The gleaning script's model-0
text output works directly:

```bash
python ../../script/glean_datasets.py --model 2 --limit 500
# -> ../../data/model/2/train_text.txt   (translation pairs, see TRANSLATION_DATA.md)
python ../../script/glean_datasets.py --model 0 --limit 500
# -> ../../data/model/0/train_text.txt
```

With no `--data`, training falls back to a deterministic synthetic corpus
(counting, patterns, small sums) — sufficient for smoke tests only.

## Training runbook

### Phase 0 — FP recurrent baseline

```bash
python train.py --config fp \
  --data ../../data/model/2/train_text.txt \
  --tokenizer ../../tokenizer/v1/tokenizer.model \
  --out checkpoints
```

Defaults follow the plan: loops sampled log-normal Poisson (mean 4, max 8),
truncated BPTT k=4, AdamW peak LR 5e-4 with warmup+cosine, stablemax cross
entropy (the actual Prieto et al. transform — a config knob via
`use_stablemax`), grad clip 1.0.

Watch the eval lines: `r=1 … r=16` validation loss. **Exit gate: loss must
improve monotonically as r grows to ~2x the training mean.** If more test-time
loops don't buy quality, stop and rethink before touching quantization. For
the iso-param / iso-FLOP feedforward comparisons, train a throwaway config
with `mean_loops=1, max_loops=1` and matched (or deepened) shapes.

Checkpoints land in `--out` every `save_interval` steps; resume with
`--resume checkpoints/model2-fp.pt`.

### Phase 1 — Halting

Halting is inference-time only, so Phase 1 is measurement, not retraining.
Sweep the threshold and read the quality/compute trade-off:

```bash
for t in 1e-3 5e-4 1e-4; do
  python generate.py --ckpt checkpoints/model2-fp.pt \
    --tokenizer ../../tokenizer/v1/tokenizer.model \
    --prompt "..." --kl-threshold $t --max-loops 16
done
```

Every run reports mean halt depth and % compute saved vs always running
`--max-loops`. Compare generation quality (and val loss at the matching fixed
r) against the always-max baseline. **Exit gate: ≥30% average compute
reduction at <1% quality loss.** Instrument depth histograms by token type
from the printed per-depth counts; a TRM-style BCE halt head is an ablation
to add only if the training-free test underdelivers.

### Phase 2 — Ternary QAT

```bash
python train.py --config ternary \
  --data ../../data/model/2/train_text.txt \
  --tokenizer ../../tokenizer/v1/tokenizer.model
```

The preset applies the whole b1.58 recipe: 2x width (512), ReLU² FFN,
`BitLinear` core (extra RMSNorm before every quantized linear, median-based
weight scale per BitNet-Reloaded, per-token int8 activations, STE), high peak
LR 1.5e-3 with the two-stage schedule (cosine → abrupt mid-run drop → low
cosine) and weight decay 0.1 → 0 — you can see `wd=0.10` flip to `wd=0.00`
at the midpoint in the logs. Prelude/coda stay FP by default
(`quantize_prelude_coda=True` to extend if stable); embeddings, norms, head,
and the halting machinery are never ternary.

**This phase holds the named novel risk**: STE noise re-entering the loop may
make latents orbit instead of converge. Re-run all Phase 1 halting
measurements on the ternary model. **Exit gate: ternary-512 within a few
percent of FP-256 on the Phase 0/1 metrics, with halting still functional.**
Ablate with `--config ternary` overridden to FP (`quantize=False`) at matched
width to isolate quantization's contribution.

Training cost is *not* reduced by ternary weights (latent FP masters + STE);
budget the same GPU time as FP plus a small overhead.

## Inference runbook

### Generation with auto-halting

```bash
python generate.py --ckpt checkpoints/model2-fp.pt \
  --tokenizer ../../tokenizer/v1/tokenizer.model \
  --prompt "your prompt" \
  --max-new-tokens 128 \
  --kl-threshold 5e-4 --max-loops 16 \
  --temperature 0.8 --top-k 40      # omit for greedy
```

Output is the text plus halt-depth metadata (mean depth, histogram, compute
saved). The per-token depths are the confidence signal the chain
router/controller consumes: tokens the model found hard ran more loops.
`--kl-threshold 0` disables early exit (always `--max-loops`) — that is the
quality-ceiling baseline every halting measurement compares against.

### Latent embedding (chain/memory hook)

```bash
python generate.py --ckpt ... --prompt "..." --embed
```

Prints the mean-pooled converged core state — model #2's equivalent of
model/0's `embed()`, usable as a memory-store key. In code:
`RecurrentLM.embed(ids)` and `RecurrentLM.generate(...)` (returns
`(ids, depths)`).

### Export (packed low-bit artifact)

```bash
python export.py --ckpt checkpoints/model2-ternary.pt \
  --out export/model2.pack.pt --verify
```

Packs every `BitLinear` to 2-bit ternary (4 weights/byte + FP32 scale),
embeddings and remaining FP linears to per-row int8, norms to FP16, and
reports bytes per section. `--verify` reloads the artifact and checks logits
against the QAT model (ternary layers round-trip exactly; int8 adds a small
bounded delta). The remaining Phase 3 work is a bitnet.cpp-style CPU
lookup-table kernel over this format, plus the two-link chain demo with
model/0. **Exit gate: a ~9 MB artifact that runs faster than FP on CPU and
composes with model/0** (the budget holds only with the tied head kept int8).

## Key knobs (`config.py`)

| Knob | Default | Meaning |
| --- | --- | --- |
| `mean_loops` / `max_loops` | 4 / 8 | training loop distribution (log-normal Poisson; 8 per Ouro's R=8 instability warning) |
| `bptt_k` | 4 | backprop through only the last k loops |
| `kl_threshold` | 5e-4 | per-token halting exit (Huginn default) |
| `max_loops_infer` | 16 | inference loop cap (~2x training mean) |
| `kv_budget` | 4 | mod-k KV slots for the looped core |
| `quantize` / `quant_scaling` | False / median | ternary QAT; median per BitNet-Reloaded (absmean = original b1.58) |
| `use_stablemax` | True | real stablemax CE — the small-data stabilizer |
| `two_stage_lr` | ternary only | b1.58 LR/WD schedule |

## Troubleshooting

- **Halting never triggers (all tokens hit max loops)** — threshold too
  strict for the model's scale; sweep upward (1e-3, 5e-3). If ternary latents
  *orbit* (KL oscillates without falling), that is the documented Phase 2
  risk: compare against the FP core ablation before blaming the threshold.
- **All tokens halt at `min_loops`** — undertrained model (distributions
  freeze early) or threshold too loose; check the Phase 0 r-sweep first —
  halting is meaningless before recursion demonstrably buys quality.
- **r-sweep flat or non-monotone** — Phase 0 gate failure. Check loop-count
  randomization is active (`r=` varies in the logs) and train longer before
  concluding recursion buys nothing at this scale.
- **Loss spikes in ternary stage 2** — expected sensitivity; the abrupt LR
  drop usually recovers it. Small models are LR/WD-sensitive under QAT
  (BitNet-Reloaded); halve the peak LR before anything else.
- **Tokenizer file not found** — deliberate: model #2 never trains a
  tokenizer. Point `--tokenizer` at `tokenizer/v1/tokenizer.model` (repo
  root) or omit it for the byte fallback (smoke only).

## Next iteration: dual-level discrete core

Small iterative LM test: **dual-level** discrete core.

See [`PLAN_CODEX.md`](PLAN_CODEX.md).

### Thesis

Same graph, two grains of forward/back:

1. **Int-wise** — normal NN dynamics on word/score nodes and int-edges  
2. **Bit/boolean** — dense bit block on each int-edge (`uint32` words; all bits talk iff the ints are connected)

Not pack-only-at-infer. Train both levels (STE / arithmetic twin on bits; ordinary grads on int scores).

### Build order

```text
M0  float NN baseline
M1  binary weights (flat)
M2  uint32 pack equivalence
M3  int graph + bit block per edge   ← thesis
M4  richer bitwise edge interior
M5  fixed-T recurrence
M6  attention (later)
M7  auto-halt (later)
```
