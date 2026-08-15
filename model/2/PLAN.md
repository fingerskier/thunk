# Model #2 — Auto-Halting Recurrent LM with Ternary Weights: Review & Plan

A review and evaluation of the model #2 stub ("auto-halting recurrent LM;
binary weights where the bits are stored in the floats"), followed by a phased
implementation plan. Evidence is current as of August 2026.

---

## Verdict up front

**Both halves of the idea are individually well-evidenced in 2025–26 work, and
their combination is genuinely novel — no published system unifies
BitNet-style low-bit weights with a depth-recurrent auto-halting LM.** The
synergy argument is real: recurrence reuses the same small weight set many
times (parameter efficiency), and low-bit weights shrink that set ~10x
(memory efficiency), so a looped ternary core can live in CPU cache and turn
edge inference from memory-bound to compute-bound. That is exactly the
"tiny, composable cognitive prosthetic" thesis.

Three corrections to the stub, each backed by evidence below:

1. **Ternary (1.58-bit), not binary.** Meta's ParetoQ sweep (the cleanest
   1/1.58/2/3/4-bit comparison) puts ternary and 2-bit on the size-accuracy
   Pareto frontier and pure 1-bit *off* it. Everything production-grade in
   this space (BitNet b1.58, Falcon-Edge, bitnet.cpp) is ternary.
2. **"Bits stored in the floats" is the right mental model for training, not
   an optimization.** That phrasing describes standard QAT: latent FP master
   weights, quantized on the fly in the forward pass, straight-through
   estimator backward. The bits have no independent existence during
   training and you cannot resume training from bits alone. There is **no
   training-time memory or compute saving** — the payoff is entirely at
   inference/export. The speculative reading (packing weight bits into float
   mantissas to batch binary ops through FP units) is not established
   practice anywhere in the literature; the proven fast path is lookup-table
   kernels (T-MAC / bitnet.cpp).
3. **Auto-halting should default to a training-free convergence test, not a
   learned mechanism.** The strongest recent looped LM (Huginn-3.5B) uses
   zero-shot per-token exit on a KL threshold between successive iterations
   and loses essentially nothing; TRM's ablation found its full ACT machinery
   slightly *worse* than a plain BCE halt head. Learned halting is an
   ablation, not the foundation.

At 5–150M params this design is ahead of published evidence on **both** axes
(sub-100M depth-recurrent LMs and sub-500M ternary LMs are each thin
territory, their combination empty). The plan therefore keeps an FP
feedforward baseline at every phase — the same discipline as model #1's plan.

---

## Where model #2 sits in the composable line

| Iteration | Role in the chain | Status |
| --- | --- | --- |
| model #0 | Translator / interface (encoder-decoder, text↔text, `embed()` memory hook) | implemented, trains on CI |
| model #1 | Reasoning diffuser (fixed answer canvas, deep supervision = denoising schedule) | smoke-test harness only |
| model #2 | **Reasoner: general LM that thinks longer on demand** (recurrent depth = test-time compute) | this plan |

Model #2 is the LM-shaped version of the recursion thesis the repo has carried
since the first prototype: model #0's original 128-dim prototype recursed until
latent cosine similarity ≥ 0.99 (found too strict in the MVP v0 smoke test —
the model rarely hit it), and model #1's plan replaced fixed-point intuition
with deep supervision. Model #2 closes the loop: recurrence over a shared
transformer core, with a principled exit criterion, as a *language model*
rather than a puzzle solver or canvas diffuser.

**Composability requirements** (so #2 can serve as a chain link per the serial
composition architecture — Perceiver → Reasoner → Actor):

- **Share model #0's tokenizer** (the SentencePiece BPE with reserved control
  tags), not a private tiny vocab like model #1's 128-token config. Chained
  models must speak the same token space; text is the inter-model interface
  until a shared embedding space is proven.
- **Expose the halt depth** per token/sequence as metadata. Downstream links
  (and the eventual router/chain controller) can use "how long it thought" as
  a confidence signal.
- **Expose the recurrent latent** the way model #0 exposes `embed()` — the
  converged core state is a natural memory-store key.

---

## Evaluation A — the auto-halting recurrent LM

### What the 2025–26 evidence says

- **Huginn-3.5B** (Geiping et al., [arXiv:2502.05171](https://arxiv.org/abs/2502.05171)):
  prelude (2 layers) → recurrent core (4 layers, looped r times) → coda
  (2 layers), trained with randomized loop counts (log-normal Poisson) and
  truncated backprop through the last k=8 iterations. Benchmarks scale
  smoothly with test-time r (ARC-E 49.1 → 69.9 from r=4 to r=32); GSM8K-CoT
  38.1 vs 1.8 for its non-recurrent baseline. The architecture template for
  model #2.
- **Ouro LoopLMs** (ByteDance Seed, [arXiv:2510.25741](https://arxiv.org/abs/2510.25741)):
  1.4B with R=4 shared-stack loops ≈ Qwen3-4B-Base (~3x parameter
  efficiency); gains attributed to knowledge *manipulation*, not storage.
  Cautionary: training was unstable at R=8; they shipped R=4.
- **Mixture-of-Recursions** ([arXiv:2507.10524](https://arxiv.org/abs/2507.10524)):
  token-level routers assign per-token recursion depth; matches vanilla
  transformers with ~50% fewer params at 135M–1.7B scale — the closest
  published evidence that recursion pays at *small* scale for LMs.
- **TRM** ([arXiv:2510.04871](https://arxiv.org/abs/2510.04871)): 7M params
  beating HRM across puzzles — already the basis of model #1's plan; its
  halting lesson transfers (below).
- **Retrofit line** ([arXiv:2511.07384](https://arxiv.org/abs/2511.07384)):
  pretrained feedforward LMs can be converted to depth-recurrent via a
  recurrence curriculum — relevant fallback if from-scratch training
  disappoints.
- **Honest cost** ([arXiv:2604.07822](https://arxiv.org/pdf/2604.07822)):
  looped models carry an inductive bias toward reasoning *at the cost of
  memorization and perplexity* at iso-FLOPs. Expect slightly worse PPL than a
  feedforward iso-FLOP baseline, better compositional behavior. Measure both.

### Halting: what actually works

| Mechanism | Evidence | Use here? |
| --- | --- | --- |
| KL/fixed-point convergence test at inference (no training) | Huginn: exit when KL between successive iteration outputs < 5e-4; negligible quality loss | **Default** |
| Simplified BCE halt head (TRM) | TRM: full ACT was *worse* (86.1% vs 87.4%); BCE head on "is answer correct" is enough, and TRM uses it only to shorten *training*, running full recursion at test | Ablation in Phase 1 |
| Q-learning ACT (HRM) | ARC Prize's independent ablation: deep supervision drove performance; ACT gave modest gains at 2x forward cost | No |
| PonderNet-style probabilistic halting | Documented collapse modes (all-halt-at-1 / never-halt, data-order sensitivity); fixes require iteration-specific gates + monotonic masks ([arXiv:2603.01914](https://arxiv.org/html/2603.01914)) | No |

This resolves the repo's oldest open question: the v0 prototype's cosine-sim
0.99 threshold was the right instinct (convergence-test halting) with the
wrong knob. The modern version is a KL threshold on output distributions,
per token, tuned on a held-out set — plus **randomized loop counts during
training** so the model is robust to whatever depth inference chooses.

Two failure modes to design against from day one:

- **Depth-extrapolation collapse**: performance peaks at some r then degrades
  with more loops. Mitigations: randomized-depth training (Huginn) and, if
  needed, spectral-radius regularization pulling latents toward stable fixed
  points (STARS, [arXiv:2605.26733](https://arxiv.org/pdf/2605.26733)).
- **KV-cache blowup**: a looped block reuses parameters but each iteration
  emits its own K/V — naive looping has the cache footprint of the unrolled
  deep model. Huginn's answer: a fixed cache budget k where iteration i
  writes entry `i mod k` (budget 4 cost ~nothing). Adopt this.

### Carry-over defects from model #1 (found in this review)

Model #2 inherits model #1's core machinery, so these must land properly here
(and ideally be fixed in model/1 too):

1. `StableMaxCrossEntropy` is defined in `model/1/model.py` but never used —
   `train.py` uses plain `F.cross_entropy`. The TRM recipe calls stablemax a
   non-optional stabilizer on small data.
2. The halt head is trained (0.05-weighted BCE) but **never consulted** — no
   early exit in training, and no inference path exists at all. "Auto-halting"
   is currently unexercised code.
3. `latent_recursion` stacks distinct `ReasoningBlock`s instead of reusing
   one weight-tied block n times. The weight-reuse-within-recursion knob (n,
   the primary memory/compute knob in the TRM recipe) does not exist as
   implemented. Model #2's core must be genuinely weight-tied — that is the
   entire premise.

---

## Evaluation B — ternary weights ("bits stored in the floats")

### The evidence at and below our scale

- **BitNet b1.58** ([arXiv:2402.17764](https://arxiv.org/pdf/2402.17764)):
  ternary {-1,0,+1} weights via absmean scaling, int8 activations. Parity
  with FP16 LLaMA "starting from 3B"; below that, consistently but modestly
  behind at iso-params (700M: PPL 12.87 vs 12.33).
- **BitNet-b1.58-2B-4T** ([arXiv:2504.12285](https://arxiv.org/pdf/2504.12285)):
  2B/4T-token native ternary model competitive with Qwen2.5-1.5B at 0.4GB
  non-embedding memory and ~10x lower energy/token on CPU.
- **Small-scale reality check**:
  - Spectra (54 models, 99M–3.9B, [arXiv:2407.12327](https://arxiv.org/abs/2407.12327)):
    at small scale FP wins at iso-params; ternary wins *per bit* above ~1B.
  - BitNet b1.58 Reloaded (100K–48M models, [arXiv:2407.09527](https://arxiv.org/abs/2407.09527)):
    ternary QAT reaches parity for small LMs **when hidden sizes are
    doubled**; small models prefer a *median*-based (not mean) quantizer and
    have different LR/WD sensitivity.
  - TernaryLM (132M native ternary, [arXiv:2602.07374](https://arxiv.org/abs/2602.07374)):
    stable training, and an implicit-regularization bonus — discrete weights
    resisted overfitting (train/val ratio 1.05 vs 3.51 FP) on limited data.
    Directly relevant: Thunk trains on small curated corpora.
- **ParetoQ** ([arXiv:2502.02631](https://arxiv.org/abs/2502.02631)): ternary,
  2-bit, 3-bit on the Pareto frontier; pure binary off it; ≤2-bit reshapes
  representations, so from-scratch QAT (which we'd do anyway at this size)
  beats fine-tune conversion.
- **Working budget rule** (synthesis, not a quoted constant): expect ~1.5–4x
  params or ~2x hidden width to match an FP16 baseline below 500M. With ~10x
  weight compression, that trade is still strongly net-positive for
  memory-bound edge inference.

### Training mechanics (what "bits in the floats" commits us to)

- Latent FP master weights; forward pass quantizes on the fly (absmean
  ternary weights, per-token absmax int8 activations); straight-through
  estimator backward; optimizer state in FP. Training cost ≈ the FP model
  plus a small overhead — **budget no training savings**.
- Stability recipe from the b1.58 line: RMSNorm immediately before every
  quantized linear, **no biases anywhere**, ReLU² instead of SwiGLU in FFNs,
  unusually high peak LR with a two-stage schedule (high-LR cosine → abrupt
  drop → low-LR cosine), weight decay 0.1 in stage 1 → 0 in stage 2. At our
  size, add BitNet-Reloaded's median-based scaling.
- Keep **embeddings, norms, the output head, and the halt/exit machinery in
  higher precision** (standard b1.58 practice; embeddings dominate a tiny
  model's byte budget anyway and quantize to int8 at export, not ternary).
- Export path: pack ternary weights 4-per-byte (plus scales); inference via
  lookup-table kernels — bitnet.cpp ([github.com/microsoft/BitNet](https://github.com/microsoft/BitNet))
  proves 1.4–6x CPU speedups and 55–82% energy reduction for exactly this
  format. A custom kernel for our small shapes is a contained project.

### The genuinely novel risk: STE noise × recursion

No published system trains a quantized depth-recurrent LM. Nearest neighbors:
MatMul-free LM (ternary weights + recurrent token mixer,
[arXiv:2406.02528](https://arxiv.org/pdf/2406.02528)) and ternary LSTMs
(ICLR 2019, [arXiv:1809.11086](https://arxiv.org/abs/1809.11086)) — evidence
that recurrence and low-bit weights *can* coexist, but nothing at
transformer-loop granularity. The specific unstudied hazard: quantization
error re-enters the loop every iteration, and convergence-test halting
assumes smoother latent dynamics than a ternary forward pass may provide
(latents may orbit rather than converge). This is why the plan quantizes
**last**, after the FP recurrent model and its halting behavior are
established, and ablates a ternary core against an FP core at matched width.

---

## Proposed reference configuration

Everything below is a starting point to be ablated, not a commitment.

| Parameter | Value | Rationale |
| --- | --- | --- |
| Shape | prelude 2 / core 2 (looped) / coda 2 | Huginn template scaled down; core stays tiny because loops supply depth |
| `d_model` | 256 (FP baseline) / 512 (ternary variant) | Reloaded: ~2x width for ternary parity |
| Heads | 4, `head_dim` 64, GQA on decode later | matches model #0 conventions |
| FFN | SwiGLU in FP phases; ReLU² in ternary phase | b1.58 recipe |
| Norms | RMSNorm pre-norm + QK-norm; extra RMSNorm before each quantized linear in ternary phase | model #0 + b1.58 conversion evidence |
| Loop count (train) | r ~ log-normal Poisson, mean 4, max 8; truncated BPTT k=4 | Huginn; Ouro's R=8 instability warning |
| Halting (infer) | per-token KL threshold ~5e-4 between successive loop outputs; mod-k KV cache, k=4 | Huginn defaults |
| Tokenizer | model #0's SentencePiece BPE (8k now, grows with #0) | composability |
| Params | ~10M FP-equivalent (≈2M embedding + ≈8M body) | trains on one consumer GPU / CI |
| Ternary body size at export | ~2–3 MB packed (+int8 embeddings ~2 MB) | the whole point |

Deep supervision (model #1 style) vs plain next-token training with randomized
loops: start with the latter — it is the proven recipe for *language models*
(Huginn/Ouro); deep supervision remains model #1's lane for canvas tasks.

---

## Phases

### Phase 0 — FP depth-recurrent LM baseline

Build prelude/core/coda with a **genuinely weight-tied core**, randomized
loop counts, truncated BPTT. Train on TinyStories-class data (the v0
prototype's corpus) or model #0's generated corpus at LM framing.
Deliverables: val PPL vs (a) iso-param feedforward, (b) iso-FLOP feedforward;
val PPL as a function of test-time r (must improve monotonically to ~2x
training mean, no collapse).
**Exit gate:** more test-time loops → better quality, from a model trained on
one GPU. If recursion buys nothing here, stop and rethink before touching
quantization.

### Phase 1 — Halting

Implement per-token KL-threshold exit + mod-k KV cache. Ablate: threshold
sweep, TRM-style BCE halt head vs the training-free test, per-token vs
per-sequence exit. Instrument depth histograms by token type.
Deliverables: quality-vs-average-depth curve; chosen exit criterion.
**Exit gate:** ≥30% average compute reduction at <1% quality loss vs always
running max loops (Huginn-class result, scaled expectations).

### Phase 2 — Ternary QAT

Apply the b1.58 recipe (absmean/median ternary linears in the core — and
prelude/coda if stable; int8 activations; extra pre-linear RMSNorm; ReLU²;
two-stage LR; WD 0.1→0) from scratch at 2x width. Ablate ternary core vs FP
core; re-run all Phase 1 halting measurements — **this is where the novel
risk lives**; check that latents still converge rather than orbit.
**Exit gate:** ternary-512 within a few percent of FP-256 on Phase 0/1
metrics, and halting still functions.

### Phase 3 — Export & composition

Pack ternary weights + int8 embeddings; a CPU LUT-kernel inference path
(bitnet.cpp-style, or bitnet.cpp itself if the architecture can be expressed
in its graph); measure bytes, tok/s, and energy on a laptop-class CPU.
Expose `embed()`-style latent access and halt-depth metadata; demonstrate one
two-link chain (model #0 translate → model #2 reason) over the shared
tokenizer.
**Exit gate:** a single-digit-MB artifact that runs faster than FP on CPU and
composes with model #0.

---

## Cross-cutting risks

- **Double-novelty**: sub-100M recurrent LMs and sub-500M ternary LMs are
  each thin literature; their combination is empty. Internal ablations, not
  literature numbers, are the ground truth here. The FP baseline at every
  phase is the insurance policy (same discipline as model #1's plan).
- **Perplexity tax**: looped models trade memorization for manipulation.
  If model #2's job in the chain is reasoning over a Perceiver's output, that
  trade is aligned — but verify on chain-relevant evals, not just PPL.
- **Halting under quantization** is the named unknown (Phase 2 gate exists
  for exactly this).
- **Training cost is not reduced** by ternary weights; CI CPU runs that
  worked for model #0's small config will be slower here (loops multiply
  FLOPs). Budget GPU time from Phase 0 (see `DISTRIBUTED_TRAINING.md` at the
  repo root for why that GPU should be local/rented, not a distributed
  network).
- **Scope discipline**: model #1's harness drifted from its plan in three
  load-bearing places (stablemax, halt usage, weight tying). Phase gates
  above are written to make the same drift visible immediately.
