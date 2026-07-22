# Model 2: Binary / Integer Core Recurrent SLM

## Goal

A small language model that eventually spends most of its compute in cheap
**bitwise** and **int-wise** cores, with a thin floating-point control plane for
stability, attention (later), and auto-halting (later).

Do **not** start complicated. Prove a normal network first. Add one idea at a
time. Attention and auto-halt are **later** phases, not part of the first train
loop.

## Build order (deliberately simple)

```text
M0  Tiny float NN baseline          ← start here
M1  Trainable binary weights (STE)  ← still ordinary tensors
M2  Packed uint32 reference path    ← inference / equivalence only
M3  Int-wise core                   ← popcount scores, int accumulators
M4  Bitwise core                    ← word logic on uint32 registers
M5  Shared recurrent cell           ← apply core repeatedly
M6  Attention (optional, later)
M7  Auto-halt (optional, later)
```

Rule: each milestone must keep the previous milestone’s tests green. If a new
stage breaks overfit or equivalence, fix or drop it before adding the next.

## What we are *not* doing first

- Not auto-halting on day one
- Not attention on day one
- Not training through packed Python bit kernels
- Not a deep pure-Boolean net
- Not storing weight bits inside float payloads as the main format
- Not binarizing embeddings + head + state all at once

## Two discrete compute styles

Both are real options. They are different, and should be added separately.

### A. Bitwise network (`uint32` registers)

Operates on packed bits with word ops:

- `XOR`, `AND`, `OR`, `NOT`
- shifts / rotates
- permutations across words
- optional XNOR against packed weight words

Good for: extremely cheap mixing, hardware-like cores, residual bit updates
(`x ^= f(x)`).

Harder to train when stacked deeply. Prefer fixed or lightly learned masks at
first; use STE on shadow bits only where needed.

### B. Int-wise network (scores / accumulators)

Operates on integer features, usually produced by popcount / binary linear:

```text
score = 2 * popcount(xnor(w_bits, x_bits)) - n   # in [-n, +n]
s <- s + score
bit <- 1 if s > threshold else 0
```

Good for: more trainable discrete depth, richer state than 1-bit, clean
`uint32` in / `int32` out boundary.

**Default preference after the float baseline:** int-wise core first, bitwise
embellishments second. Int scores are a gentler step from ordinary NNs than pure
Boolean circuits.

### Hybrid that is still simple

```text
float residual bus
  -> binarize / pack once
  -> int-wise block (binary linear + int accumulate + threshold)
  -> optional cheap bitwise mix on the packed bits
  -> float lift (scale, bias, norm)
  -> add back to residual
```

Stay in discrete space for a few micro-ops only after pack/unpack is proven.

## Representation

### Packed storage

Canonical packed container: **`uint32`**.

- one word holds 32 binary values
- prefer unsigned for bitfields
- use `int32` (or wider) for popcount scores and accumulators
- do **not** use numeric `float32` as a bit pack (NaNs, 24-bit int precision)

`float32` bitcast transport is a later experiment only, never the source of truth.

### Binary value convention

Learned binary weights / activations use `{-1, +1}` in the arithmetic twin:

- packed bit `0` → `-1`
- packed bit `1` → `+1`

Document bit order inside a word (recommend: bit `i` = dimension offset `i`).
Pad widths to multiples of 32 for the test model (`d ∈ {64, 128, 256}`).

### Train vs infer

| Form | Role |
|---|---|
| Shadow `float` weights `θ` | trained by optimizer |
| Effective `sign(θ)` | forward arithmetic twin |
| Packed `uint32` | inference + bitexact tests |

Optimizers never update packed words directly.

## Forward math (reference)

Arithmetic twin (training / correctness):

```python
b = binary_sign(theta)          # ±1
y = x @ b.T                     # or binarized x for full binary linear
```

Packed twin (inference), both sides binary:

```text
matches = popcount( ~(w ^ x) & mask )
dot     = 2 * matches - n_valid
y       = alpha * dot + bias
```

Exact agreement between arithmetic and packed paths is a hard gate before any
speed claims.

## Backprop (keep boring)

Use STE on `sign` for binary weights:

```text
forward:  b = sign(θ)           # map 0 → +1
backward: d b / d θ ≈ 1_{|θ|≤1}
```

Train the arithmetic twin. Do not require differentiable packed kernels in M0–M4.

Recurrence (when added): shared cell, BPTT over a small fixed number of steps.
No halt policy until M7.

## Milestone specs

### M0 — Tiny float NN baseline

Small LM or sequence model that can overfit a toy corpus.

Include:

- embedding
- one or two float linear / MLP blocks
- output head
- standard CE train loop
- CPU unit tests

Exit:

- overfit toy data
- deterministic forward test
- save/load round-trip

This is the scaffold everything else plugs into.

### M1 — Binary weights (arithmetic only)

Replace selected linear layers with:

```text
y = x @ sign(θ).T
```

plus learned per-out scale `alpha` and optional bias.

Still float activations and float residual.

Exit:

- effective weights are only ±1
- shadow θ gets finite grads
- bits can flip under SGD/Adam
- toy overfit still works
- quality vs M0 recorded (expect some drop)

### M2 — `uint32` pack path

Add pack/unpack utilities and a CPU packed binary-linear reference.

No new architecture yet.

Exit (property tests):

- pack∘unpack identity
- unused pad bits do not affect results
- arithmetic twin == packed twin bitexact on random cases
- odd sizes, all-zero, all-one, alternating patterns covered

### M3 — Int-wise little core

First “subnet” that is more than one matvec:

```text
# simple default (S_int)
s = BinaryLinear(x_bits)              # int scores
s = s + BinaryLinear(threshold(s))    # second int pass
y = alpha * s + bias                  # float lift
```

Or even simpler first cut: single BinaryLinear + int threshold + second
BinaryLinear.

Exit:

- arithmetic and packed int-core agree
- residual float bus still present
- overfit holds or is only mildly worse than M1
- log score histograms and threshold saturation

### M4 — Bitwise little core

Add optional word-level ops on packed activations:

- learned or fixed XOR masks
- rotates within `uint32`
- cheap bitwise residual: `x ^= P(f(x))`

Keep depth tiny (1–2 ops). This is exploratory, not required to beat M3.

Exit:

- each bitwise op has a tested arithmetic or bitexact reference
- can be toggled off
- no obligation that M4 improves loss; measure cost and stability

### M5 — Shared recurrent cell

Apply the same core repeatedly for a **fixed** step count `T` (e.g. 2–8).

```text
h0 = embed(tokens) or pooled start state
for t in 1..T:
    h = h + lift(core(pack(h), context))
logits = head(h)
```

Exit:

- shared parameters actually shared (one module object)
- grads reach core from all steps
- quality vs depth curve for fixed T
- still no halt head

### M6 — Attention (later)

Only after M0–M5 are dull and reliable.

Start with ordinary float attention over token states, or a tiny attention mixer
outside the discrete core. Do **not** invent packed attention in the same step.

Possible light versions:

- float self-attention on residual stream
- cross-attention into discrete-core readout
- later experiment: binary / low-bit attention scores (research, not default)

Exit:

- attention improves a multi-token toy task vs M5 bag/recurrent baseline
- discrete core still optional/toggleable

### M7 — Auto-halt (later)

Only after fixed-T recurrence works.

```text
p_t = sigmoid(halt_head(h_t))
loss = task + λ * expected_steps
```

Inference: stop when `p_t > τ`, always clamp to `T_max`.

Exit:

- no collapse to always-1 or always-T_max without penalty knob response
- report quality vs mean steps
- hard halt is inference-only until soft policy is stable

## Suggested first test model

Keep it embarrassingly small:

| Knob | First value |
|---|---|
| `d_model` | 128 (4 × `uint32`) |
| vocab | tiny toy / char / wordpiece playground |
| layers | 1 core block |
| recurrence T | 1 (M0–M4), then 2–4 (M5) |
| attention | off |
| halt | off |
| train path | float or ±1 arithmetic |
| pack path | test + optional infer |

Success for the whole exploration is not “beat a transformer.” Success is:

1. M0 overfits
2. M1 still learns with binary weights
3. M2 matches M1 exactly when packed
4. M3 shows int-wise depth is usable
5. M5 shows repeated core application is stable
6. only then revisit attention / halt

## Training notes

- Optimizer: AdamW is fine on shadow floats and float control params
- STE: clipped identity first
- Watch: fraction of `|θ| > 1` (dead STE region), bit-flip rate, score saturation
- Keep embeddings + LM head float through M5 at least
- Prefer pre-norm residual blocks when recurrence starts

## Benchmarks (only what matters per stage)

| Stage | Measure |
|---|---|
| M0–M1 | loss, overfit, param count |
| M2 | bitexact match, pack overhead |
| M3–M4 | loss, stability, discrete depth benefit |
| M5 | loss vs T, grad norms, state drift |
| M6 | seq task gain from attention |
| M7 | loss vs mean steps |
| any packed infer | latency breakdown: pack / core / lift / head |

## Risks (short list)

1. **Too many ideas at once** — default fail mode; follow the milestone order.
2. **Binary capacity drop** — mitigate with `alpha`, wider `d`, float head/emb.
3. **Int/bit depth kills grads** — keep cores shallow; residual float bus always on.
4. **Packing with no speedup** — acceptable in research path; do not chase kernels before M3–M5 semantics work.
5. **Halt + discrete + attention together** — forbidden as a first integration.

## Immediate next implementation

Implement **M0 only**:

1. `model.py` — tiny float LM / classifier on toy sequences
2. `train.py` — overfit loop
3. `tests/test_m0_overfit.py` — red/green
4. leave stubs or comments for M1 binary linear, nothing more

After M0 is green, add M1 binary linear behind a flag (`weight_mode=float|binary`).

## Design stance

- **Bitwise** and **int-wise** are both on the roadmap; int-wise is the first
  discrete subnet.
- **`uint32`** is the packed fabric; **`int32`** is the score fabric; **float**
  is the control bus.
- **Attention** and **auto-halt** are features of a later recurrent LM, not the
  bootstrap.
- Complexity is earned by green tests, not by the initial sketch.
