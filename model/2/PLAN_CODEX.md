# Model 2: Binary / Integer Core Recurrent SLM

## Goal

A small language model that eventually spends most of its compute in cheap
**bitwise** and **int-wise** cores, with a thin floating-point control plane for
stability, attention (later), and auto-halting (later).

Do **not** start complicated. Prove a normal network first. Add one idea at a
time. Attention and auto-halt are **later** phases, not part of the first train
loop.

## Core thesis (the point of this model test)

Run the network at **two coupled levels** on the same graph:

| Level | Object | Forward | Backward |
|---|---|---|---|
| **Int-wise** | word nodes + int-edges | normal NN-style accumulate/transform on int scores / edge weights | normal grad through those int ops (plus STE where discrete) |
| **Bit/boolean** | 32 bits inside each int, dense on each int-edge | boolean / XNOR-popcount / word logic realizing that edge | STE (or twin) grads into per-bit shadow weights and bit activations |

So it is not “pack bits only at inference.” The experiment is whether we can
**train and run both**:

1. a coarse **int graph NN** (who connects to whom, how scores mix), and
2. a fine **bit NN on each edge** (which boolean pattern implements that link),

with gradients flowing at both grains.

```text
        int path (coarse NN)
   s_i  ──────────────────────────►  s_j
          │                         ▲
          │ edge U→V                │ sum / lift
          ▼                         │
     bits(U) ── bit/boolean NN ──► contrib_V
        bit path (dense on this int-edge only)
```

### What “forward/back on both” means in practice

**Forward**

- Int level: `s_j = Σ_{i∈N(j)} f_int(s_i, e_ji)` (add scores, scales, thresholds).
- Bit level (per existing int-edge `i→j`):  
  `contrib_ji = popcount_xnor(W_ji, bits(x_i))` or a small boolean block on those 32×32 links,  
  then fold into `s_j`.

**Backward**

- Int level: ∂L/∂s and ∂L/∂(int edge params) as in a normal small NN.
- Bit level: ∂L/∂contrib flows into the edge’s bit weights via STE on `sign(θ_ji)`  
  (arithmetic ±1 twin in train; packed uint32 in infer).
- If activations are also binarized, STE into pre-bit continuous values or into  
  shadow bit logits — still scoped to int-neighbors only.

The two levels share topology: **bit paths are not a second arbitrary net**; they
are the interior of each int-edge. The int net is not merely bookkeeping; it is
the coarse activation space where residual/score dynamics live.

### Why bother (hypothesis)

- Coarse int dynamics stay closer to “normal NN” trainability.
- Fine bit blocks buy packing, cheap ops, and high fan-in per word.
- Coupling them tests whether dual-grain training beats  
  (a) float-only, (b) flat binary linear, (c) pack-at-infer-only.

### What would falsify the thesis

- Bit-level STE never moves edge patterns while int path fits alone  
  → bits are dead weight; flatten to int/float features.
- Int path adds nothing over one flat binary matmul  
  → drop hierarchical graph; keep packed binary linear only.
- Dual path trains but never matches arithmetic twin / packed twin  
  → implementation bug, not a modeling win.

### Milestone mapping

| Milestone | Dual-level status |
|---|---|
| M0 | float only (control scaffold) |
| M1 | bit weights inside ordinary linear (flat bit path; one big int) |
| M2 | packed realization of bit path |
| M3 | **explicit int graph + bit block per edge** (thesis online) |
| M4 | richer boolean interior on those same edges |
| M5+ | recurrence / attention / halt around the dual core |

M0–M2 still matter so M3 is a measured delta, not a pile of first bugs.

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

## Connectivity model (int graph owns bit edges)

Discrete topology is defined at the **integer / word** level, not as an
arbitrary bit graph.

### Rule

1. Nodes in the discrete core are **ints** (typically one `uint32` activation
   word, or an `int32` score lane associated with that word).
2. An **int-edge** `U → V` means “integer unit U feeds integer unit V.”
3. **Bitwise edges exist only along int-edges.**
4. If `U → V` is present, the default bit coupling is **dense all-to-all**:
   every bit of U may touch every bit of V (implemented as a 32×32 binary
   block, i.e. one packed-word binary linear / XNOR-popcount from U into V’s
   score).
5. There is **no** bit edge between words that are not int-connected.

```text
Int graph (coarse):          Bit graph (induced):

  U ──► V                      all 32 bits(U) ──dense──► all 32 bits(V)
  U ──► W                      all 32 bits(U) ──dense──► all 32 bits(W)
  X           (no edge)        bits(X) have no paths into V/W from X
```

### Why this is a good constraint

- **Sparse where it matters:** connectivity/search is over a small int graph
  (`d_model/32` nodes), not over thousands of bit nodes.
- **Dense where hardware is cheap:** a fully connected 32×32 binary block is
  exactly “one weight word per input bit lane into one output score,” i.e. the
  natural packed kernel, not a gather of random bit wires.
- **Matches int-wise first:** the learned object is “which ints talk,” then
  “what 32-bit pattern implements that talk.”
- **Keeps bitwise from exploding:** M4 bitwise ops are *refinements of an
  existing int-edge* (masks, rotates before the dense block, residual XOR on
  the same edge), not a second unrelated wiring language.
- **Easy to test:** int-adjacency is an explicit matrix/list; forbidden bit
  interactions are those outside the Kronecker-ish expansion of that adjacency.

### Formal view

Let int-activations be words `x_0..x_{m-1}` each in `{0,1}^32` packed as
`uint32`. Let `A[j,i] ∈ {0,1}` be int-adjacency (`i → j`).

Default dense-on-edge accumulation for output score lane `j`:

```text
s_j = sum_{i : A[j,i]=1}  popcount_xnor(W_{j,i}, x_i)
# W_{j,i} is 32 packed uint32 words (32 out-bits × 32 in-bits), or the
# arithmetic-twin ±1 matrix of shape (32, 32).
```

If several int sources feed `j`, their int contributions add in **int-wise**
space. Bits from a non-neighbor never enter `s_j`.

Optional later specializations of an existing edge (still no new bit endpoints):

| Edge refinement | Meaning |
|---|---|
| Full 32×32 binary block | default dense bit coupling |
| Shared mask + popcount | fewer parameters per edge |
| Rotate/XOR then dense | bitwise preconditioner on the same edge |
| Diagonal / bit-local | depthwise: bit k of U only to bit k of V |
| Low-rank bit block | factor 32×32 as 32×r · r×32 in ±1 or int |

Diagonal and low-rank are **restrictions** of the dense int-edge, not bypasses
of the int graph.

### What we explicitly avoid

- Bit-level sparse graphs with arbitrary cross-word wires unrelated to int `A`
- Per-bit neighbors that skip the int abstraction
- Treating bitwise and int-wise as two independently wired networks

Bitwise is not a second network. It is **implementation detail + optional
shape** on top of the int graph.

### Parameter counting (sanity)

For `m = d_model/32` int nodes and dense int graph (`m²` edges):

```text
bits in weights ≈ m² * 32 * 32 = d_model²
```

That matches a full binary `d_model × d_model` linear — good. If the int graph
is sparse (e.g. block-local, strided, or recurrent path graph), parameter count
and compute drop by the edge factor `|E|/m²`, while each existing edge stays a
cheap dense 32-bit kernel.

First test models should use either:

- **full int graph** (equivalent to ordinary binary linear), or
- **simple structured int graph** (chain / residual local / few blocks)

not a learned irregular bit mesh.

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

First subnet over the **int graph**. Default edge = dense 32×32 binary block
(all bits of source int → score lanes of dest int). No bit wires outside
int-adjacency (see Connectivity model).

```text
# simple default (S_int)
# A = int adjacency (full or structured)
s = IntGraphBinaryLinear(x_words, A)   # sum dense blocks over edges
s = s + IntGraphBinaryLinear(threshold(s), A)
y = alpha * s + bias                   # float lift
```

Even simpler first cut: one full int graph layer (== ordinary binary linear when
`A` is all-ones).

Exit:

- arithmetic and packed int-core agree
- tests that zeroing `A[j,i]` removes all influence of word `i` on score `j`
- residual float bus still present
- overfit holds or is only mildly worse than M1
- log score histograms and threshold saturation

### M4 — Bitwise refinements on existing int-edges

Bitwise ops may only precondition or residual-update along edges already in the
int graph. They must not introduce cross-talk between non-neighbors.

Examples on an edge `U → V`:

- XOR mask / rotate on `U` before the dense 32×32 block into `V`
- bit-local (diagonal) restriction of that block
- residual `U ^= f_edge(U→V)` only using that edge’s endpoints

Keep depth tiny (1–2 ops). Exploratory; not required to beat M3.

Exit:

- each bitwise op has a tested arithmetic or bitexact reference
- property test: no edge in `A` ⇒ no bit influence
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
