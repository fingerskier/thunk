# Model 2 — Review of `PLAN_CODEX.md`

Review dated 2026-07-22 against `PLAN_CODEX.md` and `README.md` as written.
This document is **critique only** — it does not replace the plan. Where it says
"edit," that edit has not been applied.

**Overall:** discipline is good. Milestone gating, arithmetic-vs-packed
equivalence tests, explicit falsifiers, and the `uint32` storage / `int32` score
/ `float` control split are all correct and worth keeping verbatim. The weakness
is in the stated goal itself — *taking advantage of nominal binary
ops/arithmetic*. Two structural holes, plus several standard BNN mechanisms that
are absent.

---

## Blockers

### B1 — The dual-level thesis, as formalized, is block-sparse binary linear

The thesis table (§Core thesis) promises an int level with **edge weights
`e_ji`** and "normal grad through those int ops." The formal view (§Formal view)
then defines:

```text
s_j = sum_{i : A[j,i]=1}  popcount_xnor(W_{j,i}, x_i)
```

`A` is a fixed 0/1 mask. `e_ji` has vanished. There are no learnable int-level
parameters and no int-level gradient — every trainable quantity lives in the bit
blocks `W_{j,i}`. So M3 is exactly M1 with a block-sparsity mask applied, and the
measured M3-vs-M1 delta the plan wants is structurally ~zero.

The plan's own falsifier — *"int path adds nothing over one flat binary matmul →
drop hierarchical graph"* — is guaranteed to fire under the current formalization,
regardless of experiment. That is not a test.

Pick one:

- **(a) Honest rename.** Call it block-sparse BNN. Still worth building, still
  gives the packing story, but drop "dual-level" and "two grains of forward/back."
- **(b) Give the int level real parameters.** Then the two grains genuinely
  exist and "does the int level move independently of the bit level" becomes
  measurable. Candidates, cheapest first:
  - per-edge float scale `alpha_ji` (one float per int-edge)
  - per-lane learnable threshold / bias at the int node
  - learned gate `g_ji` over `A` (Gumbel-sigmoid or L0), making topology itself
    a trained object

(b) is what the README's thesis actually claims. Recommend (b), with `alpha_ji`
as the minimum viable version.

### B2 — The suggested first test model cannot test the thesis

`d_model = 128` → `m = d_model/32 = 4` int nodes → 16 possible edges. Any claim
about learned or structured int-graph topology over 4 nodes is vacuous; the
graph has no room to be interesting, and sparsity patterns are indistinguishable
from noise.

Testing the thesis needs `m >= 16`, preferably `m = 32`:

| Stage | `d_model` | `m` |
|---|---|---|
| M0–M2 | 128 | 4 (fine — no graph claim yet) |
| M3+ | 512–1024 | 16–32 |

Edit §Suggested first test model accordingly, and note the reason inline so the
number doesn't get "simplified" back down later.

### B3 — No milestone binarizes activations

This is the one that blocks the stated goal.

XNOR-popcount requires **both** operands binary. The plan keeps the float
residual bus through M5 ("residual float bus always on," M3 exit: "residual
float bus still present," Training notes: "Keep embeddings + LM head float
through M5 at least"). With float activations, packed weights buy **storage
only** — there is no popcount kernel, no int accumulator, no binary arithmetic.
M2 and M3 as written therefore never exercise the thing the model exists to test.

M3's pseudocode papers over this: `IntGraphBinaryLinear(x_words, A)` takes
*packed words* as input, but nothing upstream produces them and the exit
criteria explicitly retain the float bus.

**Insert a milestone between M2 and M3** (call it M2.5 or M3a) that owns the
activation side end to end:

- `x_b = sign(norm(x))`, STE with the same clip window as the weight sign
- per-lane (or per-tensor) activation scale `alpha_x`, and where it is applied
- the exact **pack point** (float bus → `uint32`) and **unpack/lift point**
  (`int32` scores → float bus)
- exit: arithmetic twin == packed twin bitexact with *both* sides binary;
  toy overfit survives; quality drop vs M1 recorded

This is where training actually breaks — binarized activations are much harder
than binarized weights. Burying it inside M3 means the first real difficulty
arrives simultaneously with the first architectural novelty, which is precisely
the failure mode Risk #1 warns about.

---

## Missing BNN mechanisms

These are not refinements. Omitting any of them will produce a "binary doesn't
train" result that is an implementation artifact.

### M1 — Normalization before `sign` / `threshold`

The single most load-bearing trick in the BNN literature is a normalization
immediately before the sign/threshold. It keeps the score distribution centered
as shadow weights drift, which is what keeps the flip rate from collapsing to 0%
or exploding to 50%.

§M3 has bare `threshold(s)`. Add: `s -> RMSNorm/BatchNorm -> threshold`. Same for
any activation binarization from B3.

### M2 — `threshold()` has no STE

§Backprop covers `sign` only. §M3 uses `threshold`, which is equally
non-differentiable. Specify its STE (hardtanh derivative, `1_{|s - t| <= c}`)
and, if the threshold is learnable, how gradient reaches `t`.

### M3 — Weight decay on shadow `theta` is actively harmful

§Training notes says "AdamW is fine on shadow floats and float control params."

For binary nets it isn't. Decay pulls `theta` toward 0, which is the region of
**maximum bit-flip noise** — weights sitting near zero flip sign on nearly every
step, so the effective network churns while the loss looks stuck.

Use `Adam` with `weight_decay = 0` on `theta`. Apply decay only to float control
parameters (`alpha`, biases, norm params, embeddings, head).

### M4 — Clip `theta`, don't just watch it

§Training notes says to *watch* the fraction of `|theta| > 1`. Watching does not
fix it. Once `|theta| > 1` the clipped-identity STE returns exactly zero and that
weight is permanently frozen — the metric will simply record the network dying.

Clip `theta` to `[-1, 1]` after every optimizer step. Keep the `|theta| > 1`
metric as a sanity check that the clip is wired up.

### M5 — `alpha` initialization

`sign(theta)` has row norm `sqrt(n)` regardless of what the float baseline
learned. Initialize `alpha_out = mean(|theta_out|)` (the XNOR-Net analytic
scale), or calibrate on the first batch. Without this, M1 loss explodes at step 0
and reads as "binary weights can't learn."

### M6 — BPTT × STE compounding at M5

A shared binary cell unrolled `T` steps multiplies the STE approximation error
per step. Mitigations to write into §M5:

- float state (the plan already has this — keep it)
- per-step normalization inside the cell
- log grad norms **per unroll depth**, not just globally
- fallback: a BOP-style flip-threshold optimizer (accumulate momentum, flip a bit
  only when momentum exceeds a threshold) if flip churn oscillates

---

## Ordering

### O1 — Swap M4 and M5

The model is a *recurrent* SLM; §README and the title both say so. Recurrence
(M5) is load-bearing. M4 (bitwise refinements on existing int-edges) is explicitly
optional — its own exit criteria say "no obligation that M4 improves loss."

Do recurrence first. Order becomes:

```text
M0  float baseline
M1  binary weights
M2  uint32 pack equivalence
M2.5 binary activations           ← new, see B3
M3  int graph + bit block per edge
M4  fixed-T recurrence            ← was M5
M5  bitwise edge refinements      ← was M4, now optional garnish
M6  attention
M7  auto-halt
```

---

## Evaluation

### E1 — Overfit-only exit criteria are too weak

M0/M1 gate on "overfit toy data." That gates *plumbing*, not the hypothesis.
Binary nets overfit small data fine; they generalize worse. Generalization is the
entire open question, and the current criteria cannot see it.

Add a held-out split and a real small corpus as the M1/M3 comparison bed —
1MB of enwik8, or a TinyStories slice. `script/glean_datasets.py` already exists
in this repo. Report train and val loss at every milestone from M1 onward.

### E2 — Control arms must be compute-matched, not param-matched

Comparing a sparse int-graph model against a dense binary linear at equal
*parameter count* flatters the sparse one (it gets more `d_model` for the same
budget). Run at least:

- equal **bit budget** (total weight bits)
- equal **edge count** (total 32×32 blocks evaluated)

Otherwise a positive M3 result is unfalsifiable in the direction that matters.

### E3 — The packed-latency benchmark is a trap

§Benchmarks asks for "latency breakdown: pack / core / lift / head" on the packed
inference path. In this stack that number will be meaningless-to-embarrassing:

- torch has no `popcount` for integer tensors
- numpy gained `bitwise_count` only in 2.0
- a Python/numpy packed path runs orders of magnitude slower than the equivalent
  float matmul, which is dispatched to BLAS

Mark the packed path **correctness-only**. Any speed claim requires a C
extension, a CUDA kernel, or a bitserial trick — and none of those should be
attempted before M3–M4 semantics are stable (Risk #4 already says this; the
Benchmarks table contradicts it). Fix the table so nobody reports a fabricated
regression.

---

## Test additions

Cheap, high signal, none of them currently in the plan:

1. **Single-bit sensitivity.** Flipping one weight bit changes the corresponding
   output by exactly `2 * alpha`. Strong invariant on the packed path; catches
   sign-convention and bit-order bugs that pack/unpack round-trips miss.
2. **STE support test.** Gradient is exactly `0` outside `|theta| <= 1` and
   nonzero inside. Do **not** finite-difference-check the STE — it is wrong by
   construction and the test will fail correctly for the wrong reason.
3. **Init tiebreak.** Assert non-zero symmetric init so the `sign(0) -> +1`
   tiebreak never fires at step 0.
4. **Non-neighbor isolation** (plan already has this — keep). Zeroing `A[j,i]`
   removes all influence of word `i` on score `j`, verified by perturbation, not
   by reading the mask.

---

## Keep as-is

Explicitly worth not touching:

- The milestone gating rule ("each milestone keeps the previous milestone's tests
  green; fix or drop before adding the next")
- §What would falsify the thesis — the right instinct, just needs B1 fixed so the
  falsifiers can actually discriminate
- §What we are *not* doing first — every line of it
- The `uint32` packed / `int32` score / `float` control fabric split
- Parameter-count sanity: `m^2 * 32 * 32 = d_model^2` checks out
- Pack/unpack property tests including pad-bit and degenerate-pattern coverage

---

## Concrete edits to `PLAN_CODEX.md`

| # | Section | Edit |
|---|---|---|
| 1 | Core thesis / Formal view | Resolve B1 — add `alpha_ji` (min) or learned gate `g_ji`, or rename thesis to block-sparse BNN |
| 2 | Suggested first test model | `d_model` 128 for M0–M2, 512–1024 for M3+, with reason inline |
| 3 | Build order + Milestone specs | Insert binary-activation milestone before M3 (B3) |
| 4 | M3 | Add norm-before-threshold; specify threshold STE |
| 5 | Training notes | Adam `wd=0` on `theta`; clip `theta` to `[-1,1]`; `alpha` init rule |
| 6 | Build order | Swap M4 / M5 — recurrence before bitwise garnish |
| 7 | M0/M1 exits + Benchmarks | Add held-out val + real 1MB corpus; compute-matched control arms |
| 8 | Benchmarks | Mark packed latency blocked on native popcount; correctness-only until then |
| 9 | Milestone exits | Add the four tests above |
