# Model 2

Small iterative LM test: **dual-level** discrete core.

See [`PLAN_CODEX.md`](PLAN_CODEX.md).

## Thesis

Same graph, two grains of forward/back:

1. **Int-wise** — normal NN dynamics on word/score nodes and int-edges  
2. **Bit/boolean** — dense bit block on each int-edge (`uint32` words; all bits talk iff the ints are connected)

Not pack-only-at-infer. Train both levels (STE / arithmetic twin on bits; ordinary grads on int scores).

## Build order

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
