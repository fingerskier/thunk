# Model 2

Small recurrent / iterative LM experiment with cheap discrete cores.

**Start simple.** See [`PLAN_CODEX.md`](PLAN_CODEX.md).

## Direction

- Train a normal tiny NN first (M0)
- Binary weights via STE (M1)
- Pack bits in **`uint32`** for inference / tests (M2)
- **Int-wise** cores (popcount scores) then optional **bitwise** word ops (M3–M4)
- Shared recurrent application (M5)
- Attention later (M6)
- Auto-halt later (M7)

Float residual / embeddings / head stay ordinary while the core gets discrete.
