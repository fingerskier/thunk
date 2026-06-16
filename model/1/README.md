# Model #1 — Recursive Reasoning Diffuser

This directory implements the first recursive reasoning diffuser described in `PLAN.md`. It is intentionally small and experimental: the supervision loop is also the masked-denoising schedule, and the model carries only two mutable features between steps.

## Components

- `config.py` defines `ReasoningDiffuserConfig` and the tiny preset.
- `model.py` implements the shared recursive core, answer-canvas head, and halt head.
- `diffusion.py` provides the absorbing-mask schedule, masked CE objective, and confidence remasking.
- `train.py` wires TRM-style deep supervision: no-grad warmup recursions, one gradient recursion, detach between supervision steps, AdamW, and EMA support.

Run a CPU smoke test with:

```bash
python model/1/train.py
```
