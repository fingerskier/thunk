# Distributed Training for Thunk: Psyche + Solana Evaluation

An evaluation of whether Thunk should train its models on Nous Research's
Psyche network (coordinated on Solana), or any comparable decentralized
training infrastructure. Facts verified against primary sources as of
August 2026; uncertainty flags noted inline.

---

## Verdict

**Technically possible today; economically and operationally unjustified at
Thunk's scale. Recommendation: no — keep training local/CI for now, with two
specific re-evaluation triggers below.**

The short version of the math: Psyche exists to solve two problems Thunk does
not have — aggregating GPUs *across organizations* for models too expensive
for one party, and establishing *trust between strangers* contributing
compute. Thunk's models (2M–112M params) train on one consumer GPU in hours
to days; a 20M model's fp16 gradients are ~40MB per sync, shippable over
ordinary broadband without any of DisTrO's compression; and Psyche's fixed
per-round coordination overhead (on-chain phase transitions, witness windows,
warmup/cooldown) becomes the dominant cost as per-step compute shrinks —
wall-clock goes coordination-bound and GPU utilization collapses. Every
precedent for internet-distributed training at small scale (DiLoCo ~60–400M,
OpenDiLoCo 150M–1.1B, DeMo 300M–1B) is a research validation, not a
production choice; the history suggests distributed training starts paying
for itself around **~1B+ params**.

## What Psyche is

Psyche (Nous Research, launched May 2025; repo recently renamed `nousnet` —
rebrand apparently in progress) is open, fault-tolerant infrastructure for
training transformer LLMs across untrusted parties.
[nousresearch.com/nous-psyche](https://nousresearch.com/nous-psyche) ·
[github.com/PsycheFoundation/psyche](https://github.com/PsycheFoundation/psyche)

- **Coordinator = a Solana smart contract** (Anchor). It holds run config and
  the participant list, runs the epoch state machine
  (`WaitingForMembers → Warmup → RoundTrain → RoundWitness → Cooldown`),
  provides the randomness for data-batch assignment and witness election, and
  records witness attestations and reward points. **Gradients never touch the
  chain** — Solana is coordination and attestation only.
- **Clients** train assigned batches and exchange DisTrO-compressed results
  peer-to-peer over Iroh (QUIC); randomly elected witnesses submit
  Bloom-filter participation proofs.
- **Optimizer**: DisTrO/DeMo ([arXiv:2411.19870](https://arxiv.org/abs/2411.19870))
  — DCT + top-k sparsified momentum, ~2 orders of magnitude communication
  reduction (peer-reviewed at 300M–1B; marketing claims up to 857x–10,000x),
  plus 1-bit sign quantization and overlapped train/communicate in Psyche.
  Data-parallel only across nodes; tensor parallel only *within* a node — the
  model must fit on a single participant machine.
- **Track record**: Consilience 40B pretrain launched the testnet (May 2025;
  Nous states it completed, final token count unverified); Hermes 4.3
  (Dec 2025) was SFT-trained end-to-end on Psyche across 24 nodes and matched
  a centralized FSDP run on downstream evals — the strongest evidence the
  stack works as claimed.
  [nousresearch.com/introducing-hermes-4-3](https://nousresearch.com/introducing-hermes-4-3)
- **Open source**: Apache-2.0, Rust core (libtorch/TorchTitan under the
  hood), Solana programs included. Third parties **can** create runs today:
  permissionless or per-wallet-authorized joining, your own SPL token as
  optional rewards, documented run-creation CLI. Docs target Solana
  **devnet**; no verifiable mainnet migration announcement (flag). Public
  commit activity stops around March 2026 (flag: development may have moved
  private).
- **Caveats that matter**: verification of training *correctness* is an
  acknowledged open problem — witness proofs attest participation, and every
  shipped config sets `verification_percent = 0`; real runs were
  permissioned. Protocol-wide token incentives are plumbing-complete but not
  live ("once rewards are implemented"). Model support is causal decoder-only
  (native Llama and DeepSeek implementations; new architectures mean
  implementing a Rust `CausalLM` trait — the Python/HF sidecar is explicitly
  alpha). Data must be pre-tokenized fixed-size batches.

## Could Thunk run on it?

Mechanically, mostly yes — with real integration friction:

| Thunk piece | Fit |
| --- | --- |
| model/0 (encoder-decoder translator) | **Poor.** Psyche models are causal decoder-only; an encoder-decoder needs a custom Rust `CausalLM`-trait implementation that abuses the interface, or the alpha Python sidecar. |
| model/1 (recursive diffuser, deep supervision) | **Poor.** The training loop is nonstandard (supervision steps = denoising steps); Psyche's trainer abstraction covers pretraining/SFT, not custom inner loops. |
| model/2 (recurrent decoder-only LM, planned) | **Closest.** Decoder-only and next-token-trained, but the randomized loop counts and truncated BPTT still deviate from the stock trainer. |
| Scale (2M–112M) | Psyche's own dev configs are 20M-param Llamas (`config/llama2-20m-*`) run with 2–3 clients — so the software imposes no minimum, and our scale is literally its *test fixture*, not its use case. |

Notably, Psyche ships a **Solana-free centralized mode** (`architectures/
centralized/`: TCP coordinator + clients, `just local-testnet`) — the same
training stack minus the chain. If Thunk ever wants multi-node runs on
machines *we* control, that mode (or plain DiLoCo, below) is the sensible way
to use this codebase, and none of the blockchain/witness/token machinery is
load-bearing for us.

## Alternatives, briefly

- **Prime Intellect** ([INTELLECT-1](https://www.primeintellect.ai/blog/intellect-1),
  [INTELLECT-2](https://arxiv.org/abs/2505.07291)): most productionized
  alternative; OpenDiLoCo/prime frameworks, no blockchain in the loop.
  Tellingly, their INTELLECT-3 (106B) was trained on a *centralized* cluster
  — even the flagship decentralized-training company centralizes when it can.
  Their compute marketplace is a fine place to rent a single training GPU.
- **Templar / Bittensor SN3**: the only live *permissionless + incentivized*
  pretraining network (Covenant-72B completed Mar 2026,
  [arXiv:2505.21684](https://arxiv.org/abs/2505.21684)); trains one
  subnet-chosen model at a time — not a venue for running your own small run.
- **Gensyn**: verification-first protocol; testnet nodes currently paused; no
  practical path for our use.
- **Plain DiLoCo** ([arXiv:2311.08105](https://arxiv.org/abs/2311.08105)):
  inner AdamW steps + outer Nesterov averaging every ~500 steps, ~500x less
  communication, a few hundred lines of PyTorch. **The honest baseline**: if
  Thunk ever needs multi-node, DiLoCo across trusted rented nodes delivers
  the bandwidth savings with zero blockchain machinery.

## What we actually need instead

Thunk's training bottleneck is **data quality and evaluation, not compute**
(the model/0 SPEC's curated-corpus philosophy). The concrete infrastructure
gap is more modest: the current CI trains model/0's small config on a CPU
runner; model/2's looped training (see `model/2/PLAN.md`) multiplies FLOPs
per parameter and will want a single GPU. A rented A100/4090 for hours, or a
GPU CI runner, covers every model in this repo's pipeline for tens of
dollars — below the cost of even configuring a distributed run.

## Re-evaluation triggers

Revisit this decision if either becomes true:

1. **Scale crosses ~1B params** (e.g., a future chain-of-variants model or a
   large distillation teacher we want to own) — at that point DisTrO-class
   communication compression and multi-node aggregation begin to earn their
   overhead. Path: Psyche centralized mode or DiLoCo on rented nodes first;
   the public Solana network only if outside contributors join.
2. **Community co-training becomes a goal** — if Thunk variants (Maths,
   Coding, Theology…) attract contributors who want to donate GPU time to
   train "their" variant, Psyche is purpose-built for exactly that trust
   model: permissionless joining, on-chain attestation, optional SPL-token
   rewards per epoch, and public checkpoints. That is a community decision
   before it is a technical one; the technical path exists and is Apache-2.0.

An adjacent, cheaper idea in the same spirit: Thunk already publishes weights
via GitHub Releases with SHA256SUMS. If provenance/verifiability is the
attraction of "on-chain training," start by publishing training configs, data
manifests, and checkpoint hashes with each release — 95% of the auditability
at 0% of the coordination cost.

## Uncertainty flags (recap)

- Whether Psyche coordination runs on Solana devnet or mainnet today
  (docs say devnet as of the last public repo state, March 2026).
- Consilience 40B final disposition (completed per Nous; token count and
  annealed release unverified).
- No official Nous token as of mid-2026; third-party bandwidth
  requirements (10 Gb/s figures) are not from first-party sources.
- Public repo activity stops ~March 2026 (rename to `nousnet`); current
  development location unconfirmed.
