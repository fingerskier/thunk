# Thunk Translation Model Specification

## 1. Purpose

Thunk is a compact translation model for converting meaning between equivalent representations:

- English -> English semantic rewriting
- code -> code translation/porting
- English -> code generation
- code -> English explanation
- foreign language -> English
- English -> foreign language
- document -> equivalent document in another style, language, or formalism

The model is intentionally a **translator**, not a general chat agent.
Its primary job is to **preserve meaning while changing surface form, language, programming language, style, or level of formality**.


## 2. Target Model

| Item | Target |
| --- | --- |
| Architecture | encoder-decoder Transformer, deep encoder / shallow decoder |
| Parameter budget | roughly 100M trainable parameters; not a hard cap (reference config lands ~110M) |
| Input context window | 2,048 source/input tokens per segment (RoPE base chosen for 4x extension headroom) |
| Output | autoregressive target tokens; target length is limited separately at decode time |
| Primary tokenizer | shared SentencePiece/BPE vocabulary with byte fallback |
| Primary training mode | denoising pretrain + supervised seq2seq translation + teacher distillation |
| Deployment target | **local workstation first, edge/mobile after quantization (fixed constraint)** |

This translation spec supersedes the older 128-dimensional Thunk prototype assumptions. Use 512-dimensional encoder states as the default representation; Matryoshka training (Section 8) makes truncated 256/128/64-dim views available for free, which recovers the old 128-dim use case without a separate model.

Encoder-decoder remains the right architecture for compact translation in 2025-2026. Every production-grade small MT stack is encoder-decoder (OPUS-MT/Marian, Firefox Translations/Bergamot, NLLB-200-distilled, HPLT v2), and recent work reaffirms it at small scale: T5Gemma (arXiv:2504.06225) and T5Gemma 2 (arXiv:2512.14856) show encoder-decoder matching or beating decoder-only at equal inference cost; "Return of the Encoder" (arXiv:2501.16273) reports ~47% lower first-token latency and several-fold throughput gains for sub-1B encoder-decoder models on edge hardware; RedLLM (arXiv:2510.26622) confirms the pattern from 150M to 8B. Decoder-only wins accrue at 7B+, which is irrelevant to this budget.

### 2.1 Build-vs-Warm-Start Decision

Train-from-scratch is the default plan, but evaluate one warm start before committing the full training budget:

- **T5Gemma 2 270M-270M** (`google/t5gemma-2-270m-270m`): a modern pretrained encoder-decoder (UL2-adapted from Gemma 3, 140+ languages, tied embeddings, merged decoder self/cross-attention). Fine-tuning it on the Thunk corpus may beat a scratch run at a fraction of the compute, at the cost of a larger embedding table and someone else's tokenizer.
- **Distill an existing teacher** into the scratch architecture below — the production-proven path (Mozilla/Bergamot, OpusDistillery). Mind teacher licenses: MADLAD-400 (Apache-2.0) and TranslateGemma (Google, Jan 2026, Gemma terms) are clean choices; NLLB-200 models are CC-BY-NC and should be used as evaluation baselines only.

If scratch and warm-start are close on quality, prefer scratch: it keeps the tokenizer, vocabulary, and embedding budget under Thunk's control.

### 2.2 Reference Configuration

This is the concrete v0 architecture unless experiments prove a different shape is better. The exact parameter count may move above or below 100M if quality, latency, or implementation simplicity justify it.

| Parameter | Value |
| --- | ---: |
| `vocab_size` | 49,152 shared source/target tokens (range 32k-64k; see Section 5) |
| `d_model` | 512 |
| `encoder_layers` | 16 |
| `decoder_layers` | 4 |
| `attention_heads` | 8 query heads, `head_dim` 64 |
| `kv_heads (decoder self-attn)` | 2 (GQA 4:1) |
| `cross_attention` | full MHA (8 KV heads) |
| `ffn_dim` | 2,048 |
| `activation` | SwiGLU |
| `normalization` | pre-norm RMSNorm + QK-norm; optional Gemma-style post-norm as stability ablation |
| `position_encoding` | RoPE, base 100,000 (self-attention only; none on cross-attention) |
| `biases` | none on any linear projection |
| `dropout` | 0.0-0.1 during training |
| `weight_tying` | one shared token embedding for encoder input, decoder input, and decoder output head |

Approximate parameter count:

| Component | Approx params |
| --- | ---: |
| Shared token embedding / tied LM head (49,152 x 512) | 25.2M |
| 16 encoder blocks | 67.1M |
| 4 decoder blocks | 19.4M |
| Final norms and small overhead | <0.1M |
| **Total** | **~112M** |

Why the layer budget moved from the encoder-balanced 10/8 split to 16/4:

- decode latency scales with decoder depth times target length; encoder cost is paid once per segment. "Deep Encoder, Shallow Decoder" (Kasai et al., ICLR 2021, arXiv:2006.10369) showed a 12-1 split matches 6-6 BLEU at >2.5x single-sentence decode speed;
- Firefox Translations ships exactly this asymmetry in production (`enc-depth: 6, dec-depth: 2` students); T5Gemma's unbalanced 9B-2B is the same idea at LLM scale;
- the edge/local target is fixed, so latency-per-quality is the primary objective.

An aggressive ablation worth one experiment: a 2-layer decoder, optionally with an SSRU recurrent decoder cell in place of self-attention (the Bergamot student recipe), distilled from the v0 model.

## 3. Model Structure

```text
source text/code/document
        |
        v
[shared tokenizer]
        |
        v
[source tokens + source/target tags]
        |
        v
+-----------------------------+
| Encoder, 16 layers          |
| - RMSNorm (pre)             |
| - 8-head self-attention     |
|   (QK-norm, RoPE)           |
| - residual                  |
| - RMSNorm (pre)             |
| - SwiGLU feed-forward       |
| - residual                  |
+-----------------------------+
        |
        v
encoder memory states
        |
        v
+-----------------------------+
| Decoder, 4 layers           |
| - RMSNorm (pre)             |
| - causal self-attn          |
|   (8Q/2KV GQA, QK-norm,     |
|    RoPE)                    |
| - residual                  |
| - RMSNorm (pre)             |
| - 8-head cross-attention    |
|   (QK-norm, no RoPE)        |
| - residual                  |
| - RMSNorm (pre)             |
| - SwiGLU feed-forward       |
| - residual                  |
+-----------------------------+
        |
        v
[target tokens]
        |
        v
translated text/code/document
```

### 3.1 Encoder

The encoder reads the complete source segment and builds contextual states suitable for both generation and semantic indexing.

Each encoder block contains:

1. RMSNorm
2. multi-head bidirectional self-attention with QK-norm and RoPE
3. residual connection
4. RMSNorm
5. SwiGLU feed-forward network
6. residual connection

The final encoder states are also the default representation to pool/project for semantic search or memory storage (Section 8).

### 3.2 Decoder

The decoder generates the target sequence autoregressively while attending to the encoder states.

Each decoder block contains:

1. RMSNorm
2. grouped-query causal self-attention (8 query heads, 2 KV heads) with QK-norm and RoPE
3. residual connection
4. RMSNorm
5. multi-head cross-attention over encoder memory states (QK-norm, no RoPE)
6. residual connection
7. RMSNorm
8. SwiGLU feed-forward network
9. residual connection

### 3.3 Attention Decisions

Updated from the earlier "full MHA everywhere, GQA later" position. GQA and QK-norm are now standard in newly designed small models (Gemma 3 270M, Qwen3-0.6B, OLMo 2, LFM2-350M all ship both; SmolLM2-135M uses 3:1 GQA) and belong in the v0 baseline, not in a later optimization pass:

- **Encoder self-attention: full MHA.** No KV cache exists here; GQA would save parameters but not memory bandwidth at decode time, and the encoder is where translation quality is built.
- **Decoder self-attention: GQA 8Q/2KV.** The decoder self-attention KV cache is the per-token decode cost on edge hardware; 4:1 grouping cuts it 4x with negligible quality loss at this scale.
- **Cross-attention: full MHA in v0.** Cross-attention is the alignment workhorse and its K/V are computed once per source segment (cache is source-length-bound, not generated-length-bound), so GQA buys little here. Note the original GQA paper validated GQA on T5 decoder self-attention *and* cross-attention with near-MHA quality, so GQA-ing cross-attention is a legitimate later ablation — just not the conservative default.
- **QK-norm (RMSNorm on queries and keys per head) everywhere.** Cheap, removes attention-logit blowups, and replaced soft-capping in Gemma 3 (arXiv:2503.19786).
- **No biases on linear projections; z-loss (~1e-5) optional** for long-run stability.


## 4. Control Tags and Translation Interface

Use minimal language tags so one model can learn multiple translation directions without a large control vocabulary.

Example inputs:

```text
<src:english> <tgt:python>
Write a CLI that renames all .jpeg files in the current directory to .jpg.
```

```text
<src:bash> <tgt:powershell>
for f in *.jpeg; do mv "$f" "${f%.jpeg}.jpg"; done
```

Recommended reserved tags:

- source tags: `<src:english>`, `<src:python>`, `<src:typescript>`, `<src:bash>`, `<src:cmd>`, `<src:powershell>`, `<src:lean>`, `<src:hebrew>`, `<src:greek>`, `<src:latin>`, `<src:aramaic>`, etc.
- target tags: `<tgt:english>`, `<tgt:python>`, `<tgt:typescript>`, `<tgt:bash>`, `<tgt:cmd>`, `<tgt:powershell>`, `<tgt:lean>`, `<tgt:hebrew>`, `<tgt:greek>`, `<tgt:latin>`, `<tgt:aramaic>`, etc.

The target language tag is mandatory.
Source tags should be present whenever known.
Each tag must be a single reserved token in the tokenizer (never split into subwords).


## 5. Tokenization

Use one shared tokenizer for source and target text so the model can copy names, identifiers, citations, and mixed-language fragments.

Default tokenizer requirements:

- SentencePiece (BPE or unigram) with **byte fallback** and `character_coverage` 0.9995-1.0
- **48k vocabulary for v0** (up from the earlier 32k). Vocabulary scaling laws (Tao et al., NeurIPS 2024, arXiv:2407.13623) put the optimum for a ~100M-body model in the 32k-64k band, and Thunk's mix — English, Hebrew, Greek, Latin, Aramaic, plus five programming languages — is exactly the case where 32k over-fragments. Measure fertility (tokens/word) per language on the curated corpus and adjust within 32k-64k.
- split digits; preserve whitespace-sensitive code structure (no whitespace normalization that would break indentation)
- reserve only explicit source/target language tags
- train on the same curated mixture used for model pretraining/fine-tuning, with low-resource languages temperature-upsampled so they are not under-segmented
- **audit fertility (tokens/word) per language** on held-out text before freezing the vocabulary; high fertility on a target language is a direct latency and quality tax

Rejected alternatives, for the record:

- **byte-level models (ByT5/BLT):** ~4-6x slower inference at small scale (arXiv:2302.14220) — disqualifying for the edge target;
- **SuperBPE** (arXiv:2503.13423): real token-efficiency gains (~27% fewer tokens) but no production adoption as of mid-2026, and there is MT-specific evidence that cross-word tokens hurt translation; acceptable as a later tokenizer experiment, not the v0 default.

Code tokenization must preserve exact spelling and indentation well enough for generated code to be executable after detokenization.


## 6. Training

### 6.1 Objectives and Phases

Train on parallel pairs:

```text
(source tokens + source/target language tags) -> target tokens
```

The decoder is trained with teacher forcing and standard next-token cross entropy.

Training proceeds in four phases (the old three-phase plan, upgraded with the now-standard distillation and preference-optimization stages):

- **Phase 1 — denoising pretrain (replaces plain self->self copy).** UL2-style span corruption / denoising plus prefix-LM on the monolingual corpus. This is what T5Gemma used to build strong encoder-decoder checkpoints; plain identity reconstruction is too easy and teaches copying, not representation.
- **Phase 2 — supervised seq2seq translation.** All parallel directions: English -> code, code -> English, English -> English paraphrase, English <-> foreign language, code -> code. Data passes the quality gate in Section 6.3 first.
- **Phase 3 — sequence-level knowledge distillation.** Forward-translate large monolingual/code corpora with strong teachers (frontier LLM for code/explanations; MADLAD/TranslateGemma-class or LLM teachers for natural language), keep the best of the teacher's n-best/MBR candidates rather than 1-best (arXiv:2407.10456; the Bergamot pipeline selects from 8-best), filter the synthetic pairs (Section 6.3), and train on them. Seq-KD remains the standard recipe for compact students (Kim & Rush 2016; KD4MT survey, arXiv:2602.15845); it is how the 17-40M Firefox Translations models reach production quality, and NewsPaLM (arXiv:2408.06537) showed LLM-generated, quality-selected parallel data beats a 300x-larger web corpus even for training small NMT from scratch.
- **Phase 4 (optional) — QE-driven preference optimization.** CPO (arXiv:2401.08417) built the ALMA-R results at 13B, and QE-as-preference variants are demonstrated on sub-1B encoder-decoder MT models (DQO at ~500M, arXiv:2409.17673; CRPO on NLLB, arXiv:2501.13927): generate multiple candidates, score with a QE metric (CometKiwi class), train on best-vs-worst pairs. Cheap (LoRA-sized) and worth one experiment after v1.

End Phase 2/3 with a short fine-tune on the small **human-quality** parallel subset (the ALMA stage-2 finding, arXiv:2309.11674: small clean data beats large noisy data for the final polish).

### 6.2 Auxiliary Objectives

Use auxiliary tasks only when they improve translation quality:
- round-trip consistency: A -> B -> A should preserve meaning
- code execution consistency: translated code should pass equivalent tests
- contrastive encoder pooling: equivalent source/target pairs should have nearby embeddings (this also trains the Section 8 embedding hook; see Matryoshka note there)

### 6.3 Data Quality Gate

All parallel data — mined, licensed, or teacher-generated — passes two filters before training (the Tower/EuroLLM pattern, arXiv:2402.17733, arXiv:2506.04079):

1. cheap filters: dedup, language ID, length-ratio, Bicleaner-style classifier;
2. neural QE filter: CometKiwi-class reference-free score, threshold ~0.7 for natural language. QE filtering improves quality while halving training data relative to heuristic filters ("There's No Data Like Better Data", arXiv:2311.05350). For code, the equivalent gate is execution: generated/translated code must parse, compile, and where possible pass tests before it enters the training set.

Distillation data must additionally be filtered for license compatibility and semantic equivalence.

### 6.4 Optimizer, Schedule, and Scale

- **Optimizer:** Muon on hidden 2D weight matrices + AdamW on embeddings, norms, and the output head (the nanochat/Moonlight split; arXiv:2502.16982; `torch.optim.Muon` ships in PyTorch 2.12). Independent benchmarks (arXiv:2509.02046) put the realistic gain at 1.2-1.4x over well-tuned AdamW at this scale — worthwhile, but plain AdamW is an acceptable fallback.
- **Schedule:** warmup-stable-decay — ~2,000-step warmup, long constant phase, final 10-20% linear-to-zero or (1-sqrt) cooldown (arXiv:2405.18392). WSD lets training extend without committing to a total step count up front, which suits an iteratively growing curated corpus.
- **Precision:** BF16 training; gradient clip 1.0; weight decay 0.1 (none on embeddings).
- **Scale expectation:** deployed small models are heavily overtrained — SmolLM2 at ~6,500 tokens/param, Gemma 3 270M at ~22,000 tokens/param. For a ~110M model that implies hundreds of billions of training tokens, far beyond the curated corpus. Plan for many epochs over the curated data plus large Phase-3 synthetic expansion; the curated corpus is the quality anchor, not the bulk tonnage.


## 7. Data Scope

The corpus should be curated for high signal rather than web scale.

Primary domains:
- English papers, literature and masterwork especially durable/high-value works
  - bonus for books and papers that have high-quality translations in multiple languages
- multiple Bible translations where licensing permits
  - include original language versions
- high-value scientific papers
  - especially with trusted translations
- Lean proofs and formal statements
- TypeScript
- Python
- bash, cmd, and PowerShell
- ubiquitous, top-tier open-source codebases
- task-oriented script pairs and explanations

Synthetic expansion (Phase 3) draws on the same domains: teacher forward-translations of the monolingual material, back-translations, and verified task pairs (problem -> script, error -> fix) per SOURCES.md. All of it passes the Section 6.3 gate.


## 8. Memory and Semantic Search Hook

Because the architecture is encoder-decoder, the encoder output can serve two jobs:

1. condition the decoder for translation;
2. provide embeddings for semantic search and persistent memory.

Default embedding strategy:

- mean-pool final encoder states (the EmbeddingGemma recipe, arXiv:2509.20354);
- train the contrastive pooling objective (Section 6.2) with **Matryoshka Representation Learning** (arXiv:2205.13147) so the 512-dim embedding truncates to 256/128/64 dims with graceful degradation. MRL is now standard across embedding models (OpenAI, Gemini Embedding, EmbeddingGemma, Qwen3-Embedding) and costs nothing at inference; it also restores the original 128-dim Thunk idea as a truncated view;
- if embedding quality matters before the contrastive objective is trained, follow the E5/GTE two-stage recipe (weakly supervised contrastive pretrain, then supervised fine-tune with mined hard negatives);
- store embeddings with source text, target text, tags, provenance, and quality metadata;
- **storage compression: this is where TurboQuant actually applies.** TurboQuant (arXiv:2504.19874, Google Research, ICLR 2026) is an online, data-oblivious *vector* quantizer — random rotation + per-coordinate optimal scalar quantization + a 1-bit QJL residual stage — whose demonstrated applications are KV-cache compression and nearest-neighbor-search vectors, where it beats product quantization on recall with near-zero indexing time. Compressing the stored encoder embeddings with it is a natural v3 experiment. It is **not** a weight-quantization method (see Section 10).

Memory is not required for the first standalone translation baseline, but the model should expose encoder embeddings so later systems can use the same model for both semantic search and generation.

## 9. Inference

Inference flow:

1. normalize input;
2. prepend source and target language tags;
3. tokenize;
4. encode up to 2,048 source/input tokens;
5. decode target tokens autoregressively under a separate max target length;
6. detokenize;
7. optionally validate output with domain-specific checks.

Decoding defaults:

- beam search width 4 for deterministic translation (drop to beam 1-2 for the latency-sensitive path — CTranslate2's production default is 2);
- temperature 0.2-0.7 for paraphrase or creative style transfer;
- repetition penalty for long documents;
- hard stop on EOS or max target length;
- exactness-preserving mode for code, shell, and Lean;
- optional quality-aware reranking: sample N candidates, rerank with a QE metric (the Tower-v2 trick) when quality matters more than latency.

Speculative decoding is not worth implementing at this scale — a 4-layer decoder is already the fast path.

Long documents should be chunked by semantic boundaries, translated per segment, and reconciled with a second pass. A future memory-enabled version can retrieve previous chunks for terminology and style consistency.

### 9.1 Target Runtimes

| Surface | Runtime | Notes |
| --- | --- | --- |
| Desktop CPU/GPU | **CTranslate2** | the standard production runtime for encoder-decoder MT (OPUS-MT, NLLB, MADLAD, faster-whisper, Argos/LibreTranslate); int8 ~3.5x faster than fp32 on CPU; actively maintained again (v4.8.0, June 2026, with T5Gemma/T5Gemma2 support and ROCm added). ONNX Runtime (`ORTModelForSeq2SeqLM`) is the fallback |
| Mobile | **ExecuTorch** (XNNPACK / KleidiAI) or ONNX Runtime Mobile | the Llama 3.2 1B/3B edge path; T5-class export via optimum-executorch (young); supports shared/tied embedding storage and 8-bit embedding + 4-bit linear export |
| Browser | Bergamot/Marian WASM precedent; transformers.js (WASM q8; WebGPU still rough for seq2seq) | Firefox ships ~17MB int8 encoder-decoder translators in WASM to hundreds of millions of users — existence proof for the browser target |

llama.cpp/GGUF supports T5-class encoder-decoders in the CLI/library (not in `llama-server`), so a GGUF export is a nice-to-have, not a primary path.

## 10. Quantization

The baseline model is trained in BF16. Quantization is an export/deployment concern, not a reason to weaken the v0 architecture before quality is measured.

Quantization-friendly choices already in the structure:

- RMSNorm + QK-norm (no BatchNorm, bounded attention logits);
- bias-free linear projections;
- SwiGLU feed-forward blocks;
- tied embeddings (quantize the table once, ExecuTorch shared-embedding style);
- fixed `d_model=512`, `head_dim=64` tensor shapes;
- RoPE instead of learned position tables.

Deployment builds:

| Build | Intended use | Quantization |
| --- | --- | --- |
| `thunk-translate-fp16` | reference quality | FP16/BF16 weights |
| `thunk-translate-int8` | **default local/edge runtime** | INT8 per-channel weight-only (CTranslate2/intgemm style) |
| `thunk-translate-int4` | experimental small/fast runtime | INT4 groupwise (g=32) linears **via QAT**, embeddings + output head kept INT8 |

Evidence-based expectations at this scale:

- **INT8 is effectively lossless** for ~100M-class encoder-decoder MT and is the production norm (Firefox Translations ships int8 students; CTranslate2 reports no meaningful accuracy loss; the LLM.int8() activation-outlier problem only emerges around 7B);
- **INT4 needs QAT.** Sub-1B models lose disproportionately more from plain 4-bit PTQ; the working recipe is Gemma-style QAT — ~5,000 fine-tune steps distilling from the unquantized checkpoint's probabilities (Gemma 3 / Gemma 3 270M QAT releases) — or SpinQuant-style rotation PTQ, following Meta's Llama 3.2 scheme (4-bit groupwise linears, 8-bit per-channel embeddings and output);
- **do the budget math first:** with the tied 25M-param embedding table held at INT8, INT4-izing the body shrinks the model only ~25-35% beyond all-INT8. For a ~110MB int8 model that saving may not justify the QAT effort — measure before building;
- mixed precision per component is the documented NMT pattern: embeddings and decoder tolerate fewer bits worse than the encoder (arXiv:2009.07453);
- **skip entirely:** FP4 microscaling formats (MXFP4/NVFP4 — Blackwell-class datacenter hardware, no edge path) and 2-bit codebook/trellis methods (QuIP#/AQLM/QTIP — GPU-kernel-bound, demonstrated only at 7B+).

**TurboQuant correction.** The previous revision filed "Google's TurboQuant" as a candidate *weight*-quantization technique. That was a mischaracterization: TurboQuant (arXiv:2504.19874; Zandieh, Daliri, Hadian, Mirrokni; ICLR 2026) is an online vector quantizer for activation-like vectors — no calibration, no codebooks — with two demonstrated applications: (a) **KV-cache quantization** (quality-neutral at ~3.5 bits/channel) and (b) **embedding compression for nearest-neighbor search**. Neither shrinks model weights. For Thunk it is relevant to the Section 8 embedding store (strong fit) and, marginally, to decoder KV-cache compression (weak fit: in the reference config the decoder self-attention cache is ~2 KB per generated token plus a fixed ~8 KB per source token for cross-attention — single-digit MB per segment against ~110MB of int8 weights, so weights, not cache, are the memory lever at this scale). The `thunk-translate-turboquant` build is dropped accordingly.

## 11. Evaluation

Evaluation must measure whether meaning survives translation.

Natural language:

- **chrF++ via sacreBLEU** (with reported signatures) as the cheap string metric — BLEU is retained only for historical comparison; the field's position since "To Ship or Not to Ship" (arXiv:2107.10821) is that lexical metrics alone are insufficient evidence;
- **COMET-22** (`Unbabel/wmt22-comet-da`) as the default neural metric — Apache-2.0, ungated, still the most-reported reference metric. Higher-fidelity options: **xCOMET-lite** (278M distilled, runs locally) or **MetricX-24**; note xCOMET/CometKiwi weights are CC-BY-NC — fine for internal eval, check before anything commercial. MetricX-25 exists on paper but has no released weights as of mid-2026;
- **CometKiwi** for reference-free QE — doubles as the Section 6.3 training-data filter;
- **coverage caveat:** learned metrics are trained on modern-language judgments. For Hebrew/Greek/Latin/Aramaic and classic-text style transfer, fall back to chrF++ + round-trip tests + human review (WMT25 itself fell back to chrF++ where neural metrics lack coverage);
- human review for classic texts and theological/philosophical material;
- round-trip preservation tests;
- paired bootstrap significance testing (built into sacreBLEU) on every reported comparison.

Code and formal languages — **execution-based evaluation is primary** (the field's settled position; match-based metrics correlate poorly with functional correctness):

- unit tests for translated scripts/programs (pass@k / TransCoder-style computational accuracy);
- syntax checks and formatters;
- ShellCheck for shell where applicable;
- TypeScript compiler checks;
- Python execution/tests;
- Lean compilation/proof checking;
- CodeBLEU or AST similarity demoted to cheap supplementary signals only.

Operational targets:

- pass rate on curated task suites;
- exact preservation of identifiers unless instructed otherwise;
- no invented citations, imports, flags, or APIs in exactness-preserving mode;
- latency and memory measured separately for FP16, INT8, and INT4 exports, on the actual target runtimes (Section 9.1), not just in PyTorch.

Baselines to beat or match per direction: Firefox Translations students (17-40M, int8 — within ~1-5 COMET of Google Translate on covered pairs), OPUS-MT/HPLT v2 pairs, quickmt (185M, Apache-2.0), IndicTrans2-dist-200M (the best permissive ~200M distilled exemplar), and NLLB-200-distilled-600M (CC-BY-NC — eval only); T5Gemma 2 270M-270M fine-tuned on the same data as the strongest same-size comparison. Positioning note: as of mid-2026 nothing public, permissive, and multi-domain exists between the ~40M per-direction students and TranslateGemma-4B — Thunk's ~110M multi-domain translator sits squarely in that gap.

## 12. Milestones

### v0: Baseline Translator

- run the build-vs-warm-start comparison (Section 2.1) at small scale before committing;
- implement the ~110M deep-encoder/shallow-decoder Transformer (Section 2.2);
- train the 48k tokenizer; measure per-language fertility;
- denoising pretrain + train on a small curated translation/paraphrase/code dataset;
- support required source/target language tags;
- evaluate English <-> code/script and English -> English paraphrase against the Section 11 baselines.

### v1: Curated Multi-Domain Translator

- expand curated corpus; stand up the Section 6.3 QE/execution data gate;
- run Phase-3 teacher distillation at scale;
- add code execution and Lean validation loops;
- add round-trip consistency training;
- one CPO/DQO preference-optimization experiment (Section 6.1 Phase 4);
- expose encoder embedding API (with MRL truncation) for semantic search.

### v2: Quantized Local Runtime

- export FP16 reference model;
- export INT8 model to CTranslate2 (or ONNX Runtime if CTranslate2 health fails the check) and measure on-target latency/memory;
- run the INT4 budget math; if justified, QAT INT4 export via the Gemma-recipe (embeddings stay INT8);
- mobile (ExecuTorch) and browser (WASM/WebGPU) feasibility passes;
- document quality/latency/size tradeoffs per build and runtime.

### v3: Memory-Enabled Translator

- connect encoder embeddings to a vector store;
- evaluate TurboQuant-style compression of stored embeddings (its actual demonstrated application);
- use retrieved examples for document/style consistency;
- add write policy and provenance metadata for stored translations.

## 13. Resolved Clarifications and Open Questions

Resolved clarifications:

- The ~100M parameter target is a rough scale target, not a hard ceiling; the reference config is ~112M. **Only the edge/local deployment target is fixed.**
- The 2K context window means 2,048 input/source tokens; generated target length is controlled separately.
- **TurboQuant identified and re-scoped:** arXiv:2504.19874 (Google Research, ICLR 2026) is online *vector* quantization for KV caches and nearest-neighbor search — not weight quantization. It moves from the export table to the v3 embedding-store experiments (Sections 8, 10).
- Encoder-decoder with a deep encoder and shallow decoder is the evidence-backed shape for this budget; the earlier 10/8 balanced split is superseded.
- GQA + QK-norm are v0 baseline features, not deferred optimizations.
- Use 512-dimensional encoder embeddings by default; Matryoshka training provides 256/128/64-dim truncations, covering the old 128-dim Thunk compatibility without a separate model.

Open questions:

1. Should v0 prioritize English/code translation over foreign-language translation?
2. Which exact non-English languages are first-class targets beyond the generic `<src:foreign>` / `<tgt:foreign>` tags? (The tokenizer vocabulary split depends on this answer.)
3. Scratch vs T5Gemma-2-270M warm start — to be settled empirically in v0 (Section 2.1).
4. Which teacher(s) for Phase-3 distillation: per-domain specialists (NLLB/MADLAD for language, frontier LLM for code) or one frontier LLM for everything?

## 14. Key References

| Topic | Reference |
| --- | --- |
| Encoder-decoder at small scale | T5Gemma, arXiv:2504.06225; T5Gemma 2, arXiv:2512.14856; Return of the Encoder, arXiv:2501.16273; RedLLM, arXiv:2510.26622 |
| Deep encoder / shallow decoder | Kasai et al., ICLR 2021, arXiv:2006.10369; Mozilla Firefox Translations (mozilla/translations) |
| Compact MT baselines / teachers | MADLAD-400, arXiv:2309.04662; TranslateGemma, arXiv:2601.09012; IndicTrans2, arXiv:2305.16307; NLLB-200, arXiv:2207.04672 |
| Architecture components | Gemma 3, arXiv:2503.19786 (QK-norm); GQA, arXiv:2305.13245; MobileLLM, arXiv:2402.14905 |
| Vocabulary sizing | Tao et al., arXiv:2407.13623 |
| Optimizer / schedule | Muon scaling, arXiv:2502.16982; optimizer benchmark, arXiv:2509.02046; WSD analysis, arXiv:2405.18392 |
| Translation training recipe | ALMA, arXiv:2309.11674; CPO/ALMA-R, arXiv:2401.08417; DQO (~500M enc-dec), arXiv:2409.17673; Tower data filtering, arXiv:2402.17733; QE data filtering, arXiv:2311.05350; NewsPaLM, arXiv:2408.06537; MBR-n distillation, arXiv:2407.10456; KD4MT survey, arXiv:2602.15845 |
| Embeddings | MRL, arXiv:2205.13147; EmbeddingGemma, arXiv:2509.20354; E5, arXiv:2212.03533 |
| Quantization | Gemma 3 QAT (Google, Apr 2025); SpinQuant, arXiv:2405.16406; Llama 3.2 quantized (Meta, Oct 2024); on-device NMT quantization, arXiv:2009.07453 |
| TurboQuant (vector/KV-cache quantization) | arXiv:2504.19874, ICLR 2026 |
| Evaluation | COMET-22 (2022.wmt-1.52); xCOMET-lite, arXiv:2406.14553; MetricX-24, arXiv:2410.03983; CometKiwi, arXiv:2209.06243; metric guidance, arXiv:2401.06760; code-translation eval, arXiv:2308.03109 |
