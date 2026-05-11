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
| Architecture | encoder-decoder Transformer |
| Parameter budget | roughly 100M trainable parameters; not a hard cap |
| Input context window | 2,048 source/input tokens per segment |
| Output | autoregressive target tokens; target length is limited separately at decode time |
| Primary tokenizer | shared SentencePiece/BPE vocabulary |
| Primary training mode | supervised seq2seq translation + distillation |
| Deployment target | local workstation first, edge/mobile after quantization |

This translation spec supersedes the older 128-dimensional Thunk prototype assumptions. Use 512-dimensional encoder states as the default representation and only revisit smaller dimensions if deployment measurements require it.

### 2.1 Reference Configuration

This is the concrete v0 architecture unless experiments prove a different shape is better. The exact parameter count may move above or below 100M if quality, latency, or implementation simplicity justify it.

| Parameter | Value |
| --- | ---: |
| `vocab_size` | 32,000 shared source/target tokens |
| `d_model` | 512 |
| `encoder_layers` | 10 |
| `decoder_layers` | 8 |
| `attention_heads` | 8 |
| `head_dim` | 64 |
| `ffn_dim` | 2,048 |
| `activation` | SwiGLU |
| `normalization` | pre-norm RMSNorm |
| `position_encoding` | RoPE |
| `dropout` | 0.0-0.1 during training |
| `weight_tying` | shared token embedding + tied decoder output head |

Approximate parameter count:

| Component | Approx params |
| --- | ---: |
| Shared token embedding / tied LM head | 16.4M |
| 10 encoder blocks | 42.0M |
| 8 decoder blocks | 42.0M |
| Final norms and small overhead | <0.1M |
| **Total** | **~100.3M** |

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
| Encoder, 10 layers          |
| - RMSNorm                   |
| - 8-head self-attention     |
| - residual                  |
| - RMSNorm                   |
| - SwiGLU feed-forward       |
| - residual                  |
+-----------------------------+
        |
        v
encoder memory states
        |
        v
+-----------------------------+
| Decoder, 8 layers           |
| - RMSNorm                   |
| - 8-head causal self-attn   |
| - residual                  |
| - RMSNorm                   |
| - 8-head cross-attention    |
| - residual                  |
| - RMSNorm                   |
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
2. multi-head bidirectional self-attention
3. residual connection
4. RMSNorm
5. SwiGLU feed-forward network
6. residual connection

The final encoder states are also the default representation to pool/project for semantic search or memory storage.

### 3.2 Decoder

The decoder generates the target sequence autoregressively while attending to the encoder states.

Each decoder block contains:

1. RMSNorm
2. multi-head causal self-attention over generated target tokens
3. residual connection
4. RMSNorm
5. multi-head cross-attention over encoder memory states
6. residual connection
7. RMSNorm
8. SwiGLU feed-forward network
9. residual connection

### 3.3 Multi-Head Attention Decision

Use MHA in v0.

For the rough 100M-class translation model:

- 8 heads
- 64 dimensions per head
- full MHA for encoder self-attention
- full MHA for decoder causal self-attention
- full MHA for decoder cross-attention

Rationale:
- translation needs multiple simultaneous alignments: syntax, semantics, idioms, identifiers, formatting, and target-language constraints;
- cross-attention is the core mechanism that makes encoder-decoder translation work;
- 8 x 64 is a standard, hardware-friendly split for `d_model=512`;
- MQA/GQA can be evaluated later as an inference optimization, but should not be the first training baseline.


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


## 5. Tokenization

Use one shared tokenizer for source and target text so the model can copy names, identifiers, citations, and mixed-language fragments.

Default tokenizer requirements:

- SentencePiece or BPE
- 32k vocabulary for v0
- byte fallback or equivalent unknown-character strategy
- preserve whitespace-sensitive code structure
- reserve only explicit source/target language tags
- train on the same curated mixture used for model pretraining/fine-tuning

Code tokenization must preserve exact spelling and indentation well enough for generated code to be executable after detokenization.


## 6. Training Objectives

### 6.1 Primary Objective: Seq2Seq Cross Entropy

Train on parallel pairs:

```text
(source tokens + source/target language tags) -> target tokens
```

The decoder is trained with teacher forcing and standard next-token cross entropy.

Training proceeds in three phases:
- Phase 1: self -> self training, where each document is reconstructed from itself.
- Phase 2: self -> equivalent translation of source material, such as English -> code, code -> English, English -> English paraphrase, or English -> foreign language.
- Phase 3: distillation training on verifiable input -> output pairs.

### 6.2 Auxiliary Objectives

Use auxiliary tasks only when they improve translation quality:
- denoising autoencoding: corrupted input -> original input
- semantic paraphrase: document -> equivalent document
- round-trip consistency: A -> B -> A should preserve meaning
- code execution consistency: translated code should pass equivalent tests
- contrastive encoder pooling: equivalent source/target pairs should have nearby embeddings

### 6.3 Distillation

Distill common high-value tasks from stronger teacher systems using verifiable input -> output pairs:
- English -> shell script
- shell script -> English
- bash -> PowerShell
- Python -> TypeScript
- English -> Lean sketch
- Lean proof/code -> English explanation
- classic text -> modern English paraphrase
- Bible translation style transfer where licensing allows

Distillation data must be filtered for correctness, license compatibility, and semantic equivalence.


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


## 8. Memory and Semantic Search Hook

Because the architecture is encoder-decoder, the encoder output can serve two jobs:

1. condition the decoder for translation;
2. provide embeddings for semantic search and persistent memory.

Default embedding strategy:

- mean-pool or attention-pool final encoder states;
- store pooled encoder embeddings at 512 dimensions by default; only add a projection if a later deployment/runtime requires it;
- store embeddings with source text, target text, tags, provenance, and quality metadata.

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

- beam search width 4 for deterministic translation;
- temperature 0.2-0.7 for paraphrase or creative style transfer;
- repetition penalty for long documents;
- hard stop on EOS or max target length;
- exactness-preserving mode for code, shell, and Lean.

Long documents should be chunked by semantic boundaries, translated per segment, and reconciled with a second pass. A future memory-enabled version can retrieve previous chunks for terminology and style consistency.

## 10. Quantization / "Turbo-Quant"

The baseline model is trained in BF16/FP16. Quantization is an export/deployment concern, not a reason to weaken the v0 architecture before quality is measured.

Required quantization-friendly choices already in the structure:

- RMSNorm instead of BatchNorm;
- bias-free large linear projections where practical;
- SwiGLU feed-forward blocks;
- tied embeddings;
- fixed `d_model=512`, `head_dim=64` tensor shapes;
- RoPE instead of learned position tables.

Deployment targets:

| Build | Intended use | Quantization |
| --- | --- | --- |
| `thunk-translate-fp16` | reference quality | FP16/BF16 weights |
| `thunk-translate-int8` | default local/mobile runtime | INT8 weight-only or dynamic INT8 |
| `thunk-translate-int4` | experimental small/fast runtime | INT4 weight-only, AWQ/GPTQ-style |
| `thunk-translate-turboquant` | candidate optimized runtime if Google's TurboQuant technique applies | TBD after paper/runtime validation |

TurboQuant note: "turbo-quant" refers to the recent Google model-performance paper/technique, not a generic name for quantization. Treat it as a candidate optimization in the v2 export/runtime phase, not as a dependency for the baseline architecture. Before implementation, read and cite the exact paper, then verify that the technique applies to this encoder-decoder shape, cross-attention, target hardware, and chosen runtime. If applicable, compare a TurboQuant build against FP16, INT8, and INT4 on quality, latency, memory, and output stability. If not applicable, record the reason and keep the standard quantized exports.

## 11. Evaluation

Evaluation must measure whether meaning survives translation.

Natural language:

- BLEU / chrF for rough regression tracking;
- COMET or embedding-based semantic similarity where available;
- human review for classic texts and theological/philosophical material;
- round-trip preservation tests.

Code and formal languages:

- unit tests for translated scripts/programs;
- syntax checks and formatters;
- ShellCheck for shell where applicable;
- TypeScript compiler checks;
- Python execution/tests;
- Lean compilation/proof checking;
- CodeBLEU or AST similarity as secondary metrics.

Operational targets:

- pass rate on curated task suites;
- exact preservation of identifiers unless instructed otherwise;
- no invented citations, imports, flags, or APIs in exactness-preserving mode;
- latency and memory measured separately for FP16, INT8, INT4, and any TurboQuant export.

## 12. Milestones

### v0: Baseline Translator

- implement the rough 100M-class encoder-decoder Transformer;
- train tokenizer;
- train on a small curated translation/paraphrase/code dataset;
- support required source/target language tags;
- evaluate English <-> code/script and English -> English paraphrase.

### v1: Curated Multi-Domain Translator

- expand curated corpus;
- add code execution and Lean validation loops;
- add round-trip consistency training;
- expose encoder embedding API for semantic search.

### v2: Quantized Local Runtime

- export FP16 reference model;
- export stable INT8 model;
- evaluate experimental INT4 model;
- evaluate Google's TurboQuant technique if it applies to the chosen architecture/runtime;
- document quality/latency/size tradeoffs.

### v3: Memory-Enabled Translator

- connect encoder embeddings to a vector store;
- use retrieved examples for document/style consistency;
- add write policy and provenance metadata for stored translations.

## 13. Resolved Clarifications and Open Questions

Resolved clarifications:

- The ~100M parameter target is a rough scale target, not a hard ceiling.
- The 2K context window means 2,048 input/source tokens; generated target length is controlled separately.
- TurboQuant refers to the recent Google model-performance technique and should be evaluated for applicability.
- Use 512-dimensional encoder embeddings by default; ignore prior 128-dimensional Thunk compatibility unless a future constraint reintroduces it.

Open questions:

1. Should v0 prioritize English/code translation over foreign-language translation?
2. Which exact non-English languages are first-class targets beyond the generic `<src:foreign>` / `<tgt:foreign>` tags?
3. Which exact Google TurboQuant paper/version and implementation should be used for the applicability review?
