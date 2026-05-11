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

The model is intentionally a **translator**, not a general chat agent. Its primary job is to preserve meaning while changing surface form, language, programming language, style, or level of formality.

## 2. Target Model

| Item | Target |
| --- | --- |
| Architecture | encoder-decoder Transformer |
| Parameter budget | ~100M trainable parameters |
| Context window | 2,048 source tokens per segment |
| Output | autoregressive target tokens |
| Primary tokenizer | shared SentencePiece/BPE vocabulary |
| Primary training mode | supervised seq2seq translation + distillation |
| Deployment target | local workstation first, edge/mobile after quantization |

### 2.1 Reference Configuration

This is the concrete v0 architecture unless experiments prove a different shape is better.

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
[source tokens + control tags]
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

For the 100M translation model:

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

All tasks use explicit control tags so one model can learn multiple translation directions.

Example tag pattern:

```text
<src:english> <tgt:python> <task:generate_code>
Write a CLI that renames all .jpeg files in the current directory to .jpg.
```

```text
<src:bash> <tgt:powershell> <task:translate_code>
for f in *.jpeg; do mv "$f" "${f%.jpeg}.jpg"; done
```

Recommended reserved tags:

- source tags: `<src:english>`, `<src:python>`, `<src:typescript>`, `<src:bash>`, `<src:cmd>`, `<src:powershell>`, `<src:lean>`, `<src:foreign>`
- target tags: `<tgt:english>`, `<tgt:python>`, `<tgt:typescript>`, `<tgt:bash>`, `<tgt:cmd>`, `<tgt:powershell>`, `<tgt:lean>`, `<tgt:foreign>`
- task tags: `<task:translate>`, `<task:paraphrase>`, `<task:explain>`, `<task:generate_code>`, `<task:port_code>`, `<task:formalize>`, `<task:summarize_equivalent>`
- quality tags: `<literal>`, `<semantic>`, `<concise>`, `<verbose>`, `<preserve_format>`

The target language tag is mandatory. Source tags should be present whenever known.

## 5. Tokenization

Use one shared tokenizer for source and target text so the model can copy names, identifiers, citations, and mixed-language fragments.

Default tokenizer requirements:

- SentencePiece or BPE
- 32k vocabulary for v0
- byte fallback or equivalent unknown-character strategy
- preserve whitespace-sensitive code structure
- reserve explicit language/task/control tags
- train on the same curated mixture used for model pretraining/fine-tuning

Code tokenization must preserve exact spelling and indentation well enough for generated code to be executable after detokenization.

## 6. Training Objectives

### 6.1 Primary Objective: Seq2Seq Cross Entropy

Train on parallel pairs:

```text
(source tokens + control tags) -> target tokens
```

The decoder is trained with teacher forcing and standard next-token cross entropy.

### 6.2 Auxiliary Objectives

Use auxiliary tasks only when they improve translation quality:

- denoising autoencoding: corrupted input -> original input
- semantic paraphrase: document -> equivalent document
- round-trip consistency: A -> B -> A should preserve meaning
- code execution consistency: translated code should pass equivalent tests
- contrastive encoder pooling: equivalent source/target pairs should have nearby embeddings

### 6.3 Distillation

Distill common high-value tasks from stronger teacher systems:

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

- English prose, especially durable/high-value works
- multiple Bible translations where licensing permits
- foundational literature and masterworks
- high-value scientific papers, especially with trusted translations
- Lean proofs and formal statements
- TypeScript
- Python
- bash, cmd, and PowerShell
- ubiquitous open-source codebases with compatible licenses
- task-oriented script pairs and explanations

Initial restrictions:

- English is the primary natural language hub.
- Non-English support is translation to/from English first, not full many-to-many translation.
- Prefer high-quality aligned pairs over noisy scraped text.
- Do not train on code or prose whose license forbids model training or redistribution of derived artifacts.

## 8. Memory and Semantic Search Hook

Because the architecture is encoder-decoder, the encoder output can serve two jobs:

1. condition the decoder for translation;
2. provide embeddings for semantic search and persistent memory.

Default embedding strategy:

- mean-pool or attention-pool final encoder states;
- optionally project from 512 dimensions to a smaller store dimension such as 128 or 256;
- store embeddings with source text, target text, tags, provenance, and quality metadata.

Memory is not required for the first standalone translation baseline, but the model should expose encoder embeddings so later systems can use the same model for both semantic search and generation.

## 9. Inference

Inference flow:

1. normalize input;
2. prepend source, target, task, and quality tags;
3. tokenize;
4. encode up to 2,048 source tokens;
5. decode target tokens autoregressively;
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

Open clarification: if "turbo-quant" refers to a specific algorithm, runtime, or package, define it before implementation. Until then, treat it as the optimized quantized export path, with INT8 as the stable target and INT4 as experimental.

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
- latency and memory measured separately for FP16, INT8, and INT4 exports.

## 12. Milestones

### v0: Baseline Translator

- implement the 100M encoder-decoder Transformer;
- train tokenizer;
- train on a small curated translation/paraphrase/code dataset;
- support required control tags;
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
- document quality/latency/size tradeoffs.

### v3: Memory-Enabled Translator

- connect encoder embeddings to a vector store;
- use retrieved examples for document/style consistency;
- add write policy and provenance metadata for stored translations.

## 13. Open Questions

1. Is the 100M parameter target a hard ceiling, or is +/-5% acceptable?
2. Should the context limit mean 2,048 source tokens only, or 2,048 total source+target tokens?
3. Should v0 prioritize English/code translation over foreign-language translation?
4. Which exact non-English languages are first-class targets beyond the generic `<src:foreign>` / `<tgt:foreign>` tags?
5. Does "turbo-quant" mean a specific quantization technique/runtime, or just the general optimized quantized build?
6. Should encoder embeddings be stored at 512 dimensions, or projected down to the older 128-dimensional Thunk space for compatibility?
