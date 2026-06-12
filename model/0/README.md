# Thunk v0 — Baseline Translator

A runnable implementation of the v0 baseline translator from
[`SPEC.md`](../../SPEC.md): a deep-encoder / shallow-decoder encoder-decoder
Transformer that preserves meaning while changing surface form, language, or
formalism.

This replaces the earlier decoder-only recursive prototype. The architecture now
follows SPEC.md Sections 2.2 and 3 exactly.

## Architecture (`model.py`)

| Feature | Implementation |
| --- | --- |
| Shape | deep encoder / shallow decoder (reference: 16 enc / 4 dec) |
| Normalization | pre-norm RMSNorm + QK-norm (RMSNorm on per-head Q/K) |
| Position encoding | RoPE on self-attention only (base 100k); none on cross-attention |
| Encoder self-attention | full MHA, bidirectional |
| Decoder self-attention | causal grouped-query attention (8Q / 2KV) |
| Cross-attention | full MHA over encoder memory |
| Feed-forward | SwiGLU, no biases on any linear |
| Embeddings | one shared table tied to the LM head |
| Memory hook | `model.embed()` returns mean-pooled encoder states (SPEC §8) |

Two presets in `config.py`:

- `reference_config()` — the SPEC.md ~112M-parameter config (512-dim, 16/4
  layers, 49,152 vocab). `python model.py` prints **111,699,968** params,
  matching the spec's component budget (25.2M embedding + 67.1M encoder +
  19.4M decoder).
- `small_config()` — a ~9–11M-parameter config (256-dim, 6/2 layers) used as the
  default for base training, so it actually trains and learns on a CPU.

## Control tags (`config.py`, SPEC §4)

Each `<src:...>` / `<tgt:...>` tag is a single reserved tokenizer symbol, never
split into subwords. The target tag is mandatory. Inputs look like:

```
<src:digits> <tgt:english> 3247
<src:bash> <tgt:powershell> rm report.txt
```

## Tokenizer (`tokenizer.py`, SPEC §5)

Shared SentencePiece BPE with byte fallback, digit splitting, identity
normalization (to preserve code whitespace), and the control tags reserved as
user-defined symbols. Special ids are fixed: `pad=0, unk=1, bos=2, eos=3`.

## Base-training corpus (`data.py`)

Network access is restricted in the build environment, so the base-training
corpus is generated deterministically instead of downloaded. It is still a real,
meaning-preserving, exact-match-verifiable translation task across four SPEC §4
directions, both ways:

- `<src:digits>` ↔ `<tgt:english>` — number spelling (compositional, the bulk)
- `<src:english>` ↔ `<tgt:english>` — comparison paraphrase (semantic rewrite)
- `<src:python>` ↔ `<tgt:english>` — tiny code ↔ description
- `<src:bash>` ↔ `<tgt:powershell>` — shell command porting

Swap in real parallel data (SPEC §7) by replacing `build_corpus()` — the rest of
the pipeline is agnostic to where pairs come from.

## Training (`train.py`, `optim.py`, SPEC §6)

Two phases (SPEC §6.1):

1. **Denoising pretrain** — UL2-style span corruption (R-denoiser) mixed with
   prefix-LM (S-denoiser) on the monolingual side of the corpus (control tags
   stripped, texts packed into fixed-length blocks). Corrupted spans are
   replaced by reserved `<extra_id_i>` sentinels and reconstructed by the
   decoder. The encoder learns representations before it sees any translation
   tag.
2. **Supervised seq2seq** — teacher forcing with label smoothing (0.1) on the
   parallel pairs, continuing from the pretrained weights.

Optimizer is **Muon on the hidden 2D weight matrices + AdamW on embeddings,
norms, and the tied head** (SPEC §6.4), using `torch.optim.Muon` (PyTorch
2.12); `--optimizer adamw` selects the documented single-AdamW fallback. Both
phases use gradient clipping and a warmup-stable-decay LR schedule. Translation
evaluation reports val loss **and exact-match decoding accuracy**, since every
pair is verifiable.

```bash
pip install -r requirements.txt
python train.py                  # full run: denoising pretrain + supervised
python train.py --no-pretrain    # skip Phase 1
python train.py --optimizer adamw
```

Checkpoints are written to `checkpoints/` (git-ignored).

## Inference (`translate.py`, SPEC §9)

```bash
python translate.py --src digits --tgt english "3247"
# -> three thousand two hundred forty-seven

python translate.py --src bash --tgt powershell "rm report.txt"
# -> Remove-Item report.txt

python translate.py --beam 4 --src python --tgt english "x = 3 + 4"
# -> assign the sum of 3 and 4 to x
```

Greedy by default; `--beam N` enables beam search (SPEC §9 decoding defaults).

## Status vs. SPEC milestones

This is the **v0 baseline** (SPEC §12): the architecture, control-tag interface,
shared tokenizer, denoising pretrain (Phase 1) + supervised seq2seq (Phase 2),
the Muon+AdamW optimizer split (§6.4), the encoder-embedding hook, and
greedy/beam decoding. Left for later milestones: teacher distillation (Phase 3),
preference optimization (Phase 4), the QE/execution data gate, Matryoshka
contrastive training, and quantized exports.
