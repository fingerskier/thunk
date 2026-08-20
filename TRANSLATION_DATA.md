# Translation Data — Curated Language Set

Curated list of languages and paired datasets for translation-style training
(model #2 and later). Every pair is a *known* translation: aligned source and
target written by humans (docstring ↔ function, natural-language statement ↔
formal LEAN statement, same program in two languages).

## Languages

| Tag | Language |
| --- | --- |
| `english` | natural-language English |
| `lean4` | LEAN 4 (theorem statements / proofs) |
| `python` | Python |
| `javascript` | JavaScript |
| `java` | Java |
| `go` | Go |
| `php` | PHP |
| `ruby` | Ruby |
| `c` | C |
| `cpp` | C++ |
| `rust` | Rust |
| `csharp` | C# |

## Pairs and sources

| Pair | Dataset | HF repo / API | Alignment | License |
| --- | --- | --- | --- | --- |
| python/javascript/java/go/php/ruby ↔ english | CodeSearchNet (docstring ↔ function) | `code_search_net` per-language configs | strong | MIT-ish, per-repo — see card |
| lean4 ↔ english | Lean Workbook (NL problem ↔ formal statement) | `internlm/Lean-Workbook` | strong | Apache 2.0 |
| lean4 ↔ english | ProofNet (NL math ↔ formal statement) | `hoskinson-center/proofnet` via datasets-server rows API (script loader is dead in datasets ≥3) | strong, small | MIT |
| c / cpp / rust ↔ english | The Stack smol, leading doc-comment ↔ following code | `bigcode/the-stack-smol` (`data/c`, `data/c++`, `data/rust`) — **gated**: needs `HF_TOKEN` (or `huggingface-cli login`) + accepting the dataset license; falls back to synthetic otherwise | **weak** — heuristic comment pairing | permissive-licensed code only, see card |
| java ↔ csharp | CodeXGLUE CodeTrans | `google/code_x_glue_cc_code_to_code_trans` | strong | C# corpus from paired OSS projects |
| python ↔ cpp ↔ java | TransCoder GFG evaluation set (same program, three languages) | GitHub API: `facebookresearch/TransCoder` `data/transcoder_evaluation_gfg` | exact, small | CC BY-NC (eval only) |

## Rules

- **Bidirectional**: every aligned pair is emitted in both directions
  (`<src:python> <tgt:english>` and `<src:english> <tgt:python>`).
- **Tag format**: `tagged()` in `script/glean_datasets.py` —
  `<src:X> <tgt:Y> {source text}`, target is the bare translation.
- **Model 2 text format**: one example per line in
  `data/model/2/train_text.txt`: `{tagged source} <sep> {target}`.
- **Weak-alignment sources** (The Stack comment pairing) are kept in separate
  per-source `.jsonl` files so they can be excluded via `--sources`.
- **Tokenizer**: every tag above (`<src:X>`, `<tgt:X>`, `<sep>`) is a reserved
  single symbol in the pinned shared tokenizer `tokenizer/v1/tokenizer.model`
  (built by `script/train_tokenizer.py`; see `tokenizer/v1/MANIFEST.json`).
  Adding a language here means adding its tag to `MODEL2_LANGS` in that
  script and publishing a new tokenizer version.
- Regenerate with:

```bash
python script/glean_datasets.py --model 2 --limit 250
# offline smoke:
python script/glean_datasets.py --model 2 --offline --limit 5
# pinned tokenizer (once per version; never overwrite in place):
python script/train_tokenizer.py --out tokenizer/v2/tokenizer.model
python script/test_train_tokenizer.py
```

## Candidates not yet wired

- XLCoST (7-language parallel snippets, program- and snippet-level)
- MiniF2F (competition math, formal + informal, LEAN 4 port)
- Herald (large NL ↔ LEAN 4 statement corpus)
- mathlib4 docstrings (decl ↔ doc, extractable with LeanDojo)
