#!/usr/bin/env python3
"""Glean small, model-shaped datasets from public repositories and APIs.

Downloaded/generated artifacts are written under ./data/ (gitignored). The script is
model-aware: pass --model 0 for seq2seq translation pairs or --model 1 for
question/answer diffusion records. By default it samples lightweight slices from
well-known Hugging Face dataset repositories, direct public files, and no-key
public APIs, then falls back to deterministic synthetic examples when a source
cannot be fetched.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import random
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

Json = dict[str, Any]
ShapeFn = Callable[[Json], Json | None]
ApiFetchFn = Callable[[int, int], Iterable[Json]]


@dataclass(frozen=True)
class Source:
    """A public training-data source and the adapters needed to shape it."""

    name: str
    hf_repo: str | None
    hf_config: str | None
    hf_split: str
    url: str | None
    api_fetcher: ApiFetchFn | None
    model_shapes: dict[str, ShapeFn]


def clean_text(value: Any, limit: int = 512) -> str:
    """Normalize text fields and keep examples small enough for tiny models."""

    text = "" if value is None else str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def tagged(src: str, tgt: str, text: str) -> str:
    return f"<src:{src}> <tgt:{tgt}> {clean_text(text)}"


def shape_translation_pair(row: Json) -> Json | None:
    src = clean_text(row.get("translation", {}).get("en") or row.get("en"))
    tgt = clean_text(row.get("translation", {}).get("de") or row.get("de"))
    if not src or not tgt:
        return None
    return {"source": tagged("english", "german", src), "target": tgt, "task": "translation"}


def shape_instruction_pair(row: Json) -> Json | None:
    instruction = clean_text(row.get("instruction") or row.get("prompt"))
    response = clean_text(row.get("output") or row.get("response") or row.get("completion"))
    if not instruction or not response:
        return None
    return {"source": tagged("instruction", "english", instruction), "target": response, "task": "instruction"}


def shape_code_pair(row: Json) -> Json | None:
    code = clean_text(row.get("code") or row.get("content"), 384)
    doc = clean_text(row.get("docstring") or row.get("func_documentation_string") or row.get("description"))
    if not code or not doc:
        return None
    return {"source": tagged("python", "english", code), "target": doc, "task": "code_summary"}


def shape_qa(row: Json) -> Json | None:
    question = clean_text(row.get("question") or row.get("instruction") or row.get("prompt"), 256)
    answer = clean_text(row.get("answer") or row.get("output") or row.get("response") or row.get("completion"), 128)
    if not question or not answer:
        return None
    return {"question": question, "answer": answer, "task": "qa"}


def shape_text_completion(row: Json) -> Json | None:
    text = clean_text(row.get("text") or row.get("content"), 640)
    if len(text) < 80:
        return None
    question, answer = text[:256].strip(), text[256:384].strip()
    if not answer:
        return None
    return {"question": f"continue this text: {question}", "answer": answer, "task": "completion"}


def shape_api_qa_pair(row: Json) -> Json | None:
    question = clean_text(row.get("question") or row.get("prompt"), 256)
    answer = clean_text(row.get("answer") or row.get("response"), 128)
    if not question or not answer:
        return None
    return {"source": tagged("question", "answer", question), "target": answer, "task": row.get("task", "api_qa")}


def synthetic_rows(name: str) -> Iterator[Json]:
    """Deterministic fallback rows for offline smoke runs."""

    for idx in range(100):
        yield {
            "instruction": f"Summarize dataset source {name} example {idx} in one sentence.",
            "output": f"Example {idx} from {name} is a compact supervised training record.",
            "question": f"What is example {idx} from {name}?",
            "answer": f"A small fallback record for {name}.",
            "translation": {"en": f"hello world {idx}", "de": f"hallo welt {idx}"},
            "code": f"def add_{idx}(x): return x + {idx}",
            "docstring": f"Return x plus {idx}.",
            "text": (f"This is fallback text example {idx} from {name}. " * 16),
        }


def fetch_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "thunk-dataset-gleaner/1.0"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - fixed public sources below
        return json.loads(response.read().decode("utf-8", errors="replace"))


def iter_open_trivia_rows(limit: int, seed: int) -> Iterator[Json]:
    """Fetch trivia question/answer rows from the Open Trivia DB API."""

    remaining = max(0, limit)
    while remaining > 0:
        amount = min(remaining, 50)
        payload = fetch_json(f"https://opentdb.com/api.php?{urlencode({'amount': amount, 'type': 'multiple'})}")
        if payload.get("response_code") != 0:
            return
        for item in payload.get("results", []):
            question = clean_text(html.unescape(item.get("question", "")), 256)
            answer = clean_text(html.unescape(item.get("correct_answer", "")), 128)
            if question and answer:
                yield {
                    "question": question,
                    "answer": answer,
                    "task": "trivia_qa",
                    "api": "opentdb",
                    "category": clean_text(html.unescape(item.get("category", "")), 128),
                }
                remaining -= 1
                if remaining <= 0:
                    return


def iter_openlibrary_rows(limit: int, seed: int) -> Iterator[Json]:
    """Fetch book metadata rows from the Open Library Search API."""

    topics = ["science fiction", "mathematics", "programming", "history", "philosophy"]
    rng = random.Random(seed)
    rng.shuffle(topics)
    yielded = 0
    per_topic = max(1, min(50, limit))
    for topic in topics:
        params = {
            "q": topic,
            "fields": "title,author_name,first_publish_year",
            "limit": per_topic,
        }
        payload = fetch_json(f"https://openlibrary.org/search.json?{urlencode(params)}")
        for item in payload.get("docs", []):
            title = clean_text(item.get("title"), 160)
            authors = item.get("author_name") or []
            author = clean_text(authors[0] if authors else "", 128)
            if title and author:
                yield {
                    "question": f"Who wrote {title}?",
                    "answer": author,
                    "task": "book_author_qa",
                    "api": "openlibrary",
                    "topic": topic,
                    "first_publish_year": item.get("first_publish_year"),
                }
                yielded += 1
                if yielded >= limit:
                    return


def iter_datamuse_rows(limit: int, seed: int) -> Iterator[Json]:
    """Fetch word-association rows from the Datamuse words API."""

    topics = ["ocean", "music", "machine learning", "reasoning", "language", "number"]
    rng = random.Random(seed)
    rng.shuffle(topics)
    yielded = 0
    per_topic = max(1, min(50, limit))
    for topic in topics:
        params = {"ml": topic, "max": per_topic}
        payload = fetch_json(f"https://api.datamuse.com/words?{urlencode(params)}")
        for rank, item in enumerate(payload, start=1):
            word = clean_text(item.get("word"), 80)
            if word:
                yield {
                    "question": f"What is related word #{rank} for {topic}?",
                    "answer": word,
                    "task": "word_association",
                    "api": "datamuse",
                    "topic": topic,
                    "score": item.get("score"),
                }
                yielded += 1
                if yielded >= limit:
                    return


def iter_hf_rows(source: Source, limit: int, seed: int) -> Iterator[Json]:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise RuntimeError("install optional dependency: pip install datasets") from exc

    assert source.hf_repo is not None
    kwargs: dict[str, Any] = {"split": source.hf_split, "streaming": True}
    if source.hf_config:
        dataset = load_dataset(source.hf_repo, source.hf_config, **kwargs)
    else:
        dataset = load_dataset(source.hf_repo, **kwargs)
    if seed:
        dataset = dataset.shuffle(seed=seed, buffer_size=max(1_000, limit * 10))
    for row in dataset.take(limit):
        yield dict(row)


def iter_url_rows(source: Source, limit: int) -> Iterator[Json]:
    if not source.url:
        return
    request = Request(source.url, headers={"User-Agent": "thunk-dataset-gleaner/1.0"})
    with urlopen(request, timeout=30) as response:  # noqa: S310 - caller chooses public URLs
        lines = response.read().decode("utf-8", errors="replace").splitlines()
    if source.url.endswith(".csv"):
        yield from list(csv.DictReader(lines))[:limit]
    else:
        for line in lines[:limit]:
            yield json.loads(line)


SOURCES: tuple[Source, ...] = (
    Source(
        name="wmt14_en_de",
        hf_repo="wmt14",
        hf_config="de-en",
        hf_split="train",
        url=None,
        api_fetcher=None,
        model_shapes={"0": shape_translation_pair, "1": shape_qa},
    ),
    Source(
        name="databricks_dolly_15k",
        hf_repo="databricks/databricks-dolly-15k",
        hf_config=None,
        hf_split="train",
        url="https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl",
        api_fetcher=None,
        model_shapes={"0": shape_instruction_pair, "1": shape_qa},
    ),
    Source(
        name="code_search_net_python",
        hf_repo="code_search_net",
        hf_config="python",
        hf_split="train",
        url=None,
        api_fetcher=None,
        model_shapes={"0": shape_code_pair, "1": shape_qa},
    ),
    Source(
        name="openwebtext",
        hf_repo="Skylion007/openwebtext",
        hf_config=None,
        hf_split="train",
        url=None,
        api_fetcher=None,
        model_shapes={"0": shape_instruction_pair, "1": shape_text_completion},
    ),
    Source(
        name="opentdb_trivia",
        hf_repo=None,
        hf_config=None,
        hf_split="",
        url=None,
        api_fetcher=iter_open_trivia_rows,
        model_shapes={"0": shape_api_qa_pair, "1": shape_qa},
    ),
    Source(
        name="openlibrary_books",
        hf_repo=None,
        hf_config=None,
        hf_split="",
        url=None,
        api_fetcher=iter_openlibrary_rows,
        model_shapes={"0": shape_api_qa_pair, "1": shape_qa},
    ),
    Source(
        name="datamuse_words",
        hf_repo=None,
        hf_config=None,
        hf_split="",
        url=None,
        api_fetcher=iter_datamuse_rows,
        model_shapes={"0": shape_api_qa_pair, "1": shape_qa},
    ),
)


def collect_source(source: Source, model: str, limit: int, seed: int, offline: bool) -> list[Json]:
    shaper = source.model_shapes[model]
    errors: list[str] = []

    def providers() -> Iterator[Iterable[Json]]:
        if offline:
            yield synthetic_rows(source.name)
            return
        if source.hf_repo:
            yield iter_hf_rows(source, limit * 3, seed)
        if source.url:
            yield iter_url_rows(source, limit * 3)
        if source.api_fetcher:
            yield source.api_fetcher(limit * 3, seed)
        yield synthetic_rows(source.name)

    for rows in providers():
        shaped: list[Json] = []
        seen: set[str] = set()
        try:
            for row in rows:
                item = shaper(row)
                if item is None:
                    continue
                item["source_repo"] = source.hf_repo or source.url or source.name
                item["source_name"] = source.name
                digest = hashlib.sha256(json.dumps(item, sort_keys=True).encode()).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                shaped.append(item)
                if len(shaped) >= limit:
                    break
        except Exception as exc:  # keep the script useful in fresh/offline environments
            errors.append(str(exc))
            continue
        if shaped:
            if errors:
                print(f"warning: {source.name}: used fallback after: {'; '.join(errors)}", file=sys.stderr)
            return shaped
    return []


def write_jsonl(path: Path, rows: Sequence[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_model0_text(path: Path, rows: Sequence[Json]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row["source"] + "\n")
            handle.write(row["target"] + "\n")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download/sample public datasets into ./data in a model-aware shape.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              python script/glean_datasets.py --model 0 --limit 200
              python script/glean_datasets.py --model 1 --sources databricks_dolly_15k openwebtext
              python script/glean_datasets.py --model 1 --sources opentdb_trivia openlibrary_books datamuse_words --limit 20
              python script/glean_datasets.py --model all --offline --limit 20
            """
        ),
    )
    parser.add_argument("--model", choices=("0", "1", "all"), default="all", help="model shape to emit")
    parser.add_argument("--sources", nargs="*", choices=[s.name for s in SOURCES], default=[s.name for s in SOURCES])
    parser.add_argument("--limit", type=int, default=250, help="max shaped examples per source and model")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--offline", action="store_true", help="skip network/datasets and emit deterministic synthetic smoke data")
    parser.add_argument("--out-dir", type=Path, default=DATA_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    rng = random.Random(args.seed)
    selected = [s for s in SOURCES if s.name in set(args.sources)]
    models = ["0", "1"] if args.model == "all" else [args.model]

    manifest: list[Json] = []
    for model in models:
        all_rows: list[Json] = []
        for source in selected:
            rows = collect_source(source, model, args.limit, args.seed, args.offline)
            all_rows.extend(rows)
            write_jsonl(args.out_dir / f"model/{model}/{source.name}.jsonl", rows)
            manifest.append({"model": model, "source": source.name, "rows": len(rows)})
        rng.shuffle(all_rows)
        write_jsonl(args.out_dir / f"model/{model}/combined.jsonl", all_rows)
        if model == "0":
            write_model0_text(args.out_dir / "model/0/train_text.txt", all_rows)

    write_jsonl(args.out_dir / "manifest.jsonl", manifest)
    display_dir = args.out_dir if args.out_dir.is_absolute() else Path.cwd() / args.out_dir
    print(f"wrote {sum(item['rows'] for item in manifest)} rows under {display_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
