"""Tokens-per-word audit of the N-ATLaS tokenizer on FLORES-200.

FLORES is sentence-parallel, so beyond tokens/word we report the *language tax*:
tokens needed to express the same sentences as English. That ratio converts an
engine's tokens/s into comparable words/s and sentences/s per language.

Pure functions compute everything from (texts, tokenizer); `main` does the I/O.

Usage:
    uv run --with tokenizers python -m wazocore.tokens_per_word \
        --flores data/flores200_dataset --split dev \
        --tokenizer data/tokenizers/n-atlas-fp8-mirror/tokenizer.json \
        --out results/tokenizer/n-atlas-flores200-dev.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

LANGS = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}

Encode = Callable[[Sequence[str]], list[int]]  # texts -> token count per text


@dataclass(frozen=True)
class LangStats:
    lang: str
    sentences: int
    words: int
    chars: int
    tokens: int
    tokens_nfd: int
    tokens_per_word: float
    tokens_per_char: float
    tax_vs_en: float | None
    nfd_inflation: float


def normalize(texts: Sequence[str], form: str) -> list[str]:
    return [unicodedata.normalize(form, t.strip()) for t in texts]


def count_words(text: str) -> int:
    return len(text.split())


def count_chars(text: str) -> int:
    """Grapheme-ish count: NFC code points, excluding whitespace."""
    return sum(1 for c in unicodedata.normalize("NFC", text) if not c.isspace())


def lang_stats(lang: str, texts: Sequence[str], encode: Encode, en_tokens: int | None) -> LangStats:
    nfc = normalize(texts, "NFC")
    tokens = sum(encode(nfc))
    tokens_nfd = sum(encode(normalize(texts, "NFD")))
    words = sum(count_words(t) for t in nfc)
    chars = sum(count_chars(t) for t in nfc)
    return LangStats(
        lang=lang,
        sentences=len(nfc),
        words=words,
        chars=chars,
        tokens=tokens,
        tokens_nfd=tokens_nfd,
        tokens_per_word=tokens / words,
        tokens_per_char=tokens / chars,
        tax_vs_en=tokens / en_tokens if en_tokens else None,
        nfd_inflation=tokens_nfd / tokens,
    )


def audit(corpus: dict[str, Sequence[str]], encode: Encode) -> list[LangStats]:
    """English first: its token total is the reference for the tax."""
    lengths = {len(v) for v in corpus.values()}
    if len(lengths) != 1:
        raise ValueError(f"corpus is not parallel: sentence counts {lengths}")
    en = lang_stats("en", corpus["en"], encode, None)
    others = [lang_stats(l, corpus[l], encode, en.tokens) for l in corpus if l != "en"]
    return [en, *others]


def render(rows: Sequence[LangStats]) -> str:
    head = f"{'lang':<5}{'tok/word':>9}{'tok/char':>9}{'tax vs en':>10}{'NFD x':>7}{'tokens':>9}{'words':>8}"
    lines = [
        f"{r.lang:<5}{r.tokens_per_word:>9.3f}{r.tokens_per_char:>9.3f}"
        f"{(r.tax_vs_en or 1.0):>10.3f}{r.nfd_inflation:>7.3f}{r.tokens:>9}{r.words:>8}"
        for r in rows
    ]
    return "\n".join([head, *lines])


# ---- imperative shell -------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_flores(root: Path, split: str) -> dict[str, list[str]]:
    return {
        lang: (root / split / f"{code}.{split}").read_text(encoding="utf-8").splitlines()
        for lang, code in LANGS.items()
    }


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--split", default="dev", choices=["dev", "devtest"])
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--tokenizer-source", default="", help="provenance note, e.g. repo id")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from tokenizers import Tokenizer  # heavy import kept at the edge

    tok = Tokenizer.from_file(str(args.tokenizer))

    def encode(texts: Sequence[str]) -> list[int]:
        return [len(e.ids) for e in tok.encode_batch(list(texts), add_special_tokens=False)]

    rows = audit(read_flores(args.flores, args.split), encode)
    print(render(rows), file=sys.stderr)
    report = {
        "dataset": f"FLORES-200 {args.split}",
        "normalization": "NFC (tokens_nfd for comparison)",
        "tokenizer": {
            "path": str(args.tokenizer),
            "source": args.tokenizer_source,
            "sha256": sha256(args.tokenizer),
            "vocab_size": tok.get_vocab_size(with_added_tokens=True),
        },
        "languages": [asdict(r) for r in rows],
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
