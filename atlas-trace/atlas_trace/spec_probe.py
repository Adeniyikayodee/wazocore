"""Experiment: how predictable is N-ATLaS's greedy output per language?

Under greedy decoding, speculative decoding is lossless, so its gain can be
simulated exactly from the model's own greedy outputs: a drafter proposes
up to k tokens, the longest prefix that matches the model's output is
accepted, and verification adds one more token. Tokens per verification
step (tau) is the speed-up ceiling before verification overhead.

Drafters:
- lookup: llama.cpp-style prompt lookup (match the last n tokens in the
  prompt + output so far, propose what followed).
- ngram: a token n-gram table (n <= 4, longest-context backoff) built from
  FLORES-200 *dev* in the target language; outputs come from *devtest*.
- both: lookup first, n-gram when lookup finds nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Sequence

from atlas_trace.tokens_per_word import LANGS

Draft = Callable[[Sequence[int], int], list[int]]
MAX_N = 4


def ngram_table(corpus: Sequence[Sequence[int]], max_n: int = MAX_N) -> dict[tuple[int, ...], int]:
    """Most frequent next token for every context of length 1..max_n-1."""
    counts: dict[tuple[int, ...], Counter] = defaultdict(Counter)
    for seq in corpus:
        for n in range(1, max_n):
            for i in range(n, len(seq)):
                counts[tuple(seq[i - n:i])][seq[i]] += 1
    return {ctx: c.most_common(1)[0][0] for ctx, c in counts.items()}


def ngram_drafter(table: dict[tuple[int, ...], int], max_n: int = MAX_N) -> Draft:
    def draft(ctx: Sequence[int], k: int) -> list[int]:
        out: list[int] = []
        seq = list(ctx)
        for _ in range(k):
            nxt = next((table[tuple(seq[-n:])] for n in range(max_n - 1, 0, -1)
                        if len(seq) >= n and tuple(seq[-n:]) in table), None)
            if nxt is None:
                break
            out.append(nxt)
            seq.append(nxt)
        return out
    return draft


def lookup_draft(ctx: Sequence[int], k: int, n_max: int = 3) -> list[int]:
    """Prompt lookup: the most recent earlier occurrence of the last n tokens."""
    for n in range(n_max, 0, -1):
        if len(ctx) <= n:
            continue
        key = list(ctx[-n:])
        for i in range(len(ctx) - n - 1, -1, -1):
            if list(ctx[i:i + n]) == key:
                return list(ctx[i + n:i + n + k])
    return []


def combine(*drafters: Draft) -> Draft:
    def draft(ctx: Sequence[int], k: int) -> list[int]:
        return next((d for d in (f(ctx, k) for f in drafters) if d), [])
    return draft


def simulate(prompt: Sequence[int], output: Sequence[int], draft: Draft, k: int) -> tuple[int, int]:
    """(tokens produced, verification steps) for one greedy output."""
    pos, steps = 0, 0
    while pos < len(output):
        proposal = draft([*prompt, *output[:pos]], k)
        accepted = next((i for i, t in enumerate(proposal) if pos + i >= len(output) or output[pos + i] != t), len(proposal))
        pos += accepted + 1  # the verification pass always yields one more token
        steps += 1
    return len(output), steps


def continuation_share(ids: Sequence[int], tok) -> float:
    """Share of tokens that continue a word (no leading space in the piece)."""
    pieces = tok.convert_ids_to_tokens(list(ids))
    return sum(1 for p in pieces if not p.startswith("Ġ") and not p.startswith("Ċ")) / max(1, len(pieces))


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=Path, default=Path("results/baseline/m4-cpu-t6-languages.json"))
    ap.add_argument("--flores", type=Path, default=Path("data/flores200_dataset"))
    ap.add_argument("--tokenizer", default="data/models/n-atlas-bf16")
    ap.add_argument("--corpus-dir", type=Path, help="extra text per language: <dir>/<lang>.txt")
    ap.add_argument("--corpus-chars", type=int, default=0, help="characters of extra text to use")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    samples = json.loads(args.samples.read_text())["samples"]
    en_src = (args.flores / "devtest" / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
    yo_src = (args.flores / "devtest" / "yor_Latn.devtest").read_text(encoding="utf-8").splitlines()
    names = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}

    rows = []
    for lang in LANGS:
        dev = (args.flores / "dev" / f"{LANGS[lang]}.dev").read_text(encoding="utf-8").splitlines()
        extra = []
        if args.corpus_dir and args.corpus_chars:
            text = (args.corpus_dir / f"{lang}.txt").read_text(encoding="utf-8")[: args.corpus_chars]
            extra = [line for line in text.splitlines() if line.strip()]
        lines = dev + extra
        ids = tok(lines, add_special_tokens=False)["input_ids"]
        table = ngram_table(ids)
        drafters = {"lookup": lookup_draft, "ngram": ngram_drafter(table), "both": combine(lookup_draft, ngram_drafter(table))}
        mine = [s for s in samples if s["lang"] == lang]
        outs = [tok(s["output"], add_special_tokens=False)["input_ids"] for s in mine]
        prompts = []
        for s in mine:
            src_lang, src = ("yo", yo_src) if lang == "en" else ("en", en_src)
            text = (f"Translate the following {names[src_lang]} text into {names[lang]}. "
                    f"Reply with the translation only.\n\n{src[s['index']]}")
            prompts.append(tok(text, add_special_tokens=False)["input_ids"])
        cont = sum(continuation_share(o, tok) * len(o) for o in outs) / sum(len(o) for o in outs)
        row = {"lang": lang, "outputs": len(outs), "tokens": sum(map(len, outs)), "continuation_share": cont,
               "draft_corpus_tokens": sum(map(len, ids)), "tau": {}}
        for name, d in drafters.items():
            for k in (4, 8):
                made, steps = map(sum, zip(*(simulate(p, o, d, k) for p, o in zip(prompts, outs))))
                row["tau"][f"{name}-k{k}"] = made / steps
        rows.append(row)
        taus = "  ".join(f"{k} {v:.2f}" for k, v in row["tau"].items())
        print(f"{lang}: corpus {row['draft_corpus_tokens']:>9} tok | tau {taus}", file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps({"samples": str(args.samples), "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
