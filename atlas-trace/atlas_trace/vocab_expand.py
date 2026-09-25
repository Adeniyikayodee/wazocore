"""Experiment: how many tokens would Yoruba/Hausa/Igbo save if N-ATLaS's
tokenizer were expanded in place (PROJECT-PLAN-B §12, research 25 Sep)?

In-place expansion (arXiv 2607.15232) continues the existing BPE: text is
first segmented by the current tokenizer, then new merges are learned on
top, so every existing token is kept and each new token has an exact
decomposition into old ones. This measures the ceiling: tokens per word on
FLORES devtest before and after, with merges learned only on other text
(Wikipedia + FLORES dev). No model training is involved.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

Word = tuple[str, ...]


def learn_merges(words: dict[Word, int], n_merges: int) -> list[tuple[str, str]]:
    """Standard BPE continuation over pre-segmented words (weighted by
    frequency): repeatedly merge the most frequent adjacent pair. Pair
    counts are updated incrementally; a lazy max-heap picks the best pair."""
    seqs: list[list[str]] = [list(w) for w in words]
    freq: list[int] = list(words.values())
    pairs: Counter = Counter()
    where: dict[tuple[str, str], set[int]] = defaultdict(set)
    for i, s in enumerate(seqs):
        for p in zip(s, s[1:]):
            pairs[p] += freq[i]
            where[p].add(i)
    heap = [(-c, p) for p, c in pairs.items()]
    heapq.heapify(heap)
    merges: list[tuple[str, str]] = []
    while heap and len(merges) < n_merges:
        c, p = heapq.heappop(heap)
        if pairs.get(p, 0) != -c or -c < 2:
            continue  # stale entry, or nothing worth merging
        merges.append(p)
        new = p[0] + p[1]
        touched: Counter = Counter()
        for i in list(where.pop(p, ())):
            s, f = seqs[i], freq[i]
            for q in zip(s, s[1:]):
                pairs[q] -= f
                touched[q] -= f
            out, k = [], 0
            while k < len(s):
                if k + 1 < len(s) and (s[k], s[k + 1]) == p:
                    out.append(new)
                    k += 2
                else:
                    out.append(s[k])
                    k += 1
            seqs[i] = out
            for q in zip(out, out[1:]):
                pairs[q] += f
                touched[q] += f
                where[q].add(i)
        pairs.pop(p, None)
        for q, d in touched.items():
            if d > 0 and pairs.get(q, 0) > 0:
                heapq.heappush(heap, (-pairs[q], q))
    return merges


def apply_merges(tokens: Sequence[str], ranks: dict[tuple[str, str], int]) -> list[str]:
    """BPE encoding of one word, starting from its original tokens, using
    only the new merges (lowest rank first)."""
    s = list(tokens)
    while len(s) > 1:
        best = min(((ranks.get(p, 1 << 30), k) for k, p in enumerate(zip(s, s[1:]))), default=(1 << 30, 0))
        if best[0] == 1 << 30:
            break
        k = best[1]
        s[k:k + 2] = [s[k] + s[k + 1]]
    return s


# ---- shell -------------------------------------------------------------------


def segment(tok, text: str) -> Iterable[Word]:
    """Pre-tokenise with N-ATLaS's own pre-tokenizer, then tokenise each
    piece with its current BPE model."""
    for piece, _ in tok.pre_tokenizer.pre_tokenize_str(text):
        yield tuple(t.value for t in tok.model.tokenize(piece))


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tokenizer", type=Path, default=Path("data/models/n-atlas-bf16/tokenizer.json"))
    ap.add_argument("--wiki", type=Path, default=Path("data/wiki"))
    ap.add_argument("--flores", type=Path, default=Path("data/flores200_dataset"))
    ap.add_argument("--wiki-chars", type=int, default=6_000_000)
    ap.add_argument("--merges", type=int, nargs="+", default=[2000, 5000, 10000])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.tokenizer))
    codes = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}
    devtest = {l: (args.flores / "devtest" / f"{c}.devtest").read_text(encoding="utf-8") for l, c in codes.items()}
    words_per_lang = {l: sum(len(line.split()) for line in t.splitlines()) for l, t in devtest.items()}
    base = {l: sum(len(w) for w in segment(tok, t)) for l, t in devtest.items()}

    rows = []
    for lang in ["yo", "ha", "ig"]:
        train = (args.wiki / f"{lang}.txt").read_text(encoding="utf-8")[: args.wiki_chars] + "\n" + \
            (args.flores / "dev" / f"{codes[lang]}.dev").read_text(encoding="utf-8")
        words = Counter(segment(tok, train))
        merges = learn_merges(words, max(args.merges))
        test_words = list(segment(tok, devtest[lang]))
        for n in args.merges:
            ranks = {p: r for r, p in enumerate(merges[:n])}
            after = sum(len(apply_merges(w, ranks)) for w in test_words)
            rows.append({
                "lang": lang, "new_tokens": n, "train_chars": len(train),
                "tokens_before": base[lang], "tokens_after": after,
                "tokens_per_word_before": base[lang] / words_per_lang[lang],
                "tokens_per_word_after": after / words_per_lang[lang],
                "reduction": base[lang] / after,
                "tax_vs_en_before": base[lang] / base["en"], "tax_vs_en_after": after / base["en"],
            })
            r = rows[-1]
            print(f"{lang} +{n:>5} tokens: {r['tokens_per_word_before']:.2f} -> {r['tokens_per_word_after']:.2f} tok/word "
                  f"({r['reduction']:.2f}x fewer) | tax vs en {r['tax_vs_en_before']:.2f}x -> {r['tax_vs_en_after']:.2f}x", file=sys.stderr)
    if args.out:
        args.out.write_text(json.dumps({"method": "in-place BPE continuation on Wikipedia + FLORES dev; measured on FLORES devtest",
                                        "vocab_before": tok.get_vocab_size(), "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
