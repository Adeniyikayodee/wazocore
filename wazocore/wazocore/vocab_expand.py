"""In-place vocabulary expansion of the N-ATLaS tokenizer for Yoruba,
Hausa and Igbo (arXiv 2607.15232).

The existing BPE is continued: text is first segmented by the current
tokenizer, then new merges are learned on top, so every existing token
keeps its id and each new token has an exact decomposition into old ones.
`build` writes the expanded tokenizer; the default command measures the
token savings on FLORES devtest (merges learned only on other text).
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

Word = tuple[str, ...]


def learn_merges(words: dict[Word, int], n_merges: int, allowed: Callable[[str], bool] = lambda _: True) -> list[tuple[str, str]]:
    """Standard BPE continuation over pre-segmented words (weighted by
    frequency): repeatedly merge the most frequent adjacent pair whose
    result is `allowed`. Pair counts are updated incrementally; a lazy
    max-heap picks the best pair."""
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
        if not allowed(p[0] + p[1]):
            continue  # dropped from the heap; re-checked only if its count grows
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


def whole_utf8(token: str, byte_of: dict[str, int]) -> bool:
    """True if a byte-level token decodes to complete UTF-8 characters.
    A token ending mid-character lets the model emit an invalid byte
    sequence if the next token does not complete it (seen in V3/V4)."""
    try:
        bytes(byte_of[c] for c in token).decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


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


def expand(tok_json: dict, merges: Sequence[tuple[str, str]], first_id: int) -> tuple[dict, list[dict]]:
    """Returns a new tokenizer.json with `merges` appended after the
    original merges and one vocabulary entry per new token (ids from
    `first_id`), plus each new token's (left, right) decomposition. Tokens
    that already exist keep their id; their merge is still appended.

    The `tokenizers` library needs BPE vocabulary ids to be contiguous, but
    Llama-3 keeps its special tokens (ids between the BPE vocabulary and
    `first_id`) only in `added_tokens`. They are entered into the BPE
    vocabulary under their existing ids; no merge can produce them, so
    encoding is unchanged and every original id stays where it was."""
    vocab = dict(tok_json["model"]["vocab"])
    for t in tok_json.get("added_tokens", []):
        vocab.setdefault(t["content"], t["id"])
    added: list[dict] = []
    next_id = first_id
    for left, right in merges:
        token = left + right
        if token not in vocab:
            vocab[token] = next_id
            added.append({"token": token, "id": next_id, "left": left, "right": right})
            next_id += 1
    if sorted(vocab.values()) != list(range(len(vocab))):
        raise ValueError("vocabulary ids are not contiguous; the tokenizer would renumber tokens")
    model = {**tok_json["model"], "vocab": vocab, "merges": [*tok_json["model"]["merges"], *[[a, b] for a, b in merges]]}
    return {**tok_json, "model": model}, added


def balanced_corpus(texts: dict[str, str], chars: int) -> dict[str, str]:
    """Equal character budget per language, so no language dominates merges."""
    return {lang: t[:chars] for lang, t in texts.items()}


# ---- shell -------------------------------------------------------------------


def segment(tok, text: str) -> Iterable[Word]:
    """Pre-tokenise with N-ATLaS's own pre-tokenizer, then tokenise each
    piece with its current BPE model."""
    for piece, _ in tok.pre_tokenizer.pre_tokenize_str(text):
        yield tuple(t.value for t in tok.model.tokenize(piece))


def build(argv: Sequence[str]) -> int:
    """Joint merges for yo/ha/ig on balanced Wikipedia + FLORES dev text;
    writes an expanded tokenizer directory and the new-token list."""
    ap = argparse.ArgumentParser(description="build an expanded tokenizer")
    ap.add_argument("--source", type=Path, default=Path("data/models/n-atlas-bf16"))
    ap.add_argument("--wiki", type=Path, default=Path("data/wiki"))
    ap.add_argument("--flores", type=Path, default=Path("data/flores200_dataset"))
    ap.add_argument("--chars-per-lang", type=int, default=4_000_000)
    ap.add_argument("--merges", type=int, default=15_000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    import shutil

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.source / "tokenizer.json"))
    codes = {"yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}
    texts = {l: (args.flores / "dev" / f"{c}.dev").read_text(encoding="utf-8") + "\n" + (args.wiki / f"{l}.txt").read_text(encoding="utf-8")
             for l, c in codes.items()}
    words: Counter = Counter()
    for text in balanced_corpus(texts, args.chars_per_lang).values():
        words.update(segment(tok, text))
    from wazocore.vx_core import byte_decoder

    byte_of = byte_decoder()
    merges = learn_merges(words, args.merges, allowed=lambda t: whole_utf8(t, byte_of))
    tok_json = json.loads((args.source / "tokenizer.json").read_text(encoding="utf-8"))
    first_id = json.loads((args.source / "config.json").read_text())["vocab_size"]
    new_json, added = expand(tok_json, merges, first_id)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tokenizer.json").write_text(json.dumps(new_json, ensure_ascii=False), encoding="utf-8")
    for f in ["tokenizer_config.json", "special_tokens_map.json", "generation_config.json"]:
        shutil.copy(args.source / f, args.out / f)
    config = json.loads((args.source / "config.json").read_text())
    config["vocab_size"] = first_id + len(added)
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.out / "new_tokens.json").write_text(json.dumps({"first_id": first_id, "merges": len(merges), "tokens": added}, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"{len(merges)} merges, {len(added)} new tokens (ids {first_id}..{first_id + len(added) - 1}) -> {args.out}", file=sys.stderr)
    return 0


def main(argv: Sequence[str]) -> int:
    if argv and argv[0] == "build":
        return build(argv[1:])
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
