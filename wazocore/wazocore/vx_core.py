"""Pure helpers for vocabulary expansion: new-row initialisation, the
byte-level alphabet, data windows, mixing and the learning-rate schedule.
No I/O; the GPU shell is `vx_torch`, the tokenizer builder `vocab_expand`."""

from __future__ import annotations

import math
import random
from typing import Sequence

import numpy as np


def decompose(new_tokens: Sequence[dict], vocab: dict[str, int]) -> dict[int, list[int]]:
    """Original-vocabulary ids each new token is made of, by recursively
    splitting its (left, right) merge until pieces are original tokens."""
    by_token = {t["token"]: t for t in new_tokens}

    def pieces(s: str) -> list[int]:
        if s in vocab and s not in by_token:
            return [vocab[s]]
        t = by_token[s]
        return pieces(t["left"]) + pieces(t["right"])

    return {t["id"]: pieces(t["token"]) for t in new_tokens}


def init_rows(table: np.ndarray, parts: dict[int, list[int]], weights: dict[int, list[float]] | None = None) -> np.ndarray:
    """New rows in id order: the (weighted) mean of each token's piece rows."""
    ids = sorted(parts)
    rows = []
    for i in ids:
        p = np.asarray(parts[i])
        w = np.ones(len(p)) if weights is None else np.asarray(weights[i], dtype=np.float64)
        rows.append((table[p].astype(np.float64) * (w / w.sum())[:, None]).sum(0))
    return np.stack(rows).astype(table.dtype)


def char_lengths(parts: dict[int, list[int]], id_to_bytes) -> dict[int, list[float]]:
    """Weights for the output head: each piece's length in decoded bytes."""
    return {i: [max(1, len(id_to_bytes(p))) for p in ps] for i, ps in parts.items()}


def bits_per_char(nll_nats: float, chars: int) -> float:
    return nll_nats / math.log(2) / chars


def byte_decoder():
    """GPT-2 byte-level alphabet: printable stand-in character -> byte."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def chunks(ids: Sequence[int], length: int, bos: int) -> list[list[int]]:
    """Consecutive windows of `length` tokens, each starting with BOS."""
    body = length - 1
    return [[bos, *ids[i:i + body]] for i in range(0, len(ids) - body + 1, body)]


def mixture(per_lang: dict[str, list[list[int]]], shares: dict[str, float], total: int, seed: int) -> list[list[int]]:
    """Deterministic shuffled mix with each language's share of `total`
    chunks (cycling through a language's chunks if it runs short)."""
    rng = random.Random(seed)
    # Largest-remainder allocation: the counts add up to exactly `total`.
    exact = {lang: total * share / sum(shares.values()) for lang, share in shares.items()}
    counts = {lang: int(x) for lang, x in exact.items()}
    for lang in sorted(exact, key=lambda l: exact[l] - counts[l], reverse=True)[: total - sum(counts.values())]:
        counts[lang] += 1
    out: list[list[int]] = []
    for lang, n in counts.items():
        pool = per_lang[lang]
        out += [pool[i % len(pool)] for i in range(n)] if pool else []
    rng.shuffle(out)
    return out


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    """Linear warm-up, then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


ENGLISH_MARKERS = (" the ", " of the ", " and the ", " is a ", " in the ")

# Plain spellings of common Hausa words that are written with hooked letters
# (ɗan, ƙasa, ...) and undotted spellings of common Igbo words (ndị, bụ, ...):
# Wikipedia uses these careless forms often enough for a model to learn them.
CARELESS = {
    "ha": {"dan", "kasa", "kasar", "daya", "wadanda", "kungiyar", "karkashin", "yaki", "bangare", "kwallon", "kafa", "hada", "dari"},
    "ig": {"ndi", "bu", "ozo", "ulo", "ochichi", "muru"},
}
IGBO_DOTS = set("ịọụṅỊỌỤṄ")
YORUBA_MARKS = {"̀", "́", "̣"}  # grave, acute, dot below (after NFD)


def careful_line(line: str, lang: str) -> bool:
    """True if a Wikipedia line looks carefully written in `lang`: not
    English, and using the language's special letters the way its
    references do. Lines failing this taught v3 careless spellings."""
    import unicodedata

    text = unicodedata.normalize("NFC", line.strip())
    if len(text) < 20:
        return False
    if lang == "en":
        return True
    padded = f" {text.lower()} "
    if any(m in padded for m in ENGLISH_MARKERS):
        return False
    words = set("".join(c if c.isalnum() else " " for c in text.lower()).split())
    if words & CARELESS.get(lang, set()):
        return False
    if lang == "ig":
        return sum(c in IGBO_DOTS for c in text) / len(text) >= 0.03
    if lang == "yo":
        return sum(c in YORUBA_MARKS for c in unicodedata.normalize("NFD", text)) / len(text) >= 0.05
    return True


def careful_text(text: str, lang: str) -> str:
    return "\n".join(l for l in text.splitlines() if careful_line(l, lang))


def chunk_align(a: Sequence[int], b: Sequence[int]) -> list[tuple[int, int, int, int]] | None:
    """Aligns two tokenisations of the same byte string, given each token's
    length in bytes. Returns chunks (a_start, a_end, b_start, b_end): the
    shortest spans that end where both have a token boundary. Each WazoCore
    token and its original pieces fall in one chunk. None if the two do not
    cover the same number of bytes."""
    if sum(a) != sum(b):
        return None
    chunks, i, j, ia, jb, pa, pb = [], 0, 0, 0, 0, 0, 0
    while i < len(a) or j < len(b):
        if pa <= pb and i < len(a):
            pa += a[i]; i += 1
        else:
            pb += b[j]; j += 1
        if pa == pb:
            chunks.append((ia, i, jb, j))
            ia, jb = i, j
    return chunks


def chunk_sums(values: Sequence[float], spans: Sequence[tuple[int, int]]) -> list[float]:
    """Log-probability of each chunk: the sum of its tokens' log-probabilities."""
    return [sum(values[s:e]) for s, e in spans]


def nfc_rows(rows: Sequence[dict]) -> list[dict]:
    """Rows with `output` and `reference` in Unicode NFC. NTREX-128 Yoruba
    references are stored decomposed (letter + combining accent) while the
    models write composed characters; scored raw, chrF++ counts the two as
    different letters and halves every model's score."""
    import unicodedata

    return [{**r, "output": unicodedata.normalize("NFC", r["output"]), "reference": unicodedata.normalize("NFC", r["reference"])} for r in rows]


def sign_test(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value that paired disagreements (`only_a` items
    where just A shows the event, `only_b` where just B does) are balanced:
    McNemar's exact test."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n)


def looping(text: str, unit: tuple[int, int] = (2, 6), repeats: int = 6) -> bool:
    """A short unit repeated back to back (e.g. 'ereereere…'): the failure
    WazoCore showed more often than the original."""
    import re

    lo, hi = unit
    return re.search(rf"(.{{{lo},{hi}}})\1{{{repeats - 1},}}", text) is not None
