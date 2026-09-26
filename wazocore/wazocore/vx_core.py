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
