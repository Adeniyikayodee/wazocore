"""Expanded N-ATLaS: add rows for the new tokens of `vocab_expand build` to
the input embeddings and the output head (PROJECT-PLAN-B §14, V1/V2).

Initialisation (arXiv 2608.03494, best subword-composition variant):
input row = uniform mean of the token's original pieces; output row =
character-length-weighted mean of the pieces' output rows.

Stages (Apple MLX, on the 4-bit training model):
    python -m atlas_trace.vx_model build --base data/models/n-atlas-mlx-q4 \
        --vx data/models/n-atlas-vx --out data/models/n-atlas-vx-mlx-q4
    python -m atlas_trace.vx_model bpc --model M --flores F --out results/vocab/bpc-M.json
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


# ---- pure core ---------------------------------------------------------------


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


# ---- shell (MLX) -------------------------------------------------------------


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


def build(args) -> int:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load

    model, _ = load(str(args.base))
    vx_tok = json.loads((args.vx / "tokenizer.json").read_text(encoding="utf-8"))
    new = json.loads((args.vx / "new_tokens.json").read_text(encoding="utf-8"))
    orig_vocab = json.loads((args.source_tokenizer).read_text(encoding="utf-8"))["model"]["vocab"]
    parts = decompose(new["tokens"], orig_vocab)
    inv = {i: s for s, i in orig_vocab.items()}
    bd = byte_decoder()
    weights = char_lengths(parts, lambda i: bytes(bd[c] for c in inv[i]))

    # Quantisation groups run along each row, so rows are independent: only
    # the pieces' rows are dequantised, and only the new rows are quantised
    # and appended. No dense copy of the full tables is ever made.
    ids = sorted(parts)
    needed = mx.array(sorted({p for ps in parts.values() for p in ps}))
    position = {int(p): k for k, p in enumerate(np.array(needed))}

    def pieces_dense(q) -> np.ndarray:
        rows = mx.dequantize(q.weight[needed], q.scales[needed], q.biases[needed], group_size=q.group_size, bits=q.bits)
        return np.array(rows.astype(mx.float32))

    def remap(p: dict[int, list[int]]) -> dict[int, list[int]]:
        return {i: [position[x] for x in xs] for i, xs in p.items()}

    def append_rows(q, rows: np.ndarray) -> None:
        w, sc, bi = mx.quantize(mx.array(rows, dtype=mx.float16), group_size=q.group_size, bits=q.bits)
        q.weight = mx.concatenate([q.weight, w])
        q.scales = mx.concatenate([q.scales, sc])
        q.biases = mx.concatenate([q.biases, bi])

    emb, head = model.model.embed_tokens, model.lm_head
    local = remap(parts)
    append_rows(emb, init_rows(pieces_dense(emb), local))
    append_rows(head, init_rows(pieces_dense(head), local, {i: weights[i] for i in ids}))
    vocab = emb.weight.shape[0]
    if head.weight.shape[0] != vocab or vocab != new["first_id"] + len(ids):
        raise ValueError(f"row count mismatch: embed {vocab}, head {head.weight.shape[0]}")
    args.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out / "model.safetensors"), dict(tree_flatten(model.parameters())))
    config = json.loads((args.base / "config.json").read_text())
    config["vocab_size"] = vocab
    (args.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "new_tokens.json"]:
        shutil.copy(args.vx / f, args.out / f)
    print(f"expanded model: vocab {vocab}, {len(parts)} new rows -> {args.out}", file=sys.stderr)
    return 0


def bpc(args) -> int:
    """Bits per character on FLORES devtest sentences (BOS + sentence,
    sum of next-token NLL / characters): comparable across tokenizers."""
    import mlx.core as mx
    from mlx_lm import load

    model, tok = load(str(args.model))
    codes = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}
    bos = tok.bos_token_id
    rows = []
    for lang, code in codes.items():
        lines = (args.flores / "devtest" / f"{code}.devtest").read_text(encoding="utf-8").splitlines()[: args.sentences]
        nll, chars, tokens = 0.0, 0, 0
        for s in lines:
            ids = [bos, *tok.encode(s, add_special_tokens=False)]
            logits = model(mx.array([ids]))[0].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            tgt = mx.array(ids[1:])
            nll += -float(mx.take_along_axis(lp[:-1], tgt[:, None], axis=-1).sum())
            chars += len(s)
            tokens += len(ids) - 1
        rows.append({"lang": lang, "sentences": len(lines), "chars": chars, "tokens": tokens,
                     "nll_nats": nll, "bpc": bits_per_char(nll, chars), "tokens_per_char": tokens / chars})
        print(f"{lang}: BPC {rows[-1]['bpc']:.4f} ({tokens} tokens, {chars} chars)", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": str(args.model), "rows": rows}, indent=1) + "\n")
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    b = sub.add_parser("build")
    b.add_argument("--base", type=Path, required=True)
    b.add_argument("--vx", type=Path, required=True)
    b.add_argument("--source-tokenizer", type=Path, default=Path("data/models/n-atlas-bf16/tokenizer.json"))
    b.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("bpc")
    e.add_argument("--model", type=Path, required=True)
    e.add_argument("--flores", type=Path, default=Path("data/flores200_dataset"))
    e.add_argument("--sentences", type=int, default=300)
    e.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"build": build, "bpc": bpc}[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
