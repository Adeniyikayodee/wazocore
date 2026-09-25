"""Train the new-token rows of the expanded N-ATLaS on Apple MLX
(PROJECT-PLAN-B §14, V3).

The base model stays frozen and 4-bit. The 15,000 new input rows and
output rows are full-precision trainable arrays spliced into the
embedding lookup and the output head, following the low-budget recipes
(arXiv 2406.11477, 2608.03494): train the new rows first.

    python -m atlas_trace.vx_train --model data/models/n-atlas-vx-mlx-q4 \
        --tokens 2000000 --out data/runs/vx-rows-1
    python -m atlas_trace.vx_train apply --model data/models/n-atlas-vx-mlx-q4 \
        --rows data/runs/vx-rows-1/rows.safetensors --out data/models/n-atlas-vx-trained
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

# ---- pure core ---------------------------------------------------------------


def chunks(ids: Sequence[int], length: int, bos: int) -> list[list[int]]:
    """Consecutive windows of `length` tokens, each starting with BOS."""
    body = length - 1
    return [[bos, *ids[i:i + body]] for i in range(0, len(ids) - body + 1, body)]


def mixture(per_lang: dict[str, list[list[int]]], shares: dict[str, float], total: int, seed: int) -> list[list[int]]:
    """Deterministic shuffled mix with each language's share of `total`
    chunks (cycling through a language's chunks if it runs short)."""
    rng = random.Random(seed)
    out: list[list[int]] = []
    for lang, share in shares.items():
        pool = per_lang[lang]
        n = round(total * share)
        out += [pool[i % len(pool)] for i in range(n)] if pool else []
    rng.shuffle(out)
    return out


def lr_at(step: int, peak: float, warmup: int, total: int) -> float:
    """Linear warm-up, then cosine decay to 10% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


# ---- shell (MLX) -------------------------------------------------------------


def split_modules(model, first_new: int, stage: str = "both"):
    """Replaces embeddings and head with frozen-base + trainable-new-rows
    versions initialised from the model's current (quantised) new rows.
    stage "head" trains only the output rows: their gradient needs just the
    final hidden states, so no backward pass through the layers is run.
    stage "both" also trains the input rows, with gradient checkpointing
    on every layer so a 16 GB machine does not swap."""
    import mlx.core as mx
    import mlx.nn as nn

    def cut(q, rows: slice):
        return q.weight[rows], q.scales[rows], q.biases[rows]

    class SplitEmbedding(nn.Module):
        def __init__(self, q):
            super().__init__()
            w, s, b = cut(q, slice(0, first_new))
            self._base = (w, s, b, q.group_size, q.bits)
            nw, ns, nb = cut(q, slice(first_new, None))
            self.new_rows = mx.dequantize(nw, ns, nb, group_size=q.group_size, bits=q.bits).astype(mx.float32)

        def __call__(self, ids):
            w, s, b, gs, bits = self._base
            base_ids = mx.minimum(ids, first_new - 1)
            base = mx.dequantize(w[base_ids], s[base_ids], b[base_ids], group_size=gs, bits=bits)
            new = self.new_rows[mx.maximum(ids - first_new, 0)]
            return mx.where((ids >= first_new)[..., None], new.astype(base.dtype), base)

        def as_linear(self, x):  # unused (untied model)
            raise NotImplementedError

    class SplitHead(nn.Module):
        def __init__(self, q):
            super().__init__()
            w, s, b = cut(q, slice(0, first_new))
            self._base = (w, s, b, q.group_size, q.bits)
            nw, ns, nb = cut(q, slice(first_new, None))
            self.new_rows = mx.dequantize(nw, ns, nb, group_size=q.group_size, bits=q.bits).astype(mx.float32)

        def __call__(self, h):
            w, s, b, gs, bits = self._base
            base = mx.quantized_matmul(h, w, s, b, transpose=True, group_size=gs, bits=bits)
            new = h.astype(mx.float32) @ self.new_rows.T
            return mx.concatenate([base.astype(mx.float32), new], axis=-1)

    model.model.embed_tokens = SplitEmbedding(model.model.embed_tokens)
    model.lm_head = SplitHead(model.lm_head)
    model.freeze()
    model.lm_head.unfreeze(keys=["new_rows"])
    if stage == "both":
        from mlx_lm.tuner.trainer import grad_checkpoint

        model.model.embed_tokens.unfreeze(keys=["new_rows"])
        grad_checkpoint(model.model.layers[0])
    return model


def load_texts(wiki: Path, langs: Sequence[str], chars: int) -> dict[str, str]:
    return {l: (wiki / f"{l}.txt").read_text(encoding="utf-8")[:chars] for l in langs}


def bpc_subset(model, tok, flores: Path, n: int) -> dict[str, float]:
    import mlx.core as mx

    codes = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}
    out = {}
    for lang, code in codes.items():
        lines = (flores / "devtest" / f"{code}.devtest").read_text(encoding="utf-8").splitlines()[:n]
        nll, chars = 0.0, 0
        for s in lines:
            ids = [tok.bos_token_id, *tok.encode(s, add_special_tokens=False)]
            logits = model(mx.array([ids]))[0].astype(mx.float32)
            lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            nll += -float(mx.take_along_axis(lp[:-1], mx.array(ids[1:])[:, None], axis=-1).sum())
            chars += len(s)
        out[lang] = nll / math.log(2) / chars
    return out


def train(args) -> int:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from mlx_lm import load

    model, tok = load(str(args.model))
    first_new = json.loads((args.model / "new_tokens.json").read_text())["first_id"]
    model = split_modules(model, first_new, args.stage)
    if args.rows:
        start = mx.load(str(args.rows))
        model.model.embed_tokens.new_rows = start["embed_new"].astype(mx.float32)
        model.lm_head.new_rows = start["head_new"].astype(mx.float32)
        model.freeze()
        model.lm_head.unfreeze(keys=["new_rows"])
        if args.stage == "both":
            model.model.embed_tokens.unfreeze(keys=["new_rows"])
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"trainable parameters: {trainable / 1e6:.1f}M", file=sys.stderr)

    langs = ["yo", "ha", "ig", "en"]
    texts = load_texts(args.wiki, langs, args.chars_per_lang)
    per_lang = {l: chunks(tok.encode(t, add_special_tokens=False), args.seq, tok.bos_token_id) for l, t in texts.items()}
    shares = {"yo": 0.3, "ha": 0.3, "ig": 0.3, "en": 0.1}
    steps = args.tokens // args.seq
    data = mixture(per_lang, shares, steps, args.seed)
    print(f"{steps} steps of {args.seq} tokens; chunks available: " + ", ".join(f"{l} {len(c)}" for l, c in per_lang.items()), file=sys.stderr)

    def loss_fn(m, batch):
        x, y = batch[:, :-1], batch[:, 1:]
        logits = m(x).astype(mx.float32)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    opt = optim.Adam(learning_rate=args.lr)
    step_fn = nn.value_and_grad(model, loss_fn)
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")
    t0, seen, losses = time.time(), 0, []
    for step, chunk in enumerate(data):
        opt.learning_rate = lr_at(step, args.lr, args.warmup, steps)
        loss, grads = step_fn(model, mx.array([chunk]))
        opt.update(model, grads)
        mx.eval(model.trainable_parameters(), opt.state, loss)
        losses.append(float(loss))
        seen += len(chunk)
        if (step + 1) % args.report == 0:
            rec = {"step": step + 1, "tokens": seen, "loss": sum(losses[-args.report:]) / args.report,
                   "tok_s": seen / (time.time() - t0), "lr": opt.learning_rate.item() if hasattr(opt.learning_rate, "item") else opt.learning_rate}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {rec['step']} loss {rec['loss']:.3f} {rec['tok_s']:.0f} tok/s", file=sys.stderr)
        if (step + 1) % args.save_every == 0 or step + 1 == len(data):
            rows = {"embed_new": model.model.embed_tokens.new_rows.astype(mx.float16), "head_new": model.lm_head.new_rows.astype(mx.float16)}
            mx.save_safetensors(str(args.out / "rows.safetensors"), rows)
            bpc = bpc_subset(model, tok, args.flores, args.eval_sentences)
            log.write(json.dumps({"step": step + 1, "tokens": seen, "bpc": bpc}) + "\n")
            log.flush()
            print(f"checkpoint {step + 1}: BPC " + " ".join(f"{k} {v:.3f}" for k, v in bpc.items()), file=sys.stderr)
    return 0


def apply(args) -> int:
    """Writes a model directory with the trained rows quantised into the tables."""
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm import load

    model, _ = load(str(args.model))
    first_new = json.loads((args.model / "new_tokens.json").read_text())["first_id"]
    rows = mx.load(str(args.rows))
    for q, key in [(model.model.embed_tokens, "embed_new"), (model.lm_head, "head_new")]:
        w, s, b = mx.quantize(rows[key], group_size=q.group_size, bits=q.bits)
        q.weight = mx.concatenate([q.weight[:first_new], w])
        q.scales = mx.concatenate([q.scales[:first_new], s])
        q.biases = mx.concatenate([q.biases[:first_new], b])
    args.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out / "model.safetensors"), dict(tree_flatten(model.parameters())))
    for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "new_tokens.json"]:
        shutil.copy(args.model / f, args.out / f)
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def main(argv: Sequence[str]) -> int:
    if argv and argv[0] == "apply":
        ap = argparse.ArgumentParser(description="apply trained rows")
        ap.add_argument("--model", type=Path, required=True)
        ap.add_argument("--rows", type=Path, required=True)
        ap.add_argument("--out", type=Path, required=True)
        return apply(ap.parse_args(argv[1:]))
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--wiki", type=Path, default=Path("data/wiki"))
    ap.add_argument("--flores", type=Path, default=Path("data/flores200_dataset"))
    ap.add_argument("--chars-per-lang", type=int, default=12_000_000)
    ap.add_argument("--tokens", type=int, default=2_000_000)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--report", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--eval-sentences", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", choices=["head", "both"], default="both")
    ap.add_argument("--rows", type=Path, help="start from previously trained rows")
    ap.add_argument("--out", type=Path, required=True)
    return train(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
