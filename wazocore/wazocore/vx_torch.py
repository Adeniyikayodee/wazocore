"""Vocabulary expansion for N-ATLaS on a CUDA GPU (PyTorch): build the
expanded model, train it, distil the original's replies, and evaluate.

Recipe (arXiv 2406.11477, 2608.03494): new input rows = mean of the
token's original pieces; new output rows = byte-length-weighted mean; then
train the new rows plus the bottom and top two transformer layers on
target-language text with short sequences. Original vocabulary rows get
zero gradient, so every existing token keeps its exact embedding.

Stages (see docs/cloud-vx.md; `work/data` holds the laptop's data/wiki and
data/flores200_dataset, `n-atlas-vx-tok` its data/models/n-atlas-vx):
    python -m wazocore.vx_torch expand --base work/n-atlas --vx work/n-atlas-vx-tok --out work/vx-init
    python -m wazocore.vx_torch bpc    --model work/n-atlas  --flores work/data/flores200_dataset --out work/bpc-original.json
    python -m wazocore.vx_torch bpc    --model work/vx-init  --flores work/data/flores200_dataset --out work/bpc-init.json
    python -m wazocore.vx_torch train  --model work/vx-init --data work/data --out work/vx-trained
    python -m wazocore.vx_torch bpc    --model work/vx-trained --flores work/data/flores200_dataset --out work/bpc-trained.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from wazocore.vx_core import ENGLISH_MARKERS, bits_per_char, byte_decoder, careful_text, char_lengths, chunks, decompose, init_rows, lr_at, mixture

CODES = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}

# ---- model surgery (torch, but pure over its inputs) --------------------------

def expand_model(model, parts: dict[int, list[int]], out_weights: dict[int, list[float]]):
    """Appends one input and one output row per new token (ids in order),
    initialised from the token's pieces. Returns the same model object,
    resized; original rows are copied unchanged."""
    import torch

    ids = sorted(parts)
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    head = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    new_in = init_rows(emb, parts)
    new_out = init_rows(head, parts, out_weights)
    old = emb.shape[0]
    if ids != list(range(old, old + len(ids))):
        raise ValueError("new token ids must directly follow the current vocabulary")
    model.resize_token_embeddings(old + len(ids), mean_resizing=False)
    with torch.no_grad():
        dtype = model.get_input_embeddings().weight.dtype
        model.get_input_embeddings().weight[old:] = torch.from_numpy(new_in).to(dtype)
        model.get_output_embeddings().weight[old:] = torch.from_numpy(new_out).to(dtype)
    return model

def set_trainable(model, first_new: int, layers: Sequence[int]) -> list:
    """Freezes everything except the embedding/head matrices (with a
    gradient mask keeping rows below `first_new` fixed) and `layers`.
    Returns the trainable parameters."""
    model.requires_grad_(False)
    params = []
    for m in (model.get_input_embeddings(), model.get_output_embeddings()):
        w = m.weight
        w.requires_grad_(True)
        w.register_hook(lambda g, n=first_new: g.index_fill(0, _arange(n, g.device), 0.0))
        params.append(w)
    for i in layers:
        block = model.model.layers[i]
        block.requires_grad_(True)
        params += [p for p in block.parameters()]
    return params

def _arange(n: int, device):
    import torch

    return torch.arange(n, device=device)

def windows(ids: Sequence[int], length: int) -> list[list[int]]:
    """Consecutive non-overlapping `length`-token windows of a packed
    stream (conversations already carry their own BOS)."""
    return [list(ids[i:i + length]) for i in range(0, len(ids) - length + 1, length)]


def wiki_sentences(text: str, n: int, seed: int, lo: int = 40, hi: int = 300, drop_english: bool = False) -> list[str]:
    """`n` distinct sentences of `lo`-`hi` characters from a Wikipedia dump
    (split on line breaks and ". "), chosen deterministically by `seed`.
    `drop_english` skips sentences with common English phrases: the
    yo/ha/ig dumps quote English, which would teach "translating" English
    as if it were the source language."""
    import random

    pool = []
    for line in text.splitlines():
        for part in line.split(". "):
            part = part.strip()
            if lo <= len(part) <= hi and not (drop_english and any(m in f" {part.lower()} " for m in ENGLISH_MARKERS)):
                pool.append(part if part.endswith((".", "?", "!")) else part + ".")
    pool = sorted(set(pool))
    random.Random(seed).shuffle(pool)
    return pool[:n]


def paired_bootstrap(a: Sequence[str], b: Sequence[str], refs: Sequence[str], chrf, rounds: int = 1000, seed: int = 0) -> dict:
    """chrF difference b - a on the same sentences with a 95% interval from
    resampling sentences (both systems resampled together)."""
    import random

    rng, n, diffs = random.Random(seed), len(refs), []
    pick = lambda xs, idx: [xs[i] for i in idx]
    for _ in range(rounds):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(chrf(pick(b, idx), pick(refs, idx)) - chrf(pick(a, idx), pick(refs, idx)))
    diffs.sort()
    return {"delta": chrf(b, refs) - chrf(a, refs), "ci95": [diffs[int(0.025 * rounds)], diffs[int(0.975 * rounds) - 1]],
            "p_worse": sum(d < 0 for d in diffs) / rounds}


def translation_prompts(src: Sequence[str], tgt_name: str) -> list[str]:
    return [f"Translate the following English text into {tgt_name}. Reply with the translation only.\n\n{s}" for s in src]


def reverse_prompts(src: Sequence[str], src_name: str) -> list[str]:
    return [f"Translate the following {src_name} text into English. Reply with the translation only.\n\n{s}" for s in src]


def anchor_kl(student_logits, teacher_logits, n_old: int):
    """KL(teacher || student) per position over the original vocabulary.
    The student is renormalised over its first `n_old` logits, so the
    anchor keeps its preferences among old tokens without pushing mass
    away from the new ones."""
    import torch.nn.functional as F

    s = F.log_softmax(student_logits[..., :n_old].float(), dim=-1)
    t = F.log_softmax(teacher_logits[..., :n_old].float(), dim=-1)
    return (t.exp() * (t - s)).sum(-1).mean()


def lm_loss(model, batch):
    """Mean next-token cross-entropy over a [B, T] batch."""
    import torch.nn.functional as F

    logits = model(input_ids=batch[:, :-1]).logits.float()
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1))

# ---- shell -------------------------------------------------------------------

def load_model(path: Path, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype or torch.bfloat16)
    return model, tok

def flores_bpc(model, tok, flores: Path, sentences: int, device: str) -> list[dict]:
    """Bits per character on FLORES devtest (BOS + sentence, summed
    next-token NLL / characters): comparable across tokenizers."""
    import torch

    rows = []
    for lang, code in CODES.items():
        lines = (flores / "devtest" / f"{code}.devtest").read_text(encoding="utf-8").splitlines()[:sentences]
        nll, chars, tokens = 0.0, 0, 0
        with torch.no_grad():
            for s in lines:
                ids = torch.tensor([[tok.bos_token_id, *tok.encode(s, add_special_tokens=False)]], device=device)
                logits = model(input_ids=ids).logits[0, :-1].float()
                nll += float(torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="sum"))
                chars += len(s)
                tokens += ids.shape[1] - 1
        rows.append({"lang": lang, "sentences": len(lines), "chars": chars, "tokens": tokens,
                     "nll_nats": nll, "bpc": bits_per_char(nll, chars), "tokens_per_char": tokens / chars})
    return rows

def stage_expand(args) -> int:
    import shutil

    model, _ = load_model(args.base)
    new = json.loads((args.vx / "new_tokens.json").read_text(encoding="utf-8"))
    orig_vocab = json.loads((args.base / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    parts = decompose(new["tokens"], orig_vocab)
    inv = {i: s for s, i in orig_vocab.items()}
    bd = byte_decoder()
    weights = char_lengths(parts, lambda i: bytes(bd[c] for c in inv[i]))
    model = expand_model(model, parts, weights)
    model.save_pretrained(args.out, safe_serialization=True)
    for f in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "new_tokens.json"]:
        shutil.copy(args.vx / f, args.out / f)
    print(f"expanded to {model.config.vocab_size} tokens -> {args.out}", file=sys.stderr)
    return 0

def stage_bpc(args) -> int:
    model, tok = load_model(args.model)
    model = model.to(args.device).eval()
    rows = flores_bpc(model, tok, args.flores, args.sentences, args.device)
    for r in rows:
        print(f"{r['lang']}: BPC {r['bpc']:.4f} ({r['tokens']} tokens)", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": str(args.model), "rows": rows}, indent=1) + "\n")
    return 0

def stage_train(args) -> int:
    import torch

    torch.manual_seed(args.seed)
    # Trainable tensors are fp32 master weights (an lr of 5e-5 is below
    # bf16's resolution, so the updates would round away); the frozen
    # layers stay bf16 to fit an 80 GB GPU. Autocast runs matmuls in bf16.
    model, tok = load_model(args.model, torch.bfloat16)
    # Without new_tokens.json this is the control run: the original model,
    # same recipe, every vocabulary row frozen.
    nt = args.model / "new_tokens.json"
    first_new = json.loads(nt.read_text())["first_id"] if nt.exists() else model.config.vocab_size
    n_layers = model.config.num_hidden_layers
    layers = [*range(args.edge_layers), *range(n_layers - args.edge_layers, n_layers)]
    for m in [model.get_input_embeddings(), model.get_output_embeddings(), *(model.model.layers[i] for i in layers)]:
        m.float()
    model = model.to(args.device)
    amp = lambda: torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()  # checkpointing only applies in training mode
    params = set_trainable(model, first_new, layers)
    print(f"trainable: {sum(p.numel() for p in params) / 1e6:.0f}M (new rows masked to {model.config.vocab_size - first_new}), layers {layers}", file=sys.stderr)

    texts = {l: (args.data / "wiki" / f"{l}.txt").read_text(encoding="utf-8")[: args.chars_per_lang] for l in CODES}
    if args.careful_wiki:
        texts = {l: careful_text(t, l) for l, t in texts.items()}
    per_lang = {l: chunks(tok.encode(t, add_special_tokens=False), args.seq, tok.bos_token_id) for l, t in texts.items()}
    shares = {"yo": 0.3, "ha": 0.3, "ig": 0.3, "en": 0.1}
    if args.chat:
        convs = [json.loads(l)["messages"] for l in args.chat.read_text(encoding="utf-8").splitlines()]
        stream = [i for c in convs for i in tok.apply_chat_template(c, tokenize=True, return_dict=False)]
        per_lang["chat"] = windows(stream, args.seq)
        text_share = 1.0 - args.chat_share
        shares = {**{l: v * text_share for l, v in shares.items()}, "chat": args.chat_share}
        print(f"chat: {len(convs)} conversations, {len(per_lang['chat'])} windows", file=sys.stderr)
    teacher, anchor = None, []
    if args.kl_weight > 0:
        teacher, ttok = load_model(args.teacher)
        teacher = teacher.to(args.device).eval().requires_grad_(False)
        convs = [json.loads(l)["messages"] for l in args.chat.read_text(encoding="utf-8").splitlines()]
        anchor = windows([i for c in convs for i in ttok.apply_chat_template(c, tokenize=True, return_dict=False)], args.seq)
        print(f"KL anchor: weight {args.kl_weight}, {len(anchor)} windows tokenised the original way", file=sys.stderr)
    per_step = args.batch * args.accum
    steps = args.tokens // (args.seq * per_step)
    data = mixture(per_lang, shares, steps * per_step, args.seed)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")
    t0 = time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, max(1, steps // 30), steps)
        kl = torch.zeros(())
        for k in range(args.accum):
            lo = (step * args.accum + k) * args.batch
            batch = torch.tensor(data[lo:lo + args.batch], device=args.device)
            with amp():
                loss = lm_loss(model, batch)
                total = loss
                if teacher is not None:
                    j = (step * args.accum + k) * args.kl_batch
                    a = torch.tensor([anchor[(j + i) % len(anchor)] for i in range(args.kl_batch)], device=args.device)
                    with torch.no_grad():
                        t_logits = teacher(input_ids=a).logits
                    kl = anchor_kl(model(input_ids=a).logits, t_logits, teacher.config.vocab_size)
                    total = loss + args.kl_weight * kl
            (total / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.report == 0 or step + 1 == steps:
            rec = {"step": step + 1, "tokens": (step + 1) * per_step * args.seq, "loss": loss.item(), "kl": kl.item(),
                   "tok_s": (step + 1) * per_step * args.seq / (time.time() - t0), "lr": opt.param_groups[0]["lr"]}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {rec['step']}/{steps} loss {rec['loss']:.3f} kl {rec['kl']:.4f} {rec['tok_s']:.0f} tok/s", file=sys.stderr)
        if args.eval_every and (step + 1) % args.eval_every == 0 and step + 1 < steps:
            model.eval()
            with amp():
                bpc = {r["lang"]: r["bpc"] for r in flores_bpc(model, tok, args.flores, args.eval_sentences, args.device)}
            model.train()
            log.write(json.dumps({"step": step + 1, "bpc": bpc}) + "\n")
            log.flush()
            print(f"step {step + 1}: BPC " + " ".join(f"{k} {v:.3f}" for k, v in bpc.items()), file=sys.stderr)
    model.config.use_cache = True
    model.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    if nt.exists():
        (args.out / "new_tokens.json").write_text(nt.read_text())
    print(f"saved {args.out}", file=sys.stderr)
    return 0

NAMES = {"yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}


def eos_ids(tok, model) -> set[int]:
    g = model.generation_config.eos_token_id
    return {i for i in [tok.eos_token_id, *(g if isinstance(g, list) else [g])] if i is not None}


def stage_distill(args) -> int:
    """The original model's own greedy replies to translation requests
    (English <-> yo/ha/ig) on FLORES *dev* sentences; devtest stays for
    evaluation. Training the expanded model on these re-tokenised replies
    teaches it to produce the original's answers, and to end them, with
    the new tokens."""
    import torch

    model, tok = load_model(args.model)
    model = model.to(args.device).eval()
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    eos = eos_ids(tok, model)
    if args.wiki:  # Wikipedia sentences; a different sample of English per target language
        src = lambda lang, k: wiki_sentences((args.wiki / f"{lang}.txt").read_text(encoding="utf-8"), args.n, seed=k, drop_english=lang != "en")
    else:  # FLORES dev
        src = lambda lang, k: (args.flores / "dev" / f"{CODES[lang]}.dev").read_text(encoding="utf-8").splitlines()[: args.n]
    prompts = []
    for k, (lang, name) in enumerate(NAMES.items()):
        prompts += translation_prompts(src("en", k), name) + reverse_prompts(src(lang, 10 + k), name)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for b in range(0, len(prompts), args.batch):
            chunk = prompts[b:b + args.batch]
            enc = tok.apply_chat_template([[{"role": "user", "content": p}] for p in chunk], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True, padding=True).to(args.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for p, g in zip(chunk, out[:, enc["input_ids"].shape[1]:].tolist()):
                reply = tok.decode(g, skip_special_tokens=True).strip()
                stopped = any(i in eos for i in g)
                if stopped and reply:
                    f.write(json.dumps({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": reply}]}, ensure_ascii=False) + "\n")
            print(f"{b + len(chunk)}/{len(prompts)}", file=sys.stderr)
    return 0


def generation_stats(gen: Sequence[int], canonical: int, first_new: int, eos: set[int]) -> dict:
    """What one generated reply says about the expansion: did it stop, how
    many of its tokens are new, and how its length compares with the
    tokenizer's own encoding of the same text (1.0 = the model writes with
    the tokens the tokenizer would choose; >1 = it spells words out with
    old, shorter pieces and loses the speed-up)."""
    body = [i for i in gen if i not in eos]
    return {"stopped": any(i in eos for i in gen), "tokens": len(body),
            "new_tokens": sum(1 for i in body if i >= first_new),
            "canonical_tokens": canonical, "vs_canonical": len(body) / max(1, canonical)}


NTREX = {"yo": "yor", "ha": "hau", "ig": "ibo"}


def test_set(kind: str, root: Path, lang: str, n: int) -> tuple[list[str], list[str]]:
    """English sources and `lang` references: FLORES-200 devtest, or NTREX-128
    (news; never seen while WazoCore was developed)."""
    if kind == "ntrex":
        src = (root / "newstest2019-src.eng.txt").read_text(encoding="utf-8").splitlines()
        ref = (root / f"newstest2019-ref.{NTREX[lang]}.txt").read_text(encoding="utf-8").splitlines()
    else:
        src = (root / "devtest" / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
        ref = (root / "devtest" / f"{CODES[lang]}.devtest").read_text(encoding="utf-8").splitlines()
    return src[:n], ref[:n]


def stage_sample(args) -> int:
    """Greedy chat translations of English test sentences into yo/ha/ig,
    in batches, with `generation_stats` for each."""
    import torch

    model, tok = load_model(args.model)
    model = model.to(args.device).eval()
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    first_new = 128256
    eos = eos_ids(tok, model)
    kind, root = ("ntrex", args.ntrex) if args.ntrex else ("flores", args.flores)
    rows = []
    for lang, name in NAMES.items():
        src, refs = test_set(kind, root, lang, args.n)
        for b in range(0, len(src), args.batch):
            chunk = src[b:b + args.batch]
            msgs = [[{"role": "user", "content": f"Translate the following English text into {name}. Reply with the translation only.\n\n{x}"}] for x in chunk]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True).to(args.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, g in enumerate(out[:, enc["input_ids"].shape[1]:].tolist()):
                k = b + j
                text = tok.decode(g, skip_special_tokens=True).strip()
                stats = generation_stats(g, len(tok.encode(text, add_special_tokens=False)), first_new, eos)
                rows.append({"lang": lang, "i": k, "output": text, "reference": refs[k], **stats})
            print(f"{lang}: {min(b + args.batch, len(src))}/{len(src)}", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": str(args.model), "test": kind, "rows": rows}, indent=1, ensure_ascii=False) + "\n")
    return 0


def score_rows(rows: Sequence[dict], chrf) -> dict:
    """Per-language gate numbers from `sample` rows: chrF++ against the
    FLORES references (`chrf(outputs, references)`), stop count, tokens
    and characters of the replies."""
    out = {}
    for lang in NAMES:
        rs = [r for r in rows if r["lang"] == lang]
        out[lang] = {"n": len(rs), "chrf++": chrf([r["output"] for r in rs], [r["reference"] for r in rs]),
                     "stopped": sum(r["stopped"] for r in rs), "tokens": sum(r["tokens"] for r in rs),
                     "chars": sum(len(r["output"]) for r in rs), "new_tokens": sum(r["new_tokens"] for r in rs),
                     "invalid_utf8": sum("\ufffd" in r["output"] for r in rs)}
    return out


def stage_score(args) -> int:
    import sacrebleu

    chrf = lambda hyp, ref: sacrebleu.corpus_chrf(list(hyp), [list(ref)], word_order=2).score
    report = {}
    for path in args.samples:
        report[path.stem] = score_rows(json.loads(path.read_text(encoding="utf-8"))["rows"], chrf)
        for lang, r in report[path.stem].items():
            print(f"{path.stem} {lang}: chrF++ {r['chrf++']:.2f} stop {r['stopped']}/{r['n']} chars/token {r['chars'] / max(1, r['tokens']):.2f} invalid {r['invalid_utf8']}", file=sys.stderr)
    if len(args.samples) == 2:  # paired comparison: second vs first
        base, new = (json.loads(p.read_text(encoding="utf-8"))["rows"] for p in args.samples)
        report["paired"] = {}
        for lang in NAMES:
            a = sorted((r for r in base if r["lang"] == lang), key=lambda r: r["i"])
            b = sorted((r for r in new if r["lang"] == lang), key=lambda r: r["i"])
            if not a or [r["i"] for r in a] != [r["i"] for r in b]:
                continue
            report["paired"][lang] = paired_bootstrap([r["output"] for r in a], [r["output"] for r in b], [r["reference"] for r in a], chrf)
            r = report["paired"][lang]
            print(f"{lang}: delta {r['delta']:+.2f} chrF++, 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}], P(worse) {r['p_worse']:.2f}", file=sys.stderr)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--device", default="cuda")
    sub = ap.add_subparsers(dest="stage", required=True)
    e = sub.add_parser("expand")
    e.add_argument("--base", type=Path, required=True)
    e.add_argument("--vx", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("bpc")
    b.add_argument("--model", type=Path, required=True)
    b.add_argument("--flores", type=Path, required=True)
    b.add_argument("--sentences", type=int, default=300)
    b.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train")
    t.add_argument("--model", type=Path, required=True)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--tokens", type=int, default=5_000_000)
    t.add_argument("--seq", type=int, default=512)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--clip", type=float, default=1.0)
    t.add_argument("--edge-layers", type=int, default=2)
    t.add_argument("--chars-per-lang", type=int, default=12_000_000)
    t.add_argument("--report", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--flores", type=Path, default=Path("work/data/flores200_dataset"))
    t.add_argument("--chat", type=Path, help="distilled conversations (JSONL from the distill stage)")
    t.add_argument("--chat-share", type=float, default=0.4)
    t.add_argument("--eval-every", type=int, default=100, help="steps between BPC checks (0: off)")
    t.add_argument("--eval-sentences", type=int, default=50)
    t.add_argument("--accum", type=int, default=1, help="micro-batches per optimiser step")
    t.add_argument("--careful-wiki", action="store_true", help="keep only carefully spelled Wikipedia lines")
    t.add_argument("--teacher", type=Path, help="original model for the KL anchor")
    t.add_argument("--kl-weight", type=float, default=0.0)
    t.add_argument("--kl-batch", type=int, default=4, help="anchor windows per micro-batch")
    d = sub.add_parser("distill")
    d.add_argument("--model", type=Path, required=True)
    d.add_argument("--flores", type=Path, required=True)
    d.add_argument("--n", type=int, default=997, help="source sentences per language and direction")
    d.add_argument("--wiki", type=Path, help="take sources from WIKI/{en,yo,ha,ig}.txt instead of FLORES dev")
    d.add_argument("--batch", type=int, default=64)
    d.add_argument("--max-tokens", type=int, default=256)
    d.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sample")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--flores", type=Path, required=True)
    s.add_argument("--ntrex", type=Path, help="NTREX-128 folder: test on it instead of FLORES")
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--n", type=int, default=5, help="sentences per language")
    s.add_argument("--max-tokens", type=int, default=160)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("score")
    c.add_argument("samples", type=Path, nargs="+", help="JSON from the sample stage")
    c.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    stages = {"score": stage_score, "distill": stage_distill, "expand": stage_expand, "bpc": stage_bpc, "train": stage_train, "sample": stage_sample}
    return stages[args.stage](args)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
