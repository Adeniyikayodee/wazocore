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

from wazocore.vx_core import ENGLISH_MARKERS, bits_per_char, byte_decoder, careful_text, char_lengths, chunk_align, chunks, decompose, init_rows, looping, lr_at, mbr_pick, merge_table, mixture, nfc_rows, norm_calibrated, sign_test, smol_pairs

CODES = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}

# ---- model surgery (torch, but pure over its inputs) --------------------------

def expand_model(model, parts: dict[int, list[int]], out_weights: dict[int, list[float]], calibrate: bool = False):
    """Appends one input and one output row per new token (ids in order),
    initialised from the token's pieces. Returns the same model object,
    resized; original rows are copied unchanged. `calibrate` rescales the
    new input rows to the median norm of the pieces they are made of
    (output rows are left alone: calibrating them harms, arXiv 2608.03494)."""
    import numpy as np
    import torch

    ids = sorted(parts)
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    head = model.get_output_embeddings().weight.detach().float().cpu().numpy()
    new_in = init_rows(emb, parts)
    if calibrate:
        pieces = sorted({p for ps in parts.values() for p in ps})
        new_in = norm_calibrated(new_in, float(np.median(np.linalg.norm(emb[pieces], axis=1))))
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


def pooled_bootstrap(base, runs: Sequence, strata: Sequence[tuple[int, int]], score, margin: float, rounds: int = 5000, seed: int = 0) -> dict:
    """Score difference from `base`, averaged over training `runs`, on
    several test sets pooled. `base` and each run are [sentences, k] arrays
    of per-sentence statistics that `score` reduces after summing over
    sentences; sentences are resampled within each test set (`strata`,
    (start, end) rows). `lower95` is the one-sided 95% bound: the model is
    non-inferior at `margin` only if it lies above -margin. A single test
    set can fail to detect a loss that the pooled sentences show."""
    import numpy as np

    rng = np.random.default_rng(seed)
    delta = lambda idx: sum(score(r[idx]) - score(base[idx]) for r in runs) / len(runs)
    diffs = np.sort([delta(np.concatenate([lo + rng.integers(0, hi - lo, hi - lo) for lo, hi in strata])) for _ in range(rounds)])
    lower = float(diffs[int(0.05 * rounds)])
    return {"n": len(base), "runs": len(runs), "delta": float(delta(np.arange(len(base)))),
            "ci95": [float(diffs[int(0.025 * rounds)]), float(diffs[int(0.975 * rounds) - 1])],
            "lower95": lower, "margin": margin, "non_inferior": lower > -margin, "p_not_worse": float((diffs >= 0).mean())}


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


def forbidden_pairs(prev, offsets, rights):
    """For each position, the tokens that may not follow its previous token
    (`merge_table`), as flat (position, token) index tensors."""
    import torch

    counts = offsets[prev + 1] - offsets[prev]
    pos = torch.repeat_interleave(torch.arange(len(prev), device=prev.device), counts)
    within = torch.arange(len(pos), device=prev.device) - torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    return pos, rights[offsets[prev][pos] + within]


def canon_penalty(logits, prev, label, offsets, rights):
    """Mean -log(1 - mass on forbidden tokens): unlikelihood (arXiv
    1908.04319) on the piece spelling of a word that has a token of its
    own. `logits` [P, V], `prev` and `label` [P]. Positions whose own label
    is forbidden (pre-tokenizer boundaries, about 1 in 100,000) are skipped."""
    import torch

    pos, ids = forbidden_pairs(prev, offsets, rights)
    zeros = torch.zeros(len(prev), device=logits.device)
    mass = zeros.index_add(0, pos, torch.softmax(logits.float(), dim=-1)[pos, ids])
    keep = zeros.index_add(0, pos, (ids == label[pos]).float()) == 0
    return (-torch.log1p(-mass.clamp(max=1 - 1e-6)) * keep).sum() / keep.sum().clamp(min=1)


def projected_kl(student_logits, teacher_logits, first_piece, prev, offsets, rights):
    """KL(teacher || student) at positions of the new tokenisation, with
    both brought to the original vocabulary. `student_logits` [P, V],
    `teacher_logits` [P, n_old] read at the same places in the text.

    The student is projected: a new token's probability counts for its
    first original piece (`first_piece`, one id per new token), because
    that is the token the teacher would write there. The teacher reads
    pieces, so after a token it still expects the piece that completes a
    merge; in the new tokenisation that pair never occurs, so those
    entries are removed from the teacher and it is renormalised. Unlike an
    anchor on the old segmentation, this cannot reward the piece spelling."""
    import torch

    n_old = teacher_logits.shape[-1]
    p = torch.softmax(student_logits.float(), dim=-1)
    proj = p[:, :n_old].index_add(1, first_piece, p[:, n_old:])
    pos, ids = forbidden_pairs(prev, offsets, rights)
    old = ids < n_old
    t = teacher_logits.float().index_put((pos[old], ids[old]), torch.tensor(float("-inf"), device=teacher_logits.device))
    t = torch.log_softmax(t, dim=-1)
    return (t.exp() * (t.clamp(min=-1e4) - torch.log(proj.clamp(min=1e-9)))).sum(-1).mean()


def piece_stream(rows: Sequence[Sequence[int]], pieces: Sequence[Sequence[int]]) -> tuple[list[list[int]], list[list[int]]]:
    """Each row of new-tokenisation ids as the original tokenisation of the
    same text (every token replaced by its pieces; the two are identical by
    construction), and for each token the index of its last piece: where the
    teacher has read exactly as much text as the student."""
    out, ends = [], []
    for row in rows:
        flat, last = [], []
        for i in row:
            flat += pieces[i]
            last.append(len(flat) - 1)
        out.append(flat)
        ends.append(last)
    return out, ends


def teacher_at(teacher, rows: Sequence[Sequence[int]], pieces, pad: int, device: str):
    """The teacher's next-token logits after each token of `rows` except the
    last ([rows, T - 1, n_old]), read on the original tokenisation."""
    import torch

    flat, ends = piece_stream(rows, pieces)
    width = max(len(f) for f in flat)
    ids = torch.tensor([f + [pad] * (width - len(f)) for f in flat], device=device)
    with torch.no_grad():
        logits = teacher(input_ids=ids).logits  # causal: right padding cannot reach earlier positions
    at = torch.tensor([e[:-1] for e in ends], device=device)
    return logits.gather(1, at[:, :, None].expand(-1, -1, logits.shape[-1]))


def canonical_processor(offsets, rights):
    """A `generate` logits processor that forbids the pairs in
    `merge_table`, so replies are written the way the tokenizer would
    write them. For measurement only: a standard GGUF cannot carry it."""
    def process(input_ids, scores):
        for row, p in enumerate(input_ids[:, -1].tolist()):
            scores[row, rights[offsets[p]:offsets[p + 1]]] = float("-inf")
        return scores

    return process


def load_merge_table(model_dir: Path, tok, device: str):
    """`merge_table` of a model directory's new tokens as tensors on `device`."""
    import torch

    new = json.loads((model_dir / "new_tokens.json").read_text(encoding="utf-8"))
    offsets, rights = merge_table(new["tokens"], tok.get_vocab(), len(tok))
    return torch.tensor(offsets, device=device), torch.tensor(rights, device=device)


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
    weights = None if args.out_weights == "mean" else char_lengths(parts, lambda i: bytes(bd[c] for c in inv[i]), args.out_weights)
    model = expand_model(model, parts, weights, args.calibrate_input_norm)
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
        print(f"KL anchor: weight {args.kl_weight}, " + ("on the training text, student projected onto the original vocabulary" if args.anchor == "canonical"
              else f"{len(anchor)} windows tokenised the original way"), file=sys.stderr)
    canonical = args.anchor == "canonical"
    table = load_merge_table(args.model, tok, args.device) if args.canon_weight > 0 or canonical else None
    if canonical:
        if teacher is None:
            raise SystemExit("--anchor canonical needs --teacher and --kl-weight > 0")
        new = json.loads(nt.read_text(encoding="utf-8"))["tokens"]
        parts = decompose(new, ttok.get_vocab())
        pieces = [parts.get(i, [i]) for i in range(len(tok))]
        first_piece = torch.tensor([parts[i][0] for i in sorted(parts)], device=args.device)
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
                if table is not None:
                    logits = model(input_ids=batch[:, :-1]).logits.float()
                    flat, prev, label = logits.reshape(-1, logits.shape[-1]), batch[:, :-1].reshape(-1), batch[:, 1:].reshape(-1)
                    loss = torch.nn.functional.cross_entropy(flat, label)
                    canon = canon_penalty(flat, prev, label, *table)
                    total = loss + args.canon_weight * canon
                else:
                    loss = lm_loss(model, batch)
                    total = loss
                if canonical:  # anchor on the same text, read by the teacher in its own tokenisation
                    rows = batch[:args.kl_batch]
                    t_logits = teacher_at(teacher, rows.tolist(), pieces, tok.eos_token_id, args.device)
                    kl = projected_kl(logits[:len(rows)].reshape(-1, logits.shape[-1]), t_logits.reshape(-1, t_logits.shape[-1]),
                                      first_piece, rows[:, :-1].reshape(-1), *table)
                    total = total + args.kl_weight * kl
                elif teacher is not None:
                    j = (step * args.accum + k) * args.kl_batch
                    a = torch.tensor([anchor[(j + i) % len(anchor)] for i in range(args.kl_batch)], device=args.device)
                    with torch.no_grad():
                        t_logits = teacher(input_ids=a).logits
                    kl = anchor_kl(model(input_ids=a).logits, t_logits, teacher.config.vocab_size)
                    total = total + args.kl_weight * kl
            (total / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % args.report == 0 or step + 1 == steps:
            rec = {"step": step + 1, "tokens": (step + 1) * per_step * args.seq, "loss": loss.item(), "kl": kl.item(), **({"canon": canon.item()} if table is not None else {}),
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


def token_bytes(tok) -> list[bytes | None]:
    """Each id's bytes in a byte-level BPE vocabulary; None for special tokens."""
    byte_of, special = byte_decoder(), set(tok.all_special_ids) | set(tok.added_tokens_decoder)
    out = []
    for i in range(len(tok)):
        piece = tok.convert_ids_to_tokens(i)
        out.append(None if i in special or piece is None or any(c not in byte_of for c in piece)
                   else bytes(byte_of[c] for c in piece))
    return out


def prepare_trainable(model, new_tokens: Path, edge_layers: int, device: str) -> list:
    """fp32 master weights for the embeddings, head and edge layers (an lr
    below bf16 resolution would round away), frozen bf16 elsewhere, original
    vocabulary rows masked. Returns the trainable parameters."""
    first_new = json.loads(new_tokens.read_text())["first_id"] if new_tokens.exists() else model.config.vocab_size
    n_layers = model.config.num_hidden_layers
    layers = [*range(edge_layers), *range(n_layers - edge_layers, n_layers)]
    for m in [model.get_input_embeddings(), model.get_output_embeddings(), *(model.model.layers[i] for i in layers)]:
        m.float()
    model.to(device)
    return set_trainable(model, first_new, layers)


def response_logprobs(model, prompt: Sequence[int], response: Sequence[int], device: str):
    """Log-probabilities the model gives each response token after the prompt."""
    import torch

    ids = torch.tensor([list(prompt) + list(response)], device=device)
    logp = torch.log_softmax(model(input_ids=ids).logits[0, len(prompt) - 1:-1].float(), dim=-1)
    return logp.gather(1, ids[0, len(prompt):, None])[:, 0]


def stage_onpolicy(args) -> int:
    """On-policy self-distillation across tokenizers (v5). The expanded model
    writes replies with its new tokens; the frozen original scores the same
    text under its own tokenisation; the two are compared chunk by chunk
    (`chunk_align`), and each chunk the original finds less likely than the
    student is pushed down (reverse KL, REINFORCE form). A KL anchor on
    old-tokenised replies keeps the rest of the model near the original."""
    import random

    import torch

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    student, stok = load_model(args.model, torch.bfloat16)
    params = prepare_trainable(student, args.model / "new_tokens.json", args.edge_layers, args.device)
    teacher, ttok = load_model(args.teacher)
    teacher = teacher.to(args.device).eval().requires_grad_(False)
    amp = lambda: torch.autocast(device_type=args.device.split(":")[0], dtype=torch.bfloat16)
    stok.padding_side = "left"
    stok.pad_token = stok.pad_token or stok.eos_token
    s_bytes, t_bytes = token_bytes(stok), token_bytes(ttok)
    s_eos, t_eot = eos_ids(stok, student), ttok.convert_tokens_to_ids("<|eot_id|>")
    first_new = json.loads((args.model / "new_tokens.json").read_text())["first_id"]
    convs = [json.loads(l)["messages"] for l in args.chat.read_text(encoding="utf-8").splitlines()]
    prompts = [c[0]["content"] for c in convs]
    rng.shuffle(prompts)
    anchor = windows([i for c in convs for i in ttok.apply_chat_template(c, tokenize=True, return_dict=False)], args.seq)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
    args.out.mkdir(parents=True, exist_ok=True)
    log = (args.out / "log.jsonl").open("a")
    print(f"on-policy: {len(prompts)} prompts, {args.steps} steps x {args.rollouts} rollouts", file=sys.stderr)
    t0 = time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args.lr, max(1, args.steps // 20), args.steps)
        batch = [prompts[(step * args.rollouts + k) % len(prompts)] for k in range(args.rollouts)]
        # 1. rollouts from the current student
        student.eval(); student.config.use_cache = True; student.gradient_checkpointing_disable()
        enc = stok.apply_chat_template([[{"role": "user", "content": p}] for p in batch], add_generation_prompt=True,
                                       return_tensors="pt", return_dict=True, padding=True).to(args.device)
        with torch.no_grad(), amp():
            out = student.generate(**enc, do_sample=True, temperature=args.temperature, top_p=1.0,
                                   max_new_tokens=args.max_new, pad_token_id=stok.pad_token_id)
        student.train(); student.config.use_cache = False; student.gradient_checkpointing_enable()
        # 2. score chunk by chunk against the original
        samples = []
        for k, p in enumerate(batch):
            gen = out[k, enc["input_ids"].shape[1]:].tolist()
            ended = next((n for n, x in enumerate(gen) if x in s_eos), None)
            content = gen[:ended] if ended is not None else gen
            if not content or any(s_bytes[x] is None for x in content):
                continue
            try:
                text = b"".join(s_bytes[x] for x in content).decode("utf-8")
            except UnicodeDecodeError:
                continue
            t_resp = ttok.encode(text, add_special_tokens=False)
            if any(t_bytes[x] is None for x in t_resp):
                continue
            chunks_ = chunk_align([len(s_bytes[x]) for x in content], [len(t_bytes[x]) for x in t_resp])
            if chunks_ is None:
                continue
            s_prompt = stok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True, return_dict=False)
            t_prompt = ttok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=True, return_dict=False)
            samples.append((s_prompt, content + ([gen[ended]] if ended is not None else []),
                            t_prompt, t_resp + ([t_eot] if ended is not None else []), chunks_, ended is not None, text, content))
        n_chunks = sum(len(x[4]) + x[5] for x in samples)
        rkl, adv_sum = 0.0, 0.0
        for s_prompt, s_resp, t_prompt, t_resp, chunks_, ended, _, _ in samples:
            with torch.no_grad(), amp():
                lp_t = response_logprobs(teacher, t_prompt, t_resp, args.device)
            with amp():
                lp_s = response_logprobs(student, s_prompt, s_resp, args.device)
            spans_s = [(a0, a1) for a0, a1, _, _ in chunks_] + ([(len(s_resp) - 1, len(s_resp))] if ended else [])
            spans_t = [(b0, b1) for _, _, b0, b1 in chunks_] + ([(len(t_resp) - 1, len(t_resp))] if ended else [])
            s_chunk = torch.stack([lp_s[a:b].sum() for a, b in spans_s])
            t_chunk = torch.stack([lp_t[a:b].sum() for a, b in spans_t]).to(s_chunk.dtype)
            adv = (t_chunk - s_chunk.detach()).clamp(-args.clip_adv, args.clip_adv)
            (-(adv * s_chunk).sum() / max(1, n_chunks)).backward()
            rkl += float((s_chunk.detach() - t_chunk).sum())
            adv_sum += float(adv.sum())
        # 3. KL anchor on replies tokenised the original way
        kl = torch.zeros(())
        if args.kl_weight > 0:
            a = torch.tensor([anchor[(step * args.kl_batch + i) % len(anchor)] for i in range(args.kl_batch)], device=args.device)
            with torch.no_grad(), amp():
                t_logits = teacher(input_ids=a).logits
            with amp():
                kl = anchor_kl(student(input_ids=a).logits, t_logits, teacher.config.vocab_size)
            (args.kl_weight * kl).backward()
        torch.nn.utils.clip_grad_norm_(params, args.clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        texts = [x[6] for x in samples]
        rec = {"step": step + 1, "rollouts": len(samples), "reverse_kl_per_chunk": rkl / max(1, n_chunks),
               "stopped": sum(x[5] for x in samples) / max(1, len(samples)), "loops": sum(looping(t) for t in texts),
               "new_token_share": sum(sum(y >= first_new for y in x[7]) for x in samples) / max(1, sum(len(x[7]) for x in samples)),
               "kl_anchor": kl.item(), "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t0)}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (step + 1) % args.report == 0 or step + 1 == args.steps:
            print(f"step {rec['step']}/{args.steps} rKL/chunk {rec['reverse_kl_per_chunk']:.3f} stop {rec['stopped']:.2f} "
                  f"loops {rec['loops']} new {rec['new_token_share']:.2f} kl {rec['kl_anchor']:.3f} {rec['seconds']}s", file=sys.stderr)
    student.config.use_cache = True
    student.to(torch.bfloat16).save_pretrained(args.out, safe_serialization=True)
    stok.save_pretrained(args.out)
    (args.out / "new_tokens.json").write_text((args.model / "new_tokens.json").read_text())
    print(f"saved {args.out}", file=sys.stderr)
    return 0


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
        prompts += translation_prompts(src("en", k), name) + ([] if args.forward_only else reverse_prompts(src(lang, 10 + k), name))
    utility = None
    if args.candidates > 1:
        import sacrebleu

        utility = lambda hyp, ref: sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score
        torch.manual_seed(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for b in range(0, len(prompts), args.batch):
            chunk = prompts[b:b + args.batch]
            enc = tok.apply_chat_template([[{"role": "user", "content": p}] for p in chunk], add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True, padding=True).to(args.device)
            # The greedy reply first, then samples (epsilon sampling, as in MBR practice).
            passes = []
            for k in range(args.candidates):
                how = {"do_sample": False} if k == 0 else {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "epsilon_cutoff": args.epsilon}
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=args.max_tokens, pad_token_id=tok.pad_token_id, **how)
                passes.append(out[:, enc["input_ids"].shape[1]:].tolist())
            for n, p in enumerate(chunk):
                replies = [tok.decode(g[n], skip_special_tokens=True).strip() for g in passes if any(i in eos for i in g[n])]
                replies = list(dict.fromkeys(r for r in replies if r and not (utility and looping(r))))  # complete, distinct, greedy first
                if replies:
                    reply = replies[mbr_pick(replies, utility)] if utility else replies[0]
                    f.write(json.dumps({"messages": [{"role": "user", "content": p}, {"role": "assistant", "content": reply}], "candidates": len(replies)}, ensure_ascii=False) + "\n")
            print(f"{b + len(chunk)}/{len(prompts)}", file=sys.stderr)
    return 0


def stage_smol(args) -> int:
    """Human translations as training conversations, both directions, in
    the prompt format of `distill`: the signal that can lift the student
    above its teacher. Needs smolsent/ and smoldoc/ from google/smol."""
    n = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for lang, name in NAMES.items():
            pairs = [pr for kind in ("smolsent", "smoldoc") for line in (args.smol / kind / f"en_{lang}.jsonl").read_text(encoding="utf-8").splitlines()
                     for pr in smol_pairs(json.loads(line)) if max(map(len, pr)) <= args.max_chars]
            en, xx = [a for a, _ in pairs], [b for _, b in pairs]
            for prompt, reply in [*zip(translation_prompts(en, name), xx), *zip(reverse_prompts(xx, name), en)]:
                f.write(json.dumps({"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": reply}]}, ensure_ascii=False) + "\n")
            n += 2 * len(pairs)
            print(f"{lang}: {len(pairs)} sentence pairs", file=sys.stderr)
    print(f"{n} conversations -> {args.out}", file=sys.stderr)
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


def test_set(kind: str, root: Path, lang: str, n: int, start: int = 0) -> tuple[list[str], list[str]]:
    """English sources and `lang` references, lines start..start+n:
    FLORES-200 devtest, or NTREX-128 (news; lines 500 on were never
    scored before v7)."""
    if kind == "ntrex":
        src = (root / "newstest2019-src.eng.txt").read_text(encoding="utf-8").splitlines()
        ref = (root / f"newstest2019-ref.{NTREX[lang]}.txt").read_text(encoding="utf-8").splitlines()
    else:
        src = (root / "devtest" / "eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
        ref = (root / "devtest" / f"{CODES[lang]}.devtest").read_text(encoding="utf-8").splitlines()
    return src[start:start + n], ref[start:start + n]


def stage_sample(args) -> int:
    """Greedy chat translations of test sentences, English into yo/ha/ig
    (or back with `--reverse`), in batches, with `generation_stats` for each."""
    import torch

    model, tok = load_model(args.model)
    model = model.to(args.device).eval()
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    first_new = 128256
    eos = eos_ids(tok, model)
    extra = {"logits_processor": [canonical_processor(*load_merge_table(args.model, tok, args.device))]} if args.canonical else {}
    kind, root = ("ntrex", args.ntrex) if args.ntrex else ("flores", args.flores)
    rows = []
    for lang, name in NAMES.items():
        src, refs = test_set(kind, root, lang, args.n, args.start)
        if args.reverse:
            src, refs = refs, src
        prompt = reverse_prompts if args.reverse else translation_prompts
        for b in range(0, len(src), args.batch):
            chunk = src[b:b + args.batch]
            msgs = [[{"role": "user", "content": p}] for p in prompt(chunk, name)]
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True).to(args.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=args.max_tokens, do_sample=False, pad_token_id=tok.pad_token_id, **extra)
            for j, g in enumerate(out[:, enc["input_ids"].shape[1]:].tolist()):
                k = b + j
                text = tok.decode(g, skip_special_tokens=True).strip()
                stats = generation_stats(g, len(tok.encode(text, add_special_tokens=False)), first_new, eos)
                rows.append({"lang": lang, "i": args.start + k, "source": src[k], "output": text, "reference": refs[k], **stats})
            print(f"{lang}: {min(b + args.batch, len(src))}/{len(src)}", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".part")  # a lost host leaves no half-written result
    tmp.write_text(json.dumps({"model": str(args.model), "test": kind, "direction": "xx-en" if args.reverse else "en-xx",
                               "rows": rows}, indent=1, ensure_ascii=False) + "\n")
    tmp.replace(args.out)
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
        report[path.stem] = score_rows(nfc_rows(json.loads(path.read_text(encoding="utf-8"))["rows"]), chrf)
        for lang, r in report[path.stem].items():
            print(f"{path.stem} {lang}: chrF++ {r['chrf++']:.2f} stop {r['stopped']}/{r['n']} chars/token {r['chars'] / max(1, r['tokens']):.2f} invalid {r['invalid_utf8']}", file=sys.stderr)
    if len(args.samples) == 2:  # paired comparison: second vs first
        base, new = (nfc_rows(json.loads(p.read_text(encoding="utf-8"))["rows"]) for p in args.samples)
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


def stage_pooled(args) -> int:
    """All test sets and training runs in one estimate per language: chrF++
    difference from the original with a non-inferiority bound, loops
    (paired), and how many more tokens each run wrote than its tokenizer
    would for the same text."""
    import numpy as np
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=2)
    score = lambda s: metric._compute_score_from_stats(list(s.sum(0))).score
    load = lambda paths, lang: [sorted((r for r in nfc_rows(json.loads(p.read_text(encoding="utf-8"))["rows"]) if r["lang"] == lang), key=lambda r: r["i"]) for p in paths]
    stats = lambda sets: np.concatenate([np.array(metric._extract_corpus_statistics([r["output"] for r in rs], [[r["reference"] for r in rs]]), dtype=np.float64) for rs in sets])
    report = {}
    for lang in NAMES:
        base, runs = load(args.base, lang), [load(run, lang) for run in args.run]
        if any([r["i"] for r in a] != [r["i"] for r in b] for run in runs for a, b in zip(base, run)):
            raise ValueError(f"{lang}: runs and base do not cover the same sentences")
        ends = np.cumsum([len(rs) for rs in base])
        r = pooled_bootstrap(stats(base), [stats(run) for run in runs], list(zip([0, *ends[:-1]], ends)), score, args.margin, args.rounds)
        flat = lambda sets: [x for rs in sets for x in rs]
        loops = []
        for run in runs:
            pairs = [(looping(a["output"]), looping(b["output"])) for a, b in zip(flat(base), flat(run))]
            only_new, only_base = sum(b and not a for a, b in pairs), sum(a and not b for a, b in pairs)
            loops.append({"base": sum(a for a, _ in pairs), "run": sum(b for _, b in pairs), "p": sign_test(only_base, only_new)})
        excess = [sum(x["tokens"] for x in flat(run)) / max(1, sum(x["canonical_tokens"] for x in flat(run))) for run in runs]
        report[lang] = {**r, "loops": loops, "tokens_vs_canonical": excess}
        print(f"{lang}: delta {r['delta']:+.2f} chrF++ on {r['n']} sentences x {r['runs']} runs, 95% [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}], "
              f"one-sided bound {r['lower95']:+.2f} ({'non-inferior' if r['non_inferior'] else 'not shown non-inferior'} at {args.margin}); "
              f"loops {[(l['base'], l['run']) for l in loops]}; tokens vs canonical {[round(e, 3) for e in excess]}", file=sys.stderr)
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
    e.add_argument("--out-weights", choices=["bytes", "chars", "mean"], default="bytes", help="length unit weighting the output-row mean (v4: bytes; mean: unweighted, as FVT)")
    e.add_argument("--calibrate-input-norm", action="store_true", help="rescale new input rows to the median norm of their pieces")
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
    t.add_argument("--anchor", choices=["old", "canonical"], default="old",
                   help="KL anchor on replies tokenised the original way (v4), or on the training text itself with the student projected onto the original vocabulary")
    t.add_argument("--canon-weight", type=float, default=0.0, help="weight of the penalty on pairs the tokenizer would have merged (v4: 0)")
    o = sub.add_parser("onpolicy", help="v5: on-policy self-distillation across tokenizers")
    o.add_argument("--model", type=Path, required=True, help="the expanded model to refine (v4)")
    o.add_argument("--teacher", type=Path, required=True, help="the original N-ATLaS")
    o.add_argument("--chat", type=Path, required=True, help="conversations: prompts for rollouts, replies for the KL anchor")
    o.add_argument("--out", type=Path, required=True)
    o.add_argument("--steps", type=int, default=300)
    o.add_argument("--rollouts", type=int, default=32)
    o.add_argument("--max-new", type=int, default=160)
    o.add_argument("--temperature", type=float, default=1.0)
    o.add_argument("--lr", type=float, default=1e-5)
    o.add_argument("--clip", type=float, default=1.0)
    o.add_argument("--clip-adv", type=float, default=5.0, help="bound on each chunk's log-ratio")
    o.add_argument("--edge-layers", type=int, default=2)
    o.add_argument("--kl-weight", type=float, default=1.0)
    o.add_argument("--kl-batch", type=int, default=2)
    o.add_argument("--seq", type=int, default=512)
    o.add_argument("--report", type=int, default=5)
    o.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("distill")
    d.add_argument("--model", type=Path, required=True)
    d.add_argument("--flores", type=Path, required=True)
    d.add_argument("--n", type=int, default=997, help="source sentences per language and direction")
    d.add_argument("--wiki", type=Path, help="take sources from WIKI/{en,yo,ha,ig}.txt instead of FLORES dev")
    d.add_argument("--batch", type=int, default=64)
    d.add_argument("--max-tokens", type=int, default=256)
    d.add_argument("--candidates", type=int, default=1, help="replies per request: the greedy one plus samples; the MBR choice is kept (v4: 1)")
    d.add_argument("--epsilon", type=float, default=0.02, help="epsilon-sampling cutoff for the sampled candidates")
    d.add_argument("--seed", type=int, default=0)
    d.add_argument("--forward-only", action="store_true", help="only English -> yo/ha/ig requests")
    d.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("smol", help="human translations (google/smol) as training conversations")
    g.add_argument("--smol", type=Path, required=True, help="folder holding smolsent/ and smoldoc/")
    g.add_argument("--max-chars", type=int, default=600)
    g.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("sample")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--flores", type=Path, required=True)
    s.add_argument("--ntrex", type=Path, help="NTREX-128 folder: test on it instead of FLORES")
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--n", type=int, default=5, help="sentences per language")
    s.add_argument("--start", type=int, default=0, help="first test line (v7 holds out NTREX lines 500 on)")
    s.add_argument("--reverse", action="store_true", help="translate yo/ha/ig into English")
    s.add_argument("--max-tokens", type=int, default=160)
    s.add_argument("--canonical", action="store_true", help="forbid pairs the tokenizer would have merged (diagnostic)")
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("score")
    c.add_argument("samples", type=Path, nargs="+", help="JSON from the sample stage")
    c.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("pooled", help="one estimate per language over all test sets and training runs")
    p.add_argument("--base", type=Path, nargs="+", required=True, help="the original's samples, one file per test set")
    p.add_argument("--run", type=Path, nargs="+", action="append", required=True, help="one training run's samples, test sets in the same order; repeat per run")
    p.add_argument("--margin", type=float, default=1.0, help="non-inferiority margin in chrF++, chosen before looking")
    p.add_argument("--rounds", type=int, default=5000)
    p.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    stages = {"smol": stage_smol, "pooled": stage_pooled, "onpolicy": stage_onpolicy, "score": stage_score, "distill": stage_distill, "expand": stage_expand, "bpc": stage_bpc, "train": stage_train, "sample": stage_sample}
    return stages[args.stage](args)

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
