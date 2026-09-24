"""TEAL activation sparsity for N-ATLaS: per-language calibration, the
calibration x evaluation matrix behind claims C1/C2, and activation traces
for atlas-sim (PROJECT-PLAN-B §6, P1).

Sparsity is applied where the `.satl` kernels read it: the inputs of
`qkv` (q/k/v share one mask), `o`, `gate_up` (gate/up share one mask) and
`down`. A channel is kept when |x| > t for its (layer, site) threshold t,
the `(1 - p)` quantile of |x| on calibration text for target sparsity p.

Stages (run on a GPU with >= 24 GB for the bf16 8B model):

    python -m atlas_trace.teal calibrate --out work/profiles
    python -m atlas_trace.teal evaluate  --profiles work/profiles --out work/eval.json
    python -m atlas_trace.teal traces    --profile work/profiles/yo-50.json --lang yo \\
                                         --split devtest --tokens 20480 --out work/traces/yo-yo50-devtest.atr
    python -m atlas_trace.teal translate --profiles work/profiles --out work/chrf.json   # optional

Colab: `pip install torch transformers sacrebleu numpy`, `huggingface-cli
login` (gated model), upload this file, run the stages with
`--model NCAIR1/N-ATLaS`, then copy `work/` back into the repo.
Every stage also runs on CPU against a local tiny model (see tests).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

SITES = ("qkv", "o", "gate_up", "down")
TRACE_SITES = ("gate_up", "down")  # the streamable MLP bundles
LANGS = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"

# Histogram of log10|x|: 0.001-decade bins (0.23% relative resolution).
LOG_LO, LOG_HI, BINS = -6.0, 3.0, 9000


# ---- pure core ---------------------------------------------------------------


def threshold_from_hist(counts: np.ndarray, sparsity: float) -> float:
    """|x| value below which a fraction `sparsity` of the mass lies, by
    linear interpolation inside the bin that crosses the quantile."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity {sparsity} outside [0, 1)")
    if sparsity == 0.0:
        return 0.0
    total = counts.sum()
    cdf = np.cumsum(counts)
    target = sparsity * total
    i = int(np.searchsorted(cdf, target))
    below = cdf[i - 1] if i > 0 else 0.0
    frac = (target - below) / counts[i] if counts[i] > 0 else 0.0
    width = (LOG_HI - LOG_LO) / BINS
    return float(10 ** (LOG_LO + (i + frac) * width))


def profile(hists: dict[tuple[int, str], np.ndarray], layers: int, sparsity: float) -> list[list[float]]:
    """thresholds[layer][site] for one target sparsity."""
    return [[threshold_from_hist(hists[(l, s)], sparsity) for s in SITES] for l in range(layers)]


def chunks(ids: Sequence[int], n_ctx: int, max_chunks: int, bos: int) -> list[list[int]]:
    """Consecutive `n_ctx`-token windows of `ids`, each starting with BOS."""
    body = n_ctx - 1
    return [[bos, *ids[i * body:(i + 1) * body]] for i in range(min(max_chunks, len(ids) // body))]


def pack_masks(masks: np.ndarray) -> bytes:
    """Bool masks (..., channels) -> trace bitmaps: channel c is bit c % 64
    of little-endian u64 word c / 64 (docs/trace-format.md)."""
    if masks.shape[-1] % 64:
        raise ValueError("trace channels must be a multiple of 64")
    return np.packbits(masks, axis=-1, bitorder="little").tobytes()


def trace_payload(record: dict[tuple[int, str], np.ndarray], layers: int) -> bytes:
    """Token-major trace bytes for one chunk: for each token, for each layer,
    the packed masks of `TRACE_SITES` in order. `record[(l, s)]` is (T, C)."""
    per_layer = [
        np.concatenate([np.frombuffer(pack_masks(record[(l, s)]), np.uint8).reshape(record[(l, s)].shape[0], -1)
                        for s in TRACE_SITES], axis=1)
        for l in range(layers)
    ]
    return np.stack(per_layer, axis=1).tobytes()


def atr_prefix(header: dict) -> bytes:
    """`ATRC`, u32 header length, JSON header space-padded to 8 bytes."""
    raw = json.dumps(header).encode()
    pad = (-(8 + len(raw))) % 8
    return b"ATRC" + (len(raw) + pad).to_bytes(4, "little") + raw + b" " * pad


def bootstrap_ci(xs: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), means[int(0.025 * n)], means[int(0.975 * n) - 1]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---- instrumentation (the only mutable state, owned by one Control) ----------


@dataclass
class Control:
    """What the hooks do on the next forward pass.

    mode: "off" (dense), "calibrate" (accumulate |x| histograms) or
    "sparsify" (mask with `thresholds`, count kept channels, optionally
    record trace masks)."""

    mode: str = "off"
    thresholds: list[list[float]] | None = None
    hists: dict | None = None
    kept: dict | None = None  # (layer, site) -> [kept, total]
    record: dict | None = None  # (layer, site) -> list of bool arrays


def attach(model, ctl: Control) -> list:
    """Registers forward pre-hooks on every sparsified projection. Sites
    shared by several projections are counted on the first one only."""
    import torch

    def hook(layer: int, site: str, count: bool):
        def pre(_module, args):
            (x,) = args
            if ctl.mode == "calibrate" and count:
                logs = x.detach().float().abs().clamp_min(10**LOG_LO).log10()
                h = torch.histc(logs, bins=BINS, min=LOG_LO, max=LOG_HI)
                key = (layer, site)
                ctl.hists[key] = ctl.hists.get(key, 0) + h.cpu().numpy()
                return None
            if ctl.mode == "sparsify":
                keep = x.abs() > ctl.thresholds[layer][SITES.index(site)]
                if count:
                    k = ctl.kept.setdefault((layer, site), [0, 0])
                    k[0] += int(keep.sum())
                    k[1] += keep.numel()
                    if ctl.record is not None and site in TRACE_SITES:
                        ctl.record.setdefault((layer, site), []).append(keep.reshape(-1, keep.shape[-1]).cpu().numpy())
                return (x * keep,)
            return None

        return pre

    handles = []
    for l, block in enumerate(model.model.layers):
        a, m = block.self_attn, block.mlp
        for module, site, count in [
            (a.q_proj, "qkv", True), (a.k_proj, "qkv", False), (a.v_proj, "qkv", False),
            (a.o_proj, "o", True),
            (m.gate_proj, "gate_up", True), (m.up_proj, "gate_up", False),
            (m.down_proj, "down", True),
        ]:
            handles.append(module.register_forward_pre_hook(hook(l, site, count)))
    return handles


# ---- shell: data, model, stages ----------------------------------------------


def fetch_flores(root: Path) -> Path:
    base = root / "flores200_dataset"
    if not base.exists():
        root.mkdir(parents=True, exist_ok=True)
        tar = root / "flores200_dataset.tar.gz"
        urllib.request.urlretrieve(FLORES_URL, tar)
        with tarfile.open(tar) as t:
            t.extractall(root, filter="data")
    return base


def flores_text(base: Path, lang: str, split: str) -> str:
    return (base / split / f"{LANGS[lang]}.{split}").read_text(encoding="utf-8")


def load(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device).eval()
    return tok, model


def encode(tok, text: str) -> list[int]:
    return tok(text, add_special_tokens=False)["input_ids"]


def forward_logprobs(model, ids: list[int], device: str):
    import torch

    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0].float()
    return logits.log_softmax(-1)


def calib_ids(tok, base: Path, lang: str, budget: int) -> tuple[list[int], str]:
    """Calibration token ids for one language, or `pooled` (equal share of
    each language), always from the FLORES dev split."""
    if lang == "pooled":
        parts = [calib_ids(tok, base, l, budget // len(LANGS)) for l in LANGS]
        return [i for ids, _ in parts for i in ids], sha256_text("".join(h for _, h in parts))
    text = flores_text(base, lang, "dev")
    return encode(tok, text)[:budget], sha256_text(text)


def stage_calibrate(args) -> int:
    tok, model = load(args.model, args.device)
    ctl = Control()
    attach(model, ctl)
    base = fetch_flores(args.data)
    layers = len(model.model.layers)
    args.out.mkdir(parents=True, exist_ok=True)
    for lang in args.langs:
        ids, digest = calib_ids(tok, base, lang, args.calib_tokens)
        ctl.mode, ctl.hists = "calibrate", {}
        for chunk in chunks(ids, args.ctx, 10**9, tok.bos_token_id):
            forward_logprobs(model, chunk, args.device)
        ctl.mode = "off"
        for p in args.sparsities:
            doc = {
                "model": args.model, "calibration": {"lang": lang, "split": "dev", "tokens": len(ids), "text_sha256": digest},
                "sparsity": p, "sites": list(SITES), "thresholds": profile(ctl.hists, layers, p),
            }
            path = args.out / f"{lang}-{round(p * 100):02d}.json"
            path.write_text(json.dumps(doc, indent=1) + "\n")
            print(f"wrote {path}", file=sys.stderr)
    return 0


def load_profiles(folder: Path) -> list[dict]:
    return [json.loads(p.read_text()) | {"name": p.stem} for p in sorted(folder.glob("*.json"))]


def site_sparsity(kept: dict[str, list[int]]) -> dict[str, float]:
    """Fraction of channels zeroed per site, pooled over layers
    (`kept` keys are "layer.site", values [kept, total])."""
    def pooled(site: str) -> float:
        rows = [v for k, v in kept.items() if k.split(".", 1)[1] == site]
        return 1 - sum(r[0] for r in rows) / max(1, sum(r[1] for r in rows))

    return {s: pooled(s) for s in SITES}


def stage_evaluate(args) -> int:
    import torch

    tok, model = load(args.model, args.device)
    ctl = Control()
    attach(model, ctl)
    base = fetch_flores(args.data)
    profiles = load_profiles(args.profiles)
    results = []
    for lang in args.eval_langs:
        ids = encode(tok, flores_text(base, lang, "devtest"))
        per = {p["name"]: {"kl": [], "top1": [], "dnll": [], "kept": {}} for p in profiles}
        for chunk in chunks(ids, args.ctx, args.chunks, tok.bos_token_id):
            ctl.mode = "off"
            dense = forward_logprobs(model, chunk, args.device)
            targets = torch.tensor(chunk[1:], device=dense.device)
            nll_dense = -dense[:-1].gather(1, targets[:, None])[:, 0]
            for p in profiles:
                ctl.mode, ctl.thresholds, ctl.kept = "sparsify", p["thresholds"], {}
                sparse = forward_logprobs(model, chunk, args.device)
                kl = (dense.exp() * (dense - sparse)).sum(-1)
                nll_sparse = -sparse[:-1].gather(1, targets[:, None])[:, 0]
                r = per[p["name"]]
                r["kl"].append(float(kl.mean()))
                r["top1"].append(float((dense.argmax(-1) == sparse.argmax(-1)).float().mean()))
                r["dnll"].append(float((nll_sparse - nll_dense).mean()))
                for key, (k, n) in ctl.kept.items():
                    acc = r["kept"].setdefault(f"{key[0]}.{key[1]}", [0, 0])
                    acc[0] += k
                    acc[1] += n
        ctl.mode = "off"
        for p in profiles:
            r = per[p["name"]]
            by_site = site_sparsity(r["kept"])
            results.append({
                "eval_lang": lang, "profile": p["name"], "calib_lang": p["calibration"]["lang"], "target_sparsity": p["sparsity"],
                "effective_sparsity": by_site, "kl": bootstrap_ci(r["kl"]), "top1": sum(r["top1"]) / len(r["top1"]),
                "delta_nll": bootstrap_ci(r["dnll"]), "chunks": len(r["kl"]),
            })
            e = results[-1]
            print(f"{lang} <- {p['name']}: eff. sparsity {by_site['down']:.3f} (down), KL {e['kl'][0]:.4f}, "
                  f"top-1 {e['top1']:.3f}", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": args.model, "ctx": args.ctx, "results": results}, indent=1) + "\n")
    return 0


def stage_traces(args) -> int:
    tok, model = load(args.model, args.device)
    ctl = Control()
    attach(model, ctl)
    base = fetch_flores(args.data)
    prof = json.loads(args.profile.read_text())
    ids = encode(tok, flores_text(base, args.lang, args.split))
    todo = chunks(ids, args.ctx, args.tokens // args.ctx, tok.bos_token_id)
    cfg = model.config
    bytes_per_channel = {"gate_up": 2 * cfg.intermediate_size * 9 // 16, "down": cfg.hidden_size * 9 // 16}
    header = {
        "format": "atlas-trace", "version": 1, "model": args.model, "lang": args.lang,
        "source": f"FLORES-200 {args.split}", "profile": args.profile.stem,
        "layers": cfg.num_hidden_layers, "tokens": len(todo) * args.ctx,
        "matrices": [
            {"name": "gate_up", "channels": cfg.hidden_size, "bytes_per_channel": bytes_per_channel["gate_up"]},
            {"name": "down", "channels": cfg.intermediate_size, "bytes_per_channel": bytes_per_channel["down"]},
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        f.write(atr_prefix(header))
        for chunk in todo:
            ctl.mode, ctl.thresholds, ctl.kept, ctl.record = "sparsify", prof["thresholds"], {}, {}
            forward_logprobs(model, chunk, args.device)
            f.write(trace_payload({k: v[0] for k, v in ctl.record.items()}, cfg.num_hidden_layers))
    ctl.mode, ctl.record = "off", None
    print(f"wrote {args.out} ({header['tokens']} tokens)", file=sys.stderr)
    return 0


def stage_translate(args) -> int:
    import sacrebleu
    import torch

    tok, model = load(args.model, args.device)
    ctl = Control()
    attach(model, ctl)
    base = fetch_flores(args.data)
    names = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}
    src = flores_text(base, "en", "devtest").splitlines()[: args.n]
    configs = [None, *load_profiles(args.profiles)]
    out = []
    for lang in [l for l in args.eval_langs if l != "en"]:
        refs = flores_text(base, lang, "devtest").splitlines()[: args.n]
        for p in configs:
            ctl.mode, ctl.thresholds, ctl.kept = ("off", None, None) if p is None else ("sparsify", p["thresholds"], {})
            hyps = []
            for s in src:
                msgs = [{"role": "user", "content": f"Translate the following English text into {names[lang]}. "
                                                    f"Reply with the translation only.\n\n{s}"}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(args.device)
                with torch.no_grad():
                    gen = model.generate(ids, max_new_tokens=160, do_sample=False)
                hyps.append(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True).strip())
            chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
            out.append({"lang": lang, "profile": "dense" if p is None else p["name"], "chrf++": chrf})
            print(f"en->{lang} {out[-1]['profile']}: chrF++ {chrf:.2f}", file=sys.stderr)
    ctl.mode = "off"
    args.out.write_text(json.dumps({"model": args.model, "n": args.n, "results": out}, indent=1) + "\n")
    return 0


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="NCAIR1/N-ATLaS")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--ctx", type=int, default=512)
    sub = ap.add_subparsers(dest="stage", required=True)

    c = sub.add_parser("calibrate")
    c.add_argument("--langs", nargs="+", default=[*LANGS, "pooled"])
    c.add_argument("--sparsities", nargs="+", type=float, default=[0.25, 0.40, 0.50])
    c.add_argument("--calib-tokens", type=int, default=24_576)
    c.add_argument("--out", type=Path, required=True)

    e = sub.add_parser("evaluate")
    e.add_argument("--profiles", type=Path, required=True)
    e.add_argument("--eval-langs", nargs="+", default=list(LANGS))
    e.add_argument("--chunks", type=int, default=40)
    e.add_argument("--out", type=Path, required=True)

    t = sub.add_parser("traces")
    t.add_argument("--profile", type=Path, required=True)
    t.add_argument("--lang", choices=list(LANGS), required=True)
    t.add_argument("--split", choices=["dev", "devtest"], default="devtest")
    t.add_argument("--tokens", type=int, default=20_480)
    t.add_argument("--out", type=Path, required=True)

    r = sub.add_parser("translate")
    r.add_argument("--profiles", type=Path, required=True)
    r.add_argument("--eval-langs", nargs="+", default=list(LANGS))
    r.add_argument("--n", type=int, default=50)
    r.add_argument("--out", type=Path, required=True)

    args = ap.parse_args(argv)
    stages = {"calibrate": stage_calibrate, "evaluate": stage_evaluate, "traces": stage_traces, "translate": stage_translate}
    return stages[args.stage](args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
