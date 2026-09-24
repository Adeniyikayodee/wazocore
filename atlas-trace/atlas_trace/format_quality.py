"""Dense quality of `.satl` vs llama.cpp Q4_K_M, against a Q8_0 reference
(PROJECT-PLAN-B §3.2 "isolate variables").

All engines score the same FLORES-200 devtest tokens with llama.cpp's
perplexity protocol (chunks of n_ctx, BOS first, second half scored).
Per-chunk NLLs allow a paired bootstrap of each format's loss vs Q8_0.

Usage:
    uv run --project atlas-trace python -m atlas_trace.format_quality \
        --out results/quality/format-ppl.json --manifest results/probe/m4-macbook-dev.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

LANGS = {"en": "eng_Latn", "yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}


def chunk_nlls_from_running(running_ppl: Sequence[float], scored_per_chunk: int) -> list[float]:
    """llama-perplexity prints the running PPL after each chunk; recover
    each chunk's mean NLL from consecutive cumulative sums."""
    sums = [k * scored_per_chunk * math.log(p) for k, p in enumerate(running_ppl, start=1)]
    return [(s - prev) / scored_per_chunk for s, prev in zip(sums, [0.0, *sums[:-1]])]


def parse_llama_perplexity(output: str) -> list[float]:
    return [float(v) for v in re.findall(r"\[\d+\]([0-9.]+)", output)]


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (b − a) per chunk with a 95% bootstrap interval."""
    rng = random.Random(seed)
    d = [y - x for x, y in zip(a, b)]
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return sum(d) / len(d), means[int(0.025 * n)], means[int(0.975 * n) - 1]


# ---- imperative shell -------------------------------------------------------


def run_llama(binary: Path, model: Path, text: Path, ctx: int, chunks: int, threads: int) -> list[float]:
    cmd = [str(binary), "-m", str(model), "-f", str(text), "-c", str(ctx), "--chunks", str(chunks),
           "-b", str(ctx), "-t", str(threads), "-ngl", "0"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    running = parse_llama_perplexity(out.stdout + out.stderr)
    if len(running) != chunks:
        raise RuntimeError(f"{model.name}: parsed {len(running)} chunks, expected {chunks}")
    return chunk_nlls_from_running(running, ctx - 1 - ctx // 2)


def run_atlas(binary: Path, model: Path, tokenizer: Path, text: Path, ctx: int, chunks: int,
              threads: int, manifest: Path, out: Path) -> list[float]:
    cmd = [str(binary), "--threads", str(threads), "--manifest", str(manifest), "perplexity",
           "--model", str(model), "--tokenizer", str(tokenizer), "--text", str(text),
           "--ctx", str(ctx), "--chunks", str(chunks), "--out", str(out)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [math.log(p) for p in json.loads(out.read_text())["chunk_ppl"]]


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--flores", type=Path, default=Path("data/flores200_dataset"))
    ap.add_argument("--q8", type=Path, default=Path("data/models/N-ATLaS.Q8_0.gguf"))
    ap.add_argument("--q4km", type=Path, default=Path("data/models/N-ATLaS.Q4_K_M.gguf"))
    ap.add_argument("--satl", type=Path, default=Path("data/models/n-atlas-q4g64.satl"))
    ap.add_argument("--tokenizer", type=Path, default=Path("data/models/n-atlas-bf16/tokenizer.json"))
    ap.add_argument("--llama", type=Path, default=Path("vendor/llama.cpp/build-cpu/bin/llama-perplexity"))
    ap.add_argument("--atlas", type=Path, default=Path("target/release/atlas-rt"))
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--chunks", type=int, default=12)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for lang, code in LANGS.items():
        text = args.flores / "devtest" / f"{code}.devtest"
        print(f"{lang}: Q8_0", file=sys.stderr)
        q8 = run_llama(args.llama, args.q8, text, args.ctx, args.chunks, args.threads)
        print(f"{lang}: Q4_K_M", file=sys.stderr)
        q4km = run_llama(args.llama, args.q4km, text, args.ctx, args.chunks, args.threads)
        print(f"{lang}: .satl", file=sys.stderr)
        raw = args.out.with_name(f"{args.out.stem}-{lang}-atlas.json")
        satl = run_atlas(args.atlas, args.satl, args.tokenizer, text, args.ctx, args.chunks,
                         args.threads, args.manifest, raw)
        ppl = lambda nlls: math.exp(sum(nlls) / len(nlls))  # noqa: E731
        rows.append({
            "lang": lang,
            "ppl": {"q8_0": ppl(q8), "q4_k_m": ppl(q4km), "satl_q4g64": ppl(satl)},
            "delta_nll_vs_q8_0": {
                "q4_k_m": paired_bootstrap(q8, q4km),
                "satl_q4g64": paired_bootstrap(q8, satl),
            },
            "chunk_nll": {"q8_0": q8, "q4_k_m": q4km, "satl_q4g64": satl},
        })
        r = rows[-1]
        print(f"{lang}: PPL Q8_0 {r['ppl']['q8_0']:.3f}  Q4_K_M {r['ppl']['q4_k_m']:.3f}  "
              f".satl {r['ppl']['satl_q4g64']:.3f}", file=sys.stderr)

    report = {
        "protocol": f"llama.cpp perplexity: n_ctx={args.ctx}, {args.chunks} chunks, BOS per chunk, "
                    "second half scored; FLORES-200 devtest; identical token ids verified",
        "models": {"q8_0": str(args.q8), "q4_k_m": str(args.q4km), "satl_q4g64": str(args.satl)},
        "note": "delta_nll_vs_q8_0 = mean per-chunk NLL difference (nats/token), 95% paired bootstrap CI",
        "languages": rows,
        "hardware": json.loads(args.manifest.read_text()),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
