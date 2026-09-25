"""Decode speed of atlas-rt (dense and sparse) against llama.cpp on the same
machine, in interleaved rounds (PROJECT-PLAN-B §12.1b measurement rule).

Each round runs llama-bench (tg) and atlas-rt's paired dense/sparse bench
back to back, alternating which goes first; the statistics are per-round
speed ratios, so thermal drift affects both engines equally. `--ecores`
runs both under `taskpolicy -c background` (efficiency cores only), the
low-end-device proxy.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

from atlas_trace.teal import bootstrap_ci


def ratio_stats(num: Sequence[float], den: Sequence[float]) -> dict:
    ratios = [a / b for a, b in zip(num, den)]
    mean, lo, hi = bootstrap_ci(ratios)
    return {"mean": mean, "ci95": [lo, hi], "rounds": len(ratios)}


def llama_tg(bin_: Path, model: Path, threads: int, n: int, prefix: list[str]) -> float:
    cmd = [*prefix, str(bin_), "-m", str(model), "-t", str(threads), "-p", "0", "-n", str(n), "-r", "1", "-ngl", "0", "-o", "json"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)[0]["avg_ts"])


def atlas_pair(bin_: Path, model: Path, profile: Path, threads: int, n: int, prefix: list[str]) -> tuple[float, float]:
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        cmd = [*prefix, str(bin_), "--threads", str(threads), "bench", "--model", str(model), "--profile", str(profile),
               "--tokens", str(n), "--warmup", "4", "--out", f.name]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        r = json.loads(Path(f.name).read_text())
    return r["dense_tok_s_median"], r["sparse_tok_s_median"]


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--llama-bench", type=Path, default=Path("vendor/llama.cpp/build-cpu/bin/llama-bench"))
    ap.add_argument("--gguf", type=Path, default=Path("data/models/N-ATLaS.Q4_K_M.gguf"))
    ap.add_argument("--atlas", type=Path, default=Path("target/release/atlas-rt"))
    ap.add_argument("--satl", type=Path, default=Path("data/models/n-atlas-q4g64s.satl"))
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--ecores", action="store_true")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    prefix = ["taskpolicy", "-c", "background"] if args.ecores else []
    rounds = []
    for i in range(args.rounds):
        def llama() -> float:
            return llama_tg(args.llama_bench, args.gguf, args.threads, args.tokens, prefix)

        def atlas() -> tuple[float, float]:
            return atlas_pair(args.atlas, args.satl, args.profile, args.threads, args.tokens, prefix)

        if i % 2 == 0:
            l, (d, s) = llama(), atlas()
        else:
            (d, s), l = atlas(), llama()
        rounds.append({"llama_cpp": l, "atlas_dense": d, "atlas_sparse": s})
        print(f"round {i}: llama.cpp {l:.2f}  atlas dense {d:.2f}  sparse {s:.2f} tok/s", file=sys.stderr)

    col = lambda k: [r[k] for r in rounds]  # noqa: E731
    summary = {
        "sparse_vs_llama_cpp": ratio_stats(col("atlas_sparse"), col("llama_cpp")),
        "dense_vs_llama_cpp": ratio_stats(col("atlas_dense"), col("llama_cpp")),
        "sparse_vs_dense": ratio_stats(col("atlas_sparse"), col("atlas_dense")),
    }
    for k, v in summary.items():
        print(f"{k}: {v['mean']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]", file=sys.stderr)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "tier": "M4 efficiency cores (taskpolicy background)" if args.ecores else "M4 performance cores",
        "threads": args.threads, "profile": str(args.profile), "gguf": str(args.gguf), "satl": str(args.satl),
        "summary": summary, "rounds": rounds, "hardware": json.loads(args.manifest.read_text()),
    }, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
