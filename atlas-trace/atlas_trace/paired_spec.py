"""Paired A/B of two llama-server configurations on the same prompts.

The dev machine is a fanless laptop whose speed drifts by 2x under
sustained load, so sequential runs cannot be compared. Here both servers
run at once (sharing the mmap'd model), each prompt goes to both
back-to-back with the order alternating, and the statistic is the
per-prompt ratio B/A, which drift affects equally on both sides.

Usage:
    python -m atlas_trace.paired_spec --a http://127.0.0.1:8090 --b http://127.0.0.1:8091 \\
        --flores data/flores200_dataset --prompts 8 --out results/spec/paired.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

from atlas_trace.llamacpp_baseline import make_prompts, read_devtest, server_client, to_sample
from atlas_trace.teal import bootstrap_ci


def ratio_summary(ratios: Sequence[float]) -> dict:
    mean, lo, hi = bootstrap_ci(list(ratios))
    return {"n": len(ratios), "median": statistics.median(ratios), "mean": mean, "ci95": [lo, hi]}


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--a", required=True, help="baseline server URL")
    ap.add_argument("--b", required=True, help="candidate server URL")
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--meta", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    run = {"a": server_client(args.a, args.max_tokens), "b": server_client(args.b, args.max_tokens)}
    prompts = make_prompts(read_devtest(args.flores), args.prompts)
    run["a"](prompts[0].text), run["b"](prompts[0].text)  # warm-up both
    pairs = []
    for i, p in enumerate(prompts):
        order = ("a", "b") if i % 2 == 0 else ("b", "a")
        got = {k: to_sample(p, *run[k](p.text)) for k in order}
        a, b = got["a"], got["b"]
        pairs.append({
            "lang": p.lang, "index": p.index, "identical": a.output == b.output,
            "decode_ratio": (b.gen_tokens / b.gen_ms) / (a.gen_tokens / a.gen_ms),
            "a_words_s": a.gen_words / a.gen_ms * 1e3, "b_words_s": b.gen_words / b.gen_ms * 1e3,
        })
        print(f"{p.lang} #{p.index}: B/A {pairs[-1]['decode_ratio']:.2f} identical={pairs[-1]['identical']}", file=sys.stderr)

    langs = sorted({x["lang"] for x in pairs})
    summary = {l: ratio_summary([x["decode_ratio"] for x in pairs if x["lang"] == l]) for l in langs}
    for l, s in summary.items():
        print(f"{l}: speed B/A median {s['median']:.2f}, mean {s['mean']:.2f} [{s['ci95'][0]:.2f}, {s['ci95'][1]:.2f}]", file=sys.stderr)
    print(f"identical outputs: {sum(x['identical'] for x in pairs)}/{len(pairs)}", file=sys.stderr)
    meta = json.loads(args.meta.read_text()) if args.meta else {}
    args.out.write_text(json.dumps({**meta, "summary": summary, "pairs": pairs}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
