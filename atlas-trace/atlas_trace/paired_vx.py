"""Paired words/s: original N-ATLaS vs the vocabulary-expanded model in
llama.cpp (PROJECT-PLAN-B §14, V4).

Two `llama-server`s (same binary, same flags) answer the same FLORES
translation prompts. Each prompt goes to both servers back to back and the
order alternates between prompts, so device drift (a fanless laptop
throttles) falls on both models equally. The quantity compared is words
(and characters) of generated text per second of decoding, which is what a
reader sees; tokens/s would hide the point, since the expanded model
spends fewer tokens per word.

    uv run python -m atlas_trace.paired_vx --original http://127.0.0.1:8088 \
        --expanded http://127.0.0.1:8089 --flores data/flores200_dataset \
        --prompts 20 --out results/vocab/paired-words-per-s.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from atlas_trace.llamacpp_baseline import NAMES, Complete, Prompt, Sample, make_prompts, read_devtest, server_client, to_sample

# ---- pure core ---------------------------------------------------------------


def order(i: int) -> tuple[str, str]:
    """Which model answers prompt i first: alternates every prompt."""
    return ("original", "expanded") if i % 2 == 0 else ("expanded", "original")


def pooled(samples: Sequence[Sample], unit: str) -> float:
    """Generated words (or chars) per second over all samples."""
    n = sum(s.gen_words if unit == "words" else s.gen_chars for s in samples)
    return n / sum(s.gen_ms for s in samples) * 1e3


def paired_summary(lang: str, orig: Sequence[Sample], vx: Sequence[Sample]) -> dict:
    """Per-language comparison. `ratio_*` > 1 means the expanded model is
    faster; the per-prompt median of chars/s ratios is robust to one slow
    outlier, the pooled ratio weights prompts by length."""
    pairs = [(o, v) for o, v in zip(orig, vx) if o.gen_ms > 0 and v.gen_ms > 0 and o.gen_chars and v.gen_chars]
    o, v = [p[0] for p in pairs], [p[1] for p in pairs]
    return {
        "lang": lang,
        "pairs": len(pairs),
        "original": {"words_per_s": pooled(o, "words"), "chars_per_s": pooled(o, "chars"),
                     "decode_tok_s": sum(s.gen_tokens for s in o) / sum(s.gen_ms for s in o) * 1e3,
                     "tokens_per_word": sum(s.gen_tokens for s in o) / max(1, sum(s.gen_words for s in o))},
        "expanded": {"words_per_s": pooled(v, "words"), "chars_per_s": pooled(v, "chars"),
                     "decode_tok_s": sum(s.gen_tokens for s in v) / sum(s.gen_ms for s in v) * 1e3,
                     "tokens_per_word": sum(s.gen_tokens for s in v) / max(1, sum(s.gen_words for s in v))},
        "ratio_words_per_s_pooled": pooled(v, "words") / pooled(o, "words"),
        "ratio_chars_per_s_pooled": pooled(v, "chars") / pooled(o, "chars"),
        "ratio_chars_per_s_median": statistics.median(
            (b.gen_chars / b.gen_ms) / (a.gen_chars / a.gen_ms) for a, b in pairs),
    }


def render(rows: Sequence[dict]) -> str:
    head = f"{'lang':<5}{'n':>3}{'orig w/s':>10}{'vx w/s':>9}{'w/s x':>7}{'chars/s x (median)':>20}{'tok/word':>14}"
    return "\n".join([head] + [
        f"{r['lang']:<5}{r['pairs']:>3}{r['original']['words_per_s']:>10.2f}{r['expanded']['words_per_s']:>9.2f}"
        f"{r['ratio_words_per_s_pooled']:>7.2f}{r['ratio_chars_per_s_median']:>20.2f}"
        f"{r['original']['tokens_per_word']:>8.2f} -> {r['expanded']['tokens_per_word']:.2f}"
        for r in rows])


# ---- shell -------------------------------------------------------------------


def run(prompts: Sequence[Prompt], clients: dict[str, Complete]) -> tuple[dict[str, list[Sample]], list[dict]]:
    """Paired samples, plus the failures: a server error (e.g. a reply that
    is not valid UTF-8) drops that prompt's pair and is recorded."""
    import urllib.error

    for c in clients.values():
        c(prompts[0].text)  # warm-up, discarded
    out: dict[str, list[Sample]] = {k: [] for k in clients}
    failures: list[dict] = []
    for i, p in enumerate(prompts):
        pair = {}
        for name in order(i):
            try:
                timings, text = clients[name](p.text)
                pair[name] = to_sample(p, timings, text)
            except urllib.error.HTTPError as e:
                failures.append({"model": name, "lang": p.lang, "index": p.index, "error": e.read().decode("utf-8", "replace")[:300]})
                print(f"{p.lang} #{p.index}: {name} failed ({e.code})", file=sys.stderr)
        if len(pair) < 2:
            continue
        for name, sample in pair.items():
            out[name].append(sample)
        o, v = pair["original"], pair["expanded"]
        print(f"{p.lang} #{p.index}: orig {o.gen_tokens} tok/{o.gen_ms / 1e3:.1f}s  vx {v.gen_tokens} tok/{v.gen_ms / 1e3:.1f}s", file=sys.stderr)
    return out, failures


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--original", default="http://127.0.0.1:8088")
    ap.add_argument("--expanded", default="http://127.0.0.1:8089")
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=20, help="per language")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--meta", type=Path, help="JSON merged into the output")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    prompts = make_prompts(read_devtest(args.flores), args.prompts)
    clients = {"original": server_client(args.original, args.max_tokens), "expanded": server_client(args.expanded, args.max_tokens)}
    samples, failures = run(prompts, clients)
    rows = [paired_summary(l, [s for s in samples["original"] if s.lang == l], [s for s in samples["expanded"] if s.lang == l]) for l in NAMES]
    print(render(rows))
    print(f"failures: {len(failures)} " + json.dumps({m: sum(f['model'] == m for f in failures) for m in clients}))
    report = {"meta": json.loads(args.meta.read_text()) if args.meta else {}, "summary": rows, "failures": failures,
              "samples": {k: [asdict(s) for s in v] for k, v in samples.items()}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
