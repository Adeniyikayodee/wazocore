"""Regenerates every result table from the raw result files (PROJECT-PLAN-B
§4.4: "every result table regenerates from raw logs with one command").

Usage:
    uv run --project atlas-trace python -m atlas_trace.tables --results results --out results/TABLES.md
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Sequence

Section = Callable[[Path], str | None]


def table(head: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def probe(root: Path) -> str | None:
    rows = []
    for f in sorted(glob.glob(str(root / "probe" / "*.json"))):
        m = json.loads(Path(f).read_text())
        reads = [r for r in m["memory"] if r["kernel"] == "read" and r["gbps"]]
        best = max(reads, key=lambda r: r["gbps"]["median"])
        ceiling = m["decode_ceiling"]["rows"][0]["tok_per_s"] if m.get("decode_ceiling") else "?"
        rows.append([m["label"], m["host"]["cpu_model"], f"{best['gbps']['median']:.1f} ({best['threads']} thr)",
                     f"{ceiling:.1f}", f"{m['storage']['sequential']['mb_per_s']:.0f}" if m.get("storage") else "-"])
    head = ["Device", "CPU", "Read GB/s", "Dense ceiling tok/s", "Seq read MB/s"]
    return "## Hardware (atlas-probe)\n\n" + table(head, rows) if rows else None


def tokenizer(root: Path) -> str | None:
    d = load(root / "tokenizer" / "n-atlas-flores200-devtest.json")
    if not d:
        return None
    rows = [[r["lang"], f"{r['tokens_per_word']:.2f}", f"{(r['tax_vs_en'] or 1.0):.2f}x", f"{r['nfd_inflation']:.3f}"]
            for r in d["languages"]]
    return "## Tokenizer tax (FLORES-200 devtest)\n\n" + table(["Lang", "Tokens/word", "Tokens vs en", "NFD inflation"], rows)


def baseline(root: Path) -> str | None:
    rows = []
    for f in sorted(glob.glob(str(root / "baseline" / "*-languages.json"))):
        d = json.loads(Path(f).read_text())
        for s in d["summary"]:
            rows.append([d["label"], s["lang"], f"{s['prefill_tok_s_median']:.1f}", f"{s['decode_tok_s_median']:.2f}",
                         f"{s['words_per_s_median']:.2f}", f"{d.get('server_peak_rss_bytes', 0) / 2**30:.2f}"])
    head = ["Run", "Lang", "Prefill tok/s", "Decode tok/s", "Words/s", "Peak RSS GiB"]
    return "## llama.cpp baseline\n\n" + table(head, rows) if rows else None


def int8(root: Path) -> str | None:
    rows = []
    for f in sorted(glob.glob(str(root / "int8" / "*.json"))):
        d = json.loads(Path(f).read_text())
        rows.append([d["label"], d["tokens"], f"{d['kl_mean']:.2e}", f"{d['kl_p95']:.2e}", f"{100 * d['top1_agreement']:.1f}%",
                     f"{d['nll_reference']:.4f} → {d['nll_fast']:.4f}"])
    head = ["Lang", "Tokens", "Mean KL", "p95 KL", "Top-1", "NLL f32 → int8"]
    return "## Int8 activations vs f32 (same q4 weights)\n\n" + table(head, rows) if rows else None


def format_quality(root: Path) -> str | None:
    d = load(root / "quality" / "format-ppl.json")
    if not d:
        return None
    fmt = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"  # noqa: E731
    rows = [[r["lang"], f"{r['ppl']['q8_0']:.3f}", f"{r['ppl']['q4_k_m']:.3f}", f"{r['ppl']['satl_q4g64']:.3f}",
             fmt(r["delta_nll_vs_q8_0"]["q4_k_m"]), fmt(r["delta_nll_vs_q8_0"]["satl_q4g64"])] for r in d["languages"]]
    head = ["Lang", "PPL Q8_0", "PPL Q4_K_M", "PPL .satl", "ΔNLL Q4_K_M", "ΔNLL .satl"]
    return f"## Dense format quality vs Q8_0\n\n{d['protocol']}\n\n" + table(head, rows)


def convert(root: Path) -> str | None:
    f = root / "convert" / "n-atlas-q4g64-verify.txt"
    if not f.exists():
        return None
    by: dict[str, list[float]] = {}
    for line in f.read_text().splitlines():
        m = re.match(r"(\S+)\s+rel_rmse ([0-9.]+)", line)
        if m:
            by.setdefault(re.sub(r"layers\.\d+\.", "", m.group(1)), []).append(float(m.group(2)))
    rows = [[k, len(v), f"{statistics.mean(v):.4f}", f"{max(v):.4f}"] for k, v in by.items()]
    return "## .satl conversion error vs bf16\n\n" + table(["Tensor", "Count", "Mean rel RMSE", "Max"], rows)


SECTIONS: list[Section] = [probe, tokenizer, baseline, int8, format_quality, convert]


def render(root: Path) -> str:
    parts = [s(root) for s in SECTIONS]
    body = "\n\n".join(p for p in parts if p)
    return f"# Results\n\nGenerated from `{root}/` by `python -m atlas_trace.tables`. Do not edit by hand.\n\n{body}\n"


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--out", type=Path, default=Path("results/TABLES.md"))
    args = ap.parse_args(argv)
    args.out.write_text(render(args.results))
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
