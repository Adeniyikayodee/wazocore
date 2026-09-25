"""C1/C2 tables from an atlas-rt `teal-evaluate` result (pure over the JSON).

C1: with English thresholds, per language: effective sparsity at each site
and next-token loss (ΔNLL) with bootstrap CIs over chunks.
C2: own-language minus English calibration, paired per chunk.
"""

from __future__ import annotations

from typing import Sequence

from atlas_trace.format_quality import paired_bootstrap
from atlas_trace.teal import bootstrap_ci

LANGS = ("en", "yo", "ha", "ig")


def ci(t: Sequence[float]) -> str:
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def index(results: list[dict]) -> dict[tuple[str, str, float], dict]:
    return {(r["eval_lang"], r["calib_lang"], round(r["target_sparsity"], 2)): r for r in results}


def c1_rows(results: list[dict]) -> list[list[str]]:
    r = index(results)
    ps = sorted({k[2] for k in r})
    return [
        [f"{round(p * 100)}%", lang, f"{e['qkv']:.3f}", f"{e['o']:.3f}", f"{e['gate_up']:.3f}", f"{e['down']:.3f}",
         f"{x['kl'][0]:.3f}", ci(bootstrap_ci(x["delta_nll_per_chunk"])), f"{x['top1']:.3f}"]
        for p in ps for lang in LANGS if (x := r.get((lang, "en", p))) for e in [x["effective_sparsity"]]
    ]


def c2_rows(results: list[dict]) -> list[list[str]]:
    r = index(results)
    ps = sorted({k[2] for k in r})
    return [
        [lang, f"{round(p * 100)}%", ci(paired_bootstrap(a["kl_per_chunk"], b["kl_per_chunk"])),
         ci(paired_bootstrap(a["delta_nll_per_chunk"], b["delta_nll_per_chunk"])),
         f"{b['effective_sparsity']['o']:.3f} vs {a['effective_sparsity']['o']:.3f}"]
        for lang in LANGS[1:] for p in ps
        if (a := r.get((lang, "en", p))) and (b := r.get((lang, lang, p)))
    ]
