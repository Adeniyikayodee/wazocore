"""v8 comparisons from docs/PLAN-v8.md, on the pulled results.
v8 models are read from results/vocab/v8; the release, control-s0 and the
original from results/vocab/v7, as the plan fixes. Paired over sentences or
items, 5,000 bootstrap rounds, the v7 analysis code.
    uv run --project wazocore --with sacrebleu python scripts/v8_analysis.py results/vocab/v7 results/vocab/v8
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v7_analysis import CHRF2, LANGS, chrf_stats, compare, iroko, rows  # noqa: E402

V8 = ["inst", "control-inst", "long-s1"]
R_TASKS = [f"{t}_{l}" for t in ("afrixnli", "afrimgsm") for l in ("yor", "hau", "ibo")]
ACC = lambda x: 100 * x.mean()
CHRF = lambda s: CHRF2._compute_score_from_stats(list(s.sum(0))).score
IROKO = [("inst", "release"), ("inst", "n-atlas"), ("inst", "control-inst"),
         ("control-inst", "control-s0"), ("long-s1", "release")]
TRANSLATION = [("inst", "release"), ("inst", "n-atlas"), ("inst", "control-inst"), ("control-inst", "control-s0")]


def where(v7: Path, v8: Path, m: str) -> Path:
    return v8 if m in V8 else v7


def main(v7: Path, v8: Path) -> int:
    out: dict = {}
    show = lambda key, c, f=".1f": print(f"{key}: {c['delta']:+{f}} [{c['ci95'][0]:+{f}}, {c['ci95'][1]:+{f}}]"
                                         f"  lower95 {c['lower95']:+{f}}", flush=True)

    scores = {m: iroko(where(v7, v8, m), m) for m in ["n-atlas", "release", "control-s0", *V8]}
    assert all(len(scores[m]) == 12 for m in scores), {m: len(v) for m, v in scores.items()}
    pooled = {m: np.concatenate([v[t] for t in R_TASKS]) for m, v in scores.items()}
    out["R"] = {m: ACC(v) for m, v in pooled.items()}
    print("R:", {m: round(v, 1) for m, v in out["R"].items()}, f"({len(pooled['inst'])} items)")
    for a, b in IROKO:
        out[f"R | {a} vs {b}"] = c = compare(pooled, [a], [b], ACC, len(pooled[a])); show(f"R | {a} vs {b}", c)

    # Share of the release's loss on R that inst recovers, per bootstrap round
    rng, n = np.random.default_rng(0), len(pooled["inst"])
    share = lambda i: (ACC(pooled["inst"][i]) - ACC(pooled["release"][i])) / (ACC(pooled["n-atlas"][i]) - ACC(pooled["release"][i]))
    d = np.sort([share(rng.integers(0, n, n)) for _ in range(5000)])
    out["R | recovered share"] = {"delta": float(share(np.arange(n))), "ci95": [float(d[125]), float(d[4874])]}
    r = out["R | recovered share"]; print(f"R | recovered share: {r['delta']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}]")

    for task in sorted(scores["inst"]):
        s = {m: v[task] for m, v in scores.items()}
        for a, b in IROKO:
            out[f"{a} vs {b} | {task}"] = c = compare(s, [a], [b], ACC, len(s[a])); show(f"{a} vs {b} | {task}", c)

    models = sorted({m for pair in TRANSLATION for m in pair})
    for way in ["en-xx", "xx-en"]:
        for lang in LANGS:
            samples = {m: rows(where(v7, v8, m) / f"heldout-{way}-{m}.json", lang) for m in models}
            comet = {m: rows(where(v7, v8, m) / f"comet-heldout-{way}-{m}.json", lang) for m in models}
            ids = [x["i"] for x in samples["n-atlas"]]
            assert all([x["i"] for x in v] == ids for d in (samples, comet) for v in d.values()), f"{way} {lang}: sentences differ"
            chrf = {m: chrf_stats(v) for m, v in samples.items()}
            com = {m: np.array([x["comet"] for x in v]) for m, v in comet.items()}
            for a, b in TRANSLATION:
                key = f"{a} vs {b} | {way} | {lang}"
                out[key] = {"chrf": compare(chrf, [a], [b], CHRF, len(ids)), "comet": compare(com, [a], [b], ACC, len(ids))}
                show(f"{key} chrF++", out[key]["chrf"], ".2f"); show(f"{key} AfriCOMET", out[key]["comet"], ".2f")

    (v8 / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2])))
