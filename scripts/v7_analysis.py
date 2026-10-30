"""v7 comparisons from docs/PLAN-v7.md, on the pulled results.
Each side is averaged over its seeds inside every bootstrap round; the
same sentences are drawn for every model.
    uv run --project wazocore --with sacrebleu python scripts/v7_analysis.py results/vocab/v7
"""

import json
import sys
import unicodedata
from pathlib import Path

import numpy as np
from sacrebleu.metrics import CHRF

LANGS = ["yo", "ha", "ig"]
CHRF2 = CHRF(word_order=2)
NFC = lambda s: unicodedata.normalize("NFC", s)
SIDES = {
    "release vs n-atlas": (["release"], ["n-atlas"]),
    "release vs control (3 seeds)": (["release", "long-s1", "long-s2"], ["control-s0", "control-s1", "control-s2"]),
    "control vs n-atlas (3 seeds)": (["control-s0", "control-s1", "control-s2"], ["n-atlas"]),
    **{f"{a} vs release": ([a], ["release"]) for a in ["fvt", "no-canon", "no-kl", "old-anchor", "no-calib", "long-s1", "long-s2"]},
}


def rows(path: Path, lang: str) -> list[dict]:
    return sorted((r for r in json.loads(path.read_text(encoding="utf-8"))["rows"] if r["lang"] == lang), key=lambda r: r["i"])


def chrf_stats(rs: list[dict]) -> np.ndarray:
    return np.array(CHRF2._extract_corpus_statistics([NFC(r["output"]) for r in rs], [[NFC(r["reference"]) for r in rs]]), dtype=np.float64)


def compare(stats: dict, a: list[str], b: list[str], score, n: int, rounds: int = 5000) -> dict:
    side = lambda names, idx: np.mean([score(stats[m][idx]) for m in names])
    delta = lambda idx: side(a, idx) - side(b, idx)
    rng = np.random.default_rng(0)
    d = np.sort([delta(rng.integers(0, n, n)) for _ in range(rounds)])
    return {"delta": float(delta(np.arange(n))), "ci95": [float(d[int(0.025 * rounds)]), float(d[int(0.975 * rounds) - 1])], "lower95": float(d[int(0.05 * rounds)])}


def main(r: Path) -> int:
    models = sorted({m for a, b in SIDES.values() for m in a + b})
    out = {}
    for way in ["en-xx", "xx-en"]:
        for lang in LANGS:
            samples = {m: rows(r / f"heldout-{way}-{m}.json", lang) for m in models}
            comet = {m: rows(r / f"comet-heldout-{way}-{m}.json", lang) for m in models}
            ids = [x["i"] for x in samples["n-atlas"]]
            assert all([x["i"] for x in v] == ids for d in (samples, comet) for v in d.values()), f"{way} {lang}: sentences differ"
            chrf = {m: chrf_stats(v) for m, v in samples.items()}
            com = {m: np.array([x["comet"] for x in v]) for m, v in comet.items()}
            for name, (a, b) in SIDES.items():
                key = f"{name} | {way} | {lang}"
                out[key] = {"chrf": compare(chrf, a, b, lambda s: CHRF2._compute_score_from_stats(list(s.sum(0))).score, len(ids)),
                            "comet": compare(com, a, b, lambda s: 100 * s.mean(), len(ids))}
                c, k = out[key]["chrf"], out[key]["comet"]
                print(f"{key}: chrF++ {c['delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]  "
                      f"AfriCOMET {k['delta']:+.2f} [{k['ci95'][0]:+.2f}, {k['ci95'][1]:+.2f}]", flush=True)
    (r / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
