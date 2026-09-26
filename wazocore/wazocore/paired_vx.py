"""Paired words/s: original N-ATLaS vs the vocabulary-expanded model in
llama.cpp (PROJECT-PLAN-B §14, V4).

The two models (same `llama-server` binary and flags) answer the same
FLORES translation prompts. Only one model is in memory at a time (two
5 GB models on a 16 GB laptop page each other out): for each sentence
index both models answer it in every language, the first-mover
alternating, so device drift (a fanless laptop throttles) falls on both
equally. Load time is excluded; timings come from the server. The quantity compared is words
(and characters) of generated text per second of decoding, which is what a
reader sees; tokens/s would hide the point, since the expanded model
spends fewer tokens per word.

    scripts/paired_vx.sh LABEL THREADS [PROMPTS] [NGL]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import unicodedata
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

from wazocore.tokens_per_word import LANGS, count_chars, count_words

# ---- pure core ---------------------------------------------------------------


NAMES = {"en": "English", "yo": "Yoruba", "ha": "Hausa", "ig": "Igbo"}


@dataclass(frozen=True)
class Prompt:
    lang: str  # target language of the generated text
    index: int
    text: str


@dataclass(frozen=True)
class Sample:
    lang: str
    index: int
    prompt_tokens: int
    prompt_ms: float
    gen_tokens: int
    gen_ms: float
    gen_words: int
    gen_chars: int
    output: str


def build_prompt(target: str, source_lang: str, source_text: str) -> str:
    return (
        f"Translate the following {NAMES[source_lang]} text into {NAMES[target]}. "
        f"Reply with the translation only.\n\n{source_text}"
    )


def make_prompts(corpus: dict[str, Sequence[str]], n: int) -> list[Prompt]:
    """en targets translate from yo; every other target translates from en.

    Languages are interleaved (sentence i in every language, then i + 1) so
    drift in device speed over the run (thermals, background load) spreads
    evenly across languages instead of masquerading as a language effect.
    """
    def source(target: str) -> str:
        return "yo" if target == "en" else "en"

    return [
        Prompt(t, i, build_prompt(t, source(t), corpus[source(t)][i]))
        for i in range(n)
        for t in NAMES
    ]


def to_sample(p: Prompt, timings: dict, output: str) -> Sample:
    text = unicodedata.normalize("NFC", output.strip())
    return Sample(
        lang=p.lang,
        index=p.index,
        prompt_tokens=int(timings["prompt_n"]),
        prompt_ms=float(timings["prompt_ms"]),
        gen_tokens=int(timings["predicted_n"]),
        gen_ms=float(timings["predicted_ms"]),
        gen_words=count_words(text),
        gen_chars=count_chars(text),
        output=text,
    )


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


Complete = Callable[[str], tuple[dict, str]]


def server_client(url: str, max_tokens: int) -> Complete:
    def complete(prompt: str) -> tuple[dict, str]:
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "seed": 0,
            "cache_prompt": False,  # every prompt pays full prefill
        }
        req = urllib.request.Request(
            f"{url}/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.load(r)
        return resp["timings"], resp["choices"][0]["message"]["content"]

    return complete


def read_devtest(root: Path) -> dict[str, list[str]]:
    return {
        lang: (root / "devtest" / f"{code}.devtest").read_text(encoding="utf-8").splitlines()
        for lang, code in LANGS.items()
    }


def serve(bin_: str, gguf: Path, port: int, threads: int, ngl: int, log: Path):
    """Starts llama-server on one model; returns the process once healthy."""
    import subprocess
    import time
    import urllib.request

    proc = subprocess.Popen([bin_, "-m", str(gguf), "-t", str(threads), "-c", "2048", "-np", "1", "-ngl", str(ngl),
                             "--port", str(port), "--no-webui"], stdout=log.open("a"), stderr=subprocess.STDOUT)
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited; see {log}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return proc
        except OSError:
            time.sleep(0.5)


def run(prompts: Sequence[Prompt], models: dict[str, Path], args) -> tuple[dict[str, list[Sample]], list[dict]]:
    """One model in memory at a time: for each sentence index, both models
    answer that sentence in every language, first-mover alternating
    (`order`). Server errors (e.g. a reply that is not valid UTF-8) drop the
    pair and are recorded."""
    import urllib.error

    blocks: dict[int, list[Prompt]] = {}
    for p in prompts:
        blocks.setdefault(p.index, []).append(p)
    got: dict[tuple[str, str, int], Sample] = {}
    failures: list[dict] = []
    for i, block in sorted(blocks.items()):
        for name in order(i):
            proc = serve(args.server_bin, models[name], 8088, args.threads, args.ngl, args.out.with_suffix(f".{name}.log"))
            try:
                complete = server_client("http://127.0.0.1:8088", args.max_tokens)
                complete(block[0].text)  # warm-up after load, discarded
                for p in block:
                    try:
                        timings, text = complete(p.text)
                        got[(name, p.lang, p.index)] = to_sample(p, timings, text)
                    except urllib.error.HTTPError as e:
                        failures.append({"model": name, "lang": p.lang, "index": p.index, "error": e.read().decode("utf-8", "replace")[:300]})
                        print(f"{p.lang} #{p.index}: {name} failed ({e.code})", file=sys.stderr)
            finally:
                proc.terminate()
                proc.wait()
        print(f"sentence {i + 1}/{len(blocks)} done", file=sys.stderr)
    out: dict[str, list[Sample]] = {m: [] for m in models}
    for p in prompts:
        if all((m, p.lang, p.index) in got for m in models):
            for m in models:
                out[m].append(got[(m, p.lang, p.index)])
    return out, failures


def main(argv: Sequence[str]) -> int:
    import shutil

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--original", type=Path, required=True, help="GGUF")
    ap.add_argument("--expanded", type=Path, required=True, help="GGUF")
    ap.add_argument("--server-bin", default=shutil.which("llama-server"))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ngl", type=int, default=0, help="layers on the GPU (0 = CPU only)")
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=20, help="per language")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--meta", type=Path, help="JSON merged into the output")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompts = make_prompts(read_devtest(args.flores), args.prompts)
    models = {"original": args.original, "expanded": args.expanded}
    samples, failures = run(prompts, models, args)
    rows = [paired_summary(l, [s for s in samples["original"] if s.lang == l], [s for s in samples["expanded"] if s.lang == l]) for l in NAMES]
    print(render(rows))
    print(f"failures: {len(failures)} " + json.dumps({m: sum(f['model'] == m for f in failures) for m in models}))
    report = {"meta": json.loads(args.meta.read_text()) if args.meta else {}, "summary": rows, "failures": failures,
              "samples": {k: [asdict(s) for s in v] for k, v in samples.items()}}
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
