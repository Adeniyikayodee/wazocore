"""Per-language llama.cpp baseline (PROJECT-PLAN-B §6, P0).

Sends FLORES-200 devtest translation prompts to a running `llama-server` and
records prefill/decode speed from the server's own timings, plus words/s and
chars/s of the generated text: the parity metric of §2.2.

Each target language is produced by translating a parallel sentence into it
(English targets translate from Yoruba), so every language generates the same
content and differences in words/s come from tokenization, not the task.

Usage (server started by scripts/baseline_llamacpp.sh):
    uv run --project atlas-trace python -m atlas_trace.llamacpp_baseline \
        --server http://127.0.0.1:8088 --flores data/flores200_dataset \
        --prompts 10 --out results/baseline/raw.json
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

from atlas_trace.tokens_per_word import LANGS, count_chars, count_words

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


@dataclass(frozen=True)
class LangSummary:
    lang: str
    samples: int
    prefill_tok_s_median: float
    decode_tok_s_median: float
    decode_tok_s_p95: float
    words_per_s_median: float
    chars_per_s_median: float
    gen_tokens_per_word: float
    words_per_s_pooled: float


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


def p95(xs: Sequence[float]) -> float:
    return statistics.quantiles(xs, n=20, method="inclusive")[-1] if len(xs) > 1 else xs[0]


def summarize(lang: str, samples: Sequence[Sample]) -> LangSummary:
    ok = [s for s in samples if s.gen_ms > 0 and s.prompt_ms > 0 and s.gen_words > 0]
    decode = [s.gen_tokens / s.gen_ms * 1e3 for s in ok]
    return LangSummary(
        lang=lang,
        samples=len(ok),
        prefill_tok_s_median=statistics.median(s.prompt_tokens / s.prompt_ms * 1e3 for s in ok),
        decode_tok_s_median=statistics.median(decode),
        decode_tok_s_p95=p95(decode),
        words_per_s_median=statistics.median(s.gen_words / s.gen_ms * 1e3 for s in ok),
        chars_per_s_median=statistics.median(s.gen_chars / s.gen_ms * 1e3 for s in ok),
        gen_tokens_per_word=sum(s.gen_tokens for s in ok) / sum(s.gen_words for s in ok),
        words_per_s_pooled=sum(s.gen_words for s in ok) / sum(s.gen_ms for s in ok) * 1e3,
    )


def summarize_all(samples: Sequence[Sample]) -> list[LangSummary]:
    return [summarize(l, [s for s in samples if s.lang == l]) for l in NAMES]


def render(rows: Sequence[LangSummary]) -> str:
    head = f"{'lang':<5}{'n':>3}{'prefill t/s':>12}{'decode t/s':>11}{'words/s':>9}{'chars/s':>9}{'tok/word':>9}"
    return "\n".join(
        [head]
        + [
            f"{r.lang:<5}{r.samples:>3}{r.prefill_tok_s_median:>12.1f}{r.decode_tok_s_median:>11.2f}"
            f"{r.words_per_s_median:>9.2f}{r.chars_per_s_median:>9.1f}{r.gen_tokens_per_word:>9.2f}"
            for r in rows
        ]
    )


# ---- imperative shell -------------------------------------------------------

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


def run(prompts: Sequence[Prompt], complete: Complete) -> list[Sample]:
    complete(prompts[0].text)  # warm-up, discarded
    samples = []
    for p in prompts:
        timings, output = complete(p.text)
        sample = to_sample(p, timings, output)
        print(f"{p.lang} #{p.index}: {sample.gen_tokens} tok, {sample.gen_words} words", file=sys.stderr)
        samples.append(sample)
    return samples


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="http://127.0.0.1:8088")
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--prompts", type=int, default=10, help="per language")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--meta", type=Path, help="JSON merged into the output (model, build, threads)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    prompts = make_prompts(read_devtest(args.flores), args.prompts)
    samples = run(prompts, server_client(args.server, args.max_tokens))
    summary = summarize_all(samples)
    print(render(summary), file=sys.stderr)
    meta = json.loads(args.meta.read_text()) if args.meta else {}
    report = {
        **meta,
        "dataset": "FLORES-200 devtest (translation prompts)",
        "max_tokens": args.max_tokens,
        "summary": [asdict(r) for r in summary],
        "samples": [asdict(s) for s in samples],
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
