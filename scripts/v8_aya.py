"""v8 data: Aya instruction conversations mixed into v6's gold data, as
fixed in docs/PLAN-v8.md.

Rows sharing a 13-word sequence with IrokoBench, FLORES-200 (dev, devtest)
or NTREX-128 are dropped, then exact duplicates. Aya rows are added in a
fixed shuffled order until they make up 20/85 of the conversation tokens
(the release's tokenizer), so instruction data is 20% of training and
Wikipedia stays at 15%. The report gives every count and the share the
trainer will actually consume, for both tokenizers.

Needs `datasets` and `transformers`; on the pod, run with venv-eval:
  python -I scripts/v8_aya.py --vx VX_TOK --base N_ATLAS --flores F --ntrex N \
      --gold chat-v6-gold.jsonl --out chat-v8-inst.jsonl --report aya-report.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "wazocore"))
sys.path.insert(0, "/workspace/wazocore")
from wazocore.vx_core import aya_messages, fill_share, mixture, word_ngrams  # noqa: E402

AYA = ("CohereLabs/aya_dataset", "f9ea04583f02a8f86404ff6c58bf75fe637df8a2")
IROKO = {
    "masakhane/afrixnli": "e3ca06b30f3e7af2a86f6c8609ea76fee326bc56",
    "masakhane/afrimmlu": "96f247619673906ae4c321a3232afc67b98ea57e",
    "masakhane/afrimgsm": "8e4268d2b94941f18f63f694cf48e4ae26fbec65",
}
LANGUAGES = {"Yoruba": "yor", "Hausa": "hau", "Igbo": "ibo", "English": "eng"}
FLORES = ["eng_Latn", "yor_Latn", "hau_Latn", "ibo_Latn"]
N = 13
INSTRUCTION, CHAT = 0.20, 0.85
# The release recipe: 12M tokens of 512-token windows, 8 x 2 per step.
TOKENS, SEQ, PER_STEP = 12_000_000, 512, 16


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from strings(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)


def benchmark_ngrams(flores: Path, ntrex: Path) -> dict[str, set]:
    from datasets import load_dataset

    out: dict[str, set] = {}
    for name, rev in IROKO.items():
        grams = out.setdefault(name.split("/")[1], set())
        for config in LANGUAGES.values():
            for split in load_dataset(name, config, revision=rev).values():
                for row in split:
                    for text in strings(row):
                        grams |= word_ngrams(text, N)
    out["flores200"] = {g for split in ("dev", "devtest") for code in FLORES
                        for line in (flores / split / f"{code}.{split}").read_text(encoding="utf-8").splitlines()
                        for g in word_ngrams(line, N)}
    out["ntrex128"] = {g for f in sorted(ntrex.glob("*.txt"))
                       for line in f.read_text(encoding="utf-8").splitlines() for g in word_ngrams(line, N)}
    return out


def chat_ids(tok, messages) -> list[int]:
    return tok.apply_chat_template(messages, tokenize=True, return_dict=False)


def consumed_share(tok, convs: list[list[dict]], is_instruction: list[bool]) -> dict:
    """The instruction share of the conversation windows the trainer takes:
    it packs conversations in file order, cuts 512-token windows and cycles
    through them (vx_core.mixture)."""
    flags: list[bool] = []
    for c, inst in zip(convs, is_instruction):
        flags += [inst] * len(chat_ids(tok, c))
    windows = len(flags) // SEQ
    total = TOKENS // (SEQ * PER_STEP) * PER_STEP
    shares = {**{l: v * (1 - CHAT) for l, v in {"yo": 0.3, "ha": 0.3, "ig": 0.3, "en": 0.1}.items()}, "chat": CHAT}
    n = Counter(w[0] for w in mixture({l: [[l]] for l in shares}, shares, total, 0))["chat"]
    full, rest = divmod(n, windows)
    inst = full * sum(flags[: windows * SEQ]) + sum(flags[: rest * SEQ])
    return {"windows_in_file": windows, "windows_taken": n, "passes": round(n / windows, 2),
            "instruction_share_of_chat": round(inst / (n * SEQ), 4),
            "instruction_share_of_training": round(inst / (n * SEQ) * CHAT, 4)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vx", type=Path, required=True, help="the release's tokenizer")
    ap.add_argument("--base", type=Path, required=True, help="N-ATLaS, for the control's tokenizer")
    ap.add_argument("--flores", type=Path, required=True)
    ap.add_argument("--ntrex", type=Path, required=True)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    args = ap.parse_args(argv)

    from datasets import load_dataset
    from transformers import AutoTokenizer

    report: dict = {"aya": {"dataset": AYA[0], "revision": AYA[1]}, "irokobench_revisions": IROKO, "ngram": N}
    rows = [r for r in load_dataset(AYA[0], revision=AYA[1], split="train") if r["language"] in LANGUAGES]
    report["rows"] = dict(Counter(r["language"] for r in rows))

    grams = benchmark_ngrams(args.flores, args.ntrex)
    dropped, seen, kept = Counter(), set(), []
    for r in rows:
        m = aya_messages(r)
        if m is None:
            dropped["empty"] += 1
            continue
        mine = word_ngrams(m[0]["content"], N) | word_ngrams(m[1]["content"], N)
        hit = next((src for src, g in grams.items() if mine & g), None)
        if hit:
            dropped[hit] += 1
            continue
        key = (m[0]["content"], m[1]["content"])
        if key in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(key)
        kept.append((r["language"], m))
    report["dropped"] = dict(dropped)
    report["kept"] = dict(Counter(lang for lang, _ in kept))

    vx = AutoTokenizer.from_pretrained(args.vx)
    gold = args.gold.read_text(encoding="utf-8").splitlines()
    gold_tokens = sum(len(chat_ids(vx, json.loads(l)["messages"])) for l in gold)
    costs = [len(chat_ids(vx, m)) for _, m in kept]
    target = round(gold_tokens * INSTRUCTION / (CHAT - INSTRUCTION))
    picked, repeats = fill_share(costs, target, seed=0)
    report["tokens"] = {"gold": gold_tokens, "aya_available": sum(costs), "aya_target": target,
                        "aya_used": sum(costs[i] for i in picked), "aya_rows_used": len(picked),
                        "aya_repeats": repeats}
    report["used_by_language"] = dict(Counter(kept[i][0] for i in picked))

    aya = [json.dumps({"messages": kept[i][1]}, ensure_ascii=False) for i in picked]
    lines = [(l, False) for l in gold] + [(l, True) for l in aya]
    random.Random(0).shuffle(lines)
    tmp = args.out.with_suffix(".part")
    tmp.write_text("\n".join(l for l, _ in lines) + "\n", encoding="utf-8")
    tmp.replace(args.out)

    convs = [json.loads(l)["messages"] for l, _ in lines]
    flags = [inst for _, inst in lines]
    report["consumed"] = {"release_tokenizer": consumed_share(vx, convs, flags),
                          "original_tokenizer": consumed_share(AutoTokenizer.from_pretrained(args.base), convs, flags)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
