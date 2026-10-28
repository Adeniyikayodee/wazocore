"""How much the expanded tokenizer changes the tokenisation of English text.

The original tokenizer is rebuilt from the expanded one by dropping the
appended merges and the new vocabulary entries; both then encode every
non-empty line of the given text files.

    uv run --project wazocore python scripts/english_tokenisation.py \
        TOKENIZER.json NEW_TOKENS.json OUT.json TEXT_FILE...
"""

import copy
import json
import sys

from tokenizers import Tokenizer


def main(tok_path: str, new_path: str, out_path: str, *texts: str) -> int:
    expanded = json.load(open(tok_path, encoding="utf-8"))
    new = json.load(open(new_path, encoding="utf-8"))
    original = copy.deepcopy(expanded)
    original["model"]["merges"] = expanded["model"]["merges"][: -new["merges"]]
    original["model"]["vocab"] = {t: i for t, i in expanded["model"]["vocab"].items() if i < new["first_id"]}
    old, cur = Tokenizer.from_str(json.dumps(original)), Tokenizer.from_str(json.dumps(expanded))

    lines = [l for path in texts for l in open(path, encoding="utf-8").read().splitlines() if l.strip()]
    before = after = changed = 0
    for line in lines:
        a = old.encode(line, add_special_tokens=False).ids
        b = cur.encode(line, add_special_tokens=False).ids
        before, after, changed = before + len(a), after + len(b), changed + (a != b)
    report = {"files": list(texts), "lines": len(lines), "words": sum(len(l.split()) for l in lines),
              "lines_changed": changed, "tokens_original": before, "tokens_expanded": after,
              "token_change": after / before - 1}
    open(out_path, "w").write(json.dumps(report, indent=1) + "\n")
    print(f"{changed} of {len(lines)} lines tokenise differently; tokens {before} -> {after} ({report['token_change']:+.2%})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
