"""Does a retry with sampling recover a reply with a broken character?
Every prompt that gave invalid UTF-8 in an earlier run answers greedily
through llama-server; each one the server rejects is retried under sampling
(docs/METHOD.md, section 8).

    uv run --project wazocore python scripts/retry_probe.py MODEL.gguf OUT.json [RETRIES]
"""
import json, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path
from wazocore.vx_core import looping
GGUF, OUT = sys.argv[1], Path(sys.argv[2]); RETRIES = int(sys.argv[3]) if len(sys.argv) > 3 else 30
FLORES = Path("data/flores200_dataset/devtest/eng_Latn.devtest").read_text(encoding="utf-8").splitlines()
NTREX = Path("data/ntrex/newstest2019-src.eng.txt").read_text(encoding="utf-8").splitlines()
# (test set, language, sentence): the speed benchmark's failure, then the four in the full-precision samples
CASES = [("flores", "Yoruba", 0), ("ntrex", "Yoruba", 465), ("flores", "Yoruba", 88), ("flores", "Hausa", 46), ("ntrex", "Hausa", 19)]
def ask(prompt, **sampler):
    """The reply, or None when the server rejects it as invalid UTF-8."""
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0.0, "cache_prompt": False, **sampler}
    req = urllib.request.Request("http://127.0.0.1:8092/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError:
        return None
proc = subprocess.Popen(["llama-server", "-m", GGUF, "-c", "2048", "-np", "1", "-ngl", "99", "--port", "8092"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
try:
    for _ in range(300):
        try:
            if json.load(urllib.request.urlopen("http://127.0.0.1:8092/health", timeout=2)).get("status") == "ok": break
        except Exception: time.sleep(1)
    rows = []
    for test, lang, i in CASES:
        prompt = f"Translate the following English text into {lang}. Reply with the translation only.\n\n{(FLORES if test == 'flores' else NTREX)[i]}"
        row = {"test": test, "lang": lang, "i": i, "greedy_valid": ask(prompt) is not None, "retries": []}
        for t in (0.3, 0.7) if not row["greedy_valid"] else ():
            replies = [ask(prompt, temperature=t, seed=s) for s in range(1, RETRIES + 1)]
            row["retries"].append({"temperature": t, "n": RETRIES, "valid": sum(r is not None for r in replies),
                                   "loops": sum(looping(r) for r in replies if r), "replies": replies})
            print(test, lang, i, t, row["retries"][-1]["valid"], "of", RETRIES, "valid", flush=True)
        rows.append(row)
    OUT.write_text(json.dumps({"model": GGUF, "rows": rows}, ensure_ascii=False, indent=1) + "\n")
finally:
    proc.terminate(); proc.wait()
