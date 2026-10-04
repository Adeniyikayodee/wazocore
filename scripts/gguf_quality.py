"""Quality of the shipped files: greedy NTREX-128 translations from both
Q4_K_M GGUFs through llama-server, one model in memory at a time
(docs/METHOD.md, section 3). Rows suit `vx_torch score`; a stopped run
resumes where it left off.

    [SAMPLER='{"repeat_penalty": 1.1}'] python scripts/gguf_quality.py SENTENCES OUT_DIR [NAME OTHER.gguf]
    python -m wazocore.vx_torch score OUT_DIR/ntrex-gguf-n-atlas.json OUT_DIR/ntrex-gguf-wazocore.json --out OUT_DIR/summary.json
"""
import json, os, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path
N = int(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"n-atlas": "data/models/N-ATLaS.Q4_K_M.gguf", "wazocore": "data/models/gguf-vx4/WazoCore-8B-v4.Q4_K_M.gguf"}
if len(sys.argv) > 4:  # another file to score against the same original replies: NAME PATH
    MODELS[sys.argv[3]] = sys.argv[4]
SAMPLER = json.loads(os.environ.get("SAMPLER", "{}"))  # extra request fields for the NAME file, e.g. a repetition penalty
NAMES = {"ig": ("Igbo", "ibo"), "ha": ("Hausa", "hau"), "yo": ("Yoruba", "yor")}
src = Path("data/ntrex/newstest2019-src.eng.txt").read_text(encoding="utf-8").splitlines()[:N]
def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:8091" + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))
for name, gguf in MODELS.items():
    f = OUT / f"ntrex-gguf-{name}.json"
    rows = json.loads(f.read_text())["rows"] if f.exists() else []
    done = {(r["lang"], r["i"]) for r in rows}
    if len(done) == 3 * N: continue
    proc = subprocess.Popen(["llama-server", "-m", gguf, "-c", "2048", "-np", "1", "-ngl", "99", "--port", "8091"], stdout=open(OUT / f"{name}.log", "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(300):
            try:
                if json.load(urllib.request.urlopen("http://127.0.0.1:8091/health", timeout=2)).get("status") == "ok": break
            except Exception: time.sleep(1)
        for lang, (full, code) in NAMES.items():
            refs = Path(f"data/ntrex/newstest2019-ref.{code}.txt").read_text(encoding="utf-8").splitlines()[:N]
            for i, s in enumerate(src):
                if (lang, i) in done: continue
                try:
                    r = post("/v1/chat/completions", {"messages": [{"role": "user", "content": f"Translate the following English text into {full}. Reply with the translation only.\n\n{s}"}],
                                                      "max_tokens": 160, "temperature": 0.0, "cache_prompt": False, **(SAMPLER if name not in ("n-atlas", "wazocore") else {})})
                except urllib.error.HTTPError as e:  # the server rejects a reply that is not valid UTF-8: a failed request for a user
                    rows.append({"lang": lang, "i": i, "output": "", "reference": refs[i], "stopped": False, "tokens": 0, "gen_ms": 0.0, "new_tokens": 0, "error": e.code})
                    continue
                rows.append({"lang": lang, "i": i, "output": r["choices"][0]["message"]["content"].strip(), "reference": refs[i],
                             "stopped": r["choices"][0]["finish_reason"] == "stop", "tokens": r["usage"]["completion_tokens"],
                             "gen_ms": r["timings"]["predicted_ms"], "new_tokens": 0})
                if len(rows) % 20 == 0:
                    f.write_text(json.dumps({"model": gguf, "test": "ntrex", "rows": rows}, ensure_ascii=False, indent=1)); print(name, lang, i, flush=True)
        f.write_text(json.dumps({"model": gguf, "test": "ntrex", "rows": rows}, ensure_ascii=False, indent=1))
    finally:
        proc.terminate(); proc.wait()
print("DONE")
