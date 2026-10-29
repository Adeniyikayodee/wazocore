"""Segment-level AfriCOMET for `vx_torch sample` files, either direction.

Runs in its own environment (unbabel-comet pins an older transformers):
    venv-comet/bin/python africomet.py OUT_DIR SAMPLES.json...
Writes OUT_DIR/comet-<stem>.json with one score per row, in row order,
so the laptop can bootstrap paired differences like chrF++. Files
already scored are skipped.
"""

import json
import sys
from pathlib import Path

MODEL = "masakhane/africomet-stl-1.1"


def main(out: Path, paths: list[Path]) -> int:
    todo = [p for p in paths if not (out / f"comet-{p.stem}.json").exists()]
    if not todo:
        return 0
    import torch
    from comet import download_model, load_from_checkpoint

    model = load_from_checkpoint(download_model(MODEL))
    for p in todo:
        rows = json.loads(p.read_text(encoding="utf-8"))["rows"]
        data = [{"src": r["source"], "mt": r["output"], "ref": r["reference"]} for r in rows]
        scores = model.predict(data, batch_size=64, gpus=int(torch.cuda.is_available()), num_workers=2, progress_bar=False).scores
        report = {"model": MODEL, "samples": p.name, "rows": [{"lang": r["lang"], "i": r["i"], "comet": s} for r, s in zip(rows, scores)]}
        tmp = out / f"comet-{p.stem}.part"
        tmp.write_text(json.dumps(report) + "\n")
        tmp.replace(out / f"comet-{p.stem}.json")
        print(f"{p.stem}: mean {sum(scores) / len(scores):.4f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), [Path(a) for a in sys.argv[2:]]))
