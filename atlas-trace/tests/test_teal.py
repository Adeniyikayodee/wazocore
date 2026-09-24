"""TEAL calibration tests: the pure core everywhere, the hooks on the tiny
random Llama-3 fixture (CPU, needs the `ref` dependency group)."""

import json
from pathlib import Path

import numpy as np
import pytest

from atlas_trace.teal import (
    BINS, LOG_HI, LOG_LO, SITES, TRACE_SITES, atr_prefix, chunks, pack_masks, site_sparsity,
    threshold_from_hist, trace_payload,
)

TINY = Path(__file__).resolve().parents[2] / "crates/atlas-rt/tests/fixtures/tiny"


def hist_of(values: np.ndarray) -> np.ndarray:
    logs = np.log10(np.clip(np.abs(values), 10**LOG_LO, None))
    return np.histogram(logs, bins=BINS, range=(LOG_LO, LOG_HI))[0].astype(float)


def test_threshold_hits_the_requested_quantile():
    x = np.random.default_rng(0).standard_normal(200_000)
    for p in [0.25, 0.4, 0.5]:
        t = threshold_from_hist(hist_of(x), p)
        assert abs(np.mean(np.abs(x) <= t) - p) < 2e-3
    assert threshold_from_hist(hist_of(x), 0.0) == 0.0
    with pytest.raises(ValueError):
        threshold_from_hist(hist_of(x), 1.0)


def test_chunks_start_with_bos_and_do_not_overlap():
    c = chunks(list(range(10)), n_ctx=4, max_chunks=10, bos=99)
    assert c == [[99, 0, 1, 2], [99, 3, 4, 5], [99, 6, 7, 8]]
    assert len(chunks(list(range(10)), 4, 1, 99)) == 1


def test_pack_masks_follows_trace_spec():
    m = np.zeros((1, 128), dtype=bool)
    m[0, [0, 63, 64, 127]] = True
    words = np.frombuffer(pack_masks(m), dtype="<u8")
    assert list(words) == [1 | 1 << 63, 1 | 1 << 63]
    with pytest.raises(ValueError):
        pack_masks(np.zeros((1, 10), dtype=bool))


def test_trace_payload_is_token_major():
    rng = np.random.default_rng(1)
    rec = {(l, s): rng.random((3, 64 if s == "gate_up" else 128)) > 0.5 for l in range(2) for s in TRACE_SITES}
    raw = np.frombuffer(trace_payload(rec, 2), dtype=np.uint8).reshape(3, 2, 8 + 16)
    for t in range(3):
        for l in range(2):
            gate = np.unpackbits(raw[t, l, :8], bitorder="little").astype(bool)
            down = np.unpackbits(raw[t, l, 8:], bitorder="little").astype(bool)
            assert (gate == rec[(l, "gate_up")][t]).all() and (down == rec[(l, "down")][t]).all()


def test_atr_prefix_pads_to_eight_bytes():
    p = atr_prefix({"format": "atlas-trace", "version": 1})
    assert p[:4] == b"ATRC" and len(p) % 8 == 0
    n = int.from_bytes(p[4:8], "little")
    assert json.loads(p[8:8 + n])["version"] == 1


def test_site_sparsity_pools_layers_exactly():
    kept = {"0.o": [3, 10], "1.o": [1, 10], "0.down": [5, 10]}
    s = site_sparsity(kept)
    assert s["o"] == pytest.approx(0.8) and s["down"] == pytest.approx(0.5) and s["qkv"] == 1.0


# ---- hooks on the tiny model ------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    model = transformers.LlamaForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32).eval()
    ids = [int(i) for i in np.random.default_rng(2).integers(1, 256, size=8 * 63)]
    return model, chunks(ids, 64, 8, bos=0)


def run(model, chunk):
    import torch

    with torch.no_grad():
        return model(torch.tensor([chunk])).logits[0].float().log_softmax(-1)


def test_calibrated_thresholds_give_target_sparsity_on_calibration_data(tiny):
    from atlas_trace.teal import Control, attach, profile

    model, cs = tiny
    ctl = Control(mode="calibrate", hists={})
    handles = attach(model, ctl)
    try:
        for c in cs:
            run(model, c)
        layers = len(model.model.layers)
        assert set(ctl.hists) == {(l, s) for l in range(layers) for s in SITES}
        ctl.mode, ctl.thresholds, ctl.kept = "sparsify", profile(ctl.hists, layers, 0.5), {}
        for c in cs:
            run(model, c)
        # Sparsified inputs shift later layers, so only layer 0's first site
        # sees exactly the calibration distribution.
        k, n = ctl.kept[(0, "qkv")]
        assert abs(1 - k / n - 0.5) < 0.01
    finally:
        [h.remove() for h in handles]


def test_zero_sparsity_is_the_dense_model_and_masks_change_logits(tiny):
    from atlas_trace.teal import Control, attach

    model, cs = tiny
    dense = run(model, cs[0])
    layers = len(model.model.layers)
    ctl = Control(mode="sparsify", thresholds=[[0.0] * len(SITES)] * layers, kept={})
    handles = attach(model, ctl)
    try:
        assert (run(model, cs[0]) - dense).abs().max() < 1e-5
        ctl.thresholds, ctl.kept = [[0.05] * len(SITES)] * layers, {}
        kl = (dense.exp() * (dense - run(model, cs[0]))).sum(-1).mean()
        assert kl > 1e-4
    finally:
        [h.remove() for h in handles]


def test_recorded_trace_masks_match_kept_counts(tiny):
    from atlas_trace.teal import Control, attach

    model, cs = tiny
    layers = len(model.model.layers)
    ctl = Control(mode="sparsify", thresholds=[[0.05] * len(SITES)] * layers, kept={}, record={})
    handles = attach(model, ctl)
    try:
        run(model, cs[0])
        rec = {k: v[0] for k, v in ctl.record.items()}
        assert set(rec) == {(l, s) for l in range(layers) for s in TRACE_SITES}
        assert rec[(1, "down")].shape == (64, 128)
        assert int(rec[(1, "down")].sum()) == ctl.kept[(1, "down")][0]
        assert len(trace_payload(rec, layers)) == 64 * layers * (64 + 128) // 8
    finally:
        [h.remove() for h in handles]


def test_python_traces_decode_in_rust_atlas_sim(tiny, tmp_path):
    """The .atr contract (docs/trace-format.md) holds across the two languages."""
    import subprocess

    from atlas_trace.teal import Control, attach

    sim = Path(__file__).resolve().parents[2] / "target/release/atlas-sim"
    if not sim.exists():
        pytest.skip("build atlas-sim first: cargo build --release -p atlas-sim")
    model, cs = tiny
    cfg = model.config
    layers = cfg.num_hidden_layers
    header = {
        "format": "atlas-trace", "version": 1, "model": "tiny", "lang": "yo", "source": "test", "profile": "t",
        "layers": layers, "tokens": 2 * 64,
        "matrices": [
            {"name": "gate_up", "channels": cfg.hidden_size, "bytes_per_channel": 2 * cfg.intermediate_size * 9 // 16},
            {"name": "down", "channels": cfg.intermediate_size, "bytes_per_channel": cfg.hidden_size * 9 // 16},
        ],
    }
    ctl = Control(mode="sparsify", thresholds=[[0.05] * len(SITES)] * layers)
    handles = attach(model, ctl)
    path = tmp_path / "t.atr"
    try:
        with path.open("wb") as f:
            f.write(atr_prefix(header))
            for c in cs[:2]:
                ctl.kept, ctl.record = {}, {}
                run(model, c)
                f.write(trace_payload({k: v[0] for k, v in ctl.record.items()}, layers))
    finally:
        [h.remove() for h in handles]
    out = subprocess.run([str(sim), "run", "--trace", str(path), "--cache-frac", "0.5", "--policies", "lru,belady"],
                         capture_output=True, text=True, check=True)
    report = json.loads(out.stdout)
    assert report["trace"]["tokens"] == 128
    assert all(0.0 <= r["byte_hit_rate"] <= 1.0 for r in report["results"])
