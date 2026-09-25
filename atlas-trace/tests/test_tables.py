import json

from atlas_trace.tables import render, table


def test_table_renders_markdown():
    assert table(["a", "b"], [[1, 2]]) == "| a | b |\n|---|---|\n| 1 | 2 |"


def test_render_skips_missing_sections(tmp_path):
    (tmp_path / "int8").mkdir()
    (tmp_path / "int8" / "yor.json").write_text(json.dumps({
        "label": "yor", "tokens": 160, "kl_mean": 5e-3, "kl_p95": 2e-2,
        "top1_agreement": 0.96, "nll_reference": 2.67, "nll_fast": 2.66,
    }))
    out = render(tmp_path)
    assert "Int8 activations" in out and "| yor | 160 |" in out
    assert "llama.cpp baseline" not in out


def test_teal_sections_render_from_eval_results(tmp_path):
    (tmp_path / "teal").mkdir()
    def row(lang, calib, kl):
        return {"eval_lang": lang, "calib_lang": calib, "target_sparsity": 0.3, "top1": 0.8,
                "effective_sparsity": {"qkv": 0.3, "o": 0.3, "gate_up": 0.3, "down": 0.3},
                "kl": [kl, kl, kl], "kl_per_chunk": [kl, kl], "delta_nll_per_chunk": [0.01, 0.02]}
    results = [row("en", "en", 0.1), row("yo", "en", 0.1), row("yo", "yo", 0.2)]
    (tmp_path / "teal" / "eval-matrix.json").write_text(json.dumps({"results": results}))
    out = render(tmp_path)
    assert "C1 (English thresholds" in out and "| 30% | yo |" in out
    assert "| yo | 30% | +0.100 [+0.100, +0.100]" in out
