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
