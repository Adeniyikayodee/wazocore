import pytest

from atlas_trace.llamacpp_baseline import make_prompts, run, summarize_all

CORPUS = {lang: [f"{lang} sentence {i}" for i in range(3)] for lang in ("en", "yo", "ha", "ig")}


def test_prompts_cover_every_target_with_parallel_sources():
    ps = make_prompts(CORPUS, 2)
    assert [p.lang for p in ps] == ["en", "en", "yo", "yo", "ha", "ha", "ig", "ig"]
    assert "yo sentence 1" in ps[1].text  # English target translates from Yoruba
    assert "en sentence 0" in ps[2].text and "into Yoruba" in ps[2].text


def test_run_discards_warmup_and_summarizes():
    calls = []

    def fake(prompt):
        calls.append(prompt)
        return {"prompt_n": 50, "prompt_ms": 500.0, "predicted_n": 20, "predicted_ms": 1000.0}, "a b c d e"

    samples = run(make_prompts(CORPUS, 2), fake)
    assert len(calls) == 9 and len(samples) == 8
    en = summarize_all(samples)[0]
    assert en.decode_tok_s_median == pytest.approx(20.0)
    assert en.prefill_tok_s_median == pytest.approx(100.0)
    assert en.words_per_s_median == pytest.approx(5.0)
    assert en.gen_tokens_per_word == pytest.approx(4.0)
