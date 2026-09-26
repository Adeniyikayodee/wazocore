import pytest

from atlas_trace.paired_vx import Sample, make_prompts, order, paired_summary

CORPUS = {lang: [f"{lang} sentence {i}" for i in range(3)] for lang in ("en", "yo", "ha", "ig")}


def s(tokens, ms, words, chars):
    return Sample("yo", 0, 10, 1.0, tokens, ms, words, chars, "")


def test_order_alternates():
    assert order(0) == ("original", "expanded") and order(1) == ("expanded", "original")


def test_fewer_tokens_for_the_same_text_means_more_words_per_second():
    # same text (10 words, 60 chars); original needs 30 tokens, expanded 20,
    # both decode at 10 tok/s
    orig = [s(30, 3000.0, 10, 60)]
    vx = [s(20, 2000.0, 10, 60)]
    r = paired_summary("yo", orig, vx)
    assert r["ratio_words_per_s_pooled"] == pytest.approx(1.5)
    assert r["ratio_chars_per_s_median"] == pytest.approx(1.5)
    assert r["expanded"]["tokens_per_word"] == pytest.approx(2.0)


def test_prompts_interleave_languages_with_parallel_sources():
    ps = make_prompts(CORPUS, 2)
    assert [p.lang for p in ps] == ["en", "yo", "ha", "ig", "en", "yo", "ha", "ig"]
    assert "yo sentence 1" in ps[4].text  # English target translates from Yoruba
    assert "en sentence 0" in ps[1].text and "into Yoruba" in ps[1].text
