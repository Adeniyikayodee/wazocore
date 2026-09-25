import pytest

from atlas_trace.llamacpp_baseline import Sample
from atlas_trace.paired_vx import order, paired_summary


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
