import unicodedata

import pytest

from atlas_trace.tokens_per_word import audit, count_chars, count_words


def chars_as_tokens(texts):
    """Stub tokenizer: one token per code point, so NFD visibly inflates."""
    return [len(t) for t in texts]


def test_counts_words_and_chars():
    assert count_words("  a bc  d ") == 3
    assert count_chars("ọ̀ a") == 3  # NFC: ọ + combining grave, then a


def test_tax_is_relative_to_english():
    corpus = {"en": ["ab cd"], "yo": ["abcd efgh"]}
    en, yo = audit(corpus, chars_as_tokens)
    assert en.tax_vs_en is None
    assert yo.tax_vs_en == pytest.approx(9 / 5)
    assert yo.tokens_per_word == pytest.approx(4.5)


def test_nfd_inflation_detected_for_tone_marks():
    yo = unicodedata.normalize("NFC", "ẹ́")
    _, row = audit({"en": ["x"], "yo": [yo]}, chars_as_tokens)
    assert row.nfd_inflation > 1.0


def test_rejects_non_parallel_corpus():
    with pytest.raises(ValueError):
        audit({"en": ["a", "b"], "yo": ["a"]}, chars_as_tokens)
