import pytest

from atlas_trace.vocab_expand import apply_merges, balanced_corpus, expand, learn_merges


def test_learns_most_frequent_pairs_in_order():
    words = {("a", "b", "c"): 5, ("a", "b"): 3, ("x", "y"): 1}
    merges = learn_merges(words, 5)
    assert merges[0] == ("a", "b")          # 8 occurrences
    assert merges[1] == ("ab", "c")         # 5 after the first merge
    assert ("x", "y") not in merges         # a single occurrence is not worth a token


def test_apply_reproduces_training_segmentation_and_keeps_unseen_words():
    words = {("a", "b", "c"): 5, ("a", "b"): 3}
    ranks = {p: r for r, p in enumerate(learn_merges(words, 5))}
    assert apply_merges(("a", "b", "c"), ranks) == ["abc"]
    assert apply_merges(("q", "r"), ranks) == ["q", "r"]
    assert apply_merges(("a",), ranks) == ["a"]


def test_expand_appends_merges_and_keeps_ids_contiguous():
    # Special tokens (ids 3, 4) live only in added_tokens, as in Llama-3.
    tok = {"model": {"vocab": {"a": 0, "b": 1, "ab": 2}, "merges": [["a", "b"]], "type": "BPE"},
           "added_tokens": [{"content": "<s>", "id": 3}, {"content": "</s>", "id": 4}]}
    new, added = expand(tok, [("ab", "b"), ("a", "b")], first_id=5)
    assert new["model"]["merges"] == [["a", "b"], ["ab", "b"], ["a", "b"]]
    assert new["model"]["vocab"]["abb"] == 5 and new["model"]["vocab"]["ab"] == 2
    assert new["model"]["vocab"]["<s>"] == 3, "special ids unchanged"
    assert sorted(new["model"]["vocab"].values()) == list(range(6))
    assert added == [{"token": "abb", "id": 5, "left": "ab", "right": "b"}]
    assert tok["model"]["merges"] == [["a", "b"]], "input is not mutated"


def test_expand_rejects_id_gaps():
    tok = {"model": {"vocab": {"a": 0}, "merges": [], "type": "BPE"}, "added_tokens": []}
    with pytest.raises(ValueError):
        expand(tok, [("a", "a")], first_id=7)


def test_balanced_corpus_truncates_each_language():
    assert balanced_corpus({"yo": "abcdef", "ha": "xy"}, 3) == {"yo": "abc", "ha": "xy"}


def test_disallowed_merges_are_skipped_and_learning_continues():
    words = {("a", "b", "c"): 5, ("x", "y"): 4}
    merges = learn_merges(words, 5, allowed=lambda t: t != "ab")
    assert ("a", "b") not in merges
    assert merges[0] == ("b", "c") and ("x", "y") in merges


def test_whole_utf8_rejects_partial_characters():
    from atlas_trace.vocab_expand import whole_utf8
    from atlas_trace.vx_model import byte_decoder

    b = byte_decoder()
    enc = {v: k for k, v in b.items()}
    token = lambda raw: "".join(enc[x] for x in raw)
    assert whole_utf8(token(" ṣé".encode()), b)
    assert not whole_utf8(token(" ṣ".encode()[:3]), b)   # space + 2 of 3 bytes
    assert not whole_utf8(token("ṣ".encode()[1:]), b)    # starts mid-character
