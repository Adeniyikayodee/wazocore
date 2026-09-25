from atlas_trace.vocab_expand import apply_merges, learn_merges


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
