from atlas_trace.paired_spec import ratio_summary
from atlas_trace.spec_probe import combine, lookup_draft, ngram_drafter, ngram_table, simulate


def test_perfect_drafter_accepts_k_plus_one_per_step():
    out = list(range(10))
    oracle = lambda ctx, k: [t for t in out if t >= len(ctx)][:k]  # noqa: E731
    assert simulate([], out, oracle, 4) == (10, 2)  # 5 tokens per step


def test_useless_drafter_gives_one_token_per_step():
    assert simulate([], [1, 2, 3], lambda ctx, k: [], 4) == (3, 3)


def test_ngram_backs_off_to_shorter_context():
    table = ngram_table([[1, 2, 3, 4], [9, 2, 5]])
    draft = ngram_drafter(table)
    assert draft([7, 1, 2], 2) == [3, 4]  # (1, 2) -> 3, then (2, 3) -> 4
    assert draft([8], 3) == []


def test_lookup_repeats_earlier_continuation_and_combine_falls_back():
    assert lookup_draft([5, 6, 7, 5, 6], 2) == [7, 5]
    assert combine(lambda c, k: [], lambda c, k: [42])([1], 1) == [42]


def test_ratio_summary():
    s = ratio_summary([1.0, 1.0, 1.0])
    assert s["median"] == 1.0 and s["ci95"] == [1.0, 1.0] and s["n"] == 3
