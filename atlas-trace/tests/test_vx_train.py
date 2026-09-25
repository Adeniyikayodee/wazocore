import pytest

from atlas_trace.vx_train import chunks, lr_at, mixture


def test_chunks_start_with_bos_and_cover_the_stream():
    assert chunks(list(range(7)), 4, 99) == [[99, 0, 1, 2], [99, 3, 4, 5]]
    assert chunks([1], 4, 99) == []


def test_mixture_respects_shares_and_is_deterministic():
    per = {"yo": [[1]] * 3, "en": [[2]]}
    m = mixture(per, {"yo": 0.75, "en": 0.25}, 8, seed=1)
    assert sorted(x[0] for x in m) == [1] * 6 + [2] * 2
    assert m == mixture(per, {"yo": 0.75, "en": 0.25}, 8, seed=1)


def test_lr_warms_up_then_decays_to_a_tenth():
    assert lr_at(0, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(9, 1.0, 10, 100) == pytest.approx(1.0)
    assert lr_at(100, 1.0, 10, 100) == pytest.approx(0.1)
    assert lr_at(55, 1.0, 10, 100) < lr_at(20, 1.0, 10, 100)
