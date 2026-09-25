import math

import numpy as np

from atlas_trace.vx_model import bits_per_char, byte_decoder, char_lengths, decompose, init_rows


def test_decompose_recurses_to_original_pieces():
    vocab = {"a": 0, "b": 1, "c": 2}
    new = [{"token": "ab", "id": 10, "left": "a", "right": "b"},
           {"token": "abc", "id": 11, "left": "ab", "right": "c"}]
    assert decompose(new, vocab) == {10: [0, 1], 11: [0, 1, 2]}


def test_init_rows_uniform_and_weighted_means():
    table = np.array([[0.0, 0.0], [2.0, 4.0], [10.0, 10.0]], dtype=np.float32)
    parts = {5: [0, 1], 6: [1, 2]}
    assert np.allclose(init_rows(table, parts), [[1.0, 2.0], [6.0, 7.0]])
    w = init_rows(table, parts, {5: [1.0, 3.0], 6: [1.0, 1.0]})
    assert np.allclose(w[0], [1.5, 3.0]) and w.dtype == np.float32


def test_char_lengths_use_decoded_bytes():
    bd = byte_decoder()
    assert bd["Ġ"] == ord(" ") and bd["a"] == ord("a")
    lengths = char_lengths({9: [0, 1]}, {0: b"ab", 1: b"\xc3\xa0"}.__getitem__)
    assert lengths == {9: [2, 2]}


def test_bits_per_char():
    assert math.isclose(bits_per_char(math.log(2) * 10, 5), 2.0)
