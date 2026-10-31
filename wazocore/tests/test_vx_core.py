import math

import pytest

import numpy as np

from wazocore.vx_core import bits_per_char, byte_decoder, char_lengths, chunks, decompose, init_rows, lr_at, mixture


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


def test_careful_line_rejects_careless_spellings_and_english():
    from wazocore.vx_core import careful_line

    assert careful_line("Ndị obodo ahụ bụ ndị ọrụ ugbo nke ọma n'ụlọ ha.", "ig")
    assert not careful_line("Ndi obodo ahu bu ndi oru ugbo nke oma n'ulo ha.", "ig")
    assert careful_line("Ɗan wasan ƙwallon ƙafa ne daga ƙasar Najeriya.", "ha")
    assert not careful_line("Dan wasan kwallon kafa ne daga kasar Najeriya.", "ha")
    assert careful_line("Àwọn ọmọ ilé-ìwé náà lọ sí ọjà ní àárọ̀ ọjọ́ Ajé.", "yo")
    assert not careful_line("Awon omo ile-iwe naa lo si oja ni aaro ojo Aje ana.", "yo")
    assert not careful_line("This is one of the lines in the dump that is English.", "ha")
    assert careful_line("Any English line of reasonable length is kept.", "en")


def test_mixture_always_returns_exactly_total():
    per = {"a": [[1]], "b": [[2]], "c": [[3]], "chat": [[4]]}
    shares = {"a": 0.045, "b": 0.045, "c": 0.045, "chat": 0.865}
    for total in [1, 7, 16, 101]:
        assert len(mixture(per, shares, total, seed=0)) == total


def test_chunk_align_groups_new_tokens_with_their_pieces():
    from wazocore.vx_core import chunk_align

    # bytes: "ndị" (4) + " obodo" (6)  vs  "nd" (2) + "ị" (2) + " ob" (3) + "odo" (3)
    assert chunk_align([4, 6], [2, 2, 3, 3]) == [(0, 1, 0, 2), (1, 2, 2, 4)]
    assert chunk_align([2, 2], [2, 2]) == [(0, 1, 0, 1), (1, 2, 1, 2)]
    assert chunk_align([3, 3], [2, 4]) == [(0, 2, 0, 2)]      # boundaries meet only at the end
    assert chunk_align([4], [3]) is None


def test_chunk_sums_and_looping():
    from wazocore.vx_core import chunk_sums, looping

    assert chunk_sums([-1.0, -2.0, -0.5], [(0, 2), (2, 3)]) == [-3.0, -0.5]
    assert looping("ahịaereereereereereereere")
    assert not looping("Ndị obodo ahụ bụ ndị ọrụ ugbo.")


def test_nfc_rows_compose_decomposed_accents():
    from wazocore.vx_core import nfc_rows

    rows = nfc_rows([{"lang": "yo", "output": "pé", "reference": "é"}])
    assert rows == [{"lang": "yo", "output": "pé", "reference": "é"}]


def test_sign_test_is_exact_and_two_sided():
    from wazocore.vx_core import sign_test

    assert sign_test(0, 0) == 1.0 and sign_test(3, 3) == 1.0
    assert sign_test(1, 7) == sign_test(7, 1) == pytest.approx(18 / 256)


def test_char_lengths_can_count_characters_instead_of_bytes():
    pieces = {1: "ị".encode(), 2: b"na", 3: b"\xe1\xbb"}  # 3 is the first two bytes of a three-byte letter
    assert char_lengths({9: [1, 2]}, pieces.get) == {9: [3, 2]}
    assert char_lengths({9: [1, 2, 3]}, pieces.get, "chars") == {9: [1, 2, 1]}


def test_norm_calibrated_sets_every_row_to_the_target_norm():
    from wazocore.vx_core import norm_calibrated

    rows = norm_calibrated(np.array([[3.0, 4.0], [0.3, 0.4]], dtype=np.float32), 2.0)
    assert np.allclose(np.linalg.norm(rows, axis=1), 2.0) and np.allclose(rows[0], rows[1])


def test_mbr_pick_prefers_the_consensus_and_the_earliest_on_ties():
    from wazocore.vx_core import mbr_pick

    overlap = lambda a, b: len(set(a.split()) & set(b.split()))
    assert mbr_pick(["x y z", "a b c", "a b d", "a b e"], overlap) == 1
    assert mbr_pick(["same", "same", "same"], overlap) == 0
    assert mbr_pick(["greedy", "other"], overlap) == 0  # two candidates cannot outvote the greedy reply


def test_merge_table_lists_what_may_not_follow_each_token():
    from wazocore.vx_core import merge_table

    vocab = {"a": 0, "b": 1, "c": 2, "ab": 3, "abc": 4, "ac": 5}
    new = [{"token": "ab", "left": "a", "right": "b"}, {"token": "abc", "left": "ab", "right": "c"}, {"token": "ac", "left": "a", "right": "c"}]
    offsets, rights = merge_table(new, vocab, 6)
    assert rights[offsets[0]:offsets[1]] == [1, 2] and rights[offsets[3]:offsets[4]] == [2] and offsets[1] == offsets[2] == offsets[3]


def test_smol_pairs_read_sentence_and_document_records_in_nfc():
    from wazocore.vx_core import smol_pairs

    assert smol_pairs({"id": 1, "sl": "en", "tl": "yo", "src": " Good morning. ", "trg": "É "}) == [("Good morning.", "É")]
    doc = {"id": "t", "srcs": ["One.", "", "Three."], "trgs": ["Ọkan.", "x", "Mẹta."], "sep": []}
    assert smol_pairs(doc) == [("One.", "Ọkan."), ("Three.", "Mẹta.")]


def test_word_ngrams_normalise_and_skip_short_text():
    from wazocore.vx_core import word_ngrams

    decomposed = "Ọmọ".replace("Ọ", "Ọ")  # letter + combining dot below
    assert word_ngrams(f"{decomposed} lọ si ile", 2) == word_ngrams("ọmọ LỌ si ile", 2)
    assert word_ngrams("one two three", 4) == set()
    assert len(word_ngrams(" ".join(str(i) for i in range(15)), 13)) == 3


def test_aya_messages_need_both_sides():
    from wazocore.vx_core import aya_messages

    assert aya_messages({"inputs": " Kí ni? ", "targets": "Bẹ́ẹ̀ni"}) == [
        {"role": "user", "content": "Kí ni?"}, {"role": "assistant", "content": "Bẹ́ẹ̀ni"}]
    assert aya_messages({"inputs": "q", "targets": "  "}) is None
    assert aya_messages({"inputs": None, "targets": "a"}) is None


def test_fill_share_stops_at_target_and_counts_repeats():
    from wazocore.vx_core import fill_share

    picked, repeats = fill_share([5, 5, 5, 5], 12, seed=0)
    assert len(picked) == 3 and len(set(picked)) == 3 and repeats == 0
    picked, repeats = fill_share([5, 5], 21, seed=0)
    assert len(picked) == 5 and repeats == 2 and picked[:2] == picked[2:4]
    assert fill_share([3, 4], 7, seed=1) == fill_share([3, 4], 7, seed=1)
    assert fill_share([1], 0, seed=0) == ([], 0) and fill_share([0, 0], 5, seed=0) == ([], 0)
