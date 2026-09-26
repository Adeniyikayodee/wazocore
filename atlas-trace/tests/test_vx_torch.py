"""The PyTorch expansion recipe on the tiny random Llama-3 fixture (CPU)."""

from pathlib import Path

import numpy as np
import pytest

TINY = Path(__file__).resolve().parent / "fixtures/tiny"


@pytest.fixture()
def tiny():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    return transformers.LlamaForCausalLM.from_pretrained(TINY, torch_dtype=torch.float32)


def test_expand_appends_initialised_rows_and_keeps_originals(tiny):
    from atlas_trace.vx_torch import expand_model

    emb0 = tiny.get_input_embeddings().weight.detach().clone()
    head0 = tiny.get_output_embeddings().weight.detach().clone()
    parts = {256: [3, 5], 257: [7, 8, 9]}
    m = expand_model(tiny, parts, {256: [1.0, 3.0], 257: [1.0, 1.0, 1.0]})
    emb, head = m.get_input_embeddings().weight, m.get_output_embeddings().weight
    assert emb.shape[0] == head.shape[0] == 258 and m.config.vocab_size == 258
    assert np.allclose(emb[:256].detach(), emb0)
    assert np.allclose(emb[256].detach(), (emb0[3] + emb0[5]) / 2, atol=1e-6)
    assert np.allclose(head[256].detach(), 0.25 * head0[3] + 0.75 * head0[5], atol=1e-6)


def test_training_moves_only_new_rows_and_chosen_layers(tiny):
    import torch

    from atlas_trace.vx_torch import expand_model, lm_loss, set_trainable

    m = expand_model(tiny, {256: [3, 5], 257: [7, 8]}, {256: [1.0, 1.0], 257: [1.0, 1.0]})
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    params = set_trainable(m, 256, [0])
    opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.0)
    batch = torch.tensor([[0, 256, 257, 10, 256, 11, 257, 12]] * 2)
    losses = []
    for _ in range(5):
        loss = lm_loss(m, batch)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
    assert losses[-1] < losses[0]
    after = dict(m.named_parameters())
    emb = "model.embed_tokens.weight"
    assert torch.equal(after[emb][:256], before[emb][:256]), "original rows must not move"
    assert not torch.equal(after[emb][256:], before[emb][256:]), "new rows must train"
    assert torch.equal(after["lm_head.weight"][:256], before["lm_head.weight"][:256])
    assert not torch.equal(after["model.layers.0.mlp.up_proj.weight"], before["model.layers.0.mlp.up_proj.weight"])
    assert torch.equal(after["model.layers.1.mlp.up_proj.weight"], before["model.layers.1.mlp.up_proj.weight"])


def test_generation_stats_counts_new_tokens_and_stop():
    from atlas_trace.vx_torch import generation_stats

    s = generation_stats([5, 130000, 130001, 9, 2], canonical=2, first_new=128256, eos={2})
    assert s == {"stopped": True, "tokens": 4, "new_tokens": 2, "canonical_tokens": 2, "vs_canonical": 2.0}
    assert not generation_stats([5, 6], 2, 128256, {2})["stopped"]


def test_windows_pack_without_overlap_or_padding():
    from atlas_trace.vx_torch import windows

    assert windows(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5]]
    assert windows([1, 2], 3) == []


def test_prompts_name_the_language_and_carry_the_sentence():
    from atlas_trace.vx_torch import reverse_prompts, translation_prompts

    assert translation_prompts(["Hi."], "Hausa")[0].startswith("Translate the following English text into Hausa.")
    assert reverse_prompts(["Sannu."], "Hausa")[0].endswith("\n\nSannu.")


def test_score_rows_counts_per_language():
    from atlas_trace.vx_torch import score_rows

    rows = [{"lang": "yo", "output": "ab�", "reference": "ab", "stopped": True, "tokens": 2, "new_tokens": 1},
            {"lang": "ha", "output": "xyz", "reference": "xyz", "stopped": False, "tokens": 3, "new_tokens": 0}]
    r = score_rows(rows, lambda hyp, ref: float(len(hyp)))
    assert r["yo"] == {"n": 1, "chrf++": 1.0, "stopped": 1, "tokens": 2, "chars": 3, "new_tokens": 1, "invalid_utf8": 1}
    assert r["ha"]["stopped"] == 0 and r["ig"]["n"] == 0


def test_wiki_sentences_are_deterministic_and_bounded():
    from atlas_trace.vx_torch import wiki_sentences

    text = "Short. " + ". ".join(f"This is sentence number {i} about a topic" for i in range(50)) + "\nAnother line that is long enough to keep"
    a = wiki_sentences(text, 10, seed=1)
    assert a == wiki_sentences(text, 10, seed=1) and a != wiki_sentences(text, 10, seed=2)
    assert len(a) == 10 and all(40 <= len(x) <= 301 and x.endswith(".") for x in a)
    assert "Short." not in a


def test_paired_bootstrap_sees_a_consistent_improvement():
    from atlas_trace.vx_torch import paired_bootstrap

    refs = ["x"] * 20
    score = lambda hyp, ref: sum(h == r for h, r in zip(hyp, ref)) / len(ref) * 100
    r = paired_bootstrap(["y"] * 10 + ["x"] * 10, ["x"] * 20, refs, score, rounds=200)
    assert r["delta"] == 50.0 and r["ci95"][0] > 0 and r["p_worse"] == 0.0


def test_wiki_sentences_can_drop_english():
    from atlas_trace.vx_torch import wiki_sentences

    text = "This is one of the English sentences in the dump\nỌ natara mmachibido iwu nke abụọ maka afọ asatọ"
    assert wiki_sentences(text, 5, 0, drop_english=True) == ["Ọ natara mmachibido iwu nke abụọ maka afọ asatọ."]
    assert len(wiki_sentences(text, 5, 0)) == 2
