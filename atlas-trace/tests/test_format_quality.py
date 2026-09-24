import math

import pytest

from atlas_trace.format_quality import chunk_nlls_from_running, paired_bootstrap, parse_llama_perplexity


def test_recovers_per_chunk_nll_from_running_ppl():
    per_chunk = [2.0, 3.0, 1.0]
    m = 127
    running = [math.exp(sum(per_chunk[: k + 1]) / (k + 1)) for k in range(3)]
    assert chunk_nlls_from_running(running, m) == pytest.approx(per_chunk)


def test_parses_llama_output():
    out = "perplexity: calculating...\n[1]12.5000,[2]10.2500,[3]9.1000,\nFinal estimate: PPL = 9.1000 +/- 0.3"
    assert parse_llama_perplexity(out) == [12.5, 10.25, 9.1]


def test_paired_bootstrap_centres_on_mean_difference():
    mean, lo, hi = paired_bootstrap([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 4.1])
    assert mean == pytest.approx(0.1)
    assert lo == pytest.approx(0.1) and hi == pytest.approx(0.1)
