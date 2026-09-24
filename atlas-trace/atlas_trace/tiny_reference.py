"""Tiny random Llama-3 + PyTorch reference logits, for validating atlas-rt's
forward pass exactly (PROJECT-PLAN-B §3.5: dense f32 path must match).

The config is chosen so every code path is exercised: GQA (4 query heads
share 2 KV heads) and all three llama3 RoPE frequency bands (unchanged,
smoothed, divided) at head_dim 16 with original context 64.

Usage:
    uv run --project atlas-trace --group ref python -m atlas_trace.tiny_reference \
        --out crates/atlas-rt/tests/fixtures/tiny
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

TINY = dict(
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    vocab_size=256,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    rope_scaling={
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
    },
    max_position_embeddings=512,
    initializer_range=0.1,
    tie_word_embeddings=False,
)


def token_ids(n: int, vocab: int, seed: int) -> list[int]:
    """Deterministic pseudo-random ids (LCG), independent of torch's RNG."""
    ids, x = [], seed
    for _ in range(n):
        x = (1103515245 * x + 12345) % (1 << 31)
        ids.append(x % vocab)
    return ids


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(args.seed)
    config = LlamaConfig(**TINY, attn_implementation="eager", torch_dtype="float32")
    model = LlamaForCausalLM(config).eval()
    # Norm weights init to 1; perturb them so the test sees their effect.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.add_(0.2 * torch.randn_like(p))

    ids = token_ids(args.tokens, TINY["vocab_size"], args.seed + 1)
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0]

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True)
    reference = {
        "tokens": ids,
        "logits": logits.tolist(),
        "argmax": logits.argmax(-1).tolist(),
        "torch": torch.__version__,
    }
    (args.out / "reference.json").write_text(json.dumps(reference) + "\n")
    print(f"wrote {args.out} ({len(ids)} tokens)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
