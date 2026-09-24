# Project Plan B v2: Language-Calibrated Sparse Inference for N-ATLAS (Rust)

Revised 24 Sep 2026 after a first-principles review of v1. Target: NAIC 2026, **Problem Statement 1: Developer Infrastructure**. Deadline **12 Oct 2026, 11:59 PM WAT**.

See also:
- `~/Desktop/PROJECT-PLAN.md`: Plan A, the safer fallback (lossless speculative decoding)
- `NAIC-2026-and-N-ATLAS-notes.md`: challenge rules, model cards, licenses

All hardware figures marked *(est.)* are estimates to be replaced by day-1 measurements. The byte arithmetic in §2 is exact given the stated bit-widths.

---

## 0. Verdict on v1

| v1 claim | Verdict | Why |
|---|---|---|
| "Run 8B N-ATLAS on a 3–4 GB phone by streaming from storage" | **Cut** | With ~2 GB usable app memory *(est.)*, only ~18% of the MLP weights fit in RAM. That needs a ≥95% cache hit rate from an 18% cache, which is implausible for SwiGLU models under TEAL. Physics, not engineering effort, rules it out (§2.4). |
| "PowerInfer-2 shows up to 27.8×" | **Doesn't transfer** | That result relied on (a) ReLU-fied models with ~90% activation sparsity, (b) flagship UFS 4.0 storage, (c) an NPU. N-ATLAS is SwiGLU; training-free TEAL gives 40–50% sparsity. Budget phones have eMMC/UFS 2.x and no usable NPU path. |
| "Per-language cache is the novelty" | **Keep, but demote** | Plausible but unmeasured. It may be a small effect (language-specific neurons are a small fraction of all neurons). Becomes a hypothesis tested by trace simulation before any engine work depends on it. |
| "Per-language sparsity thresholds" | **Promote to core** | Cheap, testable, and addresses a real unmeasured gap: TEAL was calibrated on English only. English thresholds applied to Yoruba give a *different effective sparsity*: either quality loss or speed disparity. Either outcome is a finding. |
| "Sparse 4-bit CPU kernel" | **Keep: the engineering core** | TEAL has GPU kernels only. [2511.04477](https://arxiv.org/abs/2511.04477) solved sparsity × group-quantization on GPU only. A CPU/ARM version for SwiGLU with training-free sparsity is genuinely missing. |
| Neuralink co-activation layout | **Cut** | A per-language disk layout means 3 copies of ~3.2 GB of MLP weights. Stretch at most. |
| Missing from v1 | **Added** | Prefill cost (tens of seconds on phones), words/sec parity metric, code-switching, knowledge-retention eval (IrokoBench), a trace-driven simulator, honest baseline vs llama.cpp, KV snapshots, device tiers, decision gates. |

**Re-scoped thesis:**

> Training-free activation sparsity makes N-ATLAS decode 1.4–1.8× faster on CPU. But calibrating it on English creates a hidden, measurable disparity for Yoruba, Hausa and Igbo. We quantify that disparity, fix it with per-language calibration, and ship a Rust CPU/ARM engine that delivers the speed-up with per-language quality guarantees on real edge hardware. Streaming weights from storage for 6 GB phones is a gated stretch goal, decided by a simulator, not by hope.

---

## 1. The claims we will make, and how each is falsifiable

| # | Claim | Evidence required | Kill / downgrade condition |
|---|---|---|---|
| C1 | English-calibrated TEAL gives different effective sparsity and larger quality loss on yo/ha/ig than on en | Per-language effective sparsity + KL/chrF++/AfriMMLU at matched thresholds | No difference beyond CI → report as a null result (still valuable), drop "disparity" framing |
| C2 | Per-language calibration closes most of that gap at equal sparsity | Same metrics with per-language thresholds | Gap closure < 50% → report; keep English thresholds |
| C3 | Rust sparse engine decodes faster than llama.cpp Q4_K_M on the same device at the quality gate | Words/sec and tokens/sec, median of ≥5 runs, 95% CI, same prompts | < 1.2× on Pi 5 by Gate G2 → ship as research tool + study |
| C4 *(stretch)* | Per-language cache beats a language-agnostic cache for storage streaming | Trace-simulated byte-hit-rate vs LRU/LFU/global-static/Belady optimum | Simulated ≥ 2 tok/s on a 6 GB-class device not reached → streaming stays out |

---

## 2. Physics first: the byte budget

N-ATLAS = Llama-3-8B shape: hidden 4,096 · FFN 14,336 · 32 layers · 32 query / 8 KV heads · head dim 128 · vocab 128,256.

### 2.1 Parameters and bytes (4.5 bits/weight ≈ Q4 group-wise; lm_head at Q6_K 6.56 bpw)

| Part | Params | Bytes | Read per decoded token? |
|---|---|---|---|
| Attention Q/K/V/O | 1.342 B | 0.755 GB | Yes (sparsifiable) |
| MLP gate/up/down | 5.637 B | 3.171 GB | Yes (sparsifiable): **73% of per-token bytes** |
| lm_head | 0.525 B | 0.431 GB | Yes (dense) |
| Token embedding | 0.525 B | 0.296 GB | **No**: one 2.3 KB row per token, so it can live on storage **losslessly** |
| **Total** | 8.03 B | 4.65 GB | **4.36 GB read per token (dense)** |

### 2.2 Decode is memory-bandwidth-bound

| Sparsity (attn + MLP inputs) | Bytes/token | Byte reduction | tok/s @ 8 GB/s | @ 10 GB/s | @ 13 GB/s |
|---|---|---|---|---|---|
| 0% (dense) | 4.36 GB | 1.00× | 1.84 | 2.30 | 2.98 |
| 25% | 3.38 GB | 1.29× | | | |
| 40% | 2.79 GB | 1.56× | 2.87 | 3.59 | 4.67 |
| 50% | 2.39 GB | 1.82× | 3.34 | 4.18 | 5.43 |

Effective CPU memory bandwidth for Pi 5 / budget phone SoCs is **8–13 GB/s *(est.)***; measure it on day 1. These are ceilings. The kernel's job is to reach ≥ 80% of them.

**Report words/sec, not just tokens/sec.** Yoruba needs more tokens per word than English (1.85× on GPT-4o's tokenizer; measure N-ATLAS's own on day 1). Tokens/sec hides the language tax; words/sec and chars/sec expose it. This is the parity metric.

### 2.3 Prefill is compute-bound (and v1 ignored it)

2 × 8.03 B × N FLOPs. At ~150 GOPS effective int8 on a phone/Pi CPU *(est.)*:

| Prompt tokens | Work | Time to first token |
|---|---|---|
| 40 (short Yoruba turn) | 0.64 TOP | ~4 s |
| 200 (system prompt + turn) | 3.21 TOP | ~21 s |

Sparsity barely helps prefill: the union of active channels across many tokens is nearly dense. **Mitigation, lossless:** *KV snapshots.* Precompute the system prompt's KV cache once and save it to disk (128 KB/token fp16, so a 150-token system prompt = 19 MB). Load it at session start. Only the user turn is prefilled.

### 2.4 Storage streaming: where it works and where it can't

App memory budgets *(est.)*: 4 GB phone ≈ 2 GB · 6 GB phone ≈ 3 GB · 8 GB phone ≈ 4.5 GB. Resident non-MLP set = attention 0.755 + lm_head 0.431 + KV (q8, 2k ctx) 0.134 + runtime ~0.1 = **1.42 GB**.

| Device class | MLP cache that fits | Verdict |
|---|---|---|
| 4 GB phone | 0.58 GB = **18% of MLP** | **Out of scope for 8B.** Would need ≥95% byte-hit-rate from an 18% cache. |
| 6 GB phone | 1.58 GB = **50% of MLP** | **Stretch, gated (G1).** At 50% sparsity, MLP bytes needed/token = 1.59 GB. Misses/token: h=0.8 → 317 MB (1.06 s @ 300 MB/s eMMC, 0.32 s @ 1 GB/s UFS); h=0.9 → 159 MB; h=0.95 → 79 MB. **Viable only with UFS storage and h ≥ 0.9.** |
| 8 GB phone, Pi 5 8 GB, 8–16 GB laptop | Everything resident | **Primary target.** Sparsity = pure speed. |

---

## 3. Architecture

### 3.1 Components

| Component | Language | Role |
|---|---|---|
| `atlas-probe` | Rust | Day-1 hardware probes: memory bandwidth (STREAM-style), storage read profile (seq + 4/16/64 KB random), core topology. Emits a JSON hardware manifest. |
| `atlas-trace` | Python (PyTorch, Colab) | TEAL hooks on N-ATLAS; per-language calibration; quality eval; **activation-trace recorder** (per token, per layer, per matrix: active-channel bitmaps). |
| `atlas-sim` | Rust | **Trace-driven simulator.** Replays traces against cache policies (LRU, LFU, global-static, per-language-static, Belady optimum) and a storage/bandwidth model. Predicts bytes/token and tok/s per device class. Decides Gate G1 before any streaming code exists. |
| `atlas-convert` | Rust | bf16 safetensors → `.satl` weight file (column-major sparse-friendly quantization); per-language profile builder. |
| `atlas-rt` | Rust | The runtime: forward pass, sparse kernels, per-language profiles, KV snapshots, CLI + C ABI + Android build. |

### 3.2 Weight format `.satl` (designed for sparse reads)

- Header: magic, format version, source-model SHA-256, quant scheme, tensor table. Every tensor segment **4 KiB-aligned** for page/flash-friendly reads.
- **MLP and attention projections stored column-major:** each *input channel's* column is contiguous and quantized in groups of 32 along the *output* dimension (f16 scale + min per group).
  - TEAL uses **input sparsity** (y = W·x, most of x zeroed), so a skipped channel means a skipped contiguous column: no gather, prefetcher-friendly.
  - **gate and up share the same input x**, so channel *i*'s gate column and up column are stored **adjacent as one bundle** (2 × 14,336 × 4.5 bits ≈ 16 KB): one read serves both.
  - down_proj: columns over the 14,336 intermediate channels (4,096 × 4.5 bits ≈ 2.3 KB each).
- lm_head: row-major dense Q6 (always fully read).
- Token embedding: separate segment, read by row on demand (not resident).
- **Quantize from bf16, never from GGUF.** Converting Q4_K_M → our layout would stack two quantization errors.
- **Isolate variables:** measure *dense quality of `.satl`* vs llama.cpp Q4_K_M before adding sparsity. Otherwise format loss and sparsity loss get confused. Stretch: activation-aware scaling (AWQ-style) using per-language activation statistics (links to Plan A).

### 3.3 Per-language profiles (tiny, separate from weights)

- `profile-{yo,ha,ig,en,mix}.bin`: per layer × per matrix magnitude thresholds (TEAL-style; runtime mask = |x| > t, no sorting) + (stretch) hot-set bitmaps and frequencies for caching.
- A few KB each. They're tuned and shipped without re-exporting weights.
- **Selection:** explicit (the app knows which ASR model ran), otherwise a lightweight language ID on the prompt.
- **Code-switching** (Yoruba–English mixing is normal): a `mix` profile calibrated on code-switched text, plus a fallback to the most conservative (least sparse) thresholds when language ID is uncertain.

### 3.4 Kernels

- **Dense Q4 GEMV** (attention when not sparsified, lm_head): row-major, int8 activations, ARM `SDOT` (dotprod; Cortex-A76/A55 support it) and AVX2 `VPMADDUBSW` paths.
- **Sparse column-axpy GEMV** (the novel part):
  1. Build the active-channel index list: |x_i| > t_i. Quantize the active activations to int8.
  2. For each thread's **output row range**, loop over active columns: unpack 4-bit group → int8, widening multiply-accumulate into int32 accumulators, apply group scale/min, accumulate f32.
  3. Process active columns **4 at a time and transpose in registers** (NEON `TRN1/TRN2`/`ZIP`) so `SDOT` can reduce over 4 columns per lane. This recovers dot-product throughput for an axpy access pattern.
- **Threading:** split by output-row blocks, so there are no write conflicts and each thread streams a contiguous slice of every active column. On big.LITTLE phones, measure big-cores-only vs all-cores; slow A55 cores often straggle. Use small work chunks with stealing.
- **Correctness ladder:** scalar reference (bit-exact spec) → NEON → AVX2. Property tests over random shapes, sparsity masks and quant blocks. **At 0% sparsity the sparse kernel must equal the dense kernel within float tolerance.** Fixed reduction order in test mode for determinism.

### 3.5 Forward pass

Minimal Llama-3 op set: RMSNorm, RoPE (Llama-3 scaling), GQA attention with a KV cache (q8 option), SwiGLU MLP, sampling. Validate end to end against PyTorch reference logits on fixed prompts: **100% greedy top-1 agreement on the dense f32 path**, reported KL on the quantized path.

### 3.6 Storage streaming (stretch, only if Gate G1 passes)

- Resident set: attention + lm_head + KV + per-language hot MLP bundles, `mlock`ed where permitted. Cold bundles via a `pread` thread pool into a slab cache.
  - io_uring is generally unavailable to Android apps; don't depend on it.
  - Use `O_DIRECT` with aligned buffers where supported, so the page cache doesn't double-buffer.
- Cross-layer read-ahead: layer *l*'s activations predict layer *l+1..l+2* misses (ActiveFlow-style).
- **Cache-aware masking** ([DIP, 2412.01380](https://arxiv.org/abs/2412.01380)): when a channel is near the threshold, prefer keeping it if it's cached. Raises hit rate at a small quality cost.
  - **Our twist:** language-agnostic cache-aware masking likely hurts yo/ha/ig most, because the cache fills with English-hot channels. Per-language caches should fix that. This is where C4 lives.
- Respond to Android memory pressure (`onTrimMemory`) by shrinking the cache instead of being killed.

---

## 4. Research protocol

### 4.1 Data (Unicode NFC everywhere)

| Use | Source |
|---|---|
| Calibration (thresholds, profiles) | FLORES-200 **dev**, Aya, MENYO-20k train (yo), Leipzig corpora (ha/ig) |
| Evaluation: text | FLORES-200 **devtest** (yo/ha/ig/en), MENYO-20k test |
| Evaluation: knowledge retention | **IrokoBench** ([2406.03368](https://arxiv.org/abs/2406.03368)): AfriMMLU / AfriXNLI in yo/ha/ig. Answers the warning in [2408.14398](https://arxiv.org/abs/2408.14398) that pruning keeps perplexity but loses knowledge. |
| Evaluation: code-switching | ~50 hand-written Yoruba–English mixed prompts (native speaker) |
| Human | Blind A/B, 30 prompts × 3 languages, sparse vs dense, native speakers |

Calibration and test splits are never mixed. N-ATLAS's training data is undisclosed, so possible FLORES contamination is noted as a limitation.

### 4.2 Metrics

- **Fidelity to the dense model** (primary, cheap, sensitive): mean per-token **KL(dense ‖ sparse)** on teacher-forced text; greedy top-1 agreement.
- **Task quality:** chrF++ (en→yo/ha/ig translation), AfriMMLU accuracy, perplexity (secondary; known to mislead).
- **Speed:** time to first token, decode tokens/sec, **words/sec, chars/sec**, peak RSS, bytes read from storage per token (streaming), energy (Pi via USB power meter).
- **Disparity:** (quality loss or effective sparsity for language L) ÷ (same for English).

### 4.3 Quality gates (initial targets, recalibrated on day 3 from the dense model's own variance)

| Metric | Gate per language |
|---|---|
| Mean KL(dense ‖ sparse) | ≤ 0.05 nats/token |
| chrF++ drop vs dense | ≤ 1.0 point |
| AfriMMLU drop vs dense | ≤ 2.0 points |
| Native A/B | Non-inferior: sparse preferred-or-tied ≥ 40% |

Sparsity level is chosen **per language** as the highest that passes all gates. It's published as a quality-vs-speed curve, not one number.

### 4.4 Statistics and reproducibility

- Paired bootstrap (1,000 resamples) over sentences for quality deltas; median + p95 over ≥5 runs for speed, warm-up discarded.
- Hardware manifest from `atlas-probe` in every result file: device, SoC, RAM, storage type, OS/kernel, governor, thermal state, commit hashes.
- Pi 5: `performance` CPU governor and active cooling. Phones: can't pin clocks, so record thermals and use cool-down intervals.
- Every result table regenerates from raw logs with one command.

---

## 5. Device tiers

| Tier | Device | Role |
|---|---|---|
| **A** | x86 laptop 8–16 GB (AVX2) | Development, CI, first speed numbers |
| **B** | **Raspberry Pi 5, 8 GB** (Cortex-A76) | **Primary benchmark.** Same big core as many budget phone SoCs, but fully reproducible: fixed clocks, no Android memory killer, cheap for beta testers to replicate. |
| **C** | Android phone, 8 GB (Termux + `cargo-ndk`) | Real-phone proof |
| **D** *(stretch)* | Android phone, 6 GB, UFS storage | Storage streaming, only if Gate G1 passes |
| — | 4 GB phones | **Out of scope for 8B** (§2.4) |

---

## 6. Plan with decision gates (24 Sep → 12 Oct)

| Phase | Dates | Work | Exit gate |
|---|---|---|---|
| **P0 Physics & baselines** | 24–25 Sep | `atlas-probe` on laptop / Pi 5 / phone(s). llama.cpp N-ATLAS Q4_K_M baseline per language: prefill tok/s, decode tok/s, words/s, RSS. Tokens per word per language with N-ATLAS's tokenizer. **Recruit the 2 beta testers now.** | **G0 (25 Sep):** device tiers fixed from measured bandwidth and storage |
| **P1 Science (GPU)** | 25–28 Sep | `atlas-trace`: TEAL with English vs per-language calibration; effective sparsity per language; KL / chrF++ / AfriMMLU. Record traces (~20k tokens/language). `atlas-sim`: hit rates per matrix and policy vs Belady; predicted tok/s per tier. | **G1 (28 Sep):** per-language sparsity levels chosen; **streaming in/out** (in only if simulated ≥ 2 tok/s on Tier D with h ≥ 0.9) |
| **P2 Engine** | 28 Sep – 4 Oct | `atlas-convert` + `.satl`; dense Rust forward pass matches PyTorch; `.satl` dense quality vs Q4_K_M; scalar → NEON → AVX2 sparse kernels; profiles; KV snapshots; CLI | **G2 (2 Oct):** Pi 5 decode ≥ 1.2× llama.cpp at the quality gates. If not: ship as study + reference engine and put the remaining days into rigour and docs. |
| **P3 Devices** | 4–7 Oct | Android build; Tier C runs; code-switching profile; (if G1 passed) streaming on Tier D | **G3 (7 Oct):** feature freeze |
| **P4 Validation & packaging** | 7–10 Oct | Beta testers reproduce on their hardware; native-speaker A/B; docs; benchmark report; 3–5 min video | — |
| Buffer | 11 Oct | Fixes, final runs | Submit by 12 Oct 11:59 PM WAT |

**Critical path:** P1 traces → G1 → P2 kernels. Everything else can slip without killing the submission.

---

## 7. Deliverables and NAIC mapping

| NAIC requirement | Deliverable |
|---|---|
| Working artefact | Public repo: `atlas-rt` (CLI, library, C ABI, Android build), `atlas-convert`, `.satl` N-ATLAS weights, per-language profiles, `atlas-probe`, `atlas-sim`, `atlas-trace` |
| N-ATLAS integration evidence | Engine runs N-ATLAS natively; profiles are calibrated on N-ATLAS activations; logits parity report vs the reference model |
| Real-world validation (PS1: ≥ 2 external beta testers) | Testers reproduce the Pi 5 / laptop benchmarks and integrate the library; native-speaker A/B study |
| Technical documentation | Architecture, `.satl` spec, kernel design, benchmark methodology, hardware manifests |
| Video (3–5 min) | Same Pi 5 or phone: llama.cpp vs `atlas-rt` on a Yoruba prompt, words/sec counters, then the per-language quality table |
| Team profile / registration | Per chosen track |

**Evaluation criteria coverage:**
- technical rigour: gates, tests, statistics
- N-ATLAS integration depth: native engine and calibration
- real-world validation: devices, testers, native speakers
- impact: speed parity for Nigerian languages
- scalability: runs on commodity CPUs with no GPU
- team capability: shown by the artefact itself

---

## 8. Risk register

| Risk | Likelihood | Impact | Early signal | Response |
|---|---|---|---|---|
| Rust dense path much slower than llama.cpp, eating the sparsity gain | Medium | High | Day-8 dense benchmark | Profile; optimize the dense GEMV first; report speed-up vs both our dense and llama.cpp honestly |
| Per-language calibration shows no effect (C1/C2 null) | Medium | Medium | G1 | Publish the null result; keep English thresholds; the engine still stands |
| Quality gates fail above ~25% sparsity for yo | Medium | Medium | P1 | Lower sparsity for yo (≈1.29× ceiling); report honestly per language |
| Column-axpy kernel compute-bound on A55/A76 | Medium | Medium | Early microbench | Register transpose + `SDOT`; restrict to big cores; tune column block size |
| Storage too slow for streaming | High | Low (stretch) | P0 / G1 | Streaming stays out; Tiers B/C carry the submission |
| Colab GPU limits (bf16 8B needs a ≥24 GB GPU: L4/A100) | Medium | High | Day 1 | Secure Colab Pro / a cloud GPU on day 1; keep traces small |
| Beta testers unavailable late | Medium | High | Day 1 | Recruit on day 1; Pi 5 tier makes replication cheap |
| License | Low | High | — | `.satl` weights and profiles are N-ATLaS derivatives: same license, required attribution, "Powered by Awarri." if renamed, 1,000 active-user cap |

---

## 9. Decisions log (what we are not doing, and why)

| Not doing | Reason |
|---|---|
| 8B on 4 GB phones | §2.4: an 18% cache can't serve a ≥95% hit rate |
| ReLU-fication / distillation (TurboSparse, ActiveFlow's self-distillation) | Needs training compute and time we don't have; stays training-free |
| NPU offload | Vendor-specific, fragmented on budget SoCs, poor tooling |
| Per-language disk layouts (Neuralink) | Triples storage; stretch at most |
| Combining with Plan A's speculative decoding | Verifying k tokens unions their active channels and erodes sparsity; would need its own study |
| Patching llama.cpp instead of our own runtime | Considered (Rust kernels via C ABI into ggml custom ops). Rejected: the `.satl` layout and profile/cache control need a runtime we own. llama.cpp stays the baseline. |

---

## 10. Day-1 actions (24 Sep)

- [ ] Secure GPU access (Colab Pro L4/A100 or equivalent); accept the `NCAIR1/N-ATLaS` gated-repo conditions; download the bf16 weights
- [ ] Order or borrow a Raspberry Pi 5 8 GB with active cooler and USB power meter; identify an 8 GB Android phone (and optionally a 6 GB UFS one)
- [ ] Message 2–3 potential beta testers (developers with a laptop or Pi); confirm availability 7–10 Oct
- [ ] Write `atlas-probe`: bandwidth + storage + topology → JSON manifest
- [ ] Build llama.cpp; N-ATLAS Q4_K_M baseline on laptop (and Pi/phone when available)
- [ ] Tokens-per-word audit for yo/ha/ig/en on FLORES-200 dev with N-ATLAS's tokenizer
- [ ] Line up one native speaker per language for the code-switching set and the A/B study

---

## 11. Key references

| arXiv | Title | Use |
|---|---|---|
| [2408.14690](https://arxiv.org/abs/2408.14690) | TEAL: Training-Free Activation Sparsity | Core method; Llama-3-8B: ppl 5.87 → 6.21 at 40%; 1.53–1.8× decode on GPU; English-only calibration |
| [2412.01380](https://arxiv.org/abs/2412.01380) | Dynamic Input Pruning & Cache-Aware Masking | Cache-aware masking for SwiGLU on mobile |
| [2511.04477](https://arxiv.org/abs/2511.04477) | Dynamic Sparsity in Quantized LLM Inference | Sparsity × group-quant layout (GPU); motivates `.satl` |
| [2504.08378](https://arxiv.org/abs/2504.08378) | ActiveFlow | Cross-layer read-ahead for non-ReLU models |
| [2406.06282](https://arxiv.org/abs/2406.06282) | PowerInfer-2 | Smartphone offload; relies on ReLU sparsity + flagship hardware |
| [2312.11514](https://arxiv.org/abs/2312.11514) | LLM in a Flash | Bundling, windowed caching |
| [2402.16438](https://arxiv.org/abs/2402.16438) | Language-Specific Neurons | Language neurons concentrated in top/bottom layers |
| [2402.16367](https://arxiv.org/abs/2402.16367) | Unraveling Babel | Language-specific activation frequency patterns |
| [2408.14398](https://arxiv.org/abs/2408.14398) | Limitations of Language-Targeted Pruning | Perplexity kept, knowledge lost: why we cache, never prune, and eval AfriMMLU |
| [2601.18306](https://arxiv.org/abs/2601.18306) | Calibrating Beyond English | Activation ranges differ by language |
| [2407.03211](https://arxiv.org/abs/2407.03211) | How Does Quantization Affect Multilingual LLMs? | Automatic metrics underestimate damage; human eval needed |
| [2406.03368](https://arxiv.org/abs/2406.03368) | IrokoBench | AfriMMLU / AfriXNLI for yo/ha/ig |
| [2606.24460](https://arxiv.org/abs/2606.24460) | The African Language Tax | Tokens-per-word premiums; motivates the words/sec metric |
| [2407.00088](https://arxiv.org/abs/2407.00088) | T-MAC | CPU LUT kernels; alternative dense-path optimization |
