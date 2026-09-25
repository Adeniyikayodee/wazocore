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

> **Update 24 Sep:** the implemented `.satl` stores lm_head as q8 (8.5 bpw), not Q6 (6.56 bpw). Dense bytes per token become about 4.48 GB (+3%), so the ceilings in §2.2 are about 3% optimistic until a q6 lm_head codec lands. See §12.

### 2.2 Decode is memory-bandwidth-bound

| Sparsity (attn + MLP inputs) | Bytes/token | Byte reduction | tok/s @ 8 GB/s | @ 10 GB/s | @ 13 GB/s |
|---|---|---|---|---|---|
| 0% (dense) | 4.36 GB | 1.00× | 1.84 | 2.30 | 2.98 |
| 25% | 3.38 GB | 1.29× | | | |
| 40% | 2.79 GB | 1.56× | 2.87 | 3.59 | 4.67 |
| 50% | 2.39 GB | 1.82× | 3.34 | 4.18 | 5.43 |

Effective CPU memory bandwidth for Pi 5 / budget phone SoCs is **8–13 GB/s *(est.)***; measure it on day 1. These are ceilings. The kernel's job is to reach ≥ 80% of them.

**Report words/sec, not just tokens/sec.** Yoruba needs more tokens per word than English (1.85× on GPT-4o's tokenizer; measure N-ATLAS's own on day 1). Tokens/sec hides the language tax; words/sec and chars/sec expose it. This is the parity metric.

> **Update 24 Sep (measured):** N-ATLaS's tokenizer (identical to Llama-3's 128,256-token vocabulary) needs **2.68× (yo), 2.35× (ig) and 1.99× (ha)** the tokens of English for the same FLORES content, not 1.85×. On llama.cpp this gives Yoruba 2.7 words/s against English's 6.0 at equal tokens/s. Decomposed (NFD) Yoruba costs another 19%, so the runtime must NFC-normalise its input. See §12.

### 2.3 Prefill is compute-bound (and v1 ignored it)

2 × 8.03 B × N FLOPs. At ~150 GOPS effective int8 on a phone/Pi CPU *(est.)*:

| Prompt tokens | Work | Time to first token |
|---|---|---|
| 40 (short Yoruba turn) | 0.64 TOP | ~4 s |
| 200 (system prompt + turn) | 3.21 TOP | ~21 s |

Sparsity barely helps prefill: the union of active channels across many tokens is nearly dense. **Mitigation, lossless:** *KV snapshots.* Precompute the system prompt's KV cache once and save it to disk (128 KB/token fp16, so a 150-token system prompt = 19 MB). Load it at session start. Only the user turn is prefilled.

> **Update 24 Sep (measured):** on the M4 CPU, llama.cpp prefill runs at 26–44 tok/s. The Pi 5 will be several times slower, so KV snapshots are mandatory, not optional. Yoruba's token tax makes a "40-token turn" about 15 words.

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

> **Update 24 Sep:** "groups of 32 with f16 scale + min" is 5.0 bpw, not 4.5, which contradicts §2. The implementation uses **group 64 (4.5 bpw)**; group 32 remains an option. lm_head is q8 for now. The authoritative layout is `docs/satl-format.md`.
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

> **Update 24 Sep (implemented):**
> - SDOT reduces across columns only if each column's per-group scale is first **folded into the activation** (`x_j · s_{j,g}`) and the folded values are **re-quantised to int8 per 32-column block**. That is what is implemented. It costs a mean KL of 5–8e-3 nats against f32 activations (§12), with no measurable NLL change.
> - Threading uses **2D tiles** (≤ 4,096 output rows × a column slice). The tiling depends only on the matrix shape, so logits are bit-identical at any thread count and on any device. NEON is checked bit for bit against a scalar int8 spec.
> - Dense decode on the M4 (6 threads) is 8.3 tok/s, level with llama.cpp. Single-core efficiency is still about 2.6× below llama.cpp.

### 3.5 Forward pass

Minimal Llama-3 op set: RMSNorm, RoPE (Llama-3 scaling), GQA attention with a KV cache (q8 option), SwiGLU MLP, sampling. Validate end to end against PyTorch reference logits on fixed prompts: **100% greedy top-1 agreement on the dense f32 path**, reported KL on the quantized path.

> **Update 24 Sep:** met on a tiny random Llama-3 fixture that exercises GQA and all llama3-RoPE bands: logits within 1e-4, and 100% top-1 at every position. Not yet run on N-ATLaS itself, whose f32 reference needs more than 16 GB of RAM (Colab).

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

> **Update 24 Sep (gate recalibrated, decision C + B):** the KL ≤ 0.05 target was a guess. Our TEAL implementation reproduces the paper's next-token loss (ΔNLL about +0.07–0.10 nats at 40% sparsity, against TEAL's +0.056 for Llama-3-8B), but KL at 40% is about 0.27 nats, so that gate would allow only about 10–15% sparsity.
> - **C (reported):** the full per-language curve of KL, ΔNLL and top-1 against sparsity, with calibration on English vs the language itself. C1/C2 compare languages at matched sparsity and do not depend on any gate.
> - **B (operating point):** per language, the highest sparsity whose ΔNLL does not exceed that language's Q4_K_M loss vs Q8_0 (§12: en 0.036, yo 0.008, ha 0.011, ig 0.020 nats). Sparsity must cost no more than the quantisation users already accept.
> - **A (also reported):** where KL = 0.05 falls on each curve.

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

> **Update 24 Sep:** the development machine is an **Apple M4 (arm64)**, not x86. The NEON path is therefore the tested one. Until an AVX2 kernel exists, x86 runs the scalar int8 spec (estimated < 1 tok/s), which blocks x86 beta testers.

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

- [ ] Secure GPU access (Colab Pro L4/A100 or equivalent)
- [x] Accept the `NCAIR1/N-ATLaS` gated-repo conditions; download the bf16 weights
- [ ] Order or borrow a Raspberry Pi 5 8 GB with active cooler and USB power meter; identify an 8 GB Android phone (and optionally a 6 GB UFS one)
- [ ] Message 2–3 potential beta testers (developers with a laptop or Pi); confirm availability 7–10 Oct
- [x] Write `atlas-probe`: bandwidth + storage + topology → JSON manifest
- [x] Build llama.cpp; N-ATLAS Q4_K_M baseline on laptop (Pi/phone pending hardware)
- [x] Tokens-per-word audit for yo/ha/ig/en on FLORES-200 dev with N-ATLAS's tokenizer
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

---

## 12. Findings and deviations (living log)

Every number here is regenerated in `results/TABLES.md` (`python -m atlas_trace.tables`).

### 12.1 Measured (24 Sep, Apple M4 dev machine)

| Topic | Plan assumed | Measured | Consequence |
|---|---|---|---|
| Token tax vs English | 1.85× | yo 2.68×, ig 2.35×, ha 1.99× | Words/s parity gap is larger than planned; the motivation strengthens |
| llama.cpp decode (M4, 6 threads) | – | 7.4–8.1 tok/s in every language; words/s en 6.0, yo 2.7 | Yoruba users get 45% of English's content speed |
| llama.cpp peak RSS | – | 7.2 GiB with weight repacking, 4.9 GiB with `--no-repack` (half the prefill speed) | Report both on 8 GB devices |
| `.satl` q4g64 conversion error | – | 0.091–0.10 relative RMSE on every projection; q8 0.0055; norms exact | In line with 4-bit theory; no broken tensors |
| **Dense `.satl` vs Q4_K_M** (perplexity against a Q8_0 reference, same tokens) | assumed equal | en **+0.050 nats [0.015, 0.082] worse**; ha +0.035 [0.000, 0.070]; yo and ig equal | Must be fixed in the format **before** any sparsity result (§3.2 "isolate variables"). Likely causes: Q4_K_M keeps half the layers' `attn_v`/`ffn_down` at Q6_K and uses finer scales, and int8 activations hurt English most |
| Int8 activations vs f32 | not considered | mean KL 5–8e-3 nats; top-1 92–97%; NLL change within ±0.017 | Uses 10–16% of the 0.05-nat gate; no disadvantage for yo/ha/ig |
| atlas-rt dense decode | – | 8.3 tok/s at 6 threads (llama.cpp parity); 2.8 tok/s on 1 thread (llama.cpp 7.5) | Single-core efficiency matters on the 4-core Pi |
| Storage streaming (synthetic traces) | – | h ≥ 0.9 at a 50% cache needs highly concentrated activations, even for Belady | Early sign the stretch goal stays out; real traces decide G1 |

### 12.1b Negative result: speculative decoding does not fix the token tax on CPU (24 Sep)

Hypothesis: most of Yoruba's, Hausa's and Igbo's extra tokens continue a word already started (64%, 54% and 62% of tokens, against 23% for English), so a cheap language drafter could narrow the words/s gap.

| Test | Result |
|---|---|
| Offline simulation (exact under greedy decoding), n-gram drafter from FLORES dev + 6M chars of Wikipedia | Tokens per verification step: yo 1.67, ha 1.60, ig 1.83, en 1.26. Promising on paper |
| **Real llama.cpp**, `--spec-type ngram-cache` with that corpus, **paired A/B** (8 prompts per language) | Speed ratio medians: yo 0.87, ha 0.60, ig 1.32, en 0.89. All CIs include 1. Real acceptance was 31% of drafted tokens |
| Real llama.cpp, untrained Llama-3.2-1B draft | Acceptance 6–23%; decode fell to about 1.6 tok/s |

**Decision:** fails the ≥ 1.25× Yoruba criterion, so it is dropped. Batch verification on CPU costs more than cheap drafts save. Only 21/32 outputs were identical, because llama.cpp's batched verification changes arithmetic enough to flip greedy choices. Scripts: `atlas-trace/atlas_trace/spec_probe.py` and `paired_spec.py`; data: `results/spec/`.

**Measurement rule (applies to every speed claim from now on):** the dev machine is a **fanless MacBook Air** whose decode speed swung from 17 to 7 tok/s within 20 minutes. Sequential A/B runs are invalid (the sequential files `results/spec/base1|ngram|base2.json` show this). **Every speed comparison must use the paired, interleaved design** of `paired_spec.py`.

### 12.1c TEAL on N-ATLaS: C1 supported, C2 null (25 Sep, local, q4 model)

Calibration used 8,192 FLORES dev tokens per language. Evaluation used 8 chunks × 256 devtest tokens per language, fidelity measured against the dense fast path. Tables are in `results/TABLES.md`; raw data in `results/teal/eval-matrix.json`.

- **C1 (supported):**
  - English thresholds over-prune the **attention-output** input for yo/ha/ig by 15–20% relative (for example 47% instead of 40%); the other inputs are within 1–3 points.
  - At 40%, the Nigerian languages lose more next-token quality than English: ΔNLL en +0.016 [−0.032, +0.082], yo +0.051 [+0.012, +0.096], ha +0.028 [+0.000, +0.062], ig +0.084 [+0.037, +0.135].
  - At ≤ 30%, losses are near zero or negative for every language.
- **C2 (null, reported per §1):** own-language thresholds restore the target sparsity at `o`, but quality is equal or **worse** (for example yo 40%: ΔKL +0.038 [+0.015, +0.064]). The extra loss is not a calibration artefact: these languages appear intrinsically more sensitive at high sparsity, and the `o` input tolerates more pruning than a uniform rule gives it. **English thresholds are kept.** The lever this points to is per-site allocation (TEAL's greedy method), not per-language calibration.
- **Operating points:**
  - **B** (ΔNLL ≤ that language's Q4_K_M loss) = **30%** for all four (English alone would allow 40%). Igbo's upper CI bound (+0.046) exceeds its budget (0.020), so Igbo at 30% is marginal.
  - **A** (KL ≤ 0.05) = 10%.
  - Byte-reduction ceiling at 30% ≈ 1.35×.
- **Unexplained:** ΔNLL is significantly *negative* at 10–30% for en and ha (sparse gives the true token more probability than dense). The sign and alignment have been checked, but the effect needs more chunks before it is interpreted.

### 12.1d G1: storage streaming is OUT (25 Sep, real N-ATLaS traces)

The simulation used 30% sparsity with English thresholds, 4,096 devtest tokens per language, a 50% MLP cache (the 6 GB-phone budget), hot sets from dev traces, and this Mac's SSD (best case).
- **About 70% of MLP channels are active every token.**
- **Even Belady's offline optimum reaches only 70–71% byte hit rate**, far below the ≥ 90% G1 requires.
- **The best practical policy (static hot set) gets 52–53%**, a predicted **0.25 tok/s**.
- **LRU gets 0%**, because the model sweeps every layer in the same order each token.
- **C4 is null:** own-language and mixed hot sets differ by < 1 point.

Streaming stays out, as §2.4 predicted from the byte budget; the storage stretch goal is closed. `results/sim/`.

### 12.1e S4: sparse decode works, but C3 fails (25 Sep)

- **Sparse vs our own dense decode (paired in-process, P-cores):** 10% 1.02×, 20% 1.09×, **30% 1.17× [1.15, 1.18]**, 40% 1.27×.
- **Against llama.cpp (interleaved rounds, P-cores):** atlas-rt dense 0.29× [0.28, 0.30], sparse 0.34× [0.32, 0.35]. llama.cpp decodes at 12–16 tok/s; atlas-rt at 3.6–5.6.
  - **Correction:** the earlier "dense parity" (§12.1) came from sequential comparisons against llama-server and throttled llama-bench runs. It was wrong. llama.cpp's CPU kernels are about 3× faster than ours.
- **Efficiency cores:** inconclusive. llama.cpp is unstable (0.5–2.1 tok/s), atlas-rt about 1 tok/s.
- **Shipped-path quality at 30%** (one 512-token passage per language): KL 0.15–0.16, consistent with S3. ΔNLL of +0.05 for en, yo and ha exceeds the B budgets; this needs re-checking on S3's chunks.
- **Verdict:** C3 fails; the sparsity gain cannot overcome a 3× kernel gap. Per the G2 rule, the engine ships as a validated reference and study.

### 12.2 Design deviations

| Plan | Implemented | Why |
|---|---|---|
| §3.2 group 32, "4.5 bpw" | group 64 = 4.5 bpw (group 32 = 5.0 bpw) | Group 32 breaks the §2 byte budget by 11% |
| §3.2 lm_head Q6 | q8 | Simpler first codec; costs about 3% of bytes per token |
| §3.4 SDOT over 4 transposed columns | Same, plus scale folding and int8 re-quantisation per 32 columns | Per-column group scales otherwise block integer reduction |
| §3.4 threads split by output rows | 2D tiles (rows × column slices), shape-determined | gate_up's 112 KB output exceeds the A76's L1; thread-count-dependent tiling changed results |
| §5 Tier A x86 | Apple M4 | Available hardware; x86 needs an AVX2 kernel before beta testing |

### 12.3 Status against the gates (end of 24 Sep)

- **G0:** open. Needs Pi 5 and phone probe manifests.
- **G1 (28 Sep):** at risk. P1 science (TEAL calibration, traces) has not started; it needs a GPU.
- **G2 (2 Oct):** the engine is at dense *speed* parity on the M4, but dense *quality* trails Q4_K_M in English and Hausa, sparsity is not in the engine yet, and there is no Pi to measure on.

---

## 13. v3 scope lock (24 Sep evening): local-only execution

**Why:** there is no GPU, no Raspberry Pi and no phone. Everything runs on the Apple M4 development machine. Two things only the author can do: **recruit 2 external beta testers** (a NAIC PS1 requirement) and **register** for a track.

**Not the contribution (already solved):** running N-ATLAS offline on a laptop. Community GGUFs plus llama.cpp, Ollama or LM Studio already do this, with OpenAI-compatible servers. We do not rebuild it.

**Headline (unsolved problems only):** *Language-fair compression for N-ATLAS.*
1. **A CPU engine that decodes faster than llama.cpp** through training-free activation sparsity. No CPU/ARM sparse 4-bit kernel exists for SwiGLU models; TEAL ships GPU kernels only.
2. **The disparity finding (C1/C2):** does English-calibrated sparsity hurt Yoruba, Hausa and Igbo, and does per-language calibration fix it? This has not been measured for any African language.
3. **A per-language report card for any compressed N-ATLaS build** (GGUF or `.satl`): fidelity against a reference, words/s and token tax, per language. Published N-ATLaS quantizations currently ship with no per-language evidence, and we already found that losses differ by language.

### Execution order (each step has a done-criterion; nothing else starts until the current step is done)

| # | Step | Done when |
|---|---|---|
| S1 | **Batched prefill** (int8 GEMM over decoded column tiles). An enabler for S2 and S3, not a contribution, so no gold-plating | Logits match the decode path within int8 tolerance; the NEON kernel equals its scalar spec bit for bit; prefill ≥ 20 tok/s on 6 M4 threads |
| S2 | **Mixed-precision `.satl`** for sensitive tensors | Dense `.satl` − Q4_K_M ΔNLL confidence interval includes 0 (or better) for en, yo, ha and ig |
| S3 | **Local TEAL in atlas-rt**: calibrate, evaluate, traces (replaces the GPU/Colab path, which stays optional) | Calibration × evaluation matrix for {en, yo, ha, ig} at p ∈ {0.4, 0.5}, calibrated on {en, own language, pooled}; C1/C2 written up; atlas-sim G1 decision |
| S4 | **Sparsity in the engine**: per-language profiles, masking, active-column kernels | Decode speed vs llama.cpp on M4 P-cores and E-cores (`taskpolicy -c background` as the low-end proxy), at the quality gate |
| S5 | **Per-language report card** (`atlas-eval`): one command scores any GGUF or `.satl` build against a reference, per language (KL/ΔNLL, top-1, words/s, token tax) | Report cards for Q4_K_M, dense `.satl` and sparse `.satl`, reproducible from one command |
| S6 | **x86 path**: AVX2 kernel equal to the int8 spec; Linux/Windows builds | Spec equality under Rosetta; binaries build |
| S7 | **Quality breadth**: chrF++ and AfriMMLU through atlas-rt | Per-language tables at the chosen sparsity |
| S8 | **Packaging**: README with N-ATLaS attribution, LICENSE/NOTICE, hosted `.satl` + profiles, tester kit (testers run the report card and the speed benchmark on their own machines), outreach message, docs, video script | A tester goes from zero to a report card and a speed comparison in < 15 minutes |

### Step log
- **S1 done (24 Sep), with one shortfall accepted.** Batched prefill equals stepwise decode bit for bit, and NEON equals its scalar spec. Prefill runs at 11–13 tok/s on 6 threads on the throttled M4 (about 2.8× decode), **below the ≥ 20 tok/s target**. Profiling shows about a third of thread time idle (serial per-token work) and about 15% in per-token folding. Accepted because S1 is only an enabler: S2/S3 runs take hours, not days.

- **S3 done (25 Sep):** C1 supported, C2 null, operating point 30% (§12.1c); G1 = streaming out, C4 null (§12.1d).
- **S2 done (24 Sep).** Least-squares q4 fitting (converter only: same bytes, layout and kernels) lowers reconstruction error 7.4% and closes the English gap: `.satl` − Q4_K_M = +0.005 nats [−0.036, +0.041] (was +0.050 [+0.015, +0.082]). All four languages' intervals include 0. Point estimates remain slightly positive (+0.004 to +0.021), so `.satl` is at parity, not ahead. A fallback exists if needed: group 32 for qkv/down in Q4_K_M's sensitive layers (+1.5% bytes). `data/models/n-atlas-q4g64s.satl` is now the model for S3/S4.

### Cut (stated as limitations, not attempted)
Storage streaming beyond the atlas-sim G1 decision · Android / Tier C · Pi 5 measurements (left for testers) · human A/B and the code-switching set (replaced by automatic metrics) · C ABI · **our own chat server or offline-runner product** (already solved by llama.cpp, Ollama and LM Studio).

### Deviations this causes
- Thresholds are calibrated on the shipped q4 model, not bf16.
- C3 is claimed for M4 P-core and E-core tiers only.
