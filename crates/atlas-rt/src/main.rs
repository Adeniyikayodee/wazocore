use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use memmap2::Mmap;
use tokenizers::Tokenizer;

use atlas_rt::forward::{Kernels, KvCache, Plan, Want, prefill, step};
use atlas_rt::weights::weights;
use atlas_rt::{chat, ops};
use atlas_satl::format;

#[derive(Parser)]
#[command(about = "N-ATLAS CPU runtime")]
struct Cli {
    /// Worker threads (default: all logical CPUs)
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Use the scalar single-threaded reference kernels
    #[arg(long, global = true)]
    reference: bool,
    /// atlas-probe manifest of this machine, embedded in result files
    #[arg(long, global = true)]
    manifest: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print logits for a token sequence as JSON (validation)
    Logits {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, value_delimiter = ',')]
        tokens: Vec<u32>,
        /// Every position, not just the last
        #[arg(long)]
        all: bool,
    },
    /// Greedy chat generation
    Generate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value = "")]
        system: String,
        #[arg(long, default_value_t = 64)]
        max_tokens: usize,
        #[arg(long, default_value_t = 2048)]
        ctx: usize,
    },
    /// Teacher-forced fidelity of the fast path against the f32 reference
    Compare {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        /// Plain text, one sentence per line
        #[arg(long)]
        text: PathBuf,
        #[arg(long)]
        label: String,
        /// Tokens to evaluate (the text is truncated)
        #[arg(long, default_value_t = 160)]
        tokens: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Perplexity with llama.cpp's protocol (chunks of n_ctx, BOS first,
    /// second half scored), for comparison with llama-perplexity
    Perplexity {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        text: PathBuf,
        #[arg(long, default_value_t = 256)]
        ctx: usize,
        #[arg(long, default_value_t = 6)]
        chunks: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// TEAL: per-language thresholds from FLORES dev activation histograms
    TealCalibrate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        flores: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "en,yo,ha,ig")]
        langs: Vec<String>,
        /// Calibration tokens per language
        #[arg(long, default_value_t = 8192)]
        tokens: usize,
        #[arg(long, default_value_t = 256)]
        ctx: usize,
        #[arg(long, value_delimiter = ',', default_value = "0.4,0.5")]
        sparsities: Vec<f64>,
        #[arg(long)]
        out: PathBuf,
    },
    /// TEAL: calibration x evaluation matrix on FLORES devtest (C1/C2)
    TealEvaluate {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        flores: PathBuf,
        #[arg(long)]
        profiles: PathBuf,
        #[arg(long, value_delimiter = ',', default_value = "en,yo,ha,ig")]
        langs: Vec<String>,
        #[arg(long, value_delimiter = ',', default_value = "0.4,0.5")]
        sparsities: Vec<f64>,
        #[arg(long, default_value_t = 12)]
        chunks: usize,
        #[arg(long, default_value_t = 256)]
        ctx: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// TEAL: write an activation trace (.atr) for atlas-sim
    TealTrace {
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        flores: PathBuf,
        #[arg(long)]
        profile: PathBuf,
        #[arg(long)]
        lang: String,
        #[arg(long, default_value = "devtest")]
        split: String,
        #[arg(long, default_value_t = 4096)]
        tokens: usize,
        #[arg(long, default_value_t = 256)]
        ctx: usize,
        #[arg(long)]
        out: PathBuf,
    },
    /// Time each kernel of layer 0 and lm_head in isolation
    Kernels {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 20)]
        reps: usize,
        /// Tokens per call: 1 = decode kernels, > 1 = batched prefill kernels
        #[arg(long, default_value_t = 1)]
        batch: usize,
    },
    /// Time decode steps on synthetic tokens (no tokenizer)
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        #[arg(long, default_value_t = 4)]
        warmup: usize,
        /// Also time batched prefill of this many prompt tokens (3 runs)
        #[arg(long)]
        prefill_tokens: Option<usize>,
    },
}

type AnyError = Box<dyn std::error::Error>;

fn main() -> ExitCode {
    let cli = Cli::parse();
    if let Some(n) = cli.threads
        && let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(n).build_global()
    {
        eprintln!("error: thread pool: {e}");
        return ExitCode::FAILURE;
    }
    let kernels = if cli.reference { Kernels::Reference } else { Kernels::Fast };
    let hardware = match cli.manifest.as_deref().map(read_manifest).transpose() {
        Ok(h) => h.unwrap_or(serde_json::Value::Null),
        Err(e) => {
            eprintln!("error: manifest: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = match cli.cmd {
        Cmd::Logits { model, tokens, all } => logits(&model, &tokens, all, kernels),
        Cmd::Generate { model, tokenizer, prompt, system, max_tokens, ctx } => {
            generate(&model, &tokenizer, &system, &prompt, max_tokens, ctx, kernels)
        }
        Cmd::Bench { model, tokens, warmup, prefill_tokens } => bench(&model, tokens, warmup, prefill_tokens, kernels),
        Cmd::Kernels { model, reps, batch } => kernel_bench(&model, reps, batch, kernels),
        Cmd::TealCalibrate { model, tokenizer, flores, langs, tokens, ctx, sparsities, out } => {
            teal_calibrate(&model, &tokenizer, &flores, &langs, tokens, ctx, &sparsities, &out)
        }
        Cmd::TealEvaluate { model, tokenizer, flores, profiles, langs, sparsities, chunks, ctx, out } => {
            teal_evaluate(&model, &tokenizer, &flores, &profiles, &langs, &sparsities, chunks, ctx, &out, &hardware)
        }
        Cmd::TealTrace { model, tokenizer, flores, profile, lang, split, tokens, ctx, out } => {
            teal_trace(&model, &tokenizer, &flores, &profile, &lang, &split, tokens, ctx, &out)
        }
        Cmd::Compare { model, tokenizer, text, label, tokens, out } => {
            compare(&model, &tokenizer, &text, &label, tokens, &out, &hardware)
        }
        Cmd::Perplexity { model, tokenizer, text, ctx, chunks, out } => {
            perplexity(&model, &tokenizer, &text, ctx, chunks, kernels, &out, &hardware)
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn open(path: &Path) -> Result<(format::Header, u64, Mmap), AnyError> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: read-only mapping of a model file not modified while mapped.
    let map = unsafe { Mmap::map(&file)? };
    let (header, data) = format::decode_prefix(&map)?;
    Ok((header, data, map))
}

fn logits(model: &Path, tokens: &[u32], all: bool, kernels: Kernels) -> Result<(), AnyError> {
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let plan = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?, kernels);
    let mut kv = KvCache::new(&w.config, tokens.len());
    let per_pos = tokens.iter().map(|&t| step(&w, &plan, &mut kv, t)).collect::<Result<Vec<_>, _>>()?;
    let out: Vec<&Vec<f32>> = if all { per_pos.iter().collect() } else { per_pos.last().into_iter().collect() };
    println!("{}", serde_json::to_string(&serde_json::json!({ "tokens": tokens, "logits": out }))?);
    Ok(())
}

fn generate(
    model: &Path,
    tok_path: &Path,
    system: &str,
    user: &str,
    max_tokens: usize,
    ctx: usize,
    kernels: Kernels,
) -> Result<(), AnyError> {
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let plan = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?, kernels);
    let tok = Tokenizer::from_file(tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
    let text = chat::prompt(system, user);
    let ids = tok.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();
    let mut kv = KvCache::new(&w.config, ctx);

    let t0 = Instant::now();
    let last = prefill(&w, &plan, &mut kv, &ids, Want::Last)?.pop().ok_or("empty prompt")?;
    let prefill_secs = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut out: Vec<u32> = Vec::new();
    let mut next = ops::argmax(&last) as u32;
    while !chat::EOS.contains(&next) && out.len() < max_tokens && !kv.is_full() {
        out.push(next);
        let piece = tok.decode(&[next], false).map_err(|e| e.to_string())?;
        print!("{piece}");
        std::io::stdout().flush()?;
        next = ops::argmax(&step(&w, &plan, &mut kv, next)?) as u32;
    }
    let decode = t1.elapsed().as_secs_f64();
    let reply = tok.decode(&out, true).map_err(|e| e.to_string())?;
    let words = reply.split_whitespace().count();
    eprintln!(
        "\n\nprefill {} tok in {:.1}s ({:.2} tok/s) | decode {} tok in {:.1}s ({:.2} tok/s, {:.2} words/s)",
        ids.len(),
        prefill_secs,
        ids.len() as f64 / prefill_secs,
        out.len(),
        decode,
        out.len() as f64 / decode,
        words as f64 / decode
    );
    Ok(())
}

/// Decode-step timing: `warmup` untimed steps, then `tokens` timed steps,
/// reporting median and p95 per-step tok/s and implied weight bandwidth.
fn bench(model: &Path, tokens: usize, warmup: usize, prefill_tokens: Option<usize>, kernels: Kernels) -> Result<(), AnyError> {
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let plan = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?, kernels);
    if let Some(n) = prefill_tokens {
        let prompt: Vec<u32> = (0..n).map(|i| (i * 7919 % 50_000 + 1000) as u32).collect();
        let runs = (0..3)
            .map(|_| {
                let mut kv = KvCache::new(&w.config, n);
                let t0 = Instant::now();
                prefill(&w, &plan, &mut kv, &prompt, Want::Last).map(|_| n as f64 / t0.elapsed().as_secs_f64())
            })
            .collect::<Result<Vec<f64>, _>>()?;
        let shown: Vec<String> = runs.iter().map(|r| format!("{r:.1}")).collect();
        println!("{} threads, {kernels:?}: prefill {n} tokens: {} tok/s", rayon::current_num_threads(), shown.join(", "));
    }
    let mut kv = KvCache::new(&w.config, warmup + tokens);
    let ids = (0..warmup + tokens).map(|i| (i * 7919 % 50_000 + 1000) as u32);
    let secs: Vec<f64> = ids
        .enumerate()
        .map(|(i, t)| {
            let t0 = Instant::now();
            step(&w, &plan, &mut kv, t).map(|_| (i, t0.elapsed().as_secs_f64()))
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|(i, _)| *i >= warmup)
        .map(|(_, s)| s)
        .collect();
    let mut sorted = secs.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted[sorted.len() / 2];
    let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
    let weight_bytes: u64 = header
        .tensors
        .iter()
        .filter(|t| t.name != "embed")
        .map(|t| t.length)
        .sum();
    println!(
        "{} threads, {:?}: median {:.2} tok/s (p95 step {:.0} ms), {:.1} GB/s of weights",
        rayon::current_num_threads(),
        kernels,
        1.0 / median,
        p95 * 1e3,
        weight_bytes as f64 / median / 1e9
    );
    Ok(())
}

type KernelResult = Result<Vec<f32>, atlas_rt::kernels::KernelError>;

/// A named kernel run over `bytes` of weights.
type Case<'a> = (&'static str, u64, Box<dyn FnMut() -> KernelResult + 'a>);

/// Median seconds of `reps` runs of `f` after one warm-up run.
fn time_median(reps: usize, mut f: impl FnMut() -> KernelResult) -> Result<f64, AnyError> {
    std::hint::black_box(f()?);
    let mut secs = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            std::hint::black_box(f()?);
            Ok(t0.elapsed().as_secs_f64())
        })
        .collect::<Result<Vec<f64>, AnyError>>()?;
    secs.sort_by(f64::total_cmp);
    Ok(secs[secs.len() / 2])
}

fn kernel_bench(model: &Path, reps: usize, batch: usize, kernels: Kernels) -> Result<(), AnyError> {
    use atlas_rt::{fast, gemm, kernels as reference};
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let c = &w.config;
    let l = &w.layers[0];
    let x = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.37).sin()).collect() };
    let all = |n: usize| -> Vec<u32> { (0..n as u32).collect() };
    let (xh, xq, xf) = (x(c.hidden), x(c.n_heads * c.head_dim), x(c.ffn));
    let (ah, aq, af) = (all(c.hidden), all(c.n_heads * c.head_dim), all(c.ffn));
    let cols = |w: &atlas_rt::weights::Columns, x: &[f32], a: &[u32]| match (kernels, batch) {
        (Kernels::Reference, _) => reference::columns_gemv_ref(w, x, a),
        (Kernels::Fast, 1) => fast::columns_gemv(w, x, a),
        (Kernels::Fast, n) => gemm::columns_gemm(w, &vec![x.to_vec(); n], a).map(|mut ys| ys.pop().unwrap_or_default()),
    };
    let cases: [Case; 5] = [
        ("qkv", l.qkv.bytes() as u64, Box::new(|| cols(&l.qkv, &xh, &ah))),
        ("o", l.o.bytes() as u64, Box::new(|| cols(&l.o, &xq, &aq))),
        ("gate_up", l.gate_up.bytes() as u64, Box::new(|| cols(&l.gate_up, &xh, &ah))),
        ("down", l.down.bytes() as u64, Box::new(|| cols(&l.down, &xf, &af))),
        ("lm_head", w.lm_head.bytes() as u64, Box::new(|| match kernels {
            Kernels::Reference => reference::rows_gemv_ref(&w.lm_head, &xh),
            Kernels::Fast => fast::rows_gemv(&w.lm_head, &xh),
        })),
    ];
    println!("{} threads, {:?}, batch {batch}", rayon::current_num_threads(), kernels);
    for (name, bytes, mut f) in cases {
        let s = time_median(reps, &mut f)?;
        let per_token = s / batch as f64;
        println!("  {name:<8} {:>8.3} ms/token  {:>6.1} GB/s-equivalent  ({:.1} MB)", per_token * 1e3, bytes as f64 / per_token / 1e9, bytes as f64 / 1e6);
    }
    Ok(())
}

/// Runs the reference and fast kernels teacher-forced over the same tokens
/// and reports per-token KL(reference ‖ fast), top-1 agreement and the NLL
/// of the true next token under each.
fn compare(
    model: &Path,
    tok_path: &Path,
    text: &Path,
    label: &str,
    n: usize,
    out: &Path,
    hardware: &serde_json::Value,
) -> Result<(), AnyError> {
    use atlas_rt::metrics::{kl, nll};
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let tok = Tokenizer::from_file(tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
    let body = std::fs::read_to_string(text)?;
    let ids: Vec<u32> = std::iter::once(128_000u32) // <|begin_of_text|>
        .chain(tok.encode(body, false).map_err(|e| e.to_string())?.get_ids().iter().copied())
        .take(n + 1)
        .collect();
    let inv_freq = ops::rope_inv_freq(&w.config)?;
    let run = |k: Kernels| -> Result<Vec<Vec<f32>>, AnyError> {
        let plan = Plan::dense(&w.config, inv_freq.clone(), k);
        let mut kv = KvCache::new(&w.config, ids.len());
        let t0 = Instant::now();
        let logits = prefill(&w, &plan, &mut kv, &ids[..ids.len() - 1], Want::All)?;
        eprintln!("{label}: {k:?} {} tokens in {:.0}s", logits.len(), t0.elapsed().as_secs_f64());
        Ok(logits)
    };
    let fast = run(Kernels::Fast)?;
    let reference = run(Kernels::Reference)?;
    let next = &ids[1..];
    let kls: Vec<f64> = reference.iter().zip(&fast).map(|(r, f)| kl(r, f)).collect();
    let agree = reference.iter().zip(&fast).filter(|(r, f)| ops::argmax(r) == ops::argmax(f)).count();
    let nll_ref: Vec<f64> = reference.iter().zip(next).map(|(r, &t)| nll(r, t as usize)).collect();
    let nll_fast: Vec<f64> = fast.iter().zip(next).map(|(f, &t)| nll(f, t as usize)).collect();
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let mut sorted = kls.clone();
    sorted.sort_by(f64::total_cmp);
    let report = serde_json::json!({
        "label": label,
        "model_sha256_files": header.source.files,
        "tokens": kls.len(),
        "kl_mean": mean(&kls),
        "kl_p95": sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)],
        "kl_max": sorted.last(),
        "top1_agreement": agree as f64 / kls.len() as f64,
        "nll_reference": mean(&nll_ref),
        "nll_fast": mean(&nll_fast),
        "kl_per_token": kls,
        "hardware": hardware,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!(
        "{label}: KL mean {:.2e} p95 {:.2e} | top-1 {:.1}% | NLL ref {:.4} fast {:.4}",
        mean(&kls),
        report["kl_p95"].as_f64().unwrap_or(f64::NAN),
        100.0 * agree as f64 / kls.len() as f64,
        mean(&nll_ref),
        mean(&nll_fast)
    );
    Ok(())
}

fn read_manifest(path: &Path) -> Result<serde_json::Value, AnyError> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(serde_json::from_slice(&bytes)?)
}

#[allow(clippy::too_many_arguments)]
fn perplexity(
    model: &Path,
    tok_path: &Path,
    text: &Path,
    n_ctx: usize,
    max_chunks: usize,
    kernels: Kernels,
    out: &Path,
    hardware: &serde_json::Value,
) -> Result<(), AnyError> {
    use atlas_rt::metrics::{nll, ppl_chunks};
    const BOS: u32 = 128_000;
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let plan = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?, kernels);
    let tok = Tokenizer::from_file(tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
    let body = std::fs::read_to_string(text)?;
    let ids: Vec<u32> = std::iter::once(BOS)
        .chain(tok.encode(body, false).map_err(|e| e.to_string())?.get_ids().iter().copied())
        .collect();
    let t0 = Instant::now();
    let per_chunk = ppl_chunks(&ids, n_ctx, max_chunks, BOS)
        .iter()
        .map(|c| {
            let mut kv = KvCache::new(&w.config, n_ctx);
            let logits = prefill(&w, &plan, &mut kv, &c.tokens, Want::All)?;
            Ok(c.scored.iter().map(|&(i, target)| nll(&logits[i], target as usize)).collect::<Vec<f64>>())
        })
        .collect::<Result<Vec<_>, AnyError>>()?;
    let nlls: Vec<f64> = per_chunk.concat();
    let mean = nlls.iter().sum::<f64>() / nlls.len() as f64;
    let ppl = mean.exp();
    let chunk_ppl: Vec<f64> = per_chunk.iter().map(|c| (c.iter().sum::<f64>() / c.len() as f64).exp()).collect();
    let report = serde_json::json!({
        "engine": "atlas-rt",
        "kernels": format!("{kernels:?}"),
        "model": model.display().to_string(),
        "model_source_files": header.source.files,
        "text": text.display().to_string(),
        "text_tokens": ids.len(),
        "n_ctx": n_ctx,
        "chunks": per_chunk.len(),
        "scored_tokens": nlls.len(),
        "mean_nll": mean,
        "ppl": ppl,
        "chunk_ppl": chunk_ppl,
        "seconds": t0.elapsed().as_secs_f64(),
        "hardware": hardware,
    });
    std::fs::write(out, serde_json::to_string_pretty(&report)? + "\n")?;
    eprintln!("{}: PPL {ppl:.4} over {} tokens ({} chunks)", text.display(), nlls.len(), per_chunk.len());
    Ok(())
}

// ---- TEAL stages (shell) ------------------------------------------------------

/// Per-chunk fidelity of one sparsity profile, plus kept-channel counts.
#[derive(Default)]
struct ConfigAcc {
    kl: Vec<f64>,
    top1: Vec<f64>,
    dnll: Vec<f64>,
    kept: [(u64, u64); 4],
}

const BOS: u32 = 128_000;

fn flores_code(lang: &str) -> Result<&'static str, AnyError> {
    Ok(match lang {
        "en" => "eng_Latn",
        "yo" => "yor_Latn",
        "ha" => "hau_Latn",
        "ig" => "ibo_Latn",
        other => return Err(format!("unknown language {other}").into()),
    })
}

fn flores_text(flores: &Path, lang: &str, split: &str) -> Result<String, AnyError> {
    let f = flores.join(split).join(format!("{}.{split}", flores_code(lang)?));
    Ok(std::fs::read_to_string(&f).map_err(|e| format!("{}: {e}", f.display()))?)
}

/// `n_ctx`-token windows: BOS then `n_ctx - 1` consecutive text tokens.
fn text_chunks(tok: &Tokenizer, text: &str, n_ctx: usize, max_chunks: usize) -> Result<Vec<Vec<u32>>, AnyError> {
    let ids = tok.encode(text, false).map_err(|e| e.to_string())?.get_ids().to_vec();
    Ok(ids.chunks_exact(n_ctx - 1).take(max_chunks).map(|c| std::iter::once(BOS).chain(c.iter().copied()).collect()).collect())
}

fn sha256_hex(text: &str) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

fn load_model(path: &Path) -> Result<(format::Header, u64, Mmap), AnyError> {
    open(path)
}

#[allow(clippy::too_many_arguments)]
fn teal_calibrate(model: &Path, tok_path: &Path, flores: &Path, langs: &[String], budget: usize, ctx: usize, ps: &[f64], out: &Path) -> Result<(), AnyError> {
    use atlas_rt::forward::{Taps, prefill_tapped};
    use atlas_rt::teal::{self, Calibration, Profile, SITES};
    let (header, data, map) = load_model(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let plan = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?, Kernels::Fast);
    let tok = Tokenizer::from_file(tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
    std::fs::create_dir_all(out)?;
    let write = |lang: &str, hist: &teal::Histograms, tokens: usize, digest: &str| -> Result<(), AnyError> {
        for &p in ps {
            let profile = Profile {
                model: model.display().to_string(),
                calibration: Calibration { lang: lang.into(), split: "dev".into(), tokens, text_sha256: digest.into() },
                sparsity: p,
                sites: SITES.iter().map(|s| s.to_string()).collect(),
                thresholds: teal::thresholds(hist, p)?,
            };
            let path = out.join(format!("{lang}-{:02}.json", (p * 100.0).round() as u32));
            std::fs::write(&path, serde_json::to_string_pretty(&profile)? + "\n")?;
            eprintln!("wrote {}", path.display());
        }
        Ok(())
    };
    let mut all = Vec::new();
    let mut digests = Vec::new();
    for lang in langs {
        let text = flores_text(flores, lang, "dev")?;
        let chunks = text_chunks(&tok, &text, ctx, budget / ctx)?;
        let mut taps = Taps { hist: Some(teal::empty_histograms(w.config.layers)), ..Taps::default() };
        let t0 = Instant::now();
        for c in &chunks {
            let mut kv = KvCache::new(&w.config, ctx);
            prefill_tapped(&w, &plan, &mut kv, c, Want::Last, &mut taps)?;
        }
        let tokens = chunks.len() * ctx;
        eprintln!("{lang}: {tokens} calibration tokens in {:.0}s", t0.elapsed().as_secs_f64());
        let hist = taps.hist.ok_or("histogram missing")?;
        let digest = sha256_hex(&text);
        write(lang, &hist, tokens, &digest)?;
        digests.push(digest);
        all.push((hist, tokens));
    }
    if all.len() > 1 {
        let tokens = all.iter().map(|a| a.1).sum();
        let hists: Vec<_> = all.into_iter().map(|a| a.0).collect();
        write("pooled", &teal::pool(&hists).ok_or("no histograms")?, tokens, &sha256_hex(&digests.concat()))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn teal_evaluate(
    model: &Path,
    tok_path: &Path,
    flores: &Path,
    profiles: &Path,
    langs: &[String],
    ps: &[f64],
    max_chunks: usize,
    ctx: usize,
    out: &Path,
    hardware: &serde_json::Value,
) -> Result<(), AnyError> {
    use atlas_rt::forward::{Taps, prefill_tapped};
    use atlas_rt::metrics::log_softmax;
    use atlas_rt::teal::{self, Profile, SITES};
    use rayon::prelude::*;
    let (header, data, map) = load_model(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let dense = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?, Kernels::Fast);
    let tok = Tokenizer::from_file(tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
    let load_profile = |calib: &str, p: f64| -> Result<Profile, AnyError> {
        let f = profiles.join(format!("{calib}-{:02}.json", (p * 100.0).round() as u32));
        Ok(serde_json::from_slice(&std::fs::read(&f).map_err(|e| format!("{}: {e}", f.display()))?)?)
    };
    let mut rows = Vec::new();
    for lang in langs {
        let mut calibs: Vec<&str> = vec!["en", lang.as_str(), "pooled"];
        calibs.dedup();
        let configs: Vec<Profile> = calibs.iter().flat_map(|c| ps.iter().map(move |&p| (*c, p))).map(|(c, p)| load_profile(c, p)).collect::<Result<_, _>>()?;
        let chunks = text_chunks(&tok, &flores_text(flores, lang, "devtest")?, ctx, max_chunks)?;
        // per config: per-chunk means of KL, top-1, ΔNLL; kept counts per site
        let mut acc: Vec<ConfigAcc> = configs.iter().map(|_| ConfigAcc::default()).collect();
        let t0 = Instant::now();
        for c in &chunks {
            let mut kv = KvCache::new(&w.config, ctx);
            let dense_lp: Vec<Vec<f64>> = prefill_tapped(&w, &dense, &mut kv, c, Want::All, &mut Taps::default())?.par_iter().map(|l| log_softmax(l)).collect();
            for (cfg, a) in configs.iter().zip(acc.iter_mut()) {
                let plan = Plan { sparsity: Some(cfg.thresholds.clone()), ..dense.clone() };
                let mut taps = Taps::default();
                let mut kv = KvCache::new(&w.config, ctx);
                let sparse = prefill_tapped(&w, &plan, &mut kv, c, Want::All, &mut taps)?;
                let per_pos: Vec<(f64, bool, f64)> = dense_lp
                    .par_iter()
                    .zip(sparse.par_iter())
                    .enumerate()
                    .map(|(i, (d, s))| {
                        let s = log_softmax(s);
                        let kl = d.iter().zip(&s).map(|(a, b)| a.exp() * (a - b)).sum::<f64>().max(0.0);
                        let arg = |v: &[f64]| v.iter().enumerate().fold((0, f64::NEG_INFINITY), |b, (j, &x)| if x > b.1 { (j, x) } else { b }).0;
                        let dnll = if i + 1 < c.len() { d[c[i + 1] as usize] - s[c[i + 1] as usize] } else { 0.0 };
                        (kl, arg(d) == arg(&s), dnll)
                    })
                    .collect();
                let n = per_pos.len() as f64;
                a.kl.push(per_pos.iter().map(|p| p.0).sum::<f64>() / n);
                a.top1.push(per_pos.iter().filter(|p| p.1).count() as f64 / n);
                a.dnll.push(per_pos[..per_pos.len() - 1].iter().map(|p| p.2).sum::<f64>() / (n - 1.0));
                for (k, t) in a.kept.iter_mut().zip(taps.kept) {
                    k.0 += t.0;
                    k.1 += t.1;
                }
            }
        }
        eprintln!("{lang}: {} chunks x {} configs in {:.0}s", chunks.len(), configs.len(), t0.elapsed().as_secs_f64());
        for (cfg, a) in configs.iter().zip(&acc) {
            let eff: serde_json::Map<String, serde_json::Value> =
                SITES.iter().zip(a.kept).map(|(s, (k, t))| (s.to_string(), serde_json::json!(1.0 - k as f64 / t.max(1) as f64))).collect();
            let kl = teal::bootstrap(&a.kl, 2000);
            let dnll = teal::bootstrap(&a.dnll, 2000);
            let top1 = a.top1.iter().sum::<f64>() / a.top1.len() as f64;
            eprintln!(
                "  {lang} <- {}-{:.0}: eff. sparsity down {:.3} qkv {:.3} | KL {:.4} [{:.4}, {:.4}] | ΔNLL {:+.4} | top-1 {:.3}",
                cfg.calibration.lang,
                cfg.sparsity * 100.0,
                eff["down"].as_f64().unwrap_or(f64::NAN),
                eff["qkv"].as_f64().unwrap_or(f64::NAN),
                kl.0, kl.1, kl.2, dnll.0, top1
            );
            rows.push(serde_json::json!({
                "eval_lang": lang, "calib_lang": cfg.calibration.lang, "target_sparsity": cfg.sparsity,
                "effective_sparsity": eff, "kl": [kl.0, kl.1, kl.2], "delta_nll": [dnll.0, dnll.1, dnll.2],
                "top1": top1, "chunks": a.kl.len(), "ctx": ctx,
                "kl_per_chunk": a.kl, "delta_nll_per_chunk": a.dnll,
            }));
        }
    }
    let report = serde_json::json!({ "model": model.display().to_string(), "profiles": profiles.display().to_string(), "results": rows, "hardware": hardware });
    std::fs::write(out, serde_json::to_string_pretty(&report)? + "\n")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn teal_trace(model: &Path, tok_path: &Path, flores: &Path, profile: &Path, lang: &str, split: &str, budget: usize, ctx: usize, out: &Path) -> Result<(), AnyError> {
    use atlas_rt::forward::{Taps, prefill_tapped};
    use atlas_rt::teal::Profile;
    let (header, data, map) = load_model(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let prof: Profile = serde_json::from_slice(&std::fs::read(profile)?)?;
    let plan = Plan { sparsity: Some(prof.thresholds.clone()), ..Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?, Kernels::Fast) };
    let tok = Tokenizer::from_file(tok_path).map_err(|e| format!("{}: {e}", tok_path.display()))?;
    let chunks = text_chunks(&tok, &flores_text(flores, lang, split)?, ctx, budget / ctx)?;
    let mut taps = Taps { trace: Some(Vec::new()), ..Taps::default() };
    for c in &chunks {
        let mut kv = KvCache::new(&w.config, ctx);
        prefill_tapped(&w, &plan, &mut kv, c, Want::Last, &mut taps)?;
    }
    let c = &w.config;
    let q4 = |rows: usize| rows / 2 + rows / 64 * 4;
    let header_json = serde_json::json!({
        "format": "atlas-trace", "version": 1, "model": model.display().to_string(), "lang": lang,
        "source": format!("FLORES-200 {split}"),
        "profile": profile.file_stem().and_then(|s| s.to_str()).unwrap_or("?"),
        "layers": c.layers, "tokens": chunks.len() * ctx,
        "matrices": [
            {"name": "gate_up", "channels": c.hidden, "bytes_per_channel": 2 * q4(c.ffn)},
            {"name": "down", "channels": c.ffn, "bytes_per_channel": q4(c.hidden)},
        ],
    });
    let json = serde_json::to_vec(&header_json)?;
    let pad = (8 - (8 + json.len()) % 8) % 8;
    let words = taps.trace.ok_or("trace missing")?;
    let bytes: Vec<u8> = b"ATRC"
        .iter()
        .copied()
        .chain(((json.len() + pad) as u32).to_le_bytes())
        .chain(json)
        .chain(std::iter::repeat_n(b' ', pad))
        .chain(words.iter().flat_map(|w| w.to_le_bytes()))
        .collect();
    std::fs::write(out, bytes)?;
    eprintln!("wrote {} ({} tokens)", out.display(), chunks.len() * ctx);
    Ok(())
}
