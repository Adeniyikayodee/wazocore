use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use memmap2::Mmap;
use tokenizers::Tokenizer;

use atlas_rt::forward::{Kernels, KvCache, Plan, step};
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
    /// Time each kernel of layer 0 and lm_head in isolation
    Kernels {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 20)]
        reps: usize,
    },
    /// Time decode steps on synthetic tokens (no tokenizer)
    Bench {
        #[arg(long)]
        model: PathBuf,
        #[arg(long, default_value_t = 32)]
        tokens: usize,
        #[arg(long, default_value_t = 4)]
        warmup: usize,
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
        Cmd::Bench { model, tokens, warmup } => bench(&model, tokens, warmup, kernels),
        Cmd::Kernels { model, reps } => kernel_bench(&model, reps, kernels),
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
    let last = ids.iter().try_fold(Vec::new(), |_, &t| step(&w, &plan, &mut kv, t))?;
    let prefill = t0.elapsed().as_secs_f64();

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
        prefill,
        ids.len() as f64 / prefill,
        out.len(),
        decode,
        out.len() as f64 / decode,
        words as f64 / decode
    );
    Ok(())
}

/// Decode-step timing: `warmup` untimed steps, then `tokens` timed steps,
/// reporting median and p95 per-step tok/s and implied weight bandwidth.
fn bench(model: &Path, tokens: usize, warmup: usize, kernels: Kernels) -> Result<(), AnyError> {
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let plan = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?, kernels);
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

fn kernel_bench(model: &Path, reps: usize, kernels: Kernels) -> Result<(), AnyError> {
    use atlas_rt::{fast, kernels as reference};
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let c = &w.config;
    let l = &w.layers[0];
    let x = |n: usize| -> Vec<f32> { (0..n).map(|i| ((i as f32) * 0.37).sin()).collect() };
    let all = |n: usize| -> Vec<u32> { (0..n as u32).collect() };
    let (xh, xq, xf) = (x(c.hidden), x(c.n_heads * c.head_dim), x(c.ffn));
    let (ah, aq, af) = (all(c.hidden), all(c.n_heads * c.head_dim), all(c.ffn));
    let cols = |w: &atlas_rt::weights::Columns, x: &[f32], a: &[u32]| match kernels {
        Kernels::Reference => reference::columns_gemv_ref(w, x, a),
        Kernels::Fast => fast::columns_gemv(w, x, a),
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
    println!("{} threads, {:?}", rayon::current_num_threads(), kernels);
    for (name, bytes, mut f) in cases {
        let s = time_median(reps, &mut f)?;
        println!("  {name:<8} {:>8.3} ms  {:>6.1} GB/s  ({:.1} MB)", s * 1e3, bytes as f64 / s / 1e9, bytes as f64 / 1e6);
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
        let logits = ids[..ids.len() - 1].iter().map(|&t| step(&w, &plan, &mut kv, t)).collect::<Result<Vec<_>, _>>()?;
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
            let logits = c.tokens.iter().map(|&t| step(&w, &plan, &mut kv, t)).collect::<Result<Vec<_>, _>>()?;
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
