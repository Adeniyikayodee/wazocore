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
    let result = match cli.cmd {
        Cmd::Logits { model, tokens, all } => logits(&model, &tokens, all, kernels),
        Cmd::Generate { model, tokenizer, prompt, system, max_tokens, ctx } => {
            generate(&model, &tokenizer, &system, &prompt, max_tokens, ctx, kernels)
        }
        Cmd::Bench { model, tokens, warmup } => bench(&model, tokens, warmup, kernels),
        Cmd::Kernels { model, reps } => kernel_bench(&model, reps, kernels),
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

/// A named kernel run over `bytes` of weights.
type Case<'a> = (&'static str, u64, Box<dyn FnMut() -> Vec<f32> + 'a>);

/// Median seconds of `reps` runs of `f` after one warm-up run.
fn time_median(reps: usize, mut f: impl FnMut() -> Vec<f32>) -> f64 {
    std::hint::black_box(f());
    let mut secs: Vec<f64> = (0..reps)
        .map(|_| {
            let t0 = Instant::now();
            std::hint::black_box(f());
            t0.elapsed().as_secs_f64()
        })
        .collect();
    secs.sort_by(f64::total_cmp);
    secs[secs.len() / 2]
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
        ("qkv", (l.qkv.count * l.qkv.stride) as u64, Box::new(|| cols(&l.qkv, &xh, &ah))),
        ("o", (l.o.count * l.o.stride) as u64, Box::new(|| cols(&l.o, &xq, &aq))),
        ("gate_up", (l.gate_up.count * l.gate_up.stride) as u64, Box::new(|| cols(&l.gate_up, &xh, &ah))),
        ("down", (l.down.count * l.down.stride) as u64, Box::new(|| cols(&l.down, &xf, &af))),
        ("lm_head", w.lm_head.data.len() as u64, Box::new(|| match kernels {
            Kernels::Reference => reference::rows_gemv_ref(&w.lm_head, &xh),
            Kernels::Fast => fast::rows_gemv(&w.lm_head, &xh),
        })),
    ];
    println!("{} threads, {:?}", rayon::current_num_threads(), kernels);
    for (name, bytes, mut f) in cases {
        let s = time_median(reps, &mut f);
        println!("  {name:<8} {:>8.3} ms  {:>6.1} GB/s  ({:.1} MB)", s * 1e3, bytes as f64 / s / 1e9, bytes as f64 / 1e6);
    }
    Ok(())
}
