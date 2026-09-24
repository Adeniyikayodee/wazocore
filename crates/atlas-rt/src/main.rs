use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};
use memmap2::Mmap;
use tokenizers::Tokenizer;

use atlas_rt::forward::{KvCache, Plan, step};
use atlas_rt::weights::weights;
use atlas_rt::{chat, ops};
use atlas_satl::format;

#[derive(Parser)]
#[command(about = "N-ATLAS CPU runtime (scalar reference path)")]
struct Cli {
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
}

type AnyError = Box<dyn std::error::Error>;

fn main() -> ExitCode {
    let result = match Cli::parse().cmd {
        Cmd::Logits { model, tokens, all } => logits(&model, &tokens, all),
        Cmd::Generate { model, tokenizer, prompt, system, max_tokens, ctx } => {
            generate(&model, &tokenizer, &system, &prompt, max_tokens, ctx)
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

fn logits(model: &Path, tokens: &[u32], all: bool) -> Result<(), AnyError> {
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let plan = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?);
    let mut kv = KvCache::new(&w.config, tokens.len());
    let per_pos = tokens.iter().map(|&t| step(&w, &plan, &mut kv, t)).collect::<Result<Vec<_>, _>>()?;
    let out: Vec<&Vec<f32>> = if all { per_pos.iter().collect() } else { per_pos.last().into_iter().collect() };
    println!("{}", serde_json::to_string(&serde_json::json!({ "tokens": tokens, "logits": out }))?);
    Ok(())
}

fn generate(model: &Path, tok_path: &Path, system: &str, user: &str, max_tokens: usize, ctx: usize) -> Result<(), AnyError> {
    let (header, data, map) = open(model)?;
    let w = weights(&header, &map[data as usize..])?;
    let plan = Plan::dense(&w.config, ops::rope_inv_freq(&w.config)?);
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
