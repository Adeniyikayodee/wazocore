use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use serde::Serialize;

use atlas_probe::budget;
use atlas_sim::device::{self, Prediction};
use atlas_sim::freq;
use atlas_sim::sim::{self, PolicySpec, SimConfig, SimStats};
use atlas_sim::synth::{self, SynthParams};
use atlas_sim::trace::{self, Trace};

#[derive(Parser)]
#[command(about = "Trace-driven cache simulator for N-ATLAS weight streaming")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Write a synthetic activation trace (N-ATLAS MLP shape)
    Synth(SynthArgs),
    /// Replay a trace against cache policies
    Run(RunArgs),
}

#[derive(Args)]
struct SynthArgs {
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value = "yo")]
    lang: String,
    #[arg(long, default_value_t = 1000)]
    tokens: u64,
    #[arg(long, default_value_t = 32)]
    layers: u32,
    #[arg(long, default_value_t = 0.5)]
    sparsity: f64,
    /// Std-dev of the shared per-channel logit (heavier skew = hotter hot set)
    #[arg(long, default_value_t = 1.5)]
    skew: f64,
    /// Fraction of channels whose rate depends on the language
    #[arg(long, default_value_t = 0.1)]
    lang_frac: f64,
    /// Std-dev of the language-specific logit offset
    #[arg(long, default_value_t = 2.0)]
    lang_skew: f64,
    /// Probability a channel repeats its previous token's state
    #[arg(long, default_value_t = 0.3)]
    persist: f64,
    #[arg(long, default_value_t = 1)]
    model_seed: u64,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Args)]
struct RunArgs {
    /// Evaluation trace
    #[arg(long)]
    trace: PathBuf,
    /// Calibration traces (one per language) for the static policies
    #[arg(long)]
    calib: Vec<PathBuf>,
    /// Cache size as a fraction of all cacheable bytes
    #[arg(long, conflicts_with = "cache_gb")]
    cache_frac: Option<f64>,
    /// Cache size in GB (1e9 bytes)
    #[arg(long)]
    cache_gb: Option<f64>,
    /// Matrices that stream from storage (default: all in the trace)
    #[arg(long, value_delimiter = ',')]
    cacheable: Vec<String>,
    /// Any of lru, lfu, belady, static-lang, static-mix
    #[arg(long, value_delimiter = ',', default_value = "lru,lfu,static-lang,static-mix,belady")]
    policies: Vec<String>,
    /// atlas-probe manifest of the target device, for tok/s predictions
    #[arg(long)]
    probe: Option<PathBuf>,
    /// Attention input sparsity, for the resident bytes read per token
    #[arg(long, default_value_t = 0.5)]
    attn_sparsity: f64,
    /// Concurrent storage readers
    #[arg(long, default_value_t = 4)]
    qd: usize,
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Serialize)]
struct RunReport {
    trace: trace::Header,
    capacity_bytes: u64,
    cacheable_bytes: u64,
    results: Vec<SimStats>,
    predictions: Vec<Prediction>,
}

fn main() -> ExitCode {
    let result = match Cli::parse().cmd {
        Cmd::Synth(a) => synth_cmd(&a),
        Cmd::Run(a) => run_cmd(&a),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

type AnyError = Box<dyn std::error::Error>;

fn load(path: &Path) -> Result<Trace, AnyError> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(trace::decode(&bytes).map_err(|e| format!("{}: {e}", path.display()))?)
}

fn synth_cmd(a: &SynthArgs) -> Result<(), AnyError> {
    let params = SynthParams {
        lang: a.lang.clone(),
        tokens: a.tokens,
        layers: a.layers,
        matrices: trace::n_atlas_mlp(),
        sparsity: a.sparsity,
        skew: a.skew,
        lang_frac: a.lang_frac,
        lang_skew: a.lang_skew,
        persist: a.persist,
        model_seed: a.model_seed,
        seed: a.seed,
    };
    let t = synth::generate(&params)?;
    fs::write(&a.out, trace::encode(&t))?;
    eprintln!("wrote {} ({} tokens, {})", a.out.display(), a.tokens, t.header.source);
    Ok(())
}

fn cacheable_flags(t: &Trace, names: &[String]) -> Result<Vec<bool>, AnyError> {
    let known: Vec<&str> = t.header.matrices.iter().map(|m| m.name.as_str()).collect();
    match names.iter().find(|n| !known.contains(&n.as_str())) {
        Some(bad) => Err(format!("unknown matrix {bad}; trace has {known:?}").into()),
        None => Ok(known.iter().map(|k| names.is_empty() || names.iter().any(|n| n == k)).collect()),
    }
}

fn cacheable_bytes(t: &Trace, flags: &[bool]) -> u64 {
    let lay = &t.layout;
    (0..lay.matrices())
        .filter(|&m| flags[m])
        .map(|m| lay.channels[m] as u64 * lay.bytes[m] as u64 * lay.layers as u64)
        .sum()
}

fn policy_specs(a: &RunArgs, t: &Trace, flags: &[bool], cap: u64) -> Result<Vec<PolicySpec>, AnyError> {
    let calib: Vec<Trace> = a.calib.iter().map(|p| load(p)).collect::<Result<_, _>>()?;
    if let Some(bad) = calib.iter().find(|c| !c.layout.same_shape(&t.layout)) {
        return Err(format!("calibration trace for {} has a different shape", bad.header.lang).into());
    }
    let freqs: Vec<(String, Vec<f64>)> =
        calib.iter().map(|c| (c.header.lang.clone(), freq::activation_freqs(c))).collect();
    a.policies
        .iter()
        .map(|p| -> Result<PolicySpec, AnyError> {
            Ok(match p.as_str() {
                "lru" => PolicySpec::Lru,
                "lfu" => PolicySpec::Lfu,
                "belady" => PolicySpec::Belady,
                "static-lang" => {
                    let (_, f) = freqs
                        .iter()
                        .find(|(lang, _)| *lang == t.header.lang)
                        .ok_or(format!("static-lang needs a --calib trace for {}", t.header.lang))?;
                    PolicySpec::Static { name: p.clone(), hot: freq::hot_set(f, &t.layout, flags, cap) }
                }
                "static-mix" => {
                    let all: Vec<Vec<f64>> = freqs.iter().map(|(_, f)| f.clone()).collect();
                    let f = freq::mean_freqs(&all).ok_or("static-mix needs --calib traces")?;
                    PolicySpec::Static { name: p.clone(), hot: freq::hot_set(&f, &t.layout, flags, cap) }
                }
                other => return Err(format!("unknown policy {other}").into()),
            })
        })
        .collect()
}

fn resident_bytes_per_token(attn_sparsity: f64) -> f64 {
    let (s, p) = (budget::N_ATLAS, budget::Q4_GROUP);
    budget::attn_params(&s) as f64 * p.weight_bits / 8.0 * (1.0 - attn_sparsity)
        + budget::head_params(&s) as f64 * p.head_bits / 8.0
}

fn run_cmd(a: &RunArgs) -> Result<(), AnyError> {
    let t = load(&a.trace)?;
    let flags = cacheable_flags(&t, &a.cacheable)?;
    let total = cacheable_bytes(&t, &flags);
    let cap = match (a.cache_frac, a.cache_gb) {
        (Some(f), _) => (total as f64 * f) as u64,
        (_, Some(g)) => (g * 1e9) as u64,
        _ => return Err("give --cache-frac or --cache-gb".into()),
    };
    let specs = policy_specs(a, &t, &flags, cap)?;
    let cfg = SimConfig { capacity_bytes: cap, cacheable: flags };
    let results = specs
        .iter()
        .map(|p| {
            eprintln!("simulating {}...", p.name());
            sim::simulate(&t, &cfg, p)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let device = a
        .probe
        .as_ref()
        .map(|p| -> Result<device::Device, AnyError> {
            let v: serde_json::Value = serde_json::from_slice(&fs::read(p)?)?;
            Ok(device::from_probe(&v)?)
        })
        .transpose()?;
    let resident = resident_bytes_per_token(a.attn_sparsity);
    let predictions: Vec<Prediction> = device
        .iter()
        .flat_map(|d| results.iter().filter_map(|s| device::predict(d, s, resident, a.qd)))
        .collect();

    eprint!("{}", render(&t, cap, total, &results, &predictions));
    let report = RunReport { trace: t.header.clone(), capacity_bytes: cap, cacheable_bytes: total, results, predictions };
    let json = serde_json::to_string_pretty(&report)?;
    match &a.out {
        Some(p) => fs::write(p, json + "\n")?,
        None => println!("{json}"),
    }
    Ok(())
}

fn render(t: &Trace, cap: u64, total: u64, results: &[SimStats], preds: &[Prediction]) -> String {
    let active: Vec<String> = results
        .first()
        .map(|r| r.matrices.iter().map(|m| format!("{} {:.1}%", m.name, 100.0 * m.active_fraction)).collect())
        .unwrap_or_default();
    let head = format!(
        "\n{} [{}], {} tokens | active: {} | cache {:.2} GB = {:.0}% of {:.2} GB\n{:<12}{:>10}{:>14}{}\n",
        t.header.lang,
        t.header.profile,
        t.layout.tokens,
        active.join(", "),
        cap as f64 / 1e9,
        100.0 * cap as f64 / total.max(1) as f64,
        total as f64 / 1e9,
        "policy",
        "hit rate",
        "miss MB/tok",
        if preds.is_empty() { "" } else { "   tok/s serial / overlapped" },
    );
    let rows: String = results
        .iter()
        .map(|r| {
            let pred = preds
                .iter()
                .find(|p| p.policy == r.policy)
                .map_or(String::new(), |p| format!("   {:>6.2} / {:>6.2}", p.tok_s_serial, p.tok_s_overlapped));
            format!("{:<12}{:>9.1}%{:>14.1}{pred}\n", r.policy, 100.0 * r.byte_hit_rate, r.miss_bytes_per_token / 1e6)
        })
        .collect();
    head + &rows
}
