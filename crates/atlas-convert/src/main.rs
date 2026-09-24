use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::thread;

use clap::{Parser, Subcommand};
use memmap2::Mmap;
use safetensors::{Dtype as StDtype, SafeTensors};
use sha2::{Digest, Sha256};

use atlas_convert::codec::error_stats;
use atlas_convert::encode::{self, Dtype, Matrix};
use atlas_convert::format::{self, FORMAT, Header, Kind, Source, SourceFile, VERSION};
use atlas_convert::plan::{self, Options};

#[derive(Parser)]
#[command(about = "Convert bf16 safetensors checkpoints to the sparse-friendly .satl format")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a Hugging Face Llama checkpoint directory to .satl
    Convert {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Model id recorded in the header
        #[arg(long, default_value = "NCAIR1/N-ATLaS")]
        model_name: String,
        /// Rows per (scale, min) in q4 columns: 64 = 4.5 bpw, 32 = 5.0 bpw
        #[arg(long, default_value_t = 64)]
        q4_group: usize,
        #[arg(long, default_value_t = 32)]
        q8_group: usize,
        /// Lossless f32 everywhere (reference files for runtime validation)
        #[arg(long)]
        f32: bool,
        /// q4 parameter fit: min-max, or search (least-squares, same format)
        #[arg(long, default_value = "min-max", value_parser = ["min-max", "search"])]
        q4_fit: String,
        /// Skip SHA-256 of source files (tests only; breaks provenance)
        #[arg(long)]
        no_hash: bool,
    },
    /// Print a .satl header summary
    Inspect { file: PathBuf },
    /// Compare a .satl file against its source checkpoint
    Verify {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long)]
        satl: PathBuf,
        /// Check every Nth column/row
        #[arg(long, default_value_t = 97)]
        every: usize,
    },
}

type AnyError = Box<dyn std::error::Error>;

fn main() -> ExitCode {
    let result = match Cli::parse().cmd {
        Cmd::Convert { model_dir, out, model_name, q4_group, q8_group, f32, q4_fit, no_hash } => {
            let fit = if q4_fit == "search" { atlas_convert::codec::Fit::Search } else { atlas_convert::codec::Fit::MinMax };
            convert(&model_dir, &out, &model_name, Options { q4_group, q8_group, q4_fit: fit, f32 }, !no_hash)
        }
        Cmd::Inspect { file } => inspect(&file),
        Cmd::Verify { model_dir, satl, every } => verify(&model_dir, &satl, every.max(1)),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Memory-mapped safetensors shards of a checkpoint directory.
struct Checkpoint {
    files: Vec<(String, Mmap)>,
}

fn shard_names(dir: &Path) -> Result<Vec<String>, AnyError> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let v: serde_json::Value = serde_json::from_slice(&fs::read(&index)?)?;
        let map = v.get("weight_map").and_then(|m| m.as_object()).ok_or("index lacks weight_map")?;
        let mut names: Vec<String> = map.values().filter_map(|f| f.as_str().map(str::to_owned)).collect();
        names.sort();
        names.dedup();
        Ok(names)
    } else {
        Ok(vec!["model.safetensors".into()])
    }
}

fn open_checkpoint(dir: &Path) -> Result<Checkpoint, AnyError> {
    let files = shard_names(dir)?
        .into_iter()
        .map(|name| {
            let file = File::open(dir.join(&name)).map_err(|e| format!("{name}: {e}"))?;
            // SAFETY: read-only mapping of a file we do not modify during the run.
            let map = unsafe { Mmap::map(&file)? };
            Ok((name, map))
        })
        .collect::<Result<_, AnyError>>()?;
    Ok(Checkpoint { files })
}

fn dtype(d: StDtype) -> Result<Dtype, AnyError> {
    match d {
        StDtype::BF16 => Ok(Dtype::Bf16),
        StDtype::F16 => Ok(Dtype::F16),
        StDtype::F32 => Ok(Dtype::F32),
        other => Err(format!("unsupported source dtype {other:?}").into()),
    }
}

impl Checkpoint {
    fn parsed(&self) -> Result<Vec<SafeTensors<'_>>, AnyError> {
        self.files.iter().map(|(_, m)| Ok(SafeTensors::deserialize(m)?)).collect()
    }
}

fn shapes(parsed: &[SafeTensors]) -> HashMap<String, Vec<usize>> {
    parsed.iter().flat_map(|st| st.tensors().into_iter().map(|(n, v)| (n, v.shape().to_vec()))).collect()
}

fn matrix<'a>(parsed: &'a [SafeTensors], name: &str) -> Result<Matrix<'a>, AnyError> {
    let view = parsed.iter().find_map(|st| st.tensor(name).ok()).ok_or_else(|| format!("missing tensor {name}"))?;
    let (rows, cols) = match view.shape() {
        [n] => (1, *n),
        [r, c] => (*r, *c),
        s => return Err(format!("{name}: unsupported rank {}", s.len()).into()),
    };
    Ok(Matrix { rows, cols, dtype: dtype(view.dtype())?, data: view.data() })
}

fn sha256_file(path: &Path) -> Result<String, AnyError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn threads() -> usize {
    thread::available_parallelism().map_or(1, |n| n.get())
}

/// Splits `0..n` into `k` contiguous ranges.
fn ranges(n: usize, k: usize) -> Vec<std::ops::Range<usize>> {
    let step = n.div_ceil(k.max(1)).max(1);
    (0..n).step_by(step).map(|s| s..(s + step).min(n)).collect()
}

fn encode_tensor(kind: &Kind, parsed: &[SafeTensors], fit: atlas_convert::codec::Fit) -> Result<Vec<u8>, AnyError> {
    let sources: Vec<Matrix> = plan::sources(kind).iter().map(|(n, _)| matrix(parsed, n)).collect::<Result<_, _>>()?;
    let chunks: Vec<Vec<u8>> = match kind {
        Kind::Vector { .. } => vec![encode::encode_vector(&sources[0])],
        Kind::Columns { count, .. } => thread::scope(|s| {
            ranges(*count, threads())
                .into_iter()
                .map(|r| s.spawn(|| encode::encode_columns(kind, &sources, r, fit)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("encoder thread panicked"))
                .collect::<Result<_, _>>()
        })?,
        Kind::Rows { count, .. } => thread::scope(|s| {
            ranges(*count, threads())
                .into_iter()
                .map(|r| s.spawn(|| encode::encode_rows(kind, &sources[0], r)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|h| h.join().expect("encoder thread panicked"))
                .collect::<Result<_, _>>()
        })?,
    };
    Ok(chunks.concat())
}

fn convert(dir: &Path, out: &Path, model: &str, opts: Options, hash: bool) -> Result<(), AnyError> {
    let cfg = plan::config_from_hf(&serde_json::from_slice(&fs::read(dir.join("config.json"))?)?)?;
    let ckpt = open_checkpoint(dir)?;
    let parsed = ckpt.parsed()?;
    let tensors = plan::plan(&cfg, &shapes(&parsed), &opts)?;
    let files = ckpt
        .files
        .iter()
        .map(|(name, _)| {
            eprintln!("hashing {name}...");
            let sha256 = if hash { sha256_file(&dir.join(name))? } else { "unhashed".into() };
            Ok(SourceFile { name: name.clone(), sha256 })
        })
        .collect::<Result<_, AnyError>>()?;
    let header = Header {
        format: FORMAT.into(),
        version: VERSION,
        source: Source { model: model.into(), files },
        q4_fit: Some(if opts.q4_fit == atlas_convert::codec::Fit::Search { "search" } else { "min-max" }.into()),
        config: cfg,
        tensors,
    };
    let prefix = format::encode_prefix(&header);
    let data = format::data_start(prefix.len());
    let tmp = out.with_extension("satl.partial");
    let file = File::create(&tmp)?;
    file.write_all_at(&prefix, 0)?;
    for (i, t) in header.tensors.iter().enumerate() {
        eprintln!("[{}/{}] {} ({:.1} MB)", i + 1, header.tensors.len(), t.name, t.length as f64 / 1e6);
        let bytes = encode_tensor(&t.kind, &parsed, opts.q4_fit)?;
        if bytes.len() as u64 != t.length {
            return Err(format!("{}: encoded {} bytes, planned {}", t.name, bytes.len(), t.length).into());
        }
        file.write_all_at(&bytes, data + t.offset)?;
    }
    let end = header.tensors.last().map_or(data, |t| data + t.offset + t.length);
    file.set_len(end)?;
    file.sync_all()?;
    fs::rename(&tmp, out)?;
    eprintln!("wrote {} ({:.2} GB)", out.display(), end as f64 / 1e9);
    Ok(())
}

fn read_satl(path: &Path) -> Result<(Header, u64, Mmap), AnyError> {
    let file = File::open(path)?;
    // SAFETY: read-only mapping; the file is not modified while mapped.
    let map = unsafe { Mmap::map(&file)? };
    let (header, data) = format::decode_prefix(&map)?;
    Ok((header, data, map))
}

fn inspect(path: &Path) -> Result<(), AnyError> {
    let (h, data, map) = read_satl(path)?;
    let c = &h.config;
    println!("{} v{} from {} ({} files)", h.format, h.version, h.source.model, h.source.files.len());
    println!(
        "hidden {} ffn {} layers {} heads {}/{} head_dim {} vocab {} rope_theta {}",
        c.hidden, c.ffn, c.layers, c.n_heads, c.n_kv_heads, c.head_dim, c.vocab, c.rope_theta
    );
    println!("data at {data}, file {:.2} GB, {} tensors", map.len() as f64 / 1e9, h.tensors.len());
    for t in h.tensors.iter().filter(|t| !t.name.starts_with("layers.") || t.name.starts_with("layers.0.")) {
        let detail = match &t.kind {
            Kind::Columns { count, column_bytes, stride, codec, .. } => {
                format!("{count} columns x {column_bytes} B (stride {stride}), {codec:?}")
            }
            Kind::Rows { count, row_bytes, codec, .. } => format!("{count} rows x {row_bytes} B, {codec:?}"),
            Kind::Vector { len, .. } => format!("f32[{len}]"),
        };
        println!("  {:<22} {:>10.2} MB  {detail}", t.name, t.length as f64 / 1e6);
    }
    Ok(())
}

fn verify(dir: &Path, satl: &Path, every: usize) -> Result<(), AnyError> {
    let (h, data, map) = read_satl(satl)?;
    let ckpt = open_checkpoint(dir)?;
    let parsed = ckpt.parsed()?;
    let mut worst = 0.0f64;
    for t in &h.tensors {
        let seg = &map[(data + t.offset) as usize..(data + t.offset + t.length) as usize];
        let sources: Vec<Matrix> = plan::sources(&t.kind).iter().map(|(n, _)| matrix(&parsed, n)).collect::<Result<_, _>>()?;
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = match &t.kind {
            Kind::Columns { count, .. } => (0..*count)
                .step_by(every)
                .map(|j| {
                    let dec = encode::decode_column(&t.kind, seg, j)?;
                    Ok(sources.iter().zip(dec).map(|(m, d)| ((0..m.rows).map(|r| m.get(r, j)).collect(), d)).collect::<Vec<_>>())
                })
                .collect::<Result<Vec<_>, AnyError>>()?
                .concat(),
            Kind::Rows { count, .. } => (0..*count)
                .step_by(every)
                .map(|r| Ok((sources[0].row(r), encode::decode_row(&t.kind, seg, r)?)))
                .collect::<Result<_, AnyError>>()?,
            Kind::Vector { len, .. } => {
                let dec = (0..*len).map(|i| f32::from_le_bytes(seg[i * 4..i * 4 + 4].try_into().expect("4 bytes"))).collect();
                vec![(sources[0].row(0), dec)]
            }
        };
        let (reference, approx): (Vec<f32>, Vec<f32>) =
            pairs.into_iter().fold((Vec::new(), Vec::new()), |(mut a, mut b), (x, y)| {
                a.extend(x);
                b.extend(y);
                (a, b)
            });
        let e = error_stats(&reference, &approx);
        worst = worst.max(e.rel_rmse);
        println!("{:<22} rel_rmse {:.4}  max_abs {:.3e}", t.name, e.rel_rmse, e.max_abs);
    }
    println!("worst rel_rmse {worst:.4}");
    Ok(())
}
