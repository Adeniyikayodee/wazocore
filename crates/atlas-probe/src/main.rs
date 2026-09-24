use std::fmt;
use std::fs;
use std::io;
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use atlas_probe::config::{self, Config, Parsed};
use atlas_probe::storage::StorageError;
use atlas_probe::{host, manifest, memory, report, storage};

enum AppError {
    Storage(StorageError),
    Json(serde_json::Error),
    Write(io::Error),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Storage(e) => write!(f, "{e}"),
            Self::Json(e) => write!(f, "could not serialise manifest: {e}"),
            Self::Write(e) => write!(f, "could not write manifest: {e}"),
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match config::parse(&args) {
        Ok(Parsed::Help) => {
            print!("{}", config::USAGE);
            ExitCode::SUCCESS
        }
        Ok(Parsed::Run(cfg)) => match run(&cfg) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        Err(e) => {
            eprintln!("error: {e}\n\n{}", config::USAGE);
            ExitCode::from(2)
        }
    }
}

fn run(cfg: &Config) -> Result<(), AppError> {
    eprintln!("probing host...");
    let host = host::collect();
    let threads = config::resolve_threads(&cfg.threads, host.logical_cpus);
    eprintln!("memory bandwidth: {} MiB arrays, threads {:?}...", cfg.mem_mib, threads);
    let memory = memory::run(cfg.mem_mib, cfg.reps, &threads);
    let storage = match &cfg.storage {
        Some(s) => {
            eprintln!("storage: {} MiB test file in {}...", s.file_mib, s.dir.display());
            Some(storage::run(s).map_err(AppError::Storage)?)
        }
        None => None,
    };
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs());
    let manifest = manifest::build(cfg.label.clone(), timestamp, host, memory, storage);
    eprint!("\n{}", report::render(&manifest));
    let json = serde_json::to_string_pretty(&manifest).map_err(AppError::Json)?;
    match &cfg.out {
        Some(path) => {
            fs::write(path, json + "\n").map_err(AppError::Write)?;
            eprintln!("\nmanifest written to {}", path.display());
        }
        None => println!("{json}"),
    }
    Ok(())
}
