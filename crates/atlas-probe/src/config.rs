//! Command-line parsing as a pure function of the argument list.

use std::fmt;
use std::path::PathBuf;

pub const USAGE: &str = "\
atlas-probe: measure memory bandwidth, storage reads and CPU topology

USAGE: atlas-probe [OPTIONS]

  --label TEXT          free-form tag stored in the manifest (e.g. pi5-cooled)
  --out PATH            write JSON here instead of stdout
  --mem-mib N           size of each bandwidth array in MiB        [512]
  --reps N              timed repetitions per test, first discarded [8]
  --threads LIST        thread counts, e.g. 1,4,8                   [1,<all>]
  --storage-dir PATH    directory on the storage under test         [.]
  --storage-mib N       size of the temporary test file in MiB      [512]
  --random-reads N      random reads per block size and depth       [2000]
  --queue-depths LIST   concurrent readers, e.g. 1,4                [1,4]
  --keep-file           keep the test file after the run
  --no-storage          skip the storage probe
  -h, --help            show this help
";

#[derive(Debug, Clone, PartialEq)]
pub struct StorageConfig {
    pub dir: PathBuf,
    pub file_mib: usize,
    pub random_reads: usize,
    pub queue_depths: Vec<usize>,
    pub keep: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub label: Option<String>,
    pub out: Option<PathBuf>,
    pub mem_mib: usize,
    pub reps: usize,
    /// Empty means "1 and all logical CPUs".
    pub threads: Vec<usize>,
    pub storage: Option<StorageConfig>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            dir: PathBuf::from("."),
            file_mib: 512,
            random_reads: 2000,
            queue_depths: vec![1, 4],
            keep: false,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            label: None,
            out: None,
            mem_mib: 512,
            reps: 8,
            threads: Vec::new(),
            storage: Some(StorageConfig::default()),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    Run(Config),
    Help,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    UnknownFlag(String),
    MissingValue(String),
    BadValue { flag: String, value: String, expected: String },
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownFlag(s) => write!(f, "unknown option {s}"),
            Self::MissingValue(s) => write!(f, "{s} needs a value"),
            Self::BadValue { flag, value, expected } => {
                write!(f, "{flag} {value}: expected {expected}")
            }
        }
    }
}

pub fn parse(args: &[String]) -> Result<Parsed, ConfigError> {
    parse_from(Config::default(), args)
}

fn parse_from(cfg: Config, args: &[String]) -> Result<Parsed, ConfigError> {
    match args {
        [] => Ok(Parsed::Run(cfg)),
        [flag, ..] if flag == "-h" || flag == "--help" => Ok(Parsed::Help),
        [flag, rest @ ..] if flag == "--no-storage" => {
            parse_from(Config { storage: None, ..cfg }, rest)
        }
        [flag, rest @ ..] if flag == "--keep-file" => {
            let storage = cfg.storage.clone().map(|s| StorageConfig { keep: true, ..s });
            parse_from(Config { storage, ..cfg }, rest)
        }
        [flag, value, rest @ ..] => parse_from(apply(cfg, flag, value)?, rest),
        [flag] if is_value_flag(flag) => Err(ConfigError::MissingValue(flag.clone())),
        [flag] => Err(ConfigError::UnknownFlag(flag.clone())),
    }
}

const VALUE_FLAGS: [&str; 9] = [
    "--label",
    "--out",
    "--mem-mib",
    "--reps",
    "--threads",
    "--storage-dir",
    "--storage-mib",
    "--random-reads",
    "--queue-depths",
];

fn is_value_flag(flag: &str) -> bool {
    VALUE_FLAGS.contains(&flag)
}

fn bad(flag: &str, value: &str, expected: &str) -> ConfigError {
    ConfigError::BadValue { flag: flag.to_owned(), value: value.to_owned(), expected: expected.to_owned() }
}

fn at_least(flag: &str, value: &str, min: usize) -> Result<usize, ConfigError> {
    value
        .parse::<usize>()
        .ok()
        .filter(|n| *n >= min)
        .ok_or_else(|| bad(flag, value, &format!("an integer >= {min}")))
}

fn positive_list(flag: &str, value: &str) -> Result<Vec<usize>, ConfigError> {
    value
        .split(',')
        .map(|s| s.trim().parse::<usize>().ok().filter(|n| *n > 0))
        .collect::<Option<Vec<_>>>()
        .filter(|v| !v.is_empty())
        .ok_or_else(|| bad(flag, value, "a comma-separated list of positive integers"))
}

fn with_storage(cfg: Config, f: impl FnOnce(StorageConfig) -> StorageConfig) -> Config {
    let storage = Some(f(cfg.storage.clone().unwrap_or_default()));
    Config { storage, ..cfg }
}

fn apply(cfg: Config, flag: &str, value: &str) -> Result<Config, ConfigError> {
    Ok(match flag {
        "--label" => Config { label: Some(value.to_owned()), ..cfg },
        "--out" => Config { out: Some(PathBuf::from(value)), ..cfg },
        "--mem-mib" => Config { mem_mib: at_least(flag, value, 1)?, ..cfg },
        "--reps" => Config { reps: at_least(flag, value, 2)?, ..cfg },
        "--threads" => Config { threads: positive_list(flag, value)?, ..cfg },
        "--storage-dir" => with_storage(cfg, |s| StorageConfig { dir: PathBuf::from(value), ..s }),
        "--storage-mib" => {
            let file_mib = at_least(flag, value, 16)?;
            with_storage(cfg, |s| StorageConfig { file_mib, ..s })
        }
        "--random-reads" => {
            let random_reads = at_least(flag, value, 1)?;
            with_storage(cfg, |s| StorageConfig { random_reads, ..s })
        }
        "--queue-depths" => {
            let queue_depths = positive_list(flag, value)?;
            with_storage(cfg, |s| StorageConfig { queue_depths, ..s })
        }
        _ => return Err(ConfigError::UnknownFlag(flag.to_owned())),
    })
}

/// Thread counts to benchmark: the explicit list, or 1 and all CPUs.
pub fn resolve_threads(requested: &[usize], logical_cpus: usize) -> Vec<usize> {
    match requested {
        [] if logical_cpus > 1 => vec![1, logical_cpus],
        [] => vec![1],
        list => list.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_owned).collect()
    }

    fn run(s: &str) -> Config {
        match parse(&args(s)).unwrap() {
            Parsed::Run(c) => c,
            Parsed::Help => panic!("unexpected help"),
        }
    }

    #[test]
    fn defaults() {
        assert_eq!(run(""), Config::default());
    }

    #[test]
    fn parses_all_value_flags() {
        let c = run("--label pi5 --mem-mib 64 --reps 3 --threads 1,4 --storage-mib 32 --queue-depths 2 --keep-file");
        let s = c.storage.unwrap();
        assert_eq!(c.label.as_deref(), Some("pi5"));
        assert_eq!((c.mem_mib, c.reps, c.threads), (64, 3, vec![1, 4]));
        assert_eq!((s.file_mib, s.queue_depths, s.keep), (32, vec![2], true));
    }

    #[test]
    fn no_storage_then_storage_flag_reenables() {
        assert!(run("--no-storage").storage.is_none());
        assert!(run("--no-storage --storage-dir /x").storage.is_some());
    }

    #[test]
    fn errors_are_values() {
        assert_eq!(parse(&args("--bogus 1")), Err(ConfigError::UnknownFlag("--bogus".into())));
        assert_eq!(parse(&args("--reps")), Err(ConfigError::MissingValue("--reps".into())));
        assert!(matches!(parse(&args("--reps 1")), Err(ConfigError::BadValue { .. })));
        assert!(matches!(parse(&args("--threads 1,0")), Err(ConfigError::BadValue { .. })));
        assert_eq!(parse(&args("--help")), Ok(Parsed::Help));
    }

    #[test]
    fn resolves_default_threads() {
        assert_eq!(resolve_threads(&[], 8), vec![1, 8]);
        assert_eq!(resolve_threads(&[], 1), vec![1]);
        assert_eq!(resolve_threads(&[2], 8), vec![2]);
    }
}
