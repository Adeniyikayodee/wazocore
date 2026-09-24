//! Storage read profile: sequential 1 MiB reads and random 4/16/64 KiB reads
//! at several queue depths, bypassing the page cache where the OS allows it.
//! These are the access sizes of `.satl` column bundles (PROJECT-PLAN-B §3.2).

use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::hint::black_box;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::thread;
use std::time::Instant;

use crate::config::StorageConfig;
use crate::manifest::{CacheBypass, RandResult, SeqResult, StorageReport};
use crate::stats::{rate, summarize};

const MIB: u64 = 1024 * 1024;
const ALIGN: usize = 4096;
pub const RANDOM_BLOCKS: [u64; 3] = [4096, 16384, 65536];

#[derive(Debug)]
pub struct StorageError {
    pub step: &'static str,
    pub source: io::Error,
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "storage probe failed to {}: {}", self.step, self.source)
    }
}

impl std::error::Error for StorageError {}

fn at(step: &'static str) -> impl FnOnce(io::Error) -> StorageError {
    move |source| StorageError { step, source }
}

pub fn run(cfg: &StorageConfig) -> Result<StorageReport, StorageError> {
    let path = cfg.dir.join(format!(".atlas-probe-{}.bin", std::process::id()));
    let bytes = cfg.file_mib as u64 * MIB;
    let written = write_test_file(&path, bytes).map_err(at("write the test file"));
    let report = written.and_then(|()| measure(&path, bytes, cfg));
    let removed = match (&report, cfg.keep) {
        (Ok(_), true) => Ok(()),
        _ => fs::remove_file(&path).or_else(ignore_missing).map_err(at("remove the test file")),
    };
    report.and_then(|r| removed.map(|()| r))
}

fn ignore_missing(e: io::Error) -> io::Result<()> {
    if e.kind() == io::ErrorKind::NotFound { Ok(()) } else { Err(e) }
}

fn measure(path: &Path, bytes: u64, cfg: &StorageConfig) -> Result<StorageReport, StorageError> {
    let (file, cache_bypass) = open_uncached(path).map_err(at("open the test file"))?;
    drop_cache(&file).map_err(at("drop the page cache"))?;
    let sequential = sequential(&file, bytes).map_err(at("read sequentially"))?;
    let random = RANDOM_BLOCKS
        .iter()
        .flat_map(|&b| cfg.queue_depths.iter().map(move |&q| (b, q)))
        .map(|(block, qd)| {
            drop_cache(&file)
                .and_then(|()| random(&file, bytes, block, cfg.random_reads, qd))
                .map_err(at("read randomly"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(StorageReport {
        dir: cfg.dir.display().to_string(),
        file_bytes: bytes,
        cache_bypass,
        sequential,
        random,
    })
}

/// xorshift64* stream; incompressible so flash controllers cannot shortcut it.
fn next_rand(state: u64) -> u64 {
    let x = state ^ (state >> 12);
    let x = x ^ (x << 25);
    let x = x ^ (x >> 27);
    x.wrapping_mul(0x2545_f491_4f6c_dd1d)
}

fn rand_stream(seed: u64) -> impl Iterator<Item = u64> {
    std::iter::successors(Some(next_rand(seed | 1)), |&s| Some(next_rand(s)))
}

fn pattern(seed: u64, len: usize) -> Vec<u8> {
    rand_stream(seed).flat_map(u64::to_le_bytes).take(len).collect()
}

/// `count` block-aligned offsets drawn uniformly from the file.
pub fn random_offsets(seed: u64, file_bytes: u64, block: u64, count: usize) -> Vec<u64> {
    let blocks = (file_bytes / block).max(1);
    rand_stream(seed).take(count).map(|r| (r % blocks) * block).collect()
}

/// A zeroed buffer plus the offset at which `len` bytes are `ALIGN`-aligned,
/// as O_DIRECT requires.
fn aligned(len: usize) -> (Vec<u8>, usize) {
    let raw = vec![0u8; len + ALIGN];
    let off = raw.as_ptr().align_offset(ALIGN);
    (raw, off)
}

fn write_test_file(path: &Path, bytes: u64) -> io::Result<()> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    set_nocache(&file)?;
    (0..bytes / MIB).try_for_each(|i| file.write_all_at(&pattern(i, MIB as usize), i * MIB))?;
    file.sync_all()
}

fn sequential(file: &File, bytes: u64) -> io::Result<SeqResult> {
    let (mut raw, off) = aligned(MIB as usize);
    let buf = &mut raw[off..off + MIB as usize];
    let t0 = Instant::now();
    (0..bytes / MIB).try_for_each(|i| file.read_exact_at(buf, i * MIB))?;
    let secs = t0.elapsed().as_secs_f64();
    black_box(&buf);
    Ok(SeqResult { block_bytes: MIB, mb_per_s: rate(bytes, secs).unwrap_or(0.0) / 1e6 })
}

fn random(file: &File, bytes: u64, block: u64, count: usize, qd: usize) -> io::Result<RandResult> {
    let offsets = random_offsets(block ^ ((qd as u64) << 32), bytes, block, count);
    let t0 = Instant::now();
    let per_reader: io::Result<Vec<Vec<f64>>> = thread::scope(|s| {
        offsets
            .chunks(count.div_ceil(qd).max(1))
            .map(|chunk| s.spawn(move || read_batch(file, chunk, block as usize)))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("storage reader thread panicked"))
            .collect()
    });
    let secs = t0.elapsed().as_secs_f64();
    let latencies = per_reader?.concat();
    Ok(RandResult {
        block_bytes: block,
        queue_depth: qd,
        reads: count,
        mb_per_s: rate(count as u64 * block, secs).unwrap_or(0.0) / 1e6,
        iops: rate(count as u64, secs).unwrap_or(0.0),
        latency_us: summarize(&latencies),
    })
}

/// Reads each offset once; returns per-read latency in microseconds.
fn read_batch(file: &File, offsets: &[u64], block: usize) -> io::Result<Vec<f64>> {
    let (mut raw, off) = aligned(block);
    let buf = &mut raw[off..off + block];
    offsets
        .iter()
        .map(|&o| {
            let t0 = Instant::now();
            file.read_exact_at(buf, o)?;
            Ok(t0.elapsed().as_secs_f64() * 1e6)
        })
        .collect()
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn open_uncached(path: &Path) -> io::Result<(File, CacheBypass)> {
    use std::os::unix::fs::OpenOptionsExt;
    match OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path) {
        Ok(f) => Ok((f, CacheBypass::ODirect)),
        Err(_) => File::open(path).map(|f| (f, CacheBypass::FadviseOnly)),
    }
}

#[cfg(target_os = "macos")]
fn open_uncached(path: &Path) -> io::Result<(File, CacheBypass)> {
    let f = File::open(path)?;
    set_nocache(&f)?;
    Ok((f, CacheBypass::FNocache))
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_os = "macos")))]
fn open_uncached(path: &Path) -> io::Result<(File, CacheBypass)> {
    File::open(path).map(|f| (f, CacheBypass::None))
}

#[cfg(target_os = "macos")]
fn set_nocache(f: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: fcntl on a valid, owned descriptor with an integer argument.
    let rc = unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
    if rc == -1 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

#[cfg(not(target_os = "macos"))]
fn set_nocache(_: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn drop_cache(f: &File) -> io::Result<()> {
    use std::os::fd::AsRawFd;
    // SAFETY: posix_fadvise on a valid, owned descriptor; returns an errno value.
    let rc = unsafe { libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    if rc == 0 { Ok(()) } else { Err(io::Error::from_raw_os_error(rc)) }
}

#[cfg(not(any(target_os = "linux", target_os = "android")))]
fn drop_cache(_: &File) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_are_aligned_in_range_and_deterministic() {
        let a = random_offsets(7, 64 * MIB, 16384, 500);
        assert_eq!(a, random_offsets(7, 64 * MIB, 16384, 500));
        assert!(a.iter().all(|&o| o % 16384 == 0 && o + 16384 <= 64 * MIB));
        assert_ne!(a, random_offsets(8, 64 * MIB, 16384, 500));
    }

    #[test]
    fn aligned_buffer_is_aligned() {
        let (raw, off) = aligned(4096);
        assert_eq!(raw[off..].as_ptr() as usize % ALIGN, 0);
        assert!(raw.len() - off >= 4096);
    }

    #[test]
    fn pattern_has_requested_length() {
        assert_eq!(pattern(1, 1000).len(), 1000);
        assert_ne!(pattern(1, 64), pattern(2, 64));
    }

    #[test]
    fn end_to_end_on_temp_dir() {
        let cfg = StorageConfig {
            dir: std::env::temp_dir(),
            file_mib: 16,
            random_reads: 50,
            queue_depths: vec![1, 2],
            keep: false,
        };
        let r = run(&cfg).unwrap();
        assert_eq!(r.random.len(), RANDOM_BLOCKS.len() * 2);
        assert!(r.sequential.mb_per_s > 0.0);
    }
}
