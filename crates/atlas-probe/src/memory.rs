//! STREAM-style memory bandwidth. Worker threads persist across reps and
//! synchronise on barriers, so thread start-up cost never enters a timing.

use std::hint::black_box;
use std::sync::Barrier;
use std::thread;
use std::time::Instant;

use crate::manifest::{MemKernel, MemResult};
use crate::stats::{rate, summarize};

const F64_BYTES: u64 = 8;

/// Runs each kernel at each thread count over three arrays of `mem_mib` MiB.
pub fn run(mem_mib: usize, reps: usize, threads: &[usize]) -> Vec<MemResult> {
    let n = mem_mib * 1024 * 1024 / F64_BYTES as usize;
    let a = vec![1.0f64; n];
    let b = vec![2.0f64; n];
    let mut c = vec![0.5f64; n];
    threads
        .iter()
        .flat_map(|&t| {
            let read = measure(MemKernel::Read, t, n, reps, timed_read(&a, t, reps));
            let copy = measure(MemKernel::Copy, t, n, reps, timed_copy(&mut c, &a, t, reps));
            let triad = measure(MemKernel::Triad, t, n, reps, timed_triad(&mut c, &a, &b, t, reps));
            [read, copy, triad]
        })
        .collect()
}

/// Bytes moved per rep under the STREAM convention (write-allocate not counted).
fn arrays_touched(kernel: MemKernel) -> u64 {
    match kernel {
        MemKernel::Read => 1,
        MemKernel::Copy => 2,
        MemKernel::Triad => 3,
    }
}

fn measure(kernel: MemKernel, threads: usize, n: usize, reps: usize, secs: Vec<f64>) -> MemResult {
    let bytes_per_rep = arrays_touched(kernel) * n as u64 * F64_BYTES;
    let gbps: Vec<f64> = secs
        .iter()
        .skip(1) // warm-up: page faults, frequency ramp
        .filter_map(|&s| rate(bytes_per_rep, s).map(|r| r / 1e9))
        .collect();
    debug_assert_eq!(gbps.len(), reps.saturating_sub(1));
    MemResult { kernel, threads, bytes_per_rep, gbps: summarize(&gbps) }
}

/// Runs every worker once per rep; returns wall seconds per rep, measured
/// from the earliest worker start to the latest worker finish. Workers stamp
/// their own spans so a descheduled coordinator cannot shorten a rep.
fn timed_reps<W: FnMut() + Send>(workers: Vec<W>, reps: usize) -> Vec<f64> {
    let barrier = Barrier::new(workers.len());
    let spans: Vec<Vec<(Instant, Instant)>> = thread::scope(|s| {
        workers
            .into_iter()
            .map(|mut work| {
                let barrier = &barrier;
                s.spawn(move || {
                    (0..reps)
                        .map(|_| {
                            barrier.wait();
                            let t0 = Instant::now();
                            work();
                            (t0, Instant::now())
                        })
                        .collect()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().expect("bandwidth worker panicked"))
            .collect()
    });
    rep_durations(&spans, reps)
}

/// Per rep: latest end minus earliest start across workers.
fn rep_durations(spans: &[Vec<(Instant, Instant)>], reps: usize) -> Vec<f64> {
    (0..reps)
        .filter_map(|r| {
            let start = spans.iter().filter_map(|w| w.get(r).map(|s| s.0)).min()?;
            let end = spans.iter().filter_map(|w| w.get(r).map(|s| s.1)).max()?;
            Some(end.duration_since(start).as_secs_f64())
        })
        .collect()
}

fn chunk_len(n: usize, threads: usize) -> usize {
    n.div_ceil(threads).max(1)
}

/// Eight independent accumulators so the sum vectorises and is not
/// latency-bound on the FP add chain.
fn sum8(xs: &[f64]) -> f64 {
    let (chunks, tail) = xs.as_chunks::<8>();
    let acc = chunks.iter().fold([0.0f64; 8], |acc, c| std::array::from_fn(|i| acc[i] + c[i]));
    acc.iter().sum::<f64>() + tail.iter().sum::<f64>()
}

fn timed_read(a: &[f64], threads: usize, reps: usize) -> Vec<f64> {
    let workers = a
        .chunks(chunk_len(a.len(), threads))
        .map(|chunk| move || {
            black_box(sum8(black_box(chunk)));
        })
        .collect();
    timed_reps(workers, reps)
}

fn timed_copy(dst: &mut [f64], src: &[f64], threads: usize, reps: usize) -> Vec<f64> {
    let len = chunk_len(src.len(), threads);
    let workers = dst
        .chunks_mut(len)
        .zip(src.chunks(len))
        .map(|(d, s)| move || {
            d.copy_from_slice(black_box(s));
            black_box(&d);
        })
        .collect();
    timed_reps(workers, reps)
}

fn timed_triad(dst: &mut [f64], b: &[f64], c: &[f64], threads: usize, reps: usize) -> Vec<f64> {
    let len = chunk_len(b.len(), threads);
    let scalar = black_box(3.0f64);
    let workers = dst
        .chunks_mut(len)
        .zip(b.chunks(len).zip(c.chunks(len)))
        .map(|(d, (b, c))| move || {
            d.iter_mut().zip(b.iter().zip(c)).for_each(|(d, (b, c))| *d = b + scalar * c);
            black_box(&d);
        })
        .collect();
    timed_reps(workers, reps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sum8_matches_naive_sum() {
        let xs: Vec<f64> = (0..1003).map(f64::from).collect();
        assert_eq!(sum8(&xs), xs.iter().sum::<f64>());
    }

    #[test]
    fn small_run_reports_every_kernel() {
        let rs = run(1, 3, &[1, 2]);
        assert_eq!(rs.len(), 6);
        assert!(rs.iter().all(|r| r.gbps.is_some_and(|s| s.n == 2 && s.median > 0.0)));
    }
}
