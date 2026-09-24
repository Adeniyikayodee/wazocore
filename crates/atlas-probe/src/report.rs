//! Human-readable summary of a manifest, as a pure `Manifest -> String`.

use std::fmt::Write;

use crate::manifest::{Manifest, MemKernel};

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

pub fn render(m: &Manifest) -> String {
    let h = &m.host;
    let mut out = String::new();
    let _ = writeln!(out, "== {} ==", m.label.as_deref().unwrap_or("atlas-probe"));
    let _ = writeln!(
        out,
        "{} | {} | {}/{} | {} CPUs | RAM {}",
        h.device_model.as_deref().unwrap_or("?"),
        h.cpu_model.as_deref().unwrap_or("?"),
        h.os,
        h.arch,
        h.logical_cpus,
        h.ram_total_bytes.map_or("?".into(), |b| format!("{:.1} GiB", b as f64 / GIB)),
    );
    let clusters: Vec<String> = h
        .clusters
        .iter()
        .map(|c| match c.max_khz {
            Some(k) => format!("{}x {} @ {:.2} GHz", c.count, c.core, k as f64 / 1e6),
            None => format!("{}x {}", c.count, c.core),
        })
        .collect();
    let _ = writeln!(out, "clusters: {} | isa: {}", clusters.join(", "), h.isa_features.join(" "));
    if let Some(g) = &h.cpu_governor {
        let _ = writeln!(out, "governor: {g}");
    }

    let _ = writeln!(out, "\nmemory bandwidth (GB/s, median / max):");
    for r in &m.memory {
        let kernel = match r.kernel {
            MemKernel::Read => "read ",
            MemKernel::Copy => "copy ",
            MemKernel::Triad => "triad",
        };
        if let Some(s) = r.gbps {
            let _ = writeln!(out, "  {kernel} {:>3} thr  {:>7.2} / {:>7.2}", r.threads, s.median, s.max);
        }
    }

    if let Some(c) = &m.decode_ceiling {
        let _ = writeln!(
            out,
            "\nN-ATLAS decode ceiling at {:.2} GB/s read ({} thr):",
            c.read_gbps, c.threads
        );
        for row in &c.rows {
            let _ = writeln!(
                out,
                "  sparsity {:>3.0}%  {:.2} GB/token  {:>6.2} tok/s",
                row.sparsity * 100.0,
                row.gb_per_token,
                row.tok_per_s
            );
        }
    }

    if let Some(s) = &m.storage {
        let _ = writeln!(
            out,
            "\nstorage ({}, cache bypass {:?}): sequential 1 MiB {:.0} MB/s",
            s.dir, s.cache_bypass, s.sequential.mb_per_s
        );
        for r in &s.random {
            let p50 = r.latency_us.map_or(f64::NAN, |l| l.median);
            let _ = writeln!(
                out,
                "  random {:>3} KiB qd{}  {:>7.1} MB/s  {:>8.0} IOPS  p50 {:>7.1} us",
                r.block_bytes / 1024,
                r.queue_depth,
                r.mb_per_s,
                r.iops,
                p50
            );
        }
    }
    out
}
