//! Plain, immutable data describing one probe run. Producers live in the
//! shell modules; every consumer is a pure function.

use serde::Serialize;

use crate::budget;
use crate::stats::Summary;

/// Bumped whenever a field changes meaning, so old result files stay readable.
pub const SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub tool: String,
    pub label: Option<String>,
    pub timestamp_unix: u64,
    pub host: Host,
    pub memory: Vec<MemResult>,
    pub storage: Option<StorageReport>,
    pub decode_ceiling: Option<DecodeCeiling>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Host {
    pub hostname: Option<String>,
    pub device_model: Option<String>,
    pub os: &'static str,
    pub arch: &'static str,
    pub kernel: Option<String>,
    pub cpu_model: Option<String>,
    pub logical_cpus: usize,
    pub clusters: Vec<Cluster>,
    pub isa_features: Vec<&'static str>,
    pub ram_total_bytes: Option<u64>,
    pub ram_available_bytes: Option<u64>,
    pub cpu_governor: Option<String>,
    pub thermal_c: Vec<Thermal>,
    pub block_devices: Vec<BlockDevice>,
    pub git_commit: Option<String>,
}

/// CPUs sharing a core type and max clock (a big.LITTLE cluster).
/// `cpus` is empty where the OS does not expose per-CPU identity (macOS).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Cluster {
    pub core: String,
    pub max_khz: Option<u64>,
    pub count: usize,
    pub cpus: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Thermal {
    pub zone: String,
    pub celsius: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BlockDevice {
    pub name: String,
    pub kind: &'static str,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MemKernel {
    /// Read-only sum: the access pattern of decode (weights are only read).
    Read,
    Copy,
    Triad,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemResult {
    pub kernel: MemKernel,
    pub threads: usize,
    pub bytes_per_rep: u64,
    /// Decimal GB/s over the timed reps (warm-up discarded).
    pub gbps: Option<Summary>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CacheBypass {
    ODirect,
    FNocache,
    /// O_DIRECT refused; page cache dropped with posix_fadvise before each test.
    FadviseOnly,
    /// Numbers may include page-cache hits.
    None,
}

#[derive(Debug, Clone, Serialize)]
pub struct StorageReport {
    pub dir: String,
    pub file_bytes: u64,
    pub cache_bypass: CacheBypass,
    pub sequential: SeqResult,
    pub random: Vec<RandResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SeqResult {
    pub block_bytes: u64,
    pub mb_per_s: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct RandResult {
    pub block_bytes: u64,
    pub queue_depth: usize,
    pub reads: usize,
    pub mb_per_s: f64,
    pub iops: f64,
    pub latency_us: Option<Summary>,
}

/// Bandwidth-bound decode ceilings for N-ATLAS implied by the measured
/// read bandwidth (PROJECT-PLAN-B §2.2).
#[derive(Debug, Clone, Serialize)]
pub struct DecodeCeiling {
    pub read_gbps: f64,
    pub threads: usize,
    pub assumptions: String,
    pub rows: Vec<CeilingRow>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CeilingRow {
    pub sparsity: f64,
    pub gb_per_token: f64,
    pub tok_per_s: f64,
}

pub const CEILING_SPARSITIES: [f64; 4] = [0.0, 0.25, 0.40, 0.50];

/// Ceilings from the best median read bandwidth across thread counts.
pub fn decode_ceiling(memory: &[MemResult]) -> Option<DecodeCeiling> {
    let (threads, read_gbps) = memory
        .iter()
        .filter(|m| m.kernel == MemKernel::Read)
        .filter_map(|m| m.gbps.map(|s| (m.threads, s.median)))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    let shape = budget::N_ATLAS;
    let precision = budget::Q4_GROUP;
    let rows = CEILING_SPARSITIES
        .iter()
        .filter_map(|&sparsity| {
            let bytes = budget::bytes_per_token(&shape, &precision, sparsity)?;
            Some(CeilingRow {
                sparsity,
                gb_per_token: bytes / 1e9,
                tok_per_s: budget::decode_ceiling_tok_s(bytes, read_gbps * 1e9)?,
            })
        })
        .collect();
    Some(DecodeCeiling {
        read_gbps,
        threads,
        assumptions: format!(
            "Llama-3-8B shape; {} bpw attention/MLP, {} bpw lm_head; sparsity on attention+MLP inputs; KV reads excluded",
            precision.weight_bits, precision.head_bits
        ),
        rows,
    })
}

pub fn build(
    label: Option<String>,
    timestamp_unix: u64,
    host: Host,
    memory: Vec<MemResult>,
    storage: Option<StorageReport>,
) -> Manifest {
    let decode_ceiling = decode_ceiling(&memory);
    Manifest {
        schema_version: SCHEMA_VERSION,
        tool: format!("atlas-probe {}", env!("CARGO_PKG_VERSION")),
        label,
        timestamp_unix,
        host,
        memory,
        storage,
        decode_ceiling,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(threads: usize, median: f64) -> MemResult {
        MemResult {
            kernel: MemKernel::Read,
            threads,
            bytes_per_rep: 1,
            gbps: Some(Summary { n: 1, min: median, median, p95: median, max: median }),
        }
    }

    #[test]
    fn ceiling_uses_best_read_median() {
        let c = decode_ceiling(&[read(1, 5.0), read(4, 10.0)]).unwrap();
        assert_eq!(c.threads, 4);
        // Plan §2.2: dense at 10 GB/s ≈ 2.30 tok/s, 50% ≈ 4.18 tok/s.
        assert!((c.rows[0].tok_per_s - 2.30).abs() < 0.01);
        assert!((c.rows[3].tok_per_s - 4.18).abs() < 0.01);
    }

    #[test]
    fn ceiling_absent_without_read_results() {
        assert!(decode_ceiling(&[]).is_none());
    }
}
