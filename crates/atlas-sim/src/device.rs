//! Turns simulated cache behaviour into predicted decode speed on a device
//! measured by `atlas-probe` (PROJECT-PLAN-B §2.4, Gate G1).
//!
//! Per token: RAM bytes (resident weights + cache hits) stream at the
//! measured read bandwidth; misses are read from storage in page-aligned
//! blocks at the measured random-read throughput for that block size.
//! `serial` adds the two times; `overlapped` assumes perfect prefetch (max).

use std::fmt;

use serde::Serialize;
use serde_json::Value;

use crate::sim::SimStats;

pub const PAGE: u64 = 4096;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct StoragePoint {
    pub block_bytes: u64,
    pub queue_depth: usize,
    pub mb_per_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Device {
    pub label: String,
    pub mem_read_gbps: f64,
    pub storage: Vec<StoragePoint>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceError {
    Missing(&'static str),
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing(what) => write!(f, "probe manifest lacks {what}"),
        }
    }
}

impl std::error::Error for DeviceError {}

/// Extracts the fields the model needs from an `atlas-probe` manifest.
pub fn from_probe(v: &Value) -> Result<Device, DeviceError> {
    let mem_read_gbps = v
        .pointer("/decode_ceiling/read_gbps")
        .and_then(Value::as_f64)
        .ok_or(DeviceError::Missing("decode_ceiling.read_gbps"))?;
    let storage = v
        .pointer("/storage/random")
        .and_then(Value::as_array)
        .ok_or(DeviceError::Missing("storage.random"))?
        .iter()
        .filter_map(|r| {
            Some(StoragePoint {
                block_bytes: r.get("block_bytes")?.as_u64()?,
                queue_depth: r.get("queue_depth")?.as_u64()? as usize,
                mb_per_s: r.get("mb_per_s")?.as_f64()?,
            })
        })
        .collect::<Vec<_>>();
    if storage.is_empty() {
        return Err(DeviceError::Missing("storage.random entries"));
    }
    let label = v.get("label").and_then(Value::as_str).unwrap_or("device").to_owned();
    Ok(Device { label, mem_read_gbps, storage })
}

/// Bytes actually read for one bundle: whole pages (O_DIRECT granularity).
pub fn read_bytes(bundle: u64) -> u64 {
    bundle.div_ceil(PAGE) * PAGE
}

/// Random-read MB/s for `block` at the closest measured queue depth,
/// interpolated in log(block size) and clamped to the measured range.
pub fn storage_mb_per_s(d: &Device, block: u64, qd: usize) -> Option<f64> {
    let best_qd = d.storage.iter().map(|p| p.queue_depth).min_by_key(|q| q.abs_diff(qd))?;
    let mut pts: Vec<&StoragePoint> = d.storage.iter().filter(|p| p.queue_depth == best_qd).collect();
    pts.sort_by_key(|p| p.block_bytes);
    let x = (block as f64).ln();
    let (first, last) = (pts.first()?, pts.last()?);
    if block <= first.block_bytes {
        return Some(first.mb_per_s);
    }
    if block >= last.block_bytes {
        return Some(last.mb_per_s);
    }
    pts.windows(2).find(|w| w[0].block_bytes <= block && block <= w[1].block_bytes).map(|w| {
        let (x0, x1) = ((w[0].block_bytes as f64).ln(), (w[1].block_bytes as f64).ln());
        w[0].mb_per_s + (w[1].mb_per_s - w[0].mb_per_s) * (x - x0) / (x1 - x0)
    })
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Prediction {
    pub device: String,
    pub policy: String,
    pub ram_gb_per_token: f64,
    pub storage_mb_per_token: f64,
    /// Bytes read from storage ÷ bytes needed (page rounding).
    pub read_amplification: f64,
    pub tok_s_serial: f64,
    pub tok_s_overlapped: f64,
}

/// `resident_bytes_per_token`: weights always in RAM that one token reads
/// (sparse attention + dense lm_head), from the byte budget.
pub fn predict(d: &Device, s: &SimStats, resident_bytes_per_token: f64, qd: usize) -> Option<Prediction> {
    let tokens = s.tokens.max(1) as f64;
    let ram = resident_bytes_per_token + s.hit_bytes as f64 / tokens;
    let ram_secs = ram / (d.mem_read_gbps * 1e9);
    let per_matrix: Vec<(f64, f64)> = s
        .matrices
        .iter()
        .filter(|m| m.cacheable)
        .map(|m| {
            let block = read_bytes(m.bytes_per_channel as u64);
            let bytes = m.misses as f64 * block as f64 / tokens;
            Some((bytes, bytes / (storage_mb_per_s(d, block, qd)? * 1e6)))
        })
        .collect::<Option<_>>()?;
    let storage_bytes: f64 = per_matrix.iter().map(|p| p.0).sum();
    let storage_secs: f64 = per_matrix.iter().map(|p| p.1).sum();
    let needed = (s.requested_bytes - s.hit_bytes) as f64 / tokens;
    Some(Prediction {
        device: d.label.clone(),
        policy: s.policy.clone(),
        ram_gb_per_token: ram / 1e9,
        storage_mb_per_token: storage_bytes / 1e6,
        read_amplification: if needed > 0.0 { storage_bytes / needed } else { 1.0 },
        tok_s_serial: 1.0 / (ram_secs + storage_secs),
        tok_s_overlapped: 1.0 / ram_secs.max(storage_secs),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::MatrixStats;

    fn device() -> Device {
        Device {
            label: "d".into(),
            mem_read_gbps: 10.0,
            storage: vec![
                StoragePoint { block_bytes: 4096, queue_depth: 4, mb_per_s: 100.0 },
                StoragePoint { block_bytes: 65536, queue_depth: 4, mb_per_s: 400.0 },
                StoragePoint { block_bytes: 4096, queue_depth: 1, mb_per_s: 50.0 },
            ],
        }
    }

    #[test]
    fn interpolates_in_log_block_size() {
        let d = device();
        assert_eq!(storage_mb_per_s(&d, 4096, 4), Some(100.0));
        assert_eq!(storage_mb_per_s(&d, 16384, 4), Some(250.0)); // halfway in log2
        assert_eq!(storage_mb_per_s(&d, 1 << 20, 4), Some(400.0));
        assert_eq!(storage_mb_per_s(&d, 4096, 2), Some(50.0)); // nearest qd is 1
    }

    #[test]
    fn page_rounding() {
        assert_eq!(read_bytes(2304), 4096);
        assert_eq!(read_bytes(16128), 16384);
    }

    #[test]
    fn prediction_adds_ram_and_storage_time() {
        let s = SimStats {
            policy: "p".into(),
            tokens: 1,
            capacity_bytes: 0,
            requested_bytes: 2304 * 10,
            hit_bytes: 0,
            byte_hit_rate: 0.0,
            miss_bytes_per_token: 23040.0,
            matrices: vec![MatrixStats {
                name: "down".into(),
                cacheable: true,
                bytes_per_channel: 2304,
                active_fraction: 1.0,
                requested_bytes: 23040,
                hit_bytes: 0,
                misses: 10,
            }],
        };
        let p = predict(&device(), &s, 1e9, 4).unwrap();
        // RAM: 1e9 B / 10 GB/s = 0.1 s. Storage: 10 × 4096 B / 100 MB/s = 0.4096 ms.
        assert!((1.0 / p.tok_s_serial - 0.1004096).abs() < 1e-9);
        assert!((1.0 / p.tok_s_overlapped - 0.1).abs() < 1e-12);
        assert!((p.read_amplification - 4096.0 / 2304.0).abs() < 1e-12);
    }

    #[test]
    fn parses_probe_manifest() {
        let v: Value = serde_json::json!({
            "label": "pi5",
            "decode_ceiling": {"read_gbps": 9.5},
            "storage": {"random": [{"block_bytes": 4096, "queue_depth": 1, "mb_per_s": 20.0}]}
        });
        let d = from_probe(&v).unwrap();
        assert_eq!((d.label.as_str(), d.mem_read_gbps, d.storage.len()), ("pi5", 9.5, 1));
        assert!(from_probe(&serde_json::json!({})).is_err());
    }
}
