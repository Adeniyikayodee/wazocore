//! Day-1 hardware probes for the N-ATLAS edge runtime: memory bandwidth,
//! storage read profile and CPU topology, emitted as a JSON manifest that is
//! attached to every benchmark result.
//!
//! Pure core: `manifest` (plain data), `config`, `stats`, `budget`,
//! `cpuinfo`, `report`. Imperative shell: `host`, `memory`, `storage`.

pub mod budget;
pub mod config;
pub mod cpuinfo;
pub mod host;
pub mod manifest;
pub mod memory;
pub mod report;
pub mod stats;
pub mod storage;
