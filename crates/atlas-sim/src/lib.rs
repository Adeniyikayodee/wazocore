//! Trace-driven simulator for streaming N-ATLAS MLP weights from storage.
//! Decides Gate G1 (PROJECT-PLAN-B §6) before any streaming code exists.
//!
//! Pure core: `trace`, `rng`, `synth`, `freq`, `sim`, `device`.
//! Imperative shell: `main.rs` (file I/O, CLI).

pub mod device;
pub mod freq;
pub mod rng;
pub mod sim;
pub mod synth;
pub mod trace;
