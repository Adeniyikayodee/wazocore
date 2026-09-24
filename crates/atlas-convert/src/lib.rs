//! bf16 safetensors -> `.satl` (docs/satl-format.md): column-major q4
//! projections for input-sparse GEMV, row-major q8 embeddings and lm_head.
//!
//! Pure core: `plan`, `encode` (with `codec` and `format` from atlas-satl).
//! Imperative shell: `main.rs` (mmap, hashing, parallel encode, file writes).

pub use atlas_satl::{codec, format};

pub mod encode;
pub mod plan;
