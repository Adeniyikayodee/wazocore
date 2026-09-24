//! bf16 safetensors -> `.satl` (docs/satl-format.md): column-major q4
//! projections for input-sparse GEMV, row-major q8 embeddings and lm_head.
//!
//! Pure core: `codec`, `format`, `plan`, `encode`. Imperative shell:
//! `main.rs` (mmap, hashing, parallel encode, file writes).

pub mod codec;
pub mod encode;
pub mod format;
pub mod plan;
