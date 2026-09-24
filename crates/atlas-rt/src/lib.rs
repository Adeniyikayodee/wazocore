//! N-ATLAS CPU runtime. Reference (scalar) path first: every faster kernel
//! must match it (PROJECT-PLAN-B §3.4).
//!
//! Pure core: `weights` (views over mapped bytes), `kernels` (scalar
//! reference), `int8` (activation-quantised inner kernels and their
//! scalar spec), `fast` (parallel, checked against both), `ops`,
//! `chat`, and `forward` (state only in an explicit `&mut KvCache`).
//! Imperative shell: `main.rs`.

pub mod chat;
pub mod fast;
pub mod gemm;
pub(crate) mod int8;
pub mod forward;
pub mod kernels;
pub mod metrics;
pub mod ops;
pub mod weights;
