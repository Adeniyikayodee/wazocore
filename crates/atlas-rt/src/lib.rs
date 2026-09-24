//! N-ATLAS CPU runtime. Reference (scalar) path first: every faster kernel
//! must match it (PROJECT-PLAN-B §3.4).
//!
//! Pure core: `weights` (views over mapped bytes), `kernels`, `ops`,
//! `chat`, and `forward` (state only in an explicit `&mut KvCache`).
//! Imperative shell: `main.rs`.

pub mod chat;
pub mod forward;
pub mod kernels;
pub mod ops;
pub mod weights;
