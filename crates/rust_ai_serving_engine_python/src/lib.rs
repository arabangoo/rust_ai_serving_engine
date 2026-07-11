//! Python extension module for `rust_ai_serving_engine`.
//!
//! The Rust core remains usable without Python. This crate is enabled only by
//! maturin or an explicit `--features python` build.

#[cfg(feature = "python")]
mod python;
