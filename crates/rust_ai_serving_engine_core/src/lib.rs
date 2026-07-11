//! Local model registry primitives.
//!
//! This crate deliberately contains no inference backend. It establishes the
//! durable contract that Candle and GGUF runtime crates will consume later.

mod error;
mod generation;
mod hub;
mod manifest;
mod registry;
mod runtime;

pub use error::{EngineError, Result};
pub use generation::{
    GenerationConfig, GenerationResult, GenerationStopReason, LogitsSampler, ModelAdapter,
    TokenDecoder, generate, generate_with,
};
pub use hub::HuggingFaceHub;
pub use manifest::{ModelFormat, ModelKind, ModelManifest};
pub use registry::{ImportedModel, ModelRegistry};
pub use runtime::{DevicePreference, RuntimeDevice, RuntimeDeviceKind};
