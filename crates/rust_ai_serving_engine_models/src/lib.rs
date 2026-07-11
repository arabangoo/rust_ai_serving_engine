//! Candle-backed adapters for locally stored model weights.
//!
//! The first adapter targets GGUF files compatible with Candle's quantized
//! Llama decoder. This includes the common Llama and Mistral GGUF layouts.
//! Qwen2-compatible GGUF files use Candle's quantized Qwen2 decoder.

mod chat;
mod llama_gguf;
mod qwen2_gguf;
mod session;
mod tokenizer;

pub use chat::{ChatMessage, ChatTemplate};
pub use llama_gguf::{LlamaGgufAdapter, LlamaGgufDecoder};
pub use qwen2_gguf::{Qwen2GgufAdapter, Qwen2GgufDecoder};
pub use session::{ModelSession, SessionCache};
pub use tokenizer::LocalTokenizer;

use std::path::Path;

use candle_core::quantized::gguf_file;
use rust_ai_serving_engine_core::{EngineError, Result, RuntimeDevice, TokenDecoder};

/// Opens a GGUF decoder selected by the registered architecture name.
pub fn load_gguf_decoder(
    architecture: &str,
    weights: impl AsRef<Path>,
    runtime: &RuntimeDevice,
) -> Result<Box<dyn TokenDecoder>> {
    match architecture.to_ascii_lowercase().as_str() {
        "llama" | "llama2" | "llama3" | "llama-gguf" | "mistral" | "mixtral" => {
            Ok(Box::new(LlamaGgufDecoder::load(weights, runtime)?))
        }
        "qwen" | "qwen2" | "qwen2.5" | "qwen2-gguf" => {
            Ok(Box::new(Qwen2GgufDecoder::load(weights, runtime)?))
        }
        "phi" | "phi2" | "phi3" | "phi4" => Err(EngineError::UnsupportedArchitecture(
            "Phi GGUF is not yet enabled because Candle's Phi variants do not expose a safe KV-cache reset API".to_owned(),
        )),
        other => Err(EngineError::UnsupportedArchitecture(other.to_owned())),
    }
}

/// Reads the end-of-sequence token id recorded in GGUF metadata, when present.
pub(crate) fn gguf_eos_token(content: &gguf_file::Content) -> Option<u32> {
    content
        .metadata
        .get("tokenizer.ggml.eos_token_id")
        .and_then(|value| value.to_u32().ok())
}
