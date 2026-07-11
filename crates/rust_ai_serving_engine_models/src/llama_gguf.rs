use std::{
    fs::File,
    io::{BufReader, Seek},
    path::Path,
};

use candle_core::{Tensor, quantized::gguf_file};
use candle_transformers::models::quantized_llama::ModelWeights;
use rust_ai_serving_engine_core::{
    EngineError, ModelAdapter, ModelFormat, Result, RuntimeDevice, TokenDecoder,
};

use crate::gguf_eos_token;

/// Declares the GGUF layout handled by Candle's quantized Llama decoder.
#[derive(Debug, Default)]
pub struct LlamaGgufAdapter;

impl ModelAdapter for LlamaGgufAdapter {
    fn architecture(&self) -> &'static str {
        "llama-gguf"
    }

    fn supports_format(&self, format: ModelFormat) -> bool {
        matches!(format, ModelFormat::Gguf)
    }
}

/// Stateful prefill/decode decoder with a KV cache owned by Candle.
pub struct LlamaGgufDecoder {
    model: ModelWeights,
    device: candle_core::Device,
    next_position: usize,
    eos_token: Option<u32>,
}

impl LlamaGgufDecoder {
    /// Loads one local GGUF checkpoint into the selected Candle runtime.
    pub fn load(path: impl AsRef<Path>, runtime: &RuntimeDevice) -> Result<Self> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let content = gguf_file::Content::read(&mut reader)
            .map_err(|error| EngineError::Candle(error.to_string()))?;
        reader.rewind().map_err(EngineError::Io)?;
        let eos_token = gguf_eos_token(&content);
        let model = ModelWeights::from_gguf(content, &mut reader, runtime.device())
            .map_err(|error| EngineError::Candle(error.to_string()))?;

        Ok(Self {
            model,
            device: runtime.device().clone(),
            next_position: 0,
            eos_token,
        })
    }

    /// Clears the KV cache before starting an unrelated conversation.
    pub fn clear_cache(&mut self) {
        self.model.clear_kv_cache();
        self.next_position = 0;
    }

    fn forward_tokens(&mut self, tokens: &[u32], position: usize) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(EngineError::InvalidGenerationConfig(
                "the prompt must contain at least one token".to_owned(),
            ));
        }

        let input = Tensor::new(tokens, &self.device)
            .and_then(|tensor| tensor.unsqueeze(0))
            .map_err(|error| EngineError::Candle(error.to_string()))?;
        // Candle returns last-position logits shaped (batch, vocabulary);
        // squeeze the batch dimension before flattening to a plain vector.
        self.model
            .forward(&input, position)
            .and_then(|logits| logits.squeeze(0))
            .and_then(|logits| logits.to_vec1::<f32>())
            .map_err(|error| EngineError::Candle(error.to_string()))
    }
}

impl TokenDecoder for LlamaGgufDecoder {
    fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>> {
        self.clear_cache();
        let logits = self.forward_tokens(prompt, 0)?;
        self.next_position = prompt.len();
        Ok(logits)
    }

    fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        let logits = self.forward_tokens(&[token], self.next_position)?;
        self.next_position += 1;
        Ok(logits)
    }

    fn eos_token(&self) -> Option<u32> {
        self.eos_token
    }
}
