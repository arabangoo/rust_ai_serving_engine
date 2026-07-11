use std::path::Path;

use rust_ai_serving_engine_core::{EngineError, Result};
use tokenizers::Tokenizer;

/// Local Hugging Face `tokenizer.json` adapter.
pub struct LocalTokenizer {
    inner: Tokenizer,
}

impl LocalTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let inner = Tokenizer::from_file(path)
            .map_err(|error| EngineError::Tokenizer(error.to_string()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|error| EngineError::Tokenizer(error.to_string()))
    }

    pub fn decode(&self, tokens: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(tokens, skip_special_tokens)
            .map_err(|error| EngineError::Tokenizer(error.to_string()))
    }
}
