use crate::{EngineError, Result};
use hf_hub::api::sync::Api;
use std::path::PathBuf;

/// Downloads model artifacts from the Hugging Face Hub into its managed cache.
#[derive(Clone, Debug, Default)]
pub struct HuggingFaceHub;

impl HuggingFaceHub {
    pub fn download(&self, repository: &str, filename: &str) -> Result<PathBuf> {
        if repository.trim().is_empty() || filename.trim().is_empty() {
            return Err(EngineError::HuggingFaceHub(
                "repository and filename must not be empty".to_owned(),
            ));
        }
        let api = Api::new().map_err(|error| EngineError::HuggingFaceHub(error.to_string()))?;
        api.model(repository.to_owned())
            .get(filename)
            .map_err(|error| EngineError::HuggingFaceHub(error.to_string()))
    }
}
