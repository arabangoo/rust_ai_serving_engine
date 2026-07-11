use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelKind {
    Generator,
    Embedding,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Safetensors,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct ModelManifest {
    pub id: String,
    pub kind: ModelKind,
    pub format: ModelFormat,
    pub weights: String,
    pub sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokenizer_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chat_template: Option<String>,
}
