use std::io;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("model id must contain only ASCII letters, digits, hyphens, or underscores: {0}")]
    InvalidModelId(String),

    #[error("unsupported model file extension: {0}")]
    UnsupportedFormat(String),

    #[error("unsupported model architecture: {0}")]
    UnsupportedArchitecture(String),

    #[error("model manifest does not exist: {0}")]
    ModelNotFound(String),

    #[error("model file does not exist: {0}")]
    ModelFileNotFound(String),

    #[error("model integrity verification failed for {id}: expected {expected}, actual {actual}")]
    IntegrityMismatch {
        id: String,
        expected: String,
        actual: String,
    },

    #[error("requested backend is unavailable: {0}")]
    BackendUnavailable(String),

    #[error("Candle runtime error: {0}")]
    Candle(String),

    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Hugging Face Hub error: {0}")]
    HuggingFaceHub(String),

    #[error("invalid generation configuration: {0}")]
    InvalidGenerationConfig(String),

    #[error("model returned invalid logits")]
    InvalidLogits,

    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    #[error("manifest serialization error: {0}")]
    TomlSerialize(#[from] toml::ser::Error),

    #[error("manifest parsing error: {0}")]
    TomlDeserialize(#[from] toml::de::Error),
}

pub type Result<T> = std::result::Result<T, EngineError>;
