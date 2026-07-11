//! Loaded-model sessions and the process-wide session cache.
//!
//! Loading a GGUF checkpoint costs seconds and hashing its weights costs
//! seconds more, so neither may happen once per request. A session pairs a
//! loaded decoder with its tokenizer and template, and the cache hands out
//! the same session as long as the registered manifest hash is unchanged.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex, PoisonError},
};

use rust_ai_serving_engine_core::{
    DevicePreference, EngineError, ModelFormat, ModelManifest, ModelRegistry, Result,
    RuntimeDevice, TokenDecoder,
};

use crate::{ChatTemplate, LocalTokenizer, load_gguf_decoder};

/// One loaded GGUF model with everything needed to serve requests.
pub struct ModelSession {
    pub decoder: Box<dyn TokenDecoder>,
    pub tokenizer: LocalTokenizer,
    pub eos_token: Option<u32>,
    pub chat_template: Option<ChatTemplate>,
}

impl ModelSession {
    /// Loads the decoder and tokenizer bundle described by a manifest.
    pub fn load(manifest: &ModelManifest, runtime: &RuntimeDevice) -> Result<Self> {
        if !matches!(manifest.format, ModelFormat::Gguf) {
            return Err(EngineError::UnsupportedFormat(
                "only GGUF model bundles are currently executable".to_owned(),
            ));
        }
        let architecture = manifest.architecture.as_deref().ok_or_else(|| {
            EngineError::UnsupportedArchitecture(
                "the registered model has no architecture metadata".to_owned(),
            )
        })?;
        let tokenizer_path = manifest.tokenizer.as_deref().ok_or_else(|| {
            EngineError::ModelFileNotFound(
                "tokenizer.json is not attached; attach it before generating".to_owned(),
            )
        })?;
        let tokenizer = LocalTokenizer::from_file(tokenizer_path)?;
        let decoder = load_gguf_decoder(architecture, &manifest.weights, runtime)?;
        let eos_token = decoder.eos_token();
        let chat_template = manifest
            .chat_template
            .as_deref()
            .and_then(ChatTemplate::from_name)
            .or_else(|| ChatTemplate::for_architecture(architecture));
        Ok(Self {
            decoder,
            tokenizer,
            eos_token,
            chat_template,
        })
    }
}

/// Process-wide cache of loaded model sessions keyed by model and device.
///
/// Generation mutates the KV cache, so each session sits behind its own
/// mutex and concurrent requests to the same model serialize. Different
/// models generate concurrently.
#[derive(Default)]
pub struct SessionCache {
    sessions: Mutex<HashMap<String, CachedSession>>,
}

struct CachedSession {
    sha256: String,
    session: Arc<Mutex<ModelSession>>,
}

impl SessionCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the cached session for a registered model, loading it when
    /// absent. Weights are integrity-checked once per load, not per call;
    /// a changed manifest hash evicts and reloads the session.
    pub fn get_or_load(
        &self,
        registry: &ModelRegistry,
        id: &str,
        device: DevicePreference,
    ) -> Result<Arc<Mutex<ModelSession>>> {
        let manifest = registry.get(id)?;
        let key = format!("{id}@{device:?}");
        {
            let sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(cached) = sessions.get(&key) {
                if cached.sha256 == manifest.sha256 {
                    return Ok(cached.session.clone());
                }
            }
        }

        let manifest = registry.verify(id)?;
        let runtime = RuntimeDevice::select(device)?;
        let session = Arc::new(Mutex::new(ModelSession::load(&manifest, &runtime)?));
        let mut sessions = self.sessions.lock().unwrap_or_else(PoisonError::into_inner);
        sessions.insert(
            key,
            CachedSession {
                sha256: manifest.sha256,
                session: session.clone(),
            },
        );
        Ok(session)
    }

    /// Drops every cached session, forcing the next call to reload.
    pub fn clear(&self) {
        self.sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clear();
    }
}
