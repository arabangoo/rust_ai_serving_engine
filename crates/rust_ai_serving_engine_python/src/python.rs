use std::{
    collections::HashMap,
    sync::{OnceLock, PoisonError},
};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rust_ai_serving_engine_core::{
    DevicePreference, GenerationConfig, HuggingFaceHub, ModelKind, ModelRegistry, RuntimeDevice,
    generate, generate_with,
};
use rust_ai_serving_engine_models::{ChatMessage, LlamaGgufDecoder, LocalTokenizer, SessionCache};

/// Loaded models are cached for the lifetime of the Python process so that
/// repeated generate calls skip re-hashing and re-loading multi-gigabyte
/// weights.
static SESSION_CACHE: OnceLock<SessionCache> = OnceLock::new();

fn session_cache() -> &'static SessionCache {
    SESSION_CACHE.get_or_init(SessionCache::new)
}

fn python_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn parse_model_kind(kind: &str) -> PyResult<ModelKind> {
    match kind {
        "generator" => Ok(ModelKind::Generator),
        "embedding" => Ok(ModelKind::Embedding),
        _ => Err(PyRuntimeError::new_err(
            "kind must be either 'generator' or 'embedding'",
        )),
    }
}

fn parse_device_preference(device: &str) -> PyResult<DevicePreference> {
    match device {
        "auto" => Ok(DevicePreference::Auto),
        "cpu" => Ok(DevicePreference::Cpu),
        "cuda" => Ok(DevicePreference::Cuda),
        "metal" => Ok(DevicePreference::Metal),
        _ => Err(PyRuntimeError::new_err(
            "device must be one of 'auto', 'cpu', 'cuda', or 'metal'",
        )),
    }
}

fn parse_chat_messages(messages: Vec<HashMap<String, String>>) -> PyResult<Vec<ChatMessage>> {
    messages
        .into_iter()
        .map(|mut message| {
            let role = message
                .remove("role")
                .ok_or_else(|| PyRuntimeError::new_err("every chat message needs a 'role' key"))?;
            let content = message.remove("content").ok_or_else(|| {
                PyRuntimeError::new_err("every chat message needs a 'content' key")
            })?;
            Ok(ChatMessage { role, content })
        })
        .collect()
}

/// Registers a local GGUF or Safetensors file and returns its TOML manifest.
#[pyfunction]
#[pyo3(signature = (store, path, id, kind="generator", architecture=None, context_length=None, chat_template=None))]
fn import_model(
    py: Python<'_>,
    store: &str,
    path: &str,
    id: &str,
    kind: &str,
    architecture: Option<String>,
    context_length: Option<u32>,
    chat_template: Option<String>,
) -> PyResult<String> {
    let kind = parse_model_kind(kind)?;
    py.allow_threads(|| {
        let registry = ModelRegistry::open(store).map_err(python_error)?;
        let imported = registry
            .import_local(id, path, kind, architecture, context_length, chat_template)
            .map_err(python_error)?;
        toml::to_string_pretty(&imported.manifest).map_err(python_error)
    })
}

/// Downloads one model artifact from Hugging Face Hub, registers it, and returns its TOML manifest.
#[pyfunction]
#[pyo3(signature = (store, repo, file, id, kind="generator", architecture=None, context_length=None, chat_template=None))]
fn pull_model(
    py: Python<'_>,
    store: &str,
    repo: &str,
    file: &str,
    id: &str,
    kind: &str,
    architecture: Option<String>,
    context_length: Option<u32>,
    chat_template: Option<String>,
) -> PyResult<String> {
    let kind = parse_model_kind(kind)?;
    py.allow_threads(|| {
        let weights = HuggingFaceHub.download(repo, file).map_err(python_error)?;
        let registry = ModelRegistry::open(store).map_err(python_error)?;
        let imported = registry
            .import_local(
                id,
                weights,
                kind,
                architecture,
                context_length,
                chat_template,
            )
            .map_err(python_error)?;
        toml::to_string_pretty(&imported.manifest).map_err(python_error)
    })
}

/// Returns the TOML manifest for a registered model.
#[pyfunction]
fn inspect_model(store: &str, id: &str) -> PyResult<String> {
    let registry = ModelRegistry::open(store).map_err(python_error)?;
    toml::to_string_pretty(&registry.get(id).map_err(python_error)?).map_err(python_error)
}

/// Returns registered model identifiers in sorted order.
#[pyfunction]
fn list_models(store: &str) -> PyResult<Vec<String>> {
    let registry = ModelRegistry::open(store).map_err(python_error)?;
    Ok(registry
        .list()
        .map_err(python_error)?
        .into_iter()
        .map(|manifest| manifest.id)
        .collect())
}

/// Recomputes a registered model's SHA-256 digest and returns its TOML manifest.
#[pyfunction]
fn verify_model(py: Python<'_>, store: &str, id: &str) -> PyResult<String> {
    py.allow_threads(|| {
        let registry = ModelRegistry::open(store).map_err(python_error)?;
        toml::to_string_pretty(&registry.verify(id).map_err(python_error)?).map_err(python_error)
    })
}

/// Attaches and hashes the `tokenizer.json` required by a registered model.
#[pyfunction]
fn attach_tokenizer(py: Python<'_>, store: &str, id: &str, tokenizer: &str) -> PyResult<String> {
    py.allow_threads(|| {
        let registry = ModelRegistry::open(store).map_err(python_error)?;
        toml::to_string_pretty(
            &registry
                .attach_tokenizer(id, tokenizer)
                .map_err(python_error)?,
        )
        .map_err(python_error)
    })
}

/// Selects a Candle runtime backend and performs a minimal Tensor operation.
#[pyfunction]
#[pyo3(signature = (device="auto"))]
fn probe_runtime(device: &str) -> PyResult<String> {
    let runtime = RuntimeDevice::select(parse_device_preference(device)?).map_err(python_error)?;
    runtime.smoke_test().map_err(python_error)?;
    Ok(format!(
        "backend={:?}; accelerated={}",
        runtime.kind(),
        runtime.is_accelerated()
    ))
}

/// Generates text from a local Llama or Mistral compatible GGUF file.
///
/// `weights` and `tokenizer` are deliberately explicit; registered model
/// bundles should use `generate_registered_gguf` instead, which also caches
/// the loaded model across calls.
#[pyfunction]
#[pyo3(signature = (weights, tokenizer, prompt, max_tokens=256, temperature=0.7, top_k=Some(40), seed=0, stop_tokens=Vec::new(), device="auto"))]
fn generate_llama_gguf(
    py: Python<'_>,
    weights: &str,
    tokenizer: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    seed: u64,
    stop_tokens: Vec<u32>,
    device: &str,
) -> PyResult<String> {
    let device = parse_device_preference(device)?;
    py.allow_threads(|| {
        let tokenizer = LocalTokenizer::from_file(tokenizer).map_err(python_error)?;
        let prompt_tokens = tokenizer.encode(prompt, true).map_err(python_error)?;
        let runtime = RuntimeDevice::select(device).map_err(python_error)?;
        let mut decoder = LlamaGgufDecoder::load(weights, &runtime).map_err(python_error)?;
        let config = with_eos(
            GenerationConfig {
                max_tokens,
                temperature,
                top_k,
                seed,
                stop_tokens,
            },
            rust_ai_serving_engine_core::TokenDecoder::eos_token(&decoder),
        );
        let generated =
            generate(&mut decoder, &prompt_tokens, &config, || false).map_err(python_error)?;
        tokenizer
            .decode(&generated.tokens, true)
            .map_err(python_error)
    })
}

/// Generates through a registered GGUF model bundle using its model identifier.
///
/// The loaded model is cached process-wide; the first call hashes and loads
/// the weights, subsequent calls reuse them.
#[pyfunction]
#[pyo3(signature = (store, id, prompt, max_tokens=256, temperature=0.7, top_k=Some(40), seed=0, stop_tokens=Vec::new(), device="auto"))]
fn generate_registered_gguf(
    py: Python<'_>,
    store: &str,
    id: &str,
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    seed: u64,
    stop_tokens: Vec<u32>,
    device: &str,
) -> PyResult<String> {
    let device = parse_device_preference(device)?;
    py.allow_threads(|| {
        let registry = ModelRegistry::open(store).map_err(python_error)?;
        let session = session_cache()
            .get_or_load(&registry, id, device)
            .map_err(python_error)?;
        let mut session = session.lock().unwrap_or_else(PoisonError::into_inner);
        let prompt_tokens = session
            .tokenizer
            .encode(prompt, true)
            .map_err(python_error)?;
        let config = with_eos(
            GenerationConfig {
                max_tokens,
                temperature,
                top_k,
                seed,
                stop_tokens,
            },
            session.eos_token,
        );
        let generated = generate(session.decoder.as_mut(), &prompt_tokens, &config, || false)
            .map_err(python_error)?;
        session
            .tokenizer
            .decode(&generated.tokens, true)
            .map_err(python_error)
    })
}

/// Chat-completes through a registered GGUF model bundle.
///
/// `messages` is a list of `{"role": ..., "content": ...}` dictionaries. The
/// model's chat template (from its manifest or its architecture default) is
/// applied before generation, and generation stops at the model's
/// end-of-sequence token.
#[pyfunction]
#[pyo3(signature = (store, id, messages, max_tokens=256, temperature=0.7, top_k=Some(40), seed=0, device="auto"))]
fn generate_chat_registered_gguf(
    py: Python<'_>,
    store: &str,
    id: &str,
    messages: Vec<HashMap<String, String>>,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    seed: u64,
    device: &str,
) -> PyResult<String> {
    let device = parse_device_preference(device)?;
    let messages = parse_chat_messages(messages)?;
    py.allow_threads(|| {
        let registry = ModelRegistry::open(store).map_err(python_error)?;
        let session = session_cache()
            .get_or_load(&registry, id, device)
            .map_err(python_error)?;
        let mut session = session.lock().unwrap_or_else(PoisonError::into_inner);
        let template = session.chat_template.ok_or_else(|| {
            PyRuntimeError::new_err(
                "no chat template is known for this model; \
                 set chat_template in its manifest (chatml, llama3, or mistral)",
            )
        })?;
        let prompt = template.render(&messages).map_err(python_error)?;
        // The template spells out every special token itself.
        let prompt_tokens = session
            .tokenizer
            .encode(&prompt, false)
            .map_err(python_error)?;
        let config = with_eos(
            GenerationConfig {
                max_tokens,
                temperature,
                top_k,
                seed,
                stop_tokens: Vec::new(),
            },
            session.eos_token,
        );
        let generated = generate(session.decoder.as_mut(), &prompt_tokens, &config, || false)
            .map_err(python_error)?;
        session
            .tokenizer
            .decode(&generated.tokens, true)
            .map_err(python_error)
    })
}

/// Streams a chat completion through a registered GGUF model bundle.
///
/// `on_delta` is called with each printable text fragment as it is decoded.
/// Returning `False` from the callback cancels generation; any other return
/// value (including `None`) continues it. Exceptions raised by the callback
/// also cancel generation and propagate to the caller. Returns the full text
/// generated so far, so a cancelled call yields the partial answer.
///
/// The generation loop runs with the GIL released; the GIL is re-acquired
/// only for the duration of each callback invocation.
#[pyfunction]
#[pyo3(signature = (store, id, messages, on_delta, max_tokens=256, temperature=0.7, top_k=Some(40), seed=0, device="auto"))]
#[allow(clippy::too_many_arguments)]
fn generate_chat_stream_registered_gguf(
    py: Python<'_>,
    store: &str,
    id: &str,
    messages: Vec<HashMap<String, String>>,
    on_delta: Py<PyAny>,
    max_tokens: usize,
    temperature: f32,
    top_k: Option<usize>,
    seed: u64,
    device: &str,
) -> PyResult<String> {
    let device = parse_device_preference(device)?;
    let messages = parse_chat_messages(messages)?;
    py.allow_threads(|| {
        let registry = ModelRegistry::open(store).map_err(python_error)?;
        let session = session_cache()
            .get_or_load(&registry, id, device)
            .map_err(python_error)?;
        let mut session = session.lock().unwrap_or_else(PoisonError::into_inner);
        let template = session.chat_template.ok_or_else(|| {
            PyRuntimeError::new_err(
                "no chat template is known for this model; \
                 set chat_template in its manifest (chatml, llama3, or mistral)",
            )
        })?;
        let prompt = template.render(&messages).map_err(python_error)?;
        // The template spells out every special token itself.
        let prompt_tokens = session
            .tokenizer
            .encode(&prompt, false)
            .map_err(python_error)?;
        let config = with_eos(
            GenerationConfig {
                max_tokens,
                temperature,
                top_k,
                seed,
                stop_tokens: Vec::new(),
            },
            session.eos_token,
        );

        // Reborrow as a plain reference so decoder/tokenizer field borrows split.
        let session = &mut *session;
        let decoder = session.decoder.as_mut();
        let tokenizer = &session.tokenizer;
        let mut tokens: Vec<u32> = Vec::new();
        let mut emitted = 0usize;
        let mut callback_error: Option<PyErr> = None;
        let mut decode_error: Option<rust_ai_serving_engine_core::EngineError> = None;

        // Delivers `delta` to the Python callback. Ok(true) = keep generating.
        // Only an explicit `False` return cancels; `None` and others continue.
        let emit = |delta: &str| -> PyResult<bool> {
            Python::with_gil(|py| {
                let value = on_delta.call1(py, (delta,))?;
                Ok(value.extract::<bool>(py).unwrap_or(true))
            })
        };

        generate_with(
            decoder,
            &prompt_tokens,
            &config,
            || false,
            |token| {
                tokens.push(token);
                let full = match tokenizer.decode(&tokens, true) {
                    Ok(full) => full,
                    Err(error) => {
                        decode_error = Some(error);
                        return false;
                    }
                };
                // An incomplete UTF-8 sequence at the tail means the next
                // token carries the rest of the character; wait for it.
                if full.ends_with('\u{FFFD}') {
                    return true;
                }
                if full.len() > emitted {
                    let delta = full[emitted..].to_owned();
                    emitted = full.len();
                    return match emit(&delta) {
                        Ok(keep_going) => keep_going,
                        Err(error) => {
                            callback_error = Some(error);
                            false
                        }
                    };
                }
                true
            },
        )
        .map_err(python_error)?;

        if let Some(error) = callback_error {
            return Err(error);
        }
        if let Some(error) = decode_error {
            return Err(python_error(error));
        }
        let text = tokenizer.decode(&tokens, true).map_err(python_error)?;
        if text.len() > emitted {
            emit(&text[emitted..])?;
        }
        Ok(text)
    })
}

/// Drops every cached model session, releasing their memory.
#[pyfunction]
fn unload_models() -> PyResult<()> {
    session_cache().clear();
    Ok(())
}

/// Reports the wgpu prefill-offload status: "active: <adapter>",
/// "fallback(runtime-failure): <adapter>", or "inactive" (no usable GPU,
/// software adapter excluded, or RASE_GPU=0).
#[pyfunction]
fn gpu_info() -> PyResult<String> {
    Ok(rust_ai_serving_engine_models::gpu_gemm::status())
}

/// Returns forward-pass phase counters as a JSON string.
///
/// Profiling is collected only when the process was started with
/// RASE_PROFILE=1; otherwise every counter stays zero (and the default
/// inference path takes no timing probes). `reset=True` zeroes the counters
/// after reading, so successive calls delimit measurement windows.
#[pyfunction]
#[pyo3(signature = (reset=true))]
fn profiling_snapshot(reset: bool) -> PyResult<String> {
    Ok(rust_ai_serving_engine_models::profiling::snapshot(reset))
}

fn with_eos(mut config: GenerationConfig, eos_token: Option<u32>) -> GenerationConfig {
    if let Some(eos) = eos_token {
        if !config.stop_tokens.contains(&eos) {
            config.stop_tokens.push(eos);
        }
    }
    config
}

/// Python module exposed by maturin as `rust_ai_serving_engine`.
#[pymodule]
fn rust_ai_serving_engine(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add_function(wrap_pyfunction!(import_model, module)?)?;
    module.add_function(wrap_pyfunction!(pull_model, module)?)?;
    module.add_function(wrap_pyfunction!(inspect_model, module)?)?;
    module.add_function(wrap_pyfunction!(list_models, module)?)?;
    module.add_function(wrap_pyfunction!(verify_model, module)?)?;
    module.add_function(wrap_pyfunction!(attach_tokenizer, module)?)?;
    module.add_function(wrap_pyfunction!(probe_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(generate_llama_gguf, module)?)?;
    module.add_function(wrap_pyfunction!(generate_registered_gguf, module)?)?;
    module.add_function(wrap_pyfunction!(generate_chat_registered_gguf, module)?)?;
    module.add_function(wrap_pyfunction!(
        generate_chat_stream_registered_gguf,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(unload_models, module)?)?;
    module.add_function(wrap_pyfunction!(profiling_snapshot, module)?)?;
    module.add_function(wrap_pyfunction!(gpu_info, module)?)?;
    Ok(())
}
