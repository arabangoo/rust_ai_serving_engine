//! OpenAI-compatible HTTP surface for locally registered GGUF models.
//!
//! Loaded models are cached in a process-wide `SessionCache`; weights are
//! hashed and loaded once, not per request. Generation runs on blocking
//! worker threads, and chat streaming pushes tokens through a channel that
//! ends decoding as soon as the client disconnects.

use std::{
    convert::Infallible,
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex, PoisonError},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use rust_ai_serving_engine_core::{
    DevicePreference, EngineError, GenerationConfig, GenerationStopReason, ModelRegistry,
    generate_with,
};
use rust_ai_serving_engine_models::{ChatMessage, ModelSession, SessionCache};
use serde::{Deserialize, Serialize};
use tokio_stream::{StreamExt, wrappers::ReceiverStream};

#[derive(Clone)]
pub struct ApiState {
    store: PathBuf,
    device: DevicePreference,
    cache: Arc<SessionCache>,
}

impl ApiState {
    pub fn new(store: impl Into<PathBuf>, device: DevicePreference) -> Self {
        Self {
            store: store.into(),
            device,
            cache: Arc::new(SessionCache::new()),
        }
    }

    fn session(&self, id: &str) -> Result<Arc<Mutex<ModelSession>>, ApiError> {
        let registry = ModelRegistry::open(&self.store).map_err(api_error)?;
        self.cache
            .get_or_load(&registry, id, self.device)
            .map_err(api_error)
    }
}

pub fn router(state: ApiState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/models/{id}", get(inspect_model))
        .route("/v1/completions", post(completion))
        .route("/v1/chat/completions", post(chat_completion))
        .with_state(Arc::new(state))
}

pub async fn serve(address: SocketAddr, state: ApiState) -> Result<(), std::io::Error> {
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, router(state)).await
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn list_models(State(state): State<Arc<ApiState>>) -> ApiResult<ModelsResponse> {
    let registry = ModelRegistry::open(&state.store).map_err(api_error)?;
    let data = registry
        .list()
        .map_err(api_error)?
        .into_iter()
        .map(|manifest| ModelResponse {
            id: manifest.id,
            object: "model",
        })
        .collect();
    Ok(Json(ModelsResponse {
        object: "list",
        data,
    }))
}

async fn inspect_model(
    State(state): State<Arc<ApiState>>,
    Path(id): Path<String>,
) -> ApiResult<ModelResponse> {
    let registry = ModelRegistry::open(&state.store).map_err(api_error)?;
    let manifest = registry.get(&id).map_err(api_error)?;
    Ok(Json(ModelResponse {
        id: manifest.id,
        object: "model",
    }))
}

async fn completion(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<CompletionRequest>,
) -> ApiResult<CompletionResponse> {
    if request.stream.unwrap_or(false) {
        return Err(ApiError::not_implemented(
            "stream=true is only implemented for /v1/chat/completions",
        ));
    }
    let state = (*state).clone();
    let output = tokio::task::spawn_blocking(move || complete(state, request))
        .await
        .map_err(|error| ApiError::internal(error.to_string()))??;
    Ok(Json(output))
}

fn complete(state: ApiState, request: CompletionRequest) -> Result<CompletionResponse, ApiError> {
    let session = state.session(&request.model)?;
    let mut session = session.lock().unwrap_or_else(PoisonError::into_inner);
    let prompt_tokens = session
        .tokenizer
        .encode(&request.prompt, true)
        .map_err(api_error)?;
    let params = GenerationParams {
        config: GenerationConfig {
            max_tokens: request.max_tokens.unwrap_or(256),
            temperature: request.temperature.unwrap_or(0.7),
            top_k: request.top_k.or(Some(40)),
            seed: request.seed.unwrap_or(0),
            stop_tokens: Vec::new(),
        },
        stop_strings: request.stop.map(StopField::into_vec).unwrap_or_default(),
    };
    let outcome = run_generation(&mut session, &prompt_tokens, &params, |_| true)?;
    Ok(CompletionResponse {
        id: format!("cmpl-{}", request.model),
        object: "text_completion",
        created: unix_timestamp(),
        model: request.model,
        choices: vec![CompletionChoice {
            text: outcome.text,
            index: 0,
            finish_reason: outcome.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: outcome.completion_tokens,
            total_tokens: prompt_tokens.len() + outcome.completion_tokens,
        },
    })
}

async fn chat_completion(
    State(state): State<Arc<ApiState>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let stream = request.stream.unwrap_or(false);
    let state = (*state).clone();
    if !stream {
        let output = tokio::task::spawn_blocking(move || chat_complete(state, request))
            .await
            .map_err(|error| ApiError::internal(error.to_string()))??;
        return Ok(Json(output).into_response());
    }

    // Resolve and load the model before opening the SSE response so that
    // registry and load failures still surface as plain HTTP errors.
    let model = request.model.clone();
    let session = {
        let state = state.clone();
        let model = model.clone();
        tokio::task::spawn_blocking(move || state.session(&model))
            .await
            .map_err(|error| ApiError::internal(error.to_string()))??
    };

    let (sender, receiver) = tokio::sync::mpsc::channel::<Event>(64);
    tokio::task::spawn_blocking(move || chat_stream(session, request, sender));
    let stream = ReceiverStream::new(receiver).map(Ok::<Event, Infallible>);
    Ok(Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response())
}

fn chat_complete(
    state: ApiState,
    request: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, ApiError> {
    let session = state.session(&request.model)?;
    let mut session = session.lock().unwrap_or_else(PoisonError::into_inner);
    let (prompt_tokens, params) = prepare_chat(&session, &request)?;
    let outcome = run_generation(&mut session, &prompt_tokens, &params, |_| true)?;
    Ok(ChatCompletionResponse {
        id: format!("chatcmpl-{}", request.model),
        object: "chat.completion",
        created: unix_timestamp(),
        model: request.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessageBody {
                role: "assistant".to_owned(),
                content: outcome.text,
            },
            finish_reason: outcome.finish_reason,
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens.len(),
            completion_tokens: outcome.completion_tokens,
            total_tokens: prompt_tokens.len() + outcome.completion_tokens,
        },
    })
}

/// Runs one streaming chat generation, pushing OpenAI-style chunks into the
/// SSE channel. A closed channel (client disconnect) cancels decoding.
fn chat_stream(
    session: Arc<Mutex<ModelSession>>,
    request: ChatCompletionRequest,
    sender: tokio::sync::mpsc::Sender<Event>,
) {
    let model = request.model.clone();
    let created = unix_timestamp();
    let chunk = |delta: serde_json::Value, finish_reason: Option<&str>| {
        Event::default().data(
            serde_json::json!({
                "id": format!("chatcmpl-{model}"),
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
            })
            .to_string(),
        )
    };

    let mut session = session.lock().unwrap_or_else(PoisonError::into_inner);
    let outcome = prepare_chat(&session, &request).and_then(|(prompt_tokens, params)| {
        if sender
            .blocking_send(chunk(serde_json::json!({ "role": "assistant" }), None))
            .is_err()
        {
            return Err(ApiError::internal("client disconnected"));
        }
        run_generation(&mut session, &prompt_tokens, &params, |delta| {
            sender
                .blocking_send(chunk(serde_json::json!({ "content": delta }), None))
                .is_ok()
        })
    });

    match outcome {
        Ok(outcome) => {
            let _ = sender.blocking_send(chunk(serde_json::json!({}), Some(outcome.finish_reason)));
            let _ = sender.blocking_send(Event::default().data("[DONE]"));
        }
        Err(error) => {
            let _ = sender.blocking_send(
                Event::default().data(
                    serde_json::json!({
                        "error": { "message": error.message, "type": "server_error" }
                    })
                    .to_string(),
                ),
            );
        }
    }
}

/// Renders the chat template and assembles generation parameters.
fn prepare_chat(
    session: &ModelSession,
    request: &ChatCompletionRequest,
) -> Result<(Vec<u32>, GenerationParams), ApiError> {
    let template = session.chat_template.ok_or_else(|| {
        ApiError::bad_request(
            "no chat template is known for this model; \
             set chat_template in its manifest (chatml, llama3, or mistral)",
        )
    })?;
    let messages: Vec<ChatMessage> = request
        .messages
        .iter()
        .map(|message| ChatMessage {
            role: message.role.clone(),
            content: message.content.clone(),
        })
        .collect();
    let prompt = template.render(&messages).map_err(api_error)?;
    // The template already spells out every special token, so the tokenizer
    // must not inject its own.
    let prompt_tokens = session
        .tokenizer
        .encode(&prompt, false)
        .map_err(api_error)?;
    let params = GenerationParams {
        config: GenerationConfig {
            max_tokens: request.max_tokens.unwrap_or(256),
            temperature: request.temperature.unwrap_or(0.7),
            top_k: request.top_k.or(Some(40)),
            seed: request.seed.unwrap_or(0),
            stop_tokens: Vec::new(),
        },
        stop_strings: request
            .stop
            .clone()
            .map(StopField::into_vec)
            .unwrap_or_default(),
    };
    Ok((prompt_tokens, params))
}

struct GenerationParams {
    config: GenerationConfig,
    stop_strings: Vec<String>,
}

struct GenerationOutcome {
    text: String,
    completion_tokens: usize,
    finish_reason: &'static str,
}

/// Shared generation core for chat and completion surfaces.
///
/// Decodes incrementally, holds back enough text to catch stop strings that
/// span chunk boundaries, and reports printable deltas through `on_delta`.
/// Returning `false` from `on_delta` cancels decoding.
fn run_generation(
    session: &mut ModelSession,
    prompt_tokens: &[u32],
    params: &GenerationParams,
    mut on_delta: impl FnMut(&str) -> bool,
) -> Result<GenerationOutcome, ApiError> {
    let mut config = params.config.clone();
    if let Some(eos) = session.eos_token {
        if !config.stop_tokens.contains(&eos) {
            config.stop_tokens.push(eos);
        }
    }
    let holdback_chars = params
        .stop_strings
        .iter()
        .map(|stop| stop.chars().count())
        .max()
        .unwrap_or(0)
        .saturating_sub(1);

    let decoder = session.decoder.as_mut();
    let tokenizer = &session.tokenizer;
    let mut tokens: Vec<u32> = Vec::new();
    let mut emitted = 0usize;
    let mut truncated: Option<String> = None;
    let mut decode_error: Option<EngineError> = None;

    let result = generate_with(
        decoder,
        prompt_tokens,
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
            // An incomplete UTF-8 sequence at the tail means the next token
            // carries the rest of the character; wait instead of emitting it.
            if full.ends_with('\u{FFFD}') {
                return true;
            }
            if let Some(position) = earliest_stop(&full, &params.stop_strings) {
                let cut = &full[..position];
                if cut.len() > emitted {
                    on_delta(&cut[emitted..]);
                    emitted = cut.len();
                }
                truncated = Some(cut.to_owned());
                return false;
            }
            let safe_end = holdback_boundary(&full, holdback_chars);
            if safe_end > emitted {
                let delta = &full[emitted..safe_end];
                emitted = safe_end;
                return on_delta(delta);
            }
            true
        },
    )
    .map_err(api_error)?;

    if let Some(error) = decode_error {
        return Err(api_error(error));
    }

    let (text, stopped_on_string) = match truncated {
        Some(text) => (text, true),
        None => (tokenizer.decode(&tokens, true).map_err(api_error)?, false),
    };
    if text.len() > emitted {
        on_delta(&text[emitted..]);
    }
    let finish_reason = match result.stop_reason {
        GenerationStopReason::StopToken => "stop",
        GenerationStopReason::MaxTokens => "length",
        GenerationStopReason::Cancelled if stopped_on_string => "stop",
        GenerationStopReason::Cancelled => "cancelled",
    };
    Ok(GenerationOutcome {
        text,
        completion_tokens: result.tokens.len(),
        finish_reason,
    })
}

fn earliest_stop(text: &str, stop_strings: &[String]) -> Option<usize> {
    stop_strings
        .iter()
        .filter(|stop| !stop.is_empty())
        .filter_map(|stop| text.find(stop.as_str()))
        .min()
}

/// Returns the byte offset that keeps the last `holdback` characters unemitted.
fn holdback_boundary(text: &str, holdback: usize) -> usize {
    if holdback == 0 {
        return text.len();
    }
    text.char_indices()
        .rev()
        .nth(holdback - 1)
        .map(|(index, _)| index)
        .unwrap_or(0)
}

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

type ApiResult<T> = Result<Json<T>, ApiError>;

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }
    fn not_implemented(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_IMPLEMENTED,
            message: message.into(),
        }
    }
    fn internal(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorResponse {
                error: ErrorBody {
                    message: self.message,
                    kind: "server_error",
                },
            }),
        )
            .into_response()
    }
}

fn api_error(error: impl std::fmt::Display) -> ApiError {
    ApiError::bad_request(error.to_string())
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
}
#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelResponse>,
}
#[derive(Serialize)]
struct ModelResponse {
    id: String,
    object: &'static str,
}

/// OpenAI's `stop` accepts either one string or a list of strings.
#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
enum StopField {
    One(String),
    Many(Vec<String>),
}

impl StopField {
    fn into_vec(self) -> Vec<String> {
        match self {
            Self::One(stop) => vec![stop],
            Self::Many(stops) => stops,
        }
    }
}

#[derive(Deserialize)]
struct CompletionRequest {
    model: String,
    prompt: String,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    stop: Option<StopField>,
    stream: Option<bool>,
}
#[derive(Serialize)]
struct CompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}
#[derive(Serialize)]
struct CompletionChoice {
    text: String,
    index: usize,
    finish_reason: &'static str,
}

#[derive(Deserialize)]
struct ChatCompletionRequest {
    model: String,
    messages: Vec<ChatMessageBody>,
    max_tokens: Option<usize>,
    temperature: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    stop: Option<StopField>,
    stream: Option<bool>,
}
#[derive(Clone, Deserialize, Serialize)]
struct ChatMessageBody {
    role: String,
    content: String,
}
#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}
#[derive(Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessageBody,
    finish_reason: &'static str,
}

#[derive(Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}
#[derive(Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}
#[derive(Serialize)]
struct ErrorBody {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_field_accepts_string_or_list() {
        let one: StopField = serde_json::from_str("\"END\"").unwrap();
        assert_eq!(one.into_vec(), vec!["END".to_owned()]);
        let many: StopField = serde_json::from_str("[\"a\", \"b\"]").unwrap();
        assert_eq!(many.into_vec(), vec!["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn holdback_keeps_trailing_characters() {
        assert_eq!(holdback_boundary("hello", 0), 5);
        assert_eq!(holdback_boundary("hello", 2), 3);
        assert_eq!(holdback_boundary("한글자", 1), "한글".len());
        assert_eq!(holdback_boundary("ab", 5), 0);
    }

    #[test]
    fn earliest_stop_finds_the_first_occurrence() {
        let stops = vec!["</s>".to_owned(), "END".to_owned()];
        assert_eq!(earliest_stop("output END </s>", &stops), Some(7));
        assert_eq!(earliest_stop("plain output", &stops), None);
    }
}
