# rust_ai_serving_engine

> **Rust 기반 로컬 AI 모델 서빙 엔진**
>
> Hugging Face 에서 내려받은 GGUF 모델을 `등록 → 무결성 검증 → 로드 → 추론 → OpenAI 호환 API 서빙`으로 흘려보낸다.
> Ollama · llama.cpp · LM Studio 가 제공하는 로컬 모델 실행 경험을
> **순수 Rust 단일 바이너리 + Python 한 줄 import** 로 구현한다.

이 문서는 엔진의 **완결된 개발자 매뉴얼**이다. 설계 원칙, 공개 API, 지원 모델,
채팅 템플릿과 생성 제어, HTTP/CLI/Python 사용법, 서비스 통합, 새 아키텍처 추가법, 빌드·테스트 절차를 담는다.

[주요 참고 논문]

1. Attention Is All You Need (Transformer 구조의 원전) - https://arxiv.org/abs/1706.03762
2. LLaMA: Open and Efficient Foundation Language Models (Llama 계열 디코더 구조) - https://arxiv.org/abs/2302.13971
3. Qwen3 Technical Report (실모델 종단 간 검증에 사용한 Qwen3 계열) - https://arxiv.org/abs/2505.09388
4. The Case for 4-bit Precision: k-bit Inference Scaling Laws (4비트 양자화 추론의 근거) - https://arxiv.org/abs/2212.09720
5. Efficient Memory Management for Large Language Model Serving with PagedAttention (LLM 서빙과 KV 캐시 관리) - https://arxiv.org/abs/2309.06180

---

## 목차

1. [핵심 특징](#1-핵심-특징)
2. [빠른 시작](#2-빠른-시작)
3. [설치와 Cargo Feature](#3-설치와-cargo-feature)
4. [아키텍처](#4-아키텍처)
5. [모델 매니페스트와 레지스트리](#5-모델-매니페스트와-레지스트리)
6. [공개 API 레퍼런스](#6-공개-api-레퍼런스)
7. [지원 모델](#7-지원-모델)
8. [채팅 템플릿과 생성 제어](#8-채팅-템플릿과-생성-제어)
9. [HTTP API (OpenAI 호환)](#9-http-api-openai-호환)
10. [CLI 도구](#10-cli-도구)
11. [Python 바인딩 (PyO3)](#11-python-바인딩-pyo3)
12. [서비스 파이프라인에 붙이기](#12-서비스-파이프라인에-붙이기)
13. [새 모델 아키텍처 추가하기](#13-새-모델-아키텍처-추가하기)
14. [빌드 · Feature 조합 · 테스트](#14-빌드--feature-조합--테스트)
15. [디렉토리 구조](#15-디렉토리-구조)
16. [라이선스와 모델 책임](#16-라이선스와-모델-책임)

---

## 1. 핵심 특징

로컬 대규모 언어모델(LLM) 실행에서 과소평가되는 영역이 **모델 수명주기와 서빙 계약**이다.
추론 커널이 아무리 좋아도 "어떤 파일이 실행 가능한 모델인지, 지금 메모리에 무엇이 올라 있는지,
같은 입력이 같은 출력을 내는지"가 관리되지 않으면 로컬 AI 는 재현 불가능한 장난감이 된다.
이 엔진은 추론 커널을 새로 만드는 대신, 그 위아래의 시스템 엔지니어링을 책임지는 **런타임 틀**을 지향한다.

| 원칙 | 의미 |
|---|---|
| **커널은 만들지 않고 조립한다** | 텐서 연산·모델 구현은 Hugging Face 의 Rust 프레임워크 Candle 을 쓴다. 엔진의 차별점은 모델 수명주기(등록·검증·로드·캐시·언로드)와 서빙 계약이다. |
| **매니페스트가 곧 계약** | 모델 파일은 SHA-256 해시·아키텍처·토크나이저·채팅 템플릿이 기록된 TOML 매니페스트로만 실행된다. "실행 가능한 모델"과 "그냥 큰 파일"을 구분한다. |
| **결정적 생성** | 같은 모델·프롬프트·시드·샘플링 설정이면 같은 출력. 고정 시드 샘플러와 결정적 생성 루프로 회귀 시험이 가능하다. |
| **한 번 로드, 계속 재사용** | 프로세스 전역 세션 캐시가 해시 검증·모델 로드를 최초 1회만 수행한다. 요청마다 수 GB 를 다시 읽지 않는다. |
| **순수 Rust / zero 외부 런타임** | C++ llama.cpp 래퍼가 아니다. Python·Node.js·외부 프로세스 없이 단일 바이너리로 동작하고, Python 은 PyO3 확장 모듈로 붙는다. |

### Ollama 와 무엇이 같고 무엇이 다른가

사용자 경험의 목표는 같다 — 모델을 받아서, 등록하고, 로컬에서 대화한다. 구현 철학이 다르다.

- Ollama 는 llama.cpp(C++)를 감싼 Go 서버다. 이 엔진은 **전 계층이 Rust** 라서 하나의
  Cargo 워크스페이스에서 타입 안전하게 조립되고, 라이브러리·CLI·Python 확장이 같은 코어를 공유한다.
- 모델 관리가 암묵적 캐시가 아니라 **명시적 매니페스트**다. 가중치·토크나이저의 해시가 기록되고,
  로드 전 무결성이 검증되며, 파일이 바뀌면 캐시가 자동 무효화된다.
- 임베드가 1급 시나리오다. 서버를 따로 켜지 않아도 Rust crate 또는 Python 모듈로
  **호스트 서비스 프로세스 안에서** 직접 추론할 수 있다.

---

## 2. 빠른 시작

세 표면(CLI 서버, Python, Rust) 모두 같은 흐름이다 — 모델을 받아 등록하고, 토크나이저를 붙이고, 생성한다.

### CLI — 모델 받기부터 OpenAI 호환 서버까지

```bash
cargo build --release

# 1) Hugging Face 에서 가중치 + tokenizer.json 을 받아 실행 가능한 번들로 등록
./target/release/rust-ai-serving-engine model pull \
  --repo unsloth/Qwen3-4B-Instruct-2507-GGUF \
  --file Qwen3-4B-Instruct-2507-Q4_K_M.gguf \
  --id qwen3-4b \
  --architecture qwen3 \
  --tokenizer-repo Qwen/Qwen3-4B-Instruct-2507 \
  --tokenizer-file tokenizer.json

# 2) OpenAI 호환 서버 기동
./target/release/rust-ai-serving-engine serve --port 8080
```

```bash
# 3) 어떤 OpenAI 클라이언트로든 대화 (스트리밍은 "stream": true)
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-4b", "messages": [{"role": "user", "content": "안녕?"}]}'
```

### Python

```python
import rust_ai_serving_engine as engine

# 등록 (최초 1회) — 가중치 다운로드 + 로컬 tokenizer.json 연결
engine.pull_model("./models", "unsloth/Qwen3-4B-Instruct-2507-GGUF",
                  "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", "qwen3-4b", architecture="qwen3")
engine.attach_tokenizer("./models", "qwen3-4b", "./tokenizer.json")

# 대화 — 채팅 템플릿·종료 토큰 자동 적용, 모델은 프로세스 캐시에 상주
answer = engine.generate_chat_registered_gguf(
    "./models", "qwen3-4b",
    [{"role": "user", "content": "한 문장으로 자기소개 해줘."}],
    max_tokens=64,
)
print(answer)
```

### Rust 라이브러리

```rust
use rust_ai_serving_engine_core::{DevicePreference, GenerationConfig, ModelRegistry, generate};
use rust_ai_serving_engine_models::{ChatMessage, SessionCache};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let registry = ModelRegistry::open("./models")?;
    let cache = SessionCache::new();

    // 최초 호출: 해시 검증 + 로드 / 이후: 메모리 상주 세션 재사용
    let session = cache.get_or_load(&registry, "qwen3-4b", DevicePreference::Auto)?;
    let mut session = session.lock().unwrap();

    let template = session.chat_template.expect("chat template resolved from manifest");
    let prompt = template.render(&[ChatMessage {
        role: "user".into(),
        content: "What is the capital of France?".into(),
    }])?;
    let prompt_tokens = session.tokenizer.encode(&prompt, false)?;

    let mut config = GenerationConfig::default();
    if let Some(eos) = session.eos_token {
        config.stop_tokens.push(eos); // 종료 토큰에서 생성 자동 종료
    }
    let result = generate(session.decoder.as_mut(), &prompt_tokens, &config, || false)?;
    println!("{}", session.tokenizer.decode(&result.tokens, true)?);
    Ok(())
}
```

---

## 3. 설치와 Cargo Feature

Rust 프로젝트의 `Cargo.toml`:

```toml
[dependencies]
rust-ai-serving-engine-core = { git = "https://github.com/arabangoo/rust_ai_serving_engine" }
rust-ai-serving-engine-models = { git = "https://github.com/arabangoo/rust_ai_serving_engine" }
```

### 워크스페이스 크레이트 구성

| 크레이트 | 역할 | 주요 의존 |
|---|---|---|
| `rust_ai_serving_engine_core` | 매니페스트·레지스트리·생성 루프·샘플러·장치 선택·에러 계약 | `candle-core`, `hf-hub`, `sha2` |
| `rust_ai_serving_engine_models` | GGUF 디코더(Llama·Qwen3)·토크나이저·채팅 템플릿·세션 캐시 | `candle-transformers`, `tokenizers` |
| `rust_ai_serving_engine_api` | OpenAI 호환 HTTP API 와 SSE 스트리밍 | `axum`, `tokio` |
| `rust_ai_serving_engine_cli` | `model`·`runtime`·`serve` 명령줄 (바이너리명 `rust-ai-serving-engine`) | `clap` |
| `rust_ai_serving_engine_python` | PyO3 확장 모듈 (모듈명 `rust_ai_serving_engine`) | `pyo3`(abi3) |

### Feature 목록

| Feature | 크레이트 | 활성화 대상 | 비고 |
|---|---|---|---|
| **`cpu`** | core, models | CPU 실행 (기본 활성) | 순수 Rust, 외부 런타임 없음 |
| `cuda` | core, models | NVIDIA GPU 실행 경로 | `candle-core/cuda` 전달 |
| `metal` | core, models | Apple Silicon GPU 실행 경로 | `candle-core/metal` 전달 |
| **`python`** | python | PyO3 cdylib 바인딩 | maturin 이 자동 활성화 |

> 기본(CPU) 빌드는 외부 공유 라이브러리나 subprocess 를 요구하지 않는다. 모델 파일과 바이너리 하나면
> 오프라인·에어갭 환경에서도 동작한다 (Hugging Face 다운로드는 `model pull` 사용 시에만 필요).

---

## 4. 아키텍처

```text
요청(HTTP/CLI/Python)
  → 레지스트리: 매니페스트 조회 (아키텍처·토크나이저·템플릿·해시)
  → 세션 캐시: 최초 1회만 해시 검증 + 디코더·토크나이저 로드
  → 채팅 템플릿 렌더 → 토크나이즈
  → prefill: 프롬프트 전체 평가 + KV 캐시 생성
  → decode: 토큰 샘플링 → KV 캐시 갱신 반복 (종료 토큰·stop 문자열·취소 확인)
  → 토큰 콜백 → SSE 델타 전송 또는 완성 텍스트 조립
```

핵심은 **계약의 분리**다. `core` 는 추론 백엔드를 포함하지 않고 매니페스트·생성 루프·트레이트만 정의한다.
`models` 는 그 계약의 Candle 구현체다. HTTP·CLI·Python 은 같은 `models` 위의 세 가지 표면일 뿐이다.

- **등록** — [`ModelRegistry`](#61-modelregistry--core) 가 가중치를 해시하고 TOML 매니페스트를 원자적으로 기록한다.
- **로드** — [`SessionCache`](#63-modelsession--sessioncache--models) 가 매니페스트 해시가 같으면 메모리 상주 세션을 재사용하고, 다르면 검증 후 재로드한다.
- **생성** — [`generate` / `generate_with`](#62-생성-계약--core) 가 아키텍처 중립 디코드 루프를 돌린다. KV(Key-Value) 캐시는 디코더가 소유한다.
- **직렬화** — 같은 모델에 대한 동시 요청은 세션 뮤텍스로 직렬화된다 (KV 캐시는 공유 불가). 서로 다른 모델은 동시 생성된다.

---

## 5. 모델 매니페스트와 레지스트리

모델 저장소(store)는 폴더 하나다. `manifests/` 아래 모델당 TOML 파일 하나가 기록된다.

```toml
id = "qwen3-4b"
kind = "generator"                 # generator | embedding
format = "gguf"                    # gguf | safetensors
weights = "<가중치 파일 절대 경로>"
sha256 = "<가중치 SHA-256>"
tokenizer = "<tokenizer.json 절대 경로>"
tokenizer_sha256 = "<토크나이저 SHA-256>"
architecture = "qwen3"
context_length = 262144
chat_template = "chatml"           # chatml | llama3 | mistral (생략 시 아키텍처 기본값)
```

매니페스트는 실행 가능한 모델과 단순 파일을 구분하는 계약이다:

- **무결성** — `verify` 는 가중치·토크나이저의 SHA-256 을 재계산해 매니페스트와 대조한다.
  세션 캐시도 로드 시점에 같은 검증을 수행하고, 해시가 달라지면 캐시를 버리고 재로드한다.
- **실행 가능 조건** — 생성에는 `format = "gguf"` + `architecture` + `tokenizer` 세 가지가 필요하다.
  하나라도 없으면 로드 단계에서 무엇이 빠졌는지 명시하고 거절한다.
- **가중치 파일 위치** — `model pull` 은 Hugging Face 캐시에 내려받은 파일을 매니페스트로 가리킨다.
  캐시를 정리하면 모델을 다시 받아야 한다. 장기 보관할 모델은 원하는 폴더로 옮긴 뒤 `model import` 로 등록한다.

---

## 6. 공개 API 레퍼런스

### 6.1 `ModelRegistry` — core

```rust
ModelRegistry::open(root) -> Result<Self>          // store 폴더 열기(없으면 생성)

fn import_local(&self, id, weights, kind: ModelKind,
                architecture: Option<String>, context_length: Option<u32>,
                chat_template: Option<String>) -> Result<ImportedModel>
fn attach_tokenizer(&self, id, tokenizer_path) -> Result<ModelManifest>
fn get(&self, id) -> Result<ModelManifest>         // 매니페스트 조회 (해시 재계산 없음)
fn list(&self) -> Result<Vec<ModelManifest>>       // id 정렬 목록
fn verify(&self, id) -> Result<ModelManifest>      // 가중치·토크나이저 해시 재검증
```

`HuggingFaceHub::download(repo, file) -> Result<PathBuf>` 가 Hugging Face Hub 공개 파일을
관리 캐시에 내려받는다 (core, `hf-hub` 기반).

### 6.2 생성 계약 — core

```rust
/// 로드된 모델이 구현하는 아키텍처 중립 디코더 계약.
pub trait TokenDecoder: Send {
    fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>>;  // KV 캐시 초기화 + 첫 logits
    fn decode(&mut self, token: u32) -> Result<Vec<f32>>;       // 토큰 1개 평가 → 다음 logits
    fn eos_token(&self) -> Option<u32> { None }                 // 모델 파일이 선언한 종료 토큰
}

pub struct GenerationConfig {
    pub max_tokens: usize,        // 기본 256
    pub temperature: f32,         // 기본 0.7 (0.0 = 탐욕적 선택)
    pub top_k: Option<usize>,     // 기본 Some(40)
    pub seed: u64,                // 기본 0 — 같은 시드는 같은 출력
    pub stop_tokens: Vec<u32>,    // 이 토큰이 나오면 즉시 종료
}

// 완성형 생성 — 취소 콜백이 true 를 반환하면 중단
generate(decoder, prompt, &config, cancelled) -> Result<GenerationResult>

// 스트리밍용 — 토큰마다 on_token 이 불리고, false 반환 시 디코딩 중단
generate_with(decoder, prompt, &config, cancelled, on_token) -> Result<GenerationResult>

pub struct GenerationResult { pub tokens: Vec<u32>, pub stop_reason: GenerationStopReason }
pub enum GenerationStopReason { MaxTokens, StopToken, Cancelled }
```

### 6.3 `ModelSession` · `SessionCache` — models

```rust
/// 로드된 모델 1개 — 디코더 + 토크나이저 + 종료 토큰 + 채팅 템플릿.
pub struct ModelSession {
    pub decoder: Box<dyn TokenDecoder>,
    pub tokenizer: LocalTokenizer,
    pub eos_token: Option<u32>,
    pub chat_template: Option<ChatTemplate>,
}
ModelSession::load(&manifest, &runtime) -> Result<Self>

/// 프로세스 전역 세션 캐시. 키 = 모델 id + 장치.
SessionCache::new() -> Self
fn get_or_load(&self, registry, id, device: DevicePreference)
    -> Result<Arc<Mutex<ModelSession>>>   // 해시 불변이면 재사용, 변경이면 검증 후 재로드
fn clear(&self)                            // 전체 언로드 (메모리 해제)
```

### 6.4 디코더·토크나이저 — models

```rust
// 등록된 아키텍처 이름으로 GGUF 디코더 선택
load_gguf_decoder(architecture, weights, &runtime) -> Result<Box<dyn TokenDecoder>>

LlamaGgufDecoder::load(path, &runtime) -> Result<Self>   // Llama·Mistral 호환 GGUF
Qwen3GgufDecoder::load(path, &runtime) -> Result<Self>   // Qwen3 호환 GGUF

LocalTokenizer::from_file(path) -> Result<Self>           // Hugging Face tokenizer.json
fn encode(&self, text, add_special_tokens: bool) -> Result<Vec<u32>>
fn decode(&self, tokens, skip_special_tokens: bool) -> Result<String>
```

### 6.5 장치 선택 — core

```rust
pub enum DevicePreference { Auto, Cpu, Cuda, Metal }   // Auto = CUDA → Metal → CPU 폴백

RuntimeDevice::select(preference) -> Result<RuntimeDevice>
fn smoke_test(&self) -> Result<()>     // 실제 텐서 연산으로 백엔드 동작 확인
fn is_accelerated(&self) -> bool
```

### 6.6 에러 타입 — core

```rust
pub enum EngineError {
    InvalidModelId(String), UnsupportedFormat(String), UnsupportedArchitecture(String),
    ModelNotFound(String), ModelFileNotFound(String),
    IntegrityMismatch { id, expected, actual },
    BackendUnavailable(String), Candle(String), Tokenizer(String), HuggingFaceHub(String),
    InvalidGenerationConfig(String), InvalidLogits,
    Io(std::io::Error), TomlSerialize(..), TomlDeserialize(..),
}
```

> 로드 실패는 원인별로 구분된다 — 손상 파일·해시 불일치·미지원 아키텍처·백엔드 부재가 각각 다른
> 에러로 보고되므로, 호출자는 "왜 안 되는지"를 사용자에게 그대로 전달할 수 있다.

---

## 7. 지원 모델

실행 형식은 GGUF 양자화 모델이고, 아키텍처 이름이 디코더를 고른다.

| 아키텍처 (`--architecture`) | 디코더 | 대표 모델 |
|---|---|---|
| `qwen3` | `Qwen3GgufDecoder` (Candle quantized_qwen3) | Qwen3-4B-Instruct-2507 — 실모델로 채팅 완성·SSE 스트리밍·한국어 다중 바이트·세션 캐시 재사용까지 종단 간 검증 (16코어 CPU 노트북 기준 약 초당 5토큰) |
| `llama` `llama2` `llama3` `mistral` `mixtral` | `LlamaGgufDecoder` (Candle quantized_llama) | Llama 2·3, Mistral, Mixtral instruct 계열 |

동작 메모:

- **지원 밖 아키텍처는 잘못된 출력 대신 명확한 에러로 거절한다.** `qwen2` 는 Qwen3 GGUF 사용을 안내하는
  에러를, `phi` 는 지원 제외 사유를 담은 에러를 반환한다. Safetensors 는 레지스트리 등록·해시 검증 대상이고
  실행은 GGUF 로 한다.
- **Qwen3 는 non-thinking instruct 변형을 쓴다.** Qwen3 기본판은 답변 전 `<think>` 추론 블록을 길게
  생성하는 하이브리드 모델이라 CPU 에서 체감이 크게 나빠진다. Qwen3-4B-Instruct-2507 처럼
  thinking 이 제거된 instruct 변형은 ChatML 템플릿 그대로 동작한다 (검증 완료).
- **생성 중 같은 모델은 직렬화된다.** KV 캐시가 요청 간 공유될 수 없어 세션 뮤텍스로 순차 처리한다.
  서로 다른 모델은 동시 생성된다. 다중 사용자 대규모 배치는 이 엔진의 비목표다 (vLLM 의 영역).
- **토크나이저는 외부 `tokenizer.json` 을 쓴다.** 양자화 GGUF 저장소에 tokenizer.json 이 없으면
  원본 모델 저장소에서 받아 붙인다 (`model pull --tokenizer-repo` 가 이를 한 번에 처리).

---

## 8. 채팅 템플릿과 생성 제어

instruct 모델은 학습 때 쓰인 대화 마크업을 그대로 재현해야 정상 동작한다. 엔진은 대화 메시지 목록을
모델별 마크업으로 렌더하고, 종료 토큰에서 생성을 자동으로 끝낸다.

### 템플릿 선택 규칙

1. 매니페스트의 `chat_template` 값(`chatml` | `llama3` | `mistral`)이 있으면 그것을 쓴다.
2. 없으면 아키텍처 기본값 — `qwen3` → ChatML, `llama3` → Llama3, `llama`·`llama2`·`mistral`·`mixtral` → Mistral `[INST]`.
3. 어느 쪽도 없으면 채팅 요청을 거절한다 (완성 API 는 템플릿 없이 동작).

| 템플릿 | 마크업 | 대상 |
|---|---|---|
| `chatml` | `<|im_start|>role ... <|im_end|>` | Qwen 계열, 다수 ChatML 파인튜닝 |
| `llama3` | `<|start_header_id|>role<|end_header_id|> ... <|eot_id|>` | Llama 3 instruct |
| `mistral` | `<s>[INST] ... [/INST]` (system 은 다음 user 턴에 병합) | Mistral·Llama 2 instruct |

템플릿이 특수 토큰을 직접 표기하므로, 채팅 프롬프트는 토크나이저의 자동 특수 토큰 없이 인코딩된다.

### 종료 토큰(EOS) 자동 종료

GGUF 메타데이터의 `tokenizer.ggml.eos_token_id` 를 로드 시 읽어 두고, HTTP·Python 채팅 표면이
자동으로 stop 토큰에 추가한다. 사용자가 토큰 id 를 알 필요가 없다.

### stop 문자열

OpenAI 호환 `stop`(문자열 하나 또는 배열)을 지원한다. 생성 텍스트에 stop 문자열이 나타나면
그 직전까지만 반환하고 종료한다. 스트리밍에서는 stop 문자열 길이만큼 텍스트를 홀드백(hold-back)해서
**청크 경계에 걸친 stop 문자열도 클라이언트로 새어 나가지 않는다.**

### 샘플링

- `temperature = 0.0` — 결정적 탐욕 선택 (회귀 시험용)
- `temperature > 0` + `top_k` — 고정 시드(`seed`) 기반 확률 샘플링. 같은 시드는 같은 출력
- 다중 바이트 문자(한글 등)가 토큰 경계에 걸치면 완성될 때까지 방출을 보류한다 — 깨진 문자가 스트림에 나가지 않는다

---

## 9. HTTP API (OpenAI 호환)

`serve` 명령으로 기동한다. 기본 바인딩은 `127.0.0.1:8080` (로컬 전용 — 외부 노출은 사용자 책임).

| 경로 | 메서드 | 역할 |
|---|---|---|
| `/health` | GET | 프로세스 생존 확인 |
| `/v1/models` | GET | 등록 모델 목록 (OpenAI list 형식) |
| `/v1/models/{id}` | GET | 모델 존재 확인 |
| `/v1/completions` | POST | 프롬프트 완성 (비스트리밍) |
| `/v1/chat/completions` | POST | 채팅 완성 — `stream: true` 면 SSE 토큰 스트리밍 |

### 채팅 완성

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-4b",
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "stop": ["\n\n"]
  }'
```

응답은 OpenAI `chat.completion` 형식이다 — `choices[0].message.content`, `finish_reason`(`stop` | `length`), `usage` 토큰 집계.

### SSE 스트리밍

`"stream": true` 를 주면 `text/event-stream` 으로 OpenAI `chat.completion.chunk` 를 흘려보낸다.
첫 청크는 role, 이후 청크는 `delta.content`, 마지막 청크는 `finish_reason`, 종료 표시는 `data: [DONE]`.
클라이언트가 연결을 끊으면 서버는 다음 토큰 경계에서 디코딩을 중단한다 (낭비 계산 없음).

```bash
curl -sN http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-4b", "messages": [{"role": "user", "content": "Count to 5."}], "stream": true}'
```

### 요청 파라미터

| 파라미터 | 타입 | 기본값 | 비고 |
|---|---|---|---|
| `model` | string | 필수 | 등록된 모델 id |
| `messages` / `prompt` | array / string | 필수 | 채팅 / 완성 |
| `max_tokens` | int | 256 | |
| `temperature` | float | 0.7 | 0.0 = 결정적 |
| `top_k` | int | 40 | OpenAI 표준 외 확장 |
| `seed` | int | 0 | 고정 시드 재현 |
| `stop` | string 또는 array | 없음 | stop 문자열. 종료 토큰은 별도로 항상 적용 |
| `stream` | bool | false | 채팅 완성만 지원 |

---

## 10. CLI 도구

바이너리명은 `rust-ai-serving-engine`, 모델 저장소는 `--store <폴더>` (기본 `.rust_ai_serving_engine`).

| 명령 | 인자 | 동작 |
|---|---|---|
| `model import` | `<path>` `--id` `[--kind]` `[--architecture]` `[--context-length]` `[--chat-template]` | 로컬 GGUF·Safetensors 를 해시와 함께 등록 |
| `model pull` | `--repo --file --id` `[--architecture]` `[--chat-template]` `[--tokenizer-repo --tokenizer-file]` | Hugging Face 에서 받아 등록. 토크나이저 옵션을 주면 다운로드 후 자동 연결 |
| `model attach-tokenizer` | `<id>` `--tokenizer <path>` | 등록 모델에 로컬 tokenizer.json 연결 |
| `model list` | | 등록 모델 목록 |
| `model inspect` | `<id>` | 매니페스트 TOML 출력 |
| `model verify` | `<id>` | 가중치·토크나이저 해시 재검증 |
| `runtime probe` | `[--device auto\|cpu\|cuda\|metal]` | 장치 선택 + 실제 텐서 연산 스모크 테스트 |
| `serve` | `[--host]` `[--port]` `[--device]` | OpenAI 호환 API 서버 기동 |

```bash
# 로컬 파일 등록 후 무결성 검증
rust-ai-serving-engine model import ./my-model.gguf --id my-model --architecture llama3
rust-ai-serving-engine model verify my-model

# 장치 확인
rust-ai-serving-engine runtime probe --device auto
```

---

## 11. Python 바인딩 (PyO3)

**abi3(stable ABI)** 로 빌드되어 Python 3.9 이상 단일 휠로 호환된다. 모듈명은 `rust_ai_serving_engine`.

### 설치

```bash
# PyPI 게시 후 — Rust 툴체인 불필요
pip install rust_ai_serving_engine

# 소스에서(최신 main / 게시 전) — 설치 머신에 Rust 툴체인 필요
pip install "git+https://github.com/arabangoo/rust_ai_serving_engine"
```

### API

```python
import rust_ai_serving_engine as engine

engine.__version__                                  # "0.1.0"
engine.probe_runtime(device="auto")                 # 장치 선택 + 텐서 스모크 테스트

# 모델 수명주기 (store = 모델 저장소 폴더)
engine.pull_model(store, repo, file, id, kind="generator",
                  architecture=None, context_length=None, chat_template=None)
engine.import_model(store, path, id, ...)           # 로컬 파일 등록 (인자 동일)
engine.attach_tokenizer(store, id, tokenizer_path)
engine.list_models(store)                           # ["qwen3-4b", ...]
engine.inspect_model(store, id)                     # 매니페스트 TOML 문자열
engine.verify_model(store, id)                      # 해시 재검증
engine.unload_models()                              # 프로세스 캐시 전체 해제

# 생성 — 등록 모델 (프로세스 캐시 상주, 종료 토큰 자동)
engine.generate_registered_gguf(store, id, prompt, max_tokens=256,
                                temperature=0.7, top_k=40, seed=0,
                                stop_tokens=[], device="auto")

# 채팅 생성 — 템플릿 자동 적용
engine.generate_chat_registered_gguf(store, id,
    [{"role": "user", "content": "..."}], max_tokens=256, ...)

# 채팅 스트리밍 — 텍스트 조각마다 콜백 호출. 콜백이 False 를 반환하면 중단되고
# 그때까지의 부분 텍스트가 반환된다 (None 등 다른 반환값은 계속 진행).
engine.generate_chat_stream_registered_gguf(store, id, messages, on_delta,
                                            max_tokens=256, ...)

# 생성 — 파일 직접 지정 (레지스트리 없이 1회성)
engine.generate_llama_gguf(weights_path, tokenizer_path, prompt, ...)
```

### 스트리밍 통합 레시피

콜백이 원시 API 다. 서버 전송 이벤트(SSE)나 제너레이터가 필요하면 스레드 + 큐로 감싼다 —
생성 루프는 GIL 을 해제한 채 돌고 콜백 호출 순간에만 GIL 을 잡으므로, 호스트 서비스와 자연스럽게 병행된다.

```python
import queue
import threading

def stream_chat(messages):
    """토큰 조각을 순서대로 내놓는 제너레이터 (FastAPI StreamingResponse 등에 직결)."""
    q: queue.Queue = queue.Queue()
    done = object()

    def worker():
        try:
            engine.generate_chat_stream_registered_gguf(
                "./models", "qwen3-4b", messages,
                lambda delta: q.put(delta) or True,
            )
        finally:
            q.put(done)

    threading.Thread(target=worker, daemon=True).start()
    while (item := q.get()) is not done:
        yield item
```

### 호스트 서비스를 멈추지 않는다 — GIL 해제

다운로드·해시 검증·모델 로드·토큰 생성 같은 장시간 작업은 모두 **GIL(Global Interpreter Lock)을
해제한 채** Rust 에서 수행된다. FastAPI·Flask 같은 호스트 서비스에 임베드해도 생성 중에
다른 요청 스레드가 멈추지 않는다 (생성 중 파이썬 하트비트 스레드가 정상 동작함을 실측 확인).

### 캐시 동작

등록 모델 생성 함수의 최초 호출이 해시 검증 + 로드를 수행하고, 이후 호출은 메모리 상주 모델을
재사용한다. 매니페스트의 해시가 바뀌면(모델 파일 교체) 자동으로 재검증·재로드된다.
메모리를 돌려받으려면 `unload_models()` 를 부른다.

---

## 12. 서비스 파이프라인에 붙이기

이 엔진은 단독 앱이 아니라 **로컬 추론이 필요한 자리에 박아 넣는 코어 의존성**이다.
호스트 환경에 따라 아래 표면 중 하나를 고른다.

| 호스트 | 표면 | 통합 방법 |
|---|---|---|
| 기존 OpenAI 클라이언트 코드 | HTTP 서버 | `base_url` 만 로컬로 변경 |
| Python 서비스 (FastAPI 등) | Python 모듈 | 서버 없이 in-process 추론 |
| Rust 서비스 | crate | 레지스트리 + 세션 캐시 직접 사용 |
| 타 언어 / 배치 / 오케스트레이션 | CLI + HTTP | `serve` 를 사이드카로 |

### 12.1 OpenAI SDK — base_url 교체 한 줄

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="unused")
out = client.chat.completions.create(
    model="qwen3-4b",
    messages=[{"role": "user", "content": "요약해줘: ..."}],
    stream=True,
)
for chunk in out:
    print(chunk.choices[0].delta.content or "", end="")
```

LangChain 도 같은 방식이다 — `ChatOpenAI(base_url="http://127.0.0.1:8080/v1", model="qwen3-4b")`.

### 12.2 Python 서비스에 in-process 임베드

서버 프로세스를 따로 두지 않고 서비스 안에서 직접 추론한다. GIL 이 해제되므로
이벤트 루프는 `run_in_executor`(또는 FastAPI 의 스레드풀)로 감싸면 된다.

```python
import asyncio
import rust_ai_serving_engine as engine

STORE = "./models"

async def answer(messages: list[dict]) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: engine.generate_chat_registered_gguf(STORE, "qwen3-4b", messages, max_tokens=256),
    )
```

### 12.3 Rust 서비스에 임베드

생성은 동기·CPU 바운드이므로 async 서버(axum 등)에서는 `spawn_blocking` 으로 감싼다.
`SessionCache` 를 `Arc` 로 공유하면 모델은 프로세스에서 한 번만 로드된다.

```rust
use std::sync::Arc;
use rust_ai_serving_engine_core::{DevicePreference, GenerationConfig, ModelRegistry, generate};
use rust_ai_serving_engine_models::SessionCache;

// 기동 시 1회
let cache = Arc::new(SessionCache::new());

// 핸들러
let cache = cache.clone();
let text = tokio::task::spawn_blocking(move || -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let registry = ModelRegistry::open("./models")?;
    let session = cache.get_or_load(&registry, "qwen3-4b", DevicePreference::Auto)?;
    let mut session = session.lock().unwrap();
    let tokens = session.tokenizer.encode("The capital of France is", true)?;
    let mut config = GenerationConfig::default();
    if let Some(eos) = session.eos_token { config.stop_tokens.push(eos); }
    let out = generate(session.decoder.as_mut(), &tokens, &config, || false)?;
    Ok(session.tokenizer.decode(&out.tokens, true)?)
}).await??;
```

기성 HTTP 표면이 필요하면 `rust_ai_serving_engine_api::{router, serve, ApiState}` 를 그대로
자기 서버에 합성할 수도 있다.

### 12.4 타 언어 / 배치 — 사이드카

Java·Node·Go 등에서는 `serve` 를 사이드카 프로세스로 띄우고 OpenAI 클라이언트로 붙는 것이
가장 단순하다. 단일 바이너리라 컨테이너에 실행 파일 하나 + 모델 폴더만 넣으면 된다.

---

## 13. 새 모델 아키텍처 추가하기

새 GGUF 아키텍처는 세 단계로 붙는다. 코어 생성 루프·API·CLI 는 건드리지 않는다.

1. **디코더 구현** — `TokenDecoder` 를 구현한다. Candle 의 quantized 모델 구현을 감싸는 것이 기본형이다.

```rust
use rust_ai_serving_engine_core::{Result, TokenDecoder};

pub struct MyArchDecoder { /* ModelWeights + device + position */ }

impl TokenDecoder for MyArchDecoder {
    fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>> {
        // KV 캐시 초기화 → 프롬프트 전체 forward → 마지막 위치 logits (rank-1 Vec<f32>)
    }
    fn decode(&mut self, token: u32) -> Result<Vec<f32>> {
        // 토큰 1개 forward (KV 캐시 누적) → 다음 logits
    }
    fn eos_token(&self) -> Option<u32> { /* GGUF 메타데이터에서 읽은 값 */ }
}
```

2. **아키텍처 매핑 등록** — `load_gguf_decoder` 의 match 에 아키텍처 이름을 추가한다.
   지원하지 않는 조합은 잘못된 출력 대신 `UnsupportedArchitecture` 로 거절하는 것이 규약이다.

3. **채팅 템플릿 연결** — 기존 3종으로 충분하면 `ChatTemplate::for_architecture` 에 기본값만 추가하고,
   새 마크업이 필요하면 variant 와 렌더 함수를 추가한다 (렌더는 순수 함수라 단위 테스트로 고정한다).

주의할 계약 두 가지 — `prefill` 은 반드시 KV 캐시를 초기화해야 하고(이전 대화 오염 방지),
logits 는 **rank-1 벡터**로 반환해야 한다 (Candle forward 가 `(batch, vocab)` rank-2 를 주면 squeeze 필요 —
실제로 이 처리 누락이 치명 버그였고 실모델 종단 간 시험으로 잡았다).

---

## 14. 빌드 · Feature 조합 · 테스트

이 저장소를 clone 한 경우, **Rust 툴체인(stable)** 으로 한 번 빌드해야 한다.

| 쓰는 방식 | 빌드 명령 | 결과물 |
|---|---|---|
| CLI + 서버 | `cargo build --release` | `target/release/rust-ai-serving-engine` 단일 바이너리 |
| Python 모듈 | `pip install maturin && maturin develop --release` | 현재 venv 에 `import rust_ai_serving_engine` |
| Rust 라이브러리 | `Cargo.toml` 에 `git`/`path` 의존성 | 다른 Rust 프로젝트에 링크 |

```bash
# 전체 워크스페이스 빌드·테스트
cargo build --release
cargo test --workspace
cargo clippy --all-targets

# Python 확장 게이트 컴파일 확인
cargo check -p rust-ai-serving-engine-python --features python

# 배포 휠 빌드
maturin build --release          # dist/ 에 abi3 휠
```

테스트는 모델 파일 없이 동작하는 범위를 결정적으로 검증한다 — 생성 루프(정지·취소), 레지스트리
(등록·해시 변조 감지·토크나이저 연결), 채팅 템플릿 렌더 3종, stop 문자열 파싱·홀드백 경계.

### 실모델 스모크 (수동)

코드 변경 후 실모델 회귀는 [2장](#2-빠른-시작)의 CLI 흐름 그대로 — 소형 GGUF 를 pull 해
`serve` 를 띄우고 채팅 완성(비스트리밍·스트리밍)을 호출한다. `temperature: 0.0` + 고정 `seed` 로
같은 입력의 출력 안정성까지 확인한다.

---

## 15. 디렉토리 구조

```text
rust_ai_serving_engine/
  Cargo.toml                              # 워크스페이스 정의
  pyproject.toml                          # maturin 빌드 메타데이터 (PyPI 패키지)
  README.md                               # 이 문서
  crates/
    rust_ai_serving_engine_core/
      src/
        lib.rs                            # 크레이트 루트 · re-export
        manifest.rs                       # ModelManifest / ModelKind / ModelFormat
        registry.rs                       # ModelRegistry (등록·해시·검증, 원자적 기록)
        generation.rs                     # TokenDecoder / GenerationConfig / generate(_with) / 샘플러
        runtime.rs                        # DevicePreference / RuntimeDevice (CPU·CUDA·Metal)
        hub.rs                            # HuggingFaceHub 다운로드
        error.rs                          # EngineError
    rust_ai_serving_engine_models/
      src/
        lib.rs                            # load_gguf_decoder · GGUF EOS 추출
        llama_gguf.rs                     # Llama·Mistral GGUF 디코더
        qwen3_gguf.rs                     # Qwen3 GGUF 디코더
        chat.rs                           # ChatTemplate (ChatML·Llama3·Mistral) + 렌더 테스트
        session.rs                        # ModelSession / SessionCache
        tokenizer.rs                      # LocalTokenizer (tokenizer.json)
    rust_ai_serving_engine_api/
      src/lib.rs                          # OpenAI 호환 HTTP API + SSE 스트리밍
    rust_ai_serving_engine_cli/
      src/main.rs                         # model / runtime / serve 명령
    rust_ai_serving_engine_python/
      src/
        lib.rs                            # feature 게이트
        python.rs                         # PyO3 바인딩 (GIL 해제 + 프로세스 세션 캐시)
```

---

## 16. 라이선스와 모델 책임

엔진 코드는 Apache-2.0 이다.

모델 가중치·토크나이저·GGUF 변환물의 라이선스는 엔진과 별개다. 레지스트리에 등록하는 각 모델의
출처와 라이선스 조건(재배포 가능 여부 포함)은 사용자가 확인해야 하며, 상업 배포에서는 모델별 조건을
확인한 뒤에만 번들에 포함한다.
