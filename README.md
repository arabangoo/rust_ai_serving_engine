# rust_ai_serving_engine

Rust로 구현하는 로컬 AI 모델 엔진

`rust_ai_serving_engine`은 Ollama, llama.cpp, LM Studio가 제공하는 로컬 모델 실행 경험을 Rust 단일 바이너리로 구현하는 엔진이다. Llama·Mistral·Phi·Qwen 계열 생성 모델과 BERT·nomic 계열 임베딩 모델을 로컬에서 실행하고, CPU·CUDA·Metal 가속, GGUF 양자화 모델, OpenAI 호환 API, 모델 관리와 토큰 스트리밍을 제공한다.

문서 변환·OCR·검색 증강 생성(Retrieval-Augmented Generation, RAG)은 엔진의 선택 플러그인이다. 이 프로젝트의 본체는 모델을 다운로드 또는 가져오기하고, 검증하고, 메모리에 올리고, 추론하고, API로 서빙하는 **Rust 로컬 모델 런타임**이다.

## 1. 제품 정의

| 비교 대상 | 사용자가 기대하는 기능 | `rust_ai_serving_engine`의 구현 방향 |
| --- | --- | --- |
| Ollama | 모델 관리, 로컬 실행, REST API, 모델별 설정 | 모델 레지스트리·매니페스트·로컬 서버·명령줄 도구 |
| llama.cpp | GGUF, 양자화, CPU·GPU 로컬 추론 | Candle 기반 실행 계층과 순수 Rust 양자화 모델 경로 |
| LM Studio | 모델 선택, 다운로드, 서버 실행, 실시간 출력 | 서버·명령줄을 먼저 구현하고 사용자 인터페이스는 별도 클라이언트로 분리 |

이 프로젝트는 세 제품의 화면을 복제하는 것이 아니라, 그 핵심인 **로컬 모델 실행 플랫폼**을 Rust 생태계에 맞게 만든다.

## 2. 목표

- Python, Node.js, 외부 런타임 없이 실행되는 Rust 단일 바이너리
- Llama·Mistral·Phi·Qwen 생성 모델과 BERT·nomic 임베딩 모델의 공통 실행 계약
- CPU 기본 실행과 CUDA·Metal 선택 가속
- GGUF 양자화 모델과 Hugging Face Safetensors 모델을 다루는 모델 저장소
- OpenAI 호환 채팅·완성·임베딩 API와 서버 전송 이벤트(Server-Sent Events, SSE) 스트리밍
- 오프라인 모델 등록·실행과 선택적인 원격 모델 저장소 연동
- 모델 파일·토크나이저·템플릿·양자화 정보의 해시 검증과 재현 가능한 매니페스트
- USB·에어갭·개인 PC 배포에 적합한 상대 경로와 로컬 우선 데이터 경계

## 3. 비목표

- 대규모 언어 모델 학습이나 분산 학습 클러스터
- vLLM과 같은 다중 사용자 대규모 연속 배치 스케줄러
- C++ llama.cpp를 호출하는 얇은 래퍼
- 특정 상용 모델 또는 모델 허브에 종속되는 제품
- 처음부터 데스크톱 그래픽 사용자 인터페이스를 엔진에 내장하는 것

초기 버전은 한 명 또는 소수 사용자가 자신의 컴퓨터에서 모델을 안정적으로 실행하도록 만드는 데 집중한다. 데스크톱 UI, 편집기 확장, 모바일 앱은 같은 OpenAI 호환 API를 사용하는 별도 클라이언트로 추가한다.

## 4. 핵심 기술 선택: Candle

기본 추론 계층은 Hugging Face의 Rust 머신러닝 프레임워크인 Candle을 채택한다. Candle은 Rust 네이티브 텐서 연산과 모델 구현을 제공하며, CPU·CUDA·Metal 장치를 같은 추상화로 다룰 수 있다. 이는 순수 Rust·단일 바이너리·외부 런타임 없음이라는 프로젝트 철학과 맞는다.

| 구성요소 | 역할 |
| --- | --- |
| `candle-core` | Tensor, Device, CPU·CUDA·Metal 연산 기반 |
| `candle-nn` | 신경망 계층과 공용 모델 구성요소 |
| `candle-transformers` | Transformer 계열 모델·토크나이저 통합 경로 |
| `tokenizers` | Hugging Face 토크나이저 로드와 인코딩·디코딩 |
| `hf-hub` | 선택적 Hugging Face 모델 저장소 다운로드·캐시 |
| GGUF 모듈 | GGUF 헤더·메타데이터·양자화 텐서 로드와 검증 |
| `axum`·`tokio` | HTTP API, SSE, 요청 취소, 작업 큐 |

실행 커널을 직접 새로 만들지 않는다. 모델 아키텍처, 양자화 텐서, 토크나이저, 샘플링, KV 캐시를 Candle과 검증된 Rust 크레이트 위에서 조립하고, 프로젝트의 차별점은 모델 수명주기·서빙 계약·배포·플러그인 경계에 둔다.

## 5. 지원 모델 범위

### 5.1 생성 모델

초기 모델 어댑터는 다음 계열을 대상으로 한다.

| 계열 | 용도 | 우선 지원 형식 |
| --- | --- | --- |
| Llama | 범용 대화·지시 수행 | GGUF, Safetensors |
| Mistral | 범용 대화·코드 | GGUF, Safetensors |
| Phi | 소형 로컬 모델 | GGUF, Safetensors |
| Qwen | 다국어·코드·소형부터 대형까지 | GGUF, Safetensors |

모델마다 서로 다른 채팅 템플릿, 특수 토큰, 위치 인코딩, KV 헤드 구성이 있다. 따라서 모델 파일 확장자만 보고 실행하지 않는다. `ModelAdapter`가 아키텍처 메타데이터를 검증하고, 토크나이저·채팅 템플릿·생성 파라미터와 결합한 뒤에만 로드한다.

### 5.2 임베딩 모델

임베딩은 생성 모델의 부가기능이 아니라 독립적인 1급 모델 유형이다.

| 계열 | 용도 |
| --- | --- |
| BERT 계열 | 문장·문서 임베딩, 분류, 검색 |
| nomic 계열 | 장문 문서와 검색 증강 생성용 임베딩 |
| 다국어 임베딩 모델 | 한국어 포함 다국어 검색 |

임베딩 모델은 별도 배치 큐와 별도 메모리 예산으로 운영한다. 생성 모델의 KV 캐시와 경쟁시키지 않으며, 인덱스에는 모델 식별자·차원 수·정규화 방식·버전을 함께 기록한다.

## 6. 모델 형식과 저장소

### 6.1 GGUF 양자화 모델

GGUF는 로컬 배포에 적합한 단일 파일 모델 형식이며, 양자화 모델을 다루기에 편리하다. 엔진은 GGUF를 로드할 때 다음을 확인한다.

- 매직 넘버와 형식 버전
- 아키텍처와 지원하는 모델 어댑터의 일치 여부
- 양자화 텐서 형식과 현재 백엔드의 지원 여부
- 토크나이저·채팅 템플릿·컨텍스트 길이 메타데이터
- 모델 파일의 SHA-256 해시와 매니페스트 일치 여부

모든 GGUF 양자화 방식을 즉시 지원한다고 주장하지 않는다. 지원하는 텐서 형식과 모델 아키텍처의 조합을 명시적으로 등록하고, 미지원 조합은 잘못된 결과 대신 로드 단계에서 거절한다.

### 6.2 Safetensors 모델

Hugging Face 형식의 Safetensors 모델은 Candle의 모델 구현 경로와 직접 연결한다. 다중 파일 가중치, 설정 JSON, 토크나이저 파일, 생성 설정을 하나의 모델 매니페스트로 묶어 관리한다.

### 6.3 모델 매니페스트

```toml
id = "qwen-local-7b-q4"
kind = "generator"
format = "gguf"
architecture = "qwen2"
weights = "models/qwen-local-7b-q4.gguf"
sha256 = "<model-file-sha256>"
context_length = 8192
chat_template = "chatml"

[tokenizer]
source = "embedded"

[runtime]
default_temperature = 0.7
default_top_p = 0.9
default_max_tokens = 1024
```

매니페스트는 실행 가능한 모델과 단순 파일을 구분하는 계약이다. 모델을 추가·업데이트·공유할 때 가중치, 토크나이저, 라이선스, 해시, 호환 엔진 버전이 함께 이동해야 한다.

## 7. 하드웨어와 백엔드 선택

엔진은 시작할 때 장치와 메모리를 탐지하고, 사용 가능한 최선의 실행 경로를 선택한다.

| 우선순위 | 백엔드 | 대상 |
| --- | --- | --- |
| 1 | CUDA | NVIDIA GPU가 있는 Linux·Windows 호스트 |
| 2 | Metal | Apple Silicon과 macOS GPU |
| 3 | CPU SIMD | AVX2·AVX-512·NEON을 사용할 수 있는 호스트 |
| 4 | CPU 기본 | 가속 명령어가 제한된 호스트 |

선택 결과에는 장치 이름, 사용 가능한 메모리, 모델 예상 메모리, 선택된 양자화 형식, 컨텍스트 길이를 포함한다. 사용자는 자동 선택을 그대로 쓰거나 명시적으로 장치를 고정할 수 있다.

### 7.1 메모리 예산

생성 모델의 총 메모리 요구량은 다음 세 부분으로 계산한다.

```text
총 요구 메모리 = 모델 가중치
               + KV 캐시
               + 활성화·작업 버퍼
               + 런타임 안전 여유
```

KV 캐시는 컨텍스트 길이와 동시 생성 수에 비례한다. 모델 등록 단계에서 모델의 레이어 수, KV 헤드 수, 헤드 차원, 캐시 정밀도를 기록하고, 요청을 받기 전에 가능한 컨텍스트 길이와 동시 요청 수를 판정한다.

### 7.2 로딩 정책

- 충분한 RAM 또는 GPU 메모리가 있으면 모델을 순차 프리로드한다.
- 메모리가 부족하면 메모리 매핑을 사용하거나 더 작은 양자화 모델을 제안한다.
- 모델마다 하나의 로드 상태와 제한된 요청 큐를 유지한다.
- 사용하지 않는 모델은 유휴 시간 또는 명시 명령에 따라 언로드한다.
- 로드 실패는 손상 파일, 해시 불일치, 메모리 부족, 미지원 아키텍처로 구분해 보고한다.

## 8. 요청 수명주기

```text
HTTP 또는 CLI 요청 수신
  → 모델과 어댑터 선택
  → 채팅 템플릿 적용
  → 토크나이즈
  → prefill: 프롬프트 전체 처리와 KV 캐시 생성
  → decode: 토큰 하나 생성, 샘플링, KV 캐시 갱신 반복
  → SSE 또는 CLI로 토큰 전송
  → 종료 사유·사용량·성능 지표 기록
```

첫 토큰까지 시간(Time To First Token, TTFT)과 초당 토큰 수는 각각 prefill과 decode 성능을 보여 주는 독립 지표다. 엔진은 둘을 별도로 계측한다.

클라이언트의 연결 종료나 사용자의 중단 요청은 `CancellationToken`으로 전파한다. decode 루프는 매 토큰 경계에서 취소를 확인하며, 종료 후 KV 캐시 예약과 요청 슬롯을 반드시 반환한다.

## 9. 핵심 Rust 계약

```rust
use async_trait::async_trait;
use futures_core::Stream;
use std::pin::Pin;

pub type TokenStream = Pin<Box<dyn Stream<Item = Result<GenerateEvent, EngineError>> + Send>>;

#[async_trait]
pub trait ModelBackend: Send + Sync {
    async fn probe(&self) -> BackendCapability;
    async fn load(&self, manifest: ModelManifest) -> Result<ModelHandle, EngineError>;
    async fn generate(&self, request: GenerateRequest) -> Result<TokenStream, EngineError>;
    async fn unload(&self, model: ModelId) -> Result<(), EngineError>;
}

#[async_trait]
pub trait GeneratorModel: Send + Sync {
    async fn prefill(&mut self, tokens: &[u32]) -> Result<(), EngineError>;
    async fn decode_next(&mut self) -> Result<Logits, EngineError>;
}

#[async_trait]
pub trait EmbeddingModel: Send + Sync {
    async fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>, EngineError>;
    fn dimension(&self) -> usize;
    fn model_id(&self) -> &str;
}
```

`ModelBackend`는 장치와 배포 형식의 경계를 담당한다. `GeneratorModel`은 모델별 Transformer 구현을 감싼다. `EmbeddingModel`은 생성 모델과 다른 수명주기·배치 정책을 가진다. 이 분리 덕분에 Qwen 생성 어댑터와 nomic 임베딩 어댑터는 같은 서버에서 독립적으로 진화할 수 있다.

## 10. API와 명령줄

### 10.1 OpenAI 호환 API

| 경로 | 역할 |
| --- | --- |
| `GET /v1/models` | 로드됨·등록됨·다운로드 중인 모델 목록 |
| `POST /v1/chat/completions` | 채팅 템플릿 기반 생성과 SSE 스트리밍 |
| `POST /v1/completions` | 프롬프트 기반 완성 |
| `POST /v1/embeddings` | BERT·nomic 등 임베딩 생성 |
| `POST /v1/models/pull` | 원격 저장소에서 모델 가져오기 |
| `POST /v1/models/import` | 로컬 GGUF·Safetensors 등록 |
| `POST /v1/models/{id}/load` | 모델 로드 |
| `POST /v1/models/{id}/unload` | 모델 언로드 |
| `GET /healthz` | 프로세스 생존 확인 |
| `GET /readyz` | 백엔드·모델·저장소 준비 상태 확인 |

`/v1/chat/completions`와 `/v1/embeddings`는 기존 OpenAI 클라이언트가 쉽게 연결할 수 있는 호환 표면이다. 모델 관리 API는 엔진 고유 기능으로 명확히 분리한다.

### 10.2 명령줄 경험

```bash
# 모델 등록과 무결성 확인
rust-ai-serving-engine model import ./qwen.gguf --id qwen-local-7b-q4
rust-ai-serving-engine model verify qwen-local-7b-q4

# 모델 실행과 대화
rust-ai-serving-engine run qwen-local-7b-q4

# 로컬 API 서버
rust-ai-serving-engine serve --model qwen-local-7b-q4 --port 11434

# 임베딩 모델 등록
rust-ai-serving-engine model pull nomic-embed-text-v1.5
```

명령줄은 Ollama처럼 짧고 예측 가능해야 하지만, 모든 명령은 매니페스트·해시·장치 선택·라이선스 정보를 확인한 뒤 동작한다.

## 11. Cargo 워크스페이스

```text
rust_ai_serving_engine/
  Cargo.toml
  crates/
    rust_ai_serving_engine_core/        공용 타입, 오류, 모델 매니페스트, 트레이트
    rust_ai_serving_engine_candle/      Candle 장치·텐서·모델 실행 공통 계층
    rust_ai_serving_engine_gguf/        GGUF 파싱, 양자화 텐서, 무결성 검증
    rust_ai_serving_engine_models/      Llama·Mistral·Phi·Qwen·BERT·nomic 어댑터
    rust_ai_serving_engine_registry/    모델 저장소, 다운로드, 캐시, 매니페스트
    rust_ai_serving_engine_scheduler/   모델 큐, 취소, 메모리 예산, 언로드 정책
    rust_ai_serving_engine_api/         Axum HTTP API와 SSE 스트리밍
    rust_ai_serving_engine_cli/         run, serve, model, inspect 명령
    rust_ai_serving_engine_plugins/     RAG·문서 변환·OCR 선택 플러그인
  tests/
    fixtures/                           작은 테스트 모델·토크나이저·매니페스트
    contract/                           API·스트림·모델 어댑터 호환성 시험
```

`core`에는 Candle, HTTP, 파일시스템, 모델 허브 구현을 넣지 않는다. 모델과 백엔드가 공유하는 최소 계약만 둔다. 이 경계가 CPU 전용 빌드, CUDA 빌드, Metal 빌드, 플러그인 없는 USB 배포를 단순하게 만든다.

## 12. 선택 플러그인: 문서·OCR·RAG

`rust_markdown_transformer`와 `rust_ocr_transformer`는 엔진의 모델 실행 코어를 대체하지 않는다. 이들은 모델 엔진 위에 지식 베이스 기능을 추가하는 Rust 플러그인이다.

| 플러그인 | 역할 | 엔진과의 연결점 |
| --- | --- | --- |
| `rust_markdown_transformer` | 문서를 공통 IR·Markdown·의미 단위 청크로 변환 | 임베딩 모델에 전달할 텍스트와 인용 메타데이터 생성 |
| `rust_ocr_transformer` | 스캔·사진·화면 캡처를 구조화 OCR 결과로 변환 | 임베딩 전 텍스트화, 이미지 경계 상자 인용 보존 |
| 검색 플러그인 | 키워드·벡터·그래프 검색 | 생성 요청에 근거 컨텍스트 선택 삽입 |

플러그인은 기본 바이너리에 강제하지 않는다. `--features rag` 또는 별도 서비스 프로세스로 선택할 수 있어야 하며, 모델 실행만 필요한 사용자는 문서 파서·OCR 모델·벡터 인덱스 없이 작은 런타임을 사용할 수 있다.

## 13. 기능 게이트

```toml
[features]
default = ["cpu", "api", "cli"]
cpu = ["rust_ai_serving_engine_candle/cpu"]
cuda = ["rust_ai_serving_engine_candle/cuda"]
metal = ["rust_ai_serving_engine_candle/metal"]
gguf = ["dep:rust_ai_serving_engine_gguf"]
hf-hub = ["dep:hf-hub"]
api = ["dep:axum", "dep:tokio"]
cli = ["dep:clap"]
rag = ["dep:rust_markdown_transformer", "dep:rust_ocr_transformer"]
```

기능 게이트는 미구현 백엔드를 미리 노출하는 장식이 아니다. 실제로 컴파일·테스트되는 조합만 공개한다. CUDA와 Metal은 같은 바이너리에 억지로 묶지 않고, 대상 OS와 하드웨어에 맞는 배포 산출물로 제공한다.

## 14. 구현 순서와 완료 기준

| 단계 | 범위 | 완료 기준 |
| --- | --- | --- |
| 1단계: 런타임 골격 | 매니페스트, 모델 레지스트리, 장치 프로빙, 모의 백엔드 | 모델 없이 `import`·`inspect`·`serve` 계약 테스트 통과 |
| 2단계: Candle 생성 | 하나의 소형 Llama 또는 Qwen 계열 Safetensors 모델 | 고정 시드·CPU에서 prefill·decode·취소·SSE 동작 |
| 3단계: GGUF | GGUF 검증과 지원 양자화 형식 로드 | 손상·미지원·해시 불일치 모델이 명확히 거절됨 |
| 4단계: 임베딩 | BERT 또는 nomic 계열 모델 | `/v1/embeddings`의 배치·차원·정규화 계약 통과 |
| 5단계: 가속 | CUDA 또는 Metal 중 한 경로 | CPU 대비 성능 계측과 백엔드별 회귀 시험 |
| 6단계: 모델 수명주기 | 가져오기·캐시·로드·언로드·메모리 큐 | 모델 교체 중 요청 격리와 복구 시험 통과 |
| 7단계: 선택 플러그인 | 문서 변환·OCR·검색 증강 생성 | 모델 코어를 변경하지 않고 RAG 질의가 동작 |

## 15. 검증과 계측

| 영역 | 검증 항목 |
| --- | --- |
| 모델 등록 | 매니페스트 유효성, 해시, 라이선스 정보, 아키텍처 판정 |
| 생성 | 토크나이저 일치, 채팅 템플릿, 중단 문자열, 시드, 취소 |
| 임베딩 | 차원 수, 정규화, 배치 순서, 모델 버전 격리 |
| GGUF | 헤더, 텐서 형식, 양자화 지원, 손상 파일 거절 |
| 백엔드 | CPU·CUDA·Metal별 로드·생성·오류·성능 회귀 |
| 서버 | OpenAI 호환 요청, SSE 종료, 큐 한도, 시간 초과 |
| 성능 | 첫 토큰까지 시간, 초당 토큰, 모델 로드 시간, 메모리 사용량 |

동일한 모델 파일, 백엔드, 토크나이저, 프롬프트, 시드, 샘플링 설정에서는 회귀 비교가 가능해야 한다. 서로 다른 장치의 부동소수점 계산은 출력이 다를 수 있으므로, 골든 테스트는 백엔드별로 분리한다.

## 16. 배포

배포 단위는 실행 바이너리 하나와 모델 저장소 폴더다. 모델 파일은 실행 파일에 억지로 포함하지 않으며, 매니페스트를 통해 등록·검증·선택한다.

- CPU 단일 바이너리: 가장 넓은 호스트 호환성과 USB·에어갭 배포
- CUDA 빌드: NVIDIA GPU가 있는 호스트용
- Metal 빌드: Apple Silicon과 macOS용
- 선택 플러그인 빌드: 문서 변환·OCR·RAG를 함께 제공하는 확장 배포

Python 사용자를 위한 PyPI 배포가 필요하면 `rust_ai_serving_engine`이라는 PyO3 패키지는 엔진의 Rust API와 로컬 서버 제어 API를 노출한다. 모델 실행의 핵심은 여전히 Rust 바이너리와 Rust 라이브러리에 남긴다.

## 17. 라이선스와 모델 책임

엔진 코드의 라이선스와 모델 가중치·토크나이저·GGUF 변환물·OCR 모델·임베딩 모델의 라이선스는 별개다. 모델 레지스트리는 각 항목의 출처, 라이선스, 재배포 가능 여부, 해시, 생성 시점, 호환 엔진 버전을 기록해야 한다. 상업 배포에서는 모델별 조건을 확인한 뒤에만 번들에 포함한다.
