use crate::{EngineError, ModelFormat, Result};
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Common contract implemented by each loaded Llama, Qwen, or Mistral model.
pub trait TokenDecoder: Send {
    /// Evaluates the prompt and initializes the model-specific KV cache.
    fn prefill(&mut self, prompt: &[u32]) -> Result<Vec<f32>>;

    /// Evaluates one sampled token and returns logits for the next token.
    fn decode(&mut self, token: u32) -> Result<Vec<f32>>;

    /// End-of-sequence token declared by the model file, when it exposes one.
    fn eos_token(&self) -> Option<u32> {
        None
    }
}

/// Factory contract for an architecture-specific model adapter.
pub trait ModelAdapter: Send + Sync {
    fn architecture(&self) -> &'static str;
    fn supports_format(&self, format: ModelFormat) -> bool;
}

#[derive(Clone, Debug, PartialEq)]
pub struct GenerationConfig {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub seed: u64,
    pub stop_tokens: Vec<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_k: Some(40),
            seed: 0,
            stop_tokens: Vec::new(),
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_tokens == 0 {
            return Err(EngineError::InvalidGenerationConfig(
                "max_tokens must be greater than zero".to_owned(),
            ));
        }
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(EngineError::InvalidGenerationConfig(
                "temperature must be finite and non-negative".to_owned(),
            ));
        }
        if self.top_k == Some(0) {
            return Err(EngineError::InvalidGenerationConfig(
                "top_k must be greater than zero when supplied".to_owned(),
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationStopReason {
    MaxTokens,
    StopToken,
    Cancelled,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenerationResult {
    pub tokens: Vec<u32>,
    pub stop_reason: GenerationStopReason,
}

pub struct LogitsSampler {
    rng: StdRng,
}

impl LogitsSampler {
    pub fn seeded(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn sample(&mut self, logits: &[f32], config: &GenerationConfig) -> Result<u32> {
        if logits.is_empty() || logits.iter().any(|logit| !logit.is_finite()) {
            return Err(EngineError::InvalidLogits);
        }
        if config.temperature == 0.0 {
            return Ok(argmax(logits) as u32);
        }

        let mut candidates: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        candidates.sort_by(|left, right| right.1.total_cmp(&left.1));
        if let Some(top_k) = config.top_k {
            candidates.truncate(top_k.min(candidates.len()));
        }

        let peak = candidates[0].1 / config.temperature;
        let weights: Vec<f32> = candidates
            .iter()
            .map(|(_, logit)| ((logit / config.temperature) - peak).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        if !total.is_finite() || total <= 0.0 {
            return Err(EngineError::InvalidLogits);
        }
        let mut threshold = self.rng.random::<f32>() * total;
        for ((index, _), weight) in candidates.iter().zip(weights) {
            if threshold <= weight {
                return Ok(*index as u32);
            }
            threshold -= weight;
        }
        Ok(candidates.last().expect("non-empty candidates").0 as u32)
    }
}

/// Runs the architecture-neutral generation loop.
pub fn generate<D, C>(
    decoder: &mut D,
    prompt: &[u32],
    config: &GenerationConfig,
    cancelled: C,
) -> Result<GenerationResult>
where
    D: TokenDecoder + ?Sized,
    C: FnMut() -> bool,
{
    generate_with(decoder, prompt, config, cancelled, |_| true)
}

/// Generation loop variant that reports each sampled token as it is produced.
///
/// `on_token` receives every non-stop sampled token. Returning `false` ends
/// the loop with a `Cancelled` stop reason, which lets streaming callers stop
/// decoding as soon as their client disconnects.
pub fn generate_with<D, C, F>(
    decoder: &mut D,
    prompt: &[u32],
    config: &GenerationConfig,
    mut cancelled: C,
    mut on_token: F,
) -> Result<GenerationResult>
where
    D: TokenDecoder + ?Sized,
    C: FnMut() -> bool,
    F: FnMut(u32) -> bool,
{
    config.validate()?;
    let mut logits = decoder.prefill(prompt)?;
    let mut sampler = LogitsSampler::seeded(config.seed);
    let mut tokens = Vec::with_capacity(config.max_tokens);

    for _ in 0..config.max_tokens {
        if cancelled() {
            return Ok(GenerationResult {
                tokens,
                stop_reason: GenerationStopReason::Cancelled,
            });
        }
        let token = sampler.sample(&logits, config)?;
        tokens.push(token);
        if config.stop_tokens.contains(&token) {
            return Ok(GenerationResult {
                tokens,
                stop_reason: GenerationStopReason::StopToken,
            });
        }
        if !on_token(token) {
            return Ok(GenerationResult {
                tokens,
                stop_reason: GenerationStopReason::Cancelled,
            });
        }
        logits = decoder.decode(token)?;
    }
    Ok(GenerationResult {
        tokens,
        stop_reason: GenerationStopReason::MaxTokens,
    })
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(right.1))
        .expect("non-empty logits")
        .0
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockDecoder;

    impl TokenDecoder for MockDecoder {
        fn prefill(&mut self, _: &[u32]) -> Result<Vec<f32>> {
            Ok(vec![0.0, 3.0, 1.0])
        }

        fn decode(&mut self, _: u32) -> Result<Vec<f32>> {
            Ok(vec![2.0, 1.0, 0.0])
        }
    }

    #[test]
    fn greedy_generation_stops_on_a_stop_token() {
        let config = GenerationConfig {
            max_tokens: 4,
            temperature: 0.0,
            stop_tokens: vec![1],
            ..GenerationConfig::default()
        };
        let result = generate(&mut MockDecoder, &[42], &config, || false).unwrap();
        assert_eq!(result.tokens, vec![1]);
        assert_eq!(result.stop_reason, GenerationStopReason::StopToken);
    }

    #[test]
    fn generation_honors_cancellation() {
        let config = GenerationConfig {
            max_tokens: 4,
            temperature: 0.0,
            ..GenerationConfig::default()
        };
        let mut checks = 0;
        let result = generate(&mut MockDecoder, &[42], &config, || {
            checks += 1;
            checks > 1
        })
        .unwrap();
        assert_eq!(result.tokens, vec![1]);
        assert_eq!(result.stop_reason, GenerationStopReason::Cancelled);
    }
}
