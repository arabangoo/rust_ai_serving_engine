//! Env-gated phase timers for the forward pass (RASE_PROFILE=1).
//!
//! Purpose: decompose prefill wall time into "quantized-linear GEMM work"
//! (dequantize + f32 matmul) versus "attention kernel work", and decode wall
//! time into "quantized matvec" versus "fused attention". These numbers set
//! the upper bound of a GPU prefill-GEMM offload (Amdahl) before any kernel
//! is written.
//!
//! Design constraints:
//! - Zero behavioral change: timing never alters computation or outputs.
//! - Near-zero cost when disabled: a single cached bool check per probe;
//!   no `Instant::now()` is taken unless RASE_PROFILE=1.
//! - Counters are global atomics (nanoseconds / counts) so instrumentation
//!   needs no plumbing through candle-facing call signatures.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

/// True when the RASE_PROFILE environment variable is exactly "1".
/// Read once per process; changing the variable afterwards has no effect.
pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("RASE_PROFILE").map(|v| v == "1").unwrap_or(false))
}

/// Accumulated phase counters (ns unless suffixed otherwise).
pub struct Phases {
    /// Whole-model forward calls with sequence length > 1 (prefill chunks).
    pub prefill_calls: AtomicU64,
    /// Prompt tokens processed by prefill forwards.
    pub prefill_tokens: AtomicU64,
    /// Wall time of prefill forwards (embedding to logits).
    pub prefill_forward_ns: AtomicU64,
    /// Dequantize step of the hybrid GEMM path (per linear, per prefill call).
    pub gemm_dequant_ns: AtomicU64,
    /// f32 GEMM step of the hybrid GEMM path.
    pub gemm_matmul_ns: AtomicU64,
    /// wgpu offloaded GEMM wall time (upload + dispatch + readback).
    pub gemm_gpu_ns: AtomicU64,
    /// Number of GEMM calls served by the GPU path.
    pub gemm_gpu_calls: AtomicU64,
    /// Blocked causal prefill attention kernel.
    pub attn_blocked_ns: AtomicU64,
    /// Candle flash prefill attention kernel (short prefill / continuation).
    pub attn_flash_ns: AtomicU64,
    /// Whole-model forward calls with sequence length == 1 (decode steps).
    pub decode_steps: AtomicU64,
    /// Wall time of decode forwards.
    pub decode_forward_ns: AtomicU64,
    /// Quantized single-row matvec time inside decode forwards.
    pub decode_matvec_ns: AtomicU64,
    /// Fused interleaved-cache decode attention time.
    pub decode_attn_ns: AtomicU64,
}

pub fn phases() -> &'static Phases {
    static PHASES: Phases = Phases {
        prefill_calls: AtomicU64::new(0),
        prefill_tokens: AtomicU64::new(0),
        prefill_forward_ns: AtomicU64::new(0),
        gemm_dequant_ns: AtomicU64::new(0),
        gemm_matmul_ns: AtomicU64::new(0),
        gemm_gpu_ns: AtomicU64::new(0),
        gemm_gpu_calls: AtomicU64::new(0),
        attn_blocked_ns: AtomicU64::new(0),
        attn_flash_ns: AtomicU64::new(0),
        decode_steps: AtomicU64::new(0),
        decode_forward_ns: AtomicU64::new(0),
        decode_matvec_ns: AtomicU64::new(0),
        decode_attn_ns: AtomicU64::new(0),
    };
    &PHASES
}

/// Starts a probe. `None` (and therefore no timing cost) when disabled.
#[inline]
pub fn probe() -> Option<Instant> {
    if enabled() { Some(Instant::now()) } else { None }
}

/// Accumulates an optional probe into a counter.
#[inline]
pub fn commit(counter: &AtomicU64, probe: Option<Instant>) {
    if let Some(start) = probe {
        counter.fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

#[inline]
pub fn count(counter: &AtomicU64, amount: u64) {
    if enabled() {
        counter.fetch_add(amount, Ordering::Relaxed);
    }
}

/// Returns all counters as a JSON object; optionally resets them to zero.
pub fn snapshot(reset: bool) -> String {
    let p = phases();
    let read = |c: &AtomicU64| -> u64 {
        if reset {
            c.swap(0, Ordering::Relaxed)
        } else {
            c.load(Ordering::Relaxed)
        }
    };
    format!(
        concat!(
            "{{\"enabled\":{},",
            "\"prefill_calls\":{},",
            "\"prefill_tokens\":{},",
            "\"prefill_forward_ns\":{},",
            "\"gemm_dequant_ns\":{},",
            "\"gemm_matmul_ns\":{},",
            "\"gemm_gpu_ns\":{},",
            "\"gemm_gpu_calls\":{},",
            "\"attn_blocked_ns\":{},",
            "\"attn_flash_ns\":{},",
            "\"decode_steps\":{},",
            "\"decode_forward_ns\":{},",
            "\"decode_matvec_ns\":{},",
            "\"decode_attn_ns\":{}}}"
        ),
        enabled(),
        read(&p.prefill_calls),
        read(&p.prefill_tokens),
        read(&p.prefill_forward_ns),
        read(&p.gemm_dequant_ns),
        read(&p.gemm_matmul_ns),
        read(&p.gemm_gpu_ns),
        read(&p.gemm_gpu_calls),
        read(&p.attn_blocked_ns),
        read(&p.attn_flash_ns),
        read(&p.decode_steps),
        read(&p.decode_forward_ns),
        read(&p.decode_matvec_ns),
        read(&p.decode_attn_ns),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_is_valid_json_shape_and_resets() {
        phases().prefill_tokens.store(1700, Ordering::Relaxed);
        let s = snapshot(true);
        assert!(s.starts_with('{') && s.ends_with('}'));
        assert!(s.contains("\"prefill_tokens\":1700"));
        let s2 = snapshot(false);
        assert!(s2.contains("\"prefill_tokens\":0"));
    }
}
