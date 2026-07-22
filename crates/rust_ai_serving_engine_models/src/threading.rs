//! Decode thread-count default for heterogeneous (hybrid) CPUs.
//!
//! Candle's quantized matvec and fused decode attention run on a barrier
//! pool sized by CANDLE_NUM_THREADS (default: all physical cores) with a
//! static equal work split. On hybrid CPUs (performance + efficiency +
//! low-power-efficiency cores) every barrier then waits for the slowest
//! core, which measurably halves decode throughput: on a 16-core
//! Core Ultra 7 255H, Qwen3-4B decode ran at 10 tok/s with 16 threads and
//! 20 tok/s with 12, and Qwen3-1.7B at 13 vs 46 tok/s. The cliff appears
//! exactly when the low-power cores join the pool.
//!
//! This module sets a bandwidth-friendly default before the pool is first
//! created: with 12 or more physical cores, leave 4 cores out. Memory-bound
//! decode saturates DRAM bandwidth below the full core count, so on
//! homogeneous many-core CPUs the cap costs little, while on hybrid CPUs it
//! removes the straggler penalty. An explicit CANDLE_NUM_THREADS always
//! wins; the default is only applied when the variable is unset.

use std::sync::Once;

/// Applies the decode thread-count default once per process. Must run
/// before the first generation (the barrier pool is created lazily on the
/// first quantized matmul / decode attention call and never resized).
pub fn apply_decode_thread_default() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        if std::env::var_os("CANDLE_NUM_THREADS").is_some() {
            return; // 사용자 명시 설정 존중
        }
        let physical = num_cpus::get_physical();
        if physical >= 12 {
            let capped = physical - 4;
            // SAFETY: 모델 로드 시점(첫 생성 전)에 1회 호출된다. 표준 라이브러리의
            // env 접근은 내부 잠금으로 직렬화되며, 이 시점에는 엔진이 만든 워커
            // 스레드가 아직 없다 (barrier pool 은 첫 추론에서 lazy 생성).
            unsafe { std::env::set_var("CANDLE_NUM_THREADS", capped.to_string()) };
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn respects_existing_env_and_is_idempotent() {
        // ONCE 가 이미 소모됐을 수 있으므로 두 번 불러도 패닉이 없어야 한다는
        // 것과, 명시 설정이 있으면 그대로 남는다는 것만 확인한다.
        unsafe { std::env::set_var("CANDLE_NUM_THREADS", "5") };
        apply_decode_thread_default();
        apply_decode_thread_default();
        assert_eq!(std::env::var("CANDLE_NUM_THREADS").unwrap(), "5");
    }
}
