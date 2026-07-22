//! wgpu prefill GEMM offload: quantized weights resident on the GPU,
//! dequantized inside the shader, multiplied against f32 activations.
//!
//! Scope and shape of the offload:
//! - Only the prefill GEMM path uses this (many rows). Decode stays on the
//!   CPU quantized matvec, which is memory-bandwidth-bound and gains nothing
//!   from an integrated GPU sharing the same DRAM.
//! - Only Q4_K weight matrices are offloaded in this iteration. Q4_K_M
//!   checkpoints keep 70-80% of linear-layer elements in Q4_K; the rest
//!   (Q6_K halves of attn_v / ffn_down) falls back to the CPU GEMM path.
//! - Copying dequantized f32 weights per call would move gigabytes, so the
//!   raw quantized blocks are uploaded once per matrix and stay resident;
//!   every call uploads only activations and reads back only outputs.
//!
//! Safety ladder (자동 감지 3중 안전장치):
//! 1. RASE_GPU=0 disables the GPU path entirely (kill switch).
//! 2. Adapter selection rejects software rasterizers (DeviceType::Cpu,
//!    e.g. WARP / llvmpipe) — those are slower than the CPU kernels.
//! 3. Any runtime failure (device lost, allocation, mapping) marks the
//!    context unhealthy; from then on every call reports unavailable and
//!    the caller keeps using the CPU path.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, OnceLock};

use candle_core::quantized::{GgmlDType, QTensor};
use rayon::prelude::*;

use crate::profiling;

/// Q4_K super-block: 256 weights in 144 bytes = 36 u32 words.
const Q4K_BLOCK_U32: usize = 36;
const Q4K_BLOCK_ELEMS: usize = 256;
/// Rows of the activation matrix processed per workgroup pass (shared tile).
const TILE_M: usize = 16;
/// Threads per workgroup, one output column each.
const WG_COLS: usize = 64;
/// Activation rows uploaded per dispatch chunk (bounds staging memory).
const CHUNK_M: usize = 1024;

const SHADER: &str = r#"
// a 는 [k][mp] 로 전치·패딩(mp = 16의 배수)돼 올라온다 — 타일 적재가
// 행 방향 vec4 합체 로드 1회가 되게 하기 위한 배치. mp4 = mp / 4.
struct Params { m: u32, mp4: u32, n: u32, kb: u32 }

@group(0) @binding(0) var<storage, read> wq: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

// 활성값 타일. [원소 e][행 4개 묶음 c] 전치 배치 = vec4 로드로 16행을 4개의
// vec4 누산기에 상수 인덱스로만 누적한다 (동적 인덱스 배열 = 레지스터 스필 방지).
var<workgroup> a_tile: array<vec4<f32>, 1024>; // 256 elems x 4 vec4

// ggml get_scale_min_k4: 8개 서브블록의 6비트 scale/min 언팩.
fn scale_min(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<f32> {
    // scales[12]바이트 = s0|s1|s2 (리틀엔디언 u32 3개)
    // scales 12바이트: s0 = [0..4), s1 = [4..8), s2 = [8..12)
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        let sj = (s0 >> (8u * j)) & 0xffu;         // scales[j]
        let sj4 = (s1 >> (8u * j)) & 0xffu;        // scales[j+4]
        sc = sj & 63u;
        mn = sj4 & 63u;
    } else {
        let j4 = j - 4u;
        let sj8 = (s2 >> (8u * j4)) & 0xffu;       // scales[j+4] (인덱스 8..11)
        let sjm4 = (s0 >> (8u * j4)) & 0xffu;      // scales[j-4]
        let sj = (s1 >> (8u * j4)) & 0xffu;        // scales[j]
        sc = (sj8 & 0xfu) | ((sjm4 >> 6u) << 4u);
        mn = (sj8 >> 4u) | ((sj >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(mn));
}

// 스레드당 컬럼 4개(stride 64) 처리: a_tile 공유 로드 1세트로 FMA 를 4배
// 뽑아 공유 메모리 대역폭 병목을 완화한다.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col_a = wg.x * 256u + lid.x;
    let col_b = col_a + 64u;
    let col_c = col_a + 128u;
    let col_d = col_a + 192u;
    let row0 = wg.y * 16u;
    let rows = min(16u, p.m - row0);

    var accA0 = vec4<f32>(0.0);
    var accA1 = vec4<f32>(0.0);
    var accA2 = vec4<f32>(0.0);
    var accA3 = vec4<f32>(0.0);
    var accB0 = vec4<f32>(0.0);
    var accB1 = vec4<f32>(0.0);
    var accB2 = vec4<f32>(0.0);
    var accB3 = vec4<f32>(0.0);
    var accC0 = vec4<f32>(0.0);
    var accC1 = vec4<f32>(0.0);
    var accC2 = vec4<f32>(0.0);
    var accC3 = vec4<f32>(0.0);
    var accD0 = vec4<f32>(0.0);
    var accD1 = vec4<f32>(0.0);
    var accD2 = vec4<f32>(0.0);
    var accD3 = vec4<f32>(0.0);

    for (var kb = 0u; kb < p.kb; kb++) {
        // 256 x 16 활성값 타일 협동 적재 — 전치 배치라 vec4 합체 로드 1회
        for (var t = lid.x; t < 1024u; t += 64u) {
            let e = t / 4u;
            let cq = t % 4u;
            a_tile[t] = a[(kb * 256u + e) * p.mp4 + wg.y * 4u + cq];
        }
        workgroupBarrier();

        let use_a = col_a < p.n;
        let use_b = col_b < p.n;
        let use_c = col_c < p.n;
        let use_d = col_d < p.n;
        let base_a = (col_a * p.kb + kb) * 36u;
        let base_b = (col_b * p.kb + kb) * 36u;
        let base_c = (col_c * p.kb + kb) * 36u;
        let base_d = (col_d * p.kb + kb) * 36u;

        var dA = vec2<f32>(0.0);
        var sA0 = 0u; var sA1 = 0u; var sA2 = 0u;
        if (use_a) {
            dA = unpack2x16float(wq[base_a]);
            sA0 = wq[base_a + 1u]; sA1 = wq[base_a + 2u]; sA2 = wq[base_a + 3u];
        }
        var dB = vec2<f32>(0.0);
        var sB0 = 0u; var sB1 = 0u; var sB2 = 0u;
        if (use_b) {
            dB = unpack2x16float(wq[base_b]);
            sB0 = wq[base_b + 1u]; sB1 = wq[base_b + 2u]; sB2 = wq[base_b + 3u];
        }
        var dC = vec2<f32>(0.0);
        var sC0 = 0u; var sC1 = 0u; var sC2 = 0u;
        if (use_c) {
            dC = unpack2x16float(wq[base_c]);
            sC0 = wq[base_c + 1u]; sC1 = wq[base_c + 2u]; sC2 = wq[base_c + 3u];
        }
        var dD = vec2<f32>(0.0);
        var sD0 = 0u; var sD1 = 0u; var sD2 = 0u;
        if (use_d) {
            dD = unpack2x16float(wq[base_d]);
            sD0 = wq[base_d + 1u]; sD1 = wq[base_d + 2u]; sD2 = wq[base_d + 3u];
        }

        for (var g = 0u; g < 4u; g++) {              // 64원소 그룹 4개
            let smlA = scale_min(2u * g, sA0, sA1, sA2);
            let smhA = scale_min(2u * g + 1u, sA0, sA1, sA2);
            let wloA_s = dA.x * smlA.x;
            let wloA_m = dA.y * smlA.y;
            let whiA_s = dA.x * smhA.x;
            let whiA_m = dA.y * smhA.y;
            let smlB = scale_min(2u * g, sB0, sB1, sB2);
            let smhB = scale_min(2u * g + 1u, sB0, sB1, sB2);
            let wloB_s = dB.x * smlB.x;
            let wloB_m = dB.y * smlB.y;
            let whiB_s = dB.x * smhB.x;
            let whiB_m = dB.y * smhB.y;
            let smlC = scale_min(2u * g, sC0, sC1, sC2);
            let smhC = scale_min(2u * g + 1u, sC0, sC1, sC2);
            let wloC_s = dC.x * smlC.x;
            let wloC_m = dC.y * smlC.y;
            let whiC_s = dC.x * smhC.x;
            let whiC_m = dC.y * smhC.y;
            let smlD = scale_min(2u * g, sD0, sD1, sD2);
            let smhD = scale_min(2u * g + 1u, sD0, sD1, sD2);
            let wloD_s = dD.x * smlD.x;
            let wloD_m = dD.y * smlD.y;
            let whiD_s = dD.x * smhD.x;
            let whiD_m = dD.y * smhD.y;
            for (var b = 0u; b < 8u; b++) {          // 그룹당 qs 32바이트 = u32 8개
                var wordA = 0u;
                if (use_a) { wordA = wq[base_a + 4u + g * 8u + b]; }
                var wordB = 0u;
                if (use_b) { wordB = wq[base_b + 4u + g * 8u + b]; }
                var wordC = 0u;
                if (use_c) { wordC = wq[base_c + 4u + g * 8u + b]; }
                var wordD = 0u;
                if (use_d) { wordD = wq[base_d + 4u + g * 8u + b]; }
                for (var by = 0u; by < 4u; by++) {
                    let bi = b * 4u + by;            // 그룹 내 위치 0..31
                    let elo = (g * 64u + bi) * 4u;
                    let ehi = (g * 64u + 32u + bi) * 4u;
                    let alo0 = a_tile[elo];
                    let alo1 = a_tile[elo + 1u];
                    let alo2 = a_tile[elo + 2u];
                    let alo3 = a_tile[elo + 3u];
                    let ahi0 = a_tile[ehi];
                    let ahi1 = a_tile[ehi + 1u];
                    let ahi2 = a_tile[ehi + 2u];
                    let ahi3 = a_tile[ehi + 3u];

                    let byteA = (wordA >> (8u * by)) & 0xffu;
                    let wloA = wloA_s * f32(byteA & 0xfu) - wloA_m;
                    let whiA = whiA_s * f32(byteA >> 4u) - whiA_m;
                    accA0 += alo0 * wloA + ahi0 * whiA;
                    accA1 += alo1 * wloA + ahi1 * whiA;
                    accA2 += alo2 * wloA + ahi2 * whiA;
                    accA3 += alo3 * wloA + ahi3 * whiA;

                    let byteB = (wordB >> (8u * by)) & 0xffu;
                    let wloB = wloB_s * f32(byteB & 0xfu) - wloB_m;
                    let whiB = whiB_s * f32(byteB >> 4u) - whiB_m;
                    accB0 += alo0 * wloB + ahi0 * whiB;
                    accB1 += alo1 * wloB + ahi1 * whiB;
                    accB2 += alo2 * wloB + ahi2 * whiB;
                    accB3 += alo3 * wloB + ahi3 * whiB;

                    let byteC = (wordC >> (8u * by)) & 0xffu;
                    let wloC = wloC_s * f32(byteC & 0xfu) - wloC_m;
                    let whiC = whiC_s * f32(byteC >> 4u) - whiC_m;
                    accC0 += alo0 * wloC + ahi0 * whiC;
                    accC1 += alo1 * wloC + ahi1 * whiC;
                    accC2 += alo2 * wloC + ahi2 * whiC;
                    accC3 += alo3 * wloC + ahi3 * whiC;

                    let byteD = (wordD >> (8u * by)) & 0xffu;
                    let wloD = wloD_s * f32(byteD & 0xfu) - wloD_m;
                    let whiD = whiD_s * f32(byteD >> 4u) - whiD_m;
                    accD0 += alo0 * wloD + ahi0 * whiD;
                    accD1 += alo1 * wloD + ahi1 * whiD;
                    accD2 += alo2 * wloD + ahi2 * whiD;
                    accD3 += alo3 * wloD + ahi3 * whiD;
                }
            }
        }
        workgroupBarrier();
    }

    if (col_a < p.n) {
        let accs = array<vec4<f32>, 4>(accA0, accA1, accA2, accA3);
        for (var r = 0u; r < rows; r++) {
            c[(row0 + r) * p.n + col_a] = accs[r / 4u][r % 4u];
        }
    }
    if (col_b < p.n) {
        let accs = array<vec4<f32>, 4>(accB0, accB1, accB2, accB3);
        for (var r = 0u; r < rows; r++) {
            c[(row0 + r) * p.n + col_b] = accs[r / 4u][r % 4u];
        }
    }
    if (col_c < p.n) {
        let accs = array<vec4<f32>, 4>(accC0, accC1, accC2, accC3);
        for (var r = 0u; r < rows; r++) {
            c[(row0 + r) * p.n + col_c] = accs[r / 4u][r % 4u];
        }
    }
    if (col_d < p.n) {
        let accs = array<vec4<f32>, 4>(accD0, accD1, accD2, accD3);
        for (var r = 0u; r < rows; r++) {
            c[(row0 + r) * p.n + col_d] = accs[r / 4u][r % 4u];
        }
    }
}
"#;

/// f16 variant of the GEMM shader (adapters with SHADER_F16). Activations
/// arrive as f16 (half the upload), the inner FMA runs in packed f16, and
/// partial sums flush to f32 every 64-element group to bound rounding error.
const SHADER_F16: &str = r#"
enable f16;

struct Params { m: u32, mp4: u32, n: u32, kb: u32 }

@group(0) @binding(0) var<storage, read> wq: array<u32>;
@group(0) @binding(1) var<storage, read> a: array<vec4<f16>>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> a_tile: array<vec4<f16>, 1024>; // 256 elems x 4 vec4

fn scale_min(j: u32, s0: u32, s1: u32, s2: u32) -> vec2<f32> {
    var sc: u32;
    var mn: u32;
    if (j < 4u) {
        let sj = (s0 >> (8u * j)) & 0xffu;
        let sj4 = (s1 >> (8u * j)) & 0xffu;
        sc = sj & 63u;
        mn = sj4 & 63u;
    } else {
        let j4 = j - 4u;
        let sj8 = (s2 >> (8u * j4)) & 0xffu;
        let sjm4 = (s0 >> (8u * j4)) & 0xffu;
        let sj = (s1 >> (8u * j4)) & 0xffu;
        sc = (sj8 & 0xfu) | ((sjm4 >> 6u) << 4u);
        mn = (sj8 >> 4u) | ((sj >> 6u) << 4u);
    }
    return vec2<f32>(f32(sc), f32(mn));
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col_a = wg.x * 128u + lid.x;
    let col_b = col_a + 64u;
    let row0 = wg.y * 16u;
    let rows = min(16u, p.m - row0);

    var accA0 = vec4<f32>(0.0);
    var accA1 = vec4<f32>(0.0);
    var accA2 = vec4<f32>(0.0);
    var accA3 = vec4<f32>(0.0);
    var accB0 = vec4<f32>(0.0);
    var accB1 = vec4<f32>(0.0);
    var accB2 = vec4<f32>(0.0);
    var accB3 = vec4<f32>(0.0);

    for (var kb = 0u; kb < p.kb; kb++) {
        for (var t = lid.x; t < 1024u; t += 64u) {
            let e = t / 4u;
            let cq = t % 4u;
            a_tile[t] = a[(kb * 256u + e) * p.mp4 + wg.y * 4u + cq];
        }
        workgroupBarrier();

        let use_a = col_a < p.n;
        let use_b = col_b < p.n;
        let base_a = (col_a * p.kb + kb) * 36u;
        let base_b = (col_b * p.kb + kb) * 36u;

        var dA = vec2<f32>(0.0);
        var sA0 = 0u; var sA1 = 0u; var sA2 = 0u;
        if (use_a) {
            dA = unpack2x16float(wq[base_a]);
            sA0 = wq[base_a + 1u]; sA1 = wq[base_a + 2u]; sA2 = wq[base_a + 3u];
        }
        var dB = vec2<f32>(0.0);
        var sB0 = 0u; var sB1 = 0u; var sB2 = 0u;
        if (use_b) {
            dB = unpack2x16float(wq[base_b]);
            sB0 = wq[base_b + 1u]; sB1 = wq[base_b + 2u]; sB2 = wq[base_b + 3u];
        }

        for (var g = 0u; g < 4u; g++) {
            let smlA = scale_min(2u * g, sA0, sA1, sA2);
            let smhA = scale_min(2u * g + 1u, sA0, sA1, sA2);
            let wloA_s = dA.x * smlA.x;
            let wloA_m = dA.y * smlA.y;
            let whiA_s = dA.x * smhA.x;
            let whiA_m = dA.y * smhA.y;
            let smlB = scale_min(2u * g, sB0, sB1, sB2);
            let smhB = scale_min(2u * g + 1u, sB0, sB1, sB2);
            let wloB_s = dB.x * smlB.x;
            let wloB_m = dB.y * smlB.y;
            let whiB_s = dB.x * smhB.x;
            let whiB_m = dB.y * smhB.y;

            // f16 부분합 (64항) — g 그룹마다 f32 로 플러시해 오차를 묶는다
            var hA0 = vec4<f16>(0.0);
            var hA1 = vec4<f16>(0.0);
            var hA2 = vec4<f16>(0.0);
            var hA3 = vec4<f16>(0.0);
            var hB0 = vec4<f16>(0.0);
            var hB1 = vec4<f16>(0.0);
            var hB2 = vec4<f16>(0.0);
            var hB3 = vec4<f16>(0.0);

            for (var b = 0u; b < 8u; b++) {
                var wordA = 0u;
                if (use_a) { wordA = wq[base_a + 4u + g * 8u + b]; }
                var wordB = 0u;
                if (use_b) { wordB = wq[base_b + 4u + g * 8u + b]; }
                for (var by = 0u; by < 4u; by++) {
                    let bi = b * 4u + by;
                    let elo = (g * 64u + bi) * 4u;
                    let ehi = (g * 64u + 32u + bi) * 4u;
                    let alo0 = a_tile[elo];
                    let alo1 = a_tile[elo + 1u];
                    let alo2 = a_tile[elo + 2u];
                    let alo3 = a_tile[elo + 3u];
                    let ahi0 = a_tile[ehi];
                    let ahi1 = a_tile[ehi + 1u];
                    let ahi2 = a_tile[ehi + 2u];
                    let ahi3 = a_tile[ehi + 3u];

                    let byteA = (wordA >> (8u * by)) & 0xffu;
                    let wloA = f16(wloA_s * f32(byteA & 0xfu) - wloA_m);
                    let whiA = f16(whiA_s * f32(byteA >> 4u) - whiA_m);
                    hA0 += alo0 * wloA + ahi0 * whiA;
                    hA1 += alo1 * wloA + ahi1 * whiA;
                    hA2 += alo2 * wloA + ahi2 * whiA;
                    hA3 += alo3 * wloA + ahi3 * whiA;

                    let byteB = (wordB >> (8u * by)) & 0xffu;
                    let wloB = f16(wloB_s * f32(byteB & 0xfu) - wloB_m);
                    let whiB = f16(whiB_s * f32(byteB >> 4u) - whiB_m);
                    hB0 += alo0 * wloB + ahi0 * whiB;
                    hB1 += alo1 * wloB + ahi1 * whiB;
                    hB2 += alo2 * wloB + ahi2 * whiB;
                    hB3 += alo3 * wloB + ahi3 * whiB;
                }
            }

            accA0 += vec4<f32>(hA0);
            accA1 += vec4<f32>(hA1);
            accA2 += vec4<f32>(hA2);
            accA3 += vec4<f32>(hA3);
            accB0 += vec4<f32>(hB0);
            accB1 += vec4<f32>(hB1);
            accB2 += vec4<f32>(hB2);
            accB3 += vec4<f32>(hB3);
        }
        workgroupBarrier();
    }

    if (col_a < p.n) {
        let accs = array<vec4<f32>, 4>(accA0, accA1, accA2, accA3);
        for (var r = 0u; r < rows; r++) {
            c[(row0 + r) * p.n + col_a] = accs[r / 4u][r % 4u];
        }
    }
    if (col_b < p.n) {
        let accs = array<vec4<f32>, 4>(accB0, accB1, accB2, accB3);
        for (var r = 0u; r < rows; r++) {
            c[(row0 + r) * p.n + col_b] = accs[r / 4u][r % 4u];
        }
    }
}
"#;

/// Uploaded, resident quantized weight matrix.
#[derive(Debug)]
pub struct GpuWeight {
    buffer: wgpu::Buffer,
    /// Output features (rows of the weight matrix).
    n: usize,
    /// Input features; k % 256 == 0 guaranteed at upload.
    k: usize,
}

/// Growable staging buffers reused across dispatches (allocation per call
/// costs milliseconds on some drivers; the pool amortizes it to zero).
#[derive(Default)]
struct BufPool {
    a: Option<(wgpu::Buffer, u64)>,
    c: Option<(wgpu::Buffer, u64)>,
    read: Option<(wgpu::Buffer, u64)>,
    params: Option<wgpu::Buffer>,
}

pub struct GpuGemm {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    pool: std::sync::Mutex<BufPool>,
    healthy: AtomicBool,
    /// True = f16 커널 사용 (활성값 f16 업로드). False = f32 커널.
    pub use_f16: bool,
    pub adapter_summary: String,
}

fn ensure_buffer<'p>(
    device: &wgpu::Device,
    slot: &'p mut Option<(wgpu::Buffer, u64)>,
    size: u64,
    usage: wgpu::BufferUsages,
    label: &str,
) -> &'p wgpu::Buffer {
    let grow = match slot {
        Some((_, cap)) => *cap < size,
        None => true,
    };
    if grow {
        let cap = size.next_power_of_two().max(1 << 16);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: cap,
            usage,
            mapped_at_creation: false,
        });
        *slot = Some((buffer, cap));
    }
    &slot.as_ref().unwrap().0
}

static CONTEXT: OnceLock<Option<Arc<GpuGemm>>> = OnceLock::new();

/// Returns the process-wide GPU context, or None when the GPU path is
/// disabled, absent, software-only, or has failed at runtime.
pub fn context() -> Option<Arc<GpuGemm>> {
    let ctx = CONTEXT.get_or_init(init_context).clone()?;
    if ctx.healthy.load(std::sync::atomic::Ordering::Relaxed) {
        Some(ctx)
    } else {
        None
    }
}

/// Human-readable GPU path status for diagnostics.
pub fn status() -> String {
    match CONTEXT.get_or_init(init_context) {
        Some(ctx) => {
            if ctx.healthy.load(std::sync::atomic::Ordering::Relaxed) {
                format!("active: {}", ctx.adapter_summary)
            } else {
                format!("fallback(runtime-failure): {}", ctx.adapter_summary)
            }
        }
        None => "inactive (opt-in: RASE_GPU=1)".to_owned(),
    }
}

fn init_context() -> Option<Arc<GpuGemm>> {
    // 실측 게이트(2026-07-21): 현 f32 커널은 CPU GEMM 과 패리티 수준이라
    // 기본 비활성(옵트인 RASE_GPU=1). f16 커널이 CPU 를 상회하면 기본값을 뒤집는다.
    if !std::env::var("RASE_GPU").map(|v| v == "1").unwrap_or(false) {
        return None;
    }
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let info = adapter.get_info();
    // 소프트웨어 래스터라이저(WARP·llvmpipe)는 CPU 커널보다 느리다 — 제외.
    if matches!(info.device_type, wgpu::DeviceType::Cpu) {
        return None;
    }
    let limits = adapter.limits();
    // f16 커널은 이 장비(Arc/Vulkan/naga) 실측에서 f32 보다 느렸다(356 vs 572
    // GFLOPS — 패킹 f16 연산으로 안 내려가는 것으로 추정). 실험용 옵트인으로만 남긴다.
    let use_f16 = std::env::var("RASE_GPU_F16").map(|v| v == "1").unwrap_or(false)
        && adapter.features().contains(wgpu::Features::SHADER_F16);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("rase-prefill-gemm"),
        required_features: if use_f16 {
            wgpu::Features::SHADER_F16
        } else {
            wgpu::Features::empty()
        },
        required_limits: limits.clone(),
        ..Default::default()
    }))
    .ok()?;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("q4k-gemm"),
        source: wgpu::ShaderSource::Wgsl(if use_f16 { SHADER_F16 } else { SHADER }.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("q4k-gemm-bgl"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("q4k-gemm-pl"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("q4k-gemm-pipeline"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let summary = format!(
        "{} ({:?}/{:?}, {})",
        info.name,
        info.device_type,
        info.backend,
        if use_f16 { "f16" } else { "f32" },
    );
    Some(Arc::new(GpuGemm {
        device,
        queue,
        pipeline,
        layout,
        pool: std::sync::Mutex::new(BufPool::default()),
        healthy: AtomicBool::new(true),
        use_f16,
        adapter_summary: summary,
    }))
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuWeight {
    pub fn out_features(&self) -> usize {
        self.n
    }
}

impl GpuGemm {
    fn mark_unhealthy(&self) {
        self.healthy.store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// Uploads a Q4_K weight matrix for residency. Returns None (CPU 폴백)
    /// when the tensor is not offloadable.
    pub fn upload_weight(&self, weight: &QTensor) -> Option<GpuWeight> {
        if weight.dtype() != GgmlDType::Q4K {
            return None;
        }
        let dims = weight.shape().dims();
        let (n, k) = match *dims {
            [n, k] => (n, k),
            _ => return None,
        };
        if k % Q4K_BLOCK_ELEMS != 0 {
            return None;
        }
        let data = weight.data().ok()?;
        let expected = n * (k / Q4K_BLOCK_ELEMS) * Q4K_BLOCK_U32 * 4;
        if data.len() != expected {
            return None;
        }
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q4k-weights"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, &data);
        Some(GpuWeight { buffer, n, k })
    }

    /// C[m,n] = A[m,k] x W[n,k]^T with in-shader dequantization.
    pub fn gemm(&self, weight: &GpuWeight, a: &[f32], m: usize) -> Option<Vec<f32>> {
        let (k, n) = (weight.k, weight.n);
        if a.len() != m * k || m == 0 {
            return None;
        }
        let mut out = vec![0f32; m * n];
        for chunk_start in (0..m).step_by(CHUNK_M) {
            let rows = CHUNK_M.min(m - chunk_start);
            // 셰이더 합체 로드를 위해 청크를 [k][mp](행 패딩) 로 전치해 올린다.
            let mp = rows.div_ceil(TILE_M) * TILE_M;
            let a_chunk = &a[chunk_start * k..(chunk_start + rows) * k];
            let c_chunk = &mut out[chunk_start * n..(chunk_start + rows) * n];
            let ok = if self.use_f16 {
                let mut a_t = vec![0u16; k * mp];
                a_t.par_chunks_mut(mp).enumerate().for_each(|(e, col)| {
                    for (r, slot) in col.iter_mut().enumerate().take(rows) {
                        *slot = half::f16::from_f32(a_chunk[r * k + e]).to_bits();
                    }
                });
                self.dispatch(weight, bytemuck::cast_slice(&a_t), rows, mp, c_chunk)
            } else {
                let mut a_t = vec![0f32; k * mp];
                a_t.par_chunks_mut(mp).enumerate().for_each(|(e, col)| {
                    for (r, slot) in col.iter_mut().enumerate().take(rows) {
                        *slot = a_chunk[r * k + e];
                    }
                });
                self.dispatch(weight, bytemuck::cast_slice(&a_t), rows, mp, c_chunk)
            };
            if !ok {
                self.mark_unhealthy();
                return None;
            }
        }
        Some(out)
    }

    fn dispatch(&self, weight: &GpuWeight, a_bytes: &[u8], m: usize, mp: usize, c_out: &mut [f32]) -> bool {
        let (k, n) = (weight.k, weight.n);
        let c_bytes = (m * n * 4) as u64;

        let mut pool = self.pool.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let pool = &mut *pool;
        let a_buf = ensure_buffer(
            &self.device,
            &mut pool.a,
            a_bytes.len() as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            "q4k-activations",
        );
        self.queue.write_buffer(a_buf, 0, a_bytes);

        let c_buf = ensure_buffer(
            &self.device,
            &mut pool.c,
            c_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "q4k-output",
        );
        let read_buf = ensure_buffer(
            &self.device,
            &mut pool.read,
            c_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            "q4k-readback",
        );
        let params = [m as u32, (mp / 4) as u32, n as u32, (k / Q4K_BLOCK_ELEMS) as u32];
        let p_buf = pool.params.get_or_insert_with(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("q4k-params"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        self.queue.write_buffer(p_buf, 0, bytemuck::cast_slice(&params));

        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("q4k-bind"),
            layout: &self.layout,
            entries: &[
                bind_entry(0, &weight.buffer),
                bind_entry(1, a_buf),
                bind_entry(2, c_buf),
                bind_entry(3, p_buf),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("q4k-enc") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("q4k-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind, &[]);
            // f32 커널 = 스레드당 4컬럼, f16 커널 = 2컬럼 (레지스터 압력 균형)
            let cols_per_wg = if self.use_f16 { WG_COLS * 2 } else { WG_COLS * 4 };
            let wg_x = n.div_ceil(cols_per_wg) as u32;
            let wg_y = m.div_ceil(TILE_M) as u32;
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        encoder.copy_buffer_to_buffer(&c_buf, 0, &read_buf, 0, c_bytes);
        self.queue.submit([encoder.finish()]);

        let (tx, rx) = std::sync::mpsc::channel();
        // 풀 버퍼는 용량이 요청보다 클 수 있어 결과 구간만 매핑한다.
        read_buf.map_async(wgpu::MapMode::Read, 0..c_bytes, move |r| {
            let _ = tx.send(r);
        });
        if self.device.poll(wgpu::PollType::wait_indefinitely()).is_err() {
            return false;
        }
        match rx.recv() {
            Ok(Ok(())) => {}
            _ => return false,
        }
        {
            let view = match read_buf.get_mapped_range(0..c_bytes) {
                Ok(view) => view,
                Err(_) => return false,
            };
            c_out.copy_from_slice(bytemuck::cast_slice(&view));
        }
        read_buf.unmap();
        true
    }
}

fn bind_entry<'a>(binding: u32, buffer: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// Records GPU GEMM phase time when profiling is enabled.
pub fn commit_gpu_probe(probe: Option<std::time::Instant>) {
    profiling::commit(&profiling::phases().gemm_gpu_ns, probe);
    profiling::count(&profiling::phases().gemm_gpu_calls, 1);
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// 고정 오버헤드 분리용 수동 마이크로벤치 (cargo test --release -- --ignored --nocapture)
    #[test]
    #[ignore]
    fn bench_call_overhead() {
        unsafe { std::env::set_var("RASE_GPU", "1") };
        let Some(ctx) = context() else {
            eprintln!("skip: no GPU");
            return;
        };
        let device = Device::Cpu;
        let (n, k) = (6144usize, 2048usize);
        let w = Tensor::randn(0f32, 1f32, (n, k), &device).unwrap();
        let qw = QTensor::quantize(&w, GgmlDType::Q4K).unwrap();
        let gw = ctx.upload_weight(&qw).unwrap();
        for m in [16usize, 64, 256, 704, 1024] {
            let a = vec![0.5f32; m * k];
            let _ = ctx.gemm(&gw, &a, m); // 워밍업
            let t0 = std::time::Instant::now();
            let iters = 5;
            for _ in 0..iters {
                let _ = ctx.gemm(&gw, &a, m);
            }
            let per = t0.elapsed().as_secs_f64() / iters as f64;
            let gflops = (2.0 * m as f64 * n as f64 * k as f64) / per / 1e9;
            eprintln!("m={m:5}: {:.2} ms/call, {:.0} GFLOPS", per * 1e3, gflops);
        }
    }

    /// GPU 셰이더 역양자화 GEMM은 CPU 역양자화 + f32 행렬곱과 허용오차
    /// 안에서 일치해야 한다 (연산 순서 차이로 비트 동일은 기대하지 않음).
    /// GPU 가 없는 환경(CI 등)에서는 조용히 통과한다 — CPU 회귀는
    /// qwen3_model 쪽 기존 테스트가 보증한다.
    #[test]
    fn gpu_gemm_matches_cpu_dequant_reference() {
        // 옵트인 기본값이므로 테스트에서 명시 활성화
        unsafe { std::env::set_var("RASE_GPU", "1") };
        let Some(ctx) = context() else {
            eprintln!("skip: no usable GPU adapter");
            return;
        };
        let device = Device::Cpu;
        let (n, k, m) = (96usize, 512usize, 70usize); // k = 256의 배수, m > 워크그룹 타일
        let fill = |len: usize, phase: f32| -> Vec<f32> {
            (0..len).map(|i| ((i as f32) * 0.71 + phase).sin()).collect()
        };
        let weight = Tensor::from_vec(fill(n * k, 0.0), (n, k), &device).unwrap();
        let qweight = QTensor::quantize(&weight, GgmlDType::Q4K).unwrap();

        let gpu_weight = ctx.upload_weight(&qweight).expect("Q4K upload");
        let a = fill(m * k, 1.3);
        let got = ctx.gemm(&gpu_weight, &a, m).expect("gpu gemm");

        // CPU 레퍼런스: 같은 양자화 가중치를 역양자화해 f32 행렬곱
        let w_deq = qweight.dequantize(&device).unwrap();
        let a_t = Tensor::from_vec(a.clone(), (m, k), &device).unwrap();
        let want: Vec<f32> = a_t
            .matmul(&w_deq.t().unwrap())
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();

        let mut max_abs = 0f32;
        let mut ref_scale = 0f32;
        for (g, w) in got.iter().zip(want.iter()) {
            max_abs = max_abs.max((g - w).abs());
            ref_scale = ref_scale.max(w.abs());
        }
        // f16 커널은 활성값·곱셈이 반정밀이라 허용오차를 넓힌다 (g 그룹마다
        // f32 플러시로 묶어도 f32 커널 대비 한 자릿수 큰 오차가 정상)
        let tol = if ctx.use_f16 { 2e-2 } else { 1e-3 };
        assert!(
            max_abs <= tol * ref_scale.max(1.0),
            "GPU/CPU divergence: max_abs={max_abs}, ref_scale={ref_scale}, tol={tol}"
        );
    }
}
