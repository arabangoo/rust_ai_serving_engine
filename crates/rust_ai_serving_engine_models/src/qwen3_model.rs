//! Qwen3 GGUF forward pass with a hybrid prefill path.
//!
//! Forked from candle-transformers 0.11 `quantized_qwen3` with one change:
//! every quantized linear goes through [`HybridQMatMul`], which keeps the
//! quantized kernel for decode (single token — memory-optimal) but switches
//! to a transient dequantize + f32 GEMM for prefill (many tokens).
//!
//! Candle's quantized matmul streams and dequantizes the full weight matrix
//! once per input row, so prefilling N prompt tokens costs N decode steps.
//! The GEMM path dequantizes each layer's weights exactly once per forward
//! call and multiplies all rows against it, which is how llama.cpp keeps
//! prompt processing an order of magnitude faster than decoding. The
//! dequantized weights are dropped immediately after the matmul, so resident
//! memory does not grow (peak transient buffer is one weight matrix).

use std::io::{Read, Seek};
use std::sync::Arc;

use candle_core::cpu::kernels::VecOps;
use candle_core::quantized::{QMatMul, QTensor, gguf_file};
use candle_core::{DType, Device, Result, Storage, Tensor};
use candle_nn::attention::cpu_flash::causal::causal_decode_f32_interleaved;
use candle_nn::attention::{AttnMask, flash_attn};
use candle_nn::kv_cache::{ConcatKvCache, InterleavedKvCache, RawInterleavedKvCache};
use candle_nn::{Activation, Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;
use candle_transformers::utils::repeat_kv;
use rayon::prelude::*;

/// Prompt lengths at or above this use the dequantize + GEMM path.
/// Below it, the per-row quantized kernel is cheaper than paying the
/// fixed cost of one full weight dequantization. Break-even measured on a
/// 16-core AVX2 laptop with Qwen3-4B q4_k_m: quantized ≈ 0.06-0.13 s/token,
/// GEMM ≈ 4 s fixed + 0.02 s/token → crossover in the 40-100 token range.
const PREFILL_GEMM_MIN_SEQ: usize = 64;

/// Prompt lengths at or above this use the blocked causal attention kernel
/// instead of candle's row-wise flash kernel. Short prompts keep candle's
/// kernel (equivalent cost below a few hundred tokens).
const PREFILL_BLOCKED_ATTN_MIN_SEQ: usize = 256;

/// Query rows processed together per K/V pass in the blocked prefill kernel.
/// Larger blocks amortize K/V reads further; 16 rows × d=128 keeps the
/// per-task accumulator at 8 KB (L1-resident).
const ATTN_QUERY_BLOCK: usize = 16;

/// One streamed (score, v_row) online-softmax update. Port of candle-nn's
/// crate-private `online_softmax_step` (Apache-2.0).
#[inline(always)]
fn online_softmax_step(score: f32, m: &mut f32, ssum: &mut f32, acc: &mut [f32], v_row: &[f32]) {
    if score > *m {
        let scale_old = (*m - score).exp();
        for a in acc.iter_mut() {
            *a *= scale_old;
        }
        *ssum = *ssum * scale_old + 1.0;
        *m = score;
        for (a, &e) in acc.iter_mut().zip(v_row) {
            *a += e;
        }
    } else {
        let w = (score - *m).exp();
        for (a, &e) in acc.iter_mut().zip(v_row) {
            *a += e * w;
        }
        *ssum += w;
    }
}

/// Blocked causal flash attention for prefill (f32, batch 1, offset 0).
///
/// candle's CPU flash kernel streams the full K/V history once per query
/// row, so prompt attention pays L² strided K/V reads (~13 µs/token²
/// measured on Qwen3-4B). Processing `ATTN_QUERY_BLOCK` query rows per K/V
/// pass reuses each K/V row across the whole block — same exact online
/// softmax, a fraction of the memory traffic. No score matrix is ever
/// materialized.
///
/// Layouts: `q` is `[h_q][l][d]`, `k`/`v` are `[h_kv][l][d]` (GQA shared),
/// and the returned buffer is `[h_q][l][d]`.
fn blocked_causal_prefill_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    h_q: usize,
    h_kv: usize,
    l: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let groups = h_q / h_kv;
    let mut out = vec![0f32; h_q * l * d];
    out.par_chunks_mut(l * d)
        .enumerate()
        .for_each(|(h, out_h)| {
            let kv_h = h / groups;
            let q_h = &q[h * l * d..(h + 1) * l * d];
            let k_h = &k[kv_h * l * d..(kv_h + 1) * l * d];
            let v_h = &v[kv_h * l * d..(kv_h + 1) * l * d];
            out_h
                .par_chunks_mut(ATTN_QUERY_BLOCK * d)
                .enumerate()
                .for_each(|(blk, out_blk)| {
                    let q0 = blk * ATTN_QUERY_BLOCK;
                    let rows = (l - q0).min(ATTN_QUERY_BLOCK);
                    let mut m = [f32::NEG_INFINITY; ATTN_QUERY_BLOCK];
                    let mut ssum = [0f32; ATTN_QUERY_BLOCK];
                    let mut acc = vec![0f32; rows * d];
                    // Causal bound of the last row in the block.
                    for j in 0..(q0 + rows) {
                        let k_row = &k_h[j * d..(j + 1) * d];
                        let v_row = &v_h[j * d..(j + 1) * d];
                        // Row i may attend to j only when q0 + i >= j.
                        let first = j.saturating_sub(q0);
                        for i in first..rows {
                            let q_row = &q_h[(q0 + i) * d..(q0 + i + 1) * d];
                            let mut score = 0f32;
                            // SAFETY: q_row/k_row are both exactly `d` long.
                            unsafe {
                                f32::vec_dot(q_row.as_ptr(), k_row.as_ptr(), &mut score, d);
                            }
                            score *= scale;
                            online_softmax_step(
                                score,
                                &mut m[i],
                                &mut ssum[i],
                                &mut acc[i * d..(i + 1) * d],
                                v_row,
                            );
                        }
                    }
                    for i in 0..rows {
                        let inv = if ssum[i] > 0.0 { 1.0 / ssum[i] } else { 0.0 };
                        let dst = &mut out_blk[i * d..(i + 1) * d];
                        let a = &acc[i * d..(i + 1) * d];
                        for t in 0..d {
                            dst[t] = a[t] * inv;
                        }
                    }
                });
        });
    out
}

/// Quantized linear that picks its kernel by sequence length.
#[derive(Debug, Clone)]
struct HybridQMatMul {
    quantized: QMatMul,
    weight: Arc<QTensor>,
}

impl HybridQMatMul {
    fn from_qtensor(weight: QTensor) -> Result<Self> {
        let weight = Arc::new(weight);
        let quantized = QMatMul::from_arc(weight.clone())?;
        Ok(Self { quantized, weight })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let seq = match *x.dims() {
            [_, seq, _] => seq,
            _ => 1,
        };
        if seq >= PREFILL_GEMM_MIN_SEQ && x.device().is_cpu() {
            // Prefill: dequantize the weights once, GEMM all rows against them.
            let w = self.weight.dequantize(x.device())?; // [out, in] f32
            let (b, l, hidden) = x.dims3()?;
            let flat = x.reshape((b * l, hidden))?;
            flat.matmul(&w.t()?)?.reshape((b, l, ()))
        } else {
            self.quantized.forward(x)
        }
    }
}

pub struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<HybridQMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        HybridQMatMul::from_qtensor(ws)
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: HybridQMatMul,
    up_proj: HybridQMatMul,
    down_proj: HybridQMatMul,
    act_fn: Activation,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: Activation::Silu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
    cos_f32: Vec<f32>,
    sin_f32: Vec<f32>,
    half_d: usize,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        let sin_t = freqs.sin()?;
        let cos_t = freqs.cos()?;
        let cos_f32 = cos_t
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let sin_f32 = sin_t
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        Ok(Self {
            sin: sin_t,
            cos: cos_t,
            cos_f32,
            sin_f32,
            half_d: dim / 2,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }

    #[inline]
    fn cos_sin_at(&self, pos: usize) -> (&[f32], &[f32]) {
        let start = pos * self.half_d;
        let end = start + self.half_d;
        (&self.cos_f32[start..end], &self.sin_f32[start..end])
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: HybridQMatMul,
    k_proj: HybridQMatMul,
    v_proj: HybridQMatMul,
    o_proj: HybridQMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: Option<ConcatKvCache>,
    interleaved_cache: Option<InterleavedKvCache>,
    raw_cache: Option<RawInterleavedKvCache>,
}

impl AttentionWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        device: &Device,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let hidden_size = num_heads * head_dim;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        // CPU: interleaved + raw caches feed the flash-attention kernels.
        // Other devices: standard concat KV cache.
        let on_cpu = device.is_cpu();
        let kv_cache = if on_cpu {
            None
        } else {
            Some(ConcatKvCache::new(2))
        };
        let interleaved_cache = if on_cpu {
            Some(InterleavedKvCache::new(head_dim))
        } else {
            None
        };
        let raw_cache = if on_cpu {
            Some(RawInterleavedKvCache::new(num_kv_heads, head_dim, 4096))
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            hidden_size,
            rotary_emb,
            kv_cache,
            interleaved_cache,
            raw_cache,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        if x.device().is_cpu() && b == 1 {
            let scale = 1.0 / (self.head_dim as f32).sqrt();

            if l == 1 && q.dtype() == DType::F32 {
                // Fused single-token decode over the raw interleaved cache.
                let q_cont = q.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (q_g, q_l) = q_cont.storage_and_layout();
                let q_data: &[f32] = match &*q_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[q_l.start_offset()..],
                    _ => candle_core::bail!("Expected CPU storage"),
                };

                let k_cont = k.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (k_g, k_l) = k_cont.storage_and_layout();
                let k_data: &[f32] = match &*k_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[k_l.start_offset()..],
                    _ => candle_core::bail!("Expected CPU storage"),
                };

                let v_cont = v.squeeze(0)?.squeeze(1)?.contiguous()?;
                let (v_g, v_l) = v_cont.storage_and_layout();
                let v_data: &[f32] = match &*v_g {
                    Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[v_l.start_offset()..],
                    _ => candle_core::bail!("Expected CPU storage"),
                };

                let k_len = self.num_kv_heads * self.head_dim;
                let rc = self.raw_cache.as_mut().unwrap();
                rc.write_kv(&k_data[..k_len], &v_data[..k_len]);

                let kv_len = rc.len();
                let q_len = self.num_heads * self.head_dim;
                let ctx = causal_decode_f32_interleaved(
                    &q_data[..q_len],
                    rc.data(),
                    self.num_heads,
                    self.num_kv_heads,
                    self.head_dim,
                    kv_len,
                    scale,
                )?;

                let ctx = ctx.reshape((b, l, self.hidden_size))?;
                self.o_proj.forward(&ctx)
            } else {
                // Prefill: populate the interleaved + raw caches for the
                // subsequent fused decode steps (identical for both paths).
                let ic = self.interleaved_cache.as_mut().unwrap();
                let kv = ic.append(&k, &v)?;

                {
                    let k_cont = k.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                    let v_cont = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
                    let (kg, kl) = k_cont.storage_and_layout();
                    let k_d: &[f32] = match &*kg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[kl.start_offset()..],
                        _ => candle_core::bail!("Expected CPU"),
                    };
                    let (vg, vl) = v_cont.storage_and_layout();
                    let v_d: &[f32] = match &*vg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[vl.start_offset()..],
                        _ => candle_core::bail!("Expected CPU"),
                    };
                    self.raw_cache.as_mut().unwrap().write_kv_batch(k_d, v_d, l);
                }

                if l >= PREFILL_BLOCKED_ATTN_MIN_SEQ && offset == 0 {
                    // Long prefill: blocked causal kernel (see its docs).
                    let q_c = q.contiguous()?; // (1, h_q, l, d)
                    let k_c = k.contiguous()?; // (1, h_kv, l, d)
                    let v_c = v.contiguous()?;
                    let (qg, ql) = q_c.storage_and_layout();
                    let q_d: &[f32] = match &*qg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[ql.start_offset()..],
                        _ => candle_core::bail!("Expected CPU"),
                    };
                    let (kg, kl) = k_c.storage_and_layout();
                    let k_d: &[f32] = match &*kg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[kl.start_offset()..],
                        _ => candle_core::bail!("Expected CPU"),
                    };
                    let (vg, vl) = v_c.storage_and_layout();
                    let v_d: &[f32] = match &*vg {
                        Storage::Cpu(cpu) => &cpu.as_slice::<f32>()?[vl.start_offset()..],
                        _ => candle_core::bail!("Expected CPU"),
                    };
                    let out = blocked_causal_prefill_f32(
                        q_d,
                        k_d,
                        v_d,
                        self.num_heads,
                        self.num_kv_heads,
                        l,
                        self.head_dim,
                        scale,
                    );
                    let ctx =
                        Tensor::from_vec(out, (self.num_heads, l, self.head_dim), x.device())?
                            .transpose(0, 1)?
                            .contiguous()?
                            .reshape((b, l, self.hidden_size))?;
                    self.o_proj.forward(&ctx)
                } else {
                    // Short prefill (or cache continuation): flash kernel.
                    let kv_k = kv.narrow(2, 0, self.head_dim)?.unsqueeze(0)?;
                    let kv_v = kv.narrow(2, self.head_dim, self.head_dim)?.unsqueeze(0)?;

                    let q = q.transpose(1, 2)?.contiguous()?;
                    let k = kv_k.contiguous()?;
                    let v = kv_v.contiguous()?;

                    let ctx = flash_attn::<f32>(
                        &q,
                        &k,
                        &v,
                        scale,
                        AttnMask::causal_with_offset(offset),
                        None,
                        None,
                    )?;
                    let ctx = ctx.transpose(1, 2)?;
                    let ctx = ctx.reshape((b, l, self.hidden_size))?;
                    self.o_proj.forward(&ctx)
                }
            }
        } else {
            // Standard matmul attention (no flash kernels).
            let (k, v) = self.kv_cache.as_mut().unwrap().append(&k, &v)?;

            let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
            let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

            let scale = 1.0 / (self.head_dim as f64).sqrt();
            let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
            if let Some(m) = attn_mask {
                let scores_dtype = scores.dtype();
                let mask = if m.dtype() != scores_dtype {
                    m.to_dtype(scores_dtype)?
                } else {
                    m.clone()
                };
                scores = scores.broadcast_add(&mask)?;
            }
            let probs = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = probs.matmul(&v)?;
            let reshaped_ctx = ctx.transpose(1, 2)?.reshape((b, l, self.hidden_size))?;
            self.o_proj.forward(&reshaped_ctx)
        }
    }

    fn clear_kv_cache(&mut self) {
        if let Some(c) = &mut self.kv_cache {
            c.reset();
        }
        if let Some(c) = &mut self.interleaved_cache {
            c.reset();
        }
        if let Some(c) = &mut self.raw_cache {
            c.reset();
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    #[allow(clippy::too_many_arguments)]
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        device: &Device,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            device,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

/// Qwen3 GGUF weights with hybrid (quantized decode / GEMM prefill) linears.
pub struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: HybridQMatMul,
    device: Device,
    dtype: DType,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                device,
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Output projection, falling back to tied embeddings.
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = HybridQMatMul::from_qtensor(lm_head_tensor)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
        })
    }

    fn causal_mask(&self, b: usize, tgt: usize, offset: usize) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| (0..(tgt + offset)).map(move |j| if j <= i + offset { 0. } else { minf }))
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        // Mask materialization is skipped on CPU (kernels apply causality).
        let causal_mask = if l == 1 || self.device.is_cpu() {
            None
        } else {
            Some(self.causal_mask(b, l, offset)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::GgmlDType;

    /// The blocked causal kernel must match a naive exact-softmax reference.
    #[test]
    fn blocked_causal_prefill_matches_reference() {
        let (h_q, h_kv, l, d) = (4usize, 2usize, 37usize, 8usize);
        let scale = 0.35f32;
        let fill = |n: usize, phase: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * 0.37 + phase).sin()).collect()
        };
        let q = fill(h_q * l * d, 0.0);
        let k = fill(h_kv * l * d, 1.3);
        let v = fill(h_kv * l * d, 2.6);

        let got = blocked_causal_prefill_f32(&q, &k, &v, h_q, h_kv, l, d, scale);

        let groups = h_q / h_kv;
        for h in 0..h_q {
            let kv_h = h / groups;
            for i in 0..l {
                // Naive causal softmax attention for row (h, i).
                let q_row = &q[(h * l + i) * d..(h * l + i + 1) * d];
                let scores: Vec<f32> = (0..=i)
                    .map(|j| {
                        let k_row = &k[(kv_h * l + j) * d..(kv_h * l + j + 1) * d];
                        q_row.iter().zip(k_row).map(|(a, b)| a * b).sum::<f32>() * scale
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f32 = exps.iter().sum();
                for t in 0..d {
                    let want: f32 = (0..=i)
                        .map(|j| {
                            let v_row = &v[(kv_h * l + j) * d..(kv_h * l + j + 1) * d];
                            exps[j] / total * v_row[t]
                        })
                        .sum();
                    let have = got[(h * l + i) * d + t];
                    assert!(
                        (want - have).abs() < 1e-4,
                        "mismatch at h={h} i={i} t={t}: want {want}, have {have}"
                    );
                }
            }
        }
    }

    /// The blocked kernel must agree with candle's flash kernel at the real
    /// model dimensions (32 query heads, 8 KV heads, head_dim 128).
    #[test]
    fn blocked_kernel_matches_candle_flash_at_model_dims() {
        let (h_q, h_kv, l, d) = (32usize, 8usize, 300usize, 128usize);
        let scale = 1.0 / (d as f32).sqrt();
        let fill = |n: usize, phase: f32| -> Vec<f32> {
            (0..n)
                .map(|i| ((i as f32) * 0.61 + phase).sin() * 0.3)
                .collect()
        };
        let q = fill(h_q * l * d, 0.0);
        let k = fill(h_kv * l * d, 1.1);
        let v = fill(h_kv * l * d, 2.2);

        let got = blocked_causal_prefill_f32(&q, &k, &v, h_q, h_kv, l, d, scale);

        // candle flash expects (B, S, H, D); ours is [h][l][d].
        let device = Device::Cpu;
        let to_bshd = |data: &[f32], h: usize| -> Tensor {
            Tensor::from_vec(data.to_vec(), (1, h, l, d), &device)
                .unwrap()
                .transpose(1, 2)
                .unwrap()
                .contiguous()
                .unwrap()
        };
        let q_t = to_bshd(&q, h_q);
        let k_t = to_bshd(&k, h_kv);
        let v_t = to_bshd(&v, h_kv);
        let want = flash_attn::<f32>(
            &q_t,
            &k_t,
            &v_t,
            scale,
            AttnMask::causal_with_offset(0),
            None,
            None,
        )
        .unwrap(); // (B, H, S, D)
        let want: Vec<f32> = want.flatten_all().unwrap().to_vec1().unwrap();

        let mut max_diff = 0f32;
        for (a, b) in got.iter().zip(want.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff < 1e-4,
            "blocked kernel diverges from candle flash: max_diff={max_diff}"
        );
    }

    /// The GEMM prefill path must agree with the quantized kernel within the
    /// activation-quantization error of the quantized path itself.
    #[test]
    fn hybrid_prefill_matches_quantized_kernel() {
        let device = Device::Cpu;
        // Q4K 블록 크기(256)에 맞춰 in_dim 은 256 의 배수여야 한다.
        // 값은 결정적으로 생성한다 — 시드 없는 randn 은 간헐 실패를 만든다.
        let (out_dim, in_dim, seq) = (64, 256, PREFILL_GEMM_MIN_SEQ);
        let fill = |n: usize, phase: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32) * 0.83 + phase).sin()).collect()
        };
        let weight =
            Tensor::from_vec(fill(out_dim * in_dim, 0.0), (out_dim, in_dim), &device).unwrap();
        let qweight = QTensor::quantize(&weight, GgmlDType::Q4K).unwrap();
        let hybrid = HybridQMatMul::from_qtensor(qweight).unwrap();

        let x = Tensor::from_vec(fill(seq * in_dim, 1.7), (1, seq, in_dim), &device).unwrap();
        let gemm = hybrid.forward(&x).unwrap(); // seq >= threshold → GEMM path
        let quantized = hybrid.quantized.forward(&x).unwrap();

        let diff = (gemm - quantized)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(diff < 0.5, "GEMM/quantized divergence too large: {diff}");
    }
}
