//! Multi-threaded CPU kernels for the hot elementwise work (candle 0.11
//! runs unary ops and its own reductions on one thread): GELU (erf),
//! LayerNorm over the last dimension, and the im2col gather of the conv
//! feature encoder in channel-last layout. F32 only; other dtypes and
//! devices use candle's ops.

use candle_core::{CpuStorage, CustomOp1, DType, Layout, Result, Shape, Tensor, bail};
use rayon::prelude::*;

const CHUNK: usize = 4096;

fn f32_slice<'a>(s: &'a CpuStorage, l: &Layout) -> Result<&'a [f32]> {
    let CpuStorage::F32(v) = s else {
        bail!("expected f32 storage")
    };
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => bail!("expected a contiguous tensor"),
    }
}

fn is_fast(x: &Tensor) -> bool {
    x.device().is_cpu() && x.dtype() == DType::F32
}

#[inline(always)]
fn gelu(v: f32) -> f32 {
    // the same formula and erf as candle's gelu_erf
    (candle_core::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
}

struct GeluErf;

impl CustomOp1 for GeluErf {
    fn name(&self) -> &'static str {
        "gelu-erf-par"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let src = f32_slice(s, l)?;
        let mut out = vec![0f32; src.len()];
        out.par_chunks_mut(CHUNK)
            .zip(src.par_chunks(CHUNK))
            .for_each(|(o, i)| {
                for (o, &i) in o.iter_mut().zip(i) {
                    *o = gelu(i);
                }
            });
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }
}

/// Exact (erf) GELU.
pub fn gelu_erf(x: &Tensor) -> Result<Tensor> {
    if is_fast(x) {
        x.contiguous()?.apply_op1_no_bwd(&GeluErf)
    } else {
        x.gelu_erf()
    }
}

struct LayerNormOp<'a> {
    weight: &'a [f32],
    bias: &'a [f32],
    eps: f64,
}

impl CustomOp1 for LayerNormOp<'_> {
    fn name(&self) -> &'static str {
        "layer-norm-par"
    }
    fn cpu_fwd(&self, s: &CpuStorage, l: &Layout) -> Result<(CpuStorage, Shape)> {
        let src = f32_slice(s, l)?;
        let d = self.weight.len();
        if d == 0 || !src.len().is_multiple_of(d) {
            bail!("layer norm: last dim is not {d}");
        }
        let mut out = vec![0f32; src.len()];
        let rows = (CHUNK / d).max(1) * d;
        out.par_chunks_mut(rows)
            .zip(src.par_chunks(rows))
            .for_each(|(o, i)| {
                for (o, x) in o.chunks_exact_mut(d).zip(i.chunks_exact(d)) {
                    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / d as f64;
                    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / d as f64;
                    let inv = 1.0 / (var + self.eps).sqrt();
                    for k in 0..d {
                        o[k] = ((x[k] as f64 - mean) * inv) as f32 * self.weight[k] + self.bias[k];
                    }
                }
            });
        Ok((CpuStorage::F32(out), l.shape().clone()))
    }
}

/// LayerNorm over the last dimension with F32 `weight` and `bias` vectors.
pub fn layer_norm(x: &Tensor, weight: &[f32], bias: &[f32], eps: f64) -> Result<Tensor> {
    x.contiguous()?
        .apply_op1_no_bwd(&LayerNormOp { weight, bias, eps })
}

pub fn fast(x: &Tensor) -> bool {
    is_fast(x)
}

/// im2col of a channel-last `[T, C]` F32 tensor for a 1-D convolution
/// (kernel `k`, stride `s`, no padding): `[T', k * C]`, row t = rows
/// `t * s .. t * s + k` of the input, which are contiguous.
pub fn im2col(x: &Tensor, k: usize, s: usize) -> Result<Tensor> {
    let (t, c) = x.dims2()?;
    if t < k {
        bail!("im2col: {t} frames < kernel {k}");
    }
    let t_out = (t - k) / s + 1;
    let x = x.contiguous()?;
    let (st, l) = x.storage_and_layout();
    let candle_core::Storage::Cpu(cpu) = &*st else {
        bail!("im2col: CPU tensors only")
    };
    let src = f32_slice(cpu, l)?;
    let w = k * c;
    let mut out = vec![0f32; t_out * w];
    let rows = (CHUNK / w).max(1);
    out.par_chunks_mut(rows * w)
        .enumerate()
        .for_each(|(ci, o)| {
            for (r, row) in o.chunks_exact_mut(w).enumerate() {
                let t0 = (ci * rows + r) * s;
                row.copy_from_slice(&src[t0 * c..t0 * c + w]);
            }
        });
    drop(st);
    Tensor::from_vec(out, (t_out, w), x.device())
}
