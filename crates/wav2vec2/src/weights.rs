//! Weight store shared by the safetensors and GGUF loaders, and the layers
//! built from it: a linear layer that runs dense or quantized (`QMatMul`),
//! and LayerNorm (computed in F32).

use anyhow::{Context, Result};
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{D, DType, Device, Module, Tensor};
use std::collections::HashMap;
use std::sync::Arc;

pub struct Weights {
    map: HashMap<String, Arc<QTensor>>,
    pub device: Device,
}

impl Weights {
    pub fn new(map: HashMap<String, Arc<QTensor>>, device: Device) -> Self {
        Self { map, device }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }

    fn raw(&self, name: &str) -> Result<&Arc<QTensor>> {
        self.map
            .get(name)
            .with_context(|| format!("missing tensor '{name}'"))
    }

    /// Dense tensor in `dtype` on the model device.
    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor> {
        let q = self.raw(name)?;
        let t = q.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
        Ok(t.to_device(&self.device)?)
    }

    pub fn linear(&self, prefix: &str, bias: bool, dtype: DType) -> Result<Linear> {
        let name = format!("{prefix}.weight");
        let q = self.raw(&name)?;
        let bias = if bias {
            Some(self.get(&format!("{prefix}.bias"), dtype)?)
        } else {
            None
        };
        let inner = match q.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => {
                let t = q.dequantize(&Device::Cpu)?.to_dtype(dtype)?;
                LinearInner::Dense(t.to_device(&self.device)?)
            }
            _ => {
                let q = if q.device().same_device(&self.device) {
                    q.clone()
                } else {
                    // GGUF tensors are read straight to the device; this is the
                    // checkpoint path (never quantized), kept for completeness.
                    let t = q.dequantize(&Device::Cpu)?;
                    Arc::new(QTensor::quantize(&t.to_device(&self.device)?, q.dtype())?)
                };
                LinearInner::Quant(QMatMul::from_arc(q)?)
            }
        };
        Ok(Linear { inner, bias })
    }
}

#[derive(Debug, Clone)]
enum LinearInner {
    /// `[out, in]` weight in the layer's compute dtype.
    Dense(Tensor),
    /// Block-quantized weight; takes and returns F32 activations.
    Quant(QMatMul),
}

#[derive(Debug, Clone)]
pub struct Linear {
    inner: LinearInner,
    bias: Option<Tensor>,
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Flatten to 2-D: candle 0.11 matmul against a stride-0
        // (broadcast_left) batch returns wrong values.
        let dims = xs.dims().to_vec();
        let k = dims[dims.len() - 1];
        let ys = match &self.inner {
            LinearInner::Dense(w) => {
                let x2 = xs.to_dtype(w.dtype())?.reshape(((), k))?;
                x2.matmul(&w.t()?)?
            }
            LinearInner::Quant(q) => {
                let x2 = xs.to_dtype(DType::F32)?.reshape(((), k))?;
                q.forward(&x2)?.to_dtype(xs.dtype())?
            }
        };
        let ys = match &self.bias {
            Some(b) => ys.broadcast_add(&b.to_dtype(ys.dtype())?)?,
            None => ys,
        };
        let mut out = dims;
        *out.last_mut().unwrap() = ys.dim(1)?;
        ys.reshape(out)
    }
}

/// `nn.LayerNorm` over the last dimension, computed in F32, output in the
/// input dtype.
#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    /// The same, for the multi-threaded CPU kernel.
    w_cpu: Vec<f32>,
    b_cpu: Vec<f32>,
}

impl LayerNorm {
    pub fn load(w: &Weights, prefix: &str, eps: f64) -> Result<Self> {
        let weight = w.get(&format!("{prefix}.weight"), DType::F32)?;
        let bias = w.get(&format!("{prefix}.bias"), DType::F32)?;
        let (w_cpu, b_cpu) = if weight.device().is_cpu() {
            (weight.to_vec1()?, bias.to_vec1()?)
        } else {
            (Vec::new(), Vec::new())
        };
        Ok(Self {
            weight,
            bias,
            eps,
            w_cpu,
            b_cpu,
        })
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        if crate::cpu_ops::fast(xs) && !self.w_cpu.is_empty() {
            return crate::cpu_ops::layer_norm(xs, &self.w_cpu, &self.b_cpu, self.eps);
        }
        let dtype = xs.dtype();
        let x = xs.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_sub(&mean)?;
        let var = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x = x.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        x.broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)?
            .to_dtype(dtype)
    }
}
