//! `Wav2Vec2ForCTC` forward pass (inference), as Hugging Face transformers'
//! `modeling_wav2vec2.py`:
//!
//! waveform → conv feature encoder (7 strided Conv1d, LayerNorm per layer
//! or GroupNorm on the first, GELU) → feature projection (LayerNorm,
//! Linear) → + convolutional positional embedding (grouped Conv1d with
//! weight norm folded at conversion, `SamePad`, GELU) → transformer
//! (pre-LN "stable layer norm" for XLS-R / large, post-LN for base) →
//! `lm_head` → CTC logits.

use crate::config::Config;
use crate::cpu_ops;
use crate::weights::{LayerNorm, Linear, Weights};
use anyhow::{Result, bail};
use candle_core::{D, DType, Module, Tensor};

enum ConvNorm {
    None,
    /// LayerNorm over channels (`feat_extract_norm = "layer"`).
    Layer(LayerNorm),
    /// GroupNorm with one group per channel: each channel normalized over
    /// time (`feat_extract_norm = "group"`, first layer only).
    Group {
        weight: Tensor,
        bias: Tensor,
        eps: f64,
    },
}

struct ConvLayer {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    norm: ConvNorm,
    /// CPU F32 channel-last path: the kernel as a `[k * C_in, C_out]` matrix.
    wmat: Option<Tensor>,
}

impl ConvLayer {
    /// `[B, C_in, L]` → `[B, C_out, L']`.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut y = x.conv1d(&self.weight, 0, self.stride, 1, 1)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(&b.reshape((1, (), 1))?)?;
        }
        let y = match &self.norm {
            ConvNorm::None => y,
            ConvNorm::Layer(ln) => ln.forward(&y.transpose(1, 2)?)?.transpose(1, 2)?,
            ConvNorm::Group { weight, bias, eps } => {
                let dtype = y.dtype();
                let x = y.to_dtype(DType::F32)?;
                let mean = x.mean_keepdim(D::Minus1)?;
                let x = x.broadcast_sub(&mean)?;
                let var = x.sqr()?.mean_keepdim(D::Minus1)?;
                let x = x.broadcast_div(&(var + *eps)?.sqrt()?)?;
                x.broadcast_mul(&weight.reshape((1, (), 1))?)?
                    .broadcast_add(&bias.reshape((1, (), 1))?)?
                    .to_dtype(dtype)?
            }
        };
        Ok(cpu_ops::gelu_erf(&y)?)
    }

    /// Channel-last `[T, C_in]` → `[T', C_out]` (CPU, F32, LayerNorm).
    fn forward_cl(&self, x: &Tensor, wmat: &Tensor) -> Result<Tensor> {
        let k = self.weight.dim(2)?;
        let cols = cpu_ops::im2col(x, k, self.stride)?;
        let mut y = cols.matmul(wmat)?;
        if let Some(b) = &self.bias {
            y = y.broadcast_add(b)?;
        }
        let y = match &self.norm {
            ConvNorm::Layer(ln) => ln.forward(&y)?,
            _ => bail!("channel-last path needs LayerNorm"),
        };
        Ok(cpu_ops::gelu_erf(&y)?)
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    heads: usize,
    scale: f64,
}

impl Attention {
    fn load(w: &Weights, p: &str, cfg: &Config, dtype: DType) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q: w.linear(&format!("{p}.q_proj"), true, dtype)?,
            k: w.linear(&format!("{p}.k_proj"), true, dtype)?,
            v: w.linear(&format!("{p}.v_proj"), true, dtype)?,
            out: w.linear(&format!("{p}.out_proj"), true, dtype)?,
            heads: cfg.num_attention_heads,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// `[B, T, C]` → `[B, T, C]`, full (unmasked) self-attention.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, c) = x.dims3()?;
        let hd = c / self.heads;
        let split = |y: Tensor| -> Result<Tensor> {
            Ok(y.reshape((b, t, self.heads, hd))?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b * self.heads, t, hd))?)
        };
        let q = split((self.q.forward(x)? * self.scale)?)?;
        let k = split(self.k.forward(x)?)?;
        let v = split(self.v.forward(x)?)?;
        let scores = q.matmul(&k.t()?.contiguous()?)?;
        let probs =
            candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32)?)?.to_dtype(v.dtype())?;
        let ctx = probs
            .matmul(&v)?
            .reshape((b, self.heads, t, hd))?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, c))?;
        Ok(self.out.forward(&ctx)?)
    }
}

struct EncoderLayer {
    attn: Attention,
    layer_norm: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
    final_layer_norm: LayerNorm,
    stable: bool,
}

impl EncoderLayer {
    fn feed_forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self
            .ff_out
            .forward(&cpu_ops::gelu_erf(&self.ff_in.forward(x)?)?)?)
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.stable {
            let h = (x + self.attn.forward(&self.layer_norm.forward(x)?)?)?;
            let ff = self.feed_forward(&self.final_layer_norm.forward(&h)?)?;
            Ok((h + ff)?)
        } else {
            let h = self.layer_norm.forward(&(x + self.attn.forward(x)?)?)?;
            let h = (&h + self.feed_forward(&h)?)?;
            Ok(self.final_layer_norm.forward(&h)?)
        }
    }
}

pub struct Wav2Vec2ForCtc {
    conv: Vec<ConvLayer>,
    proj_norm: LayerNorm,
    proj: Linear,
    pos_conv_weight: Tensor,
    pos_conv_bias: Tensor,
    pos_groups: usize,
    pos_kernel: usize,
    encoder_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
    lm_head: Linear,
    stable: bool,
    pub dtype: DType,
}

impl Wav2Vec2ForCtc {
    pub fn load(w: &Weights, cfg: &Config, dtype: DType) -> Result<Self> {
        Self::load_opts(w, cfg, dtype, true)
    }

    /// `fast_cpu`: the multi-threaded channel-last conv encoder on the CPU
    /// in F32 (else candle's `conv1d`; same results to rounding).
    pub fn load_opts(w: &Weights, cfg: &Config, dtype: DType, fast_cpu: bool) -> Result<Self> {
        let eps = cfg.layer_norm_eps;
        let mut conv = Vec::new();
        for (i, (&k, &s)) in cfg.conv_kernel.iter().zip(&cfg.conv_stride).enumerate() {
            let p = format!("wav2vec2.feature_extractor.conv_layers.{i}");
            let weight = w.get(&format!("{p}.conv.weight"), dtype)?;
            let (_, _, kk) = weight.dims3()?;
            if kk != k {
                bail!("{p}: kernel {kk}, config says {k}");
            }
            let bias = if cfg.conv_bias {
                Some(w.get(&format!("{p}.conv.bias"), dtype)?)
            } else {
                None
            };
            let norm = match (cfg.feat_extract_norm.as_str(), i) {
                ("layer", _) => {
                    ConvNorm::Layer(LayerNorm::load(w, &format!("{p}.layer_norm"), eps)?)
                }
                ("group", 0) => ConvNorm::Group {
                    weight: w.get(&format!("{p}.layer_norm.weight"), DType::F32)?,
                    bias: w.get(&format!("{p}.layer_norm.bias"), DType::F32)?,
                    eps,
                },
                _ => ConvNorm::None,
            };
            let fast = fast_cpu
                && weight.device().is_cpu()
                && dtype == DType::F32
                && cfg.feat_extract_norm == "layer";
            let wmat = if fast {
                let (o, c, k) = weight.dims3()?;
                Some(
                    weight
                        .permute((2, 1, 0))?
                        .contiguous()?
                        .reshape((k * c, o))?,
                )
            } else {
                None
            };
            conv.push(ConvLayer {
                weight,
                bias,
                stride: s,
                norm,
                wmat,
            });
        }
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            let p = format!("wav2vec2.encoder.layers.{i}");
            layers.push(EncoderLayer {
                attn: Attention::load(w, &format!("{p}.attention"), cfg, dtype)?,
                layer_norm: LayerNorm::load(w, &format!("{p}.layer_norm"), eps)?,
                ff_in: w.linear(&format!("{p}.feed_forward.intermediate_dense"), true, dtype)?,
                ff_out: w.linear(&format!("{p}.feed_forward.output_dense"), true, dtype)?,
                final_layer_norm: LayerNorm::load(w, &format!("{p}.final_layer_norm"), eps)?,
                stable: cfg.do_stable_layer_norm,
            });
        }
        Ok(Self {
            conv,
            proj_norm: LayerNorm::load(w, "wav2vec2.feature_projection.layer_norm", eps)?,
            proj: w.linear("wav2vec2.feature_projection.projection", true, dtype)?,
            pos_conv_weight: w.get("wav2vec2.encoder.pos_conv_embed.conv.weight", dtype)?,
            pos_conv_bias: w.get("wav2vec2.encoder.pos_conv_embed.conv.bias", dtype)?,
            pos_groups: cfg.num_conv_pos_embedding_groups,
            pos_kernel: cfg.num_conv_pos_embeddings,
            encoder_norm: LayerNorm::load(w, "wav2vec2.encoder.layer_norm", eps)?,
            layers,
            lm_head: w.linear("lm_head", true, dtype)?,
            stable: cfg.do_stable_layer_norm,
            dtype,
        })
    }

    /// Conv feature encoder: waveform `[B, N]` → `[B, T, C]`.
    fn features(&self, input: &Tensor) -> Result<Tensor> {
        if self.conv.iter().all(|c| c.wmat.is_some()) {
            let mut out = Vec::new();
            for b in 0..input.dim(0)? {
                let mut x = input.get(b)?.to_dtype(DType::F32)?.unsqueeze(1)?;
                for c in &self.conv {
                    x = c.forward_cl(&x, c.wmat.as_ref().unwrap())?;
                }
                out.push(x);
            }
            return Ok(Tensor::stack(&out, 0)?);
        }
        let mut x = input.to_dtype(self.dtype)?.unsqueeze(1)?;
        for c in &self.conv {
            x = c.forward(&x)?;
        }
        // [B, C, T] -> [B, T, C]
        Ok(x.transpose(1, 2)?.contiguous()?)
    }

    /// Normalized waveform `[B, N]` → CTC logits `[B, T, vocab]` (F32).
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let x = self.features(input)?;
        let x = self.proj.forward(&self.proj_norm.forward(&x)?)?;
        let mut h = (&x + self.pos_embed(&x)?)?;
        if !self.stable {
            h = self.encoder_norm.forward(&h)?;
        }
        for l in &self.layers {
            h = l.forward(&h)?;
        }
        if self.stable {
            h = self.encoder_norm.forward(&h)?;
        }
        Ok(self.lm_head.forward(&h)?.to_dtype(DType::F32)?)
    }

    /// Intermediate tensors for parity checks (batch 1): `conv` [C, T],
    /// `proj`, `pos`, `hs{i}` (input of layer i; `hs{N}` the encoder
    /// output) [T, C], `logits`.
    pub fn forward_debug(&self, input: &Tensor) -> Result<Vec<(String, Tensor)>> {
        let mut out = Vec::new();
        let x = self.features(input)?;
        out.push(("conv".to_string(), x.squeeze(0)?.t()?));
        let x = self.proj.forward(&self.proj_norm.forward(&x)?)?;
        out.push(("proj".to_string(), x.squeeze(0)?));
        let pos = self.pos_embed(&x)?;
        out.push(("pos".to_string(), pos.squeeze(0)?));
        let mut h = (&x + pos)?;
        if !self.stable {
            h = self.encoder_norm.forward(&h)?;
        }
        for (i, l) in self.layers.iter().enumerate() {
            out.push((format!("hs{i}"), h.squeeze(0)?));
            h = l.forward(&h)?;
        }
        if self.stable {
            h = self.encoder_norm.forward(&h)?;
        }
        out.push((format!("hs{}", self.layers.len()), h.squeeze(0)?));
        out.push(("logits".to_string(), self.lm_head.forward(&h)?.squeeze(0)?));
        Ok(out)
    }

    fn pos_embed(&self, x: &Tensor) -> Result<Tensor> {
        let t = x.dim(1)?;
        let xt = x.transpose(1, 2)?.contiguous()?;
        let y = xt.conv1d(
            &self.pos_conv_weight,
            self.pos_kernel / 2,
            1,
            1,
            self.pos_groups,
        )?;
        let y = y.broadcast_add(&self.pos_conv_bias.reshape((1, (), 1))?)?;
        // SamePad: an even kernel gives one extra frame
        let y = cpu_ops::gelu_erf(&y.narrow(2, 0, t)?)?;
        Ok(y.transpose(1, 2)?)
    }
}

/// Zero-mean, unit-variance normalization (`Wav2Vec2FeatureExtractor`,
/// population variance, eps 1e-7), computed in f64.
pub fn normalize(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / n;
    let d = (var + 1e-7).sqrt();
    samples
        .iter()
        .map(|&v| ((v as f64 - mean) / d) as f32)
        .collect()
}
