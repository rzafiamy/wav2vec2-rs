//! Model configuration: the subset of a Hugging Face `Wav2Vec2Config` the
//! CTC forward pass needs, the CTC vocabulary, and the decoder defaults.
//! Stored as JSON in the GGUF (`wav2vec2.config`).

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub conv_bias: bool,
    /// "layer" (every conv layer has a LayerNorm) or "group" (GroupNorm on
    /// the first layer only, wav2vec2-base style).
    pub feat_extract_norm: String,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    /// Pre-LN transformer (XLS-R, large-lv60) instead of post-LN (base).
    pub do_stable_layer_norm: bool,
    pub layer_norm_eps: f64,
    pub vocab_size: usize,
    pub sample_rate: u32,
    /// Zero-mean, unit-variance waveform normalization.
    pub do_normalize: bool,
    pub vocab: Vocab,
    pub decoder: DecoderDefaults,
}

/// CTC labels, by token id, as the decoder sees them: the word delimiter
/// is " ", the blank is "", `<unk>` is "⁇" (pyctcdecode's normalization).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Vocab {
    pub labels: Vec<String>,
    pub blank_id: usize,
    pub word_delimiter_id: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DecoderDefaults {
    pub alpha: f64,
    pub beta: f64,
    pub beam_width: usize,
    /// Lowercase the transcript (the reference transcribe.py does).
    pub lowercase: bool,
}

impl Default for DecoderDefaults {
    fn default() -> Self {
        // pyctcdecode's defaults
        Self {
            alpha: 0.5,
            beta: 1.5,
            beam_width: 100,
            lowercase: false,
        }
    }
}

fn usize_of(v: &Value, k: &str) -> Result<usize> {
    v.get(k)
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .with_context(|| format!("config.json: missing integer '{k}'"))
}

fn vec_of(v: &Value, k: &str) -> Result<Vec<usize>> {
    v.get(k)
        .and_then(Value::as_array)
        .with_context(|| format!("config.json: missing array '{k}'"))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as usize).context(k.to_string()))
        .collect()
}

pub const UNK_LABEL: &str = "⁇";

impl Vocab {
    /// From a Hugging Face `vocab.json` (+ `added_tokens.json` entries),
    /// with the tokenizer's special tokens.
    pub fn from_hf(
        vocab: &Value,
        added: Option<&Value>,
        pad: &str,
        unk: &str,
        word_delimiter: &str,
        size: usize,
    ) -> Result<Self> {
        let mut by_id: Vec<Option<String>> = vec![None; size];
        let mut put = |map: &Value| -> Result<()> {
            for (tok, id) in map.as_object().context("vocab is not a JSON object")? {
                let id = id.as_u64().context("vocab id is not an integer")? as usize;
                if id >= size {
                    bail!("token '{tok}' has id {id} >= vocab_size {size}");
                }
                by_id[id] = Some(tok.clone());
            }
            Ok(())
        };
        put(vocab)?;
        if let Some(a) = added {
            put(a)?;
        }
        let mut labels = Vec::with_capacity(size);
        let (mut blank_id, mut word_delimiter_id) = (None, None);
        for (i, t) in by_id.into_iter().enumerate() {
            let t = t.with_context(|| format!("no token for id {i}"))?;
            let label = if t == word_delimiter {
                word_delimiter_id = Some(i);
                " ".to_string()
            } else if t == pad {
                blank_id = Some(i);
                String::new()
            } else if t == unk {
                UNK_LABEL.to_string()
            } else {
                t
            };
            labels.push(label);
        }
        let mut seen = std::collections::HashSet::new();
        for l in &labels {
            if !seen.insert(l) {
                bail!("duplicate label '{l}' in the vocabulary");
            }
        }
        Ok(Self {
            labels,
            blank_id: blank_id.context("no pad (CTC blank) token in the vocabulary")?,
            word_delimiter_id,
        })
    }
}

impl Config {
    /// From the Hugging Face files of a `Wav2Vec2ForCTC` model.
    pub fn from_hf(
        config: &Value,
        preprocessor: Option<&Value>,
        vocab: Vocab,
        decoder: DecoderDefaults,
    ) -> Result<Self> {
        let arch = config
            .get("architectures")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .unwrap_or("");
        if !arch.is_empty() && arch != "Wav2Vec2ForCTC" {
            bail!("unsupported architecture '{arch}' (Wav2Vec2ForCTC only)");
        }
        let s = |k: &str, d: &str| {
            config
                .get(k)
                .and_then(Value::as_str)
                .unwrap_or(d)
                .to_string()
        };
        for (k, want) in [("feat_extract_activation", "gelu"), ("hidden_act", "gelu")] {
            let got = s(k, "gelu");
            if got != want {
                bail!("unsupported {k} '{got}' (gelu only)");
            }
        }
        if config.get("add_adapter").and_then(Value::as_bool) == Some(true) {
            bail!("add_adapter models are not supported");
        }
        if config.get("adapter_attn_dim").is_some_and(|v| !v.is_null()) {
            bail!("MMS attention adapters are not supported yet");
        }
        let pre = |k: &str| preprocessor.and_then(|p| p.get(k));
        let cfg = Self {
            hidden_size: usize_of(config, "hidden_size")?,
            num_hidden_layers: usize_of(config, "num_hidden_layers")?,
            num_attention_heads: usize_of(config, "num_attention_heads")?,
            intermediate_size: usize_of(config, "intermediate_size")?,
            conv_dim: vec_of(config, "conv_dim")?,
            conv_kernel: vec_of(config, "conv_kernel")?,
            conv_stride: vec_of(config, "conv_stride")?,
            conv_bias: config
                .get("conv_bias")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            feat_extract_norm: s("feat_extract_norm", "group"),
            num_conv_pos_embeddings: usize_of(config, "num_conv_pos_embeddings")?,
            num_conv_pos_embedding_groups: usize_of(config, "num_conv_pos_embedding_groups")?,
            do_stable_layer_norm: config
                .get("do_stable_layer_norm")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            layer_norm_eps: config
                .get("layer_norm_eps")
                .and_then(Value::as_f64)
                .unwrap_or(1e-5),
            vocab_size: usize_of(config, "vocab_size")?,
            sample_rate: pre("sampling_rate")
                .and_then(Value::as_u64)
                .unwrap_or(16_000) as u32,
            do_normalize: pre("do_normalize").and_then(Value::as_bool).unwrap_or(true),
            vocab,
            decoder,
        };
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn from_json(s: &str) -> Result<Self> {
        let cfg: Self = serde_json::from_str(s).context("wav2vec2.config")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        let n = self.conv_dim.len();
        if n == 0 || self.conv_kernel.len() != n || self.conv_stride.len() != n {
            bail!("conv_dim, conv_kernel and conv_stride must have the same length");
        }
        if !matches!(self.feat_extract_norm.as_str(), "layer" | "group") {
            bail!("unknown feat_extract_norm '{}'", self.feat_extract_norm);
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            bail!("hidden_size is not a multiple of num_attention_heads");
        }
        if self.vocab.labels.len() != self.vocab_size {
            bail!(
                "vocabulary has {} labels, model vocab_size is {}",
                self.vocab.labels.len(),
                self.vocab_size
            );
        }
        if self.vocab.blank_id >= self.vocab_size {
            bail!("blank id out of range");
        }
        Ok(())
    }

    /// Samples per output frame (320 for wav2vec2: 20 ms at 16 kHz).
    pub fn hop(&self) -> usize {
        self.conv_stride.iter().product()
    }

    /// Samples one output frame sees (400 for wav2vec2).
    pub fn receptive_field(&self) -> usize {
        let mut rf = 1;
        let mut jump = 1;
        for (k, s) in self.conv_kernel.iter().zip(&self.conv_stride) {
            rf += (k - 1) * jump;
            jump *= s;
        }
        rf
    }

    /// Output frames for `n` input samples.
    pub fn frames_for(&self, n: usize) -> usize {
        let mut len = n;
        for (k, s) in self.conv_kernel.iter().zip(&self.conv_stride) {
            if len < *k {
                return 0;
            }
            len = (len - k) / s + 1;
        }
        len
    }

    /// Seconds per output frame.
    pub fn frame_secs(&self) -> f64 {
        self.hop() as f64 / self.sample_rate as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn xlsr() -> Config {
        Config {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            conv_dim: vec![512; 7],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            conv_bias: true,
            feat_extract_norm: "layer".into(),
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
            do_stable_layer_norm: true,
            layer_norm_eps: 1e-5,
            vocab_size: 3,
            sample_rate: 16_000,
            do_normalize: true,
            vocab: Vocab {
                labels: vec!["".into(), " ".into(), "a".into()],
                blank_id: 0,
                word_delimiter_id: Some(1),
            },
            decoder: DecoderDefaults::default(),
        }
    }

    #[test]
    fn geometry() {
        let c = xlsr();
        assert_eq!(c.hop(), 320);
        assert_eq!(c.receptive_field(), 400);
        assert_eq!(c.frames_for(16_000), 49);
        assert_eq!(c.frames_for(400), 1);
        assert_eq!(c.frames_for(399), 0);
        // frame j starts at sample 320 j: 400 + 320 (F - 1) samples give F frames
        for f in 1..50 {
            assert_eq!(c.frames_for(400 + 320 * (f - 1)), f);
            assert_eq!(c.frames_for(400 + 320 * (f - 1) + 319), f);
        }
    }

    #[test]
    fn vocab_labels() {
        let v: Value = serde_json::json!({"a": 0, "|": 1, "<unk>": 2, "<pad>": 3});
        let added: Value = serde_json::json!({"<s>": 4, "</s>": 5});
        let voc = Vocab::from_hf(&v, Some(&added), "<pad>", "<unk>", "|", 6).unwrap();
        assert_eq!(voc.labels, ["a", " ", UNK_LABEL, "", "<s>", "</s>"]);
        assert_eq!(voc.blank_id, 3);
        assert_eq!(voc.word_delimiter_id, Some(1));
    }
}
