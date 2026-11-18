//! Single-file GGUF models, converted from a Hugging Face `Wav2Vec2ForCTC`
//! directory.
//!
//! GGUF layout:
//! - metadata `general.architecture = "wav2vec2"`, `general.name`,
//!   `wav2vec2.config` (the [`Config`] as JSON: hyperparameters, CTC labels,
//!   decoder defaults), and optionally `wav2vec2.lm.arpa`: an ARPA n-gram
//!   language model, stored as text so the file is the whole recognizer;
//! - the inference tensors under their Hugging Face names (pretraining-only
//!   tensors dropped), with the positional convolution's weight norm folded
//!   into one `wav2vec2.encoder.pos_conv_embed.conv.weight`. 2-D linear
//!   weights use the requested dtype when their rows fit its block (else
//!   Q8_0, F16 or F32); conv kernels are F16 unless F32 is asked for (the
//!   first, 10-sample one stays F32); norms and biases stay F32.

use crate::config::{Config, DecoderDefaults, Vocab};
use crate::weights::Weights;
use anyhow::{Context, Result, bail};
use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
use candle_core::{DType, Device, Tensor};
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const ARCH: &str = "wav2vec2";
pub const KEY_CONFIG: &str = "wav2vec2.config";
pub const KEY_ARPA: &str = "wav2vec2.lm.arpa";

/// Parses a dtype name: f32, f16, bf16, q8_0.
pub fn parse_dtype(name: &str) -> Result<GgmlDType> {
    Ok(match name.to_ascii_lowercase().as_str() {
        "f32" => GgmlDType::F32,
        "f16" => GgmlDType::F16,
        "bf16" => GgmlDType::BF16,
        "q8_0" | "q8" => GgmlDType::Q8_0,
        other => bail!("unknown dtype '{other}' (f32, f16, bf16, q8_0)"),
    })
}

fn fit_dtype(wanted: GgmlDType, row: usize) -> GgmlDType {
    for dt in [wanted, GgmlDType::Q8_0, GgmlDType::F16] {
        if row.is_multiple_of(dt.block_size()) {
            return dt;
        }
    }
    GgmlDType::F32
}

/// The files of a Hugging Face model, found from a repository root (with a
/// `decoding_config.json` naming its subfolders, as in
/// malagasy-asr/xlsr-mg-norm-v2-kenlm) or from a plain model directory.
#[derive(Debug, Default)]
pub struct HfFiles {
    pub config: PathBuf,
    pub weights: PathBuf,
    pub vocab: PathBuf,
    pub added_tokens: Option<PathBuf>,
    pub tokenizer_config: Option<PathBuf>,
    pub preprocessor: Option<PathBuf>,
    pub decoding: Option<Value>,
    /// ARPA language model named by `decoding_config.json`.
    pub arpa: Option<PathBuf>,
}

fn first_existing(cands: &[PathBuf]) -> Option<PathBuf> {
    cands.iter().find(|p| p.is_file()).cloned()
}

impl HfFiles {
    pub fn find(root: &Path) -> Result<Self> {
        let decoding: Option<Value> =
            match std::fs::read_to_string(root.join("decoding_config.json")) {
                Ok(s) => Some(serde_json::from_str(&s).context("decoding_config.json")?),
                Err(_) => None,
            };
        let sub = |k: &str, d: &str| -> PathBuf {
            let name = decoding
                .as_ref()
                .and_then(|v| v.get(k))
                .and_then(Value::as_str)
                .unwrap_or(d);
            root.join(name)
        };
        let model_dir = sub("model_subfolder", "checkpoint");
        let proc_dir = sub("processor_subfolder", "processor");
        let dirs = [root.to_path_buf(), model_dir, proc_dir];
        let find =
            |name: &str| first_existing(&dirs.iter().map(|d| d.join(name)).collect::<Vec<_>>());
        let config =
            find("config.json").with_context(|| format!("no config.json under {root:?}"))?;
        let weights = find("model.safetensors")
            .or_else(|| find("pytorch_model.bin"))
            .with_context(|| format!("no model.safetensors or pytorch_model.bin under {root:?}"))?;
        let vocab = find("vocab.json").with_context(|| format!("no vocab.json under {root:?}"))?;
        let arpa = decoding
            .as_ref()
            .and_then(|v| v.get("kenlm_file"))
            .and_then(Value::as_str)
            .map(|f| root.join(f))
            .filter(|p| p.is_file());
        Ok(Self {
            config,
            weights,
            vocab,
            added_tokens: find("added_tokens.json"),
            tokenizer_config: find("tokenizer_config.json"),
            preprocessor: find("preprocessor_config.json"),
            decoding,
            arpa,
        })
    }

    fn json(p: &Path) -> Result<Value> {
        let s = std::fs::read_to_string(p).with_context(|| format!("reading {p:?}"))?;
        serde_json::from_str(&s).with_context(|| format!("parsing {p:?}"))
    }

    pub fn config(&self) -> Result<Config> {
        let config = Self::json(&self.config)?;
        let tok = self
            .tokenizer_config
            .as_deref()
            .map(Self::json)
            .transpose()?;
        let tok_str = |k: &str, d: &str| -> String {
            tok.as_ref()
                .and_then(|t| t.get(k))
                .and_then(|v| {
                    v.as_str()
                        .or_else(|| v.get("content").and_then(Value::as_str))
                })
                .unwrap_or(d)
                .to_string()
        };
        let size = config
            .get("vocab_size")
            .and_then(Value::as_u64)
            .context("config.json: no vocab_size")? as usize;
        let vocab = Vocab::from_hf(
            &Self::json(&self.vocab)?,
            self.added_tokens
                .as_deref()
                .map(Self::json)
                .transpose()?
                .as_ref(),
            &tok_str("pad_token", "<pad>"),
            &tok_str("unk_token", "<unk>"),
            &tok_str("word_delimiter_token", "|"),
            size,
        )?;
        let mut dec = DecoderDefaults::default();
        if let Some(d) = &self.decoding {
            if let Some(v) = d.get("alpha").and_then(Value::as_f64) {
                dec.alpha = v;
            }
            if let Some(v) = d.get("beta").and_then(Value::as_f64) {
                dec.beta = v;
            }
            if let Some(v) = d.get("beam_width").and_then(Value::as_u64) {
                dec.beam_width = v as usize;
            }
            // the reference decoding lowercases its output
            dec.lowercase = true;
        }
        let pre = self.preprocessor.as_deref().map(Self::json).transpose()?;
        Config::from_hf(&config, pre.as_ref(), vocab, dec)
    }
}

/// Inference tensors of a checkpoint, F32 on the CPU, weight norm folded.
pub fn read_weights(path: &Path) -> Result<Vec<(String, Tensor)>> {
    let raw: Vec<(String, Tensor)> = if path.extension().is_some_and(|e| e == "safetensors") {
        candle_core::safetensors::load(path, &Device::Cpu)?
            .into_iter()
            .collect()
    } else {
        candle_core::pickle::read_all(path).with_context(|| format!("reading {path:?}"))?
    };
    let mut map: HashMap<String, Tensor> = HashMap::new();
    for (n, t) in raw {
        let keep = n.starts_with("wav2vec2.feature_extractor.")
            || n.starts_with("wav2vec2.feature_projection.")
            || n.starts_with("wav2vec2.encoder.")
            || n.starts_with("lm_head.");
        if keep {
            map.insert(n, t.to_dtype(DType::F32)?);
        }
    }
    if !map.contains_key("lm_head.weight") {
        bail!("{path:?} is not a Wav2Vec2ForCTC checkpoint (no lm_head.weight)");
    }
    // weight norm (dim = 2): w = g * v / ||v||, the norm over dims 0 and 1
    let p = "wav2vec2.encoder.pos_conv_embed.conv";
    let pairs = [
        (
            format!("{p}.parametrizations.weight.original0"),
            format!("{p}.parametrizations.weight.original1"),
        ),
        (format!("{p}.weight_g"), format!("{p}.weight_v")),
    ];
    for (gk, vk) in pairs {
        if let (Some(g), Some(v)) = (map.remove(&gk), map.remove(&vk)) {
            let norm = v.sqr()?.sum_keepdim(0)?.sum_keepdim(1)?.sqrt()?;
            let w = v.broadcast_mul(&g.broadcast_div(&norm)?)?;
            map.insert(format!("{p}.weight"), w);
        }
    }
    if !map.contains_key(&format!("{p}.weight")) {
        bail!("no positional convolution weight in {path:?}");
    }
    let mut v: Vec<(String, Tensor)> = map.into_iter().collect();
    v.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(v)
}

fn to_map(tensors: &[(String, Tensor)]) -> Result<HashMap<String, Arc<QTensor>>> {
    tensors
        .iter()
        .map(|(n, t)| Ok((n.clone(), Arc::new(QTensor::quantize(t, GgmlDType::F32)?))))
        .collect()
}

/// Loads a Hugging Face model directory directly (F32; parity checks).
pub fn load_hf(root: &Path, device: &Device) -> Result<(Weights, Config, HfFiles)> {
    let files = HfFiles::find(root)?;
    let cfg = files.config()?;
    let map = to_map(&read_weights(&files.weights)?)?;
    Ok((Weights::new(map, device.clone()), cfg, files))
}

pub struct ConvertReport {
    pub counts: Vec<(String, usize)>,
    pub lm_bytes: usize,
}

/// Converts a Hugging Face model directory to one GGUF. `lm`: an ARPA file
/// to embed (default: the one `decoding_config.json` names, unless
/// `no_lm`).
pub fn convert(
    root: &Path,
    out: &Path,
    dtype: GgmlDType,
    name: &str,
    lm: Option<&Path>,
    no_lm: bool,
) -> Result<ConvertReport> {
    let files = HfFiles::find(root)?;
    let cfg = files.config()?;
    let tensors = read_weights(&files.weights)?;
    // build the model once: fails here, not at load, on a mismatch
    {
        let w = Weights::new(to_map(&tensors)?, Device::Cpu);
        crate::model::Wav2Vec2ForCtc::load(&w, &cfg, DType::F32)
            .context("checkpoint does not match its config")?;
    }
    let arpa = if no_lm {
        None
    } else {
        lm.map(Path::to_path_buf).or(files.arpa.clone())
    };
    let arpa_text = match &arpa {
        Some(p) => {
            let text = std::fs::read_to_string(p).with_context(|| format!("reading {p:?}"))?;
            // fail now on a malformed file
            crate::lm::NgramLm::from_arpa(&text).with_context(|| format!("parsing {p:?}"))?;
            Some(text)
        }
        None => None,
    };

    let mut q = Vec::new();
    let mut counts: HashMap<String, usize> = HashMap::new();
    let conv_dt = if dtype == GgmlDType::F32 {
        GgmlDType::F32
    } else {
        GgmlDType::F16
    };
    for (n, t) in &tensors {
        let dt = match t.rank() {
            2 if n.ends_with(".weight") => fit_dtype(dtype, t.dim(1)?),
            3 if !n.contains("conv_layers.0.") => conv_dt,
            _ => GgmlDType::F32,
        };
        *counts.entry(format!("{dt:?}")).or_default() += 1;
        q.push((n.clone(), QTensor::quantize(t, dt)?));
    }
    let mut meta = vec![
        (
            "general.architecture",
            gguf_file::Value::String(ARCH.into()),
        ),
        ("general.name", gguf_file::Value::String(name.into())),
        (
            KEY_CONFIG,
            gguf_file::Value::String(serde_json::to_string(&cfg)?),
        ),
    ];
    let lm_bytes = arpa_text.as_ref().map_or(0, String::len);
    if let Some(t) = arpa_text {
        meta.push((KEY_ARPA, gguf_file::Value::String(t)));
    }
    let meta: Vec<(&str, &gguf_file::Value)> = meta.iter().map(|(k, v)| (*k, v)).collect();
    let refs: Vec<(&str, &QTensor)> = q.iter().map(|(n, t)| (n.as_str(), t)).collect();
    let mut file = std::io::BufWriter::new(std::fs::File::create(out)?);
    gguf_file::write(&mut file, &meta, &refs)?;
    let mut counts: Vec<_> = counts.into_iter().collect();
    counts.sort();
    Ok(ConvertReport { counts, lm_bytes })
}

pub struct Loaded {
    pub weights: Weights,
    pub config: Config,
    pub name: String,
    /// The embedded ARPA language model, if any.
    pub arpa: Option<String>,
}

/// Loads a wav2vec2 GGUF.
pub fn load(path: &Path, device: &Device) -> Result<Loaded> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {path:?}"))?;
    let mut content =
        gguf_file::Content::read(&mut file).with_context(|| format!("reading {path:?}"))?;
    let mut meta = |k: &str| -> Option<String> {
        match content.metadata.remove(k) {
            Some(gguf_file::Value::String(s)) => Some(s),
            _ => None,
        }
    };
    if meta("general.architecture").as_deref() != Some(ARCH) {
        bail!("{path:?} is not a {ARCH} GGUF");
    }
    let config = Config::from_json(
        &meta(KEY_CONFIG).with_context(|| format!("{path:?}: no {KEY_CONFIG}"))?,
    )?;
    let name = meta("general.name").unwrap_or_default();
    let arpa = meta(KEY_ARPA);
    let mut map = HashMap::new();
    let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
    for n in names {
        // quantized tensors go straight to the device (used as is); dense
        // ones stay on the CPU until a layer converts them
        let quantized = !matches!(
            content.tensor_infos[&n].ggml_dtype,
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
        );
        let dev = if quantized { device } else { &Device::Cpu };
        map.insert(n.clone(), Arc::new(content.tensor(&mut file, &n, dev)?));
    }
    Ok(Loaded {
        weights: Weights::new(map, device.clone()),
        config,
        name,
        arpa,
    })
}
