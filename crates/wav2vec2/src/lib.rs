//! wav2vec 2.0 / XLS-R CTC speech recognition in Rust/Candle: one GGUF holds
//! the network, its CTC vocabulary, decoder settings and (optionally) an
//! n-gram language model; decoding is greedy or pyctcdecode-compatible beam
//! search with the LM.
//!
//! ```no_run
//! # fn main() -> anyhow::Result<()> {
//! let device = candle_core::Device::Cpu;
//! let rec = wav2vec2::Recognizer::open("models/xlsr-mg-norm-v2-f16.gguf".as_ref(), &device, None, None)?;
//! let wav = wav2vec2::audio::load(std::fs::read("speech.mp3")?, Some("mp3"), rec.cfg.sample_rate)?;
//! let t = rec.transcribe(&wav, &rec.default_options())?;
//! println!("{}", t.text);
//! # Ok(()) }
//! ```

pub mod audio;
pub mod config;
pub mod cpu_ops;
pub mod ctc;
pub mod gguf;
pub mod lm;
pub mod model;
#[doc(hidden)]
pub mod testing;
pub mod transcribe;
pub mod weights;

pub use config::Config;
pub use transcribe::{Chunking, DecodeOptions, Recognizer, TimedWord, Transcript};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use std::path::Path;

/// Which language model to use.
pub enum LmSource<'a> {
    /// The one embedded in the GGUF, if any.
    Embedded,
    /// An ARPA file (replaces the embedded one).
    Arpa(&'a Path),
    /// None: beam search without LM.
    Disabled,
}

impl Recognizer {
    /// Opens a wav2vec2 GGUF. `dtype` defaults to F16 on a GPU, F32 on the
    /// CPU; `lm` defaults to the embedded language model.
    pub fn open(
        path: &Path,
        device: &Device,
        dtype: Option<DType>,
        lm: Option<LmSource>,
    ) -> Result<Self> {
        let loaded = gguf::load(path, device)?;
        let dtype = dtype.unwrap_or(if device.is_cpu() {
            DType::F32
        } else {
            DType::F16
        });
        let model = model::Wav2Vec2ForCtc::load(&loaded.weights, &loaded.config, dtype)?;
        let scorer = match lm.unwrap_or(LmSource::Embedded) {
            LmSource::Embedded => loaded
                .arpa
                .as_deref()
                .map(ctc::Scorer::from_arpa)
                .transpose()
                .context("embedded language model")?,
            LmSource::Arpa(p) => {
                let text = std::fs::read_to_string(p).with_context(|| format!("reading {p:?}"))?;
                Some(ctc::Scorer::from_arpa(&text).with_context(|| format!("parsing {p:?}"))?)
            }
            LmSource::Disabled => None,
        };
        Ok(Self {
            name: loaded.name,
            cfg: loaded.config,
            model,
            scorer,
            device: device.clone(),
            chunking: Chunking::default(),
        })
    }
}

/// The device for `cpu` = false: CUDA (or Metal) when built in, else the CPU.
pub fn pick_device(cpu: bool) -> Result<Device> {
    if cpu {
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "cuda")]
    {
        return Ok(Device::new_cuda(0)?);
    }
    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        return Ok(Device::new_metal(0)?);
    }
    #[allow(unreachable_code)]
    Ok(Device::Cpu)
}
