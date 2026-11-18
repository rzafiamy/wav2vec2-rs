//! Speech to text: waveform → (chunked) CTC logits → greedy or beam search
//! decoding → words with timestamps.
//!
//! The model attends over the whole input, so memory grows with the square
//! of its length. Audio longer than `chunk_secs` is cut into windows on
//! frame boundaries (multiples of the 320-sample hop, so window frames are
//! global frames); each window adds `context_secs` of audio on both sides,
//! and only its central frames are kept. The waveform is normalized once,
//! over the whole input, as the reference does.

use crate::config::Config;
use crate::ctc::{self, BeamParams, Decoded, Scorer};
use crate::model::{Wav2Vec2ForCtc, normalize};
use anyhow::{Result, bail};
use candle_core::{Device, Tensor};

#[derive(Debug, Clone, Copy)]
pub struct Chunking {
    pub chunk_secs: f64,
    pub context_secs: f64,
}

impl Default for Chunking {
    fn default() -> Self {
        Self {
            chunk_secs: 30.0,
            context_secs: 3.0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    /// Argmax decoding, no beam search, no language model.
    pub greedy: bool,
    pub beam: BeamParams,
    pub lowercase: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TimedWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<TimedWord>,
    /// Seconds of audio.
    pub duration: f64,
}

pub struct Recognizer {
    /// `general.name` of the GGUF.
    pub name: String,
    pub cfg: Config,
    pub model: Wav2Vec2ForCtc,
    pub scorer: Option<Scorer>,
    pub device: Device,
    pub chunking: Chunking,
}

impl Recognizer {
    pub fn default_options(&self) -> DecodeOptions {
        let d = &self.cfg.decoder;
        DecodeOptions {
            greedy: false,
            beam: BeamParams {
                beam_width: d.beam_width,
                alpha: d.alpha,
                beta: d.beta,
                ..BeamParams::default()
            },
            lowercase: d.lowercase,
        }
    }

    fn run(&self, samples: &[f32]) -> Result<Vec<f32>> {
        let x = Tensor::from_slice(samples, (1, samples.len()), &self.device)?;
        let logits = self.model.forward(&x)?.squeeze(0)?;
        Ok(logits.flatten_all()?.to_vec1::<f32>()?)
    }

    /// CTC logits `[T, vocab]` (row-major) for 16 kHz mono samples.
    pub fn logits(&self, samples: &[f32]) -> Result<(Vec<f32>, usize)> {
        let cfg = &self.cfg;
        let v = cfg.vocab_size;
        let rf = cfg.receptive_field();
        let mut wav = if cfg.do_normalize {
            normalize(samples)
        } else {
            samples.to_vec()
        };
        if wav.len() < rf {
            // shorter than one frame: pad with silence to one frame
            wav.resize(rf, 0.0);
        }
        let total = cfg.frames_for(wav.len());
        let fps = 1.0 / cfg.frame_secs();
        let chunk = ((self.chunking.chunk_secs * fps).round() as usize).max(1);
        let ctx = (self.chunking.context_secs * fps).round() as usize;
        if total <= chunk || chunk <= 2 * ctx {
            let out = self.run(&wav)?;
            let t = out.len() / v;
            return Ok((out, t));
        }
        let hop = cfg.hop();
        let step = chunk - 2 * ctx;
        let mut out = Vec::with_capacity(total * v);
        let mut core = 0;
        while core < total {
            let core_end = (core + step).min(total);
            let w0 = core.saturating_sub(ctx);
            let w1 = (core_end + ctx).min(total);
            let s0 = w0 * hop;
            let s1 = ((w1 - 1) * hop + rf).min(wav.len());
            let l = self.run(&wav[s0..s1])?;
            let t = l.len() / v;
            let a = core - w0;
            let b = (core_end - w0).min(t);
            if a >= b {
                bail!("chunking produced no frames at {core}");
            }
            out.extend_from_slice(&l[a * v..b * v]);
            core = core_end;
        }
        Ok((out, total))
    }

    /// Decodes logits from [`Recognizer::logits`].
    pub fn decode(&self, logits: &[f32], frames: usize, opts: &DecodeOptions) -> Decoded {
        let _ = frames;
        if opts.greedy {
            ctc::greedy(logits, &self.cfg.vocab)
        } else {
            ctc::beam_search(logits, &self.cfg.vocab, self.scorer.as_ref(), &opts.beam)
        }
    }

    pub fn transcribe(&self, samples: &[f32], opts: &DecodeOptions) -> Result<Transcript> {
        let duration = samples.len() as f64 / self.cfg.sample_rate as f64;
        let (logits, frames) = self.logits(samples)?;
        let d = self.decode(&logits, frames, opts);
        Ok(self.timed(d, duration, opts.lowercase))
    }

    pub fn timed(&self, d: Decoded, duration: f64, lowercase: bool) -> Transcript {
        let fs = self.cfg.frame_secs();
        let lc = |s: String| if lowercase { s.to_lowercase() } else { s };
        let words = d
            .words
            .into_iter()
            .map(|w| TimedWord {
                text: lc(w.text),
                start: w.start as f64 * fs,
                end: (w.end as f64 * fs).min(duration.max(0.0)),
            })
            .collect();
        Transcript {
            text: lc(d.text),
            words,
            duration,
        }
    }
}
