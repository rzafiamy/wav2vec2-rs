//! Parity with the Python reference (scripts/ref/ref.py).
//!
//!     cargo run --release --example parity -- <hf-dir | model.gguf> tmp/ref [--cpu] [--greedy-only]
//!
//! Per clip:
//! 1. decoder parity: the Rust beam search on the *reference* logits must
//!    give pyctcdecode's transcript exactly (and greedy HF's greedy one);
//! 2. model parity: Rust logits vs reference logits (SNR, max |diff|,
//!    argmax agreement), and the Rust end-to-end transcript.

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use serde_json::Value;
use std::path::Path;
use wav2vec2::ctc::{self, Scorer};
use wav2vec2::{Chunking, LmSource, Recognizer, audio};

fn snr(reference: &[f32], got: &[f32]) -> f64 {
    let (mut s, mut n) = (0f64, 0f64);
    for (a, b) in reference.iter().zip(got) {
        s += (*a as f64).powi(2);
        n += (*a as f64 - *b as f64).powi(2);
    }
    10.0 * (s / n.max(1e-30)).log10()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        bail!("usage: parity <hf-dir | model.gguf> <ref-dir> [--cpu] [--dtype f16]");
    }
    let src = Path::new(&args[0]);
    let refdir = Path::new(&args[1]);
    let cpu = args.iter().any(|a| a == "--cpu");
    let dtype = args
        .iter()
        .position(|a| a == "--dtype")
        .and_then(|i| args.get(i + 1))
        .map(|s| match s.as_str() {
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            _ => DType::F32,
        });
    let device = if cpu {
        Device::Cpu
    } else {
        wav2vec2::pick_device(false)?
    };

    let rec = if src.is_dir() {
        let (w, cfg, files) = wav2vec2::gguf::load_hf(src, &device)?;
        let arpa = files
            .arpa
            .as_ref()
            .context("no kenlm_file in decoding_config.json")?;
        let scorer = Scorer::from_arpa(&std::fs::read_to_string(arpa)?)?;
        let model = wav2vec2::model::Wav2Vec2ForCtc::load(&w, &cfg, dtype.unwrap_or(DType::F32))?;
        Recognizer {
            name: "hf".into(),
            cfg,
            model,
            scorer: Some(scorer),
            device: device.clone(),
            chunking: Chunking::default(),
        }
    } else {
        Recognizer::open(src, &device, dtype, Some(LmSource::Embedded))?
    };
    let mut rec = rec;
    rec.chunking.chunk_secs = 60.0; // one pass per clip, as the reference
    let opts = rec.default_options();

    let refs: Value = serde_json::from_str(&std::fs::read_to_string(refdir.join("ref.json"))?)?;
    let tensors = candle_core::safetensors::load(refdir.join("ref.safetensors"), &Device::Cpu)?;
    let v = rec.cfg.vocab_size;
    let (mut dec_ok, mut greedy_ok, mut e2e_ok, mut n) = (0, 0, 0, 0);
    let (mut worst_snr, mut argmax_total, mut argmax_same) = (f64::INFINITY, 0usize, 0usize);
    let (mut audio_secs, mut model_secs, mut decode_secs) = (0f64, 0f64, 0f64);
    for clip in refs.as_array().context("ref.json")? {
        let key = clip["key"].as_str().unwrap();
        let path = clip["path"].as_str().unwrap();
        let want_beam = clip["beam"].as_str().unwrap();
        let want_greedy = clip["greedy"].as_str().unwrap();
        let ref_logits = tensors[key]
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        n += 1;

        // 1. decoder parity on the reference logits
        let t = std::time::Instant::now();
        let d = ctc::beam_search(&ref_logits, &rec.cfg.vocab, rec.scorer.as_ref(), &opts.beam);
        let beam = d.text.to_lowercase();
        decode_secs += t.elapsed().as_secs_f64();
        let g = ctc::greedy(&ref_logits, &rec.cfg.vocab).text.to_lowercase();
        let g = g.replace(wav2vec2::config::UNK_LABEL, "<unk>");
        if beam == want_beam {
            dec_ok += 1;
        } else {
            println!("{key} DECODER MISMATCH\n  want {want_beam}\n  got  {beam}");
        }
        if g == want_greedy {
            greedy_ok += 1;
        } else {
            println!("{key} GREEDY MISMATCH\n  want {want_greedy}\n  got  {g}");
        }

        // 2. model parity
        let bytes = std::fs::read(path)?;
        let ext = Path::new(path)
            .extension()
            .map(|e| e.to_string_lossy().to_string());
        let wav = audio::load(bytes, ext.as_deref(), rec.cfg.sample_rate)?;
        audio_secs += wav.len() as f64 / rec.cfg.sample_rate as f64;
        let t = std::time::Instant::now();
        let (logits, frames) = rec.logits(&wav)?;
        model_secs += t.elapsed().as_secs_f64();
        if logits.len() != ref_logits.len() {
            println!("{key}: {frames} frames, reference {}", ref_logits.len() / v);
            continue;
        }
        let s = snr(&ref_logits, &logits);
        let maxd = ref_logits
            .iter()
            .zip(&logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        worst_snr = worst_snr.min(s);
        for (a, b) in ref_logits.chunks_exact(v).zip(logits.chunks_exact(v)) {
            let am = |r: &[f32]| (0..v).max_by(|&i, &j| r[i].total_cmp(&r[j])).unwrap();
            argmax_total += 1;
            argmax_same += (am(a) == am(b)) as usize;
        }
        let e2e = rec.decode(&logits, frames, &opts).text.to_lowercase();
        if e2e == want_beam {
            e2e_ok += 1;
        }
        println!(
            "{key}: SNR {s:.1} dB, max|d| {maxd:.2e}, e2e {}",
            if e2e == want_beam {
                "same"
            } else {
                "DIFFERENT"
            }
        );
        if e2e != want_beam {
            println!("  want {want_beam}\n  got  {e2e}");
        }
    }
    println!(
        "\ndecoder parity {dec_ok}/{n}, greedy parity {greedy_ok}/{n}, end-to-end {e2e_ok}/{n}; \
         worst logit SNR {worst_snr:.1} dB, argmax agreement {:.4}%",
        100.0 * argmax_same as f64 / argmax_total.max(1) as f64
    );
    println!(
        "model {:.1}x realtime, beam search {:.0}x realtime ({audio_secs:.0}s of audio)",
        audio_secs / model_secs,
        audio_secs / decode_secs
    );
    Ok(())
}
