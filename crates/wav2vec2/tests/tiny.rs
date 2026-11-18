//! End-to-end checks on a tiny random model (no download).

use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;
use wav2vec2::gguf::{self, parse_dtype};
use wav2vec2::model::Wav2Vec2ForCtc;
use wav2vec2::{LmSource, Recognizer, testing};

fn tmp(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("wav2vec2-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let s: f64 = a.iter().map(|&x| (x as f64).powi(2)).sum();
    let n: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum();
    10.0 * (s / n.max(1e-30)).log10()
}

fn logits(m: &Wav2Vec2ForCtc, wav: &[f32]) -> Vec<f32> {
    let x = Tensor::from_slice(wav, (1, wav.len()), &Device::Cpu).unwrap();
    m.forward(&x)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap()
}

#[test]
fn convert_load_and_paths_agree() {
    for stable in [true, false] {
        let dir = tmp(if stable { "stable" } else { "base" });
        testing::write_tiny_hf(&dir, stable, 7).unwrap();
        let wav = wav2vec2::model::normalize(&testing::test_audio(1.3));

        // straight from the Hugging Face files, both conv paths
        let (w, cfg, _) = gguf::load_hf(&dir, &Device::Cpu).unwrap();
        assert_eq!(cfg.vocab.labels, ["", "⁇", " ", "a", "b", "<s>", "</s>"]);
        assert_eq!(cfg.frames_for(wav.len()), cfg.frames_for(20_800));
        let fast = Wav2Vec2ForCtc::load_opts(&w, &cfg, DType::F32, true).unwrap();
        let slow = Wav2Vec2ForCtc::load_opts(&w, &cfg, DType::F32, false).unwrap();
        let reference = logits(&slow, &wav);
        assert_eq!(reference.len(), cfg.frames_for(wav.len()) * cfg.vocab_size);
        assert!(reference.iter().all(|v| v.is_finite()));
        let s = snr(&reference, &logits(&fast, &wav));
        assert!(s > 100.0, "fast CPU path: SNR {s}");

        // GGUF at each dtype
        for (dt, min_snr) in [("f32", 120.0), ("f16", 40.0), ("q8_0", 25.0)] {
            let out = dir.join(format!("m-{dt}.gguf"));
            let r = gguf::convert(
                &dir,
                &out,
                parse_dtype(dt).unwrap(),
                "tiny",
                Some(&dir.join("lm.arpa")),
                false,
            )
            .unwrap();
            assert_eq!(r.lm_bytes, testing::ARPA.len());
            let l = gguf::load(&out, &Device::Cpu).unwrap();
            assert_eq!(l.config, cfg);
            assert_eq!(l.arpa.as_deref(), Some(testing::ARPA));
            assert!(
                l.weights
                    .contains("wav2vec2.encoder.pos_conv_embed.conv.weight")
            );
            assert!(!l.weights.contains("wav2vec2.masked_spec_embed"));
            let m = Wav2Vec2ForCtc::load(&l.weights, &l.config, DType::F32).unwrap();
            let s = snr(&reference, &logits(&m, &wav));
            assert!(s > min_snr, "{dt}: SNR {s}");
        }
    }
}

#[test]
fn weight_norm_is_folded() {
    let dir = tmp("wn");
    testing::write_tiny_hf(&dir, true, 3).unwrap();
    let raw = candle_core::safetensors::load(dir.join("model.safetensors"), &Device::Cpu).unwrap();
    let p = "wav2vec2.encoder.pos_conv_embed.conv";
    let g = raw[&format!("{p}.parametrizations.weight.original0")]
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let v = &raw[&format!("{p}.parametrizations.weight.original1")];
    let (o, i, k) = v.dims3().unwrap();
    let vv = v.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let folded: std::collections::HashMap<_, _> =
        gguf::read_weights(&dir.join("model.safetensors"))
            .unwrap()
            .into_iter()
            .collect();
    let w = folded[&format!("{p}.weight")]
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    for kk in 0..k {
        let norm: f32 = (0..o * i)
            .map(|j| vv[j * k + kk].powi(2))
            .sum::<f32>()
            .sqrt();
        for j in 0..o * i {
            let want = g[kk] * vv[j * k + kk] / norm;
            assert!((w[j * k + kk] - want).abs() < 1e-5);
        }
    }
}

#[test]
fn recognizer_end_to_end() {
    let dir = tmp("rec");
    testing::write_tiny_hf(&dir, true, 11).unwrap();
    let out = dir.join("m.gguf");
    gguf::convert(
        &dir,
        &out,
        parse_dtype("f16").unwrap(),
        "tiny",
        Some(&dir.join("lm.arpa")),
        false,
    )
    .unwrap();
    let wav = testing::test_audio(2.0);
    let mut rec = Recognizer::open(&out, &Device::Cpu, None, None).unwrap();
    assert!(rec.scorer.is_some());
    assert_eq!(rec.name, "tiny");
    for greedy in [false, true] {
        let mut o = rec.default_options();
        o.greedy = greedy;
        let t = rec.transcribe(&wav, &o).unwrap();
        assert!((t.duration - 2.0).abs() < 1e-9);
        assert_eq!(
            t.text,
            t.words
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        );
        for w in &t.words {
            assert!(w.start <= w.end && w.end <= t.duration + 1e-9, "{w:?}");
            assert!(w.text.chars().all(|c| "ab⁇<>/s".contains(c)), "{w:?}");
        }
    }
    // chunked: same frame count as one pass
    let long = testing::test_audio(7.0);
    let (one, f1) = rec.logits(&long).unwrap();
    rec.chunking.chunk_secs = 2.0;
    rec.chunking.context_secs = 0.5;
    let (chunked, f2) = rec.logits(&long).unwrap();
    assert_eq!(f1, f2);
    assert_eq!(one.len(), chunked.len());
    assert!(chunked.iter().all(|v| v.is_finite()));
    // tiny input: one frame of padding
    let (l, f) = rec.logits(&[0.1; 30]).unwrap();
    assert_eq!((f, l.len()), (1, rec.cfg.vocab_size));
    // without LM
    let rec2 = Recognizer::open(&out, &Device::Cpu, None, Some(LmSource::Disabled)).unwrap();
    assert!(rec2.scorer.is_none());
}
