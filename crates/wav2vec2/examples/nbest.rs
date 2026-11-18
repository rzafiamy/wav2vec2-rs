//! Final beams for one clip of a parity reference:
//!     cargo run --release --example nbest -- <hf-dir> tmp/ref clip018 [n]
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use wav2vec2::ctc::{self, BeamParams, Scorer};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let files = wav2vec2::gguf::HfFiles::find(a[0].as_ref())?;
    let cfg = files.config()?;
    let scorer = Scorer::from_arpa(&std::fs::read_to_string(files.arpa.context("no LM")?)?)?;
    let t = candle_core::safetensors::load(format!("{}/ref.safetensors", a[1]), &Device::Cpu)?;
    let l = t[&a[2]]
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let n = a.get(3).and_then(|s| s.parse().ok()).unwrap_or(5);
    let p = BeamParams {
        beam_width: cfg.decoder.beam_width,
        alpha: cfg.decoder.alpha,
        beta: cfg.decoder.beta,
        prune_history: std::env::var("NO_PRUNE_HISTORY").is_err(),
        ..Default::default()
    };
    for h in ctc::beam_search_nbest(&l, &cfg.vocab, Some(&scorer), &p, n) {
        let t = &h.decoded.text;
        println!(
            "{:.4} {:.4} {}",
            h.score,
            h.logit_score,
            &t[t.len().saturating_sub(60)..]
        );
    }
    Ok(())
}
