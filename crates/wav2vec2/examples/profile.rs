//! Times the parts of one forward pass:  profile <model.gguf> <seconds> [--cpu]
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dev = wav2vec2::pick_device(a.iter().any(|x| x == "--cpu"))?;
    let secs: f64 = a[1].parse()?;
    let rec = wav2vec2::Recognizer::open(
        a[0].as_ref(),
        &dev,
        None,
        Some(wav2vec2::LmSource::Disabled),
    )?;
    let n = (secs * 16000.0) as usize;
    let wav: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() * 0.1).collect();
    let x = Tensor::from_slice(&wav, (1, n), &dev)?;
    for _ in 0..3 {
        let t = Instant::now();
        let parts = rec.model.forward_debug(&x)?;
        let _ = parts
            .last()
            .unwrap()
            .1
            .to_dtype(DType::F32)?
            .sum_all()?
            .to_scalar::<f32>()?;
        println!("forward_debug {:.3}s", t.elapsed().as_secs_f64());
        let t = Instant::now();
        let y = rec.model.forward(&x)?;
        let _ = y.sum_all()?.to_scalar::<f32>()?;
        println!(
            "forward {:.3}s ({:.1}x realtime)",
            t.elapsed().as_secs_f64(),
            secs / t.elapsed().as_secs_f64()
        );
    }
    let _ = Device::Cpu;
    Ok(())
}
