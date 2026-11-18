//! Compare intermediate tensors with a dump from the reference model:
//!     cargo run --release --example layers -- <hf-dir> dump.safetensors
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let dev = Device::Cpu;
    let (w, cfg, _) = wav2vec2::gguf::load_hf(a[0].as_ref(), &dev)?;
    let m = wav2vec2::model::Wav2Vec2ForCtc::load(&w, &cfg, DType::F32)?;
    let r = candle_core::safetensors::load(&a[1], &dev)?;
    let input = r["input"].unsqueeze(0)?;
    for (name, t) in m.forward_debug(&input)? {
        let Some(want) = r.get(&name) else { continue };
        let t: Tensor = t.to_dtype(DType::F32)?;
        if t.dims() != want.dims() {
            println!("{name}: shape {:?} vs {:?}", t.dims(), want.dims());
            continue;
        }
        let d = (&t - want)?;
        let n = d.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let s = want.sqr()?.sum_all()?.to_scalar::<f32>()?;
        let mx = d.abs()?.max_all()?.to_scalar::<f32>()?;
        println!(
            "{name:7} SNR {:6.1} dB  max|d| {mx:.2e}",
            10.0 * (s / n.max(1e-30)).log10()
        );
    }
    Ok(())
}
