//! Audio in, pure Rust: decoding (WAV, FLAC, MP3, Ogg Vorbis, AAC/ALAC in
//! MP4/M4A) with symphonia, polyphase resampling as scipy's
//! `resample_poly`; WAV writing (tests) with hound.

/// `[channel][sample]`.
pub type Planar = Vec<Vec<f32>>;
use anyhow::{Context, Result, bail};
use std::io::Cursor;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub struct Audio {
    pub sample_rate: u32,
    /// `[channel][sample]`.
    pub data: Planar,
}

impl Audio {
    pub fn duration_secs(&self) -> f64 {
        self.data
            .first()
            .map_or(0.0, |c| c.len() as f64 / self.sample_rate as f64)
    }
}

/// Decodes an in-memory audio file. `ext` (e.g. "mp3") helps the probe.
pub fn decode(bytes: Vec<u8>, ext: Option<&str>) -> Result<Audio> {
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    if let Some(e) = ext {
        hint.with_extension(e);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .context("unsupported or unreadable audio (wav, flac, mp3, ogg, m4a)")?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .context("no audio track")?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .context("unsupported codec")?;
    let mut sample_rate = track.codec_params.sample_rate.unwrap_or(0);
    let mut data: Planar = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(e).context("reading audio"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // a corrupt frame: skip it, as ffmpeg does
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(e).context("decoding audio"),
        };
        let spec = *decoded.spec();
        sample_rate = spec.rate;
        let ch = spec.channels.count();
        if data.is_empty() {
            data = vec![Vec::new(); ch];
        }
        let b = buf.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if b.capacity() < decoded.capacity() * ch {
            *b = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        b.copy_interleaved_ref(decoded);
        for frame in b.samples().chunks_exact(ch) {
            for (c, v) in frame.iter().enumerate() {
                data[c].push(*v);
            }
        }
    }
    if data.is_empty() || data[0].is_empty() || sample_rate == 0 {
        bail!("no audio decoded");
    }
    Ok(Audio { sample_rate, data })
}

/// Mono (channels averaged, as the reference transcribe.py) at `rate` Hz.
pub fn to_mono(audio: Audio, rate: u32) -> Result<Vec<f32>> {
    let mut data = audio.data;
    if data.len() > 1 {
        let n = data[0].len();
        let k = data.len() as f64;
        let mono = (0..n)
            .map(|i| (data.iter().map(|c| c[i] as f64).sum::<f64>() / k) as f32)
            .collect();
        data = vec![mono];
    }
    if audio.sample_rate != rate {
        data = resample(&data, audio.sample_rate, rate)?;
    }
    let mono = data.pop().unwrap_or_default();
    if mono.iter().any(|v| !v.is_finite()) {
        bail!("non-finite audio samples");
    }
    Ok(mono)
}

/// Decodes and converts to mono at `rate` Hz.
pub fn load(bytes: Vec<u8>, ext: Option<&str>, rate: u32) -> Result<Vec<f32>> {
    to_mono(decode(bytes, ext)?, rate)
}

/// Planar audio from `from` Hz to `to` Hz with [`resample_poly`].
pub fn resample(data: &Planar, from: u32, to: u32) -> Result<Planar> {
    if from == 0 || to == 0 {
        bail!("sample rate 0");
    }
    Ok(data.iter().map(|c| resample_poly(c, to, from)).collect())
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

/// Modified Bessel function of the first kind, order 0.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let q = x * x / 4.0;
    for k in 1..500 {
        term *= q / (k as f64 * k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// `scipy.signal.firwin(numtaps, cutoff, window=("kaiser", beta))`:
/// low-pass, cutoff relative to Nyquist, unit gain at DC.
fn firwin_kaiser(numtaps: usize, cutoff: f64, beta: f64) -> Vec<f64> {
    let alpha = 0.5 * (numtaps as f64 - 1.0);
    let sinc = |x: f64| {
        if x == 0.0 {
            1.0
        } else {
            let px = std::f64::consts::PI * x;
            px.sin() / px
        }
    };
    let i0b = bessel_i0(beta);
    let mut h: Vec<f64> = (0..numtaps)
        .map(|n| {
            let m = n as f64 - alpha;
            let r = if numtaps > 1 {
                2.0 * n as f64 / (numtaps as f64 - 1.0) - 1.0
            } else {
                0.0
            };
            let w = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0b;
            cutoff * sinc(cutoff * m) * w
        })
        .collect();
    let s: f64 = h.iter().sum();
    for v in &mut h {
        *v /= s;
    }
    h
}

/// `scipy.signal.resample_poly(x, up, down)` (Kaiser window, beta 5, zero
/// padding), computed in f64: what the reference transcribe.py uses.
pub fn resample_poly(x: &[f32], up: u32, down: u32) -> Vec<f32> {
    let g = gcd(up as u64, down as u64).max(1);
    let (up, down) = ((up as u64 / g) as usize, (down as u64 / g) as usize);
    if up == 1 && down == 1 {
        return x.to_vec();
    }
    let n_in = x.len();
    let n_out = (n_in * up).div_ceil(down);
    let max_rate = up.max(down);
    let half_len = 10 * max_rate;
    let mut h = firwin_kaiser(2 * half_len + 1, 1.0 / max_rate as f64, 5.0);
    for v in &mut h {
        *v *= up as f64;
    }
    let n_pre_pad = down - half_len % down;
    let n_pre_remove = (half_len + n_pre_pad) / down;
    // padded filter: n_pre_pad zeros, then h (post padding only adds zeros)
    let hp = |k: usize| -> f64 {
        if k < n_pre_pad || k - n_pre_pad >= h.len() {
            0.0
        } else {
            h[k - n_pre_pad]
        }
    };
    let taps = n_pre_pad + h.len();
    let mut y = Vec::with_capacity(n_out);
    for m in 0..n_out {
        // y[m] = sum_k hp[k] * xu[(m + n_pre_remove) * down - k], xu = x
        // upsampled by `up` (zeros between samples)
        let j = (m + n_pre_remove) * down;
        let mut acc = 0f64;
        // k such that (j - k) % up == 0 and 0 <= (j - k) / up < n_in
        let mut k = j % up;
        while k < taps && k <= j {
            let i = (j - k) / up;
            if i < n_in {
                acc += hp(k) * x[i] as f64;
            }
            k += up;
        }
        y.push(acc as f32);
    }
    y
}

/// 16-bit PCM mono WAV in memory.
pub fn encode_wav(samples: &[f32], rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cur = Cursor::new(Vec::with_capacity(samples.len() * 2 + 64));
    {
        let mut w = hound::WavWriter::new(&mut cur, spec)?;
        for v in samples {
            w.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16)?;
        }
        w.finalize()?;
    }
    Ok(cur.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_roundtrip_resample() {
        let sr = 8_000;
        let tone: Vec<f32> = (0..sr).map(|i| (i as f32 * 0.05).sin() * 0.5).collect();
        let bytes = encode_wav(&tone, sr as u32).unwrap();
        let a = decode(bytes, Some("wav")).unwrap();
        assert_eq!(a.sample_rate, sr as u32);
        let m = to_mono(a, 16_000).unwrap();
        assert_eq!(m.len(), 2 * tone.len());
        let peak = m[1000..15_000].iter().fold(0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.01, "peak {peak}");
    }

    #[test]
    fn resample_poly_matches_scipy() {
        // scipy.signal.resample_poly(np.arange(1., 11.), 1, 3) and (…, 2, 1)[:6]
        let x: Vec<f32> = (1..=10).map(|v| v as f32).collect();
        let down = resample_poly(&x, 1, 3);
        let want = [1.194, 3.6312, 7.7961, 6.4106];
        assert_eq!(down.len(), 4);
        for (a, b) in down.iter().zip(want) {
            assert!((a - b).abs() < 1e-3, "{down:?}");
        }
        let up = resample_poly(&x, 2, 1);
        assert_eq!(up.len(), 20);
        let want = [1.0005, 1.5609, 2.001, 2.4391, 3.0016, 3.5857];
        for (a, b) in up.iter().zip(want) {
            assert!((a - b).abs() < 1e-3, "{up:?}");
        }
    }
}
