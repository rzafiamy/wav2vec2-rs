//! `wav2vec2`: convert a Hugging Face wav2vec2 CTC model to GGUF,
//! transcribe files, or serve the OpenAI-compatible HTTP API.

mod server;

use anyhow::{Context, Result, bail};
use candle_core::DType;
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use std::time::Instant;
use wav2vec2::{Chunking, DecodeOptions, LmSource, Recognizer, audio};

#[derive(Parser)]
#[command(
    name = "wav2vec2",
    version,
    about = "wav2vec 2.0 / XLS-R CTC speech recognition, GGUF weights, n-gram LM beam search"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct RunOpts {
    /// wav2vec2 GGUF (from `wav2vec2 convert`)
    #[arg(short, long, env = "WAV2VEC2_MODEL")]
    model: PathBuf,
    /// Run on the CPU even when a GPU build finds a GPU
    #[arg(long)]
    cpu: bool,
    /// Compute dtype: f32, f16, bf16 (default: f16 on a GPU, f32 on the CPU)
    #[arg(long)]
    dtype: Option<String>,
    /// CPU threads (default: all cores)
    #[arg(long)]
    threads: Option<usize>,
    /// ARPA language model to use instead of the embedded one
    #[arg(long, conflicts_with = "no_lm")]
    lm: Option<PathBuf>,
    /// Beam search without the language model
    #[arg(long)]
    no_lm: bool,
    /// Greedy (argmax) decoding: fastest, no beam search, no LM
    #[arg(long)]
    greedy: bool,
    /// Beam width (default: the model's, 500 for xlsr-mg)
    #[arg(long)]
    beam_width: Option<usize>,
    /// LM weight (default: the model's)
    #[arg(long)]
    alpha: Option<f64>,
    /// Word insertion bonus (default: the model's)
    #[arg(long)]
    beta: Option<f64>,
    /// Longest audio the model sees at once, in seconds; longer input is
    /// cut into overlapping windows
    #[arg(long, default_value_t = 40.0)]
    chunk_secs: f64,
    /// Extra audio on each side of a window, in seconds
    #[arg(long, default_value_t = 3.0)]
    context_secs: f64,
}

#[derive(Subcommand)]
enum Cmd {
    /// Convert a Hugging Face Wav2Vec2ForCTC directory (config.json,
    /// model.safetensors, vocab.json, ...; or a repository root with a
    /// decoding_config.json) to one GGUF
    Convert {
        source: PathBuf,
        #[arg(short, long)]
        output: PathBuf,
        /// Linear weight dtype: f16 (default), q8_0, bf16, f32
        #[arg(long, default_value = "f16")]
        dtype: String,
        /// Model name stored in the file (default: output file stem)
        #[arg(long)]
        name: Option<String>,
        /// ARPA language model to embed (default: decoding_config.json's kenlm_file)
        #[arg(long, conflicts_with = "no_lm")]
        lm: Option<PathBuf>,
        /// Do not embed a language model
        #[arg(long)]
        no_lm: bool,
    },
    /// Transcribe audio files (wav, flac, mp3, ogg, m4a)
    Transcribe {
        #[command(flatten)]
        run: RunOpts,
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        /// One JSON object per file, with word timestamps
        #[arg(long)]
        json: bool,
    },
    /// Serve the HTTP API (POST /v1/audio/transcriptions)
    Serve {
        #[command(flatten)]
        run: RunOpts,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8791)]
        port: u16,
        /// Model id reported by /v1/models (default: the GGUF's name)
        #[arg(long)]
        model_id: Option<String>,
        /// Language code reported in verbose_json
        #[arg(long, default_value = "mg")]
        language: String,
    },
    /// Show what a GGUF holds
    Info {
        #[arg(short, long)]
        model: PathBuf,
    },
}

fn parse_compute_dtype(s: &str) -> Result<DType> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "f32" => DType::F32,
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        o => bail!("unknown compute dtype '{o}' (f32, f16, bf16)"),
    })
}

fn device_name(d: &candle_core::Device) -> &'static str {
    if d.is_cuda() {
        "cuda"
    } else if d.is_metal() {
        "metal"
    } else {
        "cpu"
    }
}

pub struct Opened {
    pub rec: Recognizer,
    pub name: String,
    pub opts: DecodeOptions,
    pub device: &'static str,
}

fn open(run: &RunOpts) -> Result<Opened> {
    if let Some(n) = run.threads {
        // candle's CPU kernels use rayon's global pool, which reads this
        // once, at its first use (later than here)
        unsafe { std::env::set_var("RAYON_NUM_THREADS", n.to_string()) };
    }
    if run.chunk_secs <= 2.0 * run.context_secs {
        bail!("--chunk-secs must be more than twice --context-secs");
    }
    let device = wav2vec2::pick_device(run.cpu)?;
    let dtype = run.dtype.as_deref().map(parse_compute_dtype).transpose()?;
    let t = Instant::now();
    let lm = if run.no_lm || run.greedy {
        LmSource::Disabled
    } else if let Some(p) = &run.lm {
        LmSource::Arpa(p)
    } else {
        LmSource::Embedded
    };
    let mut rec = Recognizer::open(&run.model, &device, dtype, Some(lm))?;
    rec.chunking = Chunking {
        chunk_secs: run.chunk_secs,
        context_secs: run.context_secs,
    };
    let mut opts = rec.default_options();
    opts.greedy = run.greedy;
    if let Some(w) = run.beam_width {
        opts.beam.beam_width = w.max(1);
    }
    if let Some(a) = run.alpha {
        opts.beam.alpha = a;
    }
    if let Some(b) = run.beta {
        opts.beam.beta = b;
    }
    let name = if rec.name.is_empty() {
        run.model
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "wav2vec2".into())
    } else {
        rec.name.clone()
    };
    let decoding = match (&rec.scorer, opts.greedy) {
        (_, true) => "greedy".to_string(),
        (Some(s), _) => format!("beam {} + {}-gram LM", opts.beam.beam_width, s.lm.order()),
        (None, _) => format!("beam {}, no LM", opts.beam.beam_width),
    };
    eprintln!(
        "loaded {name} ({}, {:?}, {decoding}) in {:.1}s",
        device_name(&device),
        rec.model.dtype,
        t.elapsed().as_secs_f64()
    );
    Ok(Opened {
        rec,
        name,
        opts,
        device: device_name(&device),
    })
}

fn ext_of(p: &Path) -> Option<String> {
    p.extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
}

pub fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

fn transcribe_files(run: &RunOpts, inputs: &[PathBuf], json: bool) -> Result<()> {
    let o = open(run)?;
    for input in inputs {
        let t = Instant::now();
        let bytes = std::fs::read(input).with_context(|| format!("reading {input:?}"))?;
        let wav = audio::load(bytes, ext_of(input).as_deref(), o.rec.cfg.sample_rate)
            .with_context(|| format!("decoding {input:?}"))?;
        let tr = o.rec.transcribe(&wav, &o.opts)?;
        let el = t.elapsed().as_secs_f64();
        if json {
            let words: Vec<_> = tr
                .words
                .iter()
                .map(
                    |w| serde_json::json!({"word": w.text, "start": r3(w.start), "end": r3(w.end)}),
                )
                .collect();
            println!(
                "{}",
                serde_json::json!({
                    "file": input.display().to_string(),
                    "duration": r3(tr.duration),
                    "text": tr.text,
                    "words": words,
                })
            );
        } else {
            println!("{}", tr.text);
        }
        eprintln!(
            "{}: {:.1}s of audio in {el:.2}s ({:.1}x realtime)",
            input.display(),
            tr.duration,
            tr.duration / el.max(1e-9)
        );
    }
    Ok(())
}

fn info(model: &Path) -> Result<()> {
    let l = wav2vec2::gguf::load(model, &candle_core::Device::Cpu)?;
    let c = &l.config;
    println!("name: {}", l.name);
    println!(
        "model: {} layers, hidden {}, {} heads, ffn {}, {} conv layers ({} norm), {}",
        c.num_hidden_layers,
        c.hidden_size,
        c.num_attention_heads,
        c.intermediate_size,
        c.conv_dim.len(),
        c.feat_extract_norm,
        if c.do_stable_layer_norm {
            "pre-LN"
        } else {
            "post-LN"
        }
    );
    println!(
        "audio: {} Hz, normalize {}, {} ms per frame",
        c.sample_rate,
        c.do_normalize,
        c.frame_secs() * 1000.0
    );
    println!("vocab: {} labels: {:?}", c.vocab_size, c.vocab.labels);
    println!(
        "decoder: alpha {}, beta {}, beam {}, lowercase {}",
        c.decoder.alpha, c.decoder.beta, c.decoder.beam_width, c.decoder.lowercase
    );
    match &l.arpa {
        Some(a) => {
            let lm = wav2vec2::lm::NgramLm::from_arpa(a)?;
            println!(
                "language model: {}-gram, {} unigrams ({:.1} MB of ARPA)",
                lm.order(),
                lm.unigram_words().len(),
                a.len() as f64 / 1e6
            );
        }
        None => println!("language model: none"),
    }
    Ok(())
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,symphonia=warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Convert {
            source,
            output,
            dtype,
            name,
            lm,
            no_lm,
        } => {
            let dt = wav2vec2::gguf::parse_dtype(&dtype)?;
            let name = name.unwrap_or_else(|| {
                output
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            });
            let t = Instant::now();
            let r = wav2vec2::gguf::convert(&source, &output, dt, &name, lm.as_deref(), no_lm)?;
            let size = std::fs::metadata(&output)?.len() as f64 / 1e6;
            eprintln!(
                "wrote {} ({size:.0} MB, LM {:.0} MB) in {:.1}s: {}",
                output.display(),
                r.lm_bytes as f64 / 1e6,
                t.elapsed().as_secs_f64(),
                r.counts
                    .iter()
                    .map(|(d, n)| format!("{n} {d}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        Cmd::Transcribe { run, inputs, json } => transcribe_files(&run, &inputs, json)?,
        Cmd::Serve {
            run,
            host,
            port,
            model_id,
            language,
        } => {
            let o = open(&run)?;
            let id = model_id.unwrap_or_else(|| o.name.clone());
            let rt = tokio::runtime::Runtime::new()?;
            rt.block_on(server::serve(o, id, language, &host, port))?;
        }
        Cmd::Info { model } => info(&model)?,
    }
    Ok(())
}
