# AGENTS.md

Guidance for AI agents working in this repository.

## Project

Rust/Candle port of Hugging Face `Wav2Vec2ForCTC` inference (wav2vec 2.0,
XLS-R) with pyctcdecode's beam search and a KenLM-compatible ARPA n-gram LM.
One binary (`wav2vec2`), models in one GGUF written by `wav2vec2 convert`
from a Hugging Face directory. Engine of zallama's `wav2vec2-server` backend
(`POST /v1/audio/transcriptions`).

Pipeline: decode → mono, 16 kHz (`audio.rs`: scipy `resample_poly`
semantics) → zero-mean/unit-variance normalization over the whole input →
40 s windows with 3 s context on frame boundaries (`transcribe.rs`) →
model (`model.rs`) → CTC logits → greedy or beam search (`ctc.rs`) with
the LM (`lm.rs`) → words with frame timestamps (20 ms).

## Where things are

- `crates/wav2vec2/src/config.rs`: HF config + vocab → `Config` (JSON in
  the GGUF), CTC labels as pyctcdecode sees them (`|` → " ", pad → "",
  `<unk>` → "⁇").
- `crates/wav2vec2/src/model.rs`: the network; `forward_debug` for
  per-layer parity dumps.
- `crates/wav2vec2/src/cpu_ops.rs`: multi-threaded CPU GELU, LayerNorm and
  im2col (candle 0.11 runs elementwise ops on one thread).
- `crates/wav2vec2/src/gguf.rs`: HF directory discovery, weight-norm
  folding, conversion, loading.
- `crates/wav2vec2/src/lm.rs`: ARPA reader, KenLM `BaseScore` semantics.
- `crates/wav2vec2/src/ctc.rs`: greedy and pyctcdecode 0.5 beam search.
- `crates/wav2vec2/src/testing.rs`: tiny random HF model for tests.
- `crates/wav2vec2-cli/src/`: CLI (`main.rs`), HTTP server (`server.rs`).
- `scripts/ref/`: Python reference (`ref.py`), WER (`wer.py`).

## Commands

```bash
cargo build --release -p wav2vec2-cli
cargo test --release --workspace
cargo clippy --workspace --all-targets -- -D warnings
wav2vec2 convert <hf-dir> -o models/x.gguf [--dtype f16|q8_0] [--lm x.arpa|--no-lm]
wav2vec2 transcribe -m models/x.gguf speech.wav [--json] [--greedy]

# parity (venv: torch, transformers, pyctcdecode, kenlm, soundfile, scipy,
# safetensors, numpy 1.26 as the reference environment, pyarrow, jiwer)
python -I scripts/ref/ref.py <hf-dir> tmp/ref --parquet <badrex/malagasy-speech-full shard> -n 30
cargo run --release --example parity -- <hf-dir | x.gguf> tmp/ref [--cpu] [--dtype f16]
cargo run --release --example nbest -- <hf-dir> tmp/ref clip018      # final beams
python -I scripts/ref/wer.py refs.json hyp.jsonl
```

CUDA: `--features wav2vec2/cuda,wav2vec2-cli/cuda`, build into
`--target-dir target-cuda`, needs `/usr/local/cuda/bin` in PATH and
`CUDA_COMPUTE_CAP`.

## Rules

- transformers' `modeling_wav2vec2.py` and pyctcdecode 0.5's `decoder.py` /
  `language_model.py` are the spec. F32 CPU logits must match the reference
  (SNR well above 100 dB); the decoder must give pyctcdecode's transcript
  on the reference logits for every clip.
- pyctcdecode details that change transcripts: `decode()` uses
  `prune_history=True`; merged beams take the *latest* beam's fields; the
  last word closed by a space is scored as the empty word (an `<unk>` plus
  `unk_score_offset`); unigrams are the ARPA 1-grams with a back-off field.
- Keep the CPU path dependency-free (no ffmpeg, no MKL, no system libs).
- Tests need no model download (`testing::write_tiny_hf`).
