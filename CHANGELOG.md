# Changelog

## 0.1.0 (2026-11-18)

- wav2vec 2.0 / XLS-R `Wav2Vec2ForCTC` inference in Rust/Candle (pre-LN and
  post-LN encoders, LayerNorm and GroupNorm feature encoders), CUDA, Metal,
  CPU; F16 / BF16 / F32 compute.
- `wav2vec2 convert`: Hugging Face directory → one GGUF (F16, Q8_0, BF16,
  F32 weights; positional-conv weight norm folded; vocabulary, decoder
  settings and an ARPA LM embedded).
- Pure-Rust ARPA n-gram LM with KenLM `BaseScore` semantics, and
  pyctcdecode 0.5's beam search: identical transcripts to the Python stack
  on 30/30 reference recordings.
- scipy-exact `resample_poly`; symphonia decoding.
- 40 s windows with 3 s context for long audio.
- `wav2vec2 serve`: OpenAI-compatible `/v1/audio/transcriptions` (json,
  text, verbose_json with word timestamps, srt, vtt, SSE stream),
  `/health`, `/v1/models`.
- Multi-threaded CPU GELU / LayerNorm / im2col.
