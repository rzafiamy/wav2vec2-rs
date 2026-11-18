# wav2vec2-rs

Speech recognition in Rust: **wav2vec 2.0 / XLS-R** CTC models on
[Candle](https://github.com/huggingface/candle), one GGUF file per model,
with a pure-Rust n-gram language model and a beam search that gives the
same transcripts as pyctcdecode + KenLM.

Built first for Malagasy:
[malagasy-asr/xlsr-mg-norm-v2-kenlm](https://huggingface.co/malagasy-asr/xlsr-mg-norm-v2-kenlm)
(XLS-R 300M fine-tuned on Malagasy, 4-gram KenLM).

- One binary, `wav2vec2`: `convert`, `transcribe`, `serve` (OpenAI-compatible
  HTTP), `info`.
- One file holds everything: network, CTC vocabulary, decoder settings and
  the ARPA language model.
- Pure Rust: audio decoding (WAV, FLAC, MP3, Ogg Vorbis, AAC/ALAC in
  MP4/M4A), scipy-exact `resample_poly`, the LM (no KenLM library), the
  decoder. No Python, no ffmpeg.
- CUDA, Metal or CPU. F16 / BF16 / F32 compute; F16 or Q8_0 weights.
- Any Hugging Face `Wav2Vec2ForCTC` with a character vocabulary: XLS-R and
  large models (pre-LN, LayerNorm convs) and base models (post-LN,
  GroupNorm). MMS language adapters are not supported yet.
- The engine behind zallama's `wav2vec2-server` ASR backend
  ([rzafiamy/zallamars](https://github.com/rzafiamy/zallamars)).

## Build

```bash
./build.sh --cuda      # NVIDIA (needs nvcc), or --metal, or nothing for CPU
```

## Models

Ready to use:
[rleo/xlsr-mg-norm-v2-kenlm-GGUF](https://huggingface.co/rleo/xlsr-mg-norm-v2-kenlm-GGUF).

Or convert a Hugging Face model directory yourself. A repository root with a
`decoding_config.json` (as malagasy-asr/xlsr-mg-norm-v2-kenlm) gives the
subfolders, the LM file, alpha, beta and beam width; a plain model
directory needs `config.json`, `model.safetensors` (or `pytorch_model.bin`,
read as data, no code runs) and `vocab.json`:

```bash
hf download malagasy-asr/xlsr-mg-norm-v2-kenlm --local-dir xlsr-mg
wav2vec2 convert xlsr-mg -o models/xlsr-mg-norm-v2-f16.gguf
wav2vec2 convert xlsr-mg -o models/xlsr-mg-norm-v2-q8_0.gguf --dtype q8_0
wav2vec2 convert some/model --lm my.arpa -o m.gguf      # or --no-lm
```

| dtype | file (with the 68 MB LM) | WER* | GPU peak** |
|---|---|---|---|
| `f16` (default) | 699 MB | 16.34 % | 2.5 GB |
| `q8_0` | 416 MB | 16.25 % | 2.4 GB |
| Python reference | 1.33 GB | 16.37 % | |

\* 135 recordings (44 min) of badrex/malagasy-speech-full, test shard 0,
with the model card's text normalization; beam 500 + 4-gram LM.
\*\* RTX 4090, 40 s windows. The source model's card states no licence:
ask its authors before any commercial use.

## Transcribe

```bash
wav2vec2 transcribe -m models/xlsr-mg-norm-v2-f16.gguf speech.mp3
wav2vec2 transcribe -m … --json *.wav        # word timestamps
wav2vec2 transcribe -m … --greedy a.wav      # no beam search, no LM
```

Options: `--beam-width N`, `--alpha`, `--beta` (defaults from the model),
`--lm other.arpa`, `--no-lm`, `--chunk-secs` (default 40),
`--context-secs` (3), `--dtype f16|bf16|f32`, `--cpu`, `--threads N`.

Long audio is cut into 40 s windows with 3 s of context on each side (only
the centre frames are kept). Do not raise `--chunk-secs` far: memory grows
with the square of the window, and the model, trained on short utterances,
degrades on long ones (a 200 s recording in one pass: 33.6 % WER, 21 GB;
in 40 s windows: 17.9 %, 2.5 GB).

## HTTP API

```bash
wav2vec2 serve -m models/xlsr-mg-norm-v2-f16.gguf --port 8791
curl -F file=@speech.wav -F model=xlsr-mg http://127.0.0.1:8791/v1/audio/transcriptions
```

- `POST /v1/audio/transcriptions` (multipart, OpenAI-compatible): `file`,
  `response_format` = `json` (default) | `text` | `verbose_json` | `srt` |
  `vtt`, `timestamp_granularities[]=word` (words in verbose_json),
  `stream=true` (server-sent `transcript.text.delta` per segment, then
  `transcript.text.done`). `model`, `language`, `prompt`, `temperature` are
  accepted and ignored.
- `GET /health`, `GET /v1/models`.

One transcription at a time; the others wait.

## Speed

| | 201 s of audio | |
|---|---|---|
| RTX 4090, F16 | 0.75 s | 266× realtime (model alone ~470×) |
| RTX 4090, Q8_0 | 0.83 s | 243× |
| CPU, 32 threads, F32 | ~30 s | ~7× |

Beam search with the LM costs ~0.3 ms per second of audio. The CPU path is
limited by Candle's matrix multiply (PyTorch with MKL is ~2.5× faster
there).

## Accuracy against the reference

`scripts/ref/ref.py` runs the Python stack (transformers, pyctcdecode,
kenlm) and dumps logits and transcripts; `examples/parity.rs` compares:

- network, F32 on the CPU: logit SNR ≥ 118 dB on 30 recordings, 100 %
  argmax agreement; F16 GGUF ≥ 74 dB;
- decoder on the reference logits: 30/30 identical transcripts (beam 500,
  4-gram LM), greedy 30/30;
- end to end (F32 CPU and F16 CPU): 30/30 identical.

See [AGENTS.md](AGENTS.md) for the commands.

## Licence

Apache-2.0. Model weights keep their own licences. See
[CREDITS.md](CREDITS.md).
