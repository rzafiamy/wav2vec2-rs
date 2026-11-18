"""Reference outputs from the Python stack (transformers + pyctcdecode +
kenlm), for parity checks of the Rust port.

    python -I scripts/ref/ref.py <hf-model-dir> <out-dir> --parquet <file> [-n 20]
    python -I scripts/ref/ref.py <hf-model-dir> <out-dir> a.wav b.flac ...

Writes <out-dir>/clips/*.wav|flac (from a parquet of the
badrex/malagasy-speech-full layout), <out-dir>/ref.safetensors (logits per
clip, F32 [T, V]) and <out-dir>/ref.json (per clip: path, reference text
if known, greedy and beam transcripts). Audio loading, decoder settings and
output normalization follow the model's transcribe.py.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pyctcdecode import build_ctcdecoder
from safetensors.numpy import save_file
from scipy.signal import resample_poly
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


def extract(parquet, out, n):
    import pyarrow.parquet as pq

    t = pq.read_table(parquet, columns=["audio_id", "audio", "transcription"]).slice(0, n)
    clips = []
    for row in t.to_pylist():
        a = row["audio"]
        ext = Path(a.get("path") or "x.wav").suffix or ".wav"
        p = out / f"{row['audio_id']}{ext}"
        p.write_bytes(a["bytes"])
        clips.append((p, row["transcription"]))
    return clips


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("audio", nargs="*")
    ap.add_argument("--parquet")
    ap.add_argument("-n", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.model_dir)
    out = Path(args.out_dir)
    (out / "clips").mkdir(parents=True, exist_ok=True)
    cfg = json.loads((root / "decoding_config.json").read_text())
    processor = Wav2Vec2Processor.from_pretrained(root / cfg["processor_subfolder"], local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(root / cfg["model_subfolder"], local_files_only=True).eval()

    tok = processor.tokenizer
    vocab = tok.get_vocab()
    id2tok = {i: t for t, i in vocab.items()}
    labels = []
    for i in range(max(id2tok) + 1):
        t = id2tok[i]
        labels.append(" " if t == tok.word_delimiter_token else "" if t == tok.pad_token else "<unk>" if t == tok.unk_token else t)
    decoder = build_ctcdecoder(labels=labels, kenlm_model_path=str(root / cfg["kenlm_file"]), alpha=cfg["alpha"], beta=cfg["beta"])

    clips = [(Path(a), None) for a in args.audio]
    if args.parquet:
        clips += extract(args.parquet, out / "clips", args.n)

    tensors, meta = {}, []
    for k, (path, ref) in enumerate(clips):
        audio, sr = sf.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != cfg["sampling_rate"]:
            g = math.gcd(sr, cfg["sampling_rate"])
            audio = resample_poly(audio, cfg["sampling_rate"] // g, sr // g)
        audio = audio.astype(np.float32)
        inputs = processor(audio, sampling_rate=cfg["sampling_rate"], return_tensors="pt", padding=True)
        t0 = time.time()
        with torch.no_grad():
            logits = model(inputs.input_values).logits[0].numpy()
        t1 = time.time()
        beam = decoder.decode(logits, beam_width=cfg["beam_width"])
        beam = " ".join(beam.lower().strip().split())
        t2 = time.time()
        greedy = processor.batch_decode(np.argmax(logits, axis=-1)[None])[0]
        key = f"clip{k:03d}"
        tensors[key] = logits.astype(np.float32)
        meta.append({
            "key": key, "path": str(path), "reference": ref, "seconds": len(audio) / cfg["sampling_rate"],
            "greedy": " ".join(greedy.lower().split()), "beam": beam,
            "model_secs": t1 - t0, "decode_secs": t2 - t1,
        })
        print(f"{key} {meta[-1]['seconds']:.1f}s model {t1 - t0:.1f}s beam {t2 - t1:.1f}s: {beam[:80]}", flush=True)
    save_file(tensors, str(out / "ref.safetensors"))
    (out / "ref.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
