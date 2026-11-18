# Credits

- **wav2vec 2.0**: Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
  Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
  Representations" (NeurIPS 2020, arXiv:2006.11477). **XLS-R**: Arun Babu
  et al., "XLS-R: Self-supervised Cross-lingual Speech Representation
  Learning at Scale" (arXiv:2111.09296). Meta AI.
- **Hugging Face transformers** (Apache-2.0): `modeling_wav2vec2.py` and
  `Wav2Vec2FeatureExtractor`, the reference for the network.
- **pyctcdecode** (Apache-2.0, Kensho Technologies): the beam search this
  port reproduces.
- **KenLM** (LGPL-2.1, Kenneth Heafield): the ARPA format and `BaseScore`
  semantics, reimplemented here; no KenLM code is used.
- **SciPy** (BSD-3): `resample_poly` / `firwin` semantics.
- **Candle** (Apache-2.0 / MIT, Hugging Face): tensors, CUDA/Metal kernels,
  GGUF, safetensors and pickle readers.
- **symphonia**, **hound**, **rayon**, **axum**, **clap**.

## Model weights

Weights are not part of this repository and keep their own licences. Tested:

- [malagasy-asr/xlsr-mg-norm-v2-kenlm](https://huggingface.co/malagasy-asr/xlsr-mg-norm-v2-kenlm):
  XLS-R 300M fine-tuned on Malagasy, with a 4-gram KenLM model (its card
  states no licence).
- Evaluation audio: [badrex/malagasy-speech-full](https://huggingface.co/datasets/badrex/malagasy-speech-full).
