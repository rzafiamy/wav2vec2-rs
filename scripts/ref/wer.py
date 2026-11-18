"""WER / CER of transcripts against references, with the model card's
normalization (NFC, lowercase, typographic apostrophes and hyphens
unified, punctuation except ' and - replaced by spaces, whitespace
collapsed).

    python -I scripts/ref/wer.py refs.json hyp.jsonl [hyp2.jsonl|ref.json ...]

refs.json: {audio_id: text}. A hypothesis file is `wav2vec2 transcribe
--json` output (JSON lines with "file" and "text") or a scripts/ref/ref.py
ref.json (its "beam" transcripts).
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import jiwer


def norm(t):
    t = unicodedata.normalize("NFC", t).lower()
    t = re.sub("[‘’ʼ`´]", "'", t)
    t = re.sub("[‐‑‒–—−]", "-", t)
    t = "".join(c if (c.isalnum() or c in "'-⁇" or c.isspace()) else " " for c in t)
    return " ".join(t.split())


def hyps(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return {Path(r["file"]).stem: r["text"] for r in rows}
    return {Path(r["path"]).stem: r["beam"] for r in json.loads(p.read_text())}


refs = json.loads(Path(sys.argv[1]).read_text())
for h in sys.argv[2:]:
    hy = hyps(h)
    keys = sorted(k for k in hy if k in refs)
    r = [norm(refs[k]) for k in keys]
    y = [norm(hy[k]) for k in keys]
    print(f"{h}: {len(keys)} clips, WER {100 * jiwer.wer(r, y):.2f}%, CER {100 * jiwer.cer(r, y):.2f}%")
