# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy>=1.24,<3",
#   "onnxruntime>=1.18",
#   "transformers>=4.40",
#   "huggingface_hub>=0.23",
# ]
# ///
"""Smart Turn v3.2 (Pipecat, BSD-2) on the s6-ear turn-end test split: the bar of
docs/s6-ear-prereg.md §3 (gate 1), measured on exactly the clips s6-ear is tested on.

    uv run scripts/ear-baseline.py [--records data/ear/jev/turn_end.jsonl] [--max 0]
                                   [--json runs/s6-ear-nano/eval/baseline-smart-turn.json]

Reads the `test` records of the turn-end jev (`sky-data audio-jev`); each names the clip's
16 kHz WAV (`wav`, relative to the records file, written by `sky-data fetch-audio`). The model is
`smart-turn-v3.2-cpu.onnx` of `pipecat-ai/smart-turn-v3` at a pinned revision, run as Pipecat's
reference inference does: the last 8 s of audio, Whisper log-mel features (WhisperFeatureExtractor,
chunk_length 8, padded to 8 s, normalized), `probability > 0.5` = turn finished.

Prints and writes {n, accuracy, precision, recall, f1, positive_rate, median_ms, model, revision}.
Light: CPU only, about 10 ms a clip.
"""
import argparse
import json
import statistics
import struct
import time
from pathlib import Path

import numpy as np

REPO = "pipecat-ai/smart-turn-v3"
REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
FILE = "smart-turn-v3.2-cpu.onnx"
RATE = 16000
SECONDS = 8


def read_wav(path: Path) -> np.ndarray:
    """16-bit PCM WAV (as written by sky-data) -> float32 mono in [-1, 1]."""
    b = path.read_bytes()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    i, fmt = 12, None
    while i + 8 <= len(b):
        cid, size = b[i : i + 4], struct.unpack("<I", b[i + 4 : i + 8])[0]
        if cid == b"fmt ":
            tag, ch, rate = struct.unpack("<HHI", b[i + 8 : i + 16])
            bits = struct.unpack("<H", b[i + 22 : i + 24])[0]
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            tag, ch, rate, bits = fmt
            if tag != 1 or bits != 16 or rate != RATE:
                raise ValueError(f"{path}: need 16-bit PCM at {RATE} Hz")
            x = np.frombuffer(b[i + 8 : i + 8 + size], dtype="<i2").astype(np.float32) / 32768.0
            return x.reshape(-1, ch).mean(axis=1)
        i += 8 + size + (size & 1)
    raise ValueError(f"{path}: no data chunk")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--records", default="data/ear/jev/turn_end.jsonl")
    ap.add_argument("--max", type=int, default=0, help="clips evaluated (0: every test clip)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from transformers import WhisperFeatureExtractor

    model = hf_hub_download(REPO, FILE, revision=REVISION)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(model, sess_options=so, providers=["CPUExecutionProvider"])
    fe = WhisperFeatureExtractor(chunk_length=SECONDS)

    records = Path(a.records)
    rows = [json.loads(l) for l in records.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r.get("split") == "test" and r.get("wav")]
    if a.max:
        rows = rows[: a.max]
    if not rows:
        raise SystemExit(f"no test record with a wav in {records}")
    tp = fp = tn = fn = 0
    ms = []
    for r in rows:
        x = read_wav(records.parent / r["wav"])[-SECONDS * RATE :]
        feats = fe(
            x,
            sampling_rate=RATE,
            return_tensors="np",
            padding="max_length",
            max_length=SECONDS * RATE,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        t0 = time.perf_counter()
        p = float(np.asarray(sess.run(None, {"input_features": feats})[0]).reshape(-1)[0])
        ms.append((time.perf_counter() - t0) * 1e3)
        pred, gold = p > 0.5, r["target"][1] > 0.5
        tp += pred and gold
        fp += pred and not gold
        tn += (not pred) and not gold
        fn += (not pred) and gold
    n = len(rows)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out = {
        "n": n,
        "accuracy": (tp + tn) / n,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "positive_rate": (tp + fn) / n,
        "median_ms": statistics.median(ms),
        "model": f"{REPO}/{FILE}",
        "revision": REVISION,
        "records": str(records),
    }
    print(json.dumps(out, indent=2))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
