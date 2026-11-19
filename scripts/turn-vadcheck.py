# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy>=1.24,<3",
#   "onnxruntime>=1.18",
# ]
# ///
"""The second detector of s6-ear v3's label cross-check (docs/s6-ear3-prereg.md §3.7): Silero VAD
v6.2.3 on the validation sessions of one shard of `sky-data fetch-conv`, for the cross-check only.

    uv run --python 3.12 scripts/turn-vadcheck.py --conv DIR/conv --tag 00012 [--models DIR/hf/silero]
        [--pin runs/s6-ear3-nano/silero.json]

* **The file.** `src/silero_vad/data/silero_vad.onnx` of the tag v6.2.3 (commit
  5cd7945676eb32225748052e2e6a0580e4686a08), downloaded once from GitHub: its size (2,327,524 bytes) and
  its git blob hash (80c5592e...) must be the tag's. Its sha256 is recorded in the pin file at the first
  use and checked at every later use (§3.1).
* **Input.** `DIR/xcheck-<tag>.jsonl` names each validation session and its two transient 16 kHz,
  gain-normalised channels (32-bit float WAV, `wav-<tag>/`).
* **Settings.** Silero's ONNX model as its own wrapper runs it: 512-sample windows (32 ms) with the
  previous 64 samples as context and the recurrent state carried, the state reset per channel; a window
  is speech when its probability is >= 0.5. Only whole windows are read.
* **Output.** `DIR/silero-<tag>.jsonl`: per session and channel, the raw speech runs `[start, end)` in
  samples (consecutive speech windows). The cleaning (150 / 150 ms), the offsets, the label rule and the
  matching are those of `sky-data conv-crosscheck`, the same code as the records'. Then the WAVs are
  removed. A shard without validation sessions has nothing to do.

Silero is used for this check only: never for a training label, a selection or a weight (§13).
"""
import argparse
import datetime
import hashlib
import json
import shutil
import struct
import sys
import urllib.request
from pathlib import Path

TAG = "v6.2.3"
COMMIT = "5cd7945676eb32225748052e2e6a0580e4686a08"
PATH = "src/silero_vad/data/silero_vad.onnx"
URL = f"https://raw.githubusercontent.com/snakers4/silero-vad/{COMMIT}/{PATH}"
SIZE = 2_327_524
GIT_BLOB = "80c5592ef1f4c9ede3e357bbd02eb863358a6a9d"
RATE = 16000
WINDOW = 512
CONTEXT = 64
THRESHOLD = 0.5


def git_blob_sha1(b: bytes) -> str:
    """The git object id of a blob: sha1 of `blob <size>\\0` and the bytes."""
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def pinned_model(models: Path, pin: Path) -> Path:
    """The model file, downloaded if absent, checked against the tag (size, git blob) and the pin."""
    models.mkdir(parents=True, exist_ok=True)
    f = models / "silero_vad.onnx"
    if not f.exists():
        tmp = f.with_suffix(".part")
        with urllib.request.urlopen(URL, timeout=120) as r, open(tmp, "wb") as out:
            shutil.copyfileobj(r, out)
        tmp.rename(f)
    b = f.read_bytes()
    if len(b) != SIZE or git_blob_sha1(b) != GIT_BLOB:
        raise SystemExit(f"{f}: not the file of Silero VAD {TAG} ({len(b)} bytes, blob {git_blob_sha1(b)})")
    sha = hashlib.sha256(b).hexdigest()
    if pin.exists():
        want = json.loads(pin.read_text())["sha256"]
        if sha != want:
            raise SystemExit(f"{f}: sha256 {sha}, {pin} pins {want}")
    else:
        pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text(json.dumps({"file": PATH, "tag": TAG, "commit": COMMIT, "url": URL, "bytes": SIZE,
                                   "git_blob": GIT_BLOB, "sha256": sha, "licence": "MIT",
                                   "first_use": datetime.date.today().isoformat(),
                                   "use": "docs/s6-ear3-prereg.md §3.7, the label cross-check only"}, indent=2) + "\n")
    return f


def read_wav_f32(path: Path):
    """A mono 32-bit float WAV at 16 kHz (as sky-data writes them) -> array('f')."""
    import array

    b = path.read_bytes()
    if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    i, fmt = 12, None
    while i + 8 <= len(b):
        cid, size = b[i:i + 4], struct.unpack("<I", b[i + 4:i + 8])[0]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", b[i + 8:i + 24])
        elif cid == b"data":
            tag, ch, rate, _, _, bits = fmt
            if (tag, ch, rate, bits) != (3, 1, RATE, 32):
                raise ValueError(f"{path}: need mono 32-bit float at {RATE} Hz, not {fmt}")
            x = array.array("f")
            x.frombytes(b[i + 8:i + 8 + size])
            if sys.byteorder != "little":
                x.byteswap()
            return x
        i += 8 + size + (size & 1)
    raise ValueError(f"{path}: no data chunk")


def runs_of(flags, step=WINDOW):
    """The runs of consecutive true flags as [start, end) samples, `step` samples a flag."""
    out, start = [], None
    for k, s in enumerate(flags):
        if s and start is None:
            start = k
        elif not s and start is not None:
            out.append([start * step, k * step])
            start = None
    if start is not None:
        out.append([start * step, len(flags) * step])
    return out


class Silero:
    """Silero's ONNX model, run as its OnnxWrapper runs it at 16 kHz."""

    def __init__(self, path):
        import numpy as np
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        names = {i.name for i in self.sess.get_inputs()}
        if names != {"input", "state", "sr"}:
            raise SystemExit(f"{path}: inputs {sorted(names)}, not Silero v5/v6's input, state and sr")
        self.np = np

    def probs(self, x):
        """The speech probability of each whole 512-sample window of `x`."""
        np = self.np
        x = np.asarray(x, dtype=np.float32)
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, CONTEXT), dtype=np.float32)
        sr = np.array(RATE, dtype=np.int64)
        out = []
        for k in range(len(x) // WINDOW):
            w = np.asarray(x[k * WINDOW:(k + 1) * WINDOW], dtype=np.float32).reshape(1, WINDOW)
            inp = np.concatenate([context, w], axis=1)
            p, state = self.sess.run(None, {"input": inp, "state": state, "sr": sr})
            context = inp[:, -CONTEXT:]
            out.append(float(np.asarray(p).reshape(-1)[0]))
        return out


def run(conv: Path, tag: str, model) -> int:
    xc = conv / f"xcheck-{tag}.jsonl"
    out = conv / f"silero-{tag}.jsonl"
    wav_dir = conv / f"wav-{tag}"
    if out.exists():
        shutil.rmtree(wav_dir, ignore_errors=True)
        print(f"vadcheck {tag}: done")
        return 0
    if not xc.exists():
        if (conv / f"conv-{tag}.json").exists():
            print(f"vadcheck {tag}: no validation session")
            return 0
        raise SystemExit(f"no {xc} (sky-data fetch-conv --shards {int(tag)})")
    vad = model() if callable(model) else model
    lines = []
    for line in xc.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        for ch, name in enumerate(s["wav"]):
            flags = [p >= THRESHOLD for p in vad.probs(read_wav_f32(wav_dir / name))]
            lines.append(json.dumps({"session": s["session"], "channel": ch, "runs": runs_of(flags),
                                     "windows": len(flags)}))
    tmp = out.with_suffix(".part")
    tmp.write_text("".join(l + "\n" for l in lines))
    tmp.rename(out)
    shutil.rmtree(wav_dir, ignore_errors=True)
    print(f"vadcheck {tag}: {len(lines)} channels")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--conv", required=True, help="DIR/conv of sky-data fetch-conv")
    ap.add_argument("--tag", required=True, help="the shard's tag (five digits)")
    ap.add_argument("--models", default=None, help="where the ONNX file is kept (default: DIR/../hf/silero)")
    ap.add_argument("--pin", default="runs/s6-ear3-nano/silero.json")
    a = ap.parse_args()
    conv = Path(a.conv)
    models = Path(a.models) if a.models else conv.parent / "hf" / "silero"
    return run(conv, a.tag, lambda: Silero(pinned_model(models, Path(a.pin))))


if __name__ == "__main__":
    sys.exit(main())
