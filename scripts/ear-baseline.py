# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy>=1.24,<3",
#   "onnxruntime>=1.18",
#   "transformers>=4.40",
#   "huggingface_hub>=0.23",
#   "datasets>=2.19,<4",
#   "soundfile>=0.12",
#   "librosa>=0.10",
#   "pyarrow>=14",
# ]
# ///
"""Smart Turn v3.2 (Pipecat, BSD-2) on the Smart Turn test split, as Pipecat's reference does: the
bar of docs/s6-ear-prereg.md §3 and the harness check of docs/s6-ear2-prereg.md §8.1.

    uv run --python 3.12 scripts/ear-baseline.py [--clips data/ear/clips/smart-turn.jsonl]
        [--max 0] [--json OUT.json] [--dump OUT.jsonl] [--pad left|right]
    uv run --python 3.12 scripts/ear-baseline.py --records data/ear/jev/turn_end.jsonl ...   (v1's input)
    uv run --python 3.12 scripts/ear-baseline.py --eotbench DIR/eotbench --tag ar-00000|all
    uv run --python 3.12 scripts/ear-baseline.py --bench 1000 --json OUT.json

Test clips: the `test` lines of v1's clip manifest (`sky-data fetch-audio`), each naming the clip's
16 kHz WAV (`wav`, relative to the manifest's data folder) with its language, dataset and label;
or v1's turn-end jev (`--records`, no dataset). The model is `smart-turn-v3.2-cpu.onnx` of
`pipecat-ai/smart-turn-v3` at a pinned revision, fed as Pipecat's inference does: the last 8 s of
audio **left-padded** with zeros to 8 s (`audio_utils.truncate_audio_to_last_n_seconds`, the fix
of docs/s6-ear2-prereg.md C8), Whisper log-mel features (WhisperFeatureExtractor, chunk_length 8,
normalized), `probability > 0.5` = turn finished. `--pad right` gives v1's defective harness back
(WhisperFeatureExtractor's own end padding), for the record only.

Prints and writes {n, accuracy, precision, recall, f1, fpr_all, fnr_all, positive_rate,
timer_accuracy, median_ms, by_language, by_dataset, model, revision, pad}; `--dump` writes one
line per clip {id, prob, label, lang, dataset}. `timer_accuracy` is the always-finished rule's
accuracy (the test's share of finished clips, §8.1's second check).

`--eotbench` runs the cuts of `sky-data turn-eotbench` (docs/s6-ear2-prereg.md §8.4) of one file:
for each cut of `<store>/manifest-<tag>.jsonl`, the turn's audio (HF `datasets` decoding of
`<store>/../parquet/<tag>.parquet`) cut at the span's start, the last 8 s left-padded. Writes
`<store>/smartturn-<tag>.jsonl` {id, prob} and `smartturn-<tag>.done`; no label is read.
CPU only, one thread, about 30 ms a clip. `--offset-ms 200` (s6-ear v3, docs/s6-ear3-prereg.md
§8.3) cuts each span at its start + 0.2 s instead, for the span records whose `dmax_ms` reaches it
(`sky-data turn-eotbench --spans`), into `smartturn200-<tag>.jsonl` and `.done`.
"""
import argparse
import json
import statistics
import sys
import struct
import time
from collections import defaultdict
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


def last_seconds(x: np.ndarray, pad: str = "left", n: int = SECONDS * RATE) -> np.ndarray:
    """The last n samples; a shorter clip is zero-padded at the start (Pipecat's reference) or,
    with pad="right", left as it is for the extractor to pad at the end (v1's defect)."""
    x = np.asarray(x, dtype=np.float32)[-n:]
    if pad == "left" and len(x) < n:
        x = np.concatenate([np.zeros(n - len(x), dtype=np.float32), x])
    return x


class SmartTurn:
    def __init__(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import WhisperFeatureExtractor

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(
            hf_hub_download(REPO, FILE, revision=REVISION), sess_options=so, providers=["CPUExecutionProvider"]
        )
        self.fe = WhisperFeatureExtractor(chunk_length=SECONDS)

    def prob(self, x: np.ndarray, pad: str = "left") -> tuple[float, float]:
        feats = self.fe(
            last_seconds(x, pad),
            sampling_rate=RATE,
            return_tensors="np",
            padding="max_length",
            max_length=SECONDS * RATE,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        t0 = time.perf_counter()
        p = float(np.asarray(self.sess.run(None, {"input_features": feats})[0]).reshape(-1)[0])
        return p, (time.perf_counter() - t0) * 1e3


def counts(rows):
    tp = sum(1 for p, y in rows if p > 0.5 and y)
    fp = sum(1 for p, y in rows if p > 0.5 and not y)
    tn = sum(1 for p, y in rows if p <= 0.5 and not y)
    fn = sum(1 for p, y in rows if p <= 0.5 and y)
    return tp, fp, tn, fn


def summary(rows):
    tp, fp, tn, fn = counts(rows)
    n = max(len(rows), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "n": len(rows),
        "accuracy": (tp + tn) / n,
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / max(prec + rec, 1e-12),
        "fpr_all": fp / n,
        "fnr_all": fn / n,
        "positive_rate": (tp + fn) / n,
    }


def test_rows(a):
    """(id, wav path, label, lang, dataset) of the test clips."""
    if a.records:
        records = Path(a.records)
        rows = [json.loads(l) for l in records.read_text().splitlines() if l.strip()]
        return [
            (r.get("clip", ""), records.parent / r["wav"], r["target"][1] > 0.5, r.get("lang", ""), "")
            for r in rows
            if r.get("split") == "test" and r.get("wav")
        ]
    clips = Path(a.clips)
    base = clips.parent.parent
    out = []
    for line in clips.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        if c.get("split") == "test" and c.get("wav"):
            lab = c.get("label") or {}
            out.append((c["id"], base / c["wav"], bool(lab.get("endpoint")), c.get("lang", ""), lab.get("dataset") or ""))
    return out


def run_test(a):
    st = SmartTurn()
    rows = test_rows(a)
    if a.max:
        rows = rows[: a.max]
    if not rows:
        raise SystemExit("no test clip with a wav")
    res, ms = [], []
    dump = open(a.dump, "w") if a.dump else None
    for cid, wav, y, lang, ds in rows:
        p, t = st.prob(read_wav(wav), a.pad)
        ms.append(t)
        res.append((cid, p, y, lang, ds))
        if dump:
            dump.write(json.dumps({"id": cid, "prob": p, "label": y, "lang": lang, "dataset": ds}) + "\n")
    if dump:
        dump.close()
    out = summary([(p, y) for _, p, y, _, _ in res])
    out["timer_accuracy"] = out["positive_rate"]
    out["median_ms"] = statistics.median(ms)
    for key, idx in (("by_language", 3), ("by_dataset", 4)):
        groups = defaultdict(list)
        for r in res:
            groups[r[idx]].append((r[1], r[2]))
        out[key] = {g: summary(v) for g, v in sorted(groups.items())}
    out.update({"model": f"{REPO}/{FILE}", "revision": REVISION, "pad": a.pad,
                "source": a.records or a.clips})
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("by_")}, indent=2))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")


def run_bench(a):
    """§8.3's reference: Smart Turn's decision on one 8 s window whose features are computed,
    timed around `session.run` (one thread), median and p95 over --bench runs."""
    st = SmartTurn()
    rng = np.random.default_rng(0)
    x = (0.05 * rng.standard_normal(SECONDS * RATE)).astype(np.float32)
    feats = st.fe(x, sampling_rate=RATE, return_tensors="np", padding="max_length",
                  max_length=SECONDS * RATE, truncation=True, do_normalize=True).input_features.astype(np.float32)
    for _ in range(20):
        st.sess.run(None, {"input_features": feats})
    ms = []
    for _ in range(a.bench):
        t0 = time.perf_counter()
        st.sess.run(None, {"input_features": feats})
        ms.append((time.perf_counter() - t0) * 1e3)
    ms.sort()
    out = {"model": f"{REPO}/{FILE}", "revision": REVISION, "runs": a.bench, "threads": 1,
           "median_ms": ms[len(ms) // 2], "p95_ms": ms[min(len(ms) - 1, round(0.95 * (len(ms) - 1)))]}
    print(json.dumps(out))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")


def run_eotbench(a):
    store = Path(a.eotbench) / "store"
    if a.tag == "all":
        tags = sorted(p.name[len("manifest-"):-len(".jsonl")] for p in store.glob("manifest-*.jsonl"))
        for tag in tags:
            if (Path(a.eotbench) / "parquet" / f"{tag}.parquet").exists() or (store / f"{stem_of(a)}-{tag}.done").exists():
                run_eotbench_tag(a, tag)
            else:
                print(f"smart turn eot-bench {tag}: no parquet file and no result", file=sys.stderr)
        return
    run_eotbench_tag(a, a.tag)


def stem_of(a):
    """The name of a Smart Turn pass: `smartturn` at the span's start, `smartturn<ms>` later."""
    return "smartturn" if not a.offset_ms else f"smartturn{a.offset_ms}"


def run_eotbench_tag(a, tag):
    import pyarrow.parquet as pq
    from datasets import Audio

    store = Path(a.eotbench) / "store"
    stem = stem_of(a)
    done = store / f"{stem}-{tag}.done"
    if done.exists():
        print(f"smart turn eot-bench {tag}: done")
        return
    cuts = defaultdict(list)
    for line in (store / f"manifest-{tag}.jsonl").read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            if a.offset_ms and int(v.get("dmax_ms", 0)) < a.offset_ms:
                continue  # the ask does not fit inside the span
            cuts[int(v["row"])].append((v["id"], int(v["cut"]) + a.offset_ms * RATE // 1000))
    st = SmartTurn()
    dec = Audio(sampling_rate=RATE)
    out = open(store / f"{stem}-{tag}.jsonl.part", "w")
    n, row = 0, 0
    f = pq.ParquetFile(Path(a.eotbench) / "parquet" / f"{tag}.parquet")
    for batch in f.iter_batches(batch_size=16, columns=["audio"]):
        for au in batch.column("audio").to_pylist():
            if row in cuts:
                x = dec.decode_example({"bytes": au["bytes"], "path": None})["array"]
                for cid, cut in cuts[row]:
                    p, _ = st.prob(np.asarray(x[:cut], dtype=np.float32)) if cut > 0 else (float("nan"), 0)
                    out.write(json.dumps({"id": cid, "prob": p}) + "\n")
                    n += 1
            row += 1
    out.close()
    (store / f"{stem}-{tag}.jsonl.part").rename(store / f"{stem}-{tag}.jsonl")
    done.write_text(json.dumps({"tag": tag, "cuts": n, "offset_ms": a.offset_ms, "model": f"{REPO}/{FILE}", "revision": REVISION}) + "\n")
    print(f"smart turn eot-bench {tag}: {n} cuts")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--clips", default="data/ear/clips/smart-turn.jsonl")
    ap.add_argument("--records", default=None, help="v1's turn-end jev instead of the clip manifest")
    ap.add_argument("--max", type=int, default=0, help="clips evaluated (0: every test clip)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump", default=None, help="one line per clip: id, prob, label, lang, dataset")
    ap.add_argument("--pad", choices=["left", "right"], default="left")
    ap.add_argument("--eotbench", default=None, help="DIR/eotbench of sky-data turn-eotbench")
    ap.add_argument("--tag", default=None, help="the eot-bench file's tag (e.g. ar-00000), or all")
    ap.add_argument("--offset-ms", type=int, default=0, help="cut each span at its start + this (s6-ear v3: 200)")
    ap.add_argument("--bench", type=int, default=0, help="time N decisions on computed features (§8.3)")
    a = ap.parse_args()
    if a.bench:
        run_bench(a)
    elif a.eotbench:
        if not a.tag:
            raise SystemExit("--eotbench needs --tag")
        run_eotbench(a)
    else:
        run_test(a)


if __name__ == "__main__":
    main()
