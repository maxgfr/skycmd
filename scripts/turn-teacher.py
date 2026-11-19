# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "numpy>=1.24,<3",
#   "onnx>=1.15",
#   "onnxruntime>=1.18",
#   "transformers>=4.40",
#   "huggingface_hub>=0.23",
#   "datasets>=2.19,<4",
#   "soundfile>=0.12",
#   "librosa>=0.10",
#   "pyarrow>=14",
# ]
# ///
"""Smart Turn v3.2's targets on the s6-ear v2 training shards (docs/s6-ear2-prereg.md §5.1).

    uv run --python 3.12 scripts/turn-teacher.py --data DIR --shard 0 [--procs 4] [--min-free-gb 8]
    uv run --python 3.12 scripts/turn-teacher.py --data DIR --shard 0 --check 20
    uv run --python 3.12 scripts/turn-teacher.py --records DIR/conv --tag 00012 [--procs 4]

For shard <tag> (five digits), reads the clips `sky-data fetch-turn` kept
(`DIR/train/manifest-<tag>.jsonl`: id and parquet row) from `DIR/shards/train-<tag>.parquet`,
prepares each one as Pipecat does — HF `datasets` audio decoding at 16 kHz, the last 8 s
left-padded with zeros, WhisperFeatureExtractor(chunk_length=8), normalized — and runs the pinned
`smart-turn-v3.2-cpu.onnx` (the file of the baseline) with three graph outputs appended:

* `linear_28_DequantizeLinear_Output`: the logit z_T (its sigmoid is the `logits` output);
* `sum_1`: the pooled embedding e_T (384);
* `layer_norm_8_DequantizeLinear_Output`: the encoder states (400 x 384), whose last 200 frames
  (the last 4 s) are average-pooled by 8 into S_T (25 x 384).

Writes `DIR/train/teacher-<tag>.bin`, one record per manifest line in its order (the layout of
`sky_infer::turn::store::TEACHER_RECORD`: the logit f32, e_T as 384 f16, S_T as 25 x 384 u8 codes
`round(x / scale) + 128` with `scale = max|x| / 127` per step, then the 25 scales f32), then
`teacher-<tag>.done` with the counts and the largest gap between sigmoid(z_T) and `logits`.
Train and validation clips only: the manifest holds no test clip (§3.2). CPU only (onnxruntime's
CPU provider, one thread per process). `--check N` runs N clips and prints the outputs' check
without writing anything.

`--records` (s6-ear v3, docs/s6-ear3-prereg.md §5): Smart Turn's opinion of the conversation
validation and test records of one shard of `sky-data fetch-conv`, recorded only, never a target.
Each record's tail (`DIR/tails-<tag>.bin`, 8 s of float32 PCM up to o + 0.2 s, already left-padded
with zeros as Pipecat does; `tails-<tag>.jsonl` names them) goes through the same preprocessing and
the same graph; writes `DIR/teacher-conv-<tag>.jsonl` {id, logit, prob} and `.done`, then removes
the tails (the transient files of the shard). The number of tails must be the number of validation
and test records the shard wrote (`conv-<tag>.json`, `written`); a shard that wrote none has no tails
and its `.done` is written with 0 records, and a shard whose tails are missing or short is an error
(exit 1), never 0 records (docs/s6-ear3-prereg.md amendment 3). `scripts/turn3-guard.py teacher` checks
the whole folder after the fact.
"""
import argparse
import json
import multiprocessing as mp
import shutil
import struct
import sys
import time
from pathlib import Path

REPO = "pipecat-ai/smart-turn-v3"
REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
FILE = "smart-turn-v3.2-cpu.onnx"
RATE = 16000
SECONDS = 8
DIM = 384
STEPS = 25
POOL = 8
OUT_PROB = "logits"
OUT_LOGIT = "linear_28_DequantizeLinear_Output"
OUT_POOLED = "sum_1"
OUT_STATES = "layer_norm_8_DequantizeLinear_Output"
RECORD = 4 + DIM * 2 + STEPS * DIM + STEPS * 4
ZERO = 128


def left_pad(x, n=SECONDS * RATE):
    """The last n samples, zeros at the start when shorter (Pipecat's
    audio_utils.truncate_audio_to_last_n_seconds)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    if len(x) >= n:
        return x[-n:]
    return np.concatenate([np.zeros(n - len(x), dtype=np.float32), x])


def pool_states(states):
    """(400, 384) encoder states -> (25, 384): the last 200 frames averaged by 8."""
    import numpy as np

    s = np.asarray(states, dtype=np.float32).reshape(-1, DIM)[-STEPS * POOL:]
    return s.reshape(STEPS, POOL, DIM).mean(axis=1)


def encode_record(logit, pooled, states):
    """One teacher record (the layout of sky_infer::turn::store::encode_teacher)."""
    import numpy as np

    pooled = np.asarray(pooled, dtype=np.float32).reshape(DIM)
    states = np.asarray(states, dtype=np.float32).reshape(STEPS, DIM)
    m = np.abs(states).max(axis=1)
    scale = np.where(m > 0, m / 127.0, 1.0).astype(np.float32)
    codes = np.clip(np.round(states / scale[:, None] + ZERO), 0, 255).astype(np.uint8)
    out = struct.pack("<f", float(logit)) + pooled.astype("<f2").tobytes() + codes.tobytes()
    out += scale.astype("<f4").tobytes()
    assert len(out) == RECORD
    return out


def decode_record(b):
    """The inverse of encode_record (tests)."""
    import numpy as np

    logit = struct.unpack("<f", b[:4])[0]
    at = 4
    pooled = np.frombuffer(b[at : at + DIM * 2], dtype="<f2").astype(np.float32)
    at += DIM * 2
    codes = np.frombuffer(b[at : at + STEPS * DIM], dtype=np.uint8).reshape(STEPS, DIM)
    at += STEPS * DIM
    scale = np.frombuffer(b[at : at + STEPS * 4], dtype="<f4")
    return logit, pooled, (codes.astype(np.float32) - ZERO) * scale[:, None]


_W = {}


def _init():
    import onnx
    import onnxruntime as ort
    from datasets import Audio
    from huggingface_hub import hf_hub_download
    from transformers import WhisperFeatureExtractor

    model = onnx.load(hf_hub_download(REPO, FILE, revision=REVISION))
    have = {o.name for o in model.graph.output}
    for name in (OUT_LOGIT, OUT_POOLED, OUT_STATES):
        if name not in have:
            model.graph.output.append(onnx.helper.make_tensor_value_info(name, onnx.TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    _W["sess"] = ort.InferenceSession(model.SerializeToString(), sess_options=so, providers=["CPUExecutionProvider"])
    _W["fe"] = WhisperFeatureExtractor(chunk_length=SECONDS)
    _W["audio"] = Audio(sampling_rate=RATE)


def score(item):
    """(row, id, audio bytes) -> (row, id, record bytes, |sigmoid(z) - logits|)."""
    import numpy as np

    if not _W:
        _init()
    row, cid, data = item
    a = _W["audio"].decode_example({"bytes": data, "path": None})
    x = left_pad(a["array"])
    feats = _W["fe"](
        x,
        sampling_rate=RATE,
        return_tensors="np",
        padding="max_length",
        max_length=SECONDS * RATE,
        truncation=True,
        do_normalize=True,
    ).input_features.astype(np.float32)
    prob, z, pooled, states = _W["sess"].run([OUT_PROB, OUT_LOGIT, OUT_POOLED, OUT_STATES], {"input_features": feats})
    z = float(np.asarray(z).reshape(-1)[0])
    gap = abs(1.0 / (1.0 + np.exp(-z)) - float(np.asarray(prob).reshape(-1)[0]))
    return row, cid, encode_record(z, np.asarray(pooled).reshape(-1), pool_states(states)), gap


def score_tail(item):
    """(index, id, 8 s of float32 PCM) -> (index, id, logit, prob)."""
    import numpy as np

    if not _W:
        _init()
    idx, cid, data = item
    x = left_pad(np.frombuffer(data, dtype="<f4"))
    feats = _W["fe"](
        x,
        sampling_rate=RATE,
        return_tensors="np",
        padding="max_length",
        max_length=SECONDS * RATE,
        truncation=True,
        do_normalize=True,
    ).input_features.astype(np.float32)
    prob, z = _W["sess"].run([OUT_PROB, OUT_LOGIT], {"input_features": feats})
    return idx, cid, float(np.asarray(z).reshape(-1)[0]), float(np.asarray(prob).reshape(-1)[0])


def tail_items(binf, rows):
    """(index, id, bytes) of each tail, in the order of its lines."""
    size = SECONDS * RATE * 4
    with open(binf, "rb") as f:
        for r in rows:
            f.seek(int(r["index"]) * size)
            b = f.read(size)
            if len(b) != size:
                raise SystemExit(f"{binf}: the tail of {r['id']} is cut short")
            yield int(r["index"]), r["id"], b


def expected_tails(conv, tag):
    """The validation and test records shard `tag` wrote, each with a tail (`conv-<tag>.json`, `written`);
    None without the shard's report."""
    p = Path(conv) / f"conv-{tag}.json"
    if not p.exists():
        return None
    w = json.loads(p.read_text()).get("written") or {}
    return int(w.get("val", 0)) + int(w.get("test", 0))


def run_records(a):
    conv = Path(a.records)
    tag = a.tag
    lines, binf = conv / f"tails-{tag}.jsonl", conv / f"tails-{tag}.bin"
    out, done = conv / f"teacher-conv-{tag}.jsonl", conv / f"teacher-conv-{tag}.done"
    expected = expected_tails(conv, tag)
    if done.exists():
        had = json.loads(done.read_text()).get("records")
        if expected is not None and had != expected:
            # the tails, if any, are kept: they may be the only copy of what is not scored
            raise SystemExit(f"{done}: {had} records, but conv-{tag}.json says the shard wrote {expected} validation "
                             "and test records: remove the .done and score the shard again, or rebuild the shard")
        for p in (lines, binf):
            p.unlink(missing_ok=True)
        print(f"teacher-conv {tag}: done")
        return 0
    if expected is None:
        raise SystemExit(f"no {conv / f'conv-{tag}.json'} (sky-data fetch-conv --shards {int(tag)})")
    if not lines.exists():
        if expected == 0:
            done.write_text(json.dumps({"shard": tag, "records": 0}) + "\n")
            print(f"teacher-conv {tag}: no validation or test record")
            return 0
        raise SystemExit(f"no {lines}: the shard wrote {expected} validation and test records and their tails are "
                         f"gone; rebuild the shard (sky-data fetch-conv --shards {int(tag)})")
    rows = [json.loads(l) for l in lines.read_text().splitlines() if l.strip()]
    if len(rows) != expected:
        raise SystemExit(f"{lines} names {len(rows)} tails, the shard wrote {expected} validation and test records; "
                         f"rebuild the shard (sky-data fetch-conv --shards {int(tag)})")
    t0 = time.time()
    tmp = out.with_suffix(".part")
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.procs) as pool, open(tmp, "w") as f:
        n = 0
        for _, cid, z, p in pool.imap(score_tail, tail_items(binf, rows), chunksize=8):
            f.write(json.dumps({"id": cid, "logit": z, "prob": p}) + "\n")
            n += 1
    if n != len(rows):
        raise SystemExit(f"{n} tails scored, {lines} has {len(rows)}")
    tmp.rename(out)
    done.write_text(json.dumps({"shard": tag, "records": n, "seconds": round(time.time() - t0, 1),
                                "model": f"{REPO}/{FILE}", "revision": REVISION,
                                "use": "recorded only, never a target (docs/s6-ear3-prereg.md §5)"}) + "\n")
    for p in (lines, binf):
        p.unlink(missing_ok=True)
    print(f"teacher-conv {tag}: {n} records")
    return 0


def manifest(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            rows.append((int(v["row"]), v["id"]))
    return rows


def items(parquet, wanted):
    """(row, id, bytes) of the wanted rows, in row order, one record batch at a time."""
    import pyarrow.parquet as pq

    want = dict(wanted)
    f = pq.ParquetFile(parquet)
    row = 0
    for batch in f.iter_batches(batch_size=64, columns=["audio", "id"]):
        audio = batch.column("audio").to_pylist()
        ids = batch.column("id").to_pylist()
        for a, cid in zip(audio, ids):
            if row in want:
                if want[row] != cid:
                    raise SystemExit(f"{parquet} row {row}: id {cid}, the manifest says {want[row]}")
                yield row, cid, a["bytes"]
            row += 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data")
    ap.add_argument("--shard", type=int)
    ap.add_argument("--records", default=None, help="s6-ear v3: DIR/conv of sky-data fetch-conv")
    ap.add_argument("--tag", default=None, help="s6-ear v3: the shard's tag (five digits)")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    ap.add_argument("--check", type=int, default=0, help="score N clips, print the check, write nothing")
    a = ap.parse_args()
    if a.records:
        if not a.tag:
            raise SystemExit("--records needs --tag")
        return run_records(a)
    if a.data is None or a.shard is None:
        raise SystemExit("--data and --shard are needed (or --records and --tag)")
    data = Path(a.data)
    tag = f"{a.shard:05d}"
    man = data / "train" / f"manifest-{tag}.jsonl"
    parquet = data / "shards" / f"train-{tag}.parquet"
    out = data / "train" / f"teacher-{tag}.bin"
    done = data / "train" / f"teacher-{tag}.done"
    if done.exists() and not a.check:
        print(f"teacher {tag}: done")
        return 0
    if not man.exists() or not parquet.exists():
        raise SystemExit(f"need {man} and {parquet} (sky-data fetch-turn --shards {a.shard})")
    wanted = manifest(man)
    if a.check:
        wanted = wanted[: a.check]
    need = len(wanted) * RECORD
    free = shutil.disk_usage(data).free
    if not a.check and free - need < a.min_free_gb * (1 << 30):
        print(f"teacher {tag}: stopped by the disk rule ({free / 1e9:.2f} GB free)", file=sys.stderr)
        return 3
    t0 = time.time()
    ctx = mp.get_context("spawn")
    gaps = []
    tmp = out.with_suffix(".part")
    with ctx.Pool(a.procs) as pool, (open(tmp, "wb") if not a.check else open("/dev/null", "wb")) as f:
        n = 0
        for row, cid, rec, gap in pool.imap(score, items(parquet, wanted), chunksize=8):
            if row != wanted[n][0]:
                raise SystemExit(f"row {row} out of order (expected {wanted[n][0]})")
            f.write(rec)
            gaps.append(gap)
            n += 1
            if a.check:
                z, pooled, states = decode_record(rec)
                print(json.dumps({"row": row, "id": cid, "logit": z, "sigmoid_gap": gap,
                                  "pooled_norm": float((pooled ** 2).sum() ** 0.5),
                                  "states_abs_mean": float(abs(states).mean())}))
    if n != len(wanted):
        raise SystemExit(f"{n} clips scored, the manifest has {len(wanted)}")
    rep = {"shard": tag, "clips": n, "max_sigmoid_gap": max(gaps) if gaps else 0.0,
           "seconds": round(time.time() - t0, 1), "procs": a.procs, "model": f"{REPO}/{FILE}",
           "revision": REVISION, "record_bytes": RECORD}
    if a.check:
        print(json.dumps(rep))
        return 0
    tmp.rename(out)
    done.write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    sys.exit(main())
