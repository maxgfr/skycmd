# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval@cfaefe09c6bb4195b38c720bc3c8ea382048241d",
#   "gluonts>=0.15,<0.17",
#   "numpy>=1.26,<3",
#   "pandas>=2.1",
#   "pyarrow>=15,<18",
# ]
# ///
"""The per-window items of s6-cast v2's report (docs/s6-cast2-prereg.md §9), by the method of
docs/s6-cast2-study.md §7, for scripts/s6-cast2-gate.py (`--diagnostics`):

    uv run --python 3.12 scripts/s6-cast2-diagnostics.py --data data/gift-eval-src \
        --windows data/gift-eval --out runs/s6-cast2-pico/diagnostics.json \
        --model v1=data/gift-eval-out/s6-cast-pico --model v2=data/gift-eval-out/s6-cast2-pico \
        --model port=data/gift-eval-out/tinycast --model sn=data/gift-eval-out/seasonal-naive

- Coverage: per quantile level, and for the 80% interval, the share of valid label points at or
  below the quantile (inside the interval), per configuration, averaged over the configurations
  of a term.
- The horizon profile: each configuration's horizon cut into tenths; per tenth, the ratio of the
  summed scaled errors of two models, averaged geometrically over the configurations of a term.
- Window-level splits: per configuration, the ratio of v2's to the port's summed scaled errors
  over the windows of one kind (a period detected or not by TinyCast's detector on the port's
  prepared window of the last 2,048 points; the period aligned to the 32-point patch grid or
  not; the observed points in the last 2,048), averaged geometrically over the configurations
  with at least 20 such windows.

The scaled error of a window is the mean absolute error of its median over the valid label
points divided by the in-sample seasonal-naive error of its history at the dataset's gluonts
seasonality (the MASE of the window). Only the forecasts' quantile files and GIFT-Eval's labels
are read; nothing is scored against a bar here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TERMS = ["short", "medium", "long"]
L = 2048


def gift_eval_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("gift_eval_script", Path(__file__).resolve().parent / "gift-eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prepare_context(h: np.ndarray) -> np.ndarray:
    """TinyCast's prepared window (sky_infer::cast::tinycast::prepare_context)."""
    h = np.asarray(h, dtype=np.float32)
    a = h[-L:] if h.size >= L else np.concatenate([np.full(L - h.size, h[0] if h.size else 0.0, np.float32), h])
    nan = np.isnan(a)
    if nan.any():
        valid = np.flatnonzero(~nan)
        if valid.size >= 2:
            a = np.interp(np.arange(L), valid, a[valid].astype(np.float64)).astype(np.float32)
        else:
            a = np.where(nan, a[valid[0]] if valid.size else 0.0, a).astype(np.float32)
    return np.nan_to_num(a, nan=0.0)


def dominant_period(h: np.ndarray, alpha: float = 0.05) -> int:
    """p1 of TinyCast's significant_periods on the min-max normalized prepared window (0: none)."""
    x = prepare_context(h)
    x = np.where(np.isfinite(x), x, 0.0)
    lo, hi = x.min(), x.max()
    x = (x - lo) / max(hi - lo, 1e-5)
    n_fft = 1 << (L - 1).bit_length()
    spec = np.fft.rfft((x - x.mean()).astype(np.float64), n=n_fft)
    power = np.abs(spec[1 : n_fft // 2 + 1]) ** 2
    score = power / max(power.sum(), 1e-12)
    n_bins = power.size
    k_lo, k_hi = n_fft // (L // 2) - 1, min(n_bins - 1, n_fft // 2 - 1)
    thr = math.log(n_bins / alpha) / n_bins
    i = np.arange(1, n_bins - 1)
    ok = (i >= k_lo) & (i <= k_hi) & (score[i] > score[i - 1]) & (score[i] > score[i + 1]) & (score[i] >= thr)
    if not ok.any():
        return 0
    best = i[ok][np.argmax(score[i[ok]])]
    return int(round(n_fft / (best + 1)))


def scale(hist: np.ndarray, m: int) -> float | None:
    m = max(m, 1)
    d = np.abs(hist[m:] - hist[:-m]) if hist.size > m else np.empty(0)
    d = d[np.isfinite(d)]
    return float(d.mean()) if d.size and d.mean() > 0 else None


def gmean(v):
    v = [x for x in v if x and x > 0 and math.isfinite(x)]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--windows", required=True)
    p.add_argument("--model", action="append", required=True, help="label=forecast dir")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    from gluonts.time_feature import get_seasonality

    ge = gift_eval_module()
    os.environ["GIFT_EVAL"] = str(Path(a.data).resolve())
    models = dict(m.split("=", 1) for m in a.model)
    cov = {lab: {t: [] for t in TERMS} for lab in models}
    tenths: dict[tuple, dict[str, list]] = {}
    splits: dict[str, list] = {}
    for ds_name, term, cfg, _ in ge.configs(None):
        ds = ge.dataset(ds_name, term)
        h = int(ds.prediction_length)
        m = int(get_seasonality(ds.freq))
        pairs = [(np.asarray(i["target"], np.float32), np.asarray(l["target"], np.float32)) for i, l in ds.test_data]
        q = {}
        for lab, d in models.items():
            f = Path(d) / f"{ge.file_stem(cfg)}.q.bin"
            if f.exists():
                q[lab] = np.fromfile(f, dtype="<f4").reshape(len(pairs), h, len(LEVELS))
        if not q:
            continue
        labels = np.stack([y for _, y in pairs])
        valid = np.isfinite(labels)
        for lab, qq in q.items():
            per = [float(((labels <= qq[:, :, k]) & valid).sum() / max(valid.sum(), 1)) for k in range(len(LEVELS))]
            inside = ((labels >= qq[:, :, 0]) & (labels <= qq[:, :, 8]) & valid).sum() / max(valid.sum(), 1)
            cov[lab][term].append(per + [float(inside)])
        # scaled errors per window and step
        sc = np.array([scale(hh, m) or np.nan for hh, _ in pairs])
        err = {lab: np.abs(labels - qq[:, :, 4]) / sc[:, None] for lab, qq in q.items()}
        cuts = np.array_split(np.arange(h), 10)
        for num, den in (("v2", "port"), ("v2", "sn")):
            if num in err and den in err:
                for k, idx in enumerate(cuts):
                    e1 = np.nansum(np.where(valid[:, idx], err[num][:, idx], np.nan))
                    e0 = np.nansum(np.where(valid[:, idx], err[den][:, idx], np.nan))
                    tenths.setdefault((term, f"{num} / {den}"), {}).setdefault(k, []).append(e1 / e0 if e0 > 0 else None)
        if "v2" in err and "port" in err:
            ctxs = np.fromfile(Path(a.windows) / f"{ge.file_stem(cfg)}.bin", dtype="<f4")
            meta = json.loads((Path(a.windows) / f"{ge.file_stem(cfg)}.json").read_text())
            kinds: dict[str, list[int]] = {}
            for w, (o, n) in enumerate(zip(meta["offsets"], meta["lengths"])):
                c = ctxs[o : o + n]
                p1 = dominant_period(c)
                obs = int(np.isfinite(c[-L:]).sum())
                ks = ["a period detected" if p1 else "no period detected"]
                if p1:
                    ks.append("period < 32" if p1 < 32 else (
                        "period a multiple of 32 (+-1)" if min(p1 % 32, 32 - p1 % 32) <= 1 else "period >= 32, not a multiple of 32"))
                ks.append("observed < 128" if obs < 128 else "observed 128-511" if obs < 512 else
                          "observed 512-2,047" if obs < L else "observed 2,048")
                for k in ks:
                    kinds.setdefault(k, []).append(w)
            for k, ws in kinds.items():
                if len(ws) >= 20:
                    e1, e0 = np.nansum(err["v2"][ws] * valid[ws]), np.nansum(err["port"][ws] * valid[ws])
                    if e0 > 0:
                        splits.setdefault(k, []).append(e1 / e0)
        print(f"{cfg}: {len(pairs)} windows", flush=True)
    tables = []
    rows = []
    for lab in models:
        for t in TERMS:
            if cov[lab][t]:
                mean = np.mean(np.array(cov[lab][t]), axis=0)
                rows.append([t, lab] + [fmt(v) for v in mean])
    tables.append(["Coverage by term (share of labels at or below each level; last: inside the 80% interval)", rows,
                   ["term", "model"] + [str(l) for l in LEVELS] + ["80% interval"]])
    rows = [[t, r] + [fmt(gmean(v.get(k, []))) for k in range(10)] for (t, r), v in sorted(tenths.items())]
    tables.append(["The horizon profile by tenths (ratio of summed scaled errors)", rows,
                   ["term", "ratio"] + [str(k + 1) for k in range(10)]])
    rows = [[k, str(len(v)), fmt(gmean(v))] for k, v in sorted(splits.items())]
    tables.append(["Window-level splits, v2 / TinyCast port (configurations with >= 20 such windows)", rows,
                   ["windows", "configs", "v2 / port MASE"]])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"tables": tables}, indent=2) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
