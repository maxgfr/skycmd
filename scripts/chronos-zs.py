# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "fev==0.6.0",
#   "datasets>=2.15,<4",
#   "numpy>=1.26,<3",
#   "pyyaml>=6",
# ]
# ///
"""Chronos-ZS for s6-cast and the reference forecasters (docs/s6-cast2-prereg.md §7.3).

The 27 tasks of the Chronos zero-shot benchmark as fev defines them (`eval/chronos-zs/tasks.yaml`,
autogluon/fev, Apache-2.0), scored with fev 0.6.0's own metrics (MASE and WQL over the 9 deciles),
then divided task by task by fev's seasonal-naive results (`eval/chronos-zs/seasonal_naive.csv`)
and averaged geometrically: the relative MASE and WQL TinyCast reports (0.8800 / 0.7218 for the
port's model, with flip). The forecasts come from Rust (`sky-cast-eval run`), as for GIFT-Eval:

    uv run --python 3.12 scripts/chronos-zs.py export --cache data/chronos-zs-src --out data/chronos-zs
    sky-cast-eval run --model m.gguf --windows data/chronos-zs --out data/chronos-zs-out/<name> [--flip]
    uv run --python 3.12 scripts/chronos-zs.py score --cache data/chronos-zs-src \
        --forecasts data/chronos-zs-out/<name> --name <name>
    uv run --python 3.12 scripts/chronos-zs.py check --forecasts data/chronos-zs-out/tinycast

`export` writes, per task and evaluation window, `<task>__w<k>.bin` (every context as f32, its
last 16,384 points) and `<task>__w<k>.json` (offsets, lengths, prediction length, season), the
format `sky-cast-eval` reads. `score` replays the quantiles (`<stem>.q.bin`, f32
`[series][horizon][9]`) through `fev.Task.evaluation_summary`, the median as the point forecast,
and writes `results.csv` and `summary.json`. `check` is the harness check of §7.3: the TinyCast
port with flip must give 0.880 and 0.722 within 0.02.

The benchmark's data stays local and is never trained on or redistributed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
KEEP = 16384
REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "eval" / "chronos-zs"
# TinyCast's published Chronos-ZS row (arXiv 2608.15767; raws-labs/tinycast
# evidence/chronos-zs/README.md) and the tolerance of the harness check (§7.3).
PUBLISHED = {"MASE": 0.880, "WQL": 0.722}
CHECK_TOL = 0.02


def benchmark():
    import fev

    return fev.Benchmark.from_yaml(str(REFERENCE / "tasks.yaml"))


def task_name(task) -> str:
    return task.dataset_config


def univariate_targets(past) -> tuple[str, list[np.ndarray]]:
    """The target column of a window's past data and its series (fev turns multivariate tasks
    into univariate ones through `generate_univariate_targets_from`)."""
    cols = [c for c in past.column_names if c == "target"]
    if not cols:
        raise SystemExit(f"no target column in {past.column_names}")
    return cols[0], [np.asarray(v, dtype=np.float32) for v in past.with_format("numpy")["target"]]


def cmd_tasks(a) -> None:
    for t in benchmark().tasks:
        print(f"{task_name(t)}: horizon {t.horizon}, seasonality {t.seasonality}, windows {t.num_windows}")


def cmd_export(a) -> None:
    import os

    if a.cache:
        os.environ["HF_DATASETS_CACHE"] = str(Path(a.cache).resolve())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for task in benchmark().tasks:
        for k, window in enumerate(task.iter_windows()):
            past, _ = window.get_input_data()
            _, series = univariate_targets(past)
            stem = f"{task_name(task)}__w{k}"
            offsets, lengths, pos = [], [], 0
            with open(out / f"{stem}.bin", "wb") as f:
                for y in series:
                    y = y[-KEEP:]
                    f.write(y.astype("<f4").tobytes())
                    offsets.append(pos)
                    lengths.append(int(y.shape[0]))
                    pos += int(y.shape[0])
            meta = {
                "config": stem,
                "task": task_name(task),
                "window": k,
                "season": int(task.seasonality),
                "prediction_length": int(task.horizon),
                "windows": len(offsets),
                "offsets": offsets,
                "lengths": lengths,
            }
            (out / f"{stem}.json").write_text(json.dumps(meta))
            total += len(offsets)
            print(f"{stem}: {len(offsets)} series, horizon {task.horizon}", flush=True)
    print(f"exported {total} series to {out}")


def cmd_score(a) -> None:
    import os

    if a.cache:
        os.environ["HF_DATASETS_CACHE"] = str(Path(a.cache).resolve())
    fdir = Path(a.forecasts)
    rows = []
    for task in benchmark().tasks:
        preds = []
        missing = False
        for k, window in enumerate(task.iter_windows()):
            qfile = fdir / f"{task_name(task)}__w{k}.q.bin"
            if not qfile.exists():
                missing = True
                break
            q = np.fromfile(qfile, dtype="<f4").reshape(-1, task.horizon, len(LEVELS)).astype(np.float64)
            med = LEVELS.index(0.5)
            preds.append(
                [
                    {"predictions": q[i, :, med], **{str(l): q[i, :, j] for j, l in enumerate(LEVELS)}}
                    for i in range(q.shape[0])
                ]
            )
        if missing:
            print(f"{task_name(task)}: no forecasts, skipped", file=sys.stderr)
            continue
        s = task.evaluation_summary(preds, model_name=a.name)
        rows.append({"task_name": task_name(task), "MASE": s["MASE"], "WQL": s["WQL"],
                     "fev_version": s["fev_version"], "dataset_fingerprint": s["dataset_fingerprint"]})
        print(f"{task_name(task)}: MASE {s['MASE']:.4f} WQL {s['WQL']:.4f}", flush=True)
    out = fdir / "results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["task_name"])
        w.writeheader()
        w.writerows(rows)
    summary = summarize(out)
    (fdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def summarize(results_csv: Path) -> dict:
    """Relative MASE and WQL: geometric means over the tasks of the ratio to fev's seasonal naive."""
    ref = {}
    with open(REFERENCE / "seasonal_naive.csv", newline="") as f:
        for row in csv.DictReader(f):
            ref[row["task_name"]] = row
    logs = {"MASE": [], "WQL": []}
    unmatched = []
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            base = ref.get(row["task_name"])
            if base is None:
                unmatched.append(row["task_name"])
                continue
            for m in logs:
                num, den = float(row[m]), float(base[m])
                if num > 0 and den > 0 and math.isfinite(num) and math.isfinite(den):
                    logs[m].append(math.log(num / den))
    out = {f"relative_{m}": (math.exp(sum(v) / len(v)) if v else None) for m, v in logs.items()}
    out["tasks"] = {m: len(v) for m, v in logs.items()}
    out["unmatched"] = unmatched
    out["complete"] = all(len(v) == 27 for v in logs.values())
    return out


def cmd_check(a) -> None:
    s = json.loads((Path(a.forecasts) / "summary.json").read_text())
    ok = s.get("complete", False)
    lines = [f"tasks scored: {s.get('tasks')}{'' if ok else ' (not all 27)'}"]
    for m, want in PUBLISHED.items():
        got = s.get(f"relative_{m}")
        good = got is not None and abs(got - want) <= CHECK_TOL
        ok &= good
        lines.append(f"relative {m}: {got} (published {want}, tolerance {CHECK_TOL}): {'ok' if good else 'FAIL'}")
    naive = None
    if a.naive and (Path(a.naive) / "summary.json").exists():
        # reported, not the criterion: our seasonal naive copies its point forecast on every
        # level, so its relative WQL is not 1 (as on GIFT-Eval, 0.888)
        naive = json.loads((Path(a.naive) / "summary.json").read_text())
        lines.append(f"our seasonal naive: relative MASE {naive.get('relative_MASE')}, "
                     f"relative WQL {naive.get('relative_WQL')} (reported)")
    verdict = {"validated": bool(ok), "criterion": "docs/s6-cast2-prereg.md §7.3: the TinyCast port with flip "
               "within 0.02 of 0.880 / 0.722", "lines": lines, "summary": s, "seasonal_naive": naive,
               "checked": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z")}
    (Path(a.forecasts) / "check.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print("\n".join(lines))
    print("harness validated" if ok else "harness NOT validated")
    sys.exit(0 if ok else 1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("tasks")
    e = sub.add_parser("export")
    e.add_argument("--cache")
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--cache")
    s.add_argument("--forecasts", required=True)
    s.add_argument("--name", required=True)
    c = sub.add_parser("check")
    c.add_argument("--forecasts", required=True)
    c.add_argument("--naive", help="our seasonal naive's forecasts, reported next to the check")
    a = p.parse_args()
    {"tasks": cmd_tasks, "export": cmd_export, "score": cmd_score, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    main()
