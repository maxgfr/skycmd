# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "salesforce-gift-eval @ git+https://github.com/SalesforceAIResearch/gift-eval@cfaefe09c6bb4195b38c720bc3c8ea382048241d",
#   "gluonts>=0.15,<0.17",
#   "huggingface_hub>=0.25",
#   "numpy>=1.26,<3",
#   "pandas>=2.1",
#   "pyarrow>=15,<18",
# ]
# ///
"""GIFT-Eval for s6-cast and the reference forecasters (docs/SPEC-S6-CAST.md §6).

The forecasts come from Rust (`sky-cast-eval`); this script only moves data and scores it with
the official protocol (gift-eval's datasets, gluonts' metrics), so the numbers are comparable
with the leaderboard and with TinyCast's published ones:

    uv run --python 3.12 scripts/gift-eval.py download --data data/gift-eval-src
    uv run --python 3.12 scripts/gift-eval.py export   --data data/gift-eval-src --out data/gift-eval
    sky-cast-eval run --model model.gguf --windows data/gift-eval --out data/gift-eval-out/<name>
    uv run --python 3.12 scripts/gift-eval.py score    --data data/gift-eval-src \
        --forecasts data/gift-eval-out/<name> --name <name>

`export` writes, per configuration, `<config>.bin` (every test context as f32, concatenated) and
`<config>.json` (offsets, lengths, prediction length, season, frequency). `score` reads the
quantiles `sky-cast-eval` wrote (`<config>.q.bin`, f32 `[windows][prediction_length][9]`),
replays them through gluonts' `evaluate_model` exactly as TinyCast's driver does, writes the
leaderboard-format CSV, and normalizes MASE, WQL (CRPS) and MSIS by the seasonal-naive reference
(`eval/gift-eval/seasonal_naive.csv`, from TinyCast, Apache-2.0) with geometric means.

GIFT-Eval's data is released for research use (Apache-2.0 metadata); it stays local and is never
redistributed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# Points of history kept per window: the 2048 the models read, times the largest period-alignment
# downsampling the reference predictors apply (k up to 7), rounded up.
KEEP = 16384
REPO = Path(__file__).resolve().parent.parent
REFERENCE = REPO / "eval" / "gift-eval"

# The GIFT-Eval layout (as in gift-eval's notebooks and TinyCast's driver).
PRETTY = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}
SHORT = (
    "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly "
    "electricity/15T electricity/H electricity/D electricity/W "
    "solar/10T solar/H solar/D solar/W hospital covid_deaths "
    "us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W "
    "temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D "
    "car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D "
    "ett1/15T ett1/H ett1/D ett1/W ett2/W ett2/D jena_weather/10T jena_weather/H jena_weather/D "
    "bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H "
    "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
MED_LONG = (
    "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H "
    "LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H "
    "jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T "
    "bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
).split()
COLUMNS = [
    "dataset", "model", "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]",
    "eval_metrics/MAE[0.5]", "eval_metrics/MASE[0.5]", "eval_metrics/MAPE[0.5]",
    "eval_metrics/sMAPE[0.5]", "eval_metrics/MSIS", "eval_metrics/RMSE[mean]",
    "eval_metrics/NRMSE[mean]", "eval_metrics/ND[0.5]",
    "eval_metrics/mean_weighted_sum_quantile_loss", "domain", "num_variates",
]
NORMALIZED = [
    ("nGMASE", "eval_metrics/MASE[0.5]"),
    ("nWQL", "eval_metrics/mean_weighted_sum_quantile_loss"),
    ("nMSIS", "eval_metrics/MSIS"),
]


def properties() -> dict:
    return json.loads((REFERENCE / "dataset_properties.json").read_text())


def configs(only: list[str] | None = None):
    """(ds_name, term, config_key, ds_key) for the 97 configurations."""
    props = properties()
    for ds_name in sorted(set(SHORT + MED_LONG)):
        if "/" in ds_name:
            base, freq = ds_name.split("/")
            key = PRETTY.get(base.lower(), base.lower())
        else:
            key = PRETTY.get(ds_name.lower(), ds_name.lower())
            freq = props[key]["frequency"]
        for term in ("short", "medium", "long"):
            if term != "short" and ds_name not in MED_LONG:
                continue
            cfg = f"{key}/{freq}/{term}"
            if only and cfg not in only:
                continue
            yield ds_name, term, cfg, key


def file_stem(cfg: str) -> str:
    return cfg.replace("/", "__")


def dataset(ds_name: str, term: str):
    from gift_eval.data import Dataset

    probe = Dataset(name=ds_name, term=term, to_univariate=False)
    return Dataset(name=ds_name, term=term, to_univariate=probe.target_dim != 1)


def cmd_download(a) -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="Salesforce/GiftEval", repo_type="dataset", local_dir=a.data
    )
    print(f"GIFT-Eval in {path}")


def cmd_export(a) -> None:
    from gluonts.time_feature import get_seasonality

    os.environ["GIFT_EVAL"] = str(Path(a.data).resolve())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for ds_name, term, cfg, _ in configs(a.configs):
        ds = dataset(ds_name, term)
        offsets, lengths = [], []
        pos = 0
        with open(out / f"{file_stem(cfg)}.bin", "wb") as f:
            for entry in ds.test_data.input:
                y = np.asarray(entry["target"], dtype=np.float32)
                if y.ndim != 1:
                    raise SystemExit(f"{cfg}: a test input is not univariate")
                y = y[-KEEP:]
                f.write(y.astype("<f4").tobytes())
                offsets.append(pos)
                lengths.append(int(y.shape[0]))
                pos += int(y.shape[0])
        meta = {
            "config": cfg,
            "dataset": ds_name,
            "term": term,
            "freq": ds.freq,
            "season": int(get_seasonality(ds.freq)),
            "prediction_length": int(ds.prediction_length),
            "windows": len(offsets),
            "offsets": offsets,
            "lengths": lengths,
        }
        (out / f"{file_stem(cfg)}.json").write_text(json.dumps(meta))
        total += len(offsets)
        print(f"{cfg}: {len(offsets)} windows, horizon {ds.prediction_length}", flush=True)
    print(f"exported {total} windows to {out}")


class Replay:
    """A gluonts predictor that hands back precomputed quantiles, window by window."""

    def __init__(self, quantiles: np.ndarray, prediction_length: int):
        self.q = quantiles
        self.prediction_length = prediction_length
        self.i = 0

    def predict(self, dataset, **kwargs):
        from gluonts.model.forecast import QuantileForecast

        for entry in dataset:
            q = self.q[self.i]
            self.i += 1
            start = entry["start"] + len(entry["target"])
            yield QuantileForecast(
                forecast_arrays=q.T.astype(np.float64),
                start_date=start,
                forecast_keys=[str(l) for l in LEVELS],
                item_id=entry.get("item_id"),
            )


def metrics():
    from gluonts.ev.metrics import (
        MAE, MAPE, MASE, MSE, MSIS, ND, NRMSE, RMSE, SMAPE, MeanWeightedSumQuantileLoss,
    )

    return [
        MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(), MAPE(), SMAPE(),
        MSIS(), RMSE(), NRMSE(), ND(), MeanWeightedSumQuantileLoss(quantile_levels=LEVELS),
    ]


def cmd_score(a) -> None:
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    os.environ["GIFT_EVAL"] = str(Path(a.data).resolve())
    props = properties()
    fdir = Path(a.forecasts)
    rows = []
    for ds_name, term, cfg, key in configs(a.configs):
        qfile = fdir / f"{file_stem(cfg)}.q.bin"
        if not qfile.exists():
            print(f"{cfg}: no forecasts, skipped", file=sys.stderr)
            continue
        ds = dataset(ds_name, term)
        h = int(ds.prediction_length)
        q = np.fromfile(qfile, dtype="<f4").reshape(-1, h, len(LEVELS))
        res = evaluate_model(
            Replay(q, h),
            test_data=ds.test_data,
            metrics=metrics(),
            batch_size=1024,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(ds.freq),
        )
        row = [
            cfg, a.name,
            *(res[c].iloc[0] for c in (
                "MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]", "sMAPE[0.5]",
                "MSIS", "RMSE[mean]", "NRMSE[mean]", "ND[0.5]", "mean_weighted_sum_quantile_loss",
            )),
            props[key]["domain"], props[key]["num_variates"],
        ]
        rows.append(row)
        print(f"{cfg}: MASE {row[5]:.4f} WQL {row[12]:.4f}", flush=True)
    out = fdir / "all_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    summary = summarize(out)
    (fdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def summarize(results_csv: Path) -> dict:
    """Geometric means over the configurations of each metric's ratio to seasonal naive."""
    ref = {}
    with open(REFERENCE / "seasonal_naive.csv", newline="") as f:
        for row in csv.DictReader(f):
            ref[row["dataset"]] = row
    logs = {name: [] for name, _ in NORMALIZED}
    missing = []
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            base = ref.get(row["dataset"])
            if base is None:
                missing.append(row["dataset"])
                continue
            for name, col in NORMALIZED:
                try:
                    num, den = float(row[col]), float(base[col])
                except (TypeError, ValueError):
                    continue
                if num > 0 and den > 0 and math.isfinite(num) and math.isfinite(den):
                    logs[name].append(math.log(num / den))
    out = {name: (math.exp(sum(v) / len(v)) if v else None) for name, v in logs.items()}
    out["configs"] = {name: len(v) for name, v in logs.items()}
    out["unmatched"] = missing
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--data", required=True)
    e = sub.add_parser("export")
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--configs", nargs="*")
    s = sub.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--forecasts", required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--configs", nargs="*")
    a = p.parse_args()
    {"download": cmd_download, "export": cmd_export, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
