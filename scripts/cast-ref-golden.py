# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "torch==2.8.0",
#   "numpy==2.1.3",
#   "pandas==2.3.3",
#   "gluonts==0.16.2",
#   "safetensors==0.6.2",
#   "packaging>=24",
# ]
# ///
"""Golden forecasts of the official TinyCast predictor, for the Rust port's parity test.

TinyCast (RAWS Labs, Apache-2.0, https://github.com/raws-labs/tinycast) is imported from a local
clone of its repository, never copied here, and run on its released weights
(https://huggingface.co/raws-labs/tinycast: `model.safetensors` + `config.json`):

    uv run --python 3.12 scripts/cast-ref-golden.py --tinycast /path/to/tinycast \
        --weights /path/to/tinycast-weights --out crates/sky-infer/tests/golden/tinycast.json

The deployed path is `TinyCastPredictor.predict` on CPU (fp32): left-pad or truncate to 2048,
NaN imputation, 48-step autoregressive rollout feeding the median back, optional
flip-invariance symmetrization, quantiles sorted per step. Eight deterministic contexts (short,
long, longer than the window, with gaps, constant, strongly periodic, large offset) are
forecast at horizons 48 (one chunk) and 100 (three chunks), with and without flip.
`crates/sky-infer/tests/golden_tinycast.rs` checks `sky_infer::cast::tinycast` against the file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def f32_list(a: np.ndarray) -> list:
    """f32 values as their shortest round-trip decimal (NaN as null): small, exact JSON."""
    return [None if np.isnan(v) else float(str(np.float32(v))) for v in np.asarray(a, np.float32)]


def contexts() -> list[tuple[str, np.ndarray]]:
    """The test contexts, rounded to 3 decimals so the JSON stays small."""
    rng = np.random.default_rng(20261026)
    out = []

    t = np.arange(30)
    out.append(("short-30", 3.0 + np.sin(2 * np.pi * t / 7) + 0.1 * rng.standard_normal(30)))

    t = np.arange(500)
    out.append((
        "daily-500",
        10 + 0.01 * t + 2 * np.sin(2 * np.pi * t / 24) + 0.3 * rng.standard_normal(500),
    ))

    t = np.arange(2048)
    out.append((
        "multi-2048",
        50
        + 8 * np.sin(2 * np.pi * t / 24)
        + 4 * np.sin(2 * np.pi * t / 168 + 0.5)
        + rng.standard_normal(2048),
    ))

    out.append(("walk-2600", np.cumsum(rng.standard_normal(2600))))

    t = np.arange(700)
    x = 1 + np.cos(2 * np.pi * t / 12) + 0.2 * rng.standard_normal(700)
    x[100:140] = np.nan
    x[rng.choice(700, 35, replace=False)] = np.nan
    x[-3:-1] = np.nan
    out.append(("gaps-700", x))

    out.append(("constant-300", np.full(300, 5.0)))

    t = np.arange(1200)
    out.append(("square-p7-1200", np.where(t % 7 < 3, 1.0, -1.0)))

    t = np.arange(200)
    x = 1e4 + 50 * np.exp(t / 120) + 5 * rng.standard_normal(200)
    x[:25] = np.nan
    out.append(("offset-nanhead-200", x))

    return [(n, np.round(x, 3).astype(np.float32)) for n, x in out]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tinycast", required=True, help="clone of github.com/raws-labs/tinycast")
    ap.add_argument("--weights", required=True, help="directory with model.safetensors, config.json")
    ap.add_argument("--out", default="crates/sky-infer/tests/golden/tinycast.json")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.tinycast).resolve()))
    import pandas as pd
    import torch

    torch.set_num_threads(1)
    from tinycast import TinyCastPredictor

    weights = Path(args.weights) / "model.safetensors"
    cases = []
    ctxs = contexts()
    for horizon in (48, 100):
        for flip in (False, True):
            pred = TinyCastPredictor(
                prediction_length=horizon,
                checkpoint_path=str(weights),
                device="cpu",
                force_flip_invariance=flip,
            )
            data = [
                {"target": x, "start": pd.Period("2020-01-01 00:00", freq="h")}
                for _, x in ctxs
            ]
            for (name, x), fc in zip(ctxs, pred.predict(data)):
                q = np.asarray(fc.forecast_array, np.float32).T  # [horizon][levels]
                assert q.shape == (horizon, 9), q.shape
                cases.append({
                    "context": name,
                    "horizon": horizon,
                    "flip": flip,
                    "quantiles": f32_list(q.reshape(-1)),
                })

    doc = {
        "source": "TinyCastPredictor.predict (github.com/raws-labs/tinycast, Apache-2.0), "
        "CPU fp32, weights huggingface.co/raws-labs/tinycast",
        "torch": torch.__version__,
        "levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "contexts": {name: f32_list(x) for name, x in ctxs},
        "cases": cases,
    }
    out = Path(args.out)
    out.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"{out}: {len(cases)} cases, {out.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
