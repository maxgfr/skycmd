# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "torch==2.14.1",
#   "onnxscript==0.7.2",
#   "onnx==1.23.2",
#   "onnxruntime==1.31.0",
#   "numpy>=1.24,<3",
#   "safetensors>=0.4",
#   "huggingface_hub>=0.23",
#   "timestral @ git+https://github.com/mistralai/timestral@a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
# ]
# ///
"""Export Mistral's Timestral-5M to ONNX for the browser (onnxruntime-web), univariate, no covariates.

    uv run --python 3.12 scripts/timestral-onnx.py [--cache-dir DIR] [--out-dir site/public/models/ref]

The graph reproduces `TimestralForecaster.forecast(context, prediction_length=H)` with the
forecaster defaults (symmetric averaging, non-negative clipping, real-range cap), for one
series and a fixed context of `--context` steps (default 2048):

    input   context    float32 [1, C]      raw values, NaN (or +/-inf) = missing / left padding
    output  quantiles  float32 [1, H, 21]  raw units, sorted along the last axis

Everything (causal float64 scaling, asinh squash, mirrored pass, advancing refine frame,
real-range cap, clipping) runs inside the graph. H defaults to 128 steps (4 patches): the
decoder tokens are causal among themselves and the refine frame of patch j only reads
patches < j, so the first H steps of the native 1024-step block do not depend on the
rest; the script checks this against the official forecaster.

Then an int8 dynamic-quantized variant is built and the smaller file is kept only if its
parity stays under `--int8-tolerance`. Writes `timestral-5m.onnx`, `timestral-5m.json`
and `TIMESTRAL-LICENSE.txt` to `--out-dir`.

The weights are Mistral AI's, under their Modified MIT license (attribution notice
required; not licensed to companies above $20M monthly revenue). The `timestral` package
(Apache-2.0) is imported, not vendored.
"""

import argparse
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from huggingface_hub import snapshot_download
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import Tensor, nn

import timestral
from timestral import TimestralForecaster, load_model
from timestral.masks import build_forecast_positions, build_forecast_time_mask, build_two_stream_variate_mask
from timestral.scaler import denormalize_quantiles

HF_REPO = "mistralai/Timestral-5M"
HF_REVISION = "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f"  # main on 2026-10-15
TIMESTRAL_COMMIT = "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"
OPSET = 18
INPUT_NAME = "context"
OUTPUT_NAME = "quantiles"
REPO_ROOT = Path(__file__).resolve().parent.parent


class UnivariateTimestral(nn.Module):
    """`[1, C]` raw context with NaN = missing to `[1, H, Q]` raw-space quantiles.

    Same computation as `TimestralForecaster.forecast` for one univariate series without
    covariates and `H <= block_size * patch_size` (a single decode block). The two clampsym
    passes (context and its negation) run as two rows of one batch, which is what the
    forecaster does for a batch of two series: rows of different series never attend each
    other. Three trace-unfriendly spots of the package are expressed differently, with the
    same values:

    - forward fill of missing values: every consumer of the context (causal statistics,
      patch embedding, refine frame, cap) weights unobserved steps by 0, so 0 stands in for
      the forward-filled value;
    - `compute_enc_key_validity` (`cummax`): a patch is a valid key once the series has
      started, i.e. `cumsum(observed) > 0`;
    - `group_first_observed_patch` (`unique` + `scatter_reduce`): one row per series, so it
      is the first observed patch of the row (`argmax`).
    """

    def __init__(self, forecaster: TimestralForecaster, context_length: int, horizon: int) -> None:
        super().__init__()
        c = forecaster.config
        assert c.attend_missing_patches, "this wrapper assumes attend_missing_patches=True"
        assert context_length % c.patch_size == 0 and horizon % c.patch_size == 0
        assert 1 <= horizon // c.patch_size <= forecaster.per_shot_horizon_patches
        assert context_length <= forecaster.max_context
        self.forecaster = forecaster
        self.model = forecaster.model
        self.config = c
        self.context_length = context_length
        self.horizon_patches = horizon // c.patch_size
        self.register_buffer("group_ids", torch.arange(2), persistent=False)

    def _predict(self, inputs: Tensor, observed: Tensor) -> Tensor:
        """`TimestralModel._predict_context_only` for `(R, T)` rows, one series per row."""
        c, m = self.config, self.model
        ps, h = c.patch_size, self.horizon_patches
        rows, t = inputs.shape
        ll = t // ps
        scaled, loc, scale = m.scale(inputs, observed)
        x_enc = m.representation(scaled, observed)
        x_dec = m.representation.forecast_tokens(rows, h)
        obs_per_patch = observed.unflatten(-1, (ll, ps)).any(dim=-1)
        key_valid = torch.cumsum(obs_per_patch.to(torch.int32), dim=-1) > 0
        time_mask = build_forecast_time_mask(ll, h, key_valid, inputs.device)
        variate_mask = build_two_stream_variate_mask(self.group_ids, key_valid, h)
        first_obs = obs_per_patch.to(torch.float32).argmax(dim=-1)
        positions = build_forecast_positions(ll, h, first_obs)
        preds = m._run_two_stream(x_enc, x_dec, time_mask, variate_mask, positions)
        anchor_loc = loc.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
        anchor_scale = scale.unflatten(-1, (ll, ps))[..., 0][:, ll - 1 : ll]
        return denormalize_quantiles(preds, anchor_loc.unsqueeze(-1), anchor_scale.unsqueeze(-1), c.use_asinh)

    def forward(self, context: Tensor) -> Tensor:
        fc = self.forecaster
        observed = torch.isfinite(context)
        x = torch.where(observed, context, torch.zeros_like(context))
        rows = torch.cat([x, -x], dim=0)  # clampsym: the context and its mirror
        mask = torch.cat([observed, observed], dim=0)
        q = self._predict(rows, mask)
        q = fc._refine_block(q, rows, mask, None)
        q = fc._cap_quantiles(q[:, None], rows[:, None], mask[:, None])[:, 0]
        q = 0.5 * (q[0:1] + (-q[1:2]).flip(-1))
        eligible = observed.any(dim=-1) & ((x >= 0) | ~observed).all(dim=-1)
        return torch.where(eligible[:, None, None], q.clamp_min(0.0), q)


def test_series(context_length: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Name -> (official context, padded ONNX input). Deterministic."""
    rng = np.random.default_rng(20261026)
    t = np.arange(context_length, dtype=np.float64)
    cases: dict[str, np.ndarray] = {}
    cases["sine_trend"] = 50 + 0.02 * t + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, context_length)
    cases["random_walk"] = rng.normal(0, 1, context_length).cumsum()
    cases["constant"] = np.full(context_length, 42.0)
    gaps = 100 + 20 * np.sin(2 * np.pi * t / 168) + 5 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, context_length)
    gaps[rng.random(context_length) < 0.05] = np.nan
    gaps[1500:1640] = np.nan  # a gap spanning several whole patches
    gaps[-7:-3] = np.nan
    cases["nan_gaps"] = gaps
    counts = rng.poisson(3 + 2 * np.sin(2 * np.pi * t / 7) ** 2).astype(np.float64)
    cases["nonneg_counts"] = counts
    cases["large_level"] = 1.0e5 + np.cumsum(rng.normal(0, 0.5, context_length)) + 3 * np.sin(2 * np.pi * t / 12)
    out = {name: (v.astype(np.float32), v.astype(np.float32)) for name, v in cases.items()}
    short = (20 + 5 * np.sin(2 * np.pi * np.arange(301) / 12) + rng.normal(0, 0.5, 301)).astype(np.float32)
    padded = np.full(context_length, np.nan, dtype=np.float32)
    padded[-short.size :] = short
    out["short_301_padded"] = (short, padded)
    return out


def series_scale(x: np.ndarray) -> float:
    """Std of the observed values, or max(|mean|, 1) for a flat series."""
    obs = x[np.isfinite(x)].astype(np.float64)
    s = float(obs.std()) if obs.size > 1 else 0.0
    return s if s > 0 else max(abs(float(obs.mean())) if obs.size else 0.0, 1.0)


def run_session(path: Path, threads: int | None = None) -> ort.InferenceSession:
    so = ort.SessionOptions()
    if threads is not None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def rel_err(a: np.ndarray, b: np.ndarray, scale: float) -> tuple[float, float]:
    """(max abs error over all quantiles, over the median only), relative to `scale`."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return float(d.max() / scale), float(d[..., d.shape[-1] // 2].max() / scale)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--revision", default=HF_REVISION, help="HF revision of the weights (pinned)")
    ap.add_argument("--cache-dir", default=None, help="download directory for the weights (default: HF cache)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "site/public/models/ref"))
    ap.add_argument("--context", type=int, default=2048, help="fixed context length (multiple of 32)")
    ap.add_argument("--horizon", type=int, default=128, help="returned steps (multiple of 32, <= 1024)")
    ap.add_argument("--int8-tolerance", type=float, default=0.02,
                    help="keep int8 if its max error vs the official forecast, over all quantiles and test "
                    "series, stays under this fraction of the series std")
    args = ap.parse_args()

    torch.manual_seed(0)
    local = Path(snapshot_download(HF_REPO, revision=args.revision, cache_dir=args.cache_dir,
                                   allow_patterns=["config.json", "model.safetensors", "LICENSE"]))
    model = load_model(local)
    fc = TimestralForecaster(model)
    cfg = model.config
    levels = [float(q) for q in cfg.quantile_knots]
    native = fc.per_shot_horizon_patches * cfg.patch_size
    print(f"weights {HF_REPO}@{args.revision}: patch {cfg.patch_size}, d_model {cfg.d_model}, "
          f"{sum(p.numel() for p in model.parameters()):,} params, native block {native} steps, Q={len(levels)}")

    wrapper = UnivariateTimestral(fc, args.context, args.horizon).eval()
    cases = test_series(args.context)

    # Official reference: the forecaster with its defaults, on the unpadded context.
    ref: dict[str, np.ndarray] = {}
    checks: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for name, (ctx, padded) in cases.items():
            ref[name] = fc.forecast(ctx, prediction_length=args.horizon).quantiles[0, 0].numpy()
            full = fc.forecast(ctx, prediction_length=native).quantiles[0, 0, : args.horizon].numpy()
            pad = fc.forecast(padded, prediction_length=args.horizon).quantiles[0, 0].numpy()
            eager = wrapper(torch.from_numpy(padded)[None])[0].numpy()
            sc = series_scale(ctx)
            checks[name] = {f"official_H{args.horizon}_vs_first_{args.horizon}_of_H{native}": rel_err(full, ref[name], sc)[0],
                            "official_nan_padded_vs_unpadded": rel_err(pad, ref[name], sc)[0],
                            "torch_wrapper_vs_official": rel_err(eager, ref[name], sc)[0]}
            print(f"  [torch] {name:18s} " + "  ".join(f"{k}={v:.2e}" for k, v in checks[name].items()))

    work = Path(tempfile.mkdtemp(prefix="timestral-onnx-"))
    fp32_path = work / "timestral-5m.fp32.onnx"
    int8_path = work / "timestral-5m.int8.onnx"
    example = torch.from_numpy(cases["sine_trend"][1])[None]
    with torch.no_grad():
        torch.onnx.export(wrapper, (example,), str(fp32_path), input_names=[INPUT_NAME], output_names=[OUTPUT_NAME],
                          opset_version=OPSET, dynamo=True, external_data=False, optimize=True)
    onnx.checker.check_model(str(fp32_path), full_check=True)
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, per_channel=True)

    parity: dict[str, dict[str, dict[str, float]]] = {"fp32": {}, "int8": {}}
    for variant, path in (("fp32", fp32_path), ("int8", int8_path)):
        sess = run_session(path)
        for name, (ctx, padded) in cases.items():
            out = sess.run([OUTPUT_NAME], {INPUT_NAME: padded[None]})[0][0]
            assert out.shape == (args.horizon, len(levels)) and np.isfinite(out).all(), (variant, name)
            sc = series_scale(ctx)
            e_all, e_med = rel_err(out, ref[name], sc)
            parity[variant][name] = {"max_abs_err_rel_all_quantiles": e_all, "max_abs_err_rel_median": e_med,
                                     "max_abs_err": float(np.abs(out.astype(np.float64) - ref[name]).max()),
                                     "series_scale": sc}
            print(f"  [ort {variant}] {name:18s} max|err|/scale all={e_all:.2e} median={e_med:.2e} "
                  f"(scale {sc:.3g}, sorted={bool((np.diff(out, axis=-1) >= 0).all())})")

    def worst(v: str) -> float:
        return max(r["max_abs_err_rel_all_quantiles"] for r in parity[v].values())

    sizes = {"fp32": fp32_path.stat().st_size, "int8": int8_path.stat().st_size}
    chosen = "int8" if worst("int8") <= args.int8_tolerance else "fp32"
    print(f"sizes fp32 {sizes['fp32'] / 1e6:.2f} MB, int8 {sizes['int8'] / 1e6:.2f} MB; worst rel error "
          f"fp32 {worst('fp32'):.2e}, int8 {worst('int8'):.2e} (tolerance {args.int8_tolerance}) -> keep {chosen}")

    # Latency of one forward (CPU EP), 1 thread and default threads.
    bench: dict[str, dict[str, float]] = {}
    x = cases["sine_trend"][1][None]
    for variant, path in (("fp32", fp32_path), ("int8", int8_path)):
        for label, threads in (("1_thread", 1), ("default_threads", None)):
            sess = run_session(path, threads)
            for _ in range(5):
                sess.run(None, {INPUT_NAME: x})
            ts = []
            for _ in range(30):
                t0 = time.perf_counter()
                sess.run(None, {INPUT_NAME: x})
                ts.append((time.perf_counter() - t0) * 1e3)
            bench.setdefault(variant, {})[f"{label}_median_ms"] = round(float(np.median(ts)), 2)
    print(f"latency (ms, median of 30): {json.dumps(bench)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_out = out_dir / "timestral-5m.onnx"
    shutil.copyfile(int8_path if chosen == "int8" else fp32_path, onnx_out)
    ops = sorted({(n.domain or "ai.onnx", n.op_type) for n in onnx.load(str(onnx_out)).graph.node})
    op_types = sorted({f"{d}:{o}" if d != "ai.onnx" else o for d, o in ops})
    print(f"ops in {onnx_out.name}: {', '.join(op_types)}")

    license_text = (local / "LICENSE").read_text(encoding="utf-8")
    (out_dir / "TIMESTRAL-LICENSE.txt").write_text(
        f"Timestral-5M model weights by Mistral AI (https://huggingface.co/{HF_REPO}, revision {args.revision}), "
        f"converted to ONNX ({onnx_out.name}); used under the Modified MIT License below.\n\n{license_text}",
        encoding="utf-8",
    )

    meta = {
        "model": "Timestral-5M",
        "file": onnx_out.name,
        "variant": chosen,
        "file_size_bytes": onnx_out.stat().st_size,
        "sha256": hashlib.sha256(onnx_out.read_bytes()).hexdigest(),
        "license": "Modified MIT (Mistral AI), see TIMESTRAL-LICENSE.txt",
        "hf_repo": HF_REPO,
        "hf_revision": args.revision,
        "timestral_code": f"https://github.com/mistralai/timestral@{TIMESTRAL_COMMIT}",
        "export": {"exporter": f"torch.onnx (dynamo) torch {torch.__version__}", "onnx_opset": OPSET,
                   "onnx_ir_version": onnx.load(str(onnx_out), load_external_data=False).ir_version,
                   "onnxruntime_python": ort.__version__, "timestral": timestral.__version__},
        "input": {"name": INPUT_NAME, "shape": [1, args.context], "dtype": "float32",
                  "semantics": "raw values, oldest first, last element = most recent step; NaN (or +/-inf) = "
                               "missing; left-pad shorter series with NaN; at least one finite value required"},
        "output": {"name": OUTPUT_NAME, "shape": [1, args.horizon, len(levels)], "dtype": "float32",
                   "semantics": "quantiles in raw units for the next steps after the context, "
                                "[0, h, k] = step h+1 at level quantile_levels[k]; sorted along k"},
        "quantile_levels": levels,
        "median_index": levels.index(0.5),
        "context_length": args.context,
        "horizon": args.horizon,
        "native_block_horizon": native,
        "patch_size": cfg.patch_size,
        "in_graph": ["NaN/inf -> observation mask", "causal per-patch float64 mean/std + asinh normalization",
                     "denormalization", "advancing refine frame", "symmetric averaging (context and -context)",
                     "clamp to [min - 1e4*s, max + 1e4*s] of the context", "clip at 0 if the observed context is >= 0"],
        "reference": f"timestral.TimestralForecaster.forecast(context, prediction_length={args.horizon}) defaults",
        "torch_checks": {"metric": "max |a - b| / std(observed context), all quantiles", "cases": checks},
        "parity_vs_official": {"metric": "max |onnx - official| / std(observed context)", "int8_tolerance":
                               args.int8_tolerance, "candidates": {k: {"size_bytes": sizes[k], "worst": worst(k),
                                                                      "cases": parity[k]} for k in parity}},
        "benchmark_ort_python_cpu": {"machine": "macOS arm64", **bench},
        "ops": op_types,
    }
    (out_dir / "timestral-5m.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    shutil.rmtree(work, ignore_errors=True)
    print(f"wrote {onnx_out} ({meta['file_size_bytes'] / 1e6:.2f} MB), timestral-5m.json, TIMESTRAL-LICENSE.txt")


if __name__ == "__main__":
    main()
