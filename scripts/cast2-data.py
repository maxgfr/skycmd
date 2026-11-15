# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy>=1.26,<3",
#   "datasets>=2.15,<4",
#   "pyarrow>=15,<18",
# ]
# ///
"""The new real series of s6-cast v2 (docs/s6-cast2-prereg.md §3.2, §3.3, C3).

    python3 -I scripts/cast2-data.py list  [--out runs/s6-cast2-data/list.json]
    python3 -I scripts/cast2-data.py fetch --source uscrn_subhourly --out $CAST2_DATA/series2
    HF_HOME=$CAST2_DATA/hf uv run --python 3.12 scripts/cast2-data.py leak-check \
        --series $CAST2_DATA/series2 --gift-eval data/gift-eval-src --out $CAST2_DATA/leak.json

`list` is the dry run: it reads only metadata (directory listings, object listings, HTTP
headers, licence files) and applies the selection rule of §3.2 to every candidate source:

1. the licence, as its publisher states it, is CC0, CC-BY (any version), public domain,
   Apache-2.0, MIT or BSD;
2. its name is in neither GIFT-Eval's 97 configurations nor the 27 Chronos-ZS tasks, nor in a
   dataset either benchmark draws from (the v1 exclusion list of DATA_LICENSES.md);
3. it passes the content check of §3.3 (`leak-check`; it needs the data, so `list` reports it
   as pending).

It prints, per source, the licence and what states it, the two rules, the files, bytes, and the
series and points the fetch would write. Nothing is downloaded but listings and licence files.

`fetch` downloads one source and writes `<out>/<source>.jsonl` (`{"id", "values"}` per line,
`null` = missing, as `sky-data fetch-series`). It refuses a source that fails rule 1 or 2.

`leak-check` is the content check of §3.3: every run of 32 consecutive finite values of every
new series, as float32, is hashed (64-bit) and compared with the same hashes over every series
of GIFT-Eval (`data/gift-eval-src`, all 97 configurations' datasets) and of Chronos-ZS (the 27
datasets of eval/chronos-zs/tasks.yaml, downloaded into `$HF_HOME`). The new series' hashes
are held, the benchmark series streamed past them. Runs whose 32 values are all equal are skipped. A
source with any matching run is dropped whole. The benchmark series are read only to be hashed.
It writes the counts per source; the pre-registration has them recorded before training.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import io
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
UA = {"User-Agent": "skycmd-cast2-data/1 (metadata listing)"}
# v1's limits (crates/sky-data/src/series.rs): the last 8,192 points of a series, 64 finite ones
# at least.
MAX_LEN = 8192
MIN_POINTS = 64
RUN = 32

# Rule 1: the licences the pre-registration accepts.
ALLOWED = ("CC0", "CC-BY", "public domain", "Apache-2.0", "MIT", "BSD")

# Rule 2: GIFT-Eval's datasets (eval/gift-eval/dataset_properties.json), the 27 Chronos-ZS
# tasks (eval/chronos-zs/tasks.yaml), the datasets the two benchmarks draw from and the v1
# exclusion list of DATA_LICENSES.md. Matched on the lower-cased name with separators removed.
V1_EXCLUDED = (
    "m1 m3 m4 m5 nn5 tourism traffic weather electricity solar hospital covid_deaths car_parts "
    "ett exchange_rate fred_md dominick ercot cif_2016 australian_electricity kdd_cup_2018 "
    "temperature_rain saugeen us_births"
).split()
DRAWN_FROM = (
    "bitbrains bizitobs loop_seattle sz_taxi m_dense jena_weather restaurant hierarchical_sales "
    "beijing_air_quality monash libcity"
).split()


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def excluded_names() -> list[str]:
    props = json.loads((REPO / "eval/gift-eval/dataset_properties.json").read_text())
    names = set(props) | set(V1_EXCLUDED) | set(DRAWN_FROM)
    tasks = REPO / "eval/chronos-zs/tasks.yaml"
    for line in tasks.read_text().splitlines():
        m = re.match(r"\s*-?\s*dataset_config:\s*(\S+)", line)
        if m:
            names.add(m.group(1))
    return sorted(norm(n) for n in names)


def rule2(names: list[str]) -> tuple[bool, list[str]]:
    """The candidate's names (source, dataset, publisher's name) against the exclusion list."""
    bad = excluded_names()
    hits = sorted({b for n in names for b in bad if b and (b in norm(n) or norm(n) in b)})
    return not hits, hits


def get(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def head_size(url: str) -> int | None:
    req = urllib.request.Request(url, headers=UA, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            n = r.headers.get("Content-Length")
            return int(n) if n else None
    except Exception:
        return None


# ----------------------------------------------------------------------------- the candidates

USCRN = "https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01"
# The measured columns of a USCRN sub-hourly record (HEADERS.txt; flags and ids left out), and
# the value it writes for a missing measurement.
USCRN_FIELDS = {
    "AIR_TEMPERATURE": 8,
    "PRECIPITATION": 9,
    "SOLAR_RADIATION": 10,
    "SURFACE_TEMPERATURE": 12,
    "RELATIVE_HUMIDITY": 15,
    "SOIL_MOISTURE_5": 17,
    "SOIL_TEMPERATURE_5": 18,
    "WETNESS": 19,
    "WIND_1_5": 21,
}
USCRN_MISSING = {-9999.0, -99.0, -99.999}
# The year the fetch reads: the last complete one, so every station-variable series is its last
# 8,192 five-minute points of that year (v1's MAX_LEN rule).
USCRN_YEAR = 2025


def uscrn_listing(year: int) -> list[tuple[str, int]]:
    """(file, bytes) of one year's station files, from the directory listing."""
    text = get(f"{USCRN}/{year}/").decode("utf-8", "replace")
    out = []
    for m in re.finditer(
        r'href="(CRNS0101-05-\d{4}-[^"]+\.txt)".*?<td[^>]*>[^<]*</td>\s*<td[^>]*>\s*([\d.]+[KMG]?)\s*</td>',
        text,
        flags=re.S,
    ):
        size = m.group(2)
        mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}.get(size[-1], 1)
        out.append((m.group(1), int(float(size.rstrip("KMG")) * mult)))
    if not out:
        # a listing without a size column: the files, sized by HEAD
        for f in sorted(set(re.findall(r'href="(CRNS0101-05-\d{4}-[^"]+\.txt)"', text))):
            out.append((f, head_size(f"{USCRN}/{year}/{f}") or 0))
    return out


def list_uscrn() -> dict:
    readme = get(f"{USCRN}/README.txt").decode("utf-8", "replace")
    files = uscrn_listing(USCRN_YEAR)
    stations = len(files)
    nbytes = sum(b for _, b in files)
    per = 365 * 24 * 12
    return {
        "source": "uscrn_subhourly",
        "band": "subhourly (5 minutes)",
        "publisher": "NOAA National Centers for Environmental Information (U.S. Climate Reference Network)",
        "url": f"{USCRN}/",
        "licence": "public domain",
        "licence_stated_by": "a U.S. Government work (17 U.S.C. 105), the basis of v1's USHCN entry "
        "in DATA_LICENSES.md; the product README states no other terms",
        "readme_terms": [l.strip() for l in readme.splitlines() if re.search(r"licen|copyright|terms", l, re.I)],
        "names": ["uscrn", "U.S. Climate Reference Network", "USCRN subhourly"],
        "fetch": f"the {USCRN_YEAR} file of every station; one series per station and measured "
        f"column ({len(USCRN_FIELDS)}), its last {MAX_LEN} points, -9999 as missing",
        "files": stations,
        "bytes": nbytes,
        "series_at_most": stations * len(USCRN_FIELDS),
        "points_at_most": stations * len(USCRN_FIELDS) * min(per, MAX_LEN),
        "fetchable": True,
    }


# Considered and not kept, with the rule that fails (no network needed to say so).
REJECTED = [
    {
        "source": "azure_vm_traces_2017",
        "publisher": "Microsoft, Azure Public Dataset V1",
        "licence": "CC-BY-4.0",
        "why": "passes rules 1 and 2; not used by amendment 1 of the pre-registration: 84 GB of "
        "per-VM records for which §3.2 gives no rule to build series",
    },
    {
        "source": "borg_cluster_data_2011",
        "publisher": "Google, cluster-data ClusterData2011_2",
        "licence": "CC-BY-4.0",
        "why": "passes rules 1 and 2; not used by amendment 1 of the pre-registration: 42 GB of "
        "per-task records for which §3.2 gives no rule to build series",
    },
    {
        "source": "alibaba_cluster_trace_2018",
        "publisher": "Alibaba (github.com/alibaba/clusterdata, cluster-trace-v2018)",
        "licence": None,
        "why": "rule 1: no licence file; the README allows use 'as long as it is for research or "
        "study purpose'",
    },
    {
        "source": "Salesforce/lotsa_data and Salesforce/GiftEvalPretrain copies (alibaba, azure, borg, "
        "PEMS, LargeST, buildings_900k, ideal, ...)",
        "publisher": "Salesforce",
        "licence": "apache-2.0 card, 'for research purposes only'",
        "why": "open choice O3 (not used); its constituents are taken from their publishers instead",
    },
    {
        "source": "LargeST (PEMS 5-minute traffic)",
        "publisher": "LargeST authors",
        "licence": "CC-BY-NC-4.0",
        "why": "rule 1 (non-commercial)",
    },
    {
        "source": "Caltrans PeMS",
        "publisher": "California Department of Transportation",
        "licence": None,
        "why": "rule 1: no licence among those of the rule is stated",
    },
    {
        "source": "autogluon/chronos_datasets sub-hourly configs (electricity_15min, solar, solar_1h, "
        "taxi_30min)",
        "publisher": "AutoGluon",
        "licence": "tag 'other'",
        "why": "rule 2: electricity and solar are GIFT-Eval datasets; taxi_30min is 30-minute and a "
        "Chronos training-corpus dataset",
    },
]


def cmd_list(a) -> None:
    out = {"rule": "docs/s6-cast2-prereg.md §3.2", "excluded_names": excluded_names(), "candidates": []}
    # amendment 1: USCRN is the one new source (the others considered are in REJECTED)
    for fn in (list_uscrn,):
        try:
            c = fn()
        except Exception as e:  # a listing that fails is reported, not guessed
            c = {"source": fn.__name__.removeprefix("list_"), "error": repr(e)}
        if "licence" in c:
            c["rule1"] = any(c["licence"].lower().startswith(x.lower()) for x in ALLOWED)
            c["rule2"], c["rule2_hits"] = rule2([c["source"], *c.get("names", [])])
            c["rule3"] = "pending: leak-check after the fetch, before training"
        out["candidates"].append(c)
    out["rejected"] = REJECTED
    text = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    print(f"{'source':<26} {'licence':<14} r1    r2    files  bytes            series<=  points<=")
    for c in out["candidates"]:
        if "error" in c:
            print(f"{c['source']:<26} listing failed: {c['error']}")
            continue
        print(
            f"{c['source']:<26} {c['licence']:<14} {str(c['rule1']):<5} {str(c['rule2']):<5} "
            f"{c['files']:>6} {c['bytes'] if c['bytes'] is not None else '?':>15}  "
            f"{c.get('series_at_most', '?'):>8}  {c.get('points_at_most', '?')}"
        )
    for r in REJECTED:
        print(f"rejected: {r['source']}: {r['why']}")


# ----------------------------------------------------------------------------- fetch


def uscrn_series(text: str, station: str) -> list[tuple[str, list[float | None]]]:
    cols: dict[str, list[float | None]] = {k: [] for k in USCRN_FIELDS}
    for line in text.splitlines():
        f = line.split()
        if len(f) < 22:
            continue
        for k, i in USCRN_FIELDS.items():
            try:
                v = float(f[i])
            except ValueError:
                v = None
            cols[k].append(None if v is None or v in USCRN_MISSING else v)
    return [(f"uscrn_subhourly:{station}:{k}", v) for k, v in cols.items()]


def series_line(id_: str, values: list[float | None]) -> str | None:
    tail = values[-MAX_LEN:]
    if sum(v is not None for v in tail) < MIN_POINTS:
        return None
    if len({v for v in tail if v is not None}) < 2:
        return None  # a constant column (an absent sensor) is no series
    return json.dumps({"id": id_, "values": tail}, separators=(",", ":"))


def cmd_fetch(a) -> None:
    if a.source != "uscrn_subhourly":
        sys.exit(f"{a.source}: no fetch is written for it (see `list`)")
    c = list_uscrn()
    ok, hits = rule2([c["source"], *c["names"]])
    if not ok:
        sys.exit(f"{a.source} fails rule 2: {hits}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lines, points = [], 0
    files = uscrn_listing(USCRN_YEAR)
    if a.dry_run:
        print(f"would fetch {len(files)} files ({sum(b for _, b in files)} bytes) to {out}/uscrn_subhourly.jsonl")
        return
    # NCEI serves one file at about 0.2 MB/s: a few downloads at a time, parsed in memory and
    # dropped (nothing but the JSONL is written), in the listing's order
    from concurrent.futures import ThreadPoolExecutor

    def fetch_one(f: str) -> str:
        for attempt in range(4):
            try:
                return get(f"{USCRN}/{USCRN_YEAR}/{f}", timeout=600).decode("utf-8", "replace")
            except Exception:
                if attempt == 3:
                    raise
        raise AssertionError

    pool = ThreadPoolExecutor(max_workers=a.jobs)
    texts = pool.map(fetch_one, [f for f, _ in files])
    for k, ((f, _), text) in enumerate(zip(files, texts)):
        station = re.sub(r"^CRNS0101-05-\d{4}-|\.txt$", "", f)
        for id_, vals in uscrn_series(text, station):
            line = series_line(id_, vals)
            if line:
                lines.append(line)
                points += min(len(vals), MAX_LEN)
        print(f"[{k + 1}/{len(files)}] {station}: {len(lines)} series so far", flush=True)
    (out / "uscrn_subhourly.jsonl").write_text("\n".join(lines) + "\n")
    manifest = {
        "source": "uscrn_subhourly",
        "year": USCRN_YEAR,
        "files": len(files),
        "series": len(lines),
        "points": points,
        "sha256": hashlib.sha256((out / "uscrn_subhourly.jsonl").read_bytes()).hexdigest(),
    }
    (out / "uscrn_subhourly.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


# ----------------------------------------------------------------------------- leak check


# Amendment 2: a run is compared only if it is informative: at least MIN_DISTINCT distinct
# values, and its most frequent value on at most half of its points.
MIN_DISTINCT = 8
MAX_MODE = RUN // 2
# A source is dropped whole when more than this share of its series is.
MAX_DROPPED_SHARE = 0.05


def informative(win):
    """Per run (row of `win`): at least MIN_DISTINCT distinct values and no value on more than
    MAX_MODE of its points (on the sorted run: no value equal to the one MAX_MODE places on)."""
    import numpy as np

    srt = np.sort(win, axis=1)
    distinct = 1 + (np.diff(srt, axis=1) != 0).sum(axis=1)
    mode_over_half = (srt[:, MAX_MODE:] == srt[:, :-MAX_MODE]).any(axis=1)
    return (distinct >= MIN_DISTINCT) & ~mode_over_half


def run_hashes(x, only_informative: bool = True):
    """64-bit hashes of every run of RUN consecutive finite values: the informative ones
    (amendment 2), or with `only_informative=False` every non-constant one (amendment 1)."""
    import numpy as np

    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 1 or x.size < RUN:
        return np.empty(0, dtype=np.uint64)
    win = np.lib.stride_tricks.sliding_window_view(x, RUN)
    keep = np.isfinite(win).all(axis=1)
    if only_informative:
        keep[keep] = informative(win[keep])
    else:
        keep &= win.max(axis=1, initial=-np.inf) != win.min(axis=1, initial=np.inf)
    if not keep.any():
        return np.empty(0, dtype=np.uint64)
    bits = np.ascontiguousarray(win[keep]).view(np.uint32).astype(np.uint64)
    # a polynomial hash of the 32 bit patterns, wrapping in 64 bits, then a final mix
    mult = np.uint64(0x9E3779B97F4A7C15)
    powers = np.ones(RUN, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(1, RUN):
            powers[i] = powers[i - 1] * mult
        h = (bits * powers).sum(axis=1, dtype=np.uint64)
        h ^= h >> np.uint64(31)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(29)
    return h


def lookup(sorted_unique, h):
    """Which of the hashes `h` are in the sorted, unique array (a binary search per hash)."""
    import numpy as np

    if not h.size or not sorted_unique.size:
        return np.zeros(h.size, dtype=bool)
    idx = np.minimum(np.searchsorted(sorted_unique, h), sorted_unique.size - 1)
    return sorted_unique[idx] == h


def check(sources: dict, bench) -> dict:
    """The content check of §3.3 with amendment 2's rule. `sources`: name -> [(id, values)];
    `bench`: an iterable of (benchmark, dataset, series). Every new series' informative runs are
    hashed; the benchmark series stream past them. A series with a matching informative run is
    dropped; a source whose dropped share exceeds MAX_DROPPED_SHARE is dropped whole."""
    import numpy as np

    hashed = {name: [(i, run_hashes(x)) for i, x in rows] for name, rows in sources.items()}
    parts = [h for rows in hashed.values() for _, h in rows if h.size]
    pool = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)
    matched, counts, where = [], {}, {}
    for b, name, s in bench:
        h = run_hashes(s)
        c = counts.setdefault(b, [0, 0])
        c[0] += 1
        c[1] += int(h.size)
        hit = h[lookup(pool, h)]
        if hit.size:
            matched.append(np.unique(hit))
            where[f"{b}:{name}"] = where.get(f"{b}:{name}", 0) + int(hit.size)
    matched_h = np.unique(np.concatenate(matched)) if matched else np.empty(0, np.uint64)
    out = {"benchmarks": {b: {"series": c[0], "informative_runs": c[1]} for b, c in counts.items()},
           "benchmark_datasets_with_a_match": where, "sources": {}}
    for name, rows in hashed.items():
        hits = [int(lookup(matched_h, h).sum()) for _, h in rows]
        dropped = [i for (i, _), n in zip(rows, hits) if n]
        share = len(dropped) / max(len(rows), 1)
        out["sources"][name] = {
            "series": len(rows),
            "informative_runs": sum(int(h.size) for _, h in rows),
            "matching_runs": sum(hits),
            "series_dropped": len(dropped),
            "dropped_share": share,
            "dropped_series": dropped,
            "series_kept": 0 if share > MAX_DROPPED_SHARE else len(rows) - len(dropped),
            "kept": share <= MAX_DROPPED_SHARE,
        }
    return out


def series_of(values) -> list:
    """The 1-D series of a target value (a list, or a list of lists for a multivariate one)."""
    import numpy as np

    a = np.asarray(values, dtype=np.float64) if not isinstance(values, np.ndarray) else values
    if a.ndim == 1:
        return [a]
    if a.ndim == 2:
        return list(a)
    return []


def benchmark_series(gift_eval: Path):
    """(benchmark, name, 1-D series) of every GIFT-Eval and Chronos-ZS series."""
    import datasets

    # GIFT-Eval: every dataset directory under data/gift-eval-src (arrow, `target` column)
    for d in sorted(gift_eval.rglob("dataset_info.json")):
        ds = datasets.load_from_disk(str(d.parent))
        for row in ds.select_columns(["target"]).with_format("numpy"):
            for s in series_of(row["target"]):
                yield "gift_eval", str(d.parent.relative_to(gift_eval)), s
    # Chronos-ZS: the 27 datasets, every numeric sequence column but the timestamps
    tasks = REPO / "eval/chronos-zs/tasks.yaml"
    pairs = re.findall(r"dataset_path:\s*(\S+)\s*\n\s*dataset_config:\s*(\S+)", tasks.read_text())
    for path, config in pairs:
        # chronos_datasets_extra (ETTh, ETTm) is built by its loading script (see chronos-zs.py)
        ds = datasets.load_dataset(path, config, split="train",
                                   trust_remote_code=path == "autogluon/chronos_datasets_extra")
        seq_cols = [
            c for c, f in ds.features.items()
            if isinstance(f, datasets.Sequence) and c != "timestamp"
            and str(getattr(f.feature, "dtype", "")).startswith(("float", "int"))
        ]
        for row in ds.select_columns(seq_cols).with_format("numpy"):
            for c in seq_cols:
                yield "chronos_zs", f"{path}/{config}", row[c]


def cmd_leak_check(a) -> None:
    """Amendment 2's check on every source of `--series`; the kept series of the kept sources
    go to `--kept-out` (what trains), and the report to `--out`, the run of amendment 1's rule
    kept in it when the file holds one."""
    import numpy as np

    sources, lines = {}, {}
    for f in sorted(Path(a.series).glob("*.jsonl")):
        rows, raw = [], {}
        for line in f.open():
            if line.strip():
                r = json.loads(line)
                rows.append((r["id"], np.array([np.nan if v is None else v for v in r["values"]], dtype=np.float32)))
                raw[r["id"]] = line.rstrip("\n")
        sources[f.stem], lines[f.stem] = rows, raw
    res = check(sources, benchmark_series(Path(a.gift_eval)))
    out = Path(a.out)
    report = {
        "rule": "docs/s6-cast2-prereg.md §3.3, amendment 2: runs of 32 float32 values with at least "
        f"{MIN_DISTINCT} distinct values and no value on more than {MAX_MODE} points; a series with a "
        f"matching run is dropped, a source with more than {MAX_DROPPED_SHARE:.0%} of its series dropped "
        "is dropped whole",
        "hash": "64-bit polynomial hash of the float32 bit patterns",
        **res,
    }
    if out.exists():
        prev = json.loads(out.read_text())
        # the file holds amendment 1's run, or a report that already carries it
        report["amendment_1_run"] = prev.get("amendment_1_run", prev)
    kept_dir = Path(a.kept_out)
    kept_dir.mkdir(parents=True, exist_ok=True)
    for name, r in res["sources"].items():
        dst = kept_dir / f"{name}.jsonl"
        if not r["kept"]:
            dst.unlink(missing_ok=True)
            print(f"{name}: {r['series_dropped']} of {r['series']} series dropped ({r['dropped_share']:.2%}) "
                  "-> DROPPED whole")
            continue
        drop = set(r["dropped_series"])
        keep = [l for i, l in lines[name].items() if i not in drop]
        dst.write_text("\n".join(keep) + "\n")
        r["kept_file"] = str(dst)
        r["kept_sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
        r["kept_points"] = sum(len(json.loads(l)["values"]) for l in keep)
        print(f"{name}: {r['series']} series, {r['informative_runs']} informative runs, {r['matching_runs']} "
              f"matching, {r['series_dropped']} series dropped ({r['dropped_share']:.2%}) -> "
              f"{'kept' if r['kept'] else 'DROPPED whole'}, {r['series_kept']} series")
    for b, c in res["benchmarks"].items():
        print(f"{b}: {c['series']} series, {c['informative_runs']} informative runs")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    l = sub.add_parser("list")
    l.add_argument("--out")
    f = sub.add_parser("fetch")
    f.add_argument("--source", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--dry-run", action="store_true")
    f.add_argument("--jobs", type=int, default=8, help="downloads at a time")
    c = sub.add_parser("leak-check")
    c.add_argument("--series", required=True)
    c.add_argument("--gift-eval", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--kept-out", required=True, help="where the kept series of the kept sources go")
    a = p.parse_args()
    {"list": cmd_list, "fetch": cmd_fetch, "leak-check": cmd_leak_check}[a.cmd](a)


if __name__ == "__main__":
    main()
