//! Real time series for s6-cast (docs/SPEC-S6-CAST.md §3): `fetch-series` writes
//! `data/series/<source>.jsonl`, one `{"id", "values"}` per line (`null` = missing).
//!
//! Only sources whose upstream license is permissive and that appear in no test set s6-cast is
//! scored on (GIFT-Eval's 97 configurations, Chronos' zero-shot benchmark) are listed: a series
//! seen in training would inflate a zero-shot score. They are read from the parquet copies of
//! `autogluon/chronos_datasets`, whose repository tag is "other"; the license that applies is
//! each upstream's (DATA_LICENSES.md).

use std::path::{Path, PathBuf};

use anyhow::Result;
use serde_json::{json, Value};

use crate::hf::{read_parquet_any, Hf};

/// A series source.
pub struct SeriesSource {
    /// Output name (`data/series/<name>.jsonl`).
    pub name: &'static str,
    /// Config of `autogluon/chronos_datasets`.
    pub config: &'static str,
    /// Upstream license.
    pub license: &'static str,
    /// Series kept at most (the first ones in file order).
    pub max_series: usize,
}

pub const REPO: &str = "autogluon/chronos_datasets";

/// Points kept per series (the last ones): four 2048-point contexts.
pub const MAX_LEN: usize = 8192;
/// Shorter series (in finite points) are dropped.
pub const MIN_POINTS: usize = 64;

pub const SOURCES: [SeriesSource; 7] = [
    SeriesSource {
        name: "london_smart_meters",
        config: "monash_london_smart_meters",
        license: "CC-BY-4.0 (Monash forecasting archive)",
        max_series: 2000,
    },
    SeriesSource {
        name: "wind_farms_hourly",
        config: "wind_farms_hourly",
        license: "CC-BY-4.0 (Monash forecasting archive)",
        max_series: 400,
    },
    SeriesSource {
        name: "wind_farms_daily",
        config: "wind_farms_daily",
        license: "CC-BY-4.0 (Monash forecasting archive)",
        max_series: 400,
    },
    SeriesSource {
        name: "pedestrian_counts",
        config: "monash_pedestrian_counts",
        license: "CC-BY-4.0 (Monash forecasting archive)",
        max_series: 100,
    },
    SeriesSource {
        name: "rideshare",
        config: "monash_rideshare",
        license: "CC-BY-4.0 (Monash forecasting archive)",
        max_series: 200,
    },
    SeriesSource {
        name: "wiki_daily",
        config: "wiki_daily_100k",
        license: "CC0-1.0 (Wikimedia pageviews)",
        max_series: 5000,
    },
    SeriesSource {
        name: "ushcn_daily",
        config: "ushcn_daily",
        license: "public domain (NOAA NCEI, U.S. Government work)",
        max_series: 1300,
    },
];

/// The numeric list columns of a row (every one but `timestamp`), as series. A column holding
/// numbers and nulls is a series; `id` names the row.
pub fn row_series(source: &str, row: &Value) -> Vec<(String, Vec<Option<f64>>)> {
    let Some(obj) = row.as_object() else {
        return Vec::new();
    };
    let id = obj
        .get("id")
        .and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| Some(v.to_string()))
        })
        .unwrap_or_default();
    let mut out = Vec::new();
    for (col, v) in obj {
        if col == "timestamp" || col == "id" {
            continue;
        }
        let Some(arr) = v.as_array() else { continue };
        if arr.is_empty() || !arr.iter().all(|x| x.is_number() || x.is_null()) {
            continue;
        }
        let values: Vec<Option<f64>> = arr.iter().map(Value::as_f64).collect();
        let name = if col == "target" {
            format!("{source}:{id}")
        } else {
            format!("{source}:{id}:{col}")
        };
        out.push((name, values));
    }
    out
}

/// One output line, or `None` when the series is too short (keeps the last [`MAX_LEN`] points).
pub fn series_line(id: &str, values: &[Option<f64>]) -> Option<String> {
    let tail = &values[values.len().saturating_sub(MAX_LEN)..];
    if tail
        .iter()
        .filter(|v| v.is_some_and(f64::is_finite))
        .count()
        < MIN_POINTS
    {
        return None;
    }
    let vals: Vec<Value> = tail
        .iter()
        .map(|v| match v {
            Some(x) if x.is_finite() => json!(*x as f32),
            _ => Value::Null,
        })
        .collect();
    Some(json!({"id": id, "values": vals}).to_string())
}

/// Per source: series written.
pub struct SeriesReport {
    pub name: &'static str,
    pub license: &'static str,
    pub series: usize,
    pub points: usize,
}

/// Downloads every selected source and writes `<out>/<name>.jsonl`.
pub fn fetch_series(hf: &Hf, out: &Path, only: &[String]) -> Result<Vec<SeriesReport>> {
    std::fs::create_dir_all(out)?;
    let mut reports = Vec::new();
    for s in SOURCES.iter() {
        if !only.is_empty() && !only.iter().any(|o| o == s.name) {
            continue;
        }
        eprintln!("[series] {} ({}) ...", s.name, s.config);
        let paths: Vec<PathBuf> = hf.parquet_split(REPO, s.config, "train", 4 << 30)?;
        let mut lines = Vec::new();
        let mut points = 0;
        'files: for p in paths {
            for row in read_parquet_any(&p, &[], usize::MAX)? {
                for (id, values) in row_series(s.name, &row) {
                    if let Some(l) = series_line(&id, &values) {
                        points += values.len().min(MAX_LEN);
                        lines.push(l);
                        if lines.len() >= s.max_series {
                            break 'files;
                        }
                    }
                }
            }
        }
        let n = lines.len();
        std::fs::write(
            out.join(format!("{}.jsonl", s.name)),
            lines.join("\n") + "\n",
        )?;
        eprintln!("[series] {}: {n} series, {points} points", s.name);
        reports.push(SeriesReport {
            name: s.name,
            license: s.license,
            series: n,
            points,
        });
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_become_series_and_short_ones_are_dropped() {
        let row = json!({
            "id": "T1",
            "timestamp": ["2020-01-01", "2020-01-02"],
            "target": (0..100).map(|x| json!(x as f64)).collect::<Vec<_>>(),
            "TMAX": [1.0, null, 3.0],
            "label": "x",
        });
        let mut s = row_series("ushcn", &row);
        s.sort_by(|a, b| a.0.cmp(&b.0));
        let names: Vec<&str> = s.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["ushcn:T1", "ushcn:T1:TMAX"]);
        assert!(
            series_line(&s[1].0, &s[1].1).is_none(),
            "3 points is too short"
        );
        let l: Value = serde_json::from_str(&series_line(&s[0].0, &s[0].1).unwrap()).unwrap();
        assert_eq!(l["values"].as_array().unwrap().len(), 100);
        let gaps: Vec<Option<f64>> = (0..70).map(|i| (i % 2 == 0).then_some(1.0)).collect();
        assert!(series_line("g", &gaps).is_none(), "35 finite points");
        let long: Vec<Option<f64>> = (0..MAX_LEN + 10).map(|i| Some(i as f64)).collect();
        let l: Value = serde_json::from_str(&series_line("l", &long).unwrap()).unwrap();
        assert_eq!(l["values"][0], json!(10.0));
    }
}
