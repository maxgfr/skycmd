//! s5 analyses of `sky-eval --dump-probs` files, as pre-registered in `docs/s5-prereg.md`. A dump
//! has one line per question (`src`, `t`, `k`, `probs`, `target`, `gold`, `uid`); an evaluation
//! directory holds `<set>.dump.jsonl` for each evaluation file `<set>.jsonl`.
//!
//! ```text
//! sky-s5-report calibrate <cal.dump> --gguf in.gguf --out out.gguf [--json t.json] [--min-bucket 100]
//! sky-s5-report gates <eval dir> --records <data/s5/eval> [--json out.json]
//! sky-s5-report coherence <eval dir>[,<eval dir>...] [--json out.json]
//! sky-s5-report robust <eval dir> [--json out.json]
//! sky-s5-report selective <cal.dump> <eval dir> [--cal-src typed_decisions_train] [--json out.json]
//! sky-s5-report pari1 --ref <dirs> --variant <dirs> [--json out.json]
//! sky-s5-report pari2 --ref <dirs> --variant <dirs> [--json out.json]
//! sky-s5-report summary --store <store> --out results.json
//! ```
//!
//! **Intervals.** 95 % percentile bootstrap, B = 2000, seed 0. With several runs of a variant
//! (seeds; comma-separated directories) the bootstrap has two levels: the items are resampled,
//! then the runs with replacement, and a draw is the mean of the statistic over the drawn runs.
//! A paired comparison resamples the same items for both arms (their runs independently); its
//! p-value is the two-sided bootstrap p-value of `sky-metrics`.
//!
//! - `calibrate`: one temperature per question type, and per `type.bucket` with at least
//!   `--min-bucket` questions (option-count buckets `2`, `3_5`, `6_10`, `11`), minimizing the NLL
//!   of the calibration dump; written into a copy of the GGUF (`skycmd-iso.temperature.*` or
//!   `<arch>.decision.temperature.*`). The input GGUF must carry no temperature yet, so that the
//!   dump was produced at T = 1.
//! - `gates` (s5-lite): fitted bench ≥ 0.65, any set above 0.735 flagged, English demo cases
//!   ≥ 90 %, no set below its majority-class baseline (the most frequent gold option id).
//! - `coherence` (2609.33209 identities on the test families of typed-decisions; unit: family):
//!   MAD_neg `|s_k + n_k − 1|`, same-side rate, logit violation, partition `Σ s_k`, cross-format
//!   `|s_k − c_k|`; and the identities never trained on: disjunction additivity
//!   `|d_ij − (s_i + s_j)|`, description invariance `|d_ij − n_m|` (K = 3), monotonicity
//!   `s_i ≤ d_ij` (share of checks violated by more than 0.05, mean excess), and the accuracy of
//!   the D questions against the gold with its majority-class rate (competence).
//! - `robust`: the two-call paraphrase test, JevOut-lite (the attack keeps, per base, the
//!   distractor that lowers p(gold) most for the evaluated model), the reassignment of
//!   2609.26758 (flip rate per arm minus the 0/1 control, AUC per arm), mini-PolicyBench and
//!   candidate coverage (detection / false rejection with `P(none)`, threshold on the
//!   calibration part at false rejection ≤ 5 %).
//! - `selective`: escalation thresholds per question type by Learn-then-Test on the calibration
//!   dump (binomial p-value of the error among the accepted, fixed-sequence testing from the
//!   highest threshold, δ = 0.05), applied to the fitted bench for α ∈ {1, 2, 5, 10} %, next to
//!   the Mondrian conformal rule "accept when the set has one answer" (`s = 1 − p_y`).
//! - `pari1`: B (variant) against A′ (reference) on the fitted bench: accuracy non-inferiority
//!   (margin 0.02) and ECE non-inferiority (margin 0.01), both required; secondary comparisons
//!   with Holm-adjusted p-values: NLL and Brier of the bench, paraphrase flip rate, JevOut-lite
//!   flip rate, macro accuracy and macro ECE of the English zero-shot sets.
//! - `pari2`: λ* (variant) against λ = 0 (reference): D competence at λ = 0 (lower bound of the
//!   D accuracy > majority rate + 0.10), MAD_neg of λ* < 0.03, partition of λ* in [0.97, 1.03],
//!   bench accuracy non-inferiority (0.02), monotonicity violations reduced by ≥ 30 % with an
//!   interval of the difference below 0; secondary (Holm): MAD_neg, |Σ s − 1|, |s − c|,
//!   additivity, description invariance, bench ECE.
//! - `summary`: one compact JSON for the results page: every report of `runs/s5-report/`, the
//!   frozen selections of `runs/s5-select/`, the validation summary of every s5 run, the log of
//!   the test passes, and the shipped model by the pre-registered rule (lowest val NLL among the
//!   three seeds of the retained arm: λ* when bet 2 succeeded, else B* when bet 1 succeeded,
//!   else A′).

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use serde_json::{json, Value};
    use sky_gguf::{Gguf, GgufWriter, OwnedValue};
    use sky_metrics::{
        accuracy, brier, ece, holm, mean, nll, p_value, pick, quantile, Better, Interval, Paired,
        Pred, Rng,
    };

    /// The fitted bench.
    pub const BENCH: &str = "typed_decisions_test";
    /// The English zero-shot sets of the plan (HWU64 and SIB-200 are reported apart).
    pub const ZERO_SHOT: [&str; 9] = [
        "wikitoxic_insult",
        "wikitoxic_obscene",
        "wikitoxic_threat",
        "wikitoxic_toxicaggregated",
        "biasframes_intent",
        "biasframes_offensive",
        "biasframes_sex",
        "amazonpolarity",
        "yahootopics",
    ];
    pub const COHERENCE: &str = "coherence_test";
    const RESAMPLES: usize = 2000;
    const LEVEL: f64 = 0.95;

    #[derive(Clone, Debug)]
    pub struct Item {
        pub src: String,
        pub t: String,
        pub probs: Vec<f64>,
        pub target: Vec<f64>,
        pub uid: String,
    }

    impl Item {
        fn gold(&self) -> usize {
            sky_metrics::argmax(&self.target)
        }
        fn pred(&self) -> usize {
            sky_metrics::argmax(&self.probs)
        }
        fn correct(&self) -> bool {
            self.pred() == self.gold()
        }
        fn to_pred(&self) -> Pred {
            Pred::new(self.probs.clone(), self.target.clone(), self.src.as_str())
        }
    }

    pub fn read(path: &Path) -> Vec<Item> {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let v: Value = serde_json::from_str(l).unwrap_or_else(|e| panic!("{path:?}: {e}"));
                let f = |k: &str| -> Vec<f64> {
                    v[k].as_array()
                        .map(|a| a.iter().filter_map(Value::as_f64).collect())
                        .unwrap_or_default()
                };
                Item {
                    src: v["src"].as_str().unwrap_or_default().to_string(),
                    t: v["t"].as_str().unwrap_or_default().to_string(),
                    probs: f("probs"),
                    target: f("target"),
                    uid: v["uid"].as_str().unwrap_or_default().to_string(),
                }
            })
            .collect()
    }

    /// The dump of `set` in an evaluation directory (empty when missing).
    fn dump(dir: &Path, set: &str) -> Vec<Item> {
        let p = dir.join(format!("{set}.dump.jsonl"));
        if p.exists() {
            read(&p)
        } else {
            Vec::new()
        }
    }

    /// The dumps of `set` in several runs, checked to hold the same questions in the same order.
    fn runs(dirs: &[PathBuf], set: &str) -> Vec<Vec<Item>> {
        let v: Vec<Vec<Item>> = dirs.iter().map(|d| dump(d, set)).collect();
        for (d, r) in dirs.iter().zip(&v).skip(1) {
            assert_eq!(
                r.len(),
                v[0].len(),
                "{set}: {d:?} has a different number of questions"
            );
            for (a, b) in r.iter().zip(&v[0]) {
                assert!(
                    a.uid == b.uid && a.target == b.target,
                    "{set}: {d:?} is not aligned"
                );
            }
        }
        v
    }

    fn preds(runs: &[Vec<Item>]) -> Vec<Vec<Pred>> {
        runs.iter()
            .map(|r| r.iter().map(Item::to_pred).collect())
            .collect()
    }

    fn iv(i: Interval) -> Value {
        json!({"value": i.value, "lo": i.lo, "hi": i.hi})
    }

    fn pv(p: &Paired) -> Value {
        json!({"reference": p.reference, "variant": p.variant, "diff": iv(p.diff), "p_value": p.p_value})
    }

    // --------------------------------------------------------------- two-level bootstrap

    fn interval(draws: &mut Vec<f64>) -> (f64, f64) {
        draws.retain(|d| d.is_finite());
        draws.sort_by(f64::total_cmp);
        let a = (1.0 - LEVEL) / 2.0;
        (quantile(draws, a), quantile(draws, 1.0 - a))
    }

    fn aligned<T>(runs: &[Vec<T>]) -> usize {
        assert!(!runs.is_empty(), "no run");
        let n = runs[0].len();
        assert!(
            runs.iter().all(|r| r.len() == n),
            "the runs hold different items"
        );
        n
    }

    fn arm<T>(
        runs: &[Vec<T>],
        drawn: &[usize],
        idx: &[usize],
        stat: &impl Fn(&[T], &[usize]) -> f64,
    ) -> f64 {
        mean(drawn.iter().map(|&r| stat(&runs[r], idx)))
    }

    /// Interval of the mean over runs of `stat(items, idx)`.
    pub fn ci_runs<T>(runs: &[Vec<T>], stat: impl Fn(&[T], &[usize]) -> f64) -> Interval {
        let n = aligned(runs);
        let all: Vec<usize> = (0..n).collect();
        let every: Vec<usize> = (0..runs.len()).collect();
        let value = arm(runs, &every, &all, &stat);
        let mut draws = Vec::with_capacity(RESAMPLES);
        if n > 0 {
            let mut rng = Rng::new(0);
            let (mut idx, mut drawn) = (vec![0; n], vec![0; runs.len()]);
            for _ in 0..RESAMPLES {
                idx.iter_mut().for_each(|i| *i = rng.below(n));
                drawn.iter_mut().for_each(|r| *r = rng.below(runs.len()));
                draws.push(arm(runs, &drawn, &idx, &stat));
            }
        }
        let (lo, hi) = interval(&mut draws);
        Interval { value, lo, hi }
    }

    /// Paired comparison, variant − reference, of the mean over runs of `stat`.
    pub fn paired_runs<T>(
        reference: &[Vec<T>],
        variant: &[Vec<T>],
        stat: impl Fn(&[T], &[usize]) -> f64,
    ) -> Paired {
        let n = aligned(reference);
        assert_eq!(n, aligned(variant), "both arms need the same items");
        let all: Vec<usize> = (0..n).collect();
        let er: Vec<usize> = (0..reference.len()).collect();
        let ev: Vec<usize> = (0..variant.len()).collect();
        let (r, v) = (
            arm(reference, &er, &all, &stat),
            arm(variant, &ev, &all, &stat),
        );
        let mut draws = Vec::with_capacity(RESAMPLES);
        if n > 0 {
            let mut rng = Rng::new(0);
            let mut idx = vec![0; n];
            let (mut dr, mut dv) = (vec![0; reference.len()], vec![0; variant.len()]);
            for _ in 0..RESAMPLES {
                idx.iter_mut().for_each(|i| *i = rng.below(n));
                dr.iter_mut().for_each(|x| *x = rng.below(reference.len()));
                dv.iter_mut().for_each(|x| *x = rng.below(variant.len()));
                draws.push(arm(variant, &dv, &idx, &stat) - arm(reference, &dr, &idx, &stat));
            }
        }
        let (lo, hi) = interval(&mut draws);
        Paired {
            reference: r,
            variant: v,
            diff: Interval {
                value: v - r,
                lo,
                hi,
            },
            p_value: p_value(v - r, &draws),
        }
    }

    fn mean_of(v: &[f64], idx: &[usize]) -> f64 {
        mean(pick(v, idx).copied())
    }

    // ------------------------------------------------------------------------- calibrate

    fn bucket(k: usize) -> &'static str {
        match k {
            0..=2 => "2",
            3..=5 => "3_5",
            6..=10 => "6_10",
            _ => "11",
        }
    }

    fn nll_at(items: &[&Item], t: f64) -> f64 {
        items
            .iter()
            .map(|it| {
                let z: Vec<f64> = it.probs.iter().map(|p| p.max(1e-12).ln() / t).collect();
                let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lse = m + z.iter().map(|x| (x - m).exp()).sum::<f64>().ln();
                let s: f64 = it.target.iter().sum();
                -it.target
                    .iter()
                    .zip(&z)
                    .map(|(y, x)| y / s * (x - lse))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / items.len().max(1) as f64
    }

    /// Temperature minimizing the NLL (golden section on ln T in [ln 0.05, ln 20]).
    pub fn fit_t(items: &[&Item]) -> f64 {
        let (mut a, mut b) = (0.05f64.ln(), 20f64.ln());
        let g = (5f64.sqrt() - 1.0) / 2.0;
        let (mut c, mut d) = (b - g * (b - a), a + g * (b - a));
        let (mut fc, mut fd) = (nll_at(items, c.exp()), nll_at(items, d.exp()));
        for _ in 0..80 {
            if fc < fd {
                b = d;
                d = c;
                fd = fc;
                c = b - g * (b - a);
                fc = nll_at(items, c.exp());
            } else {
                a = c;
                c = d;
                fc = fd;
                d = a + g * (b - a);
                fd = nll_at(items, d.exp());
            }
        }
        ((a + b) / 2.0).exp()
    }

    pub fn temperatures(items: &[Item], min_bucket: usize) -> BTreeMap<String, f64> {
        let mut out = BTreeMap::new();
        let mut by_t: BTreeMap<&str, Vec<&Item>> = BTreeMap::new();
        let mut by_b: BTreeMap<String, Vec<&Item>> = BTreeMap::new();
        for it in items.iter().filter(|it| it.probs.len() >= 2) {
            by_t.entry(it.t.as_str()).or_default().push(it);
            by_b.entry(format!("{}.{}", it.t, bucket(it.probs.len())))
                .or_default()
                .push(it);
        }
        for (t, v) in &by_t {
            let temp = fit_t(v);
            eprintln!(
                "  {t:<14} n {:>6}  T {temp:.4}  NLL {:.4} -> {:.4}",
                v.len(),
                nll_at(v, 1.0),
                nll_at(v, temp)
            );
            out.insert(t.to_string(), temp);
        }
        for (k, v) in &by_b {
            if v.len() >= min_bucket {
                out.insert(k.clone(), fit_t(v));
            }
        }
        out
    }

    /// Copy of a GGUF file with the temperatures set.
    pub fn write_temperatures(
        input: &Path,
        output: &Path,
        temps: &BTreeMap<String, f64>,
    ) -> Result<(), String> {
        let bytes = std::fs::read(input).map_err(|e| format!("{input:?}: {e}"))?;
        let g = Gguf::parse(&bytes).map_err(|e| format!("{input:?}: {e}"))?;
        let arch = g.get_str("general.architecture").unwrap_or("modern-bert");
        let prefix = if arch == "skycmd-iso" {
            "skycmd-iso.temperature.".to_string()
        } else {
            format!("{arch}.decision.temperature.")
        };
        // Exports may carry neutral temperatures (Laya keeps 1.0 per type): they are dropped, any
        // other value means the dump was not produced at T = 1.
        for (k, v) in &g.kvs {
            if k.starts_with(&prefix) && v.as_f64().is_none_or(|t| (t - 1.0).abs() > 1e-6) {
                return Err(format!(
                    "{input:?} already has the temperature {k}: calibrate a T = 1 export"
                ));
            }
        }
        let mut w = GgufWriter::new().with_alignment(g.alignment);
        for (k, v) in &g.kvs {
            if *k != "general.alignment" && !k.starts_with(&prefix) {
                w.add_kv(*k, OwnedValue::from_value(v));
            }
        }
        for (k, t) in temps {
            w.add_kv(format!("{prefix}{k}"), OwnedValue::F32(*t as f32));
        }
        for t in &g.tensors {
            w.add_tensor(t.name, &t.dims, t.dtype, g.tensor_bytes(t).to_vec())
                .map_err(|e| e.to_string())?;
        }
        std::fs::write(output, w.to_bytes().map_err(|e| e.to_string())?)
            .map_err(|e| format!("{output:?}: {e}"))
    }

    // ------------------------------------------------------------------------- gates

    /// Majority-class accuracy of an evaluation file: the share of its most frequent gold id.
    fn majority(records: &Path) -> f64 {
        let text = std::fs::read_to_string(records).unwrap_or_default();
        let mut c: BTreeMap<String, usize> = BTreeMap::new();
        let mut n = 0usize;
        for l in text.lines().filter(|l| !l.trim().is_empty()) {
            let Ok(r) = serde_json::from_str::<Value>(l) else {
                continue;
            };
            let t: Vec<f64> = r["target"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default();
            let Some(kv) = r["opts_kv"].as_array() else {
                continue;
            };
            if t.is_empty() || kv.len() != t.len() {
                continue;
            }
            let id = kv[sky_metrics::argmax(&t)][0]
                .as_str()
                .unwrap_or_default()
                .to_string();
            *c.entry(id).or_default() += 1;
            n += 1;
        }
        c.values().copied().max().unwrap_or(0) as f64 / n.max(1) as f64
    }

    pub fn gates(dir: &Path, records: &Path) -> Value {
        let mut sets = serde_json::Map::new();
        let mut below = Vec::new();
        let mut flagged = Vec::new();
        let mut names: Vec<String> = std::fs::read_dir(dir)
            .map(|d| {
                d.filter_map(|e| e.ok())
                    .filter_map(|e| {
                        e.file_name()
                            .to_string_lossy()
                            .strip_suffix(".dump.jsonl")
                            .map(str::to_string)
                    })
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        let mut demo_en = None;
        for set in &names {
            let items = dump(dir, set);
            if items.is_empty() {
                continue;
            }
            if set == "demo-cases" {
                let en: Vec<f64> = items
                    .iter()
                    .filter(|it| it.src.ends_with("_en"))
                    .map(|it| f64::from(u8::from(it.correct())))
                    .collect();
                let fr: Vec<f64> = items
                    .iter()
                    .filter(|it| it.src.ends_with("_fr"))
                    .map(|it| f64::from(u8::from(it.correct())))
                    .collect();
                demo_en = Some(mean(en.iter().copied()));
                sets.insert(
                    set.clone(),
                    json!({"en": {"n": en.len(), "accuracy": mean(en.iter().copied())},
                           "fr": {"n": fr.len(), "accuracy": mean(fr.iter().copied())}}),
                );
                continue;
            }
            let ps: Vec<Pred> = items.iter().map(Item::to_pred).collect();
            let acc = ci_runs(std::slice::from_ref(&ps), |p, idx| accuracy(pick(p, idx)));
            let e = ece(&ps);
            let maj = majority(&records.join(format!("{set}.jsonl")));
            if acc.value < maj {
                below.push(set.clone());
            }
            if acc.value > 0.735 {
                flagged.push(set.clone());
            }
            sets.insert(
                set.clone(),
                json!({"n": ps.len(), "accuracy": iv(acc), "ece": e, "nll": nll(&ps),
                       "brier": brier(&ps), "majority": maj,
                       "uniform": mean(ps.iter().map(|p| 1.0 / p.k() as f64))}),
            );
        }
        let bench = sets
            .get(BENCH)
            .and_then(|v| v["accuracy"]["value"].as_f64())
            .unwrap_or(f64::NAN);
        let demo = demo_en.unwrap_or(f64::NAN);
        json!({
            "sets": sets,
            "gates": {
                "bench_ge_0_65": {"value": bench, "pass": bench >= 0.65},
                "demo_en_ge_0_90": {"value": demo, "pass": demo >= 0.90},
                "no_set_below_majority": {"below": below, "pass": below.is_empty()},
                "above_0_735_investigate": flagged,
            },
        })
    }

    // ------------------------------------------------------------------------- coherence

    /// One family: CH probs, s_k, n_k, d_ij (p_true, correct, gold is "true").
    #[derive(Default, Debug, Clone)]
    pub struct Family {
        ch: Vec<f64>,
        ch_correct: Option<bool>,
        s: BTreeMap<usize, f64>,
        n: BTreeMap<usize, f64>,
        d: BTreeMap<(usize, usize), (f64, bool, bool)>,
    }

    pub fn families(items: &[Item]) -> Vec<Family> {
        let mut fam: BTreeMap<String, Family> = BTreeMap::new();
        for it in items {
            let Some(rest) = it.uid.strip_prefix("coh/") else {
                continue;
            };
            let mut p = rest.splitn(3, '/');
            let (Some(g), Some(kind), Some(k)) = (p.next(), p.next(), p.next()) else {
                continue;
            };
            let f = fam.entry(g.to_string()).or_default();
            let pt = it.probs.get(1).copied().unwrap_or(f64::NAN);
            match kind {
                "CH" => {
                    f.ch = it.probs.clone();
                    f.ch_correct = Some(it.correct());
                }
                "S" => {
                    f.s.insert(k.parse().unwrap_or(0), pt);
                }
                "N" => {
                    f.n.insert(k.parse().unwrap_or(0), pt);
                }
                "D" => {
                    if let Some((i, j)) = k.split_once('-') {
                        f.d.insert(
                            (i.parse().unwrap_or(0), j.parse().unwrap_or(0)),
                            (pt, it.correct(), it.gold() == 1),
                        );
                    }
                }
                _ => {}
            }
        }
        fam.into_values().filter(|f| !f.s.is_empty()).collect()
    }

    /// Per-family statistics (NaN when the family lacks the parts).
    #[derive(Debug, Clone, Copy)]
    pub struct FamStats {
        mad_neg: f64,
        same_side: f64,
        logit_viol: f64,
        part_sum: f64,
        part_dev: f64,
        xf: f64,
        additivity: f64,
        description: f64,
        mono_checks: f64,
        mono_viol: f64,
        mono_excess: f64,
        d_correct: f64,
        d_true: f64,
        d_n: f64,
        ch_correct: f64,
    }

    fn logit(p: f64) -> f64 {
        let p = p.clamp(0.01, 0.99);
        (p / (1.0 - p)).ln()
    }

    pub fn fam_stats(f: &Family) -> FamStats {
        let nan = f64::NAN;
        let (mut mad, mut same, mut lv, mut np) = (0.0, 0.0, 0.0, 0.0);
        let (mut xf, mut nx) = (0.0, 0.0);
        for (k, &s) in &f.s {
            if let Some(&n) = f.n.get(k) {
                mad += (s + n - 1.0).abs();
                same += f64::from(u8::from((s - 0.5) * (n - 0.5) > 0.0));
                lv += (logit(s) + logit(n)).abs();
                np += 1.0;
            }
            if let Some(&c) = f.ch.get(*k) {
                xf += (s - c).abs();
                nx += 1.0;
            }
        }
        let complete = !f.ch.is_empty() && f.s.len() == f.ch.len();
        let part_sum: f64 = f.s.values().sum();
        let (mut add, mut na) = (0.0, 0.0);
        let (mut desc, mut nd) = (0.0, 0.0);
        let (mut checks, mut viol, mut excess) = (0.0, 0.0, 0.0);
        let (mut dc, mut dt, mut dn) = (0.0, 0.0, 0.0);
        let k = f.ch.len();
        for (&(i, j), &(d, ok, gold_true)) in &f.d {
            dc += f64::from(u8::from(ok));
            dt += f64::from(u8::from(gold_true));
            dn += 1.0;
            if let (Some(&si), Some(&sj)) = (f.s.get(&i), f.s.get(&j)) {
                add += (d - (si + sj)).abs();
                na += 1.0;
                for s in [si, sj] {
                    checks += 1.0;
                    let e = (s - d).max(0.0);
                    excess += e;
                    viol += f64::from(u8::from(e > 0.05));
                }
            }
            if k == 3 && i + j <= 3 {
                let m = 3 - i - j;
                if let Some(&nm) = f.n.get(&m) {
                    desc += (d - nm).abs();
                    nd += 1.0;
                }
            }
        }
        let r = |a: f64, n: f64| if n > 0.0 { a / n } else { nan };
        FamStats {
            mad_neg: r(mad, np),
            same_side: r(same, np),
            logit_viol: r(lv, np),
            part_sum: if complete { part_sum } else { nan },
            part_dev: if complete {
                (part_sum - 1.0).abs()
            } else {
                nan
            },
            xf: r(xf, nx),
            additivity: r(add, na),
            description: r(desc, nd),
            mono_checks: checks,
            mono_viol: viol,
            mono_excess: excess,
            d_correct: dc,
            d_true: dt,
            d_n: dn,
            ch_correct: f.ch_correct.map_or(nan, |c| f64::from(u8::from(c))),
        }
    }

    fn fmean(st: &[FamStats], idx: &[usize], f: impl Fn(&FamStats) -> f64) -> f64 {
        let v: Vec<f64> = idx
            .iter()
            .map(|&i| f(&st[i]))
            .filter(|x| x.is_finite())
            .collect();
        if v.is_empty() {
            f64::NAN
        } else {
            mean(v)
        }
    }

    fn ratio(
        st: &[FamStats],
        idx: &[usize],
        num: impl Fn(&FamStats) -> f64,
        den: impl Fn(&FamStats) -> f64,
    ) -> f64 {
        let (a, b) = idx
            .iter()
            .fold((0.0, 0.0), |(a, b), &i| (a + num(&st[i]), b + den(&st[i])));
        if b > 0.0 {
            a / b
        } else {
            f64::NAN
        }
    }

    fn fam_runs(dirs: &[PathBuf]) -> Vec<Vec<FamStats>> {
        runs(dirs, COHERENCE)
            .iter()
            .map(|r| families(r).iter().map(fam_stats).collect())
            .collect()
    }

    /// Majority-class rate of the D questions (their gold is "true" or "false").
    fn d_majority(st: &[FamStats]) -> f64 {
        let (t, n) = st
            .iter()
            .fold((0.0, 0.0), |(t, n), s| (t + s.d_true, n + s.d_n));
        if n > 0.0 {
            (t / n).max(1.0 - t / n)
        } else {
            f64::NAN
        }
    }

    pub fn coherence(dirs: &[PathBuf]) -> Value {
        let st = fam_runs(dirs);
        let m = |f: fn(&FamStats) -> f64| iv(ci_runs(&st, |s, idx| fmean(s, idx, f)));
        json!({
            "runs": dirs.len(),
            "families": st[0].len(),
            "mad_neg": m(|s| s.mad_neg),
            "same_side_rate": m(|s| s.same_side),
            "logit_violation": m(|s| s.logit_viol),
            "partition_sum": m(|s| s.part_sum),
            "partition_abs_dev": m(|s| s.part_dev),
            "xf_abs": m(|s| s.xf),
            "choice_accuracy": m(|s| s.ch_correct),
            "held_out": {
                "additivity_abs": m(|s| s.additivity),
                "description_abs": m(|s| s.description),
                "monotonicity_violation_share": iv(ci_runs(&st, |s, idx| ratio(s, idx, |x| x.mono_viol, |x| x.mono_checks))),
                "monotonicity_mean_excess": iv(ci_runs(&st, |s, idx| ratio(s, idx, |x| x.mono_excess, |x| x.mono_checks))),
                "d_accuracy": iv(ci_runs(&st, |s, idx| ratio(s, idx, |x| x.d_correct, |x| x.d_n))),
                "d_majority": d_majority(&st[0]),
            },
        })
    }

    // ------------------------------------------------------------------------- robust

    fn parts(uid: &str) -> Vec<&str> {
        uid.split('/').collect()
    }

    /// Per base (in base order): (flip, total variation, orig correct, paraphrase correct).
    fn paraphrase_rows(items: &[Item]) -> Vec<[f64; 4]> {
        let mut by: BTreeMap<&str, (Option<&Item>, Option<&Item>)> = BTreeMap::new();
        for it in items {
            let p = parts(&it.uid);
            if p.len() == 3 && p[0] == "para" {
                let e = by.entry(p[1]).or_default();
                if p[2] == "orig" {
                    e.0 = Some(it);
                } else {
                    e.1 = Some(it);
                }
            }
        }
        by.values()
            .filter_map(|(a, b)| Some((*a)?).zip(*b))
            .map(|(a, b)| {
                let tv = 0.5
                    * a.probs
                        .iter()
                        .zip(&b.probs)
                        .map(|(x, y)| (x - y).abs())
                        .sum::<f64>();
                [
                    f64::from(u8::from(a.pred() != b.pred())),
                    tv,
                    f64::from(u8::from(a.correct())),
                    f64::from(u8::from(b.correct())),
                ]
            })
            .collect()
    }

    /// Per base: (flip under the worst distractor, attack success on a correct base or NaN,
    /// drop of p(gold)).
    fn jevout_rows(items: &[Item]) -> Vec<[f64; 3]> {
        let mut by: BTreeMap<&str, (Option<&Item>, Vec<&Item>)> = BTreeMap::new();
        for it in items {
            let p = parts(&it.uid);
            if p.len() == 3 && p[0] == "distractor" {
                let e = by.entry(p[1]).or_default();
                if p[2] == "base" {
                    e.0 = Some(it);
                } else {
                    e.1.push(it);
                }
            }
        }
        by.values()
            .filter_map(|(base, vars)| {
                let base = (*base)?;
                let g = base.gold();
                let worst = vars
                    .iter()
                    .min_by(|a, b| a.probs[g].total_cmp(&b.probs[g]))?;
                Some([
                    f64::from(u8::from(worst.pred() != base.pred())),
                    if base.correct() {
                        f64::from(u8::from(!worst.correct()))
                    } else {
                        f64::NAN
                    },
                    base.probs[g] - worst.probs[g],
                ])
            })
            .collect()
    }

    fn col<const N: usize>(rows: &[[f64; N]], i: usize) -> Vec<f64> {
        rows.iter()
            .map(|r| r[i])
            .filter(|x| x.is_finite())
            .collect()
    }

    fn paraphrase(items: &[Item]) -> Value {
        let rows = paraphrase_rows(items);
        let m = |i: usize| iv(ci_runs(&[col(&rows, i)], mean_of));
        json!({"n": rows.len(), "flip_rate": m(0), "total_variation": m(1),
               "accuracy_orig": m(2), "accuracy_paraphrased": m(3)})
    }

    fn jevout(items: &[Item]) -> Value {
        let rows = jevout_rows(items);
        let m = |i: usize| iv(ci_runs(&[col(&rows, i)], mean_of));
        json!({"n": rows.len(), "flip_rate": m(0), "attack_success_on_correct": m(1),
               "mean_gold_drop": m(2)})
    }

    /// AUC of `score` for the positive class (ties count one half).
    fn auc(pairs: &[(f64, bool)]) -> f64 {
        let pos: Vec<f64> = pairs.iter().filter(|p| p.1).map(|p| p.0).collect();
        let neg: Vec<f64> = pairs.iter().filter(|p| !p.1).map(|p| p.0).collect();
        if pos.is_empty() || neg.is_empty() {
            return f64::NAN;
        }
        let mut s = 0.0;
        for p in &pos {
            for n in &neg {
                s += if p > n {
                    1.0
                } else if p == n {
                    0.5
                } else {
                    0.0
                };
            }
        }
        s / (pos.len() * neg.len()) as f64
    }

    /// (aligned, reassigned) of one base and name pair.
    type Arms<'a> = (Option<&'a Item>, Option<&'a Item>);
    /// AUC inputs of the aligned and reassigned arms.
    type AucPairs = (Vec<(f64, bool)>, Vec<(f64, bool)>);

    fn reassign(items: &[Item]) -> Value {
        // (base, pair) -> (aligned, reassigned)
        let mut by: BTreeMap<(&str, &str), Arms> = BTreeMap::new();
        for it in items {
            let p = parts(&it.uid);
            if p.len() == 4 && p[0] == "reassign" {
                let e = by.entry((p[1], p[2])).or_default();
                if p[3] == "aligned" {
                    e.0 = Some(it);
                } else {
                    e.1 = Some(it);
                }
            }
        }
        let mut flips: BTreeMap<String, BTreeMap<&str, f64>> = BTreeMap::new();
        let mut aucs: BTreeMap<String, AucPairs> = BTreeMap::new();
        for ((base, pair), (a, r)) in &by {
            let (Some(a), Some(r)) = (a, r) else { continue };
            // The chosen definition: aligned 0 -> d_no, 1 -> d_yes; reassigned 0 -> d_yes, 1 -> d_no.
            let def_a = a.pred();
            let def_r = 1 - r.pred().min(1);
            flips
                .entry(pair.to_string())
                .or_default()
                .insert(base, f64::from(u8::from(def_a != def_r)));
            let yes = a.gold() == 1;
            let e = aucs.entry(pair.to_string()).or_default();
            e.0.push((a.probs[1], yes));
            e.1.push((r.probs[0], yes));
        }
        let control = flips
            .iter()
            .find(|(k, _)| k.starts_with("0-control"))
            .map(|(_, v)| v.clone());
        let mut out = serde_json::Map::new();
        for (pair, f) in &flips {
            let v: Vec<f64> = f.values().copied().collect();
            let mut row = json!({"n": v.len(), "flip_rate": iv(ci_runs(&[v], mean_of)),
                "auc_aligned": auc(&aucs[pair].0), "auc_reassigned": auc(&aucs[pair].1)});
            if let Some(c) = &control {
                let common: Vec<&str> = f.keys().filter(|k| c.contains_key(*k)).copied().collect();
                let rv: Vec<f64> = common.iter().map(|k| c[k]).collect();
                let vv: Vec<f64> = common.iter().map(|k| f[k]).collect();
                row["minus_control_0_1"] = iv(paired_runs(&[rv], &[vv], mean_of).diff);
            }
            out.insert(pair.clone(), row);
        }
        Value::Object(out)
    }

    fn policy(items: &[Item]) -> Value {
        let mut out = serde_json::Map::new();
        for v in ["policybench_plain", "policybench_xf"] {
            let c: Vec<f64> = items
                .iter()
                .filter(|it| it.src == v)
                .map(|it| f64::from(u8::from(it.correct())))
                .collect();
            if !c.is_empty() {
                out.insert(
                    v.into(),
                    json!({"n": c.len(), "accuracy": iv(ci_runs(&[c], mean_of))}),
                );
            }
        }
        Value::Object(out)
    }

    /// Detection / false rejection with the rejection score `P(none)` (the last option).
    fn coverage(items: &[Item]) -> Value {
        let mut out = serde_json::Map::new();
        // set -> part -> kind -> (score, gold is none)
        type Kinds = BTreeMap<String, Vec<(f64, bool)>>;
        let mut by: BTreeMap<&str, BTreeMap<&str, Kinds>> = BTreeMap::new();
        for it in items {
            let p = parts(&it.uid);
            if p.len() >= 5 && p[0] == "cov" && !it.probs.is_empty() {
                let none = it.probs.len() - 1;
                by.entry(p[1])
                    .or_default()
                    .entry(p[2])
                    .or_default()
                    .entry(p[3].to_string())
                    .or_default()
                    .push((it.probs[none], it.gold() == none));
            }
        }
        let pool = |m: &Kinds, kinds: &[&str]| -> Vec<(f64, bool)> {
            kinds
                .iter()
                .filter_map(|k| m.get(*k))
                .flatten()
                .copied()
                .collect()
        };
        let rate = |v: &[(f64, bool)], absent: bool, t: f64| {
            let s: Vec<&(f64, bool)> = v.iter().filter(|x| x.1 == absent).collect();
            s.iter().filter(|x| x.0 >= t).count() as f64 / s.len().max(1) as f64
        };
        for (set, parts) in &by {
            let (Some(cal), Some(test)) = (parts.get("cal"), parts.get("test")) else {
                continue;
            };
            let c = pool(cal, &["full", "omit"]);
            let mut cands: Vec<f64> = c.iter().map(|x| x.0).collect();
            cands.sort_by(|a, b| a.total_cmp(b));
            cands.dedup();
            // The lowest threshold whose false rejection (menus holding the gold) is ≤ 5 %.
            let t = cands
                .iter()
                .copied()
                .find(|&t| rate(&c, false, t) <= 0.05)
                .unwrap_or(f64::INFINITY);
            let mut row = serde_json::Map::new();
            row.insert("threshold".into(), json!(t));
            for kind in ["omit", "bm25-2", "bm25-5", "bm25-10"] {
                let v = match kind {
                    "omit" => pool(test, &["full", "omit"]),
                    k => pool(test, &[k]),
                };
                let covered = v.iter().filter(|x| !x.1).count() as f64 / v.len().max(1) as f64;
                row.insert(
                    kind.into(),
                    json!({"n": v.len(), "detection": rate(&v, true, t),
                           "false_rejection": rate(&v, false, t), "gold_in_menu": covered}),
                );
            }
            out.insert(set.to_string(), Value::Object(row));
        }
        Value::Object(out)
    }

    pub fn robust(dir: &Path) -> Value {
        json!({
            "paraphrase": paraphrase(&dump(dir, "paraphrase_test")),
            "jevout_lite": jevout(&dump(dir, "jevout_lite")),
            "reassign": reassign(&dump(dir, "reassign_test")),
            "policybench": policy(&dump(dir, "policy_test")),
            "coverage": coverage(&dump(dir, "coverage_test")),
        })
    }

    // ------------------------------------------------------------------------- selective

    /// `P(Bin(n, a) <= k)`.
    fn binom_cdf(k: usize, n: usize, a: f64) -> f64 {
        let mut p = (1.0 - a).powi(n as i32);
        let mut s = p;
        for i in 0..k.min(n) {
            p *= (n - i) as f64 / (i + 1) as f64 * a / (1.0 - a);
            s += p;
        }
        s.min(1.0)
    }

    /// Learn-then-Test threshold on max p for a risk level `alpha`: thresholds from high to low
    /// (fixed sequence), each tested with the binomial p-value of the errors among the accepted;
    /// the lowest threshold before the first non-rejection at level `delta`.
    pub fn ltt(cal: &[&Item], alpha: f64, delta: f64) -> Option<f64> {
        let mut best = None;
        for i in 0..200 {
            let t = 1.0 - i as f64 / 200.0;
            let acc: Vec<&&Item> = cal.iter().filter(|it| it.probs[it.pred()] >= t).collect();
            if acc.is_empty() {
                continue;
            }
            let errs = acc.iter().filter(|it| !it.correct()).count();
            if binom_cdf(errs, acc.len(), alpha) <= delta {
                best = Some(t);
            } else {
                break;
            }
        }
        best
    }

    pub fn selective(cal: &[Item], test: &[Item]) -> Value {
        let mut out = serde_json::Map::new();
        let mut types: Vec<String> = test.iter().map(|i| i.t.clone()).collect();
        types.sort();
        types.dedup();
        let rate = |e: usize, a: usize| if a > 0 { e as f64 / a as f64 } else { f64::NAN };
        for alpha in [0.01, 0.02, 0.05, 0.10] {
            let (mut acc, mut err, mut n) = (0usize, 0usize, 0usize);
            let (mut cacc, mut cerr) = (0usize, 0usize);
            let mut per = serde_json::Map::new();
            for t in &types {
                let c: Vec<&Item> = cal.iter().filter(|i| &i.t == t).collect();
                let te: Vec<&Item> = test.iter().filter(|i| &i.t == t).collect();
                n += te.len();
                let thr = ltt(&c, alpha, 0.05);
                let (mut a_t, mut e_t) = (0, 0);
                if let Some(thr) = thr {
                    for it in &te {
                        if it.probs[it.pred()] >= thr {
                            a_t += 1;
                            e_t += usize::from(!it.correct());
                        }
                    }
                }
                acc += a_t;
                err += e_t;
                // Mondrian conformal: q = the ceil((n+1)(1-alpha))-th smallest 1 - p_y on cal.
                let mut sc: Vec<f64> = c.iter().map(|it| 1.0 - it.probs[it.gold()]).collect();
                sc.sort_by(|a, b| a.total_cmp(b));
                let r = (((sc.len() + 1) as f64) * (1.0 - alpha)).ceil() as usize;
                let q = if sc.is_empty() || r > sc.len() {
                    f64::INFINITY
                } else {
                    sc[r.max(1) - 1]
                };
                for it in &te {
                    if it.probs.iter().filter(|p| 1.0 - **p <= q).count() == 1 {
                        cacc += 1;
                        cerr += usize::from(!it.correct());
                    }
                }
                per.insert(
                    t.clone(),
                    json!({"threshold": thr, "accepted": a_t, "of": te.len(), "cal": c.len(),
                           "error_accepted": rate(e_t, a_t)}),
                );
            }
            out.insert(
                format!("{:.0}%", alpha * 100.0),
                json!({
                    "ltt": {"coverage": acc as f64 / n.max(1) as f64,
                            "error_accepted": rate(err, acc), "per_type": per},
                    "conformal_singletons": {"coverage": cacc as f64 / n.max(1) as f64,
                            "error_accepted": rate(cerr, cacc)},
                }),
            );
        }
        let all: Vec<Pred> = test.iter().map(Item::to_pred).collect();
        let s = sky_metrics::selective(&all);
        json!({"by_alpha": out, "aurc": s.aurc, "coverage_at_5pct_error": s.coverage_at_5pct_error})
    }

    // ------------------------------------------------------------------------- bets

    /// The English zero-shot items of a run, tagged with their set index.
    fn zero_shot(dirs: &[PathBuf]) -> Vec<Vec<(usize, Pred)>> {
        let per_set: Vec<Vec<Vec<Pred>>> =
            ZERO_SHOT.iter().map(|s| preds(&runs(dirs, s))).collect();
        (0..dirs.len())
            .map(|r| {
                per_set
                    .iter()
                    .enumerate()
                    .flat_map(|(s, runs)| runs[r].iter().map(move |p| (s, p.clone())))
                    .collect()
            })
            .collect()
    }

    fn macro_stat(items: &[(usize, Pred)], idx: &[usize], f: fn(&[&Pred]) -> f64) -> f64 {
        let mut by: Vec<Vec<&Pred>> = vec![Vec::new(); ZERO_SHOT.len()];
        for &i in idx {
            by[items[i].0].push(&items[i].1);
        }
        mean(by.iter().filter(|v| !v.is_empty()).map(|v| f(v)))
    }

    fn acc_of(v: &[&Pred]) -> f64 {
        accuracy(v.iter().copied())
    }

    fn ece_of(v: &[&Pred]) -> f64 {
        ece(v.iter().copied())
    }

    fn per_run(runs: &[Vec<Pred>], f: fn(&[&Pred]) -> f64) -> Vec<f64> {
        runs.iter()
            .map(|r| f(&r.iter().collect::<Vec<_>>()))
            .collect()
    }

    fn holm_table(rows: Vec<(&str, Paired)>) -> Value {
        let adj = holm(&rows.iter().map(|(_, p)| p.p_value).collect::<Vec<_>>());
        Value::Object(
            rows.iter()
                .zip(adj)
                .map(|((k, p), a)| {
                    let mut v = pv(p);
                    v["p_holm"] = json!(a);
                    (k.to_string(), v)
                })
                .collect(),
        )
    }

    pub fn pari1(rd: &[PathBuf], vd: &[PathBuf], acc_margin: f64, ece_margin: f64) -> Value {
        let (rt, vt) = (preds(&runs(rd, BENCH)), preds(&runs(vd, BENCH)));
        let acc = paired_runs(&rt, &vt, |p, idx| accuracy(pick(p, idx)));
        let ec = paired_runs(&rt, &vt, |p, idx| ece(pick(p, idx)));
        let acc_ok = acc.non_inferior(Better::Higher, acc_margin);
        let ece_ok = ec.non_inferior(Better::Lower, ece_margin);
        let flips = |ds: &[PathBuf], set: &str, rows: fn(&[Item]) -> Vec<f64>| -> Vec<Vec<f64>> {
            runs(ds, set).iter().map(|r| rows(r)).collect()
        };
        let para = |r: &[Item]| col(&paraphrase_rows(r), 0);
        let jev = |r: &[Item]| col(&jevout_rows(r), 0);
        let (rz, vz) = (zero_shot(rd), zero_shot(vd));
        let secondary = holm_table(vec![
            (
                "bench_nll",
                paired_runs(&rt, &vt, |p, idx| nll(pick(p, idx))),
            ),
            (
                "bench_brier",
                paired_runs(&rt, &vt, |p, idx| brier(pick(p, idx))),
            ),
            (
                "paraphrase_flip_rate",
                paired_runs(
                    &flips(rd, "paraphrase_test", para),
                    &flips(vd, "paraphrase_test", para),
                    mean_of,
                ),
            ),
            (
                "jevout_flip_rate",
                paired_runs(
                    &flips(rd, "jevout_lite", jev),
                    &flips(vd, "jevout_lite", jev),
                    mean_of,
                ),
            ),
            (
                "zero_shot_macro_accuracy",
                paired_runs(&rz, &vz, |it, idx| macro_stat(it, idx, acc_of)),
            ),
            (
                "zero_shot_macro_ece",
                paired_runs(&rz, &vz, |it, idx| macro_stat(it, idx, ece_of)),
            ),
        ]);
        json!({
            "reference_runs": rd, "variant_runs": vd,
            "primary": {
                "bench_accuracy": {"paired": pv(&acc), "worse_upper": acc.worse_upper(Better::Higher),
                    "margin": acc_margin, "non_inferior": acc_ok,
                    "reference_per_run": per_run(&rt, acc_of), "variant_per_run": per_run(&vt, acc_of)},
                "bench_ece": {"paired": pv(&ec), "worse_upper": ec.worse_upper(Better::Lower),
                    "margin": ece_margin, "non_inferior": ece_ok,
                    "reference_per_run": per_run(&rt, ece_of), "variant_per_run": per_run(&vt, ece_of)},
            },
            "success": acc_ok && ece_ok,
            "secondary_holm": secondary,
        })
    }

    pub fn pari2(rd: &[PathBuf], vd: &[PathBuf], acc_margin: f64) -> Value {
        let (rf, vf) = (fam_runs(rd), fam_runs(vd));
        let d_acc = ci_runs(&rf, |s, idx| ratio(s, idx, |x| x.d_correct, |x| x.d_n));
        let d_maj = d_majority(&rf[0]);
        let competent = d_acc.lo > d_maj + 0.10;
        let mad = ci_runs(&vf, |s, idx| fmean(s, idx, |x| x.mad_neg));
        let part = ci_runs(&vf, |s, idx| fmean(s, idx, |x| x.part_sum));
        let (rt, vt) = (preds(&runs(rd, BENCH)), preds(&runs(vd, BENCH)));
        let acc = paired_runs(&rt, &vt, |p, idx| accuracy(pick(p, idx)));
        let acc_ok = acc.non_inferior(Better::Higher, acc_margin);
        let mono = paired_runs(&rf, &vf, |s, idx| {
            ratio(s, idx, |x| x.mono_viol, |x| x.mono_checks)
        });
        let reduction = if mono.reference > 0.0 {
            1.0 - mono.variant / mono.reference
        } else {
            f64::NAN
        };
        let mono_ok = reduction >= 0.30 && mono.diff.hi < 0.0;
        let mad_ok = mad.value < 0.03;
        let part_ok = (0.97..=1.03).contains(&part.value);
        let fm = |f: fn(&FamStats) -> f64| move |s: &[FamStats], idx: &[usize]| fmean(s, idx, f);
        let secondary = holm_table(vec![
            ("mad_neg", paired_runs(&rf, &vf, fm(|x| x.mad_neg))),
            (
                "partition_abs_dev",
                paired_runs(&rf, &vf, fm(|x| x.part_dev)),
            ),
            ("xf_abs", paired_runs(&rf, &vf, fm(|x| x.xf))),
            (
                "additivity_abs",
                paired_runs(&rf, &vf, fm(|x| x.additivity)),
            ),
            (
                "description_abs",
                paired_runs(&rf, &vf, fm(|x| x.description)),
            ),
            (
                "bench_ece",
                paired_runs(&rt, &vt, |p, idx| ece(pick(p, idx))),
            ),
        ]);
        json!({
            "reference_runs": rd, "variant_runs": vd,
            "competence": {"d_accuracy_lambda0": iv(d_acc), "d_majority": d_maj,
                "required_lower_bound": d_maj + 0.10, "competent": competent},
            "primary": {
                "mad_neg": {"interval": iv(mad), "threshold": 0.03, "pass": mad_ok},
                "partition_sum": {"interval": iv(part), "range": [0.97, 1.03], "pass": part_ok},
                "bench_accuracy": {"paired": pv(&acc), "worse_upper": acc.worse_upper(Better::Higher),
                    "margin": acc_margin, "non_inferior": acc_ok},
                "monotonicity_violation_share": {"paired": pv(&mono), "relative_reduction": reduction,
                    "required_reduction": 0.30, "pass": mono_ok,
                    "measurable": competent},
            },
            "success": competent && mad_ok && part_ok && acc_ok && mono_ok,
            "trained_identities_only": mad_ok && part_ok && acc_ok && !(competent && mono_ok),
            "secondary_holm": secondary,
        })
    }

    // ------------------------------------------------------------------------- summary

    fn read_json(p: &Path) -> Option<Value> {
        serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()
    }

    /// `file stem -> content` of the JSON files of a directory.
    fn json_dir(dir: &Path) -> serde_json::Map<String, Value> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        files.sort();
        let mut m = serde_json::Map::new();
        for p in files {
            if p.extension().is_some_and(|e| e == "json") {
                if let (Some(stem), Some(v)) = (p.file_stem(), read_json(&p)) {
                    m.insert(stem.to_string_lossy().to_string(), v);
                }
            }
        }
        m
    }

    /// The shipped run by the pre-registered rule, if bet 1 has been analysed.
    fn shipped(
        reports: &serde_json::Map<String, Value>,
        selection: &serde_json::Map<String, Value>,
        runs: &serde_json::Map<String, Value>,
    ) -> Option<String> {
        let ok = |k: &str| reports.get(k).and_then(|v| v["success"].as_bool()) == Some(true);
        reports.get("pari1")?;
        let names: Vec<String> = if ok("pari2") {
            let ls = selection.get("lambda")?;
            let (tag, l) = (ls["support"].as_str()?, ls["lambda_star"].as_str()?);
            (0..3).map(|s| format!("l-{tag}-{l}-s{s}")).collect()
        } else if ok("pari1") {
            (0..3).map(|s| format!("f-b-s{s}")).collect()
        } else {
            (0..3).map(|s| format!("f-ap-s{s}")).collect()
        };
        names
            .into_iter()
            .filter_map(|n| {
                let nll = runs.get(&n)?["all"]["nll"].as_f64()?;
                Some((n, nll))
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(n, _)| n)
    }

    pub fn summary(store: &Path) -> Value {
        let mut runs = serde_json::Map::new();
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(store.join("runs"))
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        dirs.sort();
        for d in dirs {
            let Some(name) = d
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_prefix("s5-"))
                .map(str::to_string)
            else {
                continue;
            };
            let Some(v) = read_json(&d.join("val.json")) else {
                continue;
            };
            let init = v["init"]
                .as_str()
                .and_then(|p| Path::new(p).file_name())
                .map(|n| n.to_string_lossy().to_string());
            runs.insert(
                name,
                json!({"variant": v["variant"], "init": init, "lr": v["lr"], "seed": v["seed"],
                       "coh": v["coh"], "step": v["step"], "minutes": v["minutes"],
                       "all": v["all"], "coherence": v["coherence"]}),
            );
        }
        let log: Vec<Value> = std::fs::read_to_string(store.join("runs/s5-test-log.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect();
        let mut commits: Vec<&str> = log
            .iter()
            .filter_map(|v| v["prereg_commit"].as_str())
            .collect();
        commits.sort_unstable();
        commits.dedup();
        let times: Vec<u64> = log.iter().filter_map(|v| v["unix_time"].as_u64()).collect();
        let reports = json_dir(&store.join("runs/s5-report"));
        let selection = json_dir(&store.join("runs/s5-select"));
        let ship = shipped(&reports, &selection, &runs);
        json!({
            "generated_unix": std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs()),
            "shipped": ship,
            "test_log": {"passes": log.len(), "prereg_commits": commits,
                         "first_unix": times.iter().min(), "last_unix": times.iter().max()},
            "selection": selection,
            "runs": runs,
            "reports": reports,
        })
    }

    // ------------------------------------------------------------------------- CLI

    fn arg(args: &[String], k: &str) -> Option<String> {
        args.iter()
            .position(|a| a == k)
            .and_then(|i| args.get(i + 1).cloned())
    }

    fn list(s: &str) -> Vec<PathBuf> {
        s.split(',')
            .filter(|x| !x.is_empty())
            .map(PathBuf::from)
            .collect()
    }

    fn emit(v: Value, args: &[String]) {
        let text = serde_json::to_string_pretty(&v).unwrap();
        if let Some(p) = arg(args, "--json") {
            std::fs::write(&p, text + "\n").unwrap_or_else(|e| panic!("{p}: {e}"));
            eprintln!("wrote {p}");
        } else {
            println!("{text}");
        }
    }

    pub fn main() {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let usage = || -> ! {
            eprintln!(
                "usage: sky-s5-report calibrate|gates|coherence|robust|selective|pari1|pari2|summary ... (see the source)"
            );
            std::process::exit(2)
        };
        let Some(cmd) = args.first() else { usage() };
        let pos = |i: usize| -> String { args.get(i).cloned().unwrap_or_else(|| usage()) };
        let need = |k: &str| -> String { arg(&args, k).unwrap_or_else(|| usage()) };
        let num =
            |k: &str, d: f64| -> f64 { arg(&args, k).and_then(|v| v.parse().ok()).unwrap_or(d) };
        match cmd.as_str() {
            "calibrate" => {
                let min = num("--min-bucket", 100.0) as usize;
                let temps = temperatures(&read(Path::new(&pos(1))), min);
                let (input, out) = (need("--gguf"), need("--out"));
                if let Err(e) = write_temperatures(Path::new(&input), Path::new(&out), &temps) {
                    eprintln!("sky-s5-report: {e}");
                    std::process::exit(1);
                }
                eprintln!("wrote {out}");
                emit(json!(temps), &args);
            }
            "gates" => emit(
                gates(Path::new(&pos(1)), Path::new(&need("--records"))),
                &args,
            ),
            "coherence" => emit(coherence(&list(&pos(1))), &args),
            "robust" => emit(robust(Path::new(&pos(1))), &args),
            "selective" => {
                let src = arg(&args, "--cal-src");
                let cal: Vec<Item> = read(Path::new(&pos(1)))
                    .into_iter()
                    .filter(|it| src.as_ref().is_none_or(|s| it.src.starts_with(s.as_str())))
                    .collect();
                emit(selective(&cal, &dump(Path::new(&pos(2)), BENCH)), &args)
            }
            "pari1" => emit(
                pari1(
                    &list(&need("--ref")),
                    &list(&need("--variant")),
                    num("--acc-margin", 0.02),
                    num("--ece-margin", 0.01),
                ),
                &args,
            ),
            "summary" => {
                let out = need("--out");
                let v = summary(Path::new(&need("--store")));
                std::fs::write(&out, v.to_string() + "\n").unwrap_or_else(|e| panic!("{out}: {e}"));
                eprintln!("wrote {out} (shipped: {})", v["shipped"]);
            }
            "pari2" => emit(
                pari2(
                    &list(&need("--ref")),
                    &list(&need("--variant")),
                    num("--acc-margin", 0.02),
                ),
                &args,
            ),
            _ => usage(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn it(uid: &str, probs: &[f64], gold: usize) -> Item {
            let mut target = vec![0.0; probs.len()];
            target[gold] = 1.0;
            Item {
                src: "x".into(),
                t: "noul".into(),
                probs: probs.to_vec(),
                target,
                uid: uid.into(),
            }
        }

        #[test]
        fn coherence_of_a_perfect_model_is_perfect() {
            let c = [0.6, 0.3, 0.1];
            let mut v = vec![it("coh/g/CH/0", &c, 0)];
            for (k, &ck) in c.iter().enumerate() {
                v.push(it(
                    &format!("coh/g/S/{k}"),
                    &[1.0 - ck, ck],
                    usize::from(ck > 0.5),
                ));
                v.push(it(
                    &format!("coh/g/N/{k}"),
                    &[ck, 1.0 - ck],
                    usize::from(ck < 0.5),
                ));
            }
            for (i, j) in [(0, 1), (0, 2), (1, 2)] {
                let d: f64 = c[i] + c[j];
                v.push(it(
                    &format!("coh/g/D/{i}-{j}"),
                    &[1.0 - d, d],
                    usize::from(d > 0.5),
                ));
            }
            let st = fam_stats(&families(&v)[0]);
            assert!(st.mad_neg.abs() < 1e-12);
            assert!((st.part_sum - 1.0).abs() < 1e-12);
            assert!(st.additivity.abs() < 1e-12 && st.description.abs() < 1e-12);
            assert_eq!(st.mono_viol, 0.0);
            assert_eq!((st.d_n, st.d_correct, st.d_true), (3.0, 3.0, 2.0));
            assert!((d_majority(&[st]) - 2.0 / 3.0).abs() < 1e-12);
        }

        #[test]
        fn ltt_and_binomial() {
            assert!((binom_cdf(0, 10, 0.1) - 0.9f64.powi(10)).abs() < 1e-12);
            assert!((binom_cdf(10, 10, 0.3) - 1.0).abs() < 1e-12);
            // 300 confident correct answers and 300 unsure wrong ones.
            let mut cal = Vec::new();
            for i in 0..300 {
                cal.push(it(&format!("a{i}"), &[0.05, 0.95], 1));
                cal.push(it(&format!("b{i}"), &[0.45, 0.55], 0));
            }
            let refs: Vec<&Item> = cal.iter().collect();
            let t = ltt(&refs, 0.05, 0.05).unwrap();
            assert!(t > 0.55 && t <= 0.95, "{t}");
            assert!(auc(&[(0.9, true), (0.1, false)]) == 1.0);
        }

        #[test]
        fn two_level_bootstrap() {
            // One run: the interval of a proportion is about ±1.96·√(0.25/400).
            let v: Vec<f64> = (0..400).map(|i| (i % 2) as f64).collect();
            let c = ci_runs(std::slice::from_ref(&v), mean_of);
            assert_eq!(c.value, 0.5);
            assert!((0.04..0.06).contains(&(c.value - c.lo)), "{c:?}");
            // Identical arms: a zero difference, p = 1.
            let p = paired_runs(&[v.clone(), v.clone()], std::slice::from_ref(&v), mean_of);
            assert_eq!(p.diff.value, 0.0);
            assert_eq!(p.p_value, 1.0);
            // Runs that disagree widen the interval of their mean.
            let w: Vec<f64> = (0..400).map(|i| ((i % 4) < 3) as u8 as f64).collect();
            let two = ci_runs(&[v.clone(), w.clone()], mean_of);
            assert!((two.value - 0.625).abs() < 1e-12);
            assert!(two.hi - two.lo > c.hi - c.lo);
            // A better variant: the difference interval is above zero.
            let better = paired_runs(&[v], &[w], mean_of);
            assert!(better.diff.lo > 0.0 && better.non_inferior(Better::Higher, 0.0));
        }

        #[test]
        fn temperature_recovers_a_sharpening() {
            // probs = softmax(2 z) of calibrated softmax(z): the fitted T is about 2.
            let mut items = Vec::new();
            let mut rng = Rng::new(7);
            for i in 0..4000 {
                let z = [0.0, rng.f64() * 4.0 - 2.0];
                let p1 = 1.0 / (1.0 + (-z[1]).exp());
                let gold = usize::from(rng.f64() < p1);
                let q1 = 1.0 / (1.0 + (-2.0 * z[1]).exp());
                items.push(it(&format!("{i}"), &[1.0 - q1, q1], gold));
            }
            let refs: Vec<&Item> = items.iter().collect();
            let t = fit_t(&refs);
            assert!((t - 2.0).abs() < 0.2, "{t}");
        }
    }
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    native::main();
}
