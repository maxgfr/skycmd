//! s5 analyses of `sky-eval --dump-probs` files, as pre-registered in `docs/s5-prereg.md`. A dump
//! has one line per question (`src`, `t`, `k`, `probs`, `target`, `gold`, `uid`); an evaluation
//! directory holds `<set>.dump.jsonl` for each evaluation file `<set>.jsonl`.
//!
//! ```text
//! sky-s5-report calibrate <cal.dump> --gguf in.gguf --out out.gguf [--json t.json] [--min-bucket 100]
//! sky-s5-report gates <eval dir> --records <data/s5/eval> [--json out.json]
//! sky-s5-report coherence <eval dir>[,<eval dir>...] [--set coherence_test] [--json out.json]
//! sky-s5-report robust <eval dir> [--json out.json]
//! sky-s5-report selective <cal.dump> <eval dir> [--cal-src typed_decisions_train] [--json out.json]
//! sky-s5-report pari1 --ref <dirs> --variant <dirs> [--json out.json]
//! sky-s5-report pari2 --ref <dirs> --variant <dirs> [--json out.json]
//! sky-s5-report pari3 --store <store> [--format q8_0] [--cal-src typed_decisions_train] [--json out.json] [--trajectory cascade.json]
//! sky-s5-report retemper --run <run dir> [--src-prefix typed_decisions] --out <dir> [--json out.json]
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
//!   the D questions against the gold with its majority-class rate (competence). `--set
//!   battery_2609_33209` runs it on the original battery of the paper (comparison with Jev).
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
//!   flip rate, macro accuracy and macro ECE of the English zero-shot sets (wikitoxic × 4,
//!   biasframes × 3, amazonpolarity; yahootopics waits for its license).
//! - `pari2`: λ* (variant) against λ = 0 (reference): D competence at λ = 0 (lower bound of the
//!   D accuracy > majority rate + 0.10), MAD_neg of λ* < 0.03, partition of λ* in [0.97, 1.03],
//!   bench accuracy non-inferiority (0.02), monotonicity violations reduced by ≥ 30 % with an
//!   interval of the difference below 0; secondary (Holm): MAD_neg, |Σ s − 1|, |s − c|,
//!   additivity, description invariance, bench ECE.
//! - `pari3` (exploratory, amendment 4): online escalation with audit on the bench dump of the
//!   shipped model. Every escalated question and a share π of the accepted ones, drawn at random,
//!   go to the auditor (the gold); `ê_t = err_t · O_t / P(O_t)` and, per type,
//!   `λ_{t+1} = λ_t + η · 1{accepted} · (ê_t − α)`, from the plug-in threshold of the calibration
//!   dump (`--cal-src`). The stream replays the bench twice its size with replacement, the second
//!   half under a prevalence drift (hard questions and minority answers × 3 each), for
//!   α ∈ {5, 10} %, π ∈ {0.1, 0.2, 0.5}, η = 0.005 and 20 stream seeds. Success in a cell: mean
//!   estimated error of the accepted after the drift ≤ α + 0.01, with at most 1.2 times the
//!   escalations of the fixed threshold after the drift; the bet succeeds when every cell does.
//!   `--trajectory` writes 200 steps of stream seed 0 for the /live cascade demo.
//! - `retemper` (exploratory, amendment 5): temperatures fitted as by `calibrate` on the T = 1
//!   `val` dump of a run restricted to the sources starting with `--src-prefix`, applied to its f32
//!   bench dump by the factor `T′ = T_src / T_applied` (`temperatures.f32.json`); the re-tempered
//!   bench dump goes to `--out`, the same format, for `pari1` on the bench only.
//! - `summary`: one compact JSON for the results page: every report of `runs/s5-report/`, the
//!   frozen selections of `runs/s5-select/`, the validation summary of every s5 run, the log of
//!   the test passes, and the shipped model by the pre-registered rule (lowest val NLL among the
//!   three seeds of the retained arm: λ* when bet 2 succeeded, else B* when bet 1 succeeded,
//!   else A′), and the model of the /live demo (amendment 3: the long run `long-b-s0` when its
//!   val NLL is lower).

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
    /// The English zero-shot sets of the plan whose licenses are confirmed (HWU64 and SIB-200
    /// are reported apart; yahootopics waits for its license, DATA_LICENSES.md).
    pub const ZERO_SHOT: [&str; 8] = [
        "wikitoxic_insult",
        "wikitoxic_obscene",
        "wikitoxic_threat",
        "wikitoxic_toxicaggregated",
        "biasframes_intent",
        "biasframes_offensive",
        "biasframes_sex",
        "amazonpolarity",
    ];
    /// Sets left out of every report until their license is confirmed.
    pub const UNPUBLISHED: [&str; 1] = ["yahootopics"];
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
    /// A set that some runs lack is missing for the whole arm (every run comes back empty).
    fn runs(dirs: &[PathBuf], set: &str) -> Vec<Vec<Item>> {
        let v: Vec<Vec<Item>> = dirs.iter().map(|d| dump(d, set)).collect();
        if v.iter().any(Vec::is_empty) {
            return vec![Vec::new(); v.len()];
        }
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
        if n != aligned(variant) {
            // An arm without this set (q8_0 holds only the bench): nothing to compare.
            let nan = f64::NAN;
            return Paired {
                reference: nan,
                variant: nan,
                diff: Interval {
                    value: nan,
                    lo: nan,
                    hi: nan,
                },
                p_value: nan,
            };
        }
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

    // ------------------------------------------------------------------------- retemper

    /// Temperature of a question with `k` options: `type.bucket`, else `type`, else 1 (the
    /// fallback of the engine, `Model::temperature`).
    fn temp_of(temps: &BTreeMap<String, f64>, t: &str, k: usize) -> f64 {
        temps
            .get(&format!("{t}.{}", bucket(k)))
            .or_else(|| temps.get(t))
            .copied()
            .unwrap_or(1.0)
    }

    /// Probabilities produced at temperature `applied`, recomputed at `target`:
    /// `softmax(ln p / T′)` with `T′ = target / applied`, i.e. the logits divided by `target`.
    pub fn retemper_probs(probs: &[f64], applied: f64, target: f64) -> Vec<f64> {
        let tp = target / applied;
        let z: Vec<f64> = probs.iter().map(|p| p.max(1e-12).ln() / tp).collect();
        let m = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let e: Vec<f64> = z.iter().map(|x| (x - m).exp()).collect();
        let s: f64 = e.iter().sum();
        e.iter().map(|x| x / s).collect()
    }

    /// The items whose source starts with `prefix`.
    pub fn by_source(items: Vec<Item>, prefix: &str) -> Vec<Item> {
        items
            .into_iter()
            .filter(|it| it.src.starts_with(prefix))
            .collect()
    }

    /// Amendment 5 (exploratory): temperatures fitted on the T = 1 `val` dump of a run restricted
    /// to the sources starting with `prefix` (same fit as `calibrate`), applied to its f32 bench
    /// dump, produced at the temperatures of `temperatures.f32.json`. Writes the bench dump,
    /// re-tempered and otherwise unchanged, to `out`.
    pub fn retemper(run: &Path, prefix: &str, out: &Path, min_bucket: usize) -> Value {
        let cal = by_source(read(&run.join("val.f32.dump.jsonl")), prefix);
        assert!(!cal.is_empty(), "{run:?}: no val question from {prefix}");
        let src = temperatures(&cal, min_bucket);
        let tf = run.join("temperatures.f32.json");
        let applied: BTreeMap<String, f64> = serde_json::from_str(
            &std::fs::read_to_string(&tf).unwrap_or_else(|e| panic!("{tf:?}: {e}")),
        )
        .unwrap_or_else(|e| panic!("{tf:?}: {e}"));
        let bench = run.join(format!("eval-f32/{BENCH}.dump.jsonl"));
        let text = std::fs::read_to_string(&bench).unwrap_or_else(|e| panic!("{bench:?}: {e}"));
        let mut lines = Vec::new();
        for l in text.lines().filter(|l| !l.trim().is_empty()) {
            let mut v: Value = serde_json::from_str(l).unwrap_or_else(|e| panic!("{bench:?}: {e}"));
            let probs: Vec<f64> = v["probs"]
                .as_array()
                .map(|a| a.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default();
            let t = v["t"].as_str().unwrap_or_default();
            let k = probs.len();
            v["probs"] = json!(retemper_probs(
                &probs,
                temp_of(&applied, t, k),
                temp_of(&src, t, k)
            ));
            lines.push(v.to_string());
        }
        std::fs::create_dir_all(out).unwrap_or_else(|e| panic!("{out:?}: {e}"));
        let p = out.join(format!("{BENCH}.dump.jsonl"));
        std::fs::write(&p, lines.join("\n") + "\n").unwrap_or_else(|e| panic!("{p:?}: {e}"));
        eprintln!("wrote {p:?}");
        json!({"run": run, "src_prefix": prefix, "val_questions": cal.len(),
               "bench_questions": lines.len(), "applied": applied, "src": src})
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

    fn fam_runs(dirs: &[PathBuf], set: &str) -> Vec<Vec<FamStats>> {
        runs(dirs, set)
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

    pub fn coherence(dirs: &[PathBuf], set: &str) -> Value {
        let st = fam_runs(dirs, set);
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
            if UNPUBLISHED.contains(set) {
                continue;
            }
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

    // ------------------------------------------------------------------------- bet 3

    /// Bet 3 (exploratory, amendment 4): risk levels, audit rates of the accepted questions,
    /// stream seeds and step of the threshold update. Fixed before the first run.
    pub const P3_ALPHAS: [f64; 2] = [0.05, 0.10];
    pub const P3_PIS: [f64; 3] = [0.1, 0.2, 0.5];
    pub const P3_SEEDS: u64 = 20;
    pub const P3_ETA: f64 = 0.005;
    /// Success: estimated error of the accepted after the drift ≤ α + margin, with at most
    /// `P3_ESCALATIONS` times the escalations of the fixed threshold after the drift.
    pub const P3_MARGIN: f64 = 0.01;
    pub const P3_ESCALATIONS: f64 = 1.2;
    /// Questions of the stream sampled into the /live trajectory.
    const P3_TRACE: usize = 200;

    /// Starting (and fixed) threshold of a type: the lowest max-p threshold of the 200-point grid
    /// whose empirical error among the accepted calibration questions is at most α. A plug-in
    /// choice without guarantee (Learn-then-Test certifies nothing at these α); 1 when none.
    pub fn plugin_threshold(cal: &[&Item], alpha: f64) -> f64 {
        let mut best = 1.0;
        for i in 0..200 {
            let t = 1.0 - i as f64 / 200.0;
            let acc: Vec<&&Item> = cal.iter().filter(|it| it.probs[it.pred()] >= t).collect();
            let errs = acc.iter().filter(|it| !it.correct()).count();
            if !acc.is_empty() && errs as f64 <= alpha * acc.len() as f64 {
                best = t;
            }
        }
        best
    }

    /// Sampling weights of the bench after the drift (frozen scheme): × 3 for a hard question
    /// (max p below the median of its type), × 3 for a minority answer (a gold option whose count
    /// in its type is below the median count of the gold options of that type); they multiply.
    pub fn drift_weights(bench: &[Item]) -> Vec<f64> {
        let median = |mut v: Vec<f64>| -> f64 {
            v.sort_by(f64::total_cmp);
            quantile(&v, 0.5)
        };
        let mut conf: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        let mut golds: BTreeMap<&str, BTreeMap<usize, usize>> = BTreeMap::new();
        for it in bench {
            conf.entry(&it.t).or_default().push(it.probs[it.pred()]);
            *golds
                .entry(&it.t)
                .or_default()
                .entry(it.gold())
                .or_default() += 1;
        }
        let conf_med: BTreeMap<&str, f64> = conf.into_iter().map(|(t, v)| (t, median(v))).collect();
        let gold_med: BTreeMap<&str, f64> = golds
            .iter()
            .map(|(t, g)| (*t, median(g.values().map(|&c| c as f64).collect())))
            .collect();
        bench
            .iter()
            .map(|it| {
                let hard = it.probs[it.pred()] < conf_med[it.t.as_str()];
                let count = golds[it.t.as_str()][&it.gold()] as f64;
                let minority = count < gold_med[it.t.as_str()];
                (if hard { 3.0 } else { 1.0 }) * (if minority { 3.0 } else { 1.0 })
            })
            .collect()
    }

    /// A replayed stream: `2 n` bench questions drawn with replacement, uniformly for the first
    /// half and with [`drift_weights`] for the second (`true` after the drift).
    pub fn p3_stream(n: usize, weights: &[f64], seed: u64) -> Vec<(usize, bool)> {
        let mut rng = Rng::new(seed);
        let mut cum = Vec::with_capacity(weights.len());
        let mut total = 0.0;
        for w in weights {
            total += w;
            cum.push(total);
        }
        let mut out: Vec<(usize, bool)> = (0..n).map(|_| (rng.below(n), false)).collect();
        for _ in 0..n {
            let u = rng.f64() * total;
            out.push((cum.partition_point(|&c| c <= u).min(n - 1), true));
        }
        out
    }

    /// Inverse-probability estimate of an error: `err · O / P(O)`, with `P(O) = π` for an
    /// accepted question (audited at random) and 1 for an escalated one (always audited).
    pub fn ipw(wrong: bool, audited: bool, p_audit: f64) -> f64 {
        if wrong && audited {
            1.0 / p_audit
        } else {
            0.0
        }
    }

    /// `λ_{t+1} = λ_t + η · 1{accepted} · (ê_t − α)`, kept in [0, 1].
    pub fn p3_update(lambda: f64, accepted: bool, e_hat: f64, alpha: f64, eta: f64) -> f64 {
        if accepted {
            (lambda + eta * (e_hat - alpha)).clamp(0.0, 1.0)
        } else {
            lambda
        }
    }

    /// Counts of one half of a stream, for the online and the fixed thresholds.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct P3Half {
        pub n: usize,
        pub accepted: usize,
        pub wrong: usize,
        pub est: f64,
        pub audited: usize,
        pub fixed_accepted: usize,
        pub fixed_wrong: usize,
    }

    impl P3Half {
        fn rate(a: f64, b: usize) -> f64 {
            if b > 0 {
                a / b as f64
            } else {
                f64::NAN
            }
        }
        pub fn error_est(&self) -> f64 {
            Self::rate(self.est, self.accepted)
        }
        pub fn error_true(&self) -> f64 {
            Self::rate(self.wrong as f64, self.accepted)
        }
        pub fn escalations(&self) -> usize {
            self.n - self.accepted
        }
        pub fn fixed_escalations(&self) -> usize {
            self.n - self.fixed_accepted
        }
    }

    /// One step of a recorded trajectory.
    pub struct P3Step {
        pub lambda: f64,
        pub accepted: bool,
        pub audited: bool,
        /// Estimated and true error of the accepted so far in the current half.
        pub est: f64,
        pub wrong: f64,
    }

    /// Replays `stream` with one threshold per type, starting at `lambda0` (also the fixed
    /// threshold). Returns the two halves and the steps at `trace` positions.
    pub fn p3_simulate(
        bench: &[Item],
        stream: &[(usize, bool)],
        lambda0: &BTreeMap<String, f64>,
        (alpha, pi, eta): (f64, f64, f64),
        audit_seed: u64,
        trace: &[usize],
    ) -> ([P3Half; 2], Vec<P3Step>) {
        let mut rng = Rng::new(audit_seed);
        let mut lambda = lambda0.clone();
        let mut halves = [P3Half::default(); 2];
        let mut steps = Vec::with_capacity(trace.len());
        for (pos, &(i, drifted)) in stream.iter().enumerate() {
            let it = &bench[i];
            let p = it.probs[it.pred()];
            let wrong = !it.correct();
            let l0 = lambda0.get(&it.t).copied().unwrap_or(1.0);
            let l = lambda.entry(it.t.clone()).or_insert(l0);
            let before = *l;
            let accepted = p >= before;
            let audited = !accepted || rng.f64() < pi;
            let e_hat = ipw(wrong, audited, if accepted { pi } else { 1.0 });
            *l = p3_update(before, accepted, e_hat, alpha, eta);
            let h = &mut halves[usize::from(drifted)];
            h.n += 1;
            h.audited += usize::from(audited);
            if accepted {
                h.accepted += 1;
                h.wrong += usize::from(wrong);
                h.est += e_hat;
            }
            if p >= l0 {
                h.fixed_accepted += 1;
                h.fixed_wrong += usize::from(wrong);
            }
            if trace.binary_search(&pos).is_ok() {
                steps.push(P3Step {
                    lambda: before,
                    accepted,
                    audited,
                    est: h.error_est(),
                    wrong: h.error_true(),
                });
            }
        }
        (halves, steps)
    }

    /// Mean and 95 % range over the stream seeds (NaN draws left out).
    fn over_seeds(v: &[f64]) -> Value {
        let mut s: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        s.sort_by(f64::total_cmp);
        json!({"mean": mean(s.iter().copied()), "lo": quantile(&s, 0.025), "hi": quantile(&s, 0.975)})
    }

    /// Bet 3 on the bench dump of one model, thresholds started on its calibration dump.
    /// With `trajectory`, also the /live sample of stream seed 0 for every (α, π).
    pub fn pari3(cal: &[Item], bench: &[Item], with_trajectory: bool) -> (Value, Option<Value>) {
        let mut types: Vec<String> = bench.iter().map(|i| i.t.clone()).collect();
        types.sort();
        types.dedup();
        let weights = drift_weights(bench);
        let n = bench.len();
        let streams: Vec<Vec<(usize, bool)>> =
            (0..P3_SEEDS).map(|s| p3_stream(n, &weights, s)).collect();
        let trace: Vec<usize> = (0..P3_TRACE).map(|j| j * 2 * n / P3_TRACE).collect();
        let mut cells = Vec::new();
        let mut traj_cells = Vec::new();
        let mut lambda0_all = serde_json::Map::new();
        let mut success = true;
        for alpha in P3_ALPHAS {
            let lambda0: BTreeMap<String, f64> = types
                .iter()
                .map(|t| {
                    let c: Vec<&Item> = cal.iter().filter(|i| &i.t == t).collect();
                    (t.clone(), plugin_threshold(&c, alpha))
                })
                .collect();
            lambda0_all.insert(format!("{:.0}%", alpha * 100.0), json!(lambda0));
            for pi in P3_PIS {
                let mut per: Vec<[P3Half; 2]> = Vec::new();
                for (s, stream) in streams.iter().enumerate() {
                    let tr: &[usize] = if s == 0 { &trace } else { &[] };
                    let (h, steps) = p3_simulate(
                        bench,
                        stream,
                        &lambda0,
                        (alpha, pi, P3_ETA),
                        1000 + s as u64,
                        tr,
                    );
                    if s == 0 && with_trajectory {
                        traj_cells.push(json!({
                            "alpha": alpha, "pi": pi, "lambda0": lambda0,
                            "steps": steps.iter().map(|st| json!({
                                "lambda": (st.lambda * 1e4).round() / 1e4, "accepted": st.accepted,
                                "audited": st.audited, "est": st.est, "wrong": st.wrong,
                            })).collect::<Vec<_>>(),
                            "after": {"error_est": h[1].error_est(), "error_true": h[1].error_true(),
                                      "escalations": h[1].escalations(),
                                      "fixed_escalations": h[1].fixed_escalations()},
                        }));
                    }
                    per.push(h);
                }
                let col = |f: &dyn Fn(&P3Half) -> f64, half: usize| -> Vec<f64> {
                    per.iter().map(|h| f(&h[half])).collect()
                };
                let half = |k: usize| {
                    json!({
                        "coverage": over_seeds(&col(&|h| h.accepted as f64 / h.n as f64, k)),
                        "error_est": over_seeds(&col(&P3Half::error_est, k)),
                        "error_true": over_seeds(&col(&P3Half::error_true, k)),
                        "escalations": over_seeds(&col(&|h| h.escalations() as f64, k)),
                        "audit_rate": over_seeds(&col(&|h| h.audited as f64 / h.n as f64, k)),
                        "fixed": {
                            "coverage": over_seeds(&col(&|h| h.fixed_accepted as f64 / h.n as f64, k)),
                            "error_true": over_seeds(&col(&|h| P3Half::rate(h.fixed_wrong as f64, h.fixed_accepted), k)),
                            "escalations": over_seeds(&col(&|h| h.fixed_escalations() as f64, k)),
                        },
                    })
                };
                let est_after = mean(
                    col(&P3Half::error_est, 1)
                        .into_iter()
                        .filter(|x| x.is_finite()),
                );
                let esc = mean(col(&|h| h.escalations() as f64, 1));
                let esc_fixed = mean(col(&|h| h.fixed_escalations() as f64, 1));
                let ratio = esc / esc_fixed;
                let error_ok = est_after <= alpha + P3_MARGIN;
                let esc_ok = ratio <= P3_ESCALATIONS;
                success &= error_ok && esc_ok;
                cells.push(json!({
                    "alpha": alpha, "pi": pi,
                    "before": half(0), "after": half(1),
                    "error_est_after": est_after, "escalation_ratio_after": ratio,
                    "criteria": {"error_est_after_le_alpha_plus_margin": error_ok,
                                 "escalations_le_ratio": esc_ok},
                    "seeds_error_ok": col(&P3Half::error_est, 1).iter().filter(|&&e| e <= alpha + P3_MARGIN).count(),
                    "success": error_ok && esc_ok,
                }));
            }
        }
        let report = json!({
            "exploratory": true,
            "n_bench": n, "n_cal": cal.len(), "stream_len": 2 * n, "seeds": P3_SEEDS,
            "eta": P3_ETA, "margin": P3_MARGIN, "escalation_ratio_max": P3_ESCALATIONS,
            "lambda0": lambda0_all,
            "cells": cells,
            "success": success,
        });
        let traj = with_trajectory.then(|| {
            json!({
                "eta": P3_ETA, "stream_len": 2 * n, "drift_at": n,
                "items": trace.iter().map(|&pos| {
                    let (i, drifted) = streams[0][pos];
                    let it = &bench[i];
                    json!({"pos": pos, "t": it.t, "k": it.probs.len(),
                           "probs": it.probs.iter().map(|p| (p * 1e3).round() / 1e3).collect::<Vec<_>>(),
                           "gold": it.gold(), "pred": it.pred(), "drift": drifted})
                }).collect::<Vec<_>>(),
                "cells": traj_cells,
            })
        });
        (report, traj)
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

    /// The comparisons with their Holm-adjusted p-values; a comparison that could not be made
    /// (an arm without the set) is reported with nulls and left out of the family.
    fn holm_table(rows: Vec<(&str, Paired)>) -> Value {
        let done: Vec<usize> = (0..rows.len())
            .filter(|&i| rows[i].1.p_value.is_finite())
            .collect();
        let adj = holm(&done.iter().map(|&i| rows[i].1.p_value).collect::<Vec<_>>());
        Value::Object(
            rows.iter()
                .enumerate()
                .map(|(i, (k, p))| {
                    let mut v = pv(p);
                    v["p_holm"] = json!(done.iter().position(|&j| j == i).map(|r| adj[r]));
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
        let (rf, vf) = (fam_runs(rd, COHERENCE), fam_runs(vd, COHERENCE));
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

    /// The model of the /live demo: the shipped one, or the long run of amendment 3
    /// (`long-b-s0`) when its val NLL is lower.
    fn live(runs: &serde_json::Map<String, Value>, shipped: Option<&str>) -> Option<String> {
        let nll = |r: &str| runs.get(r).and_then(|v| v["all"]["nll"].as_f64());
        let ship = shipped?;
        match (nll(ship), nll("long-b-s0")) {
            (Some(a), Some(b)) if b < a => Some("long-b-s0".into()),
            _ => Some(ship.to_string()),
        }
    }

    /// The validation summary of every s5 run of the store, by run name.
    fn run_summaries(store: &Path) -> serde_json::Map<String, Value> {
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
        runs
    }

    pub fn summary(store: &Path) -> Value {
        let runs = run_summaries(store);
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
            // Amendment 3: /live serves the long run when its val NLL is lower (outside the bets).
            "live": live(&runs, ship.as_deref()),
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
                "usage: sky-s5-report calibrate|retemper|gates|coherence|robust|selective|pari1|pari2|pari3|summary ... (see the source)"
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
            "retemper" => {
                let src = arg(&args, "--src-prefix").unwrap_or_else(|| "typed_decisions".into());
                let min = num("--min-bucket", 100.0) as usize;
                let out = PathBuf::from(need("--out"));
                emit(retemper(Path::new(&need("--run")), &src, &out, min), &args)
            }
            "gates" => emit(
                gates(Path::new(&pos(1)), Path::new(&need("--records"))),
                &args,
            ),
            "coherence" => {
                let set = arg(&args, "--set").unwrap_or_else(|| COHERENCE.to_string());
                emit(coherence(&list(&pos(1)), &set), &args)
            }
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
            "pari3" => {
                let store = PathBuf::from(need("--store"));
                let format = arg(&args, "--format").unwrap_or_else(|| "q8_0".into());
                let src = arg(&args, "--cal-src").unwrap_or_else(|| "typed_decisions_train".into());
                let reports = json_dir(&store.join("runs/s5-report"));
                let selection = json_dir(&store.join("runs/s5-select"));
                let Some(ship) = shipped(&reports, &selection, &run_summaries(&store)) else {
                    eprintln!("sky-s5-report pari3: no shipped model yet (bet 1 not analysed)");
                    std::process::exit(1)
                };
                let rd = store.join(format!("runs/s5-{ship}"));
                let cal: Vec<Item> = read(&rd.join(format!("cal.{format}.dump.jsonl")))
                    .into_iter()
                    .filter(|it| it.src.starts_with(src.as_str()))
                    .collect();
                let bench = dump(&rd.join(format!("eval-{format}")), BENCH);
                let traj_path = arg(&args, "--trajectory");
                let (mut report, traj) = pari3(&cal, &bench, traj_path.is_some());
                let about = json!({"run": ship, "format": format, "cal_src": src});
                for (k, v) in about.as_object().into_iter().flatten() {
                    report[k] = v.clone();
                }
                if let (Some(p), Some(mut t)) = (traj_path, traj) {
                    for (k, v) in about.as_object().into_iter().flatten() {
                        t[k] = v.clone();
                    }
                    std::fs::write(&p, t.to_string() + "\n").unwrap_or_else(|e| panic!("{p}: {e}"));
                    eprintln!("wrote {p}");
                }
                emit(report, &args)
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
        fn ipw_error_estimate_is_unbiased() {
            let mut rng = Rng::new(7);
            let n = 200_000;
            for (e, pi) in [(0.05, 0.1), (0.05, 0.5), (0.2, 0.2), (0.3, 1.0)] {
                let sum: f64 = (0..n).map(|_| ipw(rng.f64() < e, rng.f64() < pi, pi)).sum();
                let sd = ((e / pi - e * e) / n as f64).sqrt();
                let m = sum / n as f64;
                assert!((m - e).abs() < 4.0 * sd, "e {e} pi {pi}: mean {m}");
            }
            // an escalated question is always audited: the estimate is the error itself
            assert_eq!(ipw(true, true, 1.0), 1.0);
            assert_eq!(ipw(false, true, 1.0), 0.0);
            assert_eq!(ipw(true, false, 0.2), 0.0);
        }

        #[test]
        fn retemper_composes_the_temperatures() {
            let softmax = |z: &[f64], t: f64| -> Vec<f64> {
                let e: Vec<f64> = z.iter().map(|x| (x / t).exp()).collect();
                let s: f64 = e.iter().sum();
                e.iter().map(|x| x / s).collect()
            };
            let z = [1.3, -0.4, 0.2, 2.1];
            for (applied, target) in [(0.7, 1.4), (1.0, 0.5), (1.375, 0.72), (0.9, 0.9)] {
                // p at T, then T′ = target / T: the logits divided by target
                let got = retemper_probs(&softmax(&z, applied), applied, target);
                for (a, b) in got.iter().zip(softmax(&z, target)) {
                    assert!((a - b).abs() < 1e-12, "T {applied} -> {target}: {a} vs {b}");
                }
                assert_eq!(
                    sky_metrics::argmax(&got),
                    sky_metrics::argmax(&softmax(&z, applied))
                );
            }
            // the fallback of the engine: type.bucket, else type, else 1
            let temps: BTreeMap<String, f64> =
                [("choice".to_string(), 0.7), ("choice.2".to_string(), 1.4)].into();
            assert_eq!(temp_of(&temps, "choice", 2), 1.4);
            assert_eq!(temp_of(&temps, "choice", 4), 0.7);
            assert_eq!(temp_of(&temps, "score", 4), 1.0);
        }

        #[test]
        fn retemper_keeps_only_the_source() {
            let mk = |src: &str| Item {
                src: src.into(),
                ..it("u", &[0.5, 0.5], 0)
            };
            let items = [
                "typed_decisions_train",
                "typed_decisions_train+para",
                "typed_decisions_train+none",
                "typed_synth",
                "coh_S",
                "arc",
            ]
            .map(mk)
            .to_vec();
            let kept: Vec<String> = by_source(items, "typed_decisions")
                .into_iter()
                .map(|it| it.src)
                .collect();
            assert_eq!(
                kept,
                [
                    "typed_decisions_train",
                    "typed_decisions_train+para",
                    "typed_decisions_train+none"
                ]
            );
        }

        #[test]
        fn threshold_update_rule() {
            let (a, eta) = (0.05, 0.01);
            // escalated: no update
            assert_eq!(p3_update(0.5, false, 10.0, a, eta), 0.5);
            // accepted, no audited error: down by η α
            assert!((p3_update(0.5, true, 0.0, a, eta) - (0.5 - eta * a)).abs() < 1e-15);
            // accepted, audited error at π = 0.1: up by η (10 − α)
            assert!((p3_update(0.5, true, 10.0, a, eta) - (0.5 + eta * (10.0 - a))).abs() < 1e-12);
            assert_eq!(p3_update(0.99, true, 10.0, a, eta), 1.0);
            assert_eq!(p3_update(0.0, true, 0.0, a, eta), 0.0);
        }

        /// Calibrated binary questions: max p uniform in [0.5, 1], right with probability max p.
        fn calibrated(n: usize, seed: u64) -> Vec<Item> {
            let mut rng = Rng::new(seed);
            (0..n)
                .map(|i| {
                    let p = 0.5 + 0.5 * rng.f64();
                    let gold = usize::from(rng.f64() >= p);
                    let mut x = it(&i.to_string(), &[p, 1.0 - p], gold);
                    x.t = "noul".into();
                    x
                })
                .collect()
        }

        #[test]
        fn online_threshold_tracks_the_error_level() {
            // Without drift the estimated error of the accepted settles near α from a threshold
            // far too low (0.5 accepts everything: error 25 %).
            let bench = calibrated(5000, 3);
            let stream = p3_stream(bench.len(), &vec![1.0; bench.len()], 0);
            let stream: Vec<(usize, bool)> =
                stream.into_iter().map(|(i, _)| (i, i % 2 == 1)).collect();
            let lambda0 = BTreeMap::from([("noul".to_string(), 0.5)]);
            for pi in [0.2, 0.5] {
                let (h, _) = p3_simulate(&bench, &stream, &lambda0, (0.1, pi, 0.005), 9, &[]);
                let all = P3Half {
                    n: h[0].n + h[1].n,
                    accepted: h[0].accepted + h[1].accepted,
                    wrong: h[0].wrong + h[1].wrong,
                    est: h[0].est + h[1].est,
                    ..Default::default()
                };
                assert!((all.error_true() - 0.1).abs() < 0.03, "pi {pi}: {all:?}");
                assert!(
                    (all.error_est() - all.error_true()).abs() < 0.03,
                    "pi {pi}: {all:?}"
                );
                // the fixed threshold 0.5 keeps the error of the whole bench
                let fixed = (h[0].fixed_wrong + h[1].fixed_wrong) as f64 / 10_000.0;
                assert!((fixed - 0.25).abs() < 0.03, "{fixed}");
            }
        }

        #[test]
        fn plugin_threshold_and_drift_weights() {
            let cal = calibrated(4000, 5);
            let refs: Vec<&Item> = cal.iter().collect();
            // error of the accepted above t is (1 − t) / 2 for this model: α = 0.1 at t ≈ 0.8
            let t = plugin_threshold(&refs, 0.1);
            assert!((t - 0.8).abs() < 0.04, "{t}");
            assert!(plugin_threshold(&refs, 0.0) > 0.95);
            // no threshold keeps the error at α: 1 (escalate everything)
            let wrong = [it("w", &[0.9, 0.1], 1)];
            assert_eq!(
                plugin_threshold(&wrong.iter().collect::<Vec<_>>(), 0.1),
                1.0
            );
            // hard (below the median max p) × 3, minority gold × 3
            let b = vec![
                it("a", &[0.9, 0.1], 0),
                it("b", &[0.8, 0.2], 0),
                it("c", &[0.6, 0.4], 0),
                it("d", &[0.3, 0.7], 1),
            ];
            assert_eq!(drift_weights(&b), vec![1.0, 1.0, 3.0, 9.0]);
            // the drifted half follows the weights
            let w = drift_weights(&b);
            let s = p3_stream(4, &w, 1);
            assert!(s[..4].iter().all(|x| !x.1) && s[4..].iter().all(|x| x.1));
            let big: Vec<f64> = (0..20_000).map(|i| w[i % 4]).collect();
            let s = p3_stream(20_000, &big, 2);
            let hard_minority = s[20_000..].iter().filter(|x| x.0 % 4 == 3).count();
            assert!((hard_minority as f64 / 20_000.0 - 9.0 / 14.0).abs() < 0.02);
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
