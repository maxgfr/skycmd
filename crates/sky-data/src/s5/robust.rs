//! Robustness and invariance test sets of s5 (all `split = test`, frozen by the
//! pre-registration). Each record's `uid` names its protocol and pairing:
//!
//! - `paraphrase_test.jsonl` — the **two-call test** with paraphrased definitions: every
//!   typed-decisions test question whose definitions all have a hand-written paraphrase
//!   (`eval/s5/paraphrases.json`), as `para/<base>/orig` and `para/<base>/para`;
//! - `jevout_lite.jsonl` — **JevOut-lite**: typed-decisions test questions, each with
//!   `distractor/<base>/<j>` variants whose state starts with one of 8 sentences of a corpus
//!   training never reads (`simplestories`); the attack keeps, at evaluation, the candidate that
//!   lowers the gold probability most;
//! - `reassign_test.jsonl` — the **reassignment** protocol of 2609.26758 on binary questions:
//!   two definitions `d_no`, `d_yes` under a name pair, `aligned` (n0 → d_no, n1 → d_yes) and
//!   `reassigned` (swapped), gold following the definition; `reassign/<base>/<pair>/<arm>`;
//! - `policy_test.jsonl` — **mini-PolicyBench**: tickets over 6 attributes under 4 policies that
//!   training never sees, the rule stated only in the definition of the true branch, plus an
//!   `xf` version where the false branch states the negation; `policy/<n>/<variant>`;
//! - `coverage_test.jsonl` — **candidate coverage** (2610.03387) on HWU64 and yahootopics:
//!   `full` menus of 8 with the gold, `omit` menus where the gold is replaced by another label
//!   (same size), each with a task-aligned NONE option; and BM25 top-k menus (k = 2, 5, 10) with
//!   no gold repair; the first 30 % of each kind are the calibration part (`cov/<set>/cal/...`)
//!   for the threshold of the D/F metric at F ≤ 5 %.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

use crate::rng::Rng;
use crate::s5::{uid_of, S5Record};

fn test_rec(base: &S5Record, uid: String) -> S5Record {
    let mut r = base.clone();
    r.uid = uid;
    r.split = "test".into();
    r
}

/// The two-call paraphrase test.
pub fn paraphrases(typed: &[S5Record], table: &Value) -> Vec<S5Record> {
    let map = &table["paraphrases"];
    let mut out = Vec::new();
    for r in typed {
        let paras: Option<Vec<String>> = r
            .opts_kv
            .iter()
            .map(|(_, d)| map[d.as_str()].as_str().map(str::to_string))
            .collect();
        let Some(paras) = paras else { continue };
        let mut o = test_rec(r, format!("para/{}/orig", r.uid));
        o.src = format!("{}+orig", r.src);
        let mut p = test_rec(r, format!("para/{}/para", r.uid));
        p.src = format!("{}+para", r.src);
        for ((_, d), np) in p.opts_kv.iter_mut().zip(paras) {
            *d = np;
        }
        out.push(o);
        out.push(p);
    }
    out
}

/// JevOut-lite candidates: `n` questions, 8 distractor variants each.
pub fn jevout(typed: &[S5Record], distractors: &[String], n: usize, seed: u64) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, "s5/eval/jevout");
    let mut idx: Vec<usize> = (0..typed.len()).collect();
    rng.shuffle(&mut idx);
    idx.truncate(n);
    idx.sort_unstable();
    let mut out = Vec::new();
    if distractors.is_empty() {
        return out;
    }
    for i in idx {
        let r = &typed[i];
        out.push(test_rec(r, format!("distractor/{}/base", r.uid)));
        for j in 0..8 {
            let s = &distractors[rng.below(distractors.len())];
            let mut d = test_rec(r, format!("distractor/{}/{j}", r.uid));
            d.state = format!("{s} {}", r.state);
            d.src = format!("{}+distractor", r.src);
            out.push(d);
        }
    }
    out
}

/// Name pairs of the reassignment protocol: (kind, n0, n1).
pub const NAME_PAIRS: &[(&str, &str, &str)] = &[
    ("control", "0", "1"),
    ("control", "A", "B"),
    ("polar_seen", "no", "yes"),
    ("polar_seen", "false", "true"),
    ("polar_unseen", "absent", "present"),
    ("polar_unseen", "negative", "positive"),
    ("polar_unseen", "rejected", "accepted"),
    ("random", "xg6a6", "e97ce"),
    ("random", "q81zt", "m3kpw"),
];

/// The reassignment arms of binary questions (`noul` turned into 2-option choices, and the
/// 2-option choices): the two definitions under every name pair, aligned and reassigned.
pub fn reassign(base: &[S5Record], n: usize, seed: u64) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, "s5/eval/reassign");
    let mut cand: Vec<&S5Record> = base
        .iter()
        .filter(|r| {
            r.opts_kv.len() == 2 && r.opts_kv.iter().all(|(_, d)| !d.is_empty()) && {
                let m = r.target.iter().cloned().fold(0.0, f32::max);
                m >= 0.7
            }
        })
        .collect();
    rng.shuffle(&mut cand);
    cand.truncate(n);
    let mut out = Vec::new();
    for r in cand {
        let (d_no, d_yes) = (&r.opts_kv[0].1, &r.opts_kv[1].1);
        let gold_yes = r.target[1] > r.target[0];
        for (pi, (kind, n0, n1)) in NAME_PAIRS.iter().enumerate() {
            for arm in ["aligned", "reassigned"] {
                let (a, b) = if arm == "aligned" {
                    (d_no, d_yes)
                } else {
                    (d_yes, d_no)
                };
                // Gold follows the definition.
                let yes_at = if arm == "aligned" { 1 } else { 0 };
                let gold = if gold_yes { yes_at } else { 1 - yes_at };
                let mut t = vec![0.0; 2];
                t[gold] = 1.0;
                let mut x = test_rec(r, format!("reassign/{}/{pi}-{kind}/{arm}", r.uid));
                x.t = "choice".into();
                x.opts_kv = vec![(n0.to_string(), a.clone()), (n1.to_string(), b.clone())];
                x.target = t;
                x.src = format!("reassign_{kind}");
                out.push(x);
            }
        }
    }
    out
}

/// One held-out policy of mini-PolicyBench: the rule (en), a case generator over 6 attributes
/// returning `(attributes, rule holds)`.
struct HeldOut {
    name: &'static str,
    rule: &'static str,
    negation: &'static str,
    case: CaseFn,
}

/// A policy case generator: the state fields and whether the case violates the policy.
type CaseFn = fn(&mut Rng, bool) -> (Vec<(&'static str, String)>, bool);

fn warranty(rng: &mut Rng, want: bool) -> (Vec<(&'static str, String)>, bool) {
    loop {
        let months = 1 + rng.below(36);
        let damage = ["none", "water", "drop", "wear"][rng.below(4)];
        let receipt = rng.f64() < 0.5;
        let ok = months <= 24 && damage != "water" && damage != "drop" && receipt;
        if ok == want {
            return (
                vec![
                    (
                        "product",
                        ["phone", "laptop", "headset", "watch"][rng.below(4)].into(),
                    ),
                    ("months since purchase", months.to_string()),
                    ("damage", damage.into()),
                    (
                        "receipt provided",
                        if receipt { "yes" } else { "no" }.into(),
                    ),
                    ("store", ["online", "retail"][rng.below(2)].into()),
                    ("customer since", (2010 + rng.below(15)).to_string()),
                ],
                ok,
            );
        }
    }
}

fn travel(rng: &mut Rng, want: bool) -> (Vec<(&'static str, String)>, bool) {
    loop {
        let cost = 100 + rng.below(1900);
        let class = ["economy", "business"][rng.below(2)];
        let days = rng.below(30);
        let ok = cost <= 1200 && class == "economy" && days >= 7;
        if ok == want {
            return (
                vec![
                    (
                        "destination",
                        ["Berlin", "Madrid", "Lisbon", "Oslo"][rng.below(4)].into(),
                    ),
                    ("ticket cost", format!("{cost} EUR")),
                    ("class", class.into()),
                    ("days booked in advance", days.to_string()),
                    ("traveller level", ["junior", "senior"][rng.below(2)].into()),
                    (
                        "purpose",
                        ["client meeting", "conference"][rng.below(2)].into(),
                    ),
                ],
                ok,
            );
        }
    }
}

fn overtime(rng: &mut Rng, want: bool) -> (Vec<(&'static str, String)>, bool) {
    loop {
        let hours = rng.below(30);
        let approved = rng.f64() < 0.5;
        let weekend = rng.f64() < 0.5;
        let ok = approved && (hours <= 10 || !weekend);
        if ok == want {
            return (
                vec![
                    (
                        "employee team",
                        ["support", "ops", "sales"][rng.below(3)].into(),
                    ),
                    ("overtime hours", hours.to_string()),
                    (
                        "manager approved",
                        if approved { "yes" } else { "no" }.into(),
                    ),
                    (
                        "includes weekend",
                        if weekend { "yes" } else { "no" }.into(),
                    ),
                    ("contract", ["full-time", "part-time"][rng.below(2)].into()),
                    ("month", ["March", "June", "October"][rng.below(3)].into()),
                ],
                ok,
            );
        }
    }
}

fn discount(rng: &mut Rng, want: bool) -> (Vec<(&'static str, String)>, bool) {
    loop {
        let pct = 5 * (1 + rng.below(8));
        let tier = ["bronze", "silver", "gold"][rng.below(3)];
        let order = 50 + rng.below(950);
        let ok = pct <= 15 || (tier == "gold" && order >= 500);
        if ok == want {
            return (
                vec![
                    ("requested discount", format!("{pct} %")),
                    ("loyalty tier", tier.into()),
                    ("order value", format!("{order} EUR")),
                    ("channel", ["phone", "chat", "email"][rng.below(3)].into()),
                    ("region", ["north", "south"][rng.below(2)].into()),
                    (
                        "first order",
                        if rng.f64() < 0.5 { "yes" } else { "no" }.into(),
                    ),
                ],
                ok,
            );
        }
    }
}

const HELD_OUT: &[HeldOut] = &[
    HeldOut {
        name: "warranty",
        rule: "The repair is covered: the product is under 24 months old, has no water or drop damage, and the receipt is provided.",
        negation: "The repair is not covered: the product is older than 24 months, has water or drop damage, or no receipt is provided.",
        case: warranty,
    },
    HeldOut {
        name: "travel",
        rule: "The booking is within policy: an economy ticket of at most 1200 EUR booked at least 7 days ahead.",
        negation: "The booking is outside policy: it costs more than 1200 EUR, is not economy, or was booked less than 7 days ahead.",
        case: travel,
    },
    HeldOut {
        name: "overtime",
        rule: "The overtime is payable: it was approved, and it is at most 10 hours whenever it includes a weekend.",
        negation: "The overtime is not payable: it was not approved, or it exceeds 10 hours and includes a weekend.",
        case: overtime,
    },
    HeldOut {
        name: "discount",
        rule: "The discount is allowed: at most 15 %, or any discount for a gold member on an order of at least 500 EUR.",
        negation: "The discount is not allowed: above 15 % without being a gold member with an order of at least 500 EUR.",
        case: discount,
    },
];

/// mini-PolicyBench: `n` balanced tickets, each in the `plain` and the `xf` version.
pub fn policy(n: usize, seed: u64) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, "s5/eval/policybench");
    let mut out = Vec::new();
    for i in 0..n {
        let p = &HELD_OUT[i % HELD_OUT.len()];
        let want = i % 2 == 0;
        let (attrs, ok) = (p.case)(&mut rng, want);
        let state = attrs
            .iter()
            .map(|(k, v)| format!("{k}: {v}"))
            .collect::<Vec<_>>()
            .join("; ");
        for (variant, false_def) in [("plain", String::new()), ("xf", p.negation.to_string())] {
            out.push(S5Record {
                src: format!("policybench_{variant}"),
                t: "noul".into(),
                ins: format!("Does this {} ticket satisfy the policy?", p.name),
                opts_kv: vec![("false".into(), false_def), ("true".into(), p.rule.into())],
                state: state.clone(),
                target: if ok { vec![0.0, 1.0] } else { vec![1.0, 0.0] },
                split: "test".into(),
                uid: format!("policy/{i}/{variant}"),
                lang: String::new(),
                gold: true,
            });
        }
    }
    out
}

/// BM25 scores of `docs` for `query` (k1 = 1.2, b = 0.75), tokens = lowercase alphanumeric words.
pub fn bm25(query: &str, docs: &[String]) -> Vec<f64> {
    let tok = |s: &str| -> Vec<String> {
        s.split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 1)
            .map(str::to_lowercase)
            .collect()
    };
    let dt: Vec<Vec<String>> = docs.iter().map(|d| tok(d)).collect();
    let n = dt.len() as f64;
    let avg = dt.iter().map(Vec::len).sum::<usize>() as f64 / n.max(1.0);
    let mut df: HashMap<&str, f64> = HashMap::new();
    for d in &dt {
        let mut seen: Vec<&str> = d.iter().map(String::as_str).collect();
        seen.sort_unstable();
        seen.dedup();
        for w in seen {
            *df.entry(w).or_insert(0.0) += 1.0;
        }
    }
    let q = tok(query);
    dt.iter()
        .map(|d| {
            let len = d.len() as f64;
            q.iter()
                .map(|w| {
                    let f = d.iter().filter(|x| *x == w).count() as f64;
                    if f == 0.0 {
                        return 0.0;
                    }
                    let idf = ((n - df[w.as_str()] + 0.5) / (df[w.as_str()] + 0.5) + 1.0).ln();
                    idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len / avg.max(1e-9)))
                })
                .sum()
        })
        .collect()
}

const NONE: (&str, &str) = (
    "none_of_these",
    "None of the listed options matches the request.",
);

/// Candidate coverage menus of a full-label eval set (`classes` = every label definition).
pub fn coverage(
    set: &str,
    recs: &[S5Record],
    classes: &BTreeMap<String, String>,
    seed: u64,
) -> Vec<S5Record> {
    let mut rng = Rng::derived(seed, &format!("s5/eval/coverage/{set}"));
    let labels: Vec<(&String, &String)> = classes.iter().collect();
    let docs: Vec<String> = labels
        .iter()
        .map(|(k, d)| format!("{} {d}", k.replace('_', " ")))
        .collect();
    let mut out = Vec::new();
    let n_cal = recs.len() * 3 / 10;
    for (i, r) in recs.iter().enumerate() {
        let part = if i < n_cal { "cal" } else { "test" };
        let g = r.gold_index();
        let gold_id = &r.opts_kv[g].0;
        let with_none = |mut kv: Vec<(String, String)>, gold_in: Option<usize>| {
            kv.push((NONE.0.into(), NONE.1.into()));
            let mut t = vec![0.0f32; kv.len()];
            t[gold_in.unwrap_or(kv.len() - 1)] = 1.0;
            (kv, t)
        };
        let mk = |kind: &str, kv: Vec<(String, String)>, t: Vec<f32>| {
            let mut x = test_rec(r, format!("cov/{set}/{part}/{kind}/{}", r.uid));
            x.opts_kv = kv;
            x.target = t;
            x.src = format!("coverage_{set}_{}", kind.split('-').next().unwrap_or(kind));
            x
        };
        // Full menu (gold present) and artificial omission (gold replaced, same size).
        let (kv, t) = with_none(r.opts_kv.clone(), Some(g));
        out.push(mk("full", kv, t));
        let mut kv = r.opts_kv.clone();
        let others: Vec<&(&String, &String)> = labels
            .iter()
            .filter(|(k, _)| !kv.iter().any(|(id, _)| id == *k))
            .collect();
        if !others.is_empty() {
            let (k, d) = others[rng.below(others.len())];
            kv[g] = ((*k).clone(), (*d).clone());
            let (kv, t) = with_none(kv, None);
            out.push(mk("omit", kv, t));
        }
        // BM25 menus over every label, no gold repair.
        let scores = bm25(&r.state, &docs);
        let mut order: Vec<usize> = (0..labels.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        for k in [2usize, 5, 10] {
            let menu: Vec<(String, String)> = order
                .iter()
                .take(k)
                .map(|&j| (labels[j].0.clone(), labels[j].1.clone()))
                .collect();
            let gold_in = menu.iter().position(|(id, _)| id == gold_id);
            let (kv, t) = with_none(menu, gold_in);
            out.push(mk(&format!("bm25-{k}"), kv, t));
        }
    }
    let _ = uid_of(&[set]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(kv: &[(&str, &str)], gold: usize) -> S5Record {
        S5Record {
            src: "x".into(),
            t: "choice".into(),
            ins: "q".into(),
            opts_kv: kv
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            state: "please turn the lights off now".into(),
            target: (0..kv.len())
                .map(|i| if i == gold { 1.0 } else { 0.0 })
                .collect(),
            split: "test".into(),
            uid: "u".into(),
            lang: String::new(),
            gold: true,
        }
    }

    #[test]
    fn reassignment_arms_follow_definitions() {
        let r = rec(&[("false", "it is benign"), ("true", "it is malicious")], 1);
        let v = reassign(&[r], 1, 0);
        assert_eq!(v.len(), NAME_PAIRS.len() * 2);
        let al = &v[0];
        let re = &v[1];
        assert_eq!(al.opts_kv[1].1, "it is malicious");
        assert_eq!(al.target, vec![0.0, 1.0]);
        assert_eq!(re.opts_kv[0].1, "it is malicious");
        assert_eq!(re.target, vec![1.0, 0.0]);
        assert_eq!(al.opts_kv[0].0, re.opts_kv[0].0);
    }

    #[test]
    fn policies_are_balanced_and_held_out() {
        let p = policy(200, 1);
        assert_eq!(p.len(), 400);
        let yes = p.iter().filter(|r| r.target[1] == 1.0).count();
        assert_eq!(yes, 200);
        assert!(p.iter().all(|r| r.validate().is_ok()));
        let xf = p.iter().find(|r| r.src == "policybench_xf").unwrap();
        assert!(xf.opts_kv[0].1.contains("not"));
    }

    #[test]
    fn coverage_menus_and_bm25() {
        let classes: BTreeMap<String, String> = [
            ("iot_hue_lightoff", "Wants the lights switched off."),
            ("iot_hue_lighton", "Wants the lights switched on."),
            ("weather_query", "Asks about the weather."),
            ("play_music", "Wants music played."),
        ]
        .iter()
        .map(|(a, b)| (a.to_string(), b.to_string()))
        .collect();
        let s = bm25(
            "turn the lights off",
            &classes
                .iter()
                .map(|(k, d)| format!("{} {d}", k.replace('_', " ")))
                .collect::<Vec<_>>(),
        );
        assert!(s[0] > s[2] && s[0] > s[3]);
        let r = rec(
            &[
                ("iot_hue_lightoff", "Wants the lights switched off."),
                ("weather_query", "Asks about the weather."),
            ],
            0,
        );
        let v = coverage("hwu64", &[r.clone(), r.clone(), r.clone(), r], &classes, 3);
        let omit = v.iter().find(|x| x.uid.contains("/omit/")).unwrap();
        assert!(omit.opts_kv.iter().all(|(id, _)| id != "iot_hue_lightoff"));
        assert_eq!(omit.target.last(), Some(&1.0));
        let b2 = v.iter().find(|x| x.uid.contains("/bm25-2/")).unwrap();
        assert_eq!(b2.opts_kv.len(), 3);
        assert!(v.iter().all(|x| x.validate().is_ok()));
    }
}
