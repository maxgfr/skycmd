//! Coherence families (2609.33209) built from the `choice` questions of typed-decisions.
//!
//! For a question `q` with options `x_1..x_K` and target `p`:
//!
//! - `CH`: the choice question itself;
//! - `S_k`: the noul "the correct answer to q is: x_k", target `p_k`;
//! - `N_k`: the **reformulated** negation "the correct answer to q is not: x_k", target
//!   `1 - p_k` — a new question with the generic false / true definitions, never the same
//!   question with its two definitions swapped (that negation holds by construction in iso);
//! - `D_ij` (test only, never trained on): "the correct answer to q is either x_i or x_j",
//!   target `p_i + p_j`. Monotonicity (`s_i <= d_ij`), description invariance and additivity
//!   are measured on it as identities the training never saw.
//!
//! Train families come from the train split of typed-decisions only, test families from its
//! test split. The `uid` encodes the family: `coh/<group>/<kind>/<k>` (`kind` in `CH`, `S`,
//! `N`, `D`; `k` is `i` or `i-j`).

use crate::s5::{uid_of, S5Record};

/// The family of a uid: `(group, kind, index)`.
pub fn parse_uid(uid: &str) -> Option<(&str, &str, &str)> {
    let rest = uid.strip_prefix("coh/")?;
    let mut it = rest.splitn(3, '/');
    Some((it.next()?, it.next()?, it.next()?))
}

fn noul(base: &S5Record, ins: String, p_true: f32, uid: String, src: &str) -> S5Record {
    let p = p_true.clamp(0.0, 1.0);
    S5Record {
        src: src.into(),
        t: "noul".into(),
        ins,
        opts_kv: vec![
            ("false".into(), String::new()),
            ("true".into(), String::new()),
        ],
        state: base.state.clone(),
        target: vec![1.0 - p, p],
        split: base.split.clone(),
        uid,
        lang: base.lang.clone(),
        gold: base.gold,
    }
}

fn label(def: &str, id: &str) -> String {
    if def.is_empty() {
        id.replace('_', " ")
    } else {
        def.trim().trim_end_matches('.').to_string()
    }
}

/// The families of one choice question (`with_disjunctions` for test states).
pub fn family(base: &S5Record, with_disjunctions: bool) -> Vec<S5Record> {
    if base.t != "choice" || base.opts_kv.len() < 2 || base.opts_kv.len() > 8 {
        return Vec::new();
    }
    let s: f32 = base.target.iter().sum();
    let p: Vec<f32> = base.target.iter().map(|x| x / s).collect();
    let g = uid_of(&["coh", &base.uid, &base.ins, &base.state]);
    let fr = base.lang == "fr";
    let q = base.ins.trim();
    let src = |kind: &str| format!("coh_{kind}");
    let mut out = Vec::new();
    let mut ch = base.clone();
    ch.uid = format!("coh/{g}/CH/0");
    ch.src = src("CH");
    out.push(ch);
    for (k, (id, def)) in base.opts_kv.iter().enumerate() {
        let x = label(def, id);
        let (si, ni) = if fr {
            (
                format!("La bonne réponse à « {q} » est : {x}."),
                format!("La bonne réponse à « {q} » n'est pas : {x}."),
            )
        } else {
            (
                format!("The correct answer to \"{q}\" is: {x}."),
                format!("The correct answer to \"{q}\" is not: {x}."),
            )
        };
        out.push(noul(base, si, p[k], format!("coh/{g}/S/{k}"), &src("S")));
        out.push(noul(
            base,
            ni,
            1.0 - p[k],
            format!("coh/{g}/N/{k}"),
            &src("N"),
        ));
    }
    if with_disjunctions {
        for i in 0..base.opts_kv.len() {
            for j in i + 1..base.opts_kv.len() {
                let (a, b) = (
                    label(&base.opts_kv[i].1, &base.opts_kv[i].0),
                    label(&base.opts_kv[j].1, &base.opts_kv[j].0),
                );
                let ins = if fr {
                    format!("La bonne réponse à « {q} » est soit : {a}, soit : {b}.")
                } else {
                    format!("The correct answer to \"{q}\" is either: {a}, or: {b}.")
                };
                out.push(noul(
                    base,
                    ins,
                    p[i] + p[j],
                    format!("coh/{g}/D/{i}-{j}"),
                    &src("D"),
                ));
            }
        }
    }
    out
}

/// The families of every choice question of `base`.
pub fn families(base: &[S5Record], with_disjunctions: bool) -> Vec<S5Record> {
    base.iter()
        .flat_map(|r| family(r, with_disjunctions))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn families_and_targets() {
        let base = S5Record {
            src: "typed_decisions_train".into(),
            t: "choice".into(),
            ins: "Which team should handle it?".into(),
            opts_kv: vec![
                ("billing".into(), "A charge problem.".into()),
                ("tech".into(), "Something is broken.".into()),
                ("sales".into(), String::new()),
            ],
            state: "s".into(),
            target: vec![0.6, 0.3, 0.1],
            split: "train".into(),
            uid: "u".into(),
            lang: String::new(),
            gold: true,
        };
        let f = family(&base, false);
        assert_eq!(f.len(), 1 + 2 * 3);
        let s1 = f.iter().find(|r| r.uid.ends_with("/S/1")).unwrap();
        assert_eq!(
            s1.ins,
            "The correct answer to \"Which team should handle it?\" is: Something is broken."
        );
        assert!((s1.target[1] - 0.3).abs() < 1e-6);
        let n1 = f.iter().find(|r| r.uid.ends_with("/N/1")).unwrap();
        assert!((n1.target[1] - 0.7).abs() < 1e-6);
        assert!(f.iter().all(|r| r.validate().is_ok()));
        let t = family(&base, true);
        let d = t.iter().find(|r| r.uid.ends_with("/D/0-1")).unwrap();
        assert!((d.target[1] - 0.9).abs() < 1e-6);
        assert!(d
            .ins
            .contains("either: A charge problem, or: Something is broken"));
        let (g, kind, k) = parse_uid(&d.uid).unwrap();
        assert_eq!((kind, k), ("D", "0-1"));
        assert_eq!(g, parse_uid(&s1.uid).unwrap().0);
        assert!(t.iter().filter(|r| r.uid.contains("/D/")).count() == 3);
    }
}
