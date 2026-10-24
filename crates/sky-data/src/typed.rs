//! Typed-decision corpora on the Hub (Jev / System One style), mapped to SPEC records:
//! tasksource JEV collection, procedural and synthetic typed decisions, Open-Jev,
//! swarm-laya, helmo, calibrated typed decisions and the NOTAM benchmark.

use std::collections::BTreeMap;

use anyhow::{bail, Result};
use serde_json::Value;

use crate::decisions::{cap_by_key, jev_items, parse_json_field, str_field, Ctx, Item, SMALL};
use crate::hf::{for_each_parquet_row, Hf};
use crate::record::{normalize, one_hot, Question};
use crate::rng::{hash64, Rng};

/// States longer than this are clipped: the model never reads past ~1000 characters
/// (256 tokens), and clipping keeps the JSONL files small.
pub const STATE_MAX_CHARS: usize = 6000;

/// The first `max` characters of `s` (trimmed), on a char boundary.
pub fn clip(s: &str, max: usize) -> String {
    let s = s.trim();
    match s.char_indices().nth(max) {
        Some((i, _)) => s[..i].trim_end().to_string(),
        None => s.to_string(),
    }
}

/// Stable group id of a state text.
pub fn state_group(state: &str) -> String {
    format!("{:x}", hash64(state.as_bytes()))
}

/// Uniform `[0, 1)` value of a key, for deterministic hash sampling.
pub fn hash_frac(seed: u64, key: &str) -> f64 {
    (hash64(format!("{seed}/{key}").as_bytes()) % 1_000_000) as f64 / 1e6
}

fn f64_list(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(Value::as_f64).collect()
}

fn str_list(v: &Value) -> Option<Vec<String>> {
    v.as_array()?
        .iter()
        .map(|x| x.as_str().map(str::to_string))
        .collect()
}

/// Maps a flat typed decision (`kind`, option texts, target over them) to a question and a
/// normalized target. Noul accepts a single `p(true)` or a 2-vector over `no/yes` (or
/// `false/true`) in either order. Choice options must be distinct.
pub fn flat_question(
    kind: &str,
    question: &str,
    options: &[String],
    target: &[f64],
) -> Option<(Question, Vec<f32>)> {
    let q = question.trim();
    if q.is_empty() || target.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return None;
    }
    match kind {
        "noul" => {
            let p_true = match (options.len(), target.len()) {
                (_, 1) => target[0],
                (2, 2) => {
                    let lower: Vec<String> = options
                        .iter()
                        .map(|o| o.trim().to_ascii_lowercase())
                        .collect();
                    let s = target[0] + target[1];
                    if s <= 0.0 {
                        return None;
                    }
                    match (lower[0].as_str(), lower[1].as_str()) {
                        ("no", "yes") | ("false", "true") => target[1] / s,
                        ("yes", "no") | ("true", "false") => target[0] / s,
                        _ => return None,
                    }
                }
                _ => return None,
            };
            (0.0..=1.0)
                .contains(&p_true)
                .then(|| (Question::noul(q), normalize(&[1.0 - p_true, p_true])))
        }
        "choice" | "score" => {
            if options.len() < 2
                || options.len() != target.len()
                || target.iter().sum::<f64>() <= 0.0
            {
                return None;
            }
            let opts: Vec<String> = options.iter().map(|o| o.trim().to_string()).collect();
            if opts.iter().any(String::is_empty) {
                return None;
            }
            let question = if kind == "score" {
                Question::score(q, opts)
            } else {
                let mut uniq = opts.clone();
                uniq.sort();
                uniq.dedup();
                if uniq.len() != opts.len() {
                    return None;
                }
                Question::choice(q, opts.into_iter().map(|o| (o, String::new())).collect())
            };
            Some((question, normalize(target)))
        }
        _ => None,
    }
}

// ----------------------------------------------------------------------------- tasksource JEV

/// Sources of `tasksource/tasksource-jev-typed-decisions` that are not used even when tagged
/// commercial: they duplicate a dataset we already load directly (BoolQ, MNLI, ARC,
/// CommonsenseQA (+ CoS-E), Banking77, PAWS-X fr, HellaSwag, SNLI, HelpSteer2, the procedural
/// typed decisions), or their upstream license is non-commercial / rejected (XNLI, SciQ, RACE,
/// LogiQA, PIQA).
pub fn jev_source_allowed(source: &str) -> bool {
    const EXACT: &[&str] = &[
        "super_glue/boolq",
        "super_glue/boolq_passage",
        "boolq-natural-perturbations",
        "glue/mnli",
        "commonsense_qa",
        "cos_e/v1.0",
        "banking77",
        "multilingual/paws-x/fr",
        "hellaswag",
        "snli",
        "multilingual/xnli",
        "sciq",
        "race/middle",
        "race/high",
        "race-c",
        "logiqa",
        "logiqa-2.0-nli",
        "piqa",
        "xnli",
        "logiqa2",
        "race",
    ];
    const PREFIX: &[&str] = &["ai2_arc/", "HelpSteer2/", "procedural-typed-decisions/"];
    !EXACT.contains(&source) && !PREFIX.iter().any(|p| source.starts_with(p))
}

fn jev_row_item(row: &Value, src_prefix: &str, state_key: &str) -> Option<Item> {
    let kind = row.get("kind")?.as_str()?;
    let question = row.get("question")?.as_str()?;
    let options = str_list(row.get("options")?)?;
    let target = f64_list(row.get("target")?)?;
    let source = row.get("source")?.as_str()?;
    let state = match row.get(state_key)? {
        Value::String(s) if state_key == "state_json" => match serde_json::from_str::<Value>(s) {
            Ok(Value::String(inner)) => inner,
            Ok(other) => other.to_string(),
            Err(_) => s.clone(),
        },
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    };
    let (q, target) = flat_question(kind, question, &options, &target)?;
    let group = row
        .get("group_id")
        .and_then(Value::as_str)
        .map_or_else(|| state_group(&state), str::to_string);
    Some(Item {
        group,
        rec: q.record(
            &format!("{src_prefix}:{source}"),
            clip(&state, STATE_MAX_CHARS),
            target,
        ),
    })
}

pub fn load_tasksource_jev(cx: &Ctx) -> Result<Vec<Item>> {
    let repo = "tasksource/tasksource-jev-typed-decisions";
    let shards = cx.hf.parquet_split(repo, "default", "train", SMALL)?;
    // Pass 1: eligible rows per source, to give every source a fair share of the cap.
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let (mut total, mut commercial) = (0usize, 0usize);
    for p in &shards {
        for_each_parquet_row(p, &["source", "license_use"], |row| {
            total += 1;
            if row.get("license_use").and_then(Value::as_str) == Some("commercial") {
                commercial += 1;
                let s = row.get("source").and_then(Value::as_str).unwrap_or("");
                if jev_source_allowed(s) {
                    *counts.entry(s.to_string()).or_default() += 1;
                }
            }
            Ok(())
        })?;
    }
    let eligible: usize = counts.values().sum();
    eprintln!(
        "  tasksource_jev: {total} rows, {commercial} commercial, {eligible} eligible in {} sources",
        counts.len()
    );
    let quotas = crate::decisions::fair_quotas(&counts, cx.cap);
    // Oversample a little: some rows are invalid or duplicates.
    let keep: BTreeMap<&String, f64> = counts
        .iter()
        .map(|(k, &n)| (k, (1.15 * quotas[k] as f64 / n as f64).min(1.0)))
        .collect();
    // Pass 2: map the sampled rows.
    let cols = [
        "id",
        "group_id",
        "kind",
        "question",
        "options",
        "target",
        "source",
        "state",
        "license_use",
    ];
    let mut out = Vec::new();
    let mut unmapped = 0usize;
    for p in &shards {
        for_each_parquet_row(p, &cols, |row| {
            if row.get("license_use").and_then(Value::as_str) != Some("commercial") {
                return Ok(());
            }
            let source = row
                .get("source")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let Some(&k) = keep.get(&source) else {
                return Ok(());
            };
            let id = row.get("id").and_then(Value::as_str).unwrap_or("");
            if hash_frac(cx.seed, id) >= k {
                return Ok(());
            }
            match jev_row_item(&row, "tasksource", "state") {
                Some(it) => out.push(it),
                None => unmapped += 1,
            }
            Ok(())
        })?;
    }
    eprintln!(
        "  tasksource_jev: sampled {} records ({unmapped} unmapped)",
        out.len()
    );
    let mut rng = Rng::derived(cx.seed, "tasksource_jev");
    Ok(cap_by_key(out, cx.cap, |i| i.rec.src.clone(), &mut rng))
}

// ----------------------------------------------------------------------------- Open-Jev

/// Drone and robot control sources of Open-Jev are always kept in full.
pub fn is_control_source(source: &str) -> bool {
    let s = source.to_ascii_lowercase();
    s.contains("drone") || s.contains("robot")
}

/// Cap on the non-drone/robot Open-Jev records.
pub const OPEN_JEV_OTHERS_CAP: usize = 30_000;

pub fn load_open_jev(cx: &Ctx) -> Result<Vec<Item>> {
    let repo = "ZefanCai/Open-Jev";
    let config = "browser-drone-expansion-v1-redistributable";
    let cols = [
        "group_id",
        "kind",
        "question",
        "options",
        "target",
        "source",
        "state_json",
    ];
    let (mut control, mut others) = (Vec::new(), Vec::new());
    let mut unmapped = 0usize;
    let mut per_source: BTreeMap<String, usize> = BTreeMap::new();
    for split in ["train", "calibration", "validation"] {
        for p in cx.hf.parquet_split(repo, config, split, SMALL)? {
            for_each_parquet_row(&p, &cols, |row| {
                let source = row
                    .get("source")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                *per_source.entry(source.clone()).or_default() += 1;
                match jev_row_item(&row, "open_jev", "state_json") {
                    Some(it) if is_control_source(&source) => control.push(it),
                    Some(it) => others.push(it),
                    None => unmapped += 1,
                }
                Ok(())
            })?;
        }
    }
    eprintln!("  open_jev sources: {per_source:?}");
    eprintln!(
        "  open_jev: {} drone/robot-control records kept in full, {} others capped to {OPEN_JEV_OTHERS_CAP}, {unmapped} unmapped",
        control.len(),
        others.len()
    );
    let mut rng = Rng::derived(cx.seed, "open_jev");
    control.extend(cap_by_key(
        others,
        OPEN_JEV_OTHERS_CAP,
        |i| i.rec.src.clone(),
        &mut rng,
    ));
    Ok(control)
}

// ----------------------------------------------------------------------------- Laya-format corpora

/// Reads `questions` + label columns (JSON strings or objects) of Laya-format rows.
fn laya_rows(
    hf: &Hf,
    repo: &str,
    config: &str,
    splits: &[&str],
    src: &str,
    id_col: Option<&str>,
    label_cols: &[&str],
) -> Result<Vec<Item>> {
    let mut cols = vec!["state", "questions"];
    cols.extend_from_slice(label_cols);
    if let Some(c) = id_col {
        cols.push(c);
    }
    let mut out = Vec::new();
    for split in splits {
        for p in hf.parquet_split(repo, config, split, SMALL)? {
            for_each_parquet_row(&p, &cols, |row| {
                let state = clip(str_field(&row, "state")?, STATE_MAX_CHARS);
                let qs = parse_json_field(&row, "questions")?;
                let labels: Vec<Value> = label_cols
                    .iter()
                    .map(|c| parse_json_field(&row, c).unwrap_or(Value::Null))
                    .collect();
                let refs: Vec<&Value> = labels.iter().collect();
                let group = id_col
                    .and_then(|c| row.get(c))
                    .and_then(Value::as_str)
                    .map_or_else(|| state_group(&state), str::to_string);
                out.extend(jev_items(src, &group, &state, &qs, &refs));
                Ok(())
            })?;
        }
    }
    Ok(out)
}

pub fn load_procedural(cx: &Ctx) -> Result<Vec<Item>> {
    let repo = "tasksource/procedural-typed-decisions";
    // ~260k states with several questions each: sample states first (deterministically) so
    // that about 3x the cap is mapped, then share the cap fairly between the 13 tasks.
    let frac = (3.0 * cx.cap as f64 / (260_000.0 * 6.0)).min(1.0);
    let mut out = Vec::new();
    for split in ["train", "validation"] {
        for p in cx.hf.parquet_split(repo, "all", split, SMALL)? {
            for_each_parquet_row(
                &p,
                &["id", "task", "state", "questions", "answers"],
                |row| {
                    let id = str_field(&row, "id")?;
                    if hash_frac(cx.seed, id) >= frac {
                        return Ok(());
                    }
                    let task = str_field(&row, "task")?;
                    let state = clip(str_field(&row, "state")?, STATE_MAX_CHARS);
                    let qs = parse_json_field(&row, "questions")?;
                    let answers = parse_json_field(&row, "answers")?;
                    out.extend(jev_items(
                        &format!("procedural:{task}"),
                        id,
                        &state,
                        &qs,
                        &[&answers],
                    ));
                    Ok(())
                },
            )?;
        }
    }
    let mut rng = Rng::derived(cx.seed, "procedural_typed");
    Ok(cap_by_key(out, cx.cap, |i| i.rec.src.clone(), &mut rng))
}

pub fn load_swarm_laya(cx: &Ctx) -> Result<Vec<Item>> {
    laya_rows(
        cx.hf,
        "Aditharavind/swarm-laya-decisions",
        "default",
        &["train", "validation", "test"],
        "swarm_laya",
        None,
        &["gold"],
    )
}

pub fn load_helmo(cx: &Ctx) -> Result<Vec<Item>> {
    laya_rows(
        cx.hf,
        "helmo/synthetic-typed-decisions",
        "default",
        &["train"],
        "helmo_synth",
        None,
        &["gold"],
    )
}

pub fn load_tasksource_synth(cx: &Ctx) -> Result<Vec<Item>> {
    laya_rows(
        cx.hf,
        "tasksource/synthetic-typed-decisions",
        "default",
        &["train", "validation"],
        "tasksource_synth",
        Some("id"),
        &["answers"],
    )
}

/// EEG cognitive-workload decisions (CC-BY-4.0, derived from STEW: Lim, Sourina & Wang 2018).
pub fn load_neuro_laya(cx: &Ctx) -> Result<Vec<Item>> {
    let out = laya_rows(
        cx.hf,
        "Aditharavind/laya-neuro-decisions",
        "default",
        &["train", "validation", "test"],
        "neuro_laya",
        None,
        &["gold"],
    )?;
    let mut rng = Rng::derived(cx.seed, "neuro_laya");
    Ok(cap_by_key(out, cx.cap, |i| i.rec.src.clone(), &mut rng))
}

/// Synthetic trading decisions (Apache-2.0): action, regime and confidence questions.
pub fn load_trading_laya(cx: &Ctx) -> Result<Vec<Item>> {
    laya_rows(
        cx.hf,
        "Anay2010/laya-trading-decisions",
        "default",
        &["train", "validation"],
        "trading_laya",
        None,
        &["gold"],
    )
}

pub const TETRIS_INS: &str = "Where should the current Tetris piece be placed?";

/// A nested value as the Hub's Parquet conversion stores it: a `Json` feature inside a struct
/// arrives as a JSON string, an ordinary one as the value itself.
fn nested_json(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| v.clone()),
        _ => v.clone(),
    }
}

/// One `aaung/laya-tetris-decisions` row: the board summary becomes the state, each candidate
/// placement line (`A: clears 0, holes 0, …`) an option, and the heuristic teacher's preference
/// the soft target.
pub fn tetris_item(state: &str, answers: &Value) -> Option<Item> {
    let (board, options) = state.split_once("Options:")?;
    let crit: Vec<(String, String)> = options
        .lines()
        .filter_map(|l| l.split_once(':'))
        .map(|(k, d)| (k.trim().to_string(), d.trim().to_string()))
        .filter(|(k, _)| k.len() == 1)
        .collect();
    if crit.len() < 2 {
        return None;
    }
    let q = Question::choice(TETRIS_INS, crit);
    let placement = nested_json(answers.get("placement")?);
    let target = q.target_from(&serde_json::json!({ "probabilities": placement }))?;
    let board = board.trim();
    Some(Item {
        group: state_group(board),
        rec: q.record("tetris_laya", board, target),
    })
}

pub fn load_tetris_laya(cx: &Ctx) -> Result<Vec<Item>> {
    let mut out = Vec::new();
    for split in ["train", "validation", "test"] {
        for p in cx
            .hf
            .parquet_split("aaung/laya-tetris-decisions", "default", split, SMALL)?
        {
            for_each_parquet_row(&p, &["state", "answers"], |row| {
                let answers = parse_json_field(&row, "answers")?;
                out.extend(tetris_item(str_field(&row, "state")?, &answers));
                Ok(())
            })?;
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- calibrated typed decisions

/// Whether a per-row license string allows commercial use without share-alike restrictions
/// beyond attribution: CC0, CC-BY (incl. BY-SA), MIT, Apache, BSD, ODC-BY, CDLA-Permissive.
pub fn is_permissive(license: &str) -> bool {
    let l = license.trim().to_ascii_lowercase();
    if (l.starts_with("cc") && l.contains("nc")) || l.contains("-nd") || l.is_empty() {
        return false;
    }
    l.starts_with("cc-by")
        || l.starts_with("cc0")
        || l == "mit"
        || l.starts_with("apache")
        || l.starts_with("bsd")
        || l == "odc-by"
        || l.starts_with("cdla-permissive")
        || l == "unlicense"
}

pub fn load_calibrated(cx: &Ctx) -> Result<Vec<Item>> {
    let repo = "m-lagnajit/calibrated-typed-decisions";
    let cols = [
        "id",
        "source",
        "source_license",
        "type",
        "instructions",
        "state",
        "options",
        "gold_index",
        "teacher_probs",
    ];
    let mut out = Vec::new();
    let mut licenses: BTreeMap<String, (usize, bool)> = BTreeMap::new();
    for split in ["train", "validation"] {
        for p in cx.hf.parquet_split(repo, "default", split, SMALL)? {
            for_each_parquet_row(&p, &cols, |row| {
                let lic = row
                    .get("source_license")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string();
                let ok = is_permissive(&lic);
                licenses.entry(lic).or_insert((0, ok)).0 += 1;
                if !ok {
                    return Ok(());
                }
                let options = str_list(&row["options"]).unwrap_or_default();
                let target = match f64_list(&row["teacher_probs"]) {
                    Some(t) if t.len() == options.len() => t,
                    _ => {
                        let g = row
                            .get("gold_index")
                            .and_then(Value::as_u64)
                            .unwrap_or(u64::MAX) as usize;
                        if g >= options.len() {
                            return Ok(());
                        }
                        one_hot(options.len(), g)
                            .into_iter()
                            .map(f64::from)
                            .collect()
                    }
                };
                let kind = str_field(&row, "type")?;
                let Some((q, target)) =
                    flat_question(kind, str_field(&row, "instructions")?, &options, &target)
                else {
                    return Ok(());
                };
                let state = clip(str_field(&row, "state")?, STATE_MAX_CHARS);
                let source = str_field(&row, "source")?;
                out.push(Item {
                    group: state_group(&state),
                    rec: q.record(&format!("calibrated:{source}"), state, target),
                });
                Ok(())
            })?;
        }
    }
    eprintln!("  calibrated_typed per-row licenses (rows, kept): {licenses:?}");
    Ok(out)
}

// ----------------------------------------------------------------------------- NOTAM (test only)

pub fn load_notam(cx: &Ctx) -> Result<Vec<Item>> {
    let repo = "AirsideLabs/notam-typed-decisions";
    let cols = [
        "case_id",
        "instruction",
        "context",
        "options",
        "label",
        "question_type",
    ];
    let mut out = Vec::new();
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    for config in ["notam_multi", "deel_subject"] {
        for p in cx.hf.parquet_split(repo, config, "train", SMALL)? {
            for_each_parquet_row(&p, &cols, |row| {
                let options = str_list(&row["options"]).unwrap_or_default();
                let Some(label) = row
                    .get("label")
                    .and_then(Value::as_u64)
                    .map(|l| l as usize)
                    .filter(|&l| l < options.len())
                else {
                    let why = format!("label {} / {} options", row["label"], options.len());
                    *skipped.entry(why).or_default() += 1;
                    return Ok(());
                };
                let target: Vec<f64> = one_hot(options.len(), label)
                    .into_iter()
                    .map(f64::from)
                    .collect();
                let kind = str_field(&row, "question_type")?;
                let Some((q, target)) =
                    flat_question(kind, str_field(&row, "instruction")?, &options, &target)
                else {
                    *skipped
                        .entry(format!("unmapped {kind} {options:?}"))
                        .or_default() += 1;
                    return Ok(());
                };
                let state = clip(str_field(&row, "context")?, STATE_MAX_CHARS);
                out.push(Item {
                    group: state_group(&state),
                    rec: q.record(&format!("notam:{config}"), state, target),
                });
                Ok(())
            })?;
        }
    }
    if !skipped.is_empty() {
        eprintln!("  notam: skipped rows {skipped:?}");
    }
    if out.is_empty() {
        bail!("no NOTAM records mapped");
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::validate_record;
    use serde_json::json;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn tetris_rows_become_placement_choices() {
        let state = "Tetris. Piece L, next OJS, hold none. Stack height 0, holes 0, bumpiness 0. Options:\n\
                     A: clears 0, holes 0, bumpiness 4, height 2\n\
                     B: clears 1, holes 0, bumpiness 3, height 2\n\
                     C: clears 0, holes 2, bumpiness 3, height 3";
        let it = tetris_item(state, &json!({"placement": {"A": 0.2, "B": 0.7, "C": 0.1}})).unwrap();
        let r = &it.rec;
        assert_eq!(r.t, "choice");
        assert_eq!(r.ins, TETRIS_INS);
        assert!(r.state.ends_with("bumpiness 0.") && !r.state.contains("Options"));
        assert_eq!(r.opts.len(), 3);
        assert!(r.opts[1].starts_with("B") && r.opts[1].contains("clears 1"));
        assert!((r.target[1] - 0.7).abs() < 1e-6);
        validate_record(r).unwrap();
        assert!(tetris_item("no options here", &json!({})).is_none());
        // As the Hub's Parquet stores a `Json` feature: a JSON string inside the struct.
        let s = tetris_item(
            state,
            &json!({"placement": "{\"A\": 0.2, \"B\": 0.7, \"C\": 0.1}"}),
        )
        .unwrap();
        assert_eq!(s.rec.target, it.rec.target);
    }

    #[test]
    fn flat_mapping() {
        let (q, t) = flat_question("noul", "Holds?", &[], &[0.8]).unwrap();
        assert_eq!(q.options()[1], "true: yes, the statement holds");
        assert!((t[1] - 0.8).abs() < 1e-6);
        let (_, t) = flat_question("noul", "Holds?", &s(&["no", "yes"]), &[0.25, 0.75]).unwrap();
        assert_eq!(t, vec![0.25, 0.75]);
        let (_, t) = flat_question("noul", "Holds?", &s(&["true", "false"]), &[1.0, 0.0]).unwrap();
        assert_eq!(t, vec![0.0, 1.0]);
        assert!(flat_question("noul", "Holds?", &s(&["a", "b"]), &[1.0, 0.0]).is_none());
        let (q, t) = flat_question(
            "score",
            "Rate",
            &s(&["1 out of 3", "2 out of 3", "3 out of 3"]),
            &[0.0, 2.0, 2.0],
        )
        .unwrap();
        assert_eq!(q.options()[2], "level 2: 3 out of 3");
        assert_eq!(t, vec![0.0, 0.5, 0.5]);
        let (q, _) = flat_question(
            "choice",
            "Pick",
            &s(&["entailment", "neutral"]),
            &[1.0, 0.0],
        )
        .unwrap();
        assert_eq!(q.options(), vec!["entailment", "neutral"]);
        assert!(flat_question("choice", "Pick", &s(&["a", "a"]), &[1.0, 0.0]).is_none());
        assert!(flat_question("choice", "Pick", &s(&["a", "b"]), &[1.0]).is_none());
        assert!(flat_question("weird", "Pick", &s(&["a", "b"]), &[1.0, 0.0]).is_none());
    }

    #[test]
    fn jev_rows() {
        let row = json!({"kind":"noul","question":"Is it?","options":[],"target":[1.0],"source":"x/y","state":"st","group_id":"g1"});
        let it = jev_row_item(&row, "tasksource", "state").unwrap();
        assert_eq!(it.rec.src, "tasksource:x/y");
        assert_eq!(it.group, "g1");
        assert_eq!(it.rec.target, vec![0.0, 1.0]);
        validate_record(&it.rec).unwrap();
        let row = json!({"kind":"choice","question":"Q","options":["a","b"],"target":[0.0,1.0],"source":"s","state_json":"\"hello\\nworld\""});
        let it = jev_row_item(&row, "open_jev", "state_json").unwrap();
        assert_eq!(it.rec.state, "hello\nworld");
        let row = json!({"kind":"choice","question":"Q","options":["a","b"],"target":[0.0,1.0],"source":"s","state_json":"{\"alt\": 3}"});
        assert_eq!(
            jev_row_item(&row, "open_jev", "state_json")
                .unwrap()
                .rec
                .state,
            "{\"alt\":3}"
        );
    }

    #[test]
    fn source_filters() {
        for denied in [
            "super_glue/boolq",
            "glue/mnli",
            "ai2_arc/ARC-Easy/challenge",
            "banking77",
            "sciq",
            "race/high",
            "logiqa",
            "multilingual/xnli",
            "piqa",
            "HelpSteer2/verbosity",
        ] {
            assert!(!jev_source_allowed(denied), "{denied}");
        }
        for ok in [
            "commonsense_qa_2.0",
            "anli/a1",
            "glue/rte",
            "multilingual/massive",
        ] {
            assert!(jev_source_allowed(ok), "{ok}");
        }
        assert!(is_control_source("drone-control-v1"));
        assert!(is_control_source("Robot-arm"));
        assert!(!is_control_source("browser-control-v1"));
        for ok in ["cc-by-4.0", "CC-BY-SA-4.0", "mit", "apache-2.0", "cc0-1.0"] {
            assert!(is_permissive(ok), "{ok}");
        }
        for bad in [
            "cc-by-nc-4.0",
            "cc-by-nc-sa-4.0",
            "unknown",
            "",
            "cc-by-nd-4.0",
            "other",
        ] {
            assert!(!is_permissive(bad), "{bad}");
        }
    }

    #[test]
    fn clipping() {
        assert_eq!(clip("  héllo world ", 5), "héllo");
        assert_eq!(clip("abc", 10), "abc");
        assert!(hash_frac(1, "a") < 1.0);
    }
}
