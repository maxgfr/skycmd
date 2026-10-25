//! `fetch-decisions`: Hugging Face datasets → SPEC decision records (JSONL per source).

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;

use crate::csv::parse_csv;
use crate::hf::{read_parquet, resolve_url, Hf};
use crate::npz::read_npz_u16;
use crate::record::{
    argmax, normalize, one_hot, validate_record, write_jsonl, DecisionRecord, Question,
};
use crate::rng::{hash64, Rng};

/// A record plus the key of the state it was derived from (all questions on one
/// state land in the same split).
#[derive(Clone, Debug)]
pub struct Item {
    pub group: String,
    pub rec: DecisionRecord,
}

#[derive(Clone, Debug)]
pub struct DecisionOptions {
    pub out: PathBuf,
    pub seed: u64,
    /// Default per-source cap (sources may override it, see [`Source::cap`]).
    pub cap: usize,
    pub val_frac: f64,
    /// Restrict to these source names (all when empty).
    pub only: Vec<String>,
    /// Directory of the GLM teacher outputs (`orders_decisions.jsonl`, ...).
    pub teacher: PathBuf,
}

#[derive(Clone, Debug)]
pub struct SourceReport {
    pub src: String,
    pub raw: usize,
    pub invalid: usize,
    pub duplicates: usize,
    pub train: usize,
    pub val: usize,
    pub test: usize,
    pub secs: f64,
}

/// What a loader gets: the hub client, the seed, the effective cap of its source and the
/// teacher directory.
pub struct Ctx<'a> {
    pub hf: &'a Hf,
    pub seed: u64,
    pub cap: usize,
    pub teacher: &'a Path,
}

/// A decision source: name (= output file stem), whether it is an external test set, an
/// optional cap overriding the default one, and its loader.
pub struct Source {
    pub name: &'static str,
    pub test_only: bool,
    pub cap: Option<usize>,
    pub load: fn(&Ctx) -> Result<Vec<Item>>,
}

const fn src(name: &'static str, load: fn(&Ctx) -> Result<Vec<Item>>) -> Source {
    Source {
        name,
        test_only: false,
        cap: None,
        load,
    }
}

const fn capped(name: &'static str, cap: usize, load: fn(&Ctx) -> Result<Vec<Item>>) -> Source {
    Source {
        name,
        test_only: false,
        cap: Some(cap),
        load,
    }
}

pub fn sources() -> Vec<Source> {
    use crate::nlu::*;
    use crate::typed::*;
    vec![
        src("typed_synth", load_typed_synth),
        src("assistant_decisions", load_assistant_decisions),
        src("pngwn_typed", load_pngwn),
        src("boolq", load_boolq),
        src("mnli", load_mnli),
        src("arc", load_arc),
        src("commonsense_qa", load_csqa),
        src("banking77", load_banking77),
        Source {
            name: "typed_decisions_bench",
            test_only: true,
            cap: None,
            load: load_bench,
        },
        // The train split of the same benchmark (s5: the "fitted" bench column), split
        // train / val by state.
        src("typed_decisions_train", load_bench_train),
        // Typed-decision corpora on the Hub.
        capped("tasksource_jev", 120_000, load_tasksource_jev),
        capped("procedural_typed", 80_000, load_procedural),
        // The loader applies its own cap (all drone/robot rows + 30k others).
        capped("open_jev", usize::MAX, load_open_jev),
        capped("swarm_laya", usize::MAX, load_swarm_laya),
        src("helmo_synth", load_helmo),
        src("tasksource_synth", load_tasksource_synth),
        src("calibrated_typed", load_calibrated),
        Source {
            name: "notam",
            test_only: true,
            cap: Some(usize::MAX),
            load: load_notam,
        },
        // Classic NLU / preference sets converted to typed questions.
        src("helpsteer2", load_helpsteer2),
        src("pawsx_fr", load_pawsx_fr),
        capped("allocine", 30_000, load_allocine),
        src("when2call", load_when2call),
        capped("snli", 40_000, load_snli),
        capped("hellaswag", 30_000, load_hellaswag),
        src("qasc", load_qasc),
        capped("refusal", 30_000, load_refusal),
        // Laya-format corpora of September–October 2026 (issue #38).
        capped("neuro_laya", 20_000, load_neuro_laya),
        src("trading_laya", load_trading_laya),
        src("tetris_laya", load_tetris_laya),
        // GLM teacher data (local, no download).
        capped("glm_orders", usize::MAX, crate::augment::load_glm_orders),
    ]
}

/// Runs every (selected) source and writes `<out>/<src>.jsonl`.
pub fn fetch_decisions(hf: &Hf, opts: &DecisionOptions) -> Result<Vec<SourceReport>> {
    let all = sources();
    for o in &opts.only {
        if !all.iter().any(|s| s.name == o) {
            let names: Vec<&str> = all.iter().map(|s| s.name).collect();
            bail!("unknown source {o:?}; known: {}", names.join(", "));
        }
    }
    std::fs::create_dir_all(&opts.out)?;
    let mut reports = Vec::new();
    for s in all {
        if !opts.only.is_empty() && !opts.only.iter().any(|o| o == s.name) {
            continue;
        }
        let t0 = Instant::now();
        eprintln!("[decisions] {} ...", s.name);
        let o = DecisionOptions {
            cap: s.cap.unwrap_or(opts.cap),
            ..opts.clone()
        };
        let cx = Ctx {
            hf,
            seed: opts.seed,
            cap: o.cap,
            teacher: &opts.teacher,
        };
        let items = (s.load)(&cx).with_context(|| format!("source {}", s.name))?;
        if items.is_empty() && s.name == "glm_orders" {
            // Local teacher data that is not there: no file, so no empty source either.
            continue;
        }
        let mut report = finalize(
            s.name,
            items,
            s.test_only,
            &o,
            &opts.out.join(format!("{}.jsonl", s.name)),
        )?;
        report.secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "[decisions] {}: raw {} invalid {} dup {} -> train {} val {} test {} ({:.1}s)",
            report.src,
            report.raw,
            report.invalid,
            report.duplicates,
            report.train,
            report.val,
            report.test,
            report.secs
        );
        reports.push(report);
    }
    Ok(reports)
}

/// Validates, dedups, caps with label-balanced sampling, splits by state group and writes.
/// Records a loader already marked `test` (an official labelled test split) stay `test`.
pub fn finalize(
    src: &str,
    items: Vec<Item>,
    test_only: bool,
    opts: &DecisionOptions,
    path: &Path,
) -> Result<SourceReport> {
    let raw = items.len();
    let mut seen = HashSet::new();
    let (mut invalid, mut duplicates) = (0, 0);
    let mut kept = Vec::with_capacity(items.len());
    for it in items {
        if validate_record(&it.rec).is_err() {
            invalid += 1;
            continue;
        }
        let key = hash64(
            format!(
                "{}\u{1}{}\u{1}{}",
                it.rec.ins,
                it.rec.opts.join("\u{2}"),
                it.rec.state
            )
            .as_bytes(),
        );
        if !seen.insert(key) {
            duplicates += 1;
            continue;
        }
        kept.push(it);
    }
    let mut rng = Rng::derived(opts.seed, src);
    let mut kept = balanced_cap(kept, opts.cap, &mut rng);
    for it in kept.iter_mut() {
        it.rec.split = if test_only || it.rec.split == "test" {
            "test".into()
        } else if (hash64(format!("{src}/{}", it.group).as_bytes()) % 1_000_000) as f64 / 1e6
            < opts.val_frac
        {
            "val".into()
        } else {
            "train".into()
        };
    }
    rng.shuffle(&mut kept);
    let count = |s: &str| kept.iter().filter(|i| i.rec.split == s).count();
    let report = SourceReport {
        src: src.to_string(),
        raw,
        invalid,
        duplicates,
        train: count("train"),
        val: count("val"),
        test: count("test"),
        secs: 0.0,
    };
    write_jsonl(path, kept.iter().map(|i| &i.rec))?;
    Ok(report)
}

/// Water-filling quotas: splits `cap` over keys as evenly as possible without giving any key
/// more than it has (keys with few items keep all of them, the rest share what is left).
pub fn fair_quotas(counts: &BTreeMap<String, usize>, cap: usize) -> BTreeMap<String, usize> {
    let mut order: Vec<(&String, usize)> = counts.iter().map(|(k, &n)| (k, n)).collect();
    order.sort_by_key(|&(k, n)| (n, k.clone()));
    let mut left = cap;
    let mut out = BTreeMap::new();
    for (i, (k, n)) in order.iter().enumerate() {
        let share = left / (order.len() - i);
        let q = (*n).min(share);
        out.insert((*k).clone(), q);
        left -= q;
    }
    out
}

/// Keeps at most `cap` items with per-key fair quotas ([`fair_quotas`]), sampling uniformly
/// inside each key; e.g. keyed by `src` so one large sub-source cannot crowd out the others.
pub fn cap_by_key(
    items: Vec<Item>,
    cap: usize,
    key: impl Fn(&Item) -> String,
    rng: &mut Rng,
) -> Vec<Item> {
    if items.len() <= cap {
        return items;
    }
    let mut groups: BTreeMap<String, Vec<Item>> = BTreeMap::new();
    for it in items {
        groups.entry(key(&it)).or_default().push(it);
    }
    let counts = groups.iter().map(|(k, v)| (k.clone(), v.len())).collect();
    let quotas = fair_quotas(&counts, cap);
    let mut out = Vec::with_capacity(cap);
    for (k, mut v) in groups {
        rng.shuffle(&mut v);
        v.truncate(quotas[&k]);
        out.extend(v);
    }
    out
}

/// Keeps at most `cap` items, drawing round-robin from strata `(type, #options, argmax)`
/// so a cap never skews the label distribution towards whatever came first.
pub fn balanced_cap(items: Vec<Item>, cap: usize, rng: &mut Rng) -> Vec<Item> {
    if items.len() <= cap {
        return items;
    }
    let mut strata: BTreeMap<(String, usize, usize), Vec<Item>> = BTreeMap::new();
    for it in items {
        let k = (it.rec.t.clone(), it.rec.opts.len(), argmax(&it.rec.target));
        strata.entry(k).or_default().push(it);
    }
    let mut buckets: Vec<Vec<Item>> = strata.into_values().collect();
    for b in buckets.iter_mut() {
        rng.shuffle(b);
    }
    let mut out = Vec::with_capacity(cap);
    while out.len() < cap {
        let mut progressed = false;
        for b in buckets.iter_mut() {
            if out.len() >= cap {
                break;
            }
            if let Some(it) = b.pop() {
                out.push(it);
                progressed = true;
            }
        }
        if !progressed {
            break;
        }
    }
    out
}

pub(crate) fn str_field<'a>(row: &'a Value, k: &str) -> Result<&'a str> {
    row.get(k)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string field {k}"))
}

pub(crate) fn parse_json_field(row: &Value, k: &str) -> Result<Value> {
    match row.get(k) {
        Some(Value::String(s)) => {
            Ok(serde_json::from_str(s).with_context(|| format!("field {k}"))?)
        }
        Some(v) => Ok(v.clone()),
        None => bail!("missing field {k}"),
    }
}

/// Converts one System One / Jev case (`questions` + label objects) to items.
/// `labels` are tried in order per question (e.g. teacher soft labels before gold).
pub fn jev_items(
    src: &str,
    group: &str,
    state: &str,
    questions: &Value,
    labels: &[&Value],
) -> Vec<Item> {
    let Some(qs) = questions.as_object() else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for (qid, qv) in qs {
        let Ok(q) = Question::from_jev(qv) else {
            continue;
        };
        let target = labels
            .iter()
            .find_map(|l| l.get(qid).and_then(|v| q.target_from(v)));
        if let Some(target) = target {
            out.push(Item {
                group: group.to_string(),
                rec: q.record(src, state, target),
            });
        }
    }
    out
}

pub(crate) const SMALL: u64 = 512 << 20;

fn load_typed_synth(cx: &Ctx) -> Result<Vec<Item>> {
    let (hf, _seed) = (cx.hf, cx.seed);
    let repo = "n4ze3m/typed-decisions-synth";
    let mut out = Vec::new();
    for split in ["train", "validation"] {
        for p in hf.parquet_split(repo, "default", split, SMALL)? {
            for row in read_parquet(
                &p,
                &["state_id", "state", "questions", "gold", "teacher"],
                usize::MAX,
            )? {
                let qs = parse_json_field(&row, "questions")?;
                let gold = parse_json_field(&row, "gold")?;
                let teacher = parse_json_field(&row, "teacher").unwrap_or(Value::Null);
                out.extend(jev_items(
                    "typed_synth",
                    str_field(&row, "state_id")?,
                    str_field(&row, "state")?,
                    &qs,
                    &[&teacher, &gold],
                ));
            }
        }
    }
    Ok(out)
}

fn load_assistant_decisions(cx: &Ctx) -> Result<Vec<Item>> {
    let (hf, _seed) = (cx.hf, cx.seed);
    let repo = "kgrozdanovski/assistant-decisions";
    let mut out = Vec::new();
    for split in ["train", "validation", "test"] {
        for p in hf.parquet_split(repo, "default", split, SMALL)? {
            for row in read_parquet(&p, &["case_id", "state", "questions", "gold"], usize::MAX)? {
                let qs = parse_json_field(&row, "questions")?;
                let gold = parse_json_field(&row, "gold")?;
                out.extend(jev_items(
                    "assistant_decisions",
                    str_field(&row, "case_id")?,
                    str_field(&row, "state")?,
                    &qs,
                    &[&gold],
                ));
            }
        }
    }
    Ok(out)
}

fn load_bench(cx: &Ctx) -> Result<Vec<Item>> {
    load_typed_decisions(cx, "test", "typed_decisions_bench")
}

fn load_bench_train(cx: &Ctx) -> Result<Vec<Item>> {
    load_typed_decisions(cx, "train", "typed_decisions_train")
}

/// One split of `LocalLLaMA/typed-decisions` (config `all`).
fn load_typed_decisions(cx: &Ctx, split: &str, name: &str) -> Result<Vec<Item>> {
    let repo = "LocalLLaMA/typed-decisions";
    let mut out = Vec::new();
    for p in cx.hf.parquet_split(repo, "all", split, SMALL)? {
        for row in read_parquet(&p, &["id", "state", "questions", "gold"], usize::MAX)? {
            let qs = parse_json_field(&row, "questions")?;
            let gold = parse_json_field(&row, "gold")?;
            out.extend(jev_items(
                name,
                str_field(&row, "id")?,
                str_field(&row, "state")?,
                &qs,
                &[&gold],
            ));
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- pngwn

/// One question parsed from a `pngwn/typed-decisions` prompt.
#[derive(Debug, Clone, PartialEq)]
pub struct PngwnPrompt {
    pub state: String,
    pub questions: Vec<String>,
}

/// Parses the `### State: / ### Question(s): / ### Options: / ### Answer:` prompt layout.
pub fn parse_pngwn_prompt(text: &str) -> Option<PngwnPrompt> {
    let s = text.find("### State:")? + "### State:".len();
    let rest = &text[s..];
    if let Some(qi) = rest.find("### Questions:") {
        let state = rest[..qi].trim().to_string();
        let body = &rest[qi + "### Questions:".len()..];
        let body = &body[..body.find("### Answer:").unwrap_or(body.len())];
        let mut questions = Vec::new();
        for line in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
            // "N) question text [A) x, B) y]"
            let Some(close) = line.find(") ") else {
                continue;
            };
            if !line[..close].chars().all(|c| c.is_ascii_digit()) {
                if let Some(last) = questions.last_mut() {
                    let last: &mut String = last;
                    last.push(' ');
                    last.push_str(line);
                }
                continue;
            }
            let q = &line[close + 2..];
            let q = q.rfind(" [A) ").map_or(q, |i| &q[..i]);
            questions.push(q.trim().to_string());
        }
        return (!questions.is_empty()).then_some(PngwnPrompt { state, questions });
    }
    let qi = rest.find("### Question:")?;
    let state = rest[..qi].trim().to_string();
    let body = &rest[qi + "### Question:".len()..];
    let end = body
        .find("### Options:")
        .or_else(|| body.find("### Answer:"))
        .unwrap_or(body.len());
    Some(PngwnPrompt {
        state,
        questions: vec![body[..end].trim().to_string()],
    })
}

/// Maps one pngwn decision (option texts + target over them) to a typed question.
pub fn pngwn_question(
    question: &str,
    state: &str,
    options: &[String],
    target: &[f64],
) -> Option<(Question, String, Vec<f32>)> {
    if options.len() < 2 || options.len() != target.len() {
        return None;
    }
    let lower: Vec<String> = options
        .iter()
        .map(|o| o.trim().to_ascii_lowercase())
        .collect();
    // Long questions do not fit the 96-token head: keep them in the state instead.
    let (ins, state) = if question.chars().count() > 160 {
        (
            "Which option correctly answers the question?".to_string(),
            format!("{state}\nQuestion: {question}"),
        )
    } else {
        (question.to_string(), state.to_string())
    };
    if lower == ["yes", "no"] || lower == ["no", "yes"] {
        let p_yes = if lower[0] == "yes" {
            target[0]
        } else {
            target[1]
        };
        return Some((Question::noul(ins), state, normalize(&[1.0 - p_yes, p_yes])));
    }
    let numeric: Option<Vec<i64>> = options
        .iter()
        .map(|o| o.trim().parse::<i64>().ok())
        .collect();
    if let Some(n) = numeric {
        if n.windows(2).all(|w| w[1] == w[0] + 1) {
            let levels = options.iter().map(|o| o.trim().to_string()).collect();
            return Some((Question::score(ins, levels), state, normalize(target)));
        }
    }
    let short = options
        .iter()
        .all(|o| o.chars().count() <= 24 && o.split_whitespace().count() <= 3)
        && options.iter().collect::<HashSet<_>>().len() == options.len();
    let crit = if short {
        options
            .iter()
            .map(|o| (o.trim().to_string(), String::new()))
            .collect()
    } else {
        options
            .iter()
            .enumerate()
            .map(|(i, o)| (((b'A' + i as u8) as char).to_string(), o.trim().to_string()))
            .collect()
    };
    Some((Question::choice(ins, crit), state, normalize(target)))
}

fn load_pngwn(cx: &Ctx) -> Result<Vec<Item>> {
    let (hf, _seed) = (cx.hf, cx.seed);
    let repo = "pngwn/typed-decisions";
    let gpt2 = hf.download(
        "https://huggingface.co/openai-community/gpt2/resolve/main/tokenizer.json",
        "openai-community__gpt2/tokenizer.json",
        SMALL,
    )?;
    let tok =
        tokenizers::Tokenizer::from_file(&gpt2).map_err(|e| anyhow!("gpt2 tokenizer: {e}"))?;
    let mut out = Vec::new();
    for split in ["train", "cal"] {
        let npz = hf.download(
            &resolve_url(repo, "main", &format!("{split}.npz")),
            &format!("pngwn__typed-decisions/{split}.npz"),
            SMALL,
        )?;
        let meta = hf.download(
            &resolve_url(repo, "main", &format!("{split}_meta.jsonl")),
            &format!("pngwn__typed-decisions/{split}_meta.jsonl"),
            SMALL,
        )?;
        let prompts = read_npz_u16(&npz, "prompts")?;
        let metas = std::fs::read_to_string(&meta)?;
        let metas: Vec<&str> = metas.lines().filter(|l| !l.trim().is_empty()).collect();
        if metas.len() != prompts.rows {
            bail!(
                "pngwn {split}: {} meta rows vs {} prompts",
                metas.len(),
                prompts.rows
            );
        }
        let mut unparsed = 0;
        for (i, line) in metas.iter().enumerate() {
            let m: Value = serde_json::from_str(line)?;
            let ids: Vec<u32> = prompts
                .row(i)
                .iter()
                .map(|&x| x as u32)
                .filter(|&x| x != 50256)
                .collect();
            let text = tok
                .decode(&ids, false)
                .map_err(|e| anyhow!("decode: {e}"))?;
            let Some(p) = parse_pngwn_prompt(&text) else {
                unparsed += 1;
                continue;
            };
            let options: Vec<Vec<String>> =
                serde_json::from_value(m["options"].clone()).unwrap_or_default();
            let gold: Vec<usize> = serde_json::from_value(m["gold"].clone()).unwrap_or_default();
            let probs = m["gold_probs"].as_array().cloned().unwrap_or_default();
            if p.questions.len() != options.len() || gold.len() != options.len() {
                unparsed += 1;
                continue;
            }
            let uid = m["uid"].as_str().unwrap_or("").to_string();
            for (qi, q) in p.questions.iter().enumerate() {
                let k = options[qi].len();
                let target: Vec<f64> = match probs.get(qi).and_then(Value::as_array) {
                    Some(a) if a.len() == k => {
                        a.iter().map(|x| x.as_f64().unwrap_or(0.0)).collect()
                    }
                    _ => one_hot(k, gold[qi]).into_iter().map(f64::from).collect(),
                };
                if let Some((question, state, target)) =
                    pngwn_question(q, &p.state, &options[qi], &target)
                {
                    out.push(Item {
                        group: uid.clone(),
                        rec: question.record("pngwn_typed", state, target),
                    });
                }
            }
        }
        if unparsed > 0 {
            eprintln!("  pngwn {split}: {unparsed} prompts not parsed");
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- public NLU sets

fn capitalize(s: &str) -> String {
    let mut c = s.trim().chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// BoolQ question → statement check (`noul`).
pub fn boolq_item(question: &str, passage: &str, answer: bool) -> Item {
    let q = capitalize(question.trim().trim_end_matches('?'));
    let ins = format!("Based on the passage, the answer to \"{q}?\" is yes.");
    let rec = Question::noul(ins).record("boolq", passage.trim(), one_hot(2, answer as usize));
    Item {
        group: format!("{:x}", hash64(passage.as_bytes())),
        rec,
    }
}

fn load_boolq(cx: &Ctx) -> Result<Vec<Item>> {
    let (hf, _seed) = (cx.hf, cx.seed);
    let mut out = Vec::new();
    for split in ["train", "validation"] {
        for p in hf.parquet_split("google/boolq", "default", split, SMALL)? {
            for row in read_parquet(&p, &["question", "answer", "passage"], usize::MAX)? {
                let answer = row
                    .get("answer")
                    .and_then(Value::as_bool)
                    .ok_or_else(|| anyhow!("boolq answer"))?;
                out.push(boolq_item(
                    str_field(&row, "question")?,
                    str_field(&row, "passage")?,
                    answer,
                ));
            }
        }
    }
    Ok(out)
}

pub const MNLI_LABELS: [(&str, &str); 3] = [
    ("entailment", "the premise implies the hypothesis is true"),
    (
        "neutral",
        "the hypothesis may or may not be true given the premise",
    ),
    (
        "contradiction",
        "the premise implies the hypothesis is false",
    ),
];

pub fn mnli_item(premise: &str, hypothesis: &str, label: usize) -> Item {
    let crit = MNLI_LABELS
        .iter()
        .map(|(k, d)| (k.to_string(), d.to_string()))
        .collect();
    let ins = format!(
        "How does the premise relate to the hypothesis \"{}\"?",
        hypothesis.trim()
    );
    Item {
        group: format!("{:x}", hash64(premise.as_bytes())),
        rec: Question::choice(ins, crit).record("mnli", premise.trim(), one_hot(3, label)),
    }
}

fn load_mnli(cx: &Ctx) -> Result<Vec<Item>> {
    let (hf, _seed) = (cx.hf, cx.seed);
    let mut out = Vec::new();
    for p in hf.parquet_split("nyu-mll/glue", "mnli", "train", SMALL)? {
        for row in read_parquet(&p, &["premise", "hypothesis", "label"], usize::MAX)? {
            let label = row.get("label").and_then(Value::as_i64).unwrap_or(-1);
            if !(0..3).contains(&label) {
                continue;
            }
            out.push(mnli_item(
                str_field(&row, "premise")?,
                str_field(&row, "hypothesis")?,
                label as usize,
            ));
        }
    }
    Ok(out)
}

/// Multiple-choice row (ARC / CommonsenseQA) → choice over keys `A..` with answer texts.
pub fn mc_item(
    src: &str,
    ins: &str,
    id: &str,
    question: &str,
    choices: &Value,
    answer_key: &str,
) -> Option<Item> {
    let labels: Vec<String> = serde_json::from_value(choices.get("label")?.clone()).ok()?;
    let texts: Vec<String> = serde_json::from_value(choices.get("text")?.clone()).ok()?;
    if labels.len() != texts.len() || labels.len() < 2 || labels.len() > 26 {
        return None;
    }
    let answer = labels.iter().position(|l| l == answer_key.trim())?;
    let crit = texts
        .iter()
        .enumerate()
        .map(|(i, t)| (((b'A' + i as u8) as char).to_string(), t.trim().to_string()))
        .collect();
    Some(Item {
        group: id.to_string(),
        rec: Question::choice(ins, crit).record(
            src,
            question.trim(),
            one_hot(labels.len(), answer),
        ),
    })
}

fn load_arc(cx: &Ctx) -> Result<Vec<Item>> {
    let (hf, _seed) = (cx.hf, cx.seed);
    let mut out = Vec::new();
    for config in ["ARC-Easy", "ARC-Challenge"] {
        for split in ["train", "validation", "test"] {
            for p in hf.parquet_split("allenai/ai2_arc", config, split, SMALL)? {
                for row in
                    read_parquet(&p, &["id", "question", "choices", "answerKey"], usize::MAX)?
                {
                    out.extend(mc_item(
                        "arc",
                        "Which option correctly answers the science question?",
                        str_field(&row, "id")?,
                        str_field(&row, "question")?,
                        &row["choices"],
                        str_field(&row, "answerKey")?,
                    ));
                }
            }
        }
    }
    Ok(out)
}

fn load_csqa(cx: &Ctx) -> Result<Vec<Item>> {
    let (hf, _seed) = (cx.hf, cx.seed);
    let mut out = Vec::new();
    for split in ["train", "validation"] {
        for p in hf.parquet_split("tau/commonsense_qa", "default", split, SMALL)? {
            for row in read_parquet(&p, &["id", "question", "choices", "answerKey"], usize::MAX)? {
                out.extend(mc_item(
                    "commonsense_qa",
                    "Which option best answers the commonsense question?",
                    str_field(&row, "id")?,
                    str_field(&row, "question")?,
                    &row["choices"],
                    row.get("answerKey").and_then(Value::as_str).unwrap_or(""),
                ));
            }
        }
    }
    Ok(out)
}

/// `card_arrival` → `card arrival`, with common acronyms restored.
pub fn humanize_intent(s: &str) -> String {
    s.split('_')
        .filter(|w| !w.is_empty())
        .map(|w| match w.to_ascii_lowercase().as_str() {
            "atm" => "ATM".to_string(),
            "pin" => "PIN".to_string(),
            "apple" => "Apple".to_string(),
            "google" => "Google".to_string(),
            "visa" => "Visa".to_string(),
            "mastercard" => "Mastercard".to_string(),
            "fiat" => "fiat".to_string(),
            o => o.to_string(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Banking77 utterance → choice over the gold intent and 7 random distractors (shuffled).
pub fn banking_item(text: &str, gold: &str, intents: &[String], rng: &mut Rng) -> Item {
    let mut cands: Vec<&String> = intents.iter().filter(|i| i.as_str() != gold).collect();
    rng.shuffle(&mut cands);
    let gold_s = gold.to_string();
    let mut picked: Vec<&String> = cands.into_iter().take(7).collect();
    picked.push(&gold_s);
    rng.shuffle(&mut picked);
    let answer = picked.iter().position(|c| c.as_str() == gold).unwrap();
    let crit = picked
        .iter()
        .map(|c| (humanize_intent(c), String::new()))
        .collect();
    let n = picked.len();
    Item {
        group: format!("{:x}", hash64(text.as_bytes())),
        rec: Question::choice(
            "Which intent does the customer's banking message express?",
            crit,
        )
        .record("banking77", text.trim(), one_hot(n, answer)),
    }
}

fn load_banking77(cx: &Ctx) -> Result<Vec<Item>> {
    let (hf, seed) = (cx.hf, cx.seed);
    // PolyAI/banking77 is a script dataset; its loader reads these two CSV files.
    let base =
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data";
    let mut rows = Vec::new();
    for split in ["train", "test"] {
        let p = hf.download(
            &format!("{base}/{split}.csv"),
            &format!("PolyAI__banking77/{split}.csv"),
            SMALL,
        )?;
        let parsed = parse_csv(&std::fs::read_to_string(p)?);
        rows.extend(parsed.into_iter().skip(1).filter(|r| r.len() == 2));
    }
    let mut intents: Vec<String> = rows
        .iter()
        .map(|r| r[1].clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    intents.sort();
    if intents.len() < 8 {
        bail!("banking77: only {} intents", intents.len());
    }
    let mut rng = Rng::derived(seed, "banking77-distractors");
    Ok(rows
        .iter()
        .map(|r| banking_item(&r[0], &r[1], &intents, &mut rng))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn jev_case_to_items_prefers_teacher() {
        let qs = json!({
            "a": {"type":"noul","instructions":"Holds?","criteria":{"true":"yes","false":"no"}},
            "b": {"type":"choice","instructions":"Pick","criteria":{"x":"ex","y":"why"}},
            "c": {"type":"score","instructions":"Rate","criteria":["lo","hi"]},
            "bad": {"type":"weird","instructions":"?"}
        });
        let teacher = json!({"a":{"noul":0.9},"b":{"probabilities":{"x":0.25,"y":0.75}}});
        let gold = json!({"a":false,"b":"x","c":1});
        let items = jev_items("s", "g", "state", &qs, &[&teacher, &gold]);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].rec.opts, vec!["false: no", "true: yes"]);
        assert!((items[0].rec.target[1] - 0.9).abs() < 1e-6);
        assert_eq!(items[1].rec.target, vec![0.25, 0.75]);
        assert_eq!(items[2].rec.target, vec![0.0, 1.0]);
        assert_eq!(items[2].rec.opts, vec!["level 0: lo", "level 1: hi"]);
    }

    #[test]
    fn parses_pngwn_single_and_multi() {
        let single = "### State:\nTicket from Bob.\n\n### Question:\nShould we escalate?\n\n### Options:\nA) yes\nB) no\n\n### Answer:\n";
        let p = parse_pngwn_prompt(single).unwrap();
        assert_eq!(p.state, "Ticket from Bob.");
        assert_eq!(p.questions, vec!["Should we escalate?"]);
        let multi = "### State:\nS\n\n### Questions:\n1) How severe? [A) 1, B) 2]\n2) Review? [A) yes, B) no]\n\n### Answer:\n";
        let p = parse_pngwn_prompt(multi).unwrap();
        assert_eq!(p.questions, vec!["How severe?", "Review?"]);
    }

    #[test]
    fn pngwn_type_mapping() {
        let (q, _, t) =
            pngwn_question("Escalate?", "s", &["yes".into(), "no".into()], &[0.8, 0.2]).unwrap();
        assert_eq!(q.t.as_str(), "noul");
        assert!((t[1] - 0.8).abs() < 1e-6 && (t[0] - 0.2).abs() < 1e-6);
        let (q, _, _) = pngwn_question(
            "Sev?",
            "s",
            &["1".into(), "2".into(), "3".into()],
            &[0.0, 1.0, 0.0],
        )
        .unwrap();
        assert_eq!(q.options()[0], "level 0: 1");
        let (q, _, _) = pngwn_question(
            "Team?",
            "s",
            &["platform".into(), "billing".into()],
            &[1.0, 0.0],
        )
        .unwrap();
        assert_eq!(q.options(), vec!["platform", "billing"]);
        let long = vec![
            "a rather long answer text here".to_string(),
            "another long answer option text".to_string(),
        ];
        let (q, _, _) = pngwn_question(&"x".repeat(200), "s", &long, &[1.0, 0.0]).unwrap();
        assert!(q.options()[0].starts_with("A: "));
        assert_eq!(q.ins, "Which option correctly answers the question?");
    }

    #[test]
    fn public_set_mappings() {
        let b = boolq_item("is the sky blue", "The sky is blue.", true);
        assert_eq!(
            b.rec.ins,
            "Based on the passage, the answer to \"Is the sky blue?\" is yes."
        );
        assert_eq!(b.rec.target, vec![0.0, 1.0]);
        validate_record(&b.rec).unwrap();
        let m = mnli_item("P.", "H.", 2);
        assert_eq!(
            m.rec.opts[2],
            "contradiction: the premise implies the hypothesis is false"
        );
        let c = mc_item(
            "arc",
            "Which?",
            "id1",
            "Q?",
            &json!({"label":["1","2","3"],"text":["a","b","c"]}),
            "2",
        )
        .unwrap();
        assert_eq!(c.rec.opts, vec!["A: a", "B: b", "C: c"]);
        assert_eq!(c.rec.target, vec![0.0, 1.0, 0.0]);
        assert!(mc_item(
            "arc",
            "W",
            "i",
            "Q",
            &json!({"label":["A","B"],"text":["a","b"]}),
            ""
        )
        .is_none());
    }

    #[test]
    fn banking_candidates() {
        let intents: Vec<String> = (0..20).map(|i| format!("intent_{i}")).collect();
        let mut rng = Rng::new(3);
        let it = banking_item("hello", "intent_5", &intents, &mut rng);
        assert_eq!(it.rec.opts.len(), 8);
        assert_eq!(it.rec.opts[argmax(&it.rec.target)], "intent 5");
        assert_eq!(humanize_intent("atm_support"), "ATM support");
        assert_eq!(
            humanize_intent("apple_pay_or_google_pay"),
            "Apple pay or Google pay"
        );
    }

    #[test]
    fn cap_is_balanced_and_split_is_grouped() {
        let mut items = Vec::new();
        for i in 0..1000 {
            let label = usize::from(i % 10 == 0); // 10% positives
            let mut it = boolq_item(&format!("q{i}"), &format!("passage {}", i / 2), label == 1);
            it.group = format!("g{}", i / 2);
            items.push(it);
        }
        let capped = balanced_cap(items.clone(), 100, &mut Rng::new(1));
        assert_eq!(capped.len(), 100);
        let pos = capped.iter().filter(|i| argmax(&i.rec.target) == 1).count();
        assert_eq!(pos, 50);
        let dir = std::env::temp_dir().join(format!("sky-data-dec-{}", std::process::id()));
        let opts = DecisionOptions {
            out: dir.clone(),
            seed: 1,
            cap: 60_000,
            val_frac: 0.05,
            only: vec![],
            teacher: dir.clone(),
        };
        let rep = finalize("boolq", items, false, &opts, &dir.join("b.jsonl")).unwrap();
        assert_eq!(rep.train + rep.val, 1000);
        assert!(rep.val > 10 && rep.val < 120, "val {}", rep.val);
        let recs: Vec<DecisionRecord> = crate::record::read_jsonl(&dir.join("b.jsonl")).unwrap();
        let mut by_state: BTreeMap<String, HashSet<String>> = BTreeMap::new();
        for r in recs {
            by_state.entry(r.state).or_default().insert(r.split);
        }
        assert!(by_state.values().all(|s| s.len() == 1));
        std::fs::remove_dir_all(dir).ok();
    }
}
