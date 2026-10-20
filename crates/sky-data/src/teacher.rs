//! GLM teacher client (any endpoint speaking the Anthropic Messages protocol) and the teacher
//! tasks: question paraphrases and civil drone orders.
//!
//! The endpoint, key and model are read at runtime from the environment (`SKY_TEACHER_URL`,
//! `SKY_TEACHER_KEY`, `SKY_TEACHER_MODEL`). The API key is never printed, logged or written: it
//! lives only inside [`TeacherConfig`], whose `Debug` output redacts it, and is only ever sent as
//! the `x-api-key` request header.

use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::plan::{check_category, first_mission_action, validate_plan, ACTIONS};
use crate::record::{one_hot, read_jsonl, write_jsonl, DecisionRecord, Question};
use crate::rng::{hash64, Rng};

pub const DEFAULT_MODEL: &str = "glm-4.6";
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Endpoint settings. The key is private and redacted from `Debug`.
#[derive(Clone)]
pub struct TeacherConfig {
    pub base_url: String,
    api_key: String,
    pub model: String,
    pub max_calls: usize,
    pub batch: usize,
    pub min_interval: Duration,
    pub max_tokens: u32,
    pub temperature: f64,
    pub timeout: Duration,
}

impl fmt::Debug for TeacherConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TeacherConfig")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("max_calls", &self.max_calls)
            .field("batch", &self.batch)
            .finish()
    }
}

fn env_nonempty(k: &str) -> Option<String> {
    std::env::var(k)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

#[derive(Clone, Debug, Default)]
pub struct TeacherArgs {
    pub model: Option<String>,
    pub max_calls: usize,
    pub batch: usize,
    pub min_interval_ms: u64,
}

impl TeacherConfig {
    /// Resolves URL, key and model from the environment.
    pub fn resolve(args: &TeacherArgs) -> Result<Self> {
        let base_url = env_nonempty("SKY_TEACHER_URL")
            .ok_or_else(|| {
                anyhow!("no teacher endpoint: set SKY_TEACHER_URL (Anthropic Messages protocol)")
            })?
            .trim_end_matches('/')
            .to_string();
        let api_key = env_nonempty("SKY_TEACHER_KEY")
            .ok_or_else(|| anyhow!("no teacher API key: set SKY_TEACHER_KEY"))?;
        let model = args
            .model
            .clone()
            .or_else(|| env_nonempty("SKY_TEACHER_MODEL"))
            .unwrap_or_else(|| "auto".into());
        let mut cfg = Self {
            base_url,
            api_key,
            model,
            max_calls: args.max_calls,
            batch: args.batch.max(1),
            min_interval: Duration::from_millis(args.min_interval_ms),
            max_tokens: 16_000,
            temperature: 0.9,
            timeout: Duration::from_secs(300),
        };
        if cfg.model.eq_ignore_ascii_case("auto") {
            cfg.model = pick_model(&cfg).unwrap_or_else(|e| {
                eprintln!("[teacher] model list unavailable ({e:#}); using {DEFAULT_MODEL}");
                DEFAULT_MODEL.to_string()
            });
        }
        Ok(cfg)
    }
}

/// Picks the newest full GLM model from the endpoint's model list (`GET /v1/models`),
/// preferring the full model over Air/Flash/Turbo variants.
fn pick_model(cfg: &TeacherConfig) -> Result<String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(30)))
        .build()
        .into();
    let mut resp = agent
        .get(&format!("{}/v1/models", cfg.base_url))
        .header("x-api-key", &cfg.api_key)
        .header("anthropic-version", ANTHROPIC_VERSION)
        .call()?;
    let v: Value = resp.body_mut().read_json()?;
    choose_model(&v).ok_or_else(|| anyhow!("no GLM model in list"))
}

/// Model choice from a `/v1/models` listing (pure, for testing).
pub fn choose_model(list: &Value) -> Option<String> {
    let mut models: Vec<(String, String)> = list
        .get("data")?
        .as_array()?
        .iter()
        .filter_map(|m| {
            Some((
                m.get("id")?.as_str()?.to_string(),
                m.get("created_at")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .to_string(),
            ))
        })
        .filter(|(id, _)| id.to_ascii_lowercase().contains("glm"))
        .collect();
    let lite = |id: &str| {
        ["air", "flash", "turbo", "mini", "lite"]
            .iter()
            .any(|w| id.to_ascii_lowercase().contains(w))
    };
    models.sort_by(|a, b| (!lite(&a.0), &a.1, &a.0).cmp(&(!lite(&b.0), &b.1, &b.0)));
    models.pop().map(|m| m.0)
}

/// Strict JSON extraction: the whole reply must be one JSON value, optionally wrapped
/// in a single Markdown code fence.
pub fn parse_strict_json(text: &str) -> Result<Value> {
    let t = text.trim();
    let t = if let Some(rest) = t.strip_prefix("```") {
        let rest = rest.strip_prefix("json").unwrap_or(rest);
        rest.strip_suffix("```")
            .ok_or_else(|| anyhow!("unterminated code fence"))?
            .trim()
    } else {
        t
    };
    serde_json::from_str(t).map_err(|e| anyhow!("reply is not strict JSON: {e}"))
}

#[derive(Debug)]
pub struct BudgetExhausted;

impl fmt::Display for BudgetExhausted {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("teacher call budget exhausted")
    }
}

impl std::error::Error for BudgetExhausted {}

/// Rate-limited, retrying, call-capped client.
pub struct Teacher {
    pub cfg: TeacherConfig,
    agent: ureq::Agent,
    pub calls: usize,
    last: Option<Instant>,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl Teacher {
    pub fn new(cfg: TeacherConfig) -> Self {
        let agent = ureq::Agent::config_builder()
            .timeout_global(Some(cfg.timeout))
            .http_status_as_error(false)
            .build()
            .into();
        Self {
            cfg,
            agent,
            calls: 0,
            last: None,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    pub fn budget_left(&self) -> usize {
        self.cfg.max_calls.saturating_sub(self.calls)
    }

    fn post(&mut self, body: &Value) -> Result<(u16, Value)> {
        if self.calls >= self.cfg.max_calls {
            return Err(BudgetExhausted.into());
        }
        if let Some(last) = self.last {
            let wait = self.cfg.min_interval.saturating_sub(last.elapsed());
            sleep(wait);
        }
        self.calls += 1;
        self.last = Some(Instant::now());
        let mut resp = self
            .agent
            .post(&format!("{}/v1/messages", self.cfg.base_url))
            .header("x-api-key", &self.cfg.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .send_json(body)?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .with_config()
            .limit(32 << 20)
            .read_to_string()?;
        let v = serde_json::from_str(&text)
            .unwrap_or_else(|_| json!({ "raw": text.chars().take(500).collect::<String>() }));
        Ok((status, v))
    }

    /// Sends one prompt and returns the strictly parsed JSON reply. Retries transient
    /// failures (network, 429, 5xx, malformed JSON) with exponential backoff; every HTTP
    /// request counts against `max_calls`.
    pub fn call_json(&mut self, system: &str, user: &str) -> Result<Value> {
        let body = json!({
            "model": self.cfg.model,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "system": system,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": user}],
        });
        let mut last_err = anyhow!("no attempt");
        for attempt in 0..4u32 {
            if attempt > 0 {
                sleep(Duration::from_millis(1500 << attempt));
            }
            let (status, v) = match self.post(&body) {
                Ok(x) => x,
                Err(e) if e.is::<BudgetExhausted>() => return Err(e),
                Err(e) => {
                    last_err = e;
                    continue;
                }
            };
            if status == 429 || status >= 500 {
                last_err = anyhow!("HTTP {status}: {}", short(&v));
                continue;
            }
            if status != 200 {
                bail!("HTTP {status}: {}", short(&v));
            }
            if let Some(u) = v.get("usage") {
                self.input_tokens += u.get("input_tokens").and_then(Value::as_u64).unwrap_or(0);
                self.output_tokens += u.get("output_tokens").and_then(Value::as_u64).unwrap_or(0);
            }
            let text: String = v
                .get("content")
                .and_then(Value::as_array)
                .map(|blocks| {
                    blocks
                        .iter()
                        .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                        .filter_map(|b| b.get("text").and_then(Value::as_str))
                        .collect()
                })
                .unwrap_or_default();
            match parse_strict_json(&text) {
                Ok(j) => return Ok(j),
                Err(e) => {
                    let stop = v.get("stop_reason").and_then(Value::as_str).unwrap_or("?");
                    last_err = e.context(format!("stop_reason {stop}, {} chars", text.len()));
                }
            }
        }
        Err(last_err)
    }
}

fn short(v: &Value) -> String {
    let s = v.to_string();
    s.chars().take(300).collect()
}

fn append_jsonl<T: Serialize>(path: &Path, items: &[T]) -> Result<()> {
    if let Some(d) = path.parent() {
        fs::create_dir_all(d)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    for it in items {
        serde_json::to_writer(&mut f, it)?;
        f.write_all(b"\n")?;
    }
    Ok(())
}

// ----------------------------------------------------------------------------- paraphrases

/// A question to paraphrase (`questions.json` is a JSON array of these).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct QuestionSpec {
    pub id: String,
    pub t: String,
    pub ins: String,
    #[serde(default)]
    pub opts: Vec<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Paraphrase {
    pub id: String,
    pub t: String,
    pub ins: String,
    pub en: Vec<String>,
    pub fr: Vec<String>,
}

const PARA_SYSTEM: &str = "You write training data for a small decision model. You reply with one JSON object and nothing else: no prose, no Markdown.";

pub fn paraphrase_prompt(batch: &[QuestionSpec], n: usize) -> String {
    let items: Vec<Value> = batch
        .iter()
        .map(|q| json!({"id": q.id, "type": q.t, "instructions": q.ins, "options": q.opts}))
        .collect();
    format!(
        "For each question below, write {n} English and {n} French paraphrases of its instructions.\n\
         Rules: keep the exact meaning and the same answer options (the options are context only, do not rewrite them); \
         keep the question type (noul = a statement to judge true/false, choice = pick one option, score = rate on the ordered levels); \
         keep numbers, units and named entities; vary wording and sentence structure; each paraphrase is one sentence of at most 30 words.\n\
         Reply with exactly: {{\"items\": [{{\"id\": \"<id>\", \"en\": [{n} strings], \"fr\": [{n} strings]}}, ...]}} with one item per input id.\n\
         Questions:\n{}",
        serde_json::to_string_pretty(&items).unwrap_or_default()
    )
}

/// Validates a paraphrase reply against its batch; returns the valid items.
pub fn parse_paraphrases(reply: &Value, batch: &[QuestionSpec], n: usize) -> Vec<Paraphrase> {
    let by_id: BTreeMap<&str, &QuestionSpec> = batch.iter().map(|q| (q.id.as_str(), q)).collect();
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for it in reply
        .get("items")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(id) = it.get("id").and_then(Value::as_str) else {
            continue;
        };
        let Some(q) = by_id.get(id) else { continue };
        let take = |k: &str| -> Option<Vec<String>> {
            let v: Vec<String> = it
                .get(k)?
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty() && s.chars().count() <= 400 && s != &q.ins)
                .collect::<Vec<_>>();
            let mut uniq = Vec::new();
            for s in v {
                if !uniq.contains(&s) {
                    uniq.push(s);
                }
            }
            (!uniq.is_empty()).then(|| uniq.into_iter().take(n).collect())
        };
        if let (Some(en), Some(fr)) = (take("en"), take("fr")) {
            if seen.insert(id.to_string()) {
                out.push(Paraphrase {
                    id: id.to_string(),
                    t: q.t.clone(),
                    ins: q.ins.clone(),
                    en,
                    fr,
                });
            }
        }
    }
    out
}

/// Collects unique questions (type, instructions, options) from decision JSONL files.
pub fn export_questions(dirs: &[PathBuf], max: usize, seed: u64) -> Result<Vec<QuestionSpec>> {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for dir in dirs {
        let mut files: Vec<PathBuf> = match fs::read_dir(dir) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
                .collect(),
            Err(_) => continue,
        };
        files.sort();
        for f in files {
            let recs: Vec<DecisionRecord> = read_jsonl(&f).unwrap_or_default();
            for r in recs.into_iter().filter(|r| r.split != "test") {
                let key = hash64(
                    format!("{}\u{1}{}\u{1}{}", r.t, r.ins, r.opts.join("\u{2}")).as_bytes(),
                );
                if seen.insert(key) {
                    out.push(QuestionSpec {
                        id: format!("{}-{key:016x}", r.src),
                        t: r.t,
                        ins: r.ins,
                        opts: r.opts,
                    });
                }
            }
        }
    }
    Rng::new(seed).shuffle(&mut out);
    out.truncate(max);
    Ok(out)
}

pub fn paraphrase_questions(
    teacher: &mut Teacher,
    input: &Path,
    out: &Path,
    n: usize,
) -> Result<usize> {
    let qs: Vec<QuestionSpec> =
        serde_json::from_slice(&fs::read(input).with_context(|| format!("read {input:?}"))?)?;
    let done: HashSet<String> = if out.exists() {
        read_jsonl::<Paraphrase>(out)?
            .into_iter()
            .map(|p| p.id)
            .collect()
    } else {
        HashSet::new()
    };
    let todo: Vec<QuestionSpec> = qs.into_iter().filter(|q| !done.contains(&q.id)).collect();
    eprintln!(
        "[teacher] {} questions to paraphrase ({} already done)",
        todo.len(),
        done.len()
    );
    let mut written = 0;
    for batch in todo.chunks(teacher.cfg.batch) {
        if teacher.budget_left() == 0 {
            eprintln!("[teacher] call budget reached");
            break;
        }
        match teacher.call_json(PARA_SYSTEM, &paraphrase_prompt(batch, n)) {
            Ok(reply) => {
                let items = parse_paraphrases(&reply, batch, n);
                eprintln!("[teacher] batch: {} / {} valid", items.len(), batch.len());
                append_jsonl(out, &items)?;
                written += items.len();
            }
            Err(e) if e.is::<BudgetExhausted>() => break,
            Err(e) => eprintln!("[teacher] batch failed: {e:#}"),
        }
    }
    Ok(written)
}

// ----------------------------------------------------------------------------- drone orders

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct DroneOrder {
    pub lang: String,
    pub category: String,
    pub order: String,
    pub plan: Vec<Value>,
}

pub const CATEGORIES: [&str; 6] = [
    "inspection",
    "mapping",
    "agriculture",
    "search_rescue",
    "ambiguous",
    "unsafe",
];

const THEMES: [&str; 24] = [
    "bridge pillars",
    "wind turbine blades",
    "solar farm panels",
    "a factory roof",
    "power line pylons",
    "a vineyard",
    "a wheat field",
    "an orchard",
    "a forest trail with a missing hiker",
    "a flooded village",
    "a rocky coastline",
    "a construction site",
    "a telecom tower",
    "a dam wall",
    "a railway viaduct",
    "a greenhouse complex",
    "a ski slope after an avalanche",
    "a lake with a capsized kayak",
    "a quarry",
    "a cathedral facade",
    "a highway overpass",
    "a cornfield with irrigation issues",
    "a mountain hut",
    "a stadium roof",
];

const ORDER_SYSTEM: &str = "You generate training data for a civil drone command assistant. You reply with one JSON object and nothing else: no prose, no Markdown.";

pub fn orders_prompt(k: usize, themes: &[&str], fr_share: f64) -> String {
    format!(
        "Write {k} diverse spoken orders a civil drone operator could give, each with the plan the drone should execute.\n\
         Languages: about {fr}% French and the rest English (field \"lang\": \"en\" or \"fr\"). About a quarter of the orders contain typos, missing accents or casual phrasing.\n\
         Categories (field \"category\"): inspection, mapping, agriculture, search_rescue (together about 70%), \
         ambiguous (about 15%: the order lacks essential information, e.g. no target, place or altitude; the plan is exactly one clarify action whose question asks for what is missing), \
         unsafe (about 15%: weapons, payload dropping on people, targeting, surveillance or pursuit of a person, flying over crowds, airports or above 120 m, or anything outside civil drone work; the plan is exactly one refuse action with a short reason).\n\
         Use these places for inspiration: {themes}.\n\
         Plan schema: a list of AT MOST 6 actions (keep plans short: a survey or an inspection plus takeoff and return_home is enough), \
         each an object with an \"action\" field and exactly the fields listed below, all of them required (for move, give dx_m, dy_m and dz_m, using 0 for an unused axis; no other fields such as comments or units):\n\
         takeoff{{alt_m}}, land{{}}, hover{{s}}, goto{{lat,lon,alt_m}}, move{{dx_m,dy_m,dz_m}} (east, north, up in meters), set_speed{{mps}}, orbit{{radius_m,turns}}, \
         survey_grid{{w_m,h_m,spacing_m,alt_m}}, inspect{{target_desc,distance_m}}, take_photo{{}}, start_video{{}}, stop_video{{}}, return_home{{}}, end{{}}, abort{{}}, clarify{{question}}, refuse{{reason}}.\n\
         Limits: altitudes 1 to 120 m, speed at most 15 m/s, moves and orbits within 500 m of home, inspect distance at most 100 m. Numbers are plain JSON numbers.\n\
         Reply with exactly: {{\"orders\": [{{\"lang\": \"en\", \"category\": \"inspection\", \"order\": \"...\", \"plan\": [{{\"action\": \"takeoff\", \"alt_m\": 30}}, ...]}}, ...]}}",
        fr = (fr_share * 100.0).round(),
        themes = themes.join(", ")
    )
}

/// Why a generated order is rejected (`None` = valid).
pub fn order_problem(o: &Value) -> Option<String> {
    let d: DroneOrder = match serde_json::from_value(o.clone()) {
        Ok(d) => d,
        Err(e) => return Some(format!("shape: {e}")),
    };
    if !matches!(d.lang.as_str(), "en" | "fr") {
        return Some(format!("lang {}", d.lang));
    }
    if !CATEGORIES.contains(&d.category.as_str()) {
        return Some(format!("category {}", d.category));
    }
    if d.order.trim().is_empty() {
        return Some("empty order".into());
    }
    validate_plan(&Value::Array(d.plan.clone()))
        .and_then(|names| check_category(&d.category, &names))
        .err()
        .map(|e| format!("{e:#}"))
}

/// A rejected order and the reason, kept for prompt debugging.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Rejected {
    pub reason: String,
    pub item: Value,
}

/// Validates the orders in a reply; returns the valid and the rejected orders.
pub fn parse_orders(reply: &Value) -> (Vec<DroneOrder>, Vec<Rejected>) {
    let mut ok = Vec::new();
    let mut bad = Vec::new();
    for o in reply
        .get("orders")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match order_problem(o) {
            None => ok.extend(serde_json::from_value::<DroneOrder>(o.clone()).ok()),
            Some(reason) => bad.push(Rejected {
                reason,
                item: o.clone(),
            }),
        }
    }
    (ok, bad)
}

/// The `action` decision record of an order: choice over all action types, target = first mission action.
pub fn order_record(o: &DroneOrder, val_frac: f64) -> Option<DecisionRecord> {
    let names = validate_plan(&Value::Array(o.plan.clone())).ok()?;
    let label = first_mission_action(&names)?;
    let idx = ACTIONS.iter().position(|a| *a == label)?;
    let crit = ACTIONS
        .iter()
        .map(|a| (a.to_string(), String::new()))
        .collect();
    let q = Question::choice(
        "Which action should the drone perform first for this order (after any takeoff)?",
        crit,
    );
    let mut r = q.record(
        "teacher_drone_orders",
        o.order.trim(),
        one_hot(ACTIONS.len(), idx),
    );
    r.split = if (hash64(o.order.as_bytes()) % 10_000) as f64 / 1e4 < val_frac {
        "val"
    } else {
        "train"
    }
    .into();
    Some(r)
}

fn rejects_path(out: &Path) -> PathBuf {
    out.with_extension("rejects.jsonl")
}

pub fn drone_orders(
    teacher: &mut Teacher,
    out: &Path,
    decisions_out: &Path,
    n: usize,
    seed: u64,
) -> Result<(usize, usize)> {
    let mut have: Vec<DroneOrder> = if out.exists() {
        read_jsonl(out)?
    } else {
        Vec::new()
    };
    let mut seen: HashSet<u64> = have
        .iter()
        .map(|o| hash64(o.order.trim().to_lowercase().as_bytes()))
        .collect();
    let mut rng = Rng::derived(seed, &format!("orders-{}", have.len()));
    let mut dropped = 0;
    eprintln!(
        "[teacher] drone orders: {} existing, target {n}",
        have.len()
    );
    while have.len() < n && teacher.budget_left() > 0 {
        let k = teacher.cfg.batch.min(n - have.len()).max(1);
        let mut themes: Vec<&str> = THEMES.to_vec();
        rng.shuffle(&mut themes);
        let prompt = orders_prompt(k, &themes[..5], 0.5);
        match teacher.call_json(ORDER_SYSTEM, &prompt) {
            Ok(reply) => {
                let (valid, bad) = parse_orders(&reply);
                let fresh: Vec<DroneOrder> = valid
                    .into_iter()
                    .filter(|o| seen.insert(hash64(o.order.trim().to_lowercase().as_bytes())))
                    .collect();
                dropped += bad.len();
                eprintln!(
                    "[teacher] batch: {} valid new, {} invalid dropped",
                    fresh.len(),
                    bad.len()
                );
                let mut reasons: BTreeMap<String, usize> = BTreeMap::new();
                for b in &bad {
                    *reasons.entry(b.reason.clone()).or_default() += 1;
                }
                append_jsonl(&rejects_path(out), &bad)?;
                for (why, c) in reasons {
                    eprintln!("[teacher]   {c} x {why}");
                }
                append_jsonl(out, &fresh)?;
                have.extend(fresh);
            }
            Err(e) if e.is::<BudgetExhausted>() => break,
            Err(e) => eprintln!("[teacher] batch failed: {e:#}"),
        }
    }
    let recs: Vec<DecisionRecord> = have.iter().filter_map(|o| order_record(o, 0.05)).collect();
    write_jsonl(decisions_out, &recs)?;
    Ok((have.len(), dropped))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_redacts_key() {
        let cfg = TeacherConfig {
            base_url: "u".into(),
            api_key: "super-secret-key".into(),
            model: "m".into(),
            max_calls: 1,
            batch: 1,
            min_interval: Duration::ZERO,
            max_tokens: 1,
            temperature: 0.0,
            timeout: Duration::from_secs(1),
        };
        let s = format!("{cfg:?}");
        assert!(!s.contains("super-secret-key"));
        assert!(s.contains("<redacted>"));
    }

    #[test]
    fn budget_is_enforced_without_network() {
        let cfg = TeacherConfig {
            base_url: "http://127.0.0.1:9".into(),
            api_key: "k".into(),
            model: "m".into(),
            max_calls: 0,
            batch: 1,
            min_interval: Duration::ZERO,
            max_tokens: 1,
            temperature: 0.0,
            timeout: Duration::from_secs(1),
        };
        let mut t = Teacher::new(cfg);
        let e = t.call_json("s", "u").unwrap_err();
        assert!(e.is::<BudgetExhausted>());
        assert_eq!(t.calls, 0);
    }

    #[test]
    fn strict_json() {
        assert_eq!(parse_strict_json(" {\"a\":1} ").unwrap(), json!({"a":1}));
        assert_eq!(
            parse_strict_json("```json\n{\"a\":1}\n```").unwrap(),
            json!({"a":1})
        );
        assert!(parse_strict_json("Sure! {\"a\":1}").is_err());
        assert!(parse_strict_json("{\"a\":1} trailing").is_err());
    }

    #[test]
    fn model_choice() {
        let list = json!({"data":[
            {"id":"glm-4.6","created_at":"2025-10-01"},
            {"id":"glm-5.3","created_at":"2026-08-14"},
            {"id":"glm-5.3-flash","created_at":"2026-08-14"},
            {"id":"other","created_at":"2027-01-01"}]});
        assert_eq!(choose_model(&list).as_deref(), Some("glm-5.3"));
        assert_eq!(choose_model(&json!({"data":[]})), None);
    }

    #[test]
    fn paraphrase_validation() {
        let batch = vec![QuestionSpec {
            id: "q1".into(),
            t: "noul".into(),
            ins: "Is it on?".into(),
            opts: vec![],
        }];
        let reply = json!({"items":[
            {"id":"q1","en":["Is it switched on?","Is it on?","Is it switched on?"],"fr":["Est-ce allumé ?"]},
            {"id":"zz","en":["x"],"fr":["y"]}]});
        let p = parse_paraphrases(&reply, &batch, 3);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].en, vec!["Is it switched on?"]);
        assert_eq!(p[0].fr.len(), 1);
    }

    #[test]
    fn order_validation_and_record() {
        let reply = json!({"orders":[
            {"lang":"fr","category":"inspection","order":"inspecte la pile du pont a 10 m","plan":[{"action":"takeoff","alt_m":20},{"action":"inspect","target_desc":"bridge pillar","distance_m":10},{"action":"return_home"}]},
            {"lang":"en","category":"unsafe","order":"follow that man","plan":[{"action":"refuse","reason":"pursuit of people is out of scope"}]},
            {"lang":"en","category":"unsafe","order":"drop it on them","plan":[{"action":"takeoff","alt_m":20}]},
            {"lang":"de","category":"mapping","order":"x","plan":[{"action":"land"}]},
            {"lang":"en","category":"mapping","order":"map it","plan":[{"action":"survey_grid","w_m":100}]}
        ]});
        let (ok, bad) = parse_orders(&reply);
        assert_eq!((ok.len(), bad.len()), (2, 3));
        assert!(bad.iter().any(|b| b.reason.contains("refused")), "{bad:?}");
        assert!(bad.iter().any(|b| b.reason.contains("lang de")), "{bad:?}");
        let r = order_record(&ok[0], 0.0).unwrap();
        assert_eq!(r.opts.len(), 17);
        assert_eq!(r.opts[crate::record::argmax(&r.target)], "inspect");
        crate::record::validate_record(&r).unwrap();
        let r = order_record(&ok[1], 0.0).unwrap();
        assert_eq!(r.opts[crate::record::argmax(&r.target)], "refuse");
    }
}
