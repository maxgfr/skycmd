//! `fetch-commands`: natural-language → command data for the later command decoder.
//!
//! Output: one JSONL per source under `data/commands/`, each line
//! `{src, lang, user, tools?, target_json?, intent?, split}` where `tools` is a JSON array of
//! function schemas `{name, description, parameters}`, `target_json` the compact JSON text of
//! the calls the assistant makes for `user` (`[{"name", "arguments"}]`, `[]` for a plain text
//! reply) and `intent` a MASSIVE intent label.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::decisions::SMALL;
use crate::hf::{for_each_parquet_row, Hf};
use crate::record::write_jsonl;
use crate::rng::{hash64, Rng};

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CommandRecord {
    pub src: String,
    pub lang: String,
    pub user: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_json: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    pub split: String,
}

#[derive(Clone, Debug)]
pub struct CommandOptions {
    pub out: PathBuf,
    pub seed: u64,
    pub val_frac: f64,
    /// Cap on glaive conversations whose system prompt offers no function at all.
    pub max_no_tools: usize,
    pub only: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct CommandReport {
    pub src: String,
    pub raw: usize,
    /// Rows not converted (unparseable, no user turn, or over a cap).
    pub skipped: usize,
    pub duplicates: usize,
    pub written: usize,
    pub with_calls: usize,
    pub secs: f64,
}

/// Normalizes a tool schema: OpenAI style `{"type":"function","function":{..}}` → inner object.
pub fn normalize_tool(t: &Value) -> Value {
    match t.get("function") {
        Some(f) if f.is_object() => f.clone(),
        _ => t.clone(),
    }
}

/// Calls inside `<tool_call> … </tool_call>` blocks (Hermes format).
pub fn hermes_calls(text: &str) -> Option<Vec<Value>> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(i) = rest.find("<tool_call>") {
        let after = &rest[i + "<tool_call>".len()..];
        let end = after.find("</tool_call>")?;
        let call: Value = serde_json::from_str(after[..end].trim()).ok()?;
        out.push(json!({"name": call.get("name")?.as_str()?, "arguments": call.get("arguments").cloned().unwrap_or(json!({}))}));
        rest = &after[end + "</tool_call>".len()..];
    }
    Some(out)
}

fn split_of(key: &str, seed: u64, val_frac: f64) -> String {
    if (hash64(format!("{seed}/{key}").as_bytes()) % 1_000_000) as f64 / 1e6 < val_frac {
        "val".into()
    } else {
        "train".into()
    }
}

/// One Hermes conversation → the first human turn and the calls of the reply to it, or the
/// reason it is skipped.
pub fn hermes_record(conversations: &Value, tools: &Value) -> Result<CommandRecord, &'static str> {
    let turns = conversations.as_array().ok_or("no conversation")?;
    let hi = turns
        .iter()
        .position(|t| t.get("from").and_then(Value::as_str) == Some("human"))
        .ok_or("no human turn")?;
    let user = turns[hi]
        .get("value")
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_string();
    if user.is_empty() {
        return Err("empty human turn");
    }
    let reply = turns
        .get(hi + 1)
        .filter(|t| t.get("from").and_then(Value::as_str) == Some("gpt"))
        .and_then(|t| t.get("value")?.as_str())
        .ok_or("no reply")?;
    let calls = hermes_calls(reply).ok_or("unparseable tool call")?;
    let tools: Value = match tools {
        Value::String(s) => serde_json::from_str(s).map_err(|_| "unparseable tools")?,
        v => v.clone(),
    };
    let tools: Vec<Value> = tools
        .as_array()
        .ok_or("tools not a list")?
        .iter()
        .map(normalize_tool)
        .collect();
    Ok(CommandRecord {
        src: "hermes".into(),
        lang: "en".into(),
        user,
        tools: Some(Value::Array(tools)),
        target_json: Some(Value::Array(calls).to_string()),
        intent: None,
        split: String::new(),
    })
}

/// The function schemas listed in a glaive system prompt (consecutive JSON objects).
pub fn glaive_tools(system: &str) -> Vec<Value> {
    let Some(i) = system.find('{') else {
        return Vec::new();
    };
    serde_json::Deserializer::from_str(&system[i..])
        .into_iter::<Value>()
        .map_while(|v| v.ok())
        .filter(|v| v.get("name").is_some())
        .collect()
}

/// Parses a glaive `<functioncall> {"name": "f", "arguments": '{...}'}` (the arguments are a
/// single-quoted JSON string, so the whole is not JSON).
pub fn glaive_call(text: &str) -> Option<Value> {
    let body = text.trim().strip_prefix("<functioncall>")?.trim();
    if let Ok(v) = serde_json::from_str::<Value>(body) {
        let args = match v.get("arguments")? {
            Value::String(s) => serde_json::from_str(s).ok()?,
            a => a.clone(),
        };
        return Some(json!({"name": v.get("name")?.as_str()?, "arguments": args}));
    }
    let name_at = body.find("\"name\"")?;
    let after = &body[name_at + 6..];
    let q1 = after.find('"')?;
    let q2 = after[q1 + 1..].find('"')? + q1 + 1;
    let name = &after[q1 + 1..q2];
    let a = body.find("\"arguments\"")?;
    let rest = &body[a + "\"arguments\"".len()..];
    let open = rest.find('\'')?;
    let close = rest.rfind('\'')?;
    if close <= open {
        return None;
    }
    let args: Value = serde_json::from_str(&rest[open + 1..close]).ok()?;
    Some(json!({"name": name, "arguments": args}))
}

/// One glaive v2 row → the first user turn and the assistant's call (or `[]`).
pub fn glaive_record(system: &str, chat: &str) -> Option<CommandRecord> {
    let tools = glaive_tools(system);
    let u = chat.find("USER:")? + "USER:".len();
    let rest = &chat[u..];
    let a = rest.find("ASSISTANT:")?;
    let user = rest[..a].trim().to_string();
    let reply = &rest[a + "ASSISTANT:".len()..];
    let reply = reply.split("<|endoftext|>").next()?.trim();
    let calls = if reply.starts_with("<functioncall>") {
        vec![glaive_call(reply)?]
    } else {
        Vec::new()
    };
    (!user.is_empty() && !reply.is_empty()).then(|| CommandRecord {
        src: "glaive".into(),
        lang: "en".into(),
        user,
        tools: Some(Value::Array(tools)),
        target_json: Some(Value::Array(calls).to_string()),
        intent: None,
        split: String::new(),
    })
}

fn finish(
    src: &str,
    mut recs: Vec<CommandRecord>,
    raw: usize,
    opts: &CommandOptions,
    t0: Instant,
) -> Result<CommandReport> {
    let mut seen = HashSet::new();
    let parsed = recs.len();
    recs.retain(|r| {
        seen.insert(hash64(
            format!("{}\u{1}{}\u{1}{:?}", r.lang, r.user, r.target_json).as_bytes(),
        ))
    });
    for r in recs.iter_mut().filter(|r| r.split.is_empty()) {
        r.split = split_of(&r.user, opts.seed, opts.val_frac);
    }
    Rng::derived(opts.seed, src).shuffle(&mut recs);
    let with_calls = recs
        .iter()
        .filter(|r| r.target_json.as_deref().is_some_and(|t| t != "[]"))
        .count();
    let written = write_jsonl(&opts.out.join(format!("{src}.jsonl")), &recs)?;
    Ok(CommandReport {
        src: src.into(),
        raw,
        skipped: raw.saturating_sub(parsed),
        duplicates: parsed - recs.len(),
        written,
        with_calls,
        secs: t0.elapsed().as_secs_f64(),
    })
}

fn fetch_hermes(hf: &Hf, opts: &CommandOptions) -> Result<CommandReport> {
    let t0 = Instant::now();
    let repo = "NousResearch/hermes-function-calling-v1";
    let (mut raw, mut recs) = (0, Vec::new());
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    for config in [
        "func_calling_singleturn",
        "func_calling",
        "glaive_func_calling",
    ] {
        for p in hf.parquet_split(repo, config, "train", SMALL)? {
            for_each_parquet_row(&p, &["conversations", "tools"], |row| {
                raw += 1;
                match hermes_record(&row["conversations"], &row["tools"]) {
                    Ok(r) => recs.push(r),
                    Err(why) => *skipped.entry(format!("{config}: {why}")).or_default() += 1,
                }
                Ok(())
            })?;
        }
    }
    eprintln!("  hermes skipped: {skipped:?}");
    finish("hermes", recs, raw, opts, t0)
}

fn fetch_glaive(hf: &Hf, opts: &CommandOptions) -> Result<CommandReport> {
    let t0 = Instant::now();
    let (mut raw, mut with_tools, mut no_tools) = (0, Vec::new(), Vec::new());
    for p in hf.parquet_split(
        "glaiveai/glaive-function-calling-v2",
        "default",
        "train",
        SMALL,
    )? {
        for_each_parquet_row(&p, &["system", "chat"], |row| {
            raw += 1;
            let system = row.get("system").and_then(Value::as_str).unwrap_or("");
            let chat = row.get("chat").and_then(Value::as_str).unwrap_or("");
            if let Some(r) = glaive_record(system, chat) {
                if r.tools
                    .as_ref()
                    .and_then(Value::as_array)
                    .is_some_and(|a| a.is_empty())
                {
                    no_tools.push(r);
                } else {
                    with_tools.push(r);
                }
            }
            Ok(())
        })?;
    }
    eprintln!(
        "  glaive: {} conversations with functions, {} without (capped to {})",
        with_tools.len(),
        no_tools.len(),
        opts.max_no_tools
    );
    let mut rng = Rng::derived(opts.seed, "glaive-no-tools");
    rng.shuffle(&mut no_tools);
    no_tools.truncate(opts.max_no_tools);
    with_tools.extend(no_tools);
    finish("glaive", with_tools, raw, opts, t0)
}

fn fetch_massive(hf: &Hf, opts: &CommandOptions) -> Result<CommandReport> {
    let t0 = Instant::now();
    let (mut raw, mut recs) = (0, Vec::new());
    for lang in ["en", "fr"] {
        for (split, out_split) in [("train", "train"), ("validation", "val"), ("test", "test")] {
            for p in hf.parquet_split("mteb/amazon_massive_intent", lang, split, SMALL)? {
                for_each_parquet_row(&p, &["text", "label_text"], |row| {
                    raw += 1;
                    let text = row
                        .get("text")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .trim()
                        .to_string();
                    let intent = row
                        .get("label_text")
                        .and_then(Value::as_str)
                        .ok_or_else(|| anyhow!("label_text"))?;
                    if !text.is_empty() {
                        recs.push(CommandRecord {
                            src: "massive".into(),
                            lang: lang.into(),
                            user: text,
                            tools: None,
                            target_json: None,
                            intent: Some(intent.to_string()),
                            split: out_split.into(),
                        });
                    }
                    Ok(())
                })?;
            }
        }
    }
    finish("massive", recs, raw, opts, t0)
}

/// Downloads and converts every (selected) command source.
pub fn fetch_commands(hf: &Hf, opts: &CommandOptions) -> Result<Vec<CommandReport>> {
    type Fetch = fn(&Hf, &CommandOptions) -> Result<CommandReport>;
    let all: [(&str, Fetch); 3] = [
        ("hermes", fetch_hermes),
        ("glaive", fetch_glaive),
        ("massive", fetch_massive),
    ];
    std::fs::create_dir_all(&opts.out)?;
    let mut reports = Vec::new();
    for (name, f) in all {
        if !opts.only.is_empty() && !opts.only.iter().any(|o| o == name) {
            continue;
        }
        eprintln!("[commands] {name} ...");
        reports.push(f(hf, opts)?);
    }
    Ok(reports)
}

/// Per-file summary of `data/commands` for STATS.md: records, splits, with tools, with calls, intents.
pub fn summarize(recs: &[CommandRecord]) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    m.insert("records", recs.len());
    m.insert("train", recs.iter().filter(|r| r.split == "train").count());
    m.insert("val", recs.iter().filter(|r| r.split == "val").count());
    m.insert("test", recs.iter().filter(|r| r.split == "test").count());
    m.insert("en", recs.iter().filter(|r| r.lang == "en").count());
    m.insert("fr", recs.iter().filter(|r| r.lang == "fr").count());
    m.insert(
        "with_calls",
        recs.iter()
            .filter(|r| r.target_json.as_deref().is_some_and(|t| t != "[]"))
            .count(),
    );
    m.insert(
        "intents",
        recs.iter()
            .filter_map(|r| r.intent.as_deref())
            .collect::<HashSet<_>>()
            .len(),
    );
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hermes_parsing() {
        let conv = json!([
            {"from":"system","value":"sys"},
            {"from":"human","value":" Turn on the lights "},
            {"from":"gpt","value":"<tool_call>\n{\"name\": \"lights_on\", \"arguments\": {\"room\": \"hall\"}}\n</tool_call>\n<tool_call>\n{\"name\": \"x\", \"arguments\": {}}\n</tool_call>"}
        ]);
        let tools = json!("[{\"type\": \"function\", \"function\": {\"name\": \"lights_on\", \"parameters\": {}}}]");
        let r = hermes_record(&conv, &tools).unwrap();
        assert_eq!(
            hermes_record(&json!([{"from":"human","value":"x"}]), &tools),
            Err("no reply")
        );
        assert_eq!(r.user, "Turn on the lights");
        assert_eq!(r.tools.unwrap()[0]["name"], "lights_on");
        let calls: Value = serde_json::from_str(r.target_json.as_deref().unwrap()).unwrap();
        assert_eq!(calls[0]["arguments"]["room"], "hall");
        assert_eq!(calls.as_array().unwrap().len(), 2);
        assert!(hermes_calls("<tool_call>{bad").is_none());
        assert_eq!(hermes_calls("plain answer").unwrap().len(), 0);
    }

    #[test]
    fn glaive_parsing() {
        let system = "SYSTEM: You are a helpful assistant with access to the following functions. Use them if required -\n{\n \"name\": \"get_news\",\n \"parameters\": {}\n}\n\n{\n \"name\": \"b\"\n}\n";
        assert_eq!(glaive_tools(system).len(), 2);
        let chat = "USER: News for France?\n\n\nASSISTANT: <functioncall> {\"name\": \"get_news\", \"arguments\": '{\"country\": \"France\"}'} <|endoftext|>\n\n\nFUNCTION RESPONSE: {}";
        let r = glaive_record(system, chat).unwrap();
        assert_eq!(r.user, "News for France?");
        assert_eq!(
            r.target_json.as_deref(),
            Some(r#"[{"name":"get_news","arguments":{"country":"France"}}]"#)
        );
        let calls: Value = serde_json::from_str(r.target_json.as_deref().unwrap()).unwrap();
        assert_eq!(calls[0]["name"], "get_news");
        assert_eq!(calls[0]["arguments"]["country"], "France");
        let r = glaive_record(
            "SYSTEM: You are a helpful assistant, with no access to external functions.",
            "USER: hi\n\n\nASSISTANT: Hello! <|endoftext|>",
        )
        .unwrap();
        assert_eq!(r.target_json.as_deref(), Some("[]"));
        assert_eq!(r.tools, Some(json!([])));
        assert!(glaive_call("<functioncall> {\"name\": \"f\", \"arguments\": '{bad'}").is_none());
    }

    #[test]
    fn records_serialize_without_absent_fields() {
        let r = CommandRecord {
            src: "massive".into(),
            lang: "fr".into(),
            user: "réveille-moi".into(),
            tools: None,
            target_json: None,
            intent: Some("alarm_set".into()),
            split: "train".into(),
        };
        let s = serde_json::to_string(&r).unwrap();
        assert_eq!(
            s,
            r#"{"src":"massive","lang":"fr","user":"réveille-moi","intent":"alarm_set","split":"train"}"#
        );
        assert_eq!(summarize(&[r])["intents"], 1);
    }
}
