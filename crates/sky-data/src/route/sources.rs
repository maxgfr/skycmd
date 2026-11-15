//! Every source of section 4.1 and the test sources of 7.1, read into one table of tools and one
//! of requests. Each reader takes the files `sky-data fetch-model` downloaded at the pinned
//! commits of `scripts/route/pins.json` (`<data>/src/<name>/`), or the decision and command files
//! the repository already holds (When2Call, Banking77, MASSIVE).
//!
//! A request of a tool-calling source is the user turn the first call answers; its gold tool is
//! the first call's, which must be one of the listed tools; the other tools that request calls
//! are never its distractors. A request without a call is not a `route/tools` request (only
//! When2Call says which of those no tool can serve).

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use super::{
    add, bump, mask_keys, norm_text, Counts, Request, Tool, INTENTS, TEST, TOOLS, WHEN2CALL,
};
use crate::commands::{glaive_record, hermes_record, CommandRecord};
use crate::nlu::{when2call_item, when2call_label, WHEN2CALL_OPTS};
use crate::record::{argmax, read_jsonl, DecisionRecord};

/// The inputs of the `requests` step.
#[derive(Clone, Debug)]
pub struct SourcePaths {
    /// `<data>/src`: one folder per source of the pins file.
    pub src: PathBuf,
    /// `data/decisions/when2call.jsonl` (its train / val / test partition is v1's).
    pub when2call: PathBuf,
    /// `data/decisions/banking77.jsonl`.
    pub banking77: PathBuf,
    /// `data/commands/massive.jsonl`.
    pub massive: PathBuf,
    /// The teacher's intent descriptions (`describe-intent`), `<source>:<intent>` → text.
    pub descriptions: BTreeMap<String, String>,
}

/// The table of tools, the requests, and the counts of what was read and dropped.
#[derive(Default)]
pub struct Collector {
    pub tools: BTreeMap<String, Tool>,
    pub requests: Vec<Request>,
    pub counts: Counts,
    /// Tools whose schema holds a key-like string (4.6, item 5).
    pub schema_keys: BTreeSet<String>,
    /// MetaTool's "similar choices" groups (tool name → the other tools of its group).
    pub similar: BTreeMap<String, Vec<String>>,
    /// The ids of MetaTool's described tools (`plugin_des.json`).
    pub metatool: Vec<String>,
}

/// A tool as a source lists it: name, description, schema text.
pub struct Listed {
    pub name: String,
    pub desc: String,
    pub schema: String,
}

impl Listed {
    /// From a JSON schema `{name, description, ...}`.
    pub fn from_schema(v: &Value) -> Option<Self> {
        let v = crate::commands::normalize_tool(v);
        let name = v.get("name")?.as_str()?.trim().to_string();
        if name.is_empty() {
            return None;
        }
        Some(Self {
            name,
            desc: v
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            schema: v.to_string(),
        })
    }
}

impl Collector {
    /// Registers a tool; returns its id.
    pub fn tool(&mut self, name: &str, desc: &str, schema: &str) -> String {
        let t = Tool::new(name, desc);
        let id = t.id.clone();
        if mask_keys(schema).1 > 0 || mask_keys(desc).1 > 0 {
            self.schema_keys.insert(id.clone());
        }
        self.tools.entry(id.clone()).or_insert(t);
        id
    }

    /// A request text with its key-like strings masked (the number masked is counted).
    pub fn text(&mut self, s: &str) -> String {
        let (m, n) = mask_keys(s.trim());
        add(&mut self.counts, "keys masked in requests", n);
        m
    }

    /// A request of a tool-calling source: the listed tools and the calls' names. Returns
    /// whether it was kept.
    #[allow(clippy::too_many_arguments)]
    pub fn conversation(
        &mut self,
        source: &str,
        row: usize,
        group: &str,
        split: Option<&str>,
        user: &str,
        listed: &[Listed],
        calls: &[String],
    ) -> bool {
        let user = self.text(user);
        if user.is_empty() {
            bump(&mut self.counts, format!("{source}: empty request"));
            return false;
        }
        let mut by_name: HashMap<&str, String> = HashMap::new();
        let mut own = Vec::new();
        for l in listed {
            if by_name.contains_key(l.name.as_str()) {
                continue;
            }
            let id = self.tool(&l.name, &l.desc, &l.schema);
            by_name.insert(l.name.as_str(), id.clone());
            own.push(id);
        }
        let Some(first) = calls.first() else {
            bump(&mut self.counts, format!("{source}: no call"));
            return false;
        };
        let Some(gold) = by_name.get(first.trim()).cloned() else {
            bump(&mut self.counts, format!("{source}: call not listed"));
            return false;
        };
        let mut exclude: Vec<String> = calls[1..]
            .iter()
            .filter_map(|c| by_name.get(c.trim()).cloned())
            .filter(|id| *id != gold)
            .collect();
        exclude.sort();
        exclude.dedup();
        self.requests.push(Request {
            id: format!("{source}:{row}"),
            source: source.into(),
            group: group.into(),
            split: split.map(str::to_string),
            lang: "en".into(),
            user,
            tools: own,
            gold: Some(gold),
            exclude,
            kind: "call".into(),
        });
        bump(&mut self.counts, format!("{source}: kept"));
        true
    }
}

fn read_json_file(path: &Path) -> Result<Value> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

/// The non-blank lines of a JSONL file as JSON values.
fn json_lines(path: &Path) -> Result<Vec<Value>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .enumerate()
        .map(|(i, l)| {
            serde_json::from_str(l).with_context(|| format!("{}:{}", path.display(), i + 1))
        })
        .collect()
}

fn calls_of_target(r: &CommandRecord) -> Vec<String> {
    r.target_json
        .as_deref()
        .and_then(|t| serde_json::from_str::<Value>(t).ok())
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|c| c.get("name").and_then(Value::as_str).map(str::to_string))
        .collect()
}

fn listed_of(tools: Option<&Value>) -> Vec<Listed> {
    tools
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Listed::from_schema).collect())
        .unwrap_or_default()
}

/// glaive-function-calling-v2: the first user turn and the call that answers it.
pub fn glaive(c: &mut Collector, dir: &Path) -> Result<()> {
    let rows = read_json_file(&dir.join("glaive-function-calling-v2.json"))?;
    let rows = rows
        .as_array()
        .ok_or_else(|| anyhow!("glaive: not a list"))?;
    for (i, row) in rows.iter().enumerate() {
        bump(&mut c.counts, "glaive: rows");
        let system = row.get("system").and_then(Value::as_str).unwrap_or("");
        let chat = row.get("chat").and_then(Value::as_str).unwrap_or("");
        let Some(r) = glaive_record(system, chat) else {
            bump(&mut c.counts, "glaive: unparsed");
            continue;
        };
        let listed = listed_of(r.tools.as_ref());
        let calls = calls_of_target(&r);
        c.conversation("glaive", i, TOOLS, None, &r.user, &listed, &calls);
    }
    Ok(())
}

/// The three function-calling files of hermes-function-calling-v1, as `fetch-commands` reads
/// them; a request whose normalised text is a glaive request's is dropped (4.1).
pub fn hermes(c: &mut Collector, dir: &Path) -> Result<()> {
    let glaive_texts: HashSet<String> = c
        .requests
        .iter()
        .filter(|r| r.source == "glaive")
        .map(|r| norm_text(&r.user))
        .collect();
    let mut row = 0;
    for file in [
        "func-calling-singleturn.json",
        "func-calling.json",
        "glaive-function-calling-5k.json",
    ] {
        let rows = read_json_file(&dir.join(file))?;
        for x in rows
            .as_array()
            .ok_or_else(|| anyhow!("hermes {file}: not a list"))?
        {
            bump(&mut c.counts, "hermes: rows");
            row += 1;
            let conv = x.get("conversations").cloned().unwrap_or(Value::Null);
            let tools = x.get("tools").cloned().unwrap_or(Value::Null);
            let r = match hermes_record(&conv, &tools) {
                Ok(r) => r,
                Err(why) => {
                    bump(&mut c.counts, format!("hermes: {why}"));
                    continue;
                }
            };
            if glaive_texts.contains(&norm_text(&r.user)) {
                bump(&mut c.counts, "hermes: same request as glaive");
                continue;
            }
            let listed = listed_of(r.tools.as_ref());
            let calls = calls_of_target(&r);
            c.conversation("hermes", row - 1, TOOLS, None, &r.user, &listed, &calls);
        }
    }
    Ok(())
}

/// The names of the calls of a Python-like list `[Name(a=1, b="x, y"), Other()]` (ToolACE,
/// API-Bank); `None` when it is not one.
pub fn bracket_calls(s: &str) -> Option<Vec<String>> {
    let s = s.trim();
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    let chars: Vec<char> = inner.chars().collect();
    let mut names = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        while i < chars.len() && (chars[i] == ',' || chars[i].is_whitespace()) {
            i += 1;
        }
        if i >= chars.len() {
            break;
        }
        let start = i;
        while i < chars.len() && chars[i] != '(' {
            if matches!(chars[i], ')' | '[' | ']' | '"' | '\'') {
                return None;
            }
            i += 1;
        }
        if i >= chars.len() {
            return None;
        }
        let name: String = chars[start..i]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();
        if name.is_empty() {
            return None;
        }
        // to the matching parenthesis, quotes and brackets respected
        let mut depth = 0i32;
        let mut quote: Option<char> = None;
        loop {
            let ch = *chars.get(i)?;
            match quote {
                Some(q) => {
                    if ch == '\\' {
                        i += 1;
                    } else if ch == q {
                        quote = None;
                    }
                }
                None => match ch {
                    '"' | '\'' => quote = Some(ch),
                    '(' | '[' | '{' => depth += 1,
                    ')' | ']' | '}' => {
                        depth -= 1;
                        if depth == 0 {
                            i += 1;
                            break;
                        }
                    }
                    _ => {}
                },
            }
            i += 1;
        }
        names.push(name);
    }
    (!names.is_empty()).then_some(names)
}

/// ToolACE: the tool list of the system prompt, the first user turn and the first assistant
/// turn when it is a call list.
pub fn toolace(c: &mut Collector, dir: &Path) -> Result<()> {
    let rows = read_json_file(&dir.join("data.json"))?;
    for (i, x) in rows
        .as_array()
        .ok_or_else(|| anyhow!("toolace: not a list"))?
        .iter()
        .enumerate()
    {
        bump(&mut c.counts, "toolace: rows");
        let system = x.get("system").and_then(Value::as_str).unwrap_or("");
        let listed: Vec<Listed> = match system.find("[{") {
            Some(at) => serde_json::Deserializer::from_str(&system[at..])
                .into_iter::<Value>()
                .next()
                .and_then(|v| v.ok())
                .map(|v| listed_of(Some(&v)))
                .unwrap_or_default(),
            None => Vec::new(),
        };
        if listed.is_empty() {
            bump(&mut c.counts, "toolace: no tool list");
            continue;
        }
        let conv = x
            .get("conversations")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let turn = |k: usize, who: &str| {
            conv.get(k)
                .filter(|t| t.get("from").and_then(Value::as_str) == Some(who))
                .and_then(|t| t.get("value").and_then(Value::as_str))
                .map(str::to_string)
        };
        let (Some(user), Some(reply)) = (turn(0, "user"), turn(1, "assistant")) else {
            bump(&mut c.counts, "toolace: no user turn and reply");
            continue;
        };
        let calls = bracket_calls(&reply).unwrap_or_default();
        c.conversation("toolace", i, TOOLS, None, &user, &listed, &calls);
    }
    Ok(())
}

/// Seal-Tools: its training file only; the tools are those of `tool.jsonl` and an item lists
/// none of its own.
pub fn seal_tools(c: &mut Collector, dir: &Path) -> Result<()> {
    let base = dir.join("Seal-Tools_Dataset");
    let mut tools: HashMap<String, Listed> = HashMap::new();
    for t in json_lines(&base.join("tool.jsonl"))? {
        let Some(name) = t.get("api_name").and_then(Value::as_str) else {
            continue;
        };
        tools.entry(name.to_string()).or_insert(Listed {
            name: name.to_string(),
            desc: t
                .get("api_description")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string(),
            schema: t.to_string(),
        });
    }
    for (i, x) in json_lines(&base.join("train.jsonl"))?.iter().enumerate() {
        bump(&mut c.counts, "seal-tools: rows");
        let user = x.get("query").and_then(Value::as_str).unwrap_or("");
        let calls: Vec<String> = x
            .get("calling")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|k| k.get("api").and_then(Value::as_str).map(str::to_string))
                    .collect()
            })
            .unwrap_or_default();
        // the called tools are listed so that the first can be the gold and the others excluded
        let listed: Vec<Listed> = calls
            .iter()
            .filter_map(|n| tools.get(n))
            .map(|l| Listed {
                name: l.name.clone(),
                desc: l.desc.clone(),
                schema: l.schema.clone(),
            })
            .collect();
        let n0 = c.requests.len();
        if c.conversation("seal-tools", i, TOOLS, None, user, &listed, &calls) {
            // Seal-Tools lists no tools with its items: none of its own
            c.requests[n0].tools.clear();
        }
    }
    Ok(())
}

/// API-Bank, training data of levels 1 and 2: the `apiCode` entries of the input are the
/// listed tools; the request is the last user turn; the call is the output's. Calls of the
/// retrieval tool `ToolSearcher` are not routing and are dropped.
pub fn api_bank(c: &mut Collector, dir: &Path) -> Result<()> {
    let mut row = 0;
    for file in ["lv1-api-train.json", "lv2-api-train.json"] {
        let rows = read_json_file(&dir.join("training-data").join(file))?;
        for x in rows
            .as_array()
            .ok_or_else(|| anyhow!("api-bank {file}: not a list"))?
        {
            bump(&mut c.counts, "api-bank: rows");
            row += 1;
            let input = x.get("input").and_then(Value::as_str).unwrap_or("");
            let output = x.get("output").and_then(Value::as_str).unwrap_or("");
            let mut listed = Vec::new();
            let mut turns: Vec<(String, String)> = Vec::new();
            for line in input.lines() {
                if line.starts_with("{\"apiCode\"") {
                    if let Ok(v) = serde_json::from_str::<Value>(line) {
                        if let Some(name) = v.get("apiCode").and_then(Value::as_str) {
                            listed.push(Listed {
                                name: name.to_string(),
                                desc: v
                                    .get("description")
                                    .and_then(Value::as_str)
                                    .unwrap_or("")
                                    .to_string(),
                                schema: line.to_string(),
                            });
                        }
                    }
                    continue;
                }
                let role = ["User:", "AI:", "API-Request:", "Generate API Request:"]
                    .iter()
                    .find(|r| line.starts_with(**r));
                match role {
                    Some(r) => turns.push((r.to_string(), line[r.len()..].trim().to_string())),
                    None => {
                        if let Some(t) = turns.last_mut() {
                            t.1.push('\n');
                            t.1.push_str(line);
                        }
                    }
                }
            }
            let Some((_, user)) = turns.iter().rev().find(|(r, _)| r == "User:") else {
                bump(&mut c.counts, "api-bank: no user turn");
                continue;
            };
            let calls = output
                .trim()
                .strip_prefix("API-Request:")
                .and_then(bracket_calls)
                .unwrap_or_default();
            if calls.first().map(String::as_str) == Some("ToolSearcher") {
                bump(&mut c.counts, "api-bank: ToolSearcher call");
                continue;
            }
            let user = user.clone();
            c.conversation("api-bank", row - 1, TOOLS, None, &user, &listed, &calls);
        }
    }
    Ok(())
}

/// The names inside `<TOOLCALL>[...]</TOOLCALL>` (When2Call), or of a bare JSON call or list.
pub fn toolcall_names(text: &str) -> Vec<String> {
    let body = match text.find("<TOOLCALL>") {
        Some(i) => {
            let rest = &text[i + "<TOOLCALL>".len()..];
            rest.split("</TOOLCALL>").next().unwrap_or(rest)
        }
        None => text,
    };
    let Ok(v) = serde_json::from_str::<Value>(body.trim()) else {
        return Vec::new();
    };
    let items = match v {
        Value::Array(a) => a,
        o @ Value::Object(_) => vec![o],
        _ => Vec::new(),
    };
    items
        .iter()
        .filter_map(|c| c.get("name").and_then(Value::as_str).map(str::to_string))
        .collect()
}

fn when2call_tools(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|t| t.as_str().map_or_else(|| t.to_string(), str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn first_user(messages: &Value) -> Option<String> {
    messages
        .as_array()?
        .iter()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|m| m.get("content")?.as_str().map(str::to_string))
}

/// One When2Call item as a request: tool_call → the called tool; request_for_info → the only
/// listed tool, else a tool the item's data call (`hint`); cannot_answer and direct_answer →
/// none. `None` when the tool cannot be told.
#[allow(clippy::too_many_arguments)]
fn when2call_request(
    c: &mut Collector,
    source: &str,
    row: usize,
    group: &str,
    split: &str,
    question: &str,
    tool_texts: &[String],
    label: usize,
    called: &[String],
    hint: &[String],
) -> Option<Request> {
    let listed: Vec<Listed> = tool_texts
        .iter()
        .filter_map(|t| serde_json::from_str::<Value>(t).ok())
        .filter_map(|v| Listed::from_schema(&v))
        .collect();
    let mut by_name: HashMap<String, String> = HashMap::new();
    let mut own = Vec::new();
    for l in &listed {
        if by_name.contains_key(&l.name) {
            continue;
        }
        let id = c.tool(&l.name, &l.desc, &l.schema);
        by_name.insert(l.name.clone(), id.clone());
        own.push(id);
    }
    let kind = WHEN2CALL_OPTS[label].0;
    let pick = |names: &[String]| names.iter().find_map(|n| by_name.get(n.trim()).cloned());
    let gold = match label {
        0 => Some(pick(called)?),
        1 => {
            if own.len() == 1 {
                Some(own[0].clone())
            } else {
                Some(pick(hint)?)
            }
        }
        _ => None,
    };
    let exclude = match label {
        0 => {
            let mut e: Vec<String> = called[1..]
                .iter()
                .filter_map(|n| by_name.get(n.trim()).cloned())
                .filter(|id| Some(id) != gold.as_ref())
                .collect();
            e.sort();
            e.dedup();
            e
        }
        _ => Vec::new(),
    };
    let user = c.text(question);
    Some(Request {
        id: format!("{source}:{row}"),
        source: source.into(),
        group: group.into(),
        split: Some(split.into()),
        lang: "en".into(),
        user,
        tools: own,
        gold,
        exclude,
        kind: kind.into(),
    })
}

/// A When2Call decision record's state and label.
type W2cKey = (String, usize);

/// The `(state, label)` of each When2Call decision record → its split, and the test records in
/// file order (the first 2,000 are v1's).
fn when2call_partition(path: &Path) -> Result<(HashMap<W2cKey, String>, Vec<W2cKey>)> {
    let recs: Vec<DecisionRecord> = read_jsonl(path)?;
    let mut split = HashMap::new();
    let mut test = Vec::new();
    for r in recs.into_iter().filter(|r| r.src == "when2call") {
        let key = (r.state.clone(), argmax(&r.target));
        if r.split == "test" {
            test.push(key);
        } else {
            split.entry(key).or_insert(r.split);
        }
    }
    Ok((split, test))
}

/// When2Call: the train and val requests of `data/decisions/when2call.jsonl` (each raw row is
/// rebuilt with the code that wrote that file and found there, which gives its split), and the
/// first 2,000 test records of the same file, reformatted (7.1).
pub fn when2call(c: &mut Collector, dir: &Path, decisions: &Path) -> Result<()> {
    let (split_of, test_order) = when2call_partition(decisions)?;
    let mut used: HashSet<(String, usize)> = HashSet::new();
    let mut row = 0;
    // train_sft: the response of the only assistant turn; train_pref: the chosen response, and
    // the rejected one as the hint of a request for information
    for (file, pref) in [
        ("when2call_train_sft.jsonl", false),
        ("when2call_train_pref.jsonl", true),
    ] {
        for x in json_lines(&dir.join("train").join(file))? {
            row += 1;
            bump(&mut c.counts, "when2call: rows");
            let msgs = x.get("messages").cloned().unwrap_or(Value::Null);
            let n_msgs = msgs.as_array().map_or(0, Vec::len);
            let Some(user) = first_user(&msgs) else {
                continue;
            };
            let (resp, rejected) = if pref {
                if n_msgs != 1 {
                    continue;
                }
                (
                    x["chosen_response"]
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    x["rejected_response"]
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                )
            } else {
                let last = msgs.as_array().and_then(|a| a.last());
                if n_msgs != 2
                    || last.and_then(|m| m.get("role")).and_then(Value::as_str) != Some("assistant")
                {
                    continue;
                }
                (
                    last.and_then(|m| m.get("content"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    String::new(),
                )
            };
            let tools = when2call_tools(&x["tools"]);
            let label = when2call_label(&resp);
            let item = when2call_item(&user, &tools, label);
            let key = (item.rec.state.clone(), label);
            let Some(split) = split_of.get(&key).cloned() else {
                bump(&mut c.counts, "when2call: not in the decision file");
                continue;
            };
            if !used.insert(key) {
                bump(&mut c.counts, "when2call: duplicate of a decision record");
                continue;
            }
            let called = toolcall_names(&resp);
            let hint = toolcall_names(&rejected);
            match when2call_request(
                c,
                "when2call",
                row - 1,
                WHEN2CALL,
                &split,
                &user,
                &tools,
                label,
                &called,
                &hint,
            ) {
                Some(r) => {
                    bump(&mut c.counts, format!("when2call: kept {}", r.kind));
                    c.requests.push(r);
                }
                None => bump(
                    &mut c.counts,
                    format!("when2call: {} without a tool", WHEN2CALL_OPTS[label].0),
                ),
            }
        }
    }
    let unmatched = split_of.keys().filter(|k| !used.contains(*k)).count();
    add(
        &mut c.counts,
        "when2call: decision records not found",
        unmatched,
    );

    // The test: the first 2,000 test records of the decision file, from the mcq rows.
    let want: Vec<(String, usize)> = test_order.into_iter().take(2000).collect();
    let mut rows_of: HashMap<(String, usize), Vec<Value>> = HashMap::new();
    for x in json_lines(&dir.join("test").join("when2call_test_mcq.jsonl"))? {
        let canon = |k: &str| {
            if k == "direct" {
                "direct_answer".to_string()
            } else {
                k.to_string()
            }
        };
        let key = canon(
            x.get("correct_answer")
                .and_then(Value::as_str)
                .unwrap_or(""),
        );
        let Some(label) = WHEN2CALL_OPTS.iter().position(|(k, _)| *k == key) else {
            continue;
        };
        let q = x.get("question").and_then(Value::as_str).unwrap_or("");
        let item = when2call_item(q, &when2call_tools(&x["tools"]), label);
        rows_of
            .entry((item.rec.state.clone(), label))
            .or_default()
            .push(x);
    }
    for (k, key) in want.iter().enumerate() {
        let Some(x) = rows_of
            .get_mut(key)
            .and_then(|v| (!v.is_empty()).then(|| v.remove(0)))
        else {
            bump(&mut c.counts, "when2call-mcq: record without its row");
            continue;
        };
        let tools = when2call_tools(&x["tools"]);
        let q = x.get("question").and_then(Value::as_str).unwrap_or("");
        let target = x
            .get("target_tool")
            .and_then(Value::as_str)
            .map(|s| vec![s.to_string()])
            .unwrap_or_default();
        let answer = x["answers"]
            .get("tool_call")
            .and_then(Value::as_str)
            .map(toolcall_names)
            .unwrap_or_default();
        let named: Vec<String> = target.iter().chain(answer.iter()).cloned().collect();
        let (called, hint) = match key.1 {
            0 => (named.clone(), Vec::new()),
            _ => (Vec::new(), named),
        };
        match when2call_request(
            c,
            "when2call-mcq",
            k,
            TEST,
            "test",
            q,
            &tools,
            key.1,
            &called,
            &hint,
        ) {
            Some(r) => {
                bump(&mut c.counts, "when2call-mcq: kept");
                c.requests.push(r);
            }
            None => bump(&mut c.counts, "when2call-mcq: without a tool"),
        }
    }
    Ok(())
}

/// A MASSIVE intent's scenario: the part of its name before the first `_`.
pub fn scenario(intent: &str) -> &str {
    intent.split('_').next().unwrap_or(intent)
}

/// 4.3: the four scenarios with the lowest sha256(scenario + `1234`).
pub fn held_out_scenarios(scenarios: &BTreeSet<String>) -> BTreeSet<String> {
    let mut v: Vec<(&String, [u8; 32])> = scenarios.iter().map(|s| (s, super::salted(s))).collect();
    v.sort_by_key(|a| a.1);
    v.into_iter().take(4).map(|(s, _)| s.clone()).collect()
}

/// An intent as a tool: its name and the teacher's description.
fn intent_tool(
    c: &mut Collector,
    desc: &BTreeMap<String, String>,
    source: &str,
    name: &str,
) -> Result<String> {
    let d = desc.get(&format!("{source}:{name}")).ok_or_else(|| {
        anyhow!("no teacher description of the intent {source}:{name} (describe-intent)")
    })?;
    Ok(c.tool(name, d, ""))
}

/// The intents whose names the teacher describes (4.3, 4.5): `(source, name)`.
pub fn intent_names(p: &SourcePaths) -> Result<Vec<(String, String)>> {
    let mut out = BTreeSet::new();
    for r in read_jsonl::<CommandRecord>(&p.massive)? {
        if let Some(i) = r.intent {
            out.insert(("massive".to_string(), i));
        }
    }
    for r in read_jsonl::<DecisionRecord>(&p.banking77)? {
        if r.split != "test" {
            out.insert((
                "banking77".to_string(),
                banking_name(&r.opts[argmax(&r.target)]),
            ));
        }
    }
    for n in clinc_names(&p.src.join("clinc"))? {
        if n != "oos" {
            out.insert(("clinc".to_string(), n));
        }
    }
    Ok(out.into_iter().collect())
}

/// MASSIVE (`data/commands/massive.jsonl`): the train partition of the 14 trained scenarios →
/// training, their validation partition → validation, the test partition of the 4 held-out
/// scenarios → the reported probe (7.1); everything else unused. Both languages; French
/// requests keep the English intents.
pub fn massive(c: &mut Collector, p: &SourcePaths) -> Result<BTreeSet<String>> {
    let recs: Vec<CommandRecord> = read_jsonl(&p.massive)?;
    let scenarios: BTreeSet<String> = recs
        .iter()
        .filter_map(|r| r.intent.as_deref().map(|i| scenario(i).to_string()))
        .collect();
    let held = held_out_scenarios(&scenarios);
    for (i, r) in recs.iter().enumerate() {
        let Some(intent) = r.intent.as_deref() else {
            continue;
        };
        let out = held.contains(scenario(intent));
        let (group, split, source) = match (out, r.split.as_str()) {
            (false, "train") => (INTENTS, "train", "massive"),
            (false, "val") => (INTENTS, "val", "massive"),
            (true, "test") => (TEST, "test", "massive-heldout"),
            _ => {
                bump(
                    &mut c.counts,
                    format!(
                        "massive: unused ({})",
                        if out {
                            "held-out scenario"
                        } else {
                            "trained scenario, test partition"
                        }
                    ),
                );
                continue;
            }
        };
        let gold = intent_tool(c, &p.descriptions, "massive", intent)?;
        let user = c.text(&r.user);
        c.requests.push(Request {
            id: format!("{source}:{i}"),
            source: source.into(),
            group: group.into(),
            split: Some(split.into()),
            lang: r.lang.clone(),
            user,
            tools: Vec::new(),
            gold: Some(gold),
            exclude: Vec::new(),
            kind: "intent".into(),
        });
        bump(&mut c.counts, format!("{source}: kept {}", r.lang));
    }
    Ok(held)
}

/// A Banking77 option of `data/decisions/banking77.jsonl` (`humanize_intent` of the label) →
/// the dataset's label name.
pub fn banking_name(option: &str) -> String {
    option.to_lowercase().replace(' ', "_")
}

/// Banking77: the train and val records of `data/decisions/banking77.jsonl` (never a test).
pub fn banking77(c: &mut Collector, p: &SourcePaths) -> Result<()> {
    let recs: Vec<DecisionRecord> = read_jsonl(&p.banking77)?;
    for (i, r) in recs.iter().enumerate() {
        if r.split == "test" {
            continue;
        }
        let name = banking_name(&r.opts[argmax(&r.target)]);
        let gold = intent_tool(c, &p.descriptions, "banking77", &name)?;
        let user = c.text(&r.state);
        c.requests.push(Request {
            id: format!("banking77:{i}"),
            source: "banking77".into(),
            group: INTENTS.into(),
            split: Some(r.split.clone()),
            lang: "en".into(),
            user,
            tools: Vec::new(),
            gold: Some(gold),
            exclude: Vec::new(),
            kind: "intent".into(),
        });
        bump(&mut c.counts, "banking77: kept");
    }
    Ok(())
}

/// The label names of CLINC150 `plus`, from the dataset card's `class_label` of that config.
pub fn clinc_names(dir: &Path) -> Result<Vec<String>> {
    let card = std::fs::read_to_string(dir.join("README.md"))?;
    let mut names: BTreeMap<usize, String> = BTreeMap::new();
    let mut in_plus = false;
    for line in card.lines() {
        let t = line.trim();
        if let Some(cfg) = t.strip_prefix("- config_name:") {
            if in_plus && !names.is_empty() {
                break;
            }
            in_plus = cfg.trim() == "plus";
            continue;
        }
        if !in_plus {
            continue;
        }
        // '12': name
        if let Some(rest) = t.strip_prefix('\'') {
            if let Some((num, name)) = rest.split_once("':") {
                if let Ok(k) = num.parse::<usize>() {
                    names.insert(k, name.trim().to_string());
                }
            }
        }
    }
    if names.len() != 151 || names.keys().enumerate().any(|(i, k)| i != *k) {
        bail!(
            "clinc: {} label names in the card's plus config, 151 expected",
            names.len()
        );
    }
    Ok(names.into_values().collect())
}

/// CLINC150 `plus`, test split only (7.1): in-scope queries with their intent, out-of-scope
/// queries with none.
pub fn clinc(c: &mut Collector, p: &SourcePaths) -> Result<()> {
    let dir = p.src.join("clinc");
    let names = clinc_names(&dir)?;
    let mut tools = HashMap::new();
    for n in names.iter().filter(|n| *n != "oos") {
        tools.insert(n.clone(), intent_tool(c, &p.descriptions, "clinc", n)?);
    }
    let mut rows = Vec::new();
    crate::hf::for_each_parquet_row(
        &dir.join("plus/test-00000-of-00001.parquet"),
        &["text", "intent"],
        |r| {
            rows.push(r);
            Ok(())
        },
    )?;
    for (i, r) in rows.iter().enumerate() {
        let text = r.get("text").and_then(Value::as_str).unwrap_or("");
        let k = r
            .get("intent")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("clinc: intent"))? as usize;
        let name = names.get(k).ok_or_else(|| anyhow!("clinc: label {k}"))?;
        let gold = tools.get(name).cloned();
        let user = c.text(text);
        c.requests.push(Request {
            id: format!("clinc:{i}"),
            source: "clinc".into(),
            group: TEST.into(),
            split: Some("test".into()),
            lang: "en".into(),
            user,
            tools: Vec::new(),
            kind: if gold.is_some() {
                "intent"
            } else {
                "out_of_scope"
            }
            .into(),
            gold,
            exclude: Vec::new(),
        });
        bump(&mut c.counts, "clinc: kept");
    }
    Ok(())
}

/// The tool names of a MetaTool prompt's numbered list (`1. tool name: X, tool description: …`).
pub fn metatool_listed(prompt: &str) -> Vec<String> {
    prompt
        .lines()
        .filter_map(|l| {
            let (num, rest) = l.split_once(". tool name: ")?;
            num.trim().parse::<usize>().ok()?;
            Some(rest.split(", tool description:").next()?.trim().to_string())
        })
        .collect()
}

/// MetaTool (7.1, test only): the single-tool queries of `all_clean_data.csv`, the tools of
/// `plugin_des.json`, and the "similar choices" groups of the similar-tool subtask.
pub fn metatool(c: &mut Collector, dir: &Path) -> Result<()> {
    let d = dir.join("dataset");
    let des = read_json_file(&d.join("plugin_des.json"))?;
    let des = des
        .as_object()
        .ok_or_else(|| anyhow!("metatool: plugin_des is not an object"))?;
    let mut tools = HashMap::new();
    for (name, v) in des {
        let desc = v.as_str().unwrap_or("");
        let id = c.tool(name, desc, "");
        c.metatool.push(id.clone());
        tools.insert(name.clone(), id);
    }
    c.metatool.sort();
    c.metatool.dedup();
    let sub = read_json_file(&d.join("tmp_dataset/Task2-Subtask1.json"))?;
    for e in sub
        .as_array()
        .ok_or_else(|| anyhow!("metatool: subtask 1 is not a list"))?
    {
        let (Some(tool), Some(prompt)) = (
            e.get("tool").and_then(Value::as_str),
            e.get("action_prompt").and_then(Value::as_str),
        ) else {
            continue;
        };
        if c.similar.contains_key(tool) {
            continue;
        }
        let group: Vec<String> = metatool_listed(prompt)
            .into_iter()
            .filter(|n| n != tool && tools.contains_key(n))
            .collect();
        if !group.is_empty() {
            c.similar.insert(tool.to_string(), group);
        }
    }
    let text = std::fs::read_to_string(d.join("data/all_clean_data.csv"))?;
    let rows = crate::csv::parse_csv(&text);
    let header = rows.first().ok_or_else(|| anyhow!("metatool: empty csv"))?;
    let (qi, ti) = (
        header
            .iter()
            .position(|h| h == "Query")
            .ok_or_else(|| anyhow!("metatool: no Query"))?,
        header
            .iter()
            .position(|h| h == "Tool")
            .ok_or_else(|| anyhow!("metatool: no Tool"))?,
    );
    for (i, r) in rows.iter().enumerate().skip(1) {
        let (Some(q), Some(t)) = (r.get(qi), r.get(ti)) else {
            bump(&mut c.counts, "metatool: short row");
            continue;
        };
        let Some(gold) = tools.get(t.trim()).cloned() else {
            bump(&mut c.counts, "metatool: tool not described");
            continue;
        };
        let user = c.text(q);
        if user.is_empty() {
            continue;
        }
        c.requests.push(Request {
            id: format!("metatool:{i}"),
            source: "metatool".into(),
            group: TEST.into(),
            split: Some("test".into()),
            lang: "en".into(),
            user,
            tools: Vec::new(),
            gold: Some(gold),
            exclude: Vec::new(),
            kind: "call".into(),
        });
        bump(&mut c.counts, "metatool: kept");
    }
    Ok(())
}

/// BFCL `multiple` (the function of its possible answer) and `irrelevance` (none), 7.1.
pub fn bfcl(c: &mut Collector, dir: &Path) -> Result<()> {
    let mut answer: HashMap<String, String> = HashMap::new();
    for a in json_lines(&dir.join("possible_answer/BFCL_v3_multiple.json"))? {
        let id = a
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let name = a
            .get("ground_truth")
            .and_then(Value::as_array)
            .and_then(|g| g.first())
            .and_then(Value::as_object)
            .and_then(|o| o.keys().next().cloned());
        if let Some(n) = name {
            answer.insert(id, n);
        }
    }
    for (file, source) in [
        ("BFCL_v3_multiple.json", "bfcl-multiple"),
        ("BFCL_v3_irrelevance.json", "bfcl-irrelevance"),
    ] {
        for (i, x) in json_lines(&dir.join(file))?.iter().enumerate() {
            let id = x.get("id").and_then(Value::as_str).unwrap_or("");
            let user = x
                .get("question")
                .and_then(Value::as_array)
                .and_then(|q| q.first())
                .and_then(Value::as_array)
                .and_then(|turns| {
                    turns
                        .iter()
                        .rev()
                        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
                })
                .and_then(|m| m.get("content").and_then(Value::as_str))
                .unwrap_or("")
                .to_string();
            let listed = listed_of(x.get("function"));
            if source == "bfcl-multiple" {
                let calls: Vec<String> = answer.get(id).cloned().into_iter().collect();
                c.conversation(source, i, TEST, Some("test"), &user, &listed, &calls);
            } else {
                let user = c.text(&user);
                let own: Vec<String> = listed
                    .iter()
                    .map(|l| c.tool(&l.name, &l.desc, &l.schema))
                    .collect();
                c.requests.push(Request {
                    id: format!("{source}:{i}"),
                    source: source.into(),
                    group: TEST.into(),
                    split: Some("test".into()),
                    lang: "en".into(),
                    user,
                    tools: own,
                    gold: None,
                    exclude: Vec::new(),
                    kind: "irrelevance".into(),
                });
                bump(&mut c.counts, format!("{source}: kept"));
            }
        }
    }
    Ok(())
}

/// tool-routing-data `tools/test` (7.1): every query against all 10 tools, each described by
/// its first description.
pub fn tool_routing(c: &mut Collector, dir: &Path) -> Result<()> {
    let rows = json_lines(&dir.join("tools/test.jsonl"))?;
    let mut own = Vec::new();
    for t in &rows {
        let name = t.get("name").and_then(Value::as_str).unwrap_or("");
        let desc = t
            .get("descriptions")
            .and_then(Value::as_array)
            .and_then(|d| d.first())
            .and_then(Value::as_str)
            .unwrap_or("");
        own.push(c.tool(name, desc, ""));
    }
    let mut k = 0;
    for (t, gold) in rows.iter().zip(own.clone()) {
        for q in t
            .get("queries")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
        {
            let user = c.text(q.as_str().unwrap_or(""));
            c.requests.push(Request {
                id: format!("tool-routing:{k}"),
                source: "tool-routing".into(),
                group: TEST.into(),
                split: Some("test".into()),
                lang: "en".into(),
                user,
                tools: own.clone(),
                gold: Some(gold.clone()),
                exclude: Vec::new(),
                kind: "call".into(),
            });
            k += 1;
            bump(&mut c.counts, "tool-routing: kept");
        }
    }
    Ok(())
}

/// Exact duplicates inside a group of training sources: requests with the same language,
/// normalised text and gold option, the first kept.
pub fn dedupe(c: &mut Collector) {
    let mut seen: HashSet<(String, String, String, String)> = HashSet::new();
    let tools = &c.tools;
    let mut dropped: Counts = Counts::new();
    c.requests.retain(|r| {
        if r.group == TEST {
            return true;
        }
        let gold = r
            .gold
            .as_ref()
            .and_then(|g| tools.get(g))
            .map_or(String::new(), |t| t.option());
        let key = (r.group.clone(), r.lang.clone(), norm_text(&r.user), gold);
        let keep = seen.insert(key);
        if !keep {
            bump(&mut dropped, format!("{}: exact duplicate", r.source));
        }
        keep
    });
    for (k, n) in dropped {
        add(&mut c.counts, k, n);
    }
}

/// Reads every source in order.
pub fn read_all(p: &SourcePaths) -> Result<(Collector, serde_json::Value)> {
    let mut c = Collector::default();
    glaive(&mut c, &p.src.join("glaive"))?;
    hermes(&mut c, &p.src.join("hermes"))?;
    toolace(&mut c, &p.src.join("toolace"))?;
    seal_tools(&mut c, &p.src.join("seal-tools"))?;
    api_bank(&mut c, &p.src.join("api-bank"))?;
    when2call(&mut c, &p.src.join("when2call"), &p.when2call)?;
    let held = massive(&mut c, p)?;
    banking77(&mut c, p)?;
    clinc(&mut c, p)?;
    metatool(&mut c, &p.src.join("metatool"))?;
    bfcl(&mut c, &p.src.join("bfcl"))?;
    tool_routing(&mut c, &p.src.join("tool-routing-data"))?;
    dedupe(&mut c);
    let info = json!({"massive_held_out_scenarios": held});
    Ok((c, info))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_like_call_lists() {
        assert_eq!(
            bracket_calls(
                r#"[Market Trends API(trend_type="A, B", country='us'), GetCompetitions()]"#
            ),
            Some(vec!["Market Trends API".into(), "GetCompetitions".into()])
        );
        assert_eq!(
            bracket_calls("[/madlibs-diceware(nphrase=2)]"),
            Some(vec!["/madlibs-diceware".into()])
        );
        assert_eq!(
            bracket_calls("[f(x=[1, (2)], y={'a': ')'})]"),
            Some(vec!["f".into()])
        );
        assert_eq!(bracket_calls("I cannot do that."), None);
        assert_eq!(bracket_calls("[f(x=1]"), None);
        assert_eq!(bracket_calls("[]"), None);
    }

    #[test]
    fn toolcalls() {
        assert_eq!(
            toolcall_names(
                r#"<TOOLCALL>[{"name": "a", "arguments": {}}, {"name": "b"}]</TOOLCALL>"#
            ),
            vec!["a", "b"]
        );
        assert_eq!(
            toolcall_names(r#"{"name": "c", "arguments": {}}"#),
            vec!["c"]
        );
        assert!(toolcall_names("Sorry, I can't.").is_empty());
    }

    #[test]
    fn conversations_keep_the_first_listed_call() {
        let mut c = Collector::default();
        let l = |n: &str| Listed {
            name: n.into(),
            desc: format!("Does {n}."),
            schema: "{}".into(),
        };
        assert!(c.conversation(
            "s",
            0,
            TOOLS,
            None,
            " hi ",
            &[l("a"), l("b"), l("c")],
            &["b".into(), "c".into(), "b".into()]
        ));
        let r = &c.requests[0];
        assert_eq!(r.user, "hi");
        assert_eq!(r.tools.len(), 3);
        assert_eq!(c.tools[r.gold.as_ref().unwrap()].name, "b");
        assert_eq!(r.exclude.len(), 1);
        assert_eq!(c.tools[&r.exclude[0]].name, "c");
        assert!(!c.conversation("s", 1, TOOLS, None, "x", &[l("a")], &["z".into()]));
        assert!(!c.conversation("s", 2, TOOLS, None, "x", &[l("a")], &[]));
        assert_eq!(c.counts["s: call not listed"], 1);
        assert_eq!(c.counts["s: no call"], 1);
    }

    #[test]
    fn schema_keys_are_counted_and_requests_masked() {
        let mut c = Collector::default();
        let id = c.tool(
            "t",
            "Calls the API.",
            r#"{"default": "sk-AbCdEf0123456789xyz"}"#,
        );
        assert!(c.schema_keys.contains(&id));
        assert_eq!(
            c.text("my key is AIzaSyA1b2C3d4E5f6G7h8I9j0K"),
            "my key is <KEY>"
        );
        assert_eq!(c.counts["keys masked in requests"], 1);
    }

    #[test]
    fn scenarios_and_names() {
        assert_eq!(scenario("iot_hue_lightoff"), "iot");
        assert_eq!(banking_name("Visa or Mastercard"), "visa_or_mastercard");
        assert_eq!(banking_name("ATM support"), "atm_support");
        let s: BTreeSet<String> = (0..18).map(|i| format!("s{i}")).collect();
        let h = held_out_scenarios(&s);
        assert_eq!(h.len(), 4);
        assert_eq!(h, held_out_scenarios(&s));
    }

    #[test]
    fn metatool_lists() {
        let p = "Here:\n1. tool name: A, tool description: x, y\n2. tool name: B C, tool description: z\nend";
        assert_eq!(metatool_listed(p), vec!["A", "B C"]);
    }

    #[test]
    fn duplicates_inside_training_groups() {
        let mut c = Collector::default();
        let l = Listed {
            name: "a".into(),
            desc: "A.".into(),
            schema: String::new(),
        };
        for i in 0..3 {
            let text = if i == 2 { "Other" } else { "Same, text!" };
            c.conversation(
                "s",
                i,
                TOOLS,
                None,
                text,
                std::slice::from_ref(&l),
                &["a".into()],
            );
        }
        c.conversation(
            "t",
            0,
            TEST,
            Some("test"),
            "same text",
            std::slice::from_ref(&l),
            &["a".into()],
        );
        dedupe(&mut c);
        assert_eq!(c.requests.len(), 3);
        assert_eq!(c.counts["s: exact duplicate"], 1);
    }
}
