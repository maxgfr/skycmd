//! The fresh test of s6-route v4 (`docs/s6-route4-prereg.md`, section 6): MCP tools of
//! `Agent-Ark/Toucan-1.5M`, a family no model of this project has seen, built so that the
//! surface cue of D7 cannot be in it.
//!
//! 1. [`rows`] (`sky-data route toucan-rows --input <SFT parquet files> --out <t4 dir>`): the
//!    parquet files of the SFT config read into `<t4>/stage/rows.jsonl`, one [`ToucanRow`] per
//!    qualifying row, by the rule of amendment 1 ([`parse_row`]: the SFT config has no `metadata`
//!    column; the right tool is the row's single `target_tools` entry, listed in the row's own
//!    `tools` as `<server>-<tool>`, and the server is that prefix). [`schema`]
//!    (`route toucan-schema`) prints the files' structure only.
//! 2. [`stage`] (`sky-data route toucan-stage --t4 <t4 dir>`): the tool table of every server
//!    tool and `<t4>/stage/texts.jsonl`, which `scripts/route-embed.py --build <t4 dir>` embeds
//!    with v2's pinned all-MiniLM-L6-v2.
//! 3. [`build_test`] (`sky-data --seed 1236 route toucan --v2 <v2 build> --t4 <t4 dir>`): the
//!    filters of 6.1 and 6.2, the cluster rule against v2's training and validation tools, the
//!    caps, 2,000 slates (20% removed) filled within the gold's surface cell, written once:
//!    `t4_tools.jsonl`, `t4_tools_masked.jsonl`, `t4_tools.servers.jsonl`; `FRESH4.json` (hashes
//!    and counts); the items of the probe copies' teacher job (a tool's name and description
//!    only, split `fresh-tools`, `<t4>/teacher/rewrite-description.items.jsonl`).
//! 4. The teacher (the lead runs it): `BUILD=<t4 dir> scripts/route-teacher.sh
//!    rewrite-description loop`; the guard of `sky-data teacher` takes a `fresh-tools` item
//!    only when it holds a name and a description and nothing else.
//! 5. [`probes`] (`sky-data --seed 1236 route toucan-probes --t4 <t4 dir>`): `t4_tools_rw.jsonl`
//!    and `t4_tools_masked_rw.jsonl` (6.4), written once, their hashes added.
//!
//! The builder reads no output of any system; no request goes to the teacher; the step prints
//! counts only.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::build::{
    self, answers_path, items_path, Description, RewriteItem, RouteRecord, CLUSTER_COSINE,
    REMOVED_SHARE, TEST_CAP, TEST_COSINE, TEST_SLATES,
};
use super::cluster::{self, Matrix};
use super::slates::{self, Slate};
use super::v4::surface;
use super::v5::{self, Record5};
use super::{
    bump, cut_description, mask_keys, norm_text, option_text, read_json, read_lines, sha256_file,
    sha256_hex, state_of, write_json, Counts, Tool, INSTRUCTION, INTENTS,
};
use crate::record::write_jsonl;
use crate::rng::Rng;

/// The seed of 6.2.
pub const T4_SEED: u64 = 1236;
/// 6.2, item 3.
pub const SERVER_CAP: usize = 40;
/// 6.3: below this many slates v4 is not trained.
pub const MIN_SLATES: usize = 600;
/// 6.1: the subsets whose rows are read.
pub const SUBSETS: [&str; 2] = ["single-turn-original", "single-turn-diversify"];
/// The columns of the SFT config, as its card (commit `0df3cf37`) declares them.
pub const SFT_COLUMNS: [&str; 6] = [
    "uuid",
    "subset_name",
    "question",
    "target_tools",
    "tools",
    "messages",
];
/// The `split` of every record of the fresh test.
pub const SPLIT: &str = "fresh";
/// The name of the test files (`t4_tools*.jsonl`).
pub const NAME: &str = "t4-tools";

/// The rule of 6.2, item 5: v4's (`docs/s6-route4-prereg.md`, files `t4_*`), or v5's
/// (`docs/s6-route5-prereg.md` 6.2: the fill of other servers within v5's surface cell, the
/// levels L0 to L6 at 16 candidates; files `t5_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Rule {
    #[default]
    V4,
    V5,
}

impl Rule {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "v4" => Some(Rule::V4),
            "v5" => Some(Rule::V5),
            _ => None,
        }
    }

    /// The name of the test files' records (`t4-tools`, `t5-tools`).
    pub fn name(self) -> &'static str {
        match self {
            Rule::V4 => NAME,
            Rule::V5 => "t5-tools",
        }
    }

    /// The stem of the files (`t4`, `t5`).
    pub fn stem(self) -> &'static str {
        match self {
            Rule::V4 => "t4",
            Rule::V5 => "t5",
        }
    }

    /// The manifest of the fresh folder (`FRESH4.json`, `FRESH5.json`).
    pub fn manifest(self) -> &'static str {
        match self {
            Rule::V4 => "FRESH4.json",
            Rule::V5 => "FRESH5.json",
        }
    }

    fn version(self) -> &'static str {
        match self {
            Rule::V4 => "v4",
            Rule::V5 => "v5",
        }
    }

    fn prereg(self) -> &'static str {
        match self {
            Rule::V4 => "docs/s6-route4-prereg.md",
            Rule::V5 => "docs/s6-route5-prereg.md",
        }
    }
}

/// `docs/s6-route5-prereg.md` 6: the parquet files hash-checked again against the manifest of
/// the fetch (`<data>/MANIFEST4.json`) before anything reads them: each input must be one of its
/// files by name, with its bytes and sha256.
pub fn check_manifest(inputs: &[PathBuf], manifest: &Path) -> Result<Value> {
    let m = read_json(manifest)?;
    let files = m["files"]
        .as_array()
        .with_context(|| format!("{}: no files", manifest.display()))?;
    let mut out = BTreeMap::new();
    for p in inputs {
        let name = p
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        let entry = files
            .iter()
            .find(|f| {
                f["file"]
                    .as_str()
                    .is_some_and(|x| x.rsplit('/').next() == Some(name.as_str()))
            })
            .with_context(|| format!("{} is not a file of {}", p.display(), manifest.display()))?;
        let bytes = std::fs::metadata(p)
            .with_context(|| format!("stat {}", p.display()))?
            .len();
        if entry["bytes"].as_u64() != Some(bytes) {
            bail!(
                "{}: {bytes} bytes, {} records {}",
                p.display(),
                manifest.display(),
                entry["bytes"]
            );
        }
        let h = sha256_file(p)?;
        if entry["sha256"].as_str() != Some(h.as_str()) {
            bail!(
                "{}: sha256 {h}, {} records {}",
                p.display(),
                manifest.display(),
                entry["sha256"]
            );
        }
        out.insert(name, json!({"bytes": bytes, "sha256": h}));
    }
    if inputs.is_empty() {
        bail!("no parquet file to check");
    }
    Ok(json!({"checked against": manifest.display().to_string(), "files": out}))
}

/// One tool of a server, as the server's metadata lists it.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct RawTool {
    pub name: String,
    pub description: String,
}

/// One Toucan row as the builder reads it (`<t4>/stage/rows.jsonl`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ToucanRow {
    /// `toucan:<uuid>`.
    pub id: String,
    /// `subset_name`.
    pub subset: String,
    /// `question`: the request.
    pub question: String,
    /// The server whose metadata lists the row's tools.
    pub server: String,
    /// The server's tools, in the metadata's order.
    pub tools: Vec<RawTool>,
    /// `target_tools`, as listed (a name, or `Server_Name::Tool_Name` across servers).
    pub target_tools: Vec<String>,
}

/// What the parser makes of one SFT row (amendment 1 of `docs/s6-route4-prereg.md`, 6.1).
#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    /// A row of the two subsets whose single `target_tools` entry is one tool of its own list.
    Row(ToucanRow),
    /// A row left out, and why (counted).
    Out(&'static str),
}

/// Amendment 1, 6.1: one SFT row. `target_tools` is a string of comma-separated tool names;
/// `tools` a JSON string, the list of the row's tools as `{type, function: {name, description,
/// parameters}}`, each listed name `<server>-<tool>`. A row qualifies when it is of the subsets
/// `single-turn-original` or `single-turn-diversify`, has one target, and exactly one tool of its
/// list is `<server>-<target>`; the row's server is that prefix, and every tool of its list must
/// carry it (each tool is then named without it, as `target_tools` names it). Anything that does
/// not have this form fails loudly, with the row's uuid.
pub fn parse_row(v: &Value) -> Result<Parsed> {
    let field = |k: &str| -> Result<&str> {
        v[k].as_str()
            .with_context(|| format!("a row whose `{k}` is not a string: {}", v["uuid"]))
    };
    let uuid = field("uuid")?;
    let subset = field("subset_name")?;
    if !SUBSETS.contains(&subset) {
        return Ok(Parsed::Out("6.1: another subset"));
    }
    let targets: Vec<String> = field("target_tools")?
        .split(',')
        .map(super::collapse_ws)
        .filter(|t| !t.is_empty())
        .collect();
    if targets.len() != 1 {
        return Ok(Parsed::Out("6.1: not exactly one target tool"));
    }
    let target = targets[0].clone();
    let list: Value = serde_json::from_str(field("tools")?)
        .with_context(|| format!("row {uuid}: `tools` is not JSON"))?;
    let Some(list) = list.as_array() else {
        bail!("row {uuid}: `tools` is not a JSON list");
    };
    let mut listed = Vec::with_capacity(list.len());
    for t in list {
        let f = &t["function"];
        let (Some(name), Some(desc)) = (f["name"].as_str(), f["description"].as_str()) else {
            bail!("row {uuid}: a tool without function.name and function.description");
        };
        listed.push((super::collapse_ws(name), desc.to_string()));
    }
    let names: Vec<String> = listed.iter().map(|(n, _)| n.clone()).collect();
    let server = match server_match(&target, &names) {
        Ok((server, _)) => server,
        Err(_) if target.contains("::") => {
            return Ok(Parsed::Out("6.1: a target written Server::Tool"))
        }
        Err(why) => return Ok(Parsed::Out(why)),
    };
    let prefix = format!("{server}-");
    let mut tools = Vec::with_capacity(listed.len());
    for (n, d) in listed {
        let Some(bare) = n.strip_prefix(&prefix) else {
            bail!("row {uuid}: the tool {n:?} does not carry its server's prefix {prefix:?}");
        };
        tools.push(RawTool {
            name: bare.to_string(),
            description: d,
        });
    }
    let question = field("question")?.to_string();
    Ok(Parsed::Row(ToucanRow {
        id: format!("toucan:{uuid}"),
        subset: subset.to_string(),
        question,
        server,
        tools,
        target_tools: vec![target],
    }))
}

/// Step 1: the SFT parquet files read into `<out>/stage/rows.jsonl` by [`parse_row`]: the
/// columns must be the card's six, every uuid distinct; the rows read and left out are counted
/// in `<out>/counts/toucan-rows.json`. No text is printed.
pub fn rows(inputs: &[PathBuf], out: &Path) -> Result<Value> {
    let mut read = Counts::new();
    let mut left = Counts::new();
    let mut files = BTreeMap::new();
    let mut uuids = HashSet::new();
    let mut kept = Vec::new();
    for p in inputs {
        let cols = crate::hf::parquet_columns(p)?;
        let want: BTreeSet<&str> = SFT_COLUMNS.iter().copied().collect();
        let got: BTreeSet<&str> = cols.iter().map(String::as_str).collect();
        if want != got {
            bail!(
                "{}: columns {got:?}, the SFT card declares {want:?}: not the pinned file",
                p.display()
            );
        }
        let mut n = 0usize;
        crate::hf::for_each_parquet_row(
            p,
            &["uuid", "subset_name", "question", "target_tools", "tools"],
            |v| {
                n += 1;
                bump(
                    &mut read,
                    v["subset_name"].as_str().unwrap_or("<not a string>"),
                );
                let u = v["uuid"].as_str().unwrap_or("").to_string();
                if !uuids.insert(u.clone()) {
                    bail!("{}: the uuid {u} repeats", p.display());
                }
                match parse_row(&v)? {
                    Parsed::Row(r) => kept.push(r),
                    Parsed::Out(why) => bump(&mut left, why),
                }
                Ok(())
            },
        )?;
        files.insert(p.display().to_string(), n);
    }
    if inputs.is_empty() {
        bail!("no parquet file");
    }
    write_jsonl(&out.join("stage/rows.jsonl"), &kept)?;
    let servers: BTreeSet<&str> = kept.iter().map(|r| r.server.as_str()).collect();
    let mut by_subset = Counts::new();
    for r in &kept {
        bump(&mut by_subset, &r.subset);
    }
    let counts = json!({
        "about": "s6-route v4 (docs/s6-route4-prereg.md 6.1, amendment 1): the rows of Toucan's SFT files",
        "rows by file": files, "rows read by subset": read, "rows left out, by reason": left,
        "rows kept": kept.len(), "rows kept by subset": by_subset, "servers of the kept rows": servers.len(),
    });
    write_json(&out.join("counts/toucan-rows.json"), &counts)?;
    Ok(counts)
}

/// The skeleton of a JSON value: its types and keys, every string, number and boolean replaced
/// by its type, arrays by the set of their elements' skeletons (no text of the value).
pub fn skeleton(v: &Value, depth: usize) -> String {
    if depth == 0 {
        return "...".into();
    }
    match v {
        Value::Null => "null".into(),
        Value::Bool(_) => "bool".into(),
        Value::Number(_) => "num".into(),
        Value::String(_) => "str".into(),
        Value::Array(a) => {
            let set: BTreeSet<String> = a.iter().map(|x| skeleton(x, depth - 1)).collect();
            format!("[{}]", set.into_iter().collect::<Vec<_>>().join(" | "))
        }
        Value::Object(o) => {
            let parts: Vec<String> = o
                .iter()
                .map(|(k, x)| match k.as_str() {
                    // parameter names are the tool's own: their skeleton only says it has some
                    "properties" => format!("{k}: {{..}}"),
                    _ => format!("{k}: {}", skeleton(x, depth - 1)),
                })
                .collect();
            format!("{{{}}}", parts.join(", "))
        }
    }
}

/// A string column holding JSON: the parsed value, or the string itself when it does not parse.
fn json_field(v: &Value) -> Value {
    match v {
        Value::String(t) => serde_json::from_str(t).unwrap_or_else(|_| v.clone()),
        _ => v.clone(),
    }
}

/// The structure of the SFT files, for the amendment of 6.1 (no request, no description and no
/// name is printed): the columns and their types, the rows by subset, the skeletons of
/// `target_tools` and `tools` with their counts, the uuids' uniqueness, the shape of the tool
/// names (separators), and the rows of the two subsets with exactly one target tool that names
/// exactly one tool of their own `tools` list.
pub fn schema(inputs: &[PathBuf]) -> Result<Value> {
    let mut files = BTreeMap::new();
    let mut by_subset = Counts::new();
    let mut tt_skel = Counts::new();
    let mut tools_skel = Counts::new();
    let mut tt_len = Counts::new();
    let mut uuids: HashSet<String> = HashSet::new();
    let mut dup = 0usize;
    let mut q = Counts::new();
    let mut lists: HashSet<String> = HashSet::new();
    let mut lists_one: HashSet<String> = HashSet::new();
    let mut tools_per_row = Counts::new();
    let mut names = Counts::new();
    let mut servers: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for p in inputs {
        let reader = parquet::file::reader::SerializedFileReader::new(
            std::fs::File::open(p).with_context(|| format!("open {}", p.display()))?,
        )?;
        use parquet::file::reader::FileReader;
        let md = reader.metadata();
        let cols: Vec<Value> = md
            .file_metadata()
            .schema_descr()
            .columns()
            .iter()
            .map(|c| {
                json!({"column": c.path().string(), "physical": format!("{:?}", c.physical_type()),
                             "logical": format!("{:?}", c.logical_type_ref())})
            })
            .collect();
        files.insert(
            p.display().to_string(),
            json!({"rows": md.file_metadata().num_rows(), "row groups": md.num_row_groups(), "columns": cols}),
        );
        crate::hf::for_each_parquet_row(p, &SFT_COLUMNS, |v| {
            let subset = v["subset_name"]
                .as_str()
                .unwrap_or("<not a string>")
                .to_string();
            bump(&mut by_subset, &subset);
            if let Some(u) = v["uuid"].as_str() {
                if !uuids.insert(u.to_string()) {
                    dup += 1;
                }
            }
            let tt = json_field(&v["target_tools"]);
            bump(&mut tt_skel, skeleton(&tt, 4));
            let targets: Vec<String> = match &tt {
                Value::Array(a) => a
                    .iter()
                    .filter_map(|x| x.as_str().map(str::to_string))
                    .collect(),
                Value::String(t) => t
                    .split(',')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect(),
                _ => Vec::new(),
            };
            bump(
                &mut tt_len,
                format!("{subset}: {} target(s)", targets.len()),
            );
            let tl = json_field(&v["tools"]);
            bump(&mut tools_skel, skeleton(&tl, 5));
            let tool_names: Vec<String> = tl
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|t| {
                            t["function"]["name"]
                                .as_str()
                                .or_else(|| t["name"].as_str())
                                .map(super::collapse_ws)
                        })
                        .collect()
                })
                .unwrap_or_default();
            bump(&mut tools_per_row, format!("{}", tool_names.len().min(50)));
            for n in &tool_names {
                bump(
                    &mut names,
                    if n.contains("::") {
                        "names with ::"
                    } else if n.contains("__") {
                        "names with __"
                    } else if n.contains('.') {
                        "names with ."
                    } else if n.contains('/') {
                        "names with /"
                    } else {
                        "names without a separator"
                    },
                );
            }
            let mut sorted = tool_names.clone();
            sorted.sort();
            let key = sha256_hex(sorted.join("\n").as_bytes());
            lists.insert(key.clone());
            if SUBSETS.contains(&subset.as_str()) {
                if targets.len() == 1 {
                    let t = super::collapse_ws(&targets[0]);
                    let exact = tool_names.iter().filter(|n| **n == t).count();
                    bump(
                        &mut q,
                        format!(
                            "one target, named exactly by {} tool(s) of its list",
                            exact.min(2)
                        ),
                    );
                    match server_match(&t, &tool_names) {
                        Ok((server, _)) => {
                            bump(&mut q, "one target, exactly one tool <server>-<target> of its list (qualifies)");
                            lists_one.insert(key);
                            let all = tool_names
                                .iter()
                                .all(|n| n.starts_with(&format!("{server}-")));
                            bump(
                                &mut q,
                                if all {
                                    "qualifying: every tool of the list carries the server's prefix"
                                } else {
                                    "qualifying: a tool of the list without the server's prefix"
                                },
                            );
                            servers
                                .entry(server)
                                .or_default()
                                .insert(sha256_hex(sorted_names(&tool_names).as_bytes()));
                        }
                        Err(why) => bump(&mut q, format!("one target, not qualifying: {why}")),
                    }
                    if t.contains("::") {
                        bump(&mut q, "one target written Server::Tool");
                    }
                } else {
                    bump(&mut q, format!("{} targets", targets.len()));
                }
            }
            Ok(())
        })?;
    }
    let mut by_list: HashMap<&str, usize> = HashMap::new();
    for v in servers.values() {
        for l in v {
            *by_list.entry(l.as_str()).or_default() += 1;
        }
    }
    let shared = by_list.values().filter(|&&n| n > 1).count();
    Ok(json!({
        "files": files,
        "rows by subset_name": by_subset,
        "target_tools: skeletons": tt_skel,
        "target_tools: targets per row": tt_len,
        "tools: skeletons (depth 5)": tools_skel.into_iter().map(|(k, v)| (k.chars().take(600).collect::<String>(), v)).collect::<BTreeMap<_, _>>(),
        "tools: tools per row (50: 50 or more)": tools_per_row,
        "tool names by separator": names,
        "uuids": {"distinct": uuids.len(), "repeated": dup},
        "rows of the two subsets": q,
        "distinct tool lists (sha256 of the sorted names), all rows": lists.len(),
        "distinct tool lists among the qualifying rows": lists_one.len(),
        "servers of the qualifying rows (the prefix of the gold's listed name)": servers.len(),
        "servers listed with more than one tool list": servers.values().filter(|v| v.len() > 1).count(),
        "tool lists shared by several servers": shared,
    }))
}

/// The sorted tool names of a list, one per line (the key of a tool list).
pub fn sorted_names(names: &[String]) -> String {
    let mut v = names.to_vec();
    v.sort();
    v.join("\n")
}

/// Amendment 1 (6.1): the row's right tool is the one tool of its own `tools` list whose listed
/// name is `<server>-<target>` for the row's single `target_tools` entry; returns the server (the
/// prefix) and the tool's position, or why the row does not qualify.
pub fn server_match(
    target: &str,
    names: &[String],
) -> std::result::Result<(String, usize), &'static str> {
    if target.is_empty() {
        return Err("an empty target");
    }
    let suffix = format!("-{target}");
    let m: Vec<usize> = names
        .iter()
        .enumerate()
        .filter(|(_, n)| n.as_str() == target || (n.ends_with(&suffix) && n.len() > suffix.len()))
        .map(|(i, _)| i)
        .collect();
    match m.as_slice() {
        [] => Err("no tool of its list is <server>-<target>"),
        [i] => {
            let n = &names[*i];
            if n == target {
                Err("the tool's listed name carries no server prefix")
            } else {
                Ok((n[..n.len() - suffix.len()].to_string(), *i))
            }
        }
        _ => Err("several tools of its list end with -<target>"),
    }
}

/// The tools of a stage: every server tool, by id, and the servers that list each.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct StagedTool {
    #[serde(flatten)]
    pub tool: Tool,
    pub servers: Vec<String>,
}

/// Step 2: `<t4>/stage/tools.jsonl` and `texts.jsonl` (the texts to embed, one per tool).
pub fn stage(t4: &Path) -> Result<Value> {
    let rows: Vec<ToucanRow> = read_lines(&t4.join("stage/rows.jsonl"))?;
    let mut tools: BTreeMap<String, (Tool, BTreeSet<String>)> = BTreeMap::new();
    let mut masked_keys = 0usize;
    for r in &rows {
        for t in &r.tools {
            masked_keys += mask_keys(&t.description).1;
            let tool = Tool::new(&t.name, &t.description);
            tools
                .entry(tool.id.clone())
                .or_insert_with(|| (tool, BTreeSet::new()))
                .1
                .insert(r.server.clone());
        }
    }
    let staged: Vec<StagedTool> = tools
        .into_values()
        .map(|(tool, s)| StagedTool {
            tool,
            servers: s.into_iter().collect(),
        })
        .collect();
    let texts: Vec<Value> = staged
        .iter()
        .map(|t| json!({"id": t.tool.id, "text": t.tool.option()}))
        .collect();
    write_jsonl(&t4.join("stage/tools.jsonl"), &staged)?;
    write_jsonl(&t4.join("stage/texts.jsonl"), &texts)?;
    let servers: BTreeSet<&str> = rows.iter().map(|r| r.server.as_str()).collect();
    let report = json!({"rows": rows.len(), "servers": servers.len(), "tools": staged.len(),
                        "key-like strings masked in descriptions": masked_keys});
    write_json(&t4.join("counts/toucan-stage.json"), &report)?;
    Ok(report)
}

pub struct TestOptions {
    /// v2's build folder (read only).
    pub v2: PathBuf,
    /// The Toucan folder (`stage/`, `emb/`; the test files go to `<t4>/fresh`).
    pub t4: PathBuf,
    pub seed: u64,
    /// A folder for `test-files.json` and `counts.json` (`runs/s6-route4`, `runs/s6-route5`).
    pub record: Option<PathBuf>,
    /// The rule of 6.2, item 5, and the files' names.
    pub rule: Rule,
}

/// The fresh folder of a Toucan folder.
pub fn fresh_dir(t4: &Path) -> PathBuf {
    t4.join("fresh")
}

/// The training and validation tools of v2 (its pools and intents), their normalised names and
/// MiniLM rows, and the texts of its training and validation requests.
struct V2Side {
    names: HashSet<String>,
    rows: Vec<usize>,
    minilm: Matrix,
    texts: HashSet<String>,
    norms: HashSet<String>,
}

fn v2_side(v2: &Path) -> Result<V2Side> {
    let l = build::load(v2)?;
    let a = build::assign(&l)?;
    let kept = a.kept();
    let mut tools: BTreeSet<usize> = a.pool("train").into_iter().collect();
    tools.extend(a.pool("val"));
    for k in kept.iter().filter(|k| k.r.group == INTENTS) {
        tools.extend(k.gold);
    }
    let mut texts = HashSet::new();
    let mut norms = HashSet::new();
    for k in kept
        .iter()
        .filter(|k| k.split == "train" || k.split == "val")
    {
        texts.insert(k.r.user.clone());
        norms.insert(norm_text(&k.r.user));
    }
    Ok(V2Side {
        names: tools.iter().map(|&t| l.norm[t].clone()).collect(),
        rows: tools.iter().filter_map(|&t| l.row[t]).collect(),
        minilm: l.minilm,
        texts,
        norms,
    })
}

/// For each row of `a` (`rows_a`), whether its cosine to some row of `b` (`rows_b`) is at least
/// `threshold`; on threads, the result independent of their number.
fn near_any(
    a: &Matrix,
    rows_a: &[usize],
    b: &Matrix,
    rows_b: &[usize],
    threshold: f32,
) -> Vec<bool> {
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(16);
    let mut out = vec![false; rows_a.len()];
    let parts: Vec<Vec<(usize, bool)>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                s.spawn(move || {
                    (t..rows_a.len())
                        .step_by(threads)
                        .map(|i| {
                            let ra = a.row(rows_a[i]);
                            (
                                i,
                                rows_b
                                    .iter()
                                    .any(|&j| cluster::dot(ra, b.row(j)) >= threshold),
                            )
                        })
                        .collect()
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (i, v) in parts.into_iter().flatten() {
        out[i] = v;
    }
    out
}

/// The Toucan stage loaded: its tools, rows and MiniLM rows.
struct T4 {
    tools: Vec<StagedTool>,
    index: HashMap<String, usize>,
    rows: Vec<ToucanRow>,
    minilm: Matrix,
    row: Vec<usize>,
}

fn load_t4(t4: &Path) -> Result<T4> {
    let st = t4.join("stage");
    let tools: Vec<StagedTool> = read_lines(&st.join("tools.jsonl"))?;
    let index: HashMap<String, usize> = tools
        .iter()
        .enumerate()
        .map(|(i, t)| (t.tool.id.clone(), i))
        .collect();
    let rows: Vec<ToucanRow> = read_lines(&st.join("rows.jsonl"))?;
    let texts: Vec<Value> = read_lines(&st.join("texts.jsonl"))?;
    let meta = read_json(&t4.join("emb/meta.json"))?;
    let want = sha256_file(&st.join("texts.jsonl"))?;
    if meta["texts_sha256"].as_str() != Some(want.as_str()) {
        bail!("{}/emb/meta.json was not written for its stage/texts.jsonl: run scripts/route-embed.py --build {} again", t4.display(), t4.display());
    }
    let minilm = Matrix::read_npy(&t4.join("emb/minilm.npy"))?;
    if minilm.rows != texts.len() || texts.len() != tools.len() {
        bail!(
            "{} embedded rows, {} texts, {} tools",
            minilm.rows,
            texts.len(),
            tools.len()
        );
    }
    let mut row = vec![usize::MAX; tools.len()];
    for (k, t) in texts.iter().enumerate() {
        let id = t["id"].as_str().unwrap_or("");
        let i = *index
            .get(id)
            .with_context(|| format!("texts.jsonl: unknown tool {id}"))?;
        row[i] = k;
    }
    Ok(T4 {
        tools,
        index,
        rows,
        minilm,
        row,
    })
}

/// A row kept by the filters of 6.1: its gold and its server's other tools (indices).
#[derive(Clone, Debug)]
struct Cand {
    row: usize,
    gold: usize,
    server_tools: Vec<usize>,
}

fn record(
    name: &str,
    uid: &str,
    r: &ToucanRow,
    kind: &str,
    s: &Slate,
    text: impl Fn(usize) -> String,
    ids: &[String],
) -> RouteRecord {
    let (opts, target) = s.render(text);
    RouteRecord {
        src: name.into(),
        t: "choice".into(),
        ins: INSTRUCTION.into(),
        opts,
        state: state_of(&mask_keys(&r.question).0),
        target,
        split: SPLIT.into(),
        uid: uid.into(),
        kind: kind.into(),
        lang: "und".into(),
        request: r.id.clone(),
        tools: s.tools.iter().map(|&t| ids[t].clone()).collect(),
        follow: None,
        rewritten: None,
    }
}

/// v5's records: each slate with its level and its options' roles.
fn with_levels(recs: &[RouteRecord], meta: &[(u8, Vec<String>)]) -> Vec<Record5> {
    recs.iter()
        .zip(meta)
        .map(|(r, (l, roles))| Record5 {
            r: r.clone(),
            cell_level: *l,
            roles: roles.clone(),
            from: None,
        })
        .collect()
}

/// Step 3: the test files of 6.2, written once, with their hashes and counts.
pub fn build_test(o: &TestOptions) -> Result<Value> {
    let t = load_t4(&o.t4)?;
    let v2 = v2_side(&o.v2)?;
    let n = t.tools.len();
    let norm: Vec<String> = t.tools.iter().map(|x| x.tool.norm()).collect();
    let ids: Vec<String> = t.tools.iter().map(|x| x.tool.id.clone()).collect();
    let mut drop = Counts::new();
    let mut read = Counts::new();

    // 6.2 item 1: the cluster rule against v2's training and validation tools
    let all_rows: Vec<usize> = (0..n).map(|i| t.row[i]).collect();
    let near = near_any(&t.minilm, &all_rows, &v2.minilm, &v2.rows, CLUSTER_COSINE);
    let near_v2: Vec<bool> = (0..n)
        .map(|i| near[i] || v2.names.contains(&norm[i]))
        .collect();
    // Toucan's own clusters, by v2's rule
    let rep = cluster::clusters(&t.minilm, &all_rows, &norm, CLUSTER_COSINE);
    // the label of a cluster: its smallest normalised name (v2's key), hashed
    let mut cluster_key: HashMap<usize, &str> = HashMap::new();
    for (i, &c) in rep.iter().enumerate() {
        let e = cluster_key.entry(c).or_insert(norm[i].as_str());
        if norm[i].as_str() < *e {
            *e = norm[i].as_str();
        }
    }
    let cos = |a: usize, b: usize| cluster::dot(t.minilm.row(t.row[a]), t.minilm.row(t.row[b]));

    // 6.1: the subsets, one target tool, the gold in its server's metadata
    let mut cands: Vec<Cand> = Vec::new();
    for (i, r) in t.rows.iter().enumerate() {
        bump(&mut read, &r.subset);
        if !SUBSETS.contains(&r.subset.as_str()) {
            bump(&mut drop, format!("6.1: subset {}", r.subset));
            continue;
        }
        if r.target_tools.len() != 1 {
            bump(&mut drop, "6.1: not exactly one target tool");
            continue;
        }
        if r.question.trim().is_empty() {
            bump(&mut drop, "6.1: an empty question");
            continue;
        }
        let target = r.target_tools[0].as_str();
        let listed: Vec<usize> = r
            .tools
            .iter()
            .map(|x| t.index[&Tool::new(&x.name, &x.description).id])
            .collect();
        let gold: Vec<usize> = r
            .tools
            .iter()
            .zip(&listed)
            .filter(|(x, _)| super::collapse_ws(&x.name) == super::collapse_ws(target))
            .map(|(_, &k)| k)
            .collect();
        let Some(&g) = gold.first() else {
            bump(&mut drop, "6.1: the gold is not in its server's metadata");
            continue;
        };
        if gold.len() > 1 {
            bump(
                &mut read,
                "rows whose server lists the gold's name twice (the first taken)",
            );
        }
        if near_v2[g] {
            bump(
                &mut drop,
                "6.2 item 1: the gold within v2's cluster rule of a training or validation tool",
            );
            continue;
        }
        if v2.texts.contains(&r.question) || v2.norms.contains(&norm_text(&r.question)) {
            bump(
                &mut drop,
                "4.6: the question is a training or validation request of v2",
            );
            continue;
        }
        cands.push(Cand {
            row: i,
            gold: g,
            server_tools: listed.into_iter().filter(|&k| k != g).collect(),
        });
    }
    // 6.2 item 2: one row per normalised question, the first by sha256 of the text
    let key = |c: &Cand| {
        (
            sha256_hex(t.rows[c.row].question.as_bytes()),
            t.rows[c.row].id.clone(),
        )
    };
    cands.sort_by_key(key);
    let mut seen_q = HashSet::new();
    cands.retain(|c| {
        let keep = seen_q.insert(norm_text(&t.rows[c.row].question));
        if !keep {
            bump(&mut drop, "6.2 item 2: the same normalised question");
        }
        keep
    });
    // 6.2 item 3: at most 10 per gold cluster and 40 per server, the first by sha256 of the text
    let (mut per_cluster, mut per_server): (HashMap<usize, usize>, HashMap<&str, usize>) =
        (HashMap::new(), HashMap::new());
    let mut capped = Vec::new();
    for c in &cands {
        let cl = rep[c.gold];
        let sv = t.rows[c.row].server.as_str();
        if per_cluster.get(&cl).copied().unwrap_or(0) >= TEST_CAP {
            bump(&mut drop, "6.2 item 3: over the cap of 10 per gold cluster");
            continue;
        }
        if per_server.get(sv).copied().unwrap_or(0) >= SERVER_CAP {
            bump(&mut drop, "6.2 item 3: over the cap of 40 per server");
            continue;
        }
        *per_cluster.entry(cl).or_default() += 1;
        *per_server.entry(sv).or_default() += 1;
        capped.push(c.clone());
    }
    // 6.2 item 4: 2,000 drawn by seed, 20% removed
    let rng = |label: &str| Rng::derived(o.seed, label);
    let drawn: Vec<Cand> = slates::draw(
        capped.len(),
        TEST_SLATES,
        &mut rng(&format!("route/{NAME}")),
    )
    .into_iter()
    .map(|i| capped[i].clone())
    .collect();
    let removed = slates::removed_set(
        drawn.len(),
        REMOVED_SHARE,
        &mut rng(&format!("route/{NAME}/removed")),
    );

    // 6.2 item 5: the surface cells over all Toucan tools (v4: style, end, word quintile; v5:
    // the cell of docs/s6-route5-prereg.md 4.3 on the tool as shown, the quintiles of F2 and F3
    // over Toucan's tools, the levels L0 to L6 at 16 candidates)
    let mut sorted: Vec<usize> = t
        .tools
        .iter()
        .map(|x| surface::words(&x.tool.desc))
        .collect();
    sorted.sort_unstable();
    let cell: Vec<surface::Cell> = t
        .tools
        .iter()
        .map(|x| surface::cell(&x.tool.name, &x.tool.desc, &sorted))
        .collect();
    let mut classes5 = Value::Null;
    let (table, keys5) = match o.rule {
        Rule::V4 => (Vec::new(), Vec::new()),
        Rule::V5 => {
            let cuts = surface::Cuts::new(
                t.tools
                    .iter()
                    .map(|x| (x.tool.name.as_str(), x.tool.desc.as_str())),
            );
            let k: Vec<u16> = t
                .tools
                .iter()
                .map(|x| cuts.key(&x.tool.name, &x.tool.desc).index() as u16)
                .collect();
            classes5 = json!({"names": surface::Cuts::classes(&cuts.names),
                              "words": surface::Cuts::classes(&cuts.words)});
            (surface::level_table(), k)
        }
    };
    let name = o.rule.name();
    let mut main = Vec::new();
    let mut masked = Vec::new();
    let mut meta5: Vec<(u8, Vec<String>)> = Vec::new();
    let mut servers = Vec::new();
    let mut fill_from = Counts::new();
    let mut by_level = Counts::new();
    let mut c6 = v5::LevelCounts::default();
    for (j, c) in drawn.iter().enumerate() {
        let r = &t.rows[c.row];
        let g = c.gold;
        let ok = |x: usize| {
            !near_v2[x] && norm[x] != norm[g] && cos(x, g) < TEST_COSINE && rep[x] != rep[g]
        };
        let other: Vec<usize> = (0..n)
            .filter(|&x| !t.tools[x].servers.iter().any(|s| s == &r.server))
            .collect();
        let mut rg = rng(&format!("route/{NAME}/slate/{}", r.id));
        let s = match o.rule {
            Rule::V4 => {
                // the other servers' tools, by the gold's cell, then the nearest quintiles of its
                // style and end, then its style, then any
                let (style, end, q) = cell[g];
                let mut layers: Vec<Vec<usize>> = (0..5usize)
                    .map(|d| {
                        other
                            .iter()
                            .copied()
                            .filter(|&x| {
                                cell[x].0 == style && cell[x].1 == end && cell[x].2.abs_diff(q) == d
                            })
                            .collect()
                    })
                    .collect();
                layers.push(
                    other
                        .iter()
                        .copied()
                        .filter(|&x| cell[x].0 == style)
                        .collect(),
                );
                layers.push(other.clone());
                let lay: Vec<&[usize]> = layers.iter().map(Vec::as_slice).collect();
                let s = slates::test_slate_layers(
                    Some(g),
                    &c.server_tools,
                    &lay,
                    removed.contains(&j),
                    &mut rg,
                    &ok,
                );
                for &x in &s.tools {
                    if x == g || c.server_tools.contains(&x) {
                        continue;
                    }
                    let layer = layers
                        .iter()
                        .position(|v| v.contains(&x))
                        .unwrap_or(layers.len());
                    bump(
                        &mut fill_from,
                        match layer {
                            0 => "the gold's cell",
                            1..=4 => "its style and end, another quintile",
                            5 => "its style alone",
                            _ => "any other server's tool",
                        },
                    );
                }
                s
            }
            Rule::V5 => {
                // the tools of the gold's server first (exempt), then the other servers' tools
                // within the gold's cell at the lowest level that holds 16 of them
                let kf = |x: usize| (keys5[x], keys5[x]);
                let never = |_: usize| false;
                let mut cl = v5::Cell::new(
                    &other,
                    Some(keys5[g]),
                    &table,
                    &kf,
                    &never,
                    &ok,
                    v5::MIN_CANDIDATES,
                );
                let rem = removed.contains(&j);
                let lv = cl.level;
                let (s, level, drawn_f) = v5::test_like(
                    Some(g),
                    &c.server_tools,
                    &mut cl,
                    rem.then_some(lv),
                    &[],
                    &other,
                    &mut rg,
                    &ok,
                );
                for (&x, &d) in s.tools.iter().zip(&drawn_f) {
                    if d {
                        bump(&mut fill_from, format!("drawn from L{}", cl.level_of(x)));
                    }
                }
                bump(&mut by_level, format!("L{level}"));
                c6.add(level, drawn_f.iter().any(|&d| d), !rem);
                meta5.push((level, v5::roles(&drawn_f, s.gold)));
                s
            }
        };
        if s.tools.len() < slates::TEST_TOOLS {
            bump(&mut drop, "slates of fewer than 7 tools (kept)");
        }
        let kind = if removed.contains(&j) {
            "removed"
        } else {
            "test"
        };
        let text = |x: usize| t.tools[x].tool.option();
        main.push(record(name, &r.id, r, kind, &s, text, &ids));
        let mut rm = rng(&format!("route/{NAME}/masked/{}", r.id));
        let m = slates::masked(&s, |x| t.tools[x].tool.desc.clone(), &mut rm);
        masked.push(record(
            &format!("{name}-masked"),
            &r.id,
            r,
            "masked",
            &m,
            text,
            &ids,
        ));
        let label = format!("k{}", &sha256_hex(cluster_key[&rep[g]].as_bytes())[..12]);
        servers.push(json!({"uid": r.id, "server": r.server, "cluster": label}));
    }
    // C5 (4.4): no tool of a fresh file within v2's cluster rule of a training or validation tool
    for rec in &main {
        for id in &rec.tools {
            if near_v2[t.index[id]] {
                bail!(
                    "{}: tool {id} is within v2's cluster rule of a training or validation tool",
                    rec.uid
                );
            }
        }
    }
    let dir = fresh_dir(&o.t4);
    let stem = o.rule.stem();
    let mut hashes = BTreeMap::new();
    let mut records = BTreeMap::new();
    for (f, recs) in [
        (format!("{stem}_tools.jsonl"), &main),
        (format!("{stem}_tools_masked.jsonl"), &masked),
    ] {
        let h = match o.rule {
            Rule::V4 => build::write_once(&dir.join(&f), recs)?,
            Rule::V5 => build::write_once(&dir.join(&f), &with_levels(recs, &meta5))?,
        };
        hashes.insert(f.clone(), h);
        records.insert(f, recs.len());
    }
    let sf = format!("{stem}_tools.servers.jsonl");
    hashes.insert(sf.clone(), build::write_once(&dir.join(&sf), &servers)?);
    records.insert(sf, servers.len());

    // the probe copies' teacher job: every tool a fresh slate shows, its name and description
    // only (4.2; split `fresh-tools`; never a request, never a slate)
    let shown: BTreeSet<usize> = main
        .iter()
        .flat_map(|r| r.tools.iter().map(|id| t.index[id]))
        .collect();
    let items: Vec<RewriteItem> = shown
        .iter()
        .map(|&x| RewriteItem {
            id: ids[x].clone(),
            split: crate::teacher::FRESH_TOOLS.into(),
            name: t.tools[x].tool.name.clone(),
            description: t.tools[x].tool.desc.clone(),
        })
        .collect();
    write_jsonl(&items_path(&o.t4, "rewrite-description"), &items)?;

    // ---- the counts of 6.3
    let mut per_server: BTreeMap<&str, usize> = BTreeMap::new();
    let mut per_clust: BTreeMap<&str, usize> = BTreeMap::new();
    let mut subsets = Counts::new();
    for (c, s) in drawn.iter().zip(&servers) {
        *per_server.entry(s["server"].as_str().unwrap()).or_default() += 1;
        *per_clust.entry(s["cluster"].as_str().unwrap()).or_default() += 1;
        bump(&mut subsets, &t.rows[c.row].subset);
    }
    let mut sizes: Vec<usize> = per_server.values().copied().collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let top3: usize = sizes.iter().take(3).sum();
    let none = main
        .iter()
        .filter(|r| r.target.last() == Some(&1.0))
        .count();
    let mut counts = json!({
        "about": "s6-route v4 (docs/s6-route4-prereg.md, section 6): the fresh test files, counted before any system reads them",
        "seed": o.seed,
        "rows read by subset": read,
        "rows dropped, by filter": drop,
        "rows after the filters (before the draw)": capped.len(),
        "slates": main.len(),
        "fewer than 600 slates (6.3: v4 is not trained)": main.len() < MIN_SLATES,
        "servers": per_server.len(),
        "gold clusters": per_clust.len(),
        "share of the slates on the three largest servers": top3 as f64 / main.len().max(1) as f64,
        "slates of the ten largest servers": sizes.iter().take(10).collect::<Vec<_>>(),
        "none share": none as f64 / main.len().max(1) as f64,
        "slates by subset": subsets,
        "random fill by layer (6.2 item 5)": fill_from,
        "Toucan tools": n,
        "Toucan tools within v2's cluster rule (never shown)": near_v2.iter().filter(|&&x| x).count(),
        "probe job items (fresh-tools)": items.len(),
        "the rows of the SFT files (toucan-rows)": read_json(&o.t4.join("counts/toucan-rows.json")).unwrap_or(Value::Null),
    });
    if o.rule == Rule::V5 {
        let c = counts.as_object_mut().unwrap();
        c.insert("about".into(), json!("s6-route v5 (docs/s6-route5-prereg.md, section 6): the fresh test files, counted before any system reads them"));
        let small = c
            .remove("fewer than 600 slates (6.3: v4 is not trained)")
            .unwrap_or(Value::Null);
        c.insert(
            "fewer than 600 slates (6.3: v5 is not trained)".into(),
            small,
        );
        c.insert("random fill by layer (6.2 item 5)".into(), json!(fill_from));
        c.insert(
            "slates by level (6.2 item 5, reported)".into(),
            json!(by_level),
        );
        c.insert("classes of F2 and F3".into(), classes5);
        c.insert(
            "C6 (reported, not judged: 6.2)".into(),
            json!({"counts": c6, "share of the drawn slates at L5 or L6": c6.share_l5_l6(),
                   "above 0.10": c6.share_l5_l6() > v5::C6_BAR}),
        );
    }
    let manifest = json!({
        "about": format!("s6-route {}: the fresh test files ({}, section 6), hashed before any system reads them", o.rule.version(), o.rule.prereg()),
        "files": hashes, "records": records, "counts": counts,
    });
    write_json(&dir.join(o.rule.manifest()), &manifest)?;
    record_files(o.rule, o.record.as_deref(), &dir, &manifest)?;
    Ok(json!({"files": hashes, "records": records, "counts": counts}))
}

/// `test-files.json` and `counts.json` in the record folder (`runs/s6-route4`, `runs/s6-route5`).
fn record_files(rule: Rule, record: Option<&Path>, dir: &Path, manifest: &Value) -> Result<()> {
    let Some(rd) = record else { return Ok(()) };
    std::fs::create_dir_all(rd)?;
    let v = rule.version();
    let run = if rule == Rule::V4 {
        "s6-route4"
    } else {
        "s6-route5"
    };
    write_json(
        &rd.join("test-files.json"),
        &json!({
            "about": format!("s6-route {v}: the fresh test files as the build wrote them ({} 6.2, item 6); no system reads them before runs/{run}/FRESH-OPEN", rule.prereg()),
            "build": dir.join(rule.manifest()),
            "files": manifest["files"],
            "records": manifest["records"],
        }),
    )?;
    write_json(&rd.join("counts.json"), &manifest["counts"])?;
    Ok(())
}

/// Step 5: the probe copies of 6.4, from the teacher's rewrites of the fresh tools.
pub fn probes(t4: &Path, seed: u64, record: Option<&Path>, rule: Rule) -> Result<Value> {
    let dir = fresh_dir(t4);
    let items: Vec<RewriteItem> = read_lines(&items_path(t4, "rewrite-description"))?;
    let ans_path = answers_path(t4, "rewrite-description");
    let text: HashMap<String, String> = read_lines::<Description>(&ans_path)?
        .into_iter()
        .map(|d| (d.id, cut_description(&d.description)))
        .collect();
    let refused: HashSet<String> = {
        let p = crate::distill::refused_path(&ans_path);
        if p.exists() {
            read_lines::<crate::distill::Refused>(&p)?
                .into_iter()
                .map(|r| r.id)
                .collect()
        } else {
            HashSet::new()
        }
    };
    let missing = items
        .iter()
        .filter(|i| !text.contains_key(&i.id) && !refused.contains(&i.id))
        .count();
    if missing > 0 {
        bail!("{missing} fresh tools without a rewrite: BUILD={} scripts/route-teacher.sh rewrite-description loop", t4.display());
    }
    let stem = rule.stem();
    // v5's files carry the level and the roles of every slate (Record5); v4's do not
    let read5 = |f: &str| -> Result<Vec<Record5>> {
        Ok(match rule {
            Rule::V4 => read_lines::<RouteRecord>(&dir.join(f))?
                .into_iter()
                .map(|r| Record5 {
                    r,
                    cell_level: 0,
                    roles: Vec::new(),
                    from: None,
                })
                .collect(),
            Rule::V5 => read_lines(&dir.join(f))?,
        })
    };
    let main5 = read5(&format!("{stem}_tools.jsonl"))?;
    let masked5 = read5(&format!("{stem}_tools_masked.jsonl"))?;
    let mut counts = Counts::new();
    let (mut a, mut b): (Vec<Record5>, Vec<Record5>) = (Vec::new(), Vec::new());
    for (r5, m5) in main5.iter().zip(&masked5) {
        let (r, m) = (&r5.r, &m5.r);
        if r.request != m.request {
            bail!("{stem}_tools and its masked copy do not hold the same slates");
        }
        let g = r
            .target
            .iter()
            .position(|&x| x == 1.0)
            .unwrap_or(r.tools.len());
        if g >= r.tools.len() {
            continue;
        }
        let cand: Vec<usize> = (0..r.tools.len())
            .filter(|&i| i != g && text.contains_key(&r.tools[i]))
            .collect();
        if cand.is_empty() {
            bump(
                &mut counts,
                "tool-gold slates without a rewritten distractor (left out)",
            );
            continue;
        }
        let d =
            cand[Rng::derived(seed, &format!("route/{NAME}-rw/{}", r.request)).below(cand.len())];
        for (src, dst) in [(r5, &mut a), (m5, &mut b)] {
            let mut x = src.clone();
            let name = sky_infer::route::option_name(&src.r.opts[d]).to_string();
            x.r.opts[d] = option_text(&name, &text[&src.r.tools[d]]);
            x.r.follow = Some(d);
            x.r.src = format!("{}-rw", src.r.src);
            dst.push(x);
        }
        bump(
            &mut counts,
            "tool-gold slates with one distractor rewritten",
        );
    }
    bump(
        &mut counts,
        format!("fresh tools refused by moderation: {}", refused.len()),
    );
    let mpath = dir.join(rule.manifest());
    let mut manifest = read_json(&mpath)?;
    for (f, recs) in [
        (format!("{stem}_tools_rw.jsonl"), &a),
        (format!("{stem}_tools_masked_rw.jsonl"), &b),
    ] {
        let h = match rule {
            Rule::V4 => {
                let v: Vec<&RouteRecord> = recs.iter().map(|x| &x.r).collect();
                build::write_once(&dir.join(&f), &v)?
            }
            Rule::V5 => build::write_once(&dir.join(&f), recs)?,
        };
        manifest["files"][&f] = json!(h);
        manifest["records"][&f] = json!(recs.len());
    }
    manifest["counts"]["probe copies (6.4)"] = json!(counts);
    write_json(&mpath, &manifest)?;
    record_files(rule, record, &dir, &manifest)?;
    Ok(json!({"files": manifest["files"], "probe copies": counts}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::route::build::tests::{fixture, run};

    /// A `.npy` file of float32 rows (as the build's tests write one).
    fn write_npy(path: &Path, rows: &[Vec<f32>]) {
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, {}), }}",
            rows.len(),
            rows[0].len()
        );
        let mut h = header;
        while !(10 + h.len() + 1).is_multiple_of(64) {
            h.push(' ');
        }
        h.push('\n');
        let mut buf = b"\x93NUMPY\x01\x00".to_vec();
        buf.extend_from_slice(&(h.len() as u16).to_le_bytes());
        buf.extend_from_slice(h.as_bytes());
        for r in rows {
            for x in r {
                buf.extend_from_slice(&x.to_le_bytes());
            }
        }
        std::fs::write(path, buf).unwrap();
    }

    /// An invented Toucan: 60 servers of 6 tools each in mixed styles, one row per tool and
    /// subset, plus rows that the filters drop; one tool whose name is a v2 tool's.
    fn invented(t4: &Path) -> Vec<ToucanRow> {
        let mut rows = Vec::new();
        for s in 0..60 {
            let server = format!("server-{s}");
            let tools: Vec<RawTool> = (0..6)
                .map(|k| {
                    let name = match (s + k) % 4 {
                        0 => format!("mcp_tool_{s}_{k}"),
                        1 => format!("mcpTool{s}x{k}"),
                        2 => format!("Mcp Tool {s} {k}"),
                        _ => format!("McpTool{s}X{k}"),
                    };
                    let words = 3 + (s * 7 + k * 3) % 12;
                    let mut d: Vec<String> = (0..words).map(|w| format!("w{s}k{k}n{w}")).collect();
                    d[0] = "Lists".into();
                    let mut d = d.join(" ");
                    if (s + k) % 2 == 0 {
                        d.push('.');
                    }
                    RawTool {
                        name,
                        description: d,
                    }
                })
                .collect();
            for k in 0..6 {
                for subset in [
                    "single-turn-original",
                    "single-turn-diversify",
                    "irrelevant",
                    "multi-turn",
                ] {
                    rows.push(ToucanRow {
                        id: format!("toucan:{s}-{k}-{subset}"),
                        subset: subset.into(),
                        question: format!("please run tool {k} of server {s} for {subset}"),
                        server: server.clone(),
                        tools: tools.clone(),
                        target_tools: vec![tools[k].name.clone()],
                    });
                }
            }
            // two targets, and a gold that the metadata does not list
            rows.push(ToucanRow {
                id: format!("toucan:{s}-two"),
                subset: "single-turn-original".into(),
                question: format!("two tools of server {s}"),
                server: server.clone(),
                tools: tools.clone(),
                target_tools: vec![tools[0].name.clone(), tools[1].name.clone()],
            });
            rows.push(ToucanRow {
                id: format!("toucan:{s}-missing"),
                subset: "single-turn-original".into(),
                question: format!("a tool server {s} does not list"),
                server: server.clone(),
                tools: tools.clone(),
                target_tools: vec!["not_listed".into()],
            });
        }
        // a server whose tool has the normalised name of a v2 tool (get_thing0_v0)
        let v2like = RawTool {
            name: "thing0_v0".into(),
            description: "Does a thing.".into(),
        };
        rows.push(ToucanRow {
            id: "toucan:v2like".into(),
            subset: "single-turn-original".into(),
            question: "do the v2 thing".into(),
            server: "server-v2".into(),
            tools: vec![
                v2like.clone(),
                RawTool {
                    name: "other_v2_tool".into(),
                    description: "Other.".into(),
                },
            ],
            target_tools: vec![v2like.name.clone()],
        });
        std::fs::create_dir_all(t4.join("stage")).unwrap();
        write_jsonl(&t4.join("stage/rows.jsonl"), &rows).unwrap();
        rows
    }

    /// Embeddings of the staged texts: random, away from v2's.
    fn embed(t4: &Path) {
        let texts: Vec<Value> = read_lines(&t4.join("stage/texts.jsonl")).unwrap();
        let mut rng = Rng::new(5);
        let rows: Vec<Vec<f32>> = texts
            .iter()
            .map(|_| (0..16).map(|_| rng.f64() as f32 - 0.5).collect())
            .collect();
        std::fs::create_dir_all(t4.join("emb")).unwrap();
        write_npy(&t4.join("emb/minilm.npy"), &rows);
        write_json(
            &t4.join("emb/meta.json"),
            &json!({"texts_sha256": sha256_file(&t4.join("stage/texts.jsonl")).unwrap()}),
        )
        .unwrap();
    }

    #[test]
    fn an_invented_fresh_test() {
        let root = std::env::temp_dir().join(format!("sky-route-toucan-{}", std::process::id()));
        let v2 = root.join("v2");
        fixture(&v2);
        run(&v2, 7);
        let t4 = root.join("t4");
        invented(&t4);
        stage(&t4).unwrap();
        embed(&t4);
        let o = TestOptions {
            v2: v2.clone(),
            t4: t4.clone(),
            seed: T4_SEED,
            record: Some(root.join("runs")),
            rule: Rule::V4,
        };
        let rep = build_test(&o).unwrap();
        let c = &rep["counts"];
        let dropped = c["rows dropped, by filter"].as_object().unwrap();
        assert_eq!(dropped["6.1: subset irrelevant"], 360);
        assert_eq!(dropped["6.1: not exactly one target tool"], 60);
        assert_eq!(dropped["6.1: the gold is not in its server's metadata"], 60);
        assert_eq!(
            dropped
                ["6.2 item 1: the gold within v2's cluster rule of a training or validation tool"],
            1
        );
        let dir = fresh_dir(&t4);
        let main: Vec<RouteRecord> = read_lines(&dir.join("t4_tools.jsonl")).unwrap();
        let masked: Vec<RouteRecord> = read_lines(&dir.join("t4_tools_masked.jsonl")).unwrap();
        // the two kept subsets, 720 rows, at most 10 per cluster and 40 per server
        assert_eq!(main.len(), 720);
        let none = main
            .iter()
            .filter(|r| r.target.last() == Some(&1.0))
            .count();
        assert_eq!(none, (main.len() as f64 * 0.2).round() as usize);
        let tools: Vec<StagedTool> = read_lines(&t4.join("stage/tools.jsonl")).unwrap();
        let by: HashMap<&str, &StagedTool> =
            tools.iter().map(|x| (x.tool.id.as_str(), x)).collect();
        let rows: Vec<ToucanRow> = read_lines(&t4.join("stage/rows.jsonl")).unwrap();
        let row_of: HashMap<&str, &ToucanRow> = rows.iter().map(|r| (r.id.as_str(), r)).collect();
        let mut sorted: Vec<usize> = tools.iter().map(|x| surface::words(&x.tool.desc)).collect();
        sorted.sort_unstable();
        let (mut same, mut fill) = (0, 0);
        for (r, m) in main.iter().zip(&masked) {
            assert_eq!(r.opts.len(), slates::TEST_TOOLS + 1);
            assert_eq!(r.opts.last().unwrap(), super::super::NONE_OPTION);
            assert!(m.opts[..7].iter().all(|o| o.starts_with("tool_")));
            assert_eq!(r.split, SPLIT);
            let src = row_of[r.request.as_str()];
            assert!(r
                .tools
                .iter()
                .all(|id| by[id.as_str()].tool.name != "thing0_v0"));
            let gname = &src.target_tools[0];
            let gold = tools
                .iter()
                .find(|x| &x.tool.name == gname && x.servers.contains(&src.server))
                .unwrap();
            let gc = surface::cell(&gold.tool.name, &gold.tool.desc, &sorted);
            for id in &r.tools {
                let x = by[id.as_str()];
                if x.servers.contains(&src.server) {
                    continue;
                }
                fill += 1;
                let xc = surface::cell(&x.tool.name, &x.tool.desc, &sorted);
                same += (xc.0 == gc.0 && xc.1 == gc.1) as usize;
            }
        }
        // the fill of other servers matches the gold's name style and final period
        assert!(fill > 0 && same == fill, "{same} of {fill}");
        // written once, and the counts recorded
        assert!(root.join("runs/test-files.json").exists());
        let again = build_test(&o).unwrap();
        assert_eq!(again["files"], rep["files"]);
        let o2 = TestOptions { seed: 99, ..o };
        assert!(build_test(&o2).is_err());
        // the probe job: a tool's name and description only, split fresh-tools, accepted by the
        // teacher's guard
        let ip = items_path(&t4, "rewrite-description");
        let items: Vec<Value> = read_lines(&ip).unwrap();
        assert!(!items.is_empty());
        for it in &items {
            let keys: BTreeSet<&str> = it.as_object().unwrap().keys().map(String::as_str).collect();
            assert_eq!(keys, BTreeSet::from(["id", "split", "name", "description"]));
        }
        crate::teacher::route_todo(&ip, &answers_path(&t4, "rewrite-description")).unwrap();
        // the probes need every answer
        assert!(probes(&t4, T4_SEED, None, Rule::V4).is_err());
        let ans: Vec<Value> = items
            .iter()
            .map(|i| json!({"id": i["id"], "description": format!("Returns {}.", i["name"].as_str().unwrap())}))
            .collect();
        write_jsonl(&answers_path(&t4, "rewrite-description"), &ans).unwrap();
        probes(&t4, T4_SEED, Some(&root.join("runs")), Rule::V4).unwrap();
        let rw: Vec<RouteRecord> = read_lines(&dir.join("t4_tools_rw.jsonl")).unwrap();
        let mrw: Vec<RouteRecord> = read_lines(&dir.join("t4_tools_masked_rw.jsonl")).unwrap();
        assert_eq!(rw.len(), main.len() - none);
        assert_eq!(rw.len(), mrw.len());
        for (x, y) in rw.iter().zip(&mrw) {
            let d = x.follow.unwrap();
            assert_eq!(y.follow, Some(d));
            assert!(x.opts[d].ends_with('.') && x.opts[d].contains(": Returns "));
        }
        let m = read_json(&dir.join("FRESH4.json")).unwrap();
        assert!(m["files"]["t4_tools_rw.jsonl"].is_string());
        std::fs::remove_dir_all(root).ok();
    }

    /// s6-route v5 (`docs/s6-route5-prereg.md` 6): the same rows, item 5 by v5's cell; files
    /// `t5_*` with each slate's level and its options' roles; the probe copies keep them.
    #[test]
    fn an_invented_fresh_test_by_v5s_cell() {
        let root = std::env::temp_dir().join(format!("sky-route-toucan5-{}", std::process::id()));
        let v2 = root.join("v2");
        fixture(&v2);
        run(&v2, 7);
        let t5 = root.join("t5");
        invented(&t5);
        stage(&t5).unwrap();
        embed(&t5);
        let o = TestOptions {
            v2: v2.clone(),
            t4: t5.clone(),
            seed: T4_SEED,
            record: Some(root.join("runs")),
            rule: Rule::V5,
        };
        let rep = build_test(&o).unwrap();
        let c = &rep["counts"];
        assert!(c["slates by level (6.2 item 5, reported)"].is_object());
        assert!(
            c["C6 (reported, not judged: 6.2)"]["counts"]["drawn slates"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(c["fewer than 600 slates (6.3: v5 is not trained)"].is_boolean());
        let dir = fresh_dir(&t5);
        assert!(dir.join("FRESH5.json").exists() && !dir.join("FRESH4.json").exists());
        let main: Vec<Record5> = read_lines(&dir.join("t5_tools.jsonl")).unwrap();
        let masked: Vec<Record5> = read_lines(&dir.join("t5_tools_masked.jsonl")).unwrap();
        assert_eq!(main.len(), 720);
        let none = main
            .iter()
            .filter(|r| r.r.target.last() == Some(&1.0))
            .count();
        assert_eq!(none, (main.len() as f64 * 0.2).round() as usize);
        let tools: Vec<StagedTool> = read_lines(&t5.join("stage/tools.jsonl")).unwrap();
        let by: HashMap<&str, &StagedTool> =
            tools.iter().map(|x| (x.tool.id.as_str(), x)).collect();
        let rows: Vec<ToucanRow> = read_lines(&t5.join("stage/rows.jsonl")).unwrap();
        let row_of: HashMap<&str, &ToucanRow> = rows.iter().map(|r| (r.id.as_str(), r)).collect();
        let cuts = surface::Cuts::new(
            tools
                .iter()
                .map(|x| (x.tool.name.as_str(), x.tool.desc.as_str())),
        );
        let (mut l0, mut drawn) = (0, 0);
        for (r, m) in main.iter().zip(&masked) {
            assert_eq!(r.r.src, "t5-tools");
            assert_eq!((r.cell_level, &r.roles), (m.cell_level, &m.roles));
            assert_eq!(r.r.opts.len(), slates::TEST_TOOLS + 1);
            let src = row_of[r.r.request.as_str()];
            let gname = &src.target_tools[0];
            let gold = tools
                .iter()
                .find(|x| &x.tool.name == gname && x.servers.contains(&src.server))
                .unwrap();
            let gk = cuts.key(&gold.tool.name, &gold.tool.desc);
            for (id, role) in r.r.tools.iter().zip(&r.roles) {
                let x = by[id.as_str()];
                // the tools of the gold's server are exempt; the others drawn within the cell
                let own = x.servers.contains(&src.server);
                if role == v5::RIGHT {
                    continue;
                }
                assert_eq!(role == v5::EXEMPT, own, "{}", r.r.uid);
                if role == v5::DRAWN && r.cell_level < surface::L6 {
                    let xk = cuts.key(&x.tool.name, &x.tool.desc);
                    assert!(surface::matches(gk, xk, r.cell_level), "{}", r.r.uid);
                    drawn += 1;
                }
            }
            l0 += (r.cell_level == 0) as usize;
        }
        assert!(drawn > 100 && l0 > 0, "{drawn} {l0}");
        // written once
        assert_eq!(build_test(&o).unwrap()["files"], rep["files"]);
        // the probe copies keep the level and the roles
        let ip = items_path(&t5, "rewrite-description");
        let items: Vec<Value> = read_lines(&ip).unwrap();
        let ans: Vec<Value> = items
            .iter()
            .map(|i| json!({"id": i["id"], "description": format!("Returns {}.", i["name"].as_str().unwrap())}))
            .collect();
        write_jsonl(&answers_path(&t5, "rewrite-description"), &ans).unwrap();
        probes(&t5, T4_SEED, Some(&root.join("runs")), Rule::V5).unwrap();
        let rw: Vec<Record5> = read_lines(&dir.join("t5_tools_rw.jsonl")).unwrap();
        assert_eq!(rw.len(), main.len() - none);
        assert!(rw
            .iter()
            .all(|r| r.r.follow.is_some() && r.roles.len() == r.r.tools.len()));
        let m = read_json(&dir.join("FRESH5.json")).unwrap();
        assert!(m["files"]["t5_tools_masked_rw.jsonl"].is_string());
        let tf = read_json(&root.join("runs/test-files.json")).unwrap();
        assert!(tf["about"]
            .as_str()
            .unwrap()
            .contains("runs/s6-route5/FRESH-OPEN"));
        std::fs::remove_dir_all(root).ok();
    }

    #[test]
    fn the_parquet_files_are_hash_checked_against_the_manifest() {
        let d = std::env::temp_dir().join(format!("sky-route-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("train-00000-of-00003.parquet");
        std::fs::write(&p, b"parquet bytes").unwrap();
        let m = d.join("MANIFEST4.json");
        let entry = |bytes: u64, sha: &str| json!({"files": [{"file": "SFT/train-00000-of-00003.parquet", "bytes": bytes, "sha256": sha}]});
        let h = sha256_file(&p).unwrap();
        write_json(&m, &entry(13, &h)).unwrap();
        let c = check_manifest(std::slice::from_ref(&p), &m).unwrap();
        assert_eq!(c["files"]["train-00000-of-00003.parquet"]["sha256"], h);
        write_json(&m, &entry(13, "00")).unwrap();
        assert!(check_manifest(std::slice::from_ref(&p), &m).is_err());
        write_json(&m, &entry(12, &h)).unwrap();
        assert!(check_manifest(std::slice::from_ref(&p), &m).is_err());
        let q = d.join("other.parquet");
        std::fs::write(&q, b"x").unwrap();
        write_json(&m, &entry(13, &h)).unwrap();
        assert!(check_manifest(&[p, q], &m).is_err());
        assert!(check_manifest(&[], &m).is_err());
        std::fs::remove_dir_all(d).ok();
    }

    /// The bytes of v4's fresh test of the invented Toucan (the four test files, the servers
    /// file and the probe job's items), frozen when s6-route v5's code was written
    /// (`docs/s6-route5-prereg.md` 6: v5 adds `--rule v5` and changes nothing v4's rule builds).
    #[test]
    fn t4_bytes_are_frozen() {
        let root =
            std::env::temp_dir().join(format!("sky-route-toucan-frozen-{}", std::process::id()));
        let v2 = root.join("v2");
        fixture(&v2);
        run(&v2, 7);
        let t4 = root.join("t4");
        invented(&t4);
        stage(&t4).unwrap();
        embed(&t4);
        build_test(&TestOptions {
            v2: v2.clone(),
            t4: t4.clone(),
            seed: T4_SEED,
            record: None,
            rule: Rule::V4,
        })
        .unwrap();
        let items: Vec<Value> = read_lines(&items_path(&t4, "rewrite-description")).unwrap();
        let ans: Vec<Value> = items
            .iter()
            .map(|i| json!({"id": i["id"], "description": format!("Returns {}.", i["name"].as_str().unwrap())}))
            .collect();
        write_jsonl(&answers_path(&t4, "rewrite-description"), &ans).unwrap();
        probes(&t4, T4_SEED, None, Rule::V4).unwrap();
        let mut got = Vec::new();
        for f in [
            "fresh/t4_tools.jsonl",
            "fresh/t4_tools_masked.jsonl",
            "fresh/t4_tools.servers.jsonl",
            "fresh/t4_tools_rw.jsonl",
            "fresh/t4_tools_masked_rw.jsonl",
            "stage/tools.jsonl",
            "stage/texts.jsonl",
            "teacher/rewrite-description.items.jsonl",
        ] {
            got.push(format!("{f} {}", sha256_file(&t4.join(f)).unwrap()));
        }
        std::fs::remove_dir_all(&root).ok();
        println!("FROZEN\n{}", got.join("\n"));
        assert_eq!(got.join("\n"), T4_FROZEN.trim());
    }

    const T4_FROZEN: &str = "
fresh/t4_tools.jsonl 3eb79c7eb636680a9b6086d8cc6eb4b112f8b798dece00a23b7c5dd491e1ae82
fresh/t4_tools_masked.jsonl 63acf4c94156b8c48f3c5a3974bb1927cc69b4ae3a484edf9ccf221ab62453de
fresh/t4_tools.servers.jsonl b5b44e802ed014e9518898541969872dd2129d5759beb9ad36b24877b198d049
fresh/t4_tools_rw.jsonl 9f905902cbb412f01ef9ccbc4675adc8bdfe95d6c4dc333f26e5cfe39c87144e
fresh/t4_tools_masked_rw.jsonl 3356023d368083b566503af993b297d12a0e954794b6900e91a54e7c80328ed5
stage/tools.jsonl 2fce728d86fa647191ad95027a7a36ea55a1748abb08d19f1c2c9d78d942d1f9
stage/texts.jsonl 5c1f263a3932708348a65599ae856f8efe57d7cb46e26ce92e437beea1c5b289
teacher/rewrite-description.items.jsonl e11ff73803ea64baf3d9cbcf7bc9e7ae038f393e3848d8f381deb055e42e569c
";

    fn sft_row(uuid: &str, subset: &str, targets: &str, tools: Value) -> Value {
        json!({"uuid": uuid, "subset_name": subset, "question": format!("question {uuid}"),
               "target_tools": targets, "tools": tools.to_string(), "messages": "[]"})
    }

    fn tool(name: &str, desc: &str) -> Value {
        json!({"type": "function", "function": {"name": name, "description": desc,
               "parameters": {"type": "object", "properties": {}}}})
    }

    #[test]
    fn sft_rows_are_parsed_by_amendment_1() {
        let list = json!([
            tool("time-mcp-current_time", "Now."),
            tool("time-mcp-convert_time", "Converts.")
        ]);
        let r = parse_row(&sft_row(
            "u1",
            "single-turn-original",
            "convert_time",
            list.clone(),
        ))
        .unwrap();
        let Parsed::Row(r) = r else { panic!("{r:?}") };
        assert_eq!(
            (r.id.as_str(), r.server.as_str()),
            ("toucan:u1", "time-mcp")
        );
        assert_eq!(
            r.tools[1],
            RawTool {
                name: "convert_time".into(),
                description: "Converts.".into()
            }
        );
        assert_eq!(r.target_tools, vec!["convert_time".to_string()]);
        for (row, why) in [
            (
                sft_row("u2", "irrelevant", "convert_time", list.clone()),
                "6.1: another subset",
            ),
            (
                sft_row(
                    "u3",
                    "single-turn-diversify",
                    "convert_time, current_time",
                    list.clone(),
                ),
                "6.1: not exactly one target tool",
            ),
            (
                sft_row("u4", "single-turn-original", "missing", list.clone()),
                "no tool of its list is <server>-<target>",
            ),
            (
                sft_row(
                    "u5",
                    "single-turn-original",
                    "time-mcp::convert_time",
                    list.clone(),
                ),
                "6.1: a target written Server::Tool",
            ),
        ] {
            assert_eq!(parse_row(&row).unwrap(), Parsed::Out(why));
        }
        // two tools ending with the target
        let two = json!([tool("a-x", "A."), tool("b-a-x", "B.")]);
        assert_eq!(
            parse_row(&sft_row("u6", "single-turn-original", "x", two)).unwrap(),
            Parsed::Out("several tools of its list end with -<target>")
        );
        // the loud failures: a tool without the server's prefix, `tools` not a list of functions
        let mixed = json!([
            tool("time-mcp-convert_time", "C."),
            tool("other-thing", "O.")
        ]);
        assert!(parse_row(&sft_row(
            "u7",
            "single-turn-original",
            "convert_time",
            mixed
        ))
        .is_err());
        let bad = json!({"uuid": "u8", "subset_name": "single-turn-original", "question": "q",
                         "target_tools": "x", "tools": "[{\"name\": \"s-x\"}]"});
        assert!(parse_row(&bad).is_err());
        let notjson = json!({"uuid": "u9", "subset_name": "single-turn-original", "question": "q",
                             "target_tools": "x", "tools": "not json"});
        assert!(parse_row(&notjson).is_err());
    }

    /// A parquet file of the SFT config's six string columns.
    fn write_sft(path: &Path, rows: &[Value]) {
        use parquet::data_type::{ByteArray, ByteArrayType};
        use parquet::file::properties::WriterProperties;
        use parquet::file::writer::SerializedFileWriter;
        use std::sync::Arc;
        let schema = Arc::new(
            parquet::schema::parser::parse_message_type(
                "message schema { REQUIRED BYTE_ARRAY uuid (UTF8); REQUIRED BYTE_ARRAY subset_name (UTF8); \
                 REQUIRED BYTE_ARRAY question (UTF8); REQUIRED BYTE_ARRAY target_tools (UTF8); \
                 REQUIRED BYTE_ARRAY tools (UTF8); REQUIRED BYTE_ARRAY messages (UTF8); }",
            )
            .unwrap(),
        );
        let f = std::fs::File::create(path).unwrap();
        let mut w =
            SerializedFileWriter::new(f, schema, Arc::new(WriterProperties::builder().build()))
                .unwrap();
        let mut rg = w.next_row_group().unwrap();
        for col in SFT_COLUMNS {
            let mut c = rg.next_column().unwrap().unwrap();
            let vals: Vec<ByteArray> = rows
                .iter()
                .map(|r| ByteArray::from(r[col].as_str().unwrap()))
                .collect();
            c.typed::<ByteArrayType>()
                .write_batch(&vals, None, None)
                .unwrap();
            c.close().unwrap();
        }
        rg.close().unwrap();
        w.close().unwrap();
    }

    #[test]
    fn sft_parquet_files_are_read() {
        let d = std::env::temp_dir().join(format!("sky-route-sft-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let list = json!([tool("srv-a", "Does a."), tool("srv-b", "Does b.")]);
        let mut rows = vec![
            sft_row("u1", "single-turn-original", "a", list.clone()),
            sft_row("u2", "single-turn-diversify", "b", list.clone()),
            sft_row("u3", "multi-turn", "a", list.clone()),
            sft_row("u4", "single-turn-original", "a, b", list.clone()),
        ];
        let p = d.join("train.parquet");
        write_sft(&p, &rows);
        let sch = schema(std::slice::from_ref(&p)).unwrap();
        assert_eq!(sch["uuids"]["distinct"], 4);
        let c = super::rows(std::slice::from_ref(&p), &d).unwrap();
        assert_eq!(c["rows kept"], 2);
        assert_eq!(c["servers of the kept rows"], 1);
        let kept: Vec<ToucanRow> = read_lines(&d.join("stage/rows.jsonl")).unwrap();
        assert_eq!(kept[1].tools[1].name, "b");
        // a repeated uuid fails loudly
        rows[1]["uuid"] = json!("u1");
        write_sft(&p, &rows);
        assert!(super::rows(std::slice::from_ref(&p), &d).is_err());
        assert!(super::rows(&[], &d).is_err());
        std::fs::remove_dir_all(d).ok();
    }
}
