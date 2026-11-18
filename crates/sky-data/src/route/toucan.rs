//! The fresh test of s6-route v4 (`docs/s6-route4-prereg.md`, section 6): MCP tools of
//! `Agent-Ark/Toucan-1.5M`, a family no model of this project has seen, built so that the
//! surface cue of D7 cannot be in it.
//!
//! 1. [`rows`] (`sky-data route toucan-rows --input <SFT parquet files> --out <t4 dir>`): the
//!    parquet files of the SFT config read into `<t4>/stage/rows.jsonl`, one [`ToucanRow`] per
//!    row. **Stopped** (see [`rows`]): the SFT config has no `metadata` column, so the row's
//!    server and the gold's description cannot be read as 6.1 says; the step checks the files'
//!    columns, counts the rows by subset and stops with the difference. Everything after it
//!    reads `rows.jsonl` in the format of [`ToucanRow`] and runs once an amendment says where a
//!    row's server and tool list come from.
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

/// Step 1: the SFT parquet files read into `rows.jsonl`. **Stopped**: the pre-registration (6.1)
/// reads the gold "as listed in the row's server metadata" and caps the rows per server (6.2,
/// item 3); the SFT config's card declares six string columns (`uuid`, `subset_name`, `question`,
/// `target_tools`, `tools`, `messages`), no `metadata` and no server. This step checks the
/// columns of each file, counts its rows by subset, and stops with the difference; the parser of
/// a row is written once an amendment says which column holds a row's server and its tools.
pub fn rows(inputs: &[PathBuf], out: &Path) -> Result<Value> {
    let mut by_subset = Counts::new();
    let mut files = BTreeMap::new();
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
        crate::hf::for_each_parquet_row(p, &["subset_name"], |v| {
            n += 1;
            let s = v["subset_name"].as_str().unwrap_or("<not a string>");
            bump(&mut by_subset, s);
            Ok(())
        })?;
        files.insert(p.display().to_string(), n);
    }
    let counts = json!({"rows by file": files, "rows by subset": by_subset});
    write_json(&out.join("counts/toucan-rows.json"), &counts)?;
    bail!(
        "stopped (docs/s6-route4-prereg.md 6.1): the SFT config has no `metadata` column, so a row's \
         server and the gold's description \"as listed in the row's server metadata\" cannot be read, \
         nor the cap of 40 per server applied; its columns are {SFT_COLUMNS:?}. An amendment must say \
         which column holds a row's server and its tools (counts written to {}): {counts}",
        out.join("counts/toucan-rows.json").display()
    )
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
    /// A folder for `test-files.json` and `counts.json` (`runs/s6-route4`).
    pub record: Option<PathBuf>,
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

    // 6.2 item 5: the surface cells over all Toucan tools
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
    let mut main = Vec::new();
    let mut masked = Vec::new();
    let mut servers = Vec::new();
    let mut fill_from = Counts::new();
    for (j, c) in drawn.iter().enumerate() {
        let r = &t.rows[c.row];
        let g = c.gold;
        let ok = |x: usize| {
            !near_v2[x] && norm[x] != norm[g] && cos(x, g) < TEST_COSINE && rep[x] != rep[g]
        };
        // the other servers' tools, by the gold's cell, then the nearest quintiles of its style
        // and end, then its style, then any
        let other: Vec<usize> = (0..n)
            .filter(|&x| !t.tools[x].servers.iter().any(|s| s == &r.server))
            .collect();
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
        let mut rg = rng(&format!("route/{NAME}/slate/{}", r.id));
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
        if s.tools.len() < slates::TEST_TOOLS {
            bump(&mut drop, "slates of fewer than 7 tools (kept)");
        }
        let kind = if removed.contains(&j) {
            "removed"
        } else {
            "test"
        };
        let text = |x: usize| t.tools[x].tool.option();
        main.push(record(NAME, &r.id, r, kind, &s, text, &ids));
        let mut rm = rng(&format!("route/{NAME}/masked/{}", r.id));
        let m = slates::masked(&s, |x| t.tools[x].tool.desc.clone(), &mut rm);
        masked.push(record(
            &format!("{NAME}-masked"),
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
    let mut hashes = BTreeMap::new();
    let mut records = BTreeMap::new();
    for (f, recs) in [
        ("t4_tools.jsonl", &main),
        ("t4_tools_masked.jsonl", &masked),
    ] {
        hashes.insert(f.to_string(), build::write_once(&dir.join(f), recs)?);
        records.insert(f.to_string(), recs.len());
    }
    hashes.insert(
        "t4_tools.servers.jsonl".into(),
        build::write_once(&dir.join("t4_tools.servers.jsonl"), &servers)?,
    );
    records.insert("t4_tools.servers.jsonl".into(), servers.len());

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
    let counts = json!({
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
    });
    let manifest = json!({
        "about": "s6-route v4: the fresh test files (docs/s6-route4-prereg.md, section 6), hashed before any system reads them",
        "files": hashes, "records": records, "counts": counts,
    });
    write_json(&dir.join("FRESH4.json"), &manifest)?;
    record_files(o.record.as_deref(), &dir, &manifest)?;
    Ok(json!({"files": hashes, "records": records, "counts": counts}))
}

/// `test-files.json` and `counts.json` in the record folder (`runs/s6-route4`).
fn record_files(record: Option<&Path>, dir: &Path, manifest: &Value) -> Result<()> {
    let Some(rd) = record else { return Ok(()) };
    std::fs::create_dir_all(rd)?;
    write_json(
        &rd.join("test-files.json"),
        &json!({
            "about": "s6-route v4: the fresh test files as the build wrote them (docs/s6-route4-prereg.md 6.2, item 6); no system reads them before runs/s6-route4/FRESH-OPEN",
            "build": dir.join("FRESH4.json"),
            "files": manifest["files"],
            "records": manifest["records"],
        }),
    )?;
    write_json(&rd.join("counts.json"), &manifest["counts"])?;
    Ok(())
}

/// Step 5: the probe copies of 6.4, from the teacher's rewrites of the fresh tools.
pub fn probes(t4: &Path, seed: u64, record: Option<&Path>) -> Result<Value> {
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
    let main: Vec<RouteRecord> = read_lines(&dir.join("t4_tools.jsonl"))?;
    let masked: Vec<RouteRecord> = read_lines(&dir.join("t4_tools_masked.jsonl"))?;
    let mut counts = Counts::new();
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for (r, m) in main.iter().zip(&masked) {
        if r.request != m.request {
            bail!("t4_tools and its masked copy do not hold the same slates");
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
        for (src, dst) in [(r, &mut a), (m, &mut b)] {
            let mut x = src.clone();
            let name = sky_infer::route::option_name(&src.opts[d]).to_string();
            x.opts[d] = option_text(&name, &text[&src.tools[d]]);
            x.follow = Some(d);
            x.src = format!("{}-rw", src.src);
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
    let mpath = dir.join("FRESH4.json");
    let mut manifest = read_json(&mpath)?;
    for (f, recs) in [("t4_tools_rw.jsonl", &a), ("t4_tools_masked_rw.jsonl", &b)] {
        manifest["files"][f] = json!(build::write_once(&dir.join(f), recs)?);
        manifest["records"][f] = json!(recs.len());
    }
    manifest["counts"]["probe copies (6.4)"] = json!(counts);
    write_json(&mpath, &manifest)?;
    record_files(record, &dir, &manifest)?;
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
        assert!(probes(&t4, T4_SEED, None).is_err());
        let ans: Vec<Value> = items
            .iter()
            .map(|i| json!({"id": i["id"], "description": format!("Returns {}.", i["name"].as_str().unwrap())}))
            .collect();
        write_jsonl(&answers_path(&t4, "rewrite-description"), &ans).unwrap();
        probes(&t4, T4_SEED, Some(&root.join("runs"))).unwrap();
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
        // the raw files: stopped, with the difference stated
        let err = super::rows(&[], &t4).unwrap_err().to_string();
        assert!(err.contains("no `metadata` column"), "{err}");
        std::fs::remove_dir_all(root).ok();
    }
}
