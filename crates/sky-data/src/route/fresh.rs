//! The fresh test files of s6-nano v3 (`docs/s6-nano3-prereg.md`, section 4):
//! `sky-data --seed 1235 route fresh --build <v2 build> --out <build>/fresh`.
//!
//! The builder of v2 with a new seed and the rule of each v2 file, on requests that are in no
//! slate file of v2 (training, validation or test), compared by request identifier, by exact
//! text and by normalised text ([`super::norm_text`], the rule v2 used across its sides):
//!
//! - `f_tools.jsonl`: the `route/tools` requests of v2's test clusters that no v2 slate uses
//!   (those over v2's cap of 10 per cluster, and those left out of the 2,000 of T-tools), at
//!   most [`FRESH_CAP`] per cluster (the first by sha256 of the text, v2's cap rule); the slates
//!   of T-tools (distractors among the test-side tools, 20% removed); `f_tools_masked.jsonl`, the
//!   same slates with every tool renamed `tool_<k>`; `f_tools.clusters.jsonl`, the cluster and
//!   the origin of each slate (for the readings of section 6);
//! - `f_metatool.jsonl`, `f_clinc.jsonl`: v2's rules on the queries that v2 did not draw;
//! - `f_massive.jsonl`: v2's rule on the held-out scenarios' utterances that v2 did not use
//!   (v2 used every utterance of their test partition: the fresh ones come from their training
//!   and validation partitions, which v2 never read).
//!
//! v2's split is recomputed from its stage files by the same code ([`build::assign`]); before
//! anything is written, the recomputation must reproduce every test file of v2 byte for byte.
//! The fresh files are written once; the step prints counts only. No system's output is read.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};

use super::build::{self, Ctx, Kept, RouteRecord, MASSIVE_PROBE, TEST_SLATES};
use super::{
    bump, mask_keys, norm_text, read_json, read_lines, sha256_file, sha256_hex, slates, sources,
    write_json, Counts, Request, TEST, TOOLS,
};
use crate::commands::CommandRecord;

/// The seed of section 4.
pub const FRESH_SEED: u64 = 1235;
/// v2's seed (its `BUILD.json`).
pub const V2_SEED: u64 = 1234;
/// Section 4: at most this many fresh requests per cluster.
pub const FRESH_CAP: usize = 100;
/// The `split` of every fresh record.
pub const SPLIT: &str = "fresh";
/// The two origins of a T2-tools request.
pub const OVER_CAP: &str = "over the cluster cap";
pub const LEFT_OUT: &str = "left out of the 2,000";

pub struct FreshOptions {
    /// v2's build folder (read only).
    pub build: PathBuf,
    /// Where the fresh files go.
    pub out: PathBuf,
    /// `data/commands/massive.jsonl` (none: no `f_massive.jsonl`).
    pub massive: Option<PathBuf>,
    pub seed: u64,
    pub v2_seed: u64,
    pub cap: usize,
    /// A folder for `test-files.json` and `counts.json` (`runs/s6-nano3`).
    pub record: Option<PathBuf>,
}

/// What v2's slate files hold: request identifiers, exact texts, normalised texts.
#[derive(Default, Debug)]
pub struct Seen {
    ids: HashSet<String>,
    texts: HashSet<String>,
    norms: HashSet<String>,
}

impl Seen {
    pub fn add(&mut self, id: &str, text: &str) {
        self.ids.insert(id.to_string());
        self.texts.insert(text.to_string());
        self.norms.insert(norm_text(text));
    }

    /// Why a request is not fresh: its identifier, its text or its normalised text is in a v2
    /// slate file; `None` when it is fresh.
    pub fn reason(&self, id: &str, text: &str) -> Option<&'static str> {
        if self.ids.contains(id) {
            Some("identifier")
        } else if self.texts.contains(text) {
            Some("exact text")
        } else if self.norms.contains(&norm_text(text)) {
            Some("normalised text")
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// The request of a record's state: `User: <request>`, without the BM25 line of amendment 1.
pub fn request_of_state(state: &str) -> &str {
    let s = state.strip_prefix("User: ").unwrap_or(state);
    match s.rfind(&format!("\n{}", sky_infer::route::LINE_PREFIX)) {
        Some(i) => &s[..i],
        None => s,
    }
}

/// Every slate file of a v2 build: `records/<group>/*.jsonl`, `val/*.jsonl`, `test/*.jsonl`
/// (the model's copies of the validation slates included), sorted.
pub fn v2_slate_files(build: &Path) -> Result<Vec<PathBuf>> {
    fn jsonl_in(dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }
        for e in std::fs::read_dir(dir)? {
            let p = e?.path();
            if p.is_file() && p.extension().is_some_and(|x| x == "jsonl") {
                out.push(p);
            }
        }
        Ok(())
    }
    let mut out = Vec::new();
    let records = build.join("records");
    if records.is_dir() {
        for e in std::fs::read_dir(&records)? {
            let p = e?.path();
            if p.is_dir() {
                jsonl_in(&p, &mut out)?;
            }
        }
    }
    jsonl_in(&build.join("val"), &mut out)?;
    jsonl_in(&build.join("test"), &mut out)?;
    out.sort();
    Ok(out)
}

#[derive(Deserialize)]
struct SlateLine {
    request: String,
    state: String,
}

/// The requests of every v2 slate file: each record's request identifier, the text of its
/// state, and the staged text of that identifier.
fn v2_seen(build: &Path, staged: &HashMap<&str, &str>) -> Result<(Seen, Value)> {
    let mut seen = Seen::default();
    let mut per_file = BTreeMap::new();
    for f in v2_slate_files(build)? {
        let recs: Vec<SlateLine> = read_lines(&f)?;
        for r in &recs {
            let Some(user) = staged.get(r.request.as_str()) else {
                bail!("{}: request {} is not staged", f.display(), r.request);
            };
            seen.add(&r.request, user);
            seen.add(&r.request, request_of_state(&r.state));
        }
        let rel = f
            .strip_prefix(build)
            .unwrap_or(&f)
            .to_string_lossy()
            .to_string();
        per_file.insert(rel, recs.len());
    }
    let n = seen.len();
    Ok((seen, json!({"files": per_file, "request identifiers": n})))
}

/// The cluster label of a slate in `f_tools.clusters.jsonl`: `k` and 12 hex digits of the
/// sha256 of the cluster's key (its smallest normalised name, as v2's split reads it).
fn cluster_label(l: &build::Loaded, members: &[usize]) -> String {
    let key = members
        .iter()
        .map(|&t| l.norm[t].as_str())
        .min()
        .unwrap_or("");
    format!("k{}", &sha256_hex(key.as_bytes())[..12])
}

fn none_gold(r: &RouteRecord) -> bool {
    r.target.last() == Some(&1.0)
}

/// Slates, *none* share, kinds, sources and repeated texts of a fresh file.
fn file_counts(recs: &[RouteRecord]) -> Value {
    let none = recs.iter().filter(|r| none_gold(r)).count();
    let mut kinds = Counts::new();
    let mut srcs = Counts::new();
    let mut norms: HashMap<String, usize> = HashMap::new();
    for r in recs {
        bump(&mut kinds, &r.kind);
        bump(&mut srcs, &r.src);
        *norms
            .entry(norm_text(request_of_state(&r.state)))
            .or_default() += 1;
    }
    let repeated: usize = norms.values().filter(|&&n| n > 1).map(|n| n - 1).sum();
    json!({
        "slates": recs.len(),
        "none gold": none,
        "none share": none as f64 / recs.len().max(1) as f64,
        "kinds": kinds,
        "src": srcs,
        "slates whose normalised request repeats an earlier slate of the file (kept, as v2)": repeated,
    })
}

/// The fresh files of section 4.
pub fn fresh(o: &FreshOptions) -> Result<Value> {
    let l = build::load(&o.build)?;
    if let Ok(m) = read_json(&o.build.join("BUILD.json")) {
        if let Some(s) = m["seed"].as_u64() {
            if s != o.v2_seed {
                bail!("v2's BUILD.json says seed {s}, not {}", o.v2_seed);
            }
        }
    }
    let a = build::assign(&l)?;
    let kept = a.kept();
    let cx2 = Ctx {
        l: &l,
        seed: o.v2_seed,
    };

    // the guard: the recomputed split gives v2's test files, byte for byte
    let mut reproduced = BTreeMap::new();
    for (name, recs) in build::v2_test_files(&cx2, &a, &kept)? {
        let file = o
            .build
            .join("test")
            .join(format!("{}.jsonl", name.replace('-', "_")));
        let h = sha256_hex(&build::records_bytes(&recs)?);
        let on_disk = sha256_file(&file).with_context(|| format!("hash {}", file.display()))?;
        if on_disk != h {
            bail!(
                "the recomputed split does not give {} (sha256 {h}, on disk {on_disk}): \
                 the fresh files need v2's split as it was built",
                file.display()
            );
        }
        reproduced.insert(file.file_name().unwrap().to_string_lossy().to_string(), h);
    }

    // what v2's slate files hold
    let staged: HashMap<&str, &str> = l
        .requests
        .iter()
        .map(|r| (r.id.as_str(), r.user.as_str()))
        .collect();
    let (seen, seen_report) = v2_seen(&o.build, &staged)?;
    let cx = Ctx {
        l: &l,
        seed: o.seed,
    };
    let mut excluded = Counts::new();

    // ---- T2-tools
    let drawn: HashSet<&str> = build::t_tools_drawn(&cx2, &kept)
        .iter()
        .map(|k| k.r.id.as_str())
        .collect();
    let mut candidates = Counts::new();
    let mut cands: Vec<(&Kept, &'static str)> = Vec::new();
    for (i, k) in a.all.iter().enumerate() {
        if k.r.group != TOOLS || k.split != "test" {
            continue;
        }
        let origin = if a.drop.contains(&i) {
            OVER_CAP
        } else if drawn.contains(k.r.id.as_str()) {
            continue;
        } else {
            LEFT_OUT
        };
        bump(&mut candidates, origin);
        if let Some(why) = seen.reason(&k.r.id, &k.r.user) {
            bump(
                &mut excluded,
                format!("f_tools ({origin}): in a v2 slate file ({why})"),
            );
            continue;
        }
        cands.push((k, origin));
    }
    let cluster_of = |k: &Kept| -> usize { a.cluster_of[k.gold.unwrap()].unwrap() };
    let mut by_cluster: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (j, (k, _)) in cands.iter().enumerate() {
        by_cluster.entry(cluster_of(k)).or_default().push(j);
    }
    let mut keep = HashSet::new();
    let mut before_cap = Vec::new();
    for v in by_cluster.values_mut() {
        before_cap.push(v.len());
        v.sort_by_key(|&j| {
            (
                sha256_hex(cands[j].0.r.user.as_bytes()),
                cands[j].0.r.id.clone(),
            )
        });
        for (rank, &j) in v.iter().enumerate() {
            if rank < o.cap {
                keep.insert(j);
            } else {
                bump(
                    &mut excluded,
                    format!("f_tools ({}): over the fresh cap of {}", cands[j].1, o.cap),
                );
            }
        }
    }
    let chosen: Vec<(&Kept, &'static str)> = cands
        .iter()
        .enumerate()
        .filter(|(j, _)| keep.contains(j))
        .map(|(_, c)| *c)
        .collect();
    let order = slates::draw(chosen.len(), chosen.len(), &mut cx.rng("route/f-tools"));
    let reqs: Vec<&Kept> = order.iter().map(|&i| chosen[i].0).collect();
    let origin_of: HashMap<&str, &str> =
        chosen.iter().map(|(k, o)| (k.r.id.as_str(), *o)).collect();
    let test_pool = a.pool("test");
    let mut tf = build::tool_slates(&cx, "f-tools", SPLIT, &reqs, &test_pool);
    let f_tools = tf.remove("f-tools").unwrap_or_default();
    let f_masked = tf.remove("f-tools-masked").unwrap_or_default();
    for r in &f_tools {
        for id in &r.tools {
            if a.tsplit(l.index[id]) != Some("test") {
                bail!("f_tools: tool {id} is not of a test cluster");
            }
        }
    }
    let label: HashMap<usize, String> = by_cluster
        .keys()
        .map(|&c| (c, cluster_label(&l, &a.members[&c])))
        .collect();
    let req_of: HashMap<&str, &Kept> = reqs.iter().map(|k| (k.r.id.as_str(), *k)).collect();
    let clusters: Vec<Value> = f_tools
        .iter()
        .map(|r| {
            let k = req_of[r.request.as_str()];
            json!({"uid": r.uid, "cluster": label[&cluster_of(k)], "origin": origin_of[r.request.as_str()]})
        })
        .collect();

    // ---- T2-metatool, T2-clinc
    let of =
        |source: &str| -> Vec<&Kept> { kept.iter().filter(|k| k.r.source == source).collect() };
    fn fresh_of<'k>(
        seen: &Seen,
        name: &str,
        v: Vec<&'k Kept>,
        excluded: &mut Counts,
    ) -> Vec<&'k Kept> {
        v.into_iter()
            .filter(|k| match seen.reason(&k.r.id, &k.r.user) {
                Some(why) => {
                    bump(excluded, format!("{name}: in a v2 slate file ({why})"));
                    false
                }
                None => true,
            })
            .collect()
    }
    let mt = fresh_of(&seen, "f_metatool", of("metatool"), &mut excluded);
    let f_metatool = build::metatool_file(&cx, "f-metatool", SPLIT, &mt, TEST_SLATES)?;
    let all_cl = of("clinc");
    let cl_pool = build::tool_set(&all_cl);
    let cl = fresh_of(&seen, "f_clinc", all_cl, &mut excluded);
    let f_clinc = build::clinc_file(&cx, "f-clinc", SPLIT, &cl, &cl_pool);

    // ---- T2-massive: the held-out scenarios' utterances of every partition, v2's test ones
    // excluded by identifier like any other
    let mut massive_counts = Counts::new();
    let f_massive = match &o.massive {
        None => None,
        Some(path) => {
            let ms_kept = massive_requests(&l, &kept, path, &mut massive_counts)?;
            let ms: Vec<&Kept> = ms_kept
                .iter()
                .filter(|k| match seen.reason(&k.r.id, &k.r.user) {
                    Some(why) => {
                        bump(
                            &mut excluded,
                            format!(
                                "f_massive ({} partition): in a v2 slate file ({why})",
                                k.r.split.as_deref().unwrap_or("?")
                            ),
                        );
                        false
                    }
                    None => {
                        bump(
                            &mut massive_counts,
                            format!(
                                "fresh candidates, {} partition, {}",
                                k.r.split.as_deref().unwrap_or("?"),
                                k.r.lang
                            ),
                        );
                        true
                    }
                })
                .collect();
            Some(build::massive_file(
                &cx,
                "f-massive",
                SPLIT,
                &ms,
                &build::massive_pool(&kept),
                MASSIVE_PROBE,
            ))
        }
    };

    // the last guard: no fresh request in a v2 slate file
    let mut files: Vec<(&str, &[RouteRecord])> = vec![
        ("f_tools.jsonl", &f_tools),
        ("f_tools_masked.jsonl", &f_masked),
        ("f_metatool.jsonl", &f_metatool),
        ("f_clinc.jsonl", &f_clinc),
    ];
    if let Some(m) = &f_massive {
        files.push(("f_massive.jsonl", m));
    }
    for (name, recs) in &files {
        for r in recs.iter() {
            if let Some(why) = seen.reason(&r.request, request_of_state(&r.state)) {
                bail!("{name}: {} is in a v2 slate file ({why})", r.request);
            }
            if r.split != SPLIT {
                bail!("{name}: a record of split {}", r.split);
            }
        }
    }

    // ---- written once
    let mut hashes = BTreeMap::new();
    let mut records = BTreeMap::new();
    for (name, recs) in &files {
        hashes.insert(
            name.to_string(),
            build::write_once(&o.out.join(name), recs)?,
        );
        records.insert(name.to_string(), recs.len());
    }
    let cname = "f_tools.clusters.jsonl";
    hashes.insert(
        cname.to_string(),
        build::write_once(&o.out.join(cname), &clusters)?,
    );
    records.insert(cname.to_string(), clusters.len());

    // ---- counts
    let mut per_cluster: BTreeMap<&str, usize> = BTreeMap::new();
    let mut origin_slates = Counts::new();
    for c in &clusters {
        *per_cluster
            .entry(c["cluster"].as_str().unwrap())
            .or_default() += 1;
        bump(&mut origin_slates, c["origin"].as_str().unwrap());
    }
    let mut sizes: Vec<usize> = per_cluster.values().copied().collect();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let top3: usize = sizes.iter().take(3).sum();
    let mut by_source = Counts::new();
    for k in &reqs {
        bump(&mut by_source, &k.r.source);
    }
    before_cap.sort_unstable_by(|a, b| b.cmp(a));
    let mut per_file = BTreeMap::new();
    for (name, recs) in &files {
        per_file.insert(name.to_string(), file_counts(recs));
    }
    let tools_counts = json!({
        "clusters": per_cluster.len(),
        "requests per source": by_source,
        "slates by origin": origin_slates,
        "share of the slates in the three largest clusters": top3 as f64 / f_tools.len().max(1) as f64,
        "slates of the ten largest clusters": sizes.iter().take(10).collect::<Vec<_>>(),
        "candidates by origin (before the exclusion)": candidates,
        "fresh requests of the ten largest clusters before the fresh cap": before_cap.iter().take(10).collect::<Vec<_>>(),
        "test-side tools (distractor pool)": test_pool.len(),
    });
    let counts = json!({
        "about": "s6-nano v3 (docs/s6-nano3-prereg.md, section 4): the fresh files, counted before any system reads them",
        "seed": o.seed,
        "v2 seed": o.v2_seed,
        "cap per cluster": o.cap,
        "files": per_file,
        "f_tools": tools_counts,
        "f_massive": massive_counts,
        "excluded": excluded,
        "v2 slate files read for the exclusion": seen_report,
        "v2 test files reproduced by the recomputed split": reproduced,
    });
    let manifest = json!({
        "about": "s6-nano v3: the fresh test files (docs/s6-nano3-prereg.md, section 4), hashed before any system reads them",
        "files": hashes,
        "records": records,
        "counts": counts,
    });
    write_json(&o.out.join("FRESH.json"), &manifest)?;
    if let Some(dir) = &o.record {
        std::fs::create_dir_all(dir)?;
        write_json(
            &dir.join("test-files.json"),
            &json!({
                "about": "s6-nano v3: the fresh test files as the build wrote them (docs/s6-nano3-prereg.md, section 4); \
                          scripts/route/common.py refuses a fresh file whose sha256 differs",
                "build": o.out.join("FRESH.json"),
                "files": hashes,
                "records": records,
            }),
        )?;
        write_json(&dir.join("counts.json"), &counts)?;
    }
    Ok(
        json!({"files": hashes, "records": records, "f_tools": manifest["counts"]["f_tools"], "excluded": excluded}),
    )
}

/// The utterances of MASSIVE's held-out scenarios, every partition, as v2's reader makes them
/// (`massive-heldout:<row>`, the text trimmed and its key-like strings masked, the English
/// intent as gold).
fn massive_requests(
    l: &build::Loaded,
    kept: &[Kept],
    path: &Path,
    counts: &mut Counts,
) -> Result<Vec<Kept>> {
    let recs: Vec<CommandRecord> = read_lines(path)?;
    let scenarios: BTreeSet<String> = recs
        .iter()
        .filter_map(|r| {
            r.intent
                .as_deref()
                .map(|i| sources::scenario(i).to_string())
        })
        .collect();
    let held = sources::held_out_scenarios(&scenarios);
    // the intents of v2's MASSIVE requests, by name
    let mut by_name: HashMap<&str, usize> = HashMap::new();
    for t in build::massive_pool(kept) {
        if by_name.insert(l.tools[t].name.as_str(), t).is_some() {
            bail!("two MASSIVE intents named {}", l.tools[t].name);
        }
    }
    let staged: HashMap<&str, &Request> = l
        .requests
        .iter()
        .filter(|r| r.source == "massive-heldout")
        .map(|r| (r.id.as_str(), r))
        .collect();
    let mut out = Vec::new();
    for (i, r) in recs.iter().enumerate() {
        let Some(intent) = r.intent.as_deref() else {
            continue;
        };
        if !held.contains(sources::scenario(intent)) {
            continue;
        }
        let Some(&g) = by_name.get(intent) else {
            bail!("MASSIVE intent {intent} is not one of v2's");
        };
        bump(
            counts,
            format!("held-out utterances, {} partition, {}", r.split, r.lang),
        );
        // the file must be v2's: its held-out test rows are v2's staged requests, text and gold
        let id = format!("massive-heldout:{i}");
        let user = mask_keys(r.user.trim()).0;
        if let Some(s) = staged.get(id.as_str()) {
            if s.user != user || s.gold.as_deref() != Some(l.tools[g].id.as_str()) {
                bail!("{}: row {i} is not v2's {id}", path.display());
            }
            bump(counts, "rows equal to v2's staged requests");
        } else if r.split == "test" {
            bump(counts, "test-partition rows v2 did not stage");
        }
        out.push(Kept {
            r: Request {
                id: format!("massive-heldout:{i}"),
                source: "massive-heldout".into(),
                group: TEST.into(),
                split: Some(r.split.clone()),
                lang: r.lang.clone(),
                user: mask_keys(r.user.trim()).0,
                tools: Vec::new(),
                gold: Some(l.tools[g].id.clone()),
                exclude: Vec::new(),
                kind: "intent".into(),
            },
            split: SPLIT.into(),
            gold: Some(g),
            own: Vec::new(),
            exclude: HashSet::new(),
        });
    }
    counts.insert("held-out scenarios".into(), held.len());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::write_jsonl;
    use crate::route::build::tests::{fixture, run};

    fn opts(build: &Path, out: &Path, seed: u64, cap: usize) -> FreshOptions {
        FreshOptions {
            build: build.into(),
            out: out.into(),
            massive: Some(build.join("massive.jsonl")),
            seed,
            v2_seed: 7,
            cap,
            record: None,
        }
    }

    /// The invented build of `build::tests` (seed 7), and a MASSIVE file whose one scenario
    /// is held out.
    fn v2(dir: &Path) {
        fixture(dir);
        run(dir, 7);
        let rows: Vec<Value> = (0..40)
            .map(|i| {
                let lang = if i % 2 == 0 { "en" } else { "fr" };
                let split = ["train", "val", "test"][i % 3];
                json!({"src": "massive", "lang": lang, "user": format!("massive utterance {i}"),
                       "intent": format!("intent_{}", i % 6), "split": split})
            })
            .collect();
        write_jsonl(&dir.join("massive.jsonl"), &rows).unwrap();
    }

    fn fresh_hashes(dir: &Path) -> BTreeMap<String, String> {
        let m = read_json(&dir.join("FRESH.json")).unwrap();
        serde_json::from_value(m["files"].clone()).unwrap()
    }

    #[test]
    fn requests_seen_by_identifier_or_text() {
        let mut s = Seen::default();
        s.add("glaive:1", "Book a table for two.");
        assert_eq!(s.reason("glaive:1", "anything"), Some("identifier"));
        assert_eq!(
            s.reason("glaive:2", "Book a table for two."),
            Some("exact text")
        );
        assert_eq!(
            s.reason("glaive:3", "book a TABLE for two"),
            Some("normalised text")
        );
        assert_eq!(s.reason("glaive:4", "Book a table for three."), None);
        assert_eq!(
            request_of_state("User: hello\nBM25: a score 1.00 rank 1"),
            "hello"
        );
        assert_eq!(request_of_state("User: two\nlines"), "two\nlines");
    }

    #[test]
    fn fresh_files_of_an_invented_build() {
        let root = std::env::temp_dir().join(format!("sky-route-fresh-{}", std::process::id()));
        let a = root.join("v2");
        v2(&a);
        let tools: Vec<super::super::Tool> =
            read_lines(&build::stage(&a).join("tools.jsonl")).unwrap();
        let reqs: Vec<Request> = read_lines(&build::stage(&a).join("requests.jsonl")).unwrap();

        // determinism under the seed, and the seed matters
        let (f1, f2, f3) = (root.join("f1"), root.join("f2"), root.join("f3"));
        fresh(&opts(&a, &f1, 1235, 25)).unwrap();
        fresh(&opts(&a, &f2, 1235, 25)).unwrap();
        fresh(&opts(&a, &f3, 99, 25)).unwrap();
        assert_eq!(fresh_hashes(&f1), fresh_hashes(&f2));
        assert_ne!(
            fresh_hashes(&f1)["f_tools.jsonl"],
            fresh_hashes(&f3)["f_tools.jsonl"]
        );
        // written once: a second build into the same folder must give the same bytes
        fresh(&opts(&a, &f1, 1235, 25)).unwrap();
        assert!(fresh(&opts(&a, &f1, 99, 25)).is_err());

        let ft: Vec<RouteRecord> = read_lines(&f1.join("f_tools.jsonl")).unwrap();
        let fm: Vec<RouteRecord> = read_lines(&f1.join("f_tools_masked.jsonl")).unwrap();
        let fms: Vec<RouteRecord> = read_lines(&f1.join("f_massive.jsonl")).unwrap();
        assert!(!ft.is_empty() && !fms.is_empty());
        assert_eq!(ft.len(), fm.len());
        // no fresh request is in a v2 slate file, by identifier or by text
        let mut v2_ids = HashSet::new();
        let mut v2_texts = HashSet::new();
        for f in v2_slate_files(&a).unwrap() {
            for r in read_lines::<SlateLine>(&f).unwrap() {
                v2_texts.insert(norm_text(request_of_state(&r.state)));
                v2_ids.insert(r.request);
            }
        }
        for r in ft.iter().chain(&fms) {
            assert!(!v2_ids.contains(&r.request), "{}", r.request);
            assert!(!v2_texts.contains(&norm_text(request_of_state(&r.state))));
            assert_eq!(r.split, SPLIT);
        }
        // MASSIVE: v2's identifiers (`massive-heldout:<row>`), the row's intent as gold
        let rows: Vec<Value> = read_lines(&a.join("massive.jsonl")).unwrap();
        for r in &fms {
            let i: usize = r
                .request
                .strip_prefix("massive-heldout:")
                .unwrap()
                .parse()
                .unwrap();
            let g = r.target.iter().position(|&x| x == 1.0).unwrap();
            if g < r.tools.len() {
                assert!(r.opts[g].starts_with(rows[i]["intent"].as_str().unwrap()));
            }
            assert_eq!(
                request_of_state(&r.state),
                rows[i]["user"].as_str().unwrap()
            );
        }
        // T2-tools: test-cluster requests only, the slates of T-tools, 20% removed
        let test_ids: HashSet<String> = read_lines::<RouteRecord>(&a.join("test/t_tools.jsonl"))
            .unwrap()
            .iter()
            .flat_map(|r| r.tools.clone())
            .collect();
        let by_id: HashMap<&str, &Request> = reqs.iter().map(|r| (r.id.as_str(), r)).collect();
        let thing = |tool: &str| -> String {
            let t = tools.iter().find(|t| t.id == tool).unwrap();
            t.name.split("_v").next().unwrap().to_string()
        };
        let test_things: HashSet<String> = test_ids.iter().map(|t| thing(t)).collect();
        let mut per: BTreeMap<String, usize> = BTreeMap::new();
        for r in &ft {
            assert_eq!(r.opts.len(), slates::TEST_TOOLS + 1);
            let gold = by_id[r.request.as_str()].gold.clone().unwrap();
            assert!(test_things.contains(&thing(&gold)));
            *per.entry(thing(&gold)).or_default() += 1;
        }
        let removed = ft.iter().filter(|r| r.kind == "removed").count();
        assert_eq!(removed, (ft.len() as f64 * 0.2).round() as usize);
        // the cap: 40 fresh requests per cluster (50 less the 10 of v2), cut to 25
        assert!(per.values().all(|&n| n <= 25), "{per:?}");
        assert!(per.values().any(|&n| n == 25));
        let f4 = root.join("f4");
        fresh(&opts(&a, &f4, 1235, 100)).unwrap();
        let ft4: Vec<RouteRecord> = read_lines(&f4.join("f_tools.jsonl")).unwrap();
        assert!(ft4.len() > ft.len());
        // the masked copy: every tool renamed
        for r in &fm {
            assert!(r.opts[..r.opts.len() - 1]
                .iter()
                .all(|o| o.starts_with("tool_")));
        }

        // exclusion by identifier and by text: a v2 slate file that names one fresh request by
        // its identifier and holds another's text under another identifier
        let (x, y) = (&ft4[0], &ft4[1]);
        let extra = vec![
            json!({"request": x.request, "state": "User: unrelated"}),
            json!({"request": "glaive:0-0", "state": y.state}),
        ];
        write_jsonl(&a.join("records/tools/extra.jsonl"), &extra).unwrap();
        let f5 = root.join("f5");
        let rep = fresh(&opts(&a, &f5, 1235, 100)).unwrap();
        let ft5: Vec<RouteRecord> = read_lines(&f5.join("f_tools.jsonl")).unwrap();
        assert_eq!(ft5.len(), ft4.len() - 2);
        assert!(ft5
            .iter()
            .all(|r| r.request != x.request && r.request != y.request));
        let ex = rep["excluded"].as_object().unwrap();
        assert!(ex.keys().any(|k| k.contains("(identifier)")));
        assert!(ex.keys().any(|k| k.contains("(exact text)")));

        // the guard: a v2 test file that the recomputed split does not give stops the build
        let p = a.join("test/t_clinc.jsonl");
        let mut bytes = std::fs::read(&p).unwrap();
        bytes.push(b'\n');
        std::fs::write(&p, bytes).unwrap();
        assert!(fresh(&opts(&a, &root.join("f6"), 1235, 100)).is_err());
        std::fs::remove_dir_all(root).ok();
    }
}
