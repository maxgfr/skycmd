//! s4: the corpus the Qwen3.5 teacher scores (`s4-corpus`) and the gold / teacher mix that
//! the student trains on (`s4-mix`).
//!
//! `s4-corpus` writes, in labelling priority order, `queue/p1-decisions.jsonl` (a sample of the
//! fetched typed-decision sets with randomized option lists), `queue/p2-generic.jsonl` (web and
//! story snippets under random label sets), `queue/p3-cases.jsonl` (tool routing, ticket triage
//! and rule-based moderation), plus `gold/` (the gold decision sets the student trains on
//! directly, held-out sets excluded) and `heldout/` (Banking77 and MASSIVE intent choices,
//! `split = "test"`, never trained on: the zero-shot evaluation).
//!
//! Every queued line is a SPEC decision record plus `gold`: whether `target` is a real label
//! (otherwise it is a uniform placeholder the teacher replaces). `scripts/teacher-score.py`
//! appends `uid` and `tlogits` (teacher logits over the option letters).
//!
//! `s4-mix` fits one teacher temperature on the records with a gold label (train split), then
//! writes `target = alpha * gold + (1 - alpha) * softmax(tlogits / T)` when a gold label exists
//! and the tempered teacher distribution otherwise.

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::commands::CommandRecord;
use crate::decisions::SMALL;
use crate::hf::{read_parquet, Hf};
use crate::record::{argmax, read_jsonl, write_jsonl, DecisionRecord, MlmRecord, Question};
use crate::rng::{hash64, Rng};

/// Sources never used for training: external benchmarks, the zero-shot held-out sets and
/// local teacher data of the drone task.
pub const EXCLUDED: &[&str] = &["typed_decisions_bench", "notam", "banking77", "glm_orders"];

/// Option texts the student learns as abstention ("none of these" / "I don't know").
pub const NONE_OPTS: &[&str] = &[
    "none: none of these options applies",
    "unsure: I don't know, none of the above fits",
    "other: something else not listed here",
];

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct S4Record {
    #[serde(flatten)]
    pub rec: DecisionRecord,
    /// `target` is a real label (false: uniform placeholder, teacher only).
    pub gold: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uid: Option<u64>,
    /// Teacher logits over the option letters (temperature 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tlogits: Option<Vec<f32>>,
}

impl S4Record {
    fn gold(rec: DecisionRecord) -> Self {
        Self {
            rec,
            gold: true,
            uid: None,
            tlogits: None,
        }
    }
    fn teacher(rec: DecisionRecord) -> Self {
        Self {
            rec,
            gold: false,
            uid: None,
            tlogits: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CorpusOptions {
    pub data: PathBuf,
    pub out: PathBuf,
    pub seed: u64,
    /// Records sampled per typed-decision source (p1).
    pub per_source: usize,
    /// Generic snippet records (p2).
    pub generic: usize,
    /// Tool-routing, triage and moderation records each (p3).
    pub per_case: usize,
    /// Records per held-out set.
    pub heldout: usize,
}

fn uniform(n: usize) -> Vec<f32> {
    vec![1.0 / n as f32; n]
}

fn split_of(key: &str, seed: u64, val_frac: f64) -> String {
    let h = hash64(format!("{seed}/{key}").as_bytes()) % 1_000_000;
    if (h as f64 / 1e6) < val_frac {
        "val".into()
    } else {
        "train".into()
    }
}

fn readable(label: &str) -> String {
    label.replace('_', " ")
}

fn clip(s: &str, max: usize) -> String {
    let s = s.trim();
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    match cut.rfind(['.', '!', '?']) {
        Some(i) if i > max / 3 => cut[..=i].to_string(),
        _ => match cut.rfind(' ') {
            Some(i) => format!("{}…", &cut[..i]),
            None => cut,
        },
    }
}

/// Label-set randomization of a choice record: shuffled options, extra distractors drawn
/// from `pool`, an abstention option, and sometimes the gold option removed so that the
/// abstention option becomes the answer.
pub fn randomize_choice(r: &DecisionRecord, pool: &[String], rng: &mut Rng) -> DecisionRecord {
    let mut pairs: Vec<(String, f32)> = r
        .opts
        .iter()
        .cloned()
        .zip(r.target.iter().copied())
        .collect();
    let g = argmax(&r.target);
    let confident = r.target[g] >= 0.8;
    let mut none_gold = false;
    if confident && pairs.len() >= 3 && rng.f64() < 0.12 {
        pairs.remove(g);
        none_gold = true;
    }
    if !pool.is_empty() && pairs.len() < 10 && rng.f64() < 0.3 {
        let have: HashSet<String> = pairs.iter().map(|p| p.0.clone()).collect();
        for _ in 0..1 + rng.below(2) {
            let d = &pool[rng.below(pool.len())];
            if !have.contains(d) {
                pairs.push((d.clone(), 0.0));
            }
        }
    }
    if none_gold || rng.f64() < 0.2 {
        let n = NONE_OPTS[rng.below(NONE_OPTS.len())];
        pairs.push((n.to_string(), if none_gold { 1.0 } else { 0.0 }));
    }
    rng.shuffle(&mut pairs);
    let mut out = r.clone();
    out.opts = pairs.iter().map(|p| p.0.clone()).collect();
    out.target = pairs.iter().map(|p| p.1).collect();
    let s: f32 = out.target.iter().sum();
    if s <= 0.0 {
        out.target = uniform(out.opts.len());
    } else {
        out.target.iter_mut().for_each(|x| *x /= s);
    }
    out
}

/// p1: a sample of every non-excluded typed-decision source (train and val splits), choice
/// records label-randomized. Also links the gold files the student trains on into `gold/`.
fn p1_decisions(o: &CorpusOptions) -> Result<Vec<S4Record>> {
    let dir = o.data.join("decisions");
    let gold_dir = o.out.join("gold");
    fs::create_dir_all(&gold_dir)?;
    let mut out = Vec::new();
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .with_context(|| format!("read {dir:?}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    for f in files {
        let src = f.file_stem().unwrap().to_string_lossy().to_string();
        let link = gold_dir.join(f.file_name().unwrap());
        // The benchmark stays in gold/ (test split only) so s2-eval reports it.
        if (!EXCLUDED.contains(&src.as_str()) || src == "typed_decisions_bench") && !link.exists() {
            let abs = fs::canonicalize(&f)?;
            std::os::unix::fs::symlink(abs, &link)?;
        }
        if EXCLUDED.contains(&src.as_str()) {
            continue;
        }
        let mut recs: Vec<DecisionRecord> = read_jsonl(&f)?;
        recs.retain(|r| r.split != "test" && r.opts.len() >= 2 && r.opts.len() <= 20);
        let mut rng = Rng::derived(o.seed, &format!("s4/p1/{src}"));
        rng.shuffle(&mut recs);
        let pool: Vec<String> = recs
            .iter()
            .filter(|r| r.t == "choice")
            .take(2000)
            .flat_map(|r| r.opts.iter().cloned())
            .collect();
        let mut n = 0;
        for r in recs.iter().take(o.per_source) {
            let rec = if r.t == "choice" {
                randomize_choice(r, &pool, &mut rng)
            } else {
                r.clone()
            };
            out.push(S4Record::gold(rec));
            n += 1;
        }
        eprintln!("[s4] p1 {src}: {n}");
    }
    Ok(out)
}

/// Label sets of the generic snippets (p2). None of them is the label set of a held-out
/// benchmark (Banking77, MASSIVE).
const CHOICE_SETS: &[(&str, &[(&str, &str)])] = &[
    (
        "What is the main topic of the text?",
        &[
            ("science", "natural sciences, research or experiments"),
            ("history", "past events or historical figures"),
            ("health", "medicine, illness, body or fitness"),
            ("technology", "computers, software, machines or devices"),
            ("economy", "money, markets, jobs or companies"),
            ("politics", "government, elections or laws"),
            ("education", "schools, learning or teaching"),
            ("environment", "nature, climate or ecology"),
            ("arts", "music, painting, literature or film"),
            ("family", "parents, children or home life"),
            ("food", "cooking, recipes or eating"),
            ("travel", "trips, places or tourism"),
            ("religion", "faith, beliefs or rituals"),
            ("animals", "pets, wildlife or creatures"),
            ("sport", "games, athletes or competitions"),
        ],
    ),
    (
        "What kind of text is this?",
        &[
            ("story", "a narrative with characters and events"),
            ("explanation", "explains how or why something works"),
            ("instructions", "tells the reader what to do, step by step"),
            ("opinion", "argues for a point of view"),
            ("report", "reports facts about an event"),
            ("advertisement", "promotes a product or a service"),
            ("dialogue", "mostly people talking to each other"),
            ("description", "describes a person, a place or a thing"),
        ],
    ),
    (
        "Who is the most likely intended reader?",
        &[
            ("children", "young kids"),
            ("students", "school or university learners"),
            ("experts", "specialists of the field"),
            ("general public", "any adult reader"),
            ("customers", "people who buy a product or a service"),
        ],
    ),
    (
        "What is the overall tone of the text?",
        &[
            ("positive", "cheerful, hopeful or approving"),
            ("negative", "sad, critical or worried"),
            ("neutral", "factual, without emotion"),
            ("humorous", "meant to be funny"),
            ("formal", "official or academic"),
        ],
    ),
    (
        "When does what the text describes mainly take place?",
        &[
            ("past", "events that already happened"),
            ("present", "what is happening now or is generally true"),
            ("future", "plans, predictions or possibilities"),
        ],
    ),
    (
        "What should a busy reader do with this text?",
        &[
            ("read now", "important and time-sensitive"),
            ("save for later", "useful but not urgent"),
            ("share", "interesting for friends or colleagues"),
            ("skip", "not useful for most people"),
        ],
    ),
    (
        "Which section of a website would this text fit best?",
        &[
            ("blog", "personal or informal posts"),
            ("help center", "support articles and how-tos"),
            ("news", "current events"),
            ("shop", "product pages"),
            ("kids corner", "content for children"),
            ("research", "scientific or technical papers"),
        ],
    ),
    (
        "Quel est le sujet principal du texte ?",
        &[
            ("nature", "animaux, plantes ou paysages"),
            ("société", "politique, économie ou vie publique"),
            ("vie quotidienne", "famille, maison, école"),
            ("culture", "art, livres, cinéma, musique"),
            ("sciences", "recherche, technique ou santé"),
            ("aventure", "voyages, quêtes ou découvertes"),
        ],
    ),
];

const SCORE_SETS: &[(&str, &[&str])] = &[
    (
        "How formal is the language of the text?",
        &[
            "very informal",
            "informal",
            "neutral",
            "formal",
            "very formal",
        ],
    ),
    (
        "How hard is the text to read?",
        &[
            "very easy, a young child could read it",
            "easy",
            "average",
            "hard",
            "very hard, expert level",
        ],
    ),
    (
        "How positive is the text overall?",
        &[
            "very negative",
            "negative",
            "neutral",
            "positive",
            "very positive",
        ],
    ),
    (
        "How technical is the text?",
        &[
            "not technical at all",
            "slightly technical",
            "moderately technical",
            "highly technical",
        ],
    ),
    (
        "How much does the text need a quick reply or action?",
        &["no action needed", "can wait", "soon", "urgent"],
    ),
];

const NOUL_SETS: &[&str] = &[
    "The text tells a story with characters.",
    "The text mentions a specific number, date or amount.",
    "The text is written in French.",
    "The text gives advice or instructions to the reader.",
    "The text describes a scientific fact.",
    "The text contains a question.",
    "The text talks about an animal.",
    "The text expresses a personal opinion.",
    "The text would be appropriate for a young child.",
    "The text mentions a real place or country.",
];

fn snippets(data: &Path, n: usize, seed: u64) -> Result<Vec<(String, String)>> {
    let dir = data.join("text");
    let mut all = Vec::new();
    let mut files: Vec<PathBuf> = fs::read_dir(&dir)
        .with_context(|| format!("read {dir:?}"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    let per = n.div_ceil(files.len().max(1)) * 2;
    for f in files {
        let src = f.file_stem().unwrap().to_string_lossy().to_string();
        let mut docs: Vec<MlmRecord> = read_jsonl(&f)?;
        let mut rng = Rng::derived(seed, &format!("s4/p2/{src}"));
        rng.shuffle(&mut docs);
        for d in docs.into_iter().take(per) {
            let max = 200 + rng.below(500);
            all.push((src.clone(), clip(&d.text, max)));
        }
    }
    let mut rng = Rng::derived(seed, "s4/p2/mix");
    rng.shuffle(&mut all);
    all.truncate(n);
    Ok(all)
}

/// p2: web and story snippets under random label sets (teacher labels only).
fn p2_generic(o: &CorpusOptions) -> Result<Vec<S4Record>> {
    let mut rng = Rng::derived(o.seed, "s4/p2");
    let mut out = Vec::new();
    for (src, text) in snippets(&o.data, o.generic, o.seed)? {
        let roll = rng.f64();
        let rec = if roll < 0.6 {
            let (ins, labels) = CHOICE_SETS[rng.below(CHOICE_SETS.len())];
            let mut l: Vec<&(&str, &str)> = labels.iter().collect();
            rng.shuffle(&mut l);
            let k = (3 + rng.below(6)).min(l.len());
            let mut crit: Vec<(String, String)> = l[..k]
                .iter()
                .map(|(a, b)| {
                    // Half of the label sets come without descriptions (bare names).
                    if rng.f64() < 0.5 {
                        (a.to_string(), b.to_string())
                    } else {
                        (a.to_string(), String::new())
                    }
                })
                .collect();
            if rng.f64() < 0.3 {
                let n = NONE_OPTS[rng.below(NONE_OPTS.len())];
                let (a, b) = n.split_once(": ").unwrap();
                crit.push((a.into(), b.into()));
            }
            let q = Question::choice(ins, crit);
            let n = q.options().len();
            q.record(&format!("s4_generic_{src}"), &text, uniform(n))
        } else if roll < 0.8 {
            let (ins, levels) = SCORE_SETS[rng.below(SCORE_SETS.len())];
            let q = Question::score(ins, levels.iter().map(|s| s.to_string()).collect());
            let n = levels.len();
            q.record(&format!("s4_generic_{src}"), &text, uniform(n))
        } else {
            let stmt = if rng.f64() < 0.3 {
                let (_, labels) = CHOICE_SETS[0];
                format!(
                    "The text is mainly about {}.",
                    labels[rng.below(labels.len())].0
                )
            } else {
                NOUL_SETS[rng.below(NOUL_SETS.len())].to_string()
            };
            Question::noul(stmt).record(&format!("s4_generic_{src}"), &text, uniform(2))
        };
        let mut rec = rec;
        rec.split = split_of(&text, o.seed, 0.03);
        out.push(S4Record::teacher(rec));
    }
    eprintln!("[s4] p2 generic: {}", out.len());
    Ok(out)
}

fn tool_option(t: &Value) -> Option<(String, String)> {
    let name = t.get("name")?.as_str()?.trim().to_string();
    if name.is_empty() {
        return None;
    }
    let desc = t.get("description").and_then(Value::as_str).unwrap_or("");
    Some((name, clip(desc, 110)))
}

const TOOL_INS: &[&str] = &[
    "Which tool should the assistant call to handle this request?",
    "Pick the function that best serves the user's request.",
    "Which of the available tools fits this message?",
];
const NO_TOOL: (&str, &str) = (
    "no_tool",
    "no tool is needed, reply directly or ask for details",
);

/// Tool routing: the user turn of glaive / hermes conversations, the tools offered to the
/// assistant plus distractor tools from other conversations, and a "no tool" option.
fn tool_routing(o: &CorpusOptions, rng: &mut Rng) -> Result<Vec<S4Record>> {
    let mut recs: Vec<CommandRecord> = Vec::new();
    for f in ["glaive", "hermes"] {
        let p = o.data.join("commands").join(format!("{f}.jsonl"));
        if p.exists() {
            recs.extend(read_jsonl::<CommandRecord>(&p)?);
        }
    }
    let pool: Vec<(String, String)> = recs
        .iter()
        .filter_map(|r| r.tools.as_ref()?.as_array().cloned())
        .flatten()
        .filter_map(|t| tool_option(&t))
        .collect();
    if pool.is_empty() {
        eprintln!("[s4] p3 tools: no data/commands, skipped");
        return Ok(Vec::new());
    }
    rng.shuffle(&mut recs);
    let mut out = Vec::new();
    for r in recs {
        if out.len() >= o.per_case {
            break;
        }
        let mut crit: Vec<(String, String)> = r
            .tools
            .as_ref()
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(tool_option).collect())
            .unwrap_or_default();
        let mut seen: HashSet<String> = crit.iter().map(|c| c.0.clone()).collect();
        if seen.len() != crit.len() {
            continue;
        }
        let gold_name = r
            .target_json
            .as_deref()
            .and_then(|t| serde_json::from_str::<Value>(t).ok())
            .and_then(|v| {
                v.as_array()?
                    .first()?
                    .get("name")?
                    .as_str()
                    .map(str::to_string)
            });
        if let Some(g) = &gold_name {
            if !seen.contains(g) {
                continue;
            }
        }
        let want = 3 + rng.below(6);
        let mut tries = 0;
        while crit.len() < want && tries < 50 {
            let d = &pool[rng.below(pool.len())];
            if seen.insert(d.0.clone()) {
                crit.push(d.clone());
            }
            tries += 1;
        }
        crit.push((NO_TOOL.0.into(), NO_TOOL.1.into()));
        rng.shuffle(&mut crit);
        let gi = match &gold_name {
            Some(g) => crit.iter().position(|c| &c.0 == g).unwrap(),
            None => crit.iter().position(|c| c.0 == NO_TOOL.0).unwrap(),
        };
        let mut target = vec![0.0; crit.len()];
        target[gi] = 1.0;
        let q = Question::choice(TOOL_INS[rng.below(TOOL_INS.len())], crit);
        let mut rec = q.record("s4_tools", &clip(&r.user, 900), target);
        rec.split = if r.split == "train" {
            "train".into()
        } else {
            "val".into()
        };
        out.push(S4Record::gold(rec));
    }
    eprintln!("[s4] p3 tools: {}", out.len());
    Ok(out)
}

const TRIAGE_CATS: &[(&str, &str)] = &[
    ("ACCOUNT", "account creation, login, profile or deletion"),
    ("CANCEL", "cancelling a purchase or a fee"),
    ("CONTACT", "reaching customer service or a human agent"),
    ("DELIVERY", "delivery options and delivery period"),
    ("FEEDBACK", "complaints or reviews"),
    ("INVOICE", "getting or checking an invoice"),
    ("ORDER", "placing, changing or tracking an order"),
    ("PAYMENT", "payment methods and payment problems"),
    ("REFUND", "getting money back"),
    ("SHIPPING", "shipping address changes"),
    ("SUBSCRIPTION", "newsletter subscription"),
];
const TRIAGE_INS: &[&str] = &[
    "Which inbox folder should this message go to?",
    "Route this support ticket to the right team.",
    "Which category does this customer message belong to?",
    "Dans quelle catégorie ranger ce message client ?",
];

/// Ticket / inbox triage: Bitext customer-support messages under random subsets of their
/// categories (or intents), with an "unsure" option that becomes the answer when the gold
/// category is left out.
fn triage(hf: &Hf, o: &CorpusOptions, rng: &mut Rng) -> Result<Vec<S4Record>> {
    let repo = "bitext/Bitext-customer-support-llm-chatbot-training-dataset";
    let mut rows = Vec::new();
    for p in hf.parquet_split(repo, "default", "train", SMALL)? {
        rows.extend(read_parquet(
            &p,
            &["instruction", "category", "intent"],
            usize::MAX,
        )?);
    }
    rng.shuffle(&mut rows);
    let intents: Vec<String> = {
        let mut s: Vec<String> = rows
            .iter()
            .filter_map(|r| r.get("intent")?.as_str().map(str::to_string))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        s.sort();
        s
    };
    let mut out = Vec::new();
    for r in rows {
        if out.len() >= o.per_case {
            break;
        }
        let (Some(text), Some(cat), Some(intent)) = (
            r.get("instruction").and_then(Value::as_str),
            r.get("category").and_then(Value::as_str),
            r.get("intent").and_then(Value::as_str),
        ) else {
            continue;
        };
        let by_intent = rng.f64() < 0.35;
        let mut crit: Vec<(String, String)> = if by_intent {
            let mut v: Vec<(String, String)> = intents
                .iter()
                .filter(|i| i.as_str() != intent)
                .map(|i| (readable(i), String::new()))
                .collect();
            rng.shuffle(&mut v);
            v.truncate(2 + rng.below(6));
            v
        } else {
            let mut v: Vec<(String, String)> = TRIAGE_CATS
                .iter()
                .filter(|c| c.0 != cat)
                .map(|(a, b)| {
                    let name = if rng.f64() < 0.5 {
                        a.to_lowercase()
                    } else {
                        a.to_string()
                    };
                    (
                        name,
                        if rng.f64() < 0.6 {
                            b.to_string()
                        } else {
                            String::new()
                        },
                    )
                })
                .collect();
            rng.shuffle(&mut v);
            v.truncate(2 + rng.below(5));
            v
        };
        let drop_gold = rng.f64() < 0.15;
        let gold_key = if by_intent {
            readable(intent)
        } else {
            let d = TRIAGE_CATS
                .iter()
                .find(|c| c.0 == cat)
                .map(|c| c.1)
                .unwrap_or("");
            let name = if rng.f64() < 0.5 {
                cat.to_lowercase()
            } else {
                cat.to_string()
            };
            if !drop_gold {
                crit.push((
                    name.clone(),
                    if rng.f64() < 0.6 {
                        d.to_string()
                    } else {
                        String::new()
                    },
                ));
            }
            name
        };
        if by_intent && !drop_gold {
            crit.push((gold_key.clone(), String::new()));
        }
        let with_none = drop_gold || rng.f64() < 0.4;
        if with_none {
            crit.push(("unsure".into(), "I don't know, none of these fits".into()));
        }
        rng.shuffle(&mut crit);
        let gi = if drop_gold {
            crit.iter().position(|c| c.0 == "unsure").unwrap()
        } else {
            crit.iter().position(|c| c.0 == gold_key).unwrap()
        };
        let mut target = vec![0.0; crit.len()];
        target[gi] = 1.0;
        let q = Question::choice(TRIAGE_INS[rng.below(TRIAGE_INS.len())], crit);
        let mut rec = q.record("s4_triage", text, target);
        rec.split = split_of(text, o.seed, 0.03);
        out.push(S4Record::gold(rec));
    }
    eprintln!("[s4] p3 triage: {}", out.len());
    Ok(out)
}

/// Written rules for the moderation noul; the column is the Civil Comments score used as a
/// soft gold label (`None`: teacher only).
const RULES: &[(&str, Option<&str>)] = &[
    (
        "No insults or personal attacks against other people.",
        Some("insult"),
    ),
    ("No threats of violence or harm.", Some("threat")),
    ("No obscene or vulgar language.", Some("obscene")),
    (
        "No attacks on people because of their religion, ethnicity, gender or other identity.",
        Some("identity_attack"),
    ),
    (
        "Be civil: no rude, disrespectful or toxic comments.",
        Some("toxicity"),
    ),
    ("No sexually explicit content.", Some("sexual_explicit")),
    ("No advertising or self-promotion.", None),
    ("No phone numbers, emails or home addresses.", None),
    ("No discussion of politics or elections.", None),
    (
        "Comments must be written in a polite, constructive way.",
        None,
    ),
    ("No spreading of medical misinformation.", None),
];

/// Rule-based moderation: Civil Comments (validation split) under a written rule, as a noul
/// "the message violates the rule". Toxic comments are oversampled.
fn moderation(hf: &Hf, o: &CorpusOptions, rng: &mut Rng) -> Result<Vec<S4Record>> {
    let repo = "google/civil_comments";
    let cols = [
        "text",
        "toxicity",
        "insult",
        "threat",
        "obscene",
        "identity_attack",
        "sexual_explicit",
    ];
    let mut rows = Vec::new();
    for p in hf.parquet_split(repo, "default", "validation", SMALL)? {
        rows.extend(read_parquet(&p, &cols, usize::MAX)?);
    }
    rng.shuffle(&mut rows);
    let tox = |r: &Value| r.get("toxicity").and_then(Value::as_f64).unwrap_or(0.0);
    let (hot, cold): (Vec<Value>, Vec<Value>) = rows.into_iter().partition(|r| tox(r) >= 0.3);
    let half = o.per_case / 2;
    let picked: Vec<Value> = hot
        .into_iter()
        .take(half)
        .chain(cold.into_iter().take(half))
        .collect();
    let mut out = Vec::new();
    for r in picked {
        let Some(text) = r.get("text").and_then(Value::as_str) else {
            continue;
        };
        if text.trim().len() < 20 {
            continue;
        }
        // Mapped rules with the matching column most of the time, other rules otherwise.
        let (rule, col) = if rng.f64() < 0.7 {
            RULES[rng.below(6)]
        } else {
            RULES[6 + rng.below(RULES.len() - 6)]
        };
        let ins = format!("The message violates this rule: \"{rule}\"");
        let q = Question::noul(ins);
        let (target, gold) = match col.and_then(|c| r.get(c)?.as_f64()) {
            Some(s) => (vec![1.0 - s as f32, s as f32], true),
            None => (uniform(2), false),
        };
        let mut rec = q.record("s4_moderation", &clip(text, 900), target);
        rec.split = split_of(text, o.seed, 0.03);
        out.push(S4Record {
            rec,
            gold,
            uid: None,
            tlogits: None,
        });
    }
    eprintln!("[s4] p3 moderation: {}", out.len());
    Ok(out)
}

/// Held-out zero-shot sets: Banking77 (its fetched 8-way choices) and MASSIVE intents (gold
/// plus 7 random intents), all `split = "test"`.
fn heldout(o: &CorpusOptions) -> Result<()> {
    let dir = o.out.join("heldout");
    fs::create_dir_all(&dir)?;
    let mut rng = Rng::derived(o.seed, "s4/heldout");
    let b = o.data.join("decisions").join("banking77.jsonl");
    if b.exists() {
        let mut recs: Vec<DecisionRecord> = read_jsonl(&b)?;
        rng.shuffle(&mut recs);
        recs.truncate(o.heldout);
        for r in &mut recs {
            r.src = "zs_banking77".into();
            r.split = "test".into();
        }
        write_jsonl(&dir.join("zs_banking77.jsonl"), &recs)?;
    }
    let m = o.data.join("commands").join("massive.jsonl");
    if m.exists() {
        let all: Vec<CommandRecord> = read_jsonl(&m)?;
        let mut intents: Vec<String> = all
            .iter()
            .filter_map(|r| r.intent.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        intents.sort();
        for lang in ["en", "fr"] {
            let mut recs: Vec<&CommandRecord> = all
                .iter()
                .filter(|r| r.lang == lang && r.split == "test")
                .collect();
            rng.shuffle(&mut recs);
            let n = if lang == "en" {
                o.heldout
            } else {
                o.heldout / 2
            };
            let mut out = Vec::new();
            for r in recs.into_iter().take(n) {
                let gold = r.intent.clone().unwrap();
                let mut opts: Vec<String> = vec![readable(&gold)];
                while opts.len() < 8 {
                    let d = readable(&intents[rng.below(intents.len())]);
                    if !opts.contains(&d) {
                        opts.push(d);
                    }
                }
                rng.shuffle(&mut opts);
                let gi = opts.iter().position(|x| *x == readable(&gold)).unwrap();
                let mut target = vec![0.0; 8];
                target[gi] = 1.0;
                let crit = opts.into_iter().map(|k| (k, String::new())).collect();
                let q = Question::choice(
                    "Which intent does the user's request to the voice assistant express?",
                    crit,
                );
                let mut rec = q.record(&format!("zs_massive_{lang}"), &r.user, target);
                rec.split = "test".into();
                out.push(rec);
            }
            write_jsonl(&dir.join(format!("zs_massive_{lang}.jsonl")), &out)?;
        }
    }
    Ok(())
}

pub fn build_corpus(hf: &Hf, o: &CorpusOptions) -> Result<BTreeMap<String, usize>> {
    let q = o.out.join("queue");
    fs::create_dir_all(&q)?;
    let mut counts = BTreeMap::new();
    let p1 = p1_decisions(o)?;
    counts.insert(
        "p1-decisions".to_string(),
        write_jsonl(&q.join("p1-decisions.jsonl"), &p1)?,
    );
    heldout(o)?;
    let mut rng = Rng::derived(o.seed, "s4/p3");
    let mut p3 = tool_routing(o, &mut rng)?;
    match triage(hf, o, &mut rng) {
        Ok(v) => p3.extend(v),
        Err(e) => eprintln!("[s4] p3 triage skipped: {e:#}"),
    }
    match moderation(hf, o, &mut rng) {
        Ok(v) => p3.extend(v),
        Err(e) => eprintln!("[s4] p3 moderation skipped: {e:#}"),
    }
    rng.shuffle(&mut p3);
    counts.insert(
        "p3-cases".to_string(),
        write_jsonl(&q.join("p3-cases.jsonl"), &p3)?,
    );
    let p2 = p2_generic(o)?;
    counts.insert(
        "p2-generic".to_string(),
        write_jsonl(&q.join("p2-generic.jsonl"), &p2)?,
    );
    Ok(counts)
}

pub fn softmax_t(logits: &[f32], t: f32) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::MIN, f32::max);
    let e: Vec<f32> = logits.iter().map(|x| ((x - m) / t).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|x| x / s).collect()
}

fn soft_ce(gold: &[f32], p: &[f32]) -> f64 {
    gold.iter()
        .zip(p)
        .map(|(g, q)| -(*g as f64) * (q.max(1e-9) as f64).ln())
        .sum()
}

/// The teacher temperature minimizing the soft cross-entropy against the gold labels.
pub fn fit_temperature(recs: &[S4Record]) -> f32 {
    let gold: Vec<&S4Record> = recs
        .iter()
        .filter(|r| r.gold && r.tlogits.is_some() && r.rec.split == "train")
        .collect();
    if gold.is_empty() {
        return 1.0;
    }
    let mut best = (f64::MAX, 1.0f32);
    for i in 0..=55 {
        let t = 0.25 + 0.05 * i as f32;
        let ce: f64 = gold
            .iter()
            .map(|r| soft_ce(&r.rec.target, &softmax_t(r.tlogits.as_ref().unwrap(), t)))
            .sum::<f64>()
            / gold.len() as f64;
        if ce < best.0 {
            best = (ce, t);
        }
    }
    best.1
}

#[derive(Clone, Debug)]
pub struct MixOptions {
    pub scored: Vec<PathBuf>,
    pub out: PathBuf,
    pub alpha: f32,
    /// Fixed temperature (default: fitted on the gold records).
    pub temperature: Option<f32>,
}

/// Writes the mixed training records, one file per scored input, and returns the teacher
/// report (temperature, accuracy against gold per source).
pub fn mix(o: &MixOptions) -> Result<Value> {
    let mut all: Vec<(String, Vec<S4Record>)> = Vec::new();
    for p in &o.scored {
        if !p.exists() {
            continue;
        }
        let recs: Vec<S4Record> = read_jsonl(p)?;
        let name = p.file_stem().unwrap().to_string_lossy().to_string();
        all.push((name, recs));
    }
    if all.is_empty() {
        bail!("no scored file found");
    }
    let flat: Vec<S4Record> = all.iter().flat_map(|(_, v)| v.iter().cloned()).collect();
    let t = o.temperature.unwrap_or_else(|| fit_temperature(&flat));
    let mut per_src: BTreeMap<String, (usize, usize, f64)> = BTreeMap::new();
    for r in flat.iter().filter(|r| r.gold) {
        let Some(lg) = &r.tlogits else { continue };
        let p = softmax_t(lg, t);
        let e = per_src.entry(r.rec.src.clone()).or_default();
        e.0 += 1;
        e.1 += (argmax(&p) == argmax(&r.rec.target)) as usize;
        e.2 += soft_ce(&r.rec.target, &p);
    }
    fs::create_dir_all(&o.out)?;
    let mut written = BTreeMap::new();
    for (name, recs) in all {
        let out: Vec<DecisionRecord> = recs
            .into_iter()
            .filter_map(|r| {
                let lg = r.tlogits.as_ref()?;
                if lg.len() != r.rec.opts.len() {
                    return None;
                }
                let p = softmax_t(lg, t);
                let mut rec = r.rec;
                rec.target = if r.gold {
                    rec.target
                        .iter()
                        .zip(&p)
                        .map(|(g, q)| o.alpha * g + (1.0 - o.alpha) * q)
                        .collect()
                } else {
                    p
                };
                Some(rec)
            })
            .collect();
        let n = write_jsonl(&o.out.join(format!("{name}.jsonl")), &out)?;
        written.insert(name, n);
    }
    let report = json!({
        "temperature": t,
        "alpha": o.alpha,
        "written": written,
        "teacher_vs_gold": per_src.iter().map(|(k, (n, h, ce))| (k.clone(), json!({
            "n": n, "acc": *h as f64 / *n as f64, "ce": ce / *n as f64
        }))).collect::<serde_json::Map<_, _>>(),
    });
    fs::write(
        o.out.join("teacher-report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(opts: &[&str], target: &[f32]) -> DecisionRecord {
        DecisionRecord {
            src: "x".into(),
            t: "choice".into(),
            ins: "q".into(),
            opts: opts.iter().map(|s| s.to_string()).collect(),
            state: "s".into(),
            target: target.to_vec(),
            split: "train".into(),
        }
    }

    #[test]
    fn randomized_choice_keeps_the_label_or_abstains() {
        let r = rec(&["a: x", "b: y", "c: z"], &[0.0, 1.0, 0.0]);
        let pool = vec!["d: w".to_string(), "e: v".to_string()];
        let mut rng = Rng::new(1);
        for _ in 0..500 {
            let o = randomize_choice(&r, &pool, &mut rng);
            assert_eq!(o.opts.len(), o.target.len());
            assert!((o.target.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            let g = &o.opts[argmax(&o.target)];
            assert!(g == "b: y" || NONE_OPTS.contains(&g.as_str()), "{g}");
            if !o.opts.iter().any(|x| x == "b: y") {
                assert!(NONE_OPTS.contains(&g.as_str()));
            }
        }
    }

    #[test]
    fn temperature_fit_recovers_a_sharpening() {
        // Teacher logits that are too flat by a factor 2: the fitted T is near 0.5.
        let mut recs = Vec::new();
        let mut rng = Rng::new(3);
        for _ in 0..400 {
            let g = rng.below(3);
            let mut lg = vec![0.0f32; 3];
            lg[g] = 1.0;
            let p = softmax_t(&lg, 0.5);
            // Gold drawn from the sharp distribution.
            let u = rng.f64() as f32;
            let mut acc = 0.0;
            let mut k = 2;
            for (i, q) in p.iter().enumerate() {
                acc += q;
                if u < acc {
                    k = i;
                    break;
                }
            }
            let mut target = vec![0.0; 3];
            target[k] = 1.0;
            let mut r = S4Record::gold(rec(&["a", "b", "c"], &target));
            r.tlogits = Some(lg);
            recs.push(r);
        }
        let t = fit_temperature(&recs);
        assert!((0.35..=0.7).contains(&t), "{t}");
    }

    #[test]
    fn record_with_teacher_fields_round_trips() {
        let mut r = S4Record::teacher(rec(&["a", "b"], &[0.5, 0.5]));
        r.uid = Some(4);
        r.tlogits = Some(vec![1.0, -1.0]);
        let s = serde_json::to_string(&r).unwrap();
        assert!(s.contains("\"gold\":false") && s.contains("\"tlogits\""));
        let back: S4Record = serde_json::from_str(&s).unwrap();
        assert_eq!(back.uid, Some(4));
        assert_eq!(back.rec.opts.len(), 2);
    }
}
