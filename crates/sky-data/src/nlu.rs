//! Classic NLU / preference / tool-use datasets converted to typed questions: HelpSteer2,
//! PAWS-X (fr), Allociné, When2Call, SNLI and HellaSwag.

use anyhow::{anyhow, Result};
use serde_json::Value;

use crate::decisions::{mc_item, str_field, Ctx, Item, MNLI_LABELS, SMALL};
use crate::hf::for_each_parquet_row;
use crate::record::{one_hot, Question};
use crate::rng::hash64;
use crate::typed::{clip, state_group, STATE_MAX_CHARS};

// ----------------------------------------------------------------------------- HelpSteer2

/// HelpSteer2 attributes (0–4 Likert) with the question and one description per level,
/// following the dataset card's annotation guidelines.
pub const HELPSTEER_ATTRS: [(&str, &str, [&str; 5]); 5] = [
    (
        "helpfulness",
        "How helpful is the response to the prompt overall?",
        [
            "not helpful at all",
            "slightly helpful",
            "partially helpful",
            "mostly helpful",
            "perfectly helpful",
        ],
    ),
    (
        "correctness",
        "How correct is the response, including all pertinent facts without errors?",
        [
            "completely incorrect",
            "mostly incorrect",
            "partially correct",
            "mostly correct",
            "completely correct and accurate",
        ],
    ),
    (
        "coherence",
        "How coherent and clear is the response?",
        [
            "completely incoherent",
            "mostly incoherent",
            "somewhat coherent",
            "mostly coherent and clear",
            "perfectly coherent and clear",
        ],
    ),
    (
        "complexity",
        "How much expertise does it take to write the response?",
        [
            "basic language anyone can write",
            "simple language",
            "some education needed",
            "advanced, college-level language",
            "expert, domain-specific language",
        ],
    ),
    (
        "verbosity",
        "How verbose is the response relative to what the prompt asks for?",
        [
            "very succinct",
            "rather short",
            "average length",
            "rather long",
            "very verbose",
        ],
    ),
];

/// Character budgets of the HelpSteer2 state (prompt, then response).
pub const HELPSTEER_PROMPT_CHARS: usize = 1200;
pub const HELPSTEER_RESPONSE_CHARS: usize = 2400;

/// One HelpSteer2 row → one score question per attribute (state = prompt + response).
pub fn helpsteer_items(row: &Value) -> Result<Vec<Item>> {
    let prompt = str_field(row, "prompt")?;
    let response = str_field(row, "response")?;
    let state = format!(
        "Prompt: {}\nResponse: {}",
        clip(prompt, HELPSTEER_PROMPT_CHARS),
        clip(response, HELPSTEER_RESPONSE_CHARS)
    );
    let group = state_group(prompt);
    let mut out = Vec::new();
    for (attr, ins, levels) in HELPSTEER_ATTRS {
        let Some(v) = row
            .get(attr)
            .and_then(Value::as_i64)
            .filter(|v| (0..5).contains(v))
        else {
            continue;
        };
        let q = Question::score(ins, levels.iter().map(|s| s.to_string()).collect());
        out.push(Item {
            group: group.clone(),
            rec: q.record("helpsteer2", state.clone(), one_hot(5, v as usize)),
        });
    }
    Ok(out)
}

pub fn load_helpsteer2(cx: &Ctx) -> Result<Vec<Item>> {
    let mut cols = vec!["prompt", "response"];
    cols.extend(HELPSTEER_ATTRS.iter().map(|a| a.0));
    let mut out = Vec::new();
    for split in ["train", "validation"] {
        for p in cx
            .hf
            .parquet_split("nvidia/HelpSteer2", "default", split, SMALL)?
        {
            for_each_parquet_row(&p, &cols, |row| {
                out.extend(helpsteer_items(&row)?);
                Ok(())
            })?;
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- PAWS-X fr

pub const PAWS_INS: &str = "Do these two sentences mean the same thing?";

pub fn paws_item(s1: &str, s2: &str, label: i64) -> Option<Item> {
    if !(0..2).contains(&label) || s1.trim().is_empty() || s2.trim().is_empty() {
        return None;
    }
    let state = format!("Sentence 1: {}\nSentence 2: {}", s1.trim(), s2.trim());
    Some(Item {
        group: state_group(s1.trim()),
        rec: Question::noul(PAWS_INS).record("pawsx_fr", state, one_hot(2, label as usize)),
    })
}

pub fn load_pawsx_fr(cx: &Ctx) -> Result<Vec<Item>> {
    let mut out = Vec::new();
    for split in ["train", "validation"] {
        for p in cx
            .hf
            .parquet_split("google-research-datasets/paws-x", "fr", split, SMALL)?
        {
            for_each_parquet_row(&p, &["sentence1", "sentence2", "label"], |row| {
                let label = row.get("label").and_then(Value::as_i64).unwrap_or(-1);
                out.extend(paws_item(
                    str_field(&row, "sentence1")?,
                    str_field(&row, "sentence2")?,
                    label,
                ));
                Ok(())
            })?;
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- Allociné

pub const ALLOCINE_INS: &str = "Is this review positive?";

pub fn load_allocine(cx: &Ctx) -> Result<Vec<Item>> {
    let mut out = Vec::new();
    for p in cx
        .hf
        .parquet_split("tblard/allocine", "allocine", "train", SMALL)?
    {
        for_each_parquet_row(&p, &["review", "label"], |row| {
            let label = row.get("label").and_then(Value::as_i64).unwrap_or(-1);
            let review = clip(str_field(&row, "review")?, STATE_MAX_CHARS);
            if (0..2).contains(&label) && !review.is_empty() {
                out.push(Item {
                    group: state_group(&review),
                    rec: Question::noul(ALLOCINE_INS).record(
                        "allocine",
                        review,
                        one_hot(2, label as usize),
                    ),
                });
            }
            Ok(())
        })?;
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- QASC

pub const QASC_INS: &str = "Which option correctly answers the question, given the two facts?";

/// QASC (CC-BY-4.0): eight options, the two supporting facts before the question in the state.
pub fn qasc_item(
    id: &str,
    fact1: &str,
    fact2: &str,
    question: &str,
    choices: &Value,
    answer_key: &str,
) -> Option<Item> {
    let state = format!("{} {}\n{}", fact1.trim(), fact2.trim(), question.trim());
    let mut it = mc_item("qasc", QASC_INS, id, &state, choices, answer_key)?;
    it.group = id.to_string();
    Some(it)
}

pub fn load_qasc(cx: &Ctx) -> Result<Vec<Item>> {
    let mut out = Vec::new();
    // The test split has no answer key: train + validation only.
    for split in ["train", "validation"] {
        for p in cx
            .hf
            .parquet_split("allenai/qasc", "default", split, SMALL)?
        {
            for_each_parquet_row(
                &p,
                &["id", "question", "choices", "answerKey", "fact1", "fact2"],
                |row| {
                    out.extend(qasc_item(
                        str_field(&row, "id")?,
                        str_field(&row, "fact1")?,
                        str_field(&row, "fact2")?,
                        str_field(&row, "question")?,
                        &row["choices"],
                        str_field(&row, "answerKey")?,
                    ));
                    Ok(())
                },
            )?;
        }
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- refusals

pub const REFUSAL_INS: &str = "Does the assistant refuse or decline the user's request?";

/// Characters kept of the user turn and of the assistant reply.
const REFUSAL_TURN_CHARS: usize = 600;

/// One `agentlans/refusal-classifier-data` row (CC-BY-4.0): the conversation
/// (`<|system|>…<|user|>…<|assistant|>…`) reduced to the last user turn and the reply, so that
/// clipping never cuts the reply the question is about; `Refusal` / `Non-refusal` as the label.
pub fn refusal_item(input: &str, output: &str) -> Option<Item> {
    let (before, reply) = input.rsplit_once("<|assistant|>")?;
    let user = before.rsplit_once("<|user|>").map_or(before, |(_, u)| u);
    let clean = |s: &str| clip(&s.replace("<|...|>", " … "), REFUSAL_TURN_CHARS);
    let (user, reply) = (clean(user), clean(reply));
    if reply.is_empty() {
        return None;
    }
    let refusal = match output.trim() {
        "Refusal" => 1,
        "Non-refusal" => 0,
        _ => return None,
    };
    let state = format!("User: {user}\nAssistant: {reply}");
    Some(Item {
        group: state_group(&state),
        rec: Question::noul(REFUSAL_INS).record("refusal", state, one_hot(2, refusal)),
    })
}

pub fn load_refusal(cx: &Ctx) -> Result<Vec<Item>> {
    let mut out = Vec::new();
    for p in cx.hf.parquet_split(
        "agentlans/refusal-classifier-data",
        "english_only",
        "train",
        SMALL,
    )? {
        for_each_parquet_row(&p, &["input", "output"], |row| {
            out.extend(refusal_item(
                str_field(&row, "input")?,
                str_field(&row, "output")?,
            ));
            Ok(())
        })?;
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- When2Call

pub const WHEN2CALL_INS: &str =
    "How should the assistant respond to the user's request, given the available tools?";

/// Response categories, in option order.
pub const WHEN2CALL_OPTS: [(&str, &str); 4] = [
    ("tool_call", "call one of the available tools"),
    (
        "request_for_info",
        "ask the user for missing information needed to call a tool",
    ),
    (
        "cannot_answer",
        "explain that the available tools cannot fulfil the request",
    ),
    ("direct_answer", "answer directly without calling any tool"),
];

/// Labels a When2Call assistant response: `<TOOLCALL>` → tool_call; otherwise an apology or
/// a statement of inability → cannot_answer; a question asking the user for details →
/// request_for_info; anything else → direct_answer.
pub fn when2call_label(response: &str) -> usize {
    let r = response.trim();
    if r.contains("<TOOLCALL>") || r.contains("<tool_call>") {
        return 0;
    }
    let l = r.to_lowercase();
    let unable = [
        "unable to",
        "i can't",
        "i cannot",
        "i can not",
        "i'm not able",
        "i am not able",
        "don't have the capability",
        "do not have the capability",
        "don't have access",
        "beyond my capabilities",
        "not able to",
    ]
    .iter()
    .any(|p| l.contains(p));
    let apology = ["apolog", "i'm sorry", "i am sorry", "sorry,"]
        .iter()
        .any(|p| l.starts_with(p) || l.contains(p));
    if unable || (apology && !l.contains('?')) {
        return 2;
    }
    let asks = l.contains('?')
        && [
            "could you",
            "can you",
            "please provide",
            "please specify",
            "i'll need",
            "i need",
            "i will need",
            "would you",
            "which",
            "what is the",
            "do you",
        ]
        .iter()
        .any(|p| l.contains(p));
    if asks {
        return 1;
    }
    3
}

/// Compact tool list for the state: `name: description` per tool.
pub fn tools_summary(tools: &[String]) -> String {
    if tools.is_empty() {
        return "none".into();
    }
    tools
        .iter()
        .map(|t| match serde_json::from_str::<Value>(t) {
            Ok(v) => {
                let name = v.get("name").and_then(Value::as_str).unwrap_or("?");
                let desc = v.get("description").and_then(Value::as_str).unwrap_or("");
                format!("{name}: {}", clip(desc, 200))
            }
            Err(_) => clip(t, 200),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn when2call_item(question: &str, tools: &[String], label: usize) -> Item {
    let state = clip(
        &format!(
            "User: {}\nTools:\n{}",
            question.trim(),
            tools_summary(tools)
        ),
        STATE_MAX_CHARS,
    );
    let crit = WHEN2CALL_OPTS
        .iter()
        .map(|(k, d)| (k.to_string(), d.to_string()))
        .collect();
    Item {
        group: state_group(question.trim()),
        rec: Question::choice(WHEN2CALL_INS, crit).record(
            "when2call",
            state,
            one_hot(WHEN2CALL_OPTS.len(), label),
        ),
    }
}

fn tool_list(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|t| t.as_str().map_or_else(|| t.to_string(), str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

/// The first user message of a chat.
fn first_user(messages: &Value) -> Option<String> {
    messages
        .as_array()?
        .iter()
        .find(|m| m.get("role").and_then(Value::as_str) == Some("user"))
        .and_then(|m| m.get("content")?.as_str().map(str::to_string))
}

pub fn load_when2call(cx: &Ctx) -> Result<Vec<Item>> {
    let repo = "nvidia/When2Call";
    let mut out = Vec::new();
    // Training data: SFT responses and the preferred responses of the preference set,
    // labelled by `when2call_label`.
    for p in cx.hf.parquet_split(repo, "train_sft", "train", SMALL)? {
        for_each_parquet_row(&p, &["messages", "tools"], |row| {
            let msgs = row
                .get("messages")
                .and_then(Value::as_array)
                .ok_or_else(|| anyhow!("messages"))?;
            let (Some(user), Some(last)) = (first_user(&row["messages"]), msgs.last()) else {
                return Ok(());
            };
            if last.get("role").and_then(Value::as_str) != Some("assistant") || msgs.len() != 2 {
                return Ok(());
            }
            let resp = last.get("content").and_then(Value::as_str).unwrap_or("");
            out.push(when2call_item(
                &user,
                &tool_list(&row["tools"]),
                when2call_label(resp),
            ));
            Ok(())
        })?;
    }
    for p in cx.hf.parquet_split(repo, "train_pref", "train", SMALL)? {
        for_each_parquet_row(&p, &["messages", "chosen_response", "tools"], |row| {
            let Some(user) = first_user(&row["messages"]) else {
                return Ok(());
            };
            if row["messages"].as_array().map_or(0, Vec::len) != 1 {
                return Ok(());
            }
            let resp = row["chosen_response"]
                .get("content")
                .and_then(Value::as_str)
                .unwrap_or("");
            out.push(when2call_item(
                &user,
                &tool_list(&row["tools"]),
                when2call_label(resp),
            ));
            Ok(())
        })?;
    }
    let mut train_labels = [0usize; 4];
    for it in &out {
        train_labels[crate::record::argmax(&it.rec.target)] += 1;
    }
    eprintln!("  when2call: training labels (tool_call, request_for_info, cannot_answer, direct_answer) = {train_labels:?}");
    // Official labelled benchmark: the multiple-choice test split, kept as `test`. Its
    // reference answer texts also measure the response-label heuristic (tool calls are
    // excluded: there they are bare JSON, whereas training responses carry <TOOLCALL>).
    let mut confusion = [[0usize; 4]; 4];
    for p in cx.hf.parquet_split(repo, "test", "mcq", SMALL)? {
        for_each_parquet_row(
            &p,
            &["question", "tools", "correct_answer", "answers"],
            |row| {
                let canon = |k: &str| {
                    if k == "direct" {
                        "direct_answer".to_string()
                    } else {
                        k.to_string()
                    }
                };
                let key = canon(str_field(&row, "correct_answer")?);
                let Some(label) = WHEN2CALL_OPTS.iter().position(|(k, _)| *k == key) else {
                    return Ok(());
                };
                if let Some(answers) = row.get("answers").and_then(Value::as_object) {
                    for (k, v) in answers {
                        let k = canon(k);
                        if let (Some(want), Some(text)) =
                            (WHEN2CALL_OPTS.iter().position(|(o, _)| *o == k), v.as_str())
                        {
                            if want != 0 {
                                confusion[want][when2call_label(text)] += 1;
                            }
                        }
                    }
                }
                let mut it = when2call_item(
                    str_field(&row, "question")?,
                    &tool_list(&row["tools"]),
                    label,
                );
                it.rec.split = "test".into();
                out.push(it);
                Ok(())
            },
        )?;
    }
    let right: usize = (0..4).map(|i| confusion[i][i]).sum();
    let total: usize = confusion.iter().flatten().sum();
    eprintln!(
        "  when2call: response-label heuristic on the mcq reference texts: {right}/{total} ({:.1}%), confusion rows=truth cols=pred {confusion:?}",
        100.0 * right as f64 / total.max(1) as f64
    );
    Ok(out)
}

// ----------------------------------------------------------------------------- SNLI

pub fn snli_item(premise: &str, hypothesis: &str, label: usize) -> Item {
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
        rec: Question::choice(ins, crit).record("snli", premise.trim(), one_hot(3, label)),
    }
}

pub fn load_snli(cx: &Ctx) -> Result<Vec<Item>> {
    let mut out = Vec::new();
    for p in cx
        .hf
        .parquet_split("stanfordnlp/snli", "plain_text", "train", SMALL)?
    {
        for_each_parquet_row(&p, &["premise", "hypothesis", "label"], |row| {
            let label = row.get("label").and_then(Value::as_i64).unwrap_or(-1);
            if (0..3).contains(&label) {
                out.push(snli_item(
                    str_field(&row, "premise")?,
                    str_field(&row, "hypothesis")?,
                    label as usize,
                ));
            }
            Ok(())
        })?;
    }
    Ok(out)
}

// ----------------------------------------------------------------------------- HellaSwag

pub const HELLASWAG_INS: &str = "Which ending most plausibly continues the description?";

pub fn hellaswag_item(activity: &str, ctx: &str, endings: &Value, label: &str) -> Option<Item> {
    let n = endings.as_array()?.len();
    let labels: Vec<String> = (0..n).map(|i| i.to_string()).collect();
    let texts: Vec<String> = endings
        .as_array()?
        .iter()
        .map(|e| e.as_str().map(|s| s.trim().to_string()))
        .collect::<Option<_>>()?;
    let state = format!("{}: {}", activity.trim(), ctx.trim());
    let choices = serde_json::json!({"label": labels, "text": texts});
    let mut it = mc_item("hellaswag", HELLASWAG_INS, "", &state, &choices, label)?;
    it.group = state_group(&state);
    Some(it)
}

pub fn load_hellaswag(cx: &Ctx) -> Result<Vec<Item>> {
    let mut out = Vec::new();
    for split in ["train", "validation"] {
        for p in cx
            .hf
            .parquet_split("Rowan/hellaswag", "default", split, SMALL)?
        {
            for_each_parquet_row(&p, &["activity_label", "ctx", "endings", "label"], |row| {
                out.extend(hellaswag_item(
                    str_field(&row, "activity_label")?,
                    str_field(&row, "ctx")?,
                    &row["endings"],
                    row.get("label").and_then(Value::as_str).unwrap_or(""),
                ));
                Ok(())
            })?;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{argmax, validate_record};
    use serde_json::json;

    #[test]
    fn qasc_puts_the_facts_before_the_question() {
        let choices = json!({"text": ["pearls", "streams", "beads"], "label": ["A", "B", "C"]});
        let it = qasc_item(
            "q1",
            "beads of water are formed by water vapor condensing",
            "Clouds are made of water vapor.",
            "What type of water formation is formed by clouds?",
            &choices,
            "C",
        )
        .unwrap();
        assert_eq!(it.rec.ins, QASC_INS);
        assert!(it.rec.state.starts_with("beads of water") && it.rec.state.ends_with("clouds?"));
        assert_eq!(argmax(&it.rec.target), 2);
        validate_record(&it.rec).unwrap();
        // The test split has no answer key.
        assert!(qasc_item("q2", "a", "b", "c?", &choices, "").is_none());
    }

    #[test]
    fn refusal_keeps_the_last_turn_and_the_reply() {
        let long = "x".repeat(5000);
        let input = format!(
            "<|system|>You are a recipe bot.<|user|>{long}<|assistant|>Earlier reply.<|user|>How do I join the group?<|assistant|>I'm sorry, but I can only provide cooking recipes."
        );
        let it = refusal_item(&input, "Refusal").unwrap();
        assert_eq!(it.rec.t, "noul");
        assert!(it.rec.state.starts_with("User: How do I join the group?"));
        assert!(it.rec.state.ends_with("only provide cooking recipes."));
        assert_eq!(argmax(&it.rec.target), 1);
        validate_record(&it.rec).unwrap();
        let no = refusal_item("<|user|>2+2?<|assistant|>4", "Non-refusal").unwrap();
        assert_eq!(argmax(&no.rec.target), 0);
        assert!(refusal_item("<|user|>hi", "Refusal").is_none());
        assert!(refusal_item("<|user|>hi<|assistant|>ok", "maybe").is_none());
    }

    #[test]
    fn helpsteer_one_question_per_attribute() {
        let row = json!({"prompt":"p","response":"r","helpfulness":3,"correctness":4,"coherence":4,"complexity":2,"verbosity":1});
        let items = helpsteer_items(&row).unwrap();
        assert_eq!(items.len(), 5);
        for it in &items {
            validate_record(&it.rec).unwrap();
            assert_eq!(it.rec.opts.len(), 5);
            assert_eq!(it.rec.state, "Prompt: p\nResponse: r");
        }
        assert_eq!(argmax(&items[0].rec.target), 3);
        assert_eq!(items[4].rec.opts[1], "level 1: rather short");
    }

    #[test]
    fn simple_mappings() {
        let p = paws_item("A.", "B.", 1).unwrap();
        assert_eq!(p.rec.target, vec![0.0, 1.0]);
        assert!(paws_item("A.", "B.", -1).is_none());
        let s = snli_item("P.", "H.", 0);
        assert_eq!(
            s.rec.opts[0],
            "entailment: the premise implies the hypothesis is true"
        );
        let h = hellaswag_item("Act", "ctx then", &json!(["a", "b", "c", "d"]), "2").unwrap();
        assert_eq!(h.rec.opts, vec!["A: a", "B: b", "C: c", "D: d"]);
        assert_eq!(h.rec.target, vec![0.0, 0.0, 1.0, 0.0]);
        assert_eq!(h.rec.state, "Act: ctx then");
        assert!(hellaswag_item("Act", "c", &json!(["a", "b"]), "").is_none());
    }

    #[test]
    fn when2call_labels() {
        assert_eq!(
            when2call_label("<TOOLCALL>[{\"name\": \"f\"}]</TOOLCALL>"),
            0
        );
        assert_eq!(
            when2call_label("To proceed, could you please provide the ID?"),
            1
        );
        assert_eq!(
            when2call_label("Apologies, but I'm unable to perform web searches."),
            2
        );
        assert_eq!(
            when2call_label("I'm sorry, I can't provide that information."),
            2
        );
        assert_eq!(when2call_label("The CMYK value is (0, 100, 100, 0)."), 3);
        let it = when2call_item(
            "hi",
            &[r#"{"name":"f","description":"does f"}"#.to_string()],
            1,
        );
        assert_eq!(it.rec.state, "User: hi\nTools:\nf: does f");
        assert_eq!(
            it.rec.opts[3],
            "direct_answer: answer directly without calling any tool"
        );
        validate_record(&it.rec).unwrap();
        assert_eq!(tools_summary(&[]), "none");
    }
}
