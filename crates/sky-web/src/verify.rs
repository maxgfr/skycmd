//! The secrets verifier (s6-secrets-verify-pico, `docs/s6-secrets2-prereg.md` section 4): one
//! rule candidate in, P(real credential) out, at full depth.
//!
//! A candidate is a JSON object in the terms of the generator's record (`sg-cand-1`,
//! secretgate's `scripts/rule-candidates.mjs`); a record of the generator is accepted as it is.
//! The state the model reads is built from it by `sky_infer::secret_cand`, the builder the
//! training data was made with, with the model's own tokenizer and the given word list.
//!
//! # The candidate JSON
//!
//! Required:
//! - `rule` (string): the rule id;
//! - `value` (string): the candidate value.
//!
//! Optional (absent and `null` mean the same; any other type is an error):
//! - `file` (string): the file's path relative to its repository root, `/`-separated (an
//!   absolute path brings the words of the directories above the repository into the scope).
//!   Without `scope`, both the scope words and the extension come from it
//!   (`secret_cand::path_scope`) and `ext` is ignored, as the training build did for real paths;
//! - `scope` (array of strings): the scope words themselves, each a word of CredData's
//!   `word_in_path.json` without its slashes (`test`, `src`, `conf`…), in that list's order; `[]`
//!   is no scope (written `_`). Use it when the path is not a real one (CredData's stored paths);
//! - `ext` (string): the extension, lower-case, with its dot (`.py`); used with `scope`, or when
//!   there is no `file`. Default: none;
//! - `name` (string): the variable or key name. Default: none;
//! - `lineText` (string): the line of the value's first character, without its line ending (or
//!   the generator's window of a long line). Default: empty;
//! - `start`, `end` (non-negative integers): the value's columns in the full line, in UTF-16
//!   code units (JavaScript string indices), `end` on the value's last line. Required when
//!   `lineText` is not empty;
//! - `lineFrom` (non-negative integer): the UTF-16 column at which `lineText` starts in the full
//!   line. Default 0;
//! - `line`, `endLine` (positive integers): the lines of the value's first and last characters.
//!   Default: `line` 1, `endLine` `line` plus the line breaks of the value before its last
//!   character;
//! - `prevLine`, `nextLine` (strings): the lines before the value's first and after its last.
//!   Default: none;
//! - `filters` (object of booleans): whether the default engine would have dropped the value,
//!   `entropy`, `placeholder`, `stopword`, `allowlist` (each default `false`); `context` and
//!   `pragma` are type-checked and not part of the state;
//! - `entropy` (number): type-checked and not used: the state computes the entropy itself, as
//!   the training data did;
//! - `masked` (boolean): `true` is refused (a masked record does not carry its value).
//!
//! Any other field (`generator`, `revision`, `cpStart`, `offset`, `reported`…) is ignored.

use serde_json::{json, Map, Value};
use sky_infer::prompt::{NOUL_FALSE_DEFAULT, NOUL_TRUE_DEFAULT};
use sky_infer::secret_cand::{self as sc, Candidate, Filters, WordSet, Q_REAL, WORD_IN_PATH};
use sky_infer::tokenizer::Tokenizer;
use sky_infer::QType;

use crate::{clock, Model};

/// The three operating thresholds on P(real) in the model's metadata, by the name the API gives
/// them.
pub const TAU_KEYS: [(&str, &str); 3] = [
    ("tau_f1", "skycmd-s2.secrets2.tau_f1"),
    ("tau_f05", "skycmd-s2.secrets2.tau_f05"),
    ("tau_f2", "skycmd-s2.secrets2.tau_f2"),
];

/// A candidate read from JSON, owning its text.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedCandidate {
    pub rule: String,
    pub scope: Vec<&'static str>,
    pub ext: String,
    pub name: Option<String>,
    pub value: String,
    pub line_text: String,
    pub value_start: usize,
    pub value_end: usize,
    pub prev_line: Option<String>,
    pub next_line: Option<String>,
    pub filters: Filters,
}

impl OwnedCandidate {
    pub fn as_candidate(&self) -> Candidate<'_> {
        Candidate {
            rule: &self.rule,
            scope: &self.scope,
            ext: &self.ext,
            name: self.name.as_deref(),
            value: &self.value,
            line_text: &self.line_text,
            value_start: self.value_start,
            value_end: self.value_end,
            prev_line: self.prev_line.as_deref(),
            next_line: self.next_line.as_deref(),
            filters: self.filters,
        }
    }
}

fn kind(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

/// The field `key` of `o`, `None` when absent or `null`.
fn field<'a>(o: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    o.get(key).filter(|v| !v.is_null())
}

fn opt_str(o: &Map<String, Value>, key: &str) -> Result<Option<String>, String> {
    match field(o, key) {
        None => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.clone())),
        Some(v) => Err(format!("`{key}`: expected a string, got {}", kind(v))),
    }
}

fn req_str(o: &Map<String, Value>, key: &str) -> Result<String, String> {
    opt_str(o, key)?.ok_or_else(|| format!("`{key}`: required (a string)"))
}

fn opt_uint(o: &Map<String, Value>, key: &str) -> Result<Option<u64>, String> {
    match field(o, key) {
        None => Ok(None),
        Some(v) => v.as_u64().map(Some).ok_or_else(|| {
            format!(
                "`{key}`: expected a non-negative integer, got {}",
                match v {
                    Value::Number(_) => "another number",
                    v => kind(v),
                }
            )
        }),
    }
}

fn opt_bool(o: &Map<String, Value>, key: &str, at: &str) -> Result<bool, String> {
    match field(o, key) {
        None => Ok(false),
        Some(Value::Bool(b)) => Ok(*b),
        Some(v) => Err(format!("`{at}{key}`: expected a boolean, got {}", kind(v))),
    }
}

/// The scope words of [`WORD_IN_PATH`], without their slashes.
fn scope_word(w: &str) -> Option<&'static str> {
    WORD_IN_PATH
        .iter()
        .map(|x| x.trim_matches('/'))
        .find(|x| *x == w)
}

/// Reads one candidate (see the module documentation for the accepted JSON).
pub fn parse_candidate(v: &Value) -> Result<OwnedCandidate, String> {
    let Value::Object(o) = v else {
        return Err(format!("a candidate is a JSON object, got {}", kind(v)));
    };
    if opt_bool(o, "masked", "")? {
        return Err("`masked`: a masked record does not carry its value".into());
    }
    let rule = req_str(o, "rule")?;
    let value = req_str(o, "value")?;
    let file = opt_str(o, "file")?;
    let ext = opt_str(o, "ext")?;
    let name = opt_str(o, "name")?;
    let line_text = opt_str(o, "lineText")?.unwrap_or_default();
    let prev_line = opt_str(o, "prevLine")?;
    let next_line = opt_str(o, "nextLine")?;
    match field(o, "entropy") {
        None | Some(Value::Number(_)) => {}
        Some(v) => return Err(format!("`entropy`: expected a number, got {}", kind(v))),
    }

    let scope = match field(o, "scope") {
        None => None,
        Some(Value::Array(a)) => Some(
            a.iter()
                .enumerate()
                .map(|(i, w)| match w {
                    Value::String(s) => scope_word(s).ok_or_else(|| {
                        format!("`scope[{i}]`: not a word of CredData's word_in_path.json")
                    }),
                    w => Err(format!("`scope[{i}]`: expected a string, got {}", kind(w))),
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        Some(v) => return Err(format!("`scope`: expected an array, got {}", kind(v))),
    };
    let (scope, ext) = match (scope, file) {
        (Some(scope), file) => {
            let ext =
                ext.unwrap_or_else(|| file.map_or(String::new(), |f| sc::split_extension(&f).1));
            (scope, ext)
        }
        (None, Some(file)) => sc::path_scope(&file),
        (None, None) => (Vec::new(), ext.unwrap_or_default()),
    };

    let start = opt_uint(o, "start")?;
    let end = opt_uint(o, "end")?;
    let line_from = opt_uint(o, "lineFrom")?.unwrap_or(0);
    let line = opt_uint(o, "line")?.unwrap_or(1);
    // the line of the value's last character, as the generator counts it
    let breaks = value
        .char_indices()
        .last()
        .map_or(0, |(i, _)| value[..i].matches('\n').count()) as u64;
    let end_line = opt_uint(o, "endLine")?.unwrap_or(line + breaks);
    let (value_start, value_end) = match (start, end) {
        (Some(s), Some(e)) => {
            sc::value_columns(line, end_line, s as usize, e as usize, line_from as usize)
        }
        _ if line_text.is_empty() => (0, 0),
        _ => return Err("`start` and `end` are required when `lineText` is given".into()),
    };

    let filters = match field(o, "filters") {
        None => Filters::default(),
        Some(Value::Object(f)) => {
            for extra in ["context", "pragma"] {
                opt_bool(f, extra, "filters.")?;
            }
            Filters {
                entropy: opt_bool(f, "entropy", "filters.")?,
                placeholder: opt_bool(f, "placeholder", "filters.")?,
                stopword: opt_bool(f, "stopword", "filters.")?,
                allowlist: opt_bool(f, "allowlist", "filters.")?,
            }
        }
        Some(v) => return Err(format!("`filters`: expected an object, got {}", kind(v))),
    };

    Ok(OwnedCandidate {
        rule,
        scope,
        ext,
        name,
        value,
        line_text,
        value_start,
        value_end,
        prev_line,
        next_line,
        filters,
    })
}

/// The state of a candidate given as JSON text, with a tokenizer and a word list.
pub fn state_of_json(json: &str, tok: &Tokenizer, words: &WordSet) -> Result<String, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("candidate: {e}"))?;
    let c = parse_candidate(&v)?;
    Ok(sc::state(&c.as_candidate(), tok, words))
}

/// A loaded verifier: the model, its question and the word list.
pub struct Verifier {
    model: Model,
    words: WordSet,
    options: [String; 2],
    thresholds: Value,
    window: usize,
}

impl Verifier {
    /// From the GGUF bytes of an s2 text model (s6-secrets-verify-pico) and the bytes of the
    /// word list (UTF-8, one entry per line: the build's `dictionary.txt`). The model answers at
    /// full depth, whatever its stored exit thresholds.
    pub fn load(gguf: &[u8], words: &[u8]) -> Result<Verifier, String> {
        let text = std::str::from_utf8(words).map_err(|e| format!("word list: not UTF-8 ({e})"))?;
        let words = WordSet::from_text(text);
        if words.is_empty() {
            return Err("word list: no lower-case word of 4 to 12 letters".into());
        }
        let mut model = Model::load(gguf)?;
        let window = match model.engine().s2_model() {
            Some(m) if !m.cfg.is_audio() => m.cfg.max_state_tokens,
            _ => return Err("the verifier needs an s2 text model (s6-secrets-verify-pico)".into()),
        };
        model.set_exit_threshold(Some(1.0));
        let g = sky_gguf::Gguf::parse(gguf).map_err(|e| e.to_string())?;
        let thresholds: Map<String, Value> = TAU_KEYS
            .iter()
            .map(|(name, key)| (name.to_string(), json!(g.get_f64(key))))
            .collect();
        Ok(Verifier {
            model,
            words,
            options: [
                format!("false: {NOUL_FALSE_DEFAULT}"),
                format!("true: {NOUL_TRUE_DEFAULT}"),
            ],
            thresholds: Value::Object(thresholds),
            window,
        })
    }

    /// `{tau_f1, tau_f05, tau_f2}`: the operating thresholds stored in the file (`null` when a
    /// key is absent). A candidate is accepted when `p_real >= tau`.
    pub fn thresholds(&self) -> &Value {
        &self.thresholds
    }

    /// `{name, n_params, file_bytes, window, words}`: the model and the word list loaded.
    pub fn info(&self) -> Value {
        json!({
            "name": self.model.name(),
            "n_params": self.model.n_params(),
            "file_bytes": self.model.file_bytes(),
            "window": self.window,
            "words": self.words.len(),
        })
    }

    fn state_of(&self, c: &OwnedCandidate) -> String {
        sc::state(
            &c.as_candidate(),
            self.model.engine().tokenizer(),
            &self.words,
        )
    }

    /// The state text of a candidate given as JSON.
    pub fn state(&self, json: &str) -> Result<String, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| format!("candidate: {e}"))?;
        Ok(self.state_of(&parse_candidate(&v)?))
    }

    /// Judges one candidate: `{p_real, state_tokens, truncated, micros}`. `micros` is the time
    /// from the candidate read to its answer (state, tokenization, forward).
    fn judge_one(&mut self, c: &OwnedCandidate) -> Result<Value, String> {
        let t0 = clock::now_us();
        let state = self.state_of(c);
        let state_tokens = self
            .model
            .engine_mut()
            .s2_engine_mut()
            .map_or(0, |e| e.state_len_uncut(&state));
        let probs = self
            .model
            .probs(QType::Noul, Q_REAL, &self.options, &state)?;
        let micros = clock::now_us() - t0;
        Ok(json!({
            "p_real": probs[1],
            "state_tokens": state_tokens,
            "truncated": state_tokens > self.window,
            "micros": micros,
        }))
    }

    /// Judges one candidate given as JSON.
    pub fn judge(&mut self, json: &str) -> Result<Value, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| format!("candidate: {e}"))?;
        let c = parse_candidate(&v)?;
        self.judge_one(&c)
    }

    /// Judges a JSON array of candidates: an array of answers in the same order. Every
    /// candidate is read before any is judged, and a malformed one fails the whole call with
    /// its index.
    pub fn judge_many(&mut self, json: &str) -> Result<Value, String> {
        let v: Value = serde_json::from_str(json).map_err(|e| format!("candidates: {e}"))?;
        let Value::Array(items) = v else {
            return Err(format!("candidates: expected an array, got {}", kind(&v)));
        };
        let cands = items
            .iter()
            .enumerate()
            .map(|(i, c)| parse_candidate(c).map_err(|e| format!("candidate {i}: {e}")))
            .collect::<Result<Vec<_>, _>>()?;
        cands
            .iter()
            .map(|c| self.judge_one(c))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::Array)
    }
}
