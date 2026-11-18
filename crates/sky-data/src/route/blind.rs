//! The request-blind reading of s6-route v6 (`docs/s6-route6-prereg.md` 4.3): a slate with its
//! request replaced by a placeholder P that yields no BM25 token, so that every score D4 writes
//! is 0.00 and the ranks are the options' positions.
//!
//! - [`check_placeholder`] (`sky-data route blind --check-placeholder <files> --out
//!   placeholder.json`): P is `?`, else `…`, else `-`, the first that is no request of the files
//!   (training and validation requests; T5's in the counts amendment, with `--keep`).
//! - [`blind`] (`sky-data route blind --in <slate file> --out <stem> --placeholder
//!   placeholder.json [--render bm25-line|scores-in-options]`): `<stem>.blind.jsonl`, the slate
//!   with the state `User: <P>` and its options plain; `<stem>.blind.model.jsonl`, B0, by the
//!   model file's own rendering; `<stem>.blind-bare.model.jsonl`, B1 (no score, no rank, no BM25
//!   line). Ids, targets, kinds, roles and every other field are kept.
//!
//! B0 needs no engine change: `sky-route fuse` reads `<stem>.blind.jsonl`, takes its request P,
//! computes the scores itself (all 0) and renders them as the model file says; the embedding part
//! reads P. A blind slate is never trained on (§12).

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use sky_infer::route::{plain_option, tokens, Render, LINE_PREFIX};

use super::{read_json, read_lines, sha256_file, write_json};
use crate::record::write_jsonl;

/// P and its fallbacks, in order (4.3): none yields a BM25 token, each is a token of
/// potion-base-2M.
pub const PLACEHOLDERS: [&str; 3] = ["?", "\u{2026}", "-"];

/// The request of a record's state: `User: <request>`, without v2's BM25 line, trimmed.
pub fn request_of(state: &str) -> &str {
    let s = state
        .rsplit_once(&format!("\n{LINE_PREFIX}"))
        .map_or(state, |(a, _)| a);
    s.strip_prefix("User: ").unwrap_or(s).trim()
}

#[derive(Deserialize)]
struct StateOnly {
    #[serde(default)]
    state: String,
}

/// 4.3: the first of [`PLACEHOLDERS`] that is no request of `files` (after trimming), written
/// to `out` with the files' hashes and counts; `taken` is null when every one occurs. With
/// `keep`, the placeholder of that file is checked instead (`occurs`: whether it is a request of
/// the files) and nothing is chosen. No request is printed or written.
pub fn check_placeholder(files: &[PathBuf], out: &Path, keep: Option<&Path>) -> Result<Value> {
    let mut found: HashSet<&'static str> = HashSet::new();
    let mut read = Vec::new();
    let mut n = 0usize;
    for f in files {
        let recs: Vec<StateOnly> = read_lines(f)?;
        for r in &recs {
            let q = request_of(&r.state);
            for p in PLACEHOLDERS {
                if q == p {
                    found.insert(p);
                }
            }
        }
        n += recs.len();
        read.push(json!({"file": f.display().to_string(), "records": recs.len(), "sha256": sha256_file(f)?}));
    }
    let occurs: Vec<Value> = PLACEHOLDERS
        .iter()
        .map(|p| json!({"placeholder": p, "a request": found.contains(p)}))
        .collect();
    if let Some(k) = keep {
        let p = placeholder_of(k)?;
        let hit = found.contains(p.as_str());
        let r = json!({"about": "s6-route v6 (docs/s6-route6-prereg.md 4.3, 6): the placeholder taken, checked against the requests of these files",
                       "placeholder": p, "occurs": hit, "records": n, "files": read});
        write_json(out, &r)?;
        return Ok(r);
    }
    let taken = PLACEHOLDERS.iter().find(|p| !found.contains(*p)).copied();
    let r = json!({"about": "s6-route v6 (docs/s6-route6-prereg.md 4.3): the placeholder of the blind reading, the first of ?, …, - that is no training or validation request; it never changes after training",
                   "placeholder": taken, "taken": taken, "candidates": occurs, "records": n, "files": read});
    write_json(out, &r)?;
    Ok(r)
}

/// Whether a report of [`check_placeholder`] refuses (exit 2 on the command line): with `keep`,
/// when the placeholder kept occurs as a request (its report holds `occurs` and no `taken`);
/// otherwise when every one of [`PLACEHOLDERS`] occurs (`taken` is null).
pub fn placeholder_refused(r: &Value) -> bool {
    match r.get("occurs") {
        Some(o) => o.as_bool() != Some(false),
        None => r["taken"].is_null(),
    }
}

/// The placeholder of `placeholder.json`, checked: one of [`PLACEHOLDERS`], no BM25 token.
pub fn placeholder_of(path: &Path) -> Result<String> {
    let v = read_json(path)?;
    let p = v["placeholder"]
        .as_str()
        .with_context(|| format!("{}: no placeholder", path.display()))?;
    if !PLACEHOLDERS.contains(&p) || !tokens(p).is_empty() {
        bail!("{}: {p:?} is not a placeholder of 4.3", path.display());
    }
    Ok(p.to_string())
}

/// A path under a `fresh` folder.
fn is_fresh(p: &Path) -> bool {
    p.components().any(|c| c.as_os_str() == "fresh")
}

/// The three copies of 4.3 (see the module). A file under a `fresh` folder is refused unless
/// `open` names an existing FRESH-OPEN file. Every tool option of B0 must read `score 0.00 rank
/// <i>; ` in slate order (D4) or the state's BM25 line must hold scores of 0.00 only (v2's line).
pub fn blind(
    input: &Path,
    stem: &Path,
    placeholder: &str,
    how: Render,
    open: Option<&Path>,
) -> Result<Value> {
    if is_fresh(input) && !open.is_some_and(Path::exists) {
        bail!(
            "{}: a fresh file is read blind only after its FRESH-OPEN (--open)",
            input.display()
        );
    }
    if !PLACEHOLDERS.contains(&placeholder) {
        bail!("{placeholder:?} is not a placeholder of 4.3");
    }
    let recs: Vec<Value> = read_lines(input)?;
    let mut out = Vec::with_capacity(recs.len());
    for mut r in recs {
        if !r["state"].is_string() {
            bail!("{}: a record without a state", input.display());
        }
        let opts: Vec<String> = serde_json::from_value(r["opts"].clone())
            .with_context(|| format!("{}: a record without opts", input.display()))?;
        let plain: Vec<String> = opts.iter().map(|o| plain_option(o).to_string()).collect();
        r["opts"] = json!(plain);
        r["state"] = json!(format!("User: {placeholder}"));
        out.push(r);
    }
    let name = |suffix: &str| -> PathBuf {
        let mut s = stem.as_os_str().to_owned();
        s.push(suffix);
        PathBuf::from(s)
    };
    let (plain_p, b0, b1) = (
        name(".blind.jsonl"),
        name(".blind.model.jsonl"),
        name(".blind-bare.model.jsonl"),
    );
    write_jsonl(&plain_p, &out)?;
    super::build::render_with(&plain_p, &b0, how)?;
    write_jsonl(&b1, &out)?;
    // the check: every score of B0 is 0.00 and the ranks are the positions
    let model: Vec<Value> = read_lines(&b0)?;
    for (r, x) in model.iter().zip(&out) {
        let opts: Vec<String> = serde_json::from_value(r["opts"].clone())?;
        let n = opts.len().saturating_sub(1);
        match how {
            Render::ScoresInOptions => {
                for (i, o) in opts[..n].iter().enumerate() {
                    if !o.starts_with(&format!("score 0.00 rank {}; ", i + 1)) {
                        bail!(
                            "{}: a blind option that does not read score 0.00 rank {}",
                            x["uid"],
                            i + 1
                        );
                    }
                }
            }
            Render::Line => {
                let st = r["state"].as_str().unwrap_or("");
                let line = st.split_once('\n').map_or("", |(_, l)| l);
                for (i, e) in line.trim_start_matches(LINE_PREFIX).split("; ").enumerate() {
                    if n > 0 && !e.ends_with(&format!(" score 0.00 rank {}", i + 1)) {
                        bail!(
                            "{}: a blind BM25 line with a score that is not 0.00",
                            x["uid"]
                        );
                    }
                }
            }
        }
    }
    let hashes = json!({
        plain_p.file_name().unwrap().to_string_lossy(): sha256_file(&plain_p)?,
        b0.file_name().unwrap().to_string_lossy(): sha256_file(&b0)?,
        b1.file_name().unwrap().to_string_lossy(): sha256_file(&b1)?,
    });
    Ok(
        json!({"about": "s6-route v6 (docs/s6-route6-prereg.md 4.3): the blind copies of a slate file",
              "input": input.display().to_string(), "slates": out.len(), "placeholder": placeholder,
              "render of B0": how.name(), "files": hashes}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slate(uid: &str, request: &str, gold: usize) -> Value {
        let tools = [
            "get_weather: Gets the weather of a city.",
            "send_mail: Sends a mail.",
            "lights_on: Turns the lights on.",
        ];
        let opts: Vec<String> = sky_infer::route::options_with_scores(request, &tools)
            .into_iter()
            .chain([sky_infer::route::NONE_OPTION.to_string()])
            .collect();
        let mut target = vec![0.0; 4];
        target[gold] = 1.0;
        json!({"src": "t", "t": "choice", "ins": sky_infer::route::INSTRUCTION, "opts": opts,
               "state": format!("User: {request}"), "target": target, "split": "val",
               "uid": uid, "kind": "test", "lang": "en", "request": uid,
               "tools": ["a", "b", "c"], "roles": ["right", "drawn", "drawn"]})
    }

    #[test]
    fn blind_copies_read_zero_scores_by_both_renderings() {
        let d = std::env::temp_dir().join(format!("sky-route-blind-{}", std::process::id()));
        let val = d.join("val");
        std::fs::create_dir_all(&val).unwrap();
        let f = val.join("v5_tools.jsonl");
        write_jsonl(
            &f,
            &[
                slate("u1", "what is the weather in Paris", 0),
                slate("u2", "send a mail to Bob", 3),
            ],
        )
        .unwrap();
        // the requests share tokens with the options: the scores are not all 0
        let src: Vec<Value> = read_lines(&f).unwrap();
        assert!(
            src[0]["opts"][0].as_str().unwrap().starts_with("score ")
                && !src[0]["opts"][0]
                    .as_str()
                    .unwrap()
                    .starts_with("score 0.00")
        );
        let rep = blind(
            &f,
            &val.join("v5_tools"),
            "?",
            Render::ScoresInOptions,
            None,
        )
        .unwrap();
        assert_eq!(rep["slates"], 2);
        let plain: Vec<Value> = read_lines(&val.join("v5_tools.blind.jsonl")).unwrap();
        let b0: Vec<Value> = read_lines(&val.join("v5_tools.blind.model.jsonl")).unwrap();
        let b1: Vec<Value> = read_lines(&val.join("v5_tools.blind-bare.model.jsonl")).unwrap();
        for ((p, m), (x, s)) in plain.iter().zip(&b0).zip(b1.iter().zip(&src)) {
            assert_eq!(p["state"], "User: ?");
            assert_eq!(
                (&p["uid"], &p["target"], &p["roles"]),
                (&s["uid"], &s["target"], &s["roles"])
            );
            let po: Vec<String> = serde_json::from_value(p["opts"].clone()).unwrap();
            let mo: Vec<String> = serde_json::from_value(m["opts"].clone()).unwrap();
            for (i, o) in mo[..3].iter().enumerate() {
                assert_eq!(o, &format!("score 0.00 rank {}; {}", i + 1, po[i]));
            }
            assert_eq!(mo[3], sky_infer::route::NONE_OPTION);
            assert_eq!(m["state"], "User: ?");
            // B1: no score, no rank, no line
            assert_eq!(x["opts"], p["opts"]);
            assert!(!x["state"].as_str().unwrap().contains('\n'));
            // `sky-route fuse` reads the blind plain file as it reads any slate file: the plain
            // options and the state's request, rendered by the model file's rendering. The
            // same steps here give B0's text, and the embedding part gets P.
            let opts: Vec<String> = po.iter().map(|o| plain_option(o).to_string()).collect();
            let request = request_of(p["state"].as_str().unwrap());
            assert_eq!(request, "?");
            let (fused_opts, state) = Render::ScoresInOptions.model_input(request, &opts);
            assert_eq!(fused_opts, mo);
            assert_eq!(state, "User: ?");
        }
        // v2's rendering: the state `User: ?` and a BM25 line of zeros, the options plain
        blind(&f, &d.join("v2_tools"), "?", Render::Line, None).unwrap();
        let l: Vec<Value> = read_lines(&d.join("v2_tools.blind.model.jsonl")).unwrap();
        assert_eq!(
            l[0]["state"],
            "User: ?\nBM25: get_weather score 0.00 rank 1; send_mail score 0.00 rank 2; lights_on score 0.00 rank 3"
        );
        assert!(!l[0]["opts"][0].as_str().unwrap().starts_with("score"));
        // a fresh file only after its FRESH-OPEN
        let fresh = d.join("fresh");
        std::fs::create_dir_all(&fresh).unwrap();
        let ff = fresh.join("t5_tools.jsonl");
        std::fs::copy(&f, &ff).unwrap();
        assert!(blind(
            &ff,
            &fresh.join("t5_tools"),
            "?",
            Render::ScoresInOptions,
            None
        )
        .is_err());
        let open = d.join("FRESH-OPEN");
        assert!(blind(
            &ff,
            &fresh.join("t5_tools"),
            "?",
            Render::ScoresInOptions,
            Some(&open)
        )
        .is_err());
        std::fs::write(&open, "").unwrap();
        blind(
            &ff,
            &fresh.join("t5_tools"),
            "?",
            Render::ScoresInOptions,
            Some(&open),
        )
        .unwrap();
        assert!(blind(&f, &val.join("x"), "unknown", Render::ScoresInOptions, None).is_err());
        std::fs::remove_dir_all(d).ok();
    }

    #[test]
    fn the_placeholder_and_its_fallbacks() {
        for p in PLACEHOLDERS {
            assert!(tokens(p).is_empty(), "{p}");
            assert!(sky_infer::route::bm25(p, &["get_x: Gets ? x - y \u{2026}"])
                .iter()
                .all(|&s| s == 0.0));
        }
        assert_eq!(request_of("User:  ?  \nBM25: a score 0.00 rank 1"), "?");
        let d = std::env::temp_dir().join(format!("sky-route-placeholder-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("train.jsonl");
        let out = d.join("placeholder.json");
        write_jsonl(&f, &[slate("u1", "turn on the lights", 2)]).unwrap();
        let r = check_placeholder(std::slice::from_ref(&f), &out, None).unwrap();
        assert_eq!(r["taken"], "?");
        assert_eq!(placeholder_of(&out).unwrap(), "?");
        // `?` is a request: the next one
        write_jsonl(&f, &[slate("u1", " ? ", 2), slate("u2", "lights", 2)]).unwrap();
        assert_eq!(
            check_placeholder(std::slice::from_ref(&f), &out, None).unwrap()["taken"],
            "\u{2026}"
        );
        write_jsonl(
            &f,
            &[
                slate("u1", "?", 2),
                slate("u2", "\u{2026}", 2),
                slate("u3", "-", 2),
            ],
        )
        .unwrap();
        assert!(
            check_placeholder(std::slice::from_ref(&f), &out, None).unwrap()["taken"].is_null()
        );
        // the counts amendment: the placeholder taken, checked on another file
        let keep = d.join("taken.json");
        write_json(&keep, &json!({"placeholder": "?"})).unwrap();
        let t5 = d.join("t5.jsonl");
        write_jsonl(&t5, &[slate("v1", "book a flight", 0)]).unwrap();
        let r = check_placeholder(
            std::slice::from_ref(&t5),
            &d.join("t5-check.json"),
            Some(&keep),
        )
        .unwrap();
        assert_eq!(r["occurs"], false);
        assert!(
            !placeholder_refused(&r),
            "the kept mode passes when it does not occur"
        );
        write_jsonl(&t5, &[slate("v1", "?", 0)]).unwrap();
        assert_eq!(
            check_placeholder(
                std::slice::from_ref(&t5),
                &d.join("t5-check.json"),
                Some(&keep)
            )
            .unwrap()["occurs"],
            true
        );
        write_json(&keep, &json!({"placeholder": "request"})).unwrap();
        assert!(placeholder_of(&keep).is_err());
        std::fs::remove_dir_all(d).ok();
    }
}
