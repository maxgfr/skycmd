//! `sky-data fetch-model`: a whole Hub repository (model or dataset), or the files matching
//! some patterns, pinned to a commit and checked against the hashes the Hub publishes.
//! The pinned commit and the hash of every file go to `<out>/.skycmd-fetch.json`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Result};
use serde_json::json;

use crate::hf::{Hf, RepoFile, RepoKind};

/// Name of the manifest written next to the downloaded files.
pub const MANIFEST: &str = ".skycmd-fetch.json";

pub struct FetchOptions {
    pub kind: RepoKind,
    pub repo: String,
    pub rev: String,
    pub out: PathBuf,
    /// Glob patterns (`*` matches any run of characters, `/` included); empty keeps every file.
    pub include: Vec<String>,
    pub max_bytes: u64,
}

pub struct FetchReport {
    pub commit: String,
    pub files: usize,
    pub downloaded: usize,
    pub bytes: u64,
}

/// `*`-only glob match.
pub fn glob_match(pattern: &str, s: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == s;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !s.starts_with(first) || s.len() < first.len() + last.len() || !s.ends_with(last) {
        return false;
    }
    let mut rest = &s[first.len()..s.len() - last.len()];
    for p in &parts[1..parts.len() - 1] {
        match rest.find(p) {
            Some(i) => rest = &rest[i + p.len()..],
            None => return false,
        }
    }
    true
}

fn selected<'a>(files: &'a [RepoFile], include: &[String]) -> Vec<&'a RepoFile> {
    files
        .iter()
        .filter(|f| include.is_empty() || include.iter().any(|p| glob_match(p, &f.path)))
        .collect()
}

pub fn fetch_repo(hf: &Hf, o: &FetchOptions) -> Result<FetchReport> {
    let commit = hf.repo_commit(o.kind, &o.repo, &o.rev)?;
    let all = hf.repo_files(o.kind, &o.repo, &commit)?;
    let files = selected(&all, &o.include);
    if files.is_empty() {
        bail!("{}@{commit}: no file matches {:?}", o.repo, o.include);
    }
    let bytes: u64 = files.iter().map(|f| f.size).sum();
    if bytes > o.max_bytes {
        bail!(
            "{}: {:.2} GB selected, over the {:.2} GB cap (--max-gb)",
            o.repo,
            bytes as f64 / 1e9,
            o.max_bytes as f64 / 1e9
        );
    }
    let mut downloaded = 0;
    for f in &files {
        let dest = safe_join(&o.out, &f.path)?;
        if hf.fetch_verified(o.kind, &o.repo, &commit, f, &dest)? {
            downloaded += 1;
            eprintln!("  {} ({:.1} MB) ok", f.path, f.size as f64 / 1e6);
        }
    }
    let manifest = json!({
        "repo": o.repo,
        "kind": match o.kind {
            RepoKind::Model => "model",
            RepoKind::Dataset => "dataset",
            RepoKind::GitHub => "github",
        },
        "rev": o.rev,
        "commit": commit,
        "include": o.include,
        "files": files.iter().map(|f| json!({
            "path": f.path,
            "size": f.size,
            "sha256": f.sha256,
            "git_sha1": f.git_sha1,
        })).collect::<Vec<_>>(),
    });
    fs::create_dir_all(&o.out)?;
    let mut text = serde_json::to_string_pretty(&manifest)?;
    text.push('\n');
    fs::write(o.out.join(MANIFEST), text)?;
    Ok(FetchReport {
        commit,
        files: files.len(),
        downloaded,
        bytes,
    })
}

/// `out/rel`, refusing absolute paths and `..` components from the listing.
fn safe_join(out: &Path, rel: &str) -> Result<PathBuf> {
    let p = Path::new(rel);
    if p.is_absolute()
        || p.components()
            .any(|c| !matches!(c, std::path::Component::Normal(_)))
    {
        bail!("refusing repository path {rel:?}");
    }
    Ok(out.join(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn globs() {
        assert!(glob_match("encoder/*", "encoder/config.json"));
        assert!(glob_match("*.json", "tokenizer/tokenizer.json"));
        assert!(glob_match(
            "typed-decisions/*",
            "typed-decisions/model.safetensors"
        ));
        assert!(!glob_match(
            "typed-decisions/*",
            "multilingual/model.safetensors"
        ));
        assert!(glob_match("a*b*c", "aXXbYYc"));
        assert!(!glob_match("a*b*c", "aXXcYYb"));
        assert!(glob_match("README.md", "README.md"));
        assert!(!glob_match("ab*ba", "aba"));
    }

    #[test]
    fn unsafe_paths_are_refused() {
        assert!(safe_join(Path::new("/x"), "../etc/passwd").is_err());
        assert!(safe_join(Path::new("/x"), "/etc/passwd").is_err());
        assert_eq!(
            safe_join(Path::new("/x"), "a/b.json").unwrap(),
            PathBuf::from("/x/a/b.json")
        );
    }
}
