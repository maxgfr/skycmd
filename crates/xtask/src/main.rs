//! `cargo xtask s5 <stage>`: runs the s5 pipeline stages against the persistent store.
//!
//! - **Store.** Every artefact goes to `$SKYCMD_STORE` (`data/s5/`, `runs/s5-*/`, `models/s5-*/`,
//!   `models-src/`), never to the checkout the command runs from. The store must be the main
//!   checkout of this repository (the parent of the git common directory): a worktree or any
//!   other path is refused, so that work done in a worktree survives its removal.
//! - **GPU lock.** Stages that train or score on the GPU hold an exclusive lock on
//!   `runs/s5-gpu.lock`; a second one waits.
//! - **Logs.** Each run tees its output to `runs/s5-logs/<stage>-<unix time>.log`.
//! - **Resume.** A finished step leaves `runs/s5-state/<stage>/<step>.done` holding its command
//!   line; a rerun skips the steps whose command line is unchanged (`--force` reruns them), from
//!   any worktree. Steps marked `always` (the results summary) run every time.
//! - **Experiments.** The stages of [`exp`] are built against the store (their runs depend on
//!   earlier validation results); `--jobs N` runs up to N of their independent steps at once
//!   under one GPU lock (a failed step does not stop the others), `--runs a,b` names the runs
//!   of `eval`.
//!
//! The binaries are built in release mode from the checkout `cargo xtask` runs in, and run with
//! the store as their working directory.

mod exp;
mod regress;
mod stages;

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};

use stages::{stages, Stage, Step};

/// Developer directory of the full Xcode (the Command Line Tools have no `metal` compiler).
const XCODE: &str = "/Applications/Xcode.app/Contents/Developer";

fn usage() -> ! {
    eprintln!(
        "usage:\n  cargo xtask s5 list\n  cargo xtask s5 <stage> [--force] [--dry-run] [--jobs N] [--runs a,b]\n  cargo xtask store\n  \
         cargo xtask regress [--update]\n\n\
         SKYCMD_STORE must point at the main checkout of this repository."
    );
    std::process::exit(2)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("store") => {
            println!("{}", store()?.display());
            Ok(())
        }
        Some("regress") => {
            let update = match args.get(1).map(String::as_str) {
                None => false,
                Some("--update") => true,
                Some(bad) => bail!("unknown argument {bad:?}"),
            };
            regress::regress(&checkout(), &store()?, update)
        }
        Some("s5") => {
            let Some(name) = args.get(1) else { usage() };
            if name == "list" {
                for s in stages() {
                    println!(
                        "{:<16} {}{}",
                        s.name,
                        s.about,
                        if s.steps.iter().any(|t| t.gpu) {
                            " [GPU]"
                        } else {
                            ""
                        }
                    );
                }
                for (n, about) in exp::STAGES {
                    println!("{n:<16} {about}");
                }
                return Ok(());
            }
            let (mut force, mut dry, mut jobs, mut runs) = (false, false, 1usize, None);
            let mut it = args[2..].iter();
            while let Some(a) = it.next() {
                match a.as_str() {
                    "--force" => force = true,
                    "--dry-run" => dry = true,
                    "--jobs" => {
                        jobs = it
                            .next()
                            .and_then(|v| v.parse().ok())
                            .filter(|&n| n >= 1)
                            .context("--jobs N (N >= 1)")?
                    }
                    "--runs" => runs = Some(it.next().context("--runs a,b")?.clone()),
                    bad => bail!("unknown argument {bad:?}"),
                }
            }
            let all = stages();
            if let Some(stage) = all.iter().find(|s| s.name == *name) {
                return run_stage(stage, force, dry, jobs);
            }
            match exp::stage(name, &store()?, runs.as_deref())? {
                Some(stage) => run_stage(&stage, force, dry, jobs),
                None => bail!("unknown stage {name:?} (cargo xtask s5 list)"),
            }
        }
        _ => usage(),
    }
}

/// The checkout this command runs from (the workspace root).
fn checkout() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("workspace root")
}

fn git(dir: &Path, args: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .context("running git")?;
    if !out.status.success() {
        bail!(
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok(String::from_utf8(out.stdout)?.trim().to_string())
}

/// `$SKYCMD_STORE`, checked to be the main checkout of this repository.
fn store() -> Result<PathBuf> {
    let raw = std::env::var("SKYCMD_STORE")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .context(
            "SKYCMD_STORE is not set (it must point at the main checkout of this repository)",
        )?;
    let store = PathBuf::from(raw.trim())
        .canonicalize()
        .with_context(|| format!("SKYCMD_STORE={raw:?} does not exist"))?;
    let common = git(
        &checkout(),
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )?;
    let main = Path::new(&common)
        .parent()
        .context("git common dir has no parent")?
        .canonicalize()?;
    check_store(&store, &main)?;
    Ok(store)
}

fn check_store(store: &Path, main: &Path) -> Result<()> {
    if store != main {
        bail!(
            "SKYCMD_STORE={} is refused: the store must be the main checkout {}",
            store.display(),
            main.display()
        );
    }
    if store.components().any(|c| c.as_os_str() == "worktrees") {
        bail!("SKYCMD_STORE={} is inside a worktree", store.display());
    }
    Ok(())
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// A writer shared by the terminal copy and the log file.
#[derive(Clone)]
struct Log(Arc<Mutex<File>>);

impl Log {
    fn line(&self, s: &str) {
        println!("{s}");
        if let Ok(mut f) = self.0.lock() {
            let _ = writeln!(f, "{s}");
        }
    }
}

fn run_stage(stage: &Stage, force: bool, dry: bool, jobs: usize) -> Result<()> {
    let store = store()?;
    let root = checkout();
    let bin = root.join("target/release");
    let state = store.join("runs/s5-state").join(stage.name);
    let logs = store.join("runs/s5-logs");
    fs::create_dir_all(&state)?;
    fs::create_dir_all(&logs)?;
    let log_path = logs.join(format!("{}-{}.log", stage.name, now()));
    let log = Log(Arc::new(Mutex::new(File::create(&log_path)?)));
    let head = git(&root, &["rev-parse", "HEAD"]).unwrap_or_default();
    let dirty = !git(&root, &["status", "--porcelain"])
        .unwrap_or_default()
        .is_empty();
    log.line(&format!(
        "[xtask] stage {} | store {} | checkout {} @ {head}{} | log {}",
        stage.name,
        store.display(),
        root.display(),
        if dirty { " (dirty)" } else { "" },
        log_path.display()
    ));

    // Build the binaries the stage needs, once.
    let mut packages: Vec<&str> = stage.steps.iter().filter_map(|s| s.package()).collect();
    packages.sort_unstable();
    packages.dedup();
    if !packages.is_empty() && !dry {
        let mut cmd = Command::new(env!("CARGO"));
        cmd.current_dir(&root).args(["build", "--release"]);
        // mlx (sky-train) compiles Metal kernels with the full Xcode toolchain.
        if std::env::var_os("DEVELOPER_DIR").is_none() && Path::new(XCODE).exists() {
            cmd.env("DEVELOPER_DIR", XCODE);
        }
        for p in &packages {
            cmd.args(["-p", p]);
        }
        log.line(&format!(
            "[xtask] cargo build --release -p {}",
            packages.join(" -p ")
        ));
        run(&mut cmd, &log, "").context("building the stage binaries")?;
    }

    // Experiment stages are rebuilt with other step lists: their resume key is the step name.
    let dynamic = exp::STAGES.iter().any(|(n, _)| *n == stage.name);
    let mut pending = Vec::new();
    for (i, step) in stage.steps.iter().enumerate() {
        let argv = step.argv(&store, &bin, &root);
        // The resume key names our programs without the checkout path, so that a step done
        // from one worktree is not redone from another.
        let line = std::iter::once(step.prog)
            .chain(argv[1..].iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ");
        let done = if dynamic {
            state.join(format!("{}.done", step.name))
        } else {
            state.join(format!("{i:02}-{}.done", step.name))
        };
        let key = resume_key(&line, &store);
        let unchanged =
            fs::read_to_string(&done).is_ok_and(|s| resume_key(s.trim(), &store) == key);
        if !force && !step.always && unchanged {
            log.line(&format!("[xtask] skip {} (done)", step.name));
            continue;
        }
        if dry {
            log.line(&format!("[xtask] would run {}: {line}", step.name));
            continue;
        }
        pending.push((step, argv, line, done));
    }
    let mut _gpu: Option<File> = None;
    if pending.iter().any(|p| p.0.gpu) {
        _gpu = Some(gpu_lock(&store, &log)?);
    }
    // Up to `jobs` steps at once (in order). With one job the steps form a chain and a failure
    // stops the stage; with several they are independent runs and the others go on.
    let failed: Mutex<Vec<String>> = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let mut running = 0usize;
        for (step, argv, line, done) in pending {
            if running == jobs {
                let _ = rx.recv();
                running -= 1;
            }
            if jobs == 1 && !failed.lock().map(|f| f.is_empty()).unwrap_or(false) {
                break;
            }
            running += 1;
            let (tx, log, store, failed) = (tx.clone(), log.clone(), &store, &failed);
            let prefix = if jobs > 1 {
                format!("[{}] ", step.name)
            } else {
                String::new()
            };
            scope.spawn(move || {
                log.line(&format!("[xtask] run {}: {line}", step.name));
                let t0 = std::time::Instant::now();
                let mut cmd = Command::new(&argv[0]);
                cmd.args(&argv[1..]).current_dir(store);
                for (k, v) in &step.env {
                    cmd.env(k, v);
                }
                let res = run(&mut cmd, &log, &prefix)
                    .and_then(|()| Ok(fs::write(&done, format!("{line}\n"))?));
                match res {
                    Ok(()) => log.line(&format!(
                        "[xtask] ok {} in {:.1}s",
                        step.name,
                        t0.elapsed().as_secs_f64()
                    )),
                    Err(e) => {
                        log.line(&format!("[xtask] FAILED {}: {e:#}", step.name));
                        if let Ok(mut f) = failed.lock() {
                            f.push(step.name.clone());
                        }
                    }
                }
                let _ = tx.send(());
            });
        }
    });
    let failed = failed.into_inner().unwrap_or_default();
    if !failed.is_empty() {
        bail!("stage {}: failed steps {failed:?}", stage.name);
    }
    log.line(&format!("[xtask] stage {} complete", stage.name));
    Ok(())
}

/// A step's command line with every worktree checkout of the store (`<store>/.claude/worktrees/
/// <name>`) written as the store itself: `{checkout}` arguments (the pre-registration, the demo
/// cases) then compare equal from any worktree, and a step done from one is not redone from
/// another.
fn resume_key(line: &str, store: &Path) -> String {
    let store = store.display().to_string();
    let wt = format!("{store}/.claude/worktrees/");
    let mut out = String::with_capacity(line.len());
    let mut rest = line;
    while let Some(i) = rest.find(&wt) {
        out.push_str(&rest[..i]);
        out.push_str(&store);
        let after = &rest[i + wt.len()..];
        rest = &after[after.find(['/', ' ']).unwrap_or(after.len())..];
    }
    out.push_str(rest);
    out
}

/// Exclusive lock on `runs/s5-gpu.lock`, held until the returned file is dropped.
fn gpu_lock(store: &Path, log: &Log) -> Result<File> {
    let path = store.join("runs/s5-gpu.lock");
    let f = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&path)?;
    if f.try_lock().is_err() {
        log.line(&format!(
            "[xtask] waiting for the GPU lock ({})",
            path.display()
        ));
        f.lock()?;
    }
    f.set_len(0)?;
    writeln!(&f, "pid {} since {}", std::process::id(), now())?;
    Ok(f)
}

/// Runs a command, copying its stdout and stderr to the terminal and the log line by line
/// (each line after `prefix`).
fn run(cmd: &mut Command, log: &Log, prefix: &str) -> Result<()> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("spawning {cmd:?}"))?;
    let pipes: Vec<Box<dyn std::io::Read + Send>> = vec![
        Box::new(child.stdout.take().expect("stdout")),
        Box::new(child.stderr.take().expect("stderr")),
    ];
    let threads: Vec<_> = pipes
        .into_iter()
        .map(|p| {
            let (log, prefix) = (log.clone(), prefix.to_string());
            std::thread::spawn(move || {
                for line in BufReader::new(p).lines().map_while(Result::ok) {
                    log.line(&format!("{prefix}{line}"));
                }
            })
        })
        .collect();
    let status = child.wait()?;
    for t in threads {
        let _ = t.join();
    }
    if !status.success() {
        bail!("{cmd:?} exited with {status}");
    }
    Ok(())
}

impl Step {
    /// The workspace package that builds this step's program, if it is one of ours.
    fn package(&self) -> Option<&'static str> {
        Some(match self.prog {
            "sky-data" => "sky-data",
            "sky-train" => "sky-train",
            "sky-eval" | "sky-convert" | "sky-bench" | "sky-compare" | "sky-s5-report" => {
                "sky-infer"
            }
            "skycmd" => "sky-serve",
            "sky-arena" => "sky-arena",
            _ => return None,
        })
    }

    /// The command line, with `{store}` replaced by the store path, `{checkout}` by the
    /// checkout (read-only inputs committed in the repository, e.g. `eval/s5/`) and our
    /// programs resolved to `target/release`.
    fn argv(&self, store: &Path, bin: &Path, checkout: &Path) -> Vec<String> {
        let prog = if self.package().is_some() {
            bin.join(self.prog).display().to_string()
        } else {
            self.prog.to_string()
        };
        std::iter::once(prog)
            .chain(self.args.iter().map(|a| {
                a.replace("{store}", &store.display().to_string())
                    .replace("{checkout}", &checkout.display().to_string())
            }))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_must_be_the_main_checkout() {
        let main = Path::new("/r/skycmd");
        assert!(check_store(main, main).is_ok());
        assert!(check_store(Path::new("/r/skycmd/.claude/worktrees/x"), main).is_err());
        assert!(check_store(Path::new("/tmp/store"), main).is_err());
        let wt = Path::new("/r/worktrees/skycmd");
        assert!(check_store(wt, wt).is_err());
    }

    #[test]
    fn stages_are_well_formed() {
        let all = stages();
        let mut names: Vec<_> = all.iter().map(|s| s.name).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), all.len(), "duplicate stage names");
        for s in &all {
            assert!(!s.steps.is_empty(), "{} has no step", s.name);
            let mut steps: Vec<_> = s.steps.iter().map(|t| t.name.as_str()).collect();
            steps.sort_unstable();
            steps.dedup();
            assert_eq!(
                steps.len(),
                s.steps.len(),
                "{}: duplicate step names",
                s.name
            );
            for t in &s.steps {
                // Outputs are written under the store only: relative paths resolve there
                // (working directory), absolute ones must start with {store}.
                for a in &t.args {
                    assert!(
                        !a.starts_with('/')
                            || a.starts_with("{store}")
                            || a.starts_with("{checkout}"),
                        "{}/{}: absolute path outside the store: {a}",
                        s.name,
                        t.name
                    );
                }
            }
        }
    }

    #[test]
    fn resume_key_ignores_the_worktree() {
        let st = Path::new("/r/skycmd");
        let line = |co: &str| {
            format!(
                "sky-eval m.gguf --prereg {co}/docs/s5-prereg.md --records {co}/eval/s5/d.jsonl"
            )
        };
        let main = resume_key(&line("/r/skycmd"), st);
        assert_eq!(main, line("/r/skycmd"));
        assert_eq!(
            resume_key(&line("/r/skycmd/.claude/worktrees/issue-8-x"), st),
            main
        );
        assert_eq!(
            resume_key(&line("/r/skycmd/.claude/worktrees/issue-10"), st),
            main
        );
        // other arguments are compared as they are
        assert_ne!(resume_key(&line("/elsewhere"), st), main);
        assert_eq!(
            resume_key("x --store /r/skycmd/.claude/worktrees/w", st),
            "x --store /r/skycmd"
        );
    }

    #[test]
    fn argv_substitutes_the_store() {
        let s = Step::new("x", "sky-data", &["--out", "{store}/data/s5/x"]);
        let v = s.argv(
            Path::new("/st"),
            Path::new("/co/target/release"),
            Path::new("/co"),
        );
        assert_eq!(v, ["/co/target/release/sky-data", "--out", "/st/data/s5/x"]);
    }
}
