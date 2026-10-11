//! `sky-games-gen --game {flappy,rescue,dodge,all} --n N --seed S [--out PATH]`
//!
//! Writes decision records as JSONL. With a single game, `--out` is the output file (default
//! `data/games/<game>.jsonl`); with `all`, it is the output directory (default `data/games`)
//! and `N` records are written per game. Label balance statistics go to stderr.

use sky_games::gen::generate_with;
use sky_games::GAMES;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const USAGE: &str =
    "usage: sky-games-gen --game {flappy,rescue,dodge,all} --n N --seed S [--out PATH]";

struct Args {
    game: String,
    n: usize,
    seed: u64,
    out: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        game: String::new(),
        n: 10_000,
        seed: 0,
        out: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        if flag == "-h" || flag == "--help" {
            return Err(USAGE.to_string());
        }
        let value = it
            .next()
            .ok_or_else(|| format!("missing value for {flag}\n{USAGE}"))?;
        match flag.as_str() {
            "--game" => args.game = value,
            "--n" => args.n = value.parse().map_err(|e| format!("--n: {e}"))?,
            "--seed" => args.seed = value.parse().map_err(|e| format!("--seed: {e}"))?,
            "--out" => args.out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag {flag}\n{USAGE}")),
        }
    }
    if args.game != "all" && !GAMES.contains(&args.game.as_str()) {
        return Err(format!(
            "--game must be one of flappy, rescue, dodge, all\n{USAGE}"
        ));
    }
    Ok(args)
}

fn write_game(game: &str, n: usize, seed: u64, path: &Path) -> Result<(), String> {
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let file = File::create(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut w = BufWriter::new(file);
    let mut io_err = None;
    let stats = generate_with(game, n, seed, |rec| {
        if io_err.is_none() {
            let line = serde_json::to_string(&rec).expect("record serializes");
            if let Err(e) = writeln!(w, "{line}") {
                io_err = Some(e);
            }
        }
    })?;
    if let Some(e) = io_err {
        return Err(format!("{}: {e}", path.display()));
    }
    w.flush().map_err(|e| format!("{}: {e}", path.display()))?;
    eprint!("{}", stats.report());
    eprintln!("  -> {}", path.display());
    Ok(())
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.game == "all" {
        let dir = args.out.unwrap_or_else(|| PathBuf::from("data/games"));
        for game in GAMES {
            write_game(game, args.n, args.seed, &dir.join(format!("{game}.jsonl")))?;
        }
    } else {
        let path = args
            .out
            .unwrap_or_else(|| PathBuf::from(format!("data/games/{}.jsonl", args.game)));
        write_game(&args.game, args.n, args.seed, &path)?;
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
