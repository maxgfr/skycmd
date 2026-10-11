//! `skycmd`: run System One decision models (GGUF) from the command line or over HTTP.

use std::io::Read;
use std::process::ExitCode;

use sky_serve::{LoadedModel, ServeConfig, Server, SAMPLE_REQUEST};

const USAGE: &str = "\
skycmd: run System One decision models (GGUF, s1 and s2)

Usage:
  skycmd serve  --model <file.gguf> [--host 127.0.0.1] [--port 8080] [--threads N] [--no-cors]
  skycmd decide --model <file.gguf> [--request <req.json> | -] [--pretty]
  skycmd info   <file.gguf>
  skycmd bench  <file.gguf> [--request <req.json>] [--iterations 500] [--warmup 50] [--json]
  skycmd --version | --help

serve   HTTP server: POST /v1/systemone, GET /health, GET /v1/models (same API as llama.cpp).
decide  Answer one request read from a file or from stdin and print the JSON response.
info    Print the model description (architecture, sizes, temperatures) as JSON.
bench   Time decisions on a request (default: a 3-question Flappy Drone frame).
";

/// Parsed command line: positional arguments plus `--key value` / `--flag` options.
struct Args {
    positional: Vec<String>,
    options: Vec<(String, Option<String>)>,
}

const FLAGS: &[&str] = &["pretty", "json", "no-cors", "help", "version"];

impl Args {
    fn parse(raw: impl Iterator<Item = String>) -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut options = Vec::new();
        let mut it = raw.peekable();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                if let Some((k, v)) = name.split_once('=') {
                    options.push((k.to_string(), Some(v.to_string())));
                } else if FLAGS.contains(&name) {
                    options.push((name.to_string(), None));
                } else {
                    let v = it.next().ok_or_else(|| format!("--{name} needs a value"))?;
                    options.push((name.to_string(), Some(v)));
                }
            } else if a == "-h" {
                options.push(("help".into(), None));
            } else if a == "-V" {
                options.push(("version".into(), None));
            } else if a == "-m" {
                let v = it.next().ok_or("-m needs a value")?;
                options.push(("model".into(), Some(v)));
            } else {
                positional.push(a);
            }
        }
        Ok(Args {
            positional,
            options,
        })
    }

    fn flag(&self, name: &str) -> bool {
        self.options.iter().any(|(k, _)| k == name)
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.options
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .and_then(|(_, v)| v.as_deref())
    }

    fn number(&self, name: &str, default: usize) -> Result<usize, String> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("--{name}: expected a number, got {v:?}")),
        }
    }

    /// `--model PATH`, or the first positional argument after the subcommand.
    fn model(&self) -> Result<&str, String> {
        self.get("model")
            .or_else(|| self.positional.get(1).map(String::as_str))
            .ok_or_else(|| "missing --model <file.gguf>".to_string())
    }

    fn check_known(&self, known: &[&str]) -> Result<(), String> {
        for (k, _) in &self.options {
            if !known.contains(&k.as_str()) {
                return Err(format!("unknown option --{k}"));
            }
        }
        Ok(())
    }
}

fn read_request(path: Option<&str>) -> Result<String, String> {
    match path {
        None | Some("-") => {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| format!("cannot read stdin: {e}"))?;
            Ok(s)
        }
        Some(p) => std::fs::read_to_string(p).map_err(|e| format!("cannot read {p:?}: {e}")),
    }
}

fn pretty(v: &serde_json::Value) -> String {
    serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())
}

fn run(args: &Args) -> Result<(), String> {
    let cmd = args.positional.first().map(String::as_str);
    match cmd {
        Some("serve") => {
            args.check_known(&["model", "host", "port", "threads", "no-cors"])?;
            let path = args.model()?;
            let (model, engine) = LoadedModel::open(path).map_err(|e| e.to_string())?;
            let defaults = ServeConfig::default();
            let host = args.get("host").unwrap_or("127.0.0.1");
            let port = args.number("port", 8080)?;
            let addr = if host.contains(':') && !host.starts_with('[') {
                format!("[{host}]:{port}")
            } else {
                format!("{host}:{port}")
            };
            let cfg = ServeConfig {
                addr,
                threads: args.number("threads", defaults.threads)?.max(1),
                cors: !args.flag("no-cors"),
            };
            let name = model.name.clone();
            let arch = model.architecture.clone();
            let server = Server::start(model, engine, &cfg).map_err(|e| e.to_string())?;
            eprintln!(
                "skycmd {}: serving {name} ({arch}) on http://{} with {} worker(s)\n  \
                 POST /v1/systemone  GET /health  GET /v1/models",
                env!("CARGO_PKG_VERSION"),
                server.addr(),
                cfg.threads
            );
            server.join();
            Ok(())
        }
        Some("decide") => {
            args.check_known(&["model", "request", "pretty"])?;
            let path = args.model()?;
            // `decide --model m.gguf req.json` or `decide m.gguf req.json`
            let pos = if args.get("model").is_some() { 1 } else { 2 };
            let request = read_request(
                args.get("request")
                    .or_else(|| args.positional.get(pos).map(String::as_str)),
            )?;
            let (_, mut engine) = LoadedModel::open(path).map_err(|e| e.to_string())?;
            let out = sky_serve::decide_json(&mut engine, &request).map_err(|e| e.to_string())?;
            if args.flag("pretty") {
                let v: serde_json::Value = serde_json::from_str(&out).map_err(|e| e.to_string())?;
                println!("{}", pretty(&v));
            } else {
                println!("{out}");
            }
            Ok(())
        }
        Some("info") => {
            args.check_known(&["model"])?;
            let path = args.model()?;
            let (model, mut engine) = LoadedModel::open(path).map_err(|e| e.to_string())?;
            println!("{}", pretty(&model.info(&mut engine)));
            Ok(())
        }
        Some("bench") => {
            args.check_known(&["model", "request", "iterations", "warmup", "json"])?;
            let path = args.model()?;
            let request = match args.get("request") {
                Some(p) => read_request(Some(p))?,
                None => SAMPLE_REQUEST.to_string(),
            };
            let (model, mut engine) = LoadedModel::open(path).map_err(|e| e.to_string())?;
            let iterations = args.number("iterations", 500)?;
            let warmup = args.number("warmup", 50)?;
            let r = sky_serve::bench(&mut engine, &request, warmup, iterations)
                .map_err(|e| e.to_string())?;
            if args.flag("json") {
                let mut v = r.to_json();
                v["model"] = model.name.clone().into();
                v["architecture"] = model.architecture.clone().into();
                println!("{}", pretty(&v));
            } else {
                println!(
                    "{} ({}, {} params): {} iterations, {} questions and {} tokens per request\n  \
                     mean {:.1} us  p50 {:.1} us  p90 {:.1} us  p99 {:.1} us  min {:.1} us\n  \
                     {:.0} requests/s, {:.0} decisions/s (one thread)",
                    model.name,
                    model.architecture,
                    engine.n_params(),
                    r.iterations,
                    r.questions,
                    r.input_tokens,
                    r.mean_us,
                    r.p50_us,
                    r.p90_us,
                    r.p99_us,
                    r.min_us,
                    r.requests_per_s(),
                    r.requests_per_s() * r.questions as f64
                );
            }
            Ok(())
        }
        Some(other) => Err(format!("unknown command {other:?}\n\n{USAGE}")),
        None => Err(USAGE.to_string()),
    }
}

fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("skycmd: {e}");
            return ExitCode::from(2);
        }
    };
    if args.flag("version") {
        println!("skycmd {}", env!("CARGO_PKG_VERSION"));
        return ExitCode::SUCCESS;
    }
    if args.flag("help") || args.positional.first().map(String::as_str) == Some("help") {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("skycmd: {e}");
            ExitCode::FAILURE
        }
    }
}
