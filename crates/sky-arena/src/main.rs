//! Plays every game with a decision model, natively, and prints game metrics and latency.
//!
//! ```text
//! sky-arena <model.gguf | laya_dir> [--episodes N] [--seed S] [--game flappy|rescue|dodge|doom|all] [--json out.json]
//!           [--exit-threshold T]   # s2: override the early-exit threshold (1.0 = full depth)
//! sky-arena <model> --dagger N --out data/dagger   # N oracle-labelled records per game, from the model's own play
//! ```
//!
//! Works for s1 (`modern-bert`) and s2 (`skycmd-s2`) GGUF files; for s2 the report adds the mean
//! number of state-encoder blocks used per question and how often each exit answered.
use std::collections::HashMap;
use std::time::Instant;

use sky_games::rollout::{evaluate_with, EvalOptions, Metrics, OraclePolicy, Policy};
use sky_games::{make_game, GAMES};
use sky_infer::{Engine, EngineOptions, QType};

/// A [`Policy`] backed by the System One engine: it sees only the text state, like the browser.
struct EnginePolicy {
    engine: Engine,
    questions: HashMap<(String, String), (QType, &'static str)>,
    micros: Vec<f64>,
    /// s2: state-encoder blocks used by each question.
    layers: Vec<usize>,
    /// s2: answers per exit.
    exits: Vec<usize>,
    /// Number of decisions still to print to stderr (`SKY_TRACE=N`).
    trace: usize,
}

impl EnginePolicy {
    fn new(engine: Engine) -> Self {
        let mut questions = HashMap::new();
        for game in GAMES {
            let g = make_game(game, 0).expect("known game");
            for q in g.questions() {
                let qtype = match q.qtype {
                    sky_schema::QType::Choice => QType::Choice,
                    sky_schema::QType::Score => QType::Score,
                    sky_schema::QType::Noul => QType::Noul,
                };
                questions.insert(
                    (game.to_string(), q.id.to_string()),
                    (qtype, q.instructions()),
                );
            }
        }
        let trace = std::env::var("SKY_TRACE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        EnginePolicy {
            engine,
            questions,
            micros: Vec::new(),
            layers: Vec::new(),
            exits: Vec::new(),
            trace,
        }
    }
}

impl Policy for EnginePolicy {
    fn decide(
        &mut self,
        game: &str,
        question_id: &str,
        state_text: &str,
        opts: &[String],
    ) -> Vec<f32> {
        let (qtype, ins) = self.questions[&(game.to_string(), question_id.to_string())];
        let t0 = Instant::now();
        let p = self
            .engine
            .decide_raw(qtype, ins, opts, state_text)
            .expect("decision");
        self.micros.push(t0.elapsed().as_secs_f64() * 1e6);
        let stats = self.engine.last_stats().cloned().unwrap_or_default();
        if let (Some(e), Some(l)) = (stats.exit, stats.layers_used) {
            if self.exits.len() < stats.n_exits {
                self.exits.resize(stats.n_exits, 0);
            }
            self.exits[e] += 1;
            self.layers.push(l);
        }
        if self.trace > 0 {
            self.trace -= 1;
            match stats.layers_used {
                Some(l) => eprintln!("{game}/{question_id} [{state_text}] -> {p:.2?} ({l} layers)"),
                None => eprintln!("{game}/{question_id} [{state_text}] -> {p:.2?}"),
            }
        }
        p
    }
}

/// Collects oracle-labelled records from states the model visits itself (DAgger).
fn dagger(
    policy: &mut EnginePolicy,
    games: &[&str],
    n: usize,
    seed: u64,
    out: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::Write;
    std::fs::create_dir_all(out)?;
    for &game in games {
        let probe = make_game(game, 0).expect("known game");
        let questions = probe.questions();
        let rendered: Vec<Vec<String>> = questions.iter().map(|q| q.canonical_options()).collect();
        let path = format!("{out}/{game}.jsonl");
        let mut w = std::io::BufWriter::new(std::fs::File::create(&path)?);
        let mut actor = |g: &dyn sky_games::DecisionGame, _rng: &mut sky_games::gen::SmallRng| {
            let text = g.state_text();
            let p = policy.decide(game, questions[0].id, &text, &rendered[0]);
            let abort = game == "dodge" && {
                let pa = policy.decide(game, questions[1].id, &text, &rendered[1]);
                pa[1] > pa[0]
            };
            (argmax(&p), abort)
        };
        let mut lines = Vec::new();
        let stats =
            sky_games::gen::generate_with_actor(game, n, seed ^ 0xda66e7, Some(&mut actor), |r| {
                lines.push(r)
            })?;
        for r in &lines {
            serde_json::to_writer(&mut w, r)?;
            w.write_all(b"\n")?;
        }
        eprintln!("{}\n-> {path}", stats.report());
    }
    Ok(())
}

fn argmax(p: &[f32]) -> usize {
    p.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[((sorted.len() - 1) as f64 * q).round() as usize]
}

fn metrics_json(m: &Metrics) -> serde_json::Value {
    serde_json::json!({
        "episodes": m.episodes, "mean_score": m.mean_score, "rescue_rate": m.rescue_rate, "kill_rate": m.kill_rate,
        "completion_rate": m.completion_rate, "crashes": m.crashes, "vetoes": m.vetoes,
        "aborts": m.aborts, "safe_aborts": m.safe_aborts, "mean_steps": m.mean_steps,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let Some(model) = args.first().filter(|a| !a.starts_with("--")) else {
        eprintln!("usage: sky-arena <model.gguf | laya_dir> [--episodes N] [--seed S] [--game G|all|a,b] [--json out.json] [--max-pipes N] [--exit-threshold T]");
        std::process::exit(2);
    };
    let episodes: usize = flag("--episodes").map_or(Ok(100), |v| v.parse())?;
    let seed: u64 = flag("--seed").map_or(Ok(7), |v| v.parse())?;
    let max_pipes: u32 = flag("--max-pipes").map_or(Ok(200), |v| v.parse())?;
    let game_arg = flag("--game").unwrap_or_else(|| "all".into());
    let games: Vec<&str> = GAMES
        .iter()
        .copied()
        .filter(|g| game_arg == "all" || game_arg.split(',').any(|a| a == *g))
        .collect();

    let exit_threshold: Option<f32> = flag("--exit-threshold").map(|v| v.parse()).transpose()?;
    let opts = EngineOptions {
        exit_threshold,
        ..Default::default()
    };
    let engine = if model.ends_with(".gguf") {
        Engine::from_gguf_with(&std::fs::read(model)?, opts)?
    } else {
        Engine::from_safetensors_dir_with(model, opts)?
    };
    let mut policy = EnginePolicy::new(engine);
    if let Some(n) = flag("--dagger") {
        let out = flag("--out").unwrap_or_else(|| "data/dagger".into());
        return dagger(&mut policy, &games, n.parse()?, seed, &out);
    }
    let opts = EvalOptions {
        flappy_max_pipes: max_pipes,
        veto: true,
    };
    let mut report = serde_json::Map::new();
    report.insert("model".into(), model.clone().into());
    report.insert("arch".into(), policy.engine.arch().name().into());
    if let Some(t) = exit_threshold {
        report.insert("exit_threshold".into(), t.into());
    }
    for game in games {
        policy.micros.clear();
        policy.layers.clear();
        policy.exits.clear();
        let m = evaluate_with(game, &mut policy, episodes, seed, opts);
        let oracle = evaluate_with(game, &mut OraclePolicy::default(), episodes, seed, opts);
        let mut lat = policy.micros.clone();
        lat.sort_by(|a, b| a.total_cmp(b));
        let (p50, p95) = (percentile(&lat, 0.5), percentile(&lat, 0.95));
        let mean_layers = (!policy.layers.is_empty())
            .then(|| policy.layers.iter().sum::<usize>() as f64 / policy.layers.len() as f64);
        let depth = mean_layers.map_or(String::new(), |l| {
            format!(", mean layers {l:.2}, exits {:?}", policy.exits)
        });
        let extra = if game == "doom" {
            format!(
                " kills {:.2} (oracle {:.2}) completion oracle {:.2} mean ticks {:.0} (oracle {:.0})",
                m.kill_rate, oracle.kill_rate, oracle.completion_rate, m.mean_steps, oracle.mean_steps
            )
        } else {
            String::new()
        };
        println!(
            "{game:7} model: score {:7.2} rescue {:.2} completion {:.2} crashes {:4} vetoes {:4} | oracle score {:7.2} | {} decisions, p50 {:.0} µs, p95 {:.0} µs{depth}{extra}",
            m.mean_score, m.rescue_rate, m.completion_rate, m.crashes, m.vetoes, oracle.mean_score, lat.len(), p50, p95
        );
        report.insert(
            game.into(),
            serde_json::json!({
                "model": metrics_json(&m), "oracle": metrics_json(&oracle),
                "decisions": lat.len(), "p50_us": p50, "p95_us": p95,
                "mean_layers_used": mean_layers, "exit_counts": policy.exits,
            }),
        );
    }
    if let Some(path) = flag("--json") {
        std::fs::write(path, serde_json::to_string_pretty(&report)?)?;
    }
    Ok(())
}
