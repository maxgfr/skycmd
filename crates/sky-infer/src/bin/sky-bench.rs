//! Latency benchmark: median / p95 per decision for a model file (s1 or s2).
//!
//! ```text
//! sky-bench <model.gguf | laya_dir | --random NAME> [options]
//!   --random NAME          random weights for a ladder configuration (s1-pico, s1-nano, s1-micro,
//!                          s2-pico, s2-nano, s2-micro) with a small test tokenizer
//!   --format f32|f16|ternary   storage of a random s2 model (default f32)
//!   --q8 | --q8-fast       s1: Q8_0 weights (two-pass / single-pass Q8_0 activations)
//!   --no-calib             s1: no outlier calibration of the Q8_0 matrices
//!   --ternary-compute auto|packed|f32   s2: kernels of the ternary matrices
//!   --exit-threshold T     s2: override τ (1.0 = full depth)
//!   --no-static-cache      s2: encode the static stream every decision (no LRU)
//!   --questions 1|2        questions per frame on the same state (2 = flappy action + danger)
//!   --iters N  --state TEXT  --options a,b,c  --json
//! sky-bench --kernels      ternary add/sub vs f32 products on the s2 ladder shapes
//! ```
//!
//! The question and options stay the same while the state changes every iteration, as in a game
//! loop (s1: the prefix tokenization is cached; s2: the static encoding is cached).

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::time::Instant;

    use sky_infer::kernels::ActQuant;
    use sky_infer::s2::random::WeightFormat;
    use sky_infer::{Arch, Engine, EngineOptions, QType, QuantizeSpec, TernaryCompute};

    const DANGER: &str = "Will the drone crash within half a second if it keeps gliding?";
    const DANGER_OPTS: [&str; 2] = [
        "false: no, the statement does not hold",
        "true: yes, the statement holds",
    ];

    fn usage() -> ! {
        eprintln!("usage: sky-bench <model.gguf | laya_dir | --random NAME> [--format f32|f16|ternary] [--q8 | --q8-fast] [--no-calib] [--ternary-compute auto|packed|f32] [--exit-threshold T] [--no-static-cache] [--questions 1|2] [--iters N] [--state TEXT] [--options a,b,c] [--json] | --kernels");
        std::process::exit(2);
    }

    fn pct(sorted: &[f64], q: f64) -> f64 {
        sorted[((sorted.len() - 1) as f64 * q).round() as usize]
    }

    pub fn main() {
        let mut args = std::env::args().skip(1);
        let mut path = None;
        let mut random: Option<String> = None;
        let mut format = WeightFormat::F32;
        let mut q8 = false;
        let mut calibrate = true;
        let mut act = ActQuant::default();
        let mut iters = 300usize;
        let mut ternary = TernaryCompute::Auto;
        let mut tau: Option<f32> = None;
        let mut static_cache = true;
        let mut questions = 1usize;
        let mut json = false;
        let mut state = String::from("alt 12 vy -3 dx 40 gt 20 gb 35");
        let mut options = vec![
            "flap: push the bird up".to_string(),
            "wait: let the bird fall".to_string(),
        ];
        while let Some(a) = args.next() {
            match a.as_str() {
                "--kernels" => return kernels(),
                "--q8" => q8 = true,
                "--no-calib" => calibrate = false,
                "--q8-fast" => {
                    q8 = true;
                    act = ActQuant::Q8;
                }
                "--random" => random = args.next(),
                "--format" => {
                    format = match args.next().as_deref() {
                        Some("f32") => WeightFormat::F32,
                        Some("f16") => WeightFormat::F16,
                        Some("ternary") => WeightFormat::Ternary,
                        _ => usage(),
                    }
                }
                "--ternary-compute" => {
                    ternary = match args.next().as_deref() {
                        Some("auto") => TernaryCompute::Auto,
                        Some("packed") => TernaryCompute::Packed,
                        Some("f32") => TernaryCompute::F32,
                        _ => usage(),
                    }
                }
                "--exit-threshold" => {
                    tau = Some(
                        args.next()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or_else(|| usage()),
                    )
                }
                "--no-static-cache" => static_cache = false,
                "--questions" => {
                    questions = args
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                        .clamp(1, 2)
                }
                "--json" => json = true,
                "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(iters),
                "--state" => state = args.next().unwrap_or(state),
                "--options" => {
                    options = args
                        .next()
                        .map(|v| v.split(',').map(str::to_string).collect())
                        .unwrap_or(options)
                }
                "-h" | "--help" => usage(),
                _ => path = Some(a),
            }
        }
        let path = match (path, &random) {
            (Some(p), _) => p,
            (None, Some(r)) => format!("random:{r}"),
            (None, None) => usage(),
        };
        let opts = EngineOptions {
            quantize: q8.then_some(QuantizeSpec {
                encoder: true,
                head: true,
                embeddings: false,
            }),
            calibrate,
            act_quant: act,
            ternary,
            exit_threshold: tau,
            prefix_cache: if static_cache { 16 } else { 0 },
            ..Default::default()
        };
        let t0 = Instant::now();
        let engine = if let Some(r) = &random {
            random_bytes(r, format).and_then(|b| Engine::from_gguf_with(&b, opts))
        } else if std::path::Path::new(&path).is_dir() {
            Engine::from_safetensors_dir_with(&path, opts)
        } else {
            std::fs::read(&path)
                .map_err(sky_infer::Error::from)
                .and_then(|b| Engine::from_gguf_with(&b, opts))
        };
        let mut engine = match engine {
            Ok(e) => e,
            Err(e) => {
                eprintln!("cannot load {path}: {e}");
                std::process::exit(1);
            }
        };
        let info = engine.model_info();
        let load_ms = t0.elapsed().as_secs_f64() * 1e3;
        if !json {
            println!(
                "{} ({}): {} params, d={} L={} H={} format={} q8={} (loaded in {load_ms:.0} ms)",
                info.name,
                info.arch,
                info.n_params,
                info.hidden_size,
                info.num_layers,
                info.head_layers,
                info.weight_format,
                info.q8,
            );
        }

        let ins = "Which action keeps the bird alive?";
        let (ids, _) = engine.build_sequence(
            QType::Choice,
            ins,
            &options.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &state,
        );
        if !json {
            println!("sequence: {} tokens, {} options", ids.len(), options.len());
        }
        let mut times = Vec::with_capacity(iters);
        let mut layers = Vec::with_capacity(iters);
        let mut exits = vec![0usize; info.exits.len().max(1)];
        let mut sink = 0.0f32;
        for i in 0..iters + 20 {
            // a slightly different state every frame
            let s = format!("{state} t {}", i % 50);
            let t = Instant::now();
            let p = engine
                .decide_raw(QType::Choice, ins, &options, &s)
                .expect("decision");
            let stats = engine.last_stats().cloned().unwrap_or_default();
            if questions > 1 {
                let p = engine
                    .decide_raw(QType::Noul, DANGER, &DANGER_OPTS, &s)
                    .expect("decision");
                sink += p[1];
            }
            let dt = t.elapsed().as_secs_f64() * 1e6;
            sink += p[0];
            if i >= 20 {
                times.push(dt);
                if let (Some(e), Some(l)) = (stats.exit, stats.layers_used) {
                    exits[e] += 1;
                    layers.push(l as f64);
                }
            }
        }
        times.sort_by(|a, b| a.total_cmp(b));
        let (median, p95, min) = (pct(&times, 0.5), pct(&times, 0.95), times[0]);
        let mean_layers = if layers.is_empty() {
            None
        } else {
            Some(layers.iter().sum::<f64>() / layers.len() as f64)
        };
        if json {
            println!(
                "{{\"name\":{},\"arch\":{},\"params\":{},\"weight_format\":{},\"ternary_packed\":{},\
                 \"static_cache\":{static_cache},\"questions\":{questions},\"iters\":{iters},\"median_us\":{median:.3},\
                 \"p95_us\":{p95:.3},\"min_us\":{min:.3},\"mean_layers_used\":{},\"exit_counts\":{exits:?}}}",
                sky_infer::json::escape(&info.name),
                sky_infer::json::escape(&info.arch),
                info.n_params,
                sky_infer::json::escape(&info.weight_format),
                info.ternary_packed,
                mean_layers.map_or("null".to_string(), |l| format!("{l:.4}")),
            );
            return;
        }
        println!("latency per decision ({questions} question(s) per frame): median {median:.1} us, p95 {p95:.1} us, min {min:.1} us ({iters} iterations, checksum {sink:.3})");
        if engine.arch() == Arch::S2 {
            println!(
                "mean layers used {:.2} / {}, exit counts {exits:?} (static cache {})",
                mean_layers.unwrap_or(0.0),
                info.num_layers,
                if static_cache { "on" } else { "off" }
            );
        }
    }

    fn random_bytes(name: &str, format: WeightFormat) -> sky_infer::Result<Vec<u8>> {
        if let Some(cfg) = sky_infer::s2::random::ladder_config(name) {
            let tok = sky_infer::s2::random::test_tokenizer();
            return sky_infer::s2::random::random_gguf(&cfg, &tok, 1, format);
        }
        let Some(cfg) = sky_infer::convert::ladder_config(name) else {
            eprintln!("unknown ladder configuration {name} (s1-pico, s1-nano, s1-micro, s2-pico, s2-nano, s2-micro)");
            std::process::exit(2);
        };
        let tok = sky_infer::Tokenizer::from_tokenizer_json(include_str!(
            "../../tests/fixtures/small_tokenizer.json"
        ))
        .expect("tokenizer");
        sky_infer::convert::random_gguf(&cfg, &tok, 1, sky_gguf::GgmlType::F32)
    }

    /// Ternary add/sub kernels vs f32 products on the matrix shapes of the s2 ladder, for a state
    /// stream of 20 tokens and a fusion head of 2 markers.
    fn kernels() {
        use sky_infer::ternary::TernaryMatrix;
        println!(
            "| model | shape (n x k) | rows | f32 (us) | ternary packed (us) | packed / f32 |"
        );
        println!("|---|---|---:|---:|---:|---:|");
        for name in ["s2-pico", "s2-nano", "s2-micro"] {
            let cfg = sky_infer::s2::random::ladder_config(name).expect("ladder");
            let (d, ff) = (cfg.hidden_size, cfg.intermediate_size);
            for &(n, k, rows) in &[(3 * d, d, 20), (2 * ff, d, 20), (d, ff, 20), (2 * d, d, 2)] {
                let mut seed = 0x1234_5678_u64;
                let mut rnd = || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
                };
                let w: Vec<f32> = (0..n * k).map(|_| rnd()).collect();
                let x: Vec<f32> = (0..rows * k).map(|_| rnd()).collect();
                let t = TernaryMatrix::from_f32(&w, n, k);
                let wf = t.to_f32();
                let mut y = vec![0.0; rows * n];
                let reps = (2_000_000 / (n * k * rows)).max(20);
                let time = |f: &mut dyn FnMut()| {
                    let mut best = f64::MAX;
                    for _ in 0..5 {
                        let t0 = Instant::now();
                        for _ in 0..reps {
                            f();
                        }
                        best = best.min(t0.elapsed().as_secs_f64() * 1e6 / reps as f64);
                    }
                    best
                };
                let tf = time(&mut || {
                    sky_infer::kernels::matmul_f32(&x, rows, k, &wf, n, None, &mut y, false);
                    std::hint::black_box(&y);
                });
                let tt = time(&mut || {
                    t.matmul(&x, rows, None, &mut y, false);
                    std::hint::black_box(&y);
                });
                println!(
                    "| {name} | {n} x {k} | {rows} | {tf:.2} | {tt:.2} | {:.2} |",
                    tt / tf
                );
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main()
}

/// The tools need a file system: nothing to do on wasm32.
#[cfg(target_arch = "wasm32")]
fn main() {}
