//! Latency benchmark: median / p95 per decision for a model file.
//!
//! Usage: `sky-bench <model.gguf | laya_dir | --random s1-pico|s1-nano|s1-micro> [--q8 | --q8-fast] [--no-calib]
//! [--iters N] [--state TEXT] [--options a,b,c]`
//!
//! `--random` benchmarks a ladder configuration with random weights (and a small test tokenizer).
//!
//! `--q8` quantizes the f32 weights at load (Q8_0 weights, two-pass Q8_0 activations);
//! `--q8-fast` uses single-pass Q8_0 activations. A Q8_0 GGUF file runs in Q8_0 as stored.
//!
//! The question and options stay the same while the state changes every iteration, as in a game
//! loop (the prefix tokenization is cached, the state is tokenized each time).

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::time::Instant;

    use sky_infer::kernels::ActQuant;
    use sky_infer::{Engine, EngineOptions, QType, QuantizeSpec};

    pub fn main() {
        let mut args = std::env::args().skip(1);
        let mut path = None;
        let mut random: Option<String> = None;
        let mut q8 = false;
        let mut calibrate = true;
        let mut act = ActQuant::default();
        let mut iters = 300usize;
        let mut state = String::from("alt 12 vy -3 dx 40 gt 20 gb 35");
        let mut options = vec![
            "flap: push the bird up".to_string(),
            "wait: let the bird fall".to_string(),
        ];
        while let Some(a) = args.next() {
            match a.as_str() {
                "--q8" => q8 = true,
                "--no-calib" => calibrate = false,
                "--q8-fast" => {
                    q8 = true;
                    act = ActQuant::Q8;
                }
                "--random" => random = args.next(),
                "--iters" => iters = args.next().and_then(|v| v.parse().ok()).unwrap_or(iters),
                "--state" => state = args.next().unwrap_or(state),
                "--options" => {
                    options = args
                        .next()
                        .map(|v| v.split(',').map(str::to_string).collect())
                        .unwrap_or(options)
                }
                "-h" | "--help" => {
                    println!("usage: sky-bench <model.gguf | laya_dir> [--q8 | --q8-fast] [--no-calib] [--iters N] [--state TEXT] [--options a,b,c]");
                    return;
                }
                _ => path = Some(a),
            }
        }
        let path = match (path, &random) {
            (Some(p), _) => p,
            (None, Some(r)) => format!("random:{r}"),
            (None, None) => {
                eprintln!("usage: sky-bench <model.gguf | laya_dir | --random s1-nano> [--q8 | --q8-fast] [--no-calib] [--iters N]");
                std::process::exit(2);
            }
        };
        let opts = EngineOptions {
            quantize: q8.then_some(QuantizeSpec {
                encoder: true,
                head: true,
                embeddings: false,
            }),
            calibrate,
            act_quant: act,
            ..Default::default()
        };
        let t0 = Instant::now();
        let engine = if let Some(r) = &random {
            let Some(cfg) = sky_infer::convert::ladder_config(r) else {
                eprintln!("unknown ladder configuration {r} (s1-pico, s1-nano, s1-micro)");
                std::process::exit(2);
            };
            let tok = sky_infer::Tokenizer::from_tokenizer_json(include_str!(
                "../../tests/fixtures/small_tokenizer.json"
            ))
            .expect("tokenizer");
            sky_infer::convert::random_gguf(&cfg, &tok, 1, sky_gguf::GgmlType::F32)
                .and_then(|b| Engine::from_gguf_with(&b, opts))
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
        println!(
            "{}: {} params, d={} L={} H={} q8={} (loaded in {:.0} ms)",
            info.name,
            info.n_params,
            info.hidden_size,
            info.num_layers,
            info.head_layers,
            info.q8,
            t0.elapsed().as_secs_f64() * 1e3
        );

        let ins = "Which action keeps the bird alive?";
        let (ids, _) = engine.build_sequence(
            QType::Choice,
            ins,
            &options.iter().map(|s| s.as_str()).collect::<Vec<_>>(),
            &state,
        );
        println!("sequence: {} tokens, {} options", ids.len(), options.len());
        let mut times = Vec::with_capacity(iters);
        let mut sink = 0.0f32;
        for i in 0..iters + 20 {
            // a slightly different state every frame
            let s = format!("{state} t {}", i % 50);
            let t = Instant::now();
            let p = engine
                .decide_raw(QType::Choice, ins, &options, &s)
                .expect("decision");
            let dt = t.elapsed().as_secs_f64() * 1e6;
            sink += p[0];
            if i >= 20 {
                times.push(dt);
            }
        }
        times.sort_by(|a, b| a.total_cmp(b));
        let pct = |q: f64| times[((times.len() - 1) as f64 * q).round() as usize];
        println!("latency per decision: median {:.1} us, p95 {:.1} us, min {:.1} us ({iters} iterations, checksum {sink:.3})",
                 pct(0.5), pct(0.95), times[0]);
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main()
}

/// The tools need a file system: nothing to do on wasm32.
#[cfg(target_arch = "wasm32")]
fn main() {}
