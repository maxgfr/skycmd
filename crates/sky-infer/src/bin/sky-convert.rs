//! Convert a Laya checkpoint directory (safetensors + configs + tokenizer) to GGUF with the
//! llama.cpp `modern-bert` names and keys.
//!
//! Usage: `sky-convert <laya_dir> <out.gguf> [--type f32|f16|bf16|q8_0] [--head-type f32|f16|bf16|q8_0]
//! [--pre NAME] [--name NAME]`
//!
//! `sky-convert --random s1-pico|s1-nano|s1-micro <out.gguf> [--type ...]` writes a model with
//! random weights and a small test tokenizer instead (to exercise the runtime before training).
//!
//! `sky-convert --random s2-pico|s2-nano|s2-micro|s3-pico|s3-nano <out.gguf> [--format f32|f16|ternary]
//! [--tokenizer tok-s2.json]` writes a random `skycmd-s2` model (SPEC-S2 §5), with the given s2
//! tokenizer (`[NUM]` at id 5) or a small test tokenizer.
//!
//! `--type` applies to the encoder matrices (default f16); the decision head (head blocks,
//! `token_types`, `cls.*`) keeps `--head-type` (default f32). Norms and biases are always f32.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use sky_gguf::{GgmlType, Gguf};
    use sky_infer::convert::{ConvertOptions, LayaDir};
    use sky_infer::s2::random::WeightFormat;

    fn parse_type(s: &str) -> Option<GgmlType> {
        match s.to_ascii_lowercase().as_str() {
            "f32" => Some(GgmlType::F32),
            "f16" => Some(GgmlType::F16),
            "bf16" => Some(GgmlType::BF16),
            "q8_0" | "q8" => Some(GgmlType::Q8_0),
            _ => None,
        }
    }

    const USAGE: &str =
        "usage: sky-convert <laya_dir | --random s1-pico|s1-nano|s1-micro> <out.gguf> [--type f32|f16|bf16|q8_0] [--head-type f32|f16|bf16|q8_0] [--pre NAME] [--name NAME]\n       sky-convert --random s2-pico|s2-nano|s2-micro|s3-pico|s3-nano <out.gguf> [--format f32|f16|ternary] [--tokenizer tok-s2.json] [--name NAME]";

    fn fail(msg: &str) -> ! {
        eprintln!("{msg}");
        std::process::exit(2);
    }

    pub fn main() {
        let mut args = std::env::args().skip(1);
        let mut pos = Vec::new();
        let mut opts = ConvertOptions::default();
        let mut random: Option<String> = None;
        let mut format = WeightFormat::F32;
        let mut tokenizer: Option<String> = None;
        while let Some(a) = args.next() {
            let mut value = |name: &str| {
                args.next()
                    .unwrap_or_else(|| fail(&format!("{name} needs a value\n{USAGE}")))
            };
            match a.as_str() {
                "--type" => {
                    opts.ftype = parse_type(&value("--type")).unwrap_or_else(|| fail(USAGE))
                }
                "--head-type" => {
                    opts.head_type =
                        parse_type(&value("--head-type")).unwrap_or_else(|| fail(USAGE))
                }
                "--pre" => opts.tokenizer_pre = value("--pre"),
                "--name" => opts.name = Some(value("--name")),
                "--random" => random = Some(value("--random")),
                "--format" => {
                    format = match value("--format").as_str() {
                        "f32" => WeightFormat::F32,
                        "f16" => WeightFormat::F16,
                        "ternary" => WeightFormat::Ternary,
                        _ => fail(USAGE),
                    }
                }
                "--tokenizer" => tokenizer = Some(value("--tokenizer")),
                "-h" | "--help" => {
                    println!("{USAGE}");
                    return;
                }
                _ => pos.push(a),
            }
        }
        let t0 = std::time::Instant::now();
        let (bytes, out) = match (random, pos.as_slice()) {
            (Some(name), [out]) if name.starts_with("s2-") || name.starts_with("s3-") => {
                let mut cfg = sky_infer::s2::random::ladder_config(&name)
                    .unwrap_or_else(|| fail(&format!("unknown ladder configuration {name}")));
                if let Some(n) = &opts.name {
                    cfg.name = n.clone();
                }
                let tok = match &tokenizer {
                    Some(p) => {
                        let json = std::fs::read_to_string(p)
                            .unwrap_or_else(|e| fail(&format!("{p}: {e}")));
                        sky_infer::Tokenizer::from_tokenizer_json(&json)
                            .unwrap_or_else(|e| fail(&format!("{p}: {e}")))
                    }
                    None => sky_infer::s2::random::test_tokenizer(),
                };
                let bytes = sky_infer::s2::random::random_gguf(&cfg, &tok, 1, format)
                    .unwrap_or_else(|e| fail(&format!("random model: {e}")));
                (bytes, out)
            }
            (Some(name), [out]) => {
                let cfg = sky_infer::convert::ladder_config(&name)
                    .unwrap_or_else(|| fail(&format!("unknown ladder configuration {name}")));
                let tok = sky_infer::Tokenizer::from_tokenizer_json(include_str!(
                    "../../tests/fixtures/small_tokenizer.json"
                ))
                .expect("test tokenizer");
                let bytes = sky_infer::convert::random_gguf(&cfg, &tok, 1, opts.ftype)
                    .unwrap_or_else(|e| fail(&format!("random model: {e}")));
                (bytes, out)
            }
            (None, [dir, out]) => {
                let laya = LayaDir::open(std::path::Path::new(dir))
                    .unwrap_or_else(|e| fail(&format!("{dir}: {e}")));
                let bytes = laya
                    .sources()
                    .to_gguf(&opts)
                    .unwrap_or_else(|e| fail(&format!("conversion failed: {e}")));
                (bytes, out)
            }
            _ => fail(USAGE),
        };
        std::fs::write(out, &bytes).unwrap_or_else(|e| fail(&format!("{out}: {e}")));
        let g = Gguf::parse(&bytes).expect("written file parses");
        let n_params: u64 = g.tensors.iter().map(|t| t.n_elements()).sum();
        if g.architecture() == Some(sky_infer::s2::ARCH) {
            let n = sky_infer::Engine::from_gguf(&bytes).map_or(0, |e| e.n_params());
            println!(
                "wrote {out}: {} tensors, {n} parameters, {:.1} KB, skycmd-s2 {} ({:.1} s)",
                g.tensors.len(),
                bytes.len() as f64 / 1024.0,
                format.name(),
                t0.elapsed().as_secs_f64()
            );
            return;
        }
        println!(
            "wrote {out}: {} tensors, {n_params} parameters, {:.1} MB, encoder {} / head {} ({:.1} s)",
            g.tensors.len(),
            bytes.len() as f64 / 1e6,
            opts.ftype.name(),
            opts.head_type.name(),
            t0.elapsed().as_secs_f64()
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() {
    native::main()
}

/// The tools need a file system: nothing to do on wasm32.
#[cfg(target_arch = "wasm32")]
fn main() {}
