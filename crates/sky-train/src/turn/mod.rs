//! s6-ear v2 (`s6-ear2-nano`, docs/s6-ear2-prereg.md): an end-of-turn detector distilled from
//! Smart Turn v3.2, modelled on [`crate::cast`]. The shared pure-Rust pieces (the front end, the
//! feature store, the CPU forward) live in `sky_infer::audio` and `sky_infer::turn`.

pub mod config;
pub mod data;
pub mod gguf;
pub mod golden;
pub mod model;
pub mod train;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};

use crate::s2::gguf::WeightFormat;

fn weight_format(f: &str) -> Result<WeightFormat> {
    Ok(match f {
        "f32" => WeightFormat::F32,
        "f16" => WeightFormat::F16,
        "ternary" => WeightFormat::Ternary,
        f => bail!("unknown format {f:?} (f32, f16, ternary)"),
    })
}

/// `turn-export`: a run checkpoint as an `skycmd-turn` GGUF file (§7).
#[derive(Debug, Clone, clap::Args)]
pub struct ExportArgs {
    /// Checkpoint directory (`<run>/last` or `<run>/last-ema`).
    #[arg(long)]
    pub run: PathBuf,
    /// f32 | f16 | ternary
    #[arg(long, default_value = "ternary")]
    pub format: String,
    #[arg(long)]
    pub out: PathBuf,
    /// The calibration written into the file (`{temperature, tau50, tau85}`, from
    /// `scripts/turn-gate.py calibrate`); without it: temperature 1, both thresholds 0.5.
    #[arg(long)]
    pub calib: Option<PathBuf>,
}

pub fn export(a: &ExportArgs) -> Result<()> {
    let (cfg, params) = train::load_run(&a.run)?;
    let calib = match &a.calib {
        Some(p) => serde_json::from_str(
            &std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?,
        )
        .with_context(|| format!("parsing {}", p.display()))?,
        None => gguf::Calibration::default(),
    };
    let format = weight_format(&a.format)?;
    let bytes = mlx_rs::with_device(mlx_rs::Device::cpu(), || {
        gguf::to_gguf(&params, &cfg, format, calib)
    })?;
    if let Some(d) = a.out.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(&a.out, &bytes)?;
    println!(
        "wrote {} ({} bytes, {} parameters exported, {}, temperature {}, tau50 {}, tau85 {})",
        a.out.display(),
        bytes.len(),
        model::exported_count(&cfg),
        format.name(),
        calib.temperature,
        calib.tau50,
        calib.tau85
    );
    Ok(())
}

/// `turn-probe`: one GGUF file of seeded random ternary weights per rung of a ladder config, for
/// the latency probe of §4.4 (`sky-turn --bench`, `crates/sky-web/bench/turn-bench.mjs`).
#[derive(Debug, Clone, clap::Args)]
pub struct ProbeArgs {
    #[arg(long, default_value = "configs/s6-ear2-nano.json")]
    pub config: PathBuf,
    /// Directory of the files (`<rung>.gguf`).
    #[arg(long)]
    pub out: PathBuf,
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
}

pub fn probe(a: &ProbeArgs) -> Result<()> {
    std::fs::create_dir_all(&a.out)?;
    let mut files = Vec::new();
    for r in config::rungs(&a.config)? {
        let cfg = config::TurnConfig::load(&a.config, Some(&r))?;
        // the CPU device: no GPU for a probe
        let bytes = mlx_rs::with_device(mlx_rs::Device::cpu(), || -> Result<Vec<u8>> {
            let p = model::init(&cfg, a.seed)?;
            gguf::to_gguf(&p, &cfg, WeightFormat::Ternary, Default::default())
        })?;
        let path = a.out.join(format!("{r}.gguf"));
        std::fs::write(&path, &bytes)?;
        println!(
            "{r}: d {} ff {} blocks {} hops/position {}: {} parameters, {} bytes -> {}",
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.blocks,
            cfg.hops_per_position,
            model::exported_count(&cfg),
            bytes.len(),
            path.display()
        );
        files.push(
            serde_json::json!({"rung": r, "path": path, "bytes": bytes.len(),
            "params": model::exported_count(&cfg)}),
        );
    }
    std::fs::write(
        a.out.join("files.json"),
        serde_json::to_string_pretty(&files)? + "\n",
    )?;
    Ok(())
}

/// `turn-golden`: reference outputs of a random model for the sky-infer tests.
#[derive(Debug, Clone, clap::Args)]
pub struct GoldenArgs {
    #[arg(long, default_value = "configs/s6-ear2-femto.json")]
    pub config: PathBuf,
    #[arg(long, default_value = "crates/sky-infer/tests/golden")]
    pub out: PathBuf,
    #[arg(long, default_value = "turn")]
    pub prefix: String,
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
}

pub fn golden(a: &GoldenArgs) -> Result<()> {
    let cfg = config::TurnConfig::load(&a.config, None)?;
    // the CPU device: the reference numbers do not depend on the GPU's reduction order
    mlx_rs::with_device(mlx_rs::Device::cpu(), || {
        golden::golden(&cfg, &a.out, &a.prefix, a.seed)
    })?;
    println!("wrote {}/{}.json", a.out.display(), a.prefix);
    Ok(())
}
