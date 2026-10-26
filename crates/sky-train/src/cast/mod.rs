//! s6-cast: probabilistic time-series forecasting with the s6 state encoder
//! (docs/SPEC-S6-CAST.md). The shared pure-Rust pieces (synthetic generators, normalization,
//! metrics, the CPU forward) live in `sky_infer::cast`.

pub mod config;
pub mod data;
pub mod gguf;
pub mod golden;
pub mod model;
pub mod train;

use std::path::PathBuf;

use anyhow::{bail, Result};

use crate::s2::gguf::WeightFormat;

/// `cast-export`: a run checkpoint as an `skycmd-cast` GGUF file.
#[derive(Debug, Clone, clap::Args)]
pub struct ExportArgs {
    /// Checkpoint directory (`<run>/last`, `<run>/last-ema` or `<run>/pre-qat`).
    #[arg(long)]
    pub run: PathBuf,
    /// f32 | f16 | ternary
    #[arg(long, default_value = "ternary")]
    pub format: String,
    #[arg(long)]
    pub out: PathBuf,
    /// Early-exit tolerance written into the file (`cast-calibrate` picks it; default: the
    /// config's).
    #[arg(long)]
    pub exit_tolerance: Option<f32>,
}

pub fn export(a: &ExportArgs) -> Result<()> {
    let (mut cfg, params) = train::load_run(&a.run)?;
    if let Some(t) = a.exit_tolerance {
        cfg.exit_tolerance = t;
    }
    let format = match a.format.as_str() {
        "f32" => WeightFormat::F32,
        "f16" => WeightFormat::F16,
        "ternary" => WeightFormat::Ternary,
        f => bail!("unknown format {f:?} (f32, f16, ternary)"),
    };
    let bytes = gguf::to_gguf(&params, &cfg, format)?;
    if let Some(d) = a.out.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(&a.out, &bytes)?;
    println!(
        "wrote {} ({} KB, {} parameters, {})",
        a.out.display(),
        bytes.len() / 1024,
        crate::params::count(&params),
        format.name()
    );
    Ok(())
}

/// `cast-golden`: reference outputs of a random model for the sky-infer tests.
#[derive(Debug, Clone, clap::Args)]
pub struct GoldenArgs {
    #[arg(long, default_value = "configs/s6-cast-femto.json")]
    pub config: PathBuf,
    #[arg(long, default_value = "crates/sky-infer/tests/golden")]
    pub out: PathBuf,
    #[arg(long, default_value = "cast")]
    pub prefix: String,
    #[arg(long, default_value_t = 7)]
    pub seed: u64,
}

pub fn golden(a: &GoldenArgs) -> Result<()> {
    let cfg = config::CastConfig::load(&a.config)?;
    // the CPU device: the reference numbers do not depend on the GPU's reduction order
    mlx_rs::with_device(mlx_rs::Device::cpu(), || {
        golden::golden(&cfg, &a.out, &a.prefix, a.seed)
    })?;
    println!("wrote {}/{}.json", a.out.display(), a.prefix);
    Ok(())
}

/// `cast-synth`: synthetic series as JSONL (to inspect the generators, or as demo data).
#[derive(Debug, Clone, clap::Args)]
pub struct SynthArgs {
    /// gp | pulse | tsi | sarima (default: the training mix)
    #[arg(long)]
    pub kind: Option<String>,
    #[arg(long, default_value_t = 100)]
    pub n: usize,
    #[arg(long, default_value_t = 2048)]
    pub len: usize,
    #[arg(long, default_value_t = 1)]
    pub seed: u64,
    #[arg(long)]
    pub out: PathBuf,
    /// Instead of synthetic series: the state variables of these games (flappy, rescue, dodge,
    /// doom), played by the oracle.
    #[arg(long, value_delimiter = ',')]
    pub games: Vec<String>,
    /// Episodes per game (with `--games`).
    #[arg(long, default_value_t = 50)]
    pub episodes: usize,
}

pub fn synth(a: &SynthArgs) -> Result<()> {
    if !a.games.is_empty() {
        let n = data::write_games(&a.out, &a.games, a.episodes, a.seed)?;
        println!("wrote {n} game series to {}", a.out.display());
        return Ok(());
    }
    let kind = match &a.kind {
        None => None,
        Some(k) => Some(
            sky_infer::cast::synth::Kind::parse(k)
                .ok_or_else(|| anyhow::anyhow!("unknown kind {k:?} (gp, pulse, tsi, sarima)"))?,
        ),
    };
    data::write_synth(&a.out, kind, a.n, a.len, a.seed)?;
    println!("wrote {} series to {}", a.n, a.out.display());
    Ok(())
}
