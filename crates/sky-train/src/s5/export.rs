//! `sky-train s5-export`: an iso checkpoint (variant B) as a `skycmd-iso` GGUF file for sky-infer
//! and the browser. Variant A is exported with `sky-train export` (Laya layout) and
//! `sky-convert`, which keeps it readable by llama.cpp; `skycmd-iso` runs only in sky-infer.
//!
//! Keys (prefix `skycmd-iso.`): `vocab_size`, `embedding_length`, `feed_forward_length`,
//! `block_count`, `attention.head_count`, `attention.layer_norm_epsilon`, `rope.freq_base`
//! (global layers), `rope.freq_base_swa` (local layers), `attention.sliding_window` (the full
//! local window), `attention.sliding_window_pattern` (a layer is global when `l % n == 0`),
//! `context_length` (longest sequence), `max_head_tokens`, `max_option_tokens`, `positions`
//! (`shared` or `reset`), `temperature.<type>[.<bucket>]`, and the generic `noul` definitions
//! `noul.false.<lang>` / `noul.true.<lang>`.
//!
//! Tensors: `token_embd.weight`, `token_embd_norm.weight`, `blk.<l>.{attn_norm, attn_qkv,
//! attn_output, ffn_norm, ffn_up, ffn_down}.weight` (no `attn_norm` for layer 0),
//! `output_norm.weight`, `iso.score.{0,1}.{weight,bias}` (always F32).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use sky_gguf::{encode_f32, GgmlType, GgufWriter, OwnedValue};

use crate::config::ModelConfig;
use crate::model::host_f32;
use crate::params::{self, ParamMap};
use crate::s2::gguf::write_tokenizer;
use crate::s5::iso::Positions;
use crate::s5::record::generic_noul;

/// Architecture name of the iso GGUF files.
pub const ARCH: &str = "skycmd-iso";

/// Longest option text, in tokens.
pub const MAX_OPTION_TOKENS: usize = crate::sequence::OPTION_MAX_TOKENS;

#[derive(Debug, Clone, clap::Args)]
pub struct ExportArgs {
    /// Checkpoint directory of an iso run (`<out>/last` of `s5-train --variant b-*`).
    #[arg(long)]
    pub run: PathBuf,
    #[arg(long)]
    pub out: PathBuf,
    /// f32 | f16 | q8_0 | q4_0 (matrices whose rows are a multiple of 32; the readout stays f32).
    #[arg(long, default_value = "f32")]
    pub format: String,
    /// Model name (default: the config name).
    #[arg(long)]
    pub name: Option<String>,
}

/// `(GGUF name, s1 name)` of every encoder tensor.
pub fn tensor_names(cfg: &ModelConfig) -> Vec<(String, String)> {
    let mut v = vec![
        (
            "token_embd.weight".into(),
            "encoder.embeddings.tok_embeddings.weight".into(),
        ),
        (
            "token_embd_norm.weight".into(),
            "encoder.embeddings.norm.weight".into(),
        ),
    ];
    for l in 0..cfg.num_hidden_layers {
        let e = format!("encoder.layers.{l}");
        let b = format!("blk.{l}");
        if l > 0 {
            v.push((
                format!("{b}.attn_norm.weight"),
                format!("{e}.attn_norm.weight"),
            ));
        }
        v.push((
            format!("{b}.attn_qkv.weight"),
            format!("{e}.attn.Wqkv.weight"),
        ));
        v.push((
            format!("{b}.attn_output.weight"),
            format!("{e}.attn.Wo.weight"),
        ));
        v.push((
            format!("{b}.ffn_norm.weight"),
            format!("{e}.mlp_norm.weight"),
        ));
        v.push((format!("{b}.ffn_up.weight"), format!("{e}.mlp.Wi.weight")));
        v.push((format!("{b}.ffn_down.weight"), format!("{e}.mlp.Wo.weight")));
    }
    v.push((
        "output_norm.weight".into(),
        "encoder.final_norm.weight".into(),
    ));
    for n in [
        "iso.score.0.weight",
        "iso.score.0.bias",
        "iso.score.1.weight",
        "iso.score.1.bias",
    ] {
        v.push((n.into(), n.into()));
    }
    v
}

/// Builds the GGUF bytes of an iso model.
#[allow(clippy::too_many_arguments)]
pub fn to_gguf(
    p: &ParamMap,
    cfg: &ModelConfig,
    positions: Positions,
    temperatures: &[(String, f32)],
    tokenizer_json: &str,
    format: GgmlType,
    name: &str,
) -> Result<Vec<u8>> {
    let mut w = GgufWriter::new();
    let k = |s: &str| format!("{ARCH}.{s}");
    let u = |v: usize| OwnedValue::U32(v as u32);
    w.add_kv("general.architecture", ARCH);
    w.add_kv("general.type", "model");
    w.add_kv("general.name", name);
    w.add_kv(k("vocab_size"), u(cfg.vocab_size));
    w.add_kv(k("embedding_length"), u(cfg.hidden_size));
    w.add_kv(k("feed_forward_length"), u(cfg.intermediate_size));
    w.add_kv(k("block_count"), u(cfg.num_hidden_layers));
    w.add_kv(k("attention.head_count"), u(cfg.n_heads()));
    w.add_kv(
        k("attention.layer_norm_epsilon"),
        OwnedValue::F32(cfg.norm_eps),
    );
    w.add_kv(k("rope.freq_base"), OwnedValue::F32(cfg.global_rope_theta));
    w.add_kv(
        k("rope.freq_base_swa"),
        OwnedValue::F32(cfg.local_rope_theta),
    );
    w.add_kv(k("attention.sliding_window"), u(cfg.local_attention));
    w.add_kv(
        k("attention.sliding_window_pattern"),
        u(cfg.global_attn_every_n_layers),
    );
    w.add_kv(k("context_length"), u(cfg.rl_agent.max_len));
    w.add_kv(k("max_head_tokens"), u(cfg.rl_agent.head_max_len));
    w.add_kv(k("max_option_tokens"), u(MAX_OPTION_TOKENS));
    w.add_kv(
        k("positions"),
        match positions {
            Positions::Shared => "shared",
            Positions::Reset => "reset",
        },
    );
    for (key, t) in temperatures {
        w.add_kv(k(&format!("temperature.{key}")), OwnedValue::F32(*t));
    }
    for lang in ["en", "fr"] {
        let (f, t) = generic_noul(lang);
        w.add_kv(k(&format!("noul.false.{lang}")), f);
        w.add_kv(k(&format!("noul.true.{lang}")), t);
    }
    write_tokenizer(&mut w, tokenizer_json, cfg.vocab_size)?;
    for (gguf, src) in tensor_names(cfg) {
        let a = p
            .get(src.as_str())
            .ok_or_else(|| anyhow!("missing parameter {src}"))?;
        let shape: Vec<usize> = a.shape().iter().map(|&d| d as usize).collect();
        let values = host_f32(a)?;
        let dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
        let matrix = shape.len() == 2 && !gguf.starts_with("iso.");
        let ty = match format {
            GgmlType::Q8_0 | GgmlType::Q4_0 if matrix && shape[1].is_multiple_of(32) => format,
            GgmlType::Q8_0 | GgmlType::Q4_0 | GgmlType::F16 if matrix => GgmlType::F16,
            _ => GgmlType::F32,
        };
        w.add_tensor(gguf, &dims, ty, encode_f32(ty, &values)?)?;
    }
    Ok(w.to_bytes()?)
}

/// Temperatures of a checkpoint (`temperatures.json`, `{"choice": T, "choice.3_5": T, ...}`).
fn read_temperatures(dir: &Path) -> Result<Vec<(String, f32)>> {
    let path = dir.join("temperatures.json");
    if !path.exists() {
        return Ok(Vec::new());
    }
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
    let mut out: Vec<(String, f32)> = v
        .as_object()
        .ok_or_else(|| anyhow!("{} is not an object", path.display()))?
        .iter()
        .filter_map(|(k, t)| Some((k.clone(), t.as_f64()? as f32)))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

pub fn export(a: &ExportArgs) -> Result<()> {
    let s5: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(a.run.join("s5.json")).context("s5.json of the run")?,
    )?;
    let variant = crate::s5::train::Variant::parse(
        s5["args"]["variant"]
            .as_str()
            .ok_or_else(|| anyhow!("s5.json has no variant"))?,
    )?;
    let Some(positions) = variant.iso() else {
        bail!("variant {variant:?} is not iso: use `sky-train export` + sky-convert");
    };
    let cfg: ModelConfig =
        serde_json::from_str(&std::fs::read_to_string(a.run.join("config.json"))?)?;
    let p = params::load(&a.run.join("model.safetensors"))?;
    let tok = std::fs::read_to_string(a.run.join("tokenizer.json"))?;
    let ty = match a.format.as_str() {
        "f32" => GgmlType::F32,
        "f16" => GgmlType::F16,
        "q8_0" => GgmlType::Q8_0,
        "q4_0" => GgmlType::Q4_0,
        other => bail!("unknown format {other:?} (f32, f16, q8_0, q4_0)"),
    };
    let name = a.name.clone().unwrap_or_else(|| cfg.name.clone());
    let temps = read_temperatures(&a.run)?;
    let bytes = to_gguf(&p, &cfg, positions, &temps, &tok, ty, &name)?;
    if let Some(d) = a.out.parent() {
        std::fs::create_dir_all(d)?;
    }
    std::fs::write(&a.out, &bytes)?;
    println!(
        "wrote {} ({:.1} MB, {}, positions {positions:?}, {} temperatures)",
        a.out.display(),
        bytes.len() as f64 / 1e6,
        a.format,
        temps.len()
    );
    Ok(())
}
